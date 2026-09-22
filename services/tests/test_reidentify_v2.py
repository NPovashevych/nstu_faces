import logging
import sys
from collections import defaultdict
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.enums import EmbeddingType
from db.models import DBEmbedding, DBFace, DBFaceCategory, DBFreeze, DBPerson

from services.faiss.faiss_face_index import ReferenceFaceIndex, normalize_vector


START_FACE_ID = 1
END_FACE_ID = 500000

CATEGORY_IDS = [1, 2, 3, 7]

MAX_MATCH_DISTANCE = 0.40

BATCH_SIZE = 1000
PROGRESS_STEP = 1000

LOG_FILE = "../logs/test_reidentify_unknown_clusters.log"


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)


def get_reference_person_ids_subquery():
    return select(DBEmbedding.person_id).where(DBEmbedding.embedding_type == EmbeddingType.reference_face)


def get_candidate_clusters(db: Session):
    reference_person_ids = get_reference_person_ids_subquery()

    rows = (
        db.query(
            DBPerson.id.label("person_id"),
            DBPerson.name.label("person_name"),
        )
        .join(DBFace, DBFace.person_id == DBPerson.id)
        .join(DBEmbedding, DBEmbedding.id == DBFace.embedding_id)
        .filter(
            DBFace.id >= START_FACE_ID,
            DBFace.id <= END_FACE_ID,
            DBFace.category_id.in_(CATEGORY_IDS),
            DBEmbedding.embedding_type == EmbeddingType.detected_face,
            ~DBFace.person_id.in_(reference_person_ids),
            DBPerson.name.ilike("unknown_cluster_%"),
        )
        .distinct()
        .order_by(DBPerson.id)
        .all()
    )

    return rows


def get_cluster_face_count(db: Session, cluster_person_id: int) -> int:
    return (
        db.query(DBFace.id)
        .join(DBEmbedding, DBEmbedding.id == DBFace.embedding_id)
        .filter(
            DBFace.person_id == cluster_person_id,
            DBFace.category_id.in_(CATEGORY_IDS),
            DBEmbedding.embedding_type == EmbeddingType.detected_face,
        )
        .count()
    )


def get_cluster_faces_batch(db: Session, cluster_person_id: int, last_face_id: int, batch_size: int):
    return (
        db.query(
            DBFace.id.label("face_id"),
            DBFace.embedding_id.label("embedding_id"),
            DBFaceCategory.name.label("category_name"),
            DBFreeze.freeze_path.label("freeze_path"),
            DBEmbedding.vector.label("vector"),
            DBFace.gender.label("gender"),
        )
        .join(DBEmbedding, DBEmbedding.id == DBFace.embedding_id)
        .join(DBFaceCategory, DBFaceCategory.id == DBFace.category_id)
        .join(DBFreeze, DBFreeze.id == DBFace.freeze_id)
        .filter(
            DBFace.person_id == cluster_person_id,
            DBFace.id > last_face_id,
            DBFace.category_id.in_(CATEGORY_IDS),
            DBEmbedding.embedding_type == EmbeddingType.detected_face,
        )
        .order_by(DBFace.id)
        .limit(batch_size)
        .all()
    )


def get_person_name(db: Session, person_id: int, cache: dict) -> str:
    if person_id not in cache:
        person = db.get(DBPerson, person_id)
        cache[person_id] = person.name if person else f"person_id={person_id}"

    return cache[person_id]


def main():
    db = SessionLocal()

    try:
        start_time = datetime.now()

        db.execute(text("SET max_parallel_workers_per_gather = 0"))

        logging.info("PostgreSQL parallel query disabled for this session")
        logging.info(f"Log file: {LOG_FILE}")
        logging.info("")

        logging.info("Building reference FAISS...")
        faiss_start = datetime.now()

        reference_index = ReferenceFaceIndex()
        reference_index.build(db)

        logging.info(f"Reference FAISS ready: {reference_index.size} embeddings")
        logging.info(f"FAISS build time: {datetime.now() - faiss_start}")
        logging.info(f"Face ID range for cluster discovery: {START_FACE_ID}–{END_FACE_ID}")
        logging.info(f"Categories: {CATEGORY_IDS}")
        logging.info(f"Maximum distance: {MAX_MATCH_DISTANCE}")
        logging.info(f"Batch size: {BATCH_SIZE}")
        logging.info("")

        logging.info("Searching unknown candidate clusters...")
        cluster_search_start = datetime.now()

        clusters = get_candidate_clusters(db)

        logging.info(f"Unknown candidate clusters found: {len(clusters)}")
        logging.info(f"Cluster search time: {datetime.now() - cluster_search_start}")
        logging.info("")

        person_name_cache = {}

        total_cluster_faces = 0
        total_faces_checked = 0
        total_matches = 0
        clusters_with_matches = 0
        clusters_single_reference = 0
        clusters_with_conflicts = 0

        for cluster_number, cluster in enumerate(clusters, start=1):
            cluster_person_id = cluster.person_id
            cluster_face_count = get_cluster_face_count(db, cluster_person_id)
            total_cluster_faces += cluster_face_count

            matches_by_reference = defaultdict(list)

            cluster_faces_checked = 0
            cluster_matches = 0
            last_face_id = 0
            batch_number = 0

            while True:
                batch_number += 1
                faces = get_cluster_faces_batch(db, cluster_person_id, last_face_id, BATCH_SIZE)

                if not faces:
                    break


                for row in faces:
                    cluster_faces_checked += 1
                    total_faces_checked += 1

                    emb = normalize_vector(row.vector)
                    best_ref, best_dist = reference_index.find_best_match(emb)

                    if best_ref is None or best_dist is None or best_dist > MAX_MATCH_DISTANCE:
                        continue

                    reference_person_id = best_ref["person_id"]

                    matches_by_reference[reference_person_id].append(
                        {
                            "face_id": row.face_id,
                            "distance": float(best_dist),
                            "freeze_path": row.freeze_path,
                            "category": row.category_name,
                            "gender": row.gender.value if row.gender is not None else None,
                        }
                    )

                    cluster_matches += 1
                    total_matches += 1

                last_face_id = faces[-1].face_id

                del faces

            if not matches_by_reference:
                continue

            clusters_with_matches += 1

            if len(matches_by_reference) == 1:
                clusters_single_reference += 1
                logging.info("STATUS=SINGLE_REFERENCE")
            else:
                clusters_with_conflicts += 1
                logging.info("STATUS=CONFLICT")

            sorted_references = sorted(matches_by_reference.items(), key=lambda item: (-len(item[1]), min(match["distance"] for match in item[1])))

            for reference_person_id, matches in sorted_references:
                reference_name = get_person_name(db, reference_person_id, person_name_cache)

                distances = [match["distance"] for match in matches]
                min_distance = min(distances)
                avg_distance = sum(distances) / len(distances)
                max_distance = max(distances)
                match_share = len(matches) / cluster_face_count if cluster_face_count else 0

                logging.info("")
                logging.info(f"REFERENCE | person_id={reference_person_id} | name={reference_name}")
                logging.info(f"matches={len(matches)} | share={match_share:.4%} | min={min_distance:.4f} | avg={avg_distance:.4f} | max={max_distance:.4f}")

                for match in sorted(matches, key=lambda item: item["distance"]):
                    logging.info(f"MATCH | face_id={match['face_id']} | distance={match['distance']:.4f} | category={match['category']} | gender={match['gender']} | freeze_path={match['freeze_path']}")

            logging.info("")

        logging.info("")
        logging.info("================ RESULT ================")
        logging.info(f"Face ID range for cluster discovery: {START_FACE_ID}–{END_FACE_ID}")
        logging.info(f"Unknown candidate clusters:           {len(clusters)}")
        logging.info(f"Full cluster faces:                   {total_cluster_faces}")
        logging.info(f"Faces checked:                        {total_faces_checked}")
        logging.info(f"Matches <= {MAX_MATCH_DISTANCE:.2f}:                   {total_matches}")
        logging.info(f"Clusters with matches:                {clusters_with_matches}")
        logging.info(f"Clusters single reference:            {clusters_single_reference}")
        logging.info(f"Clusters with conflicts:              {clusters_with_conflicts}")
        logging.info("========================================")
        logging.info(f"All time: {datetime.now() - start_time}")

    except Exception:
        logging.exception("SCRIPT FAILED")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
