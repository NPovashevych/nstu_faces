import logging
import sys
from collections import defaultdict
from datetime import datetime

import faiss
import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.enums import EmbeddingType
from db.models import DBEmbedding, DBFace, DBPerson

from services.faiss.faiss_face_index import ReferenceFaceIndex, normalize_vector


START_FACE_ID = 1
END_FACE_ID = 5550037

CATEGORY_IDS = [1, 2, 3, 7]

MAX_MATCH_DISTANCE = 0.40

BATCH_SIZE = 1000

LOG_FILE = "../logs/test_reidentify_v3.log"


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

    return (
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
            DBEmbedding.vector.label("vector"),
        )
        .join(DBEmbedding, DBEmbedding.id == DBFace.embedding_id)
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


def get_reference_vectors(db: Session, person_id: int):
    rows = (
        db.query(DBEmbedding.vector)
        .filter(
            DBEmbedding.person_id == person_id,
            DBEmbedding.embedding_type == EmbeddingType.reference_face,
            DBEmbedding.vector.isnot(None),
        )
        .all()
    )

    if not rows:
        return None

    vectors = np.asarray([row.vector for row in rows], dtype=np.float32)
    faiss.normalize_L2(vectors)

    return vectors


def distance_to_reference_person(vector, reference_vectors: np.ndarray) -> float:
    face_vector = normalize_vector(vector)

    similarities = reference_vectors @ face_vector
    best_similarity = float(np.max(similarities))

    return 1.0 - best_similarity


def main():
    db = SessionLocal()

    try:
        start_time = datetime.now()

        db.execute(text("SET max_parallel_workers_per_gather = 0"))

        logging.info("Building reference FAISS...")
        faiss_start = datetime.now()

        reference_index = ReferenceFaceIndex()
        reference_index.build(db)

        logging.info(f"Reference FAISS ready: {reference_index.size} embeddings")
        logging.info(f"FAISS build time: {datetime.now() - faiss_start}")
        logging.info(f"Face ID range for cluster discovery: {START_FACE_ID}–{END_FACE_ID}")
        logging.info(f"Categories: {CATEGORY_IDS}")
        logging.info(f"Maximum distance: {MAX_MATCH_DISTANCE}")
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

        for cluster in clusters:
            cluster_person_id = cluster.person_id
            cluster_name = cluster.person_name
            cluster_face_count = get_cluster_face_count(db, cluster_person_id)

            total_cluster_faces += cluster_face_count

            matches_by_reference = defaultdict(list)

            last_face_id = 0

            while True:
                faces = get_cluster_faces_batch(db, cluster_person_id, last_face_id, BATCH_SIZE)

                if not faces:
                    break

                for row in faces:
                    total_faces_checked += 1

                    emb = normalize_vector(row.vector)
                    best_ref, best_dist = reference_index.find_best_match(emb)

                    if best_ref is None or best_dist is None or best_dist > MAX_MATCH_DISTANCE:
                        continue

                    reference_person_id = best_ref["person_id"]
                    matches_by_reference[reference_person_id].append(row.face_id)
                    total_matches += 1

                last_face_id = faces[-1].face_id
                del faces

            if not matches_by_reference:
                continue

            clusters_with_matches += 1

            if len(matches_by_reference) != 1:
                clusters_with_conflicts += 1

                reference_info = ", ".join(
                    f"{get_person_name(db, person_id, person_name_cache)}={len(face_ids)}"
                    for person_id, face_ids in matches_by_reference.items()
                )

                logging.info(f"{cluster_name} | faces={cluster_face_count} | CONFLICT | {reference_info}")
                continue

            clusters_single_reference += 1

            reference_person_id = next(iter(matches_by_reference))
            reference_name = get_person_name(db, reference_person_id, person_name_cache)
            reference_vectors = get_reference_vectors(db, reference_person_id)

            if reference_vectors is None:
                logging.info(f"{cluster_name} | faces={cluster_face_count} | reference={reference_name} | ERROR: no reference vectors")
                continue

            matches_count = 0
            rest_count = 0
            rest_min = None
            rest_max = None

            last_face_id = 0

            while True:
                faces = get_cluster_faces_batch(db, cluster_person_id, last_face_id, BATCH_SIZE)

                if not faces:
                    break

                for row in faces:
                    distance = distance_to_reference_person(row.vector, reference_vectors)

                    if distance <= MAX_MATCH_DISTANCE:
                        matches_count += 1
                    else:
                        rest_count += 1

                        if rest_min is None or distance < rest_min:
                            rest_min = distance

                        if rest_max is None or distance > rest_max:
                            rest_max = distance

                last_face_id = faces[-1].face_id
                del faces

            if rest_count:
                rest_distance = f"{rest_min:.4f}–{rest_max:.4f}"
            else:
                rest_distance = "—"

            logging.info(
                f"{cluster_name} | faces={cluster_face_count} | reference={reference_name} | "
                f"matches<={MAX_MATCH_DISTANCE:.2f}={matches_count} | rest={rest_count} | rest_distance={rest_distance}"
            )

        logging.info("")
        logging.info("================ RESULT ================")
        logging.info(f"Face ID range for cluster discovery: {START_FACE_ID}–{END_FACE_ID}")
        logging.info(f"Unknown candidate clusters:           {len(clusters)}")
        logging.info(f"Full cluster faces:                   {total_cluster_faces}")
        logging.info(f"Faces checked during discovery:       {total_faces_checked}")
        logging.info(f"Discovery matches <= {MAX_MATCH_DISTANCE:.2f}:          {total_matches}")
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
