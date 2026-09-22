from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from db.enums import EmbeddingType
from db.models import DBEmbedding, DBFace, DBPerson
from db.session import SessionLocal
from services.faiss.faiss_face_index import ReferenceFaceIndex, normalize_vector
from services.config import REVIEW_JSON_FILE
from commons.common_face import get_confidence


START_FACE_ID = 1
END_FACE_ID = 5_550_037

CATEGORY_IDS = [1, 2, 3, 7]

MAX_MATCH_DISTANCE = 0.40
AUTO_CLUSTER_DISTANCE = 0.54

BATCH_SIZE = 1000

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/auto_reidentify_v1.log", mode="w", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


def get_reference_person_ids_subquery():
    return select(DBEmbedding.person_id).where(DBEmbedding.embedding_type == EmbeddingType.reference_face)


def get_candidate_clusters(db: Session):
    reference_person_ids = get_reference_person_ids_subquery()

    return (
        db.query(DBPerson.id.label("person_id"), DBPerson.name.label("person_name"))
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


def get_cluster_faces_batch(db: Session, person_id: int, last_face_id: int):
    return (
        db.query(DBFace.id.label("face_id"), DBFace.embedding_id.label("embedding_id"), DBEmbedding.vector.label("vector"))
        .join(DBEmbedding, DBEmbedding.id == DBFace.embedding_id)
        .filter(DBFace.person_id == person_id, DBFace.id > last_face_id, DBFace.category_id.in_(CATEGORY_IDS), DBEmbedding.embedding_type == EmbeddingType.detected_face)
        .order_by(DBFace.id)
        .limit(BATCH_SIZE)
        .all()
    )


def get_reference_vectors(db: Session, reference_person_id: int) -> np.ndarray:
    rows = (
        db.query(DBEmbedding.vector)
        .filter(DBEmbedding.person_id == reference_person_id, DBEmbedding.embedding_type == EmbeddingType.reference_face)
        .all()
    )

    vectors = []

    for row in rows:
        if row.vector is None:
            continue

        vector = normalize_vector(np.asarray(row.vector, dtype=np.float32))
        vectors.append(vector)

    if not vectors:
        raise RuntimeError(f"No reference embeddings for person_id={reference_person_id}")

    return np.vstack(vectors).astype(np.float32)


def distance_to_reference_person(vector, reference_vectors: np.ndarray) -> float:
    face_vector = normalize_vector(np.asarray(vector, dtype=np.float32))
    similarities = reference_vectors @ face_vector
    best_similarity = float(np.max(similarities))
    return 1.0 - best_similarity


def analyze_cluster(db: Session, reference_index: ReferenceFaceIndex, cluster_id: int, cluster_name: str) -> dict:
    faces_count = 0
    discovery_matches = []
    matched_reference_ids = set()

    last_face_id = 0

    while True:
        rows = get_cluster_faces_batch(db, cluster_id, last_face_id)

        if not rows:
            break

        for row in rows:
            faces_count += 1

            if row.vector is None:
                continue

            vector = normalize_vector(np.asarray(row.vector, dtype=np.float32))
            best_ref, distance = reference_index.find_best_match(vector)

            if best_ref is None or distance is None or distance > MAX_MATCH_DISTANCE:
                continue

            reference_person_id = int(best_ref["person_id"])
            matched_reference_ids.add(reference_person_id)

            discovery_matches.append(
                {
                    "face_id": int(row.face_id),
                    "reference_person_id": reference_person_id,
                    "distance": float(distance),
                }
            )

        last_face_id = int(rows[-1].face_id)

    if not discovery_matches:
        return {
            "status": "no_match",
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "faces_count": faces_count,
            "matches_count": 0,
        }

    if len(matched_reference_ids) > 1:
        references = []

        for reference_person_id in sorted(matched_reference_ids):
            reference_person = db.query(DBPerson).filter(DBPerson.id == reference_person_id).first()

            reference_matches = [item for item in discovery_matches if item["reference_person_id"] == reference_person_id]

            references.append(
                {
                    "reference_person_id": reference_person_id,
                    "reference_person_name": reference_person.name if reference_person else None,
                    "matches_count": len(reference_matches),
                    "best_distance": min(item["distance"] for item in reference_matches),
                }
            )

        return {
            "status": "conflict",
            "reason": "multiple_references_inside_cluster",
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "faces_count": faces_count,
            "matches_count": len(discovery_matches),
            "references": references,
        }

    reference_person_id = next(iter(matched_reference_ids))
    reference_person = db.query(DBPerson).filter(DBPerson.id == reference_person_id).first()

    if reference_person is None:
        raise RuntimeError(f"Reference person not found: person_id={reference_person_id}")

    reference_vectors = get_reference_vectors(db, reference_person_id)

    all_faces = []
    max_distance = 0.0

    last_face_id = 0

    while True:
        rows = get_cluster_faces_batch(db, cluster_id, last_face_id)

        if not rows:
            break

        for row in rows:
            if row.vector is None:
                continue

            distance = distance_to_reference_person(row.vector, reference_vectors)
            max_distance = max(max_distance, distance)

            all_faces.append(
                {
                    "face_id": int(row.face_id),
                    "embedding_id": int(row.embedding_id),
                    "distance": float(distance),
                }
            )

        last_face_id = int(rows[-1].face_id)

    review_faces = [item for item in all_faces if item["distance"] > AUTO_CLUSTER_DISTANCE]
    review_faces.sort(key=lambda item: item["distance"])

    result = {
        "status": "automatic" if max_distance <= AUTO_CLUSTER_DISTANCE else "review",
        "reason": None if max_distance <= AUTO_CLUSTER_DISTANCE else "cluster_distance",
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "faces_count": faces_count,
        "matches_count": len(discovery_matches),
        "reference_person_id": reference_person_id,
        "reference_person_name": reference_person.name,
        "max_distance": float(max_distance),
        "review_faces": review_faces,
    }

    return result


def move_duplicate_references_to_review(automatic_list: list[dict], review_list: list[dict]):
    reference_clusters = defaultdict(list)

    for item in automatic_list:
        reference_person_id = item.get("reference_person_id")

        if reference_person_id is not None:
            reference_clusters[reference_person_id].append(item)

    for item in review_list:
        if item.get("status") == "conflict":
            continue

        reference_person_id = item.get("reference_person_id")

        if reference_person_id is not None:
            reference_clusters[reference_person_id].append(item)

    duplicate_reference_ids = {reference_person_id for reference_person_id, items in reference_clusters.items() if len(items) > 1}

    if not duplicate_reference_ids:
        return

    logger.info("References used by multiple unknown clusters: %s", len(duplicate_reference_ids))

    for reference_person_id in sorted(duplicate_reference_ids):
        items = reference_clusters[reference_person_id]

        related_clusters = [
            {
                "cluster_id": item["cluster_id"],
                "cluster_name": item["cluster_name"],
                "faces_count": item["faces_count"],
                "max_distance": item.get("max_distance"),
            }
            for item in items
        ]

        logger.info("DUPLICATE REFERENCE | reference_person_id=%s | clusters=%s", reference_person_id, ", ".join(item["cluster_name"] for item in items))

        for item in items:
            item["status"] = "review"
            item["reason"] = "multiple_clusters_same_reference"
            item["related_clusters"] = related_clusters

    automatic_list[:] = [item for item in automatic_list if item.get("reference_person_id") not in duplicate_reference_ids]

    existing_review_cluster_ids = {item["cluster_id"] for item in review_list}

    for reference_person_id in duplicate_reference_ids:
        for item in reference_clusters[reference_person_id]:
            if item["cluster_id"] not in existing_review_cluster_ids:
                review_list.append(item)
                existing_review_cluster_ids.add(item["cluster_id"])


def update_automatic_cluster(db: Session, item: dict):
    cluster_id = int(item["cluster_id"])
    cluster_name = item["cluster_name"]
    reference_person_id = int(item["reference_person_id"])

    old_person = db.query(DBPerson).filter(DBPerson.id == cluster_id).with_for_update().first()

    if old_person is None:
        raise RuntimeError(f"Old person not found: person_id={cluster_id}")

    if old_person.name != cluster_name:
        raise RuntimeError(f"Old person name changed: person_id={cluster_id}, expected={cluster_name}, actual={old_person.name}")

    if not old_person.name.lower().startswith("unknown_cluster_"):
        raise RuntimeError(f"Person is no longer an unknown cluster: person_id={cluster_id}, name={old_person.name}")

    reference_person = db.query(DBPerson).filter(DBPerson.id == reference_person_id).first()

    if reference_person is None:
        raise RuntimeError(f"Reference person not found: person_id={reference_person_id}")

    reference_vectors = get_reference_vectors(db, reference_person_id)

    updated_faces = 0
    updated_embeddings = 0
    last_face_id = 0

    while True:
        rows = get_cluster_faces_batch(db, cluster_id, last_face_id)

        if not rows:
            break

        for row in rows:
            if row.vector is None:
                raise RuntimeError(f"Face has no embedding vector: face_id={row.face_id}")

            distance = distance_to_reference_person(row.vector, reference_vectors)

            if distance > AUTO_CLUSTER_DISTANCE:
                raise RuntimeError(f"Safety check failed: cluster={cluster_name}, face_id={row.face_id}, distance={distance:.4f}, limit={AUTO_CLUSTER_DISTANCE:.4f}")

            face = db.query(DBFace).filter(DBFace.id == row.face_id).first()

            if face is None:
                raise RuntimeError(f"Face not found: face_id={row.face_id}")

            if face.person_id != cluster_id:
                raise RuntimeError(f"Face person_id changed: face_id={face.id}, expected={cluster_id}, actual={face.person_id}")

            embedding = db.query(DBEmbedding).filter(DBEmbedding.id == row.embedding_id).first()

            if embedding is None:
                raise RuntimeError(f"Embedding not found: embedding_id={row.embedding_id}")

            if embedding.embedding_type != EmbeddingType.detected_face:
                raise RuntimeError(f"Unexpected embedding type: embedding_id={embedding.id}, type={embedding.embedding_type}")

            if embedding.person_id != cluster_id:
                raise RuntimeError(f"Embedding person_id changed: embedding_id={embedding.id}, expected={cluster_id}, actual={embedding.person_id}")

            face.person_id = reference_person_id
            face.confidence = get_confidence(distance)
            embedding.person_id = reference_person_id

            updated_faces += 1
            updated_embeddings += 1

        db.flush()

        last_face_id = int(rows[-1].face_id)

    if updated_faces != item["faces_count"]:
        raise RuntimeError(f"Face count mismatch: cluster={cluster_name}, analyzed={item['faces_count']}, updated={updated_faces}")

    old_person.name = f"reidentified_{reference_person_id}"

    db.flush()

    logger.info("AUTO UPDATE | %s | old_person_id=%s | reference=%s | reference_person_id=%s | faces=%s | new_old_person_name=%s", cluster_name, cluster_id, reference_person.name, reference_person_id, updated_faces, old_person.name)

    return updated_faces, updated_embeddings



def save_review_json(review_list: list[dict], reference_embeddings_count: int):
    review_list.sort(key=lambda item: (item.get("max_distance") is None, item.get("max_distance", float("inf"))))

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_embeddings": reference_embeddings_count,
        "match_distance": MAX_MATCH_DISTANCE,
        "automatic_cluster_distance": AUTO_CLUSTER_DISTANCE,
        "clusters_count": len(review_list),
        "clusters": review_list,
    }

    with REVIEW_JSON_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    started_at = datetime.now()

    db = SessionLocal()

    try:
        db.execute(text("SET max_parallel_workers_per_gather = 0"))

        logger.info("============================================================")
        logger.info("AUTO REIDENTIFY V1")
        logger.info("============================================================")
        logger.info("Face ID range: %s–%s", START_FACE_ID, END_FACE_ID)
        logger.info("Categories: %s", CATEGORY_IDS)
        logger.info("Reference match distance: %.2f", MAX_MATCH_DISTANCE)
        logger.info("Automatic cluster distance: %.2f", AUTO_CLUSTER_DISTANCE)
        logger.info("Batch size: %s", BATCH_SIZE)
        logger.info("============================================================")

        logger.info("Building reference FAISS index...")

        reference_index = ReferenceFaceIndex()
        reference_index.build(db)

        reference_embeddings_count = reference_index.index.ntotal if reference_index.index is not None else 0

        logger.info("Reference embeddings: %s", reference_embeddings_count)

        logger.info("Searching unknown candidate clusters...")

        clusters = get_candidate_clusters(db)

        logger.info("Unknown candidate clusters: %s", len(clusters))

        automatic_list = []
        review_list = []

        no_match_count = 0
        conflict_count = 0
        positive_count = 0

        for number, cluster in enumerate(clusters, start=1):
            result = analyze_cluster(db, reference_index, int(cluster.person_id), cluster.person_name)

            if result["status"] == "no_match":
                no_match_count += 1

            elif result["status"] == "conflict":
                conflict_count += 1
                positive_count += 1
                review_list.append(result)

                logger.info("[%s/%s] REVIEW CONFLICT | %s | faces=%s | references=%s", number, len(clusters), result["cluster_name"], result["faces_count"], len(result["references"]))

            elif result["status"] == "automatic":
                positive_count += 1
                automatic_list.append(result)

                logger.info("[%s/%s] AUTO CANDIDATE | %s | faces=%s | reference=%s | matches<=%.2f=%s | max_distance=%.4f", number, len(clusters), result["cluster_name"], result["faces_count"], result["reference_person_name"], MAX_MATCH_DISTANCE, result["matches_count"], result["max_distance"])

            elif result["status"] == "review":
                positive_count += 1
                review_list.append(result)

                logger.info("[%s/%s] REVIEW DISTANCE | %s | faces=%s | reference=%s | matches<=%.2f=%s | max_distance=%.4f", number, len(clusters), result["cluster_name"], result["faces_count"], result["reference_person_name"], MAX_MATCH_DISTANCE, result["matches_count"], result["max_distance"])

            if number % 5000 == 0:
                logger.info("Progress: %s/%s clusters", number, len(clusters))

        logger.info("============================================================")
        logger.info("ANALYSIS FINISHED")
        logger.info("Positive clusters before duplicate check: %s", positive_count)
        logger.info("Automatic before duplicate check: %s", len(automatic_list))
        logger.info("Review before duplicate check: %s", len(review_list))
        logger.info("Conflicts inside cluster: %s", conflict_count)
        logger.info("No match: %s", no_match_count)
        logger.info("============================================================")

        move_duplicate_references_to_review(automatic_list, review_list)

        logger.info("Automatic after duplicate check: %s", len(automatic_list))
        logger.info("Review after duplicate check: %s", len(review_list))

        save_review_json(review_list, reference_embeddings_count)

        logger.info("Review JSON saved: %s", REVIEW_JSON_FILE)

        logger.info("============================================================")
        logger.info("STARTING DATABASE UPDATES")
        logger.info("============================================================")

        total_updated_faces = 0
        total_updated_embeddings = 0
        successful_clusters = 0

        for number, item in enumerate(automatic_list, start=1):
            try:
                updated_faces, updated_embeddings = update_automatic_cluster(db, item)

                db.commit()

                total_updated_faces += updated_faces
                total_updated_embeddings += updated_embeddings
                successful_clusters += 1

                logger.info("COMMIT [%s/%s] | %s -> %s | faces=%s", number, len(automatic_list), item["cluster_name"], item["reference_person_name"], updated_faces)

            except Exception:
                db.rollback()

                logger.exception("ROLLBACK [%s/%s] | cluster=%s", number, len(automatic_list), item["cluster_name"])

                raise

        elapsed = datetime.now() - started_at

        logger.info("============================================================")
        logger.info("RESULT")
        logger.info("Unknown candidate clusters:     %s", len(clusters))
        logger.info("Positive clusters:              %s", positive_count)
        logger.info("Automatic clusters:             %s", len(automatic_list))
        logger.info("Review clusters:                %s", len(review_list))
        logger.info("Automatic clusters committed:   %s", successful_clusters)
        logger.info("Faces updated:                  %s", total_updated_faces)
        logger.info("Embeddings updated:             %s", total_updated_embeddings)
        logger.info("Review JSON:                    %s", REVIEW_JSON_FILE)
        logger.info("All time:                       %s", elapsed)
        logger.info("============================================================")

    except Exception:
        db.rollback()
        logger.exception("AUTO REIDENTIFY FAILED")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
