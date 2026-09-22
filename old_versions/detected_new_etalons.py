import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy.orm import joinedload

from db.session import SessionLocal
from db.models import DBPerson, DBEmbedding, DBFace
from db.enums import EmbeddingType, PersonStatus


AUTO_THRESHOLD = 0.42
REVIEW_THRESHOLD = 0.48

DRY_RUN = True  # якщо лог норм — False

OUT_DIR = Path("redetect_reports")
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s"
)


def normalize(vec):
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def cosine_distance(vec1, vec2):
    v1 = normalize(vec1)
    v2 = normalize(vec2)
    sim = float(np.dot(v1, v2))
    return 1 - sim


def get_confidence(distance: float):
    if distance <= 0.45:
        return 0
    if distance <= 0.48:
        return 1
    if distance <= 0.51:
        return 2
    if distance <= 0.54:
        return 3
    return None


def load_reference_embeddings(db):
    refs = (
        db.query(DBEmbedding)
        .join(DBPerson, DBEmbedding.person_id == DBPerson.id)
        .filter(DBEmbedding.type == EmbeddingType.reference_face)
        .filter(DBPerson.status != PersonStatus.unknown)
        .options(joinedload(DBEmbedding.person))
        .all()
    )

    logging.info(f"Reference embeddings: {len(refs)}")
    return refs


def load_faces_for_redetection(db):
    faces = (
        db.query(DBFace)
        .join(DBEmbedding, DBFace.embedding_id == DBEmbedding.id)
        .outerjoin(DBPerson, DBFace.person_id == DBPerson.id)
        .filter(DBEmbedding.type == EmbeddingType.detected_face)
        .filter(
            (
                DBPerson.status == PersonStatus.unknown
            )
            | (DBFace.person_id.is_(None))
            | (DBFace.confidence.is_(None))
            | (DBFace.confidence > 0)
            | (DBFace.suspicion.isnot(None))
        )
        .options(
            joinedload(DBFace.embedding),
            joinedload(DBFace.person),
            joinedload(DBFace.freeze),
        )
        .all()
    )

    logging.info(f"Faces for redetection: {len(faces)}")
    return faces


def find_best_match(face, reference_embeddings):
    best = None

    for ref in reference_embeddings:
        dist = cosine_distance(face.embedding.vector, ref.vector)

        if best is None or dist < best["distance"]:
            best = {
                "person_id": ref.person_id,
                "person_name": ref.person.name if ref.person else None,
                "reference_embedding_id": ref.id,
                "distance": dist,
            }

    return best


def redetect_faces():
    db = SessionLocal()

    auto_matches = []
    review_matches = []
    ignored = []

    try:
        refs = load_reference_embeddings(db)
        faces = load_faces_for_redetection(db)

        for face in faces:
            best = find_best_match(face, refs)

            if not best:
                continue

            distance = best["distance"]

            row = {
                "face_id": face.id,
                "freeze_id": face.freeze_id,
                "old_person_id": face.person_id,
                "old_person_name": face.person.name if face.person else None,
                "candidate_person_id": best["person_id"],
                "candidate_person_name": best["person_name"],
                "reference_embedding_id": best["reference_embedding_id"],
                "distance": round(distance, 4),
                "old_confidence": face.confidence,
                "old_suspicion": getattr(face, "suspicion", None),
            }

            if distance <= AUTO_THRESHOLD:
                auto_matches.append(row)

                logging.info(
                    f"AUTO | face_id={face.id} | "
                    f"{row['old_person_name']} -> {best['person_name']} | "
                    f"distance={distance:.4f}"
                )

                if not DRY_RUN:
                    face.person_id = best["person_id"]
                    face.confidence = get_confidence(distance)

            elif distance <= REVIEW_THRESHOLD:
                review_matches.append(row)

                logging.info(
                    f"REVIEW | face_id={face.id} | "
                    f"candidate={best['person_name']} | "
                    f"distance={distance:.4f}"
                )

            else:
                ignored.append(row)

        if not DRY_RUN:
            db.commit()
            logging.info("DB updated.")
        else:
            db.rollback()
            logging.info("DRY_RUN=True, DB not changed.")

    finally:
        db.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "created_at": datetime.now().isoformat(),
        "dry_run": DRY_RUN,
        "auto_threshold": AUTO_THRESHOLD,
        "review_threshold": REVIEW_THRESHOLD,
        "auto_count": len(auto_matches),
        "review_count": len(review_matches),
        "ignored_count": len(ignored),
        "auto_matches": auto_matches,
        "review_matches": review_matches,
    }

    report_path = OUT_DIR / f"redetect_report_{timestamp}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logging.info("--------------------------------")
    logging.info(f"AUTO:   {len(auto_matches)}")
    logging.info(f"REVIEW: {len(review_matches)}")
    logging.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    logging.info(f"Start redetection | DRY_RUN={DRY_RUN}")
    redetect_faces()
