import argparse
import logging
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.models import DBPerson, DBEmbedding, DBFace
from db.enums import PersonStatus, EmbeddingType


Path("../services/logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../services/logs/reidentify_existing_faces.log", encoding="utf-8"),
    ],
)


AUTO_THRESHOLD = 0.43
REVIEW_THRESHOLD = 0.46

DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.03


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def cosine_distance(a, b):
    return 1 - float(np.dot(a, b))


def get_confidence(dist: float) -> int | None:
    if dist <= DIST_TOLERANCE:
        return 0

    for i in range(1, 4):
        if dist <= DIST_TOLERANCE + i * STEP_TOLERANCE:
            return i

    return None


def load_known_persons(db: Session, created_after: date | None):
    query = (
        db.query(DBPerson)
        .join(DBEmbedding, DBEmbedding.person_id == DBPerson.id)
        .filter(DBPerson.status != PersonStatus.unknown)
        .filter(DBEmbedding.embedding_type == EmbeddingType.reference_face)
        .distinct()
        .order_by(DBPerson.id)
    )

    if created_after:
        query = query.filter(DBPerson.created_at >= created_after)

    return query.all()


def load_reference_embeddings(db: Session, person_id: int):
    rows = (
        db.query(DBEmbedding)
        .filter(DBEmbedding.person_id == person_id)
        .filter(DBEmbedding.embedding_type == EmbeddingType.reference_face)
        .all()
    )

    return [
        normalize(np.array(row.vector, dtype=np.float32))
        for row in rows
    ]


def load_unknown_detected_embeddings(db: Session):
    rows = (
        db.query(DBEmbedding, DBPerson)
        .join(DBPerson, DBEmbedding.person_id == DBPerson.id)
        .filter(DBPerson.status == PersonStatus.unknown)
        .filter(DBEmbedding.embedding_type == EmbeddingType.detected_face)
        .all()
    )

    items = []

    for emb, person in rows:
        items.append({
            "embedding_id": emb.id,
            "unknown_person_id": person.id,
            "unknown_name": person.name,
            "cluster_id": person.cluster_id,
            "cluster_tag": person.cluster_tag,
            "vector": normalize(np.array(emb.vector, dtype=np.float32)),
        })

    return items


def find_candidates(reference_vectors, unknown_items, review_threshold: float):
    clusters = {}

    for item in unknown_items:
        best_dist = min(
            cosine_distance(ref, item["vector"])
            for ref in reference_vectors
        )

        if best_dist > review_threshold:
            continue

        unknown_person_id = item["unknown_person_id"]

        if unknown_person_id not in clusters:
            clusters[unknown_person_id] = {
                "unknown_person_id": unknown_person_id,
                "unknown_name": item["unknown_name"],
                "cluster_id": item["cluster_id"],
                "cluster_tag": item["cluster_tag"],
                "best_distance": best_dist,
                "matched_embeddings": 1,
            }
        else:
            clusters[unknown_person_id]["matched_embeddings"] += 1
            clusters[unknown_person_id]["best_distance"] = min(
                clusters[unknown_person_id]["best_distance"],
                best_dist,
            )

    return sorted(clusters.values(), key=lambda x: x["best_distance"])


def resolve_cluster(db: Session, known_person: DBPerson, unknown_person_id: int, distance: float):
    confidence = get_confidence(distance)

    faces_updated = (
        db.query(DBFace)
        .filter(DBFace.person_id == unknown_person_id)
        .update(
            {
                DBFace.person_id: known_person.id,
                DBFace.confidence: confidence,
            },
            synchronize_session=False,
        )
    )

    embeddings_updated = (
        db.query(DBEmbedding)
        .filter(DBEmbedding.person_id == unknown_person_id)
        .filter(DBEmbedding.embedding_type == EmbeddingType.detected_face)
        .update(
            {DBEmbedding.person_id: known_person.id},
            synchronize_session=False,
        )
    )

    unknown_person = (
        db.query(DBPerson)
        .filter(DBPerson.id == unknown_person_id)
        .first()
    )

    if unknown_person:
        old_name = unknown_person.name

        unknown_person.status = PersonStatus.non_public
        unknown_person.name = f"{old_name}_resolved_to_{known_person.id}"
        unknown_person.link = known_person.link
        # cluster_tag НЕ чіпаємо

    return faces_updated, embeddings_updated


def parse_date(value: str | None):
    if not value:
        return None

    return datetime.strptime(value, "%Y-%m-%d").date()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--created-after",
        type=str,
        default=None,
        help="Перевіряти тільки персон, створених після дати YYYY-MM-DD",
    )

    parser.add_argument(
        "--today",
        action="store_true",
        help="Перевіряти тільки персон, створених сьогодні",
    )

    parser.add_argument(
        "--auto-threshold",
        type=float,
        default=AUTO_THRESHOLD,
    )

    parser.add_argument(
        "--review-threshold",
        type=float,
        default=REVIEW_THRESHOLD,
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Без --apply буде тільки dry-run",
    )

    args = parser.parse_args()

    if args.today:
        created_after = date.today()
    else:
        created_after = parse_date(args.created_after)

    db = SessionLocal()

    try:
        known_persons = load_known_persons(db, created_after)
        unknown_items = load_unknown_detected_embeddings(db)

        logging.info(f"Known persons to check: {len(known_persons)}")
        logging.info(f"Unknown detected embeddings: {len(unknown_items)}")

        resolved_unknown_ids = set()

        total_faces = 0
        total_embeddings = 0

        for known_person in known_persons:
            refs = load_reference_embeddings(db, known_person.id)

            if not refs:
                continue

            candidates = find_candidates(
                reference_vectors=refs,
                unknown_items=unknown_items,
                review_threshold=args.review_threshold,
            )

            if not candidates:
                continue

            logging.info("--------------------------------")
            logging.info(f"Known person: id={known_person.id}, name={known_person.name}")

            for c in candidates:
                if c["unknown_person_id"] in resolved_unknown_ids:
                    continue

                decision = (
                    "AUTO"
                    if c["best_distance"] <= args.auto_threshold
                    else "REVIEW"
                )

                logging.info(
                    f"{decision} | "
                    f"cluster={c['cluster_tag']} | "
                    f"unknown_person_id={c['unknown_person_id']} | "
                    f"distance={c['best_distance']:.4f} | "
                    f"matched_embeddings={c['matched_embeddings']}"
                )

                if not args.apply:
                    continue

                if decision != "AUTO":
                    continue

                faces_updated, embeddings_updated = resolve_cluster(
                    db=db,
                    known_person=known_person,
                    unknown_person_id=c["unknown_person_id"],
                    distance=c["best_distance"],
                )

                resolved_unknown_ids.add(c["unknown_person_id"])

                total_faces += faces_updated
                total_embeddings += embeddings_updated

                logging.info(
                    f"RESOLVED {c['cluster_tag']} -> {known_person.name}: "
                    f"faces={faces_updated}, embeddings={embeddings_updated}"
                )

        if args.apply:
            db.commit()
            logging.info("--------------------------------")
            logging.info(f"Total faces updated: {total_faces}")
            logging.info(f"Total embeddings updated: {total_embeddings}")
            logging.info("Done")
        else:
            logging.info("--------------------------------")
            logging.info("DRY RUN only. Щоб застосувати, додай --apply")

    except Exception:
        db.rollback()
        logging.exception("Error")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Start: {start}")

    main()

    finish = datetime.now()
    logging.info(f"Finished. Running time: {finish - start}")
