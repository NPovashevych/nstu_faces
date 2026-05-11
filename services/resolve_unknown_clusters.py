import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.models import DBPerson, DBEmbedding, DBFace
from db.enums import PersonStatus, EmbeddingType

from routers.commons import normalize, cosine_distance

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/resolve_unknown_clusters.log", encoding="utf-8"),
    ],
)


AUTO_THRESHOLD = 0.43
REVIEW_THRESHOLD = 0.46


def get_known_person(db: Session, person_id: int | None, name: str | None):
    query = db.query(DBPerson)

    if person_id is not None:
        return query.filter(DBPerson.id == person_id).first()

    if name:
        return query.filter(DBPerson.name.ilike(f"%{name}%")).first()

    return None


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


def load_unknown_embeddings(db: Session):
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

        if best_dist <= review_threshold:
            cluster_key = item["unknown_person_id"]

            if cluster_key not in clusters:
                clusters[cluster_key] = {
                    "unknown_person_id": item["unknown_person_id"],
                    "unknown_name": item["unknown_name"],
                    "cluster_id": item["cluster_id"],
                    "cluster_tag": item["cluster_tag"],
                    "best_distance": best_dist,
                    "matched_embeddings": 1,
                }
            else:
                clusters[cluster_key]["matched_embeddings"] += 1
                clusters[cluster_key]["best_distance"] = min(
                    clusters[cluster_key]["best_distance"],
                    best_dist,
                )

    return sorted(
        clusters.values(),
        key=lambda x: x["best_distance"],
    )


def resolve_cluster(
    db: Session,
    known_person: DBPerson,
    unknown_person_id: int,
):
    faces_updated = (
        db.query(DBFace)
        .filter(DBFace.person_id == unknown_person_id)
        .update(
            {DBFace.person_id: known_person.id},
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

    return faces_updated, embeddings_updated


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--person-id", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)

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
        help="Якщо не вказано --apply, буде тільки dry-run",
    )

    args = parser.parse_args()

    db = SessionLocal()

    try:
        known_person = get_known_person(
            db=db,
            person_id=args.person_id,
            name=args.name,
        )

        if not known_person:
            logging.error("Known person not found")
            return

        logging.info(f"Known person: id={known_person.id}, name={known_person.name}")

        refs = load_reference_embeddings(db, known_person.id)

        if not refs:
            logging.error("No reference embeddings for this person")
            return

        logging.info(f"Reference embeddings: {len(refs)}")

        unknown_items = load_unknown_embeddings(db)
        logging.info(f"Unknown detected embeddings: {len(unknown_items)}")

        candidates = find_candidates(
            reference_vectors=refs,
            unknown_items=unknown_items,
            review_threshold=args.review_threshold,
        )

        if not candidates:
            logging.info("No similar unknown clusters found")
            return

        logging.info("Candidates:")
        logging.info("--------------------------------")

        for c in candidates:
            decision = (
                "AUTO"
                if c["best_distance"] <= args.auto_threshold
                else "REVIEW"
            )

            logging.info(
                f"{decision} | "
                f"unknown_person_id={c['unknown_person_id']} | "
                f"cluster_tag={c['cluster_tag']} | "
                f"distance={c['best_distance']:.4f} | "
                f"matched_embeddings={c['matched_embeddings']}"
            )

        if not args.apply:
            logging.info("--------------------------------")
            logging.info("DRY RUN only. Щоб застосувати: додай --apply")
            return

        logging.info("--------------------------------")
        logging.info("APPLY MODE")

        total_faces = 0
        total_embeddings = 0

        for c in candidates:
            if c["best_distance"] > args.auto_threshold:
                logging.info(
                    f"Skip REVIEW candidate: {c['cluster_tag']} "
                    f"distance={c['best_distance']:.4f}"
                )
                continue

            faces_updated, embeddings_updated = resolve_cluster(
                db=db,
                known_person=known_person,
                unknown_person_id=c["unknown_person_id"],
            )

            total_faces += faces_updated
            total_embeddings += embeddings_updated

            logging.info(
                f"Resolved {c['cluster_tag']} -> {known_person.name}: "
                f"faces={faces_updated}, embeddings={embeddings_updated}"
            )

        db.commit()

        logging.info("--------------------------------")
        logging.info(f"Total faces updated: {total_faces}")
        logging.info(f"Total embeddings updated: {total_embeddings}")
        logging.info("Done")

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
