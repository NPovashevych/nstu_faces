from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from db.enums import EmbeddingType
from db.models import DBEmbedding, DBPerson
from db.session import SessionLocal
from services.faiss.faiss_face_index import normalize_vector


LOG_FILE = Path("../logs/auto_reidentify_v1.log")

# DRY_RUN = True
DRY_RUN = False
EXPECTED_PERSONS = 27
EXPECTED_UPDATES = 173
DISTANCE_EPS = 0.00015


def get_reidentified_person_ids() -> list[int]:
    pattern = re.compile(r"AUTO UPDATE .*?reference_person_id=(\d+)")
    person_ids = set()

    with LOG_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            match = pattern.search(line)

            if match:
                person_ids.add(int(match.group(1)))

    return sorted(person_ids)


def get_reference_vectors(db: Session, person_id: int) -> np.ndarray:
    rows = db.query(DBEmbedding.vector).filter(DBEmbedding.person_id == person_id, DBEmbedding.embedding_type == EmbeddingType.reference_face, DBEmbedding.vector.isnot(None)).all()

    vectors = [normalize_vector(np.asarray(row.vector, dtype=np.float32)) for row in rows]

    if not vectors:
        raise RuntimeError(f"No reference embeddings for person_id={person_id}")

    return np.vstack(vectors).astype(np.float32)


def distance_to_reference_person(vector, reference_vectors: np.ndarray) -> float:
    face_vector = normalize_vector(np.asarray(vector, dtype=np.float32))
    similarities = reference_vectors @ face_vector
    return 1.0 - float(np.max(similarities))


def get_changed_embeddings(db: Session, person_id: int):
    person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if person is None:
        raise RuntimeError(f"Person not found: person_id={person_id}")

    reference_vectors = get_reference_vectors(db, person_id)

    embeddings = db.query(DBEmbedding).filter(DBEmbedding.person_id == person_id, DBEmbedding.embedding_type == EmbeddingType.detected_face, DBEmbedding.vector.isnot(None)).order_by(DBEmbedding.id).all()

    changed_embeddings = []

    for embedding in embeddings:
        new_distance = distance_to_reference_person(embedding.vector, reference_vectors)
        old_distance = (embedding.source or {}).get("distance")

        if old_distance is None or abs(float(old_distance) - new_distance) > DISTANCE_EPS:
            changed_embeddings.append((embedding, old_distance, new_distance))

    print(f"{person.name} | person_id={person_id} | detected={len(embeddings)} | to_update={len(changed_embeddings)}")

    return changed_embeddings


def main():
    person_ids = get_reidentified_person_ids()

    print("=" * 100)
    print("BACKFILL REIDENTIFIED DISTANCES")
    print("=" * 100)
    print(f"Log: {LOG_FILE}")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"Persons found: {len(person_ids)}")
    print(f"Expected updates: {EXPECTED_UPDATES}")
    print("=" * 100)

    if len(person_ids) != EXPECTED_PERSONS:
        raise RuntimeError(f"Expected {EXPECTED_PERSONS} reidentified persons, found {len(person_ids)}")

    db = SessionLocal()

    try:
        updates = []

        for number, person_id in enumerate(person_ids, start=1):
            print(f"[{number}/{len(person_ids)}] ", end="")
            updates.extend(get_changed_embeddings(db, person_id))

        print()
        print("=" * 100)
        print(f"Embeddings requiring update: {len(updates)}")
        print("=" * 100)

        for embedding, old_distance, new_distance in updates:
            print(f"embedding_id={embedding.id} | person_id={embedding.person_id} | old={old_distance} | new={new_distance:.8f}")

        if len(updates) != EXPECTED_UPDATES:
            raise RuntimeError(f"SAFETY STOP: expected {EXPECTED_UPDATES} updates, found {len(updates)}. Database was not changed.")

        if DRY_RUN:
            db.rollback()
            print()
            print("DRY RUN finished. Database changed: False")
            return

        for embedding, old_distance, new_distance in updates:
            source = dict(embedding.source or {})
            source["distance"] = float(new_distance)
            embedding.source = source

        db.commit()

        print()
        print("=" * 100)
        print(f"COMMIT | embeddings updated: {len(updates)}")
        print("Database changed: True")
        print("=" * 100)

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
