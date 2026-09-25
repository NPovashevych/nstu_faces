import numpy as np

from db.enums import EmbeddingType, PersonStatus
from db.models import DBEmbedding, DBPerson
from db.session import SessionLocal
from services.faiss.faiss_face_index import normalize_vector


MAX_DISTANCE = 0.615
BATCH_SIZE = 5000
ALLOWED_STATUSES = [PersonStatus.public, PersonStatus.non_public]


def get_reference_vectors(db):
    rows = db.query(DBEmbedding.person_id, DBEmbedding.vector).join(DBPerson, DBEmbedding.person_id == DBPerson.id).filter(
        DBEmbedding.embedding_type == EmbeddingType.reference_face,
        DBEmbedding.vector.isnot(None),
        DBPerson.status.in_(ALLOWED_STATUSES),
    ).order_by(DBEmbedding.id).all()

    references = {}

    for row in rows:
        references.setdefault(row.person_id, []).append(normalize_vector(row.vector))

    for person_id in references:
        references[person_id] = np.vstack(references[person_id]).astype(np.float32)

    return references


def main():
    db = SessionLocal()

    try:
        print("=" * 100)
        print("TEST CURRENT INSIGHTFACE DISTANCES")
        print("=" * 100)

        references = get_reference_vectors(db)

        print(f"Persons with references: {len(references)}")
        print(f"Reference embeddings: {sum(len(vectors) for vectors in references.values())}")

        query = db.query(DBEmbedding.id, DBEmbedding.person_id, DBEmbedding.vector, DBEmbedding.source).join(DBPerson, DBEmbedding.person_id == DBPerson.id).filter(
            DBEmbedding.embedding_type == EmbeddingType.detected_face,
            DBEmbedding.person_id.isnot(None),
            DBEmbedding.vector.isnot(None),
            DBPerson.status.in_(ALLOWED_STATUSES),
        )

        total = query.count()

        print(f"Detected embeddings: {total}")
        print("=" * 100)

        processed = 0
        valid = 0
        over_limit = 0
        no_references = 0
        old_null = 0
        changed = 0
        unchanged = 0
        last_id = 0

        while True:
            rows = query.filter(DBEmbedding.id > last_id).order_by(DBEmbedding.id).limit(BATCH_SIZE).all()

            if not rows:
                break

            for row in rows:
                reference_vectors = references.get(row.person_id)

                if reference_vectors is None:
                    no_references += 1
                    print(f"NO REFERENCES | embedding={row.id} | person={row.person_id}")
                    continue

                vector = normalize_vector(row.vector)
                similarities = reference_vectors @ vector
                new_distance = float(1.0 - np.max(similarities))
                old_distance = (row.source or {}).get("distance")

                if old_distance is None:
                    old_null += 1
                elif abs(float(old_distance) - new_distance) <= 0.00015:
                    unchanged += 1
                else:
                    changed += 1

                if new_distance > MAX_DISTANCE:
                    over_limit += 1
                    print(f"OVER LIMIT | embedding={row.id} | person={row.person_id} | old={old_distance} | new={new_distance:.4f}")
                else:
                    valid += 1

            processed += len(rows)
            last_id = rows[-1].id

            print(
                f"Processed {processed}/{total} | valid={valid} | over_limit={over_limit} | "
                f"old_null={old_null} | changed={changed} | unchanged={unchanged} | no_refs={no_references}"
            )

        print()
        print("=" * 100)
        print("RESULT")
        print("=" * 100)
        print(f"Detected embeddings: {processed}")
        print(f"Valid distance <= {MAX_DISTANCE}: {valid}")
        print(f"Over limit > {MAX_DISTANCE}: {over_limit}")
        print(f"Old distance NULL: {old_null}")
        print(f"Old distance changed: {changed}")
        print(f"Old distance unchanged: {unchanged}")
        print(f"No reference embeddings: {no_references}")
        print("Database changed: False")
        print("=" * 100)

    finally:
        db.close()


if __name__ == "__main__":
    main()
