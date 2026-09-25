import faiss
import numpy as np

from db.enums import EmbeddingType, PersonStatus
from db.models import DBEmbedding, DBPerson
from db.session import SessionLocal
from services.faiss.faiss_face_index import normalize_vector


BATCH_SIZE = 5000
ALLOWED_STATUSES = [PersonStatus.public, PersonStatus.non_public]


def build_reference_index(db):
    rows = db.query(DBEmbedding.id, DBEmbedding.person_id, DBEmbedding.vector).join(DBPerson, DBEmbedding.person_id == DBPerson.id).filter(
        DBEmbedding.embedding_type == EmbeddingType.reference_face,
        DBEmbedding.vector.isnot(None),
        DBPerson.status.in_(ALLOWED_STATUSES),
    ).order_by(DBEmbedding.id).all()

    if not rows:
        raise RuntimeError("Reference embeddings not found")

    vectors = []
    metadata = []

    for row in rows:
        vectors.append(normalize_vector(row.vector))
        metadata.append({"embedding_id": row.id, "person_id": row.person_id})

    matrix = np.vstack(vectors).astype(np.float32)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    return index, metadata


def main():
    db = SessionLocal()

    try:
        print("=" * 80)
        print("TEST CURRENT INSIGHTFACE DISTANCES")
        print("=" * 80)

        index, metadata = build_reference_index(db)

        print(f"Reference embeddings: {index.ntotal}")

        query = db.query(DBEmbedding.id, DBEmbedding.person_id, DBEmbedding.vector, DBEmbedding.source).join(
            DBPerson, DBEmbedding.person_id == DBPerson.id
        ).filter(
            DBEmbedding.embedding_type == EmbeddingType.detected_face,
            DBEmbedding.vector.isnot(None),
            DBPerson.status.in_(ALLOWED_STATUSES),
        )

        total = query.count()

        print(f"Detected embeddings: {total}")
        print("=" * 80)

        processed = 0
        same_person = 0
        different_person = 0
        last_id = 0

        while True:
            rows = query.filter(DBEmbedding.id > last_id).order_by(DBEmbedding.id).limit(BATCH_SIZE).all()

            if not rows:
                break

            vectors = np.vstack([normalize_vector(row.vector) for row in rows]).astype(np.float32)
            similarities, positions = index.search(vectors, 1)

            for i, row in enumerate(rows):
                position = int(positions[i][0])

                if position < 0:
                    continue

                best = metadata[position]
                best_person_id = best["person_id"]
                reference_embedding_id = best["embedding_id"]
                distance = float(1.0 - similarities[i][0])
                old_distance = (row.source or {}).get("distance")

                if row.person_id == best_person_id:
                    same_person += 1
                else:
                    different_person += 1
                    print(
                        f"DIFFERENT | embedding={row.id} | current={row.person_id} | best={best_person_id} | "
                        f"reference={reference_embedding_id} | old={old_distance} | new={distance:.4f}"
                    )

            processed += len(rows)
            last_id = rows[-1].id

            print(f"Processed {processed}/{total} | same={same_person} | different={different_person}")

        print()
        print("=" * 80)
        print("RESULT")
        print("=" * 80)
        print(f"Reference embeddings: {index.ntotal}")
        print(f"Detected embeddings: {processed}")
        print(f"Same person: {same_person}")
        print(f"Different person: {different_person}")

        if processed:
            print(f"Different: {different_person / processed * 100:.2f}%")

        print("Database changed: False")
        print("=" * 80)

    finally:
        db.close()


if __name__ == "__main__":
    main()
    