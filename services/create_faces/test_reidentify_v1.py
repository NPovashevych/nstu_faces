import logging
import sys
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.enums import EmbeddingType
from db.models import DBEmbedding, DBFace, DBFaceCategory, DBFreeze, DBPerson

from services.create_faces.faiss_face_index import ReferenceFaceIndex, normalize_vector


START_FACE_ID = 1
END_FACE_ID = 20000

CATEGORY_IDS = [1, 2, 3, 7]

MAX_MATCH_DISTANCE = 0.45


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_test_faces(db: Session):

    reference_person_ids = select(DBEmbedding.person_id).where(
        DBEmbedding.embedding_type == EmbeddingType.reference_face
    )

    rows = (
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
            DBFace.id >= START_FACE_ID,
            DBFace.id <= END_FACE_ID,
            DBFace.category_id.in_(CATEGORY_IDS),
            DBEmbedding.embedding_type == EmbeddingType.detected_face,
            ~DBFace.person_id.in_(reference_person_ids),
        )
        .order_by(DBFace.id)
        .all()
    )

    return rows


def main():

    db = SessionLocal()

    try:
        faiss_time = datetime.now()
        logging.info("Building reference FAISS...")

        reference_index = ReferenceFaceIndex()
        reference_index.build(db)

        logging.info(f"Reference FAISS ready: {reference_index.size} embeddings")
        logging.info(f"Face ID range: {START_FACE_ID}–{END_FACE_ID}")
        logging.info(f"Categories: {CATEGORY_IDS}")
        logging.info(f"Maximum distance: {MAX_MATCH_DISTANCE}")

        search_time = datetime.now()
        logging.info(f"Faiss time: {search_time - faiss_time}")

        faces = get_test_faces(db)

        logging.info(f"Candidates found: {len(faces)}")
        logging.info("")

        checked = 0
        matched = 0
        person_name_cache = {}

        for row in faces:
            checked += 1

            emb = normalize_vector(row.vector)
            best_ref, best_dist = reference_index.find_best_match(emb)

            if best_ref is None or best_dist is None or best_dist > MAX_MATCH_DISTANCE:
                continue

            person_id = best_ref["person_id"]

            if person_id not in person_name_cache:
                person = db.get(DBPerson, person_id)
                person_name_cache[person_id] = person.name if person else f"person_id={person_id}"

            matched += 1

            logging.info(
                f"MATCH | face_id={row.face_id} | category={row.category_name} | new_person={person_name_cache[person_id]} | "
                f"freeze_path={row.freeze_path} | distance={best_dist:.4f} | gender={row.gender.value}"
            )

        logging.info("")
        logging.info("================ RESULT ================")
        logging.info(f"Face ID range: {START_FACE_ID}–{END_FACE_ID}")
        logging.info(f"Candidates:    {len(faces)}")
        logging.info(f"Checked:       {checked}")
        logging.info(f"Matches:       {matched}")
        logging.info("========================================")
        end_time = datetime.now()
        logging.info(f"All time: {end_time - faiss_time}")


    finally:
        db.close()


if __name__ == "__main__":
    main()
