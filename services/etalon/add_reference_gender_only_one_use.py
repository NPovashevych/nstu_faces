import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from db.session import SessionLocal
from db.enums import EmbeddingType
from db.models import DBEmbedding


os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

Path("../logs").mkdir(exist_ok=True)

logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)8s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../logs/fill_reference_gender.log", encoding="utf-8"),
    ],
)


FACE_DET_SIZE = 640
_INSIGHTFACE_CACHE = None


def load_insightface():
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))
    return app


def get_insightface():
    global _INSIGHTFACE_CACHE

    if _INSIGHTFACE_CACHE is None:
        _INSIGHTFACE_CACHE = load_insightface()

    return _INSIGHTFACE_CACHE


def get_gender_from_photo(model, image_path: Path):
    if not image_path.exists():
        logging.warning(f"Файл не знайдено: {image_path}")
        return "unknown"

    try:
        cv_img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)

    except Exception as e:
        logging.warning(f"Помилка читання {image_path}: {e}")
        return "unknown"

    if cv_img is None:
        logging.warning(f"OpenCV не зміг прочитати: {image_path}")
        return "unknown"

    faces = model.get(cv_img)

    if not faces:
        logging.warning(f"Обличчя не знайдено: {image_path}")
        return "unknown"

    # найбільше обличчя.
    face = max(faces, key=lambda f: ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
    gender = getattr(face, "gender", None)

    if gender == 1:
        return "male"

    if gender == 0:
        return "female"

    return "unknown"


def get_gender_consensus(genders):
    valid = [gender for gender in genders if gender in {"male", "female"}]

    if not valid:
        return "unknown"

    male_count = valid.count("male")
    female_count = valid.count("female")

    if male_count > female_count:
        return "male"

    if female_count > male_count:
        return "female"

    return "unknown"


def fill_reference_gender():
    db = SessionLocal()

    try:
        logging.info("Завантаження Buffalo...")
        model = get_insightface()

        embeddings = (
            db.query(DBEmbedding)
            .filter(DBEmbedding.embedding_type == EmbeddingType.reference_face)
            .order_by(DBEmbedding.person_id, DBEmbedding.id)
            .all()
        )

        logging.info(f"Reference embeddings: {len(embeddings)}")

        embeddings_by_person = defaultdict(list)

        for embedding in embeddings:
            embeddings_by_person[embedding.person_id].append(embedding)

        logging.info(f"Персон з еталонами: {len(embeddings_by_person)}")

        total_updated = persons_male = persons_female = persons_unknown = 0

        for number, (person_id, person_embeddings) in enumerate(embeddings_by_person.items(), start=1):
            detected_genders = []

            for embedding in person_embeddings:
                source = embedding.source or {}
                file_path = source.get("file_path")

                if not file_path:
                    logging.warning(f"embedding_id={embedding.id}: немає file_path")
                    detected_genders.append("unknown")
                    continue

                gender = get_gender_from_photo(model, Path(file_path))
                detected_genders.append(gender)

            person_gender = get_gender_consensus(detected_genders)

            if person_gender == "male":
                persons_male += 1

            elif person_gender == "female":
                persons_female += 1

            else:
                persons_unknown += 1

            # В усі embeddings записуємо однаковий consensus gender
            for embedding in person_embeddings:
                source = dict(embedding.source or {})
                source["gender"] = (person_gender)
                embedding.source = source
                total_updated += 1

            db.commit()

            logging.info(f"[{number}/{len(embeddings_by_person)}] person_id={person_id} | gender={person_gender} | "
                f"photos={len(person_embeddings)} | votes={detected_genders}")

        logging.info("")
        logging.info("========================================")
        logging.info("REFERENCE GENDER COMPLETED")
        logging.info("========================================")
        logging.info(f"Embeddings updated: {total_updated}")
        logging.info(f"Persons male:       {persons_male}")
        logging.info(f"Persons female:     {persons_female}")
        logging.info(f"Persons unknown:    {persons_unknown}")
        logging.info("========================================")

    except Exception:
        db.rollback()

        logging.exception("Помилка заповнення gender.")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    start = datetime.now()

    logging.info(f"Старт: {start.isoformat()}")
    fill_reference_gender()
    finish = datetime.now()
    logging.info(f"Закінчено. Час роботи: {finish - start}")
