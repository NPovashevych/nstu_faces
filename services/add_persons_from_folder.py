import os
import re
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sqlalchemy.orm import Session

from config import PERSONS_FOLDER, NEW_WIKI_PATH

from db.session import SessionLocal
from db.enums import PersonStatus, EmbeddingType
from db.models import DBEmbedding

from crud.crud_person import (
    get_person_by_code,
    get_person_by_qcode,
    create_person,
)
from crud.crud_embedding import create_embedding

from schemas.schemas_person import PersonsCreate
from schemas.schemas_embedding import EmbeddingCreate


os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

Path("logs").mkdir(exist_ok=True)

logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)8s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/add_persons_from_folder.log", encoding="utf-8"),
    ],
)


MIN_W = 15
MIN_H = 15
BLUR = 14.9
DUPLICATE_SIMILARITY = 0.98

MAX_DIST_FROM_MEAN = 0.40
MAX_PAIRWISE_DIST = 0.70


def load_people_db(path: Path) -> dict:
    if not path.exists():
        logging.warning(f"Файл people_db не знайдено: {path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    model.prepare(ctx_id=0)
    return model


def parse_name_qcode(folder_name: str):
    folder_name = folder_name.strip()

    match = re.search(r"\((Q\d+)\)\s*$", folder_name)
    if match:
        q_code = match.group(1)
        name = folder_name[:match.start()].strip()
        return name, q_code

    return folder_name, None


def make_person_code(name: str, q_code: str | None):
    if q_code:
        return q_code.lower()

    code = name.lower().strip()
    code = re.sub(r"\s+", "_", code)
    code = re.sub(r"[^\wа-яА-ЯіїєґІЇЄҐ_()-]", "", code)
    return code


def is_unknown_name(name: str) -> bool:
    return name.strip().lower().startswith("unknown")


def get_person_status(name: str, q_code: str | None):
    if q_code:
        return PersonStatus.public

    if is_unknown_name(name):
        return PersonStatus.unknown

    return PersonStatus.non_public


def get_google_search_link(name: str):
    return f"https://www.google.com/search?q={quote_plus(name)}"


def get_person_link(name: str, q_code: str | None, people_db: dict):
    if q_code:
        person_data = people_db.get(q_code)
        if person_data:
            return person_data.get("link")

        return None

    if is_unknown_name(name):
        return None

    return get_google_search_link(name)


def get_person_name_from_db_or_folder(folder_name: str, q_code: str | None, people_db: dict):
    parsed_name, _ = parse_name_qcode(folder_name)

    if q_code and q_code in people_db:
        return people_db[q_code].get("name") or parsed_name

    return parsed_name


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def normalize_embedding(embedding):
    return embedding / (np.linalg.norm(embedding) + 1e-8)


def get_embedding(model, image_path: Path):
    try:
        cv_img = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if cv_img is None:
            logging.info(f"Фото {image_path} не зміг прочитати OpenCV.")
            return None

        height, width = cv_img.shape[:2]

        if width < MIN_W or height < MIN_H:
            logging.info(f"Фото {image_path} замале ({width}x{height}).")
            return None

    except Exception as e:
        logging.info(f"Фото {image_path}: помилка читання — {e}")
        return None

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if lap_var < BLUR:
        logging.info(f"Фото {image_path} занадто розмите (blur={lap_var:.1f})")
        return None

    faces = model.get(cv_img)

    if not faces:
        logging.info(f"На фото {image_path} відсутнє обличчя")
        return None

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )

    return face.embedding


def get_existing_reference_embeddings(db: Session, person_id: int):
    return (
        db.query(DBEmbedding)
        .filter(DBEmbedding.person_id == person_id)
        .filter(DBEmbedding.embedding_type == EmbeddingType.reference_face)
        .all()
    )


def get_embedding_file_name(db_embedding: DBEmbedding):
    if not db_embedding.source:
        return None

    file_name = db_embedding.source.get("file_name")

    if file_name:
        return file_name

    file_path = db_embedding.source.get("file_path")

    if file_path:
        return Path(file_path).name

    return None


def sync_face_database(db: Session, person_folder: Path, db_person):
    existing_files = {
        photo.name
        for photo in person_folder.iterdir()
        if photo.is_file()
    }

    db_embeddings = get_existing_reference_embeddings(db, db_person.id)

    kept_embeddings = []

    for db_embedding in db_embeddings:
        file_name = get_embedding_file_name(db_embedding)

        if file_name in existing_files:
            kept_embeddings.append(db_embedding)
        else:
            logging.info(
                f"{person_folder.name}/{file_name} — embedding видалено з БД, бо файл відсутній."
            )
            db.delete(db_embedding)

    db.commit()

    return kept_embeddings


def is_duplicate_embedding(new_embedding, existing_embeddings):
    for db_embedding in existing_embeddings:
        old_vector = np.array(db_embedding.vector, dtype=np.float32)
        sim = cosine_similarity(new_embedding, old_vector)

        if sim > DUPLICATE_SIMILARITY:
            return True

    return False


def is_one_person(encodings, file_names):
    if len(encodings) < 2:
        return True, []

    encodings = [normalize_embedding(enc) for enc in encodings]

    mean_enc = np.mean(encodings, axis=0)
    mean_enc = normalize_embedding(mean_enc)

    false_photos = []

    for file_name, enc in zip(file_names, encodings):
        dist = 1 - cosine_similarity(enc, mean_enc)

        if dist > MAX_DIST_FROM_MEAN:
            false_photos.append((file_name, float(dist)))

    for i in range(len(encodings)):
        for j in range(i + 1, len(encodings)):
            pair_dist = 1 - cosine_similarity(encodings[i], encodings[j])

            if pair_dist > MAX_PAIRWISE_DIST:
                false_photos.append(
                    (
                        f"{file_names[i]} <-> {file_names[j]}",
                        float(pair_dist),
                    )
                )

    return len(false_photos) == 0, false_photos


def get_or_create_person(db: Session, person_folder: Path, people_db: dict):
    folder_name = person_folder.name

    parsed_name, q_code = parse_name_qcode(folder_name)

    person_name = get_person_name_from_db_or_folder(folder_name, q_code, people_db)
    code = make_person_code(person_name, q_code)

    db_person = None

    if q_code:
        db_person = get_person_by_qcode(db, q_code)

    if db_person is None:
        db_person = get_person_by_code(db, code)

    if db_person:
        return db_person

    person_create = PersonsCreate(
        name=person_name,
        q_code=q_code,
        link=get_person_link(person_name, q_code, people_db),
        status=get_person_status(person_name, q_code),
    )

    return create_person(db, person_create, code=code)


def process_person_folder(db: Session, model, person_folder: Path, people_db: dict):
    if not person_folder.is_dir():
        return 0, 0

    db_person = get_or_create_person(db, person_folder, people_db)

    existing_embeddings = sync_face_database(db, person_folder, db_person)

    known_files = {
        get_embedding_file_name(embedding)
        for embedding in existing_embeddings
    }

    new_encodings = []
    new_files = []
    skipped = []

    for photo in sorted(person_folder.iterdir()):
        if not photo.is_file():
            continue

        if photo.name in known_files:
            skipped.append((photo.name, "вже є в БД"))
            continue

        emb = get_embedding(model, photo)

        if emb is None:
            skipped.append((photo.name, "нема embedding"))
            continue

        if is_duplicate_embedding(emb, existing_embeddings):
            skipped.append((photo.name, "дубль"))
            continue

        new_encodings.append(emb)
        new_files.append(photo.name)

    is_ok, false_photos = is_one_person(new_encodings, new_files)

    if not is_ok:
        logging.warning(f"Папка {person_folder.name} потребує перевірки.")
        for file_name, dist in false_photos:
            logging.warning(f"Сумнівне фото: {file_name} (distance={dist:.2f})")
        return 0, len(skipped) + len(new_files)

    added_count = 0

    for emb, file_name in zip(new_encodings, new_files):
        photo_path = person_folder / file_name

        embedding_create = EmbeddingCreate(
            embedding_type=EmbeddingType.reference_face,
            source={
                "file_name": file_name,
                "file_path": str(photo_path),
                "person_folder": person_folder.name,
                "created_by": "add_persons_from_folder.py",
            },
            vector=emb.tolist(),
            person_id=db_person.id,
        )

        create_embedding(db, embedding_create)
        added_count += 1

    if added_count > 0:
        logging.info("-----------------------------------------------")
        logging.info(f"{person_folder.name}: +{added_count} / -{len(skipped)}")
        logging.info("-----------------------------------------------")

    return added_count, len(skipped)


def build_face_database_from_folder(base_folder: Path):
    people_db = load_people_db(Path(NEW_WIKI_PATH))

    model = load_model()
    db = SessionLocal()

    total_added = 0
    total_skipped = 0

    try:
        for person_folder in sorted(base_folder.iterdir()):
            added, skipped = process_person_folder(db, model, person_folder, people_db)
            total_added += added
            total_skipped += skipped

        logging.info(f"Всього додано: {total_added}, всього не додано: {total_skipped}.")

    finally:
        db.close()


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Старт: {start.isoformat()}")

    try:
        build_face_database_from_folder(Path(PERSONS_FOLDER))
    except Exception as e:
        logging.exception(f"Критична помилка при побудові бази: {e}")

    finish = datetime.now()
    logging.info(f"Закінчено. Час роботи: {finish - start}")
