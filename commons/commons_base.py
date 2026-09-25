import base64
import re
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from db.enums import EmbeddingType
from db.models import DBEmbedding, DBPerson
from services.config import INTVNEWS_FREEZE_FOLDER, PROXY_NEWS_FOLDER, TEMPORARY_FREEZES_FOLDER, TEST_FREEZE_FOLDER, TEST_MP4_LIGHT_FOLDER, USER_UPLOAD_FOLDER

# Нормалізація вектора
def normalize_vector(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


# Косинусна схожість двох векторів
def cosine_similarity(a, b) -> float:
    a = normalize_vector(a)
    b = normalize_vector(b)

    return float(np.dot(a, b))


# Косинусна відстань між двома векторами
def cosine_distance(a, b) -> float:
    return 1 - cosine_similarity(a, b)


# Перетворення відстані у відсоток схожості
def similarity_percent_from_distance(distance: float) -> float:
    return round((1 - distance) * 100, 2)


# Отримання імені персони та Q-коду з назви папки
def parse_person_folder_name(folder_name: str):
    folder_name = folder_name.strip()
    match = re.search(r"\((Q\d+)\)\s*$", folder_name)

    if match:
        q_code = match.group(1)
        name = folder_name[:match.start()].strip()

        return name, q_code

    return folder_name, None


# Формування унікального коду персони
def make_person_code(name: str, q_code: str | None):
    if q_code:
        return q_code.lower()

    code = name.lower().strip()
    code = re.sub(r"\s+", "_", code)
    code = re.sub(r"[^\wа-яА-ЯіїєґІЇЄҐ_()-]", "", code)

    return code


# Формування URL для freeze або фото
def make_image_url(freeze_path: str | None):
    if not freeze_path:
        return None

    path = Path(freeze_path)

    # Тестові freeze
    try:
        relative_path = path.relative_to(Path(TEST_FREEZE_FOLDER))
        return f"/freezes-test/{relative_path.as_posix()}"
    except ValueError:
        pass

    # INTVNEWS freeze
    try:
        relative_path = path.relative_to(Path(INTVNEWS_FREEZE_FOLDER))
        return f"/freezes-news/{relative_path.as_posix()}"
    except ValueError:
        pass

    # Фото, завантажені користувачем
    try:
        relative_path = path.relative_to(Path(USER_UPLOAD_FOLDER))
        return f"/media-user-upload/{relative_path.as_posix()}"
    except ValueError:
        pass

    # Фото з YouTube та тимчасові файли
    try:
        relative_path = path.relative_to(Path(TEMPORARY_FREEZES_FOLDER))
        return f"/media-youtube-upload/{relative_path.as_posix()}"
    except ValueError:
        pass

    return None


# Формування URL для відеофайлу
def make_media_url(media_path: str | None):
    if not media_path:
        return None

    path = Path(media_path)

    # Тестові MP4
    try:
        relative_path = path.relative_to(Path(TEST_MP4_LIGHT_FOLDER))
        return f"/media-files-test/{relative_path.as_posix()}"
    except ValueError:
        pass

    # INTVNEWS proxy
    try:
        relative_path = path.relative_to(Path(PROXY_NEWS_FOLDER))
        return f"/media-files-news/{relative_path.as_posix()}"
    except ValueError:
        pass

    return None


# Перетворення OpenCV-зображення у Base64
def image_to_base64(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")

    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


# Завантаження еталонних ембедінгів з БД
def load_reference_embeddings(db: Session):
    rows = db.query(DBEmbedding).join(DBPerson, DBEmbedding.person_id == DBPerson.id).filter(DBEmbedding.embedding_type == EmbeddingType.reference_face).all()

    refs = []

    for row in rows:
        refs.append({
            "person_id": row.person_id,
            "person_name": row.person.name,
            "person_code": row.person.code,
            "person_status": row.person.status.value if row.person.status else None,
            "q_code": row.person.q_code,
            "link": row.person.link,
            "vector": normalize_vector(row.vector),
        })

    return refs
