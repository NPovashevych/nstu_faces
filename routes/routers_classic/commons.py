import numpy as np
from pathlib import Path
from io import BytesIO
import base64

from PIL import Image
import cv2
from sqlalchemy.orm import Session

from services.config import (
    TEST_MP4_LIGHT_FOLDER,
    TEST_FREEZE_FOLDER,
    USER_UPLOAD_FOLDER,
    INTVNEWS_FREEZE_FOLDER,
    PROXY_NEWS_FOLDER,
)
from db.enums import EmbeddingType
from db.models import DBPerson, DBEmbedding


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def cosine_distance(a, b):
    return 1 - float(np.dot(a, b))


def similarity_percent_from_distance(distance: float):
    return round((1 - distance) * 100, 2)


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

    return None


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


def image_to_base64(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")

    return "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode()


def load_reference_embeddings(db: Session):
    rows = (
        db.query(DBEmbedding)
        .join(DBPerson, DBEmbedding.person_id == DBPerson.id)
        .filter(
            DBEmbedding.embedding_type == EmbeddingType.reference_face
        )
        .all()
    )

    refs = []

    for row in rows:
        refs.append({
            "person_id": row.person_id,
            "person_name": row.person.name,
            "person_code": row.person.code,
            "person_status": (
                row.person.status.value
                if row.person.status
                else None
            ),
            "q_code": row.person.q_code,
            "link": row.person.link,
            "vector": normalize(
                np.array(row.vector, dtype=np.float32)
            ),
        })

    return refs
