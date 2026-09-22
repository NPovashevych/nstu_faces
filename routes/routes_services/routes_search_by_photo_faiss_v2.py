import io
import logging
import cv2
import numpy as np
from PIL import Image

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from insightface.app import FaceAnalysis
from sqlalchemy.orm import Session

from db.session import get_db
from db.models import DBPerson

from services.faiss.faiss_face_index import REFERENCE_FACE_INDEX, UNKNOWN_FACE_INDEX, ensure_faiss_indexes
from commons.common_model import get_insightface
from commons.common_search import normalize_person, normalize_embedding


router = APIRouter(prefix="/search-photo", tags=["search by photo"])


FACE_DET_SIZE = 640
DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.055
UNKNOWN_TOLERANCE = 0.55
MAX_UPLOAD_SIZE = 15 * 1024 * 1024


_INSIGHTFACE_CACHE = None


def load_insightface():
    logging.info("Loading InsightFace buffalo_l...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))
    logging.info("InsightFace buffalo_l loaded.")

    return app


def get_insightface():
    global _INSIGHTFACE_CACHE

    if _INSIGHTFACE_CACHE is None:
        _INSIGHTFACE_CACHE = load_insightface()

    return _INSIGHTFACE_CACHE


def read_upload_file(file: UploadFile) -> bytes:
    file_bytes = file.file.read(MAX_UPLOAD_SIZE + 1)

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(file_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 15 MB.")

    return file_bytes


def read_image(file_bytes: bytes) -> np.ndarray:
    try:
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    np_image = np.asarray(pil_image)

    return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)


def get_confidence(distance: float) -> int:
    if distance <= DIST_TOLERANCE:
        return 0

    for confidence in range(1, 4):
        threshold = DIST_TOLERANCE + confidence * STEP_TOLERANCE

        if distance <= threshold:
            return confidence

    return -1


def find_known_person(db: Session, embedding: np.ndarray):
    best_ref, distance = REFERENCE_FACE_INDEX.find_best_match(embedding)

    if best_ref is None:
        return None

    confidence = get_confidence(distance)

    if confidence == -1:
        return None

    person = db.query(DBPerson).filter(DBPerson.id == best_ref["person_id"]).first()

    if not person:
        return None

    return {
        "person": person,
        "distance": round(distance, 4),
        "confidence": confidence,
    }


def find_unknown_person(db: Session, embedding: np.ndarray):
    person_id, distance = UNKNOWN_FACE_INDEX.find_best_match(embedding)

    if person_id is None or distance is None:
        return None

    if distance > UNKNOWN_TOLERANCE:
        return None

    person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if not person:
        return None

    return {
        "person": person,
        "distance": round(distance, 4),
    }


@router.post("/person")
def search_person_by_photo(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_bytes = read_upload_file(file)
    image = read_image(file_bytes)

    face_model = get_insightface()
    faces = face_model.get(image)

    if len(faces) == 0:
        return {
            "mode": "no_faces",
            "message": "На фото не знайдено облич.",
            "result": None,
        }

    if len(faces) > 1:
        return {
            "mode": "multiple_faces",
            "message": "На фото знайдено кілька облич. Скористайтесь сервісом аналіз медіа.",
            "faces_count": len(faces),
            "redirect_to": "detect_photo",
            "result": None,
        }

    face = faces[0]
    embedding = normalize_embedding(face.embedding)

    ensure_faiss_indexes(db)

    known_match = find_known_person(db=db, embedding=embedding)

    if known_match:
        person = known_match["person"]

        return {
            "mode": "person_found",
            "message": "Персону знайдено в архіві.",
            "match": {
                "type": "known",
                "distance": known_match["distance"],
                "confidence": known_match["confidence"],
            },
            "result": {
                "person": normalize_person(person),
            },
        }

    unknown_match = find_unknown_person(db=db, embedding=embedding)

    if unknown_match:
        person = unknown_match["person"]

        return {
            "mode": "person_found",
            "message": "Персону знайдено в архіві, але вона поки неідентифікована.",
            "match": {
                "type": "unknown",
                "distance": unknown_match["distance"],
            },
            "result": {
                "person": normalize_person(person),
            },
        }

    return {
        "mode": "person_not_found",
        "message": "Персону не знайдено в архіві.",
        "match": None,
        "result": None,
    }
