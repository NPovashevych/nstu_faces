from pathlib import Path
from io import BytesIO
import base64

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session

from services.config import FREEZE_FOLDER
from db.session import get_db
from db.enums import EmbeddingType
from db.models import DBPerson, DBFace, DBFreeze, DBMedia, DBEmbedding

from services.face_quality import is_good_face


router = APIRouter(prefix="/search", tags=["search"])


DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.03
MAX_ACCEPTABLE_DIST = 0.55
MIN_PHOTO_FACE_QUALITY = 0.15


_model = None


def get_model():
    global _model

    if _model is None:
        from insightface.app import FaceAnalysis

        _model = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _model.prepare(ctx_id=0, det_size=(640, 640))

    return _model


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def cosine_distance(a, b):
    return 1 - float(np.dot(a, b))


def get_confidence(dist: float):
    if dist <= DIST_TOLERANCE:
        return 0

    for i in range(1, 4):
        if dist <= DIST_TOLERANCE + i * STEP_TOLERANCE:
            return i

    return None


def make_image_url(freeze_path: str):
    relative_path = Path(freeze_path).relative_to(Path(FREEZE_FOLDER))
    return f"/freezes/{relative_path.as_posix()}"


def make_bbox_draw(bbox):
    return {
        "x": bbox[0],
        "y": bbox[1],
        "w": bbox[2] - bbox[0],
        "h": bbox[3] - bbox[1],
    }


def image_to_base64(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")

    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def load_reference_embeddings(db: Session):
    rows = (
        db.query(DBEmbedding)
        .join(DBPerson, DBEmbedding.person_id == DBPerson.id)
        .filter(DBEmbedding.embedding_type == EmbeddingType.reference_face)
        .all()
    )

    refs = []

    for row in rows:
        refs.append({
            "person_id": row.person_id,
            "person_name": row.person.name,
            "q_code": row.person.q_code,
            "link": row.person.link,
            "vector": normalize(np.array(row.vector, dtype=np.float32)),
        })

    return refs


def find_best_match(embedding, reference_embeddings):
    best = None
    best_dist = 1.0

    for ref in reference_embeddings:
        dist = cosine_distance(embedding, ref["vector"])

        if dist < best_dist:
            best_dist = dist
            best = ref

    if best is None:
        return None

    if best_dist > MAX_ACCEPTABLE_DIST:
        return {
            "recognized": False,
            "distance": round(best_dist, 4),
            "similarity_percent": round((1 - best_dist) * 100, 2),
        }

    return {
        "recognized": True,
        "person_id": best["person_id"],
        "name": best["person_name"],
        "q_code": best["q_code"],
        "link": best["link"],
        "distance": round(best_dist, 4),
        "similarity_percent": round((1 - best_dist) * 100, 2),
        "confidence": get_confidence(best_dist),
    }


def get_media_description(media: DBMedia):
    if not media.descriptions:
        return None

    d = media.descriptions[0]

    return {
        "section": d.section,
        "description": d.description,
        "date": d.date,
        "duration": d.duration,
        "journalist": d.journalist,
    }


def get_person_media_result(db: Session, person: DBPerson):
    faces = (
        db.query(DBFace)
        .join(DBFreeze, DBFace.freeze_id == DBFreeze.id)
        .join(DBMedia, DBFreeze.media_id == DBMedia.id)
        .filter(DBFace.person_id == person.id)
        .order_by(DBMedia.id, DBFreeze.time_in)
        .all()
    )

    medias_map = {}

    for face in faces:
        freeze = face.freeze
        media = freeze.media

        if media.id not in medias_map:
            medias_map[media.id] = {
                "media_id": media.id,
                "media_name": Path(media.media_path).name,
                "uploaded_by_user_id": media.user_id,
                "description": get_media_description(media),
                "frames": [],
            }

        medias_map[media.id]["frames"].append({
            "face_id": face.id,
            "freeze_id": freeze.id,
            "image_url": make_image_url(freeze.freeze_path),
            "time_in": freeze.time_in,
            "time_out": freeze.time_out,
            "bbox": face.bbox,
            "bbox_draw": make_bbox_draw(face.bbox),
            "quality": face.quality,
            "confidence": face.confidence,
        })

    return {
        "person_id": person.id,
        "name": person.name,
        "q_code": person.q_code,
        "link": person.link,
        "medias": list(medias_map.values()),
    }


@router.get("/person")
def search_person(name: str, db: Session = Depends(get_db)):
    persons = (
        db.query(DBPerson)
        .filter(DBPerson.name.ilike(f"%{name}%"))
        .order_by(DBPerson.name)
        .all()
    )

    if not persons:
        return {
            "query": name,
            "results": [],
        }

    return {
        "query": name,
        "results": [
            get_person_media_result(db, person)
            for person in persons
        ],
    }


@router.get("/person/{person_id}/media")
def search_person_by_id(person_id: int, db: Session = Depends(get_db)):
    person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    return get_person_media_result(db, person)


@router.post("/photo")
async def search_by_photo(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    contents = await file.read()

    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image")

    h, w = img.shape[:2]

    model = get_model()
    reference_embeddings = load_reference_embeddings(db)

    faces = model.get(img)
    faces = sorted(faces, key=lambda f: f.bbox[0])

    detected_faces = []

    for idx, face in enumerate(faces, start=1):
        quality = is_good_face(img, face)

        if quality < MIN_PHOTO_FACE_QUALITY:
            continue

        bbox = face.bbox.astype(float).tolist()
        emb = normalize(face.embedding)

        match = find_best_match(emb, reference_embeddings)

        detected_faces.append({
            "face_index": idx,
            "bbox": bbox,
            "bbox_draw": make_bbox_draw(bbox),
            "quality": round(quality, 4),
            "match": match,
        })

    if not detected_faces:
        return {
            "mode": "no_faces",
            "message": "На фото не знайдено придатних облич",
            "image_width": w,
            "image_height": h,
            "uploaded_image": image_to_base64(img),
            "faces": [],
        }

    recognized_faces = [
        f for f in detected_faces
        if f["match"] and f["match"].get("recognized")
    ]

    if len(detected_faces) == 1:
        one_face = detected_faces[0]
        match = one_face["match"]

        if not match or not match.get("recognized"):
            return {
                "mode": "single_unknown",
                "message": "Обличчя знайдено, але персонy не розпізнано",
                "image_width": w,
                "image_height": h,
                "uploaded_image": image_to_base64(img),
                "faces": detected_faces,
            }

        person = db.query(DBPerson).filter(DBPerson.id == match["person_id"]).first()

        return {
            "mode": "single_result",
            "image_width": w,
            "image_height": h,
            "uploaded_image": image_to_base64(img),
            "selected_face": one_face,
            "result": get_person_media_result(db, person),
        }

    return {
        "mode": "multiple_faces",
        "message": "На фото знайдено кілька облич. Оберіть потрібну людину.",
        "image_width": w,
        "image_height": h,
        "uploaded_image": image_to_base64(img),
        "faces": detected_faces,
        "recognized_count": len(recognized_faces),
    }
