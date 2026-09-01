from pathlib import Path
import json
import threading
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from routes.routers_classic.commons import (
    cosine_distance,
    normalize,
    make_image_url,
    image_to_base64,
    load_reference_embeddings,
    make_media_url,
)
from db.session import get_db
from db.models import DBPerson, DBFace, DBFreeze, DBMedia, DBEmbedding
from db.enums import PersonStatus, EmbeddingType, FaceCategory

from services.create_faces.face_quality_v3 import get_face_quality
from services.create_faces.clip_face_filter_v2 import get_clip, analyze_face_category


router = APIRouter(prefix="/search", tags=["search"])


FACE_DET_SIZE = 640

DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.05
MAX_ACCEPTABLE_DIST = 0.70

UNKNOWN_CLUSTER_TOLERANCE = 0.50
LOW_QUALITY_THRESHOLD = 0.55

BBOX_DRAW_SCALE = 1.10

CLUSTER_HINTS_PATH = Path("data/cluster_hints.json")
_hints_lock = threading.Lock()

_model = None


class ClusterHintCreate(BaseModel):
    cluster_person_id: int
    cluster_id: int | None = None
    cluster_tag: str | None = None
    suggested_name: str
    comment: str | None = None
    source: str | None = "lovable"


def safe_float(value, default=None):
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_model():
    global _model

    if _model is None:
        from insightface.app import FaceAnalysis

        _model = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _model.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))

    return _model


def to_face_category(value: str) -> FaceCategory:
    try:
        return FaceCategory(value)
    except ValueError:
        return FaceCategory.uncertain


def get_confidence_by_distance(dist: float | None):
    if dist is None:
        return None

    if dist <= DIST_TOLERANCE:
        return 0

    for i in range(1, 4):
        if dist <= DIST_TOLERANCE + i * STEP_TOLERANCE:
            return i

    return None


def normalize_bbox(bbox, scale: float = BBOX_DRAW_SCALE):
    bbox = bbox or [0, 0, 0, 0]

    x1 = safe_float(bbox[0], 0.0)
    y1 = safe_float(bbox[1], 0.0)
    x2 = safe_float(bbox[2], 0.0)
    y2 = safe_float(bbox[3], 0.0)

    width = x2 - x1
    height = y2 - y1

    center_x = x1 + width / 2
    center_y = y1 + height / 2

    draw_width = width * scale
    draw_height = height * scale

    draw_x = center_x - draw_width / 2
    draw_y = center_y - draw_height / 2

    return {
        "raw": [x1, y1, x2, y2],
        "draw": {
            "x": draw_x,
            "y": draw_y,
            "w": draw_width,
            "h": draw_height,
        },
    }


def get_face_color(category: FaceCategory, person_status: str | None) -> str:
    if category in {
        FaceCategory.low_quality,
        FaceCategory.real_unidentifiable,
    }:
        return "gray"

    if category in {
        FaceCategory.non_human,
        FaceCategory.artificial_human,
        FaceCategory.ai_generated,
        FaceCategory.uncertain,
    }:
        return "orange"

    if person_status == PersonStatus.unknown.value:
        return "red"

    if person_status in {
        PersonStatus.public.value,
        PersonStatus.non_public.value,
    }:
        return "green"

    if person_status == PersonStatus.suspicious.value:
        return "orange"

    return "orange"


def make_not_checked_clip(reason: str):
    return {
        "category": reason,
        "category_score": None,
        "best_clip_category": None,
        "best_clip_score": None,
        "clip_scores": None,
    }


def load_cluster_hints() -> dict:
    if not CLUSTER_HINTS_PATH.exists():
        return {}

    with open(CLUSTER_HINTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cluster_hint(payload: ClusterHintCreate):
    CLUSTER_HINTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _hints_lock:
        data = load_cluster_hints()
        key = str(payload.cluster_person_id)

        item = {
            "cluster_person_id": payload.cluster_person_id,
            "cluster_id": payload.cluster_id,
            "cluster_tag": payload.cluster_tag,
            "suggested_name": payload.suggested_name.strip(),
            "comment": payload.comment,
            "source": payload.source,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "pending_review",
        }

        if key not in data:
            data[key] = []

        data[key].append(item)

        with open(CLUSTER_HINTS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return item


def get_latest_cluster_hint(cluster_person_id: int):
    data = load_cluster_hints()
    items = data.get(str(cluster_person_id), [])

    if not items:
        return None

    return items[-1]


def load_unknown_cluster_embeddings(db: Session):
    rows = (
        db.query(DBEmbedding, DBPerson)
        .join(DBPerson, DBEmbedding.person_id == DBPerson.id)
        .filter(DBPerson.status == PersonStatus.unknown)
        .filter(DBEmbedding.embedding_type == EmbeddingType.detected_face)
        .all()
    )

    result = []

    for emb, person in rows:
        result.append(
            {
                "person_id": person.id,
                "person_name": person.name,
                "person_status": person.status.value if person.status else None,
                "cluster_id": person.cluster_id,
                "cluster_tag": person.cluster_tag,
                "cluster_distance": person.cluster_distance,
                "vector": normalize(np.array(emb.vector, dtype=np.float32)),
            }
        )

    return result


def find_best_known_match(embedding, reference_embeddings):
    best = None
    best_dist = 1.0

    for ref in reference_embeddings:
        dist = cosine_distance(embedding, ref["vector"])

        if dist < best_dist:
            best_dist = dist
            best = ref

    if best is None or best_dist > MAX_ACCEPTABLE_DIST:
        return None

    confidence = get_confidence_by_distance(best_dist)

    if confidence is None:
        return None

    return {
        "type": "known",
        "person": {
            "id": best["person_id"],
            "name": best["person_name"],
            "q_code": best["q_code"],
            "link": best["link"],
            "status": best.get("person_status"),
        },
        "distance": round(best_dist, 4),
        "similarity_percent": round((1 - best_dist * 0.5) * 100, 2),
        "confidence": confidence,
    }


def find_best_unknown_cluster(embedding, unknown_cluster_embeddings):
    best = None
    best_dist = 1.0

    for ref in unknown_cluster_embeddings:
        dist = cosine_distance(embedding, ref["vector"])

        if dist < best_dist:
            best_dist = dist
            best = ref

    if best is None or best_dist > UNKNOWN_CLUSTER_TOLERANCE:
        return {
            "type": "unknown",
            "person": {
                "id": None,
                "name": "Невідоме обличчя",
                "q_code": None,
                "link": None,
                "status": PersonStatus.unknown.value,
            },
            "cluster": None,
            "distance": round(best_dist, 4) if best else None,
            "similarity_percent": None,
            "confidence": None,
        }

    hint = get_latest_cluster_hint(best["person_id"])

    display_name = (
        hint["suggested_name"]
        if hint and hint.get("suggested_name")
        else best["cluster_tag"] or best["person_name"]
    )

    return {
        "type": "unknown_cluster",
        "person": {
            "id": best["person_id"],
            "name": display_name,
            "original_name": best["person_name"],
            "q_code": None,
            "link": None,
            "status": PersonStatus.unknown.value,
        },
        "cluster": {
            "id": best["cluster_id"],
            "tag": best["cluster_tag"],
            "distance": safe_float(best["cluster_distance"]),
            "suggested_name": hint["suggested_name"] if hint else None,
            "hint_status": hint["status"] if hint else None,
        },
        "distance": round(best_dist, 4),
        "similarity_percent": round((1 - best_dist * 0.5) * 100, 2),
        "confidence": None,
    }


def get_service_category_match(db: Session, category: FaceCategory):
    cluster_tag = f"{category.value}_cluster"

    person = (
        db.query(DBPerson)
        .filter(DBPerson.code == cluster_tag)
        .first()
    )

    if not person:
        return {
            "type": "service_category",
            "person": {
                "id": None,
                "name": cluster_tag,
                "q_code": None,
                "link": None,
                "status": PersonStatus.suspicious.value,
            },
            "cluster": {
                "id": None,
                "tag": cluster_tag,
                "distance": None,
                "suggested_name": None,
                "hint_status": None,
            },
            "distance": None,
            "similarity_percent": None,
            "confidence": None,
        }

    return {
        "type": "service_category",
        "person": {
            "id": person.id,
            "name": person.cluster_tag or person.name,
            "original_name": person.name,
            "q_code": person.q_code,
            "link": person.link,
            "status": person.status.value if person.status else PersonStatus.suspicious.value,
        },
        "cluster": {
            "id": person.cluster_id,
            "tag": person.cluster_tag,
            "distance": safe_float(person.cluster_distance),
            "suggested_name": None,
            "hint_status": None,
        },
        "distance": None,
        "similarity_percent": None,
        "confidence": None,
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


def normalize_db_face(face: DBFace):
    person_status = (
        face.person.status.value
        if face.person and face.person.status
        else None
    )

    category = (
        face.category
        if face.category
        else FaceCategory.uncertain
    )

    confidence = (
        face.confidence
        if person_status in {
            PersonStatus.public.value,
            PersonStatus.non_public.value,
        }
        else None
    )

    return {
        "face_id": face.id,
        "person": {
            "id": face.person_id,
            "name": face.person.name if face.person else None,
            "code": face.person.code if face.person else None,
            "q_code": face.person.q_code if face.person else None,
            "link": face.person.link if face.person else None,
            "status": person_status,
        },
        "recognition": {
            "confidence": confidence,
        },
        "category": {
            "name": category.value,
            "score": safe_float(face.category_score),
        },
        "quality": safe_float(face.quality),
        "gender": face.gender.value if face.gender else None,
        "color": get_face_color(category, person_status),
        "bbox": normalize_bbox(face.bbox),
        "analysis": face.analysis or {},
    }


def get_person_media_result(db: Session, person: DBPerson):
    faces = (
        db.query(DBFace)
        .options(joinedload(DBFace.person))
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

        media_path = media.mp4_path or media.mxf_path or media.media_path

        if media.id not in medias_map:
            medias_map[media.id] = {
                "media": {
                    "id": media.id,
                    "name": Path(media_path).name if media_path else f"media_{media.id}",
                    "path": media_path,
                    "url": make_media_url(media_path) if media_path else None,
                    "uploaded_by_user_id": media.user_id,
                    "description": get_media_description(media),
                },
                "frames": [],
            }

        medias_map[media.id]["frames"].append(
            {
                "freeze_id": freeze.id,
                "image_url": make_image_url(freeze.freeze_path),
                "time_in": safe_float(freeze.time_in, 0.0),
                "time_out": safe_float(freeze.time_out, 0.0),
                "face": normalize_db_face(face),
            }
        )

    return {
        "person": {
            "id": person.id,
            "name": person.name,
            "code": person.code,
            "q_code": person.q_code,
            "link": person.link,
            "status": person.status.value if person.status else None,
            "cluster_id": person.cluster_id,
            "cluster_tag": person.cluster_tag,
        },
        "summary": {
            "media_count": len(medias_map),
            "faces_count": len(faces),
        },
        "medias": list(medias_map.values()),
    }


def get_cluster_result_if_exists(db: Session, match: dict | None):
    if not match:
        return None

    person_id = (match.get("person") or {}).get("id")

    if not person_id:
        return None

    person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if not person:
        return None

    return get_person_media_result(db, person)


def normalize_uploaded_face(
    face_index: int,
    bbox,
    category: FaceCategory,
    category_score,
    quality,
    quality_details,
    category_result,
    match,
):
    person_status = (
        (match.get("person") or {}).get("status")
        if match
        else PersonStatus.unknown.value
    )

    confidence = (
        match.get("confidence")
        if person_status in {
            PersonStatus.public.value,
            PersonStatus.non_public.value,
        }
        else None
    )

    return {
        "face_index": face_index,
        "person": (match.get("person") if match else None),
        "recognition": {
            "type": match.get("type") if match else "unknown",
            "distance": match.get("distance") if match else None,
            "similarity_percent": match.get("similarity_percent") if match else None,
            "confidence": confidence,
        },
        "cluster": match.get("cluster") if match else None,
        "category": {
            "name": category.value,
            "score": safe_float(category_score),
        },
        "quality": safe_float(quality),
        "bbox": normalize_bbox(bbox),
        "color": get_face_color(category, person_status),
        "analysis": {
            "quality": quality_details,
            "clip": category_result,
        },
    }


@router.get("/person")
def search_person(name: str, db: Session = Depends(get_db)):
    persons = (
        db.query(DBPerson)
        .filter(DBPerson.name.ilike(f"%{name}%"))
        .order_by(DBPerson.name)
        .all()
    )

    return {
        "query": name,
        "summary": {
            "persons_count": len(persons),
        },
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


@router.post("/cluster-hint")
def create_cluster_hint(payload: ClusterHintCreate):
    if not payload.suggested_name.strip():
        raise HTTPException(status_code=400, detail="suggested_name is empty")

    item = save_cluster_hint(payload)

    return {
        "status": "ok",
        "message": "Підказку збережено для перевірки",
        "hint": item,
    }


@router.get("/cluster-hints")
def list_cluster_hints():
    return load_cluster_hints()


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

    try:
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image with PIL")

    h, w = img.shape[:2]

    model = get_model()
    clip_model, clip_preprocess, clip_text_features, clip_prompt_categories = get_clip()

    reference_embeddings = load_reference_embeddings(db)
    unknown_cluster_embeddings = load_unknown_cluster_embeddings(db)

    faces = model.get(img)
    faces = sorted(faces, key=lambda f: f.bbox[0])

    detected_faces = []

    for idx, face in enumerate(faces, start=1):
        bbox = face.bbox.astype(float).tolist()

        quality, quality_details = get_face_quality(img, face)
        emb = normalize(face.embedding)

        match = find_best_known_match(
            embedding=emb,
            reference_embeddings=reference_embeddings,
        )

        if match:
            category = FaceCategory.real_identifiable
            category_score = match["confidence"]
            category_result = make_not_checked_clip("not_checked_known_match")

        elif quality < LOW_QUALITY_THRESHOLD:
            category = FaceCategory.low_quality
            category_score = quality
            category_result = make_not_checked_clip("not_checked_low_quality")
            match = get_service_category_match(db, category)

        else:
            category_result = analyze_face_category(
                image=pil_image,
                bbox=bbox,
                model=clip_model,
                preprocess=clip_preprocess,
                text_features=clip_text_features,
                prompt_categories=clip_prompt_categories,
            )

            category = to_face_category(category_result["category"])
            category_score = category_result["category_score"]

            if category == FaceCategory.real_identifiable:
                match = find_best_unknown_cluster(
                    embedding=emb,
                    unknown_cluster_embeddings=unknown_cluster_embeddings,
                )
            else:
                match = get_service_category_match(db, category)

        cluster_result = get_cluster_result_if_exists(db, match)

        detected_faces.append(
            normalize_uploaded_face(
                face_index=idx,
                bbox=bbox,
                category=category,
                category_score=category_score,
                quality=quality,
                quality_details=quality_details,
                category_result=category_result,
                match=match,
            )
            | {
                "result": cluster_result,
            }
        )

    if not detected_faces:
        return {
            "mode": "no_faces",
            "message": "На фото не знайдено облич",
            "image": {
                "width": w,
                "height": h,
                "base64": image_to_base64(img),
            },
            "summary": {
                "faces_count": 0,
                "recognized_count": 0,
                "cluster_count": 0,
            },
            "faces": [],
            "result": None,
        }

    recognized_count = sum(
        1
        for face in detected_faces
        if face["recognition"]["type"] == "known"
    )

    cluster_count = sum(
        1
        for face in detected_faces
        if face["cluster"] is not None
    )

    if len(detected_faces) == 1:
        one_face = detected_faces[0]

        return {
            "mode": "single_result"
            if one_face["recognition"]["type"] == "known"
            else "single_unknown",
            "message": None
            if one_face["recognition"]["type"] == "known"
            else "Обличчя знайдено, але персонy не розпізнано",
            "image": {
                "width": w,
                "height": h,
                "base64": image_to_base64(img),
            },
            "summary": {
                "faces_count": 1,
                "recognized_count": recognized_count,
                "cluster_count": cluster_count,
            },
            "faces": detected_faces,
            "selected_face": one_face,
            "result": one_face["result"],
        }

    return {
        "mode": "multiple_faces",
        "message": "На фото знайдено кілька облич. Оберіть потрібну людину.",
        "image": {
            "width": w,
            "height": h,
            "base64": image_to_base64(img),
        },
        "summary": {
            "faces_count": len(detected_faces),
            "recognized_count": recognized_count,
            "cluster_count": cluster_count,
        },
        "faces": detected_faces,
        "result": None,
    }
