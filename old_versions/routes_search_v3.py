from pathlib import Path
import json
import threading
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from routers.commons import (
    cosine_distance,
    normalize,
    make_image_url,
    image_to_base64,
    load_reference_embeddings,
    make_media_url,
)
from db.session import get_db
from db.models import DBPerson, DBFace, DBFreeze, DBMedia, DBEmbedding
from db.enums import PersonStatus, EmbeddingType
from old_versions.face_quality import is_good_face


router = APIRouter(prefix="/search", tags=["search"])


DIST_TOLERANCE = 0.42
STEP_TOLERANCE = 0.03
MAX_ACCEPTABLE_DIST = 0.55
CLUSTER_TOLERANCE = 0.60
MIN_PHOTO_FACE_QUALITY = 0.00

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


def get_confidence(dist: float):
    if dist <= DIST_TOLERANCE:
        return 0

    for i in range(1, 4):
        if dist <= DIST_TOLERANCE + i * STEP_TOLERANCE:
            return i

    return None


def apply_confidence_to_name(name: str, confidence: int | None) -> str:
    if confidence is None or confidence <= 0:
        return name

    return f"{name} {'?' * min(confidence, 3)}"


def make_bbox_draw(bbox):
    return {
        "x": bbox[0],
        "y": bbox[1],
        "w": bbox[2] - bbox[0],
        "h": bbox[3] - bbox[1],
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


def load_cluster_embeddings(db: Session):
    rows = (
        db.query(DBEmbedding, DBPerson)
        .join(DBPerson, DBEmbedding.person_id == DBPerson.id)
        .filter(DBPerson.status.in_([PersonStatus.unknown, PersonStatus.suspicious]))
        .filter(DBEmbedding.embedding_type == EmbeddingType.detected_face)
        .all()
    )

    result = []

    for emb, person in rows:
        result.append({
            "person_id": person.id,
            "person_name": person.name,
            "person_status": person.status.value if person.status else None,
            "is_suspicious_cluster": person.status == PersonStatus.suspicious,
            "cluster_id": person.cluster_id,
            "cluster_tag": person.cluster_tag,
            "cluster_distance": person.cluster_distance,
            "vector": normalize(np.array(emb.vector, dtype=np.float32)),
        })

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

    confidence = get_confidence(best_dist)

    return {
        "type": "known",
        "recognized": True,
        "is_cluster": False,
        "is_unknown_cluster": False,
        "is_suspicious_cluster": False,
        "person_id": best["person_id"],
        "name": best["person_name"],
        "display_name": apply_confidence_to_name(best["person_name"], confidence),
        "q_code": best["q_code"],
        "link": best["link"],
        "distance": round(best_dist, 4),
        "similarity_percent": round((1 - best_dist * 0.5) * 100, 2),
        "confidence": confidence,
    }


def find_best_cluster(embedding, cluster_embeddings):
    best = None
    best_dist = 1.0

    for ref in cluster_embeddings:
        dist = cosine_distance(embedding, ref["vector"])

        if dist < best_dist:
            best_dist = dist
            best = ref

    if best is None or best_dist > CLUSTER_TOLERANCE:
        return {
            "type": "unknown",
            "recognized": False,
            "is_cluster": False,
            "is_unknown_cluster": False,
            "is_suspicious_cluster": False,
            "display_name": "Невідоме обличчя",
            "distance": round(best_dist, 4) if best else None,
        }

    hint = get_latest_cluster_hint(best["person_id"])

    display_name = (
        hint["suggested_name"]
        if hint and hint.get("suggested_name")
        else best["cluster_tag"] or best["person_name"]
    )

    is_suspicious_cluster = best["is_suspicious_cluster"]

    return {
        "type": "suspicious_cluster" if is_suspicious_cluster else "unknown_cluster",
        "recognized": False,
        "is_cluster": True,
        "is_unknown_cluster": not is_suspicious_cluster,
        "is_suspicious_cluster": is_suspicious_cluster,
        "person_id": best["person_id"],
        "name": best["person_name"],
        "display_name": display_name,
        "person_status": best["person_status"],
        "cluster_id": best["cluster_id"],
        "cluster_tag": best["cluster_tag"],
        "distance": round(best_dist, 4),
        "similarity_percent": round((1 - best_dist * 0.5) * 100, 2),
        "suggested_name": hint["suggested_name"] if hint else None,
        "hint_status": hint["status"] if hint else None,
    }


def find_best_match(embedding, reference_embeddings, cluster_embeddings):
    known_match = find_best_known_match(embedding, reference_embeddings)

    if known_match:
        return known_match

    return find_best_cluster(embedding, cluster_embeddings)


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
                "media_path": media.media_path,
                "media_url": make_media_url(media.media_path),
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
            "is_suspicious": face.is_suspicious,
            "suspicion_reason": face.suspicion_reason,
            "clip_category": face.clip_category,
            "clip_score": face.clip_score,
            "clip_scores": face.clip_scores,
        })

    return {
        "person_id": person.id,
        "name": person.name,
        "q_code": person.q_code,
        "link": person.link,
        "status": person.status.value if person.status else None,
        "cluster_id": person.cluster_id,
        "cluster_tag": person.cluster_tag,
        "medias": list(medias_map.values()),
    }


def get_cluster_result_if_exists(db: Session, match: dict | None):
    if not match or not match.get("is_cluster"):
        return None

    person_id = match.get("person_id")
    if not person_id:
        return None

    cluster_person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if not cluster_person:
        return None

    return get_person_media_result(db, cluster_person)


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

    h, w = img.shape[:2]

    model = get_model()
    reference_embeddings = load_reference_embeddings(db)
    cluster_embeddings = load_cluster_embeddings(db)

    faces = model.get(img)
    faces = sorted(faces, key=lambda f: f.bbox[0])

    detected_faces = []

    for idx, face in enumerate(faces, start=1):
        quality = is_good_face(img, face)
        is_low_quality = quality < MIN_PHOTO_FACE_QUALITY

        bbox = face.bbox.astype(float).tolist()
        emb = normalize(face.embedding)

        match = find_best_match(
            embedding=emb,
            reference_embeddings=reference_embeddings,
            cluster_embeddings=cluster_embeddings,
        )

        if match and match.get("is_cluster"):
            match["cluster_result"] = get_cluster_result_if_exists(db, match)

        detected_faces.append({
            "face_index": idx,
            "bbox": bbox,
            "bbox_draw": make_bbox_draw(bbox),
            "quality": round(quality, 4),
            "is_low_quality": is_low_quality,
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

    cluster_faces = [
        f for f in detected_faces
        if f["match"] and f["match"].get("is_cluster")
    ]

    suspicious_cluster_faces = [
        f for f in detected_faces
        if f["match"] and f["match"].get("is_suspicious_cluster")
    ]

    if len(detected_faces) == 1:
        one_face = detected_faces[0]
        match = one_face["match"]

        if not match or not match.get("recognized"):
            cluster_result = match.get("cluster_result") if match else None

            return {
                "mode": "single_unknown",
                "message": "Обличчя знайдено, але персонy не розпізнано",
                "image_width": w,
                "image_height": h,
                "uploaded_image": image_to_base64(img),
                "faces": detected_faces,
                "cluster_result": cluster_result,
                "result": cluster_result,
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

    cluster_results = [
        f["match"]["cluster_result"]
        for f in cluster_faces
        if f.get("match") and f["match"].get("cluster_result")
    ]

    return {
        "mode": "multiple_faces",
        "message": "На фото знайдено кілька облич. Оберіть потрібну людину.",
        "image_width": w,
        "image_height": h,
        "uploaded_image": image_to_base64(img),
        "faces": detected_faces,
        "recognized_count": len(recognized_faces),
        "cluster_count": len(cluster_faces),
        "unknown_cluster_count": len([
            f for f in cluster_faces
            if f["match"] and f["match"].get("is_unknown_cluster")
        ]),
        "suspicious_cluster_count": len(suspicious_cluster_faces),
        "cluster_results": cluster_results,
    }
