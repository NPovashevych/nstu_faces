from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from routes.routers_classic.commons import (
    similarity_percent_from_distance,
    normalize,
    cosine_distance,
    load_reference_embeddings,
    make_image_url,
)
from db.session import get_db
from db.models import DBPerson, DBFace, DBFreeze, DBMedia, DBEmbedding
from db.enums import PersonStatus, FaceCategory


router = APIRouter(prefix="/face-clusters", tags=["face_clusters"])


BBOX_DRAW_SCALE = 1.10


def safe_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_media_path(media: DBMedia) -> str:
    return media.mp4_path or media.mxf_path or ""


def get_media_name(media: DBMedia) -> str:
    path = get_media_path(media)
    return Path(path).name if path else f"media_{media.id}"


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

    return {
        "raw": [x1, y1, x2, y2],
        "draw": {
            "x": center_x - draw_width / 2,
            "y": center_y - draw_height / 2,
            "w": draw_width,
            "h": draw_height,
        },
    }


def get_face_color(face: DBFace) -> str:
    category = face.category

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

    person_status = (
        face.person.status.value
        if face.person and face.person.status
        else None
    )

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


def get_confidence(face: DBFace):
    person_status = (
        face.person.status.value
        if face.person and face.person.status
        else None
    )

    if person_status in {
        PersonStatus.public.value,
        PersonStatus.non_public.value,
    }:
        return face.confidence

    return None


def find_nearest_known(face_embedding: DBEmbedding, reference_embeddings: list[dict]):
    if face_embedding is None or not reference_embeddings:
        return None

    emb = normalize(np.array(face_embedding.vector, dtype=np.float32))

    best = None
    best_dist = 1.0

    for ref in reference_embeddings:
        dist = cosine_distance(emb, ref["vector"])

        if dist < best_dist:
            best_dist = dist
            best = ref

    if best is None:
        return None

    if best_dist <= 0.45:
        hint_level = "strong"
        hint_label = "можливо це"
    elif best_dist <= 0.55:
        hint_level = "medium"
        hint_label = "схожий на"
    else:
        return None

    return {
        "person": {
            "id": best["person_id"],
            "name": best.get("person_name") or best.get("name"),
            "code": best.get("person_code"),
            "status": best.get("person_status"),
            "q_code": best.get("q_code"),
            "link": best.get("link"),
        },
        "distance": round(best_dist, 4),
        "similarity_percent": safe_float(
            similarity_percent_from_distance(best_dist)
        ),
        "hint_level": hint_level,
        "hint_label": hint_label,
    }


def normalize_face_item(face: DBFace, reference_embeddings: list[dict]):
    freeze = face.freeze
    media = freeze.media if freeze else None

    if not freeze or not media:
        return None

    freeze_path = freeze.freeze_path or ""
    freeze_exists = bool(freeze_path and Path(freeze_path).exists())

    category = (
        face.category
        if face.category
        else FaceCategory.uncertain
    )

    person_status = (
        face.person.status.value
        if face.person and face.person.status
        else None
    )

    nearest_known = None

    # nearest_known має сенс переважно для реальних облич.
    # Для low_quality / non_human / ai_generated краще не зашумлювати.
    if category == FaceCategory.real_identifiable:
        nearest_known = find_nearest_known(
            face_embedding=face.embedding,
            reference_embeddings=reference_embeddings,
        )

    return {
        "face_id": face.id,
        "freeze_id": freeze.id,

        "image": {
            "url": make_image_url(freeze_path) if freeze_path else None,
            "exists": freeze_exists,
            "path": freeze_path,
        },

        "media": {
            "id": media.id,
            "name": get_media_name(media),
            "path": get_media_path(media),
        },

        "time": {
            "in": safe_float(freeze.time_in, 0.0),
            "out": safe_float(freeze.time_out, 0.0),
        },

        "person": {
            "id": face.person_id,
            "name": face.person.name if face.person else None,
            "code": face.person.code if face.person else None,
            "status": person_status,
        },

        "recognition": {
            "confidence": get_confidence(face),
            "nearest_known": nearest_known,
        },

        "category": {
            "name": category.value,
            "score": safe_float(face.category_score),
        },

        "quality": safe_float(face.quality),
        "gender": face.gender.value if face.gender else None,
        "color": get_face_color(face),
        "bbox": normalize_bbox(face.bbox),
        "analysis": face.analysis or {},
    }


def normalize_cluster_key(cluster_key: str) -> list[str]:
    """
    Дозволяє відкривати:
    /face-clusters/000001
    /face-clusters/unknown_cluster_000001
    /face-clusters/low_quality_cluster
    /face-clusters/ai_generated_cluster
    /face-clusters/uncertain_cluster
    """

    variants = {cluster_key}

    if cluster_key.isdigit():
        variants.add(f"unknown_cluster_{int(cluster_key):06d}")
        variants.add(f"unknown_cluster_{cluster_key}")

    if not cluster_key.endswith("_cluster"):
        variants.add(f"{cluster_key}_cluster")

    return list(variants)


def find_cluster_person(db: Session, cluster_key: str):
    variants = normalize_cluster_key(cluster_key)

    return (
        db.query(DBPerson)
        .filter(
            (DBPerson.name.in_(variants))
            | (DBPerson.cluster_tag.in_(variants))
            | (DBPerson.code.in_(variants))
        )
        .first()
    )


@router.get("/{cluster_key}")
def get_face_cluster(cluster_key: str, db: Session = Depends(get_db)):
    person = find_cluster_person(db, cluster_key)

    if not person:
        raise HTTPException(status_code=404, detail="Cluster not found")

    reference_embeddings = load_reference_embeddings(db)

    faces = (
        db.query(DBFace)
        .options(
            joinedload(DBFace.person),
            joinedload(DBFace.embedding),
            joinedload(DBFace.freeze).joinedload(DBFreeze.media),
        )
        .filter(DBFace.person_id == person.id)
        .order_by(DBFace.freeze_id, DBFace.id)
        .all()
    )

    items = []

    for face in faces:
        item = normalize_face_item(
            face=face,
            reference_embeddings=reference_embeddings,
        )

        if item:
            items.append(item)

    categories_count = {}

    for item in items:
        category_name = item["category"]["name"]
        categories_count[category_name] = categories_count.get(category_name, 0) + 1

    return {
        "cluster": {
            "person_id": person.id,
            "name": person.name,
            "code": person.code,
            "status": person.status.value if person.status else None,
            "cluster_id": person.cluster_id,
            "cluster_tag": person.cluster_tag,
            "cluster_distance": safe_float(person.cluster_distance),
        },
        "summary": {
            "faces_count": len(items),
            "categories_count": categories_count,
        },
        "items": items,
    }
