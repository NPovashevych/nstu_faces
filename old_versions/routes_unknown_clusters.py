from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from routes.routers_classic import (
    similarity_percent_from_distance,
    normalize,
    cosine_distance,
    load_reference_embeddings,
    make_image_url,
)
from db.session import get_db
from db.models import DBPerson, DBFace, DBFreeze, DBMedia, DBEmbedding
from db.enums import PersonStatus, FaceCategory


router = APIRouter(prefix="/unknown-clusters", tags=["unknown_clusters"])


def safe_float(value, default: float = 0.0) -> float:
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


def make_bbox_draw(bbox):
    bbox = bbox or [0, 0, 0, 0]

    return {
        "x": safe_float(bbox[0]),
        "y": safe_float(bbox[1]),
        "w": safe_float(bbox[2]) - safe_float(bbox[0]),
        "h": safe_float(bbox[3]) - safe_float(bbox[1]),
    }


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

    similarity = similarity_percent_from_distance(best_dist)

    if best_dist <= 0.45:
        hint_level = "strong"
        hint_label = "можливо це"
    elif best_dist <= 0.55:
        hint_level = "medium"
        hint_label = "схожий на"
    else:
        return None

    return {
        "person_id": best["person_id"],
        "name": best.get("person_name") or best.get("name"),
        "q_code": best.get("q_code"),
        "link": best.get("person_link") or best.get("link"),
        "distance": round(best_dist, 4),
        "similarity_percent": safe_float(similarity),
        "hint_level": hint_level,
        "hint_label": hint_label,
    }


@router.get("/{cluster_number}")
def get_unknown_cluster(cluster_number: str, db: Session = Depends(get_db)):
    if cluster_number.startswith("unknown_cluster_"):
        cluster_name = cluster_number
    else:
        cluster_name = f"unknown_cluster_{cluster_number}"

    person = (
        db.query(DBPerson)
        .filter(
            (DBPerson.name == cluster_name) |
            (DBPerson.cluster_tag == cluster_name) |
            (DBPerson.code == cluster_name)
        )
        .first()
    )

    if not person:
        raise HTTPException(status_code=404, detail="Cluster not found")

    if person.status != PersonStatus.unknown:
        return {
            "cluster": cluster_name,
            "person_id": person.id,
            "status": person.status.value if person.status else None,
            "message": "Кластер не є unknown-кластером",
            "resolved_name": person.name,
            "cluster_tag": person.cluster_tag,
            "count": 0,
            "items": [],
        }

    reference_embeddings = load_reference_embeddings(db)

    faces = (
        db.query(DBFace)
        .options(
            joinedload(DBFace.embedding),
            joinedload(DBFace.freeze).joinedload(DBFreeze.media),
        )
        .join(DBFreeze, DBFace.freeze_id == DBFreeze.id)
        .join(DBMedia, DBFreeze.media_id == DBMedia.id)
        .filter(DBFace.person_id == person.id)
        .order_by(DBFreeze.media_id, DBFreeze.time_in)
        .all()
    )

    items = []

    for face in faces:
        freeze = face.freeze
        media = freeze.media if freeze else None

        if not freeze or not media:
            continue

        freeze_path = freeze.freeze_path or ""
        freeze_exists = bool(freeze_path and Path(freeze_path).exists())

        nearest_known = find_nearest_known(
            face_embedding=face.embedding,
            reference_embeddings=reference_embeddings,
        )

        bbox = face.bbox or [0, 0, 0, 0]
        category = face.category.value if face.category else None

        items.append({
            "face_id": face.id,
            "freeze_id": freeze.id,

            "image_url": make_image_url(freeze_path) if freeze_path else None,
            "image_exists": freeze_exists,
            "freeze_path": freeze_path,

            "media_id": media.id,
            "media_name": get_media_name(media),
            "media_path": get_media_path(media),

            "time_in": safe_float(freeze.time_in),
            "time_out": safe_float(freeze.time_out),

            "bbox": bbox,
            "bbox_draw": make_bbox_draw(bbox),

            "category": category,
            "category_score": safe_float(face.category_score),
            "quality": safe_float(face.quality),
            "confidence": face.confidence if face.confidence is not None else 0,
            "analysis": face.analysis or {},

            "nearest_known": nearest_known,
        })

    return {
        "cluster": cluster_name,
        "person_id": person.id,
        "status": person.status.value if person.status else None,
        "cluster_id": person.cluster_id,
        "cluster_tag": person.cluster_tag,
        "count": len(items),
        "items": items,
    }
