from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.config import FREEZE_FOLDER
from db.session import get_db
from db.enums import EmbeddingType
from db.models import DBPerson, DBFace, DBFreeze, DBMedia, DBEmbedding


router = APIRouter(prefix="/unknown-clusters", tags=["unknown_clusters"])


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def cosine_distance(a, b):
    return 1 - float(np.dot(a, b))


def similarity_percent_from_distance(distance: float):
    return round((1 - distance) * 100, 2)


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
            "person_link": row.person.link,
            "q_code": row.person.q_code,
            "vector": normalize(np.array(row.vector, dtype=np.float32)),
        })

    return refs


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
        "name": best["person_name"],
        "q_code": best["q_code"],
        "link": best["person_link"],
        "distance": round(best_dist, 4),
        "similarity_percent": similarity,
        "hint_level": hint_level,
        "hint_label": hint_label,
    }


def make_image_url(freeze_path: str):
    relative_path = Path(freeze_path).relative_to(Path(FREEZE_FOLDER))
    return f"/freezes/{relative_path.as_posix()}"


@router.get("/{cluster_number}")
def get_unknown_cluster(cluster_number: str, db: Session = Depends(get_db)):
    if cluster_number.startswith("unknown_cluster_"):
        cluster_name = cluster_number
    else:
        cluster_name = f"unknown_cluster_{cluster_number}"

    person = (
        db.query(DBPerson)
        .filter(DBPerson.name == cluster_name)
        .first()
    )

    if not person:
        raise HTTPException(status_code=404, detail="Cluster not found")

    reference_embeddings = load_reference_embeddings(db)

    faces = (
        db.query(DBFace)
        .join(DBFreeze, DBFace.freeze_id == DBFreeze.id)
        .join(DBMedia, DBFreeze.media_id == DBMedia.id)
        .filter(DBFace.person_id == person.id)
        .order_by(DBFreeze.media_id, DBFreeze.time_in)
        .all()
    )

    items = []

    for face in faces:
        freeze = face.freeze
        media = freeze.media

        nearest_known = find_nearest_known(
            face_embedding=face.embedding,
            reference_embeddings=reference_embeddings,
        )

        bbox = face.bbox

        items.append({
            "face_id": face.id,
            "freeze_id": freeze.id,
            "image_url": make_image_url(freeze.freeze_path),
            "media_id": media.id,
            "media_name": Path(media.media_path).name,
            "time_in": freeze.time_in,
            "time_out": freeze.time_out,
            "bbox": bbox,
            "bbox_draw": {
                "x": bbox[0],
                "y": bbox[1],
                "w": bbox[2] - bbox[0],
                "h": bbox[3] - bbox[1],
            },
            "quality": face.quality,
            "confidence": face.confidence,
            "nearest_known": nearest_known,
        })

    return {
        "cluster": cluster_name,
        "person_id": person.id,
        "count": len(items),
        "items": items,
    }