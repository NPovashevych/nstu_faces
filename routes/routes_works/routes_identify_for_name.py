from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import cast, Float, func
from sqlalchemy.orm import Session, joinedload

from db.session import get_db
from db.models import DBPerson, DBFace, DBFreeze, DBMedia, DBEmbedding
from db.enums import EmbeddingType

from commons.commons_base import make_image_url
from commons.common_search import safe_float, format_time, get_source_label, get_media_url, normalize_person, normalize_bbox


router = APIRouter(prefix="/identify-name", tags=["identify by name"])

REFERENCE_PREVIEW_MAX_SIZE = 500
REFERENCE_PREVIEW_JPEG_QUALITY = 75


def normalize_face_preview(face: DBFace) -> dict:
    bbox_data = normalize_bbox(face.bbox)
    return {"face_id": face.id, "bbox_draw": bbox_data["bbox_draw"]}


def normalize_detected_face(face: DBFace) -> dict:
    freeze = face.freeze
    media = freeze.media
    time_in = safe_float(freeze.time_in, 0.0)
    time_out = safe_float(freeze.time_out, 0.0)

    return {
        "face": normalize_face_preview(face),
        "freeze": {"freeze_id": freeze.id, "image_url": make_image_url(freeze.freeze_path), "time_in": time_in, "time_out": time_out,
                   "time": f"{format_time(time_in)} – {format_time(time_out)}"},
        "media": {"id": media.id, "material_id": media.material_id, "name": Path(media.mp4_path or media.mxf_path or "").name,
                  "type": media.media_type.value if media.media_type else None, "source": get_source_label(media),
                  "url": get_media_url(media)},
    }


def normalize_reference_embedding(embedding: DBEmbedding) -> dict:
    source = embedding.source or {}
    return {"embedding_id": embedding.id, "file_name": source.get("file_name"), "image_url": f"/identify-name/reference-image/{embedding.id}"}



def load_detected_faces(db: Session, person_id: int, limit: int) -> tuple[list[DBFace], bool]:
    distance = cast(DBEmbedding.source["distance"].astext, Float)
    query_limit = limit + 1 if limit < 10 else 10

    faces = (
        db.query(DBFace)
        .join(DBEmbedding, DBFace.embedding_id == DBEmbedding.id)
        .filter(
            DBFace.person_id == person_id,
            DBFace.confidence == 0,
            DBEmbedding.embedding_type == EmbeddingType.detected_face,
            DBEmbedding.source.isnot(None),
            DBEmbedding.source["distance"].astext.isnot(None)
        )
        .options(joinedload(DBFace.embedding), joinedload(DBFace.freeze).joinedload(DBFreeze.media).joinedload(DBMedia.source))
        .order_by(distance.asc())
        .limit(query_limit)
        .all()
    )

    has_more = limit < 10 and len(faces) > limit
    return faces[:limit], has_more


def load_reference_embeddings(db: Session, person_id: int) -> list[DBEmbedding]:
    return db.query(DBEmbedding).filter(DBEmbedding.person_id == person_id, DBEmbedding.embedding_type == EmbeddingType.reference_face).order_by(DBEmbedding.id.asc()).all()


def get_person_stats(db: Session, person_id: int) -> dict:
    detected_confirmed = db.query(func.count(DBFace.id)).join(DBEmbedding, DBFace.embedding_id == DBEmbedding.id).filter(DBFace.person_id == person_id, DBFace.confidence == 0, DBEmbedding.embedding_type == EmbeddingType.detected_face).scalar() or 0
    reference_count = db.query(func.count(DBEmbedding.id)).filter(DBEmbedding.person_id == person_id, DBEmbedding.embedding_type == EmbeddingType.reference_face).scalar() or 0

    return {"detected_confirmed": detected_confirmed, "reference": reference_count}


@router.get("/reference-image/{embedding_id}")
def get_reference_image(embedding_id: int, db: Session = Depends(get_db)):
    embedding = db.query(DBEmbedding).filter(DBEmbedding.id == embedding_id, DBEmbedding.embedding_type == EmbeddingType.reference_face).first()

    if embedding is None:
        raise HTTPException(status_code=404, detail="Reference embedding not found")

    source = embedding.source or {}
    file_path = source.get("file_path")

    if not file_path:
        raise HTTPException(status_code=404, detail="Reference image path not found")

    image_path = Path(file_path)

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Reference image file not found")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((REFERENCE_PREVIEW_MAX_SIZE, REFERENCE_PREVIEW_MAX_SIZE))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=REFERENCE_PREVIEW_JPEG_QUALITY, optimize=True)
            buffer.seek(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot create reference preview: {e}")

    return StreamingResponse(buffer, media_type="image/jpeg")


@router.get("/persons/{person_id}/face-previews")
def get_person_face_previews(
        person_id: int, face_type: Literal["detected_face",
        "reference_face"] = Query(default="detected_face"),
        limit: int = Query(default=5, ge=1, le=10), db: Session = Depends(get_db)
):
    person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    stats = get_person_stats(db, person_id)

    if face_type == "reference_face":
        embeddings = load_reference_embeddings(db, person_id)

        return {
            "person": normalize_person(person),
            "stats": stats,
            "face_type": "reference_face",
            "faces": [normalize_reference_embedding(embedding) for embedding in embeddings],
            "has_more": False,
        }

    faces, has_more = load_detected_faces(db=db, person_id=person_id, limit=limit)

    return {
        "person": normalize_person(person),
        "stats": stats,
        "face_type": "detected_face",
        "faces": [normalize_detected_face(face) for face in faces],
        "has_more": has_more,
    }
