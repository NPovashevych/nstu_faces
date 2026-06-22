from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from db.session import get_db
from routers.commons import make_image_url
from db.models import DBMedia, DBFreeze, DBFace
from db.enums import FaceCategory, PersonStatus


router = APIRouter(prefix="/media-inspector", tags=["media inspector"])


BBOX_DRAW_SCALE = 1.10


def safe_float(value, default=None):
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_media_main_path(media: DBMedia) -> str:
    return media.mp4_path or media.mxf_path or ""


def get_media_name(media: DBMedia) -> str:
    path = get_media_main_path(media)
    return Path(path).name if path else f"media_{media.id}"


def get_person_status(face: DBFace):
    if face.person and face.person.status:
        return face.person.status.value

    return None


def get_confidence(face: DBFace):
    person_status = get_person_status(face)

    if person_status in {
        PersonStatus.public.value,
        PersonStatus.non_public.value,
    }:
        return face.confidence

    return None


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

    person_status = get_person_status(face)

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


def normalize_face(face: DBFace):
    category = (
        face.category.value
        if face.category
        else FaceCategory.uncertain.value
    )

    return {
        "face_id": face.id,

        "person": {
            "id": face.person_id,
            "name": face.person.name if face.person else None,
            "code": face.person.code if face.person else None,
            "status": get_person_status(face),
        },

        "recognition": {
            "confidence": get_confidence(face),
        },

        "category": {
            "name": category,
            "score": safe_float(face.category_score),
        },

        "quality": safe_float(face.quality),
        "gender": face.gender.value if face.gender else None,
        "color": get_face_color(face),
        "bbox": normalize_bbox(face.bbox),
        "analysis": face.analysis or {},
    }


@router.get("/list")
def media_inspector_list(db: Session = Depends(get_db)):
    rows = (
        db.query(
            DBMedia.id.label("media_id"),
            DBMedia.mxf_path.label("mxf_path"),
            DBMedia.mp4_path.label("mp4_path"),
            func.count(DBFreeze.id).label("freezes_count"),
            func.count(DBFace.id).label("faces_count"),
        )
        .outerjoin(DBFreeze, DBFreeze.media_id == DBMedia.id)
        .outerjoin(DBFace, DBFace.freeze_id == DBFreeze.id)
        .group_by(DBMedia.id, DBMedia.mxf_path, DBMedia.mp4_path)
        .order_by(DBMedia.id)
        .all()
    )

    return [
        {
            "media_id": row.media_id,
            "media_name": Path(row.mp4_path or row.mxf_path or "").name
            if row.mp4_path or row.mxf_path
            else f"media_{row.media_id}",
            "media_path": row.mp4_path or row.mxf_path or "",
            "mxf_path": row.mxf_path,
            "mp4_path": row.mp4_path,
            "freezes_count": int(row.freezes_count or 0),
            "faces_count": int(row.faces_count or 0),
        }
        for row in rows
    ]


@router.get("/{media_id}")
def media_inspector(media_id: int, db: Session = Depends(get_db)):
    media = db.query(DBMedia).filter(DBMedia.id == media_id).first()

    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    freezes = (
        db.query(DBFreeze)
        .filter(DBFreeze.media_id == media_id)
        .order_by(DBFreeze.time_in)
        .all()
    )

    frames = []

    for freeze in freezes:
        faces = (
            db.query(DBFace)
            .options(joinedload(DBFace.person))
            .filter(DBFace.freeze_id == freeze.id)
            .order_by(DBFace.id)
            .all()
        )

        frames.append(
            {
                "freeze_id": freeze.id,
                "image_url": make_image_url(freeze.freeze_path),
                "time_in": safe_float(freeze.time_in, 0.0),
                "time_out": safe_float(freeze.time_out, 0.0),
                "faces_count": len(faces),
                "faces": [normalize_face(face) for face in faces],
            }
        )

    return {
        "media": {
            "id": media.id,
            "name": get_media_name(media),
            "path": get_media_main_path(media),
            "mxf_path": media.mxf_path,
            "mp4_path": media.mp4_path,
        },
        "summary": {
            "freezes_count": len(frames),
            "faces_count": sum(frame["faces_count"] for frame in frames),
        },
        "frames": frames,
    }
