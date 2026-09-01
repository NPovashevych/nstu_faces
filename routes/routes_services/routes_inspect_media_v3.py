from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from db.session import get_db
from routes.routers_classic.commons import make_image_url
from db.models import DBMedia, DBFreeze, DBFace, DBMediaDescription
from db.enums import PersonStatus


router = APIRouter(prefix="/media-inspector", tags=["media inspector"])


BBOX_DRAW_SCALE = 1.10


def safe_float(value, default=None):
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def enum_value(value):
    if value is None:
        return None

    return value.value if hasattr(value, "value") else str(value)


def get_media_main_path(media: DBMedia) -> str:
    return media.mp4_path or media.mxf_path or ""


def get_media_name(media: DBMedia) -> str:
    path = get_media_main_path(media)
    return Path(path).name if path else f"media_{media.id}"


def get_media_type(media: DBMedia):
    return enum_value(getattr(media, "media_type", None))


def get_media_source(media: DBMedia):
    source = getattr(media, "source", None)

    if source is None:
        return None

    return {
        "id": getattr(source, "id", None),
        "code": getattr(source, "code", None),
        "name": getattr(source, "name", None),
    }


def get_media_description(db: Session, media: DBMedia):
    if not media.material_id:
        return None

    description = (
        db.query(DBMediaDescription)
        .filter(DBMediaDescription.material_id == media.material_id)
        .first()
    )

    if not description:
        return None

    return {
        "id": description.id,
        "material_id": description.material_id,
        "section": description.section,
        "shooting_date": description.shooting_date,
        "journalist": description.journalist,
        "operators": description.operators,
        "description": description.description,
        "another_info": description.another_info,
    }


def normalize_media(media: DBMedia):
    return {
        "id": media.id,
        "material_id": getattr(media, "material_id", None),
        "name": get_media_name(media),
        "path": get_media_main_path(media),
        "mxf_path": media.mxf_path,
        "mp4_path": media.mp4_path,
        "media_type": get_media_type(media),
        "duration": safe_float(getattr(media, "duration", None)),
        "source": get_media_source(media),
    }


def get_person_status(face: DBFace):
    if face.person and face.person.status:
        return face.person.status.value

    return None


def get_face_category_name(face: DBFace) -> str:
    if face.face_category:
        return face.face_category.name

    return "uncertain"


def get_face_category_group(face: DBFace):
    if face.face_category:
        return face.face_category.code

    return None


def get_confidence(face: DBFace):
    person_status = get_person_status(face)

    if person_status in {
        PersonStatus.public.value,
        PersonStatus.non_public.value,
    }:
        return face.confidence

    return None


def get_confidence_marks(confidence):
    if confidence is None or confidence == 0:
        return ""

    return "?" * int(confidence)


def get_face_color(face: DBFace) -> str:
    category_name = get_face_category_name(face)

    if category_name in {"low_quality", "unidentifiable"}:
        return "gray"

    if category_name in {
        "non_human",
        "artificial",
        "ai_generated",
        "uncertain",
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

    return {
        "raw": [x1, y1, x2, y2],
        "draw": {
            "x": center_x - draw_width / 2,
            "y": center_y - draw_height / 2,
            "w": draw_width,
            "h": draw_height,
        },
    }


def normalize_face(face: DBFace):
    confidence = get_confidence(face)

    return {
        "face_id": face.id,

        "person": {
            "id": face.person_id,
            "name": face.person.name if face.person else None,
            "code": face.person.code if face.person else None,
            "status": get_person_status(face),
        },

        "recognition": {
            "confidence": confidence,
            "confidence_marks": get_confidence_marks(confidence),
        },

        "category": {
            "name": get_face_category_name(face),
            "group": get_face_category_group(face),
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
    # media_items = (
    #     db.query(DBMedia)
    #     .options(joinedload(DBMedia.source))
    #     .order_by(DBMedia.id)
    #     .all()
    # )

    # media_items = (
    #     db.query(DBMedia)
    #     .options(joinedload(DBMedia.source))
    #     .filter(DBMedia.duration < 200)
    #     .order_by(DBMedia.id.desc())
    #     .limit(250)
    #     .all()
    # )

    media_items = (
        db.query(DBMedia)
        .options(joinedload(DBMedia.source))
        .join(DBFreeze, DBFreeze.media_id == DBMedia.id)
        .join(DBFace, DBFace.freeze_id == DBFreeze.id)
        .filter(DBMedia.duration < 200)
        .distinct()
        .order_by(DBMedia.id.asc())
        .limit(250)
        .all()
    )

    result = []

    for media in media_items:
        freezes_count = (
            db.query(func.count(DBFreeze.id))
            .filter(DBFreeze.media_id == media.id)
            .scalar()
        )

        if int(freezes_count or 0) == 0 and get_media_type(media) == "image":
            freezes_count = 1

        faces_count = (
            db.query(func.count(DBFace.id))
            .join(DBFreeze, DBFreeze.id == DBFace.freeze_id)
            .filter(DBFreeze.media_id == media.id)
            .scalar()
        )

        item = normalize_media(media)
        item.update(
            {
                "media_id": media.id,
                "media_name": item["name"],
                "media_path": item["path"],
                "freezes_count": int(freezes_count or 0),
                "faces_count": int(faces_count or 0),
            }
        )

        result.append(item)

    return result


@router.get("/{media_id}")
def media_inspector(media_id: int, db: Session = Depends(get_db)):
    media = (
        db.query(DBMedia)
        .options(joinedload(DBMedia.source))
        .filter(DBMedia.id == media_id)
        .first()
    )



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
            .options(
                joinedload(DBFace.person),
                joinedload(DBFace.face_category),
            )
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

    media_result = normalize_media(media)
    media_result["description"] = get_media_description(db, media)

    return {
        "media": media_result,
        "summary": {
            "freezes_count": len(frames),
            "faces_count": sum(frame["faces_count"] for frame in frames),
        },
        "frames": frames,
    }
