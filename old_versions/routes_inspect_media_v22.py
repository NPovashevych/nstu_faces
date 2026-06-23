from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from db.session import get_db
from routes.routers_classic import make_image_url
from db.models import DBMedia, DBFreeze, DBFace
from db.enums import FaceCategory, PersonStatus


router = APIRouter(prefix="/media-inspector", tags=["media inspector"])


LOW_QUALITY_THRESHOLD = 0.45


def get_media_main_path(media: DBMedia) -> str:
    return media.mp4_path or media.mxf_path or ""


def get_media_name(media: DBMedia) -> str:
    return Path(get_media_main_path(media)).name


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

    result = []

    for row in rows:
        main_path = row.mp4_path or row.mxf_path or ""

        result.append({
            "media_id": row.media_id,

            # старий фронт очікував media_name/media_path
            "media_name": Path(main_path).name,
            "media_path": main_path,

            # нові поля
            "mxf_path": row.mxf_path,
            "mp4_path": row.mp4_path,

            # старі поля
            "freezes_count": row.freezes_count or 0,
            "faces_count": row.faces_count or 0,
        })

    return result


def get_face_color(face: DBFace) -> str:
    category = face.category
    quality = face.quality if face.quality is not None else 0.0

    if quality < LOW_QUALITY_THRESHOLD:
        return "gray"

    if category == FaceCategory.real_unidentifiable:
        return "gray"

    if category != FaceCategory.real_identifiable:
        return "orange"

    person_status = (
        face.person.status.value
        if face.person and face.person.status
        else None
    )

    if person_status == PersonStatus.unknown.value:
        return "red"

    if face.person:
        return "green"

    return "red"


def get_face_status(face: DBFace) -> str:
    category = face.category
    quality = face.quality if face.quality is not None else 0.0

    if quality < LOW_QUALITY_THRESHOLD:
        return "low_quality"

    if category == FaceCategory.real_unidentifiable:
        return "real_unidentifiable"

    if category != FaceCategory.real_identifiable:
        return "possibly_not_human"

    person_status = (
        face.person.status.value
        if face.person and face.person.status
        else None
    )

    if person_status == PersonStatus.unknown.value:
        return "unknown"

    if face.person:
        return "known"

    return "unknown"


def get_old_style_suspicious(face: DBFace) -> bool:
    category = face.category

    if category in [
        FaceCategory.non_human,
        FaceCategory.artificial_human,
        FaceCategory.ai_generated,
        FaceCategory.uncertain,
    ]:
        return True

    person_status = (
        face.person.status.value
        if face.person and face.person.status
        else None
    )

    return person_status == PersonStatus.suspicious.value


def get_clip_from_analysis(face: DBFace):
    analysis = face.analysis or {}
    clip = analysis.get("clip") or {}

    return {
        "clip_category": clip.get("category") or clip.get("best_clip_category"),
        "clip_score": clip.get("category_score") or clip.get("best_clip_score"),
        "clip_scores": clip.get("clip_scores"),
    }


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
            .all()
        )

        faces_data = []

        for face in faces:
            bbox = face.bbox or [0, 0, 0, 0]

            person_status = (
                face.person.status.value
                if face.person and face.person.status
                else None
            )

            category = face.category.value if face.category else None
            clip_data = get_clip_from_analysis(face)

            is_suspicious = get_old_style_suspicious(face)
            is_low_quality = (
                face.quality is not None
                and face.quality < LOW_QUALITY_THRESHOLD
            )

            faces_data.append({
                # старі поля
                "face_id": face.id,
                "bbox": bbox,
                "bbox_draw": {
                    "x": bbox[0],
                    "y": bbox[1],
                    "w": bbox[2] - bbox[0],
                    "h": bbox[3] - bbox[1],
                },

                "person_id": face.person_id,
                "person_name": face.person.name if face.person else None,
                "person_status": person_status,

                "face_status": get_face_status(face),
                "is_unknown": person_status == PersonStatus.unknown.value,
                "is_suspicious": is_suspicious,

                "quality": face.quality if face.quality is not None else 0.0,
                "confidence": face.confidence,

                # старі clip-поля, але дістаємо їх з analysis
                "suspicion_reason": category,
                "clip_category": clip_data["clip_category"],
                "clip_score": clip_data["clip_score"],
                "clip_scores": clip_data["clip_scores"],

                "color": get_face_color(face),

                # нові поля
                "category": category,
                "category_score": face.category_score,
                "analysis": face.analysis,

                "is_low_quality": is_low_quality,
                "is_possibly_not_human": (
                    face.category not in [
                        FaceCategory.real_identifiable,
                        FaceCategory.real_unidentifiable,
                    ]
                ),
            })

        frames.append({
            # старі поля
            "freeze_id": freeze.id,
            "image_url": make_image_url(freeze.freeze_path),
            "time_in": freeze.time_in if freeze.time_in is not None else 0.0,
            "time_out": freeze.time_out if freeze.time_out is not None else 0.0,
            "faces": faces_data,
        })

    return {
        # старі поля
        "media_id": media.id,
        "media_name": get_media_name(media),
        "media_path": get_media_main_path(media),

        # нові поля
        "mxf_path": media.mxf_path,
        "mp4_path": media.mp4_path,

        # щоб фронт не падав на toFixed/undefined
        "freezes_count": len(frames),
        "faces_count": sum(len(frame["faces"]) for frame in frames),

        "frames": frames,
    }
