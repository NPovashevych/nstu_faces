from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from db.session import get_db
from routers.commons import make_image_url
from db.models import DBMedia, DBFreeze, DBFace
from db.enums import PersonStatus


router = APIRouter(prefix="/media-inspector", tags=["media inspector"])


@router.get("/list")
def media_inspector_list(db: Session = Depends(get_db)):
    rows = (
        db.query(
            DBMedia.id.label("media_id"),
            DBMedia.media_path.label("media_path"),
            func.count(DBFreeze.id).label("freezes_count"),
            func.count(DBFace.id).label("faces_count"),
        )
        .outerjoin(DBFreeze, DBFreeze.media_id == DBMedia.id)
        .outerjoin(DBFace, DBFace.freeze_id == DBFreeze.id)
        .group_by(DBMedia.id, DBMedia.media_path)
        .order_by(DBMedia.id)
        .all()
    )

    return [
        {
            "media_id": row.media_id,
            "media_name": Path(row.media_path).name,
            "media_path": row.media_path,
            "freezes_count": row.freezes_count,
            "faces_count": row.faces_count,
        }
        for row in rows
    ]


def get_face_color(face: DBFace) -> str:
    person_status = (
        face.person.status.value
        if face.person and face.person.status
        else None
    )

    if face.is_suspicious or person_status == "suspicious":
        return "orange"

    if person_status == "unknown":
        return "red"

    if face.person:
        return "green"

    return "gray"


def get_face_status(face: DBFace) -> str:
    person_status = (
        face.person.status.value
        if face.person and face.person.status
        else None
    )

    if face.is_suspicious or person_status == "suspicious":
        return "suspicious"

    if person_status == "unknown":
        return "unknown"

    if face.person:
        return "known"

    return "no_person"


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

            faces_data.append({
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
                "is_suspicious": (
                    face.is_suspicious
                    or person_status == PersonStatus.suspicious.value
                ),

                "quality": face.quality,
                "confidence": face.confidence,

                "suspicion_reason": face.suspicion_reason,
                "clip_category": face.clip_category,
                "clip_score": face.clip_score,
                "clip_scores": face.clip_scores,

                "color": get_face_color(face),
            })

        frames.append({
            "freeze_id": freeze.id,
            "image_url": make_image_url(freeze.freeze_path),
            "time_in": freeze.time_in,
            "time_out": freeze.time_out,
            "faces": faces_data,
        })

    return {
        "media_id": media.id,
        "media_name": Path(media.media_path).name,
        "frames": frames,
    }
