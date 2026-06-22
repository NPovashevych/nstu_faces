from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from db.session import get_db
from db.models import DBPerson, DBFace, DBFreeze, DBMedia
from routers.commons import make_image_url, make_media_url


router = APIRouter(prefix="/search-name", tags=["search by name"])


BBOX_DRAW_SCALE = 1.10
MEDIA_SOURCE_LABEL = "Тестовий пакет"


def safe_float(value, default=None):
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_time(seconds) -> str:
    seconds = safe_float(seconds, 0.0)
    total = int(seconds)

    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60

    return f"{h:02}:{m:02}:{s:02}"


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


def get_confidence_marks(confidence):
    if confidence is None or confidence == 0:
        return ""

    return "?" * int(confidence)


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


def normalize_person(person: DBPerson):
    return {
        "id": person.id,
        "name": person.name,
        "status": person.status.value if person.status else None,
        "q_code": person.q_code,
        "link": person.link,
    }


def normalize_face(face: DBFace):
    return {
        "face_id": face.id,
        "bbox": normalize_bbox(face.bbox),
        "confidence": face.confidence,
        "confidence_marks": get_confidence_marks(face.confidence),
        "frame_color": "green",
        "analysis": face.analysis or {},
    }


def build_person_result(db: Session, person: DBPerson):
    faces = (
        db.query(DBFace)
        .options(
            joinedload(DBFace.freeze)
            .joinedload(DBFreeze.media)
            .joinedload(DBMedia.descriptions),
        )
        .join(DBFreeze, DBFace.freeze_id == DBFreeze.id)
        .join(DBMedia, DBFreeze.media_id == DBMedia.id)
        .filter(DBFace.person_id == person.id)
        .order_by(DBMedia.id, DBFreeze.time_in, DBFace.id)
        .all()
    )

    medias_map = {}

    for face in faces:
        freeze = face.freeze
        media = freeze.media

        if not media.mp4_path:
            continue

        if media.id not in medias_map:
            medias_map[media.id] = {
                "media": {
                    "id": media.id,
                    "name": Path(media.mp4_path).name,
                    "source": MEDIA_SOURCE_LABEL,
                    "url": make_media_url(media.mp4_path),
                    "description": get_media_description(media),
                },
                "summary": {
                    "frames_count": 0,
                    "faces_count": 0,
                },
                "frames_map": {},
            }

        media_item = medias_map[media.id]

        if freeze.id not in media_item["frames_map"]:
            time_in = safe_float(freeze.time_in, 0.0)
            time_out = safe_float(freeze.time_out, 0.0)

            media_item["frames_map"][freeze.id] = {
                "freeze_id": freeze.id,
                "image_url": make_image_url(freeze.freeze_path),
                "time_in": time_in,
                "time_out": time_out,
                "time": f"{format_time(time_in)} – {format_time(time_out)}",
                "faces": [],
            }

        media_item["frames_map"][freeze.id]["faces"].append(normalize_face(face))
        media_item["summary"]["faces_count"] += 1

    medias = []

    for item in medias_map.values():
        frames = list(item["frames_map"].values())
        item["summary"]["frames_count"] = len(frames)

        medias.append(
            {
                "media": item["media"],
                "summary": item["summary"],
                "frames": frames,
            }
        )

    return {
        "person": normalize_person(person),
        "summary": {
            "faces_count": sum(item["summary"]["faces_count"] for item in medias),
            "frames_count": sum(item["summary"]["frames_count"] for item in medias),
            "media_count": len(medias),
        },
        "medias": medias,
    }


def get_person_candidates(db: Session, name: str):
    rows = (
        db.query(
            DBPerson.id.label("person_id"),
            DBPerson.name.label("name"),
            DBPerson.q_code.label("q_code"),
            DBPerson.link.label("link"),
            DBPerson.status.label("status"),
            func.count(DBFace.id).label("faces_count"),
            func.count(func.distinct(DBFreeze.id)).label("frames_count"),
            func.count(func.distinct(DBFreeze.media_id)).label("media_count"),
        )
        .outerjoin(DBFace, DBFace.person_id == DBPerson.id)
        .outerjoin(DBFreeze, DBFace.freeze_id == DBFreeze.id)
        .filter(DBPerson.name.ilike(f"%{name}%"))
        .group_by(
            DBPerson.id,
            DBPerson.name,
            DBPerson.q_code,
            DBPerson.link,
            DBPerson.status,
        )
        .order_by(DBPerson.name)
        .all()
    )

    return [
        {
            "person": {
                "id": row.person_id,
                "name": row.name,
                "status": row.status.value if row.status else None,
                "q_code": row.q_code,
                "link": row.link,
            },
            "summary": {
                "faces_count": int(row.faces_count or 0),
                "frames_count": int(row.frames_count or 0),
                "media_count": int(row.media_count or 0),
            },
        }
        for row in rows
    ]


@router.get("/persons")
def search_persons_by_name(
    name: str = Query(..., min_length=2),
    db: Session = Depends(get_db),
):
    candidates = get_person_candidates(db, name)

    if not candidates:
        return {
            "mode": "no_results",
            "query": name,
            "summary": {
                "persons_count": 0,
            },
            "candidates": [],
            "result": None,
        }

    if len(candidates) == 1:
        person_id = candidates[0]["person"]["id"]
        person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        return {
            "mode": "single_result",
            "query": name,
            "summary": {
                "persons_count": 1,
            },
            "candidates": candidates,
            "result": build_person_result(db, person),
        }

    return {
        "mode": "multiple_results",
        "query": name,
        "summary": {
            "persons_count": len(candidates),
        },
        "candidates": candidates,
        "result": None,
    }


@router.get("/persons/{person_id}")
def get_person_search_result(
    person_id: int,
    db: Session = Depends(get_db),
):
    person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    return {
        "mode": "person_result",
        "query": None,
        "summary": {
            "persons_count": 1,
        },
        "candidates": [],
        "result": build_person_result(db, person),
    }
