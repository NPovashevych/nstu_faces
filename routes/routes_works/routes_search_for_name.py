from pathlib import Path
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import Date, cast, func
from sqlalchemy.orm import Session, joinedload

from db.session import get_db
from db.models import DBPerson, DBFace, DBFreeze, DBMedia, DBMediaDescription, DBEmbedding
from db.enums import PersonStatus

from commons.commons_base import make_image_url
from commons.common_search import safe_float, format_time, load_media_descriptions_for_medias
from commons.common_search import get_confidence_marks, get_source_label, get_media_url
from commons.common_search import normalize_person, normalize_bbox, normalize_media_description

router = APIRouter(prefix="/search-name", tags=["search by name"])


def normalize_face_preview(face: DBFace):
    bbox_data = normalize_bbox(face.bbox)

    return {
        "face_id": face.id,
        "bbox_draw": bbox_data["bbox_draw"],
        "category": face.face_category.name if face.face_category else None,
        "quality": safe_float(face.quality),
        "gender": face.gender.value if face.gender else None,
        "confidence": face.confidence,
        "confidence_marks": get_confidence_marks(face.confidence),
        "frame_color": "green",
    }


def get_description_dates_subquery(db: Session):
    return (
        db.query(
            DBMediaDescription.material_id.label("material_id"),
            func.min(DBMediaDescription.shooting_date).label("shooting_date"),
        )
        .group_by(DBMediaDescription.material_id)
        .subquery()
    )


def get_effective_media_date(description_dates):
    return func.coalesce(
        cast(func.nullif(description_dates.c.shooting_date, ""), Date),
        cast(DBMedia.recorded_at, Date),
    )


def get_person_summary(db: Session, person_id: int):
    description_dates = get_description_dates_subquery(db)
    effective_date = get_effective_media_date(description_dates)

    row = (
        db.query(
            func.count(DBFace.id).label("faces_count"),
            func.count(func.distinct(DBFace.freeze_id)).label("frames_count"),
            func.count(func.distinct(DBFreeze.media_id)).label("media_count"),
            func.min(effective_date).label("first_date"),
            func.max(effective_date).label("last_date"),
        )
        .select_from(DBFace)
        .outerjoin(DBFreeze, DBFace.freeze_id == DBFreeze.id)
        .outerjoin(DBMedia, DBFreeze.media_id == DBMedia.id)
        .outerjoin(description_dates, description_dates.c.material_id == DBMedia.material_id)
        .filter(DBFace.person_id == person_id)
        .first()
    )

    return {
        "faces_count": int(row.faces_count or 0),
        "frames_count": int(row.frames_count or 0),
        "media_count": int(row.media_count or 0),
        "first_date": row.first_date.isoformat() if row.first_date else None,
        "last_date": row.last_date.isoformat() if row.last_date else None,
    }


def get_person_medias_page(db, person_id, limit, offset, faces_sort, duration_sort, date_sort, date_from: date | None = None, date_to: date | None = None):
    description_dates = get_description_dates_subquery(db)
    effective_date = get_effective_media_date(description_dates)

    faces_count = func.count(DBFace.id)
    frames_count = func.count(func.distinct(DBFreeze.id))

    query = (
        db.query(DBMedia.id.label("media_id"), effective_date.label("date"), faces_count.label("faces_count"), frames_count.label("frames_count"))
        .select_from(DBFace)
        .join(DBFreeze, DBFace.freeze_id == DBFreeze.id)
        .join(DBMedia, DBFreeze.media_id == DBMedia.id)
        .outerjoin(description_dates, description_dates.c.material_id == DBMedia.material_id)
        .filter(DBFace.person_id == person_id)
    )

    if date_from is not None:
        query = query.filter(effective_date >= date_from)

    if date_to is not None:
        query = query.filter(effective_date <= date_to)

    query = query.group_by(DBMedia.id, effective_date)

    total = query.count()

    # if sort == "faces_asc":
    #     query = query.order_by(faces_count.asc(), DBMedia.id)
    # else:
    #     query = query.order_by(faces_count.desc(), DBMedia.id)

    faces_order = faces_count.asc() if faces_sort == "asc" else faces_count.desc()
    duration_order = DBMedia.duration.asc().nullslast() if duration_sort == "asc" else DBMedia.duration.desc().nullslast()
    date_order = effective_date.asc().nullslast() if date_sort == "asc" else effective_date.desc().nullslast()

    query = query.order_by(faces_order, duration_order, date_order, DBMedia.id)

    rows = query.offset(offset).limit(limit).all()

    medias = [
        {
            "media_id": row.media_id,
            "date": row.date.isoformat() if row.date else None,
            "faces_count": int(row.faces_count or 0),
            "frames_count": int(row.frames_count or 0),
        }
        for row in rows
    ]

    return medias, total


def build_person_medias_preview(db: Session, medias_page: list[dict]):
    if not medias_page:
        return []

    media_ids = [item["media_id"] for item in medias_page]

    medias = db.query(DBMedia).options(joinedload(DBMedia.source)).filter(DBMedia.id.in_(media_ids)).all()

    medias_by_id = {media.id: media for media in medias}
    descriptions_by_material_id = load_media_descriptions_for_medias(db=db, medias=medias)

    result = []

    for page_item in medias_page:
        media = medias_by_id.get(page_item["media_id"])

        if not media:
            continue

        description = descriptions_by_material_id.get(media.material_id)

        result.append(
            {
                "date": page_item["date"],
                "media": {
                    "id": media.id,
                    "material_id": media.material_id,
                    "name": Path(media.mp4_path or media.mxf_path or "").name,
                    "media_type": media.media_type.value if media.media_type else None,
                    "source": get_source_label(media),
                    "url": get_media_url(media),
                    "duration": safe_float(media.duration, None),
                    "description": normalize_media_description(media=media, description=description),
                },
                "summary": {"frames_count": page_item["frames_count"], "faces_count": page_item["faces_count"]},
            }
        )

    return result


def build_person_medias_details(db: Session, person_id: int, media_ids: list[int]):
    if not media_ids:
        return []

    faces = (
        db.query(DBFace)
        .options(joinedload(DBFace.face_category), joinedload(DBFace.freeze).joinedload(DBFreeze.media).joinedload(DBMedia.source))
        .join(DBFreeze, DBFace.freeze_id == DBFreeze.id)
        .join(DBMedia, DBFreeze.media_id == DBMedia.id)
        .filter(DBFace.person_id == person_id, DBMedia.id.in_(media_ids))
        .order_by(DBMedia.id, DBFreeze.time_in, DBFace.id)
        .all()
    )

    medias_by_id = {}

    for face in faces:
        media = face.freeze.media
        medias_by_id[media.id] = media

    descriptions_by_material_id = load_media_descriptions_for_medias(db=db, medias=list(medias_by_id.values()))

    medias_map = {}

    for face in faces:
        freeze = face.freeze
        media = freeze.media

        if media.id not in medias_map:
            description = descriptions_by_material_id.get(media.material_id)

            medias_map[media.id] = {
                "media": {
                    "id": media.id,
                    "material_id": media.material_id,
                    "name": Path(media.mp4_path or media.mxf_path or "").name,
                    "source": get_source_label(media),
                    "url": get_media_url(media),
                    "mxf_path": media.mxf_path,
                    "mp4_path": media.mp4_path,
                    "duration": safe_float(media.duration, None),
                    "recorded_at": media.recorded_at.isoformat() if media.recorded_at else None,
                    "uploaded_at": media.uploaded_at.isoformat() if media.uploaded_at else None,
                    "description": normalize_media_description(media=media, description=description),
                },
                "summary": {"frames_count": 0, "faces_count": 0},
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

        media_item["frames_map"][freeze.id]["faces"].append(normalize_face_preview(face))
        media_item["summary"]["faces_count"] += 1

    result = []

    for media_id in media_ids:
        item = medias_map.get(media_id)

        if not item:
            continue

        frames = list(item["frames_map"].values())
        item["summary"]["frames_count"] = len(frames)

        result.append({"media": item["media"], "summary": item["summary"], "frames": frames})

    return result


def get_person_candidates(db: Session, name: str):
    rows = (
        db.query(
            DBPerson.id.label("person_id"),
            DBPerson.code.label("code"),
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
        .filter(DBPerson.name.ilike(f"%{name}%"),  DBPerson.status.in_([PersonStatus.public, PersonStatus.non_public]))
        .group_by(DBPerson.id, DBPerson.code, DBPerson.name, DBPerson.q_code, DBPerson.link, DBPerson.status)
        .order_by(DBPerson.name)
        .all()
    )

    return [
        {
            "person": {"id": row.person_id, "code": row.code, "name": row.name, "status": row.status.value if row.status else None, "q_code": row.q_code, "link": row.link},
            "summary": {"faces_count": int(row.faces_count or 0), "frames_count": int(row.frames_count or 0), "media_count": int(row.media_count or 0)},
        }
        for row in rows
    ]


@router.get("/persons")
def search_persons_by_name(name: str = Query(..., min_length=2), db: Session = Depends(get_db)):
    candidates = get_person_candidates(db, name)

    if not candidates:
        return {
            "mode": "no_results",
            "query": name,
            "summary": {"persons_count": 0},
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
            "summary": {"persons_count": 1},
            "candidates": candidates,
            "result": {"person": normalize_person(person), "summary": get_person_summary(db, person.id)},
        }

    return {
        "mode": "multiple_results",
        "query": name,
        "summary": {"persons_count": len(candidates)},
        "candidates": candidates,
        "result": None,
    }


@router.get("/persons/{person_id}")
def get_person_search_result(person_id: int, db: Session = Depends(get_db)):
    person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    return {
        "mode": "person_result",
        "query": None,
        "summary": {"persons_count": 1},
        "candidates": [],
        "result": {"person": normalize_person(person), "summary": get_person_summary(db, person.id)},
    }


@router.get("/persons/{person_id}/medias")
def get_person_medias(
        person_id: int, limit: int = Query(10, ge=1, le=30),
        offset: int = Query(0, ge=0),
        # sort: str = Query("faces_desc", pattern="^(faces_desc|faces_asc)$"),
        faces_sort: str = Query("desc", pattern="^(asc|desc)$"),
        duration_sort: str = Query("asc", pattern="^(asc|desc)$"),
        date_sort: str = Query("desc", pattern="^(asc|desc)$"),
        date_from: date | None = Query(None),
        date_to: date | None = Query(None),
        db: Session = Depends(get_db)
):
    person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    if date_from is not None and date_to is not None and date_from > date_to:
        raise HTTPException(status_code=400, detail="date_from cannot be later than date_to")

    medias, total = get_person_medias_page(db=db, person_id=person_id, limit=limit, offset=offset, faces_sort=faces_sort, duration_sort=duration_sort, date_sort=date_sort,date_from=date_from, date_to=date_to)
    media_previews = build_person_medias_preview(db=db, medias_page=medias)

    return {
        "person": normalize_person(person),
        "pagination": {"limit": limit, "offset": offset, "returned": len(media_previews), "total": total, "has_more": offset + len(media_previews) < total},
        "sort": {"faces": faces_sort, "duration": duration_sort, "date": date_sort},
        "medias": media_previews,
    }


@router.get("/persons/{person_id}/medias/{media_id}")
def get_person_media_details(person_id: int, media_id: int, db: Session = Depends(get_db)):
    person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    media = db.query(DBMedia).filter(DBMedia.id == media_id).first()

    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    media_details = build_person_medias_details(db=db, person_id=person_id, media_ids=[media_id])

    if not media_details:
        raise HTTPException(status_code=404, detail="Person not found in media")

    return {"person": normalize_person(person), "result": media_details[0]}


@router.get("/faces/{face_id}/analysis")
def get_face_analysis(face_id: int, db: Session = Depends(get_db)):
    face = (
        db.query(DBFace)
        .options(joinedload(DBFace.freeze).joinedload(DBFreeze.media))
        .filter(DBFace.id == face_id)
        .first()
    )

    if not face:
        raise HTTPException(status_code=404, detail="Face not found")

    embedding = None

    if face.embedding_id:
        embedding = db.query(DBEmbedding).filter(DBEmbedding.id == face.embedding_id).first()

    freeze = face.freeze
    media = freeze.media if freeze else None

    distance = None
    if embedding and embedding.source:
        distance = embedding.source.get("distance")

    return {
        "face_id": face.id,
        "freeze_id": freeze.id if freeze else None,
        "media_id": media.id if media else None,
        "distance": distance,
        "analysis": face.analysis or {},
    }
