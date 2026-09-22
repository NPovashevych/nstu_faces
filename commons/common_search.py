from pathlib import Path
import numpy as np
from sqlalchemy.orm import Session, joinedload

from db.models import DBPerson, DBFace, DBFreeze, DBMedia, DBMediaDescription
from commons.commons_base import make_image_url, make_media_url


# router = APIRouter(prefix="/search-name", tags=["search by name"])


BBOX_DRAW_SCALE = 1.10


def get_media_url(media: DBMedia):
    if not media.mp4_path:
        return None

    if media.media_type.value != "video":
        return None

    try:
        return make_media_url(media.mp4_path)
    except ValueError:
        return None


def format_time(seconds) -> str:
    seconds = safe_float(seconds, 0.0)
    total = int(seconds)

    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60

    return f"{h:02}:{m:02}:{s:02}"



def safe_float(value, default=None):
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default



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
        "bbox": [x1, y1, x2, y2],
        "bbox_draw": {
            "x": center_x - draw_width / 2,
            "y": center_y - draw_height / 2,
            "w": draw_width,
            "h": draw_height,
        },
    }


def normalize_embedding(embedding) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def normalize_face(face: DBFace):
    bbox_data = normalize_bbox(face.bbox)

    return {
        "face_id": face.id,
        "bbox": bbox_data["bbox"],
        "bbox_draw": bbox_data["bbox_draw"],
        "category": face.face_category.name if face.face_category else None,
        "category_group": face.face_category.code if face.face_category else None,
        "category_score": face.category_score,
        "quality": face.quality,
        "gender": face.gender.value if face.gender else None,
        "confidence": face.confidence,
        "confidence_marks": get_confidence_marks(face.confidence),
        "frame_color": "green",
        "analysis": face.analysis or {},
    }




def get_confidence_marks(confidence):
    if confidence is None or confidence == 0:
        return ""

    return "?" * int(confidence)


def normalize_media_description(media: DBMedia, description: DBMediaDescription | None):
    duration = safe_float(media.duration, None)

    if description is None:
        return None

    return {
        "date": description.shooting_date,
        "section": description.section,
        "journalist": description.journalist,
        "operators": description.operators,
        "duration": duration,
        "description": description.description,
        "another_info": description.another_info,
    }


def normalize_person(person: DBPerson):
    return {
        "id": person.id,
        "code": person.code,
        "name": person.name,
        "status": person.status.value if person.status else None,
        "q_code": person.q_code,
        "link": person.link,
        "cluster_id": person.cluster_id,
        "cluster_tag": person.cluster_tag,
    }


def get_source_label(media: DBMedia):
    if media.source:
        return media.source.name

    return None


def load_media_descriptions_for_medias(db: Session, medias: list[DBMedia]):
    material_ids = list({media.material_id for media in medias if media.material_id})

    if not material_ids:
        return {}

    rows = (
        db.query(DBMediaDescription)
        .filter(DBMediaDescription.material_id.in_(material_ids))
        .order_by(DBMediaDescription.material_id, DBMediaDescription.id)
        .all()
    )

    descriptions = {}

    for row in rows:
        if row.material_id not in descriptions:
            descriptions[row.material_id] = row

    return descriptions


def build_person_result(db: Session, person: DBPerson):
    faces = (
        db.query(DBFace)
        .options(joinedload(DBFace.face_category), joinedload(DBFace.freeze).joinedload(DBFreeze.media).joinedload(DBMedia.source))
        .join(DBFreeze, DBFace.freeze_id == DBFreeze.id)
        .join(DBMedia, DBFreeze.media_id == DBMedia.id)
        .filter(DBFace.person_id == person.id)
        .order_by(DBMedia.id, DBFreeze.time_in, DBFace.id)
        .all()
    )

    medias_by_id = {}

    for face in faces:
        freeze = face.freeze
        media = freeze.media
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
                    "description": normalize_media_description(media=media, description=description)
                },
                "summary": {"frames_count": 0,"faces_count": 0},
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

        medias.append({"media": item["media"], "summary": item["summary"], "frames": frames})

    return {
        "person": normalize_person(person),
        "summary": {
            "faces_count": sum(item["summary"]["faces_count"] for item in medias),
            "frames_count": sum(item["summary"]["frames_count"] for item in medias),
            "media_count": len(medias),
        },
        "medias": medias,
    }