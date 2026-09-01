from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from routes.routers_classic.commons import similarity_percent_from_distance, normalize, cosine_distance
from routes.routers_classic.commons import load_reference_embeddings, make_image_url
from db.session import get_db
from db.models import DBPerson, DBFace, DBFreeze, DBMedia, DBEmbedding, DBFaceCategory
from schemas.schemas_request import UpdateClusterRequest, AssignToPersonRequest, MoveFacesRequest
from schemas.schemas_request import SplitClusterRequest, MergeClustersRequest
from db.enums import PersonStatus


router = APIRouter(prefix="/face-clusters", tags=["face clusters"])


BBOX_DRAW_SCALE = 1.10
STRONG_HINT_DISTANCE = 0.45
MEDIUM_HINT_DISTANCE = 0.55
MAX_PAGE_SIZE = 200


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


def get_media_path(media: DBMedia) -> str:
    return media.mp4_path or media.mxf_path or ""


def get_media_name(media: DBMedia) -> str:
    path = get_media_path(media)

    return Path(path).name if path else f"media_{media.id}"


def get_media_source(media: DBMedia):
    source = getattr(media, "source", None)

    if source is None:
        return None

    return {
        "id": getattr(source, "id", None),
        "code": getattr(source, "code", None),
        "name": getattr(source, "name", None),
    }


def get_person_status(person: Optional[DBPerson]):
    if not person or not person.status:
        return None

    return enum_value(person.status)


def get_face_category_code(face: DBFace) -> str:
    """
    Для програмної логіки використовуємо стабільне технічне поле code.
    """

    if face.face_category and face.face_category.code:
        return face.face_category.code

    return "uncertain"


def get_face_category_name(face: DBFace) -> str:
    """
    Поле name передаємо на фронт як назву категорії.
    """

    if face.face_category and face.face_category.name:
        return face.face_category.name

    return "uncertain"


def normalize_bbox(bbox, scale: float = BBOX_DRAW_SCALE):
    bbox = bbox or [0, 0, 0, 0]

    # Захищаємося від неповного або пошкодженого bbox.
    if len(bbox) < 4:
        bbox = [0, 0, 0, 0]

    x1 = safe_float(bbox[0], 0.0)
    y1 = safe_float(bbox[1], 0.0)
    x2 = safe_float(bbox[2], 0.0)
    y2 = safe_float(bbox[3], 0.0)

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

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


def get_face_color(face: DBFace) -> str:
    category_code = get_face_category_code(face)

    if category_code in {
        "low_quality",
        "unidentifiable",
        "real_unidentifiable",
    }:
        return "gray"

    if category_code in {
        "non_human",
        "artificial",
        "artificial_human",
        "ai_generated",
        "uncertain",
    }:
        return "orange"

    person_status = get_person_status(face.person)

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


def get_confidence(face: DBFace):
    person_status = get_person_status(face.person)

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


def normalize_cluster_key(cluster_key: str) -> list[str]:
    cluster_key = cluster_key.strip()
    variants = {cluster_key}

    if cluster_key.isdigit():
        number = int(cluster_key)

        variants.add(str(number))
        variants.add(f"{number:06d}")
        variants.add(f"unknown_cluster_{number}")
        variants.add(f"unknown_cluster_{number:06d}")

    if not cluster_key.endswith("_cluster"):
        variants.add(f"{cluster_key}_cluster")

    return list(variants)


def find_cluster_person(db: Session, cluster_key: str):
    """
    Для прямого відкриття кластера використовуємо:
    - унікальний code;
    - cluster_id, якщо передано число.

    name і cluster_tag тут не використовуємо, оскільки вони не унікальні.
    Вони залишаються доступними для текстового пошуку в /list.
    """

    clean_key = cluster_key.strip()
    variants = normalize_cluster_key(clean_key)

    conditions = [
        DBPerson.code.in_(variants),
    ]

    if clean_key.isdigit():
        conditions.append(DBPerson.cluster_id == int(clean_key))

    return (
        db.query(DBPerson)
        .filter(or_(*conditions))
        .first()
    )


def require_cluster_person(db: Session, cluster_key: str, allow_service_cluster: bool = True):
    person = find_cluster_person(db, cluster_key)

    if not person:
        raise HTTPException(status_code=404, detail="Cluster not found")

    status = get_person_status(person)
    allowed_statuses = {PersonStatus.unknown.value}

    if allow_service_cluster:
        allowed_statuses.add(PersonStatus.suspicious.value)

    if status not in allowed_statuses:
        raise HTTPException(status_code=400, detail="Selected person is not a face cluster")

    return person


def require_known_person(db: Session, person_id: int):
    person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if not person:
        raise HTTPException(status_code=404, detail="Target person not found")

    status = get_person_status(person)

    if status not in {
        PersonStatus.public.value,
        PersonStatus.non_public.value,
    }:
        raise HTTPException(status_code=400, detail="Target person must have public or non_public status")

    return person


def get_cluster_faces_query(db: Session, person_id: int):
    """
    Embedding завантажуємо одразу, оскільки при перенесенні обличчя
    необхідно також змінити DBEmbedding.person_id.
    """

    return (
        db.query(DBFace)
        .options(joinedload(DBFace.embedding))
        .filter(DBFace.person_id == person_id)
    )


def get_selected_cluster_faces(db: Session, cluster_person_id: int, face_ids: Optional[list[int]]):
    query = get_cluster_faces_query(db=db, person_id=cluster_person_id)

    if face_ids:
        unique_face_ids = list(set(face_ids))
        query = query.filter(DBFace.id.in_(unique_face_ids))

    faces = query.all()

    if not faces:
        raise HTTPException(status_code=404, detail="No matching faces found in this cluster")

    if face_ids:
        requested_ids = set(face_ids)
        found_ids = {face.id for face in faces}
        missing_ids = sorted(requested_ids - found_ids)

        if missing_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Обличчя не належать до цього кластера",
                    "face_ids": missing_ids,
                },
            )

    return faces


def count_person_faces(db: Session, person_id: int) -> int:
    return int(
        db.query(func.count(DBFace.id))
        .filter(DBFace.person_id == person_id)
        .scalar()
        or 0
    )


def assign_face_to_person(face: DBFace, person_id: int, confidence=None):
    """
    Одночасно переприв'язує:
    - запис DBFace;
    - embedding цього обличчя.

    DBFace.person_id та DBEmbedding.person_id повинні завжди
    посилатися на ту саму персону.
    """

    face.person_id = person_id
    face.confidence = confidence

    if face.embedding:
        face.embedding.person_id = person_id


def append_archivist_action(face: DBFace, action: dict):
    """
    Зберігає не тільки останню дію архівіста, а повну історію дій
    над конкретним обличчям у полі analysis.archivist_actions.
    """

    analysis = dict(face.analysis or {})
    actions = list(analysis.get("archivist_actions") or [])

    action_data = dict(action)
    action_data["created_at"] = datetime.now(timezone.utc).isoformat()

    actions.append(action_data)

    analysis["archivist_actions"] = actions
    face.analysis = analysis


def delete_person_if_empty(db: Session, person: DBPerson, enabled: bool) -> bool:
    if not enabled:
        return False

    # Автоматично видаляємо тільки звичайний unknown-кластер.
    # Службові suspicious-кластери повинні існувати постійно.
    if get_person_status(person) != PersonStatus.unknown.value:
        return False

    faces_count = count_person_faces(db, person.id)

    if faces_count > 0:
        return False

    db.delete(person)

    return True


def generate_unknown_cluster_code(person_id: int) -> str:
    return f"unknown_cluster_{person_id:06d}"


def create_unknown_cluster_person(db: Session, name: Optional[str] = None, cluster_tag: Optional[str] = None):
    temporary_code = f"temporary_unknown_{uuid4().hex}"

    person = DBPerson(
        code=temporary_code,
        name=name or temporary_code,
        status=PersonStatus.unknown,
        cluster_tag=cluster_tag,
        cluster_distance=None,
    )

    db.add(person)
    db.flush()

    final_code = generate_unknown_cluster_code(person.id)

    person.code = final_code

    if not name:
        person.name = final_code

    person.cluster_id = person.id

    return person


def find_nearest_known(face_embedding: Optional[DBEmbedding], reference_embeddings: list[dict]):
    if face_embedding is None or not reference_embeddings:
        return None

    if not face_embedding.vector:
        return None

    emb = normalize(np.array(face_embedding.vector, dtype=np.float32))

    best = None
    best_dist = float("inf")

    for ref in reference_embeddings:
        ref_vector = ref.get("vector")

        if ref_vector is None:
            continue

        dist = cosine_distance(emb, ref_vector)

        if dist < best_dist:
            best_dist = dist
            best = ref

    if best is None:
        return None

    if best_dist <= STRONG_HINT_DISTANCE:
        hint_level = "strong"
        hint_label = "можливо це"
    elif best_dist <= MEDIUM_HINT_DISTANCE:
        hint_level = "medium"
        hint_label = "схожий на"
    else:
        return None

    return {
        "person": {
            "id": best.get("person_id"),
            "name": best.get("person_name") or best.get("name"),
            "code": best.get("person_code"),
            "status": best.get("person_status"),
            "q_code": best.get("q_code"),
            "link": best.get("link"),
        },
        "distance": round(float(best_dist), 4),
        "similarity_percent": safe_float(
            similarity_percent_from_distance(best_dist)
        ),
        "hint_level": hint_level,
        "hint_label": hint_label,
    }


def normalize_cluster(person: DBPerson, faces_count: Optional[int] = None):
    result = {
        "person_id": person.id,
        "name": person.name,
        "code": person.code,
        "status": get_person_status(person),
        "cluster_id": person.cluster_id,
        "cluster_tag": person.cluster_tag,
        "cluster_distance": safe_float(person.cluster_distance),
    }

    if faces_count is not None:
        result["faces_count"] = int(faces_count)

    return result


def normalize_face_item(face: DBFace, reference_embeddings: list[dict]):
    freeze = face.freeze
    media = freeze.media if freeze else None

    # Обличчя без freeze або media не можна коректно показати.
    if not freeze or not media:
        return None

    freeze_path = freeze.freeze_path or ""
    freeze_exists = bool(freeze_path and Path(freeze_path).exists())

    category_code = get_face_category_code(face)
    category_name = get_face_category_name(face)

    nearest_known = None

    if category_code in {
        "identifiable",
        "real_identifiable",
    }:
        nearest_known = find_nearest_known(
            face_embedding=face.embedding,
            reference_embeddings=reference_embeddings,
        )

    confidence = get_confidence(face)

    return {
        "face_id": face.id,
        "freeze_id": freeze.id,

        "image": {
            "url": make_image_url(freeze_path) if freeze_path else None,
            "exists": freeze_exists,
            "path": freeze_path,
        },

        "media": {
            "id": media.id,
            "material_id": media.material_id,
            "name": get_media_name(media),
            "path": get_media_path(media),
            "mxf_path": media.mxf_path,
            "mp4_path": media.mp4_path,
            "media_type": enum_value(media.media_type),
            "duration": safe_float(media.duration),
            "source": get_media_source(media),
        },

        "time": {
            "in": safe_float(freeze.time_in, 0.0),
            "out": safe_float(freeze.time_out, 0.0),
        },

        "person": {
            "id": face.person_id,
            "name": face.person.name if face.person else None,
            "code": face.person.code if face.person else None,
            "status": get_person_status(face.person),
        },

        "recognition": {
            "confidence": confidence,
            "confidence_marks": get_confidence_marks(confidence),
            "nearest_known": nearest_known,
        },

        "category": {
            "id": face.category_id,
            "code": category_code,
            "name": category_name,
            "score": safe_float(face.category_score),
        },

        "quality": safe_float(face.quality),
        "gender": enum_value(face.gender),
        "color": get_face_color(face),
        "bbox": normalize_bbox(face.bbox),
        "analysis": face.analysis or {},
    }


@router.get("/list")
def get_face_clusters_list(
    status: Optional[str] = Query(default=PersonStatus.unknown.value, description="unknown, suspicious або all"),
    search: Optional[str] = Query(default=None, description="Пошук за name, code або cluster_tag"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
    db: Session = Depends(get_db),
):
    faces_count_subquery = (
        db.query(
            DBFace.person_id.label("person_id"),
            func.count(DBFace.id).label("faces_count"),
        )
        .group_by(DBFace.person_id)
        .subquery()
    )

    query = (
        db.query(
            DBPerson,
            faces_count_subquery.c.faces_count,
        )
        .join(
            faces_count_subquery,
            faces_count_subquery.c.person_id == DBPerson.id,
        )
    )

    if status == "all":
        query = query.filter(
            DBPerson.status.in_(
                [
                    PersonStatus.unknown,
                    PersonStatus.suspicious,
                ]
            )
        )
    elif status == PersonStatus.unknown.value:
        query = query.filter(DBPerson.status == PersonStatus.unknown)
    elif status == PersonStatus.suspicious.value:
        query = query.filter(DBPerson.status == PersonStatus.suspicious)
    else:
        raise HTTPException(status_code=400, detail="status must be unknown, suspicious or all")

    if search:
        pattern = f"%{search.strip()}%"

        query = query.filter(
            or_(
                DBPerson.name.ilike(pattern),
                DBPerson.code.ilike(pattern),
                DBPerson.cluster_tag.ilike(pattern),
            )
        )

    total = query.count()

    rows = (
        query
        .order_by(
            faces_count_subquery.c.faces_count.desc(),
            DBPerson.id,
        )
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "pagination": {
            "offset": offset,
            "limit": limit,
            "total": total,
        },
        "items": [
            normalize_cluster(
                person=person,
                faces_count=faces_count,
            )
            for person, faces_count in rows
        ],
    }


@router.get("/{cluster_key}")
def get_face_cluster(
    cluster_key: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=MAX_PAGE_SIZE),
    include_nearest_known: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    person = require_cluster_person(db=db, cluster_key=cluster_key, allow_service_cluster=True)

    total_faces = count_person_faces(db, person.id)

    reference_embeddings = []

    if include_nearest_known and get_person_status(person) == PersonStatus.unknown.value:
        reference_embeddings = load_reference_embeddings(db)

    faces = (
        db.query(DBFace)
        .options(
            joinedload(DBFace.person),
            joinedload(DBFace.embedding),
            joinedload(DBFace.face_category),
            joinedload(DBFace.freeze)
            .joinedload(DBFreeze.media)
            .joinedload(DBMedia.source),
        )
        .filter(DBFace.person_id == person.id)
        .order_by(DBFace.freeze_id, DBFace.id)
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = []

    for face in faces:
        item = normalize_face_item(
            face=face,
            reference_embeddings=reference_embeddings,
        )

        if item:
            items.append(item)

    category_rows = (
        db.query(
            DBFaceCategory.id.label("category_id"),
            DBFaceCategory.code.label("category_code"),
            DBFaceCategory.name.label("category_name"),
            func.count(DBFace.id).label("faces_count"),
        )
        .join(
            DBFace,
            DBFace.category_id == DBFaceCategory.id,
        )
        .filter(DBFace.person_id == person.id)
        .group_by(
            DBFaceCategory.id,
            DBFaceCategory.code,
            DBFaceCategory.name,
        )
        .order_by(DBFaceCategory.id)
        .all()
    )

    categories_count = {
        row.category_code: int(row.faces_count)
        for row in category_rows
    }

    categories = [
        {
            "id": row.category_id,
            "code": row.category_code,
            "name": row.category_name,
            "faces_count": int(row.faces_count),
        }
        for row in category_rows
    ]

    return {
        "cluster": normalize_cluster(
            person=person,
            faces_count=total_faces,
        ),

        "pagination": {
            "offset": offset,
            "limit": limit,
            "total": total_faces,
            "returned": len(items),
        },

        "summary": {
            "faces_count": total_faces,
            "categories_count": categories_count,
            "categories": categories,
        },

        "items": items,
    }


@router.patch("/{cluster_key}")
def update_face_cluster(
    cluster_key: str,
    payload: UpdateClusterRequest,
    db: Session = Depends(get_db),
):
    """
    Змінює зрозумілу назву та тег unknown-кластера.
    Системний code не змінюється.
    """

    person = require_cluster_person(db=db, cluster_key=cluster_key, allow_service_cluster=False)

    update_data = payload.model_dump(exclude_unset=True)

    if "name" in update_data:
        person.name = update_data["name"].strip()

    if "cluster_tag" in update_data:
        cluster_tag = update_data["cluster_tag"]
        person.cluster_tag = cluster_tag.strip() if cluster_tag else None

    try:
        db.commit()
        db.refresh(person)

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(status_code=409, detail="Could not update cluster") from exc

    return {
        "message": "Cluster updated",
        "cluster": normalize_cluster(
            person=person,
            faces_count=count_person_faces(db, person.id),
        ),
    }


@router.post("/{cluster_key}/assign-person")
def assign_cluster_faces_to_known_person(
    cluster_key: str,
    payload: AssignToPersonRequest,
    db: Session = Depends(get_db),
):
    source_cluster = require_cluster_person(db=db, cluster_key=cluster_key, allow_service_cluster=False)
    target_person = require_known_person(db=db, person_id=payload.person_id)

    if source_cluster.id == target_person.id:
        raise HTTPException(status_code=400, detail="Source and target person are the same")

    faces = get_selected_cluster_faces(
        db=db,
        cluster_person_id=source_cluster.id,
        face_ids=payload.face_ids,
    )

    try:
        for face in faces:
            assign_face_to_person(
                face=face,
                person_id=target_person.id,
                confidence=payload.confidence,
            )

            append_archivist_action(
                face=face,
                action={
                    "action": "assign_to_known_person",
                    "source_cluster_person_id": source_cluster.id,
                    "target_person_id": target_person.id,
                    "confidence": payload.confidence,
                },
            )

        # Записуємо нові person_id у БД до перевірки порожнього кластера.
        db.flush()

        source_deleted = delete_person_if_empty(
            db=db,
            person=source_cluster,
            enabled=payload.delete_empty_cluster,
        )

        db.commit()

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(status_code=409, detail="Database conflict while assigning faces") from exc

    except Exception:
        db.rollback()
        raise

    return {
        "message": "Faces assigned to known person",
        "moved_faces_count": len(faces),
        "face_ids": [face.id for face in faces],
        "target_person": {
            "id": target_person.id,
            "name": target_person.name,
            "code": target_person.code,
            "status": get_person_status(target_person),
        },
        "source_cluster_deleted": source_deleted,
    }


@router.post("/{cluster_key}/move-faces")
def move_faces_to_cluster(
    cluster_key: str,
    payload: MoveFacesRequest,
    db: Session = Depends(get_db),
):
    source_cluster = require_cluster_person(db=db, cluster_key=cluster_key, allow_service_cluster=True)
    target_cluster = require_cluster_person(db=db, cluster_key=payload.target_cluster_key, allow_service_cluster=True)

    if source_cluster.id == target_cluster.id:
        raise HTTPException(status_code=400, detail="Source and target clusters are the same")

    faces = get_selected_cluster_faces(
        db=db,
        cluster_person_id=source_cluster.id,
        face_ids=payload.face_ids,
    )

    try:
        for face in faces:
            assign_face_to_person(
                face=face,
                person_id=target_cluster.id,
                confidence=None,
            )

            append_archivist_action(
                face=face,
                action={
                    "action": "move_to_cluster",
                    "source_cluster_person_id": source_cluster.id,
                    "target_cluster_person_id": target_cluster.id,
                },
            )

        db.flush()

        source_deleted = delete_person_if_empty(
            db=db,
            person=source_cluster,
            enabled=payload.delete_empty_cluster,
        )

        db.commit()

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(status_code=409, detail="Database conflict while moving faces") from exc

    except Exception:
        db.rollback()
        raise

    return {
        "message": "Faces moved to target cluster",
        "moved_faces_count": len(faces),
        "face_ids": [face.id for face in faces],
        "target_cluster": normalize_cluster(
            person=target_cluster,
            faces_count=count_person_faces(db, target_cluster.id),
        ),
        "source_cluster_deleted": source_deleted,
    }


@router.post("/{cluster_key}/split")
def split_face_cluster(
    cluster_key: str,
    payload: SplitClusterRequest,
    db: Session = Depends(get_db),
):
    source_cluster = require_cluster_person(db=db, cluster_key=cluster_key, allow_service_cluster=False)

    faces = get_selected_cluster_faces(
        db=db,
        cluster_person_id=source_cluster.id,
        face_ids=payload.face_ids,
    )

    source_faces_count = count_person_faces(db, source_cluster.id)

    if len(faces) >= source_faces_count:
        raise HTTPException(status_code=400, detail="Split must leave at least one face in the source cluster")

    try:
        new_cluster = create_unknown_cluster_person(
            db=db,
            name=payload.name.strip() if payload.name else None,
            cluster_tag=payload.cluster_tag.strip() if payload.cluster_tag else None,
        )

        for face in faces:
            assign_face_to_person(
                face=face,
                person_id=new_cluster.id,
                confidence=None,
            )

            append_archivist_action(
                face=face,
                action={
                    "action": "split_to_new_cluster",
                    "source_cluster_person_id": source_cluster.id,
                    "new_cluster_person_id": new_cluster.id,
                },
            )

        db.flush()

        source_deleted = delete_person_if_empty(
            db=db,
            person=source_cluster,
            enabled=payload.delete_empty_cluster,
        )

        db.commit()
        db.refresh(new_cluster)

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(status_code=409, detail="Could not create a unique cluster") from exc

    except Exception:
        db.rollback()
        raise

    return {
        "message": "New cluster created",
        "moved_faces_count": len(faces),
        "face_ids": [face.id for face in faces],
        "new_cluster": normalize_cluster(
            person=new_cluster,
            faces_count=count_person_faces(db, new_cluster.id),
        ),
        "source_cluster_deleted": source_deleted,
    }


@router.post("/{cluster_key}/merge")
def merge_face_clusters(
    cluster_key: str,
    payload: MergeClustersRequest,
    db: Session = Depends(get_db),
):
    source_cluster = require_cluster_person(db=db, cluster_key=cluster_key, allow_service_cluster=False)
    target_cluster = require_cluster_person(db=db, cluster_key=payload.target_cluster_key, allow_service_cluster=True)

    if source_cluster.id == target_cluster.id:
        raise HTTPException(status_code=400, detail="Source and target clusters are the same")

    faces = (
        db.query(DBFace)
        .options(joinedload(DBFace.embedding))
        .filter(DBFace.person_id == source_cluster.id)
        .all()
    )

    if not faces:
        raise HTTPException(status_code=400, detail="Source cluster is empty")

    try:
        for face in faces:
            assign_face_to_person(
                face=face,
                person_id=target_cluster.id,
                confidence=None,
            )

            append_archivist_action(
                face=face,
                action={
                    "action": "merge_clusters",
                    "source_cluster_person_id": source_cluster.id,
                    "target_cluster_person_id": target_cluster.id,
                },
            )

        db.flush()

        source_deleted = delete_person_if_empty(
            db=db,
            person=source_cluster,
            enabled=payload.delete_source_cluster,
        )

        db.commit()

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(status_code=409, detail="Database conflict while merging clusters") from exc

    except Exception:
        db.rollback()
        raise

    return {
        "message": "Clusters merged",
        "moved_faces_count": len(faces),
        "target_cluster": normalize_cluster(
            person=target_cluster,
            faces_count=count_person_faces(db, target_cluster.id),
        ),
        "source_cluster_deleted": source_deleted,
    }


@router.delete("/{cluster_key}")
def delete_empty_face_cluster(cluster_key: str, db: Session = Depends(get_db)):
    """
    Видаляє тільки порожній unknown-кластер.
    Службові suspicious-кластери видаляти через цей роут не можна.
    """

    person = require_cluster_person(db=db, cluster_key=cluster_key, allow_service_cluster=False)
    faces_count = count_person_faces(db, person.id)

    if faces_count > 0:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Cluster cannot be deleted while it contains faces",
                "faces_count": faces_count,
            },
        )

    person_id = person.id
    person_code = person.code

    try:
        db.delete(person)
        db.commit()

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(status_code=409, detail="Cluster has related database records") from exc

    except Exception:
        db.rollback()
        raise

    return {
        "message": "Empty cluster deleted",
        "person_id": person_id,
        "code": person_code,
    }