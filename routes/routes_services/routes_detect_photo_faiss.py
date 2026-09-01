import io
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from insightface.app import FaceAnalysis
from sqlalchemy import func
from sqlalchemy.orm import Session

from db.session import get_db
from db.enums import (
    EmbeddingType,
    FaceGender,
    IterationStatus,
    MediaType,
    PersonStatus,
    UserRole,
)
from db.models import (
    DBFaceCategory,
    DBFreeze,
    DBMedia,
    DBPerson,
    DBSource,
    DBUser,
)

from crud.crud_embedding import create_embedding
from crud.crud_face import create_face
from crud.crud_history import log_history
from crud.crud_iteration import create_iteration, update_iteration
from crud.crud_person import create_person

from schemas.schemas_embedding import EmbeddingCreate
from schemas.schemas_face import FaceCreate
from schemas.schemas_iteration import IterationCreate, IterationUpdate
from schemas.schemas_person import PersonsCreate

from routes.routers_classic.commons import normalize
from routes.routes_services.routes_search_for_name import build_person_result

from services.config import USER_UPLOAD_FOLDER
from services.create_faces.face_quality_v3 import get_face_quality
from services.create_faces.clip_face_filter_v2 import (
    get_clip,
    analyze_face_category,
)
from services.create_faces.clip_face_categories import (
    DEFAULT_FACE_CATEGORY,
    CATEGORY_IDENTIFIABLE,
    CATEGORY_LOW_QUALITY,
)
from services.create_faces.faiss_face_index import (
    REFERENCE_FACE_INDEX,
    UNKNOWN_FACE_INDEX,
    ensure_faiss_indexes,
)


router = APIRouter(
    prefix="/detect-photo",
    tags=["detect photo"],
)


SERVICE_NAME = "routes_detect_photo_faiss.py"

FACE_DET_SIZE = 640
MIN_DET_SCORE = 0.60
DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.055
UNKNOWN_TOLERANCE = 0.55
LOW_QUALITY_THRESHOLD = 0.60

BBOX_DRAW_SCALE = 1.10

USER_UPLOAD_SOURCE_CODE = "test"
USER_UPLOAD_SOURCE_NAME = "user_upload"

_INSIGHTFACE_CACHE = None


# -------------------------------------------------------------------------
# INSIGHTFACE
# -------------------------------------------------------------------------

def load_insightface():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    app.prepare(
        ctx_id=0,
        det_size=(FACE_DET_SIZE, FACE_DET_SIZE),
    )

    return app


def get_insightface():
    global _INSIGHTFACE_CACHE

    if _INSIGHTFACE_CACHE is None:
        _INSIGHTFACE_CACHE = load_insightface()

    return _INSIGHTFACE_CACHE


# -------------------------------------------------------------------------
# COMMON HELPERS
# -------------------------------------------------------------------------

def can_save_uploaded_media(user: DBUser) -> bool:
    return user.role == UserRole.developer


def safe_float(value, default=None):
    if value is None:
        return default

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def normalize_bbox(
    bbox,
    scale: float = BBOX_DRAW_SCALE,
):
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


def get_confidence(dist: float) -> int:
    if dist <= DIST_TOLERANCE:
        return 0

    for i in range(1, 4):
        if dist <= DIST_TOLERANCE + i * STEP_TOLERANCE:
            return i

    return -1


def get_confidence_marks(confidence):
    if confidence is None or confidence == 0:
        return ""

    return "?" * int(confidence)


# -------------------------------------------------------------------------
# CATEGORIES
# -------------------------------------------------------------------------

def load_face_categories(db: Session):
    rows = (
        db.query(DBFaceCategory)
        .filter(DBFaceCategory.is_active.is_(True))
        .all()
    )

    return {
        row.name: row
        for row in rows
    }


def get_category(
    face_categories: dict,
    category_name: str,
) -> DBFaceCategory:

    category = face_categories.get(category_name)

    if category:
        return category

    fallback = face_categories.get(DEFAULT_FACE_CATEGORY)

    if fallback:
        return fallback

    raise RuntimeError(
        f"Face category '{category_name}' not found and fallback "
        f"'{DEFAULT_FACE_CATEGORY}' not found."
    )


def is_identifiable_category(
    category: DBFaceCategory,
) -> bool:

    return (
        category.name == CATEGORY_IDENTIFIABLE
        or category.code == CATEGORY_IDENTIFIABLE
    )


def is_identifiable_analysis(
    category: DBFaceCategory,
    analysis: dict,
) -> bool:

    if is_identifiable_category(category):
        return True

    clip = (analysis or {}).get(
        "clip",
        {},
    )

    best_clip_category = clip.get(
        "best_clip_category"
    )

    best_clip_score = safe_float(
        clip.get("best_clip_score"),
        0.0,
    )

    return (
        best_clip_category == CATEGORY_IDENTIFIABLE
        and best_clip_score >= 0.50
    )


# -------------------------------------------------------------------------
# GENDER / COLORS
# -------------------------------------------------------------------------

def map_gender(
    face,
    category: DBFaceCategory,
) -> FaceGender:

    if not is_identifiable_category(category):
        return FaceGender.unknown

    gender = getattr(
        face,
        "gender",
        None,
    )

    if gender == 1:
        return FaceGender.male

    if gender == 0:
        return FaceGender.female

    return FaceGender.unknown


def get_person_status_value(
    person: DBPerson | None,
):
    if not person or not person.status:
        return None

    return (
        person.status.value
        if hasattr(person.status, "value")
        else str(person.status)
    )


def get_face_color(
    category: DBFaceCategory | None,
    person: DBPerson | None,
) -> str:

    category_name = (
        category.name
        if category
        else None
    )

    category_code = (
        category.code
        if category
        else None
    )

    if (
        category_name in {
            "low_quality",
            "unidentifiable",
        }
        or (
            category_code == "human"
            and category_name == "unidentifiable"
        )
    ):
        return "gray"

    if category_name in {
        "non_human",
        "artificial",
        "ai_generated",
        "uncertain",
    }:
        return "orange"

    person_status = get_person_status_value(
        person
    )

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


# -------------------------------------------------------------------------
# ANALYSIS HELPERS
# -------------------------------------------------------------------------

def make_analysis(
    quality_details: dict,
    category_result: dict,
) -> dict:

    return {
        "quality": quality_details,
        "clip": {
            "category": category_result["category"],
            "category_score": category_result["category_score"],
            "best_clip_category": category_result["best_clip_category"],
            "best_clip_score": category_result["best_clip_score"],
            "clip_scores": category_result["clip_scores"],
        },
    }


def make_analysis_without_clip(
    quality_details: dict,
    reason: str,
    category_name: str,
) -> dict:

    return {
        "quality": quality_details,
        "clip": {
            "category": category_name,
            "category_score": None,
            "best_clip_category": reason,
            "best_clip_score": None,
            "clip_scores": None,
        },
    }


# -------------------------------------------------------------------------
# UNKNOWN / SERVICE PERSONS
# -------------------------------------------------------------------------

def get_next_cluster_number(
    db: Session,
    status: PersonStatus,
) -> int:

    max_cluster_id = (
        db.query(
            func.max(DBPerson.cluster_id)
        )
        .filter(
            DBPerson.status == status
        )
        .scalar()
    )

    return (max_cluster_id or 0) + 1


def create_unknown_cluster_person(
    db: Session,
):
    cluster_id = get_next_cluster_number(
        db,
        PersonStatus.unknown,
    )

    cluster_tag = (
        f"unknown_cluster_{cluster_id:06d}"
    )

    person_create = PersonsCreate(
        name=cluster_tag,
        q_code=None,
        link=None,
        status=PersonStatus.unknown,
    )

    db_person = create_person(
        db,
        person_create,
        code=cluster_tag,
    )

    db_person.cluster_id = cluster_id
    db_person.cluster_tag = cluster_tag
    db_person.cluster_distance = 0.0

    db.commit()
    db.refresh(db_person)

    return db_person


def get_or_create_service_cluster_person(
    db: Session,
    category_name: str,
):
    cluster_tag = (
        f"{category_name}_cluster"
    )

    db_person = (
        db.query(DBPerson)
        .filter(
            DBPerson.code == cluster_tag
        )
        .first()
    )

    if db_person:
        return db_person

    person_create = PersonsCreate(
        name=cluster_tag,
        q_code=None,
        link=None,
        status=PersonStatus.suspicious,
    )

    db_person = create_person(
        db,
        person_create,
        code=cluster_tag,
    )

    db_person.cluster_id = 0
    db_person.cluster_tag = cluster_tag
    db_person.cluster_distance = 0.0

    db.commit()
    db.refresh(db_person)

    return db_person


# -------------------------------------------------------------------------
# FAISS UNKNOWN SEARCH
# -------------------------------------------------------------------------

def find_existing_unknown_cluster_person(
    db: Session,
    embedding,
):
    ensure_faiss_indexes(db)

    person_id, best_dist = (
        UNKNOWN_FACE_INDEX.find_best_match(
            embedding
        )
    )

    if person_id is None or best_dist is None:
        return None, None

    if best_dist > UNKNOWN_TOLERANCE:
        return None, None

    db_person = (
        db.query(DBPerson)
        .filter(
            DBPerson.id == person_id
        )
        .first()
    )

    if not db_person:
        return None, None

    return db_person, best_dist


def find_or_create_unknown_cluster_person(
    db: Session,
    embedding,
):
    db_person, best_dist = (
        find_existing_unknown_cluster_person(
            db,
            embedding,
        )
    )

    if db_person:
        db_person.cluster_tag = (
            db_person.cluster_tag
            or db_person.name
        )

        db_person.cluster_distance = round(
            best_dist,
            4,
        )

        db.commit()
        db.refresh(db_person)

        return db_person, best_dist

    return (
        create_unknown_cluster_person(db),
        None,
    )


# -------------------------------------------------------------------------
# INPUT IMAGE
# -------------------------------------------------------------------------

def read_image_from_upload(
    file_bytes: bytes,
):
    try:
        pil_image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )

    np_image = np.array(
        pil_image
    )

    img = cv2.cvtColor(
        np_image,
        cv2.COLOR_RGB2BGR,
    )

    return pil_image, img


# -------------------------------------------------------------------------
# PAYLOAD
# -------------------------------------------------------------------------

def normalize_person_short(
    person: DBPerson | None,
):
    if not person:
        return None

    return {
        "id": person.id,
        "code": person.code,
        "name": person.name,
        "status": (
            person.status.value
            if person.status
            else None
        ),
        "q_code": person.q_code,
        "link": person.link,
        "cluster_id": person.cluster_id,
        "cluster_tag": person.cluster_tag,
        "cluster_distance": person.cluster_distance,
    }


def make_face_payload(
    face_index: int,
    face,
    category: DBFaceCategory,
    person: DBPerson | None,
    category_score,
    quality,
    det_score,
    confidence,
    distance,
    analysis: dict,
):
    bbox = (
        face.bbox
        .astype(float)
        .tolist()
    )

    bbox_data = normalize_bbox(
        bbox
    )

    return {
        "face_index": face_index,
        "bbox": bbox_data["bbox"],
        "bbox_draw": bbox_data["bbox_draw"],
        "frame_color": get_face_color(
            category=category,
            person=person,
        ),
        "person": normalize_person_short(
            person
        ),
        "person_id": (
            person.id
            if person
            else None
        ),
        "category": (
            category.name
            if category
            else None
        ),
        "category_group": (
            category.code
            if category
            else None
        ),
        "category_score": category_score,
        "quality": quality,
        "det_score": det_score,
        "gender": (
            map_gender(
                face,
                category,
            ).value
            if category
            else FaceGender.unknown.value
        ),
        "confidence": confidence,
        "confidence_marks": (
            get_confidence_marks(
                confidence
            )
        ),
        "distance": distance,
        "analysis": analysis or {},
    }


# -------------------------------------------------------------------------
# SINGLE FACE ANALYSIS
# -------------------------------------------------------------------------

def process_single_face(
    db: Session,
    pil_image: Image.Image,
    img,
    face,
    face_index: int,
    face_categories: dict,
    clip_bundle=None,
    create_unknown: bool = False,
):
    bbox = (
        face.bbox
        .astype(float)
        .tolist()
    )

    det_score = float(
        getattr(
            face,
            "det_score",
            0.0,
        )
        or 0.0
    )

    emb = normalize(
        face.embedding
    )

    # ---------------------------------------------------------------------
    # 1. KNOWN FIRST — FAISS
    # ---------------------------------------------------------------------

    ensure_faiss_indexes(db)

    best_ref, best_dist = (
        REFERENCE_FACE_INDEX.find_best_match(
            emb
        )
    )

    confidence = get_confidence(
        best_dist
    )

    can_use_known_match = (
        best_ref is not None
        and confidence != -1
    )

    if can_use_known_match:
        category = get_category(
            face_categories,
            CATEGORY_IDENTIFIABLE,
        )

        category_score = confidence

        quality, quality_details = (
            get_face_quality(
                img,
                face,
            )
        )

        analysis = (
            make_analysis_without_clip(
                quality_details=quality_details,
                reason="not_checked_known_match",
                category_name=category.name,
            )
        )

        person = (
            db.query(DBPerson)
            .filter(
                DBPerson.id
                == best_ref["person_id"]
            )
            .first()
        )

        if not person:
            raise HTTPException(
                status_code=404,
                detail="Matched person not found",
            )

        distance = round(
            best_dist,
            4,
        )

        return {
            "status": "known",
            "face": face,
            "embedding": emb,
            "person": person,
            "category": category,
            "category_score": category_score,
            "quality": quality,
            "det_score": det_score,
            "confidence": confidence,
            "distance": distance,
            "analysis": analysis,
            "payload": make_face_payload(
                face_index=face_index,
                face=face,
                category=category,
                person=person,
                category_score=category_score,
                quality=quality,
                det_score=det_score,
                confidence=confidence,
                distance=distance,
                analysis=analysis,
            ),
        }

    # ---------------------------------------------------------------------
    # 2. LOW DET SCORE
    # ---------------------------------------------------------------------

    if det_score < MIN_DET_SCORE:
        category = get_category(
            face_categories,
            DEFAULT_FACE_CATEGORY,
        )

        analysis = {
            "quality": {
                "bbox": [
                    int(x)
                    for x in bbox
                ],
                "det_score": det_score,
                "reason": "low_det_score",
            },
            "clip": {
                "category": category.name,
                "category_score": None,
                "best_clip_category": "not_checked_low_det_score",
                "best_clip_score": None,
                "clip_scores": None,
            },
        }

        return {
            "status": "low_det_score",
            "face": face,
            "embedding": emb,
            "person": None,
            "category": category,
            "category_score": None,
            "quality": None,
            "det_score": det_score,
            "confidence": None,
            "distance": None,
            "analysis": analysis,
            "payload": make_face_payload(
                face_index=face_index,
                face=face,
                category=category,
                person=None,
                category_score=None,
                quality=None,
                det_score=det_score,
                confidence=None,
                distance=None,
                analysis=analysis,
            ),
        }

    # ---------------------------------------------------------------------
    # 3. QUALITY
    # ---------------------------------------------------------------------

    quality, quality_details = (
        get_face_quality(
            img,
            face,
        )
    )

    if quality < LOW_QUALITY_THRESHOLD:
        category = get_category(
            face_categories,
            CATEGORY_LOW_QUALITY,
        )

        category_score = quality

        analysis = (
            make_analysis_without_clip(
                quality_details=quality_details,
                reason="not_checked_low_quality",
                category_name=category.name,
            )
        )

        service_person = (
            get_or_create_service_cluster_person(
                db,
                category.name,
            )
            if create_unknown
            else None
        )

        return {
            "status": "low_quality",
            "face": face,
            "embedding": emb,
            "person": service_person,
            "category": category,
            "category_score": category_score,
            "quality": quality,
            "det_score": det_score,
            "confidence": None,
            "distance": None,
            "analysis": analysis,
            "payload": make_face_payload(
                face_index=face_index,
                face=face,
                category=category,
                person=service_person,
                category_score=category_score,
                quality=quality,
                det_score=det_score,
                confidence=None,
                distance=None,
                analysis=analysis,
            ),
        }

    # ---------------------------------------------------------------------
    # 4. CLIP
    # ---------------------------------------------------------------------

    if clip_bundle is None:
        clip_bundle = get_clip()

    (
        clip_model,
        clip_preprocess,
        clip_text_features,
        clip_prompt_categories,
    ) = clip_bundle

    category_result = (
        analyze_face_category(
            image=pil_image,
            bbox=bbox,
            model=clip_model,
            preprocess=clip_preprocess,
            text_features=clip_text_features,
            prompt_categories=clip_prompt_categories,
        )
    )

    category_name = (
        category_result.get("category")
        or DEFAULT_FACE_CATEGORY
    )

    category = get_category(
        face_categories,
        category_name,
    )

    category_score = (
        category_result["category_score"]
    )

    analysis = make_analysis(
        quality_details=quality_details,
        category_result=category_result,
    )

    # ---------------------------------------------------------------------
    # 5. IDENTIFIABLE / UNKNOWN — FAISS
    # ---------------------------------------------------------------------

    if is_identifiable_analysis(
        category=category,
        analysis=analysis,
    ):
        if create_unknown:
            person, cluster_dist = (
                find_or_create_unknown_cluster_person(
                    db=db,
                    embedding=emb,
                )
            )

        else:
            person, cluster_dist = (
                find_existing_unknown_cluster_person(
                    db=db,
                    embedding=emb,
                )
            )

        distance = (
            round(cluster_dist, 4)
            if cluster_dist is not None
            else None
        )

        return {
            "status": (
                "unknown"
                if person
                else "no_match"
            ),
            "face": face,
            "embedding": emb,
            "person": person,
            "category": category,
            "category_score": category_score,
            "quality": quality,
            "det_score": det_score,
            "confidence": None,
            "distance": distance,
            "analysis": analysis,
            "payload": make_face_payload(
                face_index=face_index,
                face=face,
                category=category,
                person=person,
                category_score=category_score,
                quality=quality,
                det_score=det_score,
                confidence=None,
                distance=distance,
                analysis=analysis,
            ),
        }

    # ---------------------------------------------------------------------
    # 6. SERVICE CATEGORY
    # ---------------------------------------------------------------------

    service_person = (
        get_or_create_service_cluster_person(
            db,
            category.name,
        )
        if create_unknown
        else None
    )

    return {
        "status": "service_category",
        "face": face,
        "embedding": emb,
        "person": service_person,
        "category": category,
        "category_score": category_score,
        "quality": quality,
        "det_score": det_score,
        "confidence": None,
        "distance": None,
        "analysis": analysis,
        "payload": make_face_payload(
            face_index=face_index,
            face=face,
            category=category,
            person=service_person,
            category_score=category_score,
            quality=quality,
            det_score=det_score,
            confidence=None,
            distance=None,
            analysis=analysis,
        ),
    }


# -------------------------------------------------------------------------
# PHOTO ANALYSIS
# -------------------------------------------------------------------------

def analyze_photo_faces(
    db: Session,
    file_bytes: bytes,
    create_unknown: bool = False,
):
    pil_image, img = (
        read_image_from_upload(
            file_bytes
        )
    )

    face_model = get_insightface()

    faces = face_model.get(
        img
    )

    ensure_faiss_indexes(db)

    face_categories = (
        load_face_categories(db)
    )

    clip_bundle = None
    results = []

    for index, face in enumerate(
        faces,
        start=1,
    ):
        result = process_single_face(
            db=db,
            pil_image=pil_image,
            img=img,
            face=face,
            face_index=index,
            face_categories=face_categories,
            clip_bundle=clip_bundle,
            create_unknown=create_unknown,
        )

        if (
            clip_bundle is None
            and result["status"] not in {
                "known",
                "low_det_score",
                "low_quality",
            }
        ):
            clip_bundle = get_clip()

        results.append(
            result
        )

    return {
        "pil_image": pil_image,
        "img": img,
        "faces": faces,
        "results": results,
    }


# -------------------------------------------------------------------------
# UPLOAD FILE
# -------------------------------------------------------------------------

def make_user_upload_url(
    path: str,
):
    relative_path = (
        Path(path)
        .relative_to(
            Path(USER_UPLOAD_FOLDER)
        )
    )

    return (
        f"/media-user-upload/"
        f"{relative_path.as_posix()}"
    )


def save_file_as_image(
    file_bytes: bytes,
    original_filename: str,
) -> Path:

    USER_UPLOAD_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex}.jpg"
    )

    path = (
        Path(USER_UPLOAD_FOLDER)
        / filename
    )

    image = Image.open(
        io.BytesIO(file_bytes)
    ).convert("RGB")

    image.save(
        path,
        format="JPEG",
        quality=95,
    )

    return path


# -------------------------------------------------------------------------
# DB SOURCE / MEDIA
# -------------------------------------------------------------------------

def get_or_create_user_upload_source(
    db: Session,
) -> DBSource:

    source = (
        db.query(DBSource)
        .filter(
            DBSource.name
            == USER_UPLOAD_SOURCE_NAME
        )
        .first()
    )

    if source:
        return source

    source = DBSource(
        code=USER_UPLOAD_SOURCE_CODE,
        name=USER_UPLOAD_SOURCE_NAME,
        description=(
            "Фото, завантажені користувачами "
            "через інтерфейс."
        ),
        is_active=True,
    )

    db.add(source)
    db.commit()
    db.refresh(source)

    return source


def create_uploaded_media_freeze_iteration(
    db: Session,
    user: DBUser,
    image_path: Path,
):
    source = (
        get_or_create_user_upload_source(
            db
        )
    )

    media = DBMedia(
        material_id=image_path.stem,
        media_type=MediaType.image,
        mxf_path=str(image_path),
        mp4_path=str(image_path),
        duration=0.0,
        recorded_at=datetime.now(),
        source_id=source.id,
        user_id=user.id,
    )

    db.add(media)
    db.commit()
    db.refresh(media)

    freeze = DBFreeze(
        time_in=0.0,
        time_out=0.0,
        freeze_path=str(image_path),
        media_id=media.id,
    )

    db.add(freeze)
    db.commit()
    db.refresh(freeze)

    iteration = create_iteration(
        db,
        IterationCreate(
            status=IterationStatus.processing,
            params={
                "service": SERVICE_NAME,
                "service_type": "detect_photo_faces",
                "source": "user_upload",
                "face_det_size": FACE_DET_SIZE,
                "min_det_score": MIN_DET_SCORE,
                "dist_tolerance": DIST_TOLERANCE,
                "step_tolerance": STEP_TOLERANCE,
                "unknown_tolerance": UNKNOWN_TOLERANCE,
                "low_quality_threshold": LOW_QUALITY_THRESHOLD,
                "logic": (
                    "upload_photo -> buffalo_detect_all_faces -> "
                    "known_match_first -> det_score -> quality -> clip -> "
                    "known_or_unknown_or_service_category"
                ),
            },
            error_message=None,
            user_id=user.id,
            media_id=media.id,
        ),
    )

    return (
        media,
        freeze,
        iteration,
    )


# -------------------------------------------------------------------------
# CREATE FACE / EMBEDDING
# -------------------------------------------------------------------------

def create_embedding_and_face(
    db: Session,
    freeze: DBFreeze,
    iteration_id: int,
    face,
    emb,
    person_id: int,
    distance,
    category: DBFaceCategory,
    category_score,
    quality: float,
    gender: FaceGender,
    confidence,
    analysis: dict,
):
    bbox = (
        face.bbox
        .astype(float)
        .tolist()
    )

    embedding_create = (
        EmbeddingCreate(
            embedding_type=EmbeddingType.detected_face,
            source={
                "freeze_id": freeze.id,
                "freeze_path": freeze.freeze_path,
                "bbox": bbox,
                "distance": distance,
                "category": category.name,
                "category_score": category_score,
                "quality": quality,
                "analysis": analysis,
                "created_by": SERVICE_NAME,
            },
            vector=emb.tolist(),
            person_id=person_id,
        )
    )

    detected_embedding = (
        create_embedding(
            db,
            embedding_create,
        )
    )

    face_create = FaceCreate(
        bbox=bbox,
        category_id=category.id,
        category_score=category_score,
        quality=quality,
        gender=gender,
        confidence=confidence,
        analysis=analysis,
        embedding_id=detected_embedding.id,
        freeze_id=freeze.id,
        person_id=person_id,
        iteration_id=iteration_id,
    )

    db_face = create_face(
        db,
        face_create,
    )

    return (
        detected_embedding,
        db_face,
    )


# -------------------------------------------------------------------------
# SAVE DETECTED FACES
# -------------------------------------------------------------------------

def save_detected_faces_to_db(
    db: Session,
    user: DBUser,
    file,
    file_bytes: bytes,
    analysis_result: dict,
):
    saved_image_path = (
        save_file_as_image(
            file_bytes=file_bytes,
            original_filename=(
                file.filename
                or "upload.jpg"
            ),
        )
    )

    media, freeze, iteration = (
        create_uploaded_media_freeze_iteration(
            db=db,
            user=user,
            image_path=saved_image_path,
        )
    )

    saved_faces = []
    skipped_faces = []

    try:
        for result in analysis_result["results"]:
            status = result["status"]

            if status == "low_det_score":
                skipped_faces.append(
                    {
                        "face_index": result[
                            "payload"
                        ]["face_index"],
                        "reason": "low_det_score",
                        "det_score": result["det_score"],
                    }
                )

                continue

            person = result.get(
                "person"
            )

            category = result[
                "category"
            ]

            if person is None:
                if is_identifiable_analysis(
                    category=category,
                    analysis=result["analysis"],
                ):
                    person, cluster_dist = (
                        find_or_create_unknown_cluster_person(
                            db=db,
                            embedding=result["embedding"],
                        )
                    )

                    result["person"] = person

                    result["distance"] = (
                        round(cluster_dist, 4)
                        if cluster_dist is not None
                        else None
                    )

                else:
                    person = (
                        get_or_create_service_cluster_person(
                            db=db,
                            category_name=category.name,
                        )
                    )

                    result["person"] = person

            detected_embedding, db_face = (
                create_embedding_and_face(
                    db=db,
                    freeze=freeze,
                    iteration_id=iteration.id,
                    face=result["face"],
                    emb=result["embedding"],
                    person_id=person.id,
                    distance=result["distance"],
                    category=category,
                    category_score=result["category_score"],
                    quality=result["quality"],
                    gender=map_gender(
                        result["face"],
                        category,
                    ),
                    confidence=result["confidence"],
                    analysis=result["analysis"],
                )
            )

            # -------------------------------------------------------------
            # LIVE UPDATE FAISS
            #
            # У FAISS додаємо тільки identifiable unknown.
            # Known faces та service categories в unknown-index не кладемо.
            # -------------------------------------------------------------

            if (
                get_person_status_value(person)
                == PersonStatus.unknown.value
                and is_identifiable_analysis(
                    category=category,
                    analysis=result["analysis"],
                )
            ):
                UNKNOWN_FACE_INDEX.add(
                    embedding=result["embedding"],
                    person_id=person.id,
                )

            payload = result[
                "payload"
            ]

            payload["face_id"] = (
                db_face.id
            )

            payload["embedding_id"] = (
                detected_embedding.id
            )

            payload["person"] = (
                normalize_person_short(
                    person
                )
            )

            payload["person_id"] = (
                person.id
            )

            payload["frame_color"] = (
                get_face_color(
                    category=category,
                    person=person,
                )
            )

            payload["distance"] = (
                result["distance"]
            )

            saved_faces.append(
                payload
            )

        update_iteration(
            db,
            iteration.id,
            IterationUpdate(
                status=IterationStatus.completed,
                finished_at=datetime.now(),
            ),
        )

        log_history(
            db,
            user.id,
            "detect.photo.add_to_db",
            {
                "filename": file.filename,
                "user_id": user.id,
                "user_role": (
                    user.role.value
                    if user.role
                    else None
                ),
                "saved_to_db": True,
                "media_id": media.id,
                "freeze_id": freeze.id,
                "iteration_id": iteration.id,
                "faces_detected": len(
                    analysis_result["results"]
                ),
                "faces_saved": len(
                    saved_faces
                ),
                "faces_skipped": len(
                    skipped_faces
                ),
            },
        )

        return {
            "mode": "detect_photo_saved",
            "message": (
                "Фото та результати детекції "
                "збережено в БД."
            ),
            "uploaded": {
                "media_id": media.id,
                "freeze_id": freeze.id,
                "image_url": (
                    make_user_upload_url(
                        str(saved_image_path)
                    )
                ),
                "path": str(
                    saved_image_path
                ),
            },
            "summary": {
                "faces_detected": len(
                    analysis_result["results"]
                ),
                "faces_saved": len(
                    saved_faces
                ),
                "faces_skipped": len(
                    skipped_faces
                ),
            },
            "faces": saved_faces,
            "skipped_faces": skipped_faces,
        }

    except Exception as e:
        update_iteration(
            db,
            iteration.id,
            IterationUpdate(
                status=IterationStatus.error,
                finished_at=datetime.now(),
                error_message=str(e),
            ),
        )

        log_history(
            db,
            user.id,
            "detect.photo.add_to_db",
            {
                "filename": file.filename,
                "user_id": user.id,
                "user_role": (
                    user.role.value
                    if user.role
                    else None
                ),
                "saved_to_db": True,
                "media_id": media.id,
                "freeze_id": freeze.id,
                "iteration_id": iteration.id,
                "result": "error",
                "error": str(e),
            },
        )

        raise


# -------------------------------------------------------------------------
# READONLY RESPONSE
# -------------------------------------------------------------------------

def make_readonly_detect_response(
    db: Session,
    user: DBUser,
    file,
    analysis_result: dict,
):
    faces_payload = [
        result["payload"]
        for result
        in analysis_result["results"]
    ]

    log_history(
        db,
        user.id,
        "detect.photo",
        {
            "filename": file.filename,
            "user_id": user.id,
            "user_role": (
                user.role.value
                if user.role
                else None
            ),
            "saved_to_db": False,
            "faces_detected": len(
                faces_payload
            ),
        },
    )

    if not faces_payload:
        return {
            "mode": "no_faces",
            "message": (
                "На фотографії не знайдено облич."
            ),
            "summary": {
                "faces_detected": 0,
            },
            "can_add_to_db": False,
            "faces": [],
            "result": None,
        }

    return {
        "mode": "detect_photo_result",
        "message": (
            "Детекцію облич на фото виконано."
        ),
        "summary": {
            "faces_detected": len(
                faces_payload
            ),
            "known_count": sum(
                1
                for face in faces_payload
                if (
                    face.get("person")
                    and face["person"].get("status")
                    in {"public", "non_public"}
                )
            ),
            "unknown_count": sum(
                1
                for face in faces_payload
                if (
                    face.get("person")
                    and face["person"].get("status")
                    == "unknown"
                )
            ),
            "service_count": sum(
                1
                for face in faces_payload
                if face.get("frame_color")
                in {"gray", "orange"}
            ),
        },
        "can_add_to_db": (
            can_save_uploaded_media(
                user
            )
        ),
        "faces": faces_payload,
        "result": None,
    }


# -------------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------------

@router.post("/faces")
def detect_faces_on_photo(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    db: Session = Depends(get_db),
):
    user = (
        db.query(DBUser)
        .filter(
            DBUser.id == user_id
        )
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    file_bytes = (
        file.file.read()
    )

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    analysis_result = (
        analyze_photo_faces(
            db=db,
            file_bytes=file_bytes,
            create_unknown=False,
        )
    )

    return (
        make_readonly_detect_response(
            db=db,
            user=user,
            file=file,
            analysis_result=analysis_result,
        )
    )


@router.post("/faces/add-to-db")
def add_detected_photo_faces_to_db(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    db: Session = Depends(get_db),
):
    user = (
        db.query(DBUser)
        .filter(
            DBUser.id == user_id
        )
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    if not can_save_uploaded_media(user):
        raise HTTPException(
            status_code=403,
            detail=(
                "Only developer can add "
                "detected photo faces to database"
            ),
        )

    file_bytes = (
        file.file.read()
    )

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    analysis_result = (
        analyze_photo_faces(
            db=db,
            file_bytes=file_bytes,
            create_unknown=True,
        )
    )

    return (
        save_detected_faces_to_db(
            db=db,
            user=user,
            file=file,
            file_bytes=file_bytes,
            analysis_result=analysis_result,
        )
    )


@router.get("/person/{person_id}")
def get_detected_person_archive(
    person_id: int,
    db: Session = Depends(get_db),
):
    person = (
        db.query(DBPerson)
        .filter(
            DBPerson.id == person_id
        )
        .first()
    )

    if not person:
        raise HTTPException(
            status_code=404,
            detail="Person not found",
        )

    return {
        "mode": "person_result",
        "query": None,
        "summary": {
            "persons_count": 1,
        },
        "candidates": [],
        "result": build_person_result(
            db,
            person,
        ),
    }
