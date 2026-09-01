# routes/routes_services/routes_search_by_photo.py

import io
import logging
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
    DBEmbedding,
    DBFaceCategory,
    DBFreeze,
    DBIteration,
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
from services.create_faces.clip_face_filter_v2 import get_clip, analyze_face_category
from services.create_faces.clip_face_categories import (
    DEFAULT_FACE_CATEGORY,
    CATEGORY_IDENTIFIABLE,
    CATEGORY_LOW_QUALITY,
)

from services.create_faces.faiss_face_index import REFERENCE_FACE_INDEX, UNKNOWN_FACE_INDEX, ensure_faiss_indexes


router = APIRouter(prefix="/search-photo", tags=["search by photo"])


SERVICE_NAME = "routes_search_by_photo_faiss.py"

FACE_DET_SIZE = 640
MIN_DET_SCORE = 0.60
DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.055
UNKNOWN_TOLERANCE = 0.55
LOW_QUALITY_THRESHOLD = 0.60

USER_UPLOAD_SOURCE_CODE = "test"
USER_UPLOAD_SOURCE_NAME = "user_upload"


# -------------------------------------------------------------------------
# GLOBAL MODEL / INDEX CACHE
# -------------------------------------------------------------------------

_INSIGHTFACE_CACHE = None

# REFERENCE_FACE_INDEX = ReferenceFaceIndex()
# UNKNOWN_FACE_INDEX = UnknownFaceIndex()

# _FAISS_INDEXES_READY = False


# -------------------------------------------------------------------------
# INSIGHTFACE
# -------------------------------------------------------------------------

def load_insightface():
    logging.info("Loading InsightFace buffalo_l...")

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

    logging.info("InsightFace buffalo_l loaded.")

    return app


def get_insightface():
    global _INSIGHTFACE_CACHE

    if _INSIGHTFACE_CACHE is None:
        _INSIGHTFACE_CACHE = load_insightface()

    return _INSIGHTFACE_CACHE

# -------------------------------------------------------------------------
# MATCHING PARAMS
# -------------------------------------------------------------------------

def get_confidence(dist: float) -> int:
    if dist <= DIST_TOLERANCE:
        return 0

    for i in range(1, 4):
        if dist <= DIST_TOLERANCE + i * STEP_TOLERANCE:
            return i

    return -1


# -------------------------------------------------------------------------
# FACE CATEGORIES
# -------------------------------------------------------------------------

def load_face_categories(db: Session):
    rows = (
        db.query(DBFaceCategory)
        .filter(DBFaceCategory.is_active.is_(True))
        .all()
    )

    return {row.name: row for row in rows}


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


def is_identifiable_category(category: DBFaceCategory) -> bool:
    return (
        category.name == CATEGORY_IDENTIFIABLE
        or category.code == CATEGORY_IDENTIFIABLE
    )


# -------------------------------------------------------------------------
# USER UPLOAD SOURCE
# -------------------------------------------------------------------------

def get_or_create_user_upload_source(db: Session) -> DBSource:
    source = (
        db.query(DBSource)
        .filter(DBSource.name == USER_UPLOAD_SOURCE_NAME)
        .first()
    )

    if source:
        return source

    source = DBSource(
        code=USER_UPLOAD_SOURCE_CODE,
        name=USER_UPLOAD_SOURCE_NAME,
        description="Фото, завантажені користувачами через інтерфейс пошуку.",
        is_active=True,
    )

    db.add(source)
    db.commit()
    db.refresh(source)

    return source


def can_save_uploaded_media(user: DBUser) -> bool:
    return user.role == UserRole.developer


def make_user_upload_url(path: str):
    relative_path = Path(path).relative_to(Path(USER_UPLOAD_FOLDER))

    return f"/media-user-upload/{relative_path.as_posix()}"


# -------------------------------------------------------------------------
# FACE ATTRIBUTES
# -------------------------------------------------------------------------

def map_gender(
    face,
    category: DBFaceCategory,
) -> FaceGender:

    if not is_identifiable_category(category):
        return FaceGender.unknown

    gender = getattr(face, "gender", None)

    if gender == 1:
        return FaceGender.male

    if gender == 0:
        return FaceGender.female

    return FaceGender.unknown


# -------------------------------------------------------------------------
# ANALYSIS
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
# UNKNOWN CLUSTERS
# -------------------------------------------------------------------------

def get_next_cluster_number(
    db: Session,
    status: PersonStatus,
) -> int:

    max_cluster_id = (
        db.query(func.max(DBPerson.cluster_id))
        .filter(DBPerson.status == status)
        .scalar()
    )

    return (max_cluster_id or 0) + 1


def create_unknown_cluster_person(db: Session):
    cluster_id = get_next_cluster_number(
        db,
        PersonStatus.unknown,
    )

    cluster_tag = f"unknown_cluster_{cluster_id:06d}"

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
    cluster_tag = f"{category_name}_cluster"

    db_person = (
        db.query(DBPerson)
        .filter(DBPerson.code == cluster_tag)
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


def find_existing_unknown_cluster_person(
    db: Session,
    embedding,
):
    ensure_faiss_indexes(db)

    person_id, best_dist = UNKNOWN_FACE_INDEX.find_best_match(
        embedding
    )

    if person_id is None or best_dist is None:
        return None, None

    if best_dist > UNKNOWN_TOLERANCE:
        return None, None

    db_person = (
        db.query(DBPerson)
        .filter(DBPerson.id == person_id)
        .first()
    )

    if not db_person:
        logging.warning(
            "Unknown FAISS returned missing person_id=%s",
            person_id,
        )

        return None, None

    return db_person, best_dist


def find_or_create_unknown_cluster_person(
    db: Session,
    embedding,
):
    db_person, best_dist = find_existing_unknown_cluster_person(
        db,
        embedding,
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

    return create_unknown_cluster_person(db), None


# -------------------------------------------------------------------------
# FILE
# -------------------------------------------------------------------------

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

    path = USER_UPLOAD_FOLDER / filename

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
# DB OBJECTS FOR USER UPLOAD
# -------------------------------------------------------------------------

def create_uploaded_media_freeze_iteration(
    db: Session,
    user: DBUser,
    image_path: Path,
):
    source = get_or_create_user_upload_source(db)

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
                "service_type": "photo_identity_search",
                "source": "user_upload",
                "face_det_size": FACE_DET_SIZE,
                "min_det_score": MIN_DET_SCORE,
                "dist_tolerance": DIST_TOLERANCE,
                "step_tolerance": STEP_TOLERANCE,
                "unknown_tolerance": UNKNOWN_TOLERANCE,
                "low_quality_threshold": LOW_QUALITY_THRESHOLD,
                "logic": (
                    "upload_photo -> buffalo_detect -> single_face_only -> "
                    "known_match_first -> det_score -> quality -> clip -> "
                    "known_or_unknown_or_service_category"
                ),
            },
            error_message=None,
            user_id=user.id,
            media_id=media.id,
        ),
    )

    return media, freeze, iteration


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
    bbox = face.bbox.astype(float).tolist()

    embedding_create = EmbeddingCreate(
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

    detected_embedding = create_embedding(
        db,
        embedding_create,
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

    create_face(
        db,
        face_create,
    )

    return detected_embedding


# -------------------------------------------------------------------------
# RESPONSE HELPERS
# -------------------------------------------------------------------------

def empty_person_result():
    return {
        "person": None,
        "summary": {
            "faces_count": 0,
            "frames_count": 0,
            "media_count": 0,
        },
        "medias": [],
    }


def make_uploaded_preview(
    image_path: Path | None,
):
    if not image_path:
        return None

    return {
        "image_url": make_user_upload_url(
            str(image_path)
        ),
        "path": str(image_path),
    }


# -------------------------------------------------------------------------
# IMAGE INPUT
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

    np_image = np.array(pil_image)

    img = cv2.cvtColor(
        np_image,
        cv2.COLOR_RGB2BGR,
    )

    return pil_image, img


def is_identifiable_clip_result(
    category: DBFaceCategory,
    analysis_result: dict,
) -> bool:

    clip = (
        analysis_result
        .get("analysis", {})
        .get("clip", {})
    )

    return (
        category.name == CATEGORY_IDENTIFIABLE
        or category.code == CATEGORY_IDENTIFIABLE
        or (
            clip.get("best_clip_category")
            == CATEGORY_IDENTIFIABLE

            and float(
                clip.get("best_clip_score")
                or 0.0
            ) >= 0.50
        )
    )


# -------------------------------------------------------------------------
# PHOTO ANALYSIS
# -------------------------------------------------------------------------

def analyze_photo(
    db: Session,
    file_bytes: bytes,
):
    pil_image, img = read_image_from_upload(
        file_bytes
    )

    face_model = get_insightface()

    faces = face_model.get(img)

    if len(faces) == 0:
        return {
            "status": "no_faces",
            "pil_image": pil_image,
            "img": img,
            "faces": faces,
        }

    if len(faces) > 1:
        return {
            "status": "multiple_faces",
            "pil_image": pil_image,
            "img": img,
            "faces": faces,
        }

    face = faces[0]

    bbox = face.bbox.astype(float).tolist()

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

    # -------------------------------------------------------------
    # FAISS REFERENCE SEARCH
    # -------------------------------------------------------------

    ensure_faiss_indexes(db)

    face_categories = load_face_categories(db)

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

    # -------------------------------------------------------------
    # KNOWN
    # -------------------------------------------------------------

    if can_use_known_match:
        category = get_category(
            face_categories,
            CATEGORY_IDENTIFIABLE,
        )

        category_score = confidence

        quality, quality_details = get_face_quality(
            img,
            face,
        )

        analysis = make_analysis_without_clip(
            quality_details=quality_details,
            reason="not_checked_known_match",
            category_name=category.name,
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

        return {
            "status": "known",
            "pil_image": pil_image,
            "img": img,
            "face": face,
            "bbox": bbox,
            "emb": emb,
            "person": person,
            "distance": round(
                best_dist,
                4,
            ),
            "confidence": confidence,
            "category": category,
            "category_score": category_score,
            "quality": quality,
            "det_score": det_score,
            "analysis": analysis,
        }

    # -------------------------------------------------------------
    # LOW DET SCORE
    # -------------------------------------------------------------

    if det_score < MIN_DET_SCORE:
        return {
            "status": "low_det_score",
            "pil_image": pil_image,
            "img": img,
            "face": face,
            "bbox": bbox,
            "emb": emb,
            "det_score": det_score,
        }

    # -------------------------------------------------------------
    # QUALITY
    # -------------------------------------------------------------

    quality, quality_details = get_face_quality(
        img,
        face,
    )

    if quality < LOW_QUALITY_THRESHOLD:
        category = get_category(
            face_categories,
            CATEGORY_LOW_QUALITY,
        )

        analysis = make_analysis_without_clip(
            quality_details=quality_details,
            reason="not_checked_low_quality",
            category_name=category.name,
        )

        return {
            "status": "low_quality",
            "pil_image": pil_image,
            "img": img,
            "face": face,
            "bbox": bbox,
            "emb": emb,
            "category": category,
            "category_score": quality,
            "quality": quality,
            "det_score": det_score,
            "analysis": analysis,
        }

    # -------------------------------------------------------------
    # CLIP
    # -------------------------------------------------------------

    (
        clip_model,
        clip_preprocess,
        clip_text_features,
        clip_prompt_categories,
    ) = get_clip()

    category_result = analyze_face_category(
        image=pil_image,
        bbox=bbox,
        model=clip_model,
        preprocess=clip_preprocess,
        text_features=clip_text_features,
        prompt_categories=clip_prompt_categories,
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

    return {
        "status": "clip_result",
        "pil_image": pil_image,
        "img": img,
        "face": face,
        "bbox": bbox,
        "emb": emb,
        "category": category,
        "category_score": category_score,
        "quality": quality,
        "det_score": det_score,
        "analysis": analysis,
    }


# -------------------------------------------------------------------------
# READONLY RESPONSE
# -------------------------------------------------------------------------

def make_readonly_response(
    db: Session,
    user: DBUser,
    file,
    analysis_result: dict,
):
    log_base = {
        "filename": file.filename,
        "user_id": user.id,
        "user_role": (
            user.role.value
            if user.role
            else None
        ),
        "saved_to_db": False,
    }

    status = analysis_result["status"]

    if status == "no_faces":
        log_history(
            db,
            user.id,
            "search.photo",
            {
                **log_base,
                "result": "no_faces",
            },
        )

        return {
            "mode": "no_faces",
            "message": "На зображенні не знайдено облич.",
            "uploaded": None,
            "match": None,
            "result": None,
        }

    if status == "multiple_faces":
        log_history(
            db,
            user.id,
            "search.photo",
            {
                **log_base,
                "result": "multiple_faces",
                "faces_count": len(
                    analysis_result["faces"]
                ),
            },
        )

        return {
            "mode": "multiple_faces",
            "message": (
                "На фото знайдено кілька облич. "
                "Для такого зображення скористайтесь "
                "сервісом детекції облич на фото."
            ),
            "faces_count": len(
                analysis_result["faces"]
            ),
            "redirect_to": "detect_photo",
            "uploaded": None,
            "match": None,
            "result": None,
        }

    if status == "known":
        person = analysis_result["person"]

        log_history(
            db,
            user.id,
            "search.photo",
            {
                **log_base,
                "result": "known",
                "person_id": person.id,
                "person_name": person.name,
                "distance": analysis_result["distance"],
                "confidence": analysis_result["confidence"],
                "det_score": analysis_result["det_score"],
                "quality": analysis_result["quality"],
            },
        )

        return {
            "mode": "photo_person_result",
            "message": "Персону знайдено в базі.",
            "uploaded": None,
            "match": {
                "type": "known",
                "person_id": person.id,
                "person_name": person.name,
                "distance": analysis_result["distance"],
                "confidence": analysis_result["confidence"],
                "det_score": analysis_result["det_score"],
                "quality": analysis_result["quality"],
                "bbox": analysis_result["bbox"],
                "analysis": analysis_result["analysis"],
            },
            "result": build_person_result(
                db,
                person,
            ),
        }

    if status == "low_det_score":
        log_history(
            db,
            user.id,
            "search.photo",
            {
                **log_base,
                "result": "low_det_score",
                "det_score": analysis_result["det_score"],
                "bbox": analysis_result["bbox"],
            },
        )

        return {
            "mode": "low_det_score",
            "message": (
                "Модель не впевнена, що на фото "
                "є людське обличчя."
            ),
            "uploaded": None,
            "match": {
                "type": "low_det_score",
                "det_score": analysis_result["det_score"],
                "bbox": analysis_result["bbox"],
            },
            "result": None,
        }

    if status == "low_quality":
        log_history(
            db,
            user.id,
            "search.photo",
            {
                **log_base,
                "result": "low_quality",
                "quality": analysis_result["quality"],
                "det_score": analysis_result["det_score"],
            },
        )

        return {
            "mode": "low_quality",
            "message": (
                "Якість фотографії недостатня для надійної "
                "ідентифікації. Ідентифікація ускладнена "
                "або неможлива."
            ),
            "uploaded": None,
            "match": {
                "type": "low_quality",
                "quality": analysis_result["quality"],
                "det_score": analysis_result["det_score"],
                "bbox": analysis_result["bbox"],
                "analysis": analysis_result["analysis"],
            },
            "result": None,
        }

    if status == "clip_result":
        category = analysis_result["category"]

        if is_identifiable_clip_result(
            category,
            analysis_result,
        ):
            person, cluster_dist = (
                find_existing_unknown_cluster_person(
                    db=db,
                    embedding=analysis_result["emb"],
                )
            )

            if person:
                distance_for_source = (
                    round(cluster_dist, 4)
                    if cluster_dist is not None
                    else None
                )

                log_history(
                    db,
                    user.id,
                    "search.photo",
                    {
                        **log_base,
                        "result": "unknown_cluster",
                        "person_id": person.id,
                        "cluster_tag": person.cluster_tag,
                        "cluster_distance": distance_for_source,
                        "category": category.name,
                        "category_score": analysis_result["category_score"],
                        "quality": analysis_result["quality"],
                        "det_score": analysis_result["det_score"],
                    },
                )

                return {
                    "mode": "photo_unknown_result",
                    "message": (
                        "Особа поки неідентифікована. "
                        "Тимчасове ім’я: unknown-кластер."
                    ),
                    "uploaded": None,
                    "match": {
                        "type": "unknown",
                        "person_id": person.id,
                        "person_name": person.name,
                        "cluster_tag": person.cluster_tag,
                        "cluster_distance": distance_for_source,
                        "category": category.name,
                        "category_score": analysis_result["category_score"],
                        "quality": analysis_result["quality"],
                        "det_score": analysis_result["det_score"],
                        "bbox": analysis_result["bbox"],
                        "analysis": analysis_result["analysis"],
                    },
                    "result": build_person_result(
                        db,
                        person,
                    ),
                }

            log_history(
                db,
                user.id,
                "search.photo",
                {
                    **log_base,
                    "result": "no_match",
                    "category": category.name,
                    "category_score": analysis_result["category_score"],
                    "quality": analysis_result["quality"],
                    "det_score": analysis_result["det_score"],
                },
            )

            return {
                "mode": "photo_no_match",
                "message": "Персону не знайдено в базі.",
                "uploaded": None,
                "can_add_to_db": can_save_uploaded_media(
                    user
                ),
                "match": {
                    "type": "no_match",
                    "category": category.name,
                    "category_score": analysis_result["category_score"],
                    "quality": analysis_result["quality"],
                    "det_score": analysis_result["det_score"],
                    "bbox": analysis_result["bbox"],
                    "analysis": analysis_result["analysis"],
                },
                "result": empty_person_result(),
            }

        log_history(
            db,
            user.id,
            "search.photo",
            {
                **log_base,
                "result": "service_category",
                "category": category.name,
                "category_score": analysis_result["category_score"],
                "quality": analysis_result["quality"],
                "det_score": analysis_result["det_score"],
            },
        )

        return {
            "mode": "service_category",
            "message": (
                "Обличчя не віднесене до категорії "
                "придатних для ідентифікації."
            ),
            "uploaded": None,
            "match": {
                "type": "service_category",
                "category": category.name,
                "category_score": analysis_result["category_score"],
                "quality": analysis_result["quality"],
                "det_score": analysis_result["det_score"],
                "bbox": analysis_result["bbox"],
                "analysis": analysis_result["analysis"],
            },
            "result": None,
        }

    raise RuntimeError(
        f"Unknown analysis status: {status}"
    )


# -------------------------------------------------------------------------
# SAVE RESULT
# -------------------------------------------------------------------------

def save_analysis_to_db(
    db: Session,
    user: DBUser,
    file,
    file_bytes: bytes,
    analysis_result: dict,
):
    status = analysis_result["status"]

    if status in {
        "no_faces",
        "multiple_faces",
        "low_det_score",
    }:
        return make_readonly_response(
            db,
            user,
            file,
            analysis_result,
        )

    saved_image_path = save_file_as_image(
        file_bytes,
        file.filename or "upload.jpg",
    )

    media, freeze, iteration = (
        create_uploaded_media_freeze_iteration(
            db=db,
            user=user,
            image_path=saved_image_path,
        )
    )

    log_base = {
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
    }

    try:

        # -------------------------------------------------------------
        # KNOWN
        # -------------------------------------------------------------

        if status == "known":
            person = analysis_result["person"]
            category = analysis_result["category"]

            create_embedding_and_face(
                db=db,
                freeze=freeze,
                iteration_id=iteration.id,
                face=analysis_result["face"],
                emb=analysis_result["emb"],
                person_id=person.id,
                distance=analysis_result["distance"],
                category=category,
                category_score=analysis_result["category_score"],
                quality=analysis_result["quality"],
                gender=map_gender(
                    analysis_result["face"],
                    category,
                ),
                confidence=analysis_result["confidence"],
                analysis=analysis_result["analysis"],
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
                "search.photo.add_to_db",
                {
                    **log_base,
                    "result": "known",
                    "person_id": person.id,
                    "person_name": person.name,
                },
            )

            return {
                "mode": "photo_person_result",
                "message": (
                    "Персону знайдено в базі. "
                    "Фото збережено в БД."
                ),
                "uploaded": make_uploaded_preview(
                    saved_image_path
                ),
                "match": {
                    "type": "known",
                    "person_id": person.id,
                    "person_name": person.name,
                    "distance": analysis_result["distance"],
                    "confidence": analysis_result["confidence"],
                    "det_score": analysis_result["det_score"],
                    "quality": analysis_result["quality"],
                    "bbox": analysis_result["bbox"],
                    "analysis": analysis_result["analysis"],
                },
                "result": build_person_result(
                    db,
                    person,
                ),
            }

        # -------------------------------------------------------------
        # LOW QUALITY
        # -------------------------------------------------------------

        if status == "low_quality":
            category = analysis_result["category"]

            service_person = (
                get_or_create_service_cluster_person(
                    db,
                    category.name,
                )
            )

            create_embedding_and_face(
                db=db,
                freeze=freeze,
                iteration_id=iteration.id,
                face=analysis_result["face"],
                emb=analysis_result["emb"],
                person_id=service_person.id,
                distance=None,
                category=category,
                category_score=analysis_result["category_score"],
                quality=analysis_result["quality"],
                gender=FaceGender.unknown,
                confidence=None,
                analysis=analysis_result["analysis"],
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
                "search.photo.add_to_db",
                {
                    **log_base,
                    "result": "low_quality",
                    "service_person_id": service_person.id,
                },
            )

            return {
                "mode": "low_quality",
                "message": (
                    "Фото низької якості збережено "
                    "у службовий кластер low_quality."
                ),
                "uploaded": make_uploaded_preview(
                    saved_image_path
                ),
                "match": {
                    "type": "low_quality",
                    "quality": analysis_result["quality"],
                    "det_score": analysis_result["det_score"],
                    "bbox": analysis_result["bbox"],
                    "analysis": analysis_result["analysis"],
                },
                "result": build_person_result(
                    db,
                    service_person,
                ),
            }

        # -------------------------------------------------------------
        # CLIP
        # -------------------------------------------------------------

        if status == "clip_result":
            category = analysis_result["category"]

            # ---------------------------------------------------------
            # IDENTIFIABLE UNKNOWN
            # ---------------------------------------------------------

            if is_identifiable_clip_result(
                category,
                analysis_result,
            ):
                person, cluster_dist = (
                    find_or_create_unknown_cluster_person(
                        db=db,
                        embedding=analysis_result["emb"],
                    )
                )

                distance_for_source = (
                    round(cluster_dist, 4)
                    if cluster_dist is not None
                    else None
                )

                detected_embedding = (
                    create_embedding_and_face(
                        db=db,
                        freeze=freeze,
                        iteration_id=iteration.id,
                        face=analysis_result["face"],
                        emb=analysis_result["emb"],
                        person_id=person.id,
                        distance=distance_for_source,
                        category=category,
                        category_score=analysis_result["category_score"],
                        quality=analysis_result["quality"],
                        gender=map_gender(
                            analysis_result["face"],
                            category,
                        ),
                        confidence=None,
                        analysis=analysis_result["analysis"],
                    )
                )

                # -----------------------------------------------------
                # LIVE UPDATE UNKNOWN FAISS
                # -----------------------------------------------------

                ensure_faiss_indexes(db)

                UNKNOWN_FACE_INDEX.add(
                    embedding=analysis_result["emb"],
                    person_id=person.id,
                )

                logging.info(
                    "Unknown embedding added to FAISS: "
                    "embedding_id=%s, person_id=%s, faiss_size=%s",
                    detected_embedding.id,
                    person.id,
                    UNKNOWN_FACE_INDEX.size,
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
                    "search.photo.add_to_db",
                    {
                        **log_base,
                        "result": "unknown_cluster",
                        "person_id": person.id,
                        "cluster_tag": person.cluster_tag,
                        "cluster_distance": distance_for_source,
                    },
                )

                return {
                    "mode": "photo_unknown_result",
                    "message": (
                        "Фото збережено в unknown-кластер."
                    ),
                    "uploaded": make_uploaded_preview(
                        saved_image_path
                    ),
                    "match": {
                        "type": "unknown",
                        "person_id": person.id,
                        "person_name": person.name,
                        "cluster_tag": person.cluster_tag,
                        "cluster_distance": distance_for_source,
                        "category": category.name,
                        "category_score": analysis_result["category_score"],
                        "quality": analysis_result["quality"],
                        "det_score": analysis_result["det_score"],
                        "bbox": analysis_result["bbox"],
                        "analysis": analysis_result["analysis"],
                    },
                    "result": build_person_result(
                        db,
                        person,
                    ),
                }

            # ---------------------------------------------------------
            # SERVICE CATEGORY
            # ---------------------------------------------------------

            service_person = (
                get_or_create_service_cluster_person(
                    db,
                    category.name,
                )
            )

            create_embedding_and_face(
                db=db,
                freeze=freeze,
                iteration_id=iteration.id,
                face=analysis_result["face"],
                emb=analysis_result["emb"],
                person_id=service_person.id,
                distance=None,
                category=category,
                category_score=analysis_result["category_score"],
                quality=analysis_result["quality"],
                gender=FaceGender.unknown,
                confidence=None,
                analysis=analysis_result["analysis"],
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
                "search.photo.add_to_db",
                {
                    **log_base,
                    "result": "service_category",
                    "service_person_id": service_person.id,
                    "category": category.name,
                },
            )

            return {
                "mode": "service_category",
                "message": (
                    "Фото збережено у службовий кластер."
                ),
                "uploaded": make_uploaded_preview(
                    saved_image_path
                ),
                "match": {
                    "type": "service_category",
                    "category": category.name,
                    "category_score": analysis_result["category_score"],
                    "quality": analysis_result["quality"],
                    "det_score": analysis_result["det_score"],
                    "bbox": analysis_result["bbox"],
                    "analysis": analysis_result["analysis"],
                },
                "result": build_person_result(
                    db,
                    service_person,
                ),
            }

        raise RuntimeError(
            f"Cannot save status: {status}"
        )

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

        raise


# -------------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------------

@router.post("/person")
def search_person_by_photo(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    db: Session = Depends(get_db),
):
    user = (
        db.query(DBUser)
        .filter(DBUser.id == user_id)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    file_bytes = file.file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    analysis_result = analyze_photo(
        db=db,
        file_bytes=file_bytes,
    )

    return make_readonly_response(
        db=db,
        user=user,
        file=file,
        analysis_result=analysis_result,
    )


@router.post("/person/add-to-db")
def add_photo_search_result_to_db(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    db: Session = Depends(get_db),
):
    user = (
        db.query(DBUser)
        .filter(DBUser.id == user_id)
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
                "Only developer can add uploaded "
                "photo to database"
            ),
        )

    file_bytes = file.file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    analysis_result = analyze_photo(
        db=db,
        file_bytes=file_bytes,
    )

    return save_analysis_to_db(
        db=db,
        user=user,
        file=file,
        file_bytes=file_bytes,
        analysis_result=analysis_result,
    )
