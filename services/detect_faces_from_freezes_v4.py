import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from sqlalchemy import func
from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.enums import (
    EmbeddingType,
    FaceCategory,
    FaceGender,
    IterationStatus,
    PersonStatus,
)
from db.models import DBEmbedding, DBFreeze, DBMedia, DBPerson

from crud.crud_embedding import create_embedding
from crud.crud_face import create_face, get_faces_by_freeze
from crud.crud_iteration import create_iteration, update_iteration
from crud.crud_person import create_person

from schemas.schemas_embedding import EmbeddingCreate
from schemas.schemas_face import FaceCreate
from schemas.schemas_iteration import IterationCreate, IterationUpdate
from schemas.schemas_person import PersonsCreate

from routers.commons import normalize, cosine_distance

from services.face_quality_v3 import get_face_quality
from services.clip_face_filter_v2 import get_clip, analyze_face_category


Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/detect_faces_from_freezes_v2.log", encoding="utf-8"),
    ],
)


FACE_DET_SIZE = 640
MIN_DET_SCORE = 0.65 # відсікти кота

DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.055 # зловити кучму з нахиленим обличчям

UNKNOWN_TOLERANCE = 0.55 # групування пиріжкова варта
LOW_QUALITY_THRESHOLD = 0.65

USER_ID = 1

_INSIGHTFACE_CACHE = None


def load_insightface():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))
    return app


def get_insightface():
    global _INSIGHTFACE_CACHE

    if _INSIGHTFACE_CACHE is None:
        _INSIGHTFACE_CACHE = load_insightface()

    return _INSIGHTFACE_CACHE


def get_confidence(dist: float) -> int:
    if dist <= DIST_TOLERANCE:
        return 0

    for i in range(1, 4):
        if dist <= DIST_TOLERANCE + i * STEP_TOLERANCE:
            return i

    return -1


def to_face_category(value: str) -> FaceCategory:
    try:
        return FaceCategory(value)
    except ValueError:
        return FaceCategory.uncertain


def map_gender(face, category: FaceCategory) -> FaceGender:
    if category != FaceCategory.real_identifiable:
        return FaceGender.unknown

    gender = getattr(face, "gender", None)

    if gender == 1:
        return FaceGender.male

    if gender == 0:
        return FaceGender.female

    return FaceGender.unknown


def load_reference_embeddings(db: Session):
    rows = (
        db.query(DBEmbedding)
        .filter(DBEmbedding.embedding_type == EmbeddingType.reference_face)
        .all()
    )

    refs = []

    for row in rows:
        refs.append(
            {
                "embedding_id": row.id,
                "person_id": row.person_id,
                "vector": normalize(np.array(row.vector, dtype=np.float32)),
            }
        )

    logging.info(f"Loaded reference embeddings: {len(refs)}")
    return refs


def find_best_known_match(embedding, reference_embeddings):
    best = None
    best_dist = 1.0

    for ref in reference_embeddings:
        dist = cosine_distance(embedding, ref["vector"])

        if dist < best_dist:
            best_dist = dist
            best = ref

    return best, best_dist


def get_next_cluster_number(db: Session, status: PersonStatus) -> int:
    max_cluster_id = (
        db.query(func.max(DBPerson.cluster_id))
        .filter(DBPerson.status == status)
        .scalar()
    )

    return (max_cluster_id or 0) + 1


def create_unknown_cluster_person(db: Session):
    cluster_id = get_next_cluster_number(db, PersonStatus.unknown)
    cluster_tag = f"unknown_cluster_{cluster_id:06d}"

    person_create = PersonsCreate(
        name=cluster_tag,
        q_code=None,
        link=None,
        status=PersonStatus.unknown,
    )

    db_person = create_person(db, person_create, code=cluster_tag)

    db_person.cluster_id = cluster_id
    db_person.cluster_tag = cluster_tag
    db_person.cluster_distance = 0.0

    db.commit()
    db.refresh(db_person)

    return db_person


def get_or_create_service_cluster_person(
    db: Session,
    category: FaceCategory,
):
    cluster_tag = f"{category.value}_cluster"

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

    db_person = create_person(db, person_create, code=cluster_tag)

    db_person.cluster_id = 0
    db_person.cluster_tag = cluster_tag
    db_person.cluster_distance = 0.0

    db.commit()
    db.refresh(db_person)

    return db_person


def load_unknown_cluster_embeddings(db: Session):
    rows = (
        db.query(DBEmbedding)
        .join(DBPerson, DBEmbedding.person_id == DBPerson.id)
        .filter(DBEmbedding.embedding_type == EmbeddingType.detected_face)
        .filter(DBPerson.status == PersonStatus.unknown)
        .all()
    )

    items = []

    for row in rows:
        items.append(
            {
                "person_id": row.person_id,
                "vector": normalize(np.array(row.vector, dtype=np.float32)),
            }
        )

    return items


def find_or_create_unknown_cluster_person(
    db: Session,
    embedding,
):
    cluster_embeddings = load_unknown_cluster_embeddings(db)

    best_person_id = None
    best_dist = 1.0

    for item in cluster_embeddings:
        dist = cosine_distance(embedding, item["vector"])

        if dist < best_dist:
            best_dist = dist
            best_person_id = item["person_id"]

    if best_person_id is not None and best_dist <= UNKNOWN_TOLERANCE:
        db_person = db.query(DBPerson).filter(DBPerson.id == best_person_id).first()

        if db_person:
            db_person.cluster_tag = db_person.cluster_tag or db_person.name
            db_person.cluster_distance = round(best_dist, 4)
            db.commit()
            db.refresh(db_person)

        return db_person, best_dist

    return create_unknown_cluster_person(db), None


def make_analysis(quality_details: dict, category_result: dict) -> dict:
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


def make_analysis_without_clip(quality_details: dict, reason: str) -> dict:
    return {
        "quality": quality_details,
        "clip": {
            "category": reason,
            "category_score": None,
            "best_clip_category": None,
            "best_clip_score": None,
            "clip_scores": None,
        },
    }


def create_detected_embedding(
    db: Session,
    person_id: int,
    embedding,
    freeze: DBFreeze,
    face,
    distance,
    category: FaceCategory,
    category_score,
    quality: float,
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
            "category": category.value,
            "category_score": category_score,
            "quality": quality,
            "analysis": analysis,
            "created_by": "detect_faces_from_freezes_v4.py",
        },
        vector=embedding.tolist(),
        person_id=person_id,
    )

    return create_embedding(db, embedding_create)


def create_face_row(
    db: Session,
    face,
    category: FaceCategory,
    category_score,
    quality: float,
    gender: FaceGender,
    confidence,
    analysis: dict,
    embedding_id: int,
    freeze_id: int,
    person_id: int,
    iteration_id: int,
):
    bbox = face.bbox.astype(float).tolist()

    face_create = FaceCreate(
        bbox=bbox,
        category=category,
        category_score=category_score,
        quality=quality,
        gender=gender,
        confidence=confidence,
        analysis=analysis,
        embedding_id=embedding_id,
        freeze_id=freeze_id,
        person_id=person_id,
        iteration_id=iteration_id,
    )

    return create_face(db, face_create)


def create_embedding_and_face(
    db: Session,
    freeze: DBFreeze,
    face,
    emb,
    person_id: int,
    distance,
    category: FaceCategory,
    category_score,
    quality: float,
    gender: FaceGender,
    confidence,
    analysis: dict,
    iteration_id: int,
):
    detected_embedding = create_detected_embedding(
        db=db,
        person_id=person_id,
        embedding=emb,
        freeze=freeze,
        face=face,
        distance=distance,
        category=category,
        category_score=category_score,
        quality=quality,
        analysis=analysis,
    )

    create_face_row(
        db=db,
        face=face,
        category=category,
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

    return detected_embedding


def process_freeze(
    db: Session,
    face_model,
    clip_model,
    clip_preprocess,
    clip_text_features,
    clip_prompt_categories,
    freeze: DBFreeze,
    iteration_id: int,
    reference_embeddings,
):
    existing_faces = get_faces_by_freeze(db, freeze.id)

    if existing_faces:
        logging.info(
            f"Skip freeze_id={freeze.id}, faces already exist: {len(existing_faces)}"
        )
        return 0

    img = cv2.imread(freeze.freeze_path)

    if img is None:
        logging.warning(f"Cannot read freeze image: {freeze.freeze_path}")
        return 0

    try:
        pil_image = Image.open(freeze.freeze_path).convert("RGB")
    except Exception as e:
        logging.warning(f"Cannot open freeze with PIL: {freeze.freeze_path} | {e}")
        return 0

    faces = face_model.get(img)

    created = 0

    for face in faces:
        bbox = face.bbox.astype(float).tolist()

        quality, quality_details = get_face_quality(img, face)
        emb = normalize(face.embedding)

        # 1. кноун
        best_ref, best_dist = find_best_known_match(emb, reference_embeddings)
        confidence = get_confidence(best_dist)

        can_use_known_match = (
            best_ref is not None
            and confidence != -1
        )

        if can_use_known_match:
            category = FaceCategory.real_identifiable
            category_score = confidence

            analysis = make_analysis_without_clip(
                quality_details=quality_details,
                reason="not_checked_known_match",
            )

            person_id = best_ref["person_id"]
            distance_for_source = round(best_dist, 4)

            create_embedding_and_face(
                db=db,
                freeze=freeze,
                face=face,
                emb=emb,
                person_id=person_id,
                distance=distance_for_source,
                category=category,
                category_score=category_score,
                quality=quality,
                gender=map_gender(face, category),
                confidence=confidence,
                analysis=analysis,
                iteration_id=iteration_id,
            )

            created += 1

            logging.info(
                f"Created known face freeze_id={freeze.id}, person_id={person_id}, "
                f"dist={distance_for_source}, confidence={confidence}, "
                f"quality={quality}"
            )

            continue

        # 2. фільтр якості.

        det_score = float(getattr(face, "det_score", 0.0) or 0.0)

        if quality < LOW_QUALITY_THRESHOLD or det_score < MIN_DET_SCORE:
            category = FaceCategory.low_quality
            category_score = quality

            reason = (
                "not_checked_low_quality"
                if quality < LOW_QUALITY_THRESHOLD
                else "not_checked_low_det_score"
            )

            analysis = make_analysis_without_clip(
                quality_details=quality_details,
                reason=reason,
            )

            service_person = get_or_create_service_cluster_person(
                db=db,
                category=category,
            )

            create_embedding_and_face(
                db=db,
                freeze=freeze,
                face=face,
                emb=emb,
                person_id=service_person.id,
                distance=None,
                category=category,
                category_score=category_score,
                quality=quality,
                gender=FaceGender.unknown,
                confidence=None,
                analysis=analysis,
                iteration_id=iteration_id,
            )

            created += 1

            logging.info(
                f"Created low_quality face freeze_id={freeze.id}, "
                f"person_id={service_person.id}, "
                f"reason={reason}, "
                f"category={category.value}:{category_score}, "
                f"quality={quality}, det_score={det_score}"
            )

            continue

        # 3.  CLIP.
        category_result = analyze_face_category(
            image=pil_image,
            bbox=bbox,
            model=clip_model,
            preprocess=clip_preprocess,
            text_features=clip_text_features,
            prompt_categories=clip_prompt_categories,
        )

        category = to_face_category(category_result["category"])
        category_score = category_result["category_score"]

        analysis = make_analysis(
            quality_details=quality_details,
            category_result=category_result,
        )

        # 4. real_identifiable — unknown,

        if category == FaceCategory.real_identifiable:
            cluster_person, cluster_dist = find_or_create_unknown_cluster_person(
                db=db,
                embedding=emb,
            )

            distance_for_source = (
                round(cluster_dist, 4)
                if cluster_dist is not None
                else None
            )

            create_embedding_and_face(
                db=db,
                freeze=freeze,
                face=face,
                emb=emb,
                person_id=cluster_person.id,
                distance=distance_for_source,
                category=category,
                category_score=category_score,
                quality=quality,
                gender=map_gender(face, category),
                confidence=None,
                analysis=analysis,
                iteration_id=iteration_id,
            )

            logging.info(
                f"Created unknown face freeze_id={freeze.id}, "
                f"person_id={cluster_person.id}, "
                f"category={category.value}:{category_score}, "
                f"cluster_dist={distance_for_source}, quality={quality}"
            )

        else:
            # 5. службові кластери

            service_person = get_or_create_service_cluster_person(
                db=db,
                category=category,
            )

            create_embedding_and_face(
                db=db,
                freeze=freeze,
                face=face,
                emb=emb,
                person_id=service_person.id,
                distance=None,
                category=category,
                category_score=category_score,
                quality=quality,
                gender=FaceGender.unknown,
                confidence=None,
                analysis=analysis,
                iteration_id=iteration_id,
            )

            logging.info(
                f"Created service-category face freeze_id={freeze.id}, "
                f"person_id={service_person.id}, "
                f"category={category.value}:{category_score}, "
                f"quality={quality}"
            )

        created += 1

    return created


def process_media(
    db: Session,
    face_model,
    clip_model,
    clip_preprocess,
    clip_text_features,
    clip_prompt_categories,
    media: DBMedia,
    reference_embeddings,
):
    freezes = (
        db.query(DBFreeze)
        .filter(DBFreeze.media_id == media.id)
        .order_by(DBFreeze.time_in)
        .all()
    )

    if not freezes:
        logging.info(f"Skip media_id={media.id}, no freezes")
        return 0

    iteration = create_iteration(
        db,
        IterationCreate(
            status=IterationStatus.processing,
            params={
                "service": "detect_faces_from_freezes_v4.py",
                "face_det_size": FACE_DET_SIZE,
                "quality_mode": "numeric",
                "category_mode": "known_first_quality_then_clip",
                "dist_tolerance": DIST_TOLERANCE,
                "step_tolerance": STEP_TOLERANCE,
                "unknown_tolerance": UNKNOWN_TOLERANCE,
                "low_quality_threshold": LOW_QUALITY_THRESHOLD,
                "logic": (
                    "detect_all -> known_match -> low_quality_service_cluster -> "
                    "clip -> unknown_similarity_only_for_real_identifiable -> "
                    "service_cluster_for_other_categories"
                ),
            },
            error_message=None,
            user_id=USER_ID,
            media_id=media.id,
        ),
    )

    total_created = 0

    try:
        for freeze in freezes:
            total_created += process_freeze(
                db=db,
                face_model=face_model,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_text_features=clip_text_features,
                clip_prompt_categories=clip_prompt_categories,
                freeze=freeze,
                iteration_id=iteration.id,
                reference_embeddings=reference_embeddings,
            )

        update_iteration(
            db,
            iteration.id,
            IterationUpdate(
                status=IterationStatus.completed,
                finished_at=datetime.now(),
            ),
        )

        logging.info(f"media_id={media.id}: faces created={total_created}")
        return total_created

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


def process_all_media():
    db = SessionLocal()

    try:
        logging.info("Loading InsightFace...")
        face_model = get_insightface()

        logging.info("Loading CLIP...")
        clip_model, clip_preprocess, clip_text_features, clip_prompt_categories = get_clip()

        reference_embeddings = load_reference_embeddings(db)

        medias = db.query(DBMedia).order_by(DBMedia.id).all()

        total_faces = 0

        for media in medias:
            total_faces += process_media(
                db=db,
                face_model=face_model,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_text_features=clip_text_features,
                clip_prompt_categories=clip_prompt_categories,
                media=media,
                reference_embeddings=reference_embeddings,
            )

        logging.info("--------------------------------")
        logging.info(f"Total faces created: {total_faces}")

    finally:
        db.close()


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Start: {start}")

    process_all_media()

    finish = datetime.now()
    logging.info(f"Finished. Running time: {finish - start}")
