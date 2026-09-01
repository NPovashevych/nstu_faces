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

from routes.routers_classic import normalize, cosine_distance

from services.create_faces.face_quality_v3 import get_face_quality
from services.create_faces.clip_face_filter_v2 import get_clip, analyze_face_category
from services.create_faces.clip_face_categories import (
    IDENTIFIABLE_CATEGORIES,
    NON_IDENTIFIABLE_CATEGORIES,
)


Path("../services/logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../services/logs/detect_faces_from_freezes_v2.log", encoding="utf-8"),
    ],
)


FACE_DET_SIZE = 960

DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.03

UNKNOWN_TOLERANCE = 0.50
SUSPICIOUS_TOLERANCE = 0.50

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


def category_is_identifiable(category: FaceCategory) -> bool:
    return category.value in IDENTIFIABLE_CATEGORIES


def category_is_non_identifiable(category: FaceCategory) -> bool:
    return category.value in NON_IDENTIFIABLE_CATEGORIES


def get_cluster_status_for_category(category: FaceCategory) -> PersonStatus:
    if category == FaceCategory.real_identifiable:
        return PersonStatus.unknown

    return PersonStatus.suspicious


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


def create_cluster_person(db: Session, status: PersonStatus):
    cluster_id = get_next_cluster_number(db, status)

    cluster_prefix = (
        "suspicious_cluster"
        if status == PersonStatus.suspicious
        else "unknown_cluster"
    )

    cluster_tag = f"{cluster_prefix}_{cluster_id:06d}"

    person_create = PersonsCreate(
        name=cluster_tag,
        q_code=None,
        link=None,
        status=status,
    )

    db_person = create_person(db, person_create, code=cluster_tag)

    db_person.cluster_id = cluster_id
    db_person.cluster_tag = cluster_tag
    db_person.cluster_distance = 0.0

    db.commit()
    db.refresh(db_person)

    return db_person


def load_cluster_embeddings(db: Session, status: PersonStatus):
    rows = (
        db.query(DBEmbedding)
        .join(DBPerson, DBEmbedding.person_id == DBPerson.id)
        .filter(DBEmbedding.embedding_type == EmbeddingType.detected_face)
        .filter(DBPerson.status == status)
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


def find_or_create_cluster_person(
    db: Session,
    embedding,
    status: PersonStatus,
):
    tolerance = (
        SUSPICIOUS_TOLERANCE
        if status == PersonStatus.suspicious
        else UNKNOWN_TOLERANCE
    )

    cluster_embeddings = load_cluster_embeddings(db, status)

    best_person_id = None
    best_dist = 1.0

    for item in cluster_embeddings:
        dist = cosine_distance(embedding, item["vector"])

        if dist < best_dist:
            best_dist = dist
            best_person_id = item["person_id"]

    if best_person_id is not None and best_dist <= tolerance:
        db_person = db.query(DBPerson).filter(DBPerson.id == best_person_id).first()

        if db_person:
            db_person.cluster_tag = db_person.cluster_tag or db_person.name
            db_person.cluster_distance = round(best_dist, 4)
            db.commit()
            db.refresh(db_person)

        return db_person, best_dist

    return create_cluster_person(db, status), None


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
            "created_by": "detect_faces_from_freezes_v2.py",
        },
        vector=embedding.tolist(),
        person_id=person_id,
    )

    return create_embedding(db, embedding_create)


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

        emb = normalize(face.embedding)

        best_ref, best_dist = find_best_known_match(emb, reference_embeddings)
        confidence = get_confidence(best_dist)

        can_use_known_match = (
            best_ref is not None
            and confidence != -1
            and category_is_identifiable(category)
        )

        if can_use_known_match:
            person_id = best_ref["person_id"]
            distance_for_source = round(best_dist, 4)
            face_confidence = confidence
            person_status_for_log = "known"

        else:
            cluster_status = get_cluster_status_for_category(category)

            cluster_person, cluster_dist = find_or_create_cluster_person(
                db=db,
                embedding=emb,
                status=cluster_status,
            )

            person_id = cluster_person.id
            distance_for_source = (
                round(cluster_dist, 4)
                if cluster_dist is not None
                else None
            )
            face_confidence = None
            person_status_for_log = cluster_status.value

        detected_embedding = create_detected_embedding(
            db=db,
            person_id=person_id,
            embedding=emb,
            freeze=freeze,
            face=face,
            distance=distance_for_source,
            category=category,
            category_score=category_score,
            quality=quality,
            analysis=analysis,
        )

        face_create = FaceCreate(
            bbox=bbox,
            category=category,
            category_score=category_score,
            quality=quality,
            gender=map_gender(face, category),
            confidence=face_confidence,
            analysis=analysis,
            embedding_id=detected_embedding.id,
            freeze_id=freeze.id,
            person_id=person_id,
            iteration_id=iteration_id,
        )

        create_face(db, face_create)
        created += 1

        logging.info(
            f"Created face freeze_id={freeze.id}, person_id={person_id}, "
            f"status={person_status_for_log}, "
            f"category={category.value}:{category_score}, "
            f"quality={quality}, "
            f"confidence={face_confidence}"
        )

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
                "service": "detect_faces_from_freezes_v3.py",
                "face_det_size": FACE_DET_SIZE,
                "quality_mode": "numeric",
                "category_mode": "clip_face_categories",
                "dist_tolerance": DIST_TOLERANCE,
                "unknown_tolerance": UNKNOWN_TOLERANCE,
                "suspicious_tolerance": SUSPICIOUS_TOLERANCE,
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
