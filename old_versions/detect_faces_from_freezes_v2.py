import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from sqlalchemy.orm import Session
from sqlalchemy import func

from db.session import SessionLocal
from db.enums import EmbeddingType, FaceGender, IterationStatus, PersonStatus
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
from old_versions.face_quality_v2 import is_good_face
from old_versions.clip_face_filter import get_clip, check_face_suspicion


Path("../services/logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../services/logs/detect_faces_from_freezes_v2.log", encoding="utf-8"),
    ],
)


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
    app.prepare(ctx_id=0, det_size=(640, 640))
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


def map_gender(face) -> FaceGender:
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

    if status == PersonStatus.suspicious:
        cluster_prefix = "suspicious_cluster"
    else:
        cluster_prefix = "unknown_cluster"

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


def create_detected_embedding(
    db: Session,
    person_id: int,
    embedding,
    freeze: DBFreeze,
    face,
    distance,
    clip_result,
):
    bbox = face.bbox.astype(float).tolist()

    embedding_create = EmbeddingCreate(
        embedding_type=EmbeddingType.detected_face,
        source={
            "freeze_id": freeze.id,
            "freeze_path": freeze.freeze_path,
            "bbox": bbox,
            "distance": distance,
            "clip_category": clip_result["category"],
            "clip_score": clip_result["score"],
            "is_suspicious": clip_result["is_suspicious"],
            "suspicion_reason": clip_result["reason"],
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
        logging.info(f"Skip freeze_id={freeze.id}, faces already exist: {len(existing_faces)}")
        return 0

    img = cv2.imread(freeze.freeze_path)

    if img is None:
        logging.warning(f"Cannot read freeze image: {freeze.freeze_path}")
        return 0

    pil_image = Image.open(freeze.freeze_path).convert("RGB")

    faces = face_model.get(img)

    created = 0

    for face in faces:
        if not is_good_face(img, face):
            logging.info(
                f"Skip low quality face freeze_id={freeze.id}, "
                f"bbox={face.bbox.astype(int).tolist()}"
            )
            continue

        clip_result = check_face_suspicion(
            image=pil_image,
            bbox=face.bbox.astype(float).tolist(),
            model=clip_model,
            preprocess=clip_preprocess,
            text_features=clip_text_features,
            prompt_categories=clip_prompt_categories,
        )

        emb = normalize(face.embedding)

        best_ref, best_dist = find_best_known_match(emb, reference_embeddings)
        confidence = get_confidence(best_dist)

        if best_ref is not None and confidence != -1 and not clip_result["is_suspicious"]:
            person_id = best_ref["person_id"]
            distance_for_source = round(best_dist, 4)
            face_confidence = confidence
            person_status_for_log = "known"

        else:
            cluster_status = (
                PersonStatus.suspicious
                if clip_result["is_suspicious"]
                else PersonStatus.unknown
            )

            cluster_person, cluster_dist = find_or_create_cluster_person(
                db=db,
                embedding=emb,
                status=cluster_status,
            )

            person_id = cluster_person.id
            distance_for_source = round(cluster_dist, 4) if cluster_dist is not None else None
            face_confidence = None
            person_status_for_log = cluster_status.value

        detected_embedding = create_detected_embedding(
            db=db,
            person_id=person_id,
            embedding=emb,
            freeze=freeze,
            face=face,
            distance=distance_for_source,
            clip_result=clip_result,
        )

        face_create = FaceCreate(
            bbox=face.bbox.astype(float).tolist(),
            gender=map_gender(face),
            quality=1.0,
            confidence=face_confidence,

            is_suspicious=clip_result["is_suspicious"],
            suspicion_reason=clip_result["reason"],
            clip_category=clip_result["category"],
            clip_score=clip_result["score"],
            clip_scores=clip_result["all_scores"],

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
            f"clip={clip_result['category']}:{clip_result['score']}, "
            f"suspicious={clip_result['is_suspicious']}"
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
                "service": "detect_faces_from_freezes_v2.py",
                "quality_filter": "face_quality_v2",
                "clip_filter": True,
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
