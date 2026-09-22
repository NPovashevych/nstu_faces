import logging
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.enums import IterationStatus
from db.models import DBFreeze, DBIteration, DBMedia

from crud.crud_iteration import create_iteration, update_iteration

from schemas.schemas_iteration import IterationCreate, IterationUpdate

from commons.clip_face_filter_v2 import get_clip
from services.faiss.faiss_face_index import ReferenceFaceIndex, UnknownFaceIndex
from commons.common_face import get_insightface, load_face_categories, process_freeze
from commons.common_face import FACE_DET_SIZE, MIN_DET_SCORE, DIST_TOLERANCE, STEP_TOLERANCE, UNKNOWN_TOLERANCE, LOW_QUALITY_THRESHOLD

from services.tests.test_speed_add_faces import PerformanceProfiler


Path("../logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../logs/detect_faces_from_freezes_v6.log", encoding="utf-8"),
    ],
)


SERVICE_NAME = "detect_faces_from_freezes_v6.py"
USER_ID = 1


def get_media_for_face_detection(db: Session):
    processed_media_ids = db.query(DBIteration.media_id).filter(DBIteration.status == IterationStatus.completed).subquery()

    medias = (
        db.query(DBMedia)
        .join(DBFreeze, DBFreeze.media_id == DBMedia.id)
        .filter(~DBMedia.id.in_(processed_media_ids))
        .distinct()
        .order_by(DBMedia.id)
        .all()
    )

    logging.info(f"Media to process: {len(medias)}")

    return medias


def process_media(db, face_model, clip_model, clip_preprocess, clip_text_features, clip_prompt_categories, media, reference_index, unknown_index, face_categories, profiler):
    freezes = db.query(DBFreeze).filter(DBFreeze.media_id == media.id).order_by(DBFreeze.time_in).all()

    if not freezes:
        logging.info(f"Skip media_id={media.id}, no freezes")
        return 0

    iteration = create_iteration(
        db,
        IterationCreate(
            status=IterationStatus.processing,
            params={
                "service": SERVICE_NAME,
                "service_type": "face_detection",
                "service_version": "v6",
                "face_det_size": FACE_DET_SIZE,
                "min_det_score": MIN_DET_SCORE,
                "dist_tolerance": DIST_TOLERANCE,
                "step_tolerance": STEP_TOLERANCE,
                "unknown_tolerance": UNKNOWN_TOLERANCE,
                "low_quality_threshold": LOW_QUALITY_THRESHOLD,
                "reference_search": "faiss_index_flat_ip_exact",
                "unknown_search": "faiss_index_flat_ip_exact_dynamic",
                "logic": (
                    "media_without_completed_iteration -> freeze -> buffalo_detect -> reference_faiss_known_match_first -> "
                    "drop_low_det_score_without_embedding -> quality -> low_quality_cluster -> clip -> "
                    "identifiable_unknown_faiss_grouping -> dynamic_unknown_faiss_add -> single_service_clusters_for_other_categories"
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
                reference_index=reference_index,
                unknown_index=unknown_index,
                face_categories=face_categories,
                profiler=profiler,
                service_name=SERVICE_NAME,
            )

        update_iteration(db, iteration.id, IterationUpdate(status=IterationStatus.completed, finished_at=datetime.now()))

        logging.info(f"media_id={media.id}: completed, faces created={total_created}")

        return total_created

    except Exception as e:
        db.rollback()

        update_iteration(db, iteration.id, IterationUpdate(status=IterationStatus.error, finished_at=datetime.now(), error_message=str(e)))

        logging.exception(f"media_id={media.id}: error during face detection")

        raise


def process_all_media():
    db = SessionLocal()

    try:
        logging.info("Loading InsightFace...")
        face_model = get_insightface()

        logging.info("Loading CLIP...")
        clip_model, clip_preprocess, clip_text_features, clip_prompt_categories = get_clip()

        face_categories = load_face_categories(db)

        logging.info("Building reference FAISS...")
        reference_index = ReferenceFaceIndex()
        reference_index.build(db)

        logging.info(f"Reference FAISS ready: {reference_index.size} embeddings")

        logging.info("Building existing unknown FAISS...")
        unknown_index = UnknownFaceIndex()
        unknown_index.build(db)

        logging.info(f"Unknown FAISS ready: {unknown_index.size} embeddings, {unknown_index.person_count} persons")

        medias = get_media_for_face_detection(db)

        total_faces = 0
        total_media = len(medias)

        profiler = PerformanceProfiler(log_every_freezes=500)

        for media_number, media in enumerate(medias, start=1):
            logging.info("--------------------------------")
            logging.info(f"MEDIA {media_number}/{total_media} | media_id={media.id}")

            total_faces += process_media(
                db=db,
                face_model=face_model,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_text_features=clip_text_features,
                clip_prompt_categories=clip_prompt_categories,
                media=media,
                reference_index=reference_index,
                unknown_index=unknown_index,
                face_categories=face_categories,
                profiler=profiler,
            )

        logging.info("--------------------------------")
        logging.info(f"Total faces created: {total_faces}")
        logging.info(f"Final reference FAISS size: {reference_index.size}")
        logging.info(f"Final unknown FAISS size: {unknown_index.size}")
        logging.info(f"Final unknown persons: {unknown_index.person_count}")

    finally:
        db.close()


if __name__ == "__main__":
    start = datetime.now()

    logging.info(f"Start: {start}")

    process_all_media()

    finish = datetime.now()

    logging.info(f"Finished: {finish}")
    logging.info(f"Running time: {finish - start}")
