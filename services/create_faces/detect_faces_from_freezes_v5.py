import logging
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from sqlalchemy import func
from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.enums import EmbeddingType, FaceGender, IterationStatus, PersonStatus
from db.models import DBEmbedding, DBFace, DBFaceCategory, DBFreeze, DBIteration, DBMedia, DBPerson


from crud.crud_face import get_faces_by_freeze
from crud.crud_iteration import create_iteration, update_iteration

from schemas.schemas_iteration import IterationCreate, IterationUpdate

from services.create_faces.face_quality_v3 import get_face_quality
from services.create_faces.clip_face_filter_v2 import get_clip, analyze_face_category
from services.create_faces.clip_face_categories import DEFAULT_FACE_CATEGORY,CATEGORY_IDENTIFIABLE, CATEGORY_LOW_QUALITY
from services.create_faces.faiss_face_index import ReferenceFaceIndex, UnknownFaceIndex, normalize_vector

from test_speed_add_faces import PerformanceProfiler

Path("../logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            "../logs/detect_faces_from_freezes_v5.log",
            encoding="utf-8",
        ),
    ],
)


SERVICE_NAME = "detect_faces_from_freezes_v5.py"

FACE_DET_SIZE = 640
MIN_DET_SCORE = 0.60
DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.055
UNKNOWN_TOLERANCE = 0.55
LOW_QUALITY_THRESHOLD = 0.60

USER_ID = 1
_INSIGHTFACE_CACHE = None


def load_insightface():
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))
    return app


def get_insightface():
    global _INSIGHTFACE_CACHE

    if _INSIGHTFACE_CACHE is None:
        _INSIGHTFACE_CACHE = load_insightface()

    return _INSIGHTFACE_CACHE


def load_face_categories(db: Session):
    rows = db.query(DBFaceCategory).filter(DBFaceCategory.is_active.is_(True)).all()
    categories = {row.name: row for row in rows}

    logging.info(f"Loaded active face categories: {len(categories)}")

    return categories


def get_category(face_categories: dict, category_name: str) -> DBFaceCategory:
    category = face_categories.get(category_name)
    if category:
        return category

    fallback = face_categories.get(DEFAULT_FACE_CATEGORY)
    if fallback:
        logging.warning(f"Face category '{category_name}' not found. Fallback to '{DEFAULT_FACE_CATEGORY}'.")
        return fallback

    raise RuntimeError(f"Face category '{category_name}' not found, fallback '{DEFAULT_FACE_CATEGORY}' not found.")


def get_media_for_face_detection(db: Session):
    processed_media_ids = db.query(DBIteration.media_id).filter(DBIteration.status == IterationStatus.completed).subquery()
    medias = db.query(DBMedia).join(DBFreeze, DBFreeze.media_id == DBMedia.id).filter(~DBMedia.id.in_(processed_media_ids)).distinct().order_by(DBMedia.id).all()

    logging.info(f"Media to process: {len(medias)}")
    return medias


def get_confidence(dist: float) -> int:
    if dist <= DIST_TOLERANCE:
        return 0

    for i in range(1, 4):
        if dist <= DIST_TOLERANCE + i * STEP_TOLERANCE:
            return i
    return -1


def get_next_cluster_number(db: Session, status: PersonStatus) -> int:
    max_cluster_id = db.query(func.max(DBPerson.cluster_id)).filter(DBPerson.status == status).scalar()
    return (max_cluster_id or 0) + 1


def create_unknown_cluster_person(db: Session):
    cluster_id = get_next_cluster_number(db, PersonStatus.unknown)
    cluster_tag = f"unknown_cluster_{cluster_id:06d}"

    db_person = DBPerson(
        code=cluster_tag,
        name=cluster_tag,
        q_code=None,
        link=None,
        status=PersonStatus.unknown,
        cluster_id=cluster_id,
        cluster_tag=cluster_tag,
        cluster_distance=0.0,
    )

    db.add(db_person)
    db.flush()

    return db_person


def find_or_create_unknown_cluster_person(db: Session, embedding, unknown_index: UnknownFaceIndex):
    best_person_id, best_dist = unknown_index.find_best_match(embedding)

    if best_person_id is not None and best_dist is not None and best_dist <= UNKNOWN_TOLERANCE:
        db_person = db.query(DBPerson).filter(DBPerson.id == best_person_id).first()

        if db_person is None:
            raise RuntimeError(f"Unknown FAISS returned person_id={best_person_id}, but DBPerson was not found.")

        db_person.cluster_tag = db_person.cluster_tag or db_person.name
        db_person.cluster_distance = round(best_dist, 4)

        return db_person, best_dist

    return create_unknown_cluster_person(db), None


def get_or_create_service_cluster_person(db: Session, category_name: str):
    cluster_tag = f"{category_name}_cluster"

    db_person = db.query(DBPerson).filter(DBPerson.code == cluster_tag).first()

    if db_person:
        return db_person

    db_person = DBPerson(
        code=cluster_tag,
        name=cluster_tag,
        q_code=None,
        link=None,
        status=PersonStatus.suspicious,
        cluster_id=0,
        cluster_tag=cluster_tag,
        cluster_distance=0.0,
    )

    db.add(db_person)
    db.flush()

    return db_person

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


def make_analysis_without_clip(quality_details: dict, reason: str, category_name: str) -> dict:
    return {
        "quality": quality_details,
        "clip": {"category": category_name, "category_score": None, "best_clip_category": reason, "best_clip_score": None, "clip_scores": None},
    }


def map_gender(face, category_name: str) -> FaceGender:

    if category_name != CATEGORY_IDENTIFIABLE:
        return FaceGender.unknown

    gender = getattr(face, "gender", None)

    if gender == 1:
        return FaceGender.male

    if gender == 0:
        return FaceGender.female

    return FaceGender.unknown


def create_detected_embedding(db, person_id, embedding, freeze: DBFreeze, face, distance, category_name, category_score, quality, analysis):
    bbox = face.bbox.astype(float).tolist()

    db_embedding = DBEmbedding(
        embedding_type=EmbeddingType.detected_face,
        source={
            "freeze_id": freeze.id,
            "freeze_path": freeze.freeze_path,
            "bbox": bbox,
            "distance": distance,
            "category": category_name,
            "category_score": category_score,
            "quality": quality,
            "analysis": analysis,
            "created_by": SERVICE_NAME,
        },
        vector=embedding.tolist(),
        person_id=person_id,
    )

    db.add(db_embedding)
    db.flush()

    return db_embedding


def create_face_row(db, face, category_id, category_score, quality, gender, confidence, analysis, embedding_id, freeze_id, person_id, iteration_id):
    bbox = face.bbox.astype(float).tolist()
    db_face = DBFace(
        bbox=bbox,
        category_id=category_id,
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

    db.add(db_face)

    return db_face

def create_embedding_and_face(db, freeze, face, emb, person_id, distance, category, category_score, quality, gender, confidence,analysis, iteration_id):
    detected_embedding = (
        create_detected_embedding(
            db=db,
            person_id=person_id,
            embedding=emb,
            freeze=freeze,
            face=face,
            distance=distance,
            category_name=category.name,
            category_score=category_score,
            quality=quality,
            analysis=analysis,
        )
    )

    create_face_row(
        db=db,
        face=face,
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

    return detected_embedding


def process_freeze(
        db, face_model, clip_model, clip_preprocess, clip_text_features, clip_prompt_categories, freeze,
        iteration_id, reference_index, unknown_index, face_categories, profiler):

    existing_faces = get_faces_by_freeze(db, freeze.id)
    if existing_faces:
        logging.info(f"Skip freeze_id={freeze.id}, faces already exist: {len(existing_faces)}")
        return 0

    t0 = perf_counter()
    img = cv2.imread(freeze.freeze_path)
    profiler.add("read_time", perf_counter() - t0)

    if img is None:
        logging.warning(f"Cannot read freeze image: {freeze.freeze_path}")
        return 0

    t0 = perf_counter()
    faces = face_model.get(img)
    profiler.add("buffalo_time", perf_counter() - t0)

    created = 0
    skipped_low_det_score = 0

    pil_image = None

    for face in faces:
        bbox = (face.bbox.astype(float).tolist())
        det_score = float(getattr(face, "det_score", 0.0) or 0.0)
        emb = normalize_vector(face.embedding)

        t0 = perf_counter()
        best_ref, best_dist  = reference_index.find_best_match(emb)
        profiler.add("reference_faiss_time", perf_counter() - t0)

        confidence = get_confidence(best_dist)

        can_use_known_match = best_ref is not None and confidence != -1

        if can_use_known_match:
            category = get_category(face_categories, CATEGORY_IDENTIFIABLE)
            category_score = confidence

            t0 = perf_counter()
            quality, quality_details = get_face_quality(img, face)
            profiler.add("quality_time",perf_counter() - t0)

            analysis = make_analysis_without_clip(quality_details=quality_details, reason="not_checked_known_match", category_name=category.name)

            person_id = best_ref["person_id"]
            distance_for_source = round(best_dist, 4)

            t0 = perf_counter()
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
                gender=map_gender(face, category.name),
                confidence=confidence,
                analysis=analysis,
                iteration_id=iteration_id,
            )
            profiler.add("db_time", perf_counter() - t0)

            created += 1

            logging.info(f"Created known face freeze_id={freeze.id}, person_id={person_id}, reference_embedding_id="
                f"{best_ref['embedding_id']}, dist={distance_for_source}, confidence={confidence}, det_score={det_score}, quality={quality}")

            continue

        if det_score < MIN_DET_SCORE:
            skipped_low_det_score += 1
            logging.info(f"Skip face low det_score freeze_id={freeze.id}, det_score={det_score}, bbox={bbox}")
            continue

        t0 = perf_counter()
        quality, quality_details = get_face_quality(img, face)
        profiler.add("quality_time", perf_counter() - t0)

        if quality < LOW_QUALITY_THRESHOLD:
            category = get_category(face_categories, CATEGORY_LOW_QUALITY)
            category_score = quality
            analysis = (
                make_analysis_without_clip(quality_details=quality_details, reason="not_checked_low_quality", category_name=category.name)
            )

            service_person = get_or_create_service_cluster_person(db=db, category_name=category.name)

            t0 = perf_counter()
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
            profiler.add("db_time", perf_counter() - t0)

            created += 1

            logging.info(f"Created low_quality face freeze_id={freeze.id}, person_id={service_person.id}, category="
                f"{category.name}: {category_score}, quality={quality}, det_score={det_score}")

            continue
        if pil_image is None:
            try:
                pil_image = Image.open(freeze.freeze_path).convert("RGB")

            except Exception as e:
                logging.warning(f"Cannot open freeze with PIL: {freeze.freeze_path} | {e}")
                return created

        t0 = perf_counter()
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
        profiler.add("clip_time", perf_counter() - t0)

        category_name = category_result.get("category") or DEFAULT_FACE_CATEGORY
        category = get_category(face_categories, category_name)
        category_score = category_result["category_score"]

        analysis = make_analysis(quality_details=quality_details, category_result=category_result)
        if category.name == CATEGORY_IDENTIFIABLE:
            t0 = perf_counter()
            cluster_person, cluster_dist = find_or_create_unknown_cluster_person(
                    db=db,
                    embedding=emb,
                    unknown_index=unknown_index)
            profiler.add("unknown_faiss_time", perf_counter() - t0)

            distance_for_source = round(cluster_dist, 4) if cluster_dist is not None else None

            t0 = perf_counter()
            detected_embedding = create_embedding_and_face(
                db=db,
                freeze=freeze,
                face=face,
                emb=emb,
                person_id=cluster_person.id,
                distance=distance_for_source,
                category=category,
                category_score=category_score,
                quality=quality,
                gender=map_gender(face, category.name),
                confidence=None,
                analysis=analysis,
                iteration_id=iteration_id)
            profiler.add("db_time", perf_counter() - t0)

            t0 = perf_counter()
            unknown_index.add(emb, cluster_person.id)
            profiler.add("unknown_faiss_time", perf_counter() - t0)

            created += 1
            logging.info(
                f"Created unknown identifiable face freeze_id={freeze.id}, person_id={cluster_person.id}, "
                f"embedding_id= {detected_embedding.id}, category={category.name}: {category_score}, cluster_dist={distance_for_source},"
                f"quality={quality}, det_score={det_score}, unknown_index_size={unknown_index.size}"
            )
            continue

        service_person = get_or_create_service_cluster_person(db=db, category_name=category.name)

        t0 = perf_counter()
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
        profiler.add("db_time", perf_counter() - t0)
        created += 1
        logging.info(
            f"Created service-category face freeze_id={freeze.id}, person_id={service_person.id}, "
            f"category={category.name}: {category_score}, quality={quality}, det_score={det_score}"
        )

    logging.info(
        f"freeze_id={freeze.id}: created={created}, skipped_low_det_score={skipped_low_det_score}, "
        f"detected_by_buffalo={len(faces)}"
    )

    profiler.freeze_completed(buffalo_faces=len(faces), created_faces=created)

    return created


def process_media(
        db, face_model, clip_model, clip_preprocess, clip_text_features, clip_prompt_categories,
        media, reference_index, unknown_index, face_categories, profiler
):

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
                "service_version": "v5",
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
                profiler=profiler
            )

        update_iteration(db, iteration.id, IterationUpdate(status=IterationStatus.completed,finished_at=datetime.now()))
        logging.info(f"media_id={media.id}: completed, faces created={total_created}")
        return total_created

    except Exception as e:
        # Відкочуємо всі embedding / face / person, які були створені для незавершеного media.
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