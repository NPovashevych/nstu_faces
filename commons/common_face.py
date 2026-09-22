import logging
from time import perf_counter

import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from sqlalchemy import func
from sqlalchemy.orm import Session

from db.enums import EmbeddingType, FaceGender, PersonStatus
from db.models import DBEmbedding, DBFace, DBFaceCategory, DBFreeze, DBPerson

from crud.crud_face import get_faces_by_freeze

from commons.face_quality_v3 import get_face_quality
from commons.clip_face_filter_v2 import get_clip, analyze_face_category
from commons.clip_face_categories import DEFAULT_FACE_CATEGORY, CATEGORY_IDENTIFIABLE, CATEGORY_LOW_QUALITY
from services.faiss.faiss_face_index import UnknownFaceIndex, normalize_vector


FACE_DET_SIZE = 640
MIN_DET_SCORE = 0.60
DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.055
UNKNOWN_TOLERANCE = 0.55
LOW_QUALITY_THRESHOLD = 0.60


_INSIGHTFACE_CACHE = None
_CLIP_CACHE = None


def load_insightface():
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))
    return app


def get_insightface():
    global _INSIGHTFACE_CACHE

    if _INSIGHTFACE_CACHE is None:
        _INSIGHTFACE_CACHE = load_insightface()

    return _INSIGHTFACE_CACHE


def get_clip_cached():
    global _CLIP_CACHE

    if _CLIP_CACHE is None:
        _CLIP_CACHE = get_clip()

    return _CLIP_CACHE


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

    db_person = DBPerson(code=cluster_tag, name=cluster_tag, q_code=None, link=None, status=PersonStatus.unknown, cluster_id=cluster_id, cluster_tag=cluster_tag, cluster_distance=0.0)

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

    db_person = DBPerson(code=cluster_tag, name=cluster_tag, q_code=None, link=None, status=PersonStatus.suspicious, cluster_id=0, cluster_tag=cluster_tag, cluster_distance=0.0)

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
        "clip": {
            "category": category_name,
            "category_score": None,
            "best_clip_category": reason,
            "best_clip_score": None,
            "clip_scores": None,
        },
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


def create_detected_embedding(db, person_id, embedding, freeze: DBFreeze, face, distance, category_name, category_score, quality, analysis, service_name):
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
            "created_by": service_name,
        },
        vector=embedding.tolist(),
        person_id=person_id,
    )

    db.add(db_embedding)
    db.flush()

    return db_embedding


def create_face_row(db, face, category_id, category_score, quality, gender, confidence, analysis, embedding_id, freeze_id, person_id, iteration_id):
    bbox = face.bbox.astype(float).tolist()

    db_face = DBFace(bbox=bbox, category_id=category_id, category_score=category_score, quality=quality, gender=gender, confidence=confidence, analysis=analysis, embedding_id=embedding_id, freeze_id=freeze_id, person_id=person_id, iteration_id=iteration_id)

    db.add(db_face)

    return db_face


def create_embedding_and_face(db, freeze, face, emb, person_id, distance, category, category_score, quality, gender, confidence, analysis, iteration_id, service_name):
    detected_embedding = create_detected_embedding(db=db, person_id=person_id, embedding=emb, freeze=freeze, face=face, distance=distance, category_name=category.name, category_score=category_score, quality=quality, analysis=analysis, service_name=service_name)

    create_face_row(db=db, face=face, category_id=category.id, category_score=category_score, quality=quality, gender=gender, confidence=confidence, analysis=analysis, embedding_id=detected_embedding.id, freeze_id=freeze.id, person_id=person_id, iteration_id=iteration_id)

    return detected_embedding


def analyze_faces(img, face_model, clip_model, clip_preprocess, clip_text_features, clip_prompt_categories, reference_index, unknown_index, face_categories, profiler=None):
    if img is None:
        return []

    t0 = perf_counter()
    faces = face_model.get(img)

    if profiler:
        profiler.add("buffalo_time", perf_counter() - t0)

    results = []
    pil_image = None

    for face_index, face in enumerate(faces):
        bbox = face.bbox.astype(float).tolist()
        det_score = float(getattr(face, "det_score", 0.0) or 0.0)
        emb = normalize_vector(face.embedding)

        t0 = perf_counter()
        best_ref, best_dist = reference_index.find_best_match(emb)

        if profiler:
            profiler.add("reference_faiss_time", perf_counter() - t0)

        confidence = get_confidence(best_dist)
        can_use_known_match = best_ref is not None and confidence != -1

        if can_use_known_match:
            category = get_category(face_categories, CATEGORY_IDENTIFIABLE)

            t0 = perf_counter()
            quality, quality_details = get_face_quality(img, face)

            if profiler:
                profiler.add("quality_time", perf_counter() - t0)

            results.append(
                {
                    "status": "known",
                    "face_index": face_index,
                    "face": face,
                    "bbox": bbox,
                    "embedding": emb,
                    "det_score": det_score,
                    "person_id": best_ref["person_id"],
                    "reference_embedding_id": best_ref["embedding_id"],
                    "distance": round(best_dist, 4),
                    "category": category,
                    "category_score": confidence,
                    "quality": quality,
                    "gender": map_gender(face, category.name),
                    "confidence": confidence,
                    "analysis": make_analysis_without_clip(quality_details=quality_details, reason="not_checked_known_match", category_name=category.name),
                }
            )

            continue

        if det_score < MIN_DET_SCORE:
            results.append(
                {
                    "status": "skipped_low_det_score",
                    "face_index": face_index,
                    "face": face,
                    "bbox": bbox,
                    "embedding": emb,
                    "det_score": det_score,
                    "person_id": None,
                    "distance": None,
                    "category": None,
                    "category_score": None,
                    "quality": None,
                    "gender": FaceGender.unknown,
                    "confidence": None,
                    "analysis": None,
                }
            )

            continue

        t0 = perf_counter()
        quality, quality_details = get_face_quality(img, face)

        if profiler:
            profiler.add("quality_time", perf_counter() - t0)

        if quality < LOW_QUALITY_THRESHOLD:
            category = get_category(face_categories, CATEGORY_LOW_QUALITY)

            results.append(
                {
                    "status": "low_quality",
                    "face_index": face_index,
                    "face": face,
                    "bbox": bbox,
                    "embedding": emb,
                    "det_score": det_score,
                    "person_id": None,
                    "distance": None,
                    "category": category,
                    "category_score": quality,
                    "quality": quality,
                    "gender": FaceGender.unknown,
                    "confidence": None,
                    "analysis": make_analysis_without_clip(quality_details=quality_details, reason="not_checked_low_quality", category_name=category.name),
                }
            )

            continue

        if pil_image is None:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

        t0 = perf_counter()
        category_result = analyze_face_category(image=pil_image, bbox=bbox, model=clip_model, preprocess=clip_preprocess, text_features=clip_text_features, prompt_categories=clip_prompt_categories)

        if profiler:
            profiler.add("clip_time", perf_counter() - t0)

        category_name = category_result.get("category") or DEFAULT_FACE_CATEGORY
        category = get_category(face_categories, category_name)
        category_score = category_result["category_score"]
        analysis = make_analysis(quality_details=quality_details, category_result=category_result)

        if category.name == CATEGORY_IDENTIFIABLE:
            t0 = perf_counter()
            unknown_person_id, cluster_dist = unknown_index.find_best_match(emb)

            if profiler:
                profiler.add("unknown_faiss_time", perf_counter() - t0)

            if unknown_person_id is None or cluster_dist is None or cluster_dist > UNKNOWN_TOLERANCE:
                unknown_person_id = None
                cluster_dist = None

            results.append(
                {
                    "status": "unknown_identifiable",
                    "face_index": face_index,
                    "face": face,
                    "bbox": bbox,
                    "embedding": emb,
                    "det_score": det_score,
                    "person_id": unknown_person_id,
                    "distance": round(cluster_dist, 4) if cluster_dist is not None else None,
                    "category": category,
                    "category_score": category_score,
                    "quality": quality,
                    "gender": map_gender(face, category.name),
                    "confidence": None,
                    "analysis": analysis,
                }
            )

            continue

        results.append(
            {
                "status": "service_category",
                "face_index": face_index,
                "face": face,
                "bbox": bbox,
                "embedding": emb,
                "det_score": det_score,
                "person_id": None,
                "distance": None,
                "category": category,
                "category_score": category_score,
                "quality": quality,
                "gender": FaceGender.unknown,
                "confidence": None,
                "analysis": analysis,
            }
        )

    return results


def process_freeze(db, face_model, clip_model, clip_preprocess, clip_text_features, clip_prompt_categories, freeze, iteration_id, reference_index, unknown_index, face_categories, profiler, service_name):
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

    results = analyze_faces(img=img, face_model=face_model, clip_model=clip_model, clip_preprocess=clip_preprocess, clip_text_features=clip_text_features, clip_prompt_categories=clip_prompt_categories, reference_index=reference_index, unknown_index=unknown_index, face_categories=face_categories, profiler=profiler)

    created = 0
    skipped_low_det_score = 0

    for result in results:
        status = result["status"]
        face = result["face"]
        emb = result["embedding"]
        bbox = result["bbox"]
        det_score = result["det_score"]

        if status == "skipped_low_det_score":
            skipped_low_det_score += 1
            logging.info(f"Skip face low det_score freeze_id={freeze.id}, det_score={det_score}, bbox={bbox}")
            continue

        if status == "known":
            t0 = perf_counter()

            create_embedding_and_face(
                db=db,
                freeze=freeze,
                face=face,
                emb=emb,
                person_id=result["person_id"],
                distance=result["distance"],
                category=result["category"],
                category_score=result["category_score"],
                quality=result["quality"],
                gender=result["gender"],
                confidence=result["confidence"],
                analysis=result["analysis"],
                iteration_id=iteration_id,
                service_name=service_name,
            )

            profiler.add("db_time", perf_counter() - t0)
            created += 1

            logging.info(f"Created known face freeze_id={freeze.id}, person_id={result['person_id']}, reference_embedding_id={result['reference_embedding_id']}, dist={result['distance']}, confidence={result['confidence']}, det_score={det_score}, quality={result['quality']}")

            continue

        if status == "low_quality":
            service_person = get_or_create_service_cluster_person(db=db, category_name=result["category"].name)

            t0 = perf_counter()

            create_embedding_and_face(
                db=db,
                freeze=freeze,
                face=face,
                emb=emb,
                person_id=service_person.id,
                distance=None,
                category=result["category"],
                category_score=result["category_score"],
                quality=result["quality"],
                gender=FaceGender.unknown,
                confidence=None,
                analysis=result["analysis"],
                iteration_id=iteration_id,
                service_name=service_name,
            )

            profiler.add("db_time", perf_counter() - t0)
            created += 1

            logging.info(f"Created low_quality face freeze_id={freeze.id}, person_id={service_person.id}, category={result['category'].name}: {result['category_score']}, quality={result['quality']}, det_score={det_score}")

            continue

        if status == "unknown_identifiable":
            cluster_person = None
            cluster_dist = result["distance"]

            if result["person_id"] is not None:
                cluster_person = db.query(DBPerson).filter(DBPerson.id == result["person_id"]).first()

                if cluster_person is None:
                    raise RuntimeError(f"Unknown FAISS returned person_id={result['person_id']}, but DBPerson was not found.")

                cluster_person.cluster_tag = cluster_person.cluster_tag or cluster_person.name
                cluster_person.cluster_distance = result["distance"]

            else:
                cluster_person = create_unknown_cluster_person(db)

            t0 = perf_counter()

            detected_embedding = create_embedding_and_face(
                db=db,
                freeze=freeze,
                face=face,
                emb=emb,
                person_id=cluster_person.id,
                distance=cluster_dist,
                category=result["category"],
                category_score=result["category_score"],
                quality=result["quality"],
                gender=result["gender"],
                confidence=None,
                analysis=result["analysis"],
                iteration_id=iteration_id,
                service_name=service_name,
            )

            profiler.add("db_time", perf_counter() - t0)

            t0 = perf_counter()
            unknown_index.add(emb, cluster_person.id)
            profiler.add("unknown_faiss_time", perf_counter() - t0)

            created += 1

            logging.info(f"Created unknown identifiable face freeze_id={freeze.id}, person_id={cluster_person.id}, embedding_id={detected_embedding.id}, category={result['category'].name}: {result['category_score']}, cluster_dist={cluster_dist}, quality={result['quality']}, det_score={det_score}, unknown_index_size={unknown_index.size}")

            continue

        if status == "service_category":
            service_person = get_or_create_service_cluster_person(db=db, category_name=result["category"].name)

            t0 = perf_counter()

            create_embedding_and_face(
                db=db,
                freeze=freeze,
                face=face,
                emb=emb,
                person_id=service_person.id,
                distance=None,
                category=result["category"],
                category_score=result["category_score"],
                quality=result["quality"],
                gender=FaceGender.unknown,
                confidence=None,
                analysis=result["analysis"],
                iteration_id=iteration_id,
                service_name=service_name,
            )

            profiler.add("db_time", perf_counter() - t0)
            created += 1

            logging.info(f"Created service-category face freeze_id={freeze.id}, person_id={service_person.id}, category={result['category'].name}: {result['category_score']}, quality={result['quality']}, det_score={det_score}")

    logging.info(f"freeze_id={freeze.id}: created={created}, skipped_low_det_score={skipped_low_det_score}, detected_by_buffalo={len(results)}")

    profiler.freeze_completed(buffalo_faces=len(results), created_faces=created)

    return created
