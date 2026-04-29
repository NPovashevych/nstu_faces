import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sqlalchemy.orm import Session

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

from services.face_quality import pass_quality, is_good_face


Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/detect_faces_from_freezes.log", encoding="utf-8"),
    ],
)


DIST_TOLERANCE = 0.45
STEP_TOLERANCE = 0.03
UNKNOWN_TOLERANCE = 0.40

USER_ID = 1


def load_model():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def cosine_distance(a, b):
    return 1 - float(np.dot(a, b))


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


def get_next_unknown_cluster_number(db: Session) -> int:
    persons = (
        db.query(DBPerson)
        .filter(DBPerson.name.like("unknown_cluster_%"))
        .all()
    )

    max_num = 0

    for person in persons:
        try:
            num = int(person.name.replace("unknown_cluster_", ""))
            max_num = max(max_num, num)
        except ValueError:
            continue

    return max_num + 1


def create_unknown_cluster_person(db: Session):
    next_num = get_next_unknown_cluster_number(db)
    cluster_tag = f"unknown_cluster_{next_num:06d}"

    person_create = PersonsCreate(
        name=cluster_tag,
        q_code=None,
        link=None,
        status=PersonStatus.unknown,
    )

    db_person = create_person(db, person_create, code=cluster_tag)

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

    unknowns = []

    for row in rows:
        unknowns.append(
            {
                "person_id": row.person_id,
                "vector": normalize(np.array(row.vector, dtype=np.float32)),
            }
        )

    return unknowns


def find_or_create_unknown_person(db: Session, embedding):
    unknown_embeddings = load_unknown_cluster_embeddings(db)

    best_person_id = None
    best_dist = 1.0

    for item in unknown_embeddings:
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


def create_detected_embedding(db: Session, person_id: int, embedding, freeze: DBFreeze, face, distance):
    bbox = face.bbox.astype(float).tolist()

    embedding_create = EmbeddingCreate(
        embedding_type=EmbeddingType.detected_face,
        source={
            "freeze_id": freeze.id,
            "freeze_path": freeze.freeze_path,
            "bbox": bbox,
            "distance": distance,
            "created_by": "detect_faces_from_freezes.py",
        },
        vector=embedding.tolist(),
        person_id=person_id,
    )

    return create_embedding(db, embedding_create)


def process_freeze(
    db: Session,
    model,
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

    faces = model.get(img)

    created = 0

    for face in faces:
        quality = is_good_face(img, face)

        if quality < 0.15:
            logging.info(
                f"Skip ghost face freeze_id={freeze.id}, quality={quality:.2f}, bbox={face.bbox.astype(int).tolist()}"
            )
            continue

        emb = normalize(face.embedding)

        best_ref, best_dist = find_best_known_match(emb, reference_embeddings)
        confidence = get_confidence(best_dist)

        if best_ref is not None and confidence != -1:
            person_id = best_ref["person_id"]
            distance_for_source = round(best_dist, 4)
            face_confidence = confidence
        else:
            unknown_person, unknown_dist = find_or_create_unknown_person(db, emb)
            person_id = unknown_person.id
            distance_for_source = round(unknown_dist, 4) if unknown_dist is not None else None
            face_confidence = None

        detected_embedding = create_detected_embedding(
            db=db,
            person_id=person_id,
            embedding=emb,
            freeze=freeze,
            face=face,
            distance=distance_for_source,
        )

        face_create = FaceCreate(
            bbox=face.bbox.astype(float).tolist(),
            gender=map_gender(face),
            quality=round(quality, 4),
            confidence=face_confidence,
            embedding_id=detected_embedding.id,
            freeze_id=freeze.id,
            person_id=person_id,
            iteration_id=iteration_id,
        )

        create_face(db, face_create)
        created += 1

    return created


def process_media(db: Session, model, media: DBMedia, reference_embeddings):
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
                "service": "detect_faces_from_freezes.py",
                "quality_filter": True,
                "dist_tolerance": DIST_TOLERANCE,
                "unknown_tolerance": UNKNOWN_TOLERANCE,
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
                model=model,
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
        model = load_model()
        reference_embeddings = load_reference_embeddings(db)

        medias = db.query(DBMedia).order_by(DBMedia.id).all()

        total_faces = 0

        for media in medias:
            total_faces += process_media(db, model, media, reference_embeddings)

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
