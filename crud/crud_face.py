from sqlalchemy.orm import Session

from db.models import DBFace
from schemas.schemas_face import FaceCreate, FaceUpdate


def get_face(db: Session, face_id: int):
    return db.query(DBFace).filter(DBFace.id == face_id).first()


def get_faces(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBFace).offset(skip).limit(limit).all()


def get_faces_by_freeze(db: Session, freeze_id: int):
    return db.query(DBFace).filter(DBFace.freeze_id == freeze_id).all()


def get_faces_by_person(db: Session, person_id: int):
    return db.query(DBFace).filter(DBFace.person_id == person_id).all()


def get_faces_by_iteration(db: Session, iteration_id: int):
    return db.query(DBFace).filter(DBFace.iteration_id == iteration_id).all()


def create_face(db: Session, face: FaceCreate):
    db_face = DBFace(
        bbox=face.bbox,
        gender=face.gender,
        confidence=face.confidence,
        embedding_id=face.embedding_id,
        freeze_id=face.freeze_id,
        person_id=face.person_id,
        iteration_id=face.iteration_id,
    )

    db.add(db_face)
    db.commit()
    db.refresh(db_face)
    return db_face


def update_face(db: Session, face_id: int, face: FaceUpdate):
    db_face = get_face(db, face_id)

    if db_face is None:
        return None

    update_data = face.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_face, field, value)

    db.commit()
    db.refresh(db_face)
    return db_face


def delete_face(db: Session, face_id: int):
    db_face = get_face(db, face_id)

    if db_face is None:
        return None

    db.delete(db_face)
    db.commit()
    return db_face
