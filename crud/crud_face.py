from sqlalchemy.orm import Session

from db.models import DBFace
from schemas.schemas_face import FaceCreate, FaceUpdate


def create_face(db: Session, face: FaceCreate):
    db_face = DBFace(**face.dict())

    db.add(db_face)
    db.commit()
    db.refresh(db_face)

    return db_face


def get_face(db: Session, face_id: int):
    return (
        db.query(DBFace)
        .filter(DBFace.id == face_id)
        .first()
    )


def get_faces(db: Session):
    return (
        db.query(DBFace)
        .order_by(DBFace.id)
        .all()
    )


def get_faces_by_freeze(db: Session, freeze_id: int):
    return (
        db.query(DBFace)
        .filter(DBFace.freeze_id == freeze_id)
        .all()
    )


def get_faces_by_person(db: Session, person_id: int):
    return (
        db.query(DBFace)
        .filter(DBFace.person_id == person_id)
        .all()
    )


def update_face(db: Session, face_id: int, payload: FaceUpdate):
    db_face = get_face(db, face_id)

    if not db_face:
        return None

    update_data = payload.dict(exclude_unset=True)

    for key, value in update_data.items():
        setattr(db_face, key, value)

    db.commit()
    db.refresh(db_face)

    return db_face


def delete_face(db: Session, face_id: int):
    db_face = get_face(db, face_id)

    if not db_face:
        return False

    db.delete(db_face)
    db.commit()

    return True
