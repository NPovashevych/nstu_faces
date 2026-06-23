from sqlalchemy.orm import Session

from db.models import DBFaceCategory
from schemas.schemas_face_category import (
    FaceCategoryCreate,
    FaceCategoryUpdate,
)


def get_face_category(db: Session, category_id: int):
    return db.query(DBFaceCategory).filter(DBFaceCategory.id == category_id).first()


def get_face_category_by_code(db: Session, code: str):
    return db.query(DBFaceCategory).filter(DBFaceCategory.code == code).first()


def get_face_categories(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBFaceCategory).offset(skip).limit(limit).all()


def get_active_face_categories(db: Session):
    return db.query(DBFaceCategory).filter(DBFaceCategory.is_active.is_(True)).order_by(DBFaceCategory.name).all()


def get_person_face_categories(db: Session):
    return (
        db.query(DBFaceCategory)
        .filter(DBFaceCategory.is_active.is_(True), DBFaceCategory.is_person.is_(True))
        .order_by(DBFaceCategory.name)
        .all()
    )


def get_non_person_face_categories(db: Session):
    return (
        db.query(DBFaceCategory)
        .filter(DBFaceCategory.is_active.is_(True), DBFaceCategory.is_person.is_(False))
        .order_by(DBFaceCategory.name)
        .all()
    )


def create_face_category(db: Session, category: FaceCategoryCreate):
    db_category = DBFaceCategory(
        code=category.code,
        name=category.name,
        description=category.description,
        is_person=category.is_person,
        is_active=category.is_active,
    )

    db.add(db_category)
    db.commit()
    db.refresh(db_category)

    return db_category


def update_face_category(db: Session, category_id: int, category: FaceCategoryUpdate):
    db_category = get_face_category(db, category_id)

    if db_category is None:
        return None

    update_data = category.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_category, field, value)

    db.commit()
    db.refresh(db_category)

    return db_category


def activate_face_category(db: Session, category_id: int):
    db_category = get_face_category(db, category_id)

    if db_category is None:
        return None

    db_category.is_active = True

    db.commit()
    db.refresh(db_category)

    return db_category


def deactivate_face_category(db: Session, category_id: int):
    db_category = get_face_category(db, category_id)

    if db_category is None:
        return None

    db_category.is_active = False

    db.commit()
    db.refresh(db_category)

    return db_category


def delete_face_category(db: Session, category_id: int):
    db_category = get_face_category(db, category_id)

    if db_category is None:
        return None

    db.delete(db_category)
    db.commit()

    return db_category
