from sqlalchemy.orm import Session

from db.models import DBMediaDescription
from schemas.schemas_media_description import (
    MediaDescriptionCreate,
    MediaDescriptionUpdate,
)


def get_media_description(db: Session, description_id: int):
    return db.query(DBMediaDescription).filter(DBMediaDescription.id == description_id).first()


def get_media_descriptions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBMediaDescription).order_by(DBMediaDescription.id).offset(skip).limit(limit).all()


def get_media_descriptions_by_material_id(db: Session, material_id: str):
    return db.query(DBMediaDescription).filter(DBMediaDescription.material_id == material_id).all()


def get_media_description_by_source_path(db: Session, source_path: str):
    return db.query(DBMediaDescription).filter(DBMediaDescription.source_path == source_path).first()


def get_media_descriptions_by_source_hash(db: Session, source_hash: str):
    return db.query(DBMediaDescription).filter(DBMediaDescription.source_hash == source_hash).all()


def create_media_description(db: Session, description: MediaDescriptionCreate):
    db_description = DBMediaDescription(
        material_id=description.material_id,
        section=description.section,
        shooting_date=description.shooting_date,
        journalist=description.journalist,
        operators=description.operators,
        description=description.description,
        another_info=description.another_info,
        source_path=description.source_path,
        source_hash=description.source_hash,
    )

    db.add(db_description)
    db.commit()
    db.refresh(db_description)

    return db_description


def update_media_description(db: Session, description_id: int, description: MediaDescriptionUpdate):
    db_description = get_media_description(db, description_id)

    if db_description is None:
        return None

    update_data = description.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_description, field, value)

    db.commit()
    db.refresh(db_description)

    return db_description


def delete_media_description(db: Session, description_id: int):
    db_description = get_media_description(db, description_id)

    if db_description is None:
        return None

    db.delete(db_description)
    db.commit()

    return db_description
