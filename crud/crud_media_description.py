from sqlalchemy.orm import Session

from db.models import DBMediaDescription
from schemas.schemas_media_description import (
    MediaDescriptionCreate,
    MediaDescriptionUpdate,
)


def get_media_description(db: Session, description_id: int):
    return (
        db.query(DBMediaDescription)
        .filter(DBMediaDescription.id == description_id)
        .first()
    )


def get_media_descriptions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBMediaDescription).offset(skip).limit(limit).all()


def get_media_descriptions_by_media(db: Session, media_id: int):
    return (
        db.query(DBMediaDescription)
        .filter(DBMediaDescription.media_id == media_id)
        .all()
    )


def create_media_description(db: Session, description: MediaDescriptionCreate):
    db_description = DBMediaDescription(
        media_id=description.media_id,
        section=description.section,
        description=description.description,
        date=description.date,
        duration=description.duration,
        journalist=description.journalist,
    )

    db.add(db_description)
    db.commit()
    db.refresh(db_description)
    return db_description


def update_media_description(
    db: Session,
    description_id: int,
    description: MediaDescriptionUpdate,
):
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
