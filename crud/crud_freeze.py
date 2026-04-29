from sqlalchemy.orm import Session

from db.models import DBFreeze
from schemas.schemas_freeze import FreezeCreate, FreezeUpdate


def get_freeze(db: Session, freeze_id: int):
    return db.query(DBFreeze).filter(DBFreeze.id == freeze_id).first()


def get_freezes(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBFreeze).offset(skip).limit(limit).all()


def get_freezes_by_media(db: Session, media_id: int):
    return db.query(DBFreeze).filter(DBFreeze.media_id == media_id).all()


def create_freeze(db: Session, freeze: FreezeCreate):
    existing = (
        db.query(DBFreeze)
        .filter(DBFreeze.freeze_path == freeze.freeze_path)
        .first()
    )

    if existing:
        return existing

    db_freeze = DBFreeze(
        time_in=freeze.time_in,
        time_out=freeze.time_out,
        media_id=freeze.media_id,
        freeze_path=freeze.freeze_path,
    )

    db.add(db_freeze)
    db.commit()
    db.refresh(db_freeze)
    return db_freeze


def update_freeze(db: Session, freeze_id: int, freeze: FreezeUpdate):
    db_freeze = get_freeze(db, freeze_id)

    if db_freeze is None:
        return None

    update_data = freeze.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_freeze, field, value)

    db.commit()
    db.refresh(db_freeze)
    return db_freeze


def delete_freeze(db: Session, freeze_id: int):
    db_freeze = get_freeze(db, freeze_id)

    if db_freeze is None:
        return None

    db.delete(db_freeze)
    db.commit()
    return db_freeze
