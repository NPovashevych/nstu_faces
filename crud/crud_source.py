from sqlalchemy.orm import Session

from db.models import DBSource
from schemas.schemas_source import SourceCreate, SourceUpdate


def get_source(db: Session, source_id: int):
    return db.query(DBSource).filter(DBSource.id == source_id).first()


def get_source_by_code(db: Session, code: str):
    return db.query(DBSource).filter(DBSource.code == code).first()


def get_sources(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBSource).offset(skip).limit(limit).all()


def get_active_sources(db: Session):
    return db.query(DBSource).filter(DBSource.is_active.is_(True)).order_by(DBSource.name).all()


def create_source(db: Session, source: SourceCreate):
    db_source = DBSource(
        code=source.code,
        name=source.name,
        description=source.description,
        is_active=source.is_active,
    )

    db.add(db_source)
    db.commit()
    db.refresh(db_source)

    return db_source


def update_source(db: Session, source_id: int, source: SourceUpdate):
    db_source = get_source(db, source_id)

    if db_source is None:
        return None

    update_data = source.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_source, field, value)

    db.commit()
    db.refresh(db_source)

    return db_source


def deactivate_source(db: Session, source_id: int):
    db_source = get_source(db, source_id)

    if db_source is None:
        return None

    db_source.is_active = False

    db.commit()
    db.refresh(db_source)

    return db_source


def activate_source(db: Session, source_id: int):
    db_source = get_source(db, source_id)

    if db_source is None:
        return None

    db_source.is_active = True

    db.commit()
    db.refresh(db_source)

    return db_source


def delete_source(db: Session, source_id: int):
    db_source = get_source(db, source_id)

    if db_source is None:
        return None

    db.delete(db_source)
    db.commit()

    return db_source
