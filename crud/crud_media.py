from sqlalchemy.orm import Session

from db.models import DBMedia
from schemas.schemas_media import MediaCreate, MediaUpdate


def get_media(db: Session, media_id: int):
    return db.query(DBMedia).filter(DBMedia.id == media_id).first()


def get_media_by_path(db: Session, media_path: str):
    return db.query(DBMedia).filter(DBMedia.media_path == media_path).first()


def get_medias(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBMedia).offset(skip).limit(limit).all()


def create_media(db: Session, media: MediaCreate, user_id: int):
    db_media = DBMedia(
        source=media.source,
        media_type=media.media_type,
        media_path=media.media_path,
        user_id=user_id,
    )

    db.add(db_media)
    db.commit()
    db.refresh(db_media)
    return db_media


def update_media(db: Session, media_id: int, media: MediaUpdate):
    db_media = get_media(db, media_id)

    if db_media is None:
        return None

    update_data = media.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_media, field, value)

    db.commit()
    db.refresh(db_media)
    return db_media


def delete_media(db: Session, media_id: int):
    db_media = get_media(db, media_id)

    if db_media is None:
        return None

    db.delete(db_media)
    db.commit()
    return db_media
