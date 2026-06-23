from sqlalchemy.orm import Session

from db.models import DBMedia
from schemas.schemas_media import MediaCreate, MediaUpdate


def get_media(db: Session, media_id: int):
    return db.query(DBMedia).filter(DBMedia.id == media_id).first()


def get_media_by_material_id(db: Session, material_id: str):
    return db.query(DBMedia).filter(DBMedia.material_id == material_id).first()


def get_media_by_mxf_path(db: Session, mxf_path: str):
    return db.query(DBMedia).filter(DBMedia.mxf_path == mxf_path).first()


def get_media_by_mp4_path(db: Session, mp4_path: str):
    return db.query(DBMedia).filter(DBMedia.mp4_path == mp4_path).first()


def get_medias(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBMedia).order_by(DBMedia.uploaded_at.desc()).offset(skip).limit(limit).all()


def get_medias_by_source(db: Session, source_id: int):
    return db.query(DBMedia).filter(DBMedia.source_id == source_id).order_by(DBMedia.uploaded_at.desc()).all()


def get_medias_by_user(db: Session, user_id: int):
    return db.query(DBMedia).filter(DBMedia.user_id == user_id).order_by(DBMedia.uploaded_at.desc()).all()


def create_media(db: Session, media: MediaCreate):
    db_media = DBMedia(
        material_id=media.material_id,
        media_type=media.media_type,
        mxf_path=media.mxf_path,
        mp4_path=media.mp4_path,
        duration=media.duration,
        recorded_at=media.recorded_at,
        source_id=media.source_id,
        user_id=media.user_id,
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
