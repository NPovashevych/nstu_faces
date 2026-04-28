from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_media import (
    get_media,
    get_media_by_path,
    get_medias,
    create_media,
    update_media,
    delete_media,
)
from crud.crud_user import get_user
from schemas.schemas_media import MediaCreate, MediaUpdate, MediaRead


router = APIRouter(prefix="/media", tags=["media"])


@router.get("/", response_model=List[MediaRead])
def read_medias(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_medias(db, skip=skip, limit=limit)


@router.get("/{media_id}", response_model=MediaRead)
def read_media(media_id: int, db: Session = Depends(get_db)):
    db_media = get_media(db, media_id)

    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")

    return db_media


@router.post("/", response_model=MediaRead)
def create_new_media(
    media: MediaCreate,
    user_id: int,
    db: Session = Depends(get_db),
):
    db_user = get_user(db, user_id)

    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    existing = get_media_by_path(db, media.media_path)

    if existing:
        raise HTTPException(status_code=400, detail="Media path already exists")

    return create_media(db, media, user_id=user_id)


@router.put("/{media_id}", response_model=MediaRead)
def update_existing_media(
    media_id: int,
    media: MediaUpdate,
    db: Session = Depends(get_db),
):
    db_media = update_media(db, media_id, media)

    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")

    return db_media


@router.delete("/{media_id}")
def delete_existing_media(media_id: int, db: Session = Depends(get_db)):
    db_media = delete_media(db, media_id)

    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")

    return {"ok": True}
