from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_media import get_media
from crud.crud_media_description import (
    get_media_description,
    get_media_descriptions,
    get_media_descriptions_by_media,
    create_media_description,
    update_media_description,
    delete_media_description,
)
from schemas.schemas_media_description import (
    MediaDescriptionCreate,
    MediaDescriptionUpdate,
    MediaDescriptionRead,
)


router = APIRouter(
    prefix="/media-descriptions",
    tags=["media descriptions"],
)


@router.get("/", response_model=List[MediaDescriptionRead])
def read_media_descriptions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    return get_media_descriptions(db, skip=skip, limit=limit)


@router.get("/{description_id}", response_model=MediaDescriptionRead)
def read_media_description(
    description_id: int,
    db: Session = Depends(get_db),
):
    db_description = get_media_description(db, description_id)

    if db_description is None:
        raise HTTPException(status_code=404, detail="Media description not found")

    return db_description


@router.get("/media/{media_id}", response_model=List[MediaDescriptionRead])
def read_media_descriptions_by_media(
    media_id: int,
    db: Session = Depends(get_db),
):
    db_media = get_media(db, media_id)

    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")

    return get_media_descriptions_by_media(db, media_id)


@router.post("/", response_model=MediaDescriptionRead)
def create_new_media_description(
    description: MediaDescriptionCreate,
    db: Session = Depends(get_db),
):
    db_media = get_media(db, description.media_id)

    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")

    return create_media_description(db, description)


@router.put("/{description_id}", response_model=MediaDescriptionRead)
def update_existing_media_description(
    description_id: int,
    description: MediaDescriptionUpdate,
    db: Session = Depends(get_db),
):
    if description.media_id is not None:
        db_media = get_media(db, description.media_id)

        if db_media is None:
            raise HTTPException(status_code=404, detail="Media not found")

    db_description = update_media_description(db, description_id, description)

    if db_description is None:
        raise HTTPException(status_code=404, detail="Media description not found")

    return db_description


@router.delete("/{description_id}")
def delete_existing_media_description(
    description_id: int,
    db: Session = Depends(get_db),
):
    db_description = delete_media_description(db, description_id)

    if db_description is None:
        raise HTTPException(status_code=404, detail="Media description not found")

    return {"ok": True}
