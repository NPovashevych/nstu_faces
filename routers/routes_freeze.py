from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_freeze import (
    get_freeze,
    get_freezes,
    get_freezes_by_media,
    create_freeze,
    update_freeze,
    delete_freeze,
)
from crud.crud_media import get_media
from schemas.schemas_freeze import FreezeCreate, FreezeUpdate, FreezeRead


router = APIRouter(prefix="/freezes", tags=["freezes"])


@router.get("/", response_model=List[FreezeRead])
def read_freezes(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_freezes(db, skip=skip, limit=limit)


@router.get("/{freeze_id}", response_model=FreezeRead)
def read_freeze(freeze_id: int, db: Session = Depends(get_db)):
    db_freeze = get_freeze(db, freeze_id)

    if db_freeze is None:
        raise HTTPException(status_code=404, detail="Freeze not found")

    return db_freeze


@router.get("/media/{media_id}", response_model=List[FreezeRead])
def read_freezes_by_media(media_id: int, db: Session = Depends(get_db)):
    db_media = get_media(db, media_id)

    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")

    return get_freezes_by_media(db, media_id)


@router.post("/", response_model=FreezeRead)
def create_new_freeze(freeze: FreezeCreate, db: Session = Depends(get_db)):
    db_media = get_media(db, freeze.media_id)

    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")

    return create_freeze(db, freeze)


@router.put("/{freeze_id}", response_model=FreezeRead)
def update_existing_freeze(
    freeze_id: int,
    freeze: FreezeUpdate,
    db: Session = Depends(get_db),
):
    if freeze.media_id is not None:
        db_media = get_media(db, freeze.media_id)

        if db_media is None:
            raise HTTPException(status_code=404, detail="Media not found")

    db_freeze = update_freeze(db, freeze_id, freeze)

    if db_freeze is None:
        raise HTTPException(status_code=404, detail="Freeze not found")

    return db_freeze


@router.delete("/{freeze_id}")
def delete_existing_freeze(freeze_id: int, db: Session = Depends(get_db)):
    db_freeze = delete_freeze(db, freeze_id)

    if db_freeze is None:
        raise HTTPException(status_code=404, detail="Freeze not found")

    return {"ok": True}
