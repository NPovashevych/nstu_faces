from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.session import get_db
from crud.crud_freeze import get_freeze, get_freeze_by_path, get_freezes, get_freezes_by_media
from crud.crud_media import get_media
from schemas.schemas_freeze import FreezeRead


router = APIRouter(prefix="/freezes", tags=["freezes"])


@router.get("/", response_model=List[FreezeRead])
def read_freezes(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_freezes(db, skip=skip, limit=limit)


@router.get("/media/{media_id}", response_model=List[FreezeRead])
def read_freezes_by_media(media_id: int, db: Session = Depends(get_db)):
    if get_media(db, media_id) is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return get_freezes_by_media(db, media_id)


@router.get("/path", response_model=FreezeRead)
def read_freeze_by_path(freeze_path: str, db: Session = Depends(get_db)):
    db_freeze = get_freeze_by_path(db, freeze_path)
    if db_freeze is None:
        raise HTTPException(status_code=404, detail="Freeze not found")
    return db_freeze


@router.get("/{freeze_id}", response_model=FreezeRead)
def read_freeze(freeze_id: int, db: Session = Depends(get_db)):
    db_freeze = get_freeze(db, freeze_id)
    if db_freeze is None:
        raise HTTPException(status_code=404, detail="Freeze not found")
    return db_freeze
