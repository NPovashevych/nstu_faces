from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_media import (
    get_media,
    get_media_by_material_id,
    get_media_by_mxf_path,
    get_media_by_mp4_path,
    get_medias,
    get_medias_by_source,
    get_medias_by_user,
    create_media,
    update_media,
    delete_media,
)
from crud.crud_user import get_user
from crud.crud_source import get_source
from schemas.schemas_media import MediaCreate, MediaUpdate, MediaRead, MediaReadWithRelations


router = APIRouter(prefix="/media", tags=["media"])


@router.get("/", response_model=List[MediaRead])
def read_medias(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_medias(db, skip=skip, limit=limit)


@router.get("/material/{material_id}", response_model=MediaRead)
def read_media_by_material_id(material_id: str, db: Session = Depends(get_db)):
    db_media = get_media_by_material_id(db, material_id)
    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return db_media


@router.get("/mxf", response_model=MediaRead)
def read_media_by_mxf_path(mxf_path: str, db: Session = Depends(get_db)):
    db_media = get_media_by_mxf_path(db, mxf_path)
    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return db_media


@router.get("/mp4", response_model=MediaRead)
def read_media_by_mp4_path(mp4_path: str, db: Session = Depends(get_db)):
    db_media = get_media_by_mp4_path(db, mp4_path)
    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return db_media


@router.get("/source/{source_id}", response_model=List[MediaRead])
def read_medias_by_source(source_id: int, db: Session = Depends(get_db)):
    if get_source(db, source_id) is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return get_medias_by_source(db, source_id)


@router.get("/user/{user_id}", response_model=List[MediaRead])
def read_medias_by_user(user_id: int, db: Session = Depends(get_db)):
    if get_user(db, user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    return get_medias_by_user(db, user_id)


@router.get("/{media_id}/full", response_model=MediaReadWithRelations)
def read_media_full(media_id: int, db: Session = Depends(get_db)):
    db_media = get_media(db, media_id)
    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return db_media


@router.get("/{media_id}", response_model=MediaRead)
def read_media(media_id: int, db: Session = Depends(get_db)):
    db_media = get_media(db, media_id)
    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return db_media


@router.post("/", response_model=MediaRead)
def create_new_media(media: MediaCreate, db: Session = Depends(get_db)):
    if get_user(db, media.user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    if get_source(db, media.source_id) is None:
        raise HTTPException(status_code=404, detail="Source not found")

    if not media.mxf_path and not media.mp4_path:
        raise HTTPException(status_code=400, detail="Either mxf_path or mp4_path is required")

    if media.material_id and get_media_by_material_id(db, media.material_id):
        raise HTTPException(status_code=400, detail="Material ID already exists")

    if media.mxf_path and get_media_by_mxf_path(db, media.mxf_path):
        raise HTTPException(status_code=400, detail="MXF path already exists")

    if media.mp4_path and get_media_by_mp4_path(db, media.mp4_path):
        raise HTTPException(status_code=400, detail="MP4 path already exists")

    return create_media(db, media)


@router.put("/{media_id}", response_model=MediaRead)
def update_existing_media(media_id: int, media: MediaUpdate, db: Session = Depends(get_db)):
    db_media = get_media(db, media_id)
    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")

    if media.source_id is not None and get_source(db, media.source_id) is None:
        raise HTTPException(status_code=404, detail="Source not found")

    if media.material_id:
        existing = get_media_by_material_id(db, media.material_id)
        if existing and existing.id != media_id:
            raise HTTPException(status_code=400, detail="Material ID already exists")

    if media.mxf_path:
        existing = get_media_by_mxf_path(db, media.mxf_path)
        if existing and existing.id != media_id:
            raise HTTPException(status_code=400, detail="MXF path already exists")

    if media.mp4_path:
        existing = get_media_by_mp4_path(db, media.mp4_path)
        if existing and existing.id != media_id:
            raise HTTPException(status_code=400, detail="MP4 path already exists")

    return update_media(db, media_id, media)


@router.delete("/{media_id}")
def delete_existing_media(media_id: int, db: Session = Depends(get_db)):
    db_media = delete_media(db, media_id)
    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return {"ok": True}
