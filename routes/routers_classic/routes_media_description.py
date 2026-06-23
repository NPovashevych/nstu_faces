from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_media_description import (
    get_media_description,
    get_media_descriptions,
    get_media_descriptions_by_material_id,
    get_media_description_by_source_path,
    get_media_descriptions_by_source_hash,
)
from schemas.schemas_media_description import MediaDescriptionRead


router = APIRouter(prefix="/media-descriptions", tags=["media descriptions"])


@router.get("/", response_model=List[MediaDescriptionRead])
def read_media_descriptions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_media_descriptions(db, skip=skip, limit=limit)


@router.get("/material/{material_id}", response_model=List[MediaDescriptionRead])
def read_media_descriptions_by_material_id(material_id: str, db: Session = Depends(get_db)):
    return get_media_descriptions_by_material_id(db, material_id)


@router.get("/source-hash/{source_hash}", response_model=List[MediaDescriptionRead])
def read_media_descriptions_by_source_hash(source_hash: str, db: Session = Depends(get_db)):
    return get_media_descriptions_by_source_hash(db, source_hash)


@router.get("/source-path", response_model=MediaDescriptionRead)
def read_media_description_by_source_path(source_path: str, db: Session = Depends(get_db)):
    db_description = get_media_description_by_source_path(db, source_path)
    if db_description is None:
        raise HTTPException(status_code=404, detail="Media description not found")
    return db_description


@router.get("/{description_id}", response_model=MediaDescriptionRead)
def read_media_description(description_id: int, db: Session = Depends(get_db)):
    db_description = get_media_description(db, description_id)
    if db_description is None:
        raise HTTPException(status_code=404, detail="Media description not found")
    return db_description
