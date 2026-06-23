from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_source import (
    get_source,
    get_source_by_code,
    get_sources,
    get_active_sources,
    create_source,
    update_source,
    activate_source,
    deactivate_source,
    delete_source,
)
from schemas.schemas_source import SourceCreate, SourceUpdate, SourceRead


router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("/", response_model=List[SourceRead])
def read_sources(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_sources(db, skip=skip, limit=limit)


@router.get("/active", response_model=List[SourceRead])
def read_active_sources(db: Session = Depends(get_db)):
    return get_active_sources(db)


@router.get("/code/{code}", response_model=SourceRead)
def read_source_by_code(code: str, db: Session = Depends(get_db)):
    db_source = get_source_by_code(db, code)
    if db_source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return db_source


@router.get("/{source_id}", response_model=SourceRead)
def read_source(source_id: int, db: Session = Depends(get_db)):
    db_source = get_source(db, source_id)
    if db_source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return db_source


@router.post("/", response_model=SourceRead)
def create_new_source(source: SourceCreate, db: Session = Depends(get_db)):
    if get_source_by_code(db, source.code):
        raise HTTPException(status_code=400, detail="Source with this code already exists")
    return create_source(db, source)


@router.put("/{source_id}", response_model=SourceRead)
def update_existing_source(source_id: int, source: SourceUpdate, db: Session = Depends(get_db)):
    if source.code is not None:
        existing = get_source_by_code(db, source.code)
        if existing and existing.id != source_id:
            raise HTTPException(status_code=400, detail="Source with this code already exists")

    db_source = update_source(db, source_id, source)
    if db_source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return db_source


@router.patch("/{source_id}/activate", response_model=SourceRead)
def activate_existing_source(source_id: int, db: Session = Depends(get_db)):
    db_source = activate_source(db, source_id)
    if db_source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return db_source


@router.patch("/{source_id}/deactivate", response_model=SourceRead)
def deactivate_existing_source(source_id: int, db: Session = Depends(get_db)):
    db_source = deactivate_source(db, source_id)
    if db_source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return db_source


@router.delete("/{source_id}")
def delete_existing_source(source_id: int, db: Session = Depends(get_db)):
    db_source = delete_source(db, source_id)
    if db_source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"ok": True}
