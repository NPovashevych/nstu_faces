from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_face_category import (
    get_face_category,
    get_face_category_by_code,
    get_face_categories,
    get_active_face_categories,
    get_person_face_categories,
    get_non_person_face_categories,
    create_face_category,
    update_face_category,
    activate_face_category,
    deactivate_face_category,
    delete_face_category,
)
from schemas.schemas_face_category import FaceCategoryCreate, FaceCategoryUpdate, FaceCategoryRead


router = APIRouter(prefix="/face-categories", tags=["face categories"])


@router.get("/", response_model=List[FaceCategoryRead])
def read_face_categories(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_face_categories(db, skip=skip, limit=limit)


@router.get("/active", response_model=List[FaceCategoryRead])
def read_active_face_categories(db: Session = Depends(get_db)):
    return get_active_face_categories(db)


@router.get("/person", response_model=List[FaceCategoryRead])
def read_person_face_categories(db: Session = Depends(get_db)):
    return get_person_face_categories(db)


@router.get("/non-person", response_model=List[FaceCategoryRead])
def read_non_person_face_categories(db: Session = Depends(get_db)):
    return get_non_person_face_categories(db)


@router.get("/code/{code}", response_model=FaceCategoryRead)
def read_face_category_by_code(code: str, db: Session = Depends(get_db)):
    db_category = get_face_category_by_code(db, code)
    if db_category is None:
        raise HTTPException(status_code=404, detail="Face category not found")
    return db_category


@router.get("/{category_id}", response_model=FaceCategoryRead)
def read_face_category(category_id: int, db: Session = Depends(get_db)):
    db_category = get_face_category(db, category_id)
    if db_category is None:
        raise HTTPException(status_code=404, detail="Face category not found")
    return db_category


@router.post("/", response_model=FaceCategoryRead)
def create_new_face_category(category: FaceCategoryCreate, db: Session = Depends(get_db)):
    if get_face_category_by_code(db, category.code):
        raise HTTPException(status_code=400, detail="Face category with this code already exists")
    return create_face_category(db, category)


@router.put("/{category_id}", response_model=FaceCategoryRead)
def update_existing_face_category(category_id: int, category: FaceCategoryUpdate, db: Session = Depends(get_db)):
    if category.code is not None:
        existing = get_face_category_by_code(db, category.code)
        if existing and existing.id != category_id:
            raise HTTPException(status_code=400, detail="Face category with this code already exists")

    db_category = update_face_category(db, category_id, category)
    if db_category is None:
        raise HTTPException(status_code=404, detail="Face category not found")
    return db_category


@router.patch("/{category_id}/activate", response_model=FaceCategoryRead)
def activate_existing_face_category(category_id: int, db: Session = Depends(get_db)):
    db_category = activate_face_category(db, category_id)
    if db_category is None:
        raise HTTPException(status_code=404, detail="Face category not found")
    return db_category


@router.patch("/{category_id}/deactivate", response_model=FaceCategoryRead)
def deactivate_existing_face_category(category_id: int, db: Session = Depends(get_db)):
    db_category = deactivate_face_category(db, category_id)
    if db_category is None:
        raise HTTPException(status_code=404, detail="Face category not found")
    return db_category


@router.delete("/{category_id}")
def delete_existing_face_category(category_id: int, db: Session = Depends(get_db)):
    db_category = delete_face_category(db, category_id)
    if db_category is None:
        raise HTTPException(status_code=404, detail="Face category not found")
    return {"ok": True}
