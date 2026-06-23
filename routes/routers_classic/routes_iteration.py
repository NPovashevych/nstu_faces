from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_iteration import (
    get_iteration,
    get_iterations,
    get_iterations_by_media,
    get_iterations_by_user,
    create_iteration,
    update_iteration,
)
from crud.crud_user import get_user
from crud.crud_media import get_media
from schemas.schemas_iteration import IterationCreate, IterationUpdate, IterationRead, IterationReadWithMediaUser


router = APIRouter(prefix="/iterations", tags=["iterations"])


@router.get("/", response_model=List[IterationRead])
def read_iterations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_iterations(db, skip=skip, limit=limit)


@router.get("/media/{media_id}", response_model=List[IterationRead])
def read_iterations_by_media(media_id: int, db: Session = Depends(get_db)):
    if get_media(db, media_id) is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return get_iterations_by_media(db, media_id)


@router.get("/user/{user_id}", response_model=List[IterationRead])
def read_iterations_by_user(user_id: int, db: Session = Depends(get_db)):
    if get_user(db, user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    return get_iterations_by_user(db, user_id)


@router.get("/{iteration_id}/full", response_model=IterationReadWithMediaUser)
def read_iteration_full(iteration_id: int, db: Session = Depends(get_db)):
    db_iteration = get_iteration(db, iteration_id)
    if db_iteration is None:
        raise HTTPException(status_code=404, detail="Iteration not found")
    return db_iteration


@router.get("/{iteration_id}", response_model=IterationRead)
def read_iteration(iteration_id: int, db: Session = Depends(get_db)):
    db_iteration = get_iteration(db, iteration_id)
    if db_iteration is None:
        raise HTTPException(status_code=404, detail="Iteration not found")
    return db_iteration


@router.post("/", response_model=IterationRead)
def create_new_iteration(iteration: IterationCreate, db: Session = Depends(get_db)):
    if get_user(db, iteration.user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    if get_media(db, iteration.media_id) is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return create_iteration(db, iteration)


@router.put("/{iteration_id}", response_model=IterationRead)
def update_existing_iteration(iteration_id: int, iteration: IterationUpdate, db: Session = Depends(get_db)):
    db_iteration = update_iteration(db, iteration_id, iteration)
    if db_iteration is None:
        raise HTTPException(status_code=404, detail="Iteration not found")
    return db_iteration
