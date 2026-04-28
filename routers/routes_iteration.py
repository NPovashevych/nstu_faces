from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_iteration import (
    get_iteration,
    get_iterations,
    get_iterations_by_media,
    create_iteration,
    update_iteration,
)
from crud.crud_user import get_user
from crud.crud_media import get_media

from schemas.schemas_iteration import (
    IterationCreate,
    IterationUpdate,
    IterationRead,
)


router = APIRouter(prefix="/iterations", tags=["iterations"])


@router.get("/", response_model=List[IterationRead])
def read_iterations(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_iterations(db, skip=skip, limit=limit)


@router.get("/{iteration_id}", response_model=IterationRead)
def read_iteration(iteration_id: int, db: Session = Depends(get_db)):
    db_iteration = get_iteration(db, iteration_id)

    if db_iteration is None:
        raise HTTPException(status_code=404, detail="Iteration not found")

    return db_iteration


@router.get("/media/{media_id}", response_model=List[IterationRead])
def read_iterations_by_media(media_id: int, db: Session = Depends(get_db)):
    db_media = get_media(db, media_id)

    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")

    return get_iterations_by_media(db, media_id)


@router.post("/", response_model=IterationRead)
def create_new_iteration(
    iteration: IterationCreate,
    db: Session = Depends(get_db),
):
    db_user = get_user(db, iteration.user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    db_media = get_media(db, iteration.media_id)
    if db_media is None:
        raise HTTPException(status_code=404, detail="Media not found")

    return create_iteration(db, iteration)


@router.put("/{iteration_id}", response_model=IterationRead)
def update_existing_iteration(
    iteration_id: int,
    iteration: IterationUpdate,
    db: Session = Depends(get_db),
):
    if iteration.user_id is not None:
        if get_user(db, iteration.user_id) is None:
            raise HTTPException(status_code=404, detail="User not found")

    if iteration.media_id is not None:
        if get_media(db, iteration.media_id) is None:
            raise HTTPException(status_code=404, detail="Media not found")

    db_iteration = update_iteration(db, iteration_id, iteration)

    if db_iteration is None:
        raise HTTPException(status_code=404, detail="Iteration not found")

    return db_iteration
