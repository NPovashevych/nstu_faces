from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_history import (
    get_history,
    get_histories,
    get_histories_by_user,
    create_history,
    update_history,
    delete_history,
)
from crud.crud_user import get_user
from schemas.schemas_history import HistoryCreate, HistoryUpdate, HistoryRead


router = APIRouter(prefix="/history", tags=["history"])


@router.get("/", response_model=List[HistoryRead])
def read_histories(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_histories(db, skip=skip, limit=limit)


@router.get("/{history_id}", response_model=HistoryRead)
def read_history(history_id: int, db: Session = Depends(get_db)):
    db_history = get_history(db, history_id)

    if db_history is None:
        raise HTTPException(status_code=404, detail="History not found")

    return db_history


@router.get("/user/{user_id}", response_model=List[HistoryRead])
def read_histories_by_user(user_id: int, db: Session = Depends(get_db)):
    if get_user(db, user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    return get_histories_by_user(db, user_id)


@router.post("/", response_model=HistoryRead)
def create_new_history(history: HistoryCreate, db: Session = Depends(get_db)):
    if get_user(db, history.user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    return create_history(db, history)


@router.put("/{history_id}", response_model=HistoryRead)
def update_existing_history(
    history_id: int,
    history: HistoryUpdate,
    db: Session = Depends(get_db),
):
    if history.user_id is not None and get_user(db, history.user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    db_history = update_history(db, history_id, history)

    if db_history is None:
        raise HTTPException(status_code=404, detail="History not found")

    return db_history


@router.delete("/{history_id}")
def delete_existing_history(history_id: int, db: Session = Depends(get_db)):
    db_history = delete_history(db, history_id)

    if db_history is None:
        raise HTTPException(status_code=404, detail="History not found")

    return {"ok": True}
