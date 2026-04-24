from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_user import (
    get_user,
    get_users,
    create_user,
    update_user,
    delete_user,
    get_user_by_email,
)
from schemas.schemas_user import UserCreate, UserUpdate, UserRead


router = APIRouter(prefix="/users", tags=["users"])


@router.get("/", response_model=List[UserRead])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_users(db, skip=skip, limit=limit)


@router.get("/{user_id}", response_model=UserRead)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = get_user(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.post("/", response_model=UserRead)
def create_new_user(user: UserCreate, db: Session = Depends(get_db)):
    existing = get_user_by_email(db, user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    return create_user(db, user)


@router.put("/{user_id}", response_model=UserRead)
def update_existing_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    db_user = update_user(db, user_id, user)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.delete("/{user_id}")
def delete_existing_user(user_id: int, db: Session = Depends(get_db)):
    db_user = delete_user(db, user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}
