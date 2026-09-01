from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import func

from db.session import get_db
from db.models import DBUser
from crud.crud_history import log_history

from routes.routes_services.history_action import ACTION_AUTH_LOGIN

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr


def normalize_email(email: str) -> str:
    return email.strip().lower()


def normalize_user(user: DBUser):
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role.value if user.role else None,
    }


@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = normalize_email(payload.email)

    user = (
        db.query(DBUser)
        .filter(func.lower(DBUser.email) == email.strip().lower())
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="User with this email is not allowed",
        )

    log_history(
        db=db,
        user_id=user.id,
        action=ACTION_AUTH_LOGIN,
        details={
            "email": email,
            "result": "success",
        },
    )

    return {
        "status": "ok",
        "user": normalize_user(user),
    }
