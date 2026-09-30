from fastapi import HTTPException
from sqlalchemy.orm import Session

from db.models import DBUser


def get_user(db: Session, user_id: int) -> DBUser:
    user = db.query(DBUser).filter(DBUser.id == user_id).first()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return user
