from sqlalchemy.orm import Session

from db.models import DBUser
from schemas.schemas_user import UserCreate, UserUpdate
from core.security import hash_password


def get_user(db: Session, user_id: int):
    return db.query(DBUser).filter(DBUser.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(DBUser).filter(DBUser.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBUser).offset(skip).limit(limit).all()


def create_user(db: Session, user: UserCreate):
    db_user = DBUser(
        name=user.name,
        email=user.email,
        password_hash=hash_password(user.password),
        role=user.role,
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user_id: int, user: UserUpdate):
    db_user = get_user(db, user_id)

    if db_user is None:
        return None

    update_data = user.model_dump(exclude_unset=True)

    if "password" in update_data:
        db_user.password_hash = hash_password(update_data.pop("password"))

    for field, value in update_data.items():
        setattr(db_user, field, value)

    db.commit()
    db.refresh(db_user)
    return db_user


def delete_user(db: Session, user_id: int):
    db_user = get_user(db, user_id)

    if db_user is None:
        return None

    db.delete(db_user)
    db.commit()
    return db_user
