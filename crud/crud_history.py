# crud/crud_history.py

from sqlalchemy.orm import Session

from db.models import DBHistory
from schemas.schemas_history import HistoryCreate


def get_history(db: Session, history_id: int):
    return db.query(DBHistory).filter(DBHistory.id == history_id).first()


def get_histories(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBHistory).order_by(DBHistory.created_at.desc()).offset(skip).limit(limit).all()


def get_histories_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(DBHistory).filter(DBHistory.user_id == user_id).order_by(DBHistory.created_at.desc()).offset(skip).limit(limit).all()


def create_history(db: Session, history: HistoryCreate):
    db_history = DBHistory(
        action=history.action,
        details=history.details,
        user_id=history.user_id,
    )

    db.add(db_history)
    db.commit()
    db.refresh(db_history)

    return db_history


def log_history(db: Session, user_id: int, action: str, details: dict | None = None):
    db_history = DBHistory(
        action=action,
        details=details,
        user_id=user_id,
    )

    db.add(db_history)
    db.commit()
    db.refresh(db_history)

    return db_history


def delete_history(db: Session, history_id: int):
    db_history = get_history(db, history_id)

    if db_history is None:
        return None

    db.delete(db_history)
    db.commit()

    return db_history
