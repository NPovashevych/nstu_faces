from sqlalchemy.orm import Session

from db.models import DBHistory
from schemas.schemas_history import HistoryCreate, HistoryUpdate


def get_history(db: Session, history_id: int):
    return db.query(DBHistory).filter(DBHistory.id == history_id).first()


def get_histories(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBHistory).offset(skip).limit(limit).all()


def get_histories_by_user(db: Session, user_id: int):
    return db.query(DBHistory).filter(DBHistory.user_id == user_id).all()


def create_history(db: Session, history: HistoryCreate):
    db_history = DBHistory(
        action=history.action,
        user_id=history.user_id,
    )

    db.add(db_history)
    db.commit()
    db.refresh(db_history)
    return db_history


def update_history(db: Session, history_id: int, history: HistoryUpdate):
    db_history = get_history(db, history_id)

    if db_history is None:
        return None

    update_data = history.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_history, field, value)

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
