from sqlalchemy.orm import Session

from db.models import DBIteration
from schemas.schemas_iteration import IterationCreate, IterationUpdate


def get_iteration(db: Session, iteration_id: int):
    return db.query(DBIteration).filter(DBIteration.id == iteration_id).first()


def get_iterations(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBIteration).offset(skip).limit(limit).all()


def get_iterations_by_media(db: Session, media_id: int):
    return db.query(DBIteration).filter(DBIteration.media_id == media_id).all()


def get_iterations_by_user(db: Session, user_id: int):
    return db.query(DBIteration).filter(DBIteration.user_id == user_id).all()


def create_iteration(db: Session, iteration: IterationCreate):
    db_iteration = DBIteration(
        status=iteration.status,
        params=iteration.params,
        error_message=iteration.error_message,
        user_id=iteration.user_id,
        media_id=iteration.media_id,
    )

    db.add(db_iteration)
    db.commit()
    db.refresh(db_iteration)
    return db_iteration


def update_iteration(db: Session, iteration_id: int, iteration: IterationUpdate):
    db_iteration = get_iteration(db, iteration_id)

    if db_iteration is None:
        return None

    update_data = iteration.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_iteration, field, value)

    db.commit()
    db.refresh(db_iteration)
    return db_iteration


def delete_iteration(db: Session, iteration_id: int):
    db_iteration = get_iteration(db, iteration_id)

    if db_iteration is None:
        return None

    db.delete(db_iteration)
    db.commit()
    return db_iteration
