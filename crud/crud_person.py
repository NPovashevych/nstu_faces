from sqlalchemy.orm import Session

from db.models import DBPerson
from schemas.schemas_person import PersonsCreate, PersonsUpdate


def get_person(db: Session, person_id: int):
    return db.query(DBPerson).filter(DBPerson.id == person_id).first()


def get_person_by_code(db: Session, code: str):
    return db.query(DBPerson).filter(DBPerson.code == code).first()


def get_person_by_qcode(db: Session, q_code: str):
    return db.query(DBPerson).filter(DBPerson.q_code == q_code).first()


def get_persons(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBPerson).offset(skip).limit(limit).all()


def create_person(db: Session, person: PersonsCreate, code: str):
    db_person = DBPerson(
        name=person.name,
        q_code=person.q_code,
        link=person.link,
        status=person.status,
        code=code,  # генерується окремо
    )

    db.add(db_person)
    db.commit()
    db.refresh(db_person)
    return db_person


def update_person(db: Session, person_id: int, person: PersonsUpdate):
    db_person = get_person(db, person_id)

    if db_person is None:
        return None

    update_data = person.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_person, field, value)

    db.commit()
    db.refresh(db_person)
    return db_person


def delete_person(db: Session, person_id: int):
    db_person = get_person(db, person_id)

    if db_person is None:
        return None

    db.delete(db_person)
    db.commit()
    return db_person
