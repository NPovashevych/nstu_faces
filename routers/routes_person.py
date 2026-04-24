from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_person import (
    get_person,
    get_persons,
    create_person,
    update_person,
    delete_person,
    get_person_by_qcode,
)
from schemas.schemas_person import PersonsCreate, PersonsUpdate, PersonsRead


router = APIRouter(prefix="/persons", tags=["persons"])


@router.get("/", response_model=List[PersonsRead])
def read_persons(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_persons(db, skip=skip, limit=limit)


@router.get("/{person_id}", response_model=PersonsRead)
def read_person(person_id: int, db: Session = Depends(get_db)):
    db_person = get_person(db, person_id)
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return db_person


@router.post("/", response_model=PersonsRead)
def create_new_person(person: PersonsCreate, db: Session = Depends(get_db)):
    if person.q_code:
        existing = get_person_by_qcode(db, person.q_code)
        if existing:
            raise HTTPException(status_code=400, detail="Person with this q_code already exists")

    # ⚠️ code треба генерувати (поки просто заглушка)
    code = f"person_{person.name.lower().replace(' ', '_')}"

    return create_person(db, person, code)


@router.put("/{person_id}", response_model=PersonsRead)
def update_existing_person(person_id: int, person: PersonsUpdate, db: Session = Depends(get_db)):
    db_person = update_person(db, person_id, person)
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return db_person


@router.delete("/{person_id}")
def delete_existing_person(person_id: int, db: Session = Depends(get_db)):
    db_person = delete_person(db, person_id)
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return {"ok": True}
