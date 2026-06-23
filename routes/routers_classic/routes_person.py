from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_person import (
    get_person,
    get_person_by_code,
    get_person_by_qcode,
    get_persons,
    get_persons_by_cluster,
    update_person,
    delete_person,
)
from schemas.schemas_person import PersonsUpdate, PersonsRead


router = APIRouter(prefix="/persons", tags=["persons"])


@router.get("/", response_model=List[PersonsRead])
def read_persons(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_persons(db, skip=skip, limit=limit)


@router.get("/code/{code}", response_model=PersonsRead)
def read_person_by_code(code: str, db: Session = Depends(get_db)):
    db_person = get_person_by_code(db, code)
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return db_person


@router.get("/qcode/{q_code}", response_model=PersonsRead)
def read_person_by_qcode(q_code: str, db: Session = Depends(get_db)):
    db_person = get_person_by_qcode(db, q_code)
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return db_person


@router.get("/cluster/{cluster_id}", response_model=List[PersonsRead])
def read_persons_by_cluster(cluster_id: int, db: Session = Depends(get_db)):
    return get_persons_by_cluster(db, cluster_id)


@router.get("/{person_id}", response_model=PersonsRead)
def read_person(person_id: int, db: Session = Depends(get_db)):
    db_person = get_person(db, person_id)
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return db_person


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
