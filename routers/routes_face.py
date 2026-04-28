from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_face import (
    get_face,
    get_faces,
    get_faces_by_freeze,
    get_faces_by_person,
    get_faces_by_iteration,
    create_face,
    update_face,
    delete_face,
)
from crud.crud_embedding import get_embedding
from crud.crud_freeze import get_freeze
from crud.crud_person import get_person
from crud.crud_iteration import get_iteration
from schemas.schemas_face import FaceCreate, FaceUpdate, FaceRead


router = APIRouter(prefix="/faces", tags=["faces"])


@router.get("/", response_model=List[FaceRead])
def read_faces(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_faces(db, skip=skip, limit=limit)


@router.get("/{face_id}", response_model=FaceRead)
def read_face(face_id: int, db: Session = Depends(get_db)):
    db_face = get_face(db, face_id)

    if db_face is None:
        raise HTTPException(status_code=404, detail="Face not found")

    return db_face


@router.get("/freeze/{freeze_id}", response_model=List[FaceRead])
def read_faces_by_freeze(freeze_id: int, db: Session = Depends(get_db)):
    if get_freeze(db, freeze_id) is None:
        raise HTTPException(status_code=404, detail="Freeze not found")

    return get_faces_by_freeze(db, freeze_id)


@router.get("/person/{person_id}", response_model=List[FaceRead])
def read_faces_by_person(person_id: int, db: Session = Depends(get_db)):
    if get_person(db, person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")

    return get_faces_by_person(db, person_id)


@router.get("/iteration/{iteration_id}", response_model=List[FaceRead])
def read_faces_by_iteration(iteration_id: int, db: Session = Depends(get_db)):
    if get_iteration(db, iteration_id) is None:
        raise HTTPException(status_code=404, detail="Iteration not found")

    return get_faces_by_iteration(db, iteration_id)


@router.post("/", response_model=FaceRead)
def create_new_face(face: FaceCreate, db: Session = Depends(get_db)):
    if get_embedding(db, face.embedding_id) is None:
        raise HTTPException(status_code=404, detail="Embedding not found")

    if get_freeze(db, face.freeze_id) is None:
        raise HTTPException(status_code=404, detail="Freeze not found")

    if get_person(db, face.person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")

    if get_iteration(db, face.iteration_id) is None:
        raise HTTPException(status_code=404, detail="Iteration not found")

    return create_face(db, face)


@router.put("/{face_id}", response_model=FaceRead)
def update_existing_face(
    face_id: int,
    face: FaceUpdate,
    db: Session = Depends(get_db),
):
    if face.embedding_id is not None and get_embedding(db, face.embedding_id) is None:
        raise HTTPException(status_code=404, detail="Embedding not found")

    if face.freeze_id is not None and get_freeze(db, face.freeze_id) is None:
        raise HTTPException(status_code=404, detail="Freeze not found")

    if face.person_id is not None and get_person(db, face.person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")

    if face.iteration_id is not None and get_iteration(db, face.iteration_id) is None:
        raise HTTPException(status_code=404, detail="Iteration not found")

    db_face = update_face(db, face_id, face)

    if db_face is None:
        raise HTTPException(status_code=404, detail="Face not found")

    return db_face


@router.delete("/{face_id}")
def delete_existing_face(face_id: int, db: Session = Depends(get_db)):
    db_face = delete_face(db, face_id)

    if db_face is None:
        raise HTTPException(status_code=404, detail="Face not found")

    return {"ok": True}
