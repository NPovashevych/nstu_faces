from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.session import get_db
from crud.crud_embedding import (
    get_embedding,
    get_embeddings,
    get_embeddings_by_person,
    create_embedding,
    update_embedding,
    delete_embedding,
)
from crud.crud_person import get_person
from schemas.schemas_embedding import (
    EmbeddingCreate,
    EmbeddingUpdate,
    EmbeddingRead,
)


router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.get("/", response_model=List[EmbeddingRead])
def read_embeddings(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_embeddings(db, skip=skip, limit=limit)


@router.get("/{embedding_id}", response_model=EmbeddingRead)
def read_embedding(embedding_id: int, db: Session = Depends(get_db)):
    db_embedding = get_embedding(db, embedding_id)
    if db_embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return db_embedding


@router.get("/person/{person_id}", response_model=List[EmbeddingRead])
def read_embeddings_by_person(person_id: int, db: Session = Depends(get_db)):
    db_person = get_person(db, person_id)
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    return get_embeddings_by_person(db, person_id)


@router.post("/", response_model=EmbeddingRead)
def create_new_embedding(embedding: EmbeddingCreate, db: Session = Depends(get_db)):
    db_person = get_person(db, embedding.person_id)
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    return create_embedding(db, embedding)


@router.put("/{embedding_id}", response_model=EmbeddingRead)
def update_existing_embedding(
    embedding_id: int,
    embedding: EmbeddingUpdate,
    db: Session = Depends(get_db),
):
    if embedding.person_id is not None:
        db_person = get_person(db, embedding.person_id)
        if db_person is None:
            raise HTTPException(status_code=404, detail="Person not found")

    db_embedding = update_embedding(db, embedding_id, embedding)
    if db_embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")

    return db_embedding


@router.delete("/{embedding_id}")
def delete_existing_embedding(embedding_id: int, db: Session = Depends(get_db)):
    db_embedding = delete_embedding(db, embedding_id)
    if db_embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")

    return {"ok": True}
