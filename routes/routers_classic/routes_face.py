from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.session import get_db
from crud.crud_embedding import get_embedding, get_embeddings, get_embeddings_by_person
from crud.crud_person import get_person
from schemas.schemas_embedding import EmbeddingRead, EmbeddingReadWithPerson


router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.get("/", response_model=List[EmbeddingRead])
def read_embeddings(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return get_embeddings(db, skip=skip, limit=limit)


@router.get("/person/{person_id}", response_model=List[EmbeddingRead])
def read_embeddings_by_person(person_id: int, db: Session = Depends(get_db)):
    if get_person(db, person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")

    return get_embeddings_by_person(db, person_id)


@router.get("/{embedding_id}/full", response_model=EmbeddingReadWithPerson)
def read_embedding_full(embedding_id: int, db: Session = Depends(get_db)):
    db_embedding = get_embedding(db, embedding_id)

    if db_embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")

    return db_embedding


@router.get("/{embedding_id}", response_model=EmbeddingRead)
def read_embedding(embedding_id: int, db: Session = Depends(get_db)):
    db_embedding = get_embedding(db, embedding_id)

    if db_embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")

    return db_embedding
