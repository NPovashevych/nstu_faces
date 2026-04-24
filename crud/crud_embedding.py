from sqlalchemy.orm import Session

from db.models import DBEmbedding
from schemas.schemas_embedding import EmbeddingCreate, EmbeddingUpdate


def get_embedding(db: Session, embedding_id: int):
    return db.query(DBEmbedding).filter(DBEmbedding.id == embedding_id).first()


def get_embeddings(db: Session, skip: int = 0, limit: int = 100):
    return db.query(DBEmbedding).offset(skip).limit(limit).all()


def get_embeddings_by_person(db: Session, person_id: int):
    return db.query(DBEmbedding).filter(DBEmbedding.person_id == person_id).all()


def create_embedding(db: Session, embedding: EmbeddingCreate):
    db_embedding = DBEmbedding(
        embedding_type=embedding.embedding_type,
        source=embedding.source,
        vector=embedding.vector,
        person_id=embedding.person_id,
    )

    db.add(db_embedding)
    db.commit()
    db.refresh(db_embedding)
    return db_embedding


def update_embedding(db: Session, embedding_id: int, embedding: EmbeddingUpdate):
    db_embedding = get_embedding(db, embedding_id)

    if db_embedding is None:
        return None

    update_data = embedding.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(db_embedding, field, value)

    db.commit()
    db.refresh(db_embedding)
    return db_embedding


def delete_embedding(db: Session, embedding_id: int):
    db_embedding = get_embedding(db, embedding_id)

    if db_embedding is None:
        return None

    db.delete(db_embedding)
    db.commit()
    return db_embedding
