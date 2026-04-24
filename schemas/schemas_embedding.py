from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel

from db.models import EmbeddingType
from schemas.schemas_person import  PersonsRead


class EmbeddingBase(BaseModel):
    embedding_type: EmbeddingType
    source: Optional[dict[str, Any]] = None
    vector: list[float]
    person_id: int


class EmbeddingCreate(EmbeddingBase):
    pass


class EmbeddingUpdate(BaseModel):
    embedding_type: Optional[EmbeddingType] = None
    source: Optional[dict[str, Any]] = None
    vector: Optional[list[float]] = None
    person_id: Optional[int] = None


class EmbeddingRead(EmbeddingBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class EmbeddingReadWithPerson(EmbeddingRead):
    person: PersonsRead
