from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel

from db.enums import FaceGender
from schemas.schemas_embedding import EmbeddingRead
from schemas.schemas_freeze import FreezeRead
from schemas.schemas_person import PersonsRead
from schemas.schemas_iteration import IterationRead
from schemas.schemas_face_category import FaceCategoryRead


class FaceBase(BaseModel):
    bbox: list[float]
    quality: Optional[float] = None
    gender: FaceGender = FaceGender.unknown
    confidence: Optional[int] = None
    analysis: Optional[dict[str, Any]] = None

    category_id: int
    category_score: Optional[float] = None

    embedding_id: int
    freeze_id: int
    person_id: Optional[int] = None
    iteration_id: int


class FaceCreate(FaceBase):
    pass


class FaceUpdate(BaseModel):
    bbox: Optional[list[float]] = None
    quality: Optional[float] = None
    gender: Optional[FaceGender] = None
    confidence: Optional[int] = None
    analysis: Optional[dict[str, Any]] = None

    category_id: Optional[int] = None
    category_score: Optional[float] = None

    embedding_id: Optional[int] = None
    freeze_id: Optional[int] = None
    person_id: Optional[int] = None
    iteration_id: Optional[int] = None


class FaceRead(FaceBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class FaceReadFull(FaceRead):
    face_category: FaceCategoryRead
    embedding: EmbeddingRead
    freeze: FreezeRead
    person: Optional[PersonsRead] = None
    iteration: IterationRead
