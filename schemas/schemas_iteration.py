from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel

from db.models import IterationStatus
from schemas.schemas_user import UserRead
from schemas.schemas_media import MediaRead


class IterationBase(BaseModel):
    status: IterationStatus = IterationStatus.processing
    params: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    user_id: int
    media_id: int


class IterationCreate(IterationBase):
    pass


class IterationUpdate(BaseModel):
    status: Optional[IterationStatus] = None
    params: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    finished_at: Optional[datetime] = None
    user_id: Optional[int] = None
    media_id: Optional[int] = None


class IterationRead(IterationBase):
    id: int
    created_at: datetime
    started_at: datetime
    finished_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class IterationReadWithMediaUser(IterationRead):
    user: UserRead
    media: MediaRead
