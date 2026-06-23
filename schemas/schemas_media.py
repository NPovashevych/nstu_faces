from datetime import datetime
from pydantic import BaseModel
from typing import Optional

from db.enums import MediaType
from schemas.schemas_user import UserRead
from schemas.schemas_source import SourceRead


class MediaBase(BaseModel):
    material_id: Optional[str] = None
    media_type: MediaType
    mxf_path: Optional[str] = None
    mp4_path: Optional[str] = None
    duration: float
    recorded_at: Optional[datetime] = None


class MediaCreate(MediaBase):
    source_id: int
    user_id: int


class MediaUpdate(BaseModel):
    material_id: Optional[str] = None
    source_id: Optional[int] = None
    media_type: Optional[MediaType] = None
    mxf_path: Optional[str] = None
    mp4_path: Optional[str] = None
    duration: Optional[float] = None
    recorded_at: Optional[datetime] = None


class MediaRead(MediaBase):
    id: int
    uploaded_at: datetime
    source_id: int
    user_id: int

    model_config = {"from_attributes": True}


class MediaReadWithRelations(MediaRead):
    source: SourceRead
    user: UserRead
