from datetime import datetime
from pydantic import BaseModel
from typing import Optional

from db.enums import MediaType, MediaSource
from schemas.schemas_user import UserRead


class MediaBase(BaseModel):
    source: MediaSource
    media_type: MediaType

    mxf_path: Optional[str] = None
    mp4_path: Optional[str] = None

    recorded_at: Optional[datetime] = None


class MediaCreate(MediaBase):
    pass


class MediaUpdate(BaseModel):
    source: Optional[MediaSource] = None
    media_type: Optional[MediaType] = None

    mxf_path: Optional[str] = None
    mp4_path: Optional[str] = None

    recorded_at: Optional[datetime] = None


class MediaRead(MediaBase):
    id: int
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class MediaReadWithUser(MediaRead):
    user_id: int
    user: UserRead
