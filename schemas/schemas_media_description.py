from pydantic import BaseModel
from typing import Optional

from schemas.schemas_media import MediaRead


class MediaDescriptionBase(BaseModel):
    media_id: int
    section: Optional[str] = None
    description: Optional[str] = None
    date: Optional[str] = None
    duration: Optional[str] = None
    journalist: Optional[str] = None


class MediaDescriptionCreate(MediaDescriptionBase):
    pass


class MediaDescriptionUpdate(BaseModel):
    media_id: Optional[int] = None
    section: Optional[str] = None
    description: Optional[str] = None
    date: Optional[str] = None
    duration: Optional[str] = None
    journalist: Optional[str] = None


class MediaDescriptionRead(MediaDescriptionBase):
    id: int

    model_config = {"from_attributes": True}


class MediaDescriptionReadWithMedia(MediaDescriptionRead):
    media: MediaRead
