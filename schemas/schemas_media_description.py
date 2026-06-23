from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel


class MediaDescriptionBase(BaseModel):
    material_id: str
    section: Optional[str] = None
    shooting_date: str
    journalist: Optional[str] = None
    operators: Optional[str] = None
    description: Optional[str] = None
    another_info: Optional[dict[str, Any]] = None

    source_path: str
    source_hash: Optional[str] = None


class MediaDescriptionCreate(MediaDescriptionBase):
    pass


class MediaDescriptionUpdate(BaseModel):
    material_id: Optional[str] = None
    section: Optional[str] = None
    shooting_date: Optional[str] = None
    journalist: Optional[str] = None
    operators: Optional[str] = None
    description: Optional[str] = None
    another_info: Optional[dict[str, Any]] = None
    source_path: Optional[str] = None
    source_hash: Optional[str] = None


class MediaDescriptionRead(MediaDescriptionBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
