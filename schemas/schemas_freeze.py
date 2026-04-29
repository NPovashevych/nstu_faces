from datetime import datetime
from pydantic import BaseModel
from typing import Optional

from schemas.schemas_media import MediaRead


class FreezeBase(BaseModel):
    time_in: float
    time_out: float
    freeze_path: str
    media_id: int


class FreezeCreate(FreezeBase):
    pass


class FreezeUpdate(BaseModel):
    time_in: Optional[float] = None
    time_out: Optional[float] = None
    freeze_path: Optional[str] = None
    media_id: Optional[int] = None


class FreezeRead(FreezeBase):
    id: int
    created_at: datetime
    media: MediaRead

    model_config = {"from_attributes": True}
