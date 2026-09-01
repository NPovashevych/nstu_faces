from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel

from schemas.schemas_user import UserRead


class HistoryBase(BaseModel):
    action: str
    details: Optional[dict[str, Any]] = None


class HistoryCreate(HistoryBase):
    user_id: int


class HistoryRead(HistoryBase):
    id: int
    user_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryReadWithUser(HistoryRead):
    user: UserRead
