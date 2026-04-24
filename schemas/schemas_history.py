from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from schemas.schemas_user import UserRead


class HistoryBase(BaseModel):
    action: str
    user_id: int


class HistoryCreate(HistoryBase):
    pass


class HistoryUpdate(BaseModel):
    action: Optional[str] = None
    user_id: Optional[int] = None


class HistoryRead(HistoryBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryReadWithUser(HistoryRead):
    user: UserRead
