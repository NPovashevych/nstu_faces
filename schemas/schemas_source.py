from typing import Optional
from pydantic import BaseModel


class SourceBase(BaseModel):
    code: str
    name: str
    description: Optional[str] = None
    is_active: bool = True


class SourceCreate(SourceBase):
    pass


class SourceUpdate(BaseModel):
    code: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class SourceRead(SourceBase):
    id: int

    model_config = {"from_attributes": True}
