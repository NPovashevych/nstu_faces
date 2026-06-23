from typing import Optional
from pydantic import BaseModel


class FaceCategoryBase(BaseModel):
    code: str
    name: str
    description: Optional[str] = None
    is_person: bool = False
    is_active: bool = True


class FaceCategoryCreate(FaceCategoryBase):
    pass


class FaceCategoryUpdate(BaseModel):
    code: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    is_person: Optional[bool] = None
    is_active: Optional[bool] = None


class FaceCategoryRead(FaceCategoryBase):
    id: int

    model_config = {"from_attributes": True}
