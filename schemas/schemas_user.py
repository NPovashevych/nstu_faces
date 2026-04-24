from datetime import datetime
from pydantic import BaseModel, EmailStr
from db.models import UserRole
from typing import Optional


class UserBase(BaseModel):
    name: str
    email: EmailStr
    role: UserRole


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    role: Optional[UserRole] = None


class UserRead(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
