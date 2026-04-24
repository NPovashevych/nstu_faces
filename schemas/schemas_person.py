from datetime import datetime
from pydantic import BaseModel
from db.models import PersonStatus
from typing import Optional


class PersonsBase(BaseModel):
    name: str
    q_code: Optional[str] = None
    link: Optional[str] = None
    status: PersonStatus = PersonStatus.unknown


class PersonsCreate(PersonsBase):
    pass


class PersonsUpdate(BaseModel):
    code: Optional[str] = None
    name: Optional[str] = None
    q_code: Optional[str] = None
    link: Optional[str] = None
    status: Optional[PersonStatus] = None


class PersonsRead(PersonsBase):
    id: int
    code: str
    created_at: datetime
    updated_at: datetime
    cluster_tag: Optional[str] = None
    cluster_distance: Optional[float] = None

    model_config = {"from_attributes": True}
