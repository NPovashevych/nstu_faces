from pydantic import BaseModel


class NewPersonDatabaseSearchRequest(BaseModel):
    user_id: int
    name: str
    q_code: str | None = None
