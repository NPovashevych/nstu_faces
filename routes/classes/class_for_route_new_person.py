from pydantic import BaseModel


class NewPersonDatabaseSearchRequest(BaseModel):
    user_id: int
    name: str
    pseudonym: str | None = None
    q_code: str | None = None


class NewPersonCheckRequest(BaseModel):
    user_id: int
    workspace_id: str
    name: str
    q_code: str | None = None


class NewPersonSaveRequest(BaseModel):
    user_id: int
    workspace_id: str
    name: str
    pseudonym: str | None = None
    q_code: str | None = None
    category: str | None = None
    like_unknown: bool = False


class NewPersonWorkspaceRequest(BaseModel):
    user_id: int
    name: str
