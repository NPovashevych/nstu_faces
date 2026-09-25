from pydantic import BaseModel


class CandidateUserRequest(BaseModel):
    user_id: int


class CandidateSaveRequest(BaseModel):
    user_id: int
    candidate_key: str
    final_name: str
    category: str | None = None


class CandidateSkipRequest(BaseModel):
    user_id: int
    candidate_key: str


class CandidateCancelRequest(BaseModel):
    user_id: int
    candidate_key: str
