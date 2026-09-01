from pydantic import BaseModel, Field
from typing import Optional


class UpdateClusterRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    cluster_tag: Optional[str] = Field(default=None, max_length=255)


class AssignToPersonRequest(BaseModel):
    person_id: int = Field(gt=0)
    face_ids: Optional[list[int]] = None
    confidence: int = Field(default=0, ge=0, le=3)     # confidence=0 означає ручне підтвердження без знаків питання.
    delete_empty_cluster: bool = True                  # Якщо після операції вихідний кластер став порожнім, його можна автоматично видалити.


class MoveFacesRequest(BaseModel):
    target_cluster_key: str = Field(min_length=1)
    face_ids: list[int] = Field(min_length=1)
    delete_empty_cluster: bool = True


class SplitClusterRequest(BaseModel):
    face_ids: list[int] = Field(min_length=1)
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)     # Необов’язкова зрозуміла  назва.
    cluster_tag: Optional[str] = Field(default=None, max_length=255)
    delete_empty_cluster: bool = True


class MergeClustersRequest(BaseModel):
    target_cluster_key: str = Field(min_length=1)
    delete_source_cluster: bool = True
