from enum import Enum as PyEnum


class UserRole(str, PyEnum):
    journalist = "journalist"
    archivist = "archivist"
    developer = "developer"
    tester = "tester"


class PersonStatus(str, PyEnum):
    public = "public"
    non_public = "non_public"
    unknown = "unknown"
    suspicious = "suspicious"


class EmbeddingType(str, PyEnum):
    reference_face = "reference_face"
    detected_face = "detected_face"


class MediaType(str, PyEnum):
    image = "image"
    video = "video"


class IterationStatus(str, PyEnum):
    processing = "processing"
    completed = "completed"
    error = "error"


class FaceGender(str, PyEnum):
    male = "male"
    female = "female"
    unknown = "unknown"
