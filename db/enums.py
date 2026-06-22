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


class FaceCategory(str, PyEnum):
    real_identifiable = "real_identifiable"
    real_unidentifiable = "real_unidentifiable"
    low_quality = "low_quality"
    non_human = "non_human"
    artificial_human = "artificial_human"
    ai_generated = "ai_generated"
    uncertain = "uncertain"


class MediaSource(str, PyEnum):
    in_media = "in_media"
    in_tv_news = "in_tv_news"
    digital = "digital"
    test_media = "test_media"
    user_upload = "user_upload"

