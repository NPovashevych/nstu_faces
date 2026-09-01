from sqlalchemy import Column, Integer, String, Enum as SAEnum, DateTime, ForeignKey, Float, text, Boolean
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from db.engine import BASE

from db.enums import UserRole, PersonStatus, EmbeddingType, MediaType, IterationStatus, FaceGender


class DBUser(BASE):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True, index=True)
    email = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(SAEnum(UserRole, name="user_role"), nullable=False)

    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    medias = relationship("DBMedia", back_populates="user")
    iterations = relationship("DBIteration", back_populates="user")
    histories = relationship("DBHistory", back_populates="user")


class DBPerson(BASE):
    __tablename__ = "person"

    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False, index=True)
    q_code = Column(String, nullable=True, unique=True, index=True)
    link = Column(String, nullable=True)
    status = Column(SAEnum(PersonStatus, name="person_status"), nullable=False, default=PersonStatus.unknown, server_default=text("'unknown'"))
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
    cluster_id = Column(Integer, nullable=True, index=True)
    cluster_tag = Column(String, nullable=True, index=True)
    cluster_distance = Column(Float, nullable=True)

    embeddings = relationship("DBEmbedding", back_populates="person", cascade="all, delete-orphan")
    faces = relationship("DBFace", back_populates="person")


class DBEmbedding(BASE):
    __tablename__ = "embedding"

    id = Column(Integer, primary_key=True)
    embedding_type = Column(SAEnum(EmbeddingType, name="embedding_type"), nullable=False)
    source = Column(JSONB, nullable=True)
    vector = Column(ARRAY(Float), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    person_id = Column(Integer, ForeignKey("person.id"), nullable=False, index=True)
    person = relationship("DBPerson", back_populates="embeddings")

    faces = relationship("DBFace", back_populates="embedding")


class DBSource(BASE):
    __tablename__ = "source"

    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(String, nullable=True)
    is_active = Column(Boolean, nullable=False, server_default=text("true"))

    medias = relationship("DBMedia", back_populates="source")


class DBMedia(BASE):
    __tablename__ = "media"

    id = Column(Integer, primary_key=True)
    material_id = Column(String, nullable=True, unique=True, index=True)
    media_type = Column(SAEnum(MediaType, name="media_type"), nullable=False, index=True)
    mxf_path = Column(String, nullable=True, unique=True, index=True)
    mp4_path = Column(String, nullable=True, unique=True, index=True)
    duration = Column(Float, nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=True)
    uploaded_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    source_id = Column(Integer, ForeignKey("source.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False, index=True)

    user = relationship("DBUser", back_populates="medias")
    source = relationship("DBSource", back_populates="medias")
    freezes = relationship("DBFreeze", back_populates="media", cascade="all, delete-orphan")
    iterations = relationship("DBIteration", back_populates="media", cascade="all, delete-orphan")


class DBFreeze(BASE):
    __tablename__ = "freeze"

    id = Column(Integer, primary_key=True)
    time_in = Column(Float, nullable=False)
    time_out = Column(Float, nullable=False)
    freeze_path = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    media_id = Column(Integer, ForeignKey("media.id"), nullable=False, index=True)
    media = relationship("DBMedia", back_populates="freezes")

    faces = relationship("DBFace", back_populates="freeze", cascade="all, delete-orphan")


class DBIteration(BASE):
    __tablename__ = "iteration"

    id = Column(Integer, primary_key=True)
    status = Column(SAEnum(IterationStatus, name="iteration_status"), nullable=False, index=True, default=IterationStatus.processing, server_default=text("'processing'"))
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    params = Column(JSONB, nullable=True)
    error_message = Column(String, nullable=True)

    user_id = Column(Integer, ForeignKey("user.id"), nullable=False, index=True)
    user = relationship("DBUser", back_populates="iterations")

    media_id = Column(Integer, ForeignKey("media.id"), nullable=False, index=True)
    media = relationship("DBMedia", back_populates="iterations")

    faces = relationship("DBFace", back_populates="iteration", cascade="all, delete-orphan")


class DBHistory(BASE):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True)
    action = Column(String, nullable=False, index=True)
    details = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    user_id = Column(Integer, ForeignKey("user.id"), nullable=False, index=True)
    user = relationship("DBUser", back_populates="histories")


class DBFaceCategory(BASE):
    __tablename__ = "face_category"

    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(String, nullable=True)

    is_person = Column(Boolean, nullable=False, index=True, server_default=text("false"))
    is_active = Column(Boolean, nullable=False, server_default=text("true"))

    faces = relationship("DBFace", back_populates="face_category")


class DBFace(BASE):
    __tablename__ = "face"

    id = Column(Integer, primary_key=True)
    bbox = Column(ARRAY(Float), nullable=False)
    quality = Column(Float, nullable=True, index=True)
    gender = Column(SAEnum(FaceGender, name="face_gender"), nullable=False, index=True, default=FaceGender.unknown, server_default=text("'unknown'"))
    confidence = Column(Integer, nullable=True, index=True)
    analysis = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    embedding_id = Column(Integer, ForeignKey("embedding.id"), nullable=False, index=True)
    embedding = relationship("DBEmbedding", back_populates="faces")

    freeze_id = Column(Integer, ForeignKey("freeze.id"), nullable=False, index=True)
    freeze = relationship("DBFreeze", back_populates="faces")

    person_id = Column(Integer, ForeignKey("person.id"), nullable=True, index=True)
    person = relationship("DBPerson", back_populates="faces")

    category_id = Column(Integer, ForeignKey("face_category.id"), nullable=False, index=True)
    face_category = relationship("DBFaceCategory", back_populates="faces")
    category_score = Column(Float, nullable=True, index=True)

    iteration_id = Column(Integer, ForeignKey("iteration.id"), nullable=False, index=True)
    iteration = relationship("DBIteration", back_populates="faces")


class DBMediaDescription(BASE):
    __tablename__ = "media_description"
    id = Column(Integer, primary_key=True)

    """ дані з firebird """
    material_id = Column(String, nullable=False, index=True)
    section = Column(String, nullable=True, index=True)
    shooting_date = Column(String, nullable=False, index=True)
    journalist = Column(String, nullable=True)
    operators = Column(String, nullable=True)
    description = Column(String, nullable=True)
    another_info = Column(JSONB, nullable=True)

    """ сервісні параметри """
    source_path = Column(String, nullable=False, index=True)
    source_hash = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

