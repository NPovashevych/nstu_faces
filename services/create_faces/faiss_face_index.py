import logging
import threading
import faiss
import numpy as np
from sqlalchemy.orm import Session

from db.enums import EmbeddingType, PersonStatus
from db.models import DBEmbedding, DBPerson


DEFAULT_EMBEDDING_DIM = 512


def normalize_vector(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


class ReferenceFaceIndex:

    def __init__(self):
        self.index = None
        self.metadata = []
        self.dimension = None

    @property
    def size(self) -> int:
        if self.index is None:
            return 0

        return int(self.index.ntotal)

    def build(self, db: Session):
        rows = (
            db.query(DBEmbedding)
            .filter(DBEmbedding.embedding_type == EmbeddingType.reference_face)
            .order_by(DBEmbedding.id)
            .all()
        )

        if not rows:
            logging.warning("Reference embeddings not found. Reference FAISS index is empty.")
            return

        vectors = []
        metadata = []

        for row in rows:
            vector = normalize_vector(row.vector)

            vectors.append(vector)
            source = row.source or {}
            gender = source.get("gender", "unknown")

            if gender not in {"male", "female", "unknown"}:
                gender = "unknown"

            metadata.append({"embedding_id": row.id, "person_id": row.person_id, "gender": gender})

        matrix = np.vstack(vectors).astype(np.float32)

        self.dimension = matrix.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(matrix)
        self.metadata = metadata

        logging.info(f"Reference FAISS built: embeddings={self.size}, dimension={self.dimension}")

    def find_best_match(self, embedding):
        if self.index is None or self.index.ntotal == 0:
            return None, 1.0

        vector = normalize_vector(embedding).reshape(1, -1)
        similarities, indices = self.index.search(vector, 1)

        position = int(indices[0][0])

        if position < 0:
            return None, 1.0

        similarity = float(similarities[0][0])
        distance = 1.0 - similarity
        metadata = self.metadata[position]

        return metadata, distance


class UnknownFaceIndex:

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIM):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)

        # position у FAISS -> person_id
        self.person_ids = []

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    @property
    def person_count(self) -> int:
        return len(set(self.person_ids))

    def build(self, db: Session):
        rows = (
            db.query(DBEmbedding)
            .join(DBPerson, DBEmbedding.person_id == DBPerson.id)
            .filter(DBEmbedding.embedding_type == EmbeddingType.detected_face)
            .filter(DBPerson.status == PersonStatus.unknown)
            .order_by(DBEmbedding.id)
            .all()
        )

        if not rows:
            logging.info("No existing unknown embeddings. Unknown FAISS starts empty.")
            return

        vectors = []
        person_ids = []

        for row in rows:
            vector = normalize_vector(row.vector)

            if vector.shape[0] != self.dimension:
                raise RuntimeError(f"Dimension err: embedding_id={row.id}, dimension={vector.shape[0]}, expected={self.dimension}")

            vectors.append(vector)
            person_ids.append(row.person_id)

        matrix = np.vstack(vectors).astype(np.float32)

        self.index.add(matrix)
        self.person_ids.extend(person_ids)

        logging.info(f"Unknown FAISS built: embeddings={self.size}, persons={self.person_count}, dimension={self.dimension}")

    def find_best_match(self, embedding):
        if self.index.ntotal == 0:
            return None, None

        vector = normalize_vector(embedding).reshape(1, -1)
        similarities, indices = self.index.search(vector, 1)

        position = int(indices[0][0])

        if position < 0:
            return None, None

        similarity = float(similarities[0][0])
        distance = 1.0 - similarity
        person_id = self.person_ids[position]

        return person_id, distance

    def add(self, embedding, person_id: int):

        vector = normalize_vector(embedding)

        if vector.shape[0] != self.dimension:
            raise RuntimeError(f"Unexpected embedding dimension: {vector.shape[0]}, expected={self.dimension}")

        self.index.add(vector.reshape(1, -1))
        self.person_ids.append(person_id)


REFERENCE_FACE_INDEX = ReferenceFaceIndex()
UNKNOWN_FACE_INDEX = UnknownFaceIndex()

_FAISS_INDEXES_READY = False
_FAISS_BUILD_LOCK = threading.Lock()


def initialize_faiss_indexes(db: Session):
    global _FAISS_INDEXES_READY

    if _FAISS_INDEXES_READY:
        return

    with _FAISS_BUILD_LOCK:
        if _FAISS_INDEXES_READY:
            return

        logging.info("Building shared FAISS indexes...")

        REFERENCE_FACE_INDEX.build(db)
        UNKNOWN_FACE_INDEX.build(db)

        _FAISS_INDEXES_READY = True

        logging.info(
            "Shared FAISS indexes ready: "
            "references=%s, unknown_embeddings=%s, unknown_persons=%s",
            REFERENCE_FACE_INDEX.size,
            UNKNOWN_FACE_INDEX.size,
            UNKNOWN_FACE_INDEX.person_count,
        )


def ensure_faiss_indexes(db: Session):
    if not _FAISS_INDEXES_READY:
        initialize_faiss_indexes(db)
