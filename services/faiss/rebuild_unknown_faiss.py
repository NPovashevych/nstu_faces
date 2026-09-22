import logging
import time

import faiss
import numpy as np

from db.enums import EmbeddingType, PersonStatus
from db.models import DBEmbedding, DBPerson
from db.session import SessionLocal

from services.config import UNKNOWN_FAISS_INDEX_PATH, UNKNOWN_FAISS_PERSON_IDS_PATH


DEFAULT_EMBEDDING_DIM = 512
PROGRESS_STEP = 100_000


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
)


def normalize_vector(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)

    if hours:
        return f"{hours:d}h {minutes:02d}m {sec:05.2f}s"

    if minutes:
        return f"{minutes:d}m {sec:05.2f}s"

    return f"{sec:.2f}s"


def rebuild_unknown_faiss():
    total_started = time.perf_counter()
    db = SessionLocal()

    try:
        query_started = time.perf_counter()

        rows = (
            db.query(
                DBEmbedding.id,
                DBEmbedding.person_id,
                DBEmbedding.vector,
            )
            .join(DBPerson, DBEmbedding.person_id == DBPerson.id)
            .filter(DBEmbedding.embedding_type == EmbeddingType.detected_face)
            .filter(DBPerson.status == PersonStatus.unknown)
            .all()
        )

        query_elapsed = time.perf_counter() - query_started

        if not rows:
            logging.warning("No unknown embeddings found. FAISS files were not created.")
            return

        prepare_started = time.perf_counter()

        vectors = []
        person_ids = []

        total_rows = len(rows)

        logging.info("Preparing vectors...")

        for i, row in enumerate(rows, start=1):
            embedding_id = row.id
            person_id = row.person_id
            vector = normalize_vector(row.vector)

            if vector.shape[0] != DEFAULT_EMBEDDING_DIM:
                raise RuntimeError(f"Unexpected embedding dimension: embedding_id={embedding_id}, dimension={vector.shape[0]}, expected={DEFAULT_EMBEDDING_DIM}")

            vectors.append(vector)
            person_ids.append(person_id)

            if i % PROGRESS_STEP == 0 or i == total_rows:
                elapsed = time.perf_counter() - prepare_started
                percent = i / total_rows * 100

                rows_per_second = i / elapsed if elapsed > 0 else 0
                remaining = total_rows - i
                eta_seconds = remaining / rows_per_second if rows_per_second > 0 else 0

                logging.info(
                    "Prepared %s / %s (%.1f%%), elapsed=%s, speed=%.0f emb/s, ETA=%s",
                    f"{i:,}",
                    f"{total_rows:,}",
                    percent,
                    format_seconds(elapsed),
                    rows_per_second,
                    format_seconds(eta_seconds),
                )

        prepare_elapsed = time.perf_counter() - prepare_started

        matrix_started = time.perf_counter()

        logging.info("Creating numpy matrix...")

        matrix = np.vstack(vectors).astype(np.float32)
        person_ids_array = np.asarray(person_ids, dtype=np.int64)

        matrix_elapsed = time.perf_counter() - matrix_started
        logging.info("Numpy matrix created: shape=%s, time=%s", matrix.shape, format_seconds(matrix_elapsed))

        faiss_started = time.perf_counter()
        logging.info("Building FAISS index: embeddings=%s, dimension=%s",  f"{matrix.shape[0]:,}", matrix.shape[1])

        index = faiss.IndexFlatIP(DEFAULT_EMBEDDING_DIM)
        index.add(matrix)

        faiss_elapsed = time.perf_counter() - faiss_started
        logging.info("FAISS index built: ntotal=%s, time=%s", f"{index.ntotal:,}", format_seconds(faiss_elapsed))


        if index.ntotal != len(person_ids_array):
            raise RuntimeError(f"FAISS/person_ids count mismatch: faiss={index.ntotal}, person_ids={len(person_ids_array)}")


        write_started = time.perf_counter()
        logging.info("Writing FAISS index to: %s", UNKNOWN_FAISS_INDEX_PATH)
        faiss.write_index(index, str(UNKNOWN_FAISS_INDEX_PATH))
        logging.info("Writing person_ids to: %s", UNKNOWN_FAISS_PERSON_IDS_PATH)
        np.save(UNKNOWN_FAISS_PERSON_IDS_PATH, person_ids_array)
        write_elapsed = time.perf_counter() - write_started


        total_elapsed = time.perf_counter() - total_started
        logging.info("================ RESULT ================")
        logging.info("FAISS file:       %s", UNKNOWN_FAISS_INDEX_PATH)
        logging.info("Person IDs file:  %s", UNKNOWN_FAISS_PERSON_IDS_PATH)
        logging.info("Embeddings:       %s", f"{index.ntotal:,}")
        logging.info("Person IDs:       %s", f"{len(person_ids_array):,}")
        logging.info("Unknown persons:  %s", f"{len(set(person_ids)):,}")
        logging.info("Dimension:        %s", DEFAULT_EMBEDDING_DIM)
        logging.info("----------------------------------------")
        logging.info("DB query:         %s", format_seconds(query_elapsed))
        logging.info("Prepare vectors:  %s", format_seconds(prepare_elapsed))
        logging.info("Numpy matrix:     %s", format_seconds(matrix_elapsed))
        logging.info("FAISS add:        %s", format_seconds(faiss_elapsed))
        logging.info("Write files:      %s", format_seconds(write_elapsed))
        logging.info("----------------------------------------")
        logging.info("TOTAL:            %s", format_seconds(total_elapsed))
        logging.info("========================================")

    finally:
        db.close()


if __name__ == "__main__":
    rebuild_unknown_faiss()
