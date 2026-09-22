from time import perf_counter
import logging


class PerformanceProfiler:
    def __init__(self, log_every_freezes=500):
        self.log_every_freezes = log_every_freezes

        self.freezes = 0
        self.buffalo_faces = 0
        self.created_faces = 0

        self.read_time = 0.0
        self.buffalo_time = 0.0
        self.reference_faiss_time = 0.0
        self.quality_time = 0.0
        self.clip_time = 0.0
        self.unknown_faiss_time = 0.0
        self.db_time = 0.0

        self.started_at = perf_counter()

    def add(self, field: str, seconds: float):
        setattr(
            self,
            field,
            getattr(self, field) + seconds,
        )

    def freeze_completed(
        self,
        buffalo_faces: int,
        created_faces: int,
    ):
        self.freezes += 1
        self.buffalo_faces += buffalo_faces
        self.created_faces += created_faces

        if self.freezes % self.log_every_freezes == 0:
            self.log()

    def log(self):
        elapsed = perf_counter() - self.started_at

        if self.freezes:
            sec_per_freeze = elapsed / self.freezes
            freezes_per_hour = (
                self.freezes / elapsed * 3600
            )
        else:
            sec_per_freeze = 0
            freezes_per_hour = 0

        measured = (
            self.read_time
            + self.buffalo_time
            + self.reference_faiss_time
            + self.quality_time
            + self.clip_time
            + self.unknown_faiss_time
            + self.db_time
        )

        other_time = max(
            0.0,
            elapsed - measured,
        )

        logging.info("")
        logging.info(
            "================ PERFORMANCE ================"
        )
        logging.info(
            f"Freezes:              {self.freezes}"
        )
        logging.info(
            f"Buffalo faces:        {self.buffalo_faces}"
        )
        logging.info(
            f"Created faces:        {self.created_faces}"
        )
        logging.info(
            f"Elapsed:              {elapsed:.1f} s"
        )
        logging.info(
            f"Seconds / freeze:     {sec_per_freeze:.3f}"
        )
        logging.info(
            f"Freezes / hour:       {freezes_per_hour:.0f}"
        )
        logging.info("")
        logging.info(
            f"READ:                 {self.read_time:.1f} s"
        )
        logging.info(
            f"BUFFALO:              {self.buffalo_time:.1f} s"
        )
        logging.info(
            f"REFERENCE FAISS:      {self.reference_faiss_time:.1f} s"
        )
        logging.info(
            f"QUALITY:              {self.quality_time:.1f} s"
        )
        logging.info(
            f"CLIP:                 {self.clip_time:.1f} s"
        )
        logging.info(
            f"UNKNOWN FAISS:        {self.unknown_faiss_time:.1f} s"
        )
        logging.info(
            f"DB:                   {self.db_time:.1f} s"
        )
        logging.info(
            f"OTHER:                {other_time:.1f} s"
        )
        logging.info(
            "============================================="
        )
        logging.info("")
