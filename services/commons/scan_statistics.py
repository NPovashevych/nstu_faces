import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta


@dataclass
class ScanStatistics:
    name: str = "scan"
    progress_step: int = 10_000

    start_time: float = field(default_factory=time.perf_counter)
    last_report_time: float = field(default_factory=time.perf_counter)

    folders_scanned: int = 0
    files_scanned: int = 0
    matched_files: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    missing: int = 0
    duplicates: int = 0
    errors: int = 0

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    def elapsed_text(self) -> str:
        return str(timedelta(seconds=int(self.elapsed_seconds())))

    def files_per_second(self) -> float:
        elapsed = self.elapsed_seconds()
        if elapsed <= 0:
            return 0.0
        return self.files_scanned / elapsed

    def matched_per_second(self) -> float:
        elapsed = self.elapsed_seconds()
        if elapsed <= 0:
            return 0.0
        return self.matched_files / elapsed

    def should_report(self) -> bool:
        return (
            self.files_scanned > 0
            and self.files_scanned % self.progress_step == 0
        )

    def report_progress(self):
        logging.info(
            f"[{self.name}] "
            f"folders={self.folders_scanned} | "
            f"files={self.files_scanned} | "
            f"matched={self.matched_files} | "
            f"elapsed={self.elapsed_text()} | "
            f"files/sec={self.files_per_second():.2f} | "
            f"matched/sec={self.matched_per_second():.2f}"
        )

    def report_summary(self):
        logging.info("--------------------------------")
        logging.info(f"[{self.name}] summary")
        logging.info(f"Elapsed: {self.elapsed_text()}")
        logging.info(f"Folders scanned: {self.folders_scanned}")
        logging.info(f"Files scanned: {self.files_scanned}")
        logging.info(f"Matched files: {self.matched_files}")
        logging.info(f"Added: {self.added}")
        logging.info(f"Updated: {self.updated}")
        logging.info(f"Unchanged: {self.unchanged}")
        logging.info(f"Missing marked: {self.missing}")
        logging.info(f"Duplicates: {self.duplicates}")
        logging.info(f"Errors: {self.errors}")
        logging.info(f"Files/sec: {self.files_per_second():.2f}")
        logging.info(f"Matched/sec: {self.matched_per_second():.2f}")
