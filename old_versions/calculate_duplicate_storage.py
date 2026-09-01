import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from services.config import (
    PROXY_INTVNEWS_DUPLICATE_FILE,
    PROXY_DIGITAL_DUPLICATE_FILE,
    HIRES_INTVNEWS_DUPLICATE_FILE,
    HIRES_DIGITAL_DUPLICATE_FILE,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../services/logs/calculate_duplicate_storage.log", encoding="utf-8"),
    ],
)


REPORT_PATH = Path(r"C:\faces\baza\duplicate_storage_report.json")


@dataclass
class DuplicateStorageStats:
    name: str
    duplicate_groups: int = 0
    files_in_duplicate_groups: int = 0
    redundant_files: int = 0
    redundant_bytes: int = 0
    same_size_groups: int = 0
    different_size_groups: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["redundant_mb"] = round(self.redundant_bytes / 1024 / 1024, 2)
        data["redundant_gb"] = round(self.redundant_bytes / 1024 / 1024 / 1024, 2)
        data["redundant_tb"] = round(self.redundant_bytes / 1024 / 1024 / 1024 / 1024, 3)
        return data


def load_json(path: Path) -> dict:
    if not path.exists():
        logging.warning(f"File not found: {path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp.json")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    temp_path.replace(path)


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} PB"


def normalize_duplicate_items(value) -> list[dict]:
    """
    Очікуваний формат:
    {
        "name": [
            {item1},
            {item2}
        ]
    }

    Але робимо трохи стійкіше на майбутнє.
    """
    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        if "items" in value and isinstance(value["items"], list):
            return value["items"]

        if "duplicates" in value and isinstance(value["duplicates"], list):
            items = []

            selected = value.get("selected")
            if isinstance(selected, dict):
                items.append(selected)

            items.extend(value["duplicates"])
            return items

    return []


def calculate_duplicate_storage(name: str, duplicate_file: Path) -> tuple[DuplicateStorageStats, dict]:
    duplicates = load_json(duplicate_file)

    stats = DuplicateStorageStats(name=name)
    details = {}

    for key, raw_items in duplicates.items():
        items = normalize_duplicate_items(raw_items)
        items = [
            item for item in items
            if isinstance(item, dict) and isinstance(item.get("size"), int)
        ]

        if len(items) < 2:
            continue

        stats.duplicate_groups += 1
        stats.files_in_duplicate_groups += len(items)

        sizes = [item["size"] for item in items]
        max_size = max(sizes)
        redundant_size = sum(sizes) - max_size
        redundant_count = len(items) - 1

        stats.redundant_files += redundant_count
        stats.redundant_bytes += redundant_size

        if len(set(sizes)) == 1:
            stats.same_size_groups += 1
        else:
            stats.different_size_groups += 1

        sorted_items = sorted(items, key=lambda item: item["size"], reverse=True)

        details[key] = {
            "keep_largest": sorted_items[0],
            "redundant_files": sorted_items[1:],
            "redundant_count": redundant_count,
            "redundant_bytes": redundant_size,
            "redundant_human": human_size(redundant_size),
            "same_size": len(set(sizes)) == 1,
        }

    return stats, details


def log_stats(stats: DuplicateStorageStats):
    data = stats.to_dict()

    logging.info("--------------------------------")
    logging.info(f"[{stats.name}]")
    logging.info(f"Duplicate groups: {stats.duplicate_groups}")
    logging.info(f"Files in duplicate groups: {stats.files_in_duplicate_groups}")
    logging.info(f"Redundant files: {stats.redundant_files}")
    logging.info(f"Same-size groups: {stats.same_size_groups}")
    logging.info(f"Different-size groups: {stats.different_size_groups}")
    logging.info(f"Redundant size: {human_size(stats.redundant_bytes)}")
    logging.info(f"Redundant GB: {data['redundant_gb']}")
    logging.info(f"Redundant TB: {data['redundant_tb']}")


def main():
    start = datetime.now()
    logging.info(f"Start: {start}")

    targets = [
        ("mxf_intvnews", HIRES_INTVNEWS_DUPLICATE_FILE),
        ("mxf_digital", HIRES_DIGITAL_DUPLICATE_FILE),
        ("mp4_intvnews", PROXY_INTVNEWS_DUPLICATE_FILE),
        ("mp4_digital", PROXY_DIGITAL_DUPLICATE_FILE),
    ]

    report = {
        "created_at": datetime.now().isoformat(),
        "summary": {},
        "details": {},
    }

    total_mxf = 0
    total_mp4 = 0

    for name, duplicate_file in targets:
        stats, details = calculate_duplicate_storage(
            name=name,
            duplicate_file=duplicate_file,
        )

        log_stats(stats)

        report["summary"][name] = stats.to_dict()
        report["details"][name] = details

        if name.startswith("mxf_"):
            total_mxf += stats.redundant_bytes

        if name.startswith("mp4_"):
            total_mp4 += stats.redundant_bytes

    report["total"] = {
        "mxf_redundant_bytes": total_mxf,
        "mxf_redundant_human": human_size(total_mxf),
        "mp4_redundant_bytes": total_mp4,
        "mp4_redundant_human": human_size(total_mp4),
        "all_redundant_bytes": total_mxf + total_mp4,
        "all_redundant_human": human_size(total_mxf + total_mp4),
    }

    logging.info("--------------------------------")
    logging.info("[TOTAL]")
    logging.info(f"MXF redundant: {human_size(total_mxf)}")
    logging.info(f"MP4 redundant: {human_size(total_mp4)}")
    logging.info(f"ALL redundant: {human_size(total_mxf + total_mp4)}")

    save_json(REPORT_PATH, report)
    logging.info(f"Report saved: {REPORT_PATH}")

    finish = datetime.now()
    logging.info(f"Finished. Running time: {finish - start}")


if __name__ == "__main__":
    main()
