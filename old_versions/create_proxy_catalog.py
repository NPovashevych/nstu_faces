import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from services.config import (
    PROXY_NEWS_FOLDER,
    PROXY_DIGITAL_FOLDER,
    PROXY_INTVNEWS_CATALOG_PATH,
    PROXY_DIGITAL_CATALOG_PATH,
    PROXY_INTVNEWS_DUPLICATE_FILE,
    PROXY_DIGITAL_DUPLICATE_FILE,
    GARBAGE_FOLDER_NAMES,
)
from services.commons.scan_statistics import ScanStatistics


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../services/logs/create_proxy_catalog.log", encoding="utf-8"),
    ],
)


VIDEO_EXTENSIONS = {".mp4"}


@dataclass
class ProxyCatalogItem:
    name: str
    mp4_path: str
    size: int
    modified: float
    modified_iso: str
    indexed_at: str
    is_missing: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def load_json(path: Path) -> dict:
    if not path.exists():
        logging.info(f"Catalog not found, create new: {path}")
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        logging.warning(f"Cannot read catalog, create new: {path} | {e}")
        return {}


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp.json")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    temp_path.replace(path)


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def make_file_item(file_path: Path) -> dict:
    stat = file_path.stat()
    now = datetime.now().isoformat()

    item = ProxyCatalogItem(
        name=file_path.stem,
        mp4_path=str(file_path),
        size=stat.st_size,
        modified=stat.st_mtime,
        modified_iso=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        indexed_at=now,
        is_missing=False,
    )

    return item.to_dict()


def scan_root_folder(
    root_folder: Path,
    stats: ScanStatistics,
    skip_folders: set[str] | None = None,
) -> tuple[dict, dict]:
    found_catalog = {}
    duplicates = {}

    skip_lower = {name.lower() for name in (skip_folders or set())}

    if not root_folder.exists():
        logging.error(f"Root folder does not exist: {root_folder}")
        stats.errors += 1
        return found_catalog, duplicates

    for child in sorted(root_folder.iterdir()):
        if not child.is_dir():
            continue

        if child.name.lower() in skip_lower:
            logging.info(f"Skip folder: {child}")
            continue

        stats.folders_scanned += 1
        logging.info(f"Scanning: {child}")

        for file_path in child.rglob("*"):
            try:
                if not file_path.is_file():
                    continue

                stats.files_scanned += 1

                if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    if stats.should_report():
                        stats.report_progress()
                    continue

                stats.matched_files += 1

                key = file_path.stem
                item = make_file_item(file_path)

                if key in found_catalog:
                    duplicates.setdefault(key, [found_catalog[key]]).append(item)
                    stats.duplicates += 1

                    logging.warning(f"Duplicate MP4 name: {key}")
                    logging.warning(f"  old: {found_catalog[key]['mp4_path']}")
                    logging.warning(f"  new: {item['mp4_path']}")

                    if stats.should_report():
                        stats.report_progress()

                    continue

                found_catalog[key] = item

                if stats.should_report():
                    stats.report_progress()

            except Exception as e:
                stats.errors += 1
                logging.warning(f"Cannot process file: {file_path} | {e}")

    return found_catalog, duplicates


def merge_catalog(
    old_catalog: dict,
    found_catalog: dict,
    stats: ScanStatistics,
) -> dict:
    merged = dict(old_catalog)

    found_paths = {
        item["mp4_path"]
        for item in found_catalog.values()
    }

    for key, new_item in found_catalog.items():
        old_item = merged.get(key)

        if old_item is None:
            merged[key] = new_item
            stats.added += 1
            continue

        old_path = old_item.get("mp4_path")
        old_size = old_item.get("size")
        old_modified = old_item.get("modified")

        if (
            old_path == new_item["mp4_path"]
            and old_size == new_item["size"]
            and old_modified == new_item["modified"]
        ):
            old_item["is_missing"] = False
            stats.unchanged += 1
            continue

        merged[key] = new_item
        stats.updated += 1

    for item in merged.values():
        if item.get("mp4_path") not in found_paths:
            item["is_missing"] = True
            stats.missing += 1

    return merged


def build_proxy_catalog(
    root_folder: Path,
    catalog_path: Path,
    duplicate_path: Path,
    name: str,
    skip_folders: set[str] | None = None,
):
    logging.info("--------------------------------")
    logging.info(f"Build catalog: {name}")
    logging.info(f"Root: {root_folder}")
    logging.info(f"Catalog: {catalog_path}")

    stats = ScanStatistics(
        name=name,
        progress_step=10_000,
    )

    old_catalog = load_json(catalog_path)

    found_catalog, duplicates = scan_root_folder(
        root_folder=root_folder,
        stats=stats,
        skip_folders=skip_folders,
    )

    merged_catalog = merge_catalog(
        old_catalog=old_catalog,
        found_catalog=found_catalog,
        stats=stats,
    )

    save_json(catalog_path, merged_catalog)
    save_json(duplicate_path, duplicates)

    logging.info(f"Found on disk: {len(found_catalog)}")
    logging.info(f"Catalog total: {len(merged_catalog)}")
    logging.info(f"Duplicates saved: {duplicate_path}")

    stats.report_summary()

    logging.info(f"Saved: {catalog_path}")


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Start: {start}")

    build_proxy_catalog(
        name="proxy_intvnews",
        root_folder=PROXY_NEWS_FOLDER,
        catalog_path=PROXY_INTVNEWS_CATALOG_PATH,
        duplicate_path=PROXY_INTVNEWS_DUPLICATE_FILE,
        skip_folders={"Digital", *GARBAGE_FOLDER_NAMES},
    )

    build_proxy_catalog(
        name="proxy_digital",
        root_folder=PROXY_DIGITAL_FOLDER,
        catalog_path=PROXY_DIGITAL_CATALOG_PATH,
        duplicate_path=PROXY_DIGITAL_DUPLICATE_FILE,
    )

    finish = datetime.now()
    logging.info("--------------------------------")
    logging.info(f"Finished. Running time: {finish - start}")
