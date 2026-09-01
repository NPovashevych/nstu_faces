import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from services.config import (
    HIRES_NEWS_FOLDER,
    HIRES_DIGITAL_FOLDER,
    HIRES_INTVNEWS_CATALOG_PATH,
    HIRES_DIGITAL_CATALOG_PATH,
    HIRES_INTVNEWS_DUPLICATE_FILE,
    HIRES_DIGITAL_DUPLICATE_FILE,
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
        logging.FileHandler("../services/logs/create_hires_catalog_test.log", encoding="utf-8"),
    ],
)


VIDEO_EXTENSIONS = {".mxf"}


@dataclass
class HiresCatalogItem:
    name: str
    mxf_path: str
    size: int
    modified: float
    modified_iso: str
    indexed_at: str
    is_missing: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def load_json(path: Path) -> dict:
    if not path.exists():
        logging.info(f"JSON not found: {path}")
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Cannot read JSON: {path} | {e}")
        return {}


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp.json")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    temp_path.replace(path)


def make_file_item(file_path: Path) -> dict:
    stat = file_path.stat()
    now = datetime.now().isoformat()

    item = HiresCatalogItem(
        name=file_path.stem,
        mxf_path=str(file_path),
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

                    logging.warning(f"Duplicate MXF name: {key}")
                    logging.warning(f"  old: {found_catalog[key]['mxf_path']}")
                    logging.warning(f"  new: {item['mxf_path']}")

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


def get_count_by_name(catalog: dict, duplicates: dict) -> dict[str, int]:
    counts = {}

    for name in catalog:
        counts[name] = 1

    for name, items in duplicates.items():
        counts[name] = len(items)

    return counts


def compare_mxf_mp4(
    hires_catalog: dict,
    hires_duplicates: dict,
    proxy_catalog: dict,
    proxy_duplicates: dict,
    name: str,
):
    mxf_counts = get_count_by_name(hires_catalog, hires_duplicates)
    mp4_counts = get_count_by_name(proxy_catalog, proxy_duplicates)

    one_mxf_one_mp4 = 0
    one_mxf_many_mp4 = 0
    many_mxf_one_mp4 = 0
    many_mxf_many_mp4 = 0

    one_mxf_no_mp4 = 0
    many_mxf_no_mp4 = 0
    mp4_without_mxf = 0

    for key, mxf_count in mxf_counts.items():
        mp4_count = mp4_counts.get(key, 0)

        if mxf_count == 1 and mp4_count == 1:
            one_mxf_one_mp4 += 1
        elif mxf_count == 1 and mp4_count >= 2:
            one_mxf_many_mp4 += 1
        elif mxf_count >= 2 and mp4_count == 1:
            many_mxf_one_mp4 += 1
        elif mxf_count >= 2 and mp4_count >= 2:
            many_mxf_many_mp4 += 1
        elif mxf_count == 1 and mp4_count == 0:
            one_mxf_no_mp4 += 1
        elif mxf_count >= 2 and mp4_count == 0:
            many_mxf_no_mp4 += 1

    for key in mp4_counts:
        if key not in mxf_counts:
            mp4_without_mxf += 1

    logging.info("--------------------------------")
    logging.info(f"[{name}] MXF ↔ MP4 comparison")
    logging.info(f"MXF names total: {len(mxf_counts)}")
    logging.info(f"MP4 names total: {len(mp4_counts)}")
    logging.info(f"1 MXF + 1 MP4: {one_mxf_one_mp4}")
    logging.info(f"1 MXF + 2+ MP4: {one_mxf_many_mp4}")
    logging.info(f"2+ MXF + 1 MP4: {many_mxf_one_mp4}")
    logging.info(f"2+ MXF + 2+ MP4: {many_mxf_many_mp4}")
    logging.info(f"1 MXF + 0 MP4: {one_mxf_no_mp4}")
    logging.info(f"2+ MXF + 0 MP4: {many_mxf_no_mp4}")
    logging.info(f"MP4 without MXF: {mp4_without_mxf}")


def build_hires_catalog(
    name: str,
    root_folder: Path,
    catalog_path: Path,
    duplicate_path: Path,
    proxy_catalog_path: Path,
    proxy_duplicate_path: Path,
    skip_folders: set[str] | None = None,
):
    logging.info("--------------------------------")
    logging.info(f"Build HIRES catalog: {name}")
    logging.info(f"Root: {root_folder}")
    logging.info(f"Catalog: {catalog_path}")

    stats = ScanStatistics(
        name=name,
        progress_step=10_000,
    )

    hires_catalog, hires_duplicates = scan_root_folder(
        root_folder=root_folder,
        stats=stats,
        skip_folders=skip_folders,
    )

    save_json(catalog_path, hires_catalog)
    save_json(duplicate_path, hires_duplicates)

    proxy_catalog = load_json(proxy_catalog_path)
    proxy_duplicates = load_json(proxy_duplicate_path)

    logging.info(f"HIRES found on disk: {len(hires_catalog)}")
    logging.info(f"HIRES duplicates: {len(hires_duplicates)}")
    logging.info(f"HIRES saved: {catalog_path}")
    logging.info(f"HIRES duplicates saved: {duplicate_path}")

    stats.report_summary()

    compare_mxf_mp4(
        hires_catalog=hires_catalog,
        hires_duplicates=hires_duplicates,
        proxy_catalog=proxy_catalog,
        proxy_duplicates=proxy_duplicates,
        name=name,
    )


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Start: {start}")

    build_hires_catalog(
        name="hires_intvnews",
        root_folder=HIRES_NEWS_FOLDER,
        catalog_path=HIRES_INTVNEWS_CATALOG_PATH,
        duplicate_path=HIRES_INTVNEWS_DUPLICATE_FILE,
        proxy_catalog_path=PROXY_INTVNEWS_CATALOG_PATH,
        proxy_duplicate_path=PROXY_INTVNEWS_DUPLICATE_FILE,
        skip_folders={"Digital", *GARBAGE_FOLDER_NAMES},
    )

    build_hires_catalog(
        name="hires_digital",
        root_folder=HIRES_DIGITAL_FOLDER,
        catalog_path=HIRES_DIGITAL_CATALOG_PATH,
        duplicate_path=HIRES_DIGITAL_DUPLICATE_FILE,
        proxy_catalog_path=PROXY_DIGITAL_CATALOG_PATH,
        proxy_duplicate_path=PROXY_DIGITAL_DUPLICATE_FILE,
    )

    finish = datetime.now()
    logging.info("--------------------------------")
    logging.info(f"Finished. Running time: {finish - start}")
