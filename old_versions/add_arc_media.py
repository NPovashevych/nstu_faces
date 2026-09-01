import csv
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy.exc import IntegrityError

from db.session import SessionLocal
from db.enums import MediaType

from crud.crud_media import get_media_by_mxf_path, create_media
from crud.crud_media_description import (
    get_media_descriptions_by_material_id,
    create_media_description,
)

from schemas.schemas_media import MediaCreate
from schemas.schemas_media_description import MediaDescriptionCreate

from services.config import (
    CSV_FOLDER,
    HIRES_INTVNEWS_DUPLICATE_FILE,
    PROXY_INTVNEWS_CATALOG_PATH,
    PROXY_INTVNEWS_DUPLICATE_FILE,
    INTVNEWS_FREEZE_FOLDER,
)

from services.commons.scan_statistics import ScanStatistics

from services.media.common_media_func import (
    safe_name,
    get_video_duration_and_scenes,
    extract_freeze_from_video,
    format_time_for_filename,
    get_or_create_freeze,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../services/logs/add_arc_media.log", encoding="utf-8"),
    ],
)


SOURCE_ID = 2
USER_ID = 1

ARC_TEST_FOLDER = Path("V:/2016/02")
NOT_FOUND_CSV_PATH = CSV_FOLDER / "NOT_FOUND.csv"


def load_json(path: Path) -> dict:
    if not path.exists():
        logging.warning(f"JSON not found: {path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_description_hash(row: dict) -> str:
    raw = "|".join(
        [
            row.get("material_id", "") or "",
            row.get("section", "") or "",
            row.get("shooting_date", "") or "",
            row.get("journalists", "") or "",
            row.get("operators", "") or "",
            row.get("description", "") or "",
        ]
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_csv_descriptions(csv_folder: Path) -> dict:
    descriptions = {}

    for csv_path in sorted(csv_folder.glob("*.csv")):
        if csv_path.name == "NOT_FOUND.csv":
            continue

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")

            for row in reader:
                material_id = (row.get("material_id") or "").strip()

                if not material_id:
                    continue

                if material_id in descriptions:
                    logging.warning(f"Duplicate material_id in CSV: {material_id}")
                    continue

                row["_source_path"] = str(csv_path)
                row["_source_hash"] = make_description_hash(row)

                descriptions[material_id] = row

    logging.info(f"CSV descriptions loaded: {len(descriptions)}")
    return descriptions


def get_arc_mxf_files(folder: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in folder.rglob("*.mxf")
        if file_path.is_file()
    )


def get_or_create_arc_media(
    db,
    mxf_path: Path,
    mp4_path: Path,
    duration: float,
):
    existing = get_media_by_mxf_path(db, str(mxf_path))

    if existing:
        return existing, False

    media_create = MediaCreate(
        source_id=SOURCE_ID,
        user_id=USER_ID,
        media_type=MediaType.video,
        mxf_path=str(mxf_path),
        mp4_path=str(mp4_path),
        duration=duration,
        recorded_at=datetime.now(),
    )

    try:
        media = create_media(db, media_create)
        return media, True

    except IntegrityError:
        db.rollback()
        media = get_media_by_mxf_path(db, str(mxf_path))
        return media, False


def get_or_create_description(
    db,
    material_id: str,
    csv_row: dict | None,
):
    existing_rows = get_media_descriptions_by_material_id(db, material_id)

    if existing_rows:
        return existing_rows[0], False

    if csv_row:
        description_create = MediaDescriptionCreate(
            material_id=material_id,
            section=csv_row.get("section") or None,
            shooting_date=csv_row.get("shooting_date") or "",
            journalist=csv_row.get("journalists") or None,
            operators=csv_row.get("operators") or None,
            description=csv_row.get("description") or None,
            another_info=None,
            source_path=csv_row["_source_path"],
            source_hash=csv_row.get("_source_hash"),
        )
    else:
        description_create = MediaDescriptionCreate(
            material_id=material_id,
            section=None,
            shooting_date="",
            journalist=None,
            operators=None,
            description=None,
            another_info=None,
            source_path=str(NOT_FOUND_CSV_PATH),
            source_hash=None,
        )

    try:
        description = create_media_description(db, description_create)
        return description, True

    except IntegrityError:
        db.rollback()
        existing_rows = get_media_descriptions_by_material_id(db, material_id)
        return existing_rows[0] if existing_rows else None, False


def process_one_arc_mxf(
    db,
    mxf_path: Path,
    proxy_catalog: dict,
    hires_duplicates: dict,
    proxy_duplicates: dict,
    descriptions: dict,
) -> dict:
    material_id = mxf_path.stem
    media_name = safe_name(material_id)

    if material_id in hires_duplicates:
        return {
            "file": mxf_path.name,
            "status": "skip",
            "media_id": None,
            "freezes": 0,
            "error": "hires_duplicate",
        }

    if material_id in proxy_duplicates:
        return {
            "file": mxf_path.name,
            "status": "skip",
            "media_id": None,
            "freezes": 0,
            "error": "proxy_duplicate",
        }

    proxy_item = proxy_catalog.get(material_id)

    if not proxy_item:
        return {
            "file": mxf_path.name,
            "status": "skip",
            "media_id": None,
            "freezes": 0,
            "error": "no_proxy_mp4",
        }

    mp4_path = Path(proxy_item["mp4_path"])

    if not mp4_path.exists():
        return {
            "file": mxf_path.name,
            "status": "skip",
            "media_id": None,
            "freezes": 0,
            "error": "proxy_mp4_missing_on_disk",
        }

    duration, scenes = get_video_duration_and_scenes(mp4_path)

    media, media_created = get_or_create_arc_media(
        db=db,
        mxf_path=mxf_path,
        mp4_path=mp4_path,
        duration=duration,
    )

    if media is None:
        return {
            "file": mxf_path.name,
            "status": "error",
            "media_id": None,
            "freezes": 0,
            "error": "media_create_failed",
        }

    freeze_folder = Path(INTVNEWS_FREEZE_FOLDER) / media_name
    created_freezes = 0

    for start, end in scenes:
        time_part = format_time_for_filename(start)
        freeze_name = f"{media_name}_{time_part}.jpg"
        freeze_path = freeze_folder / freeze_name

        if not freeze_path.exists():
            ok = extract_freeze_from_video(
                video_path=mxf_path,
                freeze_path=freeze_path,
                start_sec=start,
            )

            if not ok:
                continue

        _, created = get_or_create_freeze(
            db=db,
            media_id=media.id,
            freeze_path=freeze_path,
            time_in=start,
            time_out=end,
        )

        if created:
            created_freezes += 1

    csv_row = descriptions.get(material_id)

    _, description_created = get_or_create_description(
        db=db,
        material_id=material_id,
        csv_row=csv_row,
    )

    return {
        "file": mxf_path.name,
        "status": "ok",
        "media_id": media.id,
        "freezes": created_freezes,
        "error": "",
        "media_created": media_created,
        "description_created": description_created,
        "csv_found": csv_row is not None,
    }


def process_arc_test_folder():
    db = SessionLocal()

    total = 0
    ok_count = 0
    skipped = 0
    errors = 0

    total_freezes = 0
    csv_found = 0
    csv_not_found = 0

    media_created = 0
    description_created = 0

    stats = ScanStatistics(
        name="add_arc_media_intvnews_2016_02",
        progress_step=100,
    )

    try:
        proxy_catalog = load_json(PROXY_INTVNEWS_CATALOG_PATH)
        hires_duplicates = load_json(HIRES_INTVNEWS_DUPLICATE_FILE)
        proxy_duplicates = load_json(PROXY_INTVNEWS_DUPLICATE_FILE)
        descriptions = load_csv_descriptions(CSV_FOLDER)

        mxf_files = get_arc_mxf_files(ARC_TEST_FOLDER)

        logging.info(f"ARC test folder: {ARC_TEST_FOLDER}")
        logging.info(f"MXF files found: {len(mxf_files)}")

        for mxf_path in mxf_files:
            total += 1
            stats.files_scanned += 1

            try:
                result = process_one_arc_mxf(
                    db=db,
                    mxf_path=mxf_path,
                    proxy_catalog=proxy_catalog,
                    hires_duplicates=hires_duplicates,
                    proxy_duplicates=proxy_duplicates,
                    descriptions=descriptions,
                )

                logging.info(
                    f"{result['status'].upper()}: "
                    f"{result['file']} | "
                    f"media_id={result['media_id']} | "
                    f"freezes={result['freezes']} | "
                    f"{result['error']}"
                )

                total_freezes += result["freezes"]

                if result["status"] == "ok":
                    ok_count += 1
                    stats.matched_files += 1

                    if result.get("media_created"):
                        media_created += 1

                    if result.get("description_created"):
                        description_created += 1

                    if result.get("csv_found"):
                        csv_found += 1
                    else:
                        csv_not_found += 1

                elif result["status"] == "skip":
                    skipped += 1

                else:
                    errors += 1

                if stats.should_report():
                    stats.report_progress()

            except Exception as e:
                errors += 1
                stats.errors += 1
                logging.exception(f"Unexpected error for {mxf_path}: {e}")

        logging.info("--------------------------------")
        logging.info(f"Processed: {total}")
        logging.info(f"OK: {ok_count}")
        logging.info(f"Skipped: {skipped}")
        logging.info(f"Errors: {errors}")
        logging.info(f"Created media: {media_created}")
        logging.info(f"Created freezes: {total_freezes}")
        logging.info(f"Created descriptions: {description_created}")
        logging.info(f"CSV found: {csv_found}")
        logging.info(f"CSV not found: {csv_not_found}")

        stats.report_summary()

    finally:
        db.close()


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Start: {start}")

    process_arc_test_folder()

    finish = datetime.now()
    logging.info(f"Finished. Running time: {finish - start}")
