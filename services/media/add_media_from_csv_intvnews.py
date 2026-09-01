import csv
import json
import logging
import multiprocessing
import re
import sys

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Iterator

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.enums import MediaType
from db.models import DBFreeze, DBMedia
from db.session import SessionLocal
from schemas.schemas_freeze import FreezeCreate
from schemas.schemas_media import MediaCreate
from services.config import CSV_FOLDER, HIRES_INTVNEWS_CATALOG, INTVNEWS_FREEZE_FOLDER, PROXY_INTVNEWS_CATALOG, TEST_MP4_LIGHT_FOLDER
from services.media.common_media_func import create_light_mp4, extract_freeze_from_video, get_video_duration_and_scenes


SOURCE_ID = 2
USER_ID = 1

MAX_WORKERS = 4
MAX_PENDING_TASKS = MAX_WORKERS * 3

MAX_CONSECUTIVE_FREEZE_FAILURES = 1

LOG_PATH = Path("../logs/add_archive_media.log")

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251")

IGNORED_CSV_NAMES = {"NOT_FOUND.csv"}

# CSV_FILES_FOR_TEST = None

# Для тесту одного CSV:
# CSV_FILES_FOR_TEST = "materials_2016-07-01-2016-08-01.csv"

# Для тесту кількох CSV:
CSV_FILES_FOR_TEST = (
    "materials_2022-08-01-2022-09-01.csv",
)


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] [%(processName)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
        force=True,
    )


def safe_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    return value.strip("._") or "unnamed"


def format_time_for_filename(seconds: float) -> str:
    total_milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}-{milliseconds:03d}"


def load_json(path: Path) -> dict:
    if not path.exists():
        logging.warning(f"JSON catalog file not found: {path}")
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            logging.error(f"Invalid catalog format: {path}. Expected dict, received {type(data).__name__}")
            return {}

        return data

    except Exception:
        logging.exception(f"Cannot read JSON catalog: {path}")
        return {}


def load_catalog_folder(folder: Path, path_field: str) -> dict[str, dict]:
    catalog: dict[str, dict] = {}

    if not folder.exists():
        logging.error(f"Catalog folder not found: {folder}")
        return catalog

    json_files = sorted(folder.glob("*.json"))

    logging.info(f"Loading catalog: {folder} | JSON files: {len(json_files)}")

    for json_path in json_files:
        year_catalog = load_json(json_path)

        for raw_material_id, item in year_catalog.items():
            material_id = str(raw_material_id).strip()

            if not material_id:
                continue

            if not isinstance(item, dict):
                logging.warning(f"Invalid catalog item | file={json_path.name} | material_id={material_id}")
                continue

            file_path = item.get(path_field)

            if not file_path:
                logging.warning(f"Catalog item without {path_field} | file={json_path.name} | material_id={material_id}")
                continue

            if material_id in catalog:
                previous_path = catalog[material_id].get(path_field)

                logging.warning(
                    f"DUPLICATE MATERIAL IN CATALOG | material_id={material_id} | first_path={previous_path} | "
                    f"duplicate_path={file_path} | duplicate_file={json_path.name}"
                )
                continue

            catalog[material_id] = item

    logging.info(f"Catalog loaded: {folder} | items={len(catalog)}")

    return catalog


def resolve_csv_paths(csv_folder: Path) -> list[Path]:
    if CSV_FILES_FOR_TEST is None:
        csv_paths = sorted(path for path in csv_folder.glob("*.csv") if path.name not in IGNORED_CSV_NAMES)
        logging.info(f"CSV mode: FULL FOLDER | folder={csv_folder} | files={len(csv_paths)}")
        return csv_paths

    if isinstance(CSV_FILES_FOR_TEST, str):
        selected_names = (CSV_FILES_FOR_TEST,)
    else:
        selected_names = tuple(CSV_FILES_FOR_TEST)

    csv_paths: list[Path] = []

    for file_name in selected_names:
        csv_path = csv_folder / file_name

        if not csv_path.exists():
            logging.error(f"Selected CSV file not found: {csv_path}")
            continue

        if csv_path.name in IGNORED_CSV_NAMES:
            logging.warning(f"Selected CSV is ignored: {csv_path}")
            continue

        csv_paths.append(csv_path)

    logging.info(f"CSV mode: TEST SELECTION | requested={len(selected_names)} | existing={len(csv_paths)}")

    return csv_paths


def detect_csv_encoding(csv_path: Path) -> str:
    last_error: Exception | None = None

    for encoding in CSV_ENCODINGS:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as file:
                file.read(8192)

            return encoding

        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(f"Cannot detect CSV encoding: {csv_path}. Last error: {last_error}")


def iter_unique_material_ids(csv_folder: Path) -> Iterator[tuple[str, str, int]]:
    seen_material_ids: dict[str, tuple[str, int]] = {}
    csv_paths = resolve_csv_paths(csv_folder)

    total_rows = 0
    empty_material_ids = 0
    duplicate_material_ids = 0

    for csv_path in csv_paths:
        try:
            encoding = detect_csv_encoding(csv_path)

            logging.info(f"Reading CSV: {csv_path.name} | encoding={encoding}")

            with csv_path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file, delimiter=";")

                if not reader.fieldnames:
                    logging.warning(f"CSV without header: {csv_path}")
                    continue

                normalized_fieldnames = [str(field).strip() for field in reader.fieldnames if field is not None]

                if "material_id" not in normalized_fieldnames:
                    logging.warning(f"CSV without material_id column: {csv_path} | columns={reader.fieldnames}")
                    continue

                for row_number, row in enumerate(reader, start=2):
                    total_rows += 1

                    material_id = str(row.get("material_id") or "").strip()

                    if not material_id:
                        empty_material_ids += 1
                        continue

                    first_source = seen_material_ids.get(material_id)

                    if first_source is not None:
                        duplicate_material_ids += 1
                        first_file, first_row = first_source

                        logging.warning(
                            f"CSV DUPLICATE SKIP | material_id={material_id} | first={first_file}:{first_row} | "
                            f"duplicate={csv_path.name}:{row_number}"
                        )
                        continue

                    seen_material_ids[material_id] = (csv_path.name, row_number)

                    yield material_id, csv_path.name, row_number

        except Exception:
            logging.exception(f"Cannot process CSV: {csv_path}")

    logging.info(
        f"CSV reading finished | rows={total_rows} | unique_material_ids={len(seen_material_ids)} | "
        f"duplicates={duplicate_material_ids} | empty_material_ids={empty_material_ids}"
    )


def create_media_without_commit(db: Session, media: MediaCreate) -> DBMedia:
    db_media = DBMedia(
        material_id=media.material_id,
        media_type=media.media_type,
        mxf_path=media.mxf_path,
        mp4_path=media.mp4_path,
        duration=media.duration,
        recorded_at=media.recorded_at,
        source_id=media.source_id,
        user_id=media.user_id,
    )

    db.add(db_media)
    db.flush()

    return db_media


def get_or_create_freeze_without_commit(db: Session, freeze: FreezeCreate) -> tuple[DBFreeze, bool]:
    existing = db.query(DBFreeze).filter(DBFreeze.freeze_path == freeze.freeze_path).first()

    if existing:
        return existing, False

    db_freeze = DBFreeze(
        time_in=freeze.time_in,
        time_out=freeze.time_out,
        media_id=freeze.media_id,
        freeze_path=freeze.freeze_path,
    )

    db.add(db_freeze)
    db.flush()

    return db_freeze, True


def get_existing_archive_media(db: Session, material_id: str) -> DBMedia | None:
    return db.query(DBMedia).filter(DBMedia.material_id == material_id, DBMedia.source_id == SOURCE_ID).first()


def resolve_or_create_mp4(material_id: str, mxf_path: Path, proxy_path_value: str | None) -> tuple[Path | None, bool, str]:
    if proxy_path_value:
        proxy_path = Path(proxy_path_value)

        if proxy_path.exists():
            return proxy_path, False, ""

        reason = f"Proxy listed in catalog but missing on disk: {proxy_path}"
    else:
        reason = "Proxy not found in catalog"

    local_mp4_path = Path(TEST_MP4_LIGHT_FOLDER) / f"{safe_name(material_id)}.mp4"

    if local_mp4_path.exists():
        return local_mp4_path, False, f"{reason}. Existing local MP4 used: {local_mp4_path}"

    local_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    created = create_light_mp4(mxf_path=mxf_path, mp4_path=local_mp4_path)

    if not created:
        return None, False, f"{reason}. Local MP4 creation failed: {local_mp4_path}"

    return local_mp4_path, True, f"{reason}. Local MP4 created: {local_mp4_path}"


def build_task(material_id, csv_file, csv_row, hires_catalog: dict[str, dict], proxy_catalog: dict[str, dict]) -> tuple[str, str, str | None, str, int] | None:
    hires_item = hires_catalog.get(material_id)

    if hires_item is None:
        logging.warning(f"MXF NOT FOUND IN CATALOG | material_id={material_id} | csv={csv_file}:{csv_row}")
        return None

    mxf_path_value = hires_item.get("mxf_path")

    if not mxf_path_value:
        logging.warning(f"MXF PATH EMPTY IN CATALOG | material_id={material_id} | csv={csv_file}:{csv_row}")
        return None

    proxy_item = proxy_catalog.get(material_id)
    proxy_path_value = proxy_item.get("mp4_path") if proxy_item else None

    return material_id, str(mxf_path_value), str(proxy_path_value) if proxy_path_value else None, csv_file, csv_row


def process_one_material(task: tuple[str, str, str | None, str, int]) -> dict:
    material_id, mxf_path_value, proxy_path_value, csv_file, csv_row = task

    db = SessionLocal()

    result = {
        "material_id": material_id,
        "csv_file": csv_file,
        "csv_row": csv_row,
        "status": "error",
        "media_id": None,
        "media_created": False,
        "freezes_created": 0,
        "local_mp4_created": False,
        "message": "",
        "warnings": [],
    }

    try:
        existing_archive_media = get_existing_archive_media(db=db, material_id=material_id)

        if existing_archive_media:
            result["status"] = "already_exists"
            result["media_id"] = existing_archive_media.id
            result["message"] = f"Archive media already exists for material_id={material_id}, source_id={SOURCE_ID}"
            return result

        mxf_path = Path(mxf_path_value)

        if not mxf_path.exists():
            result["status"] = "skip"
            result["message"] = f"MXF listed in catalog but missing on disk: {mxf_path}"
            return result

        if mxf_path.stat().st_size == 0:
            result["status"] = "skip"
            result["message"] = f"MXF file is empty: {mxf_path}"
            return result

        mp4_path, local_mp4_created, mp4_note = resolve_or_create_mp4(
            material_id=material_id,
            mxf_path=mxf_path,
            proxy_path_value=proxy_path_value,
        )

        result["local_mp4_created"] = local_mp4_created

        if mp4_note:
            result["warnings"].append(mp4_note)

        if mp4_path is None:
            result["status"] = "error"
            result["message"] = "Cannot resolve or create MP4"
            return result

        duration, scenes = get_video_duration_and_scenes(mp4_path)

        media_create = MediaCreate(
            material_id=material_id,
            media_type=MediaType.video,
            mxf_path=str(mxf_path),
            mp4_path=str(mp4_path),
            duration=duration,
            recorded_at=datetime.now(),
            source_id=SOURCE_ID,
            user_id=USER_ID,
        )

        media = create_media_without_commit(db=db, media=media_create)

        result["media_id"] = media.id

        media_folder_name = safe_name(material_id)
        freeze_folder = Path(INTVNEWS_FREEZE_FOLDER) / media_folder_name

        created_freezes = 0
        failed_freezes = 0
        consecutive_failed_freezes = 0
        freeze_extraction_aborted = False
        skipped_scenes_after_abort = 0

        for scene_index, (start, end) in enumerate(scenes):
            freeze_name = f"{media_folder_name}_{format_time_for_filename(start)}.jpg"
            freeze_path = freeze_folder / freeze_name

            if not freeze_path.exists():
                extracted = extract_freeze_from_video(
                    video_path=mxf_path,
                    freeze_path=freeze_path,
                    start_sec=start,
                )

                if not extracted:
                    failed_freezes += 1
                    consecutive_failed_freezes += 1

                    if consecutive_failed_freezes >= MAX_CONSECUTIVE_FREEZE_FAILURES:
                        freeze_extraction_aborted = True
                        skipped_scenes_after_abort = len(scenes) - scene_index - 1
                        break

                    continue

            consecutive_failed_freezes = 0

            freeze_create = FreezeCreate(
                time_in=start,
                time_out=end,
                media_id=media.id,
                freeze_path=str(freeze_path),
            )

            _, created = get_or_create_freeze_without_commit(db=db, freeze=freeze_create)

            if created:
                created_freezes += 1

        db.commit()

        result["media_created"] = True
        result["freezes_created"] = created_freezes
        result["status"] = "ok"
        result["message"] = (f"duration={duration:.2f} | scenes={len(scenes)} | freeze_failures={failed_freezes}")

        if freeze_extraction_aborted:
            result["warnings"].append(
                f"FREEZE EXTRACTION ABORTED | consecutive_failures={consecutive_failed_freezes} | "
                f"failed_freezes={failed_freezes} | skipped_scenes={skipped_scenes_after_abort} | "
                f"mxf_path={mxf_path}"
            )
        elif failed_freezes:
            result["warnings"].append(f"Freeze extraction failures: {failed_freezes}")

        return result

    except IntegrityError as error:
        db.rollback()

        result["media_id"] = None
        result["media_created"] = False
        result["status"] = "error"
        result["message"] = f"IntegrityError: {error.orig or error}"
        return result

    except Exception as error:
        db.rollback()

        result["media_id"] = None
        result["media_created"] = False
        result["status"] = "error"
        result["message"] = f"{type(error).__name__}: {error}"
        return result

    finally:
        db.close()


def log_worker_result(result: dict) -> None:
    prefix = (
        f"material_id={result['material_id']} | csv={result['csv_file']}:{result['csv_row']} | media_id={result['media_id']} | "
        f"media_created={result['media_created']} | freezes_created={result['freezes_created']} | local_mp4_created={result['local_mp4_created']}"
    )

    if result["status"] == "ok":
        logging.info(f"OK | {prefix} | {result['message']}")

    elif result["status"] == "already_exists":
        logging.info(f"ALREADY EXISTS | {prefix} | {result['message']}")

    elif result["status"] == "skip":
        logging.warning(f"SKIP | {prefix} | {result['message']}")

    else:
        logging.error(f"ERROR | {prefix} | {result['message']}")

    for warning in result.get("warnings", []):
        logging.warning(f"EVENT | material_id={result['material_id']} | {warning}")


def collect_result(result: dict, counters: dict[str, int]) -> None:
    log_worker_result(result)

    counters["completed"] += 1

    if result["status"] == "ok":
        counters["ok"] += 1

    elif result["status"] == "already_exists":
        counters["already_exists"] += 1

    elif result["status"] == "skip":
        counters["skipped"] += 1

    else:
        counters["errors"] += 1

    if result.get("media_created"):
        counters["media_created"] += 1

    counters["freezes_created"] += result.get("freezes_created", 0)

    if result.get("local_mp4_created"):
        counters["local_mp4_created"] += 1


def process_archive() -> None:
    started_at = datetime.now()

    logging.info("--------------------------------------------------")
    logging.info(f"Started: {started_at}")
    logging.info(f"Workers: {MAX_WORKERS}")
    logging.info(f"Maximum pending tasks: {MAX_PENDING_TASKS}")
    logging.info(f"Maximum consecutive freeze failures: {MAX_CONSECUTIVE_FREEZE_FAILURES}")
    logging.info(f"CSV folder: {CSV_FOLDER}")
    logging.info(f"Hires catalog folder: {HIRES_INTVNEWS_CATALOG}")
    logging.info(f"Proxy catalog folder: {PROXY_INTVNEWS_CATALOG}")
    logging.info(f"Source ID: {SOURCE_ID}")
    logging.info(f"User ID: {USER_ID}")

    hires_catalog = load_catalog_folder(folder=Path(HIRES_INTVNEWS_CATALOG), path_field="mxf_path")
    proxy_catalog = load_catalog_folder(folder=Path(PROXY_INTVNEWS_CATALOG), path_field="mp4_path")

    if not hires_catalog:
        logging.error("Hires catalog is empty. Processing stopped.")
        return

    counters = {
        "submitted": 0,
        "completed": 0,
        "ok": 0,
        "already_exists": 0,
        "skipped": 0,
        "errors": 0,
        "media_created": 0,
        "freezes_created": 0,
        "local_mp4_created": 0,
        "mxf_not_found_in_catalog": 0,
    }

    pending = set()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for material_id, csv_file, csv_row in iter_unique_material_ids(Path(CSV_FOLDER)):
            task = build_task(
                material_id=material_id,
                csv_file=csv_file,
                csv_row=csv_row,
                hires_catalog=hires_catalog,
                proxy_catalog=proxy_catalog,
            )

            if task is None:
                counters["mxf_not_found_in_catalog"] += 1
                continue

            future = executor.submit(process_one_material, task)

            pending.add(future)
            counters["submitted"] += 1

            if len(pending) >= MAX_PENDING_TASKS:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)

                for finished_future in done:
                    try:
                        result = finished_future.result()
                        collect_result(result=result, counters=counters)

                    except Exception:
                        counters["completed"] += 1
                        counters["errors"] += 1
                        logging.exception("Worker crashed unexpectedly")

                    if counters["completed"] % 100 == 0:
                        logging.info(
                            f"PROGRESS | submitted={counters['submitted']} | completed={counters['completed']} | pending={len(pending)} | "
                            f"ok={counters['ok']} | already_exists={counters['already_exists']} | skipped={counters['skipped']} | errors={counters['errors']}"
                        )

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)

            for finished_future in done:
                try:
                    result = finished_future.result()
                    collect_result(result=result, counters=counters)

                except Exception:
                    counters["completed"] += 1
                    counters["errors"] += 1
                    logging.exception("Worker crashed unexpectedly")

                if counters["completed"] % 100 == 0:
                    logging.info(
                        f"PROGRESS | submitted={counters['submitted']} | completed={counters['completed']} | pending={len(pending)} | "
                        f"ok={counters['ok']} | already_exists={counters['already_exists']} | skipped={counters['skipped']} | errors={counters['errors']}"
                    )

    finished_at = datetime.now()

    logging.info("--------------------------------------------------")
    logging.info("SUMMARY")
    logging.info(f"Submitted to workers: {counters['submitted']}")
    logging.info(f"Completed: {counters['completed']}")
    logging.info(f"Successfully processed: {counters['ok']}")
    logging.info(f"Already existed for source_id={SOURCE_ID}: {counters['already_exists']}")
    logging.info(f"Skipped by workers: {counters['skipped']}")
    logging.info(f"Errors: {counters['errors']}")
    logging.info(f"MXF not found in catalog: {counters['mxf_not_found_in_catalog']}")
    logging.info(f"Created media records: {counters['media_created']}")
    logging.info(f"Created freeze records: {counters['freezes_created']}")
    logging.info(f"Created local MP4 files: {counters['local_mp4_created']}")
    logging.info(f"Started: {started_at}")
    logging.info(f"Finished: {finished_at}")
    logging.info(f"Running time: {finished_at - started_at}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    setup_logging()
    process_archive()
