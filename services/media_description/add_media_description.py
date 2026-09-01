import csv
import hashlib
import json
import logging
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import DBMediaDescription
from db.session import SessionLocal
from services.config import CSV_FOLDER


LOG_PATH = Path("../logs/import_media_descriptions.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)


BATCH_SIZE = 5_000
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251")
HASH_FIELDS = ("material_id", "section", "shooting_date", "journalist", "operators", "description")


def clean_value(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    return value


def normalize_material_id(value: str | None) -> str:
    value = clean_value(value)

    if value is None:
        return ""

    return value


def make_source_hash(data: dict) -> str:
    # Для кожного запису з firebird окримий інстанс, хоча для ідентичних буде ідентичний хеш
    hash_data = {field: data.get(field) for field in HASH_FIELDS}
    payload = json.dumps(hash_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def detect_encoding(path: Path) -> str:
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                file.read(10_000)
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Cannot detect CSV encoding: {path}")


def parse_csv_row(row: dict, source_path: Path, row_number: int) -> dict | None:
    material_id = clean_value(row.get("material_id"))
    shooting_date = clean_value(row.get("shooting_date"))

    if not material_id:
        logging.warning(f"Skip row without material_id: {source_path} | row {row_number}")
        # logging.warning(f"material_id repr={repr(row.get('material_id'))}")
        # logging.warning(row)
        return None

    if not shooting_date:
        logging.warning(f"Skip row without shooting_date: {source_path} | row {row_number} | material_id={material_id}")
        return None

    data = {
        "material_id": material_id,
        "section": clean_value(row.get("section")),
        "shooting_date": shooting_date,
        "journalist": clean_value(row.get("journalists")),
        "operators": clean_value(row.get("operators")),
        "description": clean_value(row.get("description")),
        "another_info": None,
        "source_path": str(source_path),
    }

    data["source_hash"] = make_source_hash(data)

    return data


def load_existing_records(db: Session) -> dict[str, deque[int]]:
    # захист від ідентичних рядків
    existing_by_hash: dict[str, deque[int]] = defaultdict(deque)

    statement = select(
        DBMediaDescription.id,
        DBMediaDescription.material_id,
        DBMediaDescription.section,
        DBMediaDescription.shooting_date,
        DBMediaDescription.journalist,
        DBMediaDescription.operators,
        DBMediaDescription.description,
        DBMediaDescription.source_hash,
    )

    result = db.execute(statement)
    null_hash_updates = []

    for row in result:
        source_hash = row.source_hash

        if not source_hash:
            source_hash = make_source_hash(
                {
                    "material_id": row.material_id,
                    "section": row.section,
                    "shooting_date": row.shooting_date,
                    "journalist": row.journalist,
                    "operators": row.operators,
                    "description": row.description,
                }
            )

            null_hash_updates.append(
                {
                    "id": row.id,
                    "source_hash": source_hash,
                }
            )

        existing_by_hash[source_hash].append(row.id)

    if null_hash_updates:
        db.bulk_update_mappings(
            DBMediaDescription,
            null_hash_updates,
        )

        logging.info(
            f"Filled missing hashes: "
            f"{len(null_hash_updates)}"
        )

    return existing_by_hash


def flush_batches(db: Session, inserts: list[dict], updates: list[dict]) -> tuple[int, int]:
    inserted_count = len(inserts)
    updated_count = len(updates)

    if inserts:
        db.bulk_insert_mappings(DBMediaDescription, inserts)
        inserts.clear()

    if updates:
        db.bulk_update_mappings(DBMediaDescription, updates)
        updates.clear()

    db.flush()

    return inserted_count, updated_count


def import_media_descriptions(db: Session, csv_folder: Path) -> None:
    if not csv_folder.exists():
        raise FileNotFoundError(f"CSV folder does not exist: {csv_folder}")

    csv_paths = sorted(path for path in csv_folder.glob("*.csv") if path.is_file())

    if not csv_paths:
        raise FileNotFoundError(f"CSV files not found: {csv_folder}")

    logging.info(f"CSV folder: {csv_folder}")
    logging.info(f"CSV files: {len(csv_paths)}")

    existing_by_hash = load_existing_records(db)

    inserts: list[dict] = []
    updates: list[dict] = []

    rows_read, rows_skipped, inserted, updated = 0, 0, 0, 0

    try:
        for file_number, csv_path in enumerate(csv_paths, start=1):
            encoding = detect_encoding(csv_path)

            logging.info(f"[{file_number}/{len(csv_paths)}] Read: {csv_path.name} | encoding={encoding}")

            with csv_path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file, delimiter=";")

                for row_number, raw_row in enumerate(reader, start=2):
                    rows_read += 1
                    data = parse_csv_row(raw_row, csv_path, row_number)
                    if data is None:
                        rows_skipped += 1
                        continue

                    source_hash = data["source_hash"]
                    existing_ids = existing_by_hash.get(source_hash)

                    if existing_ids:
                        record_id = existing_ids.popleft()

                        updates.append(
                            {
                                "id": record_id,
                                "material_id": data["material_id"],
                                "section": data["section"],
                                "shooting_date": data["shooting_date"],
                                "journalist": data["journalist"],
                                "operators": data["operators"],
                                "description": data["description"],
                                "another_info": None,
                                "source_path": data["source_path"],
                                "source_hash": source_hash,
                                "updated_at": datetime.now(timezone.utc),
                            }
                        )

                    else:
                        inserts.append(data)

                    if len(inserts) + len(updates) >= BATCH_SIZE:
                        batch_inserted, batch_updated = flush_batches(db=db, inserts=inserts, updates=updates)
                        inserted += batch_inserted
                        updated += batch_updated

                    if rows_read % 10_000 == 0:
                        logging.info(f"Rows read: {rows_read} | inserted: {inserted} | updated: {updated} | skipped: {rows_skipped}")

        batch_inserted, batch_updated = flush_batches(db=db, inserts=inserts, updates=updates)
        inserted += batch_inserted
        updated += batch_updated

        # Нові записи, яких немає у новому комплекті CSV
        missing_ids = [record_id for ids in existing_by_hash.values() for record_id in ids]

        deleted_marked = 0
        deleted_at = datetime.now(timezone.utc).isoformat()

        for start in range(0, len(missing_ids), BATCH_SIZE):
            batch_ids = missing_ids[start:start + BATCH_SIZE]

            mappings = [
                {
                    "id": record_id,
                    "another_info": {"deleted_from_firebird": True, "detected_at": deleted_at},
                    "updated_at": datetime.now(timezone.utc),
                }
                for record_id in batch_ids
            ]

            db.bulk_update_mappings(DBMediaDescription, mappings)

            db.flush()
            deleted_marked += len(mappings)

        db.commit()

    except Exception:
        db.rollback()
        logging.exception("Import failed. Transaction rolled back.")
        raise

    logging.info("--------------------------------")
    logging.info(f"Rows read: {rows_read}")
    logging.info(f"Inserted: {inserted}")
    logging.info(f"Updated/reconfirmed: {updated}")
    logging.info(f"Skipped: {rows_skipped}")
    logging.info(f"Marked deleted from Firebird: {deleted_marked}")


if __name__ == "__main__":
    started_at = datetime.now()

    logging.info("--------------------------------")
    logging.info(f"Started: {started_at.isoformat()}")

    with SessionLocal() as db:
        import_media_descriptions(db=db, csv_folder=CSV_FOLDER)

    finished_at = datetime.now()

    logging.info("--------------------------------")
    logging.info(f"Finished: {finished_at.isoformat()}")
    logging.info(f"Running time: {finished_at - started_at}")
