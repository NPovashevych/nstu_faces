import csv
import json
import logging
import sys

from pathlib import Path

from services.config import CSV_FOLDER, HIRES_INTVNEWS_CATALOG, INTVNEWS_STATISTIC


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251")

IGNORED_CSV_NAMES = {
    "NOT_FOUND.csv",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            logging.warning(f"Invalid JSON format: {path}")
            return {}

        return data

    except Exception:
        logging.exception(f"Cannot read JSON: {path}")
        return {}


def load_hires_material_ids(catalog_folder: Path) -> set[str]:
    material_ids: set[str] = set()

    if not catalog_folder.exists():
        raise FileNotFoundError(f"Hires catalog folder not found: {catalog_folder}")

    json_files = sorted(catalog_folder.glob("*.json"))

    logging.info(f"Hires catalog JSON files: {len(json_files)}")

    for json_path in json_files:
        data = load_json(json_path)

        for raw_material_id, item in data.items():
            material_id = str(raw_material_id).strip()

            if not material_id:
                continue

            if not isinstance(item, dict):
                continue

            mxf_path = item.get("mxf_path")

            if not mxf_path:
                continue

            material_ids.add(material_id)

    logging.info(f"Unique material_ids in hires catalog: {len(material_ids)}")

    return material_ids


def detect_csv_encoding(csv_path: Path) -> str:
    last_error = None

    for encoding in CSV_ENCODINGS:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as file:
                file.read(8192)

            return encoding

        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(f"Cannot detect encoding for {csv_path}. Last error: {last_error}")


def analyze_csv(csv_path: Path, hires_material_ids: set[str]) -> dict:
    encoding = detect_csv_encoding(csv_path)

    rows_count = 0
    empty_material_ids = 0
    duplicate_material_ids = 0

    material_ids: set[str] = set()

    with csv_path.open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        normalized_fieldnames = [str(field).strip() for field in reader.fieldnames if field is not None]

        if "material_id" not in normalized_fieldnames:
            raise ValueError(f"CSV has no material_id column. Columns: {reader.fieldnames}")

        for row in reader:
            rows_count += 1

            material_id = str(row.get("material_id") or "").strip()

            if not material_id:
                empty_material_ids += 1
                continue

            if material_id in material_ids:
                duplicate_material_ids += 1
                continue

            material_ids.add(material_id)

    unique_material_ids = len(material_ids)

    mxf_available = sum(1 for material_id in material_ids if material_id in hires_material_ids)
    mxf_not_found = unique_material_ids - mxf_available

    if unique_material_ids:
        mxf_available_percent = round(mxf_available / unique_material_ids * 100, 2)
    else:
        mxf_available_percent = 0.0

    return {
        "csv_file": csv_path.name,
        "rows_count": rows_count,
        "unique_material_ids": unique_material_ids,
        "duplicate_material_ids": duplicate_material_ids,
        "empty_material_ids": empty_material_ids,
        "mxf_available": mxf_available,
        "mxf_not_found": mxf_not_found,
        "mxf_available_percent": mxf_available_percent,
    }


def create_statistics() -> None:
    csv_folder = Path(CSV_FOLDER)
    hires_catalog_folder = Path(HIRES_INTVNEWS_CATALOG)
    output_path = Path(INTVNEWS_STATISTIC)

    if not csv_folder.exists():
        raise FileNotFoundError(f"CSV folder not found: {csv_folder}")

    hires_material_ids = load_hires_material_ids(hires_catalog_folder)

    csv_paths = sorted(path for path in csv_folder.glob("*.csv") if path.name not in IGNORED_CSV_NAMES)

    logging.info(f"CSV files found: {len(csv_paths)}")

    statistics = []

    for index, csv_path in enumerate(csv_paths, start=1):
        try:
            result = analyze_csv(csv_path=csv_path, hires_material_ids=hires_material_ids)
            statistics.append(result)

            logging.info(
                f"[{index}/{len(csv_paths)}] "
                f"{csv_path.name} | "
                f"rows={result['rows_count']} | "
                f"unique={result['unique_material_ids']} | "
                f"duplicates={result['duplicate_material_ids']} | "
                f"mxf={result['mxf_available']} | "
                f"not_found={result['mxf_not_found']} | "
                f"available={result['mxf_available_percent']}%"
            )

        except Exception:
            logging.exception(f"Cannot analyze CSV: {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "csv_file",
        "rows_count",
        "unique_material_ids",
        "duplicate_material_ids",
        "empty_material_ids",
        "mxf_available",
        "mxf_not_found",
        "mxf_available_percent",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")

        writer.writeheader()
        writer.writerows(statistics)

    logging.info("--------------------------------------------------")
    logging.info(f"Statistics created: {output_path}")
    logging.info(f"CSV files processed: {len(statistics)}")
    logging.info(f"Total rows: {sum(row['rows_count'] for row in statistics)}")
    logging.info(f"Total unique material_ids by CSV: {sum(row['unique_material_ids'] for row in statistics)}")
    logging.info(f"Total duplicates: {sum(row['duplicate_material_ids'] for row in statistics)}")
    logging.info(f"Total MXF available: {sum(row['mxf_available'] for row in statistics)}")
    logging.info(f"Total MXF not found: {sum(row['mxf_not_found'] for row in statistics)}")


if __name__ == "__main__":
    create_statistics()
