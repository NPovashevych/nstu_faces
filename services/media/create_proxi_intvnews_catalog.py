import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from services.config import (
    PROXY_NEWS_FOLDER,
    PROXY_DIGITAL_FOLDER,
    PROXY_INTVNEWS_CATALOG,
    PROXY_DIGITAL_CATALOG,
    PROXY_INTVNEWS_DUPLICATE_FILE,
    PROXY_DIGITAL_DUPLICATE_FILE,
)
from services.commons.scan_statistics import ScanStatistics


# ----------------------------------------------------------------------
# НАЛАШТУВАННЯ
# ----------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4"}

PROXY_INTVNEWS_YEARS = {
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
}

PROXY_DIGITAL_YEARS = {
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
}


LOG_PATH = Path("../logs/create_proxy_catalog.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_PATH,
            encoding="utf-8",
        ),
    ],
)


# ----------------------------------------------------------------------
# МОДЕЛЬ ЕЛЕМЕНТА КАТАЛОГУ
# ----------------------------------------------------------------------

@dataclass
class ProxyCatalogItem:
    name: str
    original_name: str
    mp4_path: str
    size: int
    modified: float
    modified_iso: str
    indexed_at: str
    catalog_year: str
    is_duplicate: bool = False
    duplicate_number: int | None = None
    is_missing: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------

def save_json(path: Path, data: dict) -> None:
    """
    Безпечно записує JSON через тимчасовий файл.

    Старий файл замінюється лише після успішного запису нового.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(f"{path.name}.tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        temp_path.replace(path)

    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

        raise


# ----------------------------------------------------------------------
# ДОПОМІЖНІ ФУНКЦІЇ
# ----------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """
    Нормалізує назву для пошуку дублікатів.

    Наприклад:
        NEWS_001
        news_001

    вважатимуться однаковою назвою.
    """
    return name.strip().casefold()


def make_catalog_key(
    original_name: str,
    duplicate_number: int,
) -> str:
    """
    Перший файл отримує звичайний ключ.

    Наступні файли з такою самою назвою:
        name_duplicate_1
        name_duplicate_2
    """
    if duplicate_number == 0:
        return original_name

    return f"{original_name}_duplicate_{duplicate_number}"


def make_file_item(
    file_path: Path,
    catalog_year: str,
    catalog_key: str,
    original_name: str,
    duplicate_number: int,
) -> dict:
    stat = file_path.stat()

    item = ProxyCatalogItem(
        name=catalog_key,
        original_name=original_name,
        mp4_path=str(file_path),
        size=stat.st_size,
        modified=stat.st_mtime,
        modified_iso=datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat(),
        indexed_at=datetime.now().isoformat(),
        catalog_year=catalog_year,
        is_duplicate=duplicate_number > 0,
        duplicate_number=(
            duplicate_number
            if duplicate_number > 0
            else None
        ),
        is_missing=False,
    )

    return item.to_dict()


def get_year_catalog_path(
    catalog_folder: Path,
    year: str,
) -> Path:
    return catalog_folder / f"{year}.json"


def get_year_folder(
    root_folder: Path,
    year: str,
) -> Path | None:
    """
    Повертає папку конкретного дозволеного року.

    Інші папки в корені сховища навіть не переглядаються.
    """
    year_folder = root_folder / year

    if not year_folder.exists():
        logging.warning(
            f"Year folder does not exist: {year_folder}"
        )
        return None

    if not year_folder.is_dir():
        logging.warning(
            f"Year path is not a folder: {year_folder}"
        )
        return None

    return year_folder


# ----------------------------------------------------------------------
# СКАНУВАННЯ ОДНОГО РОКУ
# ----------------------------------------------------------------------

def scan_year_folder(
    source_name: str,
    year: str,
    year_folder: Path,
    global_name_counts: dict[str, int],
    first_items_by_name: dict[str, dict],
    duplicates: dict[str, list[dict]],
    stats: ScanStatistics,
) -> dict:
    """
    Сканує одну папку року та повертає її Proxy-каталог.

    Дублікати визначаються глобально між усіма роками
    одного джерела.
    """
    year_catalog: dict[str, dict] = {}

    year_files_scanned = 0
    year_mp4_found = 0
    year_duplicates = 0
    year_errors = 0

    logging.info("--------------------------------")
    logging.info(f"[{source_name}] Scanning year: {year}")
    logging.info(f"Folder: {year_folder}")

    try:
        for file_path in year_folder.rglob("*"):
            try:
                if not file_path.is_file():
                    continue

                year_files_scanned += 1
                stats.files_scanned += 1

                if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    if stats.should_report():
                        stats.report_progress()

                    continue

                year_mp4_found += 1
                stats.matched_files += 1

                original_name = file_path.stem
                normalized_name = normalize_name(original_name)

                duplicate_number = global_name_counts.get(
                    normalized_name,
                    0,
                )

                catalog_key = make_catalog_key(
                    original_name=original_name,
                    duplicate_number=duplicate_number,
                )

                # Захист від збігу сформованого ключа
                # з реальною назвою іншого файла.
                while catalog_key in year_catalog:
                    duplicate_number += 1

                    catalog_key = make_catalog_key(
                        original_name=original_name,
                        duplicate_number=duplicate_number,
                    )

                item = make_file_item(
                    file_path=file_path,
                    catalog_year=year,
                    catalog_key=catalog_key,
                    original_name=original_name,
                    duplicate_number=duplicate_number,
                )

                year_catalog[catalog_key] = item

                global_name_counts[normalized_name] = (
                    duplicate_number + 1
                )

                if duplicate_number == 0:
                    first_items_by_name[normalized_name] = item

                else:
                    year_duplicates += 1
                    stats.duplicates += 1

                    if normalized_name not in duplicates:
                        duplicates[normalized_name] = []

                        first_item = first_items_by_name.get(
                            normalized_name
                        )

                        if first_item is not None:
                            duplicates[normalized_name].append(
                                first_item
                            )

                    duplicates[normalized_name].append(item)

                    logging.warning(
                        f"Duplicate MP4 name: {original_name}"
                    )
                    logging.warning(
                        f"  duplicate number: "
                        f"{duplicate_number}"
                    )
                    logging.warning(
                        f"  path: {file_path}"
                    )

                if stats.should_report():
                    stats.report_progress()

            except Exception as error:
                year_errors += 1
                stats.errors += 1

                logging.warning(
                    f"Cannot process file: "
                    f"{file_path} | {error}"
                )

    except Exception as error:
        year_errors += 1
        stats.errors += 1

        logging.error(
            f"Cannot scan folder: "
            f"{year_folder} | {error}"
        )

    logging.info(
        f"[{source_name} | {year}] "
        f"files scanned: {year_files_scanned}"
    )
    logging.info(
        f"[{source_name} | {year}] "
        f"MP4 found: {year_mp4_found}"
    )
    logging.info(
        f"[{source_name} | {year}] "
        f"duplicates: {year_duplicates}"
    )
    logging.info(
        f"[{source_name} | {year}] "
        f"errors: {year_errors}"
    )

    return year_catalog


# ----------------------------------------------------------------------
# ФОРМУВАННЯ ФАЙЛА ДУБЛІКАТІВ
# ----------------------------------------------------------------------

def prepare_duplicates_for_json(
    duplicates: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """
    Внутрішньо дублікати зберігаються за нормалізованою назвою.

    У JSON ключем стає оригінальна назва першого знайденого файла.
    """
    result: dict[str, list[dict]] = {}

    for normalized_name, items in duplicates.items():
        if not items:
            continue

        original_name = items[0].get(
            "original_name",
            normalized_name,
        )

        result[original_name] = items

    return result


# ----------------------------------------------------------------------
# ПОБУДОВА КАТАЛОГУ ОДНОГО ДЖЕРЕЛА
# ----------------------------------------------------------------------

def build_proxy_catalog(
    name: str,
    root_folder: Path,
    years: set[str],
    catalog_folder: Path,
    duplicate_path: Path,
) -> None:
    """
    Формує окремі Proxy JSON-каталоги
    для кожного дозволеного року.
    """
    logging.info("================================")
    logging.info(f"Build PROXY catalog: {name}")
    logging.info(f"Root folder: {root_folder}")
    logging.info(f"Catalog folder: {catalog_folder}")
    logging.info(
        f"Years: {', '.join(sorted(years))}"
    )

    catalog_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats = ScanStatistics(
        name=name,
        progress_step=10_000,
    )

    # Спільні для всіх років одного джерела.
    global_name_counts: dict[str, int] = {}
    first_items_by_name: dict[str, dict] = {}
    duplicates: dict[str, list[dict]] = {}

    for year in sorted(years):
        year_catalog_path = get_year_catalog_path(
            catalog_folder=catalog_folder,
            year=year,
        )

        year_folder = get_year_folder(
            root_folder=root_folder,
            year=year,
        )

        if year_folder is None:
            # Порожній JSON означає, що рік передбачений,
            # але відповідної папки в сховищі немає.
            save_json(
                path=year_catalog_path,
                data={},
            )

            stats.errors += 1

            logging.info(
                f"Empty catalog saved: "
                f"{year_catalog_path}"
            )
            continue

        stats.folders_scanned += 1

        year_catalog = scan_year_folder(
            source_name=name,
            year=year,
            year_folder=year_folder,
            global_name_counts=global_name_counts,
            first_items_by_name=first_items_by_name,
            duplicates=duplicates,
            stats=stats,
        )

        save_json(
            path=year_catalog_path,
            data=year_catalog,
        )

        logging.info(
            f"Catalog saved: {year_catalog_path}"
        )
        logging.info(
            f"Catalog records: {len(year_catalog)}"
        )

    duplicates_for_json = prepare_duplicates_for_json(
        duplicates
    )

    save_json(
        path=duplicate_path,
        data=duplicates_for_json,
    )

    duplicate_files_count = sum(
        len(items) - 1
        for items in duplicates_for_json.values()
    )

    logging.info("--------------------------------")
    logging.info(
        f"[{name}] Total MP4 files: "
        f"{stats.matched_files}"
    )
    logging.info(
        f"[{name}] Names with duplicates: "
        f"{len(duplicates_for_json)}"
    )
    logging.info(
        f"[{name}] Additional duplicate files: "
        f"{duplicate_files_count}"
    )
    logging.info(
        f"[{name}] Duplicate report saved: "
        f"{duplicate_path}"
    )

    stats.report_summary()


# ----------------------------------------------------------------------
# ЗАПУСК
# ----------------------------------------------------------------------

if __name__ == "__main__":
    start = datetime.now()

    logging.info("================================")
    logging.info(f"Start: {start.isoformat()}")

    build_proxy_catalog(
        name="proxy_intvnews",
        root_folder=PROXY_NEWS_FOLDER,
        years=PROXY_INTVNEWS_YEARS,
        catalog_folder=PROXY_INTVNEWS_CATALOG,
        duplicate_path=PROXY_INTVNEWS_DUPLICATE_FILE,
    )

    build_proxy_catalog(
        name="proxy_digital",
        root_folder=PROXY_DIGITAL_FOLDER,
        years=PROXY_DIGITAL_YEARS,
        catalog_folder=PROXY_DIGITAL_CATALOG,
        duplicate_path=PROXY_DIGITAL_DUPLICATE_FILE,
    )

    finish = datetime.now()

    logging.info("================================")
    logging.info(f"Finished: {finish.isoformat()}")
    logging.info(f"Running time: {finish - start}")
