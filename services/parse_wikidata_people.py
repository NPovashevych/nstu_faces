import json
import re
import time
import logging
from pathlib import Path
from urllib.parse import quote_plus

import requests

from config import PERSONS_FOLDER, NEW_WIKI_PATH


URL = "https://www.wikidata.org/w/api.php"
REQUEST_DELAY = 0.05

HEADERS = {
    "User-Agent": "NSTU-Faces-Parser/1.0"
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)8s: %(message)s",
    datefmt="%H:%M:%S",
)


def parse_name_qcode(folder_name: str):
    folder_name = folder_name.strip()

    match = re.search(r"\((Q\d+)\)\s*$", folder_name)
    if match:
        q_code = match.group(1)
        name = folder_name[:match.start()].strip()
        return name, q_code

    return folder_name, None


def load_people_db(path: Path) -> dict:
    if not path.exists():
        logging.info(f"Файл {path} не існує — створюю новий.")
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_people_db(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_wikipedia_link(sitelinks: dict):
    if "ukwiki" in sitelinks:
        title = sitelinks["ukwiki"]["title"]
        return f"https://uk.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"

    if "enwiki" in sitelinks:
        title = sitelinks["enwiki"]["title"]
        return f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"

    return None


def get_person_data_from_wikidata(name: str, q_code: str):
    params = {
        "action": "wbgetentities",
        "ids": q_code,
        "props": "labels|claims|sitelinks",
        "languages": "uk|en",
        "format": "json",
    }

    r = requests.get(URL, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()

    data = r.json()
    entity = data.get("entities", {}).get(q_code)

    if not entity:
        return None

    labels = entity.get("labels", {})
    claims = entity.get("claims", {})
    sitelinks = entity.get("sitelinks", {})

    wikidata_name = (
        labels.get("uk", {}).get("value")
        or labels.get("en", {}).get("value")
        or name
    )

    link = get_wikipedia_link(sitelinks)

    image = None
    if "P18" in claims:
        try:
            image = claims["P18"][0]["mainsnak"]["datavalue"]["value"]
        except Exception:
            image = None

    year_birth = None
    if "P569" in claims:
        try:
            date_value = claims["P569"][0]["mainsnak"]["datavalue"]["value"]["time"]
            year_birth = int(date_value[1:5])
        except Exception:
            year_birth = None

    return {
        "id": q_code,
        "name": wikidata_name,
        "folder_name": name,
        "link": link,
        "image": image,
        "year_birth": year_birth,
    }


def parse_people_folders():
    persons_dir = Path(PERSONS_FOLDER)
    output_path = Path(NEW_WIKI_PATH)

    people_db = load_people_db(output_path)

    added = 0
    skipped_existing = 0
    skipped_no_qcode = 0
    errors = 0

    for person_folder in sorted(persons_dir.iterdir()):
        if not person_folder.is_dir():
            continue

        name, q_code = parse_name_qcode(person_folder.name)

        if not q_code:
            skipped_no_qcode += 1
            continue

        if q_code in people_db:
            skipped_existing += 1
            continue

        try:
            person_data = get_person_data_from_wikidata(name, q_code)

            if person_data is None:
                logging.warning(f"{person_folder.name}: не знайдено у Wikidata")
                errors += 1
                continue

            people_db[q_code] = person_data
            added += 1

            logging.info(f"Додано: {name} ({q_code})")

            save_people_db(output_path, people_db)
            time.sleep(REQUEST_DELAY)

        except Exception as e:
            logging.warning(f"{person_folder.name}: помилка — {e}")
            errors += 1

    save_people_db(output_path, people_db)

    logging.info("--------------------------------")
    logging.info(f"Додано нових: {added}")
    logging.info(f"Вже були в JSON: {skipped_existing}")
    logging.info(f"Без q_code: {skipped_no_qcode}")
    logging.info(f"Помилки: {errors}")
    logging.info(f"Файл збережено: {output_path}")


if __name__ == "__main__":
    parse_people_folders()
