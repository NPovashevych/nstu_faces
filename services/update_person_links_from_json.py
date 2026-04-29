import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

from config import NEW_WIKI_PATH
from db.session import SessionLocal
from db.models import DBPerson
from db.enums import PersonStatus


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)8s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/update_person_links_from_json.log", encoding="utf-8"),
    ],
)


def is_unknown_name(name: str) -> bool:
    return name.strip().lower().startswith("unknown")


def load_people_db(path: Path) -> dict:
    if not path.exists():
        logging.warning(f"JSON файл не знайдено: {path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_google_search_link(name: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(name)}"


def get_link_for_person(person: DBPerson, people_db: dict):
    # unknown → нічого не ставимо
    if is_unknown_name(person.name):
        return None

    # є q_code → пробуємо wiki
    if person.q_code:
        data = people_db.get(person.q_code)

        if data and data.get("link"):
            return data.get("link")

    # fallback → Google
    return get_google_search_link(person.name)


def get_status_for_person(person: DBPerson) -> PersonStatus:
    if person.q_code:
        return PersonStatus.public

    if is_unknown_name(person.name):
        return PersonStatus.unknown

    return PersonStatus.non_public


def update_person_links():
    people_db = load_people_db(Path(NEW_WIKI_PATH))
    db = SessionLocal()

    updated = 0
    skipped = 0

    try:
        persons = db.query(DBPerson).order_by(DBPerson.id).all()

        for person in persons:
            new_link = get_link_for_person(person, people_db)
            new_status = get_status_for_person(person)

            changed = False

            if person.link != new_link:
                person.link = new_link
                changed = True

            if person.status != new_status:
                person.status = new_status
                changed = True

            if changed:
                updated += 1
                logging.info(f"Updated person_id={person.id}: {person.name}")
            else:
                skipped += 1

        db.commit()

        logging.info("--------------------------------")
        logging.info(f"Updated: {updated}")
        logging.info(f"Skipped: {skipped}")

    except Exception as e:
        db.rollback()
        logging.exception(f"Помилка оновлення person links: {e}")

    finally:
        db.close()


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Start: {start}")

    update_person_links()

    finish = datetime.now()
    logging.info(f"Finished. Running time: {finish - start}")
