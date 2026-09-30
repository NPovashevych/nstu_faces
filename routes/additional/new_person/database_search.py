from fuzzywuzzy import fuzz
from sqlalchemy.orm import Session

from db.enums import PersonStatus
from db.models import DBPerson


NAME_SIMILARITY_THRESHOLD = 90


def find_person_in_db(db: Session, name: str, q_code: str | None = None) -> dict:
    name = " ".join(name.strip().lower().split())

    if not name:
        raise ValueError("Person name is empty")

    q_code = (q_code or "").strip().upper()

    persons = db.query(DBPerson).filter(DBPerson.status.in_([PersonStatus.public, PersonStatus.non_public])).all()

    if q_code:
        q_code_match = next((person for person in persons if person.q_code == q_code), None)

        if q_code_match:
            return {
                "q_code_match": {
                    "person_id": q_code_match.id,
                    "name": q_code_match.name,
                    "q_code": q_code_match.q_code,
                    "status": q_code_match.status.value,
                },
                "name_matches": [],
            }

    name_matches = []

    for person in persons:
        person_name = " ".join(person.name.strip().lower().split())

        ratio = fuzz.ratio(name, person_name)
        partial_ratio = fuzz.partial_ratio(name, person_name)
        similarity = max(ratio, partial_ratio)

        if similarity >= NAME_SIMILARITY_THRESHOLD:
            name_matches.append({
                "person_id": person.id,
                "name": person.name,
                "q_code": person.q_code,
                "status": person.status.value,
                "similarity": similarity,
            })

    name_matches.sort(key=lambda item: item["similarity"], reverse=True)

    return {
        "q_code_match": None,
        "name_matches": name_matches,
    }
