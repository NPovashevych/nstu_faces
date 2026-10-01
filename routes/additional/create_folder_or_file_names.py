from unidecode import unidecode
import re


def extract_q_code(folder_name: str) -> str | None:
    match = re.search(r"\((Q\d+)\)\s*$", folder_name)
    if match is None:
        return None
    return match.group(1)


def extract_candidate_name(folder_name: str) -> str:
    return re.sub(r"\s*\(Q\d+\)\s*$", "", folder_name).strip()


def get_person_file_prefix(person_name: str) -> str:
    person_name = person_name.strip()

    if not person_name:
        raise ValueError("Person name is empty")

    first_word = person_name.split()[0]
    prefix = unidecode(first_word).lower()
    prefix = "".join(char for char in prefix if char.isalnum() or char in {"-", "_"})

    if not prefix:
        raise ValueError("Cannot create photo file prefix")

    return prefix


def normalize_photo_category(category: str | None) -> str:
    category = (category or "").strip().lower()
    return "".join(char for char in category if char.isalnum() or char in {"-", "_"})


def create_photo_file_name(person_name: str, category: str | None, number: int, extension: str) -> str:
    prefix = get_person_file_prefix(person_name)
    category = normalize_photo_category(category)
    extension = extension.lower()

    if category:
        return f"{prefix}_{category}_{number}{extension}"

    return f"{prefix}_{number}{extension}"
