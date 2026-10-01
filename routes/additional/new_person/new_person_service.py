import uuid
from pathlib import Path
import shutil

from services.config import TEMPORARY_FREEZES_FOLDER, NEW_PERSONS_FOLDER, LIKE_UNKNOWN_FOLDER
from routes.additional.image_services import add_photo, delete_photo, get_images
from routes.additional.new_person.validation import validate_ukrainian_text, validate_q_code, validate_ukrainian_pseudonym
from routes.additional.create_folder_or_file_names import create_photo_file_name, normalize_photo_category


NEW_PERSON_TEMP_FOLDER = TEMPORARY_FREEZES_FOLDER / "new_persons"


def create_new_person_workspace(user_id: int, name: str) -> dict:
    name = validate_ukrainian_text(name, "Person name")
    workspace_id = f"{name}__{uuid.uuid4().hex[:8]}"
    temp_folder = get_new_person_temp_folder(user_id, workspace_id)
    temp_folder.mkdir(parents=True, exist_ok=False)

    return {"workspace_id": workspace_id}


def get_new_person_temp_folder(user_id: int, workspace_id: str) -> Path:
    return NEW_PERSON_TEMP_FOLDER / str(user_id) / workspace_id


def add_new_person_photo(user_id: int, workspace_id: str, file_name: str, file_content: bytes) -> Path:
    temp_folder = get_new_person_temp_folder(user_id, workspace_id)

    return add_photo(folder=temp_folder, file_name=file_name, file_content=file_content)


def delete_new_person_photo(user_id: int, workspace_id: str, file_name: str) -> Path:
    temp_folder = get_new_person_temp_folder(user_id, workspace_id)

    return delete_photo(folder=temp_folder, file_name=file_name)


def make_new_person_folder_name(name: str, pseudonym: str | None = None, q_code: str | None = None) -> str:
    name = validate_ukrainian_text(name, "Person name")
    pseudonym = validate_ukrainian_pseudonym(pseudonym)
    q_code = validate_q_code(q_code)

    parts = [name]

    if pseudonym:
        parts.append(f"({pseudonym})")

    if q_code:
        parts.append(f"({q_code})")

    return " ".join(parts)


def save_new_person(user_id: int, workspace_id: str, name: str, pseudonym: str | None = None, q_code: str | None = None, category: str | None = None, like_unknown: bool = False) -> dict:
    temp_folder = get_new_person_temp_folder(user_id, workspace_id)

    if not temp_folder.exists() or not temp_folder.is_dir():
        raise FileNotFoundError("New person temporary folder not found")

    photos = get_images(temp_folder)

    if not photos:
        raise ValueError("New person must have at least 1 photo")

    folder_name = make_new_person_folder_name(name=name, pseudonym=pseudonym, q_code=q_code)
    target_root = LIKE_UNKNOWN_FOLDER if like_unknown else NEW_PERSONS_FOLDER
    final_folder = target_root / folder_name

    if final_folder.exists():
        raise FileExistsError(f"Person folder already exists: {folder_name}")

    normalized_category = normalize_photo_category(category)
    for number, photo in enumerate(photos, start=1):
        new_name = create_photo_file_name(person_name=name, category=category, number=number, extension=photo.suffix)
        photo.rename(temp_folder / new_name)

    target_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp_folder), str(final_folder))

    user_temp_folder = NEW_PERSON_TEMP_FOLDER / str(user_id)

    if user_temp_folder.exists() and not any(user_temp_folder.iterdir()):
        user_temp_folder.rmdir()

    return {
        "status": "ok",
        "folder_name": folder_name,
        "destination": "like_unknown" if like_unknown else "new_person",
        "photo_category": normalized_category,
        "photos_count": len(photos),
    }
