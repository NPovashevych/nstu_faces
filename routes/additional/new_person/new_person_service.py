import uuid
from pathlib import Path

from services.config import TEMPORARY_FREEZES_FOLDER
from routes.additional.image_services import add_photo, delete_photo


NEW_PERSON_TEMP_FOLDER = TEMPORARY_FREEZES_FOLDER / "new_persons"


def create_new_person_workspace(user_id: int) -> dict:
    workspace_id = uuid.uuid4().hex
    temp_folder = get_new_person_temp_folder(user_id, workspace_id)
    temp_folder.mkdir(parents=True, exist_ok=False)

    return {
        "workspace_id": workspace_id,
    }


def get_new_person_temp_folder(user_id: int, workspace_id: str) -> Path:
    return NEW_PERSON_TEMP_FOLDER / str(user_id) / workspace_id


def add_new_person_photo(user_id: int, workspace_id: str, file_name: str, file_content: bytes) -> Path:
    temp_folder = get_new_person_temp_folder(user_id, workspace_id)

    return add_photo(folder=temp_folder, file_name=file_name, file_content=file_content)


def delete_new_person_photo(user_id: int, workspace_id: str, file_name: str) -> Path:
    temp_folder = get_new_person_temp_folder(user_id, workspace_id)

    return delete_photo(folder=temp_folder, file_name=file_name)
