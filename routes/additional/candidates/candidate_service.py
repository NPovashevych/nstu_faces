import re
import shutil
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unidecode import unidecode

from services.config import START_CANDIDATE_FOLDER, TEMPORARY_FREEZES_FOLDER, FINISH_CANDIDATE_FOLDER, SKIPPED_CANDIDATE_FOLDER



CANDIDATE_TEMP_FOLDER = TEMPORARY_FREEZES_FOLDER / "candidates"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LOCK_TIMEOUT = timedelta(hours=2)
MAX_CANDIDATE_PHOTOS = 12


# ---------------------------------------------------------------------------
# IN-MEMORY LOCKS
# ---------------------------------------------------------------------------
#      "Чугай Катерина (Q100354851)": {          Ключ словника = точна назва вихідної папки кандидата.
#         "user_id": 17,
#         "locked_at": datetime(...),           ці lock-и існують тільки в пам'яті процесу API, після перезапуску зникають.
#         "last_activity": datetime(...),
#     }
#
# Такий механізм розрахований на запуск API в ОДНОМУ процесі/worker. Якщо в production буде кілька worker-ів або кілька екземплярів API,
# цей механізм потрібно буде перенести у спільне сховище, наприклад Redis.


_candidate_locks: dict[str, dict] = {}
_candidate_locks_mutex = threading.Lock() # Захищає операцію "знайти вільного кандидата + заблокувати його".
_candidate_cache: list[str] = []
_candidate_cache_initialized = False # Кеш в пам'яті процесу API. Після перезапуску порожній. Нові папки після ручного Refresh candidates.


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def extract_q_code(folder_name: str) -> str | None:
    match = re.search(r"\((Q\d+)\)\s*$", folder_name)
    if match is None:
        return None
    return match.group(1)


def extract_candidate_name(folder_name: str) -> str:
    return re.sub(r"\s*\(Q\d+\)\s*$", "", folder_name).strip()


def get_candidate_source_folder(candidate_key: str) -> Path:
    return START_CANDIDATE_FOLDER / candidate_key


def get_candidate_temp_folder(user_id: int, candidate_key: str) -> Path:
    return CANDIDATE_TEMP_FOLDER / str(user_id) / candidate_key


def get_candidate_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []

    return sorted(
        [file_path for file_path in folder.iterdir() if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower()
    )


def candidate_to_dict(candidate_key: str, user_id: int) -> dict:
    source_folder = get_candidate_source_folder(candidate_key)
    temp_folder = get_candidate_temp_folder(user_id, candidate_key)
    temp_images = get_candidate_images(temp_folder)

    photos = []

    for image_path in temp_images:
        photo_url = (f"/media-youtube-upload/candidates/{user_id}/{candidate_key}/{image_path.name}")
        photos.append({"file_name": image_path.name, "url": photo_url})

    return {
        "candidate_key": candidate_key,
        "original_name": extract_candidate_name(candidate_key),
        "q_code": extract_q_code(candidate_key),
        "source_folder_name": source_folder.name,
        "photos_count": len(photos),
        "photos": photos,
    }


def prepare_candidate_temp(user_id: int, candidate_key: str) -> Path:
    source_folder = get_candidate_source_folder(candidate_key)

    if not source_folder.exists() or not source_folder.is_dir():
        raise FileNotFoundError(f"Candidate source folder not found: {source_folder}")

    temp_folder = get_candidate_temp_folder(user_id, candidate_key)
    if temp_folder.exists():
        return temp_folder

    temp_folder.mkdir(parents=True, exist_ok=True)

    for source_image in get_candidate_images(source_folder):
        destination = temp_folder / source_image.name
        shutil.copy2(source_image, destination)

    return temp_folder


def cleanup_candidate_temp(user_id: int, candidate_key: str) -> None:
    temp_folder = get_candidate_temp_folder(user_id, candidate_key)

    if temp_folder.exists():
        shutil.rmtree(temp_folder)


def remove_stale_locks() -> None:
    now = utc_now()
    stale_candidates = [candidate_key for candidate_key, lock in _candidate_locks.items() if now - lock["last_activity"] > LOCK_TIMEOUT]

    for candidate_key in stale_candidates:
        _candidate_locks.pop(candidate_key, None)


def refresh_candidate_cache() -> dict:
    """
    Викликається:
    - вручну через Refresh candidates;
    - автоматично при першому Take після запуску API;
    - автоматично, якщо кеш кандидатів спорожнів.
    """

    global _candidate_cache
    global _candidate_cache_initialized

    if not START_CANDIDATE_FOLDER.exists():
        raise FileNotFoundError(f"Candidate folder not found: {START_CANDIDATE_FOLDER}")

    start_time = time.perf_counter()
    candidate_keys = [folder.name for folder in START_CANDIDATE_FOLDER.iterdir() if folder.is_dir()]
    candidate_keys.sort(key=str.lower)
    refresh_time = time.perf_counter() - start_time

    with _candidate_locks_mutex:
        _candidate_cache = candidate_keys
        _candidate_cache_initialized = True

    return {
        "status": "ok",
        "candidates_count": len(candidate_keys),
        "refresh_time": round(refresh_time, 3),
    }


def find_candidate_locked_by_user(user_id: int) -> str | None:
    for candidate_key, lock in _candidate_locks.items():
        if lock["user_id"] == user_id:
            return candidate_key

    return None


def touch_candidate(user_id: int, candidate_key: str) -> bool: # Оновлює last_activity активного lock-а. Пізніше цю функцію можна викликати при upload/delete/check/save тощо.
    with _candidate_locks_mutex:
        remove_stale_locks()

        lock = _candidate_locks.get(candidate_key)

        if lock is None:
            return False

        if lock["user_id"] != user_id:
            return False

        lock["last_activity"] = utc_now()

        return True


def release_candidate(user_id: int, candidate_key: str) -> bool:
    with _candidate_locks_mutex:
        lock = _candidate_locks.get(candidate_key)

        if lock is None:
            return False

        if lock["user_id"] != user_id:
            return False

        _candidate_locks.pop(candidate_key, None)
        # cleanup_candidate_temp(user_id, candidate_key)

        return True


def get_active_candidate(user_id: int) -> dict | None:
    with _candidate_locks_mutex:
        remove_stale_locks()

        candidate_key = find_candidate_locked_by_user(user_id)

        if candidate_key is None:
            return None

        source_folder = get_candidate_source_folder(candidate_key)

        # Якщо вихідна папка чомусь зникла поза API, старий lock більше не має сенсу.
        if not source_folder.exists() or not source_folder.is_dir():
            _candidate_locks.pop(candidate_key, None)
            return None

        _candidate_locks[candidate_key]["last_activity"] = utc_now()

    prepare_candidate_temp(user_id, candidate_key)

    return candidate_to_dict(candidate_key, user_id)


def take_candidate(user_id: int) -> dict | None:
    global _candidate_cache
    with _candidate_locks_mutex:
        remove_stale_locks()

        active_candidate_key = find_candidate_locked_by_user(user_id)

        if active_candidate_key is not None:
            _candidate_locks[active_candidate_key]["last_activity"] = utc_now()
            candidate_key = active_candidate_key
        else:
            candidate_key = None

    if candidate_key is not None:
        prepare_candidate_temp(user_id, candidate_key)
        return candidate_to_dict(candidate_key, user_id)

    # Перший Take після запуску API.
    if not _candidate_cache_initialized:
        refresh_candidate_cache()

    if not _candidate_cache:
        refresh_candidate_cache()

    with _candidate_locks_mutex:
        remove_stale_locks()
        candidate_key = None

        for current_candidate_key in _candidate_cache:
            if current_candidate_key in _candidate_locks:
                continue
            source_folder = get_candidate_source_folder(current_candidate_key)
            if not source_folder.exists() or not source_folder.is_dir():
                continue

            now = utc_now()

            _candidate_locks[current_candidate_key] = {
                "user_id": user_id,
                "locked_at": now,
                "last_activity": now,
            }

            candidate_key = current_candidate_key
            break

    if candidate_key is None:
        return None

    try:
        prepare_candidate_temp(user_id, candidate_key)

    except Exception:
        release_candidate(user_id, candidate_key)
        raise

    return candidate_to_dict(candidate_key, user_id)


def get_free_photo_name(temp_folder: Path, original_file_name: str) -> str: # тілька для temporary-папка
    original_path = Path(original_file_name)

    stem = original_path.stem.strip() or "photo"
    suffix = original_path.suffix.lower()

    candidate_name = f"{stem}{suffix}"

    if not (temp_folder / candidate_name).exists():
        return candidate_name

    number = 2

    while True:
        candidate_name = f"{stem}_{number}{suffix}"

        if not (temp_folder / candidate_name).exists():
            return candidate_name

        number += 1


def add_candidate_photo(user_id: int, candidate_key: str, file_name: str, file_content: bytes) -> dict: # тілька для temporary-папка
    if not file_name:
        raise ValueError("File name is empty")

    suffix = Path(file_name).suffix.lower()

    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image format. Allowed: jpg, jpeg, png, webp")

    with _candidate_locks_mutex:
        remove_stale_locks()

        lock = _candidate_locks.get(candidate_key)

        if lock is None:
            raise ValueError("Candidate is not locked")

        if lock["user_id"] != user_id:
            raise ValueError("Candidate is locked by another user")

        temp_folder = get_candidate_temp_folder(user_id, candidate_key)

        if not temp_folder.exists():
            raise FileNotFoundError(f"Candidate temporary folder not found: {temp_folder}")

        current_images = get_candidate_images(temp_folder)

        if len(current_images) >= MAX_CANDIDATE_PHOTOS:
            raise ValueError(f"Candidate already has maximum {MAX_CANDIDATE_PHOTOS} photos")

        safe_original_name = Path(file_name).name
        destination_name = get_free_photo_name(temp_folder=temp_folder, original_file_name=safe_original_name)

        destination = temp_folder / destination_name

        destination.write_bytes(file_content)

        lock["last_activity"] = utc_now()

    return candidate_to_dict(candidate_key, user_id)


def delete_candidate_photo(user_id: int, candidate_key: str, file_name: str) -> dict: # тілька для temporary-папка
    if not file_name:
        raise ValueError("File name is empty")

    with _candidate_locks_mutex:
        remove_stale_locks()

        lock = _candidate_locks.get(candidate_key)

        if lock is None:
            raise ValueError("Candidate is not locked")

        if lock["user_id"] != user_id:
            raise ValueError("Candidate is locked by another user")

        temp_folder = get_candidate_temp_folder(user_id, candidate_key)

        if not temp_folder.exists():
            raise FileNotFoundError(
                f"Candidate temporary folder not found: {temp_folder}"
            )

        safe_file_name = Path(file_name).name
        photo_path = temp_folder / safe_file_name

        if not photo_path.exists() or not photo_path.is_file():
            raise FileNotFoundError(f"Candidate photo not found: {safe_file_name}")

        if photo_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("File is not a candidate image")

        photo_path.unlink()

        lock["last_activity"] = utc_now()

    return candidate_to_dict(candidate_key, user_id)


def get_candidate_file_prefix(final_name: str) -> str:
    final_name = final_name.strip()

    if not final_name:
        raise ValueError("Final candidate name is empty")

    surname = final_name.split()[0]
    prefix = unidecode(surname).lower()
    prefix = "".join(char for char in prefix if char.isalnum() or char in {"-", "_"})

    if not prefix:
        raise ValueError("Cannot create photo file prefix")
    return prefix


def normalize_candidate_category(category: str | None) -> str:
    category = (category or "").strip().lower()
    return "".join(char for char in category if char.isalnum() or char in {"-", "_"})


def build_final_candidate_folder_name(final_name: str, q_code: str | None) -> str:
    final_name = final_name.strip()

    if not final_name:
        raise ValueError("Final candidate name is empty")

    if q_code:
        return f"{final_name} ({q_code})"

    return final_name


def save_candidate(user_id: int, user_name: str, candidate_key: str, final_name: str, category: str | None = None) -> dict:
    with _candidate_locks_mutex:
        remove_stale_locks()

        lock = _candidate_locks.get(candidate_key)

        if lock is None:
            raise ValueError("Candidate is not locked")

        if lock["user_id"] != user_id:
            raise ValueError("Candidate is locked by another user")

        source_folder = get_candidate_source_folder(candidate_key)
        temp_folder = get_candidate_temp_folder(user_id, candidate_key)

        if not source_folder.exists() or not source_folder.is_dir():
            raise FileNotFoundError(f"Candidate source folder not found: {source_folder}")

        if not temp_folder.exists() or not temp_folder.is_dir():
            raise FileNotFoundError(f"Candidate temporary folder not found: {temp_folder}")

        temp_images = get_candidate_images(temp_folder)

        if len(temp_images) == 0:
            raise ValueError("Candidate must have at least 1 photo")

        if len(temp_images) > MAX_CANDIDATE_PHOTOS:
            raise ValueError(f"Candidate cannot have more than {MAX_CANDIDATE_PHOTOS} photos")

        q_code = extract_q_code(candidate_key)

        final_folder_name = build_final_candidate_folder_name(final_name=final_name, q_code=q_code)

        archivist_folder = FINISH_CANDIDATE_FOLDER / user_name
        final_folder = archivist_folder / final_folder_name

        if final_folder.exists():
            raise FileExistsError(f"Final candidate folder already exists: {final_folder}")

        prefix = get_candidate_file_prefix(final_name)
        normalized_category = normalize_candidate_category(category)

        # Фото, які були у вихідній Wikipedia-папці.
        source_image_names = {image.name for image in get_candidate_images(source_folder)}

        archivist_folder.mkdir(parents=True, exist_ok=True)

        try:
            final_folder.mkdir(parents=False, exist_ok=False)
            new_photo_number = 1
            for temp_image in temp_images:
                if temp_image.name in source_image_names:
                    destination_name = temp_image.name
                else:
                    destination_name = f"{prefix}_{normalized_category}_{new_photo_number}{temp_image.suffix.lower()}"
                    new_photo_number += 1
                destination = final_folder / destination_name
                shutil.copy2(temp_image, destination)

        except Exception:
            if final_folder.exists():
                shutil.rmtree(final_folder)
            raise

        saved_images = get_candidate_images(final_folder)

        if len(saved_images) != len(temp_images):
            if final_folder.exists():
                shutil.rmtree(final_folder)

            raise RuntimeError("Not all candidate photos were saved")
        shutil.rmtree(source_folder)
        if candidate_key in _candidate_cache:
            _candidate_cache.remove(candidate_key)
        _candidate_locks.pop(candidate_key, None)
        # cleanup_candidate_temp(user_id, candidate_key)

    return {
        "status": "saved",
        "candidate_key": candidate_key,
        "original_name": extract_candidate_name(candidate_key),
        "final_name": final_name.strip(),
        "q_code": q_code,
        "category": normalized_category,
        "photos_count": len(temp_images),
        "folder_name": final_folder_name,
        "archivist": user_name,
    }


def skip_candidate(user_id: int, user_name: str, candidate_key: str) -> dict:
    with _candidate_locks_mutex:
        remove_stale_locks()

        lock = _candidate_locks.get(candidate_key)

        if lock is None:
            raise ValueError("Candidate is not locked")

        if lock["user_id"] != user_id:
            raise ValueError("Candidate is locked by another user")

        source_folder = get_candidate_source_folder(candidate_key)

        if not source_folder.exists() or not source_folder.is_dir():
            raise FileNotFoundError(f"Candidate source folder not found: {source_folder}")

        archivist_folder = SKIPPED_CANDIDATE_FOLDER / user_name
        skipped_folder = archivist_folder / candidate_key

        if skipped_folder.exists():
            raise FileExistsError(f"Skipped candidate folder already exists: {skipped_folder}")

        archivist_folder.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_folder), str(skipped_folder))
        if candidate_key in _candidate_cache:
            _candidate_cache.remove(candidate_key)
        _candidate_locks.pop(candidate_key, None)
        # cleanup_candidate_temp(user_id, candidate_key)

    return {
        "status": "skipped",
        "candidate_key": candidate_key,
        "original_name": extract_candidate_name(candidate_key),
        "q_code": extract_q_code(candidate_key),
        "folder_name": candidate_key,
        "archivist": user_name,
    }

