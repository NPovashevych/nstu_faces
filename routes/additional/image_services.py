from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PHOTOS = 12


def get_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []

    return sorted([file_path for file_path in folder.iterdir() if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS], key=lambda path: path.name.lower())


def get_free_photo_name(folder: Path, original_file_name: str) -> str:
    original_path = Path(original_file_name)
    stem = original_path.stem.strip() or "photo"
    suffix = original_path.suffix.lower()
    file_name = f"{stem}{suffix}"

    if not (folder / file_name).exists():
        return file_name

    number = 2

    while True:
        file_name = f"{stem}_{number}{suffix}"

        if not (folder / file_name).exists():
            return file_name

        number += 1


def add_photo(folder: Path, file_name: str, file_content: bytes) -> Path:
    if not file_name:
        raise ValueError("File name is empty")

    suffix = Path(file_name).suffix.lower()

    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image format. Allowed: jpg, jpeg, png, webp")

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Temporary folder not found: {folder}")

    if len(get_images(folder)) >= MAX_PHOTOS:
        raise ValueError(f"Maximum {MAX_PHOTOS} photos allowed")

    safe_file_name = Path(file_name).name
    destination_name = get_free_photo_name(folder, safe_file_name)
    destination = folder / destination_name
    destination.write_bytes(file_content)

    return destination


def delete_photo(folder: Path, file_name: str) -> Path:
    if not file_name:
        raise ValueError("File name is empty")

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Temporary folder not found: {folder}")

    safe_file_name = Path(file_name).name
    photo_path = folder / safe_file_name

    if not photo_path.exists() or not photo_path.is_file():
        raise FileNotFoundError(f"Photo not found: {safe_file_name}")

    if photo_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("File is not an image")

    photo_path.unlink()

    return photo_path
