import logging
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy.exc import IntegrityError

from db.session import SessionLocal
from db.enums import MediaType
from crud.crud_media import get_media_by_mxf_path, create_media
from schemas.schemas_media import MediaCreate
from services.config import TEST_FOLDER, TEST_MP4_LIGHT_FOLDER, TEST_FREEZE_FOLDER

from services.media.common_media_func import (
    IMAGE_EXTENSIONS,
    safe_name,
    create_light_mp4,
    get_video_duration_and_scenes,
    extract_freeze_from_video,
    copy_image_as_freeze,
    format_time_for_filename,
    get_or_create_freeze,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../logs/add_test_media.log", encoding="utf-8"),
    ],
)


TEST_SOURCE_ID = 1
TEST_USER_ID = 1


def get_test_files(test_folder: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in test_folder.iterdir()
        if file_path.is_file()
        and (
            file_path.suffix.lower() == ".mxf"
            or file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )


def get_or_create_test_media(db, original_path: Path, mp4_path: Path, media_type: MediaType, duration: float, user_id: int):
    existing = get_media_by_mxf_path(db, str(original_path))

    if existing:
        return existing

    media_create = MediaCreate(
        source_id=TEST_SOURCE_ID,
        user_id=user_id,
        media_type=media_type,
        mxf_path=str(original_path),
        mp4_path=str(mp4_path),
        duration=duration,
        recorded_at=datetime.now(),
    )

    try:
        return create_media(db, media_create)

    except IntegrityError:
        db.rollback()
        return get_media_by_mxf_path(db, str(original_path))


def process_test_mxf(db, mxf_path: Path, user_id: int) -> dict:
    media_name = safe_name(mxf_path.stem)

    mp4_path = Path(TEST_MP4_LIGHT_FOLDER) / f"{media_name}.mp4"
    freeze_folder = Path(TEST_FREEZE_FOLDER) / media_name

    if not mp4_path.exists():
        ok = create_light_mp4(
            mxf_path=mxf_path,
            mp4_path=mp4_path,
        )

        if not ok:
            return {
                "file": mxf_path.name,
                "status": "error",
                "media_id": None,
                "freezes": 0,
                "error": "mp4_failed",
            }

    duration, scenes = get_video_duration_and_scenes(mp4_path)

    media = get_or_create_test_media(
        db=db,
        original_path=mxf_path,
        mp4_path=mp4_path,
        media_type=MediaType.video,
        duration=duration,
        user_id=user_id,
    )

    if media is None:
        return {
            "file": mxf_path.name,
            "status": "error",
            "media_id": None,
            "freezes": 0,
            "error": "media_create_failed",
        }

    created_freezes = 0

    for start, end in scenes:
        time_part = format_time_for_filename(start)
        freeze_name = f"{media_name}_{time_part}.jpg"
        freeze_path = freeze_folder / freeze_name

        if not freeze_path.exists():
            ok = extract_freeze_from_video(
                video_path=mxf_path,
                freeze_path=freeze_path,
                start_sec=start,
            )

            if not ok:
                continue

        _, created = get_or_create_freeze(
            db=db,
            media_id=media.id,
            freeze_path=freeze_path,
            time_in=start,
            time_out=end,
        )

        if created:
            created_freezes += 1

    return {
        "file": mxf_path.name,
        "status": "ok",
        "media_id": media.id,
        "freezes": created_freezes,
        "error": "",
    }


def process_test_image(db, image_path: Path, user_id: int) -> dict:
    media_name = safe_name(image_path.stem)

    freeze_folder = Path(TEST_FREEZE_FOLDER) / media_name
    freeze_path = freeze_folder / f"{media_name}.jpg"

    media = get_or_create_test_media(
        db=db,
        original_path=image_path,
        mp4_path=image_path,
        media_type=MediaType.image,
        duration=0.0,
        user_id=user_id,
    )

    if media is None:
        return {
            "file": image_path.name,
            "status": "error",
            "media_id": None,
            "freezes": 0,
            "error": "media_create_failed",
        }

    if not freeze_path.exists():
        ok = copy_image_as_freeze(
            image_path=image_path,
            freeze_path=freeze_path,
        )

        if not ok:
            return {
                "file": image_path.name,
                "status": "error",
                "media_id": media.id,
                "freezes": 0,
                "error": "image_freeze_failed",
            }

    _, created = get_or_create_freeze(
        db=db,
        media_id=media.id,
        freeze_path=freeze_path,
        time_in=0.0,
        time_out=0.0,
    )

    return {
        "file": image_path.name,
        "status": "ok",
        "media_id": media.id,
        "freezes": 1 if created else 0,
        "error": "",
    }


def process_one_test_file(file_path: Path, user_id: int) -> dict:
    db = SessionLocal()

    try:
        suffix = file_path.suffix.lower()

        if suffix == ".mxf":
            return process_test_mxf(
                db=db,
                mxf_path=file_path,
                user_id=user_id,
            )

        if suffix in IMAGE_EXTENSIONS:
            return process_test_image(
                db=db,
                image_path=file_path,
                user_id=user_id,
            )

        return {
            "file": file_path.name,
            "status": "skip",
            "media_id": None,
            "freezes": 0,
            "error": "unsupported_test_file_type",
        }

    except Exception as e:
        logging.exception(f"Unexpected error for {file_path}")
        return {
            "file": file_path.name,
            "status": "error",
            "media_id": None,
            "freezes": 0,
            "error": str(e),
        }

    finally:
        db.close()


def process_test_media_folder(user_id: int = TEST_USER_ID):
    input_folder = Path(TEST_FOLDER)

    files = get_test_files(input_folder)

    logging.info(f"Found test media files: {len(files)}")

    total_freezes = 0
    errors = 0
    skipped = 0

    for file_path in files:
        result = process_one_test_file(
            file_path=file_path,
            user_id=user_id,
        )

        logging.info(
            f"{result['status'].upper()}: "
            f"{result['file']} | "
            f"media_id={result['media_id']} | "
            f"freezes={result['freezes']} | "
            f"{result['error']}"
        )

        total_freezes += result["freezes"]

        if result["status"] == "error":
            errors += 1

        if result["status"] == "skip":
            skipped += 1

    logging.info("--------------------------------")
    logging.info(f"Processed test files: {len(files)}")
    logging.info(f"Created freezes: {total_freezes}")
    logging.info(f"Skipped: {skipped}")
    logging.info(f"Errors: {errors}")


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Start: {start}")

    process_test_media_folder(user_id=TEST_USER_ID)

    finish = datetime.now()
    logging.info(f"Finished. Running time: {finish - start}")
