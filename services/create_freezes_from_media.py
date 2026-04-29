import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
from scenedetect import open_video, SceneManager, ContentDetector
from sqlalchemy.orm import Session

from config import FREEZE_FOLDER
from db.session import SessionLocal
from db.enums import MediaType
from db.models import DBMedia
from crud.crud_freeze import create_freeze, get_freezes_by_media
from schemas.schemas_freeze import FreezeCreate


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/create_freezes_from_media.log", encoding="utf-8"),
    ],
)


SCENE_THRESHOLD = 60.0


def safe_name(name: str) -> str:
    bad_chars = '<>:"/\\|?*'
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name.strip()


def format_time_for_filename(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    minutes = total_ms // 60000
    sec = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{minutes:02d}_{sec:02d}_{ms:03d}"


def get_freeze_folder(media: DBMedia) -> Path:
    media_path = Path(media.media_path)
    folder_name = safe_name(media_path.stem)
    folder = Path(FREEZE_FOLDER) / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_video_scenes(video_path: Path):
    video = open_video(str(video_path))

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=SCENE_THRESHOLD))
    scene_manager.detect_scenes(video)

    scenes = scene_manager.get_scene_list()

    if not scenes:
        duration = video.duration.get_seconds()
        return [(0.0, duration)]

    return [
        (start.get_seconds(), end.get_seconds())
        for start, end in scenes
    ]


def extract_video_frame(video_path: Path, freeze_path: Path, start_sec: float) -> bool:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logging.warning(f"Cannot open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        logging.warning(f"Bad FPS for video: {video_path}")
        cap.release()
        return False

    frame_number = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logging.warning(f"Cannot read frame at {start_sec:.2f}s from {video_path}")
        return False

    return cv2.imwrite(str(freeze_path), frame)


def create_freezes_for_video(db: Session, media: DBMedia) -> int:
    video_path = Path(media.media_path)

    if not video_path.exists():
        logging.warning(f"Media file not found: {video_path}")
        return 0

    freeze_folder = get_freeze_folder(media)
    media_name = safe_name(video_path.stem)

    scenes = get_video_scenes(video_path)

    created = 0

    for start, end in scenes:
        time_part = format_time_for_filename(start)
        freeze_name = f"{media_name}_{time_part}.jpg"
        freeze_path = freeze_folder / freeze_name

        if not freeze_path.exists():
            ok = extract_video_frame(video_path, freeze_path, start)

            if not ok:
                continue

        freeze_create = FreezeCreate(
            time_in=start,
            time_out=end,
            freeze_path=str(freeze_path),
            media_id=media.id,
        )

        create_freeze(db, freeze_create)
        created += 1

    return created


def create_freeze_for_image(db: Session, media: DBMedia) -> int:
    image_path = Path(media.media_path)

    if not image_path.exists():
        logging.warning(f"Image file not found: {image_path}")
        return 0

    freeze_folder = get_freeze_folder(media)
    media_name = safe_name(image_path.stem)

    freeze_path = freeze_folder / f"{media_name}_0.jpg"

    if not freeze_path.exists():
        shutil.copy2(image_path, freeze_path)

    freeze_create = FreezeCreate(
        time_in=0.0,
        time_out=0.0,
        freeze_path=str(freeze_path),
        media_id=media.id,
    )

    create_freeze(db, freeze_create)
    return 1


def process_media(db: Session, media: DBMedia) -> int:
    existing_freezes = get_freezes_by_media(db, media.id)

    if existing_freezes:
        logging.info(
            f"Skip media_id={media.id}, freezes already exist: {len(existing_freezes)}"
        )
        return 0

    if media.media_type == MediaType.video:
        return create_freezes_for_video(db, media)

    if media.media_type == MediaType.image:
        return create_freeze_for_image(db, media)

    logging.warning(f"Unknown media_type for media_id={media.id}: {media.media_type}")
    return 0


def process_all_media():
    db = SessionLocal()

    total_media = 0
    total_freezes = 0

    try:
        medias = db.query(DBMedia).order_by(DBMedia.id).all()

        for media in medias:
            total_media += 1
            created = process_media(db, media)
            total_freezes += created

            if created:
                logging.info(f"media_id={media.id}: created freezes={created}")

        logging.info("--------------------------------")
        logging.info(f"Processed media: {total_media}")
        logging.info(f"Created freezes: {total_freezes}")

    finally:
        db.close()


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Start: {start}")

    process_all_media()

    finish = datetime.now()
    logging.info(f"Finished. Running time: {finish - start}")

