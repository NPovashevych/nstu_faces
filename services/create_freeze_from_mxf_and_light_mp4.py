import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scenedetect import open_video, SceneManager, ContentDetector
from sqlalchemy.exc import IntegrityError

from config import MXF_FOLDER, MP4_LIGHT_FOLDER, FREEZE_FOLDER_FROM_MXF
from db.session import SessionLocal
from db.enums import MediaSource, MediaType
from crud.crud_media import get_media_by_mxf_path, create_media
from crud.crud_freeze import create_freeze
from schemas.schemas_media import MediaCreate
from schemas.schemas_freeze import FreezeCreate


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/create_media_and_freezes_from_mxf.log", encoding="utf-8"),
    ],
)


SCENE_THRESHOLD = 30.0


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def safe_name(name: str) -> str:
    bad_chars = '<>:"/\\|?*'
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name.strip()


def parse_recorded_at(file_stem: str) -> datetime | None:
    match = re.match(r"^(\d{2})(\d{2})(\d{2})-", file_stem)

    if not match:
        return None

    day, month, year = match.groups()

    year = int("20" + year)
    month = int(month)
    day = int(day)

    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def format_time_for_filename(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    sec = total_seconds % 60
    tenth = int((seconds - total_seconds) * 10)

    return f"{minutes:02d}_{sec:02d}_{tenth}"


def create_light_mp4(mxf_path: Path, mp4_path: Path) -> bool:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = mp4_path.with_suffix(".tmp.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mxf_path),

        "-map", "0:v:0",
        "-map", "0:a:0?",

        "-vf", "yadif=0:-1:0,scale=-2:720",

        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-b:a", "96k",

        "-movflags", "+faststart",
        str(temp_path),
    ]

    logging.info(f"Create light MP4: {mxf_path.name} -> {mp4_path.name}")

    result = run_cmd(cmd)

    if result.returncode != 0:
        logging.error(f"FFmpeg MP4 failed: {mxf_path}")
        logging.error(result.stderr)

        if temp_path.exists():
            temp_path.unlink()

        return False

    temp_path.replace(mp4_path)
    return True


def get_video_scenes(mp4_path: Path) -> list[tuple[float, float]]:
    video = open_video(str(mp4_path))

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=SCENE_THRESHOLD,
            min_scene_len=15,
        )
    )

    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()

    if not scenes:
        duration = video.duration.get_seconds()
        return [(0.0, duration)]

    return [
        (start.get_seconds(), end.get_seconds())
        for start, end in scenes
    ]


def extract_freeze_from_mxf(mxf_path: Path, freeze_path: Path, start_sec: float) -> bool:
    freeze_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(mxf_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(freeze_path),
    ]

    result = run_cmd(cmd)

    if result.returncode != 0:
        logging.warning(f"Cannot extract freeze at {start_sec:.2f}s from {mxf_path.name}")
        logging.warning(result.stderr)
        return False

    return freeze_path.exists()


def get_or_create_media(db, mxf_path: Path, mp4_path: Path, recorded_at: datetime | None, user_id: int):
    existing = get_media_by_mxf_path(db, str(mxf_path))

    if existing:
        return existing

    media_create = MediaCreate(
        source=MediaSource.in_media,
        media_type=MediaType.video,
        mxf_path=str(mxf_path),
        mp4_path=str(mp4_path),
        recorded_at=recorded_at,
    )

    try:
        return create_media(db, media_create, user_id=user_id)

    except IntegrityError:
        db.rollback()
        return get_media_by_mxf_path(db, str(mxf_path))


def process_one_mxf(mxf_path: Path, user_id: int) -> dict:
    db = SessionLocal()

    try:
        media_name = safe_name(mxf_path.stem)

        mp4_path = Path(MP4_LIGHT_FOLDER) / f"{media_name}.mp4"
        freeze_folder = Path(FREEZE_FOLDER_FROM_MXF) / media_name

        recorded_at = parse_recorded_at(mxf_path.stem)

        if not mp4_path.exists():
            ok = create_light_mp4(mxf_path, mp4_path)

            if not ok:
                return {
                    "file": mxf_path.name,
                    "status": "error",
                    "media_id": None,
                    "freezes": 0,
                    "error": "mp4_failed",
                }

        media = get_or_create_media(
            db=db,
            mxf_path=mxf_path,
            mp4_path=mp4_path,
            recorded_at=recorded_at,
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

        scenes = get_video_scenes(mp4_path)

        created_freezes = 0

        for start, end in scenes:
            time_part = format_time_for_filename(start)
            freeze_name = f"{media_name}_{time_part}.jpg"
            freeze_path = freeze_folder / freeze_name

            if not freeze_path.exists():
                ok = extract_freeze_from_mxf(
                    mxf_path=mxf_path,
                    freeze_path=freeze_path,
                    start_sec=start,
                )

                if not ok:
                    continue

            freeze_create = FreezeCreate(
                time_in=start,
                time_out=end,
                freeze_path=str(freeze_path),
                media_id=media.id,
            )

            try:
                create_freeze(db, freeze_create)
                created_freezes += 1

            except IntegrityError:
                db.rollback()
                logging.info(f"Freeze already exists: {freeze_path.name}")

        return {
            "file": mxf_path.name,
            "status": "ok",
            "media_id": media.id,
            "freezes": created_freezes,
            "error": "",
        }

    except Exception as e:
        logging.exception(f"Unexpected error for {mxf_path}")
        return {
            "file": mxf_path.name,
            "status": "error",
            "media_id": None,
            "freezes": 0,
            "error": str(e),
        }

    finally:
        db.close()


def process_mxf_folder(user_id: int = 1):
    input_folder = Path(MXF_FOLDER)

    mxf_files = sorted(
        list(input_folder.glob("*.mxf"))
    )

    logging.info(f"Found MXF files: {len(mxf_files)}")

    total_freezes = 0
    errors = 0

    for mxf_path in mxf_files:
        result = process_one_mxf(mxf_path, user_id=user_id)

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

    logging.info("--------------------------------")
    logging.info(f"Processed MXF: {len(mxf_files)}")
    logging.info(f"Created freezes: {total_freezes}")
    logging.info(f"Errors: {errors}")


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Start: {start}")

    process_mxf_folder(user_id=1)

    finish = datetime.now()
    logging.info(f"Finished. Running time: {finish - start}")
