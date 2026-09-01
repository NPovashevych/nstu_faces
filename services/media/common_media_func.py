import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

from scenedetect import open_video, SceneManager, ContentDetector
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.models import DBFreeze


VIDEO_EXTENSIONS = {".mxf", ".mp4", ".mov", ".avi"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SCENE_THRESHOLD = 30.0
MIN_SCENE_LEN = 15


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


def get_video_duration_and_scenes(mp4_path: Path) -> tuple[float, list[tuple[float, float]]]:
    video = open_video(str(mp4_path))

    duration = float(video.duration.get_seconds())

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=SCENE_THRESHOLD,
            min_scene_len=MIN_SCENE_LEN,
        )
    )

    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()

    if not scenes:
        return duration, [(0.0, duration)]

    return duration, [(float(start.get_seconds()), float(end.get_seconds())) for start, end in scenes]


def format_time_for_filename(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    sec = total_seconds % 60
    tenth = int((seconds - total_seconds) * 10)

    return f"{minutes:02d}_{sec:02d}_{tenth}"


def extract_freeze_from_video(video_path: Path, freeze_path: Path, start_sec: float) -> bool:
    freeze_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(freeze_path),
    ]

    result = run_cmd(cmd)

    if result.returncode != 0:
        logging.warning(f"Cannot extract freeze at {start_sec:.2f}s from {video_path.name}")
        logging.warning(result.stderr)
        return False

    return freeze_path.exists()


def copy_image_as_freeze(image_path: Path, freeze_path: Path) -> bool:
    freeze_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(image_path, freeze_path)
        return freeze_path.exists()
    except Exception as e:
        logging.warning(f"Cannot copy image freeze: {image_path} -> {freeze_path}: {e}")
        return False


def get_or_create_freeze(db: Session, media_id: int, freeze_path: Path, time_in: float, time_out: float):
    freeze_path_str = str(freeze_path)
    existing = db.query(DBFreeze).filter(DBFreeze.freeze_path == freeze_path_str).first()

    if existing:
        return existing, False

    db_freeze = DBFreeze(time_in=time_in, time_out=time_out, freeze_path=freeze_path_str, media_id=media_id)
    db.add(db_freeze)

    try:
        db.commit()
        db.refresh(db_freeze)
        return db_freeze, True

    except IntegrityError:
        db.rollback()
        existing = db.query(DBFreeze) .filter(DBFreeze.freeze_path == freeze_path_str).first()
        return existing, False


def calculate_file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    sha = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)

    return sha.hexdigest()
