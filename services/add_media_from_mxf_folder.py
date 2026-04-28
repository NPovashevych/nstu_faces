import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from config import MP4_FOLDER, MXF_FOLDER
from db.session import SessionLocal
from db.enums import MediaSource, MediaType
from crud.crud_media import get_media_by_path, create_media
from schemas.schemas_media import MediaCreate


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/add_media_from_mxf_folder.log", encoding="utf-8"),
    ],
)


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def ffprobe_info(video_path: Path) -> dict | None:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    result = run_cmd(cmd)

    if result.returncode != 0:
        logging.warning(f"ffprobe error for {video_path}: {result.stderr}")
        return None

    try:
        return json.loads(result.stdout)
    except Exception as e:
        logging.warning(f"Cannot parse ffprobe json for {video_path}: {e}")
        return None


def has_video_stream(info: dict) -> bool:
    return any(stream.get("codec_type") == "video" for stream in info.get("streams", []))


def has_audio_stream(info: dict) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in info.get("streams", []))


def check_playable(video_path: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(video_path),
        "-f", "null",
        "-",
    ]

    result = run_cmd(cmd)

    if result.returncode != 0:
        logging.warning(f"File is not playable: {video_path}")
        logging.warning(result.stderr)

    return result.returncode == 0


def check_video_quality(video_path: Path) -> bool:
    info = ffprobe_info(video_path)

    if info is None:
        return False

    if not has_video_stream(info):
        logging.warning(f"No video stream: {video_path}")
        return False

    if not has_audio_stream(info):
        logging.warning(f"No audio stream: {video_path}")

    duration = float(info.get("format", {}).get("duration", 0) or 0)

    if duration <= 0:
        logging.warning(f"Bad duration: {video_path}")
        return False

    return check_playable(video_path)


def extract_video_high_quality(mxf_path: Path, mp4_path: Path) -> bool:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mxf_path),

        "-map", "0:v:0",
        "-map", "0:a:0?",

        # легке покращення для архівного/динамічного відео
        "-vf", "yadif=0:-1:0,unsharp=3:3:0.4:3:3:0.0",

        # якісне mp4, місце не економимо
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "15",
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-b:a", "320k",

        "-movflags", "+faststart",
        str(mp4_path),
    ]

    logging.info(f"Converting: {mxf_path.name} -> {mp4_path.name}")

    result = run_cmd(cmd)

    if result.returncode != 0:
        logging.error(f"FFmpeg failed for {mxf_path}")
        logging.error(result.stderr)
        return False

    return check_video_quality(mp4_path)


def add_media_record(db: Session, mp4_path: Path, user_id: int, source: MediaSource):
    media_path = str(mp4_path)

    existing = get_media_by_path(db, media_path)

    if existing:
        logging.info(f"Already in DB: {mp4_path.name}")
        return existing

    media_create = MediaCreate(
        source=source,
        media_type=MediaType.video,
        media_path=media_path,
    )

    db_media = create_media(db, media_create, user_id=user_id)

    logging.info(f"Added to DB: {mp4_path.name} id={db_media.id}")

    return db_media


def process_mxf_folder(
    input_folder: Path,
    output_folder: Path,
    user_id: int,
    source: MediaSource,
):
    db = SessionLocal()

    created_mp4 = 0
    added_to_db = 0
    skipped = 0
    errors = 0

    try:
        for mxf_path in sorted(input_folder.glob("*.mxf")):
            mp4_path = output_folder / f"{mxf_path.stem}.mp4"

            db_media = get_media_by_path(db, str(mp4_path))

            if db_media and mp4_path.exists():
                logging.info(f"Skip, already exists in DB and disk: {mp4_path.name}")
                skipped += 1
                continue

            if not mp4_path.exists():
                ok = extract_video_high_quality(mxf_path, mp4_path)

                if not ok:
                    errors += 1
                    continue

                created_mp4 += 1
            else:
                logging.info(f"MP4 exists on disk, checking quality: {mp4_path.name}")

                if not check_video_quality(mp4_path):
                    logging.warning(f"Existing MP4 is bad, recreating: {mp4_path.name}")

                    ok = extract_video_high_quality(mxf_path, mp4_path)

                    if not ok:
                        errors += 1
                        continue

                    created_mp4 += 1

            existing_after = get_media_by_path(db, str(mp4_path))

            if not existing_after:
                add_media_record(db, mp4_path, user_id=user_id, source=source)
                added_to_db += 1
            else:
                skipped += 1

        logging.info("--------------------------------")
        logging.info(f"Created MP4: {created_mp4}")
        logging.info(f"Added to DB: {added_to_db}")
        logging.info(f"Skipped: {skipped}")
        logging.info(f"Errors: {errors}")

    finally:
        db.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-folder",
        required=True,
        help="Папка з MXF файлами",
    )

    parser.add_argument(
        "--output-folder",
        default=str(MP4_FOLDER),
        help="Папка для MP4 файлів",
    )

    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="ID користувача, який додає медіа",
    )

    parser.add_argument(
        "--source",
        default="in_media",
        choices=["in_media", "media_teca", "user_upload"],
        help="Джерело медіа",
    )

    return parser.parse_args()


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Start: {start}")

    # input_path = input("Введи шлях до папки з MXF: ").strip('" ')
    # output_path = input("Введи шлях до папки для MP4 (Enter = default): ").strip('" ')

    input_folder = MXF_FOLDER
    output_folder = MP4_FOLDER
    user_id = 1
    source = MediaSource.in_media

    # args = parse_args()

    process_mxf_folder(
        input_folder=Path(input_folder),
        output_folder=Path(output_folder),
        user_id=user_id,
        source=source,
    )

    finish = datetime.now()
    logging.info(f"Finished. Running time: {finish - start}")
