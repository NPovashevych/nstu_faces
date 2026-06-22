import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import MP4_FOLDER, MXF_FOLDER
from db.session import SessionLocal
from db.enums import MediaSource, MediaType
from crud.crud_media import get_media_by_path, create_media
from schemas.schemas_media import MediaCreate


MAX_WORKERS = 2


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../services/logs/add_media_from_mxf_v2.log", encoding="utf-8"),
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

    return True


def extract_video_fast_crf20(mxf_path: Path, mp4_path: Path) -> bool:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = mp4_path.with_suffix(".tmp.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mxf_path),

        "-map", "0:v:0",
        "-map", "0:a:0?",

        "-vf", "yadif=0:-1:0",

        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-b:a", "128k",

        "-movflags", "+faststart",
        str(temp_path),
    ]

    logging.info(f"Converting B_fast_crf20: {mxf_path.name} -> {mp4_path.name}")

    result = run_cmd(cmd)

    if result.returncode != 0:
        logging.error(f"FFmpeg failed for {mxf_path}")
        logging.error(result.stderr)
        if temp_path.exists():
            temp_path.unlink()
        return False

    if not check_video_quality(temp_path):
        logging.error(f"Converted MP4 failed quality check: {temp_path}")
        if temp_path.exists():
            temp_path.unlink()
        return False

    temp_path.replace(mp4_path)
    return True


def add_media_record(db: Session, mp4_path: Path, user_id: int, source: MediaSource):
    media_path = str(mp4_path)

    existing = get_media_by_path(db, media_path)
    if existing:
        return existing

    media_create = MediaCreate(
        source=source,
        media_type=MediaType.video,
        media_path=media_path,
    )

    try:
        db_media = create_media(db, media_create, user_id=user_id)
        db.commit()
        logging.info(f"Added to DB: {mp4_path.name} id={db_media.id}")
        return db_media

    except IntegrityError:
        db.rollback()
        existing = get_media_by_path(db, media_path)
        if existing:
            logging.info(f"Already added by another process: {mp4_path.name}")
            return existing
        raise


def process_one_mxf(mxf_path: Path, output_folder: Path, user_id: int, source: MediaSource):
    db = SessionLocal()

    try:
        mp4_path = output_folder / f"{mxf_path.stem}.mp4"

        db_media = get_media_by_path(db, str(mp4_path))

        if db_media and mp4_path.exists():
            return {
                "file": mxf_path.name,
                "status": "skipped",
                "created_mp4": 0,
                "added_to_db": 0,
                "error": "",
            }

        if not mp4_path.exists():
            ok = extract_video_fast_crf20(mxf_path, mp4_path)

            if not ok:
                return {
                    "file": mxf_path.name,
                    "status": "error",
                    "created_mp4": 0,
                    "added_to_db": 0,
                    "error": "convert_failed",
                }

            created_mp4 = 1

        else:
            logging.info(f"MP4 exists on disk, checking quality: {mp4_path.name}")

            if not check_video_quality(mp4_path):
                logging.warning(f"Existing MP4 is bad, recreating: {mp4_path.name}")

                ok = extract_video_fast_crf20(mxf_path, mp4_path)

                if not ok:
                    return {
                        "file": mxf_path.name,
                        "status": "error",
                        "created_mp4": 0,
                        "added_to_db": 0,
                        "error": "reconvert_failed",
                    }

                created_mp4 = 1
            else:
                created_mp4 = 0

        existing_after = get_media_by_path(db, str(mp4_path))

        if not existing_after:
            add_media_record(db, mp4_path, user_id=user_id, source=source)
            added_to_db = 1
        else:
            added_to_db = 0

        return {
            "file": mxf_path.name,
            "status": "ok",
            "created_mp4": created_mp4,
            "added_to_db": added_to_db,
            "error": "",
        }

    except Exception as e:
        logging.exception(f"Unexpected error for {mxf_path}")
        return {
            "file": mxf_path.name,
            "status": "error",
            "created_mp4": 0,
            "added_to_db": 0,
            "error": str(e),
        }

    finally:
        db.close()


def process_mxf_folder_parallel(
    input_folder: Path,
    output_folder: Path,
    user_id: int,
    source: MediaSource,
):
    mxf_files = sorted(input_folder.glob("*.mxf"))

    logging.info(f"Found MXF files: {len(mxf_files)}")
    logging.info(f"MAX_WORKERS: {MAX_WORKERS}")

    created_mp4 = 0
    added_to_db = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                process_one_mxf,
                mxf_path,
                output_folder,
                user_id,
                source,
            )
            for mxf_path in mxf_files
        ]

        for future in as_completed(futures):
            result = future.result()

            logging.info(
                f"{result['status'].upper()}: {result['file']} {result.get('error', '')}"
            )

            created_mp4 += result["created_mp4"]
            added_to_db += result["added_to_db"]

            if result["status"] == "skipped":
                skipped += 1
            elif result["status"] == "error":
                errors += 1

    logging.info("--------------------------------")
    logging.info(f"Created MP4: {created_mp4}")
    logging.info(f"Added to DB: {added_to_db}")
    logging.info(f"Skipped: {skipped}")
    logging.info(f"Errors: {errors}")


if __name__ == "__main__":
    start = datetime.now()
    logging.info(f"Start: {start}")

    process_mxf_folder_parallel(
        input_folder=Path(MXF_FOLDER),
        output_folder=Path(MP4_FOLDER),
        user_id=1,
        source=MediaSource.in_media,
    )

    finish = datetime.now()
    logging.info(f"Finished. Running time: {finish - start}")
