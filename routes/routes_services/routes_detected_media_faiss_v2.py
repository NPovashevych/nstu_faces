import logging
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
import yt_dlp
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from db.models import DBPerson
from db.session import get_db

from commons.common_face import analyze_faces, get_clip_cached, get_insightface, load_face_categories
from commons.common_scene import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, extract_freeze_from_video, get_video_duration_and_scenes
from commons.common_search import get_confidence_marks, normalize_bbox, normalize_person
from services.faiss.faiss_face_index import REFERENCE_FACE_INDEX, UNKNOWN_FACE_INDEX, ensure_faiss_indexes
from services.config import TEMPORARY_FREEZES_FOLDER
from commons.commons_base import make_image_url


router = APIRouter(prefix="/detect-media", tags=["detect media"])


YOUTUBE_FORMAT = "bestvideo[ext=mp4][vcodec^=avc1][height<=720]/bestvideo[ext=mp4][height<=720]/bestvideo"


def detect_media_type(file_name: str | None, content_type: str | None) -> str:
    suffix = Path(file_name or "").suffix.lower()
    content_type = (content_type or "").lower()

    if suffix in IMAGE_EXTENSIONS or content_type.startswith("image/"):
        return "image"

    if suffix in VIDEO_EXTENSIONS or content_type.startswith("video/"):
        return "video"

    raise HTTPException(status_code=400, detail="Не вдалося визначити тип медіа. Підтримуються зображення та відео.")


def download_youtube_video(url: str, destination_folder: Path) -> tuple[Path, str]:
    options = {
        "format": YOUTUBE_FORMAT,
        "outtmpl": str(destination_folder / "%(id)s.%(ext)s"),
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)

        video_id = info.get("id")
        extension = info.get("ext") or "mp4"
        title = info.get("title") or video_id or "youtube_video"

        if not video_id:
            raise HTTPException(status_code=400, detail="Не вдалося визначити YouTube video ID.")

        video_path = destination_folder / f"{video_id}.{extension}"

        if not video_path.exists() or video_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Не вдалося завантажити відео з YouTube.")

        return video_path, title

    except yt_dlp.utils.DownloadError as error:
        raise HTTPException(status_code=400, detail=f"Не вдалося завантажити відео з YouTube: {error}") from error


def load_image_from_path(image_path: Path):
    img = cv2.imread(str(image_path))

    if img is None:
        raise HTTPException(status_code=400, detail="Не вдалося прочитати зображення.")

    return img


def load_image_from_bytes(file_bytes: bytes):
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Не вдалося прочитати зображення.")

    return img


def normalize_detected_face(db: Session, result: dict):
    person = None

    if result.get("person_id") is not None:
        person = db.query(DBPerson).filter(DBPerson.id == result["person_id"]).first()

    category = result.get("category")
    gender = result.get("gender")
    confidence = result.get("confidence")
    bbox_data = normalize_bbox(result["bbox"])

    status = result["status"]

    if status == "known":
        frame_color = "green"
    elif status == "unknown_identifiable":
        frame_color = "red"
    elif status == "low_quality":
        frame_color = "gray"
    elif status == "service_category":
        frame_color = "orange"
    else:
        frame_color = "gray"

    return {
        "status": result["status"],
        "face_index": result["face_index"],
        "bbox": bbox_data["bbox"],
        "bbox_draw": bbox_data["bbox_draw"],
        "frame_color": frame_color,
        "det_score": result["det_score"],
        "distance": result["distance"],
        "category": category.name if category else None,
        "category_group": category.code if category else None,
        "category_score": result["category_score"],
        "quality": result["quality"],
        "gender": gender.value if gender else None,
        "confidence": confidence,
        "confidence_marks": get_confidence_marks(confidence),
        "analysis": result["analysis"] or {},
        "person": normalize_person(person) if person else None,
    }


def prepare_face_analysis(db: Session):
    face_model = get_insightface()
    clip_model, clip_preprocess, clip_text_features, clip_prompt_categories = get_clip_cached()

    ensure_faiss_indexes(db)
    face_categories = load_face_categories(db)

    return face_model, clip_model, clip_preprocess, clip_text_features, clip_prompt_categories, face_categories


def analyze_image(db: Session, img, face_model, clip_model, clip_preprocess, clip_text_features, clip_prompt_categories, face_categories) -> list[dict]:
    results = analyze_faces(
        img=img,
        face_model=face_model,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_text_features=clip_text_features,
        clip_prompt_categories=clip_prompt_categories,
        reference_index=REFERENCE_FACE_INDEX,
        unknown_index=UNKNOWN_FACE_INDEX,
        face_categories=face_categories,
    )

    return [normalize_detected_face(db=db, result=result) for result in results if result["status"] != "skipped_low_det_score"]


def build_image_response(source_type: str, file_name: str, analysis_id: str, image_url: str, faces: list[dict]) -> dict:
    return {
        "mode": "image",
        "source_type": source_type,
        "analysis_id": analysis_id,
        "file_name": file_name,
        "summary": {
            "frames_count": 1,
            "faces_count": len(faces),
        },
        "frames": [
            {
                "frame_index": 0,
                "image_url": image_url,
                "time_in": 0.0,
                "time_out": 0.0,
                "faces": faces,
            }
        ],
    }


def analyze_video(db: Session, video_path: Path, source_type: str, file_name: str, face_model, clip_model, clip_preprocess, clip_text_features, clip_prompt_categories, face_categories) -> dict:
    duration, scenes = get_video_duration_and_scenes(video_path)

    frames = []
    total_faces = 0

    analysis_id = uuid.uuid4().hex
    frames_folder = Path(TEMPORARY_FREEZES_FOLDER) / analysis_id
    frames_folder.mkdir(parents=True, exist_ok=True)

    for frame_index, (time_in, time_out) in enumerate(scenes):
        freeze_name = f"frame_{frame_index:06d}.jpg"
        freeze_path = frames_folder / freeze_name
        extracted = extract_freeze_from_video(video_path=video_path, freeze_path=freeze_path, start_sec=time_in)
        if not extracted:
            logging.warning(f"Cannot extract temporary frame | video={file_name} | frame_index={frame_index} | time_in={time_in}")
            continue
        img = load_image_from_path(freeze_path)
        faces = analyze_image(db=db, img=img, face_model=face_model, clip_model=clip_model, clip_preprocess=clip_preprocess, clip_text_features=clip_text_features, clip_prompt_categories=clip_prompt_categories, face_categories=face_categories)
        total_faces += len(faces)
        frames.append({"frame_index": frame_index, "image_url": make_image_url(str(freeze_path)), "time_in": round(time_in, 3), "time_out": round(time_out, 3), "faces": faces})

    return {
        "mode": "video",
        "source_type": source_type,
        "analysis_id": analysis_id,
        "file_name": file_name,
        "duration": round(duration, 3),
        "summary": {"scenes_count": len(scenes), "frames_count": len(frames), "faces_count": total_faces},
        "frames": frames,
    }


@router.post("/analyze")
async def detect_media(file: UploadFile | None = File(default=None), url: str | None = Form(default=None), db: Session = Depends(get_db)):
    if file is None and not url:
        raise HTTPException(status_code=400, detail="Передайте файл або посилання на YouTube.")

    if file is not None and url:
        raise HTTPException(status_code=400, detail="Передайте щось одне: файл або посилання на YouTube.")

    face_model, clip_model, clip_preprocess, clip_text_features, clip_prompt_categories, face_categories = prepare_face_analysis(db)

    if file is not None:
        file_name = file.filename or "uploaded_media"
        media_type = detect_media_type(file_name=file_name, content_type=file.content_type)

        if media_type == "image":
            file_bytes = await file.read()

            if not file_bytes:
                raise HTTPException(status_code=400, detail="Файл порожній.")

            img = load_image_from_bytes(file_bytes)
            analysis_id = uuid.uuid4().hex
            image_folder = Path(TEMPORARY_FREEZES_FOLDER) / analysis_id
            image_folder.mkdir(parents=True, exist_ok=True)

            suffix = Path(file_name).suffix.lower() or ".jpg"
            image_path = image_folder / f"image{suffix}"

            with image_path.open("wb") as destination:
                destination.write(file_bytes)

            faces = analyze_image(db=db, img=img, face_model=face_model, clip_model=clip_model, clip_preprocess=clip_preprocess, clip_text_features=clip_text_features, clip_prompt_categories=clip_prompt_categories, face_categories=face_categories)
            image_url = make_image_url(str(image_path))

            return build_image_response(source_type="file", file_name=file_name, analysis_id=analysis_id, image_url=image_url, faces=faces)

        suffix = Path(file_name).suffix.lower() or ".mp4"

        with tempfile.TemporaryDirectory(prefix="nstu_detect_media_") as temp_folder_value:
            video_path = Path(temp_folder_value) / f"uploaded{suffix}"

            with video_path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    destination.write(chunk)

            if not video_path.exists() or video_path.stat().st_size == 0:
                raise HTTPException(status_code=400, detail="Файл порожній.")

            return analyze_video(db=db, video_path=video_path, source_type="file", file_name=file_name, face_model=face_model, clip_model=clip_model, clip_preprocess=clip_preprocess, clip_text_features=clip_text_features, clip_prompt_categories=clip_prompt_categories, face_categories=face_categories)

    with tempfile.TemporaryDirectory(prefix="nstu_detect_youtube_") as temp_folder_value:
        temp_folder = Path(temp_folder_value)
        video_path, video_title = download_youtube_video(url=url.strip(), destination_folder=temp_folder)

        return analyze_video(db=db, video_path=video_path, source_type="youtube", file_name=video_title, face_model=face_model, clip_model=clip_model, clip_preprocess=clip_preprocess, clip_text_features=clip_text_features, clip_prompt_categories=clip_prompt_categories, face_categories=face_categories)

