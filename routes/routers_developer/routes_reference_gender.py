from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from io import BytesIO

from PIL import Image
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.session import get_db
from db.enums import EmbeddingType
from db.models import DBEmbedding, DBPerson


router = APIRouter(prefix="/developer/reference-gender", tags=["developer reference gender"])

REFERENCE_PREVIEW_MAX_SIZE = 500
REFERENCE_PREVIEW_JPEG_QUALITY = 75

class ReferenceGenderUpdate(BaseModel):
    gender: str | None = None
    reviewed: bool = True


def get_reference_embeddings(db: Session, person_id: int):
    return (
        db.query(DBEmbedding)
        .filter(DBEmbedding.person_id == person_id, DBEmbedding.embedding_type == EmbeddingType.reference_face)
        .order_by(DBEmbedding.id)
        .all()
    )


def get_person_gender(embeddings):
    genders = []

    for embedding in embeddings:
        source = embedding.source or {}
        gender = source.get("gender")

        if gender in {"male", "female", "unknown"}:
            genders.append(gender)

    valid_genders = [gender for gender in genders if gender in {"male", "female"}]

    if not valid_genders:
        return "unknown"

    male_count = valid_genders.count("male")
    female_count = valid_genders.count("female")

    if male_count > female_count:
        return "male"

    if female_count > male_count:
        return "female"

    return "unknown"


def is_gender_reviewed(embeddings):
    if not embeddings:
        return False

    return all((embedding.source or {}).get("gender_reviewed") is True for embedding in embeddings)


@router.get("")
def list_reference_gender(
    reviewed: bool | None = Query(default=None),
    single_reference: bool | None = Query(default=None),
    person_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = (
        db.query(DBPerson)
        .join(DBEmbedding, DBEmbedding.person_id == DBPerson.id)
        .filter(DBEmbedding.embedding_type == EmbeddingType.reference_face)
        .distinct()
        .order_by(DBPerson.id)
    )

    if person_id is not None:
        query = query.filter(DBPerson.id == person_id)

    persons = query.all()
    result = []

    for person in persons:
        embeddings = get_reference_embeddings(db, person.id)

        if single_reference is True and len(embeddings) != 1:
            continue

        if single_reference is False and len(embeddings) == 1:
            continue

        person_reviewed = is_gender_reviewed(embeddings)

        if reviewed is not None and person_reviewed != reviewed:
            continue

        photos = []

        for embedding in embeddings:
            source = embedding.source or {}

            photos.append(
                {
                    "embedding_id": embedding.id,
                    "file_name": source.get("file_name"),
                    "image_url": f"/developer/reference-gender/image/{embedding.id}",
                    "gender": source.get("gender"),
                }
            )

        result.append(
            {
                "person_id": person.id,
                "name": person.name,
                "gender": get_person_gender(embeddings),
                "reviewed": person_reviewed,
                "reference_count": len(embeddings),
                "photos": photos,
            }
        )

    return {
        "count": len(result),
        "items": result,
    }


@router.get("/image/{embedding_id}")
def get_reference_image(embedding_id: int, db: Session = Depends(get_db)):
    embedding = (
        db.query(DBEmbedding)
        .filter(DBEmbedding.id == embedding_id, DBEmbedding.embedding_type == EmbeddingType.reference_face)
        .first()
    )

    if embedding is None:
        raise HTTPException(status_code=404, detail="Reference embedding not found.")

    source = embedding.source or {}
    file_path = source.get("file_path")

    if not file_path:
        raise HTTPException(status_code=404, detail="Reference image path not found.")

    image_path = Path(file_path)

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Reference image file not found: {file_path}")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((REFERENCE_PREVIEW_MAX_SIZE, REFERENCE_PREVIEW_MAX_SIZE))

            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=REFERENCE_PREVIEW_JPEG_QUALITY, optimize=True)
            buffer.seek(0)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot create reference preview: {e}")

    return StreamingResponse(buffer, media_type="image/jpeg")


@router.patch("/{person_id}")
def update_reference_gender(person_id: int, payload: ReferenceGenderUpdate, db: Session = Depends(get_db)):
    person = db.query(DBPerson).filter(DBPerson.id == person_id).first()

    if person is None:
        raise HTTPException(status_code=404, detail="Person not found.")

    embeddings = get_reference_embeddings(db, person_id)

    if not embeddings:
        raise HTTPException(status_code=404, detail="Reference embeddings not found.")

    if payload.gender is not None and payload.gender not in {"male", "female", "unknown"}:
        raise HTTPException(status_code=400, detail="Gender must be male, female or unknown.")

    reviewed_at = datetime.now().isoformat()

    for embedding in embeddings:
        source = dict(embedding.source or {})

        if payload.gender is not None:
            source["gender"] = payload.gender

        source["gender_reviewed"] = payload.reviewed

        if payload.reviewed:
            source["gender_reviewed_at"] = reviewed_at
        else:
            source.pop("gender_reviewed_at", None)

        embedding.source = source

    db.commit()

    return {
        "person_id": person.id,
        "name": person.name,
        "gender": get_person_gender(embeddings),
        "reviewed": is_gender_reviewed(embeddings),
        "reference_count": len(embeddings),
    }
