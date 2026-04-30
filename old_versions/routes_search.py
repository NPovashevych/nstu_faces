from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.session import get_db
from db.models import DBPerson, DBFace, DBFreeze, DBMedia

from services.config import FREEZE_FOLDER


router = APIRouter(prefix="/search", tags=["search"])


def make_image_url(freeze_path: str):
    relative_path = Path(freeze_path).relative_to(Path(FREEZE_FOLDER))
    return f"/freezes/{relative_path.as_posix()}"


@router.get("/person")
def search_person(name: str, db: Session = Depends(get_db)):
    """
    Пошук по ПІБ (частковий)
    """

    persons = (
        db.query(DBPerson)
        .filter(DBPerson.name.ilike(f"%{name}%"))
        .all()
    )

    if not persons:
        raise HTTPException(status_code=404, detail="Персону не знайдено")

    results = []

    for person in persons:
        faces = (
            db.query(DBFace)
            .join(DBFreeze, DBFace.freeze_id == DBFreeze.id)
            .join(DBMedia, DBFreeze.media_id == DBMedia.id)
            .filter(DBFace.person_id == person.id)
            .order_by(DBFreeze.time_in)
            .all()
        )

        medias_map = {}

        for face in faces:
            freeze = face.freeze
            media = freeze.media

            if media.id not in medias_map:
                medias_map[media.id] = {
                    "media_id": media.id,
                    "media_name": Path(media.media_path).name,
                    "frames": []
                }

            medias_map[media.id]["frames"].append({
                "image_url": make_image_url(freeze.freeze_path),
                "time_in": freeze.time_in,
                "time_out": freeze.time_out,
                "bbox": face.bbox,
                "quality": face.quality,
            })

        results.append({
            "person_id": person.id,
            "name": person.name,
            "q_code": person.q_code,
            "link": person.link,
            "medias": list(medias_map.values())
        })

    return {
        "query": name,
        "results": results
    }
