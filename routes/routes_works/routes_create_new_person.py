from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException

from db.session import get_db
from routes.additional.user_services import get_user
from routes.additional.new_person.database_search import find_person_in_db
from routes.classes.database_search import NewPersonDatabaseSearchRequest
from routes.additional.new_person.new_person_service import add_new_person_photo, delete_new_person_photo
from routes.additional.image_inspections import check_image_folder
from routes.additional.new_person.new_person_service import create_new_person_workspace, get_new_person_temp_folder


router = APIRouter(prefix="/new-person", tags=["New person"])

@router.post("/find-in-db")
def find_new_person_in_db(payload: NewPersonDatabaseSearchRequest, db: Session = Depends(get_db)):
    get_user(db, payload.user_id)
    return find_person_in_db(db=db, name=payload.name, q_code=payload.q_code)


@router.post("/workspace")
def create_new_person_workspace_route(user_id: int, db: Session = Depends(get_db)):
    get_user(db, user_id)

    return create_new_person_workspace(user_id)


@router.post("/photo")
async def add_new_person_photo_route(user_id: int = Form(...), workspace_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    get_user(db, user_id)

    try:
        file_content = await file.read()
        photo_path = add_new_person_photo(user_id=user_id, workspace_id=workspace_id, file_name=file.filename or "", file_content=file_content)

    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "ok",
        "file_name": photo_path.name,
        "url": f"/media-youtube-upload/new_persons/{user_id}/{workspace_id}/{photo_path.name}",
    }


@router.delete("/photo")
def delete_new_person_photo_route(user_id: int, workspace_id: str, file_name: str, db: Session = Depends(get_db)):
    get_user(db, user_id)

    try:
        photo_path = delete_new_person_photo(user_id=user_id, workspace_id=workspace_id, file_name=file_name)

    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "ok",
        "file_name": photo_path.name,
    }


@router.post("/check")
def check_new_person_photos(user_id: int, workspace_id: str, db: Session = Depends(get_db)):
    get_user(db, user_id)

    temp_folder = get_new_person_temp_folder(user_id, workspace_id)

    if not temp_folder.exists() or not temp_folder.is_dir():
        raise HTTPException(status_code=404, detail="New person temporary folder not found")

    try:
        result = check_image_folder(temp_folder, check_references=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot check photos: {e}")

    return {
        "status": "ok",
        "workspace_id": workspace_id,
        "check": result,
    }
