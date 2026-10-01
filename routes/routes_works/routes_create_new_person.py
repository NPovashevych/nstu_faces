from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException

from db.session import get_db
from routes.additional.user_services import get_user
from routes.additional.new_person.database_search import find_person_in_db
from routes.classes.class_for_route_new_person import NewPersonDatabaseSearchRequest, NewPersonCheckRequest, NewPersonSaveRequest, NewPersonWorkspaceRequest
from routes.additional.new_person.new_person_service import add_new_person_photo, delete_new_person_photo, create_new_person_workspace
from routes.additional.new_person.new_person_service import save_new_person, get_new_person_temp_folder
from routes.additional.image_inspections import check_image_folder
from routes.additional.new_person.validation import validate_ukrainian_pseudonym


router = APIRouter(prefix="/new-person", tags=["New person"])

@router.post("/find-in-db")
def find_new_person_in_db(payload: NewPersonDatabaseSearchRequest, db: Session = Depends(get_db)):
    get_user(db, payload.user_id)

    try:
        validate_ukrainian_pseudonym(payload.pseudonym)
        return find_person_in_db(db=db, name=payload.name, q_code=payload.q_code)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@router.post("/workspace")
def create_workspace(request: NewPersonWorkspaceRequest, db: Session = Depends(get_db)):
    get_user(db, request.user_id)

    try:
        return create_new_person_workspace(user_id=request.user_id, name=request.name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


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
def check_new_person(request: NewPersonCheckRequest, db: Session = Depends(get_db)):
    get_user(db, request.user_id)

    temp_folder = get_new_person_temp_folder(request.user_id, request.workspace_id)

    if not temp_folder.exists() or not temp_folder.is_dir():
        raise HTTPException(status_code=404, detail="New person workspace not found")

    try:
        database_search = find_person_in_db(db=db, name=request.name, q_code=request.q_code)
        result = check_image_folder(temp_folder, check_references=True, check_unknown=True)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    return {
        "status": "ok",
        "workspace_id": request.workspace_id,
        "database_search": database_search,
        "check": result,
    }


@router.post("/save")
def save_new_person_route(request: NewPersonSaveRequest, db: Session = Depends(get_db)):
    get_user(db, request.user_id)

    try:
        result = save_new_person(
            user_id=request.user_id,
            workspace_id=request.workspace_id,
            name=request.name,
            pseudonym=request.pseudonym,
            q_code=request.q_code,
            category=request.category,
            like_unknown=request.like_unknown,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except (ValueError, FileExistsError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    return result
