from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from db.session import get_db
from routes.additional.candidates.candidate_service import get_active_candidate, take_candidate, add_candidate_photo, delete_candidate_photo
from routes.additional.candidates.candidate_service import save_candidate, skip_candidate, release_candidate
from routes.additional.candidates.candidate_service import touch_candidate, get_candidate_temp_folder
from routes.additional.image_inspections import check_image_folder
from routes.additional.user_services import get_user
from routes.classes.class_for_route_candidates import CandidateSaveRequest, CandidateUserRequest, CandidateSkipRequest, CandidateCancelRequest, CandidateCheckRequest

router = APIRouter(prefix="/candidates", tags=["candidates"])


@router.get("/active")
def active_candidate(user_id: int, db: Session = Depends(get_db)):
    user = get_user(db, user_id)

    try:
        candidate = get_active_candidate(user.id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot get active candidate: {e}")

    if candidate is None:
        return {"status": "empty", "candidate": None}

    return {"status": "active", "candidate": candidate}


@router.post("/take")
def take_next_candidate(payload: CandidateUserRequest, db: Session = Depends(get_db)):
    user = get_user(db, payload.user_id)
    try:
        candidate = take_candidate(user.id)

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot take candidate: {e}")

    if candidate is None:
        return {
            "status": "empty",
            "message": "Вільних кандидатів немає.",
            "candidate": None,
        }

    return {
        "status": "active",
        "candidate": candidate,
    }


@router.post("/photo")
async def upload_candidate_photo(user_id: int = Form(...), candidate_key: str = Form(...), photo: UploadFile = File(...), db: Session = Depends(get_db)):
    user = get_user(db, user_id)

    if not photo.filename:
        raise HTTPException(status_code=400, detail="File name is empty")

    try:
        file_content = await photo.read()

        if not file_content:
            raise HTTPException(status_code=400, detail="File is empty")

        candidate = add_candidate_photo(user_id=user.id, candidate_key=candidate_key, file_name=photo.filename, file_content=file_content)

    except HTTPException:
        raise

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot upload candidate photo: {e}")

    finally:
        await photo.close()

    return {
        "status": "ok",
        "candidate": candidate,
    }

@router.delete("/photo")
def remove_candidate_photo(user_id: int, candidate_key: str, file_name: str, db: Session = Depends(get_db)):
    user = get_user(db, user_id)

    try:
        candidate = delete_candidate_photo(user_id=user.id, candidate_key=candidate_key, file_name=file_name)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot delete candidate photo: {e}")

    return {
        "status": "ok",
        "candidate": candidate,
    }


@router.post("/save")
def save_current_candidate(payload: CandidateSaveRequest, db: Session = Depends(get_db)):
    user = get_user(db, payload.user_id)
    try:
        result = save_candidate(user_id=user.id, user_name=user.name, candidate_key=payload.candidate_key, final_name=payload.final_name, category=payload.category)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot save candidate: {e}")

    return result

@router.post("/skip")
def skip_current_candidate(payload: CandidateSkipRequest, db: Session = Depends(get_db)):
    user = get_user(db, payload.user_id)
    try:
        result = skip_candidate(user_id=user.id, user_name=user.name, candidate_key=payload.candidate_key)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot skip candidate: {e}")

    return result


@router.post("/cancel")
def cancel_current_candidate(payload: CandidateCancelRequest, db: Session = Depends(get_db)):
    user = get_user(db, payload.user_id)

    released = release_candidate(user_id=user.id, candidate_key=payload.candidate_key)

    if not released:
        raise HTTPException(status_code=400, detail="Candidate is not locked by this user")

    return {
        "status": "cancelled",
        "candidate_key": payload.candidate_key,
    }


@router.post("/check")
def check_current_candidate(payload: CandidateCheckRequest, db: Session = Depends(get_db)):
    user = get_user(db, payload.user_id)

    if not touch_candidate(user.id, payload.candidate_key):
        raise HTTPException(status_code=400, detail="Candidate is not locked by this user")
    candidate_folder = get_candidate_temp_folder(user.id, payload.candidate_key)

    if not candidate_folder.exists() or not candidate_folder.is_dir():
        raise HTTPException(status_code=404, detail="Candidate temporary folder not found")

    try:
        result = check_image_folder(candidate_folder)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot check candidate: {e}")

    return {
        "status": "ok",
        "candidate_key": payload.candidate_key,
        "check": result,
    }
