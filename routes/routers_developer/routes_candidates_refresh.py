from fastapi import APIRouter, HTTPException

from services.candidates.candidate_service import refresh_candidate_cache


router = APIRouter(prefix="/developer/candidates", tags=["developer refresh candidate cache"])


@router.post("/refresh")
def refresh_candidates():
    try:
        return refresh_candidate_cache()

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot refresh candidate cache: {e}")
