from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes.routers_classic import routes_user
from routes.routes_works import routes_search_for_name, routes_auth
from routes.routes_works import routes_claster_identify
from routes.routes_works import routes_detected_media_faiss_v2
from routes.routes_works import routes_search_by_photo_faiss_v2
from routes.routes_works import routes_inspect_media_v3
from routes.routes_works import routes_identify_for_name
from routes.routes_works import routes_candidates
from routes.routers_developer import routes_reference_gender, routes_candidates_refresh

from services.config import TEST_FREEZE_FOLDER, TEST_MP4_LIGHT_FOLDER, USER_UPLOAD_FOLDER, INTVNEWS_FREEZE_FOLDER, PROXY_NEWS_FOLDER
from services.config import TEMPORARY_FREEZES_FOLDER, START_CANDIDATE_FOLDER, SKIPPED_CANDIDATE_FOLDER, FINISH_CANDIDATE_FOLDER

app = FastAPI()
# app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для тесту з Lovable ок
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/freezes-test", StaticFiles(directory=str(TEST_FREEZE_FOLDER)), name="freezes-test")
app.mount("/freezes-news", StaticFiles(directory=str(INTVNEWS_FREEZE_FOLDER)), name="freezes-news")
app.mount("/media-files-test", StaticFiles(directory=str(TEST_MP4_LIGHT_FOLDER)),name="media-files-test")
app.mount("/media-files-news", StaticFiles(directory=str(PROXY_NEWS_FOLDER)), name="media-files-news")
app.mount("/media-user-upload", StaticFiles(directory=str(USER_UPLOAD_FOLDER)), name="media-user-upload")
app.mount("/media-youtube-upload", StaticFiles(directory=str(TEMPORARY_FREEZES_FOLDER)), name="media-youtube-upload")
app.mount("/start-candidates", StaticFiles(directory=str(START_CANDIDATE_FOLDER)), name="start-candidates")
app.mount("/skipped-candidates", StaticFiles(directory=str(SKIPPED_CANDIDATE_FOLDER)), name="skipped-candidates")
app.mount("/finish-candidates", StaticFiles(directory=str(FINISH_CANDIDATE_FOLDER)), name="finish-candidates")


# класичні роути
app.include_router(routes_user.router)
# app.include_router(routes_embedding.router)
# app.include_router(routes_person.router)
# app.include_router(routes_media_description.router)
# app.include_router(routes_media.router)
# app.include_router(routes_freeze.router)
# app.include_router(routes_iteration.router)
# app.include_router(routes_face.router)
# app.include_router(routes_history.router)
# app.include_router(routes_source.router)
# app.include_router(routes_face_category.router)

# сервісні роути
app.include_router(routes_claster_identify.router)
app.include_router(routes_search_for_name.router)
app.include_router(routes_search_by_photo_faiss_v2.router)
app.include_router(routes_auth.router)
app.include_router(routes_detected_media_faiss_v2.router)
app.include_router(routes_inspect_media_v3.router)
app.include_router(routes_identify_for_name.router)
app.include_router(routes_candidates.router)

# роути розробника
app.include_router(routes_reference_gender.router)
app.include_router(routes_candidates_refresh.router)
