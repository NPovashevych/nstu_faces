from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes.routers_classic import routes_freeze, routes_history, routes_iteration, routes_embedding
from routes.routers_classic import routes_face, routes_media, routes_person, routes_user, routes_media_description
from routes.routers_classic import routes_source, routes_face_category
from routes.routes_services import routes_search_for_name, routes_auth
from routes.routes_services import routes_claster_identify
from routes.routes_services import routes_search_by_photo_faiss, routes_detect_photo_faiss
from routes.routes_services import routes_inspect_media_v3
from routes.routers_developer import routes_reference_gender

from services.config import TEST_FREEZE_FOLDER, TEST_MP4_LIGHT_FOLDER, USER_UPLOAD_FOLDER, INTVNEWS_FREEZE_FOLDER, PROXY_NEWS_FOLDER

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для тесту з Lovable ок
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/freezes-test",
    StaticFiles(directory=str(TEST_FREEZE_FOLDER)),
    name="freezes-test",
)

app.mount(
    "/freezes-news",
    StaticFiles(directory=str(INTVNEWS_FREEZE_FOLDER)),
    name="freezes-news",
)

app.mount(
    "/media-files-test",
    StaticFiles(directory=str(TEST_MP4_LIGHT_FOLDER)),
    name="media-files-test",
)

app.mount(
    "/media-files-news",
    StaticFiles(directory=str(PROXY_NEWS_FOLDER)),
    name="media-files-news",
)

app.mount(
    "/media-user-upload",
    StaticFiles(directory=str(USER_UPLOAD_FOLDER)),
    name="media-user-upload",
)

# класичні роути
app.include_router(routes_user.router)
app.include_router(routes_embedding.router)
app.include_router(routes_person.router)
app.include_router(routes_media_description.router)
app.include_router(routes_media.router)
app.include_router(routes_freeze.router)
app.include_router(routes_iteration.router)
app.include_router(routes_face.router)
app.include_router(routes_history.router)
app.include_router(routes_source.router)
app.include_router(routes_face_category.router)

# сервісні роути
app.include_router(routes_claster_identify.router)
app.include_router(routes_search_for_name.router)
app.include_router(routes_search_by_photo_faiss.router)
app.include_router(routes_auth.router)
app.include_router(routes_detect_photo_faiss.router)
app.include_router(routes_inspect_media_v3.router)

# роути розробника
app.include_router(routes_reference_gender.router)
