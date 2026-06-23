from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes.routers_classic import routes_freeze, routes_history, routes_iteration, routes_embedding
from routes.routers_classic import routes_face, routes_media, routes_person, routes_user, routes_media_description
from routes.routers_classic import routes_source, routes_face_category
from routes.routes_services import routes_search_for_name, routes_inspect_media_v3, routes_service_cluster
from services.config import FREEZE_FOLDER_FROM_MXF, MP4_LIGHT_FOLDER

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для тесту з Lovable ок
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/freezes", StaticFiles(directory=str(FREEZE_FOLDER_FROM_MXF)), name="freezes")
app.mount("/media-files", StaticFiles(directory=str(MP4_LIGHT_FOLDER)), name="media-files")

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
app.include_router(routes_service_cluster.router)
app.include_router(routes_search_for_name.router)
app.include_router(routes_inspect_media_v3.router)

