from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers import routes_user, routes_embedding, routes_person, routes_media_description, routes_media
from routers import routes_freeze, routes_iteration, routes_face, routes_history
from routers import routes_search_for_name
from routers import routes_service_cluster
from routers import routes_inspect_media_v3
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


app.include_router(routes_user.router)
app.include_router(routes_embedding.router)
app.include_router(routes_person.router)
app.include_router(routes_media_description.router)
app.include_router(routes_media.router)
app.include_router(routes_freeze.router)
app.include_router(routes_iteration.router)
app.include_router(routes_face.router)
app.include_router(routes_history.router)
app.include_router(routes_service_cluster.router)
app.include_router(routes_search_for_name.router)
app.include_router(routes_inspect_media_v3.router)

