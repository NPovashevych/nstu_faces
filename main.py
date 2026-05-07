from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers import routes_user, routes_embedding, routes_person, routes_media_description, routes_media
from routers import routes_freeze, routes_iteration, routes_face, routes_history
from routers import routes_unknown_clusters
from routers import routes_search_v2, routes_inspect_media

from services.config import FREEZE_FOLDER, MP4_FOLDER


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для тесту з Lovable ок
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/freezes", StaticFiles(directory=str(FREEZE_FOLDER)), name="freezes")
app.mount("/media-files", StaticFiles(directory=str(MP4_FOLDER)), name="media-files")


app.include_router(routes_user.router)
app.include_router(routes_embedding.router)
app.include_router(routes_person.router)
app.include_router(routes_media_description.router)
app.include_router(routes_media.router)
app.include_router(routes_freeze.router)
app.include_router(routes_iteration.router)
app.include_router(routes_face.router)
app.include_router(routes_history.router)
app.include_router(routes_unknown_clusters.router)
app.include_router(routes_search_v2.router)
app.include_router(routes_inspect_media.router)
