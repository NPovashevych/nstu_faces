from fastapi import FastAPI

from routers import routes_user, routes_embedding, routes_person, routes_media_description, routes_media
from routers import routes_freeze, routes_iteration, routes_face, routes_history

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


app.include_router(routes_user.router)
app.include_router(routes_embedding.router)
app.include_router(routes_person.router)
app.include_router(routes_media_description.router)
app.include_router(routes_media.router)
app.include_router(routes_freeze.router)
app.include_router(routes_iteration.router)
app.include_router(routes_face.router)
app.include_router(routes_history.router)
