from fastapi import FastAPI

from routers import routes_user, routes_embedding, routes_person

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


