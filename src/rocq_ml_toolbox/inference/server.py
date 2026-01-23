import os

from pytanque.protocol import Request
from .rpc_registry import dispatch_rpc

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from .sessions import SessionManager
from .rpc_registry import build_registry, dispatch_rpc

@asynccontextmanager
async def lifespan(app: FastAPI):
    NUM_PET_SERVER = int(os.environ["NUM_PET_SERVER"])
    PET_SERVER_START_PORT = int(os.environ["PET_SERVER_START_PORT"])
    REDIS_URL = os.environ["REDIS_URL"]

    sm = SessionManager(
        redis_url=REDIS_URL,
        pet_server_start_port=PET_SERVER_START_PORT,
        num_pet_server=NUM_PET_SERVER
    )
    registry = build_registry(sm)
    app.state.registry = registry
    yield

    # cleanup hook?
    # sm.close()

app = FastAPI(lifespan=lifespan)

@app.post("/rpc")
def rpc_endpoint(request: Request, body: bytes, x_session_id: str | None = None):
    registry = request.app.state.registry
    result = dispatch_rpc(registry)
    return result