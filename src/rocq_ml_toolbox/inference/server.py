import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request as FastAPIRequest, HTTPException
from fastapi.responses import PlainTextResponse
import tempfile

from pydantic import BaseModel
from typing import Optional, Any
from pytanque.client import RouteName
from pytanque.routes import PETANQUE_ROUTES
from .sessions import SessionManager
from ..parser.ast.driver import load_ast_dump

class JsonRpcBody(BaseModel):
    jsonrpc: str = "2.0"
    id: int
    session_id: str
    route_name: RouteName
    params: dict[str, Any]
    timeout: Optional[float]

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
    app.state.sm = sm
    yield

    # cleanup hook?
    # sm.close()

app = FastAPI(lifespan=lifespan)

@app.get("/login")
def login(request: FastAPIRequest):
    """
    Return a session object with assigned pet-server index and unique session ID.
    """
    session_manager: SessionManager = request.app.state.sm
    session_id = session_manager.create_session()
    return {"session_id": session_id}

@app.get("/health")
def health():
    """
    Check health status.
    """
    return {"status": "ok"}

@app.post("/rpc")
def rpc_endpoint(body: JsonRpcBody, request: FastAPIRequest):
    session_manager: SessionManager = request.app.state.sm
    params_cls = PETANQUE_ROUTES.get(body.route_name).params_cls
    params_obj = params_cls.from_json(body.params)

    result = session_manager._pet_call(
        request_id=body.id,
        session_id=body.session_id,
        route_name=body.route_name,
        params=params_obj,
        timeout=body.timeout,
    )
    # result = dispatch_rpc(registry)
    return result.to_json()

class GetAstBody(BaseModel):
    path: str
    force_dump: bool=False

@app.post("/get_ast")
def get_ast(body: GetAstBody):
    """
    Extract full AST (verbatim) from document at `path`.
    """
    output = {"value": load_ast_dump(body.path, force_dump=body.force_dump)}
    return output

@app.get("/empty_file")
def empty_file():
    """
    Return the path of a new empty file.
    """
    fp = tempfile.NamedTemporaryFile(delete=False, suffix='.v')
    output = {"path": fp.name}
    return output