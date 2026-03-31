import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request as FastAPIRequest, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
import tempfile
import json
import logging

from pydantic import BaseModel, Field
from typing import Optional, Any
from pytanque.client import RouteName, PetanqueError, Response
from pytanque.protocol import Failure, Error
from pytanque.routes import PETANQUE_ROUTES
from .sessions import SessionManager, SessionManagerError
from ..parser.ast.driver import load_proof_dump
from ..parser.glob.driver import load_glob_file
from ..safeverify.core import run_safeverify

logger = logging.getLogger("session")

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

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
    SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(30 * 60)))
    SESSION_CACHE_KEEP_FEEDBACK = _env_bool("SESSION_CACHE_KEEP_FEEDBACK", default=False)
    SESSION_CLEANUP_INTERVAL_SECONDS = int(os.environ.get("SESSION_CLEANUP_INTERVAL_SECONDS", "60"))

    sm = SessionManager(
        redis_url=REDIS_URL,
        pet_server_start_port=PET_SERVER_START_PORT,
        num_pet_server=NUM_PET_SERVER,
        session_ttl_s=SESSION_TTL_SECONDS,
        cache_feedback=SESSION_CACHE_KEEP_FEEDBACK,
        session_cleanup_interval_s=SESSION_CLEANUP_INTERVAL_SECONDS,
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
def health(request: FastAPIRequest):
    """
    Check health status.
    """
    session_manager: SessionManager = request.app.state.sm
    snapshot = session_manager.health_snapshot()
    if not snapshot["ok"]:
        raise HTTPException(status_code=503, detail=snapshot)
    return snapshot

@app.post("/rpc")
def rpc_endpoint(body: JsonRpcBody, request: FastAPIRequest):
    session_manager: SessionManager = request.app.state.sm
    params_cls = PETANQUE_ROUTES[body.route_name].params_cls
    params_obj = params_cls.from_json(body.params)
    try:
        result = session_manager._pet_call(
            request_id=body.id,
            session_id=body.session_id,
            route_name=body.route_name,
            params=params_obj,
            timeout=body.timeout,
        )
    except PetanqueError as e:
        logging.error(f"[{body.session_id}] Petanque error {e.code}: {e.message}")
        result = Failure(
            body.id,
            Error(
                e.code,
                e.message
            )
        )
    except SessionManagerError as e:
        logging.error(f"[{body.session_id}] SessioManager error: {e.message}")
        result = Failure(
            body.id,
            Error(
                -30_000,
                e.message
            )
        )
    # result = dispatch_rpc(registry)
    return result.to_json()

class GetAstBody(BaseModel):
    path: str
    force_dump: bool=False
    root: Optional[str]=None

class GetGlobBody(BaseModel):
    path: str
    force_compile: bool=False

class SafeVerifyBody(BaseModel):
    source: str
    target: str
    root: str
    axiom_whitelist: list[str] = Field(default_factory=list)
    save_path: Optional[str] = None
    verbose: bool = False

@app.post("/get_dump")
def get_dump(body: GetAstBody):
    """
    Extract full AST (verbatim) from document at `path`.
    """
    logging.info(f"get_ast: {body.path}")
    proof, ast, diags = load_proof_dump(body.path, root=body.root, force_dump=body.force_dump)
    def gen():
        yield '{"proof":'
        yield json.dumps(proof)
        yield ', "ast":'
        yield json.dumps(ast)
        yield ', "diags":'
        yield json.dumps([d.to_json() for d in diags])
        yield '}'

    return StreamingResponse(gen(), media_type="application/json")

@app.post("/get_glob")
def get_glob(body: GetGlobBody):
    """
    Extract glob (verbatim) from document at `path`.
    """
    logging.info(f"get_glob: {body.path}")
    output = {"value": load_glob_file(body.path, force_compile=body.force_compile)}
    return output

@app.post("/safeverify")
def safeverify(body: SafeVerifyBody):
    """
    Run SafeVerify inside the server environment.
    """
    report = run_safeverify(
        body.source,
        body.target,
        root=body.root,
        axiom_whitelist=body.axiom_whitelist,
        save_path=body.save_path,
        verbose=body.verbose,
    )
    return report.to_json()

@app.get("/empty_file")
def empty_file():
    """
    Return the path of a new empty file.
    """
    fp = tempfile.NamedTemporaryFile(delete=False, suffix='.v')
    output = {"path": fp.name}
    return output
