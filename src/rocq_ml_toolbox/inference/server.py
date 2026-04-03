from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pytanque.client import PetanqueError, Response, RouteName
from pytanque.protocol import Error, Failure
from pytanque.routes import PETANQUE_ROUTES

from ..parser.ast.driver import load_proof_dump
from ..parser.glob.driver import load_glob_file
from ..safeverify.core import run_safeverify
from .file_api import (
    AccessLibrariesBody,
    FileAccessConfig,
    FsAccessMode,
    ReadDocstringsBody,
    ReadFileBody,
    WriteFileBody,
    access_libraries,
    read_docstrings,
    read_file,
    resolve_coq_lib_path,
    router as file_api_router,
    write_file,
)
from .sessions import SessionManager, SessionManagerError

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
    num_pet_server = int(os.environ["NUM_PET_SERVER"])
    pet_server_start_port = int(os.environ["PET_SERVER_START_PORT"])
    redis_url = os.environ["REDIS_URL"]
    session_ttl_seconds = int(os.environ.get("SESSION_TTL_SECONDS", str(30 * 60)))
    session_cache_keep_feedback = _env_bool("SESSION_CACHE_KEEP_FEEDBACK", default=False)
    session_cleanup_interval_seconds = int(os.environ.get("SESSION_CLEANUP_INTERVAL_SECONDS", "60"))

    fs_mode_raw = os.environ.get("FS_ACCESS_MODE", FsAccessMode.READ_LIB_ONLY.value).strip()
    try:
        fs_mode = FsAccessMode(fs_mode_raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid FS_ACCESS_MODE={fs_mode_raw!r}. Expected one of: "
            f"{FsAccessMode.READ_LIB_ONLY.value}, {FsAccessMode.RW_ANYWHERE.value}."
        ) from exc

    coq_lib_override = os.environ.get("COQ_LIB_PATH")
    coq_lib_path = resolve_coq_lib_path(coq_lib_override)
    read_allow_raw = os.environ.get("FS_READ_ALLOW_PATHS", "[]")
    try:
        parsed_read_allow = json.loads(read_allow_raw)
    except Exception as exc:
        raise RuntimeError(f"Invalid FS_READ_ALLOW_PATHS value: {read_allow_raw!r}") from exc
    if not isinstance(parsed_read_allow, list):
        raise RuntimeError("FS_READ_ALLOW_PATHS must be a JSON list of paths.")
    read_allow_paths: tuple[Path, ...] = tuple(Path(str(p)).expanduser().resolve() for p in parsed_read_allow)

    sm = SessionManager(
        redis_url=redis_url,
        pet_server_start_port=pet_server_start_port,
        num_pet_server=num_pet_server,
        session_ttl_s=session_ttl_seconds,
        cache_feedback=session_cache_keep_feedback,
        session_cleanup_interval_s=session_cleanup_interval_seconds,
    )
    app.state.sm = sm
    app.state.file_access = FileAccessConfig(
        mode=fs_mode,
        coq_lib_path=coq_lib_path,
        read_allow_paths=read_allow_paths,
    )
    app.state.toc_cache: dict[tuple[str, str, bool, bool], dict[str, Any]] = {}
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(file_api_router)


@app.get("/login")
def login(request: FastAPIRequest):
    """Return a session object with assigned pet-server index and unique session ID."""
    session_manager: SessionManager = request.app.state.sm
    session_id = session_manager.create_session()
    return {"session_id": session_id}


@app.get("/health")
def health(request: FastAPIRequest):
    """Check health status."""
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
    except PetanqueError as exc:
        logging.error("[%s] Petanque error %s: %s", body.session_id, exc.code, exc.message)
        result = Failure(
            body.id,
            Error(
                exc.code,
                exc.message,
            ),
        )
    except SessionManagerError as exc:
        logging.error("[%s] SessionManager error: %s", body.session_id, exc.message)
        result = Failure(
            body.id,
            Error(
                -30_000,
                exc.message,
            ),
        )
    return result.to_json()


class GetAstBody(BaseModel):
    path: str
    force_dump: bool = False
    root: Optional[str] = None


class GetGlobBody(BaseModel):
    path: str
    force_compile: bool = False


class SafeVerifyBody(BaseModel):
    source: str
    target: str
    root: str
    axiom_whitelist: list[str] = Field(default_factory=list)
    save_path: Optional[str] = None
    verbose: bool = False


class EmptyFileBody(BaseModel):
    content: Optional[str] = None
    root: Optional[str] = None


@app.post("/get_dump")
def get_dump(body: GetAstBody):
    """Extract full AST (verbatim) from document at `path`."""
    logging.info("get_ast: %s", body.path)
    proof, ast, diags = load_proof_dump(body.path, root=body.root, force_dump=body.force_dump)

    def gen():
        yield '{"proof":'
        yield json.dumps(proof)
        yield ', "ast":'
        yield json.dumps(ast)
        yield ', "diags":'
        yield json.dumps([d.to_json() for d in diags])
        yield "}"

    return StreamingResponse(gen(), media_type="application/json")


@app.post("/get_glob")
def get_glob(body: GetGlobBody):
    """Extract glob (verbatim) from document at `path`."""
    logging.info("get_glob: %s", body.path)
    return {"value": load_glob_file(body.path, force_compile=body.force_compile)}


@app.post("/safeverify")
def safeverify(body: SafeVerifyBody):
    """Run SafeVerify inside the server environment."""
    report = run_safeverify(
        body.source,
        body.target,
        root=body.root,
        axiom_whitelist=body.axiom_whitelist,
        save_path=body.save_path,
        verbose=body.verbose,
    )
    return report.to_json()


@app.post("/tmp_file")
def temp_file(body: EmptyFileBody):
    """Return the path of a new empty file."""
    root_dir: str | None = None
    if body.root is not None:
        root_path = body.root
        os.makedirs(root_path, exist_ok=True)
        root_dir = root_path

    fp = tempfile.NamedTemporaryFile(delete=False, suffix=".v", dir=root_dir)
    fp.close()
    if body.content:
        with open(fp.name, "w", encoding="utf-8") as file:
            file.write(body.content)
    return {"path": fp.name}
