from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.context import Context
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.middleware import Middleware, MiddlewareContext

THIS_DIR = Path(__file__).resolve().parent
PUTNAM_AGENT_DIR = THIS_DIR.parent
REPO_ROOT = PUTNAM_AGENT_DIR.parent.parent
if str(PUTNAM_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(PUTNAM_AGENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import DocqAgentSession, read_source_via_client  # noqa: E402
from rocq_ml_toolbox.inference.client import PytanqueExtended  # noqa: E402

DEFAULT_SOURCE = PUTNAM_AGENT_DIR / "putnam" / "mathcomp" / "putnam_1962_a6.v"
DEFAULT_MCP_HOST = "0.0.0.0"
DEFAULT_MCP_PORT = 8012
DEFAULT_PET_HOST = "127.0.0.1"
DEFAULT_PET_PORT = 5000
DEFAULT_PET_TIMEOUT = 90.0
DEFAULT_MAX_TOOL_CALLS = 4000
DEFAULT_MAX_REQUESTS = 2000
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_MODEL_TIMEOUT_SECONDS = 300.0
DEFAULT_MODEL_MAX_RETRIES = 5
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_MIN_P = 0.01
DEFAULT_REPEAT_PENALTY = 1.0
DEFAULT_SEMANTIC_ROUTE = "/search"
DEFAULT_SEMANTIC_ENV = "coq-mathcomp"

mcp = FastMCP(name="DocQ Tools MCP")


@dataclass
class RuntimeConfig:
    source: Path
    env: str | None
    pet_host: str
    pet_port: int
    pet_timeout: float
    max_tool_calls: int
    max_requests: int | None
    include_semantic_tool: bool
    semantic_base_url: str | None
    semantic_route: str
    semantic_api_key: str | None
    semantic_env: str
    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_model: str
    model_timeout_seconds: float
    model_max_retries: int
    temperature: float
    top_p: float
    min_p: float
    repeat_penalty: float


@dataclass
class SessionEntry:
    key: str
    config: RuntimeConfig
    client: PytanqueExtended
    session: DocqAgentSession
    subagent_model: Any | None
    model_error: str | None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_SESSION_REGISTRY: dict[str, SessionEntry] = {}
_SESSION_REGISTRY_LOCK = threading.Lock()


def _to_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _to_int(raw: str | None, *, default: int, name: str, minimum: int | None = None) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        out = int(raw)
    except Exception as exc:
        raise ValueError(f"Invalid integer for `{name}`: {raw!r}") from exc
    if minimum is not None and out < minimum:
        raise ValueError(f"Invalid `{name}`={out}: must be >= {minimum}")
    return out


def _to_float(raw: str | None, *, default: float, name: str, minimum: float | None = None) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        out = float(raw)
    except Exception as exc:
        raise ValueError(f"Invalid float for `{name}`: {raw!r}") from exc
    if minimum is not None and out < minimum:
        raise ValueError(f"Invalid `{name}`={out}: must be >= {minimum}")
    return out


def _request_query_param(*names: str) -> str | None:
    try:
        req = get_http_request()
    except Exception:
        return None
    for name in names:
        value = req.query_params.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return None


def _cfg_param(*query_names: str, env_name: str, default: str | None = None) -> str | None:
    value = _request_query_param(*query_names)
    if value is not None:
        return value
    env_value = os.getenv(env_name)
    if env_value is not None and env_value.strip() != "":
        return env_value.strip()
    return default


def _normalize_source_path(raw_source: str | None) -> Path:
    source_text = (raw_source or str(DEFAULT_SOURCE)).strip()
    source = Path(source_text).expanduser()
    if source.is_absolute():
        return source.resolve()

    candidates = [
        (Path.cwd() / source).resolve(),
        (PUTNAM_AGENT_DIR / source).resolve(),
        (REPO_ROOT / source).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _normalize_read_path(path: str | list[str] | None) -> str | None:
    if path is None:
        return None
    if isinstance(path, list):
        parts: list[str] = []
        for item in path:
            token = str(item).strip().strip("/")
            if token:
                parts.append(token)
        return "/".join(parts) if parts else None
    token = str(path).strip()
    return token or None


def _extract_runtime_config() -> RuntimeConfig:
    source = _normalize_source_path(
        _cfg_param("source", env_name="DOCQ_MCP_SOURCE", default=str(DEFAULT_SOURCE))
    )
    if not source.exists():
        raise ValueError(f"Source file does not exist: {source}")

    env = _cfg_param("env", env_name="DOCQ_MCP_ENV", default=None)
    pet_host = _cfg_param("pet_host", "host", env_name="DOCQ_MCP_PET_HOST", default=DEFAULT_PET_HOST)
    pet_port = _to_int(
        _cfg_param("pet_port", "port", env_name="DOCQ_MCP_PET_PORT", default=str(DEFAULT_PET_PORT)),
        default=DEFAULT_PET_PORT,
        name="pet_port",
        minimum=1,
    )
    pet_timeout = _to_float(
        _cfg_param("pet_timeout", "timeout", env_name="DOCQ_MCP_PET_TIMEOUT", default=str(DEFAULT_PET_TIMEOUT)),
        default=DEFAULT_PET_TIMEOUT,
        name="pet_timeout",
        minimum=0.1,
    )

    max_tool_calls = _to_int(
        _cfg_param("max_tool_calls", env_name="DOCQ_MCP_MAX_TOOL_CALLS", default=str(DEFAULT_MAX_TOOL_CALLS)),
        default=DEFAULT_MAX_TOOL_CALLS,
        name="max_tool_calls",
        minimum=1,
    )
    max_requests_raw = _cfg_param("max_requests", env_name="DOCQ_MCP_MAX_REQUESTS", default=str(DEFAULT_MAX_REQUESTS))
    max_requests: int | None
    if max_requests_raw is None or max_requests_raw.strip().lower() in {"none", "unbounded"}:
        max_requests = None
    else:
        max_requests = _to_int(max_requests_raw, default=DEFAULT_MAX_REQUESTS, name="max_requests", minimum=1)

    include_semantic_tool = _to_bool(
        _cfg_param("include_semantic_tool", env_name="DOCQ_MCP_INCLUDE_SEMANTIC_TOOL", default="1"),
        default=True,
    )
    if _to_bool(
        _cfg_param("disable_semantic_tool", env_name="DOCQ_MCP_DISABLE_SEMANTIC_TOOL", default="0"),
        default=False,
    ):
        include_semantic_tool = False

    semantic_base_url = _cfg_param("semantic_base_url", env_name="DOCQ_MCP_SEMANTIC_BASE_URL", default=None)
    semantic_route = _cfg_param(
        "semantic_route",
        env_name="DOCQ_MCP_SEMANTIC_ROUTE",
        default=DEFAULT_SEMANTIC_ROUTE,
    )
    semantic_api_key = _cfg_param("semantic_api_key", env_name="DOCQ_MCP_SEMANTIC_API_KEY", default=None)
    semantic_env = _cfg_param(
        "semantic_env",
        env_name="DOCQ_MCP_SEMANTIC_ENV",
        default=DEFAULT_SEMANTIC_ENV,
    )

    openrouter_api_key = _cfg_param(
        "openrouter_api_key",
        env_name="DOCQ_MCP_OPENROUTER_API_KEY",
        default=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    openrouter_base_url = _cfg_param(
        "openrouter_base_url",
        env_name="DOCQ_MCP_OPENROUTER_BASE_URL",
        default=DEFAULT_OPENROUTER_BASE_URL,
    )
    openrouter_model = _cfg_param(
        "openrouter_model",
        "model",
        env_name="DOCQ_MCP_OPENROUTER_MODEL",
        default=DEFAULT_OPENROUTER_MODEL,
    )
    model_timeout_seconds = _to_float(
        _cfg_param(
            "model_timeout_seconds",
            env_name="DOCQ_MCP_MODEL_TIMEOUT_SECONDS",
            default=str(DEFAULT_MODEL_TIMEOUT_SECONDS),
        ),
        default=DEFAULT_MODEL_TIMEOUT_SECONDS,
        name="model_timeout_seconds",
        minimum=0.1,
    )
    model_max_retries = _to_int(
        _cfg_param(
            "model_max_retries",
            env_name="DOCQ_MCP_MODEL_MAX_RETRIES",
            default=str(DEFAULT_MODEL_MAX_RETRIES),
        ),
        default=DEFAULT_MODEL_MAX_RETRIES,
        name="model_max_retries",
        minimum=0,
    )
    temperature = _to_float(
        _cfg_param("temperature", env_name="DOCQ_MCP_TEMPERATURE", default=str(DEFAULT_TEMPERATURE)),
        default=DEFAULT_TEMPERATURE,
        name="temperature",
        minimum=0.0,
    )
    top_p = _to_float(
        _cfg_param("top_p", env_name="DOCQ_MCP_TOP_P", default=str(DEFAULT_TOP_P)),
        default=DEFAULT_TOP_P,
        name="top_p",
        minimum=0.0,
    )
    min_p = _to_float(
        _cfg_param("min_p", env_name="DOCQ_MCP_MIN_P", default=str(DEFAULT_MIN_P)),
        default=DEFAULT_MIN_P,
        name="min_p",
        minimum=0.0,
    )
    repeat_penalty = _to_float(
        _cfg_param(
            "repeat_penalty",
            env_name="DOCQ_MCP_REPEAT_PENALTY",
            default=str(DEFAULT_REPEAT_PENALTY),
        ),
        default=DEFAULT_REPEAT_PENALTY,
        name="repeat_penalty",
        minimum=0.0,
    )

    return RuntimeConfig(
        source=source,
        env=env,
        pet_host=pet_host or DEFAULT_PET_HOST,
        pet_port=pet_port,
        pet_timeout=pet_timeout,
        max_tool_calls=max_tool_calls,
        max_requests=max_requests,
        include_semantic_tool=include_semantic_tool,
        semantic_base_url=semantic_base_url,
        semantic_route=semantic_route or DEFAULT_SEMANTIC_ROUTE,
        semantic_api_key=semantic_api_key,
        semantic_env=semantic_env or DEFAULT_SEMANTIC_ENV,
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=openrouter_base_url or DEFAULT_OPENROUTER_BASE_URL,
        openrouter_model=openrouter_model or DEFAULT_OPENROUTER_MODEL,
        model_timeout_seconds=model_timeout_seconds,
        model_max_retries=model_max_retries,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
    )


def _build_optional_subagent_model(config: RuntimeConfig) -> tuple[Any | None, str | None]:
    if not config.openrouter_api_key:
        return None, (
            "No OpenRouter API key configured. Set `OPENROUTER_API_KEY`/`OPENAI_API_KEY` "
            "or pass `openrouter_api_key` query param to enable lemma subagent proving tools."
        )
    try:
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.settings import ModelSettings
    except Exception as exc:
        return None, f"Unable to import OpenRouter model dependencies: {exc}"

    try:
        settings = ModelSettings(
            temperature=config.temperature,
            top_p=config.top_p,
            extra_body={
                "min_p": config.min_p,
                "repetition_penalty": config.repeat_penalty,
            },
        )
        openai_client = AsyncOpenAI(
            base_url=config.openrouter_base_url,
            api_key=config.openrouter_api_key,
            timeout=config.model_timeout_seconds,
            max_retries=config.model_max_retries,
        )
        provider = OpenAIProvider(openai_client=openai_client)
        model = OpenAIChatModel(config.openrouter_model, provider=provider, settings=settings)
        return model, None
    except Exception as exc:
        return None, f"Failed to build OpenRouter model: {exc}"


def _close_entry(entry: SessionEntry) -> None:
    base_client = getattr(entry.session.client, "_inner_client", entry.session.client)
    close = getattr(base_client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _create_entry(key: str, config: RuntimeConfig) -> SessionEntry:
    model, model_error = _build_optional_subagent_model(config)
    client = PytanqueExtended(config.pet_host, config.pet_port)
    session = DocqAgentSession.from_source(
        client,
        config.source,
        env=config.env,
        timeout=config.pet_timeout,
        connect=True,
        semantic_base_url=config.semantic_base_url,
        semantic_route=config.semantic_route,
        semantic_api_key=config.semantic_api_key,
        semantic_env=config.semantic_env,
        max_tool_calls=config.max_tool_calls,
        max_requests=config.max_requests,
        subagent_model=model,
        include_semantic_tool=config.include_semantic_tool,
        logger=None,
    )
    return SessionEntry(
        key=key,
        config=config,
        client=client,
        session=session,
        subagent_model=model,
        model_error=model_error,
    )


def _extract_openai_session_id(context: MiddlewareContext) -> str | None:
    fast_ctx = context.fastmcp_context
    if fast_ctx is None:
        return None
    req_ctx = getattr(fast_ctx, "request_context", None)
    meta = getattr(req_ctx, "meta", None)
    if meta is None:
        return None
    try:
        dump = meta.model_dump()
    except Exception:
        return None
    session_id = dump.get("openai/session")
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    return None


class DocqSessionMiddleware(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        fast_ctx = context.fastmcp_context
        if fast_ctx is not None:
            openai_session = _extract_openai_session_id(context)
            if openai_session:
                try:
                    setattr(fast_ctx.session, "_fastmcp_state_prefix", openai_session)
                except Exception:
                    pass
            existing = None
            try:
                existing = await fast_ctx.get_state("DOCQ_SESSION_KEY")
            except Exception:
                existing = None
            if isinstance(existing, str) and existing.strip():
                key = existing.strip()
            else:
                if openai_session:
                    key = f"openai:{openai_session}"
                else:
                    key = f"session:{uuid.uuid4().hex}"
                await fast_ctx.set_state("DOCQ_SESSION_KEY", key)
        return await call_next(context)


mcp.add_middleware(DocqSessionMiddleware())


async def _get_entry(ctx: Context) -> SessionEntry:
    key = await ctx.get_state("DOCQ_SESSION_KEY")
    if not isinstance(key, str) or not key.strip():
        key = f"session:{uuid.uuid4().hex}"
        await ctx.set_state("DOCQ_SESSION_KEY", key)

    with _SESSION_REGISTRY_LOCK:
        existing = _SESSION_REGISTRY.get(key)
    if existing is not None:
        return existing

    config = _extract_runtime_config()
    entry = _create_entry(key, config)
    with _SESSION_REGISTRY_LOCK:
        old = _SESSION_REGISTRY.get(key)
        if old is not None:
            return old
        _SESSION_REGISTRY[key] = entry
    return entry


async def _reset_entry(ctx: Context, *, source: str | None = None, env: str | None = None) -> SessionEntry:
    key = await ctx.get_state("DOCQ_SESSION_KEY")
    if not isinstance(key, str) or not key.strip():
        key = f"session:{uuid.uuid4().hex}"
        await ctx.set_state("DOCQ_SESSION_KEY", key)

    config = _extract_runtime_config()
    if source is not None and source.strip() != "":
        config.source = _normalize_source_path(source)
        if not config.source.exists():
            raise ValueError(f"Source file does not exist: {config.source}")
    if env is not None:
        config.env = env.strip() or None

    new_entry = _create_entry(key, config)
    with _SESSION_REGISTRY_LOCK:
        old_entry = _SESSION_REGISTRY.get(key)
        _SESSION_REGISTRY[key] = new_entry
    if old_entry is not None:
        _close_entry(old_entry)
    return new_entry


def _result_error(message: str, *, hint: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "error": message}
    if hint:
        out["hint"] = hint
    return out


@mcp.tool()
async def session_info(ctx: Context) -> dict[str, Any]:
    """Show active DocQ MCP session configuration and proof head status."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            head = entry.session.doc_manager.current_head()
            return {
                "ok": True,
                "session_key": entry.key,
                "source": str(entry.config.source),
                "env": entry.config.env,
                "pet_host": entry.config.pet_host,
                "pet_port": entry.config.pet_port,
                "pet_timeout": entry.config.pet_timeout,
                "include_semantic_tool": entry.config.include_semantic_tool,
                "semantic_enabled": entry.session.semantic_search is not None,
                "semantic_env": entry.config.semantic_env,
                "subagent_model_enabled": entry.subagent_model is not None,
                "subagent_model_error": entry.model_error,
                "manual_pending_lemma_tools_enabled": True,
                "head": head,
            }
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def reset_session(
    ctx: Context,
    source: str | None = None,
    env: str | None = None,
) -> dict[str, Any]:
    """Reset current MCP session with fresh DocQ workspace (optional new source/env)."""
    try:
        entry = await _reset_entry(ctx, source=source, env=env)
        async with entry.lock:
            head = entry.session.doc_manager.current_head()
            return {
                "ok": True,
                "session_key": entry.key,
                "source": str(entry.config.source),
                "env": entry.config.env,
                "head": head,
            }
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def list_docs(ctx: Context) -> Any:
    """List virtual documents/branches in current DocQ workspace."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            docs = entry.session.doc_manager.list_docs()
            return {
                "ok": True,
                "docs": docs,
                "doc_count": len(docs),
            }
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def checkout_doc(doc_id: int, ctx: Context) -> dict[str, Any]:
    """Switch active head document branch to `doc_id`."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.doc_manager.checkout_doc(doc_id=doc_id)
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def show_doc(doc_id: int, ctx: Context) -> dict[str, Any]:
    """Show one virtual document branch content/states."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.doc_manager.show_doc(doc_id=doc_id)
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def completion_status(ctx: Context, doc_id: int | None = None) -> dict[str, Any]:
    """Completion status of active/head proof in given doc."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.completion_status(doc_id=doc_id)
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def current_head(ctx: Context, doc_id: int | None = None) -> dict[str, Any]:
    """Return current head state metadata and recommended next action."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.doc_manager.current_head(doc_id=doc_id)
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def list_states(ctx: Context, doc_id: int | None = None) -> dict[str, Any]:
    """List detailed state nodes for a doc."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            states = entry.session.doc_manager.list_states_verbose(doc_id=doc_id)
            head = entry.session.doc_manager.current_head(doc_id=doc_id)
            return {
                "ok": True,
                "doc_id": head.get("doc_id"),
                "head_doc_id": head.get("head_doc_id"),
                "latest_state_index": head.get("latest_state_index"),
                "states": states,
                "state_count": len(states),
            }
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def get_goals(
    ctx: Context,
    state_index: int = 0,
    doc_id: int | None = None,
) -> dict[str, Any]:
    """Get goals at `(doc_id, state_index)`."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.doc_manager.get_goals(state_index=state_index, doc_id=doc_id)
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def run_tac(
    ctx: Context,
    state_index: int = 0,
    tactic: str = "idtac.",
    doc_id: int | None = None,
    branch_reason: str | None = None,
) -> dict[str, Any]:
    """Run one tactic from explicit state index."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.doc_manager.run_tac(
                state_index=state_index,
                tactic=tactic,
                doc_id=doc_id,
                branch_reason=branch_reason,
            )
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def run_tac_latest(
    ctx: Context,
    tactic: str = "idtac.",
    doc_id: int | None = None,
    branch_reason: str | None = None,
) -> dict[str, Any]:
    """Run one tactic from current latest state."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.doc_manager.run_tac_latest(
                tactic=tactic,
                doc_id=doc_id,
                branch_reason=branch_reason,
            )
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def read_workspace_source(
    ctx: Context,
    line: int | None = None,
    before: int = 20,
    after: int = 20,
    doc_id: int | None = None,
) -> dict[str, Any]:
    """Read workspace source content (full or around one line)."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.doc_manager.read_source(
                line=line,
                before=before,
                after=after,
                doc_id=doc_id,
            )
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def read_source_file(
    path: str | list[str],
    ctx: Context,
    line: int | None = None,
    before: int = 20,
    after: int = 20,
) -> dict[str, Any]:
    """Read external library file (TOC-relative) or workspace file if it matches a virtual doc path."""
    try:
        entry = await _get_entry(ctx)
        normalized = _normalize_read_path(path)
        if not normalized:
            return _result_error(
                "Invalid empty path.",
                hint="Use a TOC-relative path such as `mathcomp/fingroup/perm.v`.",
            )

        async with entry.lock:
            workspace_doc_id = entry.session.doc_manager.doc_id_for_source_path(normalized)
            if workspace_doc_id is not None:
                out = entry.session.doc_manager.read_source(
                    line=line,
                    before=before,
                    after=after,
                    doc_id=workspace_doc_id,
                )
                out["ok"] = True
                out["source_kind"] = "workspace_doc"
                out["requested_path"] = normalized
                out["resolved_path"] = str(entry.session.doc_manager.sessions[workspace_doc_id].source_path)
                return out

            out = read_source_via_client(
                entry.session.client,
                normalized,
                line=line,
                before=before,
                after=after,
            )
            out["ok"] = True
            return out
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def explore_toc(ctx: Context, path: list[str] | None = None) -> dict[str, Any]:
    """Explore library TOC tree for current environment."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.toc_explorer.explore(path or [])
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def semantic_doc_search(
    ctx: Context,
    query: str,
    k: int = 10,
) -> dict[str, Any]:
    """Semantic retrieval (if configured) with natural-language query."""
    if k < 1:
        return _result_error("k must be >= 1")
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            if entry.session.semantic_search is None:
                return _result_error(
                    "Semantic search is not configured for this session.",
                    hint="Set semantic_base_url (query or DOCQ_MCP_SEMANTIC_BASE_URL).",
                )
            results = entry.session.semantic_search.search(query=query, k=k)
            return {
                "ok": True,
                "query": query,
                "k": k,
                "env": entry.config.semantic_env,
                "results": results,
            }
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def add_import(
    libname: str,
    source: str,
    ctx: Context,
    doc_id: int | None = None,
) -> dict[str, Any]:
    """Add import in active doc (atoms or full `Require Import ...` statement)."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.doc_manager.add_import(libname=libname, source=source, doc_id=doc_id)
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def remove_import(
    libname: str,
    source: str,
    ctx: Context,
    doc_id: int | None = None,
) -> dict[str, Any]:
    """Remove import from active doc (atoms or full statement)."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.doc_manager.remove_import(libname=libname, source=source, doc_id=doc_id)
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def require_import(
    libname: str,
    source: str,
    ctx: Context,
) -> dict[str, Any]:
    """Alias of `add_import` for compatibility with subagent-style workflows."""
    out = await add_import(libname=libname, source=source, doc_id=None, ctx=ctx)
    if isinstance(out, dict):
        out.setdefault("applied_to_workspace", bool(out.get("ok", False)))
    return out


@mcp.tool()
async def prepare_intermediate_lemma(
    lemma_type: str,
    ctx: Context,
    lemma_name: str | None = None,
    doc_id: int | None = None,
    subagent_message: str | None = None,
) -> dict[str, Any]:
    """Prepare intermediate lemma workspace/subsession."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.prepare_intermediate_lemma(
                lemma_type=lemma_type,
                lemma_name=lemma_name,
                doc_id=doc_id,
                subagent_message=subagent_message,
            )
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def prove_intermediate_lemma(
    lemma_name: str,
    ctx: Context,
    prompt: str | None = None,
    subagent_message: str | None = None,
) -> dict[str, Any]:
    """Run proving subagent for a previously prepared lemma."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.prove_intermediate_lemma(
                lemma_name=lemma_name,
                prompt=prompt,
                subagent_message=subagent_message,
            )
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def add_intermediate_lemma(
    lemma_type: str,
    ctx: Context,
    lemma_name: str | None = None,
    prompt: str | None = None,
    subagent_message: str | None = None,
    doc_id: int | None = None,
) -> dict[str, Any]:
    """Prepare + prove intermediate lemma in one call."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.add_intermediate_lemma(
                lemma_type=lemma_type,
                lemma_name=lemma_name,
                prompt=prompt,
                subagent_message=subagent_message,
                doc_id=doc_id,
            )
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def drop_pending_intermediate_lemma(lemma_name: str, ctx: Context) -> dict[str, Any]:
    """Drop one pending lemma placeholder."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.drop_pending_intermediate_lemma(lemma_name=lemma_name)
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def list_pending_intermediate_lemmas(ctx: Context) -> Any:
    """List pending intermediate lemmas."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            pending = entry.session.list_pending_intermediate_lemmas()
            return {
                "ok": True,
                "pending_lemmas": pending,
                "pending_count": len(pending),
            }
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def pending_lemma_current_head(
    lemma_name: str,
    ctx: Context,
) -> dict[str, Any]:
    """Head status for one pending intermediate lemma branch."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.pending_lemma_current_head(lemma_name=lemma_name)
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def pending_lemma_list_states(
    lemma_name: str,
    ctx: Context,
) -> dict[str, Any]:
    """List state nodes for one pending intermediate lemma branch."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.pending_lemma_list_states(lemma_name=lemma_name)
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def pending_lemma_get_goals(
    lemma_name: str,
    ctx: Context,
    state_index: int | None = None,
) -> dict[str, Any]:
    """Get goals in one pending intermediate lemma branch."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.pending_lemma_get_goals(
                lemma_name=lemma_name,
                state_index=state_index,
            )
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def pending_lemma_run_tac(
    lemma_name: str,
    tactic: str,
    ctx: Context,
    state_index: int | None = None,
    branch_reason: str | None = None,
) -> dict[str, Any]:
    """Run one tactic in one pending intermediate lemma branch."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.pending_lemma_run_tac(
                lemma_name=lemma_name,
                tactic=tactic,
                state_index=state_index,
                branch_reason=branch_reason,
            )
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def remove_intermediate_lemma(
    lemma_name: str,
    ctx: Context,
    doc_id: int | None = None,
) -> dict[str, Any]:
    """Remove already materialized intermediate lemma from doc."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            return entry.session.doc_manager.remove_intermediate_lemma(
                lemma_name=lemma_name,
                doc_id=doc_id,
            )
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def materialized_source(
    ctx: Context,
    doc_id: int | None = None,
    state_index: int | None = None,
) -> dict[str, Any]:
    """Return fully materialized source with replayed proof script and final `Qed.`."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            text = entry.session.doc_manager.materialized_source(doc_id=doc_id, state_index=state_index)
            return {
                "ok": True,
                "doc_id": doc_id,
                "state_index": state_index,
                "content": text,
            }
    except Exception as exc:
        return _result_error(str(exc))


@mcp.tool()
async def validate_final_qed(
    ctx: Context,
    doc_id: int | None = None,
    state_index: int | None = None,
) -> dict[str, Any]:
    """Run final validation: explicit `Qed.` replay check plus SafeVerify."""
    try:
        entry = await _get_entry(ctx)
        async with entry.lock:
            ok, error = entry.session.validate_final_qed(doc_id=doc_id, state_index=state_index)
            return {
                "ok": bool(ok),
                "doc_id": doc_id,
                "state_index": state_index,
                "error": error,
            }
    except Exception as exc:
        return _result_error(str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DocQ FastMCP tool server")
    parser.add_argument("--host", default=os.getenv("DOCQ_MCP_BIND_HOST", DEFAULT_MCP_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("DOCQ_MCP_BIND_PORT", str(DEFAULT_MCP_PORT))))
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default=os.getenv("DOCQ_MCP_TRANSPORT", "http"),
        help="FastMCP transport.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
