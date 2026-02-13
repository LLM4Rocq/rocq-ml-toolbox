from __future__ import annotations
import os
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.context import Context
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_request

from .navigation import CodebaseNavigator

mcp = FastMCP(name="Rocq Prover MCP")

code_navigator = CodebaseNavigator(Path('annotated/export/theostos'))

def _extract_env() -> str:
    try:
        req = get_http_request()
        env = req.query_params.get("env")
        if env:
            return env.strip()
    except Exception:
        pass

    env = os.getenv("MCP_ENV", "")
    return env

class EnvMiddleware(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        # Complete MCP handshake first. After this, session-scoped state is safe to set.
        ctx = context.fastmcp_context
        if ctx is not None:
            env = _extract_env()
            ctx.set_state("MCP_ENV", env)
        return await call_next(context)

mcp.add_middleware(EnvMiddleware())

@mcp.tool()
async def open_file(path: list[str], ctx: Context) -> str:
    """
    Open the leaf source file for a given annotated path (under the current environment)
    and return its content.

    Input:
      - path: list of path segments, e.g. ["algebra", "group"].

    Output:
      - On success: the content of source_wo_proof.v for that leaf.
      - On error: an explanation, including valid children / suggestions where possible.
    """
    global code_navigator
    mcp_env = ctx.get_state("MCP_ENV")

    resp = code_navigator.open(mcp_env, path, filename="source_wo_proof.v")
    if resp.get("ok"):
        return resp["result"]
    return f"Error: {resp['result']}"

@mcp.tool()
async def explore_codebase(path: list[str], ctx: Context) -> str:
    """
    Browse the annotated (file-wise) codebase for the current environment.

    Input:
      - path: list of path segments to open, e.g. [] (root), ["algebra"], ["algebra", "group"].
        Segments must match existing subfolders exactly.

    Output:
      - On success: a compact ASCII tree of the children at the requested location (dirs and file-like leaves).
      - On error: an explanation plus the valid children of the last existing path, with close-match suggestions.

    Usage pattern:
      1) Call with [] to see top-level entries.
      2) Call again with a deeper path based on what was returned.
    """
    global code_navigator
    mcp_env = ctx.get_state("MCP_ENV")

    resp = code_navigator.explore(mcp_env, path)
    if resp['ok']:
        return resp['result']
    else:
        return f"Error: {resp['result']}"

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8001)
