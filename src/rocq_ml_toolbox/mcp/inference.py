from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Self
from pathlib import Path
from fastmcp import FastMCP
from fastmcp.server.context import Context
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.middleware import CallNext
from mcp.types import CallToolResult, Request, TextContent

from rocq_ml_toolbox.inference.client import PytanqueExtended
from pytanque.client import PetanqueError
from pytanque import State, Goal
from fastmcp.server.dependencies import get_http_request

from .navigation import CodebaseNavigator

mcp = FastMCP(name="Rocq Prover MCP")

@dataclass
class ActiveSession:
    client: PytanqueExtended
    history: List[Tuple[str, State]] = field(default_factory=list)

    @property
    def state(self) -> Optional[State]:
        if not self.history:
            return None
        return self.history[-1][1]
    
    def undo(self, step: int):
        if step > 0:
            self.history = self.history[:-step]
    
    @classmethod
    def from_json(cls, x) -> Self:
        return cls(
            PytanqueExtended.from_json(x['client']),
            [(tac, State.from_json(st)) for tac, st in x['history']]
        )

    def to_json(self) -> Any:
        return {
            "client": self.client.to_json(),
            "history": [(tac, st.to_json()) for tac, st in self.history]
        }

class PytanqueMiddleware(Middleware):
    async def on_initialize(self, context: MiddlewareContext, call_next):
        # Complete MCP handshake first. After this, session-scoped state is safe to set.
        await call_next(context)

        ctx = context.fastmcp_context
        if ctx is None:
            return

        # Avoid re-initializing if client reconnects/re-initializes
        existing = await ctx.get_state("active_session")
        if existing is not None:
            return

        pet_client = PytanqueExtended("127.0.0.1", 5000)
        pet_client.connect()
        active_session = ActiveSession(pet_client)

        await ctx.set_state("active_session", active_session.to_json())
    
    async def on_request(self, context: MiddlewareContext, call_next) -> Any:
        # very hacky, but right now issue with OpenAI MCP implementation (inconsistent session_id..)
        if context.fastmcp_context and context.fastmcp_context.request_context and context.fastmcp_context.request_context.meta:
            dump = context.fastmcp_context.request_context.meta.model_dump()
            if 'openai/session' in dump:
                openai_session:str = dump['openai/session']
                setattr(context.fastmcp_context.session, "_fastmcp_state_prefix", openai_session)
        return await call_next(context)
mcp.add_middleware(PytanqueMiddleware())

async def _set_session(active_session: ActiveSession, ctx: Context):
    await ctx.set_state("active_session", active_session.to_json())

async def _get_session(ctx: Context) -> ActiveSession:
    active_session = await ctx.get_state("active_session")
    if active_session is None:
        # Fallback if middleware wasn't run for some reason
        pet_client = PytanqueExtended("127.0.0.1", 5000)
        pet_client.connect()
        active_session = ActiveSession(pet_client)
        await _set_session(active_session, ctx)
    else:
        active_session = ActiveSession.from_json(active_session)
    return active_session

def _str_of_goals(goals: List[Goal]) -> str:
    if not goals:
        result = "Proof is finished."
    else:
        result = f"\nCurrent goals ({len(goals)}):\n"
        result += "\n".join(
            [f"Goal {i+1}:\n{goal.pp}" for i, goal in enumerate(goals)]
        )
    return result

@mcp.tool()
async def start_proof(ctx: Context) -> str:
    """Start (or reset) proof of the given theorem."""
    active_session = await _get_session(ctx)
    client = active_session.client
    path = client.empty_file()
    init_state = client.get_root_state(path)
    active_session.history = []
    state = client.run(init_state,"Require Import Ensembles Reals FinFun.")
    state = client.run(state, """Theorem putnam_1972_a2
    : (forall (S : Type) (Smul : S -> S -> S), (forall x y : S, (Smul x (Smul x y) = y /\\ Smul (Smul y x) x = y)) -> (forall x y : S, Smul x y = Smul y x)) /\\
        (exists (S : Type) (Smul : S -> S -> S), (forall x y : S, (Smul x (Smul x y) = y /\\ Smul (Smul y x) x = y)) /\\ ~(forall x y z : S, Smul x (Smul y z) = Smul (Smul x y) z)).""")
    
    active_session.history.append(("", state))
    new_goals = client.goals(state)

    if not new_goals:
        result = "Error: No current goals."
    else:
        result = "Proof started.\n" + _str_of_goals(new_goals)
    await _set_session(active_session, ctx)
    return result

@mcp.tool()
async def run_tac(cmd: str, ctx: Context) -> str:
    """Run a tactic in the active session, return new goals if succeed."""
    active_session = await _get_session(ctx)
    client = active_session.client
    state = active_session.state
    if not state:
        return "Error: Start a proof before calling run_tac."
    
    try:
        new_state = client.run(state, cmd)
    except PetanqueError as e:
        return f"Error {e.code} when running `{cmd}`:\n{e.message}"
    new_goals = client.goals(new_state)
    result = _str_of_goals(new_goals)
    active_session.history.append((cmd, new_state))
    await _set_session(active_session, ctx)
    return result

@mcp.tool()
async def undo(steps: int, ctx: Context) -> str:
    """Undo the last N steps in the active session."""
    active_session = await _get_session(ctx)
    client = active_session.client
    active_session.undo(steps)
    
    if active_session.state:
        result = "Current proof:\n" + "\n".join([s for s,_ in active_session.history])
        new_goals = client.goals(active_session.state)
        result += '\n current goals:\n' + _str_of_goals(new_goals)
        await _set_session(active_session, ctx)
        return result
    return f"Error: Undo not admissible {steps} > number of tactics"

code_navigator = CodebaseNavigator(Path('export/theostos'))

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
            await ctx.set_state("MCP_ENV", env)
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
    mcp_env = await ctx.get_state("MCP_ENV")

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
    mcp_env = await ctx.get_state("MCP_ENV")

    resp = code_navigator.explore(mcp_env, path)
    if resp['ok']:
        return resp['result']
    else:
        return f"Error: {resp['result']}"

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
