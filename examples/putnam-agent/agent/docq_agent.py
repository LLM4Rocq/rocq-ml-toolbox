from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic_ai import Agent, ModelRetry, RunContext, capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits

from .doc_manager import BranchSession, DocumentManager, StateNode, parse_last_target_layout
from .docstring_tools import SemanticDocSearchClient
from .library_tools import TocExplorer, read_source_via_client

DOCQ_SYSTEM_PROMPT = (
    "You are editing and proving Rocq code with strict tool usage.\n"
    "Rules:\n"
    "- Use `explore_toc` incrementally and discover root entries first.\n"
    "- Use `read_source_file` to inspect library files with optional line window.\n"
    "- For library reads, pass TOC-relative file paths exactly as returned by `explore_toc`.\n"
    "- Use `show_workspace` or `show_doc(doc_id)` to inspect virtual files.\n"
    "- Use `completion_status(doc_id?)` before finishing; do not end while `latest_goals_count > 0`.\n"
    "- Use `list_docs` and `checkout_doc` to navigate document branches.\n"
    "- Use `run_tac` from any known state index (optionally scoped by `doc_id`).\n"
    "- State graph semantics: every successful `run_tac` creates a new state node.\n"
    "  You can roll back/branch by re-running `run_tac` from any previous `state_index`.\n"
    "- `show_workspace` includes a `states` list with `state_index`/`parent_state_index` so rollback targets are visible.\n"
    "  Use `list_states`/`show_workspace` before branching from older states.\n"
    "- Use ASCII Coq syntax in tool arguments: `forall`, `exists`, `->`, `/\\`, `\\/`, `<=`, `>=`.\n"
    "  Do NOT use Unicode logic symbols (`∀`, `∃`, `→`, `∧`, `∨`, `≤`, `≥`, ...).\n"
    "- For imports, use `add_import(libname, source, doc_id?)` and `remove_import(...)`.\n"
    "- Import format: give only atoms, e.g. libname='mathcomp.fingroup', source='perm'.\n"
    "  Do NOT pass full commands like `From ... Require Import ...`.\n"
    "  Example call: add_import(libname='mathcomp.fingroup', source='perm').\n"
    "- For helper statements, prefer phased tools:\n"
    "  `prepare_intermediate_lemma` -> `prove_intermediate_lemma` -> `drop_pending_intermediate_lemma`.\n"
    "- You can pass lemma-proving guidance with `subagent_message` on "
    "`prepare_intermediate_lemma` / `prove_intermediate_lemma` / `add_intermediate_lemma`.\n"
    "  Use it for relevant imports, local-goal focus, and proof strategy.\n"
    "- Intermediate lemma format: `lemma_name` is only the identifier; `lemma_type` is only the proposition.\n"
    "  Do NOT include `Lemma name :` prefix nor `Proof.`/`Qed.` in `lemma_type`.\n"
    "  Example call: prepare_intermediate_lemma(lemma_name='helper_card', lemma_type='forall n : nat, n > 0 -> True').\n"
    "- Convenience path `add_intermediate_lemma` is still available.\n"
    "- If a helper lemma is problematic, call `remove_intermediate_lemma(lemma_name)`.\n"
    "- If stuck on lemma proving, the sub-agent can abort and report why.\n"
    "Reasoning quality bar:\n"
    "- Before proposing a proof step, inspect the actual current goal/hypotheses (`completion_status` + `get_goals`).\n"
    "- Choose tactics based on goal shape and available hypotheses; do not guess missing identifiers.\n"
    "- After any failure, explain the concrete blocker from the error and adapt the next step.\n"
    "- Avoid repeating the same failing tactic pattern; if blocked, gather context, import needed libraries, "
    "or reformulate with an intermediate lemma.\n"
    "Reason-before-tool protocol:\n"
    "- Before each tool call, first write a short rationale with exactly these labels:\n"
    "  `Goal:` `Observation:` `Plan:` `Expected result:`\n"
    "- Make the rationale specific to the current proof state; avoid generic text.\n"
    "- If the last step failed, your next rationale must reference that concrete error and what changes now.\n"
    "- Prefer deliberate, high-signal moves over rapid low-information tool spam."
)

LEMMA_SUBAGENT_SYSTEM_PROMPT = (
    "You are proving one intermediate lemma.\n"
    "Use tools:\n"
    "- `explore_toc`, `semantic_doc_search`, `read_source_file`\n"
    "- `show_workspace`, `read_workspace_source`\n"
    "- `list_states`, `get_goals`, `run_tac`\n"
    "- `require_import(libname, source)` if the lemma proof needs a new import.\n"
    "- `abort(explanation)` if the lemma is likely wrong/unprovable.\n"
    "Goal: finish with no remaining goals."
)

SUBAGENT_COMPRESSION_SYSTEM_PROMPT = (
    "You compress a lemma-proof subagent run for continuation after context reset.\n"
    "Return a faithful handoff in plain text.\n"
    "Keep exact names for lemmas/imports/doc_id/state and provide concrete next proof steps.\n"
    "Never invent solved goals, imported modules, or successful tactics."
)


class _CompressionRequested(Exception):
    def __init__(self, *, reason: str, req_input_tokens: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.req_input_tokens = req_input_tokens


def _lemma_subagent_prompt(*, include_semantic_tool: bool) -> str:
    retrieval_line = "`explore_toc`, `semantic_doc_search`, `read_source_file`"
    if not include_semantic_tool:
        retrieval_line = "`explore_toc`, `read_source_file`"
    return (
        "You are proving one intermediate lemma.\n"
        "Use tools:\n"
        f"- {retrieval_line}\n"
        "- For library reads, pass TOC-relative file paths exactly as returned by `explore_toc`.\n"
        "- `read_source_file` is for library files only; do NOT pass workspace `/tmp/...` paths.\n"
        "- For current proof file content, always use `read_workspace_source`.\n"
        "- `show_workspace`, `read_workspace_source`\n"
        "- `list_states`, `get_goals`, `run_tac`\n"
        "- State graph semantics: each successful `run_tac` creates a new state.\n"
        "  You may roll back/branch by invoking `run_tac` from any earlier `state_index`.\n"
        "- `show_workspace` returns `states` (with `state_index` and `parent_state_index`) so rollback points are explicit.\n"
        "- Use ASCII Coq syntax in tool arguments (`forall`, `exists`, `->`, `/\\`, `\\/`, `<=`, `>=`).\n"
        "- `run_tac` must be one short tactic step per call (single line, no pasted scripts).\n"
        "- Never send multi-line tactics, `Proof.` blocks, or `Qed.`/`Admitted.`.\n"
        "- Reason in a disciplined loop: inspect goals -> pick one justified tactic -> run it -> re-check goals.\n"
        "- When a tactic fails, use the exact error to choose a different next step (do not blind-retry).\n"
        "- Prefer mathematically meaningful progress over superficial/no-op tactics.\n"
        "- Before each tool call, write a short rationale with labels: `Goal:` `Observation:` `Plan:` `Expected result:`.\n"
        "- Rationale must mention the current local goal (or exact blocker) and why the chosen tool/tactic is next.\n"
        "- If you receive an error, explicitly change strategy in the next rationale; do not repeat unchanged attempts.\n"
        "- After each failure, immediately re-anchor with `get_goals` or `read_workspace_source`, then try one corrected step.\n"
        "- Prefer standard Coq tactics unless ssreflect imports/tactics are explicitly available in the workspace.\n"
        "- `require_import(libname, source)` if the lemma proof needs a new import.\n"
        "- Import format: only atoms; do not pass full `From ... Require Import ...` commands.\n"
        "- Example call: require_import(libname='mathcomp.fingroup', source='perm').\n"
        "- `abort(explanation)` if the lemma is likely wrong/unprovable.\n"
        "Goal: finish with no remaining goals."
    )


@dataclass
class LemmaSubSession:
    branch: BranchSession
    client: Any
    toc_explorer: TocExplorer
    semantic_search: SemanticDocSearchClient | None = None
    logger: Callable[[str], None] | None = None
    abort_reason: str | None = None
    required_imports: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PendingLemma:
    base_doc_id: int
    lemma_name: str
    lemma_type: str
    sub_branch: BranchSession
    subagent_message: str | None = None


class _LockedClientProxy:
    """Serialize all client method calls to avoid concurrent JSON-RPC id races."""

    def __init__(self, inner_client: Any, *, lock: threading.RLock | None = None):
        object.__setattr__(self, "_inner_client", inner_client)
        object.__setattr__(self, "_lock", lock or threading.RLock())

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner_client, name)
        if not callable(attr):
            return attr

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)

        return _wrapped

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_inner_client", "_lock"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._inner_client, name, value)


_IDENT_RE = re.compile(r"^[A-Za-z0-9_.]+$")
_IMPORT_RE = re.compile(r"^\s*(From\s+\S+\s+Require\s+(Import|Export)|Require\s+Import|Import|Export)\b")


def _line_numbered_from_text(content: str, *, start_line: int = 1) -> str:
    lines = content.splitlines()
    if not lines:
        return ""
    width = max(3, len(str(start_line + len(lines))))
    return "\n".join(f"{idx + start_line:>{width}}: {line}" for idx, line in enumerate(lines))


def _join_lines(lines: list[str], *, trailing_newline: bool = True) -> str:
    text = "\n".join(lines)
    if trailing_newline and not text.endswith("\n"):
        text += "\n"
    return text


def _branch_tactics_path(branch: BranchSession) -> list[str]:
    idx = branch.latest_state_index
    path: list[str] = []
    while idx > 0:
        node = branch.nodes[idx]
        if node.tactic:
            path.append(node.tactic)
        parent = node.parent_index
        if parent is None:
            break
        idx = parent
    path.reverse()
    return path


def _has_import_statement(content: str, *, libname: str, source: str) -> bool:
    pattern = re.compile(
        rf"^\s*From\s+{re.escape(libname)}\s+Require\s+Import\s+{re.escape(source)}\s*\.\s*$"
    )
    layout = parse_last_target_layout(content)
    return any(bool(pattern.match(line)) for line in layout.prefix_lines)


def _insert_import_statement(content: str, *, libname: str, source: str) -> str:
    layout = parse_last_target_layout(content)
    prefix = list(layout.prefix_lines)
    statement = f"From {libname} Require Import {source}."
    if statement in prefix:
        return content

    last_import = -1
    for idx, line in enumerate(prefix):
        if _IMPORT_RE.search(line):
            last_import = idx
    insert_idx = 0 if last_import < 0 else last_import + 1
    prefix.insert(insert_idx, statement)
    return _join_lines(prefix + layout.target_lines + layout.suffix_lines, trailing_newline=True)


def _rebuild_branch_with_import(branch: BranchSession, *, libname: str, source: str) -> tuple[BranchSession, dict[str, Any]]:
    if _has_import_statement(branch.source_content, libname=libname, source=source):
        return branch, {"added": False, "replayed_tactics": 0}

    new_content = _insert_import_statement(branch.source_content, libname=libname, source=source)
    tmp_path = Path(branch.client.tmp_file(content=new_content)).resolve()
    layout = parse_last_target_layout(new_content)
    state0 = branch.client.get_state_at_pos(
        str(tmp_path),
        layout.proof_line,
        layout.proof_character,
        timeout=branch.timeout,
    )
    rebuilt = BranchSession(
        client=branch.client,
        doc_id=branch.doc_id,
        source_path=tmp_path,
        source_content=new_content,
        layout=layout,
        timeout=branch.timeout,
        nodes=[StateNode(index=0, parent_index=None, tactic=None, state=state0)],
        logger=branch.logger,
    )
    replayed = 0
    state_index = 0
    for tactic in _branch_tactics_path(branch):
        out = rebuilt.run_tac(state_index, tactic)
        if not out.get("ok", False):
            raise ValueError(
                f"Failed to replay tactic after adding import: {out.get('error', 'unknown error')}"
            )
        state_index = int(out["new_state_index"])
        replayed += 1
    return rebuilt, {"added": True, "replayed_tactics": replayed}


def _adopt_branch_state(dst: BranchSession, src: BranchSession) -> None:
    dst.source_path = src.source_path
    dst.source_content = src.source_content
    dst.layout = src.layout
    dst.nodes = src.nodes


def _rebuild_branch_on_client(branch: BranchSession, *, client: Any) -> BranchSession:
    """Clone a branch onto another client by replaying its tactic path."""
    tmp_path = Path(client.tmp_file(content=branch.source_content)).resolve()
    layout = parse_last_target_layout(branch.source_content)
    state0 = client.get_state_at_pos(
        str(tmp_path),
        layout.proof_line,
        layout.proof_character,
        timeout=branch.timeout,
    )
    rebuilt = BranchSession(
        client=client,
        doc_id=branch.doc_id,
        source_path=tmp_path,
        source_content=branch.source_content,
        layout=layout,
        timeout=branch.timeout,
        nodes=[StateNode(index=0, parent_index=None, tactic=None, state=state0)],
        logger=branch.logger,
    )
    state_index = 0
    for tactic in _branch_tactics_path(branch):
        out = rebuilt.run_tac(state_index, tactic)
        if not out.get("ok", False):
            raise ValueError(
                "Failed to replay tactic while cloning subagent workspace: "
                f"{out.get('error', 'unknown error')}"
            )
        state_index = int(out["new_state_index"])
    return rebuilt


def _render_content_window(
    content: str,
    *,
    line: int | None = None,
    before: int = 20,
    after: int = 20,
) -> dict[str, Any]:
    lines = content.splitlines()
    if line is None:
        return {
            "mode": "full",
            "total_lines": len(lines),
            "content": _line_numbered_from_text(content, start_line=1),
        }

    if line < 1:
        raise ValueError("line must be >= 1")
    start = max(1, line - max(0, before))
    end = min(len(lines), line + max(0, after))
    snippet = "\n".join(lines[start - 1 : end])
    return {
        "mode": "around_line",
        "line": line,
        "start_line": start,
        "end_line": end,
        "total_lines": len(lines),
        "content": _line_numbered_from_text(snippet, start_line=start),
    }


def _read_source_error_payload(*, requested_path: str, error: str) -> dict[str, Any]:
    hint = (
        "If this is a library file, call `explore_toc` first and then call `read_source_file` "
        "with a TOC-relative path from the explorer entries "
        "(for example `mathcomp/fingroup/perm.v`)."
    )
    if Path(requested_path).is_absolute():
        hint += " Do not pass absolute filesystem paths."
    return {
        "ok": False,
        "requested_path": requested_path,
        "error": error,
        "hint": hint,
    }


def _normalize_read_source_path(path: str | list[str] | None) -> str | None:
    if path is None:
        return None
    if isinstance(path, list):
        parts: list[str] = []
        for item in path:
            token = str(item).strip()
            if not token:
                continue
            token = token.strip("/")
            if token:
                parts.append(token)
        return "/".join(parts)
    token = str(path).strip()
    return token or None


def build_docq_subagent(
    model: Any = None,
    *,
    retries: int = 1,
    include_semantic_tool: bool = True,
) -> Agent[LemmaSubSession, str]:
    agent = Agent(
        model=model,
        deps_type=LemmaSubSession,
        output_type=str,
        name="docq-lemma-subagent",
        system_prompt=_lemma_subagent_prompt(include_semantic_tool=include_semantic_tool),
        retries=retries,
    )

    def _sub_log(ctx: RunContext[LemmaSubSession], message: str) -> None:
        logger = ctx.deps.logger
        if logger is not None:
            logger(f"subagent | {message}")

    @agent.tool
    def list_states(ctx: RunContext[LemmaSubSession]) -> list[int]:
        states = ctx.deps.branch.available_state_indexes
        _sub_log(ctx, f"list_states -> {states}")
        return states

    @agent.tool
    def explore_toc(ctx: RunContext[LemmaSubSession], path: list[str] | None = None) -> dict[str, Any]:
        req_path = path or []
        _sub_log(ctx, f"explore_toc(path={req_path})")
        out = ctx.deps.toc_explorer.explore(req_path)
        _sub_log(ctx, f"explore_toc -> ok={out.get('ok', False)}")
        return out

    if include_semantic_tool:

        @agent.tool
        def semantic_doc_search(
            ctx: RunContext[LemmaSubSession],
            query: str,
            k: int = 5,
        ) -> dict[str, Any]:
            _sub_log(ctx, f"semantic_doc_search(query={query!r}, k={k})")
            if ctx.deps.semantic_search is None:
                raise ModelRetry("Semantic search is not configured for this session.")
            if k < 1:
                raise ModelRetry("k must be >= 1")
            results = ctx.deps.semantic_search.search(query=query, k=k)
            _sub_log(ctx, f"semantic_doc_search -> {len(results)} results")
            return {"query": query, "k": k, "results": results}

    @agent.tool
    def read_source_file(
        ctx: RunContext[LemmaSubSession],
        path: str | list[str] | None = None,
        line: int | None = None,
        before: int = 20,
        after: int = 20,
    ) -> dict[str, Any]:
        source_path = str(ctx.deps.branch.source_path)
        normalized_path = _normalize_read_source_path(path)
        requested_path = source_path if normalized_path is None else normalized_path
        path_for_log = "<workspace>" if normalized_path is None else requested_path
        _sub_log(
            ctx,
            f"read_source_file(path={path_for_log!r}, line={line}, before={before}, after={after})",
        )
        if normalized_path is None or requested_path == source_path:
            payload = {
                "ok": False,
                "error": "Workspace path is not allowed in subagent read_source_file.",
                "hint": (
                    "Use `read_workspace_source` for workspace content. "
                    "Use `read_source_file` only with TOC-relative library paths."
                ),
                "source_kind": "workspace_doc",
            }
            _sub_log(ctx, "read_source_file -> rejected (workspace path); use read_workspace_source")
            return payload
        if Path(requested_path).is_absolute():
            payload = _read_source_error_payload(
                requested_path=requested_path,
                error="Absolute filesystem paths are not supported by this tool.",
            )
            _sub_log(ctx, f"read_source_file -> failed: {payload['error']}")
            return payload
        try:
            payload = read_source_via_client(
                ctx.deps.client,
                requested_path,
                line=line,
                before=before,
                after=after,
            )
            payload["ok"] = True
            _sub_log(ctx, f"read_source_file -> ok (resolved_path={payload.get('resolved_path')!r})")
            return payload
        except Exception as exc:
            error = str(exc)
            _sub_log(ctx, f"read_source_file -> failed: {error}")
            return _read_source_error_payload(requested_path=requested_path, error=error)

    @agent.tool
    def show_workspace(ctx: RunContext[LemmaSubSession]) -> dict[str, Any]:
        _sub_log(ctx, "show_workspace()")
        out = {
            "doc_id": ctx.deps.branch.doc_id,
            "states": ctx.deps.branch.list_states(),
            "content": _line_numbered_from_text(ctx.deps.branch.source_content, start_line=1),
        }
        _sub_log(ctx, "show_workspace -> ok")
        return out

    @agent.tool
    def read_workspace_source(
        ctx: RunContext[LemmaSubSession],
        line: int | None = None,
        before: int = 20,
        after: int = 20,
    ) -> dict[str, Any]:
        _sub_log(ctx, f"read_workspace_source(line={line}, before={before}, after={after})")
        try:
            payload = _render_content_window(
                ctx.deps.branch.source_content,
                line=line,
                before=before,
                after=after,
            )
            payload["doc_id"] = ctx.deps.branch.doc_id
            payload["ok"] = True
            _sub_log(ctx, "read_workspace_source -> ok")
            return payload
        except Exception as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def get_goals(ctx: RunContext[LemmaSubSession], state_index: int = 0) -> dict[str, Any]:
        try:
            return ctx.deps.branch.get_goals(state_index)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def run_tac(
        ctx: RunContext[LemmaSubSession],
        state_index: int = 0,
        tactic: str = "idtac.",
    ) -> dict[str, Any]:
        if "\n" in tactic or "\r" in tactic:
            raise ModelRetry(
                "run_tac expects one tactic step per call (single line). "
                "Split scripts into multiple run_tac calls."
            )
        try:
            return ctx.deps.branch.run_tac(state_index, tactic)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def require_import(
        ctx: RunContext[LemmaSubSession],
        libname: str = "Stdlib",
        source: str = "List",
    ) -> dict[str, Any]:
        """Register needed import atoms only. Example: require_import(libname='mathcomp.fingroup', source='perm')."""
        _sub_log(ctx, f"require_import(libname={libname!r}, source={source!r})")
        if not _IDENT_RE.match(libname):
            raise ModelRetry(f"Invalid libname={libname!r}.")
        if not _IDENT_RE.match(source):
            raise ModelRetry(f"Invalid source={source!r}.")
        pair = (libname, source)
        if pair not in ctx.deps.required_imports:
            ctx.deps.required_imports.append(pair)
        try:
            rebuilt, info = _rebuild_branch_with_import(ctx.deps.branch, libname=libname, source=source)
            _adopt_branch_state(ctx.deps.branch, rebuilt)
            _sub_log(
                ctx,
                "require_import applied to workspace "
                f"(added={info.get('added')}, replayed_tactics={info.get('replayed_tactics')})",
            )
        except Exception as exc:
            raise ModelRetry(f"Failed to apply import in current subagent workspace: {exc}") from exc
        return {
            "ok": True,
            "applied_to_workspace": True,
            "required_imports": [
                {"libname": lib, "source": src} for (lib, src) in ctx.deps.required_imports
            ],
        }

    @agent.tool
    def abort(ctx: RunContext[LemmaSubSession], explanation: str = "not provable") -> str:
        ctx.deps.abort_reason = explanation.strip() or "not provable"
        _sub_log(ctx, f"abort({ctx.deps.abort_reason!r})")
        return f"ABORT: {ctx.deps.abort_reason}"

    return agent


@dataclass
class DocqAgentSession:
    client: Any
    source_path: Path
    env: str | None
    doc_manager: DocumentManager
    toc_explorer: TocExplorer
    semantic_search: SemanticDocSearchClient | None = None
    timeout: float = 60.0
    usage: RunUsage = field(default_factory=RunUsage)
    usage_limits: UsageLimits = field(default_factory=lambda: UsageLimits(tool_calls_limit=120, request_limit=200))
    logger: Callable[[str], None] | None = None
    subagent_model: Any = None
    subagent_retries: int = 1
    subagent_threshold_compression: int = 100_000
    subagent_max_compression_rounds: int = 6
    include_semantic_tool: bool = True
    trace_message_callback: Callable[[str, Any], None] | None = None
    trace_event_callback: Callable[[str, Any], None] | None = None
    trace_request_callback: Callable[[str, Any], None] | None = None
    pending_lemmas: dict[str, PendingLemma] = field(default_factory=dict)
    _subagent_compression_agent: Agent[Any, str] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_source(
        cls,
        client: Any,
        source_path: str | Path,
        *,
        env: str | None = None,
        timeout: float = 60.0,
        connect: bool = True,
        logger: Callable[[str], None] | None = None,
        semantic_base_url: str | None = None,
        semantic_route: str = "/search",
        semantic_api_key: str | None = None,
        max_tool_calls: int = 120,
        max_requests: int | None = 200,
        subagent_model: Any = None,
        subagent_retries: int = 1,
        threshold_compression: int = 100_000,
        subagent_max_compression_rounds: int = 6,
        include_semantic_tool: bool = True,
        trace_message_callback: Callable[[str, Any], None] | None = None,
        trace_event_callback: Callable[[str, Any], None] | None = None,
        trace_request_callback: Callable[[str, Any], None] | None = None,
    ) -> "DocqAgentSession":
        locked_client = _LockedClientProxy(client)
        if connect:
            locked_client.connect()
        source = Path(source_path).resolve()
        manager = DocumentManager(locked_client, source, timeout=timeout, logger=logger)
        explorer = TocExplorer(locked_client, env=env)
        semantic_client = None
        if semantic_base_url:
            semantic_client = SemanticDocSearchClient(
                base_url=semantic_base_url,
                route=semantic_route,
                api_key=semantic_api_key,
                timeout=timeout,
            )
        session = cls(
            client=locked_client,
            source_path=source,
            env=env,
            doc_manager=manager,
            toc_explorer=explorer,
            semantic_search=semantic_client,
            timeout=timeout,
            usage=RunUsage(),
            usage_limits=UsageLimits(tool_calls_limit=max_tool_calls, request_limit=max_requests),
            logger=logger,
            subagent_model=subagent_model,
            subagent_retries=subagent_retries,
            subagent_threshold_compression=threshold_compression,
            subagent_max_compression_rounds=subagent_max_compression_rounds,
            include_semantic_tool=include_semantic_tool,
            trace_message_callback=trace_message_callback,
            trace_event_callback=trace_event_callback,
            trace_request_callback=trace_request_callback,
        )
        session.doc_manager.mutation_validator = session._validate_mutation_in_fresh_session
        return session

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def _emit_trace_messages(self, source: str, messages: list[Any]) -> None:
        cb = self.trace_message_callback
        if cb is None:
            return
        for msg in messages:
            try:
                cb(source, msg)
            except Exception:
                return

    def _emit_trace_event(self, source: str, event: Any) -> None:
        cb = self.trace_event_callback
        if cb is None:
            return
        try:
            cb(source, event)
        except Exception:
            return

    def _emit_trace_request(self, source: str, payload: Any) -> None:
        cb = self.trace_request_callback
        if cb is None:
            return
        try:
            cb(source, payload)
        except Exception:
            return

    def _try_make_fresh_client(self) -> Any | None:
        base_client = getattr(self.client, "_inner_client", self.client)
        host = getattr(base_client, "host", None)
        port = getattr(base_client, "port", None)
        if host is None or port is None:
            return None
        try:
            fresh = type(base_client)(host, port)
        except Exception:
            return None
        if hasattr(base_client, "timeout_http") and hasattr(fresh, "timeout_http"):
            try:
                fresh.timeout_http = base_client.timeout_http
            except Exception:
                pass
        connect = getattr(fresh, "connect", None)
        if callable(connect):
            connect()
        return fresh

    @staticmethod
    def _close_client_quietly(client: Any) -> None:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _validate_mutation_in_fresh_session(self, doc_id: int, label: str) -> tuple[bool, str | None]:
        fresh = self._try_make_fresh_client()
        if fresh is None:
            return True, None
        try:
            node = self.doc_manager.nodes.get(doc_id)
            if node is None:
                return False, f"Unknown doc_id={doc_id} during validation."
            layout = parse_last_target_layout(node.content)
            tmp_path = fresh.tmp_file(content=node.content)
            state0 = fresh.get_state_at_pos(
                str(tmp_path),
                layout.proof_line,
                layout.proof_character,
                timeout=self.timeout,
            )
            if label.startswith("add_lemma:"):
                lemma_name = label.split(":", 1)[1].strip()
                if lemma_name:
                    fresh.run(state0, f"pose proof ({lemma_name}).", timeout=self.timeout)
            return True, None
        except Exception as exc:
            return False, str(exc)
        finally:
            self._close_client_quietly(fresh)

    def validate_final_qed(
        self,
        *,
        doc_id: int | None = None,
        state_index: int | None = None,
    ) -> tuple[bool, str | None]:
        try:
            status = self.doc_manager.completion_status(doc_id=doc_id)
        except Exception as exc:
            return False, f"Unable to read completion status before final Qed check: {exc}"

        resolved_doc_id = int(status.get("doc_id", self.doc_manager.head_doc_id))
        latest_state = int(
            status.get(
                "latest_state_index",
                self.doc_manager.sessions[resolved_doc_id].latest_state_index,
            )
        )
        target_state = latest_state if state_index is None else int(state_index)
        if not bool(status.get("latest_proof_finished", False)):
            return False, "Head proof is not finished yet; cannot run final Qed check."

        probe_client = self._try_make_fresh_client()
        owns_probe_client = probe_client is not None
        if probe_client is None:
            probe_client = self.client
        try:
            node = self.doc_manager.nodes[resolved_doc_id]
            branch = self.doc_manager.sessions[resolved_doc_id]
            proof_script = branch.proof_script(target_state)
            tactics = [line for line in proof_script.splitlines() if line.strip()]
            layout = parse_last_target_layout(node.content)
            tmp_path = probe_client.tmp_file(content=node.content)
            state = probe_client.get_state_at_pos(
                str(tmp_path),
                layout.proof_line,
                layout.proof_character,
                timeout=self.timeout,
            )
            for tactic in tactics:
                state = probe_client.run(state, tactic, timeout=self.timeout)
            probe_client.run(state, "Qed.", timeout=self.timeout)
            self._log(
                f"final Qed check ok(doc_id={resolved_doc_id}, state_index={target_state}, "
                f"replayed_tactics={len(tactics)})"
            )
            return True, None
        except Exception as exc:
            self._log(
                f"final Qed check failed(doc_id={resolved_doc_id}, state_index={target_state}): {exc}"
            )
            return False, str(exc)
        finally:
            if owns_probe_client:
                self._close_client_quietly(probe_client)

    @staticmethod
    def _normalize_required_imports(raw: Any) -> list[tuple[str, str]]:
        if not isinstance(raw, list):
            return []
        out: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            libname = item.get("libname")
            source = item.get("source")
            if not isinstance(libname, str) or not isinstance(source, str):
                continue
            libname = libname.strip()
            source = source.strip()
            if not _IDENT_RE.match(libname) or not _IDENT_RE.match(source):
                continue
            pair = (libname, source)
            if pair in seen:
                continue
            seen.add(pair)
            out.append(pair)
        return out

    def _doc_has_import(self, *, doc_id: int, libname: str, source: str) -> bool:
        node = self.doc_manager.nodes[doc_id]
        layout = parse_last_target_layout(node.content)
        pattern = re.compile(
            rf"^\s*From\s+{re.escape(libname)}\s+Require\s+Import\s+{re.escape(source)}\s*\.\s*$"
        )
        return any(bool(pattern.match(line)) for line in layout.prefix_lines)

    def remaining_tool_calls(self) -> int | None:
        limit = self.usage_limits.tool_calls_limit
        if limit is None:
            return None
        return max(0, limit - self.usage.tool_calls)

    def remaining_requests(self) -> int | None:
        limit = self.usage_limits.request_limit
        if limit is None:
            return None
        return max(0, limit - self.usage.requests)

    def remaining_input_tokens(self) -> int | None:
        limit = self.usage_limits.input_tokens_limit
        if limit is None:
            return None
        used = int(getattr(self.usage, "input_tokens", 0) or 0)
        return max(0, int(limit) - used)

    def remaining_output_tokens(self) -> int | None:
        limit = self.usage_limits.output_tokens_limit
        if limit is None:
            return None
        used = int(getattr(self.usage, "output_tokens", 0) or 0)
        return max(0, int(limit) - used)

    def remaining_total_tokens(self) -> int | None:
        limit = self.usage_limits.total_tokens_limit
        if limit is None:
            return None
        used = int(getattr(self.usage, "total_tokens", 0) or 0)
        return max(0, int(limit) - used)

    @staticmethod
    def _extract_last_response_usage(run_ctx: Any) -> tuple[int, int, int, int] | None:
        extracted = DocqAgentSession._extract_last_request_response(run_ctx)
        if extracted is None:
            return None
        response_index, _request_msg, _response_msg, req_input, req_output, req_total = extracted
        return response_index, req_input, req_output, req_total

    @staticmethod
    def _extract_last_request_response(
        run_ctx: Any,
    ) -> tuple[int, Any | None, Any, int, int, int] | None:
        messages = getattr(run_ctx, "messages", None)
        if not messages:
            return None
        return DocqAgentSession._extract_last_request_response_from_messages(messages)

    @staticmethod
    def _extract_last_request_response_from_messages(
        messages: list[Any],
    ) -> tuple[int, Any | None, Any, int, int, int] | None:
        # In streamed runs the tail item may be a non-response entry; scan
        # backwards to find the latest response with usage payload.
        for response_index in range(len(messages) - 1, -1, -1):
            response_msg = messages[response_index]
            usage = getattr(response_msg, "usage", None)
            if usage is None:
                continue
            has_values = getattr(usage, "has_values", None)
            if callable(has_values) and not has_values():
                continue
            req_input = int(getattr(usage, "input_tokens", 0) or 0)
            req_output = int(getattr(usage, "output_tokens", 0) or 0)
            req_total_raw = getattr(usage, "total_tokens", None)
            req_total = int(req_total_raw) if req_total_raw is not None else req_input + req_output
            request_msg = messages[response_index - 1] if response_index > 0 else None
            return response_index, request_msg, response_msg, req_input, req_output, req_total
        return None

    @staticmethod
    def _is_total_tokens_limit_error(exc: Exception) -> bool:
        return "total_tokens_limit" in str(exc)

    def _subagent_attempt_usage_limits(
        self,
        *,
        remaining_tool_calls: int | None,
        remaining_requests: int | None,
    ) -> UsageLimits:
        # Compression threshold is enforced from per-request context size
        # (`req_input_tokens`) in the subagent event handler.
        total_tokens_limit = getattr(self.usage_limits, "total_tokens_limit", None)
        return UsageLimits(
            request_limit=remaining_requests,
            tool_calls_limit=remaining_tool_calls,
            input_tokens_limit=getattr(self.usage_limits, "input_tokens_limit", None),
            output_tokens_limit=getattr(self.usage_limits, "output_tokens_limit", None),
            total_tokens_limit=total_tokens_limit,
        )

    def _get_subagent_compression_agent(self) -> Agent[Any, str]:
        if self._subagent_compression_agent is None:
            if self.subagent_model is None:
                raise RuntimeError("No subagent model configured for context compression.")
            self._subagent_compression_agent = Agent(
                model=self.subagent_model,
                output_type=str,
                system_prompt=SUBAGENT_COMPRESSION_SYSTEM_PROMPT,
                retries=1,
                name="docq-lemma-compressor",
            )
        return self._subagent_compression_agent

    @staticmethod
    def _resume_subagent_prompt(main_prompt: str, summary: str) -> str:
        return (
            f"{main_prompt.strip()}\n\n"
            "Continuation Summary:\n"
            f"{summary.strip()}\n\n"
            "Continue proving the same lemma from this summary."
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): DocqAgentSession._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DocqAgentSession._jsonable(v) for v in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return DocqAgentSession._jsonable(model_dump(mode="json"))
            except TypeError:
                return DocqAgentSession._jsonable(model_dump())
            except Exception:
                pass
        to_dict = getattr(value, "dict", None)
        if callable(to_dict):
            try:
                return DocqAgentSession._jsonable(to_dict())
            except Exception:
                pass
        return repr(value)

    @staticmethod
    def _history_excerpt(
        message_history: list[Any],
        *,
        max_items: int = 20,
        max_chars: int = 16_000,
    ) -> str:
        if not message_history:
            return "(none)"
        tail = list(message_history[-max_items:])
        base_index = max(0, len(message_history) - len(tail))
        lines: list[str] = []
        for i, msg in enumerate(tail, start=base_index):
            payload = DocqAgentSession._jsonable(msg)
            lines.append(f"[{i}] {json.dumps(payload, ensure_ascii=False)}")
        out = "\n".join(lines)
        if len(out) > max_chars:
            return out[:max_chars] + "\n... [truncated]"
        return out

    def _summarize_subagent_for_compression(
        self,
        *,
        deps: LemmaSubSession,
        main_prompt: str,
        message_history: list[Any],
        round_index: int,
        remaining_tool_calls: int | None,
        remaining_requests: int | None,
    ) -> str:
        latest_state = deps.branch.latest_state_index
        snapshot: dict[str, Any] = {
            "compression_round": round_index,
            "doc_id": deps.branch.doc_id,
            "latest_state_index": latest_state,
            "required_imports": [
                {"libname": libname, "source": source} for (libname, source) in deps.required_imports
            ],
        }
        try:
            goals = deps.branch.get_goals(latest_state).get("goals", [])
            snapshot["latest_goals_count"] = len(goals)
            preview_goals: list[str] = []
            for g in goals[:3]:
                preview = str(g)
                if len(preview) > 1200:
                    preview = preview[:1200] + " ..."
                preview_goals.append(preview)
            snapshot["latest_goals"] = preview_goals
        except Exception as exc:
            snapshot["goal_snapshot_error"] = str(exc)
        try:
            proof_lines = deps.branch.proof_script(latest_state).splitlines()
            snapshot["proof_script_tail"] = [ln for ln in proof_lines[-20:] if ln.strip()]
        except Exception as exc:
            snapshot["proof_script_tail_error"] = str(exc)

        history_excerpt = self._history_excerpt(message_history, max_items=22, max_chars=18_000)
        prompt = (
            "Summarize this failed/truncated lemma-proof run for immediate continuation.\n"
            "Write EXACTLY these sections with these headers:\n"
            "1) LEMMA OBJECTIVE\n"
            "2) CURRENT PROOF STATUS\n"
            "3) WORKING STEPS (confirmed tactics/imports)\n"
            "4) FAILING STEPS (exact error snippets)\n"
            "5) LOCAL GOAL (verbatim from snapshot)\n"
            "6) NEXT STEPS (ordered, concrete, short)\n"
            "Hard requirements:\n"
            "- Keep exact doc_id/state indexes, lemma names, and import atoms.\n"
            "- Do not claim completion unless goals are truly empty.\n"
            "- LOCAL GOAL must include at least one unresolved goal if present.\n"
            "- NEXT STEPS must begin with a re-anchor action (get_goals/list_states/read_workspace_source).\n"
            "- Target 350-700 words.\n\n"
            f"Original lemma-subagent prompt:\n{main_prompt}\n\n"
            f"Current snapshot (JSON):\n{json.dumps(snapshot, ensure_ascii=False)}\n\n"
            "Recent captured run messages (possibly truncated):\n"
            f"{history_excerpt}"
        )
        compressor = self._get_subagent_compression_agent()
        summary_limits = UsageLimits(
            request_limit=remaining_requests,
            tool_calls_limit=remaining_tool_calls,
            total_tokens_limit=None,
        )
        with capture_run_messages() as compression_messages:
            result = compressor.run_sync(
                prompt,
                message_history=None,
                usage=self.usage,
                usage_limits=summary_limits,
            )
        extracted = self._extract_last_request_response_from_messages(compression_messages)
        if extracted is not None:
            (
                response_index,
                request_message,
                response_message,
                req_input_tokens,
                req_output_tokens,
                req_total_tokens,
            ) = extracted
            self._emit_trace_request(
                "subagent-compression",
                {
                    "doc_id": deps.branch.doc_id,
                    "compression_round": round_index,
                    "response_index": response_index,
                    "req_input_tokens": req_input_tokens,
                    "req_output_tokens": req_output_tokens,
                    "req_total_tokens": req_total_tokens,
                    "context_message_count": response_index,
                    "context_messages": list(compression_messages)[:response_index],
                    "request_message": request_message,
                    "response_message": response_message,
                },
            )
        else:
            self._emit_trace_request(
                "subagent-compression",
                {
                    "doc_id": deps.branch.doc_id,
                    "compression_round": round_index,
                    "request_prompt": prompt,
                    "response_text": str(result.output),
                },
            )
        return str(result.output).strip()

    def run_lemma_subagent(
        self,
        *,
        sub_branch: BranchSession,
        prompt: str,
    ) -> dict[str, Any]:
        remaining_tool_calls = self.remaining_tool_calls()
        remaining_requests = self.remaining_requests()
        remaining_input_tokens = self.remaining_input_tokens()
        remaining_output_tokens = self.remaining_output_tokens()
        remaining_total_tokens = self.remaining_total_tokens()
        if remaining_tool_calls is not None and remaining_tool_calls <= 0:
            return {"ok": False, "error": "Tool-call budget exhausted before sub-agent run."}
        if remaining_requests is not None and remaining_requests <= 0:
            return {"ok": False, "error": "Request budget exhausted before sub-agent run."}
        self._log(
            f"subagent start(doc_id={sub_branch.doc_id}, remaining_tool_calls="
            f"{remaining_tool_calls if remaining_tool_calls is not None else 'unbounded'}, "
            f"remaining_requests={remaining_requests if remaining_requests is not None else 'unbounded'}, "
            f"remaining_input_tokens={remaining_input_tokens if remaining_input_tokens is not None else 'unbounded'}, "
            f"remaining_output_tokens={remaining_output_tokens if remaining_output_tokens is not None else 'unbounded'}, "
            f"remaining_total_tokens={remaining_total_tokens if remaining_total_tokens is not None else 'unbounded'}, "
            f"used_input_tokens={int(getattr(self.usage, 'input_tokens', 0) or 0)}, "
            f"used_output_tokens={int(getattr(self.usage, 'output_tokens', 0) or 0)}, "
            f"used_total_tokens={int(getattr(self.usage, 'total_tokens', 0) or 0)}, "
            f"compression_threshold_tokens={self.subagent_threshold_compression if self.subagent_threshold_compression > 0 else 0})"
        )
        sub_client = self._try_make_fresh_client()
        owns_sub_client = sub_client is not None
        if sub_client is None:
            # Fallback: keep existing session client (still protected by lock proxy).
            sub_client = self.client
        try:
            isolated_branch = _rebuild_branch_on_client(sub_branch, client=sub_client)
        except Exception as exc:
            if owns_sub_client:
                self._close_client_quietly(sub_client)
            return {"ok": False, "error": f"Failed to initialize isolated subagent workspace: {exc}"}

        deps = LemmaSubSession(
            branch=isolated_branch,
            client=sub_client,
            toc_explorer=self.toc_explorer,
            semantic_search=self.semantic_search,
            logger=self.logger,
        )
        subagent = build_docq_subagent(
            model=self.subagent_model,
            retries=self.subagent_retries,
            include_semantic_tool=self.include_semantic_tool,
        )
        current_prompt = prompt
        current_history: list[Any] | None = None
        compression_rounds = 0
        sub_model_call_index = 0
        sub_last_logged_requests = int(getattr(self.usage, "requests", 0) or 0)
        sub_last_logged_input_tokens = int(getattr(self.usage, "input_tokens", 0) or 0)
        sub_last_logged_output_tokens = int(getattr(self.usage, "output_tokens", 0) or 0)
        sub_last_logged_response_index = -1
        sub_runctx_message_cursor = 0
        latest_subagent_messages: list[Any] = []

        def _flush_subagent_runctx_messages(run_ctx: Any) -> None:
            nonlocal sub_runctx_message_cursor, latest_subagent_messages
            cb = self.trace_message_callback
            if cb is None:
                return
            messages = getattr(run_ctx, "messages", None)
            if not messages:
                return
            latest_subagent_messages = list(messages)
            while sub_runctx_message_cursor < len(messages):
                try:
                    cb("subagent", messages[sub_runctx_message_cursor])
                except Exception:
                    return
                sub_runctx_message_cursor += 1

        async def _sub_event_stream_handler(_run_ctx: Any, stream: Any) -> None:
            nonlocal sub_model_call_index, sub_last_logged_requests
            nonlocal sub_last_logged_input_tokens, sub_last_logged_output_tokens
            nonlocal sub_last_logged_response_index
            async for _event in stream:
                self._emit_trace_event("subagent", _event)
                _flush_subagent_runctx_messages(_run_ctx)
            _flush_subagent_runctx_messages(_run_ctx)
            usage = getattr(_run_ctx, "usage", None)
            requests_now = int(getattr(usage, "requests", 0) or 0)
            input_tokens_now = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens_now = int(getattr(usage, "output_tokens", 0) or 0)
            per_request = self._extract_last_request_response(_run_ctx)
            if per_request is not None:
                (
                    response_index,
                    request_message,
                    response_message,
                    req_input,
                    req_output,
                    req_total,
                ) = per_request
                if response_index > sub_last_logged_response_index:
                    sub_model_call_index += 1
                    self._log(
                        f"subagent model call(doc_id={sub_branch.doc_id}, "
                        f"index={sub_model_call_index}, "
                        f"run_step={getattr(_run_ctx, 'run_step', None)}, "
                        f"requests={requests_now}, "
                        f"req_input_tokens={req_input}, "
                        f"req_output_tokens={req_output}, "
                        f"req_total_tokens={req_total}, "
                        f"input_tokens={input_tokens_now}, "
                        f"output_tokens={output_tokens_now})"
                    )
                    self._emit_trace_request(
                        "subagent",
                        {
                            "doc_id": sub_branch.doc_id,
                            "index": sub_model_call_index,
                            "run_step": getattr(_run_ctx, "run_step", None),
                            "requests": requests_now,
                            "response_index": response_index,
                            "req_input_tokens": req_input,
                            "req_output_tokens": req_output,
                            "req_total_tokens": req_total,
                            "input_tokens": input_tokens_now,
                            "output_tokens": output_tokens_now,
                            "context_message_count": response_index,
                            "context_messages": list(getattr(_run_ctx, "messages", []) or [])[:response_index],
                            "request_message": request_message,
                            "response_message": response_message,
                        },
                    )
                    sub_last_logged_response_index = response_index
                    sub_last_logged_requests = requests_now
                    sub_last_logged_input_tokens = input_tokens_now
                    sub_last_logged_output_tokens = output_tokens_now
                    if (
                        self.subagent_threshold_compression > 0
                        and req_input > self.subagent_threshold_compression
                    ):
                        raise _CompressionRequested(
                            reason="req_input_tokens_threshold",
                            req_input_tokens=req_input,
                        )
                    return
            if requests_now > sub_last_logged_requests:
                delta_requests = requests_now - sub_last_logged_requests
                delta_input_tokens = input_tokens_now - sub_last_logged_input_tokens
                delta_output_tokens = output_tokens_now - sub_last_logged_output_tokens
                if delta_requests == 1 and delta_input_tokens == 0 and delta_output_tokens == 0:
                    # Ignore noisy intermediate stream callbacks; a proper
                    # per-request log usually follows once usage is attached.
                    return
                self._log(
                    f"subagent model call coalesced(doc_id={sub_branch.doc_id}, "
                    f"run_step={getattr(_run_ctx, 'run_step', None)}, "
                    f"requests={requests_now}, "
                    f"delta_requests={delta_requests}, "
                    f"delta_input_tokens={delta_input_tokens}, "
                    f"delta_output_tokens={delta_output_tokens})"
                )
                sub_last_logged_requests = requests_now
                sub_last_logged_input_tokens = input_tokens_now
                sub_last_logged_output_tokens = output_tokens_now
        try:
            while True:
                limits = self._subagent_attempt_usage_limits(
                    remaining_tool_calls=remaining_tool_calls,
                    remaining_requests=remaining_requests,
                )
                with capture_run_messages() as run_messages:
                    sub_runctx_message_cursor = 0
                    latest_subagent_messages = []
                    try:
                        _ = subagent.run_sync(
                            current_prompt,
                            deps=deps,
                            message_history=current_history,
                            usage=self.usage,
                            usage_limits=limits,
                            event_stream_handler=_sub_event_stream_handler,
                        )
                        break
                    except (UsageLimitExceeded, _CompressionRequested) as exc:
                        compression_reason = "limit"
                        compression_req_input: int | None = None
                        if isinstance(exc, _CompressionRequested):
                            if self.subagent_threshold_compression <= 0:
                                self._log(f"subagent failure(doc_id={sub_branch.doc_id}): {exc}")
                                return {"ok": False, "error": f"Sub-agent failure: {exc}"}
                            compression_reason = exc.reason
                            compression_req_input = exc.req_input_tokens
                        else:
                            if (
                                self.subagent_threshold_compression <= 0
                                or not self._is_total_tokens_limit_error(exc)
                            ):
                                self._log(f"subagent budget exceeded(doc_id={sub_branch.doc_id}): {exc}")
                                return {"ok": False, "error": f"Sub-agent budget exceeded: {exc}"}
                            compression_reason = "total_tokens_limit"
                        compression_rounds += 1
                        if compression_rounds > self.subagent_max_compression_rounds:
                            err = (
                                "Sub-agent exceeded maximum compression rounds "
                                f"({self.subagent_max_compression_rounds}). Last error: {exc}"
                            )
                            self._log(f"subagent compression failed(doc_id={sub_branch.doc_id}): {err}")
                            return {"ok": False, "error": err}
                        start_input = int(getattr(self.usage, "input_tokens", 0) or 0)
                        start_total = int(getattr(self.usage, "total_tokens", 0) or 0)
                        self._log(
                            f"subagent compression triggered(doc_id={sub_branch.doc_id}, "
                            f"round={compression_rounds}, threshold={self.subagent_threshold_compression}, "
                            f"reason={compression_reason}, req_input_tokens={compression_req_input}, "
                            f"input_tokens={start_input}, total_tokens={start_total})"
                        )
                        try:
                            summary = self._summarize_subagent_for_compression(
                                deps=deps,
                                main_prompt=prompt,
                                message_history=latest_subagent_messages or run_messages,
                                round_index=compression_rounds,
                                remaining_tool_calls=remaining_tool_calls,
                                remaining_requests=remaining_requests,
                            )
                        except Exception as summary_exc:
                            err = f"Sub-agent compression summarization failed: {summary_exc}"
                            self._log(f"subagent compression failed(doc_id={sub_branch.doc_id}): {err}")
                            return {"ok": False, "error": err}
                        end_input = int(getattr(self.usage, "input_tokens", 0) or 0)
                        end_total = int(getattr(self.usage, "total_tokens", 0) or 0)
                        self._log(
                            f"subagent compression summary done(doc_id={sub_branch.doc_id}, "
                            f"round={compression_rounds}, input_tokens={end_input}, total_tokens={end_total}, "
                            f"delta_input_tokens={end_input - start_input}, "
                            f"delta_total_tokens={end_total - start_total})"
                        )
                        current_prompt = self._resume_subagent_prompt(prompt, summary)
                        current_history = None
                        sub_last_logged_requests = int(getattr(self.usage, "requests", 0) or 0)
                        sub_last_logged_input_tokens = int(getattr(self.usage, "input_tokens", 0) or 0)
                        sub_last_logged_output_tokens = int(getattr(self.usage, "output_tokens", 0) or 0)
                        sub_last_logged_response_index = -1
                        self._log(
                            f"subagent context compressed and resumed with prompt+summary "
                            f"(doc_id={sub_branch.doc_id}, round={compression_rounds})"
                        )
                        continue
                    except Exception as exc:
                        self._log(f"subagent failure(doc_id={sub_branch.doc_id}): {exc}")
                        return {"ok": False, "error": f"Sub-agent failure: {exc}"}
        finally:
            if owns_sub_client:
                self._close_client_quietly(sub_client)

        if deps.abort_reason:
            self._log(f"subagent abort(doc_id={sub_branch.doc_id}): {deps.abort_reason}")
            return {"ok": False, "error": deps.abort_reason, "aborted": True}

        final_branch = deps.branch
        latest = final_branch.latest_state_index
        goals = final_branch.get_goals(latest).get("goals", [])
        if goals:
            self._log(f"subagent unfinished(doc_id={sub_branch.doc_id}, remaining_goals={len(goals)})")
            return {"ok": False, "error": "Sub-agent did not finish the lemma proof."}
        required_imports = [
            {"libname": libname, "source": source} for (libname, source) in deps.required_imports
        ]
        self._log(f"subagent done(doc_id={sub_branch.doc_id}, state_index={latest})")
        return {
            "ok": True,
            "proof_script": final_branch.proof_script(latest),
            "state_index": latest,
            "required_imports": required_imports,
        }

    def add_intermediate_lemma(
        self,
        *,
        lemma_type: str,
        prompt: str | None = None,
        subagent_message: str | None = None,
        lemma_name: str | None = None,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        has_subagent_message = bool((subagent_message or prompt or "").strip())
        self._log(
            f"add_intermediate_lemma called(lemma_name={lemma_name!r}, doc_id={doc_id}, "
            f"lemma_type={lemma_type!r}, has_subagent_message={has_subagent_message})"
        )
        prep = self.prepare_intermediate_lemma(
            lemma_type=lemma_type,
            lemma_name=lemma_name,
            doc_id=doc_id,
            subagent_message=subagent_message,
        )
        if not prep.get("ok", False):
            self._log(f"add_intermediate_lemma prepare failed: {prep.get('error')}")
            return prep
        proved = self.prove_intermediate_lemma(
            lemma_name=str(prep["lemma_name"]),
            prompt=prompt,
            subagent_message=subagent_message,
        )
        if not proved.get("ok", False):
            proved["prepared"] = True
            self._log(f"add_intermediate_lemma prove failed: {proved.get('error')}")
        else:
            self._log(f"add_intermediate_lemma done: {proved.get('lemma_name')}")
        return proved

    def prepare_intermediate_lemma(
        self,
        *,
        lemma_type: str,
        lemma_name: str | None = None,
        doc_id: int | None = None,
        subagent_message: str | None = None,
    ) -> dict[str, Any]:
        clean_subagent_message = (subagent_message or "").strip() or None
        self._log(
            f"prepare_intermediate_lemma called(lemma_name={lemma_name!r}, doc_id={doc_id}, "
            f"lemma_type={lemma_type!r}, has_subagent_message={bool(clean_subagent_message)})"
        )
        name = (lemma_name or "").strip()
        if not name:
            name = self.doc_manager.next_lemma_name()
        if not _IDENT_RE.match(name):
            return {"ok": False, "phase": "prepare", "lemma_name": name, "error": f"Invalid lemma_name={name!r}."}
        if name in self.pending_lemmas:
            return {"ok": False, "phase": "prepare", "lemma_name": name, "error": "Lemma is already pending."}

        try:
            base_doc_id, sub_branch = self.doc_manager.create_lemma_subsession(
                lemma_name=name,
                lemma_type=lemma_type,
                doc_id=doc_id,
            )
        except Exception as exc:
            return {"ok": False, "phase": "prepare", "lemma_name": name, "error": f"Lemma declaration/type-check failed: {exc}"}

        self.pending_lemmas[name] = PendingLemma(
            base_doc_id=base_doc_id,
            lemma_name=name,
            lemma_type=lemma_type,
            sub_branch=sub_branch,
            subagent_message=clean_subagent_message,
        )
        self._log(
            f"prepare_intermediate_lemma ok(lemma_name={name}, base_doc_id={base_doc_id}, "
            f"sub_doc_id={sub_branch.doc_id})"
        )
        return {
            "ok": True,
            "phase": "prepare",
            "lemma_name": name,
            "base_doc_id": base_doc_id,
            "sub_doc_id": sub_branch.doc_id,
            "sub_states": sub_branch.list_states(),
            "has_subagent_message": bool(clean_subagent_message),
        }

    def prove_intermediate_lemma(
        self,
        *,
        lemma_name: str,
        prompt: str | None = None,
        subagent_message: str | None = None,
    ) -> dict[str, Any]:
        clean_prompt = (prompt or "").strip() or None
        clean_subagent_message = (subagent_message or "").strip() or None
        self._log(
            f"prove_intermediate_lemma called(lemma_name={lemma_name!r}, "
            f"has_prompt={bool(clean_prompt)}, has_subagent_message={bool(clean_subagent_message)})"
        )
        pending = self.pending_lemmas.get(lemma_name)
        if pending is None:
            return {
                "ok": False,
                "phase": "prove",
                "lemma_name": lemma_name,
                "error": f"No pending lemma named `{lemma_name}`.",
                "pending_lemmas": sorted(self.pending_lemmas.keys()),
            }

        handoff = clean_subagent_message or clean_prompt or pending.subagent_message
        sub_prompt = (
            f"Prove the intermediate lemma `{lemma_name}`. "
            "Use run_tac/list_states/get_goals. Call abort if impossible."
        )
        if handoff:
            sub_prompt += f"\n\nMain-agent handoff:\n{handoff}"
        sub_result = self.run_lemma_subagent(sub_branch=pending.sub_branch, prompt=sub_prompt)
        if not sub_result.get("ok"):
            self._log(
                f"prove_intermediate_lemma subagent failed(lemma_name={lemma_name}): "
                f"{sub_result.get('error')}"
            )
            return {
                "ok": False,
                "phase": "prove",
                "lemma_name": lemma_name,
                "error": sub_result.get("error", "Sub-agent failed."),
                "aborted": bool(sub_result.get("aborted", False)),
                "pending": True,
            }

        proof_script = str(sub_result.get("proof_script", "")).strip()
        required_imports = self._normalize_required_imports(sub_result.get("required_imports"))
        base_doc_id = pending.base_doc_id
        applied_imports: list[dict[str, Any]] = []
        for (libname, source) in required_imports:
            if self._doc_has_import(doc_id=base_doc_id, libname=libname, source=source):
                continue
            try:
                added = self.doc_manager.add_import(libname=libname, source=source, doc_id=base_doc_id)
            except Exception as exc:
                return {
                    "ok": False,
                    "phase": "prove",
                    "lemma_name": lemma_name,
                    "error": f"Failed to apply required import `{libname}.{source}`: {exc}",
                    "pending": True,
                }
            base_doc_id = int(added["doc_id"])
            applied_imports.append(
                {
                    "libname": libname,
                    "source": source,
                    "doc_id": base_doc_id,
                }
            )
        try:
            reg = self.doc_manager.register_proved_lemma(
                base_doc_id=base_doc_id,
                lemma_name=pending.lemma_name,
                lemma_type=pending.lemma_type,
                proof_script=proof_script,
            )
        except Exception as exc:
            return {
                "ok": False,
                "phase": "prove",
                "lemma_name": lemma_name,
                "error": f"Lemma registration failed: {exc}",
                "pending": True,
            }

        self.pending_lemmas.pop(lemma_name, None)
        reg["phase"] = "prove"
        reg["lemma_name"] = lemma_name
        reg["proof_script"] = proof_script
        if applied_imports:
            reg["applied_imports"] = applied_imports
        self._log(f"prove_intermediate_lemma ok(lemma_name={lemma_name}, doc_id={reg.get('doc_id')})")
        return reg

    def drop_pending_intermediate_lemma(self, *, lemma_name: str) -> dict[str, Any]:
        self._log(f"drop_pending_intermediate_lemma called(lemma_name={lemma_name!r})")
        removed = self.pending_lemmas.pop(lemma_name, None)
        if removed is None:
            return {
                "ok": False,
                "phase": "drop",
                "lemma_name": lemma_name,
                "error": f"No pending lemma named `{lemma_name}`.",
                "pending_lemmas": sorted(self.pending_lemmas.keys()),
            }
        return {
            "ok": True,
            "phase": "drop",
            "lemma_name": lemma_name,
            "remaining_pending_lemmas": sorted(self.pending_lemmas.keys()),
        }

    def list_pending_intermediate_lemmas(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name in sorted(self.pending_lemmas.keys()):
            pending = self.pending_lemmas[name]
            out.append(
                {
                    "lemma_name": name,
                    "base_doc_id": pending.base_doc_id,
                    "sub_doc_id": pending.sub_branch.doc_id,
                    "sub_state_count": len(pending.sub_branch.nodes),
                    "has_subagent_message": bool(pending.subagent_message),
                }
            )
        return out

    def completion_status(self, *, doc_id: int | None = None) -> dict[str, Any]:
        return self.doc_manager.completion_status(doc_id=doc_id)


def build_docq_agent(
    model: Any = None,
    *,
    retries: int = 2,
    include_semantic_tool: bool = True,
) -> Agent[DocqAgentSession, str]:
    agent = Agent(
        model=model,
        deps_type=DocqAgentSession,
        output_type=str,
        name="docq-agent",
        system_prompt=DOCQ_SYSTEM_PROMPT,
        retries=retries,
    )

    @agent.tool
    def explore_toc(ctx: RunContext[DocqAgentSession], path: list[str] | None = None) -> dict[str, Any]:
        req_path = path or []
        ctx.deps._log(f"explore_toc(path={req_path})")
        out = ctx.deps.toc_explorer.explore(req_path)
        ctx.deps._log(f"explore_toc -> ok={out.get('ok', False)}")
        return out

    if include_semantic_tool:

        @agent.tool
        def semantic_doc_search(
            ctx: RunContext[DocqAgentSession],
            query: str,
            k: int = 5,
        ) -> dict[str, Any]:
            if ctx.deps.semantic_search is None:
                raise ModelRetry("Semantic search is not configured for this session.")
            if k < 1:
                raise ModelRetry("k must be >= 1")
            results = ctx.deps.semantic_search.search(query=query, k=k)
            return {"query": query, "k": k, "results": results}

    @agent.tool
    def read_source_file(
        ctx: RunContext[DocqAgentSession],
        path: str | list[str],
        line: int | None = None,
        before: int = 20,
        after: int = 20,
    ) -> dict[str, Any]:
        normalized_path = _normalize_read_source_path(path)
        if normalized_path is None:
            error = "Invalid empty path. Provide a TOC-relative path like `mathcomp/fingroup/perm.v`."
            ctx.deps._log(f"read_source_file -> failed: {error}")
            return _read_source_error_payload(requested_path=str(path), error=error)
        ctx.deps._log(
            f"read_source_file(path={normalized_path!r}, line={line}, before={before}, after={after})"
        )
        workspace_doc_id = ctx.deps.doc_manager.doc_id_for_source_path(normalized_path)
        if workspace_doc_id is not None:
            try:
                out = ctx.deps.doc_manager.read_source(
                    line=line,
                    before=before,
                    after=after,
                    doc_id=workspace_doc_id,
                )
                out["requested_path"] = normalized_path
                out["resolved_path"] = str(ctx.deps.doc_manager.sessions[workspace_doc_id].source_path)
                out["ok"] = True
                out["source_kind"] = "workspace_doc"
                ctx.deps._log(
                    f"read_source_file -> ok (workspace doc_id={workspace_doc_id}, "
                    f"resolved_path={out.get('resolved_path')!r})"
                )
                return out
            except Exception as exc:
                error = str(exc)
                ctx.deps._log(f"read_source_file -> failed (workspace path): {error}")
                return _read_source_error_payload(requested_path=normalized_path, error=error)
        try:
            out = read_source_via_client(
                ctx.deps.client,
                normalized_path,
                line=line,
                before=before,
                after=after,
            )
            out["ok"] = True
            ctx.deps._log(f"read_source_file -> ok (resolved_path={out.get('resolved_path')!r})")
            return out
        except Exception as exc:
            error = str(exc)
            ctx.deps._log(f"read_source_file -> failed: {error}")
            return _read_source_error_payload(requested_path=normalized_path, error=error)

    @agent.tool
    def list_docs(ctx: RunContext[DocqAgentSession]) -> list[dict[str, Any]]:
        return ctx.deps.doc_manager.list_docs()

    @agent.tool
    def checkout_doc(ctx: RunContext[DocqAgentSession], doc_id: int) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.checkout_doc(doc_id=doc_id)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def show_doc(ctx: RunContext[DocqAgentSession], doc_id: int) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.show_doc(doc_id=doc_id)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def show_workspace(ctx: RunContext[DocqAgentSession], doc_id: int | None = None) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.show_workspace(doc_id=doc_id)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def completion_status(ctx: RunContext[DocqAgentSession], doc_id: int | None = None) -> dict[str, Any]:
        try:
            return ctx.deps.completion_status(doc_id=doc_id)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def list_states(ctx: RunContext[DocqAgentSession], doc_id: int | None = None) -> list[dict[str, Any]]:
        try:
            return ctx.deps.doc_manager.list_states_verbose(doc_id=doc_id)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def get_goals(
        ctx: RunContext[DocqAgentSession],
        state_index: int = 0,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.get_goals(state_index=state_index, doc_id=doc_id)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def run_tac(
        ctx: RunContext[DocqAgentSession],
        state_index: int = 0,
        tactic: str = "idtac.",
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.run_tac(state_index=state_index, tactic=tactic, doc_id=doc_id)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def read_workspace_source(
        ctx: RunContext[DocqAgentSession],
        line: int | None = None,
        before: int = 20,
        after: int = 20,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.read_source(line=line, before=before, after=after, doc_id=doc_id)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def add_import(
        ctx: RunContext[DocqAgentSession],
        libname: str,
        source: str,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        """Add import atoms only. Example: add_import(libname='mathcomp.fingroup', source='perm')."""
        try:
            return ctx.deps.doc_manager.add_import(libname=libname, source=source, doc_id=doc_id)
        except ValueError as exc:
            error = str(exc)
            ctx.deps._log(f"add_import failed: {error}")
            return {
                "ok": False,
                "error": error,
                "hint": "Call add_import with separate arguments, e.g. libname='mathcomp.fingroup', source='perm'.",
            }

    @agent.tool
    def remove_import(
        ctx: RunContext[DocqAgentSession],
        libname: str,
        source: str,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        """Remove import atoms only. Example: remove_import(libname='mathcomp.fingroup', source='perm')."""
        try:
            return ctx.deps.doc_manager.remove_import(libname=libname, source=source, doc_id=doc_id)
        except ValueError as exc:
            error = str(exc)
            ctx.deps._log(f"remove_import failed: {error}")
            return {
                "ok": False,
                "error": error,
                "hint": "Call remove_import with separate arguments, e.g. libname='mathcomp.fingroup', source='perm'.",
            }

    @agent.tool
    def prepare_intermediate_lemma(
        ctx: RunContext[DocqAgentSession],
        lemma_type: str,
        lemma_name: str | None = None,
        doc_id: int | None = None,
        subagent_message: str | None = None,
    ) -> dict[str, Any]:
        """Prepare helper lemma and optionally store `subagent_message` guidance for the proving subagent."""
        return ctx.deps.prepare_intermediate_lemma(
            lemma_type=lemma_type,
            lemma_name=lemma_name,
            doc_id=doc_id,
            subagent_message=subagent_message,
        )

    @agent.tool
    def prove_intermediate_lemma(
        ctx: RunContext[DocqAgentSession],
        lemma_name: str,
        prompt: str | None = None,
        subagent_message: str | None = None,
    ) -> dict[str, Any]:
        """Prove prepared helper lemma. Use `subagent_message` to pass import/goal/strategy hints to the subagent."""
        return ctx.deps.prove_intermediate_lemma(
            lemma_name=lemma_name,
            prompt=prompt,
            subagent_message=subagent_message,
        )

    @agent.tool
    def drop_pending_intermediate_lemma(
        ctx: RunContext[DocqAgentSession],
        lemma_name: str,
    ) -> dict[str, Any]:
        return ctx.deps.drop_pending_intermediate_lemma(lemma_name=lemma_name)

    @agent.tool
    def list_pending_intermediate_lemmas(ctx: RunContext[DocqAgentSession]) -> list[dict[str, Any]]:
        return ctx.deps.list_pending_intermediate_lemmas()

    @agent.tool
    def add_intermediate_lemma(
        ctx: RunContext[DocqAgentSession],
        lemma_type: str,
        lemma_name: str | None = None,
        prompt: str | None = None,
        subagent_message: str | None = None,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        """Prepare+prove helper lemma. `subagent_message` is forwarded as a handoff to the proving subagent."""
        return ctx.deps.add_intermediate_lemma(
            lemma_type=lemma_type,
            lemma_name=lemma_name,
            prompt=prompt,
            subagent_message=subagent_message,
            doc_id=doc_id,
        )

    @agent.tool
    def remove_intermediate_lemma(
        ctx: RunContext[DocqAgentSession],
        lemma_name: str,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.remove_intermediate_lemma(lemma_name=lemma_name, doc_id=doc_id)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.output_validator
    def ensure_no_pending_lemmas(ctx: RunContext[DocqAgentSession], output: str) -> str:
        text = output.strip()
        # TestModel tool-trace outputs are JSON blobs; allow those so deterministic tests
        # can inspect tool payloads without forcing another model round-trip.
        if text.startswith("{") and text.endswith("}"):
            return output
        pending = ctx.deps.list_pending_intermediate_lemmas()
        if pending:
            names = ", ".join(item.get("lemma_name", "?") for item in pending)
            raise ModelRetry(
                "You cannot finish while intermediate lemmas are pending. "
                f"Pending: {names}. "
                "Call `prove_intermediate_lemma` or `drop_pending_intermediate_lemma`."
            )
        try:
            status = ctx.deps.completion_status()
        except Exception as exc:
            raise ModelRetry(
                "Could not read completion status from the proof workspace. "
                f"Tool/backend error: {exc}. "
                "Call `show_workspace` and `get_goals` on the active doc/state, then continue proving."
            ) from exc
        if not bool(status.get("latest_proof_finished", False)):
            raise ModelRetry(
                "You cannot finish yet: the current head proof is still open. "
                f"doc_id={status.get('doc_id')} latest_state_index={status.get('latest_state_index')} "
                f"latest_goals_count={status.get('latest_goals_count')}. "
                "Call `completion_status` then `get_goals` on the latest state and continue with `run_tac`. "
                "If needed, inspect/branch with `list_states`, `show_workspace`, `list_docs`, `checkout_doc`."
            )
        ok, qed_error = ctx.deps.validate_final_qed(
            doc_id=int(status.get("doc_id", ctx.deps.doc_manager.head_doc_id)),
            state_index=int(status.get("latest_state_index", 0)),
        )
        if not ok:
            raise ModelRetry(
                "You cannot finish yet: explicit final `Qed.` replay check failed. "
                f"Error: {qed_error}. "
                "Re-anchor with `completion_status`/`get_goals`, then continue `run_tac` until a clean finish."
            )
        return output

    return agent
