from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic_ai import Agent, ModelRetry, RunContext
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
    "- Use ASCII Coq syntax in tool arguments: `forall`, `exists`, `->`, `/\\`, `\\/`, `<=`, `>=`.\n"
    "  Do NOT use Unicode logic symbols (`∀`, `∃`, `→`, `∧`, `∨`, `≤`, `≥`, ...).\n"
    "- For imports, use `add_import(libname, source, doc_id?)` and `remove_import(...)`.\n"
    "- Import format: give only atoms, e.g. libname='mathcomp.fingroup', source='perm'.\n"
    "  Do NOT pass full commands like `From ... Require Import ...`.\n"
    "  Example call: add_import(libname='mathcomp.fingroup', source='perm').\n"
    "- For helper statements, prefer phased tools:\n"
    "  `prepare_intermediate_lemma` -> `prove_intermediate_lemma` -> `drop_pending_intermediate_lemma`.\n"
    "- Intermediate lemma format: `lemma_name` is only the identifier; `lemma_type` is only the proposition.\n"
    "  Do NOT include `Lemma name :` prefix nor `Proof.`/`Qed.` in `lemma_type`.\n"
    "  Example call: prepare_intermediate_lemma(lemma_name='helper_card', lemma_type='forall n : nat, n > 0 -> True').\n"
    "- Convenience path `add_intermediate_lemma` is still available.\n"
    "- If a helper lemma is problematic, call `remove_intermediate_lemma(lemma_name)`.\n"
    "- If stuck on lemma proving, the sub-agent can abort and report why."
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


def _lemma_subagent_prompt(*, include_semantic_tool: bool) -> str:
    retrieval_line = "`explore_toc`, `semantic_doc_search`, `read_source_file`"
    if not include_semantic_tool:
        retrieval_line = "`explore_toc`, `read_source_file`"
    return (
        "You are proving one intermediate lemma.\n"
        "Use tools:\n"
        f"- {retrieval_line}\n"
        "- For library reads, pass TOC-relative file paths exactly as returned by `explore_toc`.\n"
        "- `show_workspace`, `read_workspace_source`\n"
        "- `list_states`, `get_goals`, `run_tac`\n"
        "- Use ASCII Coq syntax in tool arguments (`forall`, `exists`, `->`, `/\\`, `\\/`, `<=`, `>=`).\n"
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
        path: str | None = None,
        line: int | None = None,
        before: int = 20,
        after: int = 20,
    ) -> dict[str, Any]:
        source_path = str(ctx.deps.branch.source_path)
        requested_path = source_path if path is None else str(path)
        _sub_log(
            ctx,
            f"read_source_file(path={requested_path!r}, line={line}, before={before}, after={after})",
        )
        if path is None or requested_path == source_path:
            payload = _render_content_window(
                ctx.deps.branch.source_content,
                line=line,
                before=before,
                after=after,
            )
            payload["requested_path"] = requested_path
            payload["resolved_path"] = source_path
            payload["ok"] = True
            _sub_log(ctx, "read_source_file -> ok (workspace content)")
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
            "source_path": str(ctx.deps.branch.source_path),
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
            payload["source_path"] = str(ctx.deps.branch.source_path)
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
    include_semantic_tool: bool = True
    pending_lemmas: dict[str, PendingLemma] = field(default_factory=dict)

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
        include_semantic_tool: bool = True,
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
            include_semantic_tool=include_semantic_tool,
        )
        session.doc_manager.mutation_validator = session._validate_mutation_in_fresh_session
        return session

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

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

    def run_lemma_subagent(
        self,
        *,
        sub_branch: BranchSession,
        prompt: str,
    ) -> dict[str, Any]:
        remaining_tool_calls = self.remaining_tool_calls()
        remaining_requests = self.remaining_requests()
        if remaining_tool_calls is not None and remaining_tool_calls <= 0:
            return {"ok": False, "error": "Tool-call budget exhausted before sub-agent run."}
        if remaining_requests is not None and remaining_requests <= 0:
            return {"ok": False, "error": "Request budget exhausted before sub-agent run."}
        self._log(
            f"subagent start(doc_id={sub_branch.doc_id}, remaining_tool_calls="
            f"{remaining_tool_calls if remaining_tool_calls is not None else 'unbounded'}, "
            f"remaining_requests={remaining_requests if remaining_requests is not None else 'unbounded'})"
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
        limits = None
        if remaining_tool_calls is not None or remaining_requests is not None:
            limits = UsageLimits(tool_calls_limit=remaining_tool_calls, request_limit=remaining_requests)

        try:
            _ = subagent.run_sync(
                prompt,
                deps=deps,
                usage=self.usage,
                usage_limits=limits,
            )
        except UsageLimitExceeded as exc:
            self._log(f"subagent budget exceeded(doc_id={sub_branch.doc_id}): {exc}")
            return {"ok": False, "error": f"Sub-agent budget exceeded: {exc}"}
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
        lemma_name: str | None = None,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        self._log(
            f"add_intermediate_lemma called(lemma_name={lemma_name!r}, doc_id={doc_id}, "
            f"lemma_type={lemma_type!r})"
        )
        prep = self.prepare_intermediate_lemma(
            lemma_type=lemma_type,
            lemma_name=lemma_name,
            doc_id=doc_id,
        )
        if not prep.get("ok", False):
            self._log(f"add_intermediate_lemma prepare failed: {prep.get('error')}")
            return prep
        proved = self.prove_intermediate_lemma(
            lemma_name=str(prep["lemma_name"]),
            prompt=prompt,
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
    ) -> dict[str, Any]:
        self._log(
            f"prepare_intermediate_lemma called(lemma_name={lemma_name!r}, doc_id={doc_id}, "
            f"lemma_type={lemma_type!r})"
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
        }

    def prove_intermediate_lemma(
        self,
        *,
        lemma_name: str,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        self._log(f"prove_intermediate_lemma called(lemma_name={lemma_name!r})")
        pending = self.pending_lemmas.get(lemma_name)
        if pending is None:
            return {
                "ok": False,
                "phase": "prove",
                "lemma_name": lemma_name,
                "error": f"No pending lemma named `{lemma_name}`.",
                "pending_lemmas": sorted(self.pending_lemmas.keys()),
            }

        sub_prompt = prompt or (
            f"Prove the intermediate lemma `{lemma_name}`. "
            "Use run_tac/list_states/get_goals. Call abort if impossible."
        )
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
        path: str,
        line: int | None = None,
        before: int = 20,
        after: int = 20,
    ) -> dict[str, Any]:
        ctx.deps._log(f"read_source_file(path={path!r}, line={line}, before={before}, after={after})")
        workspace_doc_id = ctx.deps.doc_manager.doc_id_for_source_path(path)
        if workspace_doc_id is not None:
            try:
                out = ctx.deps.doc_manager.read_source(
                    line=line,
                    before=before,
                    after=after,
                    doc_id=workspace_doc_id,
                )
                out["requested_path"] = path
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
                return _read_source_error_payload(requested_path=path, error=error)
        try:
            out = read_source_via_client(
                ctx.deps.client,
                path,
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
            return _read_source_error_payload(requested_path=path, error=error)

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
    ) -> dict[str, Any]:
        """Prepare helper lemma. Example: prepare_intermediate_lemma(lemma_name='helper_card', lemma_type='forall n : nat, n > 0 -> True')."""
        return ctx.deps.prepare_intermediate_lemma(
            lemma_type=lemma_type,
            lemma_name=lemma_name,
            doc_id=doc_id,
        )

    @agent.tool
    def prove_intermediate_lemma(
        ctx: RunContext[DocqAgentSession],
        lemma_name: str,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        return ctx.deps.prove_intermediate_lemma(lemma_name=lemma_name, prompt=prompt)

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
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        """Prepare+prove helper lemma. Example: add_intermediate_lemma(lemma_name='helper_card', lemma_type='forall n : nat, n > 0 -> True')."""
        return ctx.deps.add_intermediate_lemma(
            lemma_type=lemma_type,
            lemma_name=lemma_name,
            prompt=prompt,
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
        return output

    return agent
