from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits

from .doc_manager import BranchSession, DocumentManager, parse_last_target_layout
from .docstring_tools import SemanticDocSearchClient
from .library_tools import TocExplorer, read_source_via_client

DOCQ_SYSTEM_PROMPT = (
    "You are editing and proving Rocq code with strict tool usage.\n"
    "Rules:\n"
    "- Use `explore_toc` incrementally and discover root entries first.\n"
    "- Use `read_source_file` to inspect library files with optional line window.\n"
    "- Use `show_workspace` or `show_doc(doc_id)` to inspect virtual files.\n"
    "- Use `list_docs` and `checkout_doc` to navigate document branches.\n"
    "- Use `run_tac` from any known state index (optionally scoped by `doc_id`).\n"
    "- For imports, use `add_import(libname, source, doc_id?)` and `remove_import(...)`.\n"
    "- For helper statements, prefer phased tools:\n"
    "  `prepare_intermediate_lemma` -> `prove_intermediate_lemma` -> `drop_pending_intermediate_lemma`.\n"
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


@dataclass
class LemmaSubSession:
    branch: BranchSession
    client: Any
    toc_explorer: TocExplorer
    semantic_search: SemanticDocSearchClient | None = None
    abort_reason: str | None = None
    required_imports: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PendingLemma:
    base_doc_id: int
    lemma_name: str
    lemma_type: str
    sub_branch: BranchSession


_IDENT_RE = re.compile(r"^[A-Za-z0-9_.]+$")


def build_docq_subagent(model: Any = None, *, retries: int = 1) -> Agent[LemmaSubSession, str]:
    agent = Agent(
        model=model,
        deps_type=LemmaSubSession,
        output_type=str,
        name="docq-lemma-subagent",
        system_prompt=LEMMA_SUBAGENT_SYSTEM_PROMPT,
        retries=retries,
    )

    @agent.tool
    def list_states(ctx: RunContext[LemmaSubSession]) -> list[int]:
        return ctx.deps.branch.available_state_indexes

    @agent.tool
    def explore_toc(ctx: RunContext[LemmaSubSession], path: list[str] | None = None) -> dict[str, Any]:
        return ctx.deps.toc_explorer.explore(path or [])

    @agent.tool
    def semantic_doc_search(
        ctx: RunContext[LemmaSubSession],
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
        ctx: RunContext[LemmaSubSession],
        path: str | None = None,
        line: int | None = None,
        before: int = 20,
        after: int = 20,
    ) -> dict[str, Any]:
        try:
            return read_source_via_client(
                ctx.deps.client,
                path or str(ctx.deps.branch.source_path),
                line=line,
                before=before,
                after=after,
            )
        except Exception as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def show_workspace(ctx: RunContext[LemmaSubSession]) -> dict[str, Any]:
        content = read_source_via_client(ctx.deps.client, str(ctx.deps.branch.source_path))
        return {
            "doc_id": ctx.deps.branch.doc_id,
            "source_path": str(ctx.deps.branch.source_path),
            "states": ctx.deps.branch.list_states(),
            "content": content.get("content", ""),
        }

    @agent.tool
    def read_workspace_source(
        ctx: RunContext[LemmaSubSession],
        line: int | None = None,
        before: int = 20,
        after: int = 20,
    ) -> dict[str, Any]:
        try:
            payload = read_source_via_client(
                ctx.deps.client,
                str(ctx.deps.branch.source_path),
                line=line,
                before=before,
                after=after,
            )
            payload["doc_id"] = ctx.deps.branch.doc_id
            payload["source_path"] = str(ctx.deps.branch.source_path)
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
        if not _IDENT_RE.match(libname):
            raise ModelRetry(f"Invalid libname={libname!r}.")
        if not _IDENT_RE.match(source):
            raise ModelRetry(f"Invalid source={source!r}.")
        pair = (libname, source)
        if pair not in ctx.deps.required_imports:
            ctx.deps.required_imports.append(pair)
        return {
            "ok": True,
            "required_imports": [
                {"libname": lib, "source": src} for (lib, src) in ctx.deps.required_imports
            ],
        }

    @agent.tool
    def abort(ctx: RunContext[LemmaSubSession], explanation: str = "not provable") -> str:
        ctx.deps.abort_reason = explanation.strip() or "not provable"
        return f"ABORT: {ctx.deps.abort_reason}"

    return agent


@dataclass
class DocqAgentSession:
    client: Any
    source_path: Path
    env: str
    doc_manager: DocumentManager
    toc_explorer: TocExplorer
    semantic_search: SemanticDocSearchClient | None = None
    timeout: float = 60.0
    usage: RunUsage = field(default_factory=RunUsage)
    usage_limits: UsageLimits = field(default_factory=lambda: UsageLimits(tool_calls_limit=120))
    logger: Callable[[str], None] | None = None
    subagent_model: Any = None
    subagent_retries: int = 1
    pending_lemmas: dict[str, PendingLemma] = field(default_factory=dict)

    @classmethod
    def from_source(
        cls,
        client: Any,
        source_path: str | Path,
        *,
        env: str,
        timeout: float = 60.0,
        connect: bool = True,
        logger: Callable[[str], None] | None = None,
        semantic_base_url: str | None = None,
        semantic_route: str = "/search",
        semantic_api_key: str | None = None,
        max_tool_calls: int = 120,
        subagent_model: Any = None,
        subagent_retries: int = 1,
    ) -> "DocqAgentSession":
        if connect:
            client.connect()
        source = Path(source_path).resolve()
        manager = DocumentManager(client, source, timeout=timeout, logger=logger)
        explorer = TocExplorer(client, env=env)
        semantic_client = None
        if semantic_base_url:
            semantic_client = SemanticDocSearchClient(
                base_url=semantic_base_url,
                route=semantic_route,
                api_key=semantic_api_key,
                timeout=timeout,
            )
        session = cls(
            client=client,
            source_path=source,
            env=env,
            doc_manager=manager,
            toc_explorer=explorer,
            semantic_search=semantic_client,
            timeout=timeout,
            usage=RunUsage(),
            usage_limits=UsageLimits(tool_calls_limit=max_tool_calls),
            logger=logger,
            subagent_model=subagent_model,
            subagent_retries=subagent_retries,
        )
        session.doc_manager.mutation_validator = session._validate_mutation_in_fresh_session
        return session

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def _try_make_fresh_client(self) -> Any | None:
        host = getattr(self.client, "host", None)
        port = getattr(self.client, "port", None)
        if host is None or port is None:
            return None
        try:
            fresh = type(self.client)(host, port)
        except Exception:
            return None
        if hasattr(self.client, "timeout_http") and hasattr(fresh, "timeout_http"):
            try:
                fresh.timeout_http = self.client.timeout_http
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
            tmp_path = fresh.tmp_file(content=node.content, root=str(self.source_path.parent))
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

    def run_lemma_subagent(
        self,
        *,
        sub_branch: BranchSession,
        prompt: str,
    ) -> dict[str, Any]:
        remaining = self.remaining_tool_calls()
        if remaining is not None and remaining <= 0:
            return {"ok": False, "error": "Tool-call budget exhausted before sub-agent run."}

        deps = LemmaSubSession(
            branch=sub_branch,
            client=self.client,
            toc_explorer=self.toc_explorer,
            semantic_search=self.semantic_search,
        )
        subagent = build_docq_subagent(model=self.subagent_model, retries=self.subagent_retries)
        limits = None
        if remaining is not None:
            limits = UsageLimits(tool_calls_limit=remaining)

        try:
            _ = subagent.run_sync(
                prompt,
                deps=deps,
                usage=self.usage,
                usage_limits=limits,
            )
        except UsageLimitExceeded as exc:
            return {"ok": False, "error": f"Sub-agent budget exceeded: {exc}"}
        except Exception as exc:
            return {"ok": False, "error": f"Sub-agent failure: {exc}"}

        if deps.abort_reason:
            return {"ok": False, "error": deps.abort_reason, "aborted": True}

        latest = sub_branch.latest_state_index
        goals = sub_branch.get_goals(latest).get("goals", [])
        if goals:
            return {"ok": False, "error": "Sub-agent did not finish the lemma proof."}
        required_imports = [
            {"libname": libname, "source": source} for (libname, source) in deps.required_imports
        ]
        return {
            "ok": True,
            "proof_script": sub_branch.proof_script(latest),
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
        prep = self.prepare_intermediate_lemma(
            lemma_type=lemma_type,
            lemma_name=lemma_name,
            doc_id=doc_id,
        )
        if not prep.get("ok", False):
            return prep
        proved = self.prove_intermediate_lemma(
            lemma_name=str(prep["lemma_name"]),
            prompt=prompt,
        )
        if not proved.get("ok", False):
            proved["prepared"] = True
        return proved

    def prepare_intermediate_lemma(
        self,
        *,
        lemma_type: str,
        lemma_name: str | None = None,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
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
        return reg

    def drop_pending_intermediate_lemma(self, *, lemma_name: str) -> dict[str, Any]:
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


def build_docq_agent(model: Any = None, *, retries: int = 2) -> Agent[DocqAgentSession, str]:
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
        return ctx.deps.toc_explorer.explore(path or [])

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
        try:
            return read_source_via_client(
                ctx.deps.client,
                path,
                line=line,
                before=before,
                after=after,
            )
        except Exception as exc:
            raise ModelRetry(str(exc)) from exc

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
        try:
            return ctx.deps.doc_manager.add_import(libname=libname, source=source, doc_id=doc_id)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def remove_import(
        ctx: RunContext[DocqAgentSession],
        libname: str,
        source: str,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.remove_import(libname=libname, source=source, doc_id=doc_id)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def prepare_intermediate_lemma(
        ctx: RunContext[DocqAgentSession],
        lemma_type: str,
        lemma_name: str | None = None,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
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

    return agent
