from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits

from .doc_manager import BranchSession, DocumentManager
from .docstring_tools import SemanticDocSearchClient
from .library_tools import TocExplorer, read_source_via_client

DOCQ_SYSTEM_PROMPT = (
    "You are editing and proving Rocq code with strict tool usage.\n"
    "Rules:\n"
    "- Use `explore_toc` incrementally (do not request global dumps repeatedly).\n"
    "- Use `read_source_file` to inspect library files with optional line window.\n"
    "- Use `show_workspace` to inspect the current virtual manipulated file.\n"
    "- Use `run_tac` from any known state index.\n"
    "- For imports, only call `add_import(libname, source)`.\n"
    "- If an import is problematic, call `remove_import(libname, source)`.\n"
    "- For helper statements, call `add_intermediate_lemma(lemma_type)`.\n"
    "- If a helper lemma is problematic, call `remove_intermediate_lemma(lemma_name)`.\n"
    "- If stuck on lemma proving, the sub-agent can abort and report why."
)

LEMMA_SUBAGENT_SYSTEM_PROMPT = (
    "You are proving one intermediate lemma.\n"
    "Use tools:\n"
    "- `explore_toc`, `semantic_doc_search`, `read_source_file`\n"
    "- `show_workspace`, `read_workspace_source`\n"
    "- `list_states`, `get_goals`, `run_tac`\n"
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
        return cls(
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

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

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
            subagent.run_sync(
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
        return {"ok": True, "proof_script": sub_branch.proof_script(latest), "state_index": latest}

    def add_intermediate_lemma(
        self,
        *,
        lemma_type: str,
        prompt: str | None = None,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        lemma_name = self.doc_manager.next_lemma_name()
        try:
            base_doc_id, sub_branch = self.doc_manager.create_lemma_subsession(
                lemma_name=lemma_name,
                lemma_type=lemma_type,
                doc_id=doc_id,
            )
        except Exception as exc:
            return {"ok": False, "error": f"Lemma declaration/type-check failed: {exc}", "lemma_name": lemma_name}

        sub_prompt = prompt or (
            f"Prove the intermediate lemma `{lemma_name}`. "
            "Use run_tac/list_states/get_goals. Call abort if impossible."
        )
        sub_result = self.run_lemma_subagent(sub_branch=sub_branch, prompt=sub_prompt)
        if not sub_result.get("ok"):
            return {
                "ok": False,
                "lemma_name": lemma_name,
                "error": sub_result.get("error", "Sub-agent failed."),
                "aborted": bool(sub_result.get("aborted", False)),
            }

        proof_script = str(sub_result.get("proof_script", "")).strip()
        try:
            reg = self.doc_manager.register_proved_lemma(
                base_doc_id=base_doc_id,
                lemma_name=lemma_name,
                lemma_type=lemma_type,
                proof_script=proof_script,
            )
        except Exception as exc:
            return {"ok": False, "lemma_name": lemma_name, "error": f"Lemma registration failed: {exc}"}

        reg["lemma_name"] = lemma_name
        reg["proof_script"] = proof_script
        return reg


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
    def show_workspace(ctx: RunContext[DocqAgentSession]) -> dict[str, Any]:
        return ctx.deps.doc_manager.show_workspace()

    @agent.tool
    def list_states(ctx: RunContext[DocqAgentSession]) -> list[dict[str, Any]]:
        return ctx.deps.doc_manager.list_states_verbose()

    @agent.tool
    def get_goals(ctx: RunContext[DocqAgentSession], state_index: int = 0) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.get_goals(state_index=state_index)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def run_tac(
        ctx: RunContext[DocqAgentSession],
        state_index: int = 0,
        tactic: str = "idtac.",
    ) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.run_tac(state_index=state_index, tactic=tactic)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def read_workspace_source(
        ctx: RunContext[DocqAgentSession],
        line: int | None = None,
        before: int = 20,
        after: int = 20,
    ) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.read_source(line=line, before=before, after=after)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def add_import(ctx: RunContext[DocqAgentSession], libname: str, source: str) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.add_import(libname=libname, source=source)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def remove_import(ctx: RunContext[DocqAgentSession], libname: str, source: str) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.remove_import(libname=libname, source=source)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def add_intermediate_lemma(
        ctx: RunContext[DocqAgentSession],
        lemma_type: str,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        result = ctx.deps.add_intermediate_lemma(lemma_type=lemma_type, prompt=prompt)
        if not result.get("ok", False):
            raise ModelRetry(str(result.get("error", "Failed to add intermediate lemma.")))
        return result

    @agent.tool
    def remove_intermediate_lemma(
        ctx: RunContext[DocqAgentSession],
        lemma_name: str,
    ) -> dict[str, Any]:
        try:
            return ctx.deps.doc_manager.remove_intermediate_lemma(lemma_name=lemma_name)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    return agent
