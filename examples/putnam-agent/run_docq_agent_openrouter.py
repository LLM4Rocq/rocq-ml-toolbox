#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from agent import (  # noqa: E402
    DocqAgentSession,
    build_docq_agent,
    make_console_logger,
)
from rocq_ml_toolbox.inference.client import PytanqueExtended  # noqa: E402

DEFAULT_SOURCE = THIS_DIR / "putnam" / "mathcomp" / "putnam_1965_a5.v"
DEFAULT_MODEL = "openai/gpt-5.3-codex"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_MIN_P = 0.01
DEFAULT_REPEAT_PENALTY = 1.0
DEFAULT_THRESHOLD_COMPRESSION = 100_000
DEFAULT_PROMPT = (
    "Inspect the workspace and propose one useful intermediate lemma candidate, "
    "then attempt proving it."
)
COMPRESSION_SYSTEM_PROMPT = (
    "You compress theorem-proving sessions for continuation after context reset.\n"
    "Return a faithful handoff in plain text.\n"
    "Preserve exact names (theorem/lemma/import/doc_id/state), open-goal status, failures, and next actions.\n"
    "Never invent solved goals, imported modules, or completed lemmas."
)


class _CompressionRequested(Exception):
    def __init__(self, *, reason: str, req_input_tokens: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.req_input_tokens = req_input_tokens


def parse_args() -> argparse.Namespace:
    default_artifacts_dir = THIS_DIR / "interactive_test" / f"docq_batch_{time.strftime('%Y%m%d_%H%M%S')}"
    parser = argparse.ArgumentParser(description="Run the docq agent on one Rocq source file.")
    parser.add_argument("-k", "--num-agents", type=int, default=1, help="Number of agents to run.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent agents (default: k).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Path to source .v file to manipulate (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--env",
        default=None,
        help=(
            "Optional environment id for /access_libraries "
            "(needed only if multiple <coq_lib>/*.toc.json files exist)."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tool-calls", type=int, default=1000)
    parser.add_argument("--max-requests", type=int, default=500)
    parser.add_argument(
        "--threshold-compression",
        type=int,
        default=DEFAULT_THRESHOLD_COMPRESSION,
        help=(
            "Token threshold for context compaction. "
            "When exceeded, the runner asks the model for a continuation summary and restarts with "
            "'main task prompt + summary'. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-output-validation-recoveries",
        type=int,
        default=4,
        help=(
            "Automatic recovery attempts when output validation retries are exhausted "
            "(default: 4, set 0 to disable)."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Nucleus sampling top-p (default: {DEFAULT_TOP_P}).",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=DEFAULT_MIN_P,
        help=f"Minimum probability cutoff (OpenRouter `min_p`, default: {DEFAULT_MIN_P}).",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=DEFAULT_REPEAT_PENALTY,
        help=f"Repetition penalty (OpenRouter `repetition_penalty`, default: {DEFAULT_REPEAT_PENALTY}).",
    )
    parser.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--openrouter-base-url",
        default=os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    parser.add_argument("--semantic-base-url", default=os.getenv("DOCQ_SEARCH_BASE_URL"))
    parser.add_argument("--semantic-route", default=os.getenv("DOCQ_SEARCH_ROUTE", "/search"))
    parser.add_argument("--semantic-api-key", default=os.getenv("DOCQ_SEARCH_API_KEY"))
    parser.add_argument(
        "--disable-semantic-tool",
        action="store_true",
        help="Disable semantic_doc_search tool exposure (main + sub-agent).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable real-time runner/session logs.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=default_artifacts_dir,
        help=f"Directory where per-agent traces/docs are exported (default: {default_artifacts_dir}).",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Disable artifact export (trace/log/docs).",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


@dataclass(frozen=True)
class DocqAgentTask:
    prompt: str
    source: Path
    name: str = ""


@dataclass
class ScalableDocqRunner:
    client_factory: Callable[[], Any]
    agent: Any
    env: str | None
    subagent_model: Any = None
    subagent_retries: int = 1
    timeout: float = 60.0
    max_concurrency: int = 1
    max_tool_calls: int = 120
    max_requests: int | None = 200
    semantic_base_url: str | None = None
    semantic_route: str = "/search"
    semantic_api_key: str | None = None
    include_semantic_tool: bool = True
    logger: Callable[[str], None] | None = None
    log_enabled: bool = False
    log_prefix: str = "docq-runner"
    artifacts_dir: Path | None = None
    threshold_compression: int = DEFAULT_THRESHOLD_COMPRESSION
    compression_model: Any | None = None
    max_compressions_per_task: int = 8
    max_output_validation_recoveries: int = 4
    _compression_agent: Agent[Any, str] | None = field(default=None, init=False, repr=False)

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)
            return
        if not self.log_enabled:
            return
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}][{self.log_prefix}] {message}", flush=True)

    @staticmethod
    def _safe_task_label(label: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._-")
        return safe or "agent"

    @staticmethod
    def _usage_to_dict(usage: Any) -> dict[str, Any]:
        input_tokens = getattr(usage, "input_tokens", getattr(usage, "request_tokens", None))
        output_tokens = getattr(usage, "output_tokens", None)
        return {
            "requests": getattr(usage, "requests", None),
            # Keep the legacy key for backward compatibility in artifacts.
            "request_tokens": input_tokens,
            "input_tokens": input_tokens,
            # Keep the legacy key for backward compatibility in artifacts.
            "response_tokens": output_tokens,
            "output_tokens": output_tokens,
            "total_tokens": getattr(usage, "total_tokens", None),
            "tool_calls": getattr(usage, "tool_calls", None),
        }

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): ScalableDocqRunner._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ScalableDocqRunner._jsonable(v) for v in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return ScalableDocqRunner._jsonable(model_dump(mode="json"))
            except TypeError:
                return ScalableDocqRunner._jsonable(model_dump())
            except Exception:
                pass
        to_dict = getattr(value, "dict", None)
        if callable(to_dict):
            try:
                return ScalableDocqRunner._jsonable(to_dict())
            except Exception:
                pass
        return repr(value)

    @staticmethod
    def _append_jsonl(path: Path | None, payload: Any) -> None:
        if path is None:
            return
        line = json.dumps(ScalableDocqRunner._jsonable(payload), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")

    @staticmethod
    def _extract_last_response_usage(run_ctx: Any) -> tuple[int, int, int, int] | None:
        extracted = ScalableDocqRunner._extract_last_request_response(run_ctx)
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
        return ScalableDocqRunner._extract_last_request_response_from_messages(messages)

    @staticmethod
    def _extract_last_request_response_from_messages(
        messages: Sequence[Any],
    ) -> tuple[int, Any | None, Any, int, int, int] | None:
        # In some streams the last message may be a non-response artifact; walk
        # backwards to find the most recent message carrying usage.
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
    def _history_excerpt(
        message_history: Sequence[Any],
        *,
        max_items: int = 14,
        max_chars: int = 10_000,
    ) -> str:
        if not message_history:
            return "(none)"
        tail = list(message_history[-max_items:])
        base_index = max(0, len(message_history) - len(tail))
        lines: list[str] = []
        for i, msg in enumerate(tail, start=base_index):
            payload = ScalableDocqRunner._jsonable(msg)
            encoded = json.dumps(payload, ensure_ascii=False)
            lines.append(f"[{i}] {encoded}")
        out = "\n".join(lines)
        if len(out) > max_chars:
            return out[:max_chars] + "\n... [truncated]"
        return out

    def _make_attempt_usage_limits(self, session: DocqAgentSession) -> UsageLimits:
        # Compression threshold is enforced from per-request context size
        # (`req_input_tokens`) in the event handler, not via cumulative total.
        total_tokens_limit = getattr(session.usage_limits, "total_tokens_limit", None)
        return UsageLimits(
            request_limit=getattr(session.usage_limits, "request_limit", None),
            tool_calls_limit=getattr(session.usage_limits, "tool_calls_limit", None),
            input_tokens_limit=getattr(session.usage_limits, "input_tokens_limit", None),
            output_tokens_limit=getattr(session.usage_limits, "output_tokens_limit", None),
            total_tokens_limit=total_tokens_limit,
        )

    @staticmethod
    def _is_compression_limit_error(exc: Exception) -> bool:
        text = str(exc)
        return "total_tokens_limit" in text

    @staticmethod
    def _is_output_validation_retry_exhaustion(exc: Exception) -> bool:
        text = str(exc)
        if not text:
            return False
        return ("Exceeded maximum retries" in text) and ("output validation" in text)

    @staticmethod
    def _resume_prompt_after_output_validation(
        *,
        main_prompt: str,
        error_text: str,
        status: dict[str, Any] | None,
        attempt: int,
        max_attempts: int,
    ) -> str:
        status = status or {}
        return (
            f"{main_prompt.strip()}\n\n"
            "Recovery Context:\n"
            f"- Output validation retry exhaustion detected ({attempt}/{max_attempts}).\n"
            f"- Validator error: {error_text}\n"
            f"- completion_status snapshot: {json.dumps(status, ensure_ascii=False)}\n\n"
            "Strict continuation instructions:\n"
            "1) Do NOT output final text now.\n"
            "2) First call `completion_status`.\n"
            "3) Then call `get_goals` on the latest state.\n"
            "4) Continue only with proof tools (`run_tac`, imports, branch tools) until no goals remain.\n"
            "5) Output final text only after `completion_status` reports `latest_proof_finished=true` and zero goals."
        )

    def _get_compression_agent(self) -> Agent[Any, str]:
        if self._compression_agent is None:
            model = self.compression_model or self.subagent_model
            if model is None:
                raise RuntimeError("No model available for context compression summary.")
            self._compression_agent = Agent(
                model=model,
                output_type=str,
                system_prompt=COMPRESSION_SYSTEM_PROMPT,
                retries=1,
                name="docq-context-compressor",
            )
        return self._compression_agent

    @staticmethod
    def _resume_prompt(main_prompt: str, summary: str) -> str:
        return (
            f"{main_prompt.strip()}\n\n"
            "Context Summary From Previous Segment:\n"
            f"{summary.strip()}\n\n"
            "Continue the same task from this summary."
        )

    async def _summarize_for_compression(
        self,
        *,
        session: DocqAgentSession,
        main_prompt: str,
        message_history: Sequence[Any],
        round_index: int,
    ) -> tuple[str, dict[str, Any] | None]:
        status: dict[str, Any] = {}
        try:
            status = session.completion_status()
        except Exception:
            status = {"error": "completion_status unavailable"}
        head_doc_id = int(status.get("head_doc_id", session.doc_manager.head_doc_id))
        latest_state_index = int(
            status.get(
                "latest_state_index",
                session.doc_manager.sessions[head_doc_id].latest_state_index,
            )
        )
        goal_snapshot: dict[str, Any] = {}
        try:
            goal_snapshot = session.doc_manager.get_goals(doc_id=head_doc_id, state_index=latest_state_index)
        except Exception as exc:
            goal_snapshot = {"error": str(exc), "goals": []}
        goals = goal_snapshot.get("goals", []) if isinstance(goal_snapshot, dict) else []
        goal_preview: list[str] = []
        if isinstance(goals, list):
            for goal in goals[:3]:
                text = str(goal)
                if len(text) > 1200:
                    text = text[:1200] + " ..."
                goal_preview.append(text)
        proof_tail: list[str] = []
        try:
            proof_lines = (
                session.doc_manager.sessions[head_doc_id]
                .proof_script(latest_state_index)
                .splitlines()
            )
            proof_tail = [ln for ln in proof_lines[-25:] if ln.strip()]
        except Exception:
            proof_tail = []
        snapshot = {
            "compression_round": round_index,
            "completion_status": status,
            "head_doc_id": head_doc_id,
            "latest_state_index": latest_state_index,
            "latest_goals_count": len(goal_preview),
            "latest_goals": goal_preview,
            "proof_script_tail": proof_tail,
            "pending_lemmas": session.list_pending_intermediate_lemmas(),
        }
        history_excerpt = self._history_excerpt(message_history, max_items=24, max_chars=24_000)
        prompt = (
            "Create a continuation-safe handoff summary so another run can continue without full history.\n"
            "Write EXACTLY these sections with these headers:\n"
            "1) MAIN TASK\n"
            "2) GLOBAL STATUS\n"
            "3) WHAT WORKED\n"
            "4) WHAT FAILED (with exact error text snippets)\n"
            "5) LOCAL GOAL (verbatim from snapshot when available)\n"
            "6) NEXT ACTION PLAN (ordered concrete tool/tactic steps)\n"
            "Hard requirements:\n"
            "- Preserve exact symbol names, lemma names, doc_id/state indexes, and import atoms.\n"
            "- Do not claim goals are solved unless snapshot says so.\n"
            "- In LOCAL GOAL, include at least one current goal statement if any goal exists.\n"
            "- In NEXT ACTION PLAN, include immediate first action to re-anchor state (e.g. completion_status/get_goals).\n"
            "- Keep actionable detail; avoid vague strategy-only summaries.\n"
            "- Target 500-900 words.\n\n"
            f"Main task prompt:\n{main_prompt}\n\n"
            f"Current snapshot (JSON):\n{json.dumps(snapshot, ensure_ascii=False)}\n\n"
            "Recent captured run messages (possibly truncated):\n"
            f"{history_excerpt}"
        )
        compression_agent = self._get_compression_agent()
        compression_limits = UsageLimits(
            request_limit=getattr(session.usage_limits, "request_limit", None),
            tool_calls_limit=getattr(session.usage_limits, "tool_calls_limit", None),
            input_tokens_limit=getattr(session.usage_limits, "input_tokens_limit", None),
            output_tokens_limit=getattr(session.usage_limits, "output_tokens_limit", None),
            total_tokens_limit=None,
        )
        with capture_run_messages() as compression_messages:
            result = await compression_agent.run(
                prompt,
                message_history=None,
                usage=session.usage,
                usage_limits=compression_limits,
            )
        summary_text = str(result.output).strip()
        extracted = self._extract_last_request_response_from_messages(compression_messages)
        if extracted is None:
            return summary_text, None
        (
            response_index,
            request_message,
            response_message,
            req_input_tokens,
            req_output_tokens,
            req_total_tokens,
        ) = extracted
        request_trace = {
            "compression_round": round_index,
            "response_index": response_index,
            "req_input_tokens": req_input_tokens,
            "req_output_tokens": req_output_tokens,
            "req_total_tokens": req_total_tokens,
            "context_message_count": response_index,
            "context_messages": list(compression_messages)[:response_index],
            "request_message": request_message,
            "response_message": response_message,
        }
        return summary_text, request_trace

    def _next_task_artifact_dir(self, task_label: str) -> Path | None:
        if self.artifacts_dir is None:
            return None
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        stem = self._safe_task_label(task_label)
        candidate = self.artifacts_dir / stem
        suffix = 2
        while candidate.exists():
            candidate = self.artifacts_dir / f"{stem}_{suffix}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _write_task_artifacts(
        self,
        *,
        task_dir: Path | None,
        task_label: str,
        task_logs: list[str],
        session: DocqAgentSession | None,
        output: str | None,
        error: str | None,
        traceback_text: str | None,
        compression_rounds: int = 0,
    ) -> None:
        if task_dir is None:
            return

        (task_dir / "task.log").write_text("\n".join(task_logs) + ("\n" if task_logs else ""), encoding="utf-8")
        if output is not None:
            (task_dir / "agent_output.txt").write_text(output, encoding="utf-8")

        all_messages_file = task_dir / "all_messages.jsonl"
        new_messages_file = task_dir / "new_messages.jsonl"
        events_file = task_dir / "events.jsonl"
        compression_file = task_dir / "compression_summaries.jsonl"
        request_io_file = task_dir / "request_io.jsonl"

        summary: dict[str, Any] = {
            "task_label": task_label,
            "status": "ok" if error is None else "error",
            "error": error,
            "traceback": traceback_text,
            "output_file": "agent_output.txt" if output is not None else None,
            "all_messages_file": all_messages_file.name if all_messages_file.exists() else None,
            "new_messages_file": new_messages_file.name if new_messages_file.exists() else None,
            "events_file": events_file.name if events_file.exists() else None,
            "compression_summaries_file": compression_file.name if compression_file.exists() else None,
            "request_io_file": request_io_file.name if request_io_file.exists() else None,
            "compression_rounds": compression_rounds,
        }

        if session is not None:
            docs_meta: list[dict[str, Any]] = []
            docs_dir = task_dir / "docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            final_doc_path: str | None = None
            final_doc_materialized_path: str | None = None
            for doc_id, node in sorted(session.doc_manager.nodes.items()):
                doc_path = docs_dir / f"doc_{doc_id}.v"
                doc_path.write_text(node.content, encoding="utf-8")
                session_branch = session.doc_manager.sessions.get(doc_id)
                if doc_id == session.doc_manager.head_doc_id:
                    final_copy = task_dir / "final_doc.v"
                    final_copy.write_text(node.content, encoding="utf-8")
                    final_doc_path = str(final_copy.relative_to(task_dir))
                    try:
                        final_materialized = task_dir / "final_doc_materialized.v"
                        final_materialized.write_text(
                            session.doc_manager.materialized_source(doc_id=doc_id),
                            encoding="utf-8",
                        )
                        final_doc_materialized_path = str(final_materialized.relative_to(task_dir))
                    except Exception as exc:
                        summary["final_doc_materialized_error"] = str(exc)
                docs_meta.append(
                    {
                        "doc_id": doc_id,
                        "parent_doc_id": node.parent_doc_id,
                        "label": node.label,
                        "is_head": doc_id == session.doc_manager.head_doc_id,
                        "content_file": str(doc_path.relative_to(task_dir)),
                        "state_count": len(session_branch.nodes) if session_branch is not None else None,
                        "source_path": str(session_branch.source_path) if session_branch is not None else None,
                    }
                )
            summary["head_doc_id"] = session.doc_manager.head_doc_id
            summary["final_doc_file"] = final_doc_path
            summary["final_doc_materialized_file"] = final_doc_materialized_path
            summary["docs"] = docs_meta
            summary["pending_lemmas"] = session.list_pending_intermediate_lemmas()
            summary["usage"] = self._usage_to_dict(session.usage)
            summary["usage_limits"] = {
                "tool_calls_limit": getattr(session.usage_limits, "tool_calls_limit", None),
                "request_limit": getattr(session.usage_limits, "request_limit", None),
            }
            try:
                head_doc_id = session.doc_manager.head_doc_id
                status = session.doc_manager.completion_status(doc_id=head_doc_id)
                summary["head_status"] = status
                summary["head_state_index"] = status.get("latest_state_index")
                summary["head_goals_count"] = status.get("latest_goals_count")
                summary["head_proof_finished"] = status.get("latest_proof_finished")
                summary["head_has_placeholder_tactic"] = status.get("latest_has_placeholder_tactic")
            except Exception as exc:
                summary["head_status_error"] = str(exc)

        (task_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    async def run_task(self, task: DocqAgentTask) -> str:
        task_label = task.name or task.source.name
        self._log(f"task start: {task_label}")
        task_logs: list[str] = []
        task_dir = self._next_task_artifact_dir(task_label)
        if task_dir is not None:
            self._log(f"task artifacts: {task_label} -> {task_dir}")
        task_log_path = task_dir / "task.log" if task_dir is not None else None
        all_messages_path = task_dir / "all_messages.jsonl" if task_dir is not None else None
        new_messages_path = task_dir / "new_messages.jsonl" if task_dir is not None else None
        events_path = task_dir / "events.jsonl" if task_dir is not None else None
        compression_path = task_dir / "compression_summaries.jsonl" if task_dir is not None else None
        request_io_path = task_dir / "request_io.jsonl" if task_dir is not None else None
        if task_log_path is not None:
            task_log_path.write_text("", encoding="utf-8")
        if all_messages_path is not None:
            all_messages_path.write_text("", encoding="utf-8")
        if new_messages_path is not None:
            new_messages_path.write_text("", encoding="utf-8")
        if events_path is not None:
            events_path.write_text("", encoding="utf-8")
        if compression_path is not None:
            compression_path.write_text("", encoding="utf-8")
        if request_io_path is not None:
            request_io_path.write_text("", encoding="utf-8")
        captured_messages: list[Any] | None = None
        run_message_cursor = 0
        global_message_index = 0
        request_io_index = 0
        event_cursor = 0
        model_call_index = 0
        last_logged_requests = 0
        last_logged_input_tokens = 0
        last_logged_output_tokens = 0
        last_logged_main_response_index = -1
        compression_rounds = 0

        def _task_record(message: str) -> None:
            stamp = time.strftime("%H:%M:%S")
            line = f"[{stamp}] {message}"
            task_logs.append(line)
            if task_log_path is not None:
                with task_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.write("\n")

        def _task_record_echo(message: str) -> None:
            _task_record(message)
            self._log(message)

        def _flush_captured_messages() -> None:
            nonlocal run_message_cursor, global_message_index
            if captured_messages is None:
                return
            while run_message_cursor < len(captured_messages):
                payload = {
                    "message_index": global_message_index,
                    "source": "main",
                    "message": captured_messages[run_message_cursor],
                }
                self._append_jsonl(all_messages_path, payload)
                self._append_jsonl(new_messages_path, payload)
                run_message_cursor += 1
                global_message_index += 1

        def _record_session_message(source: str, message: Any) -> None:
            nonlocal global_message_index
            payload = {
                "message_index": global_message_index,
                "source": source,
                "message": message,
            }
            self._append_jsonl(all_messages_path, payload)
            self._append_jsonl(new_messages_path, payload)
            global_message_index += 1

        def _record_session_event(source: str, event: Any) -> None:
            nonlocal event_cursor
            self._append_jsonl(
                events_path,
                {
                    "event_index": event_cursor,
                    "source": source,
                    "event": event,
                },
            )
            event_cursor += 1

        def _record_request_io(source: str, payload: Any) -> None:
            nonlocal request_io_index
            entry = {
                "io_index": request_io_index,
                "source": source,
                "payload": payload,
            }
            self._append_jsonl(request_io_path, entry)
            request_io_index += 1

        async def _event_stream_handler(_run_ctx: Any, stream: Any) -> None:
            nonlocal event_cursor, model_call_index, last_logged_requests
            nonlocal last_logged_input_tokens, last_logged_output_tokens
            nonlocal last_logged_main_response_index
            async for event in stream:
                self._append_jsonl(
                    events_path,
                    {
                        "event_index": event_cursor,
                        "event": event,
                    },
                )
                event_cursor += 1
                _flush_captured_messages()
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
                if response_index > last_logged_main_response_index:
                    model_call_index += 1
                    _task_record_echo(
                        f"{task_label} | model call("
                        f"index={model_call_index}, "
                        f"run_step={getattr(_run_ctx, 'run_step', None)}, "
                        f"requests={requests_now}, "
                        f"req_input_tokens={req_input}, "
                        f"req_output_tokens={req_output}, "
                        f"req_total_tokens={req_total}, "
                        f"input_tokens={input_tokens_now}, "
                        f"output_tokens={output_tokens_now})"
                    )
                    _record_request_io(
                        "main",
                        {
                            "index": model_call_index,
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
                    last_logged_main_response_index = response_index
                    last_logged_requests = requests_now
                    last_logged_input_tokens = input_tokens_now
                    last_logged_output_tokens = output_tokens_now
                    if self.threshold_compression > 0 and req_input > self.threshold_compression:
                        raise _CompressionRequested(
                            reason="req_input_tokens_threshold",
                            req_input_tokens=req_input,
                        )
                    return
            if requests_now > last_logged_requests:
                delta_requests = requests_now - last_logged_requests
                delta_input_tokens = input_tokens_now - last_logged_input_tokens
                delta_output_tokens = output_tokens_now - last_logged_output_tokens
                if delta_requests == 1 and delta_input_tokens == 0 and delta_output_tokens == 0:
                    # Avoid noisy duplicate "coalesced" logs when stream events
                    # arrive before per-response usage is attached.
                    return
                model_call_index += 1
                _task_record_echo(
                    f"{task_label} | model call("
                    f"index={model_call_index}, "
                    f"run_step={getattr(_run_ctx, 'run_step', None)}, "
                    f"requests={requests_now}, "
                    f"input_tokens={input_tokens_now}, "
                    f"output_tokens={output_tokens_now}, "
                    f"delta_input_tokens={delta_input_tokens}, "
                    f"delta_output_tokens={delta_output_tokens}, "
                    f"delta_requests={delta_requests})"
                )
                if delta_requests > 1:
                    _task_record_echo(
                        f"{task_label} | model call coalesced(delta_requests={delta_requests}) "
                        "because shared usage includes non-main requests (e.g. subagent/compression)."
                    )
                last_logged_requests = requests_now
                last_logged_input_tokens = input_tokens_now
                last_logged_output_tokens = output_tokens_now

        _task_record(f"task start: {task_label}")

        client = self.client_factory()
        session: DocqAgentSession | None = None
        run_result: Any | None = None
        output: str | None = None
        error: str | None = None
        tb_text: str | None = None

        def session_logger(message: str, *, label: str = task_label) -> None:
            line = f"{label} | {message}"
            _task_record(line)
            self._log(line)
        try:
            session = DocqAgentSession.from_source(
                client=client,
                source_path=task.source,
                env=self.env,
                timeout=self.timeout,
                connect=True,
                logger=session_logger,
                semantic_base_url=self.semantic_base_url,
                semantic_route=self.semantic_route,
                semantic_api_key=self.semantic_api_key,
                max_tool_calls=self.max_tool_calls,
                max_requests=self.max_requests,
                subagent_model=self.subagent_model,
                subagent_retries=self.subagent_retries,
                threshold_compression=self.threshold_compression,
                include_semantic_tool=self.include_semantic_tool,
                trace_message_callback=_record_session_message,
                trace_event_callback=_record_session_event,
                trace_request_callback=_record_request_io,
            )
            session_logger(
                "subagent budget config("
                f"tool_calls_limit={getattr(session.usage_limits, 'tool_calls_limit', None)}, "
                f"request_limit={getattr(session.usage_limits, 'request_limit', None)}, "
                f"input_tokens_limit={getattr(session.usage_limits, 'input_tokens_limit', None)}, "
                f"output_tokens_limit={getattr(session.usage_limits, 'output_tokens_limit', None)}, "
                f"total_tokens_limit={getattr(session.usage_limits, 'total_tokens_limit', None)}, "
                f"compression_threshold_tokens={self.threshold_compression})"
            )
            last_logged_requests = int(getattr(session.usage, "requests", 0) or 0)
            last_logged_input_tokens = int(getattr(session.usage, "input_tokens", 0) or 0)
            last_logged_output_tokens = int(getattr(session.usage, "output_tokens", 0) or 0)
            last_logged_main_response_index = -1
            current_prompt = task.prompt
            current_history: Sequence[Any] | None = None
            output_validation_recoveries = 0
            while True:
                limits = self._make_attempt_usage_limits(session)
                with capture_run_messages() as run_messages:
                    captured_messages = run_messages
                    run_message_cursor = 0
                    try:
                        run_result = await self.agent.run(
                            current_prompt,
                            deps=session,
                            message_history=current_history,
                            usage=session.usage,
                            usage_limits=limits,
                            event_stream_handler=_event_stream_handler if task_dir is not None else None,
                        )
                    except UnexpectedModelBehavior as exc:
                        _flush_captured_messages()
                        if not self._is_output_validation_retry_exhaustion(exc):
                            raise
                        output_validation_recoveries += 1
                        if output_validation_recoveries > self.max_output_validation_recoveries:
                            raise RuntimeError(
                                "Exceeded output-validation recovery attempts "
                                f"({self.max_output_validation_recoveries}). Last error: {exc}"
                            ) from exc
                        status_snapshot: dict[str, Any]
                        try:
                            status_snapshot = session.completion_status()
                        except Exception as status_exc:
                            status_snapshot = {"error": str(status_exc)}
                        _task_record_echo(
                            f"{task_label} | output validation recovery triggered "
                            f"(attempt={output_validation_recoveries}/{self.max_output_validation_recoveries})"
                        )
                        if events_path is not None:
                            self._append_jsonl(
                                events_path,
                                {
                                    "event_index": event_cursor,
                                    "event": {
                                        "type": "output_validation_recovery",
                                        "attempt": output_validation_recoveries,
                                        "max_attempts": self.max_output_validation_recoveries,
                                        "error": str(exc),
                                        "status": status_snapshot,
                                    },
                                },
                            )
                            event_cursor += 1
                        current_prompt = self._resume_prompt_after_output_validation(
                            main_prompt=task.prompt,
                            error_text=str(exc),
                            status=status_snapshot,
                            attempt=output_validation_recoveries,
                            max_attempts=self.max_output_validation_recoveries,
                        )
                        current_history = list(run_messages)
                        last_logged_requests = int(getattr(session.usage, "requests", 0) or 0)
                        last_logged_input_tokens = int(getattr(session.usage, "input_tokens", 0) or 0)
                        last_logged_output_tokens = int(getattr(session.usage, "output_tokens", 0) or 0)
                        last_logged_main_response_index = -1
                        continue
                    except (UsageLimitExceeded, _CompressionRequested) as exc:
                        _flush_captured_messages()
                        compression_reason = "limit"
                        compression_req_input: int | None = None
                        if isinstance(exc, _CompressionRequested):
                            if self.threshold_compression <= 0:
                                raise
                            compression_reason = exc.reason
                            compression_req_input = exc.req_input_tokens
                        else:
                            if self.threshold_compression <= 0 or not self._is_compression_limit_error(exc):
                                raise
                            compression_reason = "total_tokens_limit"
                        compression_rounds += 1
                        if compression_rounds > self.max_compressions_per_task:
                            raise RuntimeError(
                                "Exceeded maximum context compression rounds "
                                f"({self.max_compressions_per_task}). Last error: {exc}"
                            ) from exc
                        _task_record_echo(
                            f"{task_label} | context compression triggered "
                            f"(round={compression_rounds}, threshold={self.threshold_compression}, "
                            f"reason={compression_reason}, req_input_tokens={compression_req_input})"
                        )
                        compression_start_requests = int(getattr(session.usage, "requests", 0) or 0)
                        compression_start_input = int(getattr(session.usage, "input_tokens", 0) or 0)
                        _task_record_echo(
                            f"{task_label} | compression summary call start("
                            f"round={compression_rounds}, "
                            f"requests={compression_start_requests}, "
                            f"input_tokens={compression_start_input})"
                        )
                        summary, compression_request_trace = await self._summarize_for_compression(
                            session=session,
                            main_prompt=task.prompt,
                            message_history=run_messages,
                            round_index=compression_rounds,
                        )
                        if compression_request_trace is not None:
                            _record_request_io("main-compression", compression_request_trace)
                        compression_end_requests = int(getattr(session.usage, "requests", 0) or 0)
                        compression_end_input = int(getattr(session.usage, "input_tokens", 0) or 0)
                        _task_record_echo(
                            f"{task_label} | compression summary call end("
                            f"round={compression_rounds}, "
                            f"requests={compression_end_requests}, "
                            f"input_tokens={compression_end_input}, "
                            f"delta_requests={compression_end_requests - compression_start_requests}, "
                            f"delta_input_tokens={compression_end_input - compression_start_input})"
                        )
                        self._append_jsonl(
                            compression_path,
                            {
                                "compression_round": compression_rounds,
                                "threshold_tokens": self.threshold_compression,
                                "usage_after_limit_trip": self._usage_to_dict(session.usage),
                                "summary": summary,
                            },
                        )
                        self._append_jsonl(
                            events_path,
                            {
                                "event_index": event_cursor,
                                "event": {
                                    "type": "context_compression",
                                    "round": compression_rounds,
                                    "threshold_tokens": self.threshold_compression,
                                },
                            },
                        )
                        event_cursor += 1
                        current_prompt = self._resume_prompt(task.prompt, summary)
                        current_history = None
                        last_logged_requests = int(getattr(session.usage, "requests", 0) or 0)
                        last_logged_input_tokens = int(getattr(session.usage, "input_tokens", 0) or 0)
                        last_logged_output_tokens = int(getattr(session.usage, "output_tokens", 0) or 0)
                        last_logged_main_response_index = -1
                        _task_record_echo(
                            f"{task_label} | context compressed and resumed with prompt+summary "
                            f"(round={compression_rounds})"
                        )
                        continue

                    _flush_captured_messages()
                    output = str(run_result.output)
                    self._log(f"task done: {task_label}")
                    _task_record(f"task done: {task_label}")
                    return output
        except Exception as exc:
            error = str(exc)
            tb_text = traceback.format_exc()
            self._log(f"task failed: {task_label} ({exc})")
            _task_record(f"task failed: {task_label} ({exc})")
            raise
        finally:
            _flush_captured_messages()
            self._write_task_artifacts(
                task_dir=task_dir,
                task_label=task_label,
                task_logs=task_logs,
                session=session,
                output=output,
                error=error,
                traceback_text=tb_text,
                compression_rounds=compression_rounds,
            )
            close = getattr(client, "close", None)
            if callable(close):
                close()

    async def run_many(self, tasks: Sequence[DocqAgentTask]) -> list[str]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _worker(task: DocqAgentTask) -> str:
            async with semaphore:
                return await self.run_task(task)

        return await asyncio.gather(*(_worker(task) for task in tasks))

    def run_many_sync(self, tasks: Sequence[DocqAgentTask]) -> list[str]:
        return asyncio.run(self.run_many(tasks))


def main() -> int:
    args = parse_args()
    if args.num_agents < 1:
        raise SystemExit("--num-agents must be >= 1")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    if args.max_requests is not None and args.max_requests < 1:
        raise SystemExit("--max-requests must be >= 1")
    if args.threshold_compression < 0:
        raise SystemExit("--threshold-compression must be >= 0")
    if args.max_output_validation_recoveries < 0:
        raise SystemExit("--max-output-validation-recoveries must be >= 0")
    if not (0.0 <= args.temperature <= 2.0):
        raise SystemExit("--temperature must be in [0, 2]")
    if not (0.0 < args.top_p <= 1.0):
        raise SystemExit("--top-p must be in (0, 1]")
    if not (0.0 <= args.min_p <= 1.0):
        raise SystemExit("--min-p must be in [0, 1]")
    if args.repeat_penalty <= 0.0:
        raise SystemExit("--repeat-penalty must be > 0")
    if not args.openrouter_api_key:
        raise SystemExit("Missing OpenRouter API key. Use --openrouter-api-key or OPENROUTER_API_KEY.")
    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"--source does not exist: {source}")
    artifacts_dir: Path | None = None if args.no_artifacts else args.artifacts_dir.resolve()
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_settings = ModelSettings(
        temperature=args.temperature,
        top_p=args.top_p,
        extra_body={
            "min_p": args.min_p,
            "repetition_penalty": args.repeat_penalty,
        },
    )
    provider = OpenAIProvider(base_url=args.openrouter_base_url, api_key=args.openrouter_api_key)
    model = OpenAIChatModel(args.model, provider=provider, settings=model_settings)
    agent = build_docq_agent(
        model=model,
        include_semantic_tool=not args.disable_semantic_tool,
    )

    logger = None if args.quiet else make_console_logger("docq-batch")
    runner = ScalableDocqRunner(
        client_factory=lambda: PytanqueExtended(args.host, args.port),
        agent=agent,
        env=args.env,
        subagent_model=model,
        timeout=args.timeout,
        logger=logger,
        log_enabled=not args.quiet,
        log_prefix="docq-batch",
        semantic_base_url=args.semantic_base_url,
        semantic_route=args.semantic_route,
        semantic_api_key=args.semantic_api_key,
        max_tool_calls=args.max_tool_calls,
        max_requests=args.max_requests,
        max_concurrency=args.max_concurrency or args.num_agents,
        include_semantic_tool=not args.disable_semantic_tool,
        artifacts_dir=artifacts_dir,
        threshold_compression=args.threshold_compression,
        compression_model=model,
        max_output_validation_recoveries=args.max_output_validation_recoveries,
    )

    tasks = [
        DocqAgentTask(
            name=f"agent-{i + 1:02d}",
            prompt=f"[agent-{i + 1:02d}] {args.prompt}",
            source=source,
        )
        for i in range(args.num_agents)
    ]

    print(
        f"Running {args.num_agents} docq agents on {source} with model={args.model} "
        f"and concurrency={runner.max_concurrency} "
        f"(threshold-compression={args.threshold_compression}, min-p={args.min_p}, "
        f"repeat-penalty={args.repeat_penalty}, "
        f"max-output-validation-recoveries={args.max_output_validation_recoveries})",
        flush=True,
    )
    if artifacts_dir is not None:
        print(f"Artifacts directory: {artifacts_dir}", flush=True)
    outputs = runner.run_many_sync(tasks)

    print("\n=== Batch outputs ===", flush=True)
    for i, out in enumerate(outputs, start=1):
        print(f"\n[agent-{i:02d}]\n{out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
