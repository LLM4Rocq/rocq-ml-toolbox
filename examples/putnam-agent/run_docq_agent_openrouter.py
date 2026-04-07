#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

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
DEFAULT_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PROMPT = (
    "Inspect the workspace and propose one useful intermediate lemma candidate, "
    "then attempt proving it."
)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--max-requests", type=int, default=200)
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

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)
            return
        if not self.log_enabled:
            return
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}][{self.log_prefix}] {message}", flush=True)

    async def run_task(self, task: DocqAgentTask) -> str:
        task_label = task.name or task.source.name
        self._log(f"task start: {task_label}")
        client = self.client_factory()
        session_logger: Callable[[str], None] | None = None
        if self.logger is not None:
            session_logger = lambda message, label=task_label: self.logger(f"{label} | {message}")
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
            include_semantic_tool=self.include_semantic_tool,
        )
        try:
            result = await self.agent.run(
                task.prompt,
                deps=session,
                usage=session.usage,
                usage_limits=session.usage_limits,
            )
            self._log(f"task done: {task_label}")
            return str(result.output)
        except Exception as exc:
            self._log(f"task failed: {task_label} ({exc})")
            raise
        finally:
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
    if not args.openrouter_api_key:
        raise SystemExit("Missing OpenRouter API key. Use --openrouter-api-key or OPENROUTER_API_KEY.")
    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"--source does not exist: {source}")

    provider = OpenAIProvider(base_url=args.openrouter_base_url, api_key=args.openrouter_api_key)
    model = OpenAIChatModel(args.model, provider=provider)
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
        f"and concurrency={runner.max_concurrency}",
        flush=True,
    )
    outputs = runner.run_many_sync(tasks)

    print("\n=== Batch outputs ===", flush=True)
    for i, out in enumerate(outputs, start=1):
        print(f"\n[agent-{i:02d}]\n{out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
