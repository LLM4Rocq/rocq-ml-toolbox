#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from agent import (  # noqa: E402
    PutnamAgentTask,
    PutnamBenchProblem,
    ScalablePutnamRunner,
    build_scalable_putnam_agent,
    make_console_logger,
)
from rocq_ml_toolbox.inference.client import PytanqueExtended  # noqa: E402

DEFAULT_PROBLEM = THIS_DIR / "putnam" / "mathcomp" / "putnam_1965_a5.v"
DEFAULT_BENCH_ROOT = THIS_DIR / "putnam"
DEFAULT_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_MIN_P = 0.01
DEFAULT_REPEAT_PENALTY = 1.0
DEFAULT_PROMPT = (
    "Prove the theorem. Start from state index 0. Use run_tac/get_goals/list_states as needed. "
    "Before finishing, call safe_verify(final_proof), then call end(final_proof). "
    "Return a short summary of the final successful proof."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a batch of K Putnam agents on putnam_1965_a5 (mathcomp) via OpenRouter.",
    )
    parser.add_argument("-k", "--num-agents", type=int, default=4, help="Number of agents to run.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent agents (default: k).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="rocq-ml-server host.")
    parser.add_argument("--port", type=int, default=5000, help="rocq-ml-server port.")
    parser.add_argument("--timeout", type=float, default=90.0, help="Per-tool timeout in seconds.")
    parser.add_argument("--problem", type=Path, default=DEFAULT_PROBLEM, help="Putnam .v file to solve.")
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT, help="Putnam benchmark root.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        help=f"OpenRouter model id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--openrouter-base-url",
        default=os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
        help=f"OpenRouter base URL (default: {DEFAULT_OPENROUTER_BASE_URL}).",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY).",
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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for every agent.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable real-time runner/session logs.",
    )
    return parser.parse_args()


def _build_model(
    base_url: str,
    api_key: str,
    model_name: str,
    *,
    temperature: float,
    top_p: float,
    min_p: float,
    repeat_penalty: float,
) -> OpenAIChatModel:
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    settings = ModelSettings(
        temperature=temperature,
        top_p=top_p,
        extra_body={
            "min_p": min_p,
            "repetition_penalty": repeat_penalty,
        },
    )
    return OpenAIChatModel(model_name, provider=provider, settings=settings)


def main() -> int:
    args = parse_args()
    if args.num_agents < 1:
        raise SystemExit("--num-agents must be >= 1")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
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

    problem = PutnamBenchProblem.from_file(args.problem, bench_root=args.bench_root)
    model = _build_model(
        args.openrouter_base_url,
        args.openrouter_api_key,
        args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
    )
    agent = build_scalable_putnam_agent(model=model)

    logger = None if args.quiet else make_console_logger("putnam-batch")
    runner = ScalablePutnamRunner(
        client_factory=lambda: PytanqueExtended(args.host, args.port),
        agent=agent,
        timeout=args.timeout,
        max_concurrency=args.max_concurrency or args.num_agents,
        logger=logger,
        log_enabled=not args.quiet,
        log_prefix="putnam-batch",
    )

    tasks = [
        PutnamAgentTask(
            name=f"agent-{i + 1:02d}",
            prompt=f"[agent-{i + 1:02d}] {args.prompt}",
            problem=problem,
        )
        for i in range(args.num_agents)
    ]

    print(
        f"Running {args.num_agents} agents on {problem.source_path} with model={args.model} "
        f"and concurrency={runner.max_concurrency} "
        f"(min-p={args.min_p}, repeat-penalty={args.repeat_penalty})",
        flush=True,
    )
    outputs = runner.run_many_sync(tasks)

    print("\n=== Batch outputs ===", flush=True)
    for i, out in enumerate(outputs, start=1):
        print(f"\n[agent-{i:02d}]\n{out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
