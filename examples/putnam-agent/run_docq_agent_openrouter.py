#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

DEFAULT_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PROMPT = (
    "Inspect the workspace and propose one useful intermediate lemma candidate, "
    "then attempt proving it."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the docq agent on one Rocq source file.")
    parser.add_argument("--source", required=True, help="Path to source .v file to manipulate.")
    parser.add_argument("--env", required=True, help="Environment id used by /access_libraries.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tool-calls", type=int, default=120)
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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.openrouter_api_key:
        raise SystemExit("Missing OpenRouter API key. Use --openrouter-api-key or OPENROUTER_API_KEY.")

    client = PytanqueExtended(args.host, args.port)
    logger = make_console_logger("docq-agent")
    session = DocqAgentSession.from_source(
        client=client,
        source_path=args.source,
        env=args.env,
        timeout=args.timeout,
        connect=True,
        logger=logger,
        semantic_base_url=args.semantic_base_url,
        semantic_route=args.semantic_route,
        semantic_api_key=args.semantic_api_key,
        max_tool_calls=args.max_tool_calls,
    )
    provider = OpenAIProvider(base_url=args.openrouter_base_url, api_key=args.openrouter_api_key)
    model = OpenAIChatModel(args.model, provider=provider)
    agent = build_docq_agent(model=model)
    result = agent.run_sync(
        args.prompt,
        deps=session,
        usage=session.usage,
        usage_limits=session.usage_limits,
    )
    print(result.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
