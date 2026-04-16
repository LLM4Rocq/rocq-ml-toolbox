from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from fastmcp import Client

DEFAULT_SERVER_URL = "http://127.0.0.1:8012/mcp"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except TypeError:
            return _jsonable(model_dump())
        except Exception:
            pass
    return repr(value)


def _tool_result_to_dict(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "is_error": bool(
            getattr(
                result,
                "is_error",
                getattr(result, "isError", False),
            )
        )
    }
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if structured is not None:
        payload["structured_content"] = _jsonable(structured)

    content = getattr(result, "content", None)
    if content is not None:
        payload["content"] = _jsonable(content)

    return payload


async def _call_tool(client: Client, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    result = await client.call_tool(name, arguments or {}, raise_on_error=False)
    payload = _tool_result_to_dict(result)
    print(f"\n== {name} ==")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


async def run(args: argparse.Namespace) -> None:
    client = Client(args.server_url)
    async with client:
        tools = await client.list_tools()
        tool_names = sorted(getattr(tool, "name", repr(tool)) for tool in tools)
        print(json.dumps({"tool_count": len(tool_names), "tools": tool_names}, indent=2, ensure_ascii=False))

        await _call_tool(client, "session_info")
        head_payload = await _call_tool(client, "current_head")

        latest_state_index = 0
        structured = head_payload.get("structured_content")
        if isinstance(structured, dict):
            latest_state_index = int(structured.get("latest_state_index", 0) or 0)

        await _call_tool(client, "get_goals", {"state_index": latest_state_index})
        await _call_tool(client, "completion_status")
        await _call_tool(client, "read_workspace_source")

        if args.tactic:
            await _call_tool(client, "run_tac_latest", {"tactic": args.tactic})
            await _call_tool(client, "current_head")
            await _call_tool(client, "get_goals", {"state_index": latest_state_index + 1})

        if args.semantic_query:
            await _call_tool(
                client,
                "semantic_doc_search",
                {
                    "query": args.semantic_query,
                    "k": args.semantic_k,
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny FastMCP test client for DocQ MCP server.")
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=(
            "MCP endpoint URL, for example: "
            "http://127.0.0.1:8012/mcp?source=examples/putnam-agent/putnam/mathcomp/putnam_1962_a6.v"
        ),
    )
    parser.add_argument(
        "--tactic",
        default="",
        help="Optional tactic to run via run_tac_latest (e.g. 'idtac.').",
    )
    parser.add_argument(
        "--semantic-query",
        default="",
        help="Optional semantic_doc_search query.",
    )
    parser.add_argument("--semantic-k", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
