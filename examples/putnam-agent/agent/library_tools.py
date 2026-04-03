from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _line_numbered(lines: list[str], start_line: int = 1) -> str:
    width = max(3, len(str(start_line + len(lines))))
    return "\n".join(f"{idx + start_line:>{width}}: {line}" for idx, line in enumerate(lines))


@dataclass
class TocExplorer:
    client: Any
    env: str
    include_theories: bool = True
    include_user_contrib: bool = True
    use_cache: bool = True

    _payload: dict[str, Any] | None = None
    _nodes_by_id: dict[str, dict[str, Any]] | None = None

    def _load(self) -> None:
        if self._payload is not None and self._nodes_by_id is not None:
            return
        payload = self.client.access_libraries(
            self.env,
            use_cache=self.use_cache,
            include_theories=self.include_theories,
            include_user_contrib=self.include_user_contrib,
        )
        nodes = payload.get("nodes", [])
        by_id: dict[str, dict[str, Any]] = {}
        for node in nodes:
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                by_id[node["id"]] = node
        self._payload = payload
        self._nodes_by_id = by_id

    def explore(self, path: list[str] | None = None) -> dict[str, Any]:
        self._load()
        assert self._payload is not None
        assert self._nodes_by_id is not None

        root_id = self._payload.get("root_id", "dir:ROOT")
        if not isinstance(root_id, str) or root_id not in self._nodes_by_id:
            root_id = "dir:ROOT"
        current = self._nodes_by_id[root_id]
        consumed: list[str] = []
        for segment in path or []:
            children_ids = current.get("children_ids", [])
            matched: dict[str, Any] | None = None
            for child_id in children_ids:
                child = self._nodes_by_id.get(child_id)
                if child and child.get("name") == segment:
                    matched = child
                    break
            if matched is None:
                available = sorted(
                    str(self._nodes_by_id[cid].get("name"))
                    for cid in children_ids
                    if cid in self._nodes_by_id
                )
                return {
                    "ok": False,
                    "path": consumed,
                    "error": f"Unknown segment {segment!r}.",
                    "available": available,
                }
            current = matched
            consumed.append(segment)

        entries: list[dict[str, Any]] = []
        for child_id in current.get("children_ids", []):
            child = self._nodes_by_id.get(child_id)
            if not child:
                continue
            kind = child.get("type")
            child_path = str(child.get("path", ""))
            if kind == "directory":
                entries.append(
                    {
                        "name": child.get("name"),
                        "kind": "directory",
                        "path": child_path,
                    }
                )
            elif kind == "file" and child_path.endswith(".v"):
                entries.append(
                    {
                        "name": child.get("name"),
                        "kind": "file",
                        "path": child_path,
                        "line_count": child.get("line_count"),
                    }
                )

        entries.sort(key=lambda x: (0 if x.get("kind") == "directory" else 1, str(x.get("name", ""))))
        return {
            "ok": True,
            "env": self.env,
            "path": consumed,
            "entries": entries,
        }


def read_source_via_client(
    client: Any,
    path: str,
    *,
    line: int | None = None,
    before: int = 20,
    after: int = 20,
    chunk_size: int = 20000,
) -> dict[str, Any]:
    offset = 0
    chunks: list[str] = []
    while True:
        chunk = client.read_file(path, offset=offset, max_chars=chunk_size)
        text = chunk.get("content", "")
        if not isinstance(text, str):
            raise ValueError("Invalid /read_file response: content must be a string.")
        chunks.append(text)
        if chunk.get("eof", False):
            break
        next_offset = chunk.get("next_offset")
        if not isinstance(next_offset, int) or next_offset <= offset:
            raise ValueError("Invalid /read_file response: bad next_offset progression.")
        offset = next_offset
    full = "".join(chunks)
    lines = full.splitlines()
    if line is None:
        return {
            "mode": "full",
            "total_lines": len(lines),
            "content": _line_numbered(lines, start_line=1),
        }

    if line < 1:
        raise ValueError("line must be >= 1")
    start = max(1, line - max(0, before))
    end = min(len(lines), line + max(0, after))
    snippet_lines = lines[start - 1 : end]
    return {
        "mode": "around_line",
        "line": line,
        "start_line": start,
        "end_line": end,
        "total_lines": len(lines),
        "content": _line_numbered(snippet_lines, start_line=start),
    }
