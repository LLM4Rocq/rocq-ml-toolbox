from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


def _line_numbered(lines: list[str], start_line: int = 1) -> str:
    width = max(3, len(str(start_line + len(lines))))
    return "\n".join(f"{idx + start_line:>{width}}: {line}" for idx, line in enumerate(lines))


def _looks_like_rocq_source_path(path: str) -> bool:
    if path.endswith(".v"):
        return True
    name = PurePosixPath(path).name
    if not name:
        return False
    # Some env-level TOCs expose logical module paths without ".v" suffix.
    return "." not in name


def _dedup_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _normalize_explore_path(path: list[str] | None) -> list[str]:
    if not path:
        return []
    out: list[str] = []
    for segment in path:
        for part in str(segment).split("/"):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _candidate_source_paths(path: str) -> list[str]:
    raw = path.strip()
    if not raw:
        return []

    candidates: list[str] = [raw]
    pure = PurePosixPath(raw)
    if pure.suffix != ".v":
        candidates.append(str(PurePosixPath(f"{raw}.v")))

    if not pure.is_absolute():
        prefixed: list[str] = []
        for prefix in ("theories", "user-contrib"):
            pref = PurePosixPath(prefix) / pure
            prefixed.append(str(pref))
            if pref.suffix != ".v":
                prefixed.append(str(PurePosixPath(f"{pref}.v")))
        candidates.extend(prefixed)
    return _dedup_preserve_order(candidates)


def _read_path_mode_for(candidate: str) -> str:
    return "auto" if Path(candidate).is_absolute() else "coq_lib_relative"


def _read_file_chunk(client: Any, path: str, *, offset: int, max_chars: int) -> dict[str, Any]:
    path_mode = _read_path_mode_for(path)
    try:
        return client.read_file(path, offset=offset, max_chars=max_chars, path_mode=path_mode)
    except TypeError:
        # Backward compatibility for local fake clients or older transports.
        return client.read_file(path, offset=offset, max_chars=max_chars)


def _can_read_path(client: Any, path: str) -> bool:
    try:
        _ = _read_file_chunk(client, path, offset=0, max_chars=1)
        return True
    except Exception:
        return False


def _resolve_source_path(client: Any, path: str) -> str:
    candidates = _candidate_source_paths(path)
    for candidate in candidates:
        if _can_read_path(client, candidate):
            return candidate
    tried_preview = candidates[:8]
    suffix = "" if len(candidates) <= len(tried_preview) else f" (+{len(candidates) - len(tried_preview)} more)"
    hint = (
        "Use `explore_toc` first and pass a TOC-relative file path "
        "(for example `mathcomp/fingroup/perm.v`)."
    )
    if Path(path).is_absolute():
        hint += " Absolute filesystem paths are not supported by this tool."
    raise ValueError(
        "Unable to resolve source path for read_file. "
        f"requested={path!r} tried={tried_preview}{suffix}. {hint}"
    )


@dataclass
class TocExplorer:
    client: Any
    env: str | None = None
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
        root_entries = sorted(
            str(self._nodes_by_id[cid].get("name"))
            for cid in current.get("children_ids", [])
            if cid in self._nodes_by_id
        )
        requested_path = path or []
        normalized_path = _normalize_explore_path(path)
        consumed: list[str] = []
        for segment in normalized_path:
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
                    "requested_path": requested_path,
                    "path": consumed,
                    "error": f"Unknown segment {segment!r}.",
                    "available": available,
                    "root_entries": root_entries,
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
            elif kind == "file" and _looks_like_rocq_source_path(child_path):
                is_logical_path = not child_path.endswith(".v")
                entries.append(
                    {
                        "name": child.get("name"),
                        "kind": "file",
                        "path": child_path,
                        "is_logical_path": is_logical_path,
                        "suggested_read_path": f"{child_path}.v" if is_logical_path else child_path,
                        "line_count": child.get("line_count"),
                    }
                )

        entries.sort(key=lambda x: (0 if x.get("kind") == "directory" else 1, str(x.get("name", ""))))
        return {
            "ok": True,
            "env": self._payload.get("env", self.env),
            "coq_lib_path": self._payload.get("coq_lib_path"),
            "read_path_mode": self._payload.get("read_path_mode"),
            "requested_path": requested_path,
            "path": consumed,
            "root_entries": root_entries,
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
    resolved_path = _resolve_source_path(client, path)
    offset = 0
    chunks: list[str] = []
    while True:
        chunk = _read_file_chunk(client, resolved_path, offset=offset, max_chars=chunk_size)
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
            "requested_path": path,
            "resolved_path": resolved_path,
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
        "requested_path": path,
        "resolved_path": resolved_path,
        "line": line,
        "start_line": start,
        "end_line": end,
        "total_lines": len(lines),
        "content": _line_numbered(snippet_lines, start_line=start),
    }
