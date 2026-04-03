from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _line_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in handle:
            count += 1
    return count


class FsAccessMode(StrEnum):
    READ_LIB_ONLY = "read_lib_only"
    RW_ANYWHERE = "rw_anywhere"


@dataclass(frozen=True)
class FileAccessConfig:
    mode: FsAccessMode
    coq_lib_path: Path
    read_allow_paths: tuple[Path, ...] = ()


def resolve_coq_lib_path(override: str | None) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"--coq-lib-path does not exist: {path}")
        return path

    try:
        output = subprocess.check_output(["coqc", "-where"], text=True)
    except Exception as exc:
        raise RuntimeError("Unable to resolve Coq lib path via `coqc -where`.") from exc

    path = Path(output.strip()).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"`coqc -where` returned missing path: {path}")
    return path


def _assert_read_allowed(path: Path, cfg: FileAccessConfig) -> Path:
    resolved = path.expanduser().resolve()
    if cfg.mode == FsAccessMode.RW_ANYWHERE:
        return resolved
    if _is_within(resolved, cfg.coq_lib_path):
        return resolved
    for allowed in cfg.read_allow_paths:
        if _is_within(resolved, allowed):
            return resolved
    raise HTTPException(
        status_code=403,
        detail={
            "error": "Read denied by fs-access policy.",
            "mode": cfg.mode.value,
            "path": str(resolved),
            "coq_lib_path": str(cfg.coq_lib_path),
            "read_allow_paths": [str(p) for p in cfg.read_allow_paths],
            "hint": "Use --fs-read-allow <path> (repeatable) or --fs-access-mode rw_anywhere.",
        },
    )


def _assert_write_allowed(path: Path, cfg: FileAccessConfig) -> Path:
    resolved = path.expanduser().resolve()
    if cfg.mode == FsAccessMode.RW_ANYWHERE:
        return resolved
    raise HTTPException(
        status_code=403,
        detail={
            "error": "Write denied by fs-access policy.",
            "mode": cfg.mode.value,
            "path": str(resolved),
            "hint": "Use --fs-access-mode rw_anywhere to enable writes.",
        },
    )


def _file_candidates_from_node_path(coq_lib_path: Path, node_path: str) -> list[Path]:
    p = Path(node_path)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
        return candidates

    candidates.extend(
        [
            coq_lib_path / p,
            coq_lib_path / "user-contrib" / p,
            coq_lib_path / "theories" / p,
        ]
    )

    parts = list(PurePosixPath(node_path).parts)
    if parts and parts[0] == "Corelib" and len(parts) > 1:
        candidates.append(coq_lib_path / "theories" / PurePosixPath(*parts[1:]))

    dedup: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        dedup.append(c)
        seen.add(key)
    return dedup


def _enrich_file_nodes_with_line_count(payload: dict[str, Any], coq_lib_path: Path) -> dict[str, Any]:
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return payload

    enriched: list[dict[str, Any]] = []
    for raw_node in nodes:
        if not isinstance(raw_node, dict):
            enriched.append(raw_node)
            continue
        node = dict(raw_node)
        if node.get("type") == "file" and "line_count" not in node:
            node_path = node.get("path")
            if isinstance(node_path, str):
                for candidate in _file_candidates_from_node_path(coq_lib_path, node_path):
                    if candidate.exists() and candidate.is_file():
                        try:
                            node["line_count"] = _line_count(candidate)
                        except Exception:
                            pass
                        break
        enriched.append(node)
    payload["nodes"] = enriched
    return payload


def _fallback_toc_from_roots(
    *,
    env: str,
    coq_lib_path: Path,
    include_theories: bool = True,
    include_user_contrib: bool = True,
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {
        "dir:ROOT": {
            "id": "dir:ROOT",
            "type": "directory",
            "name": "ROOT",
            "path": "",
            "parent_id": None,
            "children_ids": [],
            "one_liner": "",
        }
    }
    file_index: dict[str, str] = {}

    def ensure_dir(rel_path: str) -> str:
        rel_path = rel_path.strip("/")
        if not rel_path:
            return "dir:ROOT"
        node_id = f"dir:{rel_path}"
        if node_id in nodes:
            return node_id

        parent_path = str(PurePosixPath(rel_path).parent)
        if parent_path == ".":
            parent_path = ""
        parent_id = ensure_dir(parent_path)
        nodes[node_id] = {
            "id": node_id,
            "type": "directory",
            "name": PurePosixPath(rel_path).name,
            "path": rel_path,
            "parent_id": parent_id,
            "children_ids": [],
            "one_liner": "",
        }
        if node_id not in nodes[parent_id]["children_ids"]:
            nodes[parent_id]["children_ids"].append(node_id)
        return node_id

    roots: list[tuple[str, Path]] = []
    if include_theories:
        roots.append(("theories", coq_lib_path / "theories"))
    if include_user_contrib:
        roots.append(("user-contrib", coq_lib_path / "user-contrib"))

    for root_label, root_path in roots:
        if not root_path.exists() or not root_path.is_dir():
            continue
        ensure_dir(root_label)
        for file_path in root_path.rglob("*.v"):
            rel = file_path.resolve().relative_to(coq_lib_path.resolve()).as_posix()
            rel_parent = str(PurePosixPath(rel).parent)
            parent_id = ensure_dir(rel_parent)
            file_node_id = f"file:{rel}"
            line_count: int | None = None
            try:
                line_count = _line_count(file_path)
            except Exception:
                line_count = None

            node = {
                "id": file_node_id,
                "type": "file",
                "name": file_path.name,
                "path": rel,
                "parent_id": parent_id,
                "children_ids": [],
                "one_liner": "",
                "line_count": line_count,
            }
            nodes[file_node_id] = node
            if file_node_id not in nodes[parent_id]["children_ids"]:
                nodes[parent_id]["children_ids"].append(file_node_id)
            entry_key = hashlib.sha1(rel.encode("utf-8")).hexdigest()
            file_index[entry_key] = file_node_id

    node_list = sorted(
        nodes.values(),
        key=lambda n: (str(n.get("type", "")), str(n.get("path", "")), str(n.get("name", ""))),
    )
    return {
        "env": env,
        "generated_at": _utc_now_iso(),
        "root_id": "dir:ROOT",
        "nodes": node_list,
        "file_index": file_index,
    }


class AccessLibrariesBody(BaseModel):
    env: str
    use_cache: bool = True
    include_theories: bool = True
    include_user_contrib: bool = True


class ReadFileBody(BaseModel):
    path: str
    offset: int = 0
    max_chars: int = 20000


class WriteFileBody(BaseModel):
    path: str
    content: str
    offset: int = 0
    truncate: bool = False


class ReadDocstringsBody(BaseModel):
    source: str


def _extract_docstring_entries(nodes: list[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def walk(entries: list[Any]) -> None:
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            docstring = raw.get("docstring")
            if isinstance(docstring, str) and docstring.strip():
                data = raw.get("data", {})
                if not isinstance(data, dict):
                    data = {}
                output.append(
                    {
                        "uid": data.get("uid"),
                        "name": raw.get("name"),
                        "kind": raw.get("kind"),
                        "range": raw.get("range"),
                        "docstring": docstring,
                        "content": data.get("content"),
                    }
                )
            members = raw.get("members")
            if isinstance(members, list):
                walk(members)

    walk(nodes)
    return output


router = APIRouter()


@router.post("/access_libraries")
def access_libraries(body: AccessLibrariesBody, request: FastAPIRequest):
    cfg: FileAccessConfig = request.app.state.file_access
    cache: dict[tuple[str, str, bool, bool], dict[str, Any]] = request.app.state.toc_cache
    key = (
        str(cfg.coq_lib_path),
        body.env,
        bool(body.include_theories),
        bool(body.include_user_contrib),
    )
    if body.use_cache and key in cache:
        return cache[key]

    env_toc_path = cfg.coq_lib_path / f"{body.env}.toc.json"
    payload: dict[str, Any]
    if env_toc_path.exists():
        _assert_read_allowed(env_toc_path, cfg)
        with env_toc_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise HTTPException(status_code=500, detail=f"Invalid env TOC format in {env_toc_path}.")
        payload = data
    else:
        payload = _fallback_toc_from_roots(
            env=body.env,
            coq_lib_path=cfg.coq_lib_path,
            include_theories=body.include_theories,
            include_user_contrib=body.include_user_contrib,
        )

    payload = _enrich_file_nodes_with_line_count(payload, cfg.coq_lib_path)
    cache[key] = payload
    return payload


@router.post("/read_file")
def read_file(body: ReadFileBody, request: FastAPIRequest):
    if body.offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if body.max_chars <= 0:
        raise HTTPException(status_code=400, detail="max_chars must be > 0")

    cfg: FileAccessConfig = request.app.state.file_access
    path = _assert_read_allowed(Path(body.path), cfg)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    total_chars = len(content)
    chunk = content[body.offset : body.offset + body.max_chars]
    next_offset = body.offset + len(chunk)
    return {
        "path": str(path),
        "content": chunk,
        "offset": body.offset,
        "next_offset": next_offset,
        "eof": next_offset >= total_chars,
        "total_chars": total_chars,
    }


@router.post("/write_file")
def write_file(body: WriteFileBody, request: FastAPIRequest):
    if body.offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    cfg: FileAccessConfig = request.app.state.file_access
    path = _assert_write_allowed(Path(body.path), cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    if body.truncate:
        if body.offset != 0:
            raise HTTPException(status_code=400, detail="offset must be 0 when truncate=true")
        with path.open("w", encoding="utf-8") as handle:
            handle.write(body.content)
    else:
        mode = "r+" if path.exists() else "w+"
        with path.open(mode, encoding="utf-8") as handle:
            handle.seek(body.offset)
            handle.write(body.content)

    return {
        "path": str(path),
        "bytes_written": len(body.content.encode("utf-8")),
        "size": path.stat().st_size,
    }


@router.post("/read_docstrings")
def read_docstrings(body: ReadDocstringsBody, request: FastAPIRequest):
    cfg: FileAccessConfig = request.app.state.file_access
    source_path = _assert_read_allowed(Path(body.source), cfg)
    toc_path = source_path if str(source_path).endswith(".toc.json") else Path(str(source_path) + ".toc.json")
    toc_path = _assert_read_allowed(toc_path, cfg)
    if not toc_path.exists() or not toc_path.is_file():
        return {"source": str(source_path), "toc_path": str(toc_path), "docstrings": []}

    try:
        payload = json.loads(toc_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse toc file: {toc_path}: {exc}") from exc

    if not isinstance(payload, list):
        return {"source": str(source_path), "toc_path": str(toc_path), "docstrings": []}
    return {"source": str(source_path), "toc_path": str(toc_path), "docstrings": _extract_docstring_entries(payload)}
