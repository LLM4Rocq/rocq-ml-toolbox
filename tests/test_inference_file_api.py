from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.rocq_ml_toolbox.inference import server as inference_server


def _request_with_config(
    *,
    mode: inference_server.FsAccessMode,
    coq_lib_path: Path,
    read_allow_paths: tuple[Path, ...] = (),
):
    state = SimpleNamespace(
        file_access=inference_server.FileAccessConfig(
            mode=mode,
            coq_lib_path=coq_lib_path,
            read_allow_paths=read_allow_paths,
        ),
        toc_cache={},
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_access_libraries_prefers_precomputed_env_toc(tmp_path: Path):
    coq_lib = tmp_path / "coq-lib"
    coq_lib.mkdir()
    (coq_lib / "theories").mkdir()
    file_path = coq_lib / "theories" / "Demo.v"
    file_path.write_text("Lemma demo : True.\nProof.\nexact I.\nQed.\n", encoding="utf-8")

    env_toc = {
        "env": "coq-demo",
        "generated_at": "2026-01-01T00:00:00Z",
        "root_id": "dir:ROOT",
        "nodes": [
            {
                "id": "dir:ROOT",
                "type": "directory",
                "name": "ROOT",
                "path": "",
                "parent_id": None,
                "children_ids": ["file:theories/Demo.v"],
                "one_liner": "",
            },
            {
                "id": "file:theories/Demo.v",
                "type": "file",
                "name": "Demo.v",
                "path": "theories/Demo.v",
                "parent_id": "dir:ROOT",
                "children_ids": [],
                "one_liner": "",
            },
        ],
        "file_index": {"abc": "file:theories/Demo.v"},
    }
    (coq_lib / "coq-demo.toc.json").write_text(json.dumps(env_toc), encoding="utf-8")

    req = _request_with_config(mode=inference_server.FsAccessMode.READ_LIB_ONLY, coq_lib_path=coq_lib)
    result = inference_server.access_libraries(
        inference_server.AccessLibrariesBody(env="coq-demo"),
        req,
    )
    assert result["env"] == "coq-demo"
    file_nodes = [n for n in result["nodes"] if n.get("type") == "file"]
    assert file_nodes
    assert file_nodes[0].get("line_count") == 4


def test_access_libraries_fallback_scans_and_caches(tmp_path: Path):
    coq_lib = tmp_path / "coq-lib"
    (coq_lib / "theories" / "Init").mkdir(parents=True)
    (coq_lib / "user-contrib" / "MathComp").mkdir(parents=True)
    (coq_lib / "theories" / "Init" / "Logic.v").write_text("Lemma t : True.\nProof.\nexact I.\nQed.\n")
    (coq_lib / "user-contrib" / "MathComp" / "ssrbool.v").write_text("From Stdlib Require Import List.\n")

    req = _request_with_config(mode=inference_server.FsAccessMode.READ_LIB_ONLY, coq_lib_path=coq_lib)
    body = inference_server.AccessLibrariesBody(env="missing-env", use_cache=True)
    first = inference_server.access_libraries(body, req)
    second = inference_server.access_libraries(body, req)

    assert first["env"] == "missing-env"
    assert second == first
    assert req.app.state.toc_cache, "fallback payload should be cached in memory"
    assert any(n.get("path", "").endswith(".v") for n in first["nodes"] if n.get("type") == "file")


def test_read_file_chunking(tmp_path: Path):
    coq_lib = tmp_path / "coq-lib"
    coq_lib.mkdir()
    p = coq_lib / "a.v"
    p.write_text("abcdef", encoding="utf-8")
    req = _request_with_config(mode=inference_server.FsAccessMode.READ_LIB_ONLY, coq_lib_path=coq_lib)

    first = inference_server.read_file(
        inference_server.ReadFileBody(path=str(p), offset=0, max_chars=2),
        req,
    )
    assert first["content"] == "ab"
    assert first["next_offset"] == 2
    assert first["eof"] is False

    second = inference_server.read_file(
        inference_server.ReadFileBody(path=str(p), offset=first["next_offset"], max_chars=10),
        req,
    )
    assert second["content"] == "cdef"
    assert second["eof"] is True
    assert second["total_chars"] == 6


def test_read_file_allowed_by_fs_read_allow(tmp_path: Path):
    coq_lib = tmp_path / "coq-lib"
    coq_lib.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    p = outside / "a.v"
    p.write_text("demo", encoding="utf-8")
    req = _request_with_config(
        mode=inference_server.FsAccessMode.READ_LIB_ONLY,
        coq_lib_path=coq_lib,
        read_allow_paths=(outside,),
    )

    out = inference_server.read_file(
        inference_server.ReadFileBody(path=str(p), offset=0, max_chars=10),
        req,
    )
    assert out["content"] == "demo"


def test_write_file_policy_and_rw_mode(tmp_path: Path):
    coq_lib = tmp_path / "coq-lib"
    coq_lib.mkdir()
    target = tmp_path / "out.v"

    req_ro = _request_with_config(mode=inference_server.FsAccessMode.READ_LIB_ONLY, coq_lib_path=coq_lib)
    with pytest.raises(HTTPException):
        inference_server.write_file(
            inference_server.WriteFileBody(path=str(target), content="x", offset=0, truncate=True),
            req_ro,
        )

    req_rw = _request_with_config(mode=inference_server.FsAccessMode.RW_ANYWHERE, coq_lib_path=coq_lib)
    out = inference_server.write_file(
        inference_server.WriteFileBody(path=str(target), content="abc", offset=0, truncate=True),
        req_rw,
    )
    assert out["bytes_written"] == 3
    assert target.read_text(encoding="utf-8") == "abc"


def test_read_docstrings_present_and_missing(tmp_path: Path):
    coq_lib = tmp_path / "coq-lib"
    coq_lib.mkdir()
    source = coq_lib / "X.v"
    source.write_text("Lemma x : True.\nProof.\nexact I.\nQed.\n", encoding="utf-8")
    toc = [
        {
            "kind": "start_theorem_proof",
            "name": "x",
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 10}},
            "members": [],
            "data": {"uid": "x.uid", "content": "Lemma x : True."},
            "docstring": "A trivial truth.",
        }
    ]
    (coq_lib / "X.v.toc.json").write_text(json.dumps(toc), encoding="utf-8")

    req = _request_with_config(mode=inference_server.FsAccessMode.READ_LIB_ONLY, coq_lib_path=coq_lib)
    out = inference_server.read_docstrings(inference_server.ReadDocstringsBody(source=str(source)), req)
    assert len(out["docstrings"]) == 1
    assert out["docstrings"][0]["uid"] == "x.uid"

    missing = inference_server.read_docstrings(
        inference_server.ReadDocstringsBody(source=str(coq_lib / "Missing.v")),
        req,
    )
    assert missing["docstrings"] == []
