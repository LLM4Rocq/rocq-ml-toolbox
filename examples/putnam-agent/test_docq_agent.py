from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from pytanque.protocol import Goal, State

pytest.importorskip("pydantic_ai")

THIS_DIR = Path(__file__).resolve().parent

from agent.doc_manager import DocumentManager  # noqa: E402
from agent.docq_agent import DocqAgentSession  # noqa: E402
from agent.library_tools import TocExplorer, read_source_via_client  # noqa: E402


class FakeClient:
    def __init__(self):
        self.connected = False
        self.run_calls: list[tuple[int, str]] = []
        self.state_calls: list[tuple[str, int, int]] = []
        self._next_state_id = 1

    def connect(self) -> None:
        self.connected = True

    def tmp_file(self, content: str | None = None, root: str | None = None) -> str:
        target_root = Path(root or tempfile.gettempdir())
        target_root.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(dir=str(target_root), suffix=".v")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if content is not None:
                handle.write(content)
        return path

    def get_state_at_pos(self, path: str, line: int, character: int, timeout: float | None = None) -> State:
        self.state_calls.append((path, line, character))
        return State(st=0, proof_finished=False, feedback=[], hash=0)

    def run(self, state: State, cmd: str, timeout: float | None = None) -> State:
        self.run_calls.append((state.st, cmd))
        if cmd.startswith("fail"):
            raise RuntimeError("tactic failed")
        st = State(st=self._next_state_id, proof_finished=False, feedback=[], hash=self._next_state_id)
        self._next_state_id += 1
        return st

    def goals(self, state: State, timeout: float | None = None) -> list[Goal]:
        if state.st >= 2:
            return []
        return [Goal(info={}, hyps=[], ty=f"goal-{state.st}", pp=f"goal-{state.st}")]

    def access_libraries(
        self,
        env: str,
        *,
        use_cache: bool = True,
        include_theories: bool = True,
        include_user_contrib: bool = True,
    ) -> dict[str, Any]:
        return {
            "env": env,
            "root_id": "dir:ROOT",
            "nodes": [
                {
                    "id": "dir:ROOT",
                    "type": "directory",
                    "name": "ROOT",
                    "path": "",
                    "parent_id": None,
                    "children_ids": ["dir:theories", "file:theories/Demo.v"],
                },
                {
                    "id": "dir:theories",
                    "type": "directory",
                    "name": "theories",
                    "path": "theories",
                    "parent_id": "dir:ROOT",
                    "children_ids": [],
                },
                {
                    "id": "file:theories/Demo.v",
                    "type": "file",
                    "name": "Demo.v",
                    "path": "theories/Demo.v",
                    "parent_id": "dir:ROOT",
                    "children_ids": [],
                    "line_count": 42,
                },
            ],
            "file_index": {},
        }

    def read_file(self, path: str, *, offset: int = 0, max_chars: int = 20000) -> dict[str, Any]:
        content = Path(path).read_text(encoding="utf-8")
        chunk = content[offset : offset + max_chars]
        next_offset = offset + len(chunk)
        return {
            "path": path,
            "content": chunk,
            "offset": offset,
            "next_offset": next_offset,
            "eof": next_offset >= len(content),
            "total_chars": len(content),
        }


def _source_file(tmp_path: Path) -> Path:
    source = tmp_path / "demo.v"
    source.write_text(
        "From Stdlib Require Import List.\n"
        "\n"
        "Lemma helper : True.\n"
        "Proof.\n"
        "  exact I.\n"
        "Qed.\n"
        "\n"
        "Theorem target : True.\n"
        "Proof.\n"
        "Admitted.\n",
        encoding="utf-8",
    )
    return source


def test_document_manager_state_branching(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)

    a = manager.run_tac(state_index=0, tactic="idtac.")
    b = manager.run_tac(state_index=0, tactic="idtac.")
    c = manager.run_tac(state_index=1, tactic="idtac.")

    assert a["ok"] and a["new_state_index"] == 1
    assert b["ok"] and b["new_state_index"] == 2
    assert c["ok"] and c["new_state_index"] == 3

    verbose = manager.list_states_verbose()
    assert verbose[1]["parent_state_index"] == 0
    assert verbose[2]["parent_state_index"] == 0
    assert verbose[3]["parent_state_index"] == 1


def test_document_manager_auto_branch_on_mutation_from_older_doc(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)

    first = manager.add_import(libname="MathComp", source="ssreflect")
    assert first["doc_id"] == 1
    assert first["branched"] is False

    second = manager.add_import(libname="Stdlib", source="Bool", doc_id=0)
    assert second["doc_id"] == 2
    assert second["branched"] is True
    assert manager.head_doc_id == 2


def test_docq_add_intermediate_lemma_success_and_abort(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=10,
    )

    session.run_lemma_subagent = lambda **kwargs: {"ok": True, "proof_script": "exact I."}  # type: ignore[method-assign]
    ok = session.add_intermediate_lemma(lemma_type="True")
    assert ok["ok"] is True
    assert ok["lemma_name"].startswith("intermediate_lemma_")
    assert session.doc_manager.head_doc_id == 1

    session_abort = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=10,
    )
    session_abort.run_lemma_subagent = lambda **kwargs: {  # type: ignore[method-assign]
        "ok": False,
        "error": "not provable",
        "aborted": True,
    }
    failed = session_abort.add_intermediate_lemma(lemma_type="False")
    assert failed["ok"] is False
    assert failed["aborted"] is True
    assert session_abort.doc_manager.head_doc_id == 0


def test_docq_shared_budget_blocks_subagent(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=1,
    )
    session.usage.tool_calls = 1
    out = session.add_intermediate_lemma(lemma_type="True")
    assert out["ok"] is False
    assert "budget" in out["error"].lower()


def test_toc_explorer_and_read_source_trim(tmp_path: Path):
    client = FakeClient()
    source = _source_file(tmp_path)
    explorer = TocExplorer(client=client, env="coq-demo")
    root = explorer.explore([])
    assert root["ok"] is True
    assert any(entry["kind"] == "file" and entry["line_count"] == 42 for entry in root["entries"])

    full = read_source_via_client(client, str(source))
    assert full["mode"] == "full"
    assert "Theorem target" in full["content"]

    around = read_source_via_client(client, str(source), line=8, before=1, after=1)
    assert around["mode"] == "around_line"
    assert around["start_line"] == 7
