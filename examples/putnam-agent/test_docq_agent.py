from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from pytanque.protocol import Goal, State

pytest.importorskip("pydantic_ai")
from pydantic_ai.models.test import TestModel

THIS_DIR = Path(__file__).resolve().parent

from agent.doc_manager import DocumentManager  # noqa: E402
from agent.docq_agent import (  # noqa: E402
    DocqAgentSession,
    LemmaSubSession,
    _rebuild_branch_with_import,
    build_docq_agent,
    build_docq_subagent,
)
from agent.library_tools import TocExplorer, read_source_via_client  # noqa: E402


class FakeClient:
    def __init__(self):
        self.connected = False
        self.run_calls: list[tuple[int, str]] = []
        self.state_calls: list[tuple[str, int, int]] = []
        self.tmp_file_calls: list[dict[str, Any]] = []
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
        self.tmp_file_calls.append({"content": content, "root": root, "path": path})
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

    def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        max_chars: int = 20000,
        path_mode: str | None = None,
    ) -> dict[str, Any]:
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


class FakeLogicalPathClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.memory_files: dict[str, str] = {
            "mathcomp/boot/ssrbool.v": "Lemma eqxx : forall b : bool, b == b.\nProof.\nAdmitted.\n",
            "theories/Sets/Finite_sets.v": "Lemma finite_sets_demo : True.\nProof.\nexact I.\nQed.\n",
        }

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
                    "children_ids": ["dir:mathcomp"],
                },
                {
                    "id": "dir:mathcomp",
                    "type": "directory",
                    "name": "mathcomp",
                    "path": "mathcomp",
                    "parent_id": "dir:ROOT",
                    "children_ids": ["dir:mathcomp/boot"],
                },
                {
                    "id": "dir:mathcomp/boot",
                    "type": "directory",
                    "name": "boot",
                    "path": "mathcomp/boot",
                    "parent_id": "dir:mathcomp",
                    "children_ids": ["file:mathcomp/boot/ssrbool"],
                },
                {
                    "id": "file:mathcomp/boot/ssrbool",
                    "type": "file",
                    "name": "ssrbool",
                    "path": "mathcomp/boot/ssrbool",
                    "parent_id": "dir:mathcomp/boot",
                    "children_ids": [],
                    "line_count": 3,
                },
            ],
            "file_index": {},
        }

    def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        max_chars: int = 20000,
        path_mode: str | None = None,
    ) -> dict[str, Any]:
        if path not in self.memory_files:
            raise RuntimeError(f"not found: {path}")
        content = self.memory_files[path]
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


class FakeDenyTmpReadClient(FakeClient):
    def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        max_chars: int = 20000,
        path_mode: str | None = None,
    ) -> dict[str, Any]:
        if path.startswith("/tmp/tmp"):
            raise RuntimeError(f"read denied: {path}")
        return super().read_file(path, offset=offset, max_chars=max_chars, path_mode=path_mode)


class FakeMissingRefClient(FakeClient):
    def run(self, state: State, cmd: str, timeout: float | None = None) -> State:
        raise RuntimeError("Coq: The reference lia was not found in the current environment.")


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


def _tool_names(agent: Any) -> set[str]:
    toolset = getattr(agent, "_function_toolset", None)
    if toolset is not None:
        tools_dict = getattr(toolset, "tools", None)
        if isinstance(tools_dict, dict):
            return {str(name) for name in tools_dict.keys()}
    return set()


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


def test_document_manager_uses_server_tmp_root(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    manager.add_import(libname="MathComp", source="ssreflect")
    assert client.tmp_file_calls
    assert all(call.get("root") is None for call in client.tmp_file_calls)


def test_run_tac_rejects_top_level_vernac(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    out = manager.run_tac(state_index=0, tactic="Lemma rogue : True.")
    assert out["ok"] is False
    assert "top-level vernac" in out["error"]


def test_run_tac_rejects_markdown_wrapped_tactic(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    out = manager.run_tac(state_index=0, tactic="`intro n.`")
    assert out["ok"] is False
    assert "markdown-wrapped tactic" in out["error"]


def test_run_tac_missing_reference_includes_hint(tmp_path: Path):
    client = FakeMissingRefClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    out = manager.run_tac(state_index=0, tactic="lia.")
    assert out["ok"] is False
    assert "not found in the current environment" in out["error"]
    assert "add_import" in str(out.get("hint", ""))


def test_document_manager_checkout_and_list_docs(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    first = manager.add_import(libname="MathComp", source="ssreflect")
    second = manager.add_import(libname="Stdlib", source="Bool", doc_id=0)

    docs = manager.list_docs()
    assert {d["doc_id"] for d in docs} == {0, 1, 2}
    root = next(d for d in docs if d["doc_id"] == 0)
    assert sorted(root["children_doc_ids"]) == [1, 2]

    switched = manager.checkout_doc(doc_id=first["doc_id"])
    assert switched["head_doc_id"] == first["doc_id"]
    shown = manager.show_doc(doc_id=second["doc_id"])
    assert shown["doc_id"] == second["doc_id"]


def test_document_manager_remove_import(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    added = manager.add_import(libname="MathComp", source="ssreflect")
    assert added["ok"] is True

    removed = manager.remove_import(libname="MathComp", source="ssreflect")
    assert removed["ok"] is True
    workspace = manager.show_workspace()
    assert "From MathComp Require Import ssreflect." not in workspace["content"]

    with pytest.raises(ValueError, match="Import not found"):
        manager.remove_import(libname="MathComp", source="ssreflect")


def test_document_manager_add_import_accepts_full_statement_in_source(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    out = manager.add_import(
        libname="ignored",
        source="From mathcomp.fingroup Require Import perm.",
    )
    assert out["ok"] is True
    workspace = manager.show_workspace()
    assert "From mathcomp.fingroup Require Import perm." in workspace["content"]


def test_document_manager_prepare_intermediate_lemma_rejects_full_declaration(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    prep = session.prepare_intermediate_lemma(
        lemma_type="Lemma helper_full_decl : True.",
        lemma_name="helper_full_decl",
    )
    assert prep["ok"] is False
    assert "expected proposition only" in str(prep.get("error", ""))
    assert len(session.list_pending_intermediate_lemmas()) == 0


def test_document_manager_prepare_intermediate_lemma_rejects_markdown_backticks(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    before_calls = len(client.tmp_file_calls)
    prep = session.prepare_intermediate_lemma(
        lemma_type="`forall n : nat, n > 0 -> n >= 1`",
        lemma_name="helper_backtick",
    )
    assert prep["ok"] is False
    assert "markdown backticks" in str(prep.get("error", ""))
    assert len(session.list_pending_intermediate_lemmas()) == 0
    # No transient branch should be created on immediate format rejection.
    assert len(client.tmp_file_calls) == before_calls


def test_document_manager_prepare_intermediate_lemma_rejects_unicode_logic(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    before_calls = len(client.tmp_file_calls)
    prep = session.prepare_intermediate_lemma(
        lemma_type="∀ n : nat, n > 0 -> n >= 1",
        lemma_name="helper_unicode",
    )
    assert prep["ok"] is False
    assert "Unicode logical symbols" in str(prep.get("error", ""))
    assert len(session.list_pending_intermediate_lemmas()) == 0
    assert len(client.tmp_file_calls) == before_calls


def test_document_manager_doc_id_for_source_path(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    head_source = manager.sessions[manager.head_doc_id].source_path
    assert manager.doc_id_for_source_path(str(head_source)) == manager.head_doc_id
    assert manager.doc_id_for_source_path("/tmp/does-not-exist.v") is None


def test_document_manager_mutation_validator_rejects_and_rolls_back(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(
        client,
        _source_file(tmp_path),
        timeout=5.0,
        mutation_validator=lambda _doc_id, _label: (False, "reject"),
    )
    with pytest.raises(ValueError, match="fresh-session validation"):
        manager.add_import(libname="MathComp", source="ssreflect")
    assert manager.head_doc_id == 0
    assert sorted(manager.nodes.keys()) == [0]
    assert sorted(manager.sessions.keys()) == [0]


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
    assert failed["pending"] is True
    assert session_abort.list_pending_intermediate_lemmas()
    assert session_abort.doc_manager.head_doc_id == 0


def test_docq_prepare_prove_drop_intermediate_lemma(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    prep = session.prepare_intermediate_lemma(lemma_type="True", lemma_name="helper_x")
    assert prep["ok"] is True
    assert prep["lemma_name"] == "helper_x"
    assert len(session.list_pending_intermediate_lemmas()) == 1

    session.run_lemma_subagent = lambda **kwargs: {"ok": True, "proof_script": "exact I."}  # type: ignore[method-assign]
    proved = session.prove_intermediate_lemma(lemma_name="helper_x")
    assert proved["ok"] is True
    assert proved["lemma_name"] == "helper_x"
    assert session.list_pending_intermediate_lemmas() == []

    prep2 = session.prepare_intermediate_lemma(lemma_type="False", lemma_name="helper_bad")
    assert prep2["ok"] is True
    session.run_lemma_subagent = lambda **kwargs: {  # type: ignore[method-assign]
        "ok": False,
        "error": "not provable",
        "aborted": True,
    }
    failed = session.prove_intermediate_lemma(lemma_name="helper_bad")
    assert failed["ok"] is False
    assert failed["pending"] is True
    dropped = session.drop_pending_intermediate_lemma(lemma_name="helper_bad")
    assert dropped["ok"] is True


def test_docq_prove_intermediate_applies_required_imports(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    prep = session.prepare_intermediate_lemma(lemma_type="True", lemma_name="helper_with_import")
    assert prep["ok"] is True
    session.run_lemma_subagent = lambda **kwargs: {  # type: ignore[method-assign]
        "ok": True,
        "proof_script": "exact I.",
        "required_imports": [
            {"libname": "MathComp", "source": "ssreflect"},
            {"libname": "MathComp", "source": "ssreflect"},
        ],
    }
    proved = session.prove_intermediate_lemma(lemma_name="helper_with_import")
    assert proved["ok"] is True
    assert proved["applied_imports"][0]["libname"] == "MathComp"
    assert proved["applied_imports"][0]["source"] == "ssreflect"
    workspace = session.doc_manager.show_workspace()
    assert "From MathComp Require Import ssreflect." in workspace["content"]


def test_document_manager_remove_intermediate_lemma(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=10,
    )
    session.run_lemma_subagent = lambda **kwargs: {"ok": True, "proof_script": "exact I."}  # type: ignore[method-assign]
    added = session.add_intermediate_lemma(lemma_type="True")
    assert added["ok"] is True
    lemma_name = added["lemma_name"]

    removed = session.doc_manager.remove_intermediate_lemma(lemma_name=lemma_name)
    assert removed["ok"] is True
    workspace = session.doc_manager.show_workspace()
    assert lemma_name not in workspace["content"]

    with pytest.raises(ValueError, match="not found"):
        session.doc_manager.remove_intermediate_lemma(lemma_name=lemma_name)


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


def test_docq_request_budget_blocks_subagent(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        max_requests=1,
    )
    session.usage.requests = 1
    out = session.add_intermediate_lemma(lemma_type="True")
    assert out["ok"] is False
    assert "request budget" in out["error"].lower()


def test_docq_session_accepts_configured_request_limit(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        max_requests=321,
    )
    assert session.usage_limits.request_limit == 321


def test_docq_subagent_has_exploration_and_retrieval_tools(tmp_path: Path):
    client = FakeDenyTmpReadClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )

    class FakeSemantic:
        def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
            return [{"logical_path": "Demo.target", "docstring": query, "k": k}]

    session.semantic_search = FakeSemantic()
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    deps = LemmaSubSession(
        branch=branch,
        client=session.client,
        toc_explorer=session.toc_explorer,
        semantic_search=session.semantic_search,
    )
    agent = build_docq_subagent(
        model=TestModel(
            call_tools=[
                "explore_toc",
                "semantic_doc_search",
                "read_source_file",
                "show_workspace",
                "read_workspace_source",
                "list_states",
                "get_goals",
                "run_tac",
                "require_import",
                "abort",
            ]
        )
    )
    result = agent.run_sync("Use all tools.", deps=deps)
    payload = json.loads(result.output)

    assert "explore_toc" in payload
    assert payload["explore_toc"]["ok"] is True
    assert "semantic_doc_search" in payload
    assert payload["semantic_doc_search"]["results"][0]["logical_path"] == "Demo.target"
    assert "read_source_file" in payload
    assert "Theorem target" in payload["read_source_file"]["content"]
    assert "show_workspace" in payload
    assert "states" in payload["show_workspace"]
    assert "read_workspace_source" in payload
    assert payload["read_workspace_source"]["doc_id"] == branch.doc_id
    assert "list_states" in payload
    assert "get_goals" in payload
    assert "run_tac" in payload
    assert "require_import" in payload
    assert "abort" in payload


def test_docq_subagent_require_import_applies_to_workspace_and_replays(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    deps = LemmaSubSession(
        branch=branch,
        client=session.client,
        toc_explorer=session.toc_explorer,
    )
    agent = build_docq_subagent(model=TestModel(call_tools=["run_tac", "require_import", "show_workspace"]))
    result = agent.run_sync("Run tactic then require import and show workspace.", deps=deps)
    payload = json.loads(result.output)

    assert payload["run_tac"]["ok"] is True
    assert payload["require_import"]["ok"] is True
    assert payload["require_import"]["applied_to_workspace"] is True
    assert "From Stdlib Require Import List." in payload["show_workspace"]["content"]
    assert len(payload["show_workspace"]["states"]) >= 2


def test_docq_subagent_can_disable_semantic_tool():
    agent = build_docq_subagent(model=TestModel(), include_semantic_tool=False)
    names = _tool_names(agent)
    assert "semantic_doc_search" not in names
    assert "explore_toc" in names
    assert "run_tac" in names


def test_docq_agent_has_branch_and_pending_tools(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    agent = build_docq_agent(
        model=TestModel(
            call_tools=[
                "list_docs",
                "show_workspace",
                "checkout_doc",
                "list_states",
                "get_goals",
                "run_tac",
                "add_import",
                "remove_import",
                "prepare_intermediate_lemma",
                "list_pending_intermediate_lemmas",
                "drop_pending_intermediate_lemma",
            ]
        )
    )
    result = agent.run_sync("Use branch and pending tools.", deps=session)
    payload = json.loads(result.output)
    assert "list_docs" in payload
    assert "show_workspace" in payload
    assert "checkout_doc" in payload
    assert "prepare_intermediate_lemma" in payload
    assert "list_pending_intermediate_lemmas" in payload
    assert "drop_pending_intermediate_lemma" in payload


def test_docq_agent_can_disable_semantic_tool(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        include_semantic_tool=False,
    )
    agent = build_docq_agent(model=TestModel(), include_semantic_tool=False)
    names = _tool_names(agent)
    assert "semantic_doc_search" not in names
    assert "explore_toc" in names
    assert "add_import" in names
    assert session.include_semantic_tool is False


def test_toc_explorer_and_read_source_trim(tmp_path: Path):
    client = FakeClient()
    source = _source_file(tmp_path)
    explorer = TocExplorer(client=client, env="coq-demo")
    root = explorer.explore([])
    assert root["ok"] is True
    assert sorted(root["root_entries"]) == ["Demo.v", "theories"]
    assert any(entry["kind"] == "file" and entry["line_count"] == 42 for entry in root["entries"])

    full = read_source_via_client(client, str(source))
    assert full["mode"] == "full"
    assert "Theorem target" in full["content"]

    around = read_source_via_client(client, str(source), line=8, before=1, after=1)
    assert around["mode"] == "around_line"
    assert around["start_line"] == 7


def test_toc_explorer_handles_logical_source_paths():
    client = FakeLogicalPathClient()
    explorer = TocExplorer(client=client, env="coq-mathcomp")
    out = explorer.explore(["mathcomp", "boot"])
    assert out["ok"] is True
    assert len(out["entries"]) == 1
    entry = out["entries"][0]
    assert entry["kind"] == "file"
    assert entry["path"] == "mathcomp/boot/ssrbool"
    assert entry["is_logical_path"] is True
    assert entry["suggested_read_path"] == "mathcomp/boot/ssrbool.v"


def test_toc_explorer_accepts_slash_joined_segments():
    client = FakeLogicalPathClient()
    explorer = TocExplorer(client=client, env="coq-mathcomp")
    out = explorer.explore(["mathcomp/boot"])
    assert out["ok"] is True
    assert out["path"] == ["mathcomp", "boot"]
    assert out["requested_path"] == ["mathcomp/boot"]


def test_read_source_via_client_resolves_missing_v_suffix():
    client = FakeLogicalPathClient()
    out = read_source_via_client(client, "mathcomp/boot/ssrbool")
    assert out["mode"] == "full"
    assert out["requested_path"] == "mathcomp/boot/ssrbool"
    assert out["resolved_path"] == "mathcomp/boot/ssrbool.v"
    assert "eqxx" in out["content"]


def test_read_source_via_client_rejects_unknown_non_toc_path():
    client = FakeLogicalPathClient()
    with pytest.raises(ValueError, match="Unable to resolve source path") as exc:
        read_source_via_client(client, "/_std_lib/Sets/Finite_sets.v")
    msg = str(exc.value)
    assert "explore_toc" in msg
    assert "Absolute filesystem paths are not supported" in msg


def test_rebuild_branch_with_import_replays_existing_tactics(tmp_path: Path):
    client = FakeClient()
    source = _source_file(tmp_path)
    manager = DocumentManager(client, source, timeout=5.0)
    branch = manager.sessions[manager.head_doc_id]
    first = branch.run_tac(0, "idtac.")
    assert first["ok"] is True
    rebuilt, info = _rebuild_branch_with_import(branch, libname="MathComp", source="ssreflect")
    assert info["added"] is True
    assert info["replayed_tactics"] == 1
    assert "From MathComp Require Import ssreflect." in rebuilt.source_content
    assert rebuilt.latest_state_index == 1
