from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest
from pytanque.protocol import Goal, State

pytest.importorskip("pydantic_ai")
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

THIS_DIR = Path(__file__).resolve().parent

from run_docq_agent_openrouter import ScalableDocqRunner  # noqa: E402
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


class FakeMissingDeclaredLemmaClient(FakeClient):
    def run(self, state: State, cmd: str, timeout: float | None = None) -> State:
        m = re.search(r"pose\s+proof\s+\(([^)]+)\)\.", cmd)
        if m:
            name = m.group(1).strip()
            raise RuntimeError(f"Coq: The variable {name} was not found in the current environment.")
        return super().run(state, cmd, timeout=timeout)


class FakeQedFailClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.qed_attempted = False

    def run(self, state: State, cmd: str, timeout: float | None = None) -> State:
        if cmd.strip() == "Qed.":
            self.qed_attempted = True
            raise RuntimeError("Final Qed check failed in fake backend.")
        return super().run(state, cmd, timeout=timeout)


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


class _FakeMsgUsage:
    def __init__(self, *, input_tokens: int, output_tokens: int, total_tokens: int | None = None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens

    def has_values(self) -> bool:
        return True


class _FakeEmptyUsage:
    def has_values(self) -> bool:
        return False


class _FakeMessage:
    def __init__(self, usage: Any = None):
        self.usage = usage


class _FakeRunCtx:
    def __init__(self, messages: list[Any]):
        self.messages = messages


def test_extract_last_response_usage_scans_backwards():
    run_ctx = _FakeRunCtx(
        [
            _FakeMessage(_FakeMsgUsage(input_tokens=10, output_tokens=5, total_tokens=15)),
            _FakeMessage(None),
            _FakeMessage(_FakeEmptyUsage()),
        ]
    )
    got = DocqAgentSession._extract_last_response_usage(run_ctx)
    assert got == (0, 10, 5, 15)


def test_runner_extract_last_response_usage_scans_backwards():
    run_ctx = _FakeRunCtx(
        [
            _FakeMessage(None),
            _FakeMessage(_FakeMsgUsage(input_tokens=7, output_tokens=3, total_tokens=None)),
            _FakeMessage(None),
        ]
    )
    got = ScalableDocqRunner._extract_last_response_usage(run_ctx)
    assert got == (1, 7, 3, 10)


def test_runner_detects_output_validation_retry_exhaustion():
    exc = RuntimeError("Exceeded maximum retries (2) for output validation")
    assert ScalableDocqRunner._is_output_validation_retry_exhaustion(exc) is True
    other = RuntimeError("some unrelated error")
    assert ScalableDocqRunner._is_output_validation_retry_exhaustion(other) is False


def test_runner_resume_prompt_after_output_validation_includes_recovery_steps():
    text = ScalableDocqRunner._resume_prompt_after_output_validation(
        main_prompt="Solve theorem.",
        error_text="Exceeded maximum retries (2) for output validation",
        status={"doc_id": 3, "latest_state_index": 2, "latest_goals_count": 1},
        attempt=1,
        max_attempts=4,
    )
    assert "Recovery Context:" in text
    assert "completion_status snapshot" in text
    assert "Do NOT output final text now." in text
    assert "get_goals" in text


def test_runner_output_validation_status_marker():
    marker = ScalableDocqRunner._output_validation_status_marker(
        {"doc_id": 3, "latest_state_index": 9, "latest_goals_count": 2}
    )
    assert marker == (3, 9, 2)
    marker2 = ScalableDocqRunner._output_validation_status_marker(
        {"head_doc_id": 7, "latest_state_index": 4, "latest_goals_count": 1}
    )
    assert marker2 == (7, 4, 1)
    assert ScalableDocqRunner._output_validation_status_marker({"error": "unavailable"}) is None
    assert ScalableDocqRunner._output_validation_status_marker({}) is None


def test_runner_attempt_limits_preserve_explicit_total_limit(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        max_requests=20,
        threshold_compression=4321,
    )
    session.usage_limits = UsageLimits(
        tool_calls_limit=20,
        request_limit=20,
        input_tokens_limit=111,
        output_tokens_limit=222,
        total_tokens_limit=333,
    )
    session.usage.input_tokens = 90_000
    session.usage.output_tokens = 9_999
    runner = ScalableDocqRunner(
        agent=build_docq_agent(model=TestModel()),
        client_factory=FakeClient,
        env="coq-demo",
        timeout=10.0,
        max_tool_calls=20,
        max_requests=20,
        threshold_compression=4321,
        subagent_model=TestModel(),
    )
    limits = runner._make_attempt_usage_limits(session)
    assert limits.total_tokens_limit == 333


def test_subagent_attempt_limits_preserve_explicit_total_limit(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        max_requests=20,
        threshold_compression=4321,
    )
    session.usage_limits = UsageLimits(
        tool_calls_limit=20,
        request_limit=20,
        input_tokens_limit=111,
        output_tokens_limit=222,
        total_tokens_limit=444,
    )
    session.usage.tool_calls = 13
    session.usage.requests = 14
    session.usage.input_tokens = 80_000
    session.usage.output_tokens = 8_888
    limits = session._subagent_attempt_usage_limits(
        remaining_tool_calls=7,
        remaining_requests=6,
    )
    assert limits.tool_calls_limit == 20
    assert limits.request_limit == 20
    assert limits.total_tokens_limit == 444


def test_document_manager_state_branching(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)

    a = manager.run_tac(state_index=0, tactic="idtac.")
    b = manager.run_tac(state_index=0, tactic="idtac.")
    c = manager.run_tac(state_index=1, tactic="idtac.")

    assert a["ok"] and a["new_state_index"] == 1
    assert b["ok"] and b["new_state_index"] == 2
    assert c["ok"] and c["new_state_index"] == 3
    assert "feedback" in a and isinstance(a["feedback"], list)
    assert "feedback_count" in a and isinstance(a["feedback_count"], int)

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


def test_document_manager_add_import_replays_latest_tactics(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    first = manager.run_tac(state_index=0, tactic="idtac.")
    assert first["ok"] is True

    added = manager.add_import(libname="MathComp", source="ssreflect")
    assert added["ok"] is True
    assert added["replayed_tactics"] == 1
    assert manager.list_states(doc_id=added["doc_id"]) == [0, 1]
    assert manager.sessions[added["doc_id"]].nodes[1].tactic == "idtac."


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


def test_run_tac_rejects_placeholder_tactic(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    out = manager.run_tac(state_index=0, tactic="all: admit.")
    assert out["ok"] is False
    assert "placeholder tactic" in out["error"]


def test_run_tac_rejects_shelving_tactic(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    out = manager.run_tac(state_index=0, tactic="all: shelve.")
    assert out["ok"] is False
    assert "shelving tactics" in out["error"]


def test_run_tac_missing_reference_includes_hint(tmp_path: Path):
    client = FakeMissingRefClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    out = manager.run_tac(state_index=0, tactic="lia.")
    assert out["ok"] is False
    assert "not found in the current environment" in out["error"]
    assert "add_import" in str(out.get("hint", ""))


def test_run_tac_repeated_same_failure_triggers_loop_guard(tmp_path: Path):
    client = FakeMissingRefClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    out1 = manager.run_tac(state_index=0, tactic="lia.")
    out2 = manager.run_tac(state_index=0, tactic="lia.")
    out3 = manager.run_tac(state_index=0, tactic="lia.")
    assert out1["ok"] is False
    assert out2["ok"] is False
    assert out3["ok"] is False
    assert out3.get("loop_guard_triggered") is True
    assert int(out3.get("failure_repeat_count", 0)) >= 3
    assert "Loop guard:" in str(out3.get("hint", ""))


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
    first = manager.run_tac(state_index=0, tactic="idtac.")
    assert first["ok"] is True
    added = manager.add_import(libname="MathComp", source="ssreflect")
    assert added["ok"] is True

    removed = manager.remove_import(libname="MathComp", source="ssreflect", doc_id=added["doc_id"])
    assert removed["ok"] is True
    assert removed["replayed_tactics"] == 1
    assert manager.list_states(doc_id=removed["doc_id"]) == [0, 1]
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


def test_document_manager_prepare_intermediate_lemma_exposes_probe_error_details(tmp_path: Path):
    client = FakeMissingRefClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    prep = session.prepare_intermediate_lemma(
        lemma_type="True",
        lemma_name="helper_probe_error",
    )
    assert prep["ok"] is False
    assert prep["phase"] == "prepare"
    assert prep["lemma_name"] == "helper_probe_error"
    assert "probe tactic" in str(prep.get("error", "")).lower()
    assert "helper_probe_error" in str(prep.get("error", ""))
    assert "lia was not found" in str(prep.get("probe_error", ""))
    assert "explore_toc" in str(prep.get("probe_hint", ""))
    assert prep.get("probe_tactic") == "pose proof (helper_probe_error)."
    assert prep.get("lemma_statement") == "Lemma helper_probe_error : True."


def test_document_manager_prepare_intermediate_lemma_probe_missing_name_has_specific_hint(tmp_path: Path):
    client = FakeMissingDeclaredLemmaClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    prep = session.prepare_intermediate_lemma(
        lemma_type="True",
        lemma_name="helper_probe_missing",
    )
    assert prep["ok"] is False
    assert prep["phase"] == "prepare"
    assert prep["lemma_name"] == "helper_probe_missing"
    assert prep.get("probe_tactic") == "pose proof (helper_probe_missing)."
    assert "Internal probe note" in str(prep.get("probe_hint", ""))
    assert "missing library import" in str(prep.get("probe_hint", ""))
    assert "add_import" not in str(prep.get("probe_hint", ""))
    assert prep.get("statement_probe_tactic") == "assert (True)."


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

    captured: dict[str, Any] = {}

    def _fake_ok(**kwargs: Any) -> dict[str, Any]:
        captured["prompt"] = kwargs.get("prompt")
        return {"ok": True, "proof_script": "exact I."}

    session.run_lemma_subagent = _fake_ok  # type: ignore[method-assign]
    ok = session.add_intermediate_lemma(lemma_type="True", subagent_message="Use hint gamma.")
    assert ok["ok"] is True
    assert ok["lemma_name"].startswith("intermediate_lemma_")
    assert session.doc_manager.head_doc_id == 1
    assert "Main-agent handoff:" in str(captured.get("prompt", ""))
    assert "Use hint gamma." in str(captured.get("prompt", ""))

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
    assert "Subagent successfully proved the intermediate lemma" in proved["main_agent_feedback"]
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


def test_docq_prepare_intermediate_records_subagent_message(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    prep = session.prepare_intermediate_lemma(
        lemma_type="True",
        lemma_name="helper_msg",
        subagent_message="Import Lia. Focus on the first unsolved goal.",
    )
    assert prep["ok"] is True
    assert prep["has_subagent_message"] is True
    pending = session.list_pending_intermediate_lemmas()
    assert len(pending) == 1
    assert pending[0]["lemma_name"] == "helper_msg"
    assert pending[0]["has_subagent_message"] is True


def test_docq_prove_intermediate_forwards_subagent_message(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    prep = session.prepare_intermediate_lemma(
        lemma_type="True",
        lemma_name="helper_msg",
        subagent_message="Use import hint alpha.",
    )
    assert prep["ok"] is True
    captured: dict[str, Any] = {}

    def _fake_run(**kwargs: Any) -> dict[str, Any]:
        captured["prompt"] = kwargs.get("prompt")
        return {"ok": True, "proof_script": "exact I."}

    session.run_lemma_subagent = _fake_run  # type: ignore[method-assign]
    proved = session.prove_intermediate_lemma(
        lemma_name="helper_msg",
        subagent_message="Use import hint beta.",
    )
    assert proved["ok"] is True
    assert "Main-agent handoff:" in str(captured.get("prompt", ""))
    assert "Use import hint beta." in str(captured.get("prompt", ""))


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


def test_docq_prove_intermediate_replays_main_states_after_registration(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    first = session.doc_manager.run_tac(state_index=0, tactic="idtac.")
    assert first["ok"] is True

    prep = session.prepare_intermediate_lemma(lemma_type="True", lemma_name="helper_replay")
    assert prep["ok"] is True

    session.run_lemma_subagent = lambda **kwargs: {"ok": True, "proof_script": "exact I."}  # type: ignore[method-assign]
    proved = session.prove_intermediate_lemma(lemma_name="helper_replay")
    assert proved["ok"] is True
    assert proved["replayed_tactics"] == 1
    assert proved["continue_state_index"] == 1
    assert proved["available_state_indexes"] == [0, 1]
    assert "helper_replay : True" in proved["main_agent_feedback"]
    latest_doc_id = int(proved["doc_id"])
    assert session.doc_manager.sessions[latest_doc_id].nodes[1].tactic == "idtac."
    workspace = session.doc_manager.show_workspace(doc_id=latest_doc_id)
    assert "Lemma helper_replay : True." in workspace["content"]


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


def test_docq_completion_status_reports_open_and_closed_states(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    open_status = session.completion_status()
    assert open_status["latest_proof_finished"] is False
    assert open_status["latest_goals_count"] == 1
    assert open_status["solved_state_indexes"] == []

    session.doc_manager.run_tac(state_index=0, tactic="idtac.")
    session.doc_manager.run_tac(state_index=1, tactic="idtac.")
    closed_status = session.completion_status()
    assert closed_status["latest_proof_finished"] is True
    assert closed_status["latest_goals_count"] == 0
    assert closed_status["latest_has_placeholder_tactic"] is False
    assert closed_status["solved_state_indexes"] == [2]


def test_document_manager_materialized_source_uses_current_proof_trace(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    manager.run_tac(state_index=0, tactic="idtac.")
    manager.run_tac(state_index=1, tactic="idtac.")
    materialized = manager.materialized_source()
    assert "Proof." in materialized
    assert "Qed." in materialized
    assert "Admitted." not in materialized
    assert materialized.count("idtac.") == 2


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


def test_docq_session_accepts_configured_compression_threshold(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        max_requests=100,
        threshold_compression=4321,
    )
    assert session.subagent_threshold_compression == 4321


def test_docq_output_validator_blocks_when_head_proof_not_finished(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    agent = build_docq_agent(
        model=TestModel(call_tools=[], custom_output_text="I am done."),
        retries=1,
    )
    with pytest.raises(Exception, match="output validation"):
        agent.run_sync("Stop now.", deps=session)


def test_docq_output_validator_runs_explicit_final_qed_check(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    session.doc_manager.run_tac(state_index=0, tactic="idtac.")
    session.doc_manager.run_tac(state_index=1, tactic="idtac.")
    baseline_calls = len(client.run_calls)
    agent = build_docq_agent(
        model=TestModel(call_tools=[], custom_output_text="I am done."),
        retries=1,
    )
    result = agent.run_sync("Stop now.", deps=session)
    assert result.output == "I am done."
    replay_calls = client.run_calls[baseline_calls:]
    assert any(cmd.strip() == "Qed." for _, cmd in replay_calls)


def test_docq_output_validator_blocks_when_explicit_final_qed_fails(tmp_path: Path):
    client = FakeQedFailClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    session.doc_manager.run_tac(state_index=0, tactic="idtac.")
    session.doc_manager.run_tac(state_index=1, tactic="idtac.")
    agent = build_docq_agent(
        model=TestModel(call_tools=[], custom_output_text="I am done."),
        retries=1,
    )
    with pytest.raises(Exception, match="output validation"):
        agent.run_sync("Stop now.", deps=session)
    assert client.qed_attempted is True


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
    assert payload["read_source_file"]["ok"] is False
    assert payload["read_source_file"]["source_kind"] == "workspace_doc"
    assert "read_workspace_source" in payload["read_source_file"]["hint"]
    assert "show_workspace" in payload
    assert "states" in payload["show_workspace"]
    assert "source_path" not in payload["show_workspace"]
    assert "read_workspace_source" in payload
    assert payload["read_workspace_source"]["doc_id"] == branch.doc_id
    assert "source_path" not in payload["read_workspace_source"]
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


def test_docq_subagent_read_source_file_rejects_workspace_path(tmp_path: Path):
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
    agent = build_docq_subagent(model=TestModel(call_tools=["read_source_file"]))
    result = agent.run_sync("Read workspace source file path.", deps=deps)
    payload = json.loads(result.output)

    assert payload["read_source_file"]["ok"] is False
    assert payload["read_source_file"]["source_kind"] == "workspace_doc"
    assert "read_workspace_source" in payload["read_source_file"]["hint"]


def test_docq_subagent_can_disable_semantic_tool():
    agent = build_docq_subagent(model=TestModel(), include_semantic_tool=False)
    names = _tool_names(agent)
    assert "semantic_doc_search" not in names
    assert "explore_toc" in names
    assert "run_tac" in names


def test_docq_subagent_start_log_includes_token_budgets(tmp_path: Path):
    client = FakeClient()
    logs: list[str] = []
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        max_requests=20,
        logger=logs.append,
        subagent_model=TestModel(),
        threshold_compression=4321,
    )
    session.usage_limits = UsageLimits(
        tool_calls_limit=20,
        request_limit=20,
        input_tokens_limit=123456,
        output_tokens_limit=234567,
        total_tokens_limit=345678,
    )
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    _ = session.run_lemma_subagent(sub_branch=branch, prompt="Try proving.")
    joined = "\n".join(logs)
    assert "subagent start(" in joined
    assert "remaining_input_tokens=123456" in joined
    assert "remaining_output_tokens=234567" in joined
    assert "remaining_total_tokens=345678" in joined
    assert "compression_threshold_tokens=4321" in joined
    assert "subagent model call(" in joined
    assert "req_input_tokens=" in joined
    assert "req_output_tokens=" in joined


def test_docq_subagent_trace_messages_are_emitted(tmp_path: Path):
    client = FakeClient()
    traced: list[tuple[str, Any]] = []
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        max_requests=20,
        subagent_model=TestModel(call_tools=["show_workspace"]),
        trace_message_callback=lambda source, message: traced.append((source, message)),
    )
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    _ = session.run_lemma_subagent(sub_branch=branch, prompt="Inspect workspace.")
    assert traced
    assert any(source == "subagent" for source, _ in traced)


def test_docq_subagent_trace_requests_are_emitted(tmp_path: Path):
    client = FakeClient()
    traced_requests: list[tuple[str, Any]] = []
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        max_requests=20,
        subagent_model=TestModel(call_tools=["show_workspace"]),
        trace_request_callback=lambda source, payload: traced_requests.append((source, payload)),
    )
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    _ = session.run_lemma_subagent(sub_branch=branch, prompt="Inspect workspace.")
    assert traced_requests
    sources = {source for source, _ in traced_requests}
    assert "subagent" in sources
    payload = next(payload for source, payload in traced_requests if source == "subagent")
    assert isinstance(payload, dict)
    assert "context_message_count" in payload
    assert "context_messages" in payload
    assert "request_message" in payload
    assert "response_message" in payload
    assert "req_input_tokens" in payload
    assert "req_output_tokens" in payload


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
                "completion_status",
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
    assert "completion_status" in payload
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
