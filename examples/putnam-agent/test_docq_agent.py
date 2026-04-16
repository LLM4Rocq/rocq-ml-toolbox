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

from run_docq_agent_openrouter import DocqAgentTask, ScalableDocqRunner  # noqa: E402
from agent.doc_manager import DocumentManager, _clip_goal_text, _normalize_import_parts  # noqa: E402
from agent.docq_agent import (  # noqa: E402
    ABORT_MIN_CHARS,
    ABORT_MIN_WORDS,
    DocqAgentSession,
    LemmaSubSession,
    _validate_subagent_abort_explanation,
    _adopt_branch_state,
    _create_nested_lemma_subsession,
    _rebuild_branch_with_import,
    _register_nested_lemma_into_branch,
    build_docq_agent,
    build_docq_subagent,
)
from agent.docstring_tools import SemanticDocSearchClient  # noqa: E402
from agent.library_tools import TocExplorer, read_source_via_client  # noqa: E402


class FakeClient:
    def __init__(self):
        self.connected = False
        self.run_calls: list[tuple[int, str]] = []
        self.state_calls: list[tuple[str, int, int]] = []
        self.tmp_file_calls: list[dict[str, Any]] = []
        self.safeverify_calls: list[dict[str, Any]] = []
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
        # Keep fake `proof_finished` coherent with `goals`: this fake backend
        # marks states >=2 as solved.
        proof_finished = self._next_state_id >= 2
        st = State(
            st=self._next_state_id,
            proof_finished=proof_finished,
            feedback=[],
            hash=self._next_state_id,
        )
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

    def safeverify(
        self,
        source: str,
        target: str,
        root: str,
        axiom_whitelist: list[str] | None = None,
        save_path: str | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        self.safeverify_calls.append(
            {
                "source": source,
                "target": target,
                "root": root,
                "axiom_whitelist": list(axiom_whitelist or []),
                "save_path": save_path,
                "verbose": verbose,
            }
        )
        return {
            "ok": True,
            "summary": {
                "num_obligations": 1,
                "passed": 1,
                "failed": 0,
                "global_failures": 0,
            },
            "global_failures": [],
            "outcomes": [],
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


class FakeImportProbeFailClient(FakeClient):
    def run(self, state: State, cmd: str, timeout: float | None = None) -> State:
        if cmd.strip() == "From Bad Require Import Missing.":
            raise RuntimeError("Coq: Cannot find a physical path bound to logical path Missing.")
        return super().run(state, cmd, timeout=timeout)


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


class FakeSafeVerifyFailClient(FakeClient):
    def safeverify(
        self,
        source: str,
        target: str,
        root: str,
        axiom_whitelist: list[str] | None = None,
        save_path: str | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        base = super().safeverify(
            source,
            target,
            root,
            axiom_whitelist=axiom_whitelist,
            save_path=save_path,
            verbose=verbose,
        )
        base["ok"] = False
        base["summary"] = {
            "num_obligations": 1,
            "passed": 0,
            "failed": 1,
            "global_failures": 0,
        }
        base["outcomes"] = [
            {
                "ok": False,
                "obligation": {"name": "target"},
                "failure_codes": ["disallowed_axioms"],
            }
        ]
        return base


class FakeZeroGoalsNotFinishedClient(FakeClient):
    """Simulate shelved/unfocused obligations: no focused goals but proof not closed."""

    def run(self, state: State, cmd: str, timeout: float | None = None) -> State:
        self.run_calls.append((state.st, cmd))
        if cmd.startswith("fail"):
            raise RuntimeError("tactic failed")
        st = State(st=self._next_state_id, proof_finished=False, feedback=[], hash=self._next_state_id)
        self._next_state_id += 1
        return st


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


def test_runner_detects_retryable_transport_error():
    exc = RuntimeError("Exception: Response payload is not completed")
    assert ScalableDocqRunner._is_retryable_transport_error(exc) is True
    exc_history = RuntimeError(
        "Cannot provide a new user prompt when the message history contains unprocessed tool calls."
    )
    assert ScalableDocqRunner._is_retryable_transport_error(exc_history) is True
    other = RuntimeError("Coq: The reference lia was not found in the current environment.")
    assert ScalableDocqRunner._is_retryable_transport_error(other) is False


def test_runner_detects_context_window_limit_error() -> None:
    exc = RuntimeError(
        "status_code: 400, model_name: moonshotai/kimi-k2.5, body: {'message': "
        "\"This endpoint's maximum context length is 262144 tokens. However, you requested about 292148 tokens.\"}"
    )
    assert ScalableDocqRunner._is_compression_limit_error(exc) is True
    other = RuntimeError("Coq: The reference lia was not found in the current environment.")
    assert ScalableDocqRunner._is_compression_limit_error(other) is False


def test_clip_goal_text_truncates_long_goals() -> None:
    long_goal = "G" * 20050
    clipped = _clip_goal_text(long_goal, max_chars=1000)
    assert clipped.startswith("G" * 1000)
    assert "goal truncated" in clipped
    assert "omitted" in clipped
    assert len(clipped) < len(long_goal)


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


def test_semantic_doc_search_client_sends_env_and_sorts_top_k(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "results": [
                    {
                        "score": 0.2,
                        "logical_path": "A.B",
                        "relative_path": "a/b.v",
                        "line": 10,
                        "character": 2,
                        "statement": "Lemma foo : True.",
                        "localization": {"start_line": 10, "start_character": 2},
                    },
                    {
                        "score": 0.9,
                        "logical_path": "C.D",
                        "relative_path": "c/d.v",
                        "line": 3,
                        "character": 14,
                        "statement": "Theorem bar : True.",
                        "localization": {"start_line": 3, "start_character": 14},
                    },
                    {
                        "score": "0.5",
                        "logical_path": "E.F",
                        "relative_path": "e/f.v",
                        "line": "7",
                        "character": "1",
                        "statement": "Fact baz : True.",
                        "localization": {"start_line": 7, "start_character": 1},
                    },
                ]
            }

    def _fake_post(url: str, *, json: dict[str, Any], headers: dict[str, Any], timeout: float):
        captured["url"] = url
        captured["json"] = dict(json)
        captured["headers"] = dict(headers)
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("agent.docstring_tools.requests.post", _fake_post)
    client = SemanticDocSearchClient(
        base_url="http://127.0.0.1:8010",
        route="/search",
        api_key="tok",
        env="coq-mathcomp",
        timeout=12.0,
    )
    out = client.search("group homomorphism", k=2)

    assert captured["url"] == "http://127.0.0.1:8010/search"
    assert captured["json"] == {"query": "group homomorphism", "env": "coq-mathcomp", "k": 2}
    assert captured["headers"]["Authorization"] == "Bearer tok"
    assert captured["timeout"] == 12.0

    assert len(out) == 2
    assert out[0]["score"] == 0.9
    assert out[1]["score"] == 0.5
    assert out[0]["logical_path"] == "C.D"
    assert out[0]["relative_path"] == "c/d.v"
    assert out[0]["statement"] == "Theorem bar : True."
    assert out[0]["line"] == 3
    assert out[0]["character"] == 14
    assert out[0]["localization"]["line"] == 3
    assert out[0]["localization"]["character"] == 14


def test_semantic_doc_search_client_falls_back_to_localization_line_character(
    monkeypatch: pytest.MonkeyPatch,
):
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "results": [
                    {
                        "score": 0.7,
                        "logical_path": "X.Y",
                        "relative_path": "x/y.v",
                        "statement": "Lemma local : True.",
                        "localization": {"start_line": 42, "start_character": 9},
                    }
                ]
            }

    def _fake_post(url: str, *, json: dict[str, Any], headers: dict[str, Any], timeout: float):
        return _Resp()

    monkeypatch.setattr("agent.docstring_tools.requests.post", _fake_post)
    client = SemanticDocSearchClient(base_url="http://127.0.0.1:8010")
    out = client.search("some query", k=1)
    assert len(out) == 1
    assert out[0]["line"] == 42
    assert out[0]["character"] == 9
    assert out[0]["localization"]["line"] == 42
    assert out[0]["localization"]["character"] == 9


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


def test_document_manager_run_tac_latest_uses_latest_state(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)

    out1 = manager.run_tac_latest(tactic="idtac.")
    out2 = manager.run_tac_latest(tactic="idtac.")

    assert out1["ok"] is True
    assert out2["ok"] is True
    assert out1["source_state_index"] == 0
    assert out2["source_state_index"] == 1
    assert out2["new_state_index"] == 2


def test_document_manager_run_tac_rejects_empty_tactic(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)

    out = manager.run_tac(state_index=0, tactic="")

    assert out["ok"] is False
    assert "empty tactic" in out["error"]


def test_document_manager_run_tac_rejects_proof_command(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)

    out = manager.run_tac(state_index=0, tactic="Proof.")

    assert out["ok"] is False
    assert "top-level vernac commands" in out["error"]


def test_document_manager_current_head_and_stale_state_warning(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)

    first = manager.run_tac(state_index=0, tactic="idtac.")
    assert first["ok"] is True
    stale = manager.run_tac(state_index=0, tactic="idtac.")
    assert stale["ok"] is True
    assert stale["stale_state"] is True
    assert "non-latest state" in str(stale.get("stale_state_warning", ""))

    head = manager.current_head()
    assert head["doc_id"] == manager.head_doc_id
    assert isinstance(head["latest_state_index"], int)
    assert head["recommended_next_action"] in {"run_tac_latest", "finish_candidate"}


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


def test_document_manager_add_import_accepts_require_import_statement(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    out = manager.add_import(
        libname="ignored",
        source="Require Import Coq.Logic.FunctionalExtensionality.",
    )
    assert out["ok"] is True
    workspace = manager.show_workspace()
    assert "From Coq.Logic Require Import FunctionalExtensionality." in workspace["content"]


def test_normalize_import_parts_accepts_require_import_statement():
    lib, src = _normalize_import_parts(
        "ignored",
        "Require Import Coq.Logic.FunctionalExtensionality.",
    )
    assert lib == "Coq.Logic"
    assert src == "FunctionalExtensionality"


def test_document_manager_add_import_rejects_invalid_import_via_probe(tmp_path: Path):
    client = FakeImportProbeFailClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)

    with pytest.raises(ValueError, match="Import probe rejected"):
        manager.add_import(libname="Bad", source="Missing")

    assert manager.head_doc_id == 0
    assert set(manager.nodes.keys()) == {0}


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


def test_run_tac_missing_ref_hint_mentions_semantic_when_enabled(tmp_path: Path):
    manager = DocumentManager(
        FakeMissingRefClient(),
        _source_file(tmp_path),
        timeout=5.0,
        include_semantic_tool=True,
    )
    out = manager.run_tac(state_index=0, tactic="idtac.")
    assert out["ok"] is False
    assert "semantic_doc_search" in str(out.get("hint", ""))


def test_run_tac_missing_ref_hint_omits_semantic_when_disabled(tmp_path: Path):
    manager = DocumentManager(
        FakeMissingRefClient(),
        _source_file(tmp_path),
        timeout=5.0,
        include_semantic_tool=False,
    )
    out = manager.run_tac(state_index=0, tactic="idtac.")
    assert out["ok"] is False
    assert "semantic_doc_search" not in str(out.get("hint", ""))
    assert "explore_toc" in str(out.get("hint", ""))


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


def test_validate_subagent_abort_explanation_rejects_short_message() -> None:
    ok, error = _validate_subagent_abort_explanation("not provable")
    assert ok is False
    assert "Abort rejected" in error
    assert str(ABORT_MIN_CHARS) in error
    assert str(ABORT_MIN_WORDS) in error


def test_validate_subagent_abort_explanation_accepts_detailed_message() -> None:
    detailed = (
        "Context: We are proving a helper lemma with one remaining goal and no direct matching hypotheses.\n"
        "Attempt 1: Tried structured induction on n with simplification and rewriting, but the induction "
        "hypothesis does not match the transformed target shape.\n"
        "Attempt 2: Tried introducing an intermediate equality and searching imports with explore_toc and "
        "semantic_doc_search, then applied several candidate lemmas; each candidate failed due to side conditions "
        "that cannot be discharged from available assumptions.\n"
        "Observed errors: Repeated tactic errors include no matching subterm for rewrite, inability to unify "
        "goal heads after simplification, and missing algebraic preconditions despite import attempts.\n"
        "Why unlikely solvable now: Two materially different proof strategies failed with concrete blockers, "
        "and the remaining path appears to require a stronger lemma not present in the current local context.\n"
        "Suggested next steps: Main agent should either reformulate the lemma statement with stronger premises, "
        "or first prove a prerequisite algebraic helper lemma and retry from the latest state after replay."
    )
    ok, error = _validate_subagent_abort_explanation(detailed)
    assert ok is True
    assert error == ""


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


def test_docq_prove_intermediate_manual_mode_when_no_subagent_model(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    prep = session.prepare_intermediate_lemma(lemma_type="True", lemma_name="helper_manual")
    assert prep["ok"] is True

    blocked = session.prove_intermediate_lemma(lemma_name="helper_manual")
    assert blocked["ok"] is False
    assert blocked["manual_mode_required"] is True
    assert blocked["pending"] is True

    head0 = session.pending_lemma_current_head(lemma_name="helper_manual")
    assert head0["ok"] is True
    assert head0["latest_goals_count"] == 1

    step1 = session.pending_lemma_run_tac(lemma_name="helper_manual", tactic="idtac.")
    assert step1["ok"] is True
    step2 = session.pending_lemma_run_tac(lemma_name="helper_manual", tactic="idtac.")
    assert step2["ok"] is True

    head1 = session.pending_lemma_current_head(lemma_name="helper_manual")
    assert head1["ok"] is True
    assert head1["latest_goals_count"] == 0

    proved = session.prove_intermediate_lemma(lemma_name="helper_manual")
    assert proved["ok"] is True
    assert proved["lemma_name"] == "helper_manual"
    assert session.list_pending_intermediate_lemmas() == []


def test_docq_add_intermediate_lemma_manual_handoff_when_no_subagent_model(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    out = session.add_intermediate_lemma(lemma_type="True", lemma_name="helper_manual_add")
    assert out["ok"] is True
    assert out["phase"] == "prepare_pending_manual"
    assert out["manual_mode_required"] is True
    assert out["pending"] is True
    assert out["next_action"] == "pending_lemma_run_tac"
    assert out["lemma_name"] == "helper_manual_add"
    assert any(x["lemma_name"] == "helper_manual_add" for x in session.list_pending_intermediate_lemmas())

    step1 = session.pending_lemma_run_tac(lemma_name="helper_manual_add", tactic="idtac.")
    assert step1["ok"] is True
    step2 = session.pending_lemma_run_tac(lemma_name="helper_manual_add", tactic="idtac.")
    assert step2["ok"] is True

    proved = session.prove_intermediate_lemma(lemma_name="helper_manual_add")
    assert proved["ok"] is True
    assert proved["lemma_name"] == "helper_manual_add"


def test_pending_lemma_run_tac_invalid_state_returns_error_payload(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    prep = session.prepare_intermediate_lemma(lemma_type="True", lemma_name="helper_invalid_state")
    assert prep["ok"] is True

    out = session.pending_lemma_run_tac(
        lemma_name="helper_invalid_state",
        tactic="idtac.",
        state_index=999,
    )
    assert out["ok"] is False
    assert out["lemma_name"] == "helper_invalid_state"
    assert out["requested_state_index"] == 999
    assert "available_state_indexes" in out
    assert "Unknown state index" in out["error"]
    assert "pending_lemma_current_head" in out["hint"]


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


def test_document_manager_materialized_source_does_not_fake_qed_when_open(tmp_path: Path):
    client = FakeClient()
    manager = DocumentManager(client, _source_file(tmp_path), timeout=5.0)
    manager.run_tac(state_index=0, tactic="idtac.")
    materialized = manager.materialized_source()
    assert "Proof." in materialized
    assert "proof unfinished" in materialized
    assert "Admitted." not in materialized
    tail = [line for line in materialized.splitlines() if line.strip()][-1]
    assert "no final Qed" in tail


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


def test_docq_output_validator_does_not_bypass_json_when_head_not_finished(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    agent = build_docq_agent(
        model=TestModel(call_tools=[], custom_output_text='{"ok": true}'),
        retries=1,
    )
    with pytest.raises(Exception, match="output validation"):
        agent.run_sync("Stop now.", deps=session)


def test_docq_output_validator_runs_explicit_final_qed_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakePassReport:
        def to_json(self) -> dict[str, Any]:
            return {
                "ok": True,
                "summary": {
                    "num_obligations": 1,
                    "passed": 1,
                    "failed": 0,
                    "global_failures": 0,
                },
                "global_failures": [],
                "outcomes": [],
            }

    def _fake_run_safeverify(*args: Any, **kwargs: Any) -> _FakePassReport:
        return _FakePassReport()

    import src.rocq_ml_toolbox.safeverify.core as sv_core

    monkeypatch.setattr(sv_core, "run_safeverify", _fake_run_safeverify)

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


def test_docq_output_validator_blocks_when_safeverify_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakeReport:
        def to_json(self) -> dict[str, Any]:
            return {
                "ok": False,
                "summary": {
                    "num_obligations": 1,
                    "passed": 0,
                    "failed": 1,
                    "global_failures": 0,
                },
                "global_failures": [],
                "outcomes": [
                    {
                        "ok": False,
                        "obligation": {"name": "target"},
                        "failure_codes": ["disallowed_axioms"],
                    }
                ],
            }

    def _fake_run_safeverify(*args: Any, **kwargs: Any) -> _FakeReport:
        return _FakeReport()

    import src.rocq_ml_toolbox.safeverify.core as sv_core

    monkeypatch.setattr(sv_core, "run_safeverify", _fake_run_safeverify)

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
    agent = build_docq_agent(
        model=TestModel(call_tools=[], custom_output_text="I am done."),
        retries=1,
    )
    with pytest.raises(Exception, match="output validation"):
        agent.run_sync("Stop now.", deps=session)


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
        env = "coq-demo"

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
                "read_workspace_source",
                "current_head",
                "list_states",
                "get_goals",
                "run_tac_latest",
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
    assert payload["semantic_doc_search"]["env"] == "coq-demo"
    assert payload["semantic_doc_search"]["results"][0]["logical_path"] == "Demo.target"
    assert "read_source_file" in payload
    assert payload["read_source_file"]["ok"] is False
    assert payload["read_source_file"]["source_kind"] == "workspace_doc"
    assert "read_workspace_source" in payload["read_source_file"]["hint"]
    assert "read_workspace_source" in payload
    assert payload["read_workspace_source"]["doc_id"] == branch.doc_id
    assert "source_path" not in payload["read_workspace_source"]
    assert "current_head" in payload
    assert "list_states" in payload
    assert "get_goals" in payload
    assert "run_tac_latest" in payload
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
    agent = build_docq_subagent(model=TestModel(call_tools=["run_tac", "require_import", "read_workspace_source"]))
    result = agent.run_sync("Run tactic then require import and read workspace source.", deps=deps)
    payload = json.loads(result.output)

    assert payload["run_tac"]["ok"] is True
    assert payload["require_import"]["ok"] is True
    assert payload["require_import"]["applied_to_workspace"] is True
    assert "From Stdlib Require Import List." in payload["read_workspace_source"]["content"]


def test_docq_subagent_run_tac_rejected_after_proof_is_finished(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    # Reach a solved latest state first.
    assert branch.run_tac(0, "idtac.")["ok"] is True
    assert branch.run_tac(1, "idtac.")["ok"] is True
    deps = LemmaSubSession(
        branch=branch,
        client=session.client,
        toc_explorer=session.toc_explorer,
    )
    agent = build_docq_subagent(model=TestModel(call_tools=["run_tac"]))
    result = agent.run_sync("Try an extra tactic after completion.", deps=deps)
    payload = json.loads(result.output)
    assert payload["run_tac"]["ok"] is False
    assert "already finished" in payload["run_tac"]["error"]
    assert "run_tac_latest" in payload["run_tac"]["error"]
    assert "current_head" in payload["run_tac"]["hint"]
    assert "abort" in payload["run_tac"]["hint"]


def test_docq_subagent_run_tac_latest_rejected_after_proof_is_finished(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    assert branch.run_tac(0, "idtac.")["ok"] is True
    assert branch.run_tac(1, "idtac.")["ok"] is True
    deps = LemmaSubSession(
        branch=branch,
        client=session.client,
        toc_explorer=session.toc_explorer,
    )
    agent = build_docq_subagent(model=TestModel(call_tools=["run_tac_latest"]))
    result = agent.run_sync("Try run_tac_latest after completion.", deps=deps)
    payload = json.loads(result.output)
    assert payload["run_tac_latest"]["ok"] is False
    assert "already finished" in payload["run_tac_latest"]["error"]
    assert "run_tac" in payload["run_tac_latest"]["error"]
    assert "current_head" in payload["run_tac_latest"]["hint"]
    assert "abort" in payload["run_tac_latest"]["hint"]


def test_docq_subagent_abort_rejected_when_proof_already_finished(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    assert branch.run_tac(0, "idtac.")["ok"] is True
    assert branch.run_tac(1, "idtac.")["ok"] is True
    deps = LemmaSubSession(
        branch=branch,
        client=session.client,
        toc_explorer=session.toc_explorer,
    )
    agent = build_docq_subagent(model=TestModel(call_tools=["abort"]))
    result = agent.run_sync("Abort after completion.", deps=deps)
    payload = json.loads(result.output)
    assert payload["abort"].startswith("ABORT_REJECTED:")
    assert "already finished" in payload["abort"]
    assert "return final answer" in payload["abort"]


def test_docq_subagent_zero_goals_without_proof_finished_is_not_finished(tmp_path: Path):
    client = FakeZeroGoalsNotFinishedClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    assert branch.run_tac(0, "idtac.")["ok"] is True
    assert branch.run_tac(1, "idtac.")["ok"] is True
    latest = branch.latest_state_index
    latest_status = branch.get_goals(latest)
    assert len(latest_status["goals"]) == 0
    assert latest_status["proof_finished"] is False

    deps = LemmaSubSession(
        branch=branch,
        client=session.client,
        toc_explorer=session.toc_explorer,
    )
    agent = build_docq_subagent(model=TestModel(call_tools=["current_head", "run_tac_latest"]))
    result = agent.run_sync("Check finish behavior when no focused goals but proof is not finished.", deps=deps)
    payload = json.loads(result.output)

    assert payload["current_head"]["latest_goals_count"] == 0
    assert payload["current_head"]["latest_proof_finished"] is False
    assert payload["current_head"]["recommended_next_action"] == "resolve_unfocused_obligations"
    assert payload["run_tac_latest"]["ok"] is True


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


def test_docq_subagent_exposes_nested_intermediate_lemma_tools():
    agent = build_docq_subagent(model=TestModel())
    names = _tool_names(agent)
    assert "prepare_intermediate_lemma" in names
    assert "prove_intermediate_lemma" in names
    assert "drop_pending_intermediate_lemma" in names
    assert "list_pending_intermediate_lemmas" in names
    assert "pending_lemma_current_head" in names
    assert "pending_lemma_list_states" in names
    assert "pending_lemma_get_goals" in names
    assert "pending_lemma_run_tac" in names


def test_docq_subagent_can_prepare_and_register_nested_intermediate_lemma(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=40,
    )
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    sub_branch = _create_nested_lemma_subsession(
        base_branch=branch,
        lemma_name="sublemma_1",
        lemma_type="True",
    )
    assert sub_branch.run_tac(0, "idtac.")["ok"] is True
    assert sub_branch.run_tac(1, "idtac.")["ok"] is True

    rebuilt = _register_nested_lemma_into_branch(
        base_branch=branch,
        lemma_name="sublemma_1",
        lemma_type="True",
        proof_script=sub_branch.proof_script(sub_branch.latest_state_index),
        required_imports=[],
    )
    _adopt_branch_state(branch, rebuilt)
    assert "Lemma sublemma_1 : True." in branch.source_content


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
        subagent_model=TestModel(call_tools=["read_workspace_source"]),
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
        subagent_model=TestModel(call_tools=["read_workspace_source"]),
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


def test_run_lemma_subagent_persists_progress_when_unfinished(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        max_requests=20,
        subagent_model=TestModel(call_tools=["run_tac"]),
    )
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    assert branch.latest_state_index == 0

    result = session.run_lemma_subagent(sub_branch=branch, prompt="Take one proof step.")

    assert result["ok"] is False
    assert "did not finish" in result["error"]
    assert branch.latest_state_index >= 1


def test_run_lemma_subagent_recovers_unprocessed_tool_calls_history_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        max_requests=20,
        subagent_model=TestModel(),
    )
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    seen_histories: list[Any] = []

    class _FakeSubagent:
        def __init__(self) -> None:
            self.calls = 0

        def run_sync(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            seen_histories.append(kwargs.get("message_history"))
            if self.calls == 1:
                raise RuntimeError(
                    "Cannot provide a new user prompt when the message history contains unprocessed tool calls."
                )
            deps = kwargs["deps"]
            deps.branch.run_tac(deps.branch.latest_state_index, "idtac.")
            deps.branch.run_tac(deps.branch.latest_state_index, "idtac.")

            class _Result:
                output = "ok"

            return _Result()

    fake = _FakeSubagent()
    monkeypatch.setattr("agent.docq_agent.build_docq_subagent", lambda **kwargs: fake)

    result = session.run_lemma_subagent(sub_branch=branch, prompt="Try proving.")

    assert result["ok"] is True
    assert fake.calls == 2
    assert seen_histories[0] is None
    assert seen_histories[1] is None


def test_docq_agent_has_branch_and_pending_tools(tmp_path: Path):
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        _source_file(tmp_path),
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
    )
    # Pre-solve the head proof so final output validation can pass after tool calls.
    session.doc_manager.run_tac(state_index=0, tactic="idtac.")
    session.doc_manager.run_tac(state_index=1, tactic="idtac.")
    # This test focuses on tool exposure/shape, not end-to-end proof validity.
    session.validate_final_qed = lambda *, doc_id=None, state_index=None: (True, None)  # type: ignore[assignment]
    agent = build_docq_agent(
        model=TestModel(
            call_tools=[
                "list_docs",
                "completion_status",
                "current_head",
                "checkout_doc",
                "list_states",
                "get_goals",
                "run_tac_latest",
                "run_tac",
                "add_import",
                "remove_import",
                "list_pending_intermediate_lemmas",
            ]
        )
    )
    result = agent.run_sync("Use branch and pending tools.", deps=session)
    payload = json.loads(result.output)
    assert "list_docs" in payload
    assert "completion_status" in payload
    assert "current_head" in payload
    assert "checkout_doc" in payload
    assert "list_pending_intermediate_lemmas" in payload
    assert "run_tac_latest" in payload
    names = _tool_names(agent)
    assert "prepare_intermediate_lemma" in names
    assert "drop_pending_intermediate_lemma" in names
    assert "pending_lemma_current_head" in names
    assert "pending_lemma_get_goals" in names
    assert "pending_lemma_run_tac" in names


def test_runner_passes_event_stream_handler_even_without_artifacts(tmp_path: Path):
    source = _source_file(tmp_path)

    class _DummyAgent:
        def __init__(self) -> None:
            self.last_event_stream_handler = None

        async def run(self, *args: Any, **kwargs: Any) -> Any:
            self.last_event_stream_handler = kwargs.get("event_stream_handler")

            class _Result:
                output = "ok"

            return _Result()

    dummy_agent = _DummyAgent()
    runner = ScalableDocqRunner(
        client_factory=FakeClient,
        agent=dummy_agent,
        env="coq-demo",
        timeout=5.0,
        max_tool_calls=20,
        max_requests=20,
        artifacts_dir=None,
        threshold_compression=1234,
    )
    outputs = runner.run_many_sync([DocqAgentTask(name="t", prompt="p", source=source)])
    assert outputs == ["ok"]
    assert dummy_agent.last_event_stream_handler is not None


def test_runner_request_io_compacts_payload_and_tracks_deltas(tmp_path: Path):
    source = _source_file(tmp_path)
    artifacts = tmp_path / "artifacts"

    class _UsageNow:
        def __init__(self, *, requests: int, input_tokens: int, output_tokens: int):
            self.requests = requests
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    async def _empty_stream():
        if False:
            yield None

    class _TracingAgent:
        async def run(self, *args: Any, **kwargs: Any) -> Any:
            handler = kwargs.get("event_stream_handler")
            assert handler is not None

            class _RunCtx:
                pass

            run_ctx = _RunCtx()
            run_ctx.run_step = 1
            run_ctx.usage = _UsageNow(requests=1, input_tokens=100, output_tokens=10)
            run_ctx.messages = [
                _FakeMessage(None),
                _FakeMessage(_FakeMsgUsage(input_tokens=100, output_tokens=10, total_tokens=110)),
            ]
            await handler(run_ctx, _empty_stream())

            run_ctx.run_step = 2
            run_ctx.usage = _UsageNow(requests=2, input_tokens=230, output_tokens=22)
            run_ctx.messages.extend(
                [
                    _FakeMessage(None),
                    _FakeMessage(_FakeMsgUsage(input_tokens=130, output_tokens=12, total_tokens=142)),
                ]
            )
            await handler(run_ctx, _empty_stream())

            class _Result:
                output = "ok"

            return _Result()

    runner = ScalableDocqRunner(
        client_factory=FakeClient,
        agent=_TracingAgent(),
        env="coq-demo",
        timeout=5.0,
        max_tool_calls=20,
        max_requests=20,
        artifacts_dir=artifacts,
        threshold_compression=999999,
    )
    outputs = runner.run_many_sync([DocqAgentTask(name="t", prompt="p", source=source)])
    assert outputs == ["ok"]

    task_dir = sorted(artifacts.iterdir())[0]
    request_lines = [
        json.loads(line)
        for line in (task_dir / "request_io.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(request_lines) == 2
    first = request_lines[0]["payload"]
    second = request_lines[1]["payload"]
    assert "context_messages" not in first
    assert "request_message" not in first
    assert "response_message" not in first
    assert first["context_delta"]["new_count"] == 1
    assert second["context_delta"]["new_count"] == 2
    assert second["delta"]["requests"] == 1
    assert second["delta"]["response_index"] == 2
    assert "omitted" in second
    assert "request_message" in second["omitted"]
    assert "response_message" in second["omitted"]


def test_runner_checkpoints_pending_lemma_subdocs(tmp_path: Path):
    source = _source_file(tmp_path)
    artifacts = tmp_path / "artifacts"

    class _PendingLemmaAgent:
        async def run(self, *args: Any, **kwargs: Any) -> Any:
            session = kwargs["deps"]
            prepared = session.prepare_intermediate_lemma(
                lemma_name="helper_pending",
                lemma_type="True",
            )
            assert prepared.get("ok") is True

            class _Result:
                output = "ok-pending"

            return _Result()

    runner = ScalableDocqRunner(
        client_factory=FakeClient,
        agent=_PendingLemmaAgent(),
        env="coq-demo",
        timeout=5.0,
        max_tool_calls=20,
        max_requests=20,
        artifacts_dir=artifacts,
    )
    outputs = runner.run_many_sync([DocqAgentTask(name="t", prompt="p", source=source)])
    assert outputs == ["ok-pending"]

    task_dir = sorted(artifacts.iterdir())[0]
    summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
    pending = summary["pending_lemmas"]
    assert pending
    assert pending[0]["lemma_name"] == "helper_pending"
    sub_doc_id = int(pending[0]["sub_doc_id"])

    pending_doc_meta = next(doc for doc in summary["docs"] if int(doc["doc_id"]) == sub_doc_id)
    assert pending_doc_meta.get("is_pending_lemma") is True
    assert (task_dir / pending_doc_meta["content_file"]).exists()
    assert (task_dir / "docs" / f"doc_{sub_doc_id}_materialized.v").exists()

    checkpoint_lines = [
        json.loads(line)
        for line in (task_dir / "doc_checkpoints.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert checkpoint_lines
    changed_docs = checkpoint_lines[-1]["changed_docs"]
    assert any(
        int(item.get("doc_id", -1)) == sub_doc_id and bool(item.get("is_pending_lemma", False))
        for item in changed_docs
    )
    checkpoint_rel = checkpoint_lines[-1]["checkpoint_dir"]
    assert checkpoint_rel is not None
    checkpoint_dir = task_dir / checkpoint_rel
    assert (checkpoint_dir / f"doc_{sub_doc_id}.v").exists()
    assert (checkpoint_dir / f"doc_{sub_doc_id}_materialized.v").exists()


def test_runner_retries_transient_transport_failure(tmp_path: Path):
    source = _source_file(tmp_path)

    class _RetryOnceAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Exception: Response payload is not completed")

            class _Result:
                output = "ok-after-retry"

            return _Result()

    retry_agent = _RetryOnceAgent()
    runner = ScalableDocqRunner(
        client_factory=FakeClient,
        agent=retry_agent,
        env="coq-demo",
        timeout=5.0,
        max_tool_calls=20,
        max_requests=20,
        artifacts_dir=None,
        max_transport_retries=2,
    )
    outputs = runner.run_many_sync([DocqAgentTask(name="t", prompt="p", source=source)])
    assert outputs == ["ok-after-retry"]
    assert retry_agent.calls == 2


def test_runner_compression_snapshot_uses_true_goals_count(tmp_path: Path):
    source = _source_file(tmp_path)
    client = FakeClient()
    session = DocqAgentSession.from_source(
        client,
        source,
        env="coq-demo",
        connect=False,
        max_tool_calls=20,
        max_requests=20,
    )

    class _FakeCompressor:
        def __init__(self) -> None:
            self.prompt = ""

        async def run(self, prompt: str, **kwargs: Any) -> Any:
            self.prompt = prompt

            class _Result:
                output = "summary"

            return _Result()

    fake = _FakeCompressor()
    runner = ScalableDocqRunner(
        client_factory=FakeClient,
        agent=build_docq_agent(model=TestModel()),
        env="coq-demo",
        timeout=5.0,
        max_tool_calls=20,
        max_requests=20,
    )
    runner._compression_agent = fake  # type: ignore[assignment]
    session.completion_status = lambda: {  # type: ignore[method-assign]
        "doc_id": 0,
        "head_doc_id": 0,
        "latest_state_index": 0,
        "latest_goals_count": 5,
    }
    session.doc_manager.get_goals = (  # type: ignore[method-assign]
        lambda **kwargs: {"goals": [f"goal-{i}" for i in range(5)]}
    )

    import asyncio

    out_summary, out_trace = asyncio.run(
        runner._summarize_for_compression(
            session=session,
            main_prompt="prove",
            message_history=[],
            round_index=1,
        )
    )
    assert out_summary == "summary"
    assert out_trace is None
    assert '"latest_goals_count": 5' in fake.prompt


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


def test_rebuild_branch_with_import_rejects_invalid_import_via_probe(tmp_path: Path):
    client = FakeImportProbeFailClient()
    source = _source_file(tmp_path)
    manager = DocumentManager(client, source, timeout=5.0)
    branch = manager.sessions[manager.head_doc_id]
    with pytest.raises(ValueError, match="Import probe rejected"):
        _rebuild_branch_with_import(branch, libname="Bad", source="Missing")


def test_adopt_branch_state_resets_loop_guard_memory(tmp_path: Path):
    source = _source_file(tmp_path)
    session = DocqAgentSession.from_source(FakeClient(), source, env="coq-demo", connect=False)
    branch = session.doc_manager.sessions[session.doc_manager.head_doc_id]
    rebuilt, _ = _rebuild_branch_with_import(branch, libname="Stdlib", source="List")
    branch._last_failure_signature = (0, "same failure")
    branch._last_failure_count = 3
    branch._last_stagnation_signature = (0, "idtac.")
    branch._last_stagnation_count = 8
    _adopt_branch_state(branch, rebuilt)
    assert branch._last_failure_signature is None
    assert branch._last_failure_count == 0
    assert branch._last_stagnation_signature is None
    assert branch._last_stagnation_count == 0
