import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from pytanque.protocol import Goal, State

pytest.importorskip("pydantic_ai")
from pydantic_ai.models.test import TestModel

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from agent.pydantic_agent import (  # noqa: E402
    PutnamAgentSession,
    PutnamAgentTask,
    PutnamBenchProblem,
    ScalablePutnamRunner,
    build_scalable_putnam_agent,
    find_proof_end_position,
    make_console_logger,
)

PUTNAM_ROOT = THIS_DIR / "putnam"


def _first_file(track: str) -> Path:
    files = sorted((PUTNAM_ROOT / track).glob("*.v"))
    assert files, f"No Putnam files found for track={track}"
    return files[0]


class FakeClient:
    def __init__(self, *, force_safeverify_ok: bool = True):
        self.force_safeverify_ok = force_safeverify_ok
        self.connected = False
        self.get_state_calls: list[tuple[str, int, int, float | None]] = []
        self.run_calls: list[tuple[int, str, float | None]] = []
        self.tmp_file_calls: list[dict[str, str | None]] = []
        self.safeverify_calls: list[dict[str, str]] = []
        self._next_state_id = 1

    def connect(self) -> None:
        self.connected = True

    def get_state_at_pos(
        self,
        path: str,
        line: int,
        character: int,
        timeout: float | None = None,
    ) -> State:
        self.get_state_calls.append((path, line, character, timeout))
        return State(st=0, proof_finished=False, feedback=[], hash=0)

    def run(self, state: State, cmd: str, timeout: float | None = None) -> State:
        self.run_calls.append((state.st, cmd, timeout))
        if cmd.startswith("fail"):
            raise RuntimeError("tactic failed")

        new_state = State(
            st=self._next_state_id,
            proof_finished=False,
            feedback=[(0, f"ran {cmd}")],
            hash=self._next_state_id,
        )
        self._next_state_id += 1
        return new_state

    def goals(self, state: State, timeout: float | None = None) -> list[Goal]:
        if state.st >= 2:
            return []
        return [Goal(info={}, hyps=[], ty=f"goal-{state.st}", pp=f"goal-{state.st}")]

    def tmp_file(self, content: str | None = None, root: str | None = None) -> str:
        root_path = Path(root) if root is not None else Path(tempfile.mkdtemp())
        root_path.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(dir=str(root_path), suffix=".v")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if content is not None:
                handle.write(content)
        self.tmp_file_calls.append({"content": content, "root": root, "path": path})
        return path

    def safeverify(
        self,
        source: str,
        target: str,
        root: str,
        axiom_whitelist=None,
        save_path=None,
        verbose: bool = False,
    ) -> dict:
        self.safeverify_calls.append(
            {
                "source": source,
                "target": target,
                "root": root,
            }
        )
        target_content = Path(target).read_text(encoding="utf-8")
        ok = self.force_safeverify_ok and ("Qed." in target_content) and ("Admitted." not in target_content)
        return {
            "ok": ok,
            "summary": {
                "num_obligations": 1,
                "passed": 1 if ok else 0,
                "failed": 0 if ok else 1,
                "global_failures": 0 if ok else 1,
            },
            "outcomes": [],
            "global_failures": [],
        }


def _problem(track: str) -> PutnamBenchProblem:
    return PutnamBenchProblem.from_file(_first_file(track), bench_root=PUTNAM_ROOT)


def test_find_proof_end_position_on_putnam_tracks():
    for track in ("coquelicot", "mathcomp"):
        path = _first_file(track)
        line, character = find_proof_end_position(path)
        line_text = path.read_text(encoding="utf-8").splitlines()[line]
        assert line_text[:character].endswith("Proof.")


def test_session_initial_state_and_branching_indexes():
    problem = _problem("coquelicot")
    client = FakeClient()
    session = PutnamAgentSession.from_problem(client, problem, timeout=12.0)

    assert client.connected is True
    assert session.available_state_indexes == [0]
    assert client.tmp_file_calls, "tmp_file was not called to stage source"
    staged_source = client.tmp_file_calls[0]["path"]
    assert staged_source is not None
    assert client.get_state_calls == [(staged_source, problem.proof_line, problem.proof_character, 12.0)]

    first = session.run_tac(0, "idtac.")
    second = session.run_tac(0, "idtac.")
    failed = session.run_tac(1, "fail.")

    assert first["ok"] is True and first["new_state_index"] == 1
    assert second["ok"] is True and second["new_state_index"] == 2
    assert failed["ok"] is False
    assert "tactic failed" in (failed.get("error") or "")
    assert session.available_state_indexes == [0, 1, 2]


def test_safe_verify_uses_tmp_file_inside_putnam_root():
    problem = _problem("mathcomp")
    client = FakeClient()
    session = PutnamAgentSession.from_problem(client, problem)

    result = session.safe_verify("Proof.\n  exact I.\nQed.")
    assert result["ok"] is True

    assert client.tmp_file_calls, "tmp_file was not called"
    tmp_call = client.tmp_file_calls[-1]
    assert tmp_call["root"] == str(session.proof_root)

    assert client.safeverify_calls, "safeverify was not called"
    sv_call = client.safeverify_calls[-1]
    assert sv_call["source"] == str(session.source_path)
    assert sv_call["root"] == str(session.proof_root)
    assert Path(sv_call["target"]).resolve().is_relative_to(session.proof_root.resolve())


def test_end_fails_when_safeverify_fails():
    problem = _problem("coquelicot")
    client = FakeClient(force_safeverify_ok=False)
    session = PutnamAgentSession.from_problem(client, problem)

    with pytest.raises(ValueError, match="SafeVerify failed"):
        session.end("Proof.\n  exact I.\nQed.")


def test_scalable_agent_tools_against_putnam_problem():
    problem = _problem("mathcomp")
    client = FakeClient()
    session = PutnamAgentSession.from_problem(client, problem)

    agent = build_scalable_putnam_agent(
        model=TestModel(call_tools=["list_states", "get_goals", "run_tac", "safe_verify", "end"])
    )
    result = agent.run_sync("Use the tools.", deps=session)
    payload = json.loads(result.output)

    assert "list_states" in payload
    assert "get_goals" in payload
    assert "run_tac" in payload
    assert "safe_verify" in payload
    assert "end" in payload


def test_scalable_runner_runs_many_tasks():
    agent = build_scalable_putnam_agent(model=TestModel(call_tools=["list_states"]))
    runner = ScalablePutnamRunner(client_factory=FakeClient, agent=agent, max_concurrency=2)

    tasks = [
        PutnamAgentTask(prompt="Task A", problem=_problem("coquelicot")),
        PutnamAgentTask(prompt="Task B", problem=_problem("mathcomp")),
    ]
    outputs = runner.run_many_sync(tasks)

    assert len(outputs) == 2
    assert all("list_states" in output for output in outputs)


def test_realtime_logging_feedback(capsys):
    problem = _problem("coquelicot")
    client = FakeClient()
    logger = make_console_logger("demo-agent")
    session = PutnamAgentSession.from_problem(client, problem, logger=logger)

    session.get_goals(0)
    session.run_tac(0, "idtac.")
    session.safe_verify("Proof.\n  exact I.\nQed.")
    out = capsys.readouterr().out

    assert "initialized at state 0" in out
    assert "get_goals(state_index=0)" in out
    assert "run_tac(state_index=0, tactic='idtac.')" in out
    assert "safe_verify finished: ok=True" in out
