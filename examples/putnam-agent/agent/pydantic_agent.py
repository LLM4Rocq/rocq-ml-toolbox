from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Sequence

from pydantic_ai import Agent, ModelRetry, RunContext

PROOF_TOKEN = "Proof."
ADMITTED_TOKEN = "Admitted."
DEFAULT_FINAL_PROOF = "Proof.\n  exact I.\nQed.\n"
PROOF_BLOCK_RE = re.compile(r"Proof\.(?:.|\n)*?Admitted\.")

SYSTEM_PROMPT = (
    "You are proving one Putnam theorem in Rocq/Coq.\n"
    "Rules:\n"
    "- Start from state index 0 (already initialized at end of `Proof.`).\n"
    "- Use `run_tac(state_index, tactic)` from any known state index.\n"
    "- Use `get_goals(state_index)` to inspect goals.\n"
    "- Before finishing, call `safe_verify(final_proof)`.\n"
    "- Call `end(final_proof)` only when the proof is ready.\n"
    "- `end` fails if SafeVerify does not pass."
)


def make_console_logger(prefix: str = "putnam-agent") -> Callable[[str], None]:
    lock = Lock()

    def _log(message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with lock:
            print(f"[{stamp}][{prefix}] {message}", flush=True)

    return _log


def find_proof_end_position(path: str | Path) -> tuple[int, int]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_idx, line in enumerate(lines):
        col = line.find(PROOF_TOKEN)
        if col != -1:
            return line_idx, col + len(PROOF_TOKEN)
    raise ValueError(f"No `{PROOF_TOKEN}` token found in: {path}")


def _resolve_bench_root(source_path: Path, bench_root: str | Path | None) -> Path:
    if bench_root is not None:
        root = Path(bench_root).resolve()
        source_path.resolve().relative_to(root)
        return root

    for parent in source_path.resolve().parents:
        if parent.name == "putnam":
            return parent
    return source_path.resolve().parent


def _goal_text(goal: Any) -> str:
    return getattr(goal, "pp", None) or getattr(goal, "ty", "")


def _feedback_text(state: Any) -> list[str]:
    feedback = getattr(state, "feedback", None) or []
    out: list[str] = []
    for item in feedback:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            out.append(f"[{item[0]}] {item[1]}")
        else:
            out.append(str(item))
    return out


def _normalize_final_proof(final_proof: str) -> str:
    proof = final_proof.strip()
    if PROOF_TOKEN in proof and "Qed." in proof:
        return proof if proof.endswith("\n") else proof + "\n"

    body = proof or "exact I."
    if not body.endswith("."):
        body = body + "."
    lines = [f"  {line}" if line.strip() else "" for line in body.splitlines()]
    return "Proof.\n" + "\n".join(lines) + "\nQed.\n"


def _render_safeverify_error(summary: dict[str, Any]) -> str:
    failed = summary.get("failed")
    global_failures = summary.get("global_failures")
    if failed is None and global_failures is None:
        return "SafeVerify failed."
    return f"SafeVerify failed (failed={failed}, global_failures={global_failures})."


@dataclass(frozen=True)
class PutnamBenchProblem:
    source_path: Path
    bench_root: Path
    proof_line: int
    proof_character: int

    @classmethod
    def from_file(
        cls,
        source_path: str | Path,
        *,
        bench_root: str | Path | None = None,
    ) -> "PutnamBenchProblem":
        source = Path(source_path).resolve()
        root = _resolve_bench_root(source, bench_root)
        line, character = find_proof_end_position(source)
        return cls(source_path=source, bench_root=root, proof_line=line, proof_character=character)


def iter_putnam_problems(
    bench_root: str | Path,
    *,
    track: str | None = None,
    limit: int | None = None,
) -> list[PutnamBenchProblem]:
    root = Path(bench_root).resolve()
    tracks = [root / track] if track else [root / "coquelicot", root / "mathcomp"]
    files: list[Path] = []
    for track_root in tracks:
        if track_root.exists():
            files.extend(sorted(track_root.glob("*.v")))
    if limit is not None:
        files = files[:limit]
    return [PutnamBenchProblem.from_file(path, bench_root=root) for path in files]


@dataclass
class PutnamAgentSession:
    client: Any
    problem: PutnamBenchProblem
    timeout: float = 60.0
    states: list[Any] = field(default_factory=list)
    logger: Callable[[str], None] | None = None
    log_enabled: bool = False
    log_prefix: str = "putnam-agent"

    @classmethod
    def from_problem(
        cls,
        client: Any,
        problem: PutnamBenchProblem,
        *,
        timeout: float = 60.0,
        connect: bool = True,
        logger: Callable[[str], None] | None = None,
        log_enabled: bool = False,
        log_prefix: str | None = None,
    ) -> "PutnamAgentSession":
        if connect:
            client.connect()
        initial_state = client.get_state_at_pos(
            str(problem.source_path),
            problem.proof_line,
            problem.proof_character,
            timeout=timeout,
        )
        session = cls(
            client=client,
            problem=problem,
            timeout=timeout,
            states=[initial_state],
            logger=logger,
            log_enabled=log_enabled,
            log_prefix=log_prefix or problem.source_path.name,
        )
        session._log(
            "initialized at state 0 "
            f"({problem.source_path}:{problem.proof_line}:{problem.proof_character})"
        )
        return session

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)
            return
        if not self.log_enabled:
            return
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}][{self.log_prefix}] {message}", flush=True)

    @property
    def available_state_indexes(self) -> list[int]:
        return list(range(len(self.states)))

    def _state(self, state_index: int) -> Any:
        if state_index < 0 or state_index >= len(self.states):
            raise ValueError(f"Unknown state index {state_index}; available={self.available_state_indexes}")
        return self.states[state_index]

    def get_goals(self, state_index: int = 0) -> dict[str, Any]:
        goals = [_goal_text(g) for g in self.client.goals(self._state(state_index), timeout=self.timeout)]
        self._log(f"get_goals(state_index={state_index}) -> {len(goals)} goals")
        return {
            "state_index": state_index,
            "goals": goals,
            "proof_finished": len(goals) == 0,
        }

    def run_tac(self, state_index: int, tactic: str) -> dict[str, Any]:
        state = self._state(state_index)
        self._log(f"run_tac(state_index={state_index}, tactic={tactic!r})")
        try:
            new_state = self.client.run(state, tactic, timeout=self.timeout)
        except Exception as exc:
            self._log(f"run_tac failed: {exc}")
            return {
                "ok": False,
                "source_state_index": state_index,
                "error": str(exc),
                "feedback": _feedback_text(state),
            }

        self.states.append(new_state)
        new_state_index = len(self.states) - 1
        goals = [_goal_text(g) for g in self.client.goals(new_state, timeout=self.timeout)]
        self._log(f"run_tac ok -> new_state_index={new_state_index}, goals={len(goals)}")
        return {
            "ok": True,
            "source_state_index": state_index,
            "new_state_index": new_state_index,
            "goals": goals,
            "feedback": _feedback_text(new_state),
        }

    def _candidate_content(self, final_proof: str) -> str:
        source = self.problem.source_path.read_text(encoding="utf-8")
        proof = _normalize_final_proof(final_proof)

        match = PROOF_BLOCK_RE.search(source)
        if match:
            return source[: match.start()] + proof + source[match.end() :]

        one_line = f"{PROOF_TOKEN} {ADMITTED_TOKEN}"
        if one_line in source:
            return source.replace(one_line, proof.strip(), 1)
        if ADMITTED_TOKEN in source:
            return source.replace(ADMITTED_TOKEN, proof.strip(), 1)

        raise ValueError(f"No `{ADMITTED_TOKEN}` token found in source file: {self.problem.source_path}")

    def safe_verify(self, final_proof: str) -> dict[str, Any]:
        self._log("safe_verify started")
        target_content = self._candidate_content(final_proof)
        target = self.client.tmp_file(content=target_content, root=str(self.problem.bench_root))
        report = self.client.safeverify(
            source=str(self.problem.source_path),
            target=target,
            root=str(self.problem.bench_root),
            verbose=True,
        )
        summary = report.get("summary", {}) if isinstance(report, dict) else {}
        ok = bool(report.get("ok", False)) if isinstance(report, dict) else False
        self._log(f"safe_verify finished: ok={ok}, target={target}")
        return {
            "ok": ok,
            "target_path": str(target),
            "summary": summary if isinstance(summary, dict) else {},
            "report": report if isinstance(report, dict) else {},
        }

    def end(self, final_proof: str) -> dict[str, Any]:
        self._log("end called")
        result = self.safe_verify(final_proof)
        if not result["ok"]:
            self._log("end rejected (SafeVerify failed)")
            raise ValueError(_render_safeverify_error(result.get("summary", {})))
        self._log("end accepted")
        return result


def build_scalable_putnam_agent(model: Any = None, *, retries: int = 2) -> Agent[PutnamAgentSession, str]:
    agent = Agent(
        model=model,
        deps_type=PutnamAgentSession,
        output_type=str,
        name="putnam-scalable-agent",
        system_prompt=SYSTEM_PROMPT,
        retries=retries,
    )

    @agent.tool
    def list_states(ctx: RunContext[PutnamAgentSession]) -> list[int]:
        """List available proof states that can be used as `state_index` in other tools."""
        return ctx.deps.available_state_indexes

    @agent.tool
    def get_goals(ctx: RunContext[PutnamAgentSession], state_index: int = 0) -> dict[str, Any]:
        """Return goals at `state_index`."""
        try:
            return ctx.deps.get_goals(state_index)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def run_tac(
        ctx: RunContext[PutnamAgentSession],
        state_index: int = 0,
        tactic: str = "idtac.",
    ) -> dict[str, Any]:
        """Run `tactic` from `state_index`; on success returns new state index and goals."""
        try:
            return ctx.deps.run_tac(state_index, tactic)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc

    @agent.tool
    def safe_verify(
        ctx: RunContext[PutnamAgentSession],
        final_proof: str = DEFAULT_FINAL_PROOF,
    ) -> dict[str, Any]:
        """Run SafeVerify against a candidate final proof script."""
        return ctx.deps.safe_verify(final_proof)

    @agent.tool
    def end(
        ctx: RunContext[PutnamAgentSession],
        final_proof: str = DEFAULT_FINAL_PROOF,
    ) -> str:
        """Finalize the proof; fails if SafeVerify is not successful."""
        try:
            result = ctx.deps.end(final_proof)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        return f"Accepted by SafeVerify with summary={result['summary']}"

    return agent


@dataclass(frozen=True)
class PutnamAgentTask:
    prompt: str
    problem: PutnamBenchProblem
    name: str = ""


@dataclass
class ScalablePutnamRunner:
    client_factory: Callable[[], Any]
    agent: Agent[PutnamAgentSession, str]
    timeout: float = 60.0
    max_concurrency: int = 8
    logger: Callable[[str], None] | None = None
    log_enabled: bool = False
    log_prefix: str = "putnam-runner"

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)
            return
        if not self.log_enabled:
            return
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}][{self.log_prefix}] {message}", flush=True)

    async def run_task(self, task: PutnamAgentTask) -> str:
        task_label = task.name or task.problem.source_path.name
        self._log(f"task start: {task_label}")
        client = self.client_factory()
        session_logger: Callable[[str], None] | None = None
        if self.logger is not None:
            session_logger = lambda message, label=task_label: self.logger(f"{label} | {message}")
        session = PutnamAgentSession.from_problem(
            client,
            task.problem,
            timeout=self.timeout,
            logger=session_logger,
            log_enabled=self.log_enabled,
            log_prefix=task_label,
        )
        try:
            result = await self.agent.run(task.prompt, deps=session)
            self._log(f"task done: {task_label}")
            return str(result.output)
        except Exception as exc:
            self._log(f"task failed: {task_label} ({exc})")
            raise
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    async def run_many(self, tasks: Sequence[PutnamAgentTask]) -> list[str]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _worker(task: PutnamAgentTask) -> str:
            async with semaphore:
                return await self.run_task(task)

        return await asyncio.gather(*(_worker(task) for task in tasks))

    def run_many_sync(self, tasks: Sequence[PutnamAgentTask]) -> list[str]:
        return asyncio.run(self.run_many(tasks))
