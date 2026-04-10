from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROOF_TOKEN = "Proof."
END_PROOF_TOKENS = ("Qed.", "Admitted.", "Defined.", "Abort.")
TARGET_START_RE = re.compile(r"^\s*(Theorem|Lemma|Fact|Proposition|Corollary)\b")
IMPORT_RE = re.compile(r"^\s*(From\s+\S+\s+Require\s+(Import|Export)|Require\s+Import|Import|Export)\b")
IDENT_RE = re.compile(r"^[A-Za-z0-9_.]+$")
FROM_REQUIRE_IMPORT_RE = re.compile(
    r"^\s*From\s+([A-Za-z0-9_.]+)\s+Require\s+Import\s+([A-Za-z0-9_.]+)\s*\.\s*$"
)
REQUIRE_IMPORT_RE = re.compile(
    r"^\s*Require\s+Import\s+([A-Za-z0-9_.]+)\s*\.?\s*$"
)
LEMMA_DECL_PREFIX_RE = re.compile(r"^\s*(Lemma|Theorem|Fact|Proposition|Corollary)\b")
LEMMA_PROOF_TOKEN_RE = re.compile(r"\b(Proof|Qed|Admitted|Defined|Abort)\b")
FORBIDDEN_TACTIC_PREFIX_RE = re.compile(
    r"^\s*(Lemma|Theorem|Fact|Proposition|Corollary|From|Require|Import|Export|Section|End|Module|"
    r"Definition|Fixpoint|CoFixpoint|Record|Inductive|CoInductive|Class|Instance|Notation|Ltac|Axiom|"
    r"Hypothesis|Variable|Context|Proof|Qed|Admitted|Defined|Abort)\b"
)
PLACEHOLDER_TACTIC_RE = re.compile(r"\b(admit|admitted)\b", re.IGNORECASE)
SHELVING_TACTIC_RE = re.compile(r"\b(shelve_unifiable|unshelve|shelve)\b", re.IGNORECASE)
MutationValidator = Callable[[int, str], tuple[bool, str | None]]
StateEventLogger = Callable[[dict[str, Any]], None]
UNICODE_LOGIC_TOKENS = ("∀", "∃", "→", "↔", "⇒", "⇔", "∧", "∨", "≤", "≥", "≠", "¬")
MISSING_IN_ENV_RE = re.compile(
    r"(reference|variable)\s+.+\s+was not found in the current environment",
    re.IGNORECASE,
)
MAX_GOAL_TEXT_CHARS = 12_000


class LemmaSubsessionProbeError(ValueError):
    def __init__(
        self,
        *,
        lemma_name: str,
        lemma_statement: str,
        probe_tactic: str,
        probe_check: dict[str, Any],
    ):
        self.lemma_name = lemma_name
        self.lemma_statement = lemma_statement
        self.probe_tactic = probe_tactic
        self.probe_check = dict(probe_check)
        probe_error = str(probe_check.get("error", "unknown error"))
        probe_hint = probe_check.get("hint")
        message = (
            f"Lemma declaration/type-check failed for `{lemma_name}`. "
            f"Probe tactic `{probe_tactic}` failed with: {probe_error}"
        )
        if probe_hint:
            message += f" Hint: {probe_hint}"
        super().__init__(message)


def _join_lines(lines: list[str], *, trailing_newline: bool = True) -> str:
    text = "\n".join(lines)
    if trailing_newline and not text.endswith("\n"):
        text += "\n"
    return text


def _normalize_lemma_type(lemma_type: str) -> str:
    out = lemma_type.strip()
    if "`" in out:
        raise ValueError(
            "Invalid `lemma_type`: markdown backticks are not allowed. "
            "Pass only raw proposition text, e.g. "
            "lemma_type='forall n : nat, n > 0 -> n >= 1'."
        )
    if any(tok in out for tok in UNICODE_LOGIC_TOKENS):
        raise ValueError(
            "Invalid `lemma_type`: Unicode logical symbols are not supported in tool input. "
            "Use ASCII Coq syntax instead: `forall`, `exists`, `->`, `/\\`, `\\/`, `<=`, `>=`."
        )
    if LEMMA_DECL_PREFIX_RE.match(out):
        raise ValueError(
            "Invalid `lemma_type`: expected proposition only, not a full declaration. "
            "Example: lemma_name='helper_x', lemma_type='forall n : nat, n > 0 -> True'."
        )
    if LEMMA_PROOF_TOKEN_RE.search(out):
        raise ValueError(
            "Invalid `lemma_type`: do not include proof commands (`Proof.`, `Qed.`, ...). "
            "Pass only the proposition."
        )
    if out.endswith("."):
        out = out[:-1].rstrip()
    return out


def _tactic_error_hint(error: str, *, include_semantic_tool: bool = True) -> str | None:
    if MISSING_IN_ENV_RE.search(error):
        retrieval_guidance = "`explore_toc` + `read_source_file`"
        if include_semantic_tool:
            retrieval_guidance += (
                " (optionally `semantic_doc_search` with a natural-language query "
                "to find relevant modules/lemmas faster)"
            )
        return (
            "A referenced constant/tactic is missing in the current environment. "
            f"If this comes from a library, use {retrieval_guidance} to find the module, "
            "then add it via `add_import` (main) or `require_import` (subagent). "
            "Do not guess import roots; derive `libname`/`source` from TOC entries."
        )
    return None


def _normalize_error_for_loop_guard(error: str) -> str:
    text = re.sub(r"\s+", " ", error.strip().lower())
    if len(text) > 500:
        return text[:500]
    return text


def _clip_goal_text(goal_text: str, *, max_chars: int = MAX_GOAL_TEXT_CHARS) -> str:
    if len(goal_text) <= max_chars:
        return goal_text
    omitted = len(goal_text) - max_chars
    return goal_text[:max_chars] + f"\n... [goal truncated, omitted {omitted} chars]"


def _is_missing_name_error(error: str, *, name: str) -> bool:
    pattern = re.compile(
        rf"(reference|variable)\s+{re.escape(name)}\s+was not found in the current environment",
        re.IGNORECASE,
    )
    return bool(pattern.search(error))


def _format_state_feedback(state: Any) -> list[dict[str, Any]]:
    raw_feedback = getattr(state, "feedback", None)
    if not isinstance(raw_feedback, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in raw_feedback:
        level: Any = None
        message: Any = None
        if isinstance(entry, tuple | list) and len(entry) >= 2:
            level, message = entry[0], entry[1]
        elif isinstance(entry, dict):
            level = entry.get("level")
            message = entry.get("message")
        else:
            message = str(entry)
        out.append(
            {
                "level": int(level) if isinstance(level, int) else level,
                "message": str(message) if message is not None else "",
            }
        )
    return out


def _normalize_import_parts(libname: str, source: str) -> tuple[str, str]:
    lib = libname.strip()
    src = source.strip()

    def _split_logical_path(path: str) -> tuple[str, str]:
        pieces = [p for p in path.split(".") if p]
        if len(pieces) < 2:
            raise ValueError(
                "Invalid import statement: expected dotted logical path in "
                f"`Require Import ...`, got {path!r}. "
                "Use either `From A.B Require Import C.` or pass `libname='A.B', source='C'`."
            )
        return ".".join(pieces[:-1]), pieces[-1]

    stmt_in_source = FROM_REQUIRE_IMPORT_RE.match(src)
    if stmt_in_source:
        return stmt_in_source.group(1), stmt_in_source.group(2)

    stmt_in_libname = FROM_REQUIRE_IMPORT_RE.match(lib)
    if stmt_in_libname:
        return stmt_in_libname.group(1), stmt_in_libname.group(2)

    req_stmt_in_source = REQUIRE_IMPORT_RE.match(src)
    if req_stmt_in_source:
        return _split_logical_path(req_stmt_in_source.group(1))

    req_stmt_in_libname = REQUIRE_IMPORT_RE.match(lib)
    if req_stmt_in_libname:
        return _split_logical_path(req_stmt_in_libname.group(1))

    return lib, src


def _indent_proof_body(body: str) -> str:
    lines: list[str] = []
    for raw in body.splitlines():
        if raw.strip():
            lines.append(f"  {raw.strip()}")
        else:
            lines.append("")
    if not lines:
        return "  exact I."
    return "\n".join(lines)


def _extract_proof_body(proof_script: str) -> str:
    text = proof_script.strip()
    if "Proof." in text and "Qed." in text:
        after = text.split("Proof.", 1)[1]
        before_qed = after.split("Qed.", 1)[0]
        return before_qed.strip()
    return text


def _remove_blank_padding(lines: list[str]) -> list[str]:
    out = list(lines)
    i = 0
    while i < len(out) - 1:
        if out[i].strip() == "" and out[i + 1].strip() == "":
            del out[i]
            continue
        i += 1
    return out


@dataclass(frozen=True)
class ParsedDocumentLayout:
    lines: list[str]
    target_start_line: int
    proof_line: int
    proof_character: int
    target_end_line: int

    @property
    def prefix_lines(self) -> list[str]:
        return self.lines[: self.target_start_line]

    @property
    def target_lines(self) -> list[str]:
        return self.lines[self.target_start_line : self.target_end_line + 1]

    @property
    def suffix_lines(self) -> list[str]:
        return self.lines[self.target_end_line + 1 :]


def parse_last_target_layout(content: str) -> ParsedDocumentLayout:
    lines = content.splitlines()
    if not lines:
        raise ValueError("Empty source file.")

    proof_line = -1
    proof_character = -1
    for idx, line in enumerate(lines):
        col = line.find(PROOF_TOKEN)
        if col != -1:
            proof_line = idx
            proof_character = col + len(PROOF_TOKEN)
    if proof_line < 0:
        raise ValueError("No `Proof.` token found in source.")

    target_start_line = -1
    for idx in range(proof_line, -1, -1):
        if TARGET_START_RE.search(lines[idx]):
            target_start_line = idx
            break
    if target_start_line < 0:
        raise ValueError("No theorem/lemma statement found before the last `Proof.`.")

    target_end_line = len(lines) - 1
    for idx in range(proof_line + 1, len(lines)):
        if any(token in lines[idx] for token in END_PROOF_TOKENS):
            target_end_line = idx
            break

    return ParsedDocumentLayout(
        lines=lines,
        target_start_line=target_start_line,
        proof_line=proof_line,
        proof_character=proof_character,
        target_end_line=target_end_line,
    )


def _format_with_line_numbers(content: str) -> str:
    lines = content.splitlines()
    width = max(3, len(str(len(lines))))
    return "\n".join(f"{idx + 1:>{width}}: {line}" for idx, line in enumerate(lines))


@dataclass
class StateNode:
    index: int
    parent_index: int | None
    tactic: str | None
    state: Any


@dataclass
class BranchSession:
    client: Any
    doc_id: int
    source_path: Path
    source_content: str
    layout: ParsedDocumentLayout
    timeout: float
    nodes: list[StateNode] = field(default_factory=list)
    logger: Callable[[str], None] | None = None
    include_semantic_tool: bool = True
    event_logger: StateEventLogger | None = None
    _last_failure_signature: tuple[int, str] | None = None
    _last_failure_count: int = 0
    _last_stagnation_signature: tuple[int, str] | None = None
    _last_stagnation_count: int = 0

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def _emit_event(self, payload: dict[str, Any]) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger(payload)
        except Exception:
            return

    @property
    def available_state_indexes(self) -> list[int]:
        return [node.index for node in self.nodes]

    @property
    def latest_state_index(self) -> int:
        return self.nodes[-1].index

    def _state_node(self, state_index: int) -> StateNode:
        if state_index < 0 or state_index >= len(self.nodes):
            raise ValueError(f"Unknown state index {state_index}; available={self.available_state_indexes}")
        return self.nodes[state_index]

    def list_states(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for node in self.nodes:
            out.append(
                {
                    "state_index": node.index,
                    "parent_state_index": node.parent_index,
                    "tactic": node.tactic,
                    "proof_line": self.layout.proof_line + 1,
                }
            )
        return out

    def get_goals(self, state_index: int = 0) -> dict[str, Any]:
        pretty_goals = self._pretty_goals(state_index)
        proof_finished = self._state_proof_finished(state_index, goals_count=len(pretty_goals))
        self._log(f"get_goals(doc_id={self.doc_id}, state={state_index}) -> {len(pretty_goals)} goals")
        return {
            "state_index": state_index,
            "goals": pretty_goals,
            "proof_finished": proof_finished,
        }

    def _pretty_goals(self, state_index: int) -> list[str]:
        node = self._state_node(state_index)
        goals = self.client.goals(node.state, timeout=self.timeout)
        return [_clip_goal_text(getattr(goal, "pp", None) or getattr(goal, "ty", "")) for goal in goals]

    def _state_proof_finished(self, state_index: int, *, goals_count: int | None = None) -> bool:
        node = self._state_node(state_index)
        raw = getattr(node.state, "proof_finished", None)
        if isinstance(raw, bool):
            return raw
        if raw is not None:
            try:
                return bool(raw)
            except Exception:
                pass
        if goals_count is None:
            goals_count = len(self._pretty_goals(state_index))
        return goals_count == 0

    def run_tac(
        self,
        state_index: int,
        tactic: str,
        *,
        branch_reason: str | None = None,
    ) -> dict[str, Any]:
        node = self._state_node(state_index)
        latest_state_before = self.latest_state_index
        is_stale_state = state_index != latest_state_before
        goals_before = self._pretty_goals(state_index)
        goals_before_count = len(goals_before)
        normalized_tactic = re.sub(r"\s+", " ", tactic.strip())
        self._log(
            f"run_tac(doc_id={self.doc_id}, state={state_index}, head_before={latest_state_before}, "
            f"stale_state={is_stale_state}, tactic={tactic!r}, branch_reason={branch_reason!r})"
        )
        if is_stale_state and not (branch_reason or "").strip():
            stale_sig = (state_index, normalized_tactic)
            if stale_sig == self._last_stagnation_signature and self._last_stagnation_count >= 8:
                error = (
                    f"Stagnation guard: repeated stale-state tactic from state {state_index} "
                    f"(head={latest_state_before}) with no declared branch intent. "
                    "Use `run_tac_latest` for normal progress, or provide `branch_reason` when branching on purpose."
                )
                hint = (
                    "Re-anchor with `current_head` + `get_goals`, then either continue with `run_tac_latest` "
                    "or explicitly justify rollback with `branch_reason`."
                )
                self._log(
                    f"run_tac rejected(doc_id={self.doc_id}, state={state_index}): {error}"
                )
                self._emit_event(
                    {
                        "type": "stagnation_alert",
                        "doc_id": self.doc_id,
                        "state_index": state_index,
                        "head_state_index": latest_state_before,
                        "tactic": tactic,
                        "reason": "repeated_stale_state_no_branch_reason",
                        "repeat_count": self._last_stagnation_count,
                        "hint": hint,
                    }
                )
                return {
                    "ok": False,
                    "doc_id": self.doc_id,
                    "source_state_index": state_index,
                    "head_state_index": latest_state_before,
                    "stale_state": is_stale_state,
                    "error": error,
                    "hint": hint,
                    "loop_guard_triggered": True,
                    "failure_repeat_count": self._last_stagnation_count,
                }
        stripped_tactic = tactic.strip()
        if not stripped_tactic:
            error = (
                "run_tac received an empty tactic. "
                "Provide one concrete proof step (for example `intro.` or `exact H.`)."
            )
            self._log(f"run_tac rejected(doc_id={self.doc_id}, state={state_index}): {error}")
            self._emit_event(
                {
                    "type": "run_tac_rejected",
                    "doc_id": self.doc_id,
                    "state_index": state_index,
                    "head_state_index": latest_state_before,
                    "stale_state": is_stale_state,
                    "tactic": tactic,
                    "error": error,
                }
            )
            return {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "head_state_index": latest_state_before,
                "stale_state": is_stale_state,
                "error": error,
            }
        if stripped_tactic.startswith(":"):
            error = (
                "run_tac received an invalid tactic starting with `:`. "
                "Remove leading punctuation and send plain Coq tactic text."
            )
            self._log(f"run_tac rejected(doc_id={self.doc_id}, state={state_index}): {error}")
            self._emit_event(
                {
                    "type": "run_tac_rejected",
                    "doc_id": self.doc_id,
                    "state_index": state_index,
                    "head_state_index": latest_state_before,
                    "stale_state": is_stale_state,
                    "tactic": tactic,
                    "error": error,
                }
            )
            return {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "head_state_index": latest_state_before,
                "stale_state": is_stale_state,
                "error": error,
            }
        if stripped_tactic.startswith("`") or stripped_tactic.endswith("`"):
            error = (
                "run_tac received markdown-wrapped tactic. "
                "Pass raw tactic text without backticks, e.g. `intro n.` not \"`intro n.`\"."
            )
            self._log(f"run_tac rejected(doc_id={self.doc_id}, state={state_index}): {error}")
            self._emit_event(
                {
                    "type": "run_tac_rejected",
                    "doc_id": self.doc_id,
                    "state_index": state_index,
                    "head_state_index": latest_state_before,
                    "stale_state": is_stale_state,
                    "tactic": tactic,
                    "error": error,
                }
            )
            return {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "head_state_index": latest_state_before,
                "stale_state": is_stale_state,
                "error": error,
            }
        if FORBIDDEN_TACTIC_PREFIX_RE.match(tactic):
            error = (
                "run_tac only accepts proof tactics, not top-level vernac commands. "
                "Use add_import/remove_import and add_intermediate_lemma/remove_intermediate_lemma tools."
            )
            self._log(f"run_tac rejected(doc_id={self.doc_id}, state={state_index}): {error}")
            self._emit_event(
                {
                    "type": "run_tac_rejected",
                    "doc_id": self.doc_id,
                    "state_index": state_index,
                    "head_state_index": latest_state_before,
                    "stale_state": is_stale_state,
                    "tactic": tactic,
                    "error": error,
                }
            )
            return {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "head_state_index": latest_state_before,
                "stale_state": is_stale_state,
                "error": error,
            }
        if PLACEHOLDER_TACTIC_RE.search(tactic):
            error = (
                "run_tac rejected placeholder tactic (`admit`/`Admitted`). "
                "Provide a real proof step."
            )
            self._log(f"run_tac rejected(doc_id={self.doc_id}, state={state_index}): {error}")
            self._emit_event(
                {
                    "type": "run_tac_rejected",
                    "doc_id": self.doc_id,
                    "state_index": state_index,
                    "head_state_index": latest_state_before,
                    "stale_state": is_stale_state,
                    "tactic": tactic,
                    "error": error,
                }
            )
            return {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "head_state_index": latest_state_before,
                "stale_state": is_stale_state,
                "error": error,
            }
        if SHELVING_TACTIC_RE.search(tactic):
            error = (
                "run_tac rejected shelving tactics (`shelve`/`Unshelve`). "
                "These can hide unresolved obligations and produce non-replayable proofs. "
                "Use real proof steps or branch/lemma tools instead."
            )
            self._log(f"run_tac rejected(doc_id={self.doc_id}, state={state_index}): {error}")
            self._emit_event(
                {
                    "type": "run_tac_rejected",
                    "doc_id": self.doc_id,
                    "state_index": state_index,
                    "head_state_index": latest_state_before,
                    "stale_state": is_stale_state,
                    "tactic": tactic,
                    "error": error,
                }
            )
            return {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "head_state_index": latest_state_before,
                "stale_state": is_stale_state,
                "error": error,
            }
        try:
            new_state = self.client.run(node.state, tactic, timeout=self.timeout)
        except Exception as exc:
            error = str(exc)
            hint = _tactic_error_hint(error, include_semantic_tool=self.include_semantic_tool)
            signature = (state_index, _normalize_error_for_loop_guard(error))
            if signature == self._last_failure_signature:
                self._last_failure_count += 1
            else:
                self._last_failure_signature = signature
                self._last_failure_count = 1
            loop_guard_triggered = self._last_failure_count >= 3
            if loop_guard_triggered:
                loop_hint = (
                    f"Loop guard: repeated the same failure on state {state_index} "
                    f"({self._last_failure_count} times). "
                    "Re-anchor with `get_goals`/`list_states`/`read_workspace_source`, then "
                    "change strategy (import missing symbols, branch from another state, or "
                    "introduce/prove an intermediate lemma). Do not retry the same tactic pattern."
                )
                if hint:
                    hint = f"{hint} {loop_hint}"
                else:
                    hint = loop_hint
                self._log(
                    f"run_tac loop_guard(doc_id={self.doc_id}, state={state_index}): "
                    f"repeat_count={self._last_failure_count}"
                )
            self._log(f"run_tac failed(doc_id={self.doc_id}, state={state_index}): {error}")
            if hint:
                self._log(f"run_tac hint(doc_id={self.doc_id}, state={state_index}): {hint}")
            self._emit_event(
                {
                    "type": "run_tac_failure",
                    "doc_id": self.doc_id,
                    "state_index": state_index,
                    "head_state_index": latest_state_before,
                    "stale_state": is_stale_state,
                    "tactic": tactic,
                    "error": error,
                    "hint": hint,
                    "loop_guard_triggered": loop_guard_triggered,
                    "failure_repeat_count": self._last_failure_count,
                }
            )
            out = {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "head_state_index": latest_state_before,
                "stale_state": is_stale_state,
                "error": error,
            }
            if hint:
                out["hint"] = hint
            if loop_guard_triggered:
                out["loop_guard_triggered"] = True
                out["failure_repeat_count"] = self._last_failure_count
            return out

        new_idx = len(self.nodes)
        self.nodes.append(StateNode(index=new_idx, parent_index=state_index, tactic=tactic, state=new_state))
        self._last_failure_signature = None
        self._last_failure_count = 0
        goals = self.client.goals(new_state, timeout=self.timeout)
        pretty_goals = [_clip_goal_text(getattr(goal, "pp", None) or getattr(goal, "ty", "")) for goal in goals]
        goals_after_count = len(pretty_goals)
        new_state_proof_finished = self._state_proof_finished(new_idx, goals_count=goals_after_count)
        if new_state_proof_finished:
            progress_type = "solved"
        elif goals_after_count < goals_before_count:
            progress_type = "goal_reduced"
        elif goals_after_count == goals_before_count:
            progress_type = "goal_same"
        else:
            progress_type = "goal_increased"
        stagnation_sig = (state_index, normalized_tactic)
        if is_stale_state and progress_type == "goal_same":
            if stagnation_sig == self._last_stagnation_signature:
                self._last_stagnation_count += 1
            else:
                self._last_stagnation_signature = stagnation_sig
                self._last_stagnation_count = 1
        else:
            self._last_stagnation_signature = None
            self._last_stagnation_count = 0
        feedback = _format_state_feedback(new_state)
        self._log(
            f"run_tac ok(doc_id={self.doc_id}) -> new_state_index={new_idx}, "
            f"goals={goals_before_count}->{goals_after_count}, progress={progress_type}, "
            f"source_state={state_index}, head_before={latest_state_before}, stale_state={is_stale_state}"
        )
        stale_warning: str | None = None
        if is_stale_state and not (branch_reason or "").strip():
            stale_warning = (
                "You executed from a non-latest state without branch_reason. "
                "Use `run_tac_latest` unless you intentionally branch from history."
            )
        self._emit_event(
            {
                "type": "run_tac_success",
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "new_state_index": new_idx,
                "head_state_index_before": latest_state_before,
                "head_state_index_after": self.latest_state_index,
                "stale_state": is_stale_state,
                "branch_reason": (branch_reason or "").strip() or None,
                "tactic": tactic,
                "goals_before": goals_before_count,
                "goals_after": goals_after_count,
                "proof_finished_after": new_state_proof_finished,
                "progress_type": progress_type,
                "stagnation_repeat_count": self._last_stagnation_count if is_stale_state else 0,
            }
        )
        if (
            is_stale_state
            and progress_type == "goal_same"
            and self._last_stagnation_count >= 6
            and not (branch_reason or "").strip()
        ):
            self._emit_event(
                {
                    "type": "stagnation_alert",
                    "doc_id": self.doc_id,
                    "state_index": state_index,
                    "head_state_index": latest_state_before,
                    "tactic": tactic,
                    "reason": "stale_state_no_goal_change",
                    "repeat_count": self._last_stagnation_count,
                    "hint": (
                        "Repeated stale-state tactic with no goal reduction. "
                        "Switch to run_tac_latest or provide an explicit branch_reason."
                    ),
                }
            )
        return {
            "ok": True,
            "doc_id": self.doc_id,
            "source_state_index": state_index,
            "head_state_index_before": latest_state_before,
            "head_state_index_after": self.latest_state_index,
            "stale_state": is_stale_state,
            "new_state_index": new_idx,
            "goals": pretty_goals,
            "goals_before_count": goals_before_count,
            "goals_after_count": goals_after_count,
            "proof_finished_after": new_state_proof_finished,
            "progress_type": progress_type,
            "feedback": feedback,
            "feedback_count": len(feedback),
            "stale_state_warning": stale_warning,
        }

    def _tactics_path(self, state_index: int) -> list[str]:
        node = self._state_node(state_index)
        idx = node.index
        path: list[str] = []
        while idx is not None and idx > 0:
            cur = self.nodes[idx]
            if cur.tactic:
                path.append(cur.tactic)
            idx = cur.parent_index
        path.reverse()
        return path

    def proof_script(self, state_index: int | None = None) -> str:
        target = self.latest_state_index if state_index is None else state_index
        tactics = self._tactics_path(target)
        if not tactics:
            return "exact I."
        return "\n".join(tactics)

    def has_placeholder_tactic(self, state_index: int | None = None) -> bool:
        target = self.latest_state_index if state_index is None else state_index
        return any(PLACEHOLDER_TACTIC_RE.search(tac or "") for tac in self._tactics_path(target))

    def materialized_source(self, state_index: int | None = None) -> str:
        """Return source content with current proof script materialized.

        If the selected state is fully solved, emit a closing `Qed.`.
        Otherwise, keep the script but do not claim closure with `Qed.`.
        """
        target = self.latest_state_index if state_index is None else state_index
        lines = list(self.layout.lines)
        proof_line = self.layout.proof_line
        end_line = self.layout.target_end_line
        goals = self._pretty_goals(target)
        state_finished = self._state_proof_finished(target, goals_count=len(goals))
        finished = state_finished and not self.has_placeholder_tactic(target)

        # Include tokens on the same line as `Proof.` / end-token if present.
        if 0 <= proof_line < len(lines):
            line = lines[proof_line]
            pos = line.find(PROOF_TOKEN)
            if pos >= 0:
                lines[proof_line] = line[: pos + len(PROOF_TOKEN)]
        for idx in range(proof_line, len(lines)):
            if any(token in lines[idx] for token in END_PROOF_TOKENS):
                end_line = idx
                break

        script_lines = [ln.rstrip() for ln in self.proof_script(target).splitlines() if ln.strip()]
        if not script_lines:
            script_lines = ["exact I."]
        closing_line = (
            "Qed."
            if finished
            else (
                "(* proof unfinished: state not closed"
                + (f", {len(goals)} focused goal(s) remaining" if len(goals) > 0 else ", 0 focused goals")
                + "; no final Qed. *)"
            )
        )
        rebuilt = lines[: proof_line + 1] + script_lines + [closing_line] + lines[end_line + 1 :]
        return _join_lines(rebuilt, trailing_newline=True)


@dataclass
class DocumentNode:
    doc_id: int
    parent_doc_id: int | None
    label: str
    content: str


class DocumentManager:
    def __init__(
        self,
        client: Any,
        source_path: str | Path,
        *,
        timeout: float = 60.0,
        logger: Callable[[str], None] | None = None,
        mutation_validator: MutationValidator | None = None,
        include_semantic_tool: bool = True,
        event_logger: StateEventLogger | None = None,
    ):
        self.client = client
        self.source_path = Path(source_path).resolve()
        self.timeout = timeout
        self.logger = logger
        self.mutation_validator = mutation_validator
        self.include_semantic_tool = bool(include_semantic_tool)
        self.event_logger = event_logger

        root_content = self.source_path.read_text(encoding="utf-8")
        self.nodes: dict[int, DocumentNode] = {
            0: DocumentNode(doc_id=0, parent_doc_id=None, label="root", content=root_content),
        }
        self.sessions: dict[int, BranchSession] = {
            0: self._new_branch_session(doc_id=0, content=root_content),
        }
        self.head_doc_id = 0
        self._next_doc_id = 1
        self._next_lemma_id = 1
        self._next_transient_doc_id = -1

    def set_event_logger(self, event_logger: StateEventLogger | None) -> None:
        self.event_logger = event_logger
        for session in self.sessions.values():
            session.event_logger = event_logger

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def _new_branch_session(
        self,
        doc_id: int,
        content: str,
        *,
        purpose: str | None = None,
    ) -> BranchSession:
        tmp_path = Path(
            self.client.tmp_file(content=content),
        ).resolve()
        layout = parse_last_target_layout(content)
        state0 = self.client.get_state_at_pos(
            str(tmp_path),
            layout.proof_line,
            layout.proof_character,
            timeout=self.timeout,
        )
        session = BranchSession(
            client=self.client,
            doc_id=doc_id,
            source_path=tmp_path,
            source_content=content,
            layout=layout,
            timeout=self.timeout,
            nodes=[StateNode(index=0, parent_index=None, tactic=None, state=state0)],
            logger=self.logger,
            include_semantic_tool=self.include_semantic_tool,
            event_logger=self.event_logger,
        )
        if doc_id < 0:
            prefix = "initialized transient doc session"
            if purpose:
                prefix += f" ({purpose})"
            self._log(
                f"{prefix} at state 0 "
                f"(transient_doc_id={doc_id}, source={tmp_path}, "
                f"line={layout.proof_line}, character={layout.proof_character})"
            )
        else:
            self._log(
                f"initialized doc_id={doc_id} at state 0 "
                f"(source={tmp_path}, line={layout.proof_line}, character={layout.proof_character})"
            )
        return session

    def _alloc_transient_doc_id(self) -> int:
        out = self._next_transient_doc_id
        self._next_transient_doc_id -= 1
        return out

    def _resolve_existing_doc(self, doc_id: int | None) -> int:
        if doc_id is None:
            return self.head_doc_id
        if doc_id not in self.nodes:
            known = sorted(self.nodes.keys())
            raise ValueError(f"Unknown doc_id={doc_id}. available_doc_ids={known}")
        return doc_id

    def _resolve_doc_for_mutation(self, doc_id: int | None) -> int:
        return self._resolve_existing_doc(doc_id)

    def doc_id_for_source_path(self, source_path: str | Path) -> int | None:
        try:
            resolved = Path(source_path).resolve()
        except Exception:
            return None
        # Allow callers to reference the original workspace source path and
        # transparently read the current head virtual document.
        if resolved == self.source_path:
            return self.head_doc_id
        for doc_id, session in self.sessions.items():
            if session.source_path.resolve() == resolved:
                return doc_id
        return None

    def _rollback_mutation(self, *, new_doc_id: int, base_doc_id: int) -> None:
        self.nodes.pop(new_doc_id, None)
        self.sessions.pop(new_doc_id, None)
        self._next_doc_id = new_doc_id
        self.head_doc_id = base_doc_id

    def _replay_latest_tactics(self, *, source_doc_id: int, target_doc_id: int, label: str) -> int:
        source_session = self.sessions[source_doc_id]
        target_session = self.sessions[target_doc_id]
        tactics = source_session._tactics_path(source_session.latest_state_index)
        if not tactics:
            return 0

        replayed = 0
        state_index = 0
        for tactic in tactics:
            out = target_session.run_tac(state_index=state_index, tactic=tactic)
            if not out.get("ok", False):
                error = out.get("error", "unknown error")
                raise ValueError(
                    f"Replay failed for mutation `{label}` on tactic #{replayed + 1} "
                    f"(source_doc_id={source_doc_id}, target_doc_id={target_doc_id}, "
                    f"source_state={state_index}): {error}"
                )
            state_index = int(out["new_state_index"])
            replayed += 1
        return replayed

    def _create_mutation(
        self,
        *,
        base_doc_id: int,
        label: str,
        content: str,
        replay_from_doc_id: int | None = None,
    ) -> dict[str, Any]:
        branched = base_doc_id != self.head_doc_id
        new_doc_id = self._next_doc_id
        self._next_doc_id += 1
        self.nodes[new_doc_id] = DocumentNode(
            doc_id=new_doc_id,
            parent_doc_id=base_doc_id,
            label=label,
            content=content,
        )
        self.sessions[new_doc_id] = self._new_branch_session(doc_id=new_doc_id, content=content)
        self.head_doc_id = new_doc_id
        replayed_tactics = 0
        if self.mutation_validator is not None:
            ok, error = self.mutation_validator(new_doc_id, label)
            if not ok:
                self._rollback_mutation(new_doc_id=new_doc_id, base_doc_id=base_doc_id)
                detail = error or "fresh-session validation failed"
                raise ValueError(f"Mutation `{label}` rejected by fresh-session validation: {detail}")
        if replay_from_doc_id is not None:
            try:
                replayed_tactics = self._replay_latest_tactics(
                    source_doc_id=replay_from_doc_id,
                    target_doc_id=new_doc_id,
                    label=label,
                )
            except Exception as exc:
                self._rollback_mutation(new_doc_id=new_doc_id, base_doc_id=base_doc_id)
                raise ValueError(
                    f"Mutation `{label}` failed while replaying prior tactics from doc_id="
                    f"{replay_from_doc_id}: {exc}"
                ) from exc
        self._log(
            f"new doc node: {new_doc_id} (from {base_doc_id}) label={label!r} "
            f"branched={branched} replayed_tactics={replayed_tactics}"
        )
        return {
            "ok": True,
            "doc_id": new_doc_id,
            "base_doc_id": base_doc_id,
            "branched": branched,
            "head_doc_id": self.head_doc_id,
            "replayed_tactics": replayed_tactics,
        }

    def list_states(self, *, doc_id: int | None = None) -> list[int]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        return self.sessions[resolved_doc_id].available_state_indexes

    def list_states_verbose(self, *, doc_id: int | None = None) -> list[dict[str, Any]]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        states = self.sessions[resolved_doc_id].list_states()
        latest = self.sessions[resolved_doc_id].latest_state_index
        for entry in states:
            entry["is_latest"] = int(entry["state_index"]) == latest
        return states

    def get_goals(self, *, state_index: int = 0, doc_id: int | None = None) -> dict[str, Any]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        return self.sessions[resolved_doc_id].get_goals(state_index=state_index)

    def run_tac(
        self,
        *,
        state_index: int,
        tactic: str,
        doc_id: int | None = None,
        branch_reason: str | None = None,
    ) -> dict[str, Any]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        session = self.sessions[resolved_doc_id]
        try:
            out = session.run_tac(
                state_index=state_index,
                tactic=tactic,
                branch_reason=branch_reason,
            )
            out["resolved_doc_id"] = resolved_doc_id
            return out
        except ValueError as exc:
            raise ValueError(
                f"{exc} (doc_id={resolved_doc_id}, current_head={self.head_doc_id}, "
                f"available_states={session.available_state_indexes})"
            ) from exc

    def run_tac_latest(
        self,
        *,
        tactic: str,
        doc_id: int | None = None,
        branch_reason: str | None = None,
    ) -> dict[str, Any]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        latest = self.sessions[resolved_doc_id].latest_state_index
        return self.run_tac(
            state_index=latest,
            tactic=tactic,
            doc_id=resolved_doc_id,
            branch_reason=branch_reason,
        )

    def read_source(
        self,
        *,
        line: int | None = None,
        before: int = 20,
        after: int = 20,
        doc_id: int | None = None,
    ) -> dict[str, Any]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        content = self.nodes[resolved_doc_id].content
        lines = content.splitlines()
        if line is None:
            return {
                "doc_id": resolved_doc_id,
                "mode": "full",
                "total_lines": len(lines),
                "content": _format_with_line_numbers(content),
            }

        if line < 1:
            raise ValueError("line must be >= 1")
        start = max(1, line - max(0, before))
        end = min(len(lines), line + max(0, after))
        snippet = "\n".join(lines[start - 1 : end])
        return {
            "doc_id": resolved_doc_id,
            "mode": "around_line",
            "line": line,
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
            "content": _format_with_line_numbers(snippet),
        }

    def show_workspace(self, *, doc_id: int | None = None) -> dict[str, Any]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        node = self.nodes[resolved_doc_id]
        session = self.sessions[resolved_doc_id]
        head = self.completion_status(doc_id=resolved_doc_id)
        recent_transitions: list[dict[str, Any]] = []
        for state in session.list_states()[-5:]:
            if state["tactic"] is None:
                continue
            recent_transitions.append(
                {
                    "state_index": state["state_index"],
                    "parent_state_index": state["parent_state_index"],
                    "tactic": state["tactic"],
                }
            )
        recommended_next_action = (
            "Proof appears complete on latest state. Re-check with completion_status/get_goals and then finish."
            if bool(head.get("latest_proof_finished", False))
            else (
                "Re-anchor with current_head/get_goals on latest state, then continue with run_tac_latest."
            )
        )
        return {
            "doc_id": resolved_doc_id,
            "head_doc_id": self.head_doc_id,
            "parent_doc_id": node.parent_doc_id,
            "label": node.label,
            "source_path": str(session.source_path),
            "states": session.list_states(),
            "head": {
                "doc_id": head.get("doc_id"),
                "latest_state_index": head.get("latest_state_index"),
                "latest_goals_count": head.get("latest_goals_count"),
                "latest_proof_finished": head.get("latest_proof_finished"),
            },
            "recent_transitions": recent_transitions,
            "recommended_next_action": recommended_next_action,
            "content": _format_with_line_numbers(node.content),
        }

    def list_docs(self) -> list[dict[str, Any]]:
        children: dict[int, list[int]] = {doc_id: [] for doc_id in self.nodes}
        for doc_id, node in self.nodes.items():
            if node.parent_doc_id is not None and node.parent_doc_id in children:
                children[node.parent_doc_id].append(doc_id)

        out: list[dict[str, Any]] = []
        for doc_id in sorted(self.nodes.keys()):
            node = self.nodes[doc_id]
            session = self.sessions.get(doc_id)
            out.append(
                {
                    "doc_id": doc_id,
                    "parent_doc_id": node.parent_doc_id,
                    "label": node.label,
                    "is_head": doc_id == self.head_doc_id,
                    "children_doc_ids": sorted(children.get(doc_id, [])),
                    "state_count": len(session.nodes) if session is not None else 0,
                }
            )
        return out

    def checkout_doc(self, *, doc_id: int) -> dict[str, Any]:
        resolved = self._resolve_existing_doc(doc_id)
        previous = self.head_doc_id
        self.head_doc_id = resolved
        self._log(f"checkout doc: {previous} -> {resolved}")
        return {
            "ok": True,
            "previous_head_doc_id": previous,
            "head_doc_id": self.head_doc_id,
        }

    def show_doc(self, *, doc_id: int) -> dict[str, Any]:
        return self.show_workspace(doc_id=doc_id)

    def materialized_source(
        self,
        *,
        doc_id: int | None = None,
        state_index: int | None = None,
    ) -> str:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        return self.sessions[resolved_doc_id].materialized_source(state_index=state_index)

    def completion_status(self, *, doc_id: int | None = None) -> dict[str, Any]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        session = self.sessions[resolved_doc_id]
        latest_state_index = session.latest_state_index
        latest_goals = session._pretty_goals(latest_state_index)
        latest_state_proof_finished = session._state_proof_finished(
            latest_state_index,
            goals_count=len(latest_goals),
        )
        has_placeholder = session.has_placeholder_tactic(latest_state_index)
        solved_state_indexes: list[int] = []
        for idx in session.available_state_indexes:
            if idx == latest_state_index:
                done = latest_state_proof_finished and not has_placeholder
            else:
                done = session._state_proof_finished(idx)
            if done:
                solved_state_indexes.append(idx)
        preview = latest_goals[0] if latest_goals else ""
        if len(preview) > 300:
            preview = preview[:300] + " ..."
        return {
            "doc_id": resolved_doc_id,
            "head_doc_id": self.head_doc_id,
            "is_head_doc": resolved_doc_id == self.head_doc_id,
            "latest_state_index": latest_state_index,
            "latest_goals_count": len(latest_goals),
            "latest_proof_finished": latest_state_proof_finished and not has_placeholder,
            "latest_state_proof_finished": latest_state_proof_finished,
            "latest_has_placeholder_tactic": has_placeholder,
            "solved_state_indexes": solved_state_indexes,
            "available_state_indexes": session.available_state_indexes,
            "latest_goal_preview": preview,
        }

    def current_head(self, *, doc_id: int | None = None) -> dict[str, Any]:
        status = self.completion_status(doc_id=doc_id)
        recommended_next_action = (
            "finish_candidate"
            if bool(status.get("latest_proof_finished", False))
            else "run_tac_latest"
        )
        return {
            "doc_id": status.get("doc_id"),
            "head_doc_id": status.get("head_doc_id"),
            "latest_state_index": status.get("latest_state_index"),
            "latest_goals_count": status.get("latest_goals_count"),
            "latest_proof_finished": status.get("latest_proof_finished"),
            "available_state_indexes": status.get("available_state_indexes"),
            "recommended_next_action": recommended_next_action,
        }

    def _insert_block_before_target(self, base_content: str, block: str) -> str:
        layout = parse_last_target_layout(base_content)
        prefix = list(layout.prefix_lines)
        if prefix and prefix[-1].strip():
            prefix.append("")
        prefix.extend(block.rstrip("\n").splitlines())
        prefix.append("")
        lines = prefix + layout.target_lines + layout.suffix_lines
        return _join_lines(lines, trailing_newline=True)

    @staticmethod
    def _import_insert_index(prefix_lines: list[str]) -> int:
        last_import = -1
        for idx, line in enumerate(prefix_lines):
            if IMPORT_RE.search(line):
                last_import = idx
        return 0 if last_import < 0 else last_import + 1

    @staticmethod
    def _import_remove_index(prefix_lines: list[str], *, libname: str, source: str) -> int:
        pattern = re.compile(
            rf"^\s*From\s+{re.escape(libname)}\s+Require\s+Import\s+{re.escape(source)}\s*\.\s*$"
        )
        for idx in range(len(prefix_lines) - 1, -1, -1):
            if pattern.match(prefix_lines[idx]):
                return idx
        return -1

    def _probe_import_statement(
        self,
        *,
        layout: ParsedDocumentLayout,
        insert_index: int,
        statement: str,
    ) -> None:
        """Probe import with Coq before mutating documents.

        We position on the import line and execute the same `From ... Require Import ...`
        command through the raw client to surface missing-path/module errors early.
        """
        probe_prefix = list(layout.prefix_lines)
        probe_prefix.insert(insert_index, statement)
        probe_content = _join_lines(
            probe_prefix + layout.target_lines + layout.suffix_lines,
            trailing_newline=True,
        )
        probe_path = Path(self.client.tmp_file(content=probe_content)).resolve()
        probe_line = max(1, insert_index + 1)
        try:
            probe_state = self.client.get_state_at_pos(
                str(probe_path),
                probe_line,
                0,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise ValueError(
                "Import probe failed while positioning before the inserted import "
                f"(line={probe_line}, statement={statement!r}): {exc}"
            ) from exc
        try:
            self.client.run(probe_state, statement, timeout=self.timeout)
        except Exception as exc:
            raise ValueError(
                f"Import probe rejected `{statement}`: {exc}. "
                "Verify `libname`/`source` with `explore_toc` before calling add_import."
            ) from exc

    @staticmethod
    def _find_lemma_range(prefix_lines: list[str], *, lemma_name: str) -> tuple[int, int] | None:
        start_re = re.compile(rf"^\s*Lemma\s+{re.escape(lemma_name)}\s*:")
        start_idx = -1
        for idx in range(len(prefix_lines) - 1, -1, -1):
            if start_re.search(prefix_lines[idx]):
                start_idx = idx
                break
        if start_idx < 0:
            return None

        end_idx = -1
        for idx in range(start_idx + 1, len(prefix_lines)):
            if any(token in prefix_lines[idx] for token in END_PROOF_TOKENS):
                end_idx = idx
                break
        if end_idx < 0:
            return None
        return (start_idx, end_idx)

    def add_import(self, *, libname: str, source: str, doc_id: int | None = None) -> dict[str, Any]:
        libname, source = _normalize_import_parts(libname, source)
        if not IDENT_RE.match(libname):
            raise ValueError(f"Invalid libname={libname!r}.")
        if not IDENT_RE.match(source):
            raise ValueError(f"Invalid source={source!r}.")

        resolved_doc_id = self._resolve_doc_for_mutation(doc_id)
        base = self.nodes[resolved_doc_id].content
        layout = parse_last_target_layout(base)

        statement = f"From {libname} Require Import {source}."
        self._log(
            f"add_import called(doc_id={resolved_doc_id}, libname={libname!r}, source={source!r})"
        )
        prefix = list(layout.prefix_lines)
        idx = self._import_insert_index(prefix)
        self._probe_import_statement(layout=layout, insert_index=idx, statement=statement)
        prefix.insert(idx, statement)
        new_content = _join_lines(prefix + layout.target_lines + layout.suffix_lines, trailing_newline=True)

        result = self._create_mutation(
            base_doc_id=resolved_doc_id,
            label=f"add_import:{libname}.{source}",
            content=new_content,
            replay_from_doc_id=resolved_doc_id,
        )
        result["statement"] = statement
        self._log(
            f"add_import ok(doc_id={result['doc_id']}, statement={statement!r}, "
            f"replayed_tactics={result.get('replayed_tactics', 0)})"
        )
        return result

    def remove_import(self, *, libname: str, source: str, doc_id: int | None = None) -> dict[str, Any]:
        libname, source = _normalize_import_parts(libname, source)
        if not IDENT_RE.match(libname):
            raise ValueError(f"Invalid libname={libname!r}.")
        if not IDENT_RE.match(source):
            raise ValueError(f"Invalid source={source!r}.")

        resolved_doc_id = self._resolve_doc_for_mutation(doc_id)
        base = self.nodes[resolved_doc_id].content
        layout = parse_last_target_layout(base)
        prefix = list(layout.prefix_lines)
        idx = self._import_remove_index(prefix, libname=libname, source=source)
        if idx < 0:
            raise ValueError(f"Import not found: From {libname} Require Import {source}.")

        self._log(
            f"remove_import called(doc_id={resolved_doc_id}, libname={libname!r}, source={source!r})"
        )
        removed = prefix.pop(idx).strip()
        prefix = _remove_blank_padding(prefix)
        new_content = _join_lines(prefix + layout.target_lines + layout.suffix_lines, trailing_newline=True)
        result = self._create_mutation(
            base_doc_id=resolved_doc_id,
            label=f"remove_import:{libname}.{source}",
            content=new_content,
            replay_from_doc_id=resolved_doc_id,
        )
        result["removed_statement"] = removed
        self._log(
            f"remove_import ok(doc_id={result['doc_id']}, statement={removed!r}, "
            f"replayed_tactics={result.get('replayed_tactics', 0)})"
        )
        return result

    def next_lemma_name(self) -> str:
        name = f"intermediate_lemma_{self._next_lemma_id}"
        self._next_lemma_id += 1
        return name

    def create_lemma_subsession(
        self,
        *,
        lemma_name: str,
        lemma_type: str,
        doc_id: int | None = None,
    ) -> tuple[int, BranchSession]:
        normalized_type = _normalize_lemma_type(lemma_type)
        if not normalized_type:
            raise ValueError("Lemma type cannot be empty.")

        resolved_doc_id = self._resolve_doc_for_mutation(doc_id)
        base_content = self.nodes[resolved_doc_id].content
        lemma_statement = f"Lemma {lemma_name} : {normalized_type}."

        # Type-check gate: statement must parse/type-check before sub-agent starts.
        probe_block = "\n".join([lemma_statement, "Proof.", "  admit.", "Admitted."])
        probe_content = self._insert_block_before_target(base_content, probe_block)
        probe_session = self._new_branch_session(
            doc_id=self._alloc_transient_doc_id(),
            content=probe_content,
            purpose="lemma_typecheck_probe",
        )
        probe_tactic = f"pose proof ({lemma_name})."
        probe_check = probe_session.run_tac(0, probe_tactic)
        if not probe_check.get("ok", False):
            probe_error = str(probe_check.get("error", ""))
            if _is_missing_name_error(probe_error, name=lemma_name):
                statement_probe_tactic = f"assert ({normalized_type})."
                statement_probe = probe_session.run_tac(0, statement_probe_tactic)
                probe_check = dict(probe_check)
                probe_check["statement_probe_tactic"] = statement_probe_tactic
                if not statement_probe.get("ok", False):
                    probe_check["statement_probe_error"] = statement_probe.get("error")
                    probe_check["statement_probe_hint"] = statement_probe.get("hint")
                message = (
                    "Internal probe note: `prepare_intermediate_lemma` runs "
                    f"`{probe_tactic}` to verify the newly declared lemma is in scope. "
                    f"That probe failed with: {probe_error}. "
                    "This usually indicates the lemma declaration did not register as expected "
                    "(statement parsing/typechecking/scope issue), not a missing library import."
                )
                if not statement_probe.get("ok", False):
                    message += (
                        f" Secondary statement probe `{statement_probe_tactic}` failed with: "
                        f"{statement_probe.get('error')}. "
                    )
                    st_hint = statement_probe.get("hint")
                    if st_hint:
                        message += f"Hint: {st_hint}"
                else:
                    message += (
                        " Secondary statement probe succeeded, so focus on why the named lemma "
                        "was not registered in the probe context."
                    )
                probe_check["hint"] = message
            raise LemmaSubsessionProbeError(
                lemma_name=lemma_name,
                lemma_statement=lemma_statement,
                probe_tactic=probe_tactic,
                probe_check=probe_check,
            )

        layout = parse_last_target_layout(base_content)
        lines = list(layout.prefix_lines)
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([lemma_statement, "Proof.", "Admitted."])
        if layout.suffix_lines:
            lines.append("")
            lines.extend(layout.suffix_lines)
        sub_content = _join_lines(lines, trailing_newline=True)
        sub_session = self._new_branch_session(
            doc_id=self._alloc_transient_doc_id(),
            content=sub_content,
            purpose="lemma_subgoal_workspace",
        )
        goals = sub_session.get_goals(0).get("goals", [])
        if len(goals) == 0:
            raise ValueError(
                "Intermediate lemma sub-session started with zero goals. "
                "This usually indicates an invalid or non-open proof context. "
                "Check that `lemma_type` is a valid proposition and uses ASCII Coq syntax "
                "(`forall`, `exists`, `->`, `/\\`, `\\/`, `<=`, `>=`)."
            )
        return resolved_doc_id, sub_session

    def register_proved_lemma(
        self,
        *,
        base_doc_id: int,
        lemma_name: str,
        lemma_type: str,
        proof_script: str,
    ) -> dict[str, Any]:
        normalized_type = _normalize_lemma_type(lemma_type)
        proof_body = _extract_proof_body(proof_script)
        lemma_block = "\n".join(
            [
                f"Lemma {lemma_name} : {normalized_type}.",
                "Proof.",
                _indent_proof_body(proof_body),
                "Qed.",
            ]
        )
        base_content = self.nodes[base_doc_id].content
        content = self._insert_block_before_target(base_content, lemma_block)
        result = self._create_mutation(
            base_doc_id=base_doc_id,
            label=f"add_lemma:{lemma_name}",
            content=content,
            replay_from_doc_id=base_doc_id,
        )
        result["lemma_name"] = lemma_name
        return result

    def remove_intermediate_lemma(self, *, lemma_name: str, doc_id: int | None = None) -> dict[str, Any]:
        if not IDENT_RE.match(lemma_name):
            raise ValueError(f"Invalid lemma_name={lemma_name!r}.")

        resolved_doc_id = self._resolve_doc_for_mutation(doc_id)
        base = self.nodes[resolved_doc_id].content
        layout = parse_last_target_layout(base)
        prefix = list(layout.prefix_lines)
        span = self._find_lemma_range(prefix, lemma_name=lemma_name)
        if span is None:
            raise ValueError(f"Intermediate lemma `{lemma_name}` not found.")
        start, end = span
        removed_lines = prefix[start : end + 1]
        del prefix[start : end + 1]
        prefix = _remove_blank_padding(prefix)
        new_content = _join_lines(prefix + layout.target_lines + layout.suffix_lines, trailing_newline=True)
        result = self._create_mutation(
            base_doc_id=resolved_doc_id,
            label=f"remove_lemma:{lemma_name}",
            content=new_content,
            replay_from_doc_id=resolved_doc_id,
        )
        result["lemma_name"] = lemma_name
        result["removed_block"] = _join_lines(removed_lines, trailing_newline=False)
        return result
