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
LEMMA_DECL_PREFIX_RE = re.compile(r"^\s*(Lemma|Theorem|Fact|Proposition|Corollary)\b")
LEMMA_PROOF_TOKEN_RE = re.compile(r"\b(Proof|Qed|Admitted|Defined|Abort)\b")
FORBIDDEN_TACTIC_PREFIX_RE = re.compile(
    r"^\s*(Lemma|Theorem|Fact|Proposition|Corollary|From|Require|Import|Export|Section|End|Module|"
    r"Definition|Fixpoint|CoFixpoint|Record|Inductive|CoInductive|Class|Instance|Notation|Ltac|Axiom|"
    r"Hypothesis|Variable|Context|Qed|Admitted|Defined|Abort)\b"
)
PLACEHOLDER_TACTIC_RE = re.compile(r"\b(admit|admitted)\b", re.IGNORECASE)
SHELVING_TACTIC_RE = re.compile(r"\b(shelve_unifiable|unshelve|shelve)\b", re.IGNORECASE)
MutationValidator = Callable[[int, str], tuple[bool, str | None]]
UNICODE_LOGIC_TOKENS = ("∀", "∃", "→", "↔", "⇒", "⇔", "∧", "∨", "≤", "≥", "≠", "¬")
MISSING_IN_ENV_RE = re.compile(
    r"(reference|variable)\s+.+\s+was not found in the current environment",
    re.IGNORECASE,
)


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


def _tactic_error_hint(error: str) -> str | None:
    if MISSING_IN_ENV_RE.search(error):
        return (
            "A referenced constant/tactic is missing in the current environment. "
            "If this comes from a library, use `explore_toc` + `read_source_file` to find the module, "
            "then add it via `add_import` (main) or `require_import` (subagent). "
            "Do not guess import roots; derive `libname`/`source` from TOC entries."
        )
    return None


def _normalize_error_for_loop_guard(error: str) -> str:
    text = re.sub(r"\s+", " ", error.strip().lower())
    if len(text) > 500:
        return text[:500]
    return text


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

    stmt_in_source = FROM_REQUIRE_IMPORT_RE.match(src)
    if stmt_in_source:
        return stmt_in_source.group(1), stmt_in_source.group(2)

    stmt_in_libname = FROM_REQUIRE_IMPORT_RE.match(lib)
    if stmt_in_libname:
        return stmt_in_libname.group(1), stmt_in_libname.group(2)

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
    _last_failure_signature: tuple[int, str] | None = None
    _last_failure_count: int = 0

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

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
        self._log(f"get_goals(doc_id={self.doc_id}, state={state_index}) -> {len(pretty_goals)} goals")
        return {
            "state_index": state_index,
            "goals": pretty_goals,
            "proof_finished": len(pretty_goals) == 0,
        }

    def _pretty_goals(self, state_index: int) -> list[str]:
        node = self._state_node(state_index)
        goals = self.client.goals(node.state, timeout=self.timeout)
        return [getattr(goal, "pp", None) or getattr(goal, "ty", "") for goal in goals]

    def run_tac(self, state_index: int, tactic: str) -> dict[str, Any]:
        node = self._state_node(state_index)
        self._log(f"run_tac(doc_id={self.doc_id}, state={state_index}, tactic={tactic!r})")
        stripped_tactic = tactic.strip()
        if stripped_tactic.startswith("`") or stripped_tactic.endswith("`"):
            error = (
                "run_tac received markdown-wrapped tactic. "
                "Pass raw tactic text without backticks, e.g. `intro n.` not \"`intro n.`\"."
            )
            self._log(f"run_tac rejected(doc_id={self.doc_id}, state={state_index}): {error}")
            return {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "error": error,
            }
        if FORBIDDEN_TACTIC_PREFIX_RE.match(tactic):
            error = (
                "run_tac only accepts proof tactics, not top-level vernac commands. "
                "Use add_import/remove_import and add_intermediate_lemma/remove_intermediate_lemma tools."
            )
            self._log(f"run_tac rejected(doc_id={self.doc_id}, state={state_index}): {error}")
            return {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "error": error,
            }
        if PLACEHOLDER_TACTIC_RE.search(tactic):
            error = (
                "run_tac rejected placeholder tactic (`admit`/`Admitted`). "
                "Provide a real proof step."
            )
            self._log(f"run_tac rejected(doc_id={self.doc_id}, state={state_index}): {error}")
            return {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "error": error,
            }
        if SHELVING_TACTIC_RE.search(tactic):
            error = (
                "run_tac rejected shelving tactics (`shelve`/`Unshelve`). "
                "These can hide unresolved obligations and produce non-replayable proofs. "
                "Use real proof steps or branch/lemma tools instead."
            )
            self._log(f"run_tac rejected(doc_id={self.doc_id}, state={state_index}): {error}")
            return {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "error": error,
            }
        try:
            new_state = self.client.run(node.state, tactic, timeout=self.timeout)
        except Exception as exc:
            error = str(exc)
            hint = _tactic_error_hint(error)
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
            out = {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
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
        pretty_goals = [getattr(goal, "pp", None) or getattr(goal, "ty", "") for goal in goals]
        feedback = _format_state_feedback(new_state)
        self._log(f"run_tac ok(doc_id={self.doc_id}) -> new_state_index={new_idx}, goals={len(pretty_goals)}")
        return {
            "ok": True,
            "doc_id": self.doc_id,
            "source_state_index": state_index,
            "new_state_index": new_idx,
            "goals": pretty_goals,
            "feedback": feedback,
            "feedback_count": len(feedback),
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
        """Return source content with current proof script materialized as `Qed.`."""
        target = self.latest_state_index if state_index is None else state_index
        lines = list(self.layout.lines)
        proof_line = self.layout.proof_line
        end_line = self.layout.target_end_line

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
        rebuilt = lines[: proof_line + 1] + script_lines + ["Qed."] + lines[end_line + 1 :]
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
    ):
        self.client = client
        self.source_path = Path(source_path).resolve()
        self.timeout = timeout
        self.logger = logger
        self.mutation_validator = mutation_validator

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
        for doc_id, session in self.sessions.items():
            if session.source_path.resolve() == resolved:
                return doc_id
        return None

    def _create_mutation(self, *, base_doc_id: int, label: str, content: str) -> dict[str, Any]:
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
        if self.mutation_validator is not None:
            ok, error = self.mutation_validator(new_doc_id, label)
            if not ok:
                self.nodes.pop(new_doc_id, None)
                self.sessions.pop(new_doc_id, None)
                self._next_doc_id = new_doc_id
                self.head_doc_id = base_doc_id
                detail = error or "fresh-session validation failed"
                raise ValueError(f"Mutation `{label}` rejected by fresh-session validation: {detail}")
        self._log(f"new doc node: {new_doc_id} (from {base_doc_id}) label={label!r} branched={branched}")
        return {
            "ok": True,
            "doc_id": new_doc_id,
            "base_doc_id": base_doc_id,
            "branched": branched,
            "head_doc_id": self.head_doc_id,
        }

    def list_states(self, *, doc_id: int | None = None) -> list[int]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        return self.sessions[resolved_doc_id].available_state_indexes

    def list_states_verbose(self, *, doc_id: int | None = None) -> list[dict[str, Any]]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        return self.sessions[resolved_doc_id].list_states()

    def get_goals(self, *, state_index: int = 0, doc_id: int | None = None) -> dict[str, Any]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        return self.sessions[resolved_doc_id].get_goals(state_index=state_index)

    def run_tac(self, *, state_index: int, tactic: str, doc_id: int | None = None) -> dict[str, Any]:
        resolved_doc_id = self._resolve_existing_doc(doc_id)
        session = self.sessions[resolved_doc_id]
        try:
            return session.run_tac(state_index=state_index, tactic=tactic)
        except ValueError as exc:
            raise ValueError(
                f"{exc} (doc_id={resolved_doc_id}, current_head={self.head_doc_id}, "
                f"available_states={session.available_state_indexes})"
            ) from exc

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
        return {
            "doc_id": resolved_doc_id,
            "head_doc_id": self.head_doc_id,
            "parent_doc_id": node.parent_doc_id,
            "label": node.label,
            "source_path": str(session.source_path),
            "states": session.list_states(),
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
        has_placeholder = session.has_placeholder_tactic(latest_state_index)
        solved_state_indexes: list[int] = []
        for idx in session.available_state_indexes:
            if idx == latest_state_index:
                done = len(latest_goals) == 0 and not has_placeholder
            else:
                done = len(session._pretty_goals(idx)) == 0
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
            "latest_proof_finished": len(latest_goals) == 0 and not has_placeholder,
            "latest_has_placeholder_tactic": has_placeholder,
            "solved_state_indexes": solved_state_indexes,
            "available_state_indexes": session.available_state_indexes,
            "latest_goal_preview": preview,
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
        prefix.insert(idx, statement)
        new_content = _join_lines(prefix + layout.target_lines + layout.suffix_lines, trailing_newline=True)

        result = self._create_mutation(
            base_doc_id=resolved_doc_id,
            label=f"add_import:{libname}.{source}",
            content=new_content,
        )
        result["statement"] = statement
        self._log(f"add_import ok(doc_id={result['doc_id']}, statement={statement!r})")
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
        )
        result["removed_statement"] = removed
        self._log(f"remove_import ok(doc_id={result['doc_id']}, statement={removed!r})")
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
        )
        result["lemma_name"] = lemma_name
        result["removed_block"] = _join_lines(removed_lines, trailing_newline=False)
        return result
