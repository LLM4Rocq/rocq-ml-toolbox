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


def _join_lines(lines: list[str], *, trailing_newline: bool = True) -> str:
    text = "\n".join(lines)
    if trailing_newline and not text.endswith("\n"):
        text += "\n"
    return text


def _normalize_lemma_type(lemma_type: str) -> str:
    out = lemma_type.strip()
    if out.endswith("."):
        out = out[:-1].rstrip()
    return out


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
    layout: ParsedDocumentLayout
    timeout: float
    nodes: list[StateNode] = field(default_factory=list)
    logger: Callable[[str], None] | None = None

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
        node = self._state_node(state_index)
        goals = self.client.goals(node.state, timeout=self.timeout)
        pretty_goals = [getattr(goal, "pp", None) or getattr(goal, "ty", "") for goal in goals]
        return {
            "state_index": state_index,
            "goals": pretty_goals,
            "proof_finished": len(pretty_goals) == 0,
        }

    def run_tac(self, state_index: int, tactic: str) -> dict[str, Any]:
        node = self._state_node(state_index)
        self._log(f"run_tac(doc_id={self.doc_id}, state={state_index}, tactic={tactic!r})")
        try:
            new_state = self.client.run(node.state, tactic, timeout=self.timeout)
        except Exception as exc:
            return {
                "ok": False,
                "doc_id": self.doc_id,
                "source_state_index": state_index,
                "error": str(exc),
            }

        new_idx = len(self.nodes)
        self.nodes.append(StateNode(index=new_idx, parent_index=state_index, tactic=tactic, state=new_state))
        goals = self.client.goals(new_state, timeout=self.timeout)
        pretty_goals = [getattr(goal, "pp", None) or getattr(goal, "ty", "") for goal in goals]
        return {
            "ok": True,
            "doc_id": self.doc_id,
            "source_state_index": state_index,
            "new_state_index": new_idx,
            "goals": pretty_goals,
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
    ):
        self.client = client
        self.source_path = Path(source_path).resolve()
        self.timeout = timeout
        self.logger = logger

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

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def _new_branch_session(self, doc_id: int, content: str) -> BranchSession:
        tmp_path = Path(
            self.client.tmp_file(content=content, root=str(self.source_path.parent)),
        ).resolve()
        layout = parse_last_target_layout(content)
        state0 = self.client.get_state_at_pos(
            str(tmp_path),
            layout.proof_line,
            layout.proof_character,
            timeout=self.timeout,
        )
        return BranchSession(
            client=self.client,
            doc_id=doc_id,
            source_path=tmp_path,
            layout=layout,
            timeout=self.timeout,
            nodes=[StateNode(index=0, parent_index=None, tactic=None, state=state0)],
            logger=self.logger,
        )

    def _ensure_head_doc(self, doc_id: int | None) -> int:
        if doc_id is None:
            return self.head_doc_id
        if doc_id != self.head_doc_id:
            raise ValueError(
                f"Stale document reference doc_id={doc_id}; current_head={self.head_doc_id}. "
                f"No manual checkout in v1."
            )
        return doc_id

    def _resolve_doc_for_mutation(self, doc_id: int | None) -> int:
        if doc_id is None:
            return self.head_doc_id
        if doc_id not in self.nodes:
            raise ValueError(f"Unknown doc_id={doc_id}.")
        return doc_id

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
        self._log(f"new doc node: {new_doc_id} (from {base_doc_id}) label={label!r} branched={branched}")
        return {
            "ok": True,
            "doc_id": new_doc_id,
            "base_doc_id": base_doc_id,
            "branched": branched,
            "head_doc_id": self.head_doc_id,
        }

    def list_states(self, *, doc_id: int | None = None) -> list[int]:
        resolved_doc_id = self._ensure_head_doc(doc_id)
        return self.sessions[resolved_doc_id].available_state_indexes

    def list_states_verbose(self, *, doc_id: int | None = None) -> list[dict[str, Any]]:
        resolved_doc_id = self._ensure_head_doc(doc_id)
        return self.sessions[resolved_doc_id].list_states()

    def get_goals(self, *, state_index: int = 0, doc_id: int | None = None) -> dict[str, Any]:
        resolved_doc_id = self._ensure_head_doc(doc_id)
        return self.sessions[resolved_doc_id].get_goals(state_index=state_index)

    def run_tac(self, *, state_index: int, tactic: str, doc_id: int | None = None) -> dict[str, Any]:
        resolved_doc_id = self._ensure_head_doc(doc_id)
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
        resolved_doc_id = self._ensure_head_doc(doc_id)
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
        resolved_doc_id = self._ensure_head_doc(doc_id)
        node = self.nodes[resolved_doc_id]
        session = self.sessions[resolved_doc_id]
        return {
            "doc_id": resolved_doc_id,
            "head_doc_id": self.head_doc_id,
            "parent_doc_id": node.parent_doc_id,
            "label": node.label,
            "states": session.list_states(),
            "content": _format_with_line_numbers(node.content),
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

    def add_import(self, *, libname: str, source: str, doc_id: int | None = None) -> dict[str, Any]:
        if not IDENT_RE.match(libname):
            raise ValueError(f"Invalid libname={libname!r}.")
        if not IDENT_RE.match(source):
            raise ValueError(f"Invalid source={source!r}.")

        resolved_doc_id = self._resolve_doc_for_mutation(doc_id)
        base = self.nodes[resolved_doc_id].content
        layout = parse_last_target_layout(base)

        statement = f"From {libname} Require Import {source}."
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
        _ = self._new_branch_session(doc_id=-1, content=probe_content)

        layout = parse_last_target_layout(base_content)
        lines = list(layout.prefix_lines)
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([lemma_statement, "Proof.", "Admitted."])
        if layout.suffix_lines:
            lines.append("")
            lines.extend(layout.suffix_lines)
        sub_content = _join_lines(lines, trailing_newline=True)
        sub_session = self._new_branch_session(doc_id=-1, content=sub_content)
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
