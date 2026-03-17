from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class FailureCode(StrEnum):
    MISSING_OBLIGATION = "missing_obligation"
    DUPLICATE_NAME = "duplicate_name"
    STATEMENT_MISMATCH = "statement_mismatch"
    INCOMPLETE_PROOF = "incomplete_proof"
    DISALLOWED_AXIOMS = "disallowed_axioms"
    NEW_LOCAL_AXIOM = "new_local_axiom"
    PARSE_OR_COMPILE_ERROR = "parse_or_compile_error"


@dataclass(frozen=True)
class Obligation:
    name: str
    source_proof_id: int
    source_logical_name: str
    source_start_line: int
    source_start_character: int

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_proof_id": self.source_proof_id,
            "source_logical_name": self.source_logical_name,
            "source_start": {
                "line": self.source_start_line,
                "character": self.source_start_character,
            },
        }


@dataclass
class CheckOutcome:
    obligation: Obligation
    matched_target: str | None
    checks: dict[str, bool]
    failure_codes: list[FailureCode] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failure_codes

    def add_failure(self, code: FailureCode, details: Any | None = None) -> None:
        if code not in self.failure_codes:
            self.failure_codes.append(code)
        if details is not None:
            self.details[code.value] = details

    def to_json(self) -> dict[str, Any]:
        return {
            "obligation": self.obligation.to_json(),
            "matched_target": self.matched_target,
            "checks": self.checks,
            "ok": self.ok,
            "failure_codes": [code.value for code in self.failure_codes],
            "details": self.details,
        }


@dataclass
class GlobalFailure:
    code: FailureCode
    details: Any

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "details": self.details,
        }


@dataclass
class VerificationReport:
    source_path: str
    target_path: str
    root: str
    config: dict[str, Any]
    outcomes: list[CheckOutcome] = field(default_factory=list)
    global_failures: list[GlobalFailure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        if self.global_failures:
            return False
        return all(outcome.ok for outcome in self.outcomes)

    def add_global_failure(self, code: FailureCode, details: Any) -> None:
        self.global_failures.append(GlobalFailure(code=code, details=details))

    def summary(self) -> dict[str, int]:
        total = len(self.outcomes)
        failed = sum(1 for outcome in self.outcomes if not outcome.ok)
        passed = total - failed
        return {
            "num_obligations": total,
            "passed": passed,
            "failed": failed,
            "global_failures": len(self.global_failures),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source": self.source_path,
            "target": self.target_path,
            "root": self.root,
            "config": self.config,
            "summary": self.summary(),
            "global_failures": [x.to_json() for x in self.global_failures],
            "outcomes": [x.to_json() for x in self.outcomes],
        }

    def save_json(self, path: str | Path) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
