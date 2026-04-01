"""Parser for the `proofs` section of `.json.proofdepsdump` files.

This intentionally ignores AST payload parsing for now (`astdump_jsonl`).
It reuses core dataclasses from `pytanque.protocol` (notably `Position`,
`Range`, and `GoalHyp`) and defines only the extra structures needed for the
proof-dependency dump schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import json

from pytanque.protocol import GoalHyp, Range


def _expect_dict(type_name: str, x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    raise ValueError(f"{type_name} expects a JSON object, got: {x!r}")


def _expect_list(type_name: str, x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    raise ValueError(f"{type_name} expects a JSON array, got: {x!r}")


def _expect_int(type_name: str, field_name: str, x: Any) -> int:
    if isinstance(x, int):
        return x
    raise ValueError(f"{type_name}.{field_name} expects int, got: {x!r}")


def _expect_str(type_name: str, field_name: str, x: Any) -> str:
    if isinstance(x, str):
        return x
    raise ValueError(f"{type_name}.{field_name} expects str, got: {x!r}")


@dataclass
class ProofDepLocation:
    range: Range

    @classmethod
    def from_json(cls, x: Any) -> "ProofDepLocation":
        x = _expect_dict("ProofDepLocation", x)
        if "range" not in x:
            raise ValueError("ProofDepLocation is missing field 'range'")
        return cls(range=Range.from_json(x["range"]))

    def to_json(self) -> dict[str, Any]:
        return {"range": self.range.to_json()}


@dataclass
class ProofDependency:
    name: str
    logical_path: str
    physical_path: Optional[str] = None
    locations: list[ProofDepLocation] = field(default_factory=list)

    @classmethod
    def from_json(cls, x: Any) -> "ProofDependency":
        # Backward compatibility with older dumps where deps were plain strings.
        if isinstance(x, str):
            return cls(name=x, logical_path="", locations=[])

        x = _expect_dict("ProofDependency", x)
        if "name" not in x:
            raise ValueError("ProofDependency is missing field 'name'")

        locations_raw = x.get("locations", [])
        locations = [ProofDepLocation.from_json(v) for v in _expect_list("locations", locations_raw)]

        return cls(
            name=_expect_str("ProofDependency", "name", x["name"]),
            logical_path=_expect_str("ProofDependency", "logical_path", x.get("logical_path", "")),
            physical_path=(
                None
                if x.get("physical_path") is None
                else _expect_str("ProofDependency", "physical_path", x["physical_path"])
            ),
            locations=locations,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "logical_path": self.logical_path,
            "physical_path": self.physical_path,
            "locations": [loc.to_json() for loc in self.locations],
        }


@dataclass
class ProofGoal:
    evar: int
    name: Optional[str]
    hyps: list[GoalHyp]
    ty: str

    @classmethod
    def from_json(cls, x: Any) -> "ProofGoal":
        x = _expect_dict("ProofGoal", x)
        for field_name in ("evar", "hyps", "ty"):
            if field_name not in x:
                raise ValueError(f"ProofGoal is missing field '{field_name}'")
        return cls(
            evar=_expect_int("ProofGoal", "evar", x["evar"]),
            name=x["name"] if x.get("name") is None else _expect_str("ProofGoal", "name", x["name"]),
            hyps=[GoalHyp.from_json(h) for h in _expect_list("ProofGoal.hyps", x["hyps"])],
            ty=_expect_str("ProofGoal", "ty", x["ty"]),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "evar": self.evar,
            "name": self.name,
            "hyps": [h.to_json() for h in self.hyps],
            "ty": self.ty,
        }


@dataclass
class ProofGoalState:
    goals: list[ProofGoal]
    stack: list[tuple[list[ProofGoal], list[ProofGoal]]]
    bullet: Optional[str]
    shelf: list[ProofGoal]
    given_up: list[ProofGoal]

    @classmethod
    def from_json(cls, x: Any) -> "ProofGoalState":
        x = _expect_dict("ProofGoalState", x)
        for field_name in ("goals", "stack", "shelf", "given_up"):
            if field_name not in x:
                raise ValueError(f"ProofGoalState is missing field '{field_name}'")

        stack: list[tuple[list[ProofGoal], list[ProofGoal]]] = []
        for pair in _expect_list("ProofGoalState.stack", x["stack"]):
            pair_list = _expect_list("ProofGoalState.stack pair", pair)
            if len(pair_list) != 2:
                raise ValueError(
                    f"ProofGoalState.stack pair expects length 2, got: {pair_list!r}"
                )
            left = [ProofGoal.from_json(g) for g in _expect_list("ProofGoalState.stack[0]", pair_list[0])]
            right = [ProofGoal.from_json(g) for g in _expect_list("ProofGoalState.stack[1]", pair_list[1])]
            stack.append((left, right))

        bullet_raw = x.get("bullet")
        bullet = None if bullet_raw is None else _expect_str("ProofGoalState", "bullet", bullet_raw)

        return cls(
            goals=[ProofGoal.from_json(g) for g in _expect_list("ProofGoalState.goals", x["goals"])],
            stack=stack,
            bullet=bullet,
            shelf=[ProofGoal.from_json(g) for g in _expect_list("ProofGoalState.shelf", x["shelf"])],
            given_up=[ProofGoal.from_json(g) for g in _expect_list("ProofGoalState.given_up", x["given_up"])],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "goals": [g.to_json() for g in self.goals],
            "stack": [[[g.to_json() for g in l], [g.to_json() for g in r]] for (l, r) in self.stack],
            "bullet": self.bullet,
            "shelf": [g.to_json() for g in self.shelf],
            "given_up": [g.to_json() for g in self.given_up],
        }


@dataclass
class ProofStep:
    index: int
    range: Range
    raw: str
    tactic_tags: list[str] = field(default_factory=list)
    notations: list[ProofDependency] = field(default_factory=list)
    deps: list[ProofDependency] = field(default_factory=list)
    goals_after: Optional[ProofGoalState] = None

    @classmethod
    def from_json(cls, x: Any) -> "ProofStep":
        x = _expect_dict("ProofStep", x)
        for field_name in ("index", "range", "raw"):
            if field_name not in x:
                raise ValueError(f"ProofStep is missing field '{field_name}'")

        tactic_tags_raw = x.get("tactic_tags", [])
        notations_raw = x.get("notations", [])
        deps_raw = x.get("deps", [])
        goals_after_raw = x.get("goals_after")

        return cls(
            index=_expect_int("ProofStep", "index", x["index"]),
            range=Range.from_json(x["range"]),
            raw=_expect_str("ProofStep", "raw", x["raw"]),
            tactic_tags=[_expect_str("ProofStep", "tactic_tags[]", t) for t in _expect_list("ProofStep.tactic_tags", tactic_tags_raw)],
            notations=[ProofDependency.from_json(n) for n in _expect_list("ProofStep.notations", notations_raw)],
            deps=[ProofDependency.from_json(d) for d in _expect_list("ProofStep.deps", deps_raw)],
            goals_after=None if goals_after_raw is None else ProofGoalState.from_json(goals_after_raw),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "range": self.range.to_json(),
            "raw": self.raw,
            "tactic_tags": list(self.tactic_tags),
            "notations": [n.to_json() for n in self.notations],
            "deps": [d.to_json() for d in self.deps],
            "goals_after": None if self.goals_after is None else self.goals_after.to_json(),
        }


@dataclass
class ProofEntry:
    proof_id: int
    name: str
    start_range: Range
    statement: str
    statement_notations: list[ProofDependency]
    axioms: list[ProofDependency]
    initial_goals: Optional[ProofGoalState]
    steps: list[ProofStep]

    @classmethod
    def from_json(cls, x: Any) -> "ProofEntry":
        x = _expect_dict("ProofEntry", x)
        for field_name in ("proof_id", "name", "start_range", "steps"):
            if field_name not in x:
                raise ValueError(f"ProofEntry is missing field '{field_name}'")

        initial_goals_raw = x.get("initial_goals")
        steps = [ProofStep.from_json(s) for s in _expect_list("ProofEntry.steps", x["steps"])]

        statement_raw = x.get("statement")
        if statement_raw is None:
            statement = steps[0].raw if steps else ""
        else:
            statement = _expect_str("ProofEntry", "statement", statement_raw)

        statement_notations_raw = x.get("statement_notations")
        if statement_notations_raw is None:
            statement_notations = steps[0].notations if steps else []
        else:
            statement_notations = [
                ProofDependency.from_json(n)
                for n in _expect_list("ProofEntry.statement_notations", statement_notations_raw)
            ]

        axioms_raw = x.get("axioms", [])
        axioms = [
            ProofDependency.from_json(a)
            for a in _expect_list("ProofEntry.axioms", axioms_raw)
        ]

        return cls(
            proof_id=_expect_int("ProofEntry", "proof_id", x["proof_id"]),
            name=_expect_str("ProofEntry", "name", x["name"]),
            start_range=Range.from_json(x["start_range"]),
            statement=statement,
            statement_notations=statement_notations,
            axioms=axioms,
            initial_goals=None
            if initial_goals_raw is None
            else ProofGoalState.from_json(initial_goals_raw),
            steps=steps,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "name": self.name,
            "start_range": self.start_range.to_json(),
            "statement": self.statement,
            "statement_notations": [n.to_json() for n in self.statement_notations],
            "axioms": [a.to_json() for a in self.axioms],
            "initial_goals": None if self.initial_goals is None else self.initial_goals.to_json(),
            "steps": [s.to_json() for s in self.steps],
        }


@dataclass
class ProofDump:
    proofs: list[ProofEntry]

    @classmethod
    def from_json(cls, x: Any) -> "ProofDump":
        x = _expect_dict("ProofDump", x)
        if "proofs" not in x:
            raise ValueError("ProofDump is missing field 'proofs'")
        return cls(proofs=[ProofEntry.from_json(p) for p in _expect_list("ProofDump.proofs", x["proofs"])])

    @classmethod
    def from_json_string(cls, x: str) -> "ProofDump":
        return cls.from_json(json.loads(x))

    @classmethod
    def from_file(cls, path: str | Path) -> "ProofDump":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def to_json(self) -> dict[str, Any]:
        return {"proofs": [p.to_json() for p in self.proofs]}

    def to_json_string(self, **kw: Any) -> str:
        return json.dumps(self.to_json(), **kw)
