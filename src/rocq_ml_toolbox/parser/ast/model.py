from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Optional, Dict
from pytanque.protocol import Range

@dataclass
class Span:
    bp: int
    ep: int

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Span:
        """Build a Span from a dictionary representation."""
        return cls(
            bp=d['bp'],
            ep=d['ep']
        )

    def to_json(self) -> Any:
        return {
            "bp": self.bp,
            "ep": self.ep
        }


@dataclass(kw_only=True)
class AstNode:
    span: Optional[Span]=None
    range: Optional[Range]=None
    name: Optional[str]=None

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Span:
        """Build an AstNode from a dictionary representation."""
        return cls(
            span=Span.from_json(d['span']) if 'span' in d else None,
            range=Range.from_json(d['range']) if 'range' in d else None,
            name=d['name'] if 'name' in d else None
        )

    def to_json(self) -> Any:
        return {
            "span": self.span.to_json(),
            "range": self.range.to_json(),
            "name": self.name
        }

class VernacKind(StrEnum):
    REQUIRE = auto()
    IMPORT = auto()
    OPEN_CLOSE_SCOPE = auto()
    SET_OPTION = auto()
    DEFINE_MODULE = auto()
    EXTEND = auto()
    BIND_SCOPE = auto()
    END_SEGMENT = auto()
    BEGIN_SECTION = auto()
    CONTEXT = auto()
    DEFINITION = auto()
    ARGUMENTS = auto()
    SYNTACTIC_DEFINITION = auto()
    RESERVE = auto()
    START_THEOREM_PROOF = auto()
    PROOF = auto()
    END_PROOF = auto()
    ABORT = auto()
    ASSUMPTION = auto()
    HINTS = auto()
    BULLET = auto()
    NOTATION = auto()
    RESERVED_NOTATION = auto()
    FIXPOINT = auto()
    COFIXPOINT = auto()
    DECLARE_SCOPE = auto()
    DELIMITERS = auto()
    COERCION = auto()
    CANONICAL = auto()
    INSTANCE = auto()
    INCLUDE = auto()
    DECLARE_CUSTOM_ENTRY = auto()
    IDENTITY_COERCION = auto()
    DECLARE_MODULE_TYPE = auto()
    DECLARE_MODULE = auto()
    ATTRIBUTES = auto()
    REMOVE_HINTS = auto()
    SUBPROOF = auto()
    END_SUBPROOF = auto()
    DECLARE_INSTANCE = auto()
    CREATE_HINT_DB = auto()
    DECLARE_ML_MODULE = auto()
    EXISTING_INSTANCE = auto()
    SCHEME = auto()
    SET_OPACITY = auto()
    EXACT_PROOF = auto()
    REGISTER = auto()
    SET_STRATEGY = auto()
    GENERALIZABLE = auto()
    EXTRA_DEPENDENCY = auto()
    ADD_OPTION = auto()
    PROOF_STEP = auto()  # used for parsed "extend" proof steps etc.
    UNKNOWN = auto()
    INDUCTIVE = auto()
    COINDUCTIVE = auto()
    RECORD = auto()
    STRUCTURE = auto()
    VARIANT = auto()
    CLASS = auto()
    LTAC = auto()
    CONSTANT = auto()
    FIELD = auto()
    CONSTRUCTOR = auto()
    HB = auto() # TODO: right now, we don't get it from the AST
    

@dataclass
class VernacElement(AstNode):
    kind: VernacKind
    members: Optional[list[VernacElement]]=field(default_factory=list)
    data: dict[str, Any]=field(default_factory=dict)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> VernacElement:
        return cls(
            span=Span.from_json(d["span"]) if d["span"] else None,
            range=Range.from_json(d["range"]) if d["range"] else None,
            name=d["name"],
            kind=VernacKind(d["kind"]),
            members=[cls.from_json(el) for el in d.get("members", [])],
            data=d.get("data", {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "span": self.span.to_json() if self.span else None,
            "range": self.range.to_json() if self.range else None,
            "name": self.name,
            "kind": self.kind.value,
            "members": [el.to_json() for el in self.members],
            "data": self.data,
        }


@dataclass(kw_only=True)
class UnsupportedNode(VernacElement):
    keyword: str
    raw: dict[str, Any] | list[Any] | Any

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> UnsupportedNode:
        return cls(
            span=Span.from_json(d["span"]) if "span" in d else None,
            range=Range.from_json(d["range"]) if "range" in d else None,
            name=d["name"],
            kind=VernacKind(d["kind"]),
            members=[VernacElement.from_json(el) for el in d.get("members", [])],
            data=d.get("data", {}),
            keyword=d["keyword"],
            raw=d["raw"],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "span": self.span.to_json() if self.span else None,
            "range": self.range.to_json() if self.range else None,
            "name": self.name,
            "kind": self.kind.value,
            "members": [el.to_json() for el in self.members],
            "data": self.data,
            "keyword": self.keyword,
            "raw": self.raw,
        }