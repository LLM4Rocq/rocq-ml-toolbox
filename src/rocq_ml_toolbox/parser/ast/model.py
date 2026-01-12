from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from pytanque.protocol import Range

@dataclass
class Span:
    bp: int
    ep: int

@dataclass(kw_only=True)
class AstNode:
    span: Optional[Span]=None
    range: Optional[Range]=None
    name: Optional[str]=None

class VernacKind(Enum):
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

@dataclass
class VernacElement(AstNode):
    kind: VernacKind
    members: Optional[list[VernacElement]]=field(default_factory=list)
    data: Optional[dict[str, Any]]=field(default_factory=dict)

@dataclass(kw_only=True)
class UnsupportedNode(VernacElement):
    keyword: str
    raw: dict[str, Any] | list[Any] | Any
