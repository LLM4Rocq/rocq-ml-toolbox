"""Dataclasses and interfaces shared by the parser components."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from pytanque.protocol import Range, Position

@dataclass
class Element:
    """Element extracted from source file."""

    name: str
    fqn: str
    range: Range
    content: Optional[str]=None
    kind: Optional[str]=None
    physical_path: Optional[str]=None
    logical_path: Optional[str]=None
    children: Optional[Element]=None
    
    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Element:
        """Build an element from a dictionary representation."""
        return cls(
            physical_path=d.get('physical_path', None),
            logical_path=d.get('logical_path', None),
            name=d["name"],
            range=Range.from_json(d["range"]),
            content=d.get('content', None),
            kind=d.get('kind', None),
            children=d.get('children', None)
        )

@dataclass
class Dependency:
    """Reference to a premise or hypothesis used in a proof step."""

    element: Element
    range: Range

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Dependency:
        """Build a dependency from a dictionary representation."""
        return cls(
            element=Element.from_json(d["element"]),
            range=Range.from_json(d["range"])
        )

@dataclass
class Step:
    """Single proof step together with its state transitions."""

    step: str
    goals: Any
    dependencies: List[Dependency]

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Step:
        """Build a proof step from a dictionary representation."""
        return cls(
            step=d["step"],
            state_in=d["state_in"],
            state_out=d["state_out"],
            dependencies=[Dependency.from_json(x) for x in d["dependencies"]],
        )

@dataclass
class Theorem:
    """Single proof step together with its state transitions."""
    steps: List[Step]
    initial_goals: Any
    element: Element

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Step:
        """Build a proof step from a dictionary representation."""
        return cls(
            step=d["step"],
            goals=d["goals"],
            dependencies=[Dependency.from_json(x) for x in d["dependencies"]],
        )

@dataclass
class Source:
    """Source file plus helper accessors."""

    path: str
    content: str

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Source:
        """Build a source from a dictionary representation."""
        return cls(
            path=d["path"],
            content=d["content"],
        )

class ParserError(Exception):
    """Base class for parser error"""
    pass

class ProofNotFound(ParserError):
    """Raised when a completed proof script cannot be located."""
    pass

class TimeOut(ParserError):
    """Raised when a parser action takes too long."""
    pass