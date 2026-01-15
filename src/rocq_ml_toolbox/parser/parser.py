"""Dataclasses and interfaces shared by the parser components."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from functools import cached_property

from pytanque.protocol import Range, Position
from .ast.model import VernacElement

@dataclass
class Dependency:
    """Reference to a premise or hypothesis used in a proof step."""

    element: VernacElement
    range: Range

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Dependency:
        """Build a dependency from a dictionary representation."""
        raise Exception("from_json not implemented for VernacElement")
        return cls(
            element=VernacElement.from_json(d["element"]),
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
    element: VernacElement

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
    logical_path: Optional[str]=None
    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Source:
        """Build a source from a dictionary representation."""
        return cls(
            path=d["path"],
            content=d["content"],
            logical_path=d["logical_path"],
        )

    @classmethod
    def from_local_path(cls, path: str) -> Source:
        """Build a source from a local path."""
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        return cls(
            path=path,
            content=content
        )

    @cached_property
    def content_utf8(self) -> bytes:
        return self.content.encode("utf-8")

class ParserError(Exception):
    """Base class for parser error"""
    pass

class ProofNotFound(ParserError):
    """Raised when a completed proof script cannot be located."""
    pass

class TimeOut(ParserError):
    """Raised when a parser action takes too long."""
    pass