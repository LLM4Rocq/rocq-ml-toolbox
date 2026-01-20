"""Dataclasses and interfaces shared by the parser components."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from functools import cached_property

from pytanque.protocol import Range, Position, Goal
from .ast.model import VernacElement

@dataclass
class Step:
    """Single proof step together with its state transitions."""

    step: str
    goals: List[Goal]
    dependencies: List[VernacElement]

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Step:
        """Build a proof step from a dictionary representation."""
        return cls(
            step=d["step"],
            goals=[Goal.from_json(x) for x in d["goals"]],
            dependencies=[VernacElement.from_json(x) for x in d["dependencies"]]
        )

    def to_json(self) -> Any:
        return {
            "step": self.step,
            "goals": [goal.to_json() for goal in self.goals],
            "dependencies": [dep.to_json() for dep in self.dependencies]
        }

@dataclass
class Theorem:
    """Single proof step together with its state transitions."""
    steps: List[Step]
    initial_goals: List[Goal]
    element: VernacElement

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Step:
        """Build a proof step from a dictionary representation."""
        return cls(
            steps=[Step.from_json(step) for step in d["steps"]],
            initial_goals=[Goal.from_json(goal) for goal in d['initial_goals']],
            element=VernacElement.from_json(d['element'])
        )
    
    def to_json(self) -> Any:
        return {
            "steps": [step.to_json() for step in self.steps],
            "initial_goals": [goal.to_json() for goal in self.initial_goals],
            "element": self.element.to_json()
        }

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
            logical_path=d["logical_path"] if "logical_path" in d else None,
        )
    
    def to_json(self) -> Any:
        return {
            "path": self.path,
            "content": self.content,
            "logical_path": self.logical_path
        }

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