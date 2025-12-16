"""Dataclasses and interfaces shared by the parser components."""

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, asdict

from typing import List, Dict, Any

@dataclass
class Position:
    """Location in a source file."""

    line: int
    character: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Position":
        """Build a position from a JSON-friendly dictionary."""
        return cls(line=int(d["line"]), character=int(d["character"]))

@dataclass
class Range:
    """Range within a source file."""

    start: Position
    end: Position

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Range":
        """Build a range from a JSON-friendly dictionary."""
        return cls(start=Position.from_dict(d["start"]),
                   end=Position.from_dict(d["end"]))

@dataclass
class Element:
    """Theorem or lemma metadata extracted from Coq."""

    origin: str
    name: str
    statement: str
    range: Range

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Element":
        """Build an element from a dictionary representation."""
        return cls(
            origin=d["origin"],
            name=d["name"],
            statement=d["statement"],
            range=Range.from_dict(d["range"]),
        )

@dataclass
class Dependency:
    """Reference to a premise or hypothesis used in a proof step."""

    origin: str
    name: str
    range: Range
    kind: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Dependency":
        """Build a dependency from a dictionary representation."""
        return cls(
            origin=d["origin"],
            name=d["name"],
            range=Range.from_dict(d["range"]),
            kind=d["kind"],
        )

@dataclass
class Step:
    """Single proof step together with its state transitions."""

    step: str
    state_in: Any
    state_out: Any
    dependencies: List[Dependency]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step":
        """Build a proof step from a dictionary representation."""
        return cls(
            step=d["step"],
            state_in=d["state_in"],
            state_out=d["state_out"],
            dependencies=[Dependency.from_dict(x) for x in d["dependencies"]],
        )

@dataclass
class Source:
    """Raw Coq source file plus helper accessors."""

    path: Path
    content: str
    @property
    def content_lines(self) -> List[str]:
        """Return the source as a list of lines."""
        return self.content.splitlines()
    
    def to_dict(self) -> dict:
        """Serialize the source for JSON output."""
        d = asdict(self)
        d["path"] = str(d["path"])
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Source":
        """Build a source from a dictionary representation."""
        p = d.get("path")
        return cls(
            path=p if isinstance(p, Path) else Path(p),
            content=d["content"],
        )

class ProofNotFound(Exception):
    """Raised when a completed proof script cannot be located."""
    pass

class TimeOut(Exception):
    """Raised when a parser action takes too long."""
    pass

def update_statement(theorem: Element, source: Source):
    """Populate the theorem statement text from its source range."""
    try:
        lines = source.content_lines[theorem.range.start.line: theorem.range.end.line+1]
        lines[0] = lines[0][theorem.range.start.character:]
        lines[-1] = lines[-1][:theorem.range.end.character]
        theorem.statement = "\n".join(lines)
    except IndexError:
        print(f"Failed to extract statement from {theorem} in {source.path}")

class AbstractParser(ABC):
    """Interface implemented by proof parsers."""

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def __call__(self, theorem: Element, source: Source) -> List[Step]:
        """Parse proof steps for the given theorem."""
        pass
