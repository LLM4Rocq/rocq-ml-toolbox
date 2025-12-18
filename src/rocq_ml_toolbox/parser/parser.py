"""Dataclasses and interfaces shared by the parser components."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pytanque.protocol import Range, Position
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class Element:
    """Element extracted from source file."""

    origin: str
    name: str
    fqn: str
    range: Range
    content: Optional[str]=None
    kind: Optional[str]=None

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Element:
        """Build an element from a dictionary representation."""
        return cls(
            origin=d["origin"],
            name=d["name"],
            range=Range.from_json(d["range"]),
            content=d.get('content', None),
            kind=d.get('kind', None)
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
            state_in=d["state_in"],
            state_out=d["state_out"],
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

def pos_to_offset(content: str, p: Position) -> int:
    lines = content.splitlines(keepends=True)
    if p.line < 0 or p.line >= len(lines):
        raise IndexError(f"line out of bounds: {p.line}")
    offset = sum(len(lines[i]) for i in range(p.line))
    line_no_nl = lines[p.line].rstrip("\r\n")
    if p.character < 0 or p.character > len(line_no_nl):
        raise IndexError(f"character out of bounds: {p.character} on line {p.line}")
    return offset + p.character

def extract_subtext(content: str, r: Range) -> str:
    """
    Extract substring defined by Range (line/character), where:
      - line is 0-based
      - character is 0-based index within that line
      - end is treated as exclusive
    """
    start_off = pos_to_offset(content, r.start)
    end_off = pos_to_offset(content, r.end)

    sliced = content[start_off:end_off]
    return sliced

def move_position(content: str, pos: Position, length: int) -> Position:
    """
    Move a (line, character) position by `offset` characters within `text`.
    """

    lines = content.splitlines(keepends=True)

    abs_index = pos_to_offset(content, pos) + length

    line = 0
    char = abs_index
    for l in lines:
        if char < len(l):
            break
        char -= len(l)
        line += 1

    return Position(line=line, character=char)

def update_statement(theorem: Element, source: Source):
    """Populate the theorem statement text from its source range."""
    try:
        lines = source.content_lines[theorem.range.start.line: theorem.range.end.line+1]
        lines[0] = lines[0][theorem.range.start.character:]
        lines[-1] = lines[-1][:theorem.range.end.character]
        theorem.statement = "\n".join(lines)
    except IndexError:
        print(f"Failed to extract statement from {theorem} in {source.path}")

class ParserError(Exception):
    """Base class for parser error"""

class ProofNotFound(ParserError):
    """Raised when a completed proof script cannot be located."""
    pass

class TimeOut(ParserError):
    """Raised when a parser action takes too long."""
    pass

class AbstractParser(ABC):
    """Interface implemented by proof parsers."""

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def __call__(self, theorem: Element, source: Source) -> List[Step]:
        """Parse proof steps for the given theorem."""
        pass
