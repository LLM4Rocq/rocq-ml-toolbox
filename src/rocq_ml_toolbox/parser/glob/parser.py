import dataclasses
import json
from enum import StrEnum
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

class GlobKind(StrEnum):
    """
    Enumeration of standard Rocq/Coq object kinds found in .glob files.
    """
    AX = "ax"
    DEF = "def"
    COE = "coe"
    THM = "thm"
    SUB = "subclass"
    CANONSTRUC = "canonstruc"
    EX = "ex"
    SCHEME = "scheme"
    CLASS = "class"
    PROJECTION = "proj"
    INSTANCE = "inst"
    METH = "meth"
    DEFAX = "defax"
    PRFAX = "prfax"
    PRIM = "prim"
    VAR = "var"
    INDREC = "indrec"
    REC = "rec"
    COREC = "corec"
    IND = "ind"
    VARIANT = "variant"
    COIND = "coind"
    CONSTR = "constr"
    NOT = "not"
    BINDER = "binder"
    LIB = "lib"
    MOD = "mod"
    MODTYPE = "modtype"
    ABBREV = "abbrev"
    SEC = "sec"
    UNK = "unk"        # Unknown/Fallback

    @classmethod
    def _missing_(cls, value):
        return cls.UNK

# Added kw_only=True to fix the inheritance order issue
@dataclasses.dataclass(kw_only=True)
class GlobEntry:
    """Base class for .glob entries."""
    bp: int
    ep: int
    name: str
    secpath: Optional[str] = None  # None if parsed as '<>'
    
    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

@dataclasses.dataclass(kw_only=True)
class GlobDefinition(GlobEntry):
    """Represents a definition (e.g., variable, theorem, inductive)."""
    kind: GlobKind

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'GlobDefinition':
        if 'kind' in data:
            data['kind'] = GlobKind(data['kind'])
        return cls(**data)

@dataclasses.dataclass(kw_only=True)
class GlobReference(GlobEntry):
    """Represents a reference to a defined object."""
    filepath: str
    kind: GlobKind

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'GlobReference':
        if 'kind' in data:
            data['kind'] = GlobKind(data['kind'])
        return cls(**data)

@dataclasses.dataclass
class GlobFile:
    """Represents the parsed contents of a .glob file."""
    digest: Optional[str]
    module_path: str
    entries: List[Union[GlobDefinition, GlobReference]]

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'GlobFile':
        entries_data = data.get('entries', [])
        parsed_entries = []

        for entry_dict in entries_data:
            if 'filepath' in entry_dict:
                parsed_entries.append(GlobReference.from_json(entry_dict))
            else:
                parsed_entries.append(GlobDefinition.from_json(entry_dict))

        return cls(
            digest=data.get('digest'),
            module_path=data.get('module_path', ''),
            entries=parsed_entries
        )

def parse_glob_file(file_path: Union[str, Path]) -> GlobFile:
    """Parses a Rocq/Coq .glob file into structured dataclasses."""
    entries = []
    digest = None
    module_path = ""

    path_obj = Path(file_path)
    
    if not path_obj.exists():
        return GlobFile(digest=None, module_path="", entries=[])

    with path_obj.open('r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return GlobFile(digest=None, module_path="", entries=[])

    current_line_idx = 0
    if lines[0].startswith("DIGEST:"):
        digest = lines[0][7:].strip()
        current_line_idx += 1

    if current_line_idx < len(lines) and lines[current_line_idx].startswith("F"):
        module_path = lines[current_line_idx][1:].strip()
        current_line_idx += 1

    for line in lines[current_line_idx:]:
        tokens = line.split()
        if not tokens:
            continue

        first_token = tokens[0]

        def parse_secpath(token: str) -> Optional[str]:
            return None if token == '<>' else token

        # Check for Reference (Rbc:ec)
        if first_token.startswith("R") and len(first_token) > 1 and first_token[1].isdigit():
            if len(tokens) >= 5:
                loc_token = first_token[1:]
                start, end = map(int, loc_token.split(':'))
                entries.append(GlobReference(
                    bp=start,
                    ep=end,
                    filepath=tokens[1],
                    secpath=parse_secpath(tokens[2]),
                    name=tokens[3],
                    kind=GlobKind(tokens[4])
                ))
        # Check for Definition
        else:
            if len(tokens) >= 4:
                try:
                    start, end = map(int, tokens[1].split(':'))
                    entries.append(GlobDefinition(
                        kind=GlobKind(first_token),
                        bp=start,
                        ep=end,
                        secpath=parse_secpath(tokens[2]),
                        name=tokens[3]
                    ))
                except ValueError:
                    continue

    return GlobFile(digest=digest, module_path=module_path, entries=entries)