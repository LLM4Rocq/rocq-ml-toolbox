from typing import Optional, Dict
import re

from ..parser import Range, Position, Element, ParserError

def parse_loadpath(text: str) -> dict[str, str]:
    """
    Parse Coq LoadPath output into a dict mapping
    logical paths -> physical paths.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    lines = [
        l for l in lines
        if l and not l.startswith("Logical Path")
    ]

    result: dict[str, str] = {}

    i = 0
    while i < len(lines):
        line = lines[i]

        # logical + physical on the same line
        m = re.match(r'^(\S+)\s+(/.+)$', line)
        if m:
            logical, physical = m.groups()
            result[logical] = physical
            i += 1
            continue

        # logical on one line, physical on the next (indented)
        m = re.match(r'^(\S+)$', line)
        if m and i + 1 < len(lines):
            m2 = re.match(r'^\s+(/.+)$', lines[i + 1])
            if m2:
                result[m.group(1)] = m2.group(1)
                i += 2
                continue

        i += 1

    return result

def solve_logical_path(logical_path: str, map_paths:Dict[str, str]) -> Optional[str]:
    """Convert a logical path into a physical path based on LoadPath dict (logical -> physical)."""
    if logical_path in map_paths:
        return map_paths[logical_path]
    if not '.' in logical_path:
        return None
    parent, child = logical_path.rsplit('.', 1)
    for l_path, p_path in map_paths.items():
        if l_path == parent:
            return p_path + f'/{child}.v'
    return None

def solve_physical_path(physical_path: str, map_paths:Dict[str, str]) -> Optional[str]:
    """Convert a physical path into a logical path based on LoadPath dict (physical -> logical)."""
    if physical_path in map_paths:
        return map_paths[physical_path]
    parent, child = physical_path.rsplit('/', 1)
    if not child.endswith('.v'):
        None
    
    child = child[:-2]
    for p_path, l_path in map_paths.items():
        if p_path == parent:
            return l_path + f'.{child}'
    return None

def parse_about(result: str, map_l_p:Dict[str, str], map_p_l:Dict[str, str]) -> Optional[Element]:
    """Parse `About` feedback into a tuple `(path, range)`. Path corresponds to the path of the dependency (if None: current file)."""
    name = result.split(' :')[0]
    if "Hypothesis of the goal context." in result:
        return None
    if 'Declared in' in result:
        pattern = (
            r'Expands to: (?P<kind>Constant|Constructor)\s+(?P<fqn>[^,\s]+)\s.*'
            r'Declared in\s+(?:File\s+"(?P<file>[^"]+)"|library (?P<lib>[^,]+)), '
            r'line (?P<line>\d+(?:-\d+)?), characters (?P<char>\d+(?:-\d+)?)'
        )
        match = re.search(pattern, result, flags=re.DOTALL)
        if not match:
            raise ParserError(f"Issue when parsing position information from {result}")
        logical_path = match.group("lib")
        physical_path = match.group("file")
        fqn = match.group("fqn")
        kind = match.group("kind")

        if logical_path:
            physical_path = solve_logical_path(logical_path, map_l_p)
        elif physical_path:
            logical_path = solve_physical_path(physical_path, map_p_l)
            if logical_path:
                fqn = f'{logical_path}.' + fqn.split('.', maxsplit=1)[1]
        
        line_part = match.group('line')
        char_part = match.group('char')

        line_start, line_end = (map(int, line_part.split('-')) if '-' in line_part else (int(line_part), int(line_part)))
        char_start, char_end = (map(int, char_part.split('-')) if '-' in char_part else (int(char_part), int(char_part)))

        r = Range(
            start=Position(line_start, char_start),
            end=Position(line_end, char_end)
        )
        
        return Element(physical_path=physical_path, logical_path=logical_path, name=name, kind=kind, fqn=fqn, range=r)
    return None