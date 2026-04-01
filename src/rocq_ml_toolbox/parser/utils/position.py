from ..parser import Range, Position, ParserError

def pos_to_offset(content_utf_8: bytes, p: Position) -> int:
    """
    Convert a Position (line/character) to a byte offset in UTF-8 content.

    - line is 0-based
    - character is a 0-based *byte index* within the line
    """
    lines = content_utf_8.splitlines(keepends=True)
    if p.line < 0 or len(lines) <= p.line:
        return len(content_utf_8)
    offset = sum(len(lines[i]) for i in range(p.line))
    line_no_nl = lines[p.line].rstrip(b"\r\n")
    if p.character < 0 or len(line_no_nl) < p.character:
        raise ParserError(f"character out of bounds: {p.character} on line {p.line}")
    return offset + p.character

def offset_to_pos(content_utf_8: bytes, offset: int) -> Position:
    """
    Convert a byte offset in UTF-8 content to a Position (line/character).

    - line is 0-based
    - character is a 0-based *byte index* within the line
    """
    if offset < 0:
        raise ParserError(f"offset out of bounds: {offset}")

    lines = content_utf_8.splitlines(keepends=True)

    remaining = offset
    for i, line in enumerate(lines):
        if remaining <= len(line):
            line_no_nl = line.rstrip(b"\r\n")
            character = min(remaining, len(line_no_nl))
            return Position(line=i, character=character)
        remaining -= len(line)

    if lines:
        last_line = lines[-1].rstrip(b"\r\n")
        return Position(line=len(lines) - 1, character=len(last_line))
    else:
        return Position(line=0, character=0)

def extract_subtext(content_utf_8: bytes, r: Range) -> bytes:
    """
    Extract substring defined by Range (line/character), where:
      - line is 0-based
      - character is a 0-based *byte index* within that line
      - end is treated as exclusive
    """
    start_off = pos_to_offset(content_utf_8, r.start)
    end_off = pos_to_offset(content_utf_8, r.end)

    return content_utf_8[start_off:end_off]


def move_position(content: bytes, pos: Position, length: int) -> Position:
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