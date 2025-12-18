from ..parser import Range, Position, ParserError

def pos_to_offset(content: str, p: Position) -> int:
    lines = content.splitlines(keepends=True)
    if p.line < 0 or len(lines) <= p.line:
        return len(content)
    offset = sum(len(lines[i]) for i in range(p.line))
    line_no_nl = lines[p.line].rstrip("\r\n")
    if p.character < 0 or len(line_no_nl) < p.character:
        raise ParserError(f"character out of bounds: {p.character} on line {p.line}")
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