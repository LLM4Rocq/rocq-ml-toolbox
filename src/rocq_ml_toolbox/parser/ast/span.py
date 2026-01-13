from __future__ import annotations

from .model import Span


def extract_span(obj) -> Span:
    """
    Extract span from a nested structure
    Returns Span(bp=-1, ep=-1) if no bp/ep were found.
    """
    bp = float("inf")
    ep = -1
    stack = [obj]

    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "bp" in cur and "ep" in cur:
                try:
                    cbp = int(cur["bp"])
                    cep = int(cur["ep"])
                    bp = min(bp, cbp)
                    ep = max(ep, cep)
                except Exception:
                    pass
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)

    if bp == float("inf"):
        return Span(-1, -1)
    return Span(int(bp), int(ep))
