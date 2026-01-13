from __future__ import annotations

from typing import Any, Iterable, Optional


class MalformedAST(ValueError):
    pass


def jpath(obj: Any, *path: Any) -> Any:
    """
    Strict navigation helper.
    """
    cur = obj
    for p in path:
        try:
            cur = cur[p]
        except Exception as e:
            raise MalformedAST(f"JSON path failed at {p!r} in {path!r}") from e
    return cur


def jmaybe(obj: Any, *path: Any, default: Any = None) -> Any:
    cur = obj
    for p in path:
        try:
            cur = cur[p]
        except Exception:
            return default
    return cur


def as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def ensure_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    return str(x)


def first(*values: Any, default: Any = None) -> Any:
    for v in values:
        if v is not None:
            return v
    return default


def find_local_attr(attrs: Iterable[Any]) -> bool:
    for a in attrs or []:
        v = a.get("v") if isinstance(a, dict) else None
        if v and isinstance(v, list) and len(v) >= 1 and v[0] == "local":
            return True
    return False
