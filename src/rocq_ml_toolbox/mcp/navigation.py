from __future__ import annotations

import json
import os
import difflib
from enum import StrEnum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


class NodeKind(StrEnum):
    FILE = "file"
    DIR = "dir"

@dataclass
class Node:
    one_liner: Optional[str] = None
    num_elements: Optional[int] = None
    children: Dict[str, Node] = field(default_factory=dict)

    @property
    def kind(self) -> NodeKind:
        if self.one_liner:
            return NodeKind.FILE
        else:
            return NodeKind.DIR

    def child(self, name: str) -> Node:
        if name in self.children:
            return self.children[name]
        new_node = Node()
        self.children[name] = new_node
        return new_node

def list_packages(root: Path) -> Dict[str, Path]:
    """First-level packages/environments."""
    return {p.name: p for p in root.iterdir() if p.is_dir()}

def build_trie(env_root: Path) -> Node:
    """
    Index all directories containing one_liner.txt under env_root
    into a trie keyed by relative path parts.
    """
    root = Node("")

    for dirpath, _, filenames in os.walk(env_root):
        p = Path(dirpath)
        rel = p.relative_to(env_root)
        parts = [] if rel == Path(".") else list(rel.parts)

        cur = root
        for part in parts:
            cur = cur.child(part)

        try:
            cur.one_liner = (p / "one_liner.txt").read_text(encoding="utf-8").strip()
            with open(p / "all_elements.json", "r") as file:
                cur.num_elements = len(json.load(file))
        except Exception:
            cur.one_liner = None

    return root

def _normalize_path(path: Union[str, Iterable[str], None]) -> List[str]:
    if path is None:
        return []
    if isinstance(path, str):
        parts = [p for p in path.split(".") if p]
    else:
        parts = [str(p) for p in path if str(p)]

    # basic traversal / separator hardening
    return [p for p in parts if p not in (".", "..") and "/" not in p and "\\" not in p]


def _classify_child(name: str, node: Node) -> Dict[str, Any]:
    """
    Heuristic:
    - If it has children => directory-like
    - If it has no children but has one_liner => file-like leaf
    - If it has neither (should be rare) => treat as directory-like placeholder
    """
    return {
        "name": name,
        "kind": str(node.kind),
        "one_liner": node.one_liner,
        "num_elements": node.num_elements
    }

def _slice_children(node: Node, depth: int) -> Dict[str, Any]:
    """
    Returns a structured view plus a compact tree string.
    If depth > 1, recurses into subdirectories.
    """
    def sort_key(item: Tuple[str, Node]) -> Tuple[int, str]:
        name, child = item
        is_dir = 0 if child.kind == NodeKind.FILE else 1
        return (is_dir, name)

    items = sorted(node.children.items(), key=sort_key)
    entries: List[Dict[str, Any]] = []
    for name, child in items:
        entry = _classify_child(name, child)
        if depth > 1 and child.kind == NodeKind.DIR:
            entry["children"] = _slice_children(child, depth=depth - 1)
        entries.append(entry)

    tree = _render_tree(entries)

    return {
        "entries": entries,
        "depth": depth,
        "tree": tree,
    }


def _render_tree(entries: List[Dict[str, Any]], prefix: str = "") -> str:
    """
    Render a compact ASCII tree from the structured entries.
    Shows leaf 'files' with their one-liner when present.
    """
    lines: List[str] = []
    for i, e in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "

        name = e["name"]
        kind = e["kind"]
        num_elements = e['num_elements']
        one = (e.get("one_liner") or "").strip()

        label = name + ("/" if kind == "dir" else "")
        if kind == "file" and one:
            # ignore if nothing inside
            if num_elements > 0:
                label += f"  –  {one} ({num_elements} elements)"
        elif kind == "dir" and one:
            label += f"  –  {one}"

        lines.append(prefix + connector + label)

        children = e.get("children")
        if children and children.get("entries"):
            extension = "    " if is_last else "│   "
            lines.append(_render_tree(children["entries"], prefix + extension))

    return "\n".join([ln for ln in lines if ln])


class CodebaseNavigator:
    def __init__(self, annotated_root: Path) -> None:
        self.annotated_root = annotated_root
        self.env_paths = list_packages(annotated_root)
        self.env_tries: Dict[str, Node] = {}

    def refresh(self) -> None:
        self.env_paths = list_packages(self.annotated_root)
        self.env_tries.clear()

    def _get_trie(self, env: str) -> Optional[Node]:
        if env not in self.env_paths:
            return None
        if env not in self.env_tries:
            self.env_tries[env] = build_trie(self.env_paths[env])
        return self.env_tries[env]

    def list_packages(self) -> Dict[str, Any]:
        return {"packages": sorted(self.env_paths.keys())}

    def open(
        self,
        env: str,
        path: Union[str, Iterable[str], None] = None,
        *,
        filename: str = "source_wo_proof.v",
    ) -> Dict[str, Any]:
        """
        Return the content of a source file for a given annotated path.

        Accepts either:
          - path pointing to a directory (opens <dir>/<filename>)
          - path pointing directly to a file

        Produces helpful "closest path" suggestions using the trie if a segment is invalid.
        """
        trie = self._get_trie(env)
        if trie is None:
            available = "\n".join(sorted(self.env_paths.keys()))
            return {
                "ok": False,
                "result": f"Unknown environment: {env}.\nAvailable envs are:\n{available}",
            }

        parts = _normalize_path(path)
        env_root = self.env_paths[env]

        direct = env_root.joinpath(*parts)
        candidates: List[Path] = []
        if direct.is_file():
            candidates.append(direct)
        candidates.append(direct / filename)

        for fp in candidates:
            if fp.exists() and fp.is_file():
                try:
                    return {"ok": True, "result": fp.read_text(encoding="utf-8")}
                except Exception as e:
                    return {"ok": False, "result": f"Failed to read {fp}: {e}"}

        # If missing, use trie to find the first invalid segment and show options.
        cur = trie
        consumed: List[str] = []
        for part in parts:
            if part not in cur.children:
                choices = sorted(cur.children.keys())
                suggestions = difflib.get_close_matches(part, choices, n=3, cutoff=0.6)
                path_prefix = ".".join([env, *consumed]) if consumed else env
                from_path = ".".join(consumed) if consumed else env

                choices_block = "\n".join(f"- {c}" for c in choices) if choices else "- (no entries)"
                suggestion_text = f"\nDid you mean: {', '.join(suggestions)}?" if suggestions else ""

                return {
                    "ok": False,
                    "result": (
                        f"'{part}' does not exist under '{path_prefix}'.\n"
                        f"From '{from_path}' you can access:\n{choices_block}"
                        f"{suggestion_text}"
                    ),
                }

            cur = cur.children[part]
            consumed.append(part)

        expected = env_root.joinpath(*parts) / filename
        return {"ok": False, "result": f"Path exists, but source file not found: {expected}"}
        
    def explore(
        self,
        env: str,
        path: Union[str, Iterable[str], None] = None,
        *,
        depth: int = 1,
    ) -> Dict[str, Any]:
        trie = self._get_trie(env)
        if trie is None:
            available = "\n".join(sorted(self.env_paths.keys()))
            return {
                "ok": False,
                "result": f"Unknown environment: {env}.\nAvailable envs are:\n{available}",
            }

        parts = _normalize_path(path)
        cur = trie
        consumed: List[str] = []

        for part in parts:
            if part not in cur.children:
                choices = sorted(cur.children.keys())
                suggestions = difflib.get_close_matches(part, choices, n=3, cutoff=0.6)
                path_prefix = ".".join([env, *consumed]) if consumed else env
                from_path = ".".join(consumed) if consumed else env

                choices_block = "\n".join(f"- {c}" for c in choices) if choices else "- (no entries)"
                suggestion_text = f"\nDid you mean one of them: {', '.join(suggestions)}?" if suggestions else ""

                return {
                    "ok": False,
                    "result": (
                        f"'{part}' does not exist under '{path_prefix}'.\n"
                        f"From '{from_path}' you can access:\n{choices_block}"
                        f"{suggestion_text}"
                    ),
                }
            cur = cur.children[part]
            consumed.append(part)

        where = ".".join(consumed) if consumed else env
        tree = _slice_children(cur, depth=depth)["tree"]
        return {"ok": True, "result": f"Currently at {where}.\n{tree}"}

if __name__ == "__main__":
    code_nav = CodebaseNavigator(Path('annotated/export/theostos'))
    print(code_nav.explore('coq-geocoq', ['mathcomp']))
