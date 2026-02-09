from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, List, Optional
import tempfile
import shutil

from .dispatch import parse_node
from .model import VernacElement


class FccConfig:
    fcc_cmd: str = "fcc"
    plugin: str = "coq-lsp.plugin.astdump"


def ast_dump_path(filepath: str | Path) -> Path:
    p = Path(filepath)
    return p.with_suffix(p.suffix + ".jsonl.astdump")


def run_fcc_astdump(filepath: str | Path, *, root: Optional[str]=None, cfg: FccConfig = FccConfig()) -> Path:
    filepath = Path(filepath)

    out = ast_dump_path(filepath)
    cmd = [cfg.fcc_cmd, f"--root={root}", f"--plugin={cfg.plugin}", str(filepath), "--no_vo"]
    subprocess.run(cmd, capture_output=True, text=True)

    if not out.exists():
        raise RuntimeError(f"Expected ast dump not found: {out}")
    return out

def generate_ast_dump_file(filepath: str | Path, *, force_dump: bool = False, cfg: FccConfig = FccConfig()) -> Path:
    filepath = Path(filepath)
    dump = ast_dump_path(filepath)

    if force_dump or not dump.exists():
        run_fcc_astdump(filepath, cfg=cfg)

    return dump

def load_ast_dump(filepath: str | Path, *, root: Optional[str] = None, force_dump: bool = False, cfg: FccConfig = FccConfig()) -> List[dict]:
    filepath = Path(filepath)
    dump = ast_dump_path(filepath)

    if force_dump or not dump.exists():
        run_fcc_astdump(filepath, root=root, cfg=cfg)

    contents: List[dict] = []
    with dump.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            contents.append(json.loads(line))
    return contents


def parse_ast_dump(
    ast_dump: List[dict],
    *,
    on_unsupported: str = "keep",  # "keep" | "raise"
    keep_raw: bool = False,
) -> List[VernacElement]:
    out: List[VernacElement] = []
    for obj in ast_dump:
        out.append(parse_node(obj, on_unsupported=on_unsupported, keep_raw=keep_raw))
    return out


def compute_ast(
    filepath: str | Path,
    *,
    force_dump: bool = False,
    on_unsupported: str = "keep",
    keep_raw: bool = False,
    cfg: FccConfig = FccConfig(),
) -> List[VernacElement]:
    ast_dump = load_ast_dump(filepath, force_dump=force_dump, cfg=cfg)
    return parse_ast_dump(ast_dump, on_unsupported=on_unsupported, keep_raw=keep_raw)


def iter_v_files(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".v"):
                yield Path(dirpath) / name
