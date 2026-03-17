from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .dispatch import parse_node
from .model import VernacElement
from ..diags.parser import parse_diagnostics_file, Diagnostic
from ..proof.parser import ProofDump

class FccConfig:
    fcc_cmd: str = "fcc"
    plugin: str = "coq-lsp.plugin.proofdepsdump"


def proof_dump_path(filepath: str | Path) -> Path:
    p = Path(filepath)
    return p.with_suffix(p.suffix + ".json.proofdepsdump")

def ast_dump_path(filepath: str | Path) -> Path:
    p = Path(filepath)
    return p.with_suffix(p.suffix + ".json.proofdepsdump.ast")

def diags_dump_path(filepath: str | Path) -> Path:
    p = Path(filepath)
    return p.with_suffix(".diags")

def run_fcc(filepath: str | Path, *, root: Optional[str]=None, cfg: FccConfig = FccConfig(), max_errors=10_000):
    filepath = Path(filepath)

    diags = diags_dump_path(filepath)
    cmd = [cfg.fcc_cmd]
    if root:
        cmd.append(f"--root={root}")
    cmd.extend([f"--plugin={cfg.plugin}", str(filepath), "--no_vo", f"--max_errors={max_errors}"])
    subprocess.run(cmd, capture_output=True, text=True)

    if not diags.exists():
        raise RuntimeError(f"Expected diags dump not found: {diags}")

def load_jsonl(path: Path) -> List[dict]:
    content: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            content.append(json.loads(line))
    return content

def load_proof_dump(filepath: str | Path, *, root: Optional[str] = None, force_dump: bool = False, cfg: FccConfig = FccConfig()) -> Tuple[List[dict], List[dict], List[Diagnostic]]:
    filepath = Path(filepath)
    ast_dump = ast_dump_path(filepath)
    proof_dump = proof_dump_path(filepath)
    diags = diags_dump_path(filepath)

    if force_dump or not proof_dump.exists():
        run_fcc(filepath, root=root, cfg=cfg)

    proof_contents = json.loads(proof_dump.read_text())
    ast_contents = json.loads(ast_dump.read_text())['astdump_jsonl']
    return proof_contents, ast_contents, parse_diagnostics_file(diags)

def parse_proof_dump(
        proof_dump: List[dict]
) -> ProofDump:
    out = ProofDump.from_json(proof_dump)
    return out

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

def iter_v_files(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".v"):
                yield Path(dirpath) / name
