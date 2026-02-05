from __future__ import annotations

import subprocess
from pathlib import Path
import tempfile
import shutil

from .parser import parse_glob_file, GlobFile

class CoqcConfig:
    coq_cmd: str = "coqc"

def glob_path(filepath: str | Path) -> Path:
    p = Path(filepath)
    return p.with_suffix(".glob")

def run_coqc(filepath: str | Path, *, cfg: CoqcConfig = CoqcConfig()) -> Path:
    filepath = Path(filepath)
    with tempfile.TemporaryDirectory(prefix="coqc_") as tmpdir:
        tmpdir = Path(tmpdir)

        tmp_src = tmpdir / filepath.name
        shutil.copy(filepath, tmp_src)

        out = glob_path(tmp_src)

        cmd = [cfg.coq_cmd, str(tmp_src)]
        subprocess.run(cmd, capture_output=True, text=True)

        if not out.exists():
            raise RuntimeError(f"Expected ast dump not found: {out}")

        final_out = glob_path(filepath)
        shutil.copy(out, final_out)
    return final_out

def load_glob_file(filepath: str | Path, *, force_compile: bool = False, cfg: CoqcConfig = CoqcConfig()) -> GlobFile:
    filepath = Path(filepath)
    dump = glob_path(filepath)

    if force_compile or not dump.exists():
        run_coqc(filepath, cfg=cfg)

    return parse_glob_file(dump)
