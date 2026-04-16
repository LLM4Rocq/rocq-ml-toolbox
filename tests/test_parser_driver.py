import json
from pathlib import Path

from src.rocq_ml_toolbox.parser.ast import driver


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_load_proof_dump_force_dump_allows_missing_diags(monkeypatch, tmp_path: Path):
    source = _write(tmp_path / "Source.v", "Theorem t : True.\nAdmitted.\n")

    def fake_run_fcc(filepath: str | Path, **_kwargs):
        filepath = Path(filepath)
        driver.proof_dump_path(filepath).write_text("[]", encoding="utf-8")
        driver.ast_dump_path(filepath).write_text(
            json.dumps({"astdump_jsonl": []}),
            encoding="utf-8",
        )

    monkeypatch.setattr(driver, "run_fcc", fake_run_fcc)

    proofs, ast, diags = driver.load_proof_dump(source, force_dump=True)

    assert proofs == []
    assert ast == []
    assert diags == []
