import json
import subprocess
import sys
from pathlib import Path

from src.rocq_ml_toolbox.safeverify.core import _duplicate_names, run_safeverify
from src.rocq_ml_toolbox.safeverify.types import FailureCode


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _run_pair(tmp_path: Path, source_text: str, target_text: str, whitelist: str | None = None):
    source = _write(tmp_path / "Source.v", source_text)
    target = _write(tmp_path / "Target.v", target_text)

    whitelist_path = None
    if whitelist is not None:
        whitelist_path = _write(tmp_path / "whitelist.json", whitelist)

    report = run_safeverify(
        source,
        target,
        root=tmp_path,
        axiom_whitelist=whitelist_path,
        verbose=True,
    )
    return report, source, target, whitelist_path


def _codes(report):
    return [code.value for code in report.outcomes[0].failure_codes]


def test_pass_simple_admitted_solution(tmp_path: Path):
    source = """
Theorem t : True.
Admitted.
"""
    target = """
Theorem t : True.
Proof. exact I. Qed.
"""
    report, _, _, _ = _run_pair(tmp_path, source, target)

    assert report.ok is True
    assert report.summary()["num_obligations"] == 1
    assert report.summary()["failed"] == 0
    assert report.outcomes[0].checks["statement"] is True


def test_fail_target_has_admitted(tmp_path: Path):
    source = """
Theorem t : True.
Admitted.
"""
    target = """
Theorem t : True.
Admitted.
"""
    report, _, _, _ = _run_pair(tmp_path, source, target)

    assert report.ok is False
    assert FailureCode.INCOMPLETE_PROOF.value in _codes(report)


def test_fail_target_has_admit_qed(tmp_path: Path):
    source = """
Theorem t : True.
Admitted.
"""
    target = """
Theorem t : True.
Proof.
  admit.
Qed.
"""
    report, _, _, _ = _run_pair(tmp_path, source, target)

    assert report.ok is False
    assert FailureCode.INCOMPLETE_PROOF.value in _codes(report)


def test_fail_missing_obligation(tmp_path: Path):
    source = """
Theorem t : True.
Admitted.
"""
    target = """
Theorem u : True.
Proof. exact I. Qed.
"""
    report, _, _, _ = _run_pair(tmp_path, source, target)

    assert report.ok is False
    assert FailureCode.MISSING_OBLIGATION.value in _codes(report)


def test_fail_statement_mismatch(tmp_path: Path):
    source = """
Theorem t : nat.
Admitted.
"""
    target = """
Theorem t : True.
Proof. exact I. Qed.
"""
    report, _, _, _ = _run_pair(tmp_path, source, target)

    assert report.ok is False
    assert FailureCode.STATEMENT_MISMATCH.value in _codes(report)


def test_fail_new_local_axiom(tmp_path: Path):
    source = """
Theorem t : True.
Admitted.
"""
    target = """
Axiom extA : True.
Theorem t : True.
Proof. exact extA. Qed.
"""
    report, _, _, _ = _run_pair(tmp_path, source, target)

    assert report.ok is False
    assert any(gf.code == FailureCode.NEW_LOCAL_AXIOM for gf in report.global_failures)
    codes = _codes(report)
    assert FailureCode.NEW_LOCAL_AXIOM.value in codes
    assert FailureCode.DISALLOWED_AXIOMS.value in codes


def test_pass_whitelisted_axiom(tmp_path: Path):
    source = """
Theorem t : True.
Admitted.
"""
    target = """
Axiom extA : True.
Theorem t : True.
Proof. exact extA. Qed.
"""
    whitelist = json.dumps({"axioms": ["extA"]})
    report, _, _, _ = _run_pair(tmp_path, source, target, whitelist=whitelist)

    assert report.ok is True


def test_pass_with_helper_lemma(tmp_path: Path):
    source = """
Theorem t : True.
Admitted.
"""
    target = """
Lemma helper : True.
Proof. exact I. Qed.

Theorem t : True.
Proof. exact helper. Qed.
"""
    report, _, _, _ = _run_pair(tmp_path, source, target)

    assert report.ok is True


def test_duplicate_name_detection_unit():
    assert _duplicate_names(["a", "b", "a", "c", "b", "d"]) == {"a", "b"}


def test_cli_exit_code_and_save(tmp_path: Path):
    source = _write(
        tmp_path / "Source.v",
        """
Theorem t : True.
Admitted.
""",
    )
    target = _write(
        tmp_path / "Target.v",
        """
Theorem t : True.
Proof. exact I. Qed.
""",
    )
    report_path = tmp_path / "report.json"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.rocq_ml_toolbox.safeverify.cli",
            str(source),
            str(target),
            "--root",
            str(tmp_path),
            "--save",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["summary"]["num_obligations"] == 1
