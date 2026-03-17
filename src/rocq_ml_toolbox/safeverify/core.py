from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import shlex
import subprocess
import tempfile

from ..parser.ast.driver import load_proof_dump, parse_ast_dump, parse_proof_dump
from ..parser.ast.model import VernacElement, VernacKind
from ..parser.diags.parser import Diagnostic
from ..parser.proof.parser import ProofDependency, ProofEntry
from .types import CheckOutcome, FailureCode, Obligation, VerificationReport


INCOMPLETE_PROOF_SNIPPETS = (
    "attempt to save an incomplete proof",
    "given up (admitted) goals",
    "remaining open goals",
    "proof term is not complete",
)
ERROR_SEVERITY = 1


@dataclass(frozen=True)
class LoadPathBinding:
    physical_dir: Path
    logical_prefix: str
    recursive: bool


@dataclass(frozen=True)
class TheoremDecl:
    name: str
    relative_fqn: str


def _normalize_step(raw: str) -> str:
    return raw.strip()


def _pos_tuple(line: int, character: int) -> tuple[int, int]:
    return (line, character)


def _is_incomplete_diag(message: str) -> bool:
    msg = message.lower()
    return any(pattern in msg for pattern in INCOMPLETE_PROOF_SNIPPETS)


def _is_error_diag(diag: Diagnostic) -> bool:
    return diag.severity == ERROR_SEVERITY


def _diagnostic_to_json(diag: Diagnostic) -> dict[str, Any]:
    return {
        "severity": diag.severity,
        "message": diag.message,
        "range": {
            "start": {
                "line": diag.range.start.line,
                "character": diag.range.start.character,
            },
            "end": {
                "line": diag.range.end.line,
                "character": diag.range.end.character,
            },
        },
    }


def _proof_bounds(proof: ProofEntry) -> tuple[tuple[int, int], tuple[int, int]]:
    start = _pos_tuple(proof.start_range.start.line, proof.start_range.start.character)
    if proof.steps:
        last = proof.steps[-1].range.end
        end = _pos_tuple(last.line, last.character)
    else:
        end = _pos_tuple(proof.start_range.end.line, proof.start_range.end.character)
    return start, end


def _diag_overlaps_proof(diag: Diagnostic, proof: ProofEntry) -> bool:
    start, end = _proof_bounds(proof)
    dstart = _pos_tuple(diag.range.start.line, diag.range.start.character)
    dend = _pos_tuple(diag.range.end.line, diag.range.end.character)
    return not (dend < start or end < dstart)


def _proof_incomplete_diagnostics(proof: ProofEntry, diags: Sequence[Diagnostic]) -> list[str]:
    result: list[str] = []
    for diag in diags:
        if _diag_overlaps_proof(diag, proof) and _is_incomplete_diag(diag.message):
            result.append(diag.message)
    return result


def _proof_error_diagnostics(proof: ProofEntry, diags: Sequence[Diagnostic]) -> list[Diagnostic]:
    result: list[Diagnostic] = []
    for diag in diags:
        if _diag_overlaps_proof(diag, proof) and _is_error_diag(diag):
            result.append(diag)
    return result


def _proof_has_admitted_step(proof: ProofEntry) -> bool:
    return any(_normalize_step(step.raw) == "Admitted." for step in proof.steps)


def _proof_has_self_axiom(proof: ProofEntry) -> bool:
    return any(dep.name == proof.name for dep in proof.axioms)


def _is_missing_obligation(proof: ProofEntry, diags: Sequence[Diagnostic]) -> bool:
    if _proof_has_admitted_step(proof):
        return True
    return bool(_proof_error_diagnostics(proof, diags))


def _extract_theorem_decls(ast: Sequence[VernacElement]) -> list[TheoremDecl]:
    declarations: list[TheoremDecl] = []
    namespaces: list[tuple[str, str]] = []

    for entry in ast:
        kind = entry.kind

        if kind == VernacKind.BEGIN_SECTION and entry.name:
            namespaces.append(("SECTION", entry.name))
            continue

        if kind in (VernacKind.DEFINE_MODULE, VernacKind.DECLARE_MODULE_TYPE) and entry.name:
            is_alias = bool(entry.data.get("is_alias", False))
            if not is_alias:
                namespaces.append(("MODULE", entry.name))
            continue

        if kind == VernacKind.END_SEGMENT:
            if not namespaces:
                continue
            if entry.name:
                idx = None
                for k in range(len(namespaces) - 1, -1, -1):
                    if namespaces[k][1] == entry.name:
                        idx = k
                        break
                if idx is not None:
                    namespaces = namespaces[:idx]
                    continue
            namespaces.pop()
            continue

        if kind == VernacKind.START_THEOREM_PROOF and entry.name:
            modules = [name for kind_, name in namespaces if kind_ == "MODULE"]
            relative_fqn = ".".join([*modules, entry.name])
            declarations.append(TheoremDecl(name=entry.name, relative_fqn=relative_fqn))

    return declarations


def _proof_id_to_relative_fqn(proofs: Sequence[ProofEntry], theorem_decls: Sequence[TheoremDecl]) -> dict[int, str]:
    by_name: dict[str, deque[TheoremDecl]] = defaultdict(deque)
    for decl in theorem_decls:
        by_name[decl.name].append(decl)

    output: dict[int, str] = {}
    for proof in proofs:
        queue = by_name.get(proof.name)
        if queue and len(queue) > 0:
            output[proof.proof_id] = queue.popleft().relative_fqn
        else:
            output[proof.proof_id] = proof.name
    return output


def _extract_assumption_names(ast_raw: Sequence[dict[str, Any]]) -> set[str]:
    names: set[str] = set()

    for obj in ast_raw:
        expr = obj.get("v", {}).get("expr")
        if not isinstance(expr, list) or len(expr) < 2:
            continue

        data = expr[1]
        if not isinstance(data, list) or not data or data[0] != "VernacAssumption":
            continue

        if len(data) < 4 or not isinstance(data[3], list):
            continue

        assumptions = data[3]
        for clause in assumptions:
            if not isinstance(clause, list) or len(clause) < 2:
                continue
            declarations = clause[1]
            if not isinstance(declarations, list):
                continue
            for declaration in declarations:
                if not isinstance(declaration, list) or not declaration:
                    continue
                identifiers = declaration[0]
                if not isinstance(identifiers, list):
                    continue
                for ident in identifiers:
                    if isinstance(ident, dict):
                        node = ident
                    elif isinstance(ident, list) and ident:
                        node = ident[0]
                    else:
                        continue
                    if not isinstance(node, dict):
                        continue
                    value = node.get("v")
                    if (
                        isinstance(value, list)
                        and len(value) >= 2
                        and value[0] == "Id"
                        and isinstance(value[1], str)
                    ):
                        names.add(value[1])

    return names


def _strip_comment(line: str) -> str:
    if "#" not in line:
        return line
    return line.split("#", maxsplit=1)[0]


def _resolve_dir(token: str, root: Path) -> Path:
    p = Path(token)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _collect_coqproject_loadpaths(root: Path) -> tuple[list[str], list[LoadPathBinding]]:
    project_file = root / "_CoqProject"
    coqc_args: list[str] = []
    bindings: list[LoadPathBinding] = []

    if not project_file.exists():
        # Default to a direct root binding.
        coqc_args.extend(["-Q", str(root), ""])
        bindings.append(LoadPathBinding(root, "", recursive=False))
        return coqc_args, bindings

    tokens: list[str] = []
    for line in project_file.read_text(encoding="utf-8").splitlines():
        stripped = _strip_comment(line).strip()
        if not stripped:
            continue
        tokens.extend(shlex.split(stripped))

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in {"-Q", "-R"}:
            if i + 2 >= len(tokens):
                break
            directory = _resolve_dir(tokens[i + 1], root)
            logical = tokens[i + 2]
            recursive = tok == "-R"
            coqc_args.extend([tok, str(directory), logical])
            bindings.append(LoadPathBinding(directory, logical, recursive))
            i += 3
            continue
        if tok == "-I":
            if i + 1 >= len(tokens):
                break
            directory = _resolve_dir(tokens[i + 1], root)
            coqc_args.extend(["-I", str(directory)])
            i += 2
            continue
        i += 1

    if not bindings:
        coqc_args.extend(["-Q", str(root), ""])
        bindings.append(LoadPathBinding(root, "", recursive=False))

    return coqc_args, bindings


def _module_from_binding(filepath: Path, binding: LoadPathBinding) -> str | None:
    try:
        rel = filepath.relative_to(binding.physical_dir)
    except ValueError:
        return None

    if rel.suffix != ".v":
        return None

    parts = list(rel.with_suffix("").parts)
    prefix = [x for x in binding.logical_prefix.split(".") if x]
    return ".".join([*prefix, *parts])


def _compute_logical_module(filepath: Path, root: Path, bindings: Sequence[LoadPathBinding]) -> str:
    filepath = filepath.resolve()
    try:
        filepath.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"File '{filepath}' is outside root '{root}'.") from exc

    candidates: list[tuple[int, str]] = []
    for binding in bindings:
        module = _module_from_binding(filepath, binding)
        if module is None:
            continue
        candidates.append((len(binding.physical_dir.parts), module))

    if not candidates:
        rel = filepath.relative_to(root)
        if rel.suffix != ".v":
            raise ValueError(f"Expected a .v file, got '{filepath}'.")
        return ".".join(rel.with_suffix("").parts)

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_whitelist(path: str | Path | None) -> set[str]:
    if path is None:
        return set()

    path = Path(path)
    content = path.read_text(encoding="utf-8")

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML whitelist requires 'pyyaml' to be installed.") from exc
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)

    if data is None:
        return set()

    if isinstance(data, dict):
        if "axioms" in data:
            data = data["axioms"]
        else:
            data = list(data.keys())

    if isinstance(data, str):
        data = [data]

    if not isinstance(data, list):
        raise ValueError("Whitelist must be a string, a list of strings, or {\"axioms\": [...]}.")

    out: set[str] = set()
    for item in data:
        if not isinstance(item, str):
            raise ValueError(f"Whitelist entry must be a string, got: {item!r}")
        value = item.strip()
        if value:
            out.add(value)
    return out


def _dep_identifier(dep: ProofDependency) -> str:
    if dep.logical_path:
        return f"{dep.logical_path}.{dep.name}"
    return dep.name


def _dep_allowed(dep: ProofDependency, allowed_axioms: set[str]) -> bool:
    keys = {dep.name}
    if dep.logical_path:
        keys.add(f"{dep.logical_path}.{dep.name}")
    return any(k in allowed_axioms for k in keys)


def _target_proof_completeness(proof: ProofEntry, diags: Sequence[Diagnostic]) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {}

    has_admitted_step = _proof_has_admitted_step(proof)
    if has_admitted_step:
        details["has_admitted_step"] = True

    overlapping_errors = _proof_error_diagnostics(proof, diags)
    if overlapping_errors:
        details["error_diagnostics"] = [_diagnostic_to_json(diag) for diag in overlapping_errors]

    # Reporting signal only: this is not a completeness gate.
    incomplete_signals = _proof_incomplete_diagnostics(proof, diags)
    if incomplete_signals:
        details["incomplete_proof_signals"] = incomplete_signals

    has_self_axiom = _proof_has_self_axiom(proof)
    if has_self_axiom:
        details["has_self_axiom"] = True

    is_complete = not has_admitted_step and not overlapping_errors and not has_self_axiom
    if is_complete:
        # Keep output compact for successful proofs.
        return True, {}

    return False, details


def _statement_harness(
    source_module: str,
    target_module: str,
    source_ref: str,
    target_ref: str,
) -> str:
    imports = [f"Require Import {source_module}."]
    if target_module != source_module:
        imports.append(f"Require Import {target_module}.")

    imports_block = "\n".join(imports)
    return (
        f"{imports_block}\n"
        "Goal True.\n"
        f"  let Ts := type of {source_ref} in\n"
        f"  let Tt := type of {target_ref} in\n"
        "  unify Ts Tt;\n"
        "  unify Tt Ts;\n"
        "  exact I.\n"
        "Qed.\n"
    )


def _cleanup_coqc_artifacts(check_file: Path) -> None:
    for suffix in (".v", ".vo", ".glob", ".vok", ".vos"):
        try:
            check_file.with_suffix(suffix).unlink(missing_ok=True)
        except Exception:
            pass


def _run_coqc(file_path: Path, coqc_args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    cmd = ["coqc", *coqc_args, str(file_path)]
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _prepare_statement_env(source_path: Path, target_path: Path, coqc_args: Sequence[str], root: Path) -> None:
    src = _run_coqc(source_path, coqc_args, cwd=root)
    if src.returncode != 0:
        msg = src.stderr.strip() or src.stdout.strip()
        raise RuntimeError(f"coqc failed on source file: {msg}")

    if target_path.resolve() == source_path.resolve():
        return

    tgt = _run_coqc(target_path, coqc_args, cwd=root)
    if tgt.returncode != 0:
        msg = tgt.stderr.strip() or tgt.stdout.strip()
        raise RuntimeError(f"coqc failed on target file: {msg}")


def _check_statement_convertibility(
    *,
    root: Path,
    coqc_args: Sequence[str],
    source_module: str,
    target_module: str,
    source_ref: str,
    target_ref: str,
) -> tuple[bool, str | None]:
    tmp = tempfile.NamedTemporaryFile(prefix="sv_check_", suffix=".v", dir=root, delete=False)
    check_file = Path(tmp.name)
    tmp.close()

    try:
        check_file.write_text(
            _statement_harness(
                source_module=source_module,
                target_module=target_module,
                source_ref=source_ref,
                target_ref=target_ref,
            ),
            encoding="utf-8",
        )
        proc = _run_coqc(check_file, coqc_args, cwd=root)
        if proc.returncode == 0:
            return True, None

        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        message = stderr or stdout or "unknown coqc error"
        return False, message
    finally:
        _cleanup_coqc_artifacts(check_file)


def _duplicate_names(names: Iterable[str]) -> set[str]:
    counts = Counter(names)
    return {name for name, count in counts.items() if count > 1}


def _is_name_whitelisted(name: str, module: str, allowed_axioms: set[str]) -> bool:
    return name in allowed_axioms or f"{module}.{name}" in allowed_axioms


def _disallowed_axioms(proof: ProofEntry, allowed_axioms: set[str]) -> list[str]:
    disallowed: list[str] = []
    for dep in proof.axioms:
        if _dep_allowed(dep, allowed_axioms):
            continue
        disallowed.append(_dep_identifier(dep))
    # Keep deterministic order.
    return sorted(set(disallowed))


def run_safeverify(
    source_path: str | Path,
    target_path: str | Path,
    *,
    root: str | Path,
    axiom_whitelist: str | Path | None = None,
    save_path: str | Path | None = None,
    verbose: bool = False,
) -> VerificationReport:
    root_path = Path(root).resolve()
    source = Path(source_path).resolve()
    target = Path(target_path).resolve()

    whitelist: set[str] = set()
    try:
        whitelist = _load_whitelist(axiom_whitelist)
    except Exception as exc:
        report = VerificationReport(
            source_path=str(source),
            target_path=str(target),
            root=str(root_path),
            config={
                "verbose": verbose,
                "axiom_whitelist": [],
                "axiom_whitelist_path": None if axiom_whitelist is None else str(axiom_whitelist),
            },
        )
        report.add_global_failure(FailureCode.PARSE_OR_COMPILE_ERROR, f"Whitelist error: {exc}")
        if save_path is not None:
            report.save_json(save_path)
        return report

    report = VerificationReport(
        source_path=str(source),
        target_path=str(target),
        root=str(root_path),
        config={
            "verbose": verbose,
            "axiom_whitelist": sorted(whitelist),
            "axiom_whitelist_path": None if axiom_whitelist is None else str(axiom_whitelist),
        },
    )

    if not root_path.exists() or not root_path.is_dir():
        report.add_global_failure(
            FailureCode.PARSE_OR_COMPILE_ERROR,
            f"Root directory does not exist: {root_path}",
        )
        if save_path is not None:
            report.save_json(save_path)
        return report

    for fp in (source, target):
        if not fp.exists():
            report.add_global_failure(
                FailureCode.PARSE_OR_COMPILE_ERROR,
                f"File does not exist: {fp}",
            )
            if save_path is not None:
                report.save_json(save_path)
            return report

    try:
        source.relative_to(root_path)
        target.relative_to(root_path)
    except ValueError as exc:
        report.add_global_failure(
            FailureCode.PARSE_OR_COMPILE_ERROR,
            f"Both files must be inside root '{root_path}'.",
        )
        if save_path is not None:
            report.save_json(save_path)
        return report

    try:
        source_proof_raw, source_ast_raw, source_diags = load_proof_dump(
            source,
            root=str(root_path),
            force_dump=True,
        )
        target_proof_raw, target_ast_raw, target_diags = load_proof_dump(
            target,
            root=str(root_path),
            force_dump=True,
        )

        source_dump = parse_proof_dump(source_proof_raw)
        target_dump = parse_proof_dump(target_proof_raw)
        source_ast = parse_ast_dump(source_ast_raw)
        target_ast = parse_ast_dump(target_ast_raw)
    except Exception as exc:
        report.add_global_failure(FailureCode.PARSE_OR_COMPILE_ERROR, f"Failed to parse proof dumps: {exc}")
        if save_path is not None:
            report.save_json(save_path)
        return report

    try:
        coqc_args, bindings = _collect_coqproject_loadpaths(root_path)
        source_module = _compute_logical_module(source, root_path, bindings)
        target_module = _compute_logical_module(target, root_path, bindings)
    except Exception as exc:
        report.add_global_failure(
            FailureCode.PARSE_OR_COMPILE_ERROR,
            f"Unable to resolve logical modules from root '{root_path}': {exc}",
        )
        if save_path is not None:
            report.save_json(save_path)
        return report

    source_decls = _extract_theorem_decls(source_ast)
    target_decls = _extract_theorem_decls(target_ast)
    source_rel_fqn = _proof_id_to_relative_fqn(source_dump.proofs, source_decls)
    target_rel_fqn = _proof_id_to_relative_fqn(target_dump.proofs, target_decls)

    source_local_axioms = _extract_assumption_names(source_ast_raw)
    target_local_axioms = _extract_assumption_names(target_ast_raw)

    obligations: list[Obligation] = []
    for proof in source_dump.proofs:
        if not _is_missing_obligation(proof, source_diags):
            continue
        rel_fqn = source_rel_fqn.get(proof.proof_id, proof.name)
        logical_name = f"{source_module}.{rel_fqn}" if source_module else rel_fqn
        obligations.append(
            Obligation(
                name=proof.name,
                source_proof_id=proof.proof_id,
                source_logical_name=logical_name,
                source_start_line=proof.start_range.start.line,
                source_start_character=proof.start_range.start.character,
            )
        )

    source_duplicate_names = _duplicate_names(ob.name for ob in obligations)
    target_duplicate_names = _duplicate_names(proof.name for proof in target_dump.proofs)

    if source_duplicate_names:
        report.add_global_failure(
            FailureCode.DUPLICATE_NAME,
            {
                "side": "source_obligations",
                "names": sorted(source_duplicate_names),
            },
        )
    if target_duplicate_names:
        report.add_global_failure(
            FailureCode.DUPLICATE_NAME,
            {
                "side": "target",
                "names": sorted(target_duplicate_names),
            },
        )

    source_proof_names = {proof.name for proof in source_dump.proofs}

    allowed_axioms = set(whitelist)
    for name in source_local_axioms:
        allowed_axioms.add(name)
        allowed_axioms.add(f"{source_module}.{name}")

    for proof in source_dump.proofs:
        for dep in proof.axioms:
            # Exclude theorem self-axioms from admitted obligations.
            if dep.name in source_proof_names:
                continue
            allowed_axioms.add(dep.name)
            if dep.logical_path:
                allowed_axioms.add(f"{dep.logical_path}.{dep.name}")

    new_local_axioms = sorted(
        name
        for name in target_local_axioms
        if not _is_name_whitelisted(name, target_module, allowed_axioms)
    )
    if new_local_axioms:
        report.add_global_failure(
            FailureCode.NEW_LOCAL_AXIOM,
            {
                "axioms": new_local_axioms,
            },
        )

    target_by_name: dict[str, list[ProofEntry]] = defaultdict(list)
    for proof in target_dump.proofs:
        target_by_name[proof.name].append(proof)

    statement_env_error: str | None = None
    statement_env_ready = False

    def ensure_statement_env() -> None:
        nonlocal statement_env_error, statement_env_ready
        if statement_env_ready or statement_env_error is not None:
            return
        try:
            _prepare_statement_env(source, target, coqc_args, root_path)
            statement_env_ready = True
        except Exception as exc:
            statement_env_error = str(exc)
            report.add_global_failure(FailureCode.PARSE_OR_COMPILE_ERROR, statement_env_error)

    outcomes: list[CheckOutcome] = []

    for obligation in obligations:
        outcome = CheckOutcome(
            obligation=obligation,
            matched_target=None,
            checks={
                "statement": False,
                "completeness": False,
                "axioms": False,
            },
        )

        if obligation.name in source_duplicate_names or obligation.name in target_duplicate_names:
            outcome.add_failure(FailureCode.DUPLICATE_NAME, {"name": obligation.name})
            outcomes.append(outcome)
            continue

        candidates = target_by_name.get(obligation.name, [])
        if not candidates:
            outcome.add_failure(FailureCode.MISSING_OBLIGATION, {"name": obligation.name})
            outcomes.append(outcome)
            continue

        if len(candidates) != 1:
            outcome.add_failure(
                FailureCode.DUPLICATE_NAME,
                {
                    "name": obligation.name,
                    "target_candidates": [x.name for x in candidates],
                },
            )
            outcomes.append(outcome)
            continue

        target_proof = candidates[0]
        outcome.matched_target = target_proof.name

        complete, complete_details = _target_proof_completeness(target_proof, target_diags)
        outcome.checks["completeness"] = complete
        if not complete:
            outcome.add_failure(FailureCode.INCOMPLETE_PROOF, complete_details)

        disallowed = _disallowed_axioms(target_proof, allowed_axioms)
        outcome.checks["axioms"] = len(disallowed) == 0
        if disallowed:
            outcome.add_failure(FailureCode.DISALLOWED_AXIOMS, {"axioms": disallowed})

        if complete:
            ensure_statement_env()
            if statement_env_error is not None:
                outcome.add_failure(FailureCode.PARSE_OR_COMPILE_ERROR, statement_env_error)
            else:
                target_rel = target_rel_fqn.get(target_proof.proof_id, target_proof.name)
                target_logical_name = f"{target_module}.{target_rel}" if target_module else target_rel
                statement_ok, statement_error = _check_statement_convertibility(
                    root=root_path,
                    coqc_args=coqc_args,
                    source_module=source_module,
                    target_module=target_module,
                    source_ref=obligation.source_logical_name,
                    target_ref=target_logical_name,
                )
                outcome.checks["statement"] = statement_ok
                if not statement_ok:
                    outcome.add_failure(
                        FailureCode.STATEMENT_MISMATCH,
                        {
                            "source": obligation.source_logical_name,
                            "target": target_logical_name,
                            "error": statement_error,
                        },
                    )

        if new_local_axioms:
            outcome.add_failure(FailureCode.NEW_LOCAL_AXIOM, {"axioms": new_local_axioms})

        outcomes.append(outcome)

    report.outcomes = outcomes

    if save_path is not None:
        report.save_json(save_path)

    return report


__all__ = ["run_safeverify"]
