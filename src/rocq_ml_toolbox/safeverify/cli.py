from __future__ import annotations

import argparse
import json
from typing import Sequence

from .core import run_safeverify


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocq-ml-safeverify")
    parser.add_argument("source", help="Path to source .v file (contains missing proofs).")
    parser.add_argument("target", help="Path to target .v file (proposed answer).")
    parser.add_argument(
        "--root",
        required=True,
        help="Project root used for stable logical path resolution.",
    )
    parser.add_argument(
        "--axiom-whitelist",
        help="Path to a JSON file containing a list of allowed axioms.",
    )
    parser.add_argument(
        "--save",
        help="Save a detailed JSON report to this file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed failure information.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report = run_safeverify(
        args.source,
        args.target,
        root=args.root,
        axiom_whitelist=args.axiom_whitelist,
        save_path=args.save,
        verbose=args.verbose,
    )

    summary = report.summary()
    print(
        f"ok={report.ok} obligations={summary['num_obligations']} "
        f"passed={summary['passed']} failed={summary['failed']} "
        f"global_failures={summary['global_failures']}"
    )

    if args.verbose:
        for gf in report.global_failures:
            print(f"[global] {gf.code.value}: {gf.details}")
        for outcome in report.outcomes:
            if outcome.ok:
                continue
            print(
                f"[obligation {outcome.obligation.name}] "
                f"failures={','.join(code.value for code in outcome.failure_codes)}"
            )
            if outcome.details:
                print(json.dumps(outcome.details, indent=2))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
