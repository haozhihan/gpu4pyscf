#!/usr/bin/env python3
"""Generate a deterministic rank/tolerance sweep without running CCSD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "threshold_grid.json")
    p.add_argument("--eri-tol", nargs="+", type=float, default=[1e-4, 1e-6, 1e-8])
    p.add_argument("--direct-scf-tol", nargs="+", type=float, default=[1e-13])
    p.add_argument(
        "--denominator-tolerance", nargs="+", type=float, default=[1e-10]
    )
    p.add_argument(
        "--rr-cutoff",
        nargs="+",
        type=float,
        default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-11],
    )
    p.add_argument(
        "--thc-fit-tol", nargs="+", type=float,
        default=[1e-4, 1e-5, 1e-6, 1e-7],
    )
    p.add_argument(
        "--fno-thresh", nargs="+", type=float,
        default=[1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7],
    )
    p.add_argument("--dry-run", action="store_true", help="print instead of writing")
    a = p.parse_args()
    if any(
        x < 0
        for values in (
            a.eri_tol,
            a.direct_scf_tol,
            a.denominator_tolerance,
            a.rr_cutoff,
            a.thc_fit_tol,
            a.fno_thresh,
        )
        for x in values
    ):
        p.error("thresholds must be non-negative")
    if any(x == 0 for x in a.direct_scf_tol):
        p.error("direct SCF thresholds must be positive")
    grid = {
        "cases": ["water2-tz", "water4-tz", "water8-tz", "water8s4-tz"],
        "methods": [
            "canonical",
            "fno",
            "rr_cd",
            "rr_canonical",
            "thc_canonical",
        ],
        "eri_tol": a.eri_tol,
        "direct_scf_tol": a.direct_scf_tol,
        "denominator_tolerance": a.denominator_tolerance,
        "rr_eig_cutoff": a.rr_cutoff,
        "thc_fit_tol": a.thc_fit_tol,
        "fno_thresh": a.fno_thresh,
        "gate_policy": {
            "water2": "complete grid",
            "water4": "first passing candidate and next tighter setting",
            "water8": "two candidates advanced from water4; select fastest passing",
            "water8s4": "selected water8 parameters; accuracy and resource holdout only",
        },
        "validation_status": {
            "canonical": "production-canonical",
            "fno": "production-fno",
            "rr_cd": "direct-cd-compressed-rr-reference",
            "rr_canonical": "dense-projected-validation",
            "thc_canonical": "dense-two-level-validation-surrogate",
        },
        "performance_eligibility": {
            "rr_cd": {
                "eligible": False,
                "pending_gates": [
                    "GINT restricted-column water2 timing gate",
                    "RR ring contraction production kernel",
                ],
            }
        },
    }
    text = json.dumps(grid, indent=2, sort_keys=True) + "\n"
    if a.dry_run:
        print(text, end="")
    else:
        if a.output.exists():
            raise RuntimeError(f"refusing to overwrite {a.output}")
        a.output.write_text(text)
        print(a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
