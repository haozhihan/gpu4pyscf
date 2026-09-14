#!/usr/bin/env python3
"""Dependency-light validation for the A100 harness and frozen geometries."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    payload = json.loads((ROOT / "cases.json").read_text())
    cases = payload["cases"]
    assert [case["id"] for case in cases] == [
        "water2-tz", "water4-tz", "water8-tz", "water8s4-tz"
    ]
    assert [len(case["geometry_angstrom"]) for case in cases] == [6, 12, 24, 24]
    assert [case["role"] for case in cases] == [
        "sanity", "development", "target", "holdout"
    ]
    assert all(case["basis"] == "cc-pVTZ" for case in cases)
    hashes = []
    for case in cases:
        canonical = json.dumps(case["geometry_angstrom"], separators=(",", ":"), ensure_ascii=True)
        hashes.append(hashlib.sha256(canonical.encode()).hexdigest())
    assert len(set(hashes)) == 4

    cmd = [sys.executable, str(ROOT / "benchmark.py"), "--case", "water2-tz", "--method", "canonical", "--device", "cpu", "--dry-run"]
    dry = subprocess.run(cmd, check=True, capture_output=True, text=True)
    plan = json.loads(dry.stdout)
    assert plan["dry_run"] is True
    assert plan["plan"]["validation_status"] == "production-canonical"

    grid = subprocess.run([sys.executable, str(ROOT / "make_threshold_grid.py"), "--dry-run"], check=True, capture_output=True, text=True)
    assert "rr_eig_cutoff" in grid.stdout and "thc_fit_tol" in grid.stdout
    status = json.loads((ROOT / "implementation-status.json").read_text())
    assert status["schema"] == "gpu4pyscf.water8.implementation-status.v2"
    assert status["contract"]["reporting_target_seconds"] == 133.04
    assert status["contract"]["derived_exact_target_seconds"] == 133.0347
    claim_policy = status["claim_policy"]
    assert claim_policy["ten_x_claim_allowed"] is False
    assert claim_policy["reason"]
    evidence_states = set(claim_policy["evidence_states"])
    assert evidence_states == {
        "source_reported", "local_measured", "derived", "hypothesis"
    }
    performance_claim = status["performance_claim"]
    assert performance_claim["eligible"] is False
    assert performance_claim["ten_x_achieved"] is False
    assert performance_claim["candidate_water8_median_seconds"] is None
    assert performance_claim["accuracy_qualified_candidate_count"] == 0
    assert performance_claim["eligible_canonical_water8_sample_count"] == len(
        performance_claim["eligible_canonical_water8_samples_seconds"]
    )
    canonical_samples = performance_claim[
        "eligible_canonical_water8_samples_seconds"
    ]
    assert len(canonical_samples) >= 3
    assert performance_claim["matched_canonical_water8_median_seconds"] == (
        statistics.median(canonical_samples)
    )
    assert performance_claim["reason"]
    assert [gate["id"] for gate in status["gates"]] == [
        "G0", "G1", "G2", "G3", "G4", "G5", "G6"
    ]
    assert all(gate["evidence_state"] in evidence_states for gate in status["gates"])
    print(
        "self-check passed: cases, CLI dry-run, threshold grid, and "
        "fail-closed implementation status"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
