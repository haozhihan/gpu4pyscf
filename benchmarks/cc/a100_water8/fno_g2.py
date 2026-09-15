#!/usr/bin/env python3
"""Plan, submit, and audit the water4 FNO + Delta-MP2 G2 probe.

The probe is deliberately separate from the final 10x analyzer.  It runs one
fresh canonical/FNO A/B pair for every frozen-natural-orbital occupation
threshold, proves that each pair used the exact same serialized RHF orbitals,
and joins those timing records to counterpoise calculations made from one
shared cluster/ghost-monomer orbital bundle.  Only the first loose-to-tight
setting that passes every water4 gate and its immediately tighter neighbour
may be proposed for water8.

``plan`` writes a machine-readable Slurm DAG without submitting it.  ``submit``
prints that DAG by default and requires ``--execute`` to call ``sbatch``.
``analyze`` consumes synchronized JSON records and refuses to advance an
incomplete or ambiguously paired threshold grid.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

try:
    from . import analyze_results as _analysis
except ImportError:  # direct execution from this directory
    _module_root = str(Path(__file__).resolve().parent)
    if _module_root not in sys.path:
        sys.path.insert(0, _module_root)
    import analyze_results as _analysis


SCHEMA = "gpu4pyscf.water8.fno-g2.v1"
PLAN_SCHEMA = "gpu4pyscf.water8.fno-g2-plan.v1"
SUBMISSION_SCHEMA = "gpu4pyscf.water8.fno-g2-submission.v1"
CASE_ID = "water4-tz"
FNO_THRESHOLDS = (1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7)
ECORR_LIMIT_EH = 1e-4
CP_LIMIT_KCAL_MOL = 0.02
ACTIVE_RESIDUAL_LIMIT = 1e-6
HBM_LIMIT_GIB = 72.0
RSS_LIMIT_GIB = 110.0
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _json_write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {path}") from exc


def _cutoff_token(value: float) -> str:
    if value not in FNO_THRESHOLDS:
        raise ValueError(f"unsupported FNO G2 threshold {value!r}")
    coefficient, exponent = f"{value:.0e}".split("e")
    return f"{coefficient}e{int(exponent)}"


def _source_digest(source_root: Path, task_root: Path) -> str:
    source = source_root.expanduser().resolve()
    task = task_root.expanduser().resolve()
    try:
        relative = source.relative_to(task / "snapshots")
    except ValueError as exc:
        raise ValueError(
            "source root must be below <task-root>/snapshots/<sha256>/source"
        ) from exc
    if len(relative.parts) != 2 or relative.parts[1] != "source":
        raise ValueError(
            "source root must equal <task-root>/snapshots/<sha256>/source"
        )
    digest = relative.parts[0]
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError("source snapshot directory must be a lowercase SHA-256")
    return digest


def make_plan(
    *,
    task_root: Path,
    source_root: Path,
    run_id: str,
    node: str,
) -> dict[str, Any]:
    """Return the fixed seven-threshold water4 job DAG."""

    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run id must start with an alphanumeric character and contain "
            "only letters, digits, dot, underscore, or hyphen"
        )
    if not node or any(character.isspace() for character in node):
        raise ValueError("node must be one non-empty Slurm host name")
    task = task_root.expanduser().resolve()
    source = source_root.expanduser().resolve()
    digest = _source_digest(source, task)
    result_root = task / "results" / "fno-g2" / run_id
    timing_root = result_root / "timing"
    orbital_root = result_root / "orbitals"
    cp_root = result_root / "counterpoise"
    cp_bundle = cp_root / "water4-tz-canonical-orbitals.npz"
    benchmark_launcher = (
        source / "benchmarks" / "cc" / "a100_water8" /
        "run_mtu_benchmark.sbatch"
    )
    cp_launcher = (
        source / "benchmarks" / "cc" / "a100_water8" /
        "run_mtu_counterpoise.sbatch"
    )
    common = {
        "CCSD_TASK_ROOT": str(task),
        "CCSD_SOURCE_ROOT": str(source),
        "CCSD_EXPECTED_DEPLOYMENT_PROFILE": "candidate",
        "CCSD_REQUIRE_CANONICAL_PRISTINE": "0",
    }
    jobs: list[dict[str, Any]] = []
    previous_timing_job: str | None = None
    for index, cutoff in enumerate(FNO_THRESHOLDS):
        token = _cutoff_token(cutoff)
        artifact = orbital_root / f"water4-tz-{token}.npz"
        oracle_id = f"timing-{token}-canonical"
        candidate_id = f"timing-{token}-fno"
        oracle_repeat = f"{run_id}-{token}-canonical-a"
        candidate_repeat = f"{run_id}-{token}-fno-b"
        jobs.append({
            "id": oracle_id,
            "phase": "water4-timing",
            "ordinal": 2 * index,
            "launcher": str(benchmark_launcher),
            "slurm_options": [f"--nodelist={node}"],
            "depends_on": (
                [] if previous_timing_job is None else [previous_timing_job]
            ),
            "exports": {
                **common,
                "CASE": CASE_ID,
                "METHOD": "canonical",
                "REPEAT": oracle_repeat,
                "BENCHMARK_OUTPUT_DIR": str(timing_root),
                "BENCHMARK_ORBITAL_ARTIFACT_OUT": str(artifact),
            },
            "expected_outputs": [
                str(timing_root / f"{CASE_ID}__canonical__r{oracle_repeat}.json"),
                str(artifact),
            ],
            "role": "A: canonical orbital producer and timing oracle",
        })
        jobs.append({
            "id": candidate_id,
            "phase": "water4-timing",
            "ordinal": 2 * index + 1,
            "launcher": str(benchmark_launcher),
            "slurm_options": [f"--nodelist={node}"],
            "depends_on": [oracle_id],
            "exports": {
                **common,
                "CASE": CASE_ID,
                "METHOD": "fno",
                "REPEAT": candidate_repeat,
                "BENCHMARK_OUTPUT_DIR": str(timing_root),
                "BENCHMARK_ORBITAL_ARTIFACT_IN": str(artifact),
                "FNO_THRESH": repr(cutoff),
            },
            "expected_outputs": [
                str(timing_root / f"{CASE_ID}__fno__r{candidate_repeat}.json"),
            ],
            "role": "B: FNO-CCSD + Delta-MP2 timing candidate",
            "fno_thresh": cutoff,
        })
        previous_timing_job = candidate_id

    cp_oracle_id = "counterpoise-canonical"
    jobs.append({
        "id": cp_oracle_id,
        "phase": "water4-counterpoise",
        "ordinal": len(jobs),
        "launcher": str(cp_launcher),
        "slurm_options": [f"--nodelist={node}"],
        # The node has four A100s.  Without this cross-phase dependency Slurm
        # may co-schedule the CP oracle beside the timing chain and make the
        # A/B wall times depend on shared CPU, memory, and PCIe traffic.
        "depends_on": [previous_timing_job],
        "exports": {
            **common,
            "CASE": CASE_ID,
            "METHOD": "canonical",
            "CP_RUN_ID": f"{run_id}-canonical",
            "CP_OUTPUT_ROOT": str(cp_root),
            "CP_ORBITAL_BUNDLE_OUT": str(cp_bundle),
        },
        "expected_outputs": [
            str(cp_root / f"{CASE_ID}__canonical__{run_id}-canonical.json"),
            str(cp_bundle),
        ],
        "role": "canonical Boys--Bernardi oracle and shared bundle producer",
    })
    previous_cp_job = cp_oracle_id
    for index, cutoff in enumerate(FNO_THRESHOLDS):
        token = _cutoff_token(cutoff)
        job_id = f"counterpoise-{token}-fno"
        cp_run_id = f"{run_id}-{token}-fno"
        jobs.append({
            "id": job_id,
            "phase": "water4-counterpoise",
            "ordinal": len(jobs),
            "launcher": str(cp_launcher),
            "slurm_options": [f"--nodelist={node}"],
            # Serialize the CP consumers so every job has an unambiguous
            # predecessor and the fixed node never runs competing probes.
            "depends_on": [previous_cp_job],
            "exports": {
                **common,
                "CASE": CASE_ID,
                "METHOD": "fno",
                "CP_RUN_ID": cp_run_id,
                "CP_OUTPUT_ROOT": str(cp_root),
                "CP_ORBITAL_BUNDLE_IN": str(cp_bundle),
                "FNO_THRESH": repr(cutoff),
            },
            "expected_outputs": [
                str(cp_root / f"{CASE_ID}__fno__{cp_run_id}.json"),
            ],
            "role": "FNO Boys--Bernardi candidate from shared CP bundle",
            "fno_thresh": cutoff,
        })
        previous_cp_job = job_id

    return {
        "schema": PLAN_SCHEMA,
        "case_id": CASE_ID,
        "run_id": run_id,
        "task_root": str(task),
        "source_root": str(source),
        "source_tree_sha256": digest,
        "node": node,
        "threshold_order": list(FNO_THRESHOLDS),
        "result_root": str(result_root),
        "timing_result_root": str(timing_root),
        "counterpoise_result_root": str(cp_root),
        "policies": {
            "node_isolation": (
                "all 22 jobs form one afterok chain; counterpoise starts only "
                "after the final timing candidate so a four-GPU node cannot "
                "co-schedule validation traffic beside an A/B timing run"
            ),
            "timing": (
                "one fresh-process canonical/FNO pair per threshold, serialized "
                "on one node; all post-HF preprocessing and checkpoint included"
            ),
            "timing_orbitals": (
                "each A/B pair must use one exact canonical orbital artifact"
            ),
            "counterpoise": (
                "one canonical cluster-plus-ghost-monomer orbital bundle shared "
                "by the oracle and all seven FNO candidates"
            ),
            "advancement": (
                "first loose-to-tight fully passing threshold and its immediate "
                "tighter neighbour; both must independently pass"
            ),
        },
        "limits": {
            "ecorr_error_eh": ECORR_LIMIT_EH,
            "counterpoise_error_kcal_mol": CP_LIMIT_KCAL_MOL,
            "active_space_equation_residual": ACTIVE_RESIDUAL_LIMIT,
            "hbm_gib": HBM_LIMIT_GIB,
            "host_rss_gib": RSS_LIMIT_GIB,
            "water4_speedup_strictly_greater_than": 1.0,
        },
        "jobs": jobs,
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _at(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _record_path(record: dict[str, Any]) -> str | None:
    value = record.get("_fno_g2_input_path")
    return None if value is None else str(value)


def _load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for raw in paths:
        path = raw.expanduser()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for path in sorted(set(item.resolve() for item in files)):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read JSON record {path}: {exc}") from exc
        if not isinstance(value, dict):
            continue
        is_cp = value.get("schema") == "gpu4pyscf.water8.counterpoise.v1"
        is_timed = (
            value.get("case_id") == CASE_ID
            and value.get("method") in {"canonical", "fno"}
            and "post_hf_seconds" in value
        )
        if is_cp or is_timed:
            value = dict(value)
            value["_fno_g2_input_path"] = str(path)
            records.append(value)
    return records


def _threshold(record: dict[str, Any]) -> float | None:
    selection = _at(record, "approximation.selection")
    if not isinstance(selection, dict):
        selection = _at(record, "plan.approximation.selection")
    if not isinstance(selection, dict):
        return None
    if selection.get("mode") != "occupation_threshold":
        return None
    return _number(selection.get("value"))


def _same_float(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    lhs, rhs = _number(left), _number(right)
    return bool(
        lhs is not None and rhs is not None
        and math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=atol)
    )


def _orbital_pair_proof(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    left = baseline.get("canonical_orbitals")
    right = candidate.get("canonical_orbitals")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False, ["canonical orbital evidence is missing"], {}
    if left.get("mode") != "written-and-reloaded":
        reasons.append("canonical A record did not write and reload its artifact")
    if right.get("mode") != "loaded":
        reasons.append("FNO B record did not load the producer artifact")
    for side, payload in (("canonical", left), ("FNO", right)):
        if payload.get("applied_to_mean_field") is not True:
            reasons.append(f"{side} record did not apply the serialized orbitals")
        if payload.get("included_in_post_hf") is not False:
            reasons.append(f"{side} orbital setup timing boundary is ambiguous")
    for field in (
        "schema", "artifact_path", "artifact_sha256", "orbital_fingerprint",
        "identity", "producer",
    ):
        if left.get(field) != right.get(field):
            reasons.append(f"canonical/FNO orbital artifact mismatch: {field}")
    for side, record, payload in (
        ("canonical", baseline, left), ("FNO", candidate, right)
    ):
        checkpoint = record.get("checkpoint")
        if not isinstance(checkpoint, dict):
            reasons.append(f"{side} checkpoint evidence is missing")
            continue
        if checkpoint.get("included_in_post_hf") is not True:
            reasons.append(f"{side} checkpoint was excluded from post-HF time")
        if checkpoint.get("orbital_artifact_sha256") != payload.get(
            "artifact_sha256"
        ):
            reasons.append(f"{side} checkpoint artifact SHA does not match")
        if checkpoint.get("orbital_fingerprint") != payload.get(
            "orbital_fingerprint"
        ):
            reasons.append(f"{side} checkpoint orbital fingerprint does not match")
    evidence = {
        "artifact_path": left.get("artifact_path"),
        "artifact_sha256": left.get("artifact_sha256"),
        "orbital_fingerprint": left.get("orbital_fingerprint"),
        "identity": left.get("identity"),
        "canonical_mode": left.get("mode"),
        "candidate_mode": right.get("mode"),
    }
    return not reasons, reasons, evidence


def _fno_audit(
    record: dict[str, Any], expected_threshold: float
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    audit = record.get("fno")
    if not isinstance(audit, dict):
        return False, ["FNO audit metadata is missing"], {}
    if audit.get("selection_mode") != "occupation_threshold":
        reasons.append("FNO effective selector is not occupation_threshold")
    if not _same_float(audit.get("selection_value"), expected_threshold):
        reasons.append("FNO selection value does not match the threshold")
    if not _same_float(audit.get("threshold"), expected_threshold):
        reasons.append("FNO audit threshold does not match the plan")
    if audit.get("requested_nvir_act") is not None:
        reasons.append("FNO scan unexpectedly used a fixed active rank")
    if audit.get("pct_occ") is not None:
        reasons.append("FNO scan unexpectedly used cumulative occupation")
    nvir = audit.get("nvir")
    nvir_active = audit.get("nvir_active")
    if (
        isinstance(nvir, bool) or not isinstance(nvir, int) or nvir < 1
        or isinstance(nvir_active, bool) or not isinstance(nvir_active, int)
        or not 1 <= nvir_active <= nvir
    ):
        reasons.append("FNO active/full virtual dimensions are invalid")
    frozen = audit.get("frozen_virtual")
    active = audit.get("active_virtual")
    if isinstance(nvir, int) and isinstance(nvir_active, int):
        if not isinstance(frozen, list) or len(frozen) != nvir - nvir_active:
            reasons.append("FNO frozen virtual count is inconsistent")
        if not isinstance(active, list) or len(active) != nvir_active:
            reasons.append("FNO active virtual count is inconsistent")
    occupations = audit.get("virtual_occupations")
    if not isinstance(occupations, list) or (
        isinstance(nvir, int) and len(occupations) != nvir
    ):
        reasons.append("FNO virtual occupation spectrum is missing or incomplete")
    elif any(
        _number(left) is None or _number(right) is None
        or float(left) < float(right) - 1e-14
        for left, right in zip(occupations, occupations[1:])
    ):
        reasons.append("FNO virtual occupations are not finite descending values")

    delta = _number(record.get("delta_mp2"))
    raw_corr = _number(record.get("raw_fno_e_corr"))
    corrected_corr = _number(record.get("e_corr"))
    raw_total = _number(record.get("raw_fno_e_tot"))
    corrected_total = _number(record.get("e_tot"))
    mp2 = audit.get("mp2")
    if not isinstance(mp2, dict):
        reasons.append("FNO MP2 audit is missing")
        mp2 = {}
    full_mp2 = _number(mp2.get("full_correlation_energy_eh"))
    active_mp2 = _number(mp2.get("fno_correlation_energy_eh"))
    audit_delta = _number(mp2.get("delta_mp2_eh"))
    for label, value in (
        ("top-level Delta-MP2", delta),
        ("raw FNO correlation energy", raw_corr),
        ("corrected FNO correlation energy", corrected_corr),
        ("raw FNO total energy", raw_total),
        ("corrected FNO total energy", corrected_total),
        ("full-space MP2 correlation energy", full_mp2),
        ("FNO-space MP2 correlation energy", active_mp2),
        ("audited Delta-MP2", audit_delta),
    ):
        if value is None:
            reasons.append(f"{label} is missing or non-finite")
    if delta is not None and audit_delta is not None and not _same_float(
        delta, audit_delta, atol=1e-10
    ):
        reasons.append("top-level and audited Delta-MP2 values disagree")
    if (
        full_mp2 is not None and active_mp2 is not None and delta is not None
        and not _same_float(full_mp2 - active_mp2, delta, atol=1e-10)
    ):
        reasons.append("Delta-MP2 is not full-space MP2 minus FNO-space MP2")
    if (
        raw_corr is not None and delta is not None and corrected_corr is not None
        and not _same_float(raw_corr + delta, corrected_corr, atol=1e-10)
    ):
        reasons.append("reported corrected correlation energy omits/misapplies Delta-MP2")
    if (
        raw_total is not None and delta is not None and corrected_total is not None
        and not _same_float(raw_total + delta, corrected_total, atol=1e-10)
    ):
        reasons.append("reported corrected total energy omits/misapplies Delta-MP2")
    if record.get("energy_definition") != (
        "FNO-CCSD + Delta-MP2 (full-space MP2 minus FNO-space MP2)"
    ):
        reasons.append("FNO corrected-energy definition is missing or changed")

    timings = audit.get("timings_s")
    required_timings = (
        "mp2_full", "occupation_and_fno_orbitals",
        "semicanonical_fock_validation", "mp2_fno", "total",
    )
    if not isinstance(timings, dict):
        reasons.append("FNO setup timing audit is missing")
        timings = {}
    elif any(
        _number(timings.get(name)) is None or float(timings[name]) < 0.0
        for name in required_timings
    ):
        reasons.append("FNO setup timings are missing, non-finite, or negative")
    evidence = {
        "selection_mode": audit.get("selection_mode"),
        "selection_value": audit.get("selection_value"),
        "nvir": nvir,
        "nvir_active": nvir_active,
        "nvir_frozen": (
            nvir - nvir_active
            if isinstance(nvir, int) and isinstance(nvir_active, int)
            else None
        ),
        "lowest_retained_occupation": _at(
            audit, "occupation_boundary.lowest_retained"
        ),
        "highest_discarded_occupation": _at(
            audit, "occupation_boundary.highest_discarded"
        ),
        "semicanonical_offdiag_max": audit.get(
            "semicanonical_offdiag_max"
        ),
        "delta_mp2_eh": delta,
        "mp2_full_corr_eh": full_mp2,
        "mp2_fno_corr_eh": active_mp2,
        "setup_timings_s": timings,
    }
    return not reasons, reasons, evidence


def _phase_totals(record: dict[str, Any]) -> dict[str, float]:
    phases = _at(record, "run_metrics.phase_totals")
    if not isinstance(phases, dict):
        return {}
    output: dict[str, float] = {}
    for name, payload in phases.items():
        if not isinstance(payload, dict):
            continue
        value = _number(payload.get("total_s"))
        if value is not None and value >= 0.0:
            output[str(name)] = value
    return dict(sorted(output.items()))


def _candidate_row(
    *,
    threshold: float,
    candidates: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    cp_records: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    if len(candidates) != 1:
        reasons.append(
            f"expected exactly one FNO timing record; found {len(candidates)}"
        )
    candidate = candidates[0] if len(candidates) == 1 else None
    matching: list[tuple[dict[str, Any], dict[str, Any], list[str]]] = []
    if candidate is not None:
        for baseline in baselines:
            paired, orbital_reasons, orbital_evidence = _orbital_pair_proof(
                baseline, candidate
            )
            if paired:
                matching.append((baseline, orbital_evidence, orbital_reasons))
    if len(matching) != 1:
        reasons.append(
            "expected exactly one canonical timing record with the same exact "
            f"orbital artifact; found {len(matching)}"
        )
    baseline = matching[0][0] if len(matching) == 1 else None
    orbital_evidence = matching[0][1] if len(matching) == 1 else {}

    compatibility: dict[str, Any] = {"compatible": False, "reasons": []}
    fno_evidence: dict[str, Any] = {}
    fno_passed = False
    source_digest: str | None = None
    cp_error: float | None = None
    cp_reasons: list[str] = []
    ecorr_error: float | None = None
    residual: float | None = None
    resources: dict[str, Any] = {}
    baseline_seconds: float | None = None
    candidate_seconds: float | None = None
    speedup: float | None = None
    signature: str | None = None
    if candidate is not None:
        fno_passed, fno_reasons, fno_evidence = _fno_audit(
            candidate, threshold
        )
        reasons.extend(fno_reasons)
        if candidate.get("status") != "completed" or candidate.get(
            "cc_converged"
        ) is not True:
            reasons.append("FNO timing record is missing or unconverged")
        if candidate.get("performance_eligible") is False:
            reasons.append("FNO timing record is explicitly performance-ineligible")
        residual = _number(_at(candidate, "residual.equation_norm"))
        if _at(candidate, "residual.orbital_space") != "fno-active":
            reasons.append("FNO active-space residual identity is missing")
        if _at(candidate, "residual.is_full_space") is not False:
            reasons.append("FNO residual is not marked as active-space only")
        if residual is None or residual > ACTIVE_RESIDUAL_LIMIT:
            reasons.append("FNO active-space equation residual exceeds 1e-6")
        resources = _analysis._resources(candidate)
        if resources.get("passed") is not True:
            reasons.append("FNO resource or fixed-hardware gate failed")
        candidate_seconds = _number(candidate.get("post_hf_seconds"))
        signature = _analysis._approximation_signature(candidate)
        source_digest = _analysis._source_digest(candidate)
    if baseline is not None and candidate is not None:
        if baseline.get("status") != "completed" or baseline.get(
            "cc_converged"
        ) is not True:
            reasons.append("canonical timing oracle is missing or unconverged")
        compatibility = _analysis.compatibility(baseline, candidate)
        reasons.extend(str(item) for item in compatibility.get("reasons", []))
        baseline_corr = _number(baseline.get("e_corr"))
        candidate_corr = _number(candidate.get("e_corr"))
        if baseline_corr is not None and candidate_corr is not None:
            ecorr_error = abs(candidate_corr - baseline_corr)
        if ecorr_error is None or ecorr_error > ECORR_LIMIT_EH:
            reasons.append("Delta-MP2-corrected correlation error exceeds 1e-4 Eh")
        baseline_seconds = _number(baseline.get("post_hf_seconds"))
        if (
            baseline_seconds is not None and candidate_seconds is not None
            and candidate_seconds > 0.0
        ):
            speedup = baseline_seconds / candidate_seconds
        if speedup is None or speedup <= 1.0:
            reasons.append("water4 FNO post-HF time is not faster than canonical")
    if candidate is not None and signature is not None and source_digest is not None:
        cp_error, cp_reasons = _analysis._counterpoise_error(
            cp_records,
            CASE_ID,
            "canonical",
            "fno",
            signature,
            source_digest,
        )
        reasons.extend(cp_reasons)
        if cp_error is None or cp_error > CP_LIMIT_KCAL_MOL:
            reasons.append("counterpoise interaction-energy error exceeds 0.02 kcal/mol")
    else:
        reasons.append("candidate signature/source is unavailable for CP pairing")

    reasons = list(dict.fromkeys(reasons))
    return {
        "threshold": threshold,
        "threshold_token": _cutoff_token(threshold),
        "approximation_signature": signature,
        "source_tree_sha256": source_digest,
        "records": {
            "canonical": None if baseline is None else _record_path(baseline),
            "candidate": None if candidate is None else _record_path(candidate),
        },
        "canonical_orbital_pair": {
            "passed": bool(baseline is not None and candidate is not None),
            **orbital_evidence,
        },
        "compatibility": compatibility,
        "timing": {
            "canonical_post_hf_s": baseline_seconds,
            "candidate_post_hf_s": candidate_seconds,
            "speedup": speedup,
            "candidate_phase_totals_s": (
                {} if candidate is None else _phase_totals(candidate)
            ),
        },
        "accuracy": {
            "ecorr_error_eh": ecorr_error,
            "ecorr_limit_eh": ECORR_LIMIT_EH,
            "counterpoise_error_kcal_mol": cp_error,
            "counterpoise_limit_kcal_mol": CP_LIMIT_KCAL_MOL,
            "active_space_equation_residual": residual,
            "active_space_residual_limit": ACTIVE_RESIDUAL_LIMIT,
        },
        "fno": {"audit_passed": fno_passed, **fno_evidence},
        "resources": resources,
        "qualifies_for_water8": not reasons,
        "reasons": reasons,
    }


def analyze_fno_g2(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit a synchronized water4 grid and select at most two water8 probes."""

    timed = [
        item for item in records
        if item.get("case_id") == CASE_ID
        and item.get("method") in {"canonical", "fno"}
        and "post_hf_seconds" in item
    ]
    cp_records = [
        item for item in records
        if item.get("schema") == "gpu4pyscf.water8.counterpoise.v1"
    ]
    baselines = [item for item in timed if item.get("method") == "canonical"]
    fno_records = [item for item in timed if item.get("method") == "fno"]
    unexpected = [
        item for item in fno_records
        if _threshold(item) not in FNO_THRESHOLDS
    ]
    rows = [
        _candidate_row(
            threshold=cutoff,
            candidates=[
                item for item in fno_records
                if _same_float(_threshold(item), cutoff)
            ],
            baselines=baselines,
            cp_records=cp_records,
        )
        for cutoff in FNO_THRESHOLDS
    ]
    grid_complete = all(
        sum(_same_float(_threshold(item), cutoff) for item in fno_records) == 1
        for cutoff in FNO_THRESHOLDS
    ) and not unexpected
    passing_indices = [
        index for index, row in enumerate(rows)
        if row["qualifies_for_water8"]
    ]
    selection_reasons: list[str] = []
    selected: list[dict[str, Any]] = []
    first_index: int | None = None
    if not grid_complete:
        selection_reasons.append(
            "the exact seven-threshold water4 grid is incomplete or contains "
            "unexpected FNO selectors"
        )
    elif not passing_indices:
        selection_reasons.append("no water4 FNO threshold passed every gate")
    else:
        first_index = passing_indices[0]
        next_index = first_index + 1
        if next_index >= len(rows):
            selection_reasons.append(
                "the first passing threshold is the tightest grid point; no "
                "immediately tighter confirmation setting exists"
            )
        elif not rows[next_index]["qualifies_for_water8"]:
            selection_reasons.append(
                "the setting immediately tighter than the first pass did not "
                "independently pass every gate"
            )
        else:
            selected = [rows[first_index], rows[next_index]]
    water8_candidates = [
        {
            "fno_thresh": row["threshold"],
            "approximation_signature": row["approximation_signature"],
            "water4_nvir_active": row["fno"].get("nvir_active"),
            "water4_post_hf_s": row["timing"]["candidate_post_hf_s"],
            "water4_speedup": row["timing"]["speedup"],
            "water4_ecorr_error_eh": row["accuracy"]["ecorr_error_eh"],
            "water4_counterpoise_error_kcal_mol": row["accuracy"][
                "counterpoise_error_kcal_mol"
            ],
            "water4_delta_mp2_eh": row["fno"].get("delta_mp2_eh"),
        }
        for row in selected
    ]
    return {
        "schema": SCHEMA,
        "case_id": CASE_ID,
        "threshold_order": list(FNO_THRESHOLDS),
        "records_loaded": len(records),
        "timing_records_loaded": len(timed),
        "counterpoise_records_loaded": len(cp_records),
        "unexpected_fno_records": [
            _record_path(item) for item in unexpected
        ],
        "grid_complete": grid_complete,
        "rows": rows,
        "selection": {
            "ready_for_water8": len(water8_candidates) == 2,
            "first_passing_cutoff": (
                None if first_index is None else rows[first_index]["threshold"]
            ),
            "next_tighter_cutoff": (
                None
                if first_index is None or first_index + 1 >= len(rows)
                else rows[first_index + 1]["threshold"]
            ),
            "water8_candidates": water8_candidates,
            "reasons": selection_reasons,
        },
    }


def _validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported FNO G2 plan schema")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("FNO G2 plan has no jobs")
    ids = [item.get("id") for item in jobs if isinstance(item, dict)]
    if len(ids) != len(jobs) or any(not isinstance(item, str) for item in ids):
        raise ValueError("every FNO G2 job needs a string id")
    if len(set(ids)) != len(ids):
        raise ValueError("FNO G2 job ids are not unique")
    known: set[str] = set()
    for job in jobs:
        dependencies = job.get("depends_on")
        if not isinstance(dependencies, list) or any(
            item not in known for item in dependencies
        ):
            raise ValueError(
                f"job {job.get('id')!r} has a missing or forward dependency"
            )
        launcher = job.get("launcher")
        options = job.get("slurm_options")
        exports = job.get("exports")
        if not isinstance(launcher, str) or not launcher:
            raise ValueError(f"job {job.get('id')!r} has no launcher")
        if not isinstance(options, list) or not all(
            isinstance(item, str) and item for item in options
        ):
            raise ValueError(f"job {job.get('id')!r} has invalid Slurm options")
        if not isinstance(exports, dict) or not exports:
            raise ValueError(f"job {job.get('id')!r} has no exports")
        for key, value in exports.items():
            if (
                not isinstance(key, str) or not key
                or not isinstance(value, str)
                or any(character in value for character in ",\n\r")
            ):
                raise ValueError(
                    f"job {job.get('id')!r} has an unsafe sbatch export"
                )
        known.add(str(job["id"]))
    return plan


def _sbatch_argv(
    job: dict[str, Any], submitted: dict[str, str]
) -> list[str]:
    dependencies = [submitted[item] for item in job["depends_on"]]
    exports = ["ALL"] + [
        f"{key}={value}" for key, value in sorted(job["exports"].items())
    ]
    argv = ["sbatch", "--parsable", f"--job-name={job['id']}"]
    argv.extend(job["slurm_options"])
    if dependencies:
        argv.append("--dependency=afterok:" + ":".join(dependencies))
    argv.extend(["--export=" + ",".join(exports), job["launcher"]])
    return argv


def submit_plan(
    plan: dict[str, Any], *, execute: bool
) -> dict[str, Any]:
    """Print or submit a validated plan in dependency order."""

    plan = _validate_plan(plan)
    submitted: dict[str, str] = {}
    commands: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        argv = _sbatch_argv(job, submitted)
        if execute:
            output = subprocess.check_output(argv, text=True).strip()
            job_id = output.split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(
                    f"sbatch returned an invalid job id for {job['id']}: {output!r}"
                )
        else:
            # Stable placeholders let later dry-run commands show dependency
            # wiring without being mistaken for real numeric Slurm job ids.
            job_id = f"<{job['id']}>"
        submitted[job["id"]] = job_id
        commands.append({
            "job": job["id"],
            "job_id": job_id,
            "argv": argv,
            "shell_preview": shlex.join(argv),
        })
    return {
        "schema": SUBMISSION_SCHEMA,
        "executed": execute,
        "plan_run_id": plan.get("run_id"),
        "source_tree_sha256": plan.get("source_tree_sha256"),
        "jobs": commands,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write the fixed water4 Slurm DAG")
    plan.add_argument("--task-root", type=Path, required=True)
    plan.add_argument("--source-root", type=Path, required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--node", default="compute-1-6")
    plan.add_argument("--output", type=Path, default=None)

    analyze = subparsers.add_parser(
        "analyze", help="audit synchronized water4 timing and CP JSON"
    )
    analyze.add_argument("--records", type=Path, nargs="+", required=True)
    analyze.add_argument("--output", type=Path, required=True)

    submit = subparsers.add_parser(
        "submit", help="preview a plan, or call sbatch with --execute"
    )
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--execute", action="store_true")
    submit.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="required with --execute; written once after all submissions",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        payload = make_plan(
            task_root=args.task_root,
            source_root=args.source_root,
            run_id=args.run_id,
            node=args.node,
        )
        if args.output is None:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _json_write_once(args.output, payload)
            print(args.output)
        return 0
    if args.command == "analyze":
        payload = analyze_fno_g2(_load_records(args.records))
        _json_write_once(args.output, payload)
        print(json.dumps(payload["selection"], indent=2, sort_keys=True))
        return 0 if payload["selection"]["ready_for_water8"] else 1
    plan_payload = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.execute and args.receipt is None:
        raise ValueError("--execute requires --receipt")
    if not args.execute and args.receipt is not None:
        raise ValueError("--receipt is used only with --execute")
    payload = submit_plan(plan_payload, execute=args.execute)
    if args.receipt is not None:
        _json_write_once(args.receipt, payload)
        print(args.receipt)
    else:
        for item in payload["jobs"]:
            print(item["shell_preview"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
