#!/usr/bin/env python3
"""Plan, submit, and audit the staged FNO + Delta-MP2 G2 probe.

The probe is deliberately separate from the final 10x analyzer.  It runs one
fresh canonical/FNO A/B pair for every frozen-natural-orbital occupation
threshold, proves that each pair used the exact same serialized RHF orbitals,
and joins those timing records to counterpoise calculations made from one
shared cluster/ghost-monomer orbital bundle.  Only the first loose-to-tight
setting that passes every water4 gate and its immediately tighter neighbour
may be proposed for water8.

``plan`` writes a machine-readable Slurm DAG without submitting it.  ``submit``
prints that DAG by default and requires ``--execute`` to call ``sbatch``.
``analyze`` consumes synchronized JSON records plus an executed receipt and
refuses to advance an incomplete, tampered, or ambiguously paired grid.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import hashlib
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
FIXED_NODE = "compute-1-6"
FNO_THRESHOLDS = (1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7)
CASE_DIMENSIONS = {
    "water2-tz": {"nao": 116, "nocc": 10, "nvir": 106},
    "water4-tz": {"nao": 232, "nocc": 20, "nvir": 212},
    "water8-tz": {"nao": 464, "nocc": 40, "nvir": 424},
}
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_evidence(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return {
        "path": str(resolved),
        "file_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "payload_sha256": _payload_sha256(payload),
        "payload": payload,
    }


def _evidence_without_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in value if key != "payload"}


def _require_analysis_evidence(
    path: Path, *, stage: str, source_digest: str
) -> dict[str, Any]:
    evidence = _file_evidence(path, f"{stage} analysis")
    payload = evidence["payload"]
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{stage} analysis has an unsupported schema")
    if payload.get("stage") != stage:
        raise ValueError(f"{stage} analysis stage marker is missing or changed")
    if payload.get("source_tree_sha256") != source_digest:
        raise ValueError(f"{stage} analysis source digest differs from candidate")
    if stage == "water2" and payload.get("selection", {}).get("ready_for_next_stage") is not True:
        raise ValueError("water2 analysis does not pass the complete prerequisite gate")
    if payload.get("selection", {}).get("ready_for_water8") is not True and stage == "water4":
        raise ValueError("water4 analysis does not authorize water8")
    return evidence


def _require_receipt_evidence(
    path: Path, *, source_digest: str, expected_stage: str | None = None
) -> dict[str, Any]:
    evidence = _file_evidence(path, "prior submission receipt")
    payload = evidence["payload"]
    if payload.get("executed") is not True:
        raise ValueError("prior submission receipt was not executed")
    if payload.get("source_tree_sha256") != source_digest:
        raise ValueError("prior submission receipt source digest differs from candidate")
    if expected_stage is not None and payload.get("stage") != expected_stage:
        raise ValueError(f"prior submission receipt is not a {expected_stage} receipt")
    plan_path = payload.get("plan_path")
    if not isinstance(plan_path, str):
        raise ValueError("prior submission receipt is missing its immutable plan path")
    plan = _read_plan_object(Path(plan_path))
    _validate_plan(plan)
    expected_hash = _payload_sha256(plan)
    if payload.get("plan_payload_sha256") != expected_hash:
        raise ValueError("prior submission receipt plan content hash differs")
    if plan.get("stage") != expected_stage:
        raise ValueError("prior submission receipt plan stage differs")
    jobs = payload.get("jobs")
    _validate_receipt_jobs(payload, plan)
    return evidence


def _read_plan_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read immutable plan {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("immutable plan must contain one JSON object")
    return value


def _regenerate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a stage plan from its immutable inputs before any sbatch call."""
    stage = plan.get("stage")
    task_root, source_root = Path(plan["task_root"]), Path(plan["source_root"])
    common = {"run_id": plan["run_id"], "node": plan["node"], "task_root": task_root,
              "source_root": source_root}
    if stage == "water2":
        return make_water2_plan(**common)
    prerequisite = plan.get("prerequisite")
    if not isinstance(prerequisite, dict):
        raise ValueError("stage plan prerequisite is missing")
    analysis = prerequisite.get("analysis", {}).get("path")
    receipt = prerequisite.get("receipt", {}).get("path")
    if not isinstance(analysis, str) or not isinstance(receipt, str):
        raise ValueError("stage plan prerequisite paths are missing")
    if stage == "water4":
        return make_water4_plan(**common, water2_analysis=Path(analysis), water2_receipt=Path(receipt))
    if stage == "water8":
        return make_water8_plan(**common, water4_analysis=Path(analysis), water4_receipt=Path(receipt))
    raise ValueError(f"unsupported stage {stage!r}")


def _validate_receipt_jobs(receipt: dict[str, Any], plan: dict[str, Any]) -> dict[str, str]:
    if receipt.get("schema") != SUBMISSION_SCHEMA or receipt.get("executed") is not True:
        raise ValueError("receipt schema or executed marker is invalid")
    if receipt.get("stage") != plan.get("stage") or receipt.get("case_id") != plan.get("case_id"):
        raise ValueError("receipt stage/case does not match its plan")
    if receipt.get("source_tree_sha256") != plan.get("source_tree_sha256"):
        raise ValueError("receipt source digest differs from its plan")
    if receipt.get("plan_payload_sha256") != _payload_sha256(plan):
        raise ValueError("receipt plan content hash differs from the supplied plan")
    jobs = receipt.get("jobs")
    expected = [job.get("id") for job in plan.get("jobs", [])]
    if not isinstance(jobs, list) or len(jobs) != len(expected):
        raise ValueError("receipt must contain exactly one result for every plan job")
    logical: dict[str, str] = {}
    for item in jobs:
        if not isinstance(item, dict) or item.get("job") not in expected:
            raise ValueError("receipt contains a missing or extra logical job")
        name, number = item.get("job"), item.get("job_id")
        if name in logical or not isinstance(number, str) or not re.fullmatch(r"[1-9][0-9]*", number):
            raise ValueError("receipt contains duplicate or non-numeric Slurm job IDs")
        logical[name] = number
    if set(logical) != set(expected) or len(set(logical.values())) != len(logical):
        raise ValueError("receipt does not prove the exact logical job set")
    return logical


def _record_slurm_name_id(record: dict[str, Any]) -> tuple[str | None, str | None]:
    slurm = record.get("slurm") if isinstance(record.get("slurm"), dict) else {}
    name = slurm.get("SLURM_JOB_NAME") or record.get("slurm_job_name")
    job_id = slurm.get("SLURM_JOB_ID") or record.get("slurm_job_id")
    return (None if name is None else str(name), None if job_id is None else str(job_id))


def _bind_records_to_receipt(
    records: list[dict[str, Any]], receipt: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    """Require every synchronized result to identify one exact submitted job."""
    errors: list[str] = []
    logical = _validate_receipt_jobs(receipt, plan)
    reverse = {number: name for name, number in logical.items()}
    jobs_by_id = {job["id"]: job for job in plan.get("jobs", [])}
    seen: set[str] = set()
    for record in records:
        name, job_id = _record_slurm_name_id(record)
        is_cp = record.get("schema") == "gpu4pyscf.water8.counterpoise.v1"
        orchestration = record.get("orchestration") if isinstance(record.get("orchestration"), dict) else {}
        logical_name = orchestration.get("logical_job_id")
        if is_cp and not isinstance(logical_name, str):
            errors.append("counterpoise result is missing immutable logical job identity")
            continue
        if name is not None or job_id is not None:
            if name is None or job_id is None:
                errors.append("analysis result is missing Slurm job name or ID")
            elif name not in logical or logical[name] != job_id:
                errors.append("analysis result Slurm ID is outside the exact submission receipt")
            elif is_cp and logical_name != name:
                errors.append("counterpoise logical job identity differs from Slurm job name")
            else:
                job = jobs_by_id[name]
                result_path = _record_path(record)
                expected_result = job.get("expected_outputs", [None])[0]
                if result_path != expected_result:
                    errors.append("result path differs from the immutable plan")
                if is_cp:
                    expected_exports = job.get("exports", {})
                    if orchestration.get("run_id") != expected_exports.get("CP_RUN_ID"):
                        errors.append("counterpoise orchestration run ID differs from the immutable plan")
                    if orchestration.get("stage") != plan.get("stage"):
                        errors.append("counterpoise orchestration stage differs from the immutable plan")
                    bundle = record.get("orbital_bundle")
                    expected_bundle = expected_exports.get("CP_ORBITAL_BUNDLE_OUT") or expected_exports.get("CP_ORBITAL_BUNDLE_IN")
                    if not isinstance(bundle, dict) or bundle.get("path") != expected_bundle or bundle.get("complete") is not True:
                        errors.append("counterpoise orbital bundle is missing or differs from the immutable plan")
                    else:
                        bundle_path = Path(str(expected_bundle))
                        if not bundle_path.is_file():
                            errors.append("counterpoise orbital bundle file is missing")
                        else:
                            declared = bundle.get("sha256")
                            actual = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                            if declared != actual:
                                errors.append("counterpoise orbital bundle hash does not match the artifact")
                seen.add(name)
            continue
        errors.append("analysis result is missing receipt-bound Slurm job name or ID")
    missing = set(logical) - seen
    if missing:
        errors.append("submission receipt contains jobs without synchronized result records")
    return list(dict.fromkeys(errors))


def _dimension_partition(
    audit: dict[str, Any], expected: dict[str, int]
) -> list[str]:
    """Return fail-closed errors for the global MO virtual partition."""
    errors: list[str] = []
    nocc, nvir = expected["nocc"], expected["nvir"]
    if audit.get("nocc") != nocc or audit.get("nvir") != nvir:
        errors.append("FNO dimensions do not match the frozen case contract")
    active = audit.get("active_virtual")
    frozen = audit.get("frozen_virtual")
    if not isinstance(active, list) or not isinstance(frozen, list):
        return errors + ["FNO active/frozen virtual index lists are missing"]
    nvir_active = audit.get("nvir_active")
    if not isinstance(nvir_active, int) or isinstance(nvir_active, bool):
        errors.append("FNO active virtual count is not an integer")
        nvir_active = -1
    if len(active) != nvir_active or len(frozen) != nvir - nvir_active:
        errors.append("FNO active/frozen virtual list lengths are inconsistent")
    all_indexes = active + frozen
    if any(isinstance(item, bool) or not isinstance(item, int) for item in all_indexes):
        errors.append("FNO virtual indexes must be integers")
    else:
        expected_set = set(range(nocc, nocc + nvir))
        if len(all_indexes) != len(set(all_indexes)):
            errors.append("FNO active/frozen virtual indexes contain duplicates")
        if set(all_indexes) != expected_set:
            errors.append("FNO active/frozen virtual indexes are not an exact full partition")
        if set(active).intersection(frozen):
            errors.append("FNO active/frozen virtual indexes overlap")
    return errors


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
    prior_analysis: Path | None = None,
    prior_receipt: Path | None = None,
) -> dict[str, Any]:
    """Return the fixed seven-threshold water4 job DAG."""

    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run id must start with an alphanumeric character and contain "
            "only letters, digits, dot, underscore, or hyphen"
        )
    if not node or any(character.isspace() for character in node):
        raise ValueError("node must be one non-empty Slurm host name")
    if node != FIXED_NODE:
        raise ValueError(f"formal FNO G2 plans require the fixed node {FIXED_NODE}")
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
            "FNO_LOGICAL_JOB_ID": cp_oracle_id,
            "FNO_STAGE": "water4",
        },
        "expected_outputs": [
            str(cp_root / f"{CASE_ID}__canonical__{run_id}-canonical.json"),
            str(cp_bundle),
        ],
            "role": "canonical Boys--Bernardi oracle and shared bundle producer",
            "orchestration": {"logical_job_id": cp_oracle_id},
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
                "FNO_LOGICAL_JOB_ID": job_id,
                "FNO_STAGE": "water4",
            },
            "expected_outputs": [
                str(cp_root / f"{CASE_ID}__fno__{cp_run_id}.json"),
            ],
            "role": "FNO Boys--Bernardi candidate from shared CP bundle",
            "fno_thresh": cutoff,
            "orchestration": {"logical_job_id": job_id},
        })
        previous_cp_job = job_id

    plan = {
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
    if prior_analysis is not None or prior_receipt is not None:
        if prior_analysis is None or prior_receipt is None:
            raise ValueError("water4 plan requires both water2 analysis and receipt")
        analysis_evidence = _require_analysis_evidence(
            prior_analysis, stage="water2", source_digest=digest
        )
        receipt_evidence = _require_receipt_evidence(
            prior_receipt, source_digest=digest, expected_stage="water2"
        )
        if analysis_evidence["payload"].get("case_id") != "water2-tz":
            raise ValueError("water2 prerequisite analysis case differs")
        plan["stage"] = "water4"
        plan["prerequisite"] = {
            "stage": "water2",
            "analysis": _evidence_without_payload(analysis_evidence),
            "receipt": _evidence_without_payload(receipt_evidence),
        }
    else:
        # Kept for the original water4 preview API.  New orchestration callers
        # must use make_water4_plan, which always requires the prerequisite.
        plan["stage"] = "water4-legacy-preview"
    return plan


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
            value.get("case_id") in CASE_DIMENSIONS
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
    record: dict[str, Any], expected_threshold: float,
    expected_dimensions: dict[str, int] | None = None,
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
    if expected_dimensions is not None:
        reasons.extend(_dimension_partition(audit, expected_dimensions))
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
    expected_dimensions: dict[str, int] | None = None,
    strict_source: str | None = None,
    case_id: str = CASE_ID,
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
            candidate, threshold, expected_dimensions
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
        if strict_source is not None and source_digest != strict_source:
            reasons.append("candidate source digest differs from the immutable plan")
        baseline_source = _analysis._source_digest(baseline) if baseline is not None else None
        if strict_source is not None and baseline_source != strict_source:
            reasons.append("canonical source digest differs from the immutable plan")
    if baseline is not None and candidate is not None:
        if expected_dimensions is not None:
            for label, record in (("canonical", baseline), ("FNO", candidate)):
                identity = _at(record, "canonical_orbitals.identity")
                if not isinstance(identity, dict) or any(
                    identity.get(name) != expected_dimensions[name]
                    for name in ("nocc", "nvir")
                ):
                    reasons.append(f"{label} orbital identity dimensions violate the case contract")
                orbital = record.get("canonical_orbitals")
                artifact_path = orbital.get("artifact_path") if isinstance(orbital, dict) else None
                artifact_sha = orbital.get("artifact_sha256") if isinstance(orbital, dict) else None
                if not isinstance(artifact_path, str) or not Path(artifact_path).is_file():
                    reasons.append(f"{label} canonical orbital artifact is missing")
                elif not isinstance(artifact_sha, str) or hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest() != artifact_sha:
                    reasons.append(f"{label} canonical orbital artifact hash does not match")
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
            case_id,
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


def analyze_fno_g2(
    records: list[dict[str, Any]], *, case_id: str = CASE_ID,
    strict_dimensions: bool = False,
) -> dict[str, Any]:
    """Audit a synchronized seven-threshold grid.

    The historical no-argument form remains compatible with the original
    water4 unit fixtures.  Staged orchestration uses the strict case-specific
    wrappers below, which additionally enforce immutable dimensions and source
    provenance.
    """
    if case_id not in CASE_DIMENSIONS:
        raise ValueError(f"unsupported FNO case {case_id!r}")
    expected_dimensions = CASE_DIMENSIONS[case_id] if strict_dimensions else None

    timed = [
        item for item in records
        if item.get("case_id") == case_id
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
            expected_dimensions=expected_dimensions,
            case_id=case_id,
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
        "stage": "water4" if case_id == "water4-tz" else case_id.removesuffix("-tz"),
        "case_id": case_id,
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


def _strict_source_digest(records: list[dict[str, Any]]) -> str | None:
    values = {_analysis._source_digest(item) for item in records}
    values.discard(None)
    return next(iter(values)) if len(values) == 1 else None


def _analysis_receipt_gate(
    records: list[dict[str, Any]], receipt_path: Path | None, *, stage: str
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any] | None]:
    if receipt_path is None:
        return None, [f"{stage} analysis requires the executed submission receipt"], None
    try:
        evidence = _file_evidence(receipt_path, f"{stage} submission receipt")
        receipt = evidence["payload"]
        plan_path = receipt.get("plan_path")
        if not isinstance(plan_path, str):
            raise ValueError("submission receipt does not bind an immutable plan path")
        plan = _read_plan_object(Path(plan_path))
        _validate_plan(plan)
        if plan.get("stage") != stage or receipt.get("stage") != stage:
            raise ValueError("submission receipt stage differs from analysis stage")
        if receipt.get("source_tree_sha256") != plan.get("source_tree_sha256"):
            raise ValueError("submission receipt source digest differs from its plan")
        errors = _bind_records_to_receipt(records, receipt, plan)
        logical = _validate_receipt_jobs(receipt, plan)
        return receipt, errors, {
            "receipt_path": str(evidence["path"]),
            "receipt_file_sha256": evidence["file_sha256"],
            "receipt_payload_sha256": evidence["payload_sha256"],
            "plan_path": str(Path(plan_path).expanduser().resolve()),
            "plan_file_sha256": hashlib.sha256(Path(plan_path).expanduser().resolve().read_bytes()).hexdigest(),
            "plan_payload_sha256": _payload_sha256(plan),
            "run_id": plan.get("run_id"),
            "logical_job_ids": logical,
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, [f"invalid {stage} submission receipt: {exc}"], None


def _analysis_receipt_identity_matches(
    analysis: dict[str, Any], receipt_evidence: dict[str, Any]
) -> bool:
    binding = analysis.get("receipt_binding")
    if not isinstance(binding, dict):
        return False
    expected = {
        "receipt_path": receipt_evidence.get("path"),
        "receipt_file_sha256": receipt_evidence.get("file_sha256"),
        "receipt_payload_sha256": receipt_evidence.get("payload_sha256"),
        "plan_path": receipt_evidence.get("payload", {}).get("plan_path"),
        "plan_payload_sha256": receipt_evidence.get("payload", {}).get("plan_payload_sha256"),
    }
    if expected["plan_path"] is not None:
        expected["plan_path"] = str(Path(expected["plan_path"]).expanduser().resolve())
    try:
        plan = _read_plan_object(Path(expected["plan_path"]))
        expected["run_id"] = plan.get("run_id")
        expected["logical_job_ids"] = _validate_receipt_jobs(receipt_evidence["payload"], plan)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return all(binding.get(key) == value for key, value in expected.items())


def analyze_water2(
    records: list[dict[str, Any]], receipt_path: Path | None = None
) -> dict[str, Any]:
    """Fail-closed analysis for the complete water2 FNO development grid."""
    result = analyze_fno_g2(records, case_id="water2-tz", strict_dimensions=True)
    source = _strict_source_digest(records)
    result["source_tree_sha256"] = source
    result["stage"] = "water2"
    reasons = list(result["selection"].get("reasons", []))
    if source is None:
        reasons.append("all water2 records must carry one identical source digest")
    if not result["grid_complete"]:
        reasons.append("water2 requires the complete seven-threshold grid")
    if any(not row["qualifies_for_water8"] for row in result["rows"]):
        reasons.append("water2 contains a threshold failing a numerical or resource gate")
    _receipt, receipt_reasons, binding = _analysis_receipt_gate(records, receipt_path, stage="water2")
    if _receipt is not None and source is not None and _receipt.get("source_tree_sha256") != source:
        receipt_reasons.append("water2 receipt source digest differs from result records")
    reasons.extend(receipt_reasons)
    result["receipt_binding"] = binding
    result["selection"]["ready_for_next_stage"] = not reasons
    result["selection"]["reasons"] = list(dict.fromkeys(reasons))
    return result


def analyze_water4(
    records: list[dict[str, Any]], receipt_path: Path | None = None
) -> dict[str, Any]:
    """Strict water4 analysis whose result authorizes exactly two water8 probes."""
    result = analyze_fno_g2(records, case_id="water4-tz", strict_dimensions=True)
    source = _strict_source_digest(records)
    result["source_tree_sha256"] = source
    result["stage"] = "water4"
    reasons = list(result["selection"].get("reasons", []))
    if source is None:
        reasons.append("all water4 records must carry one identical source digest")
    if not result["grid_complete"]:
        reasons.append("water4 requires the complete seven-threshold grid")
    _receipt, receipt_reasons, binding = _analysis_receipt_gate(records, receipt_path, stage="water4")
    if _receipt is not None and source is not None and _receipt.get("source_tree_sha256") != source:
        receipt_reasons.append("water4 receipt source digest differs from result records")
    reasons.extend(receipt_reasons)
    result["receipt_binding"] = binding
    result["selection"]["ready_for_water8"] = (
        not reasons and result["selection"].get("ready_for_water8") is True
    )
    result["selection"]["reasons"] = list(dict.fromkeys(reasons))
    return result


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
    stage = plan.get("stage")
    if stage in {"water4", "water8"}:
        prerequisite = plan.get("prerequisite")
        if not isinstance(prerequisite, dict):
            raise ValueError(f"{stage} plan is missing prerequisite evidence")
        for label in ("analysis", "receipt"):
            evidence = prerequisite.get(label)
            if not isinstance(evidence, dict):
                raise ValueError(f"{stage} plan is missing prerequisite {label}")
            path = evidence.get("path")
            if not isinstance(path, str):
                raise ValueError(f"{stage} prerequisite {label} path is invalid")
            fresh = _file_evidence(Path(path), f"{stage} prerequisite {label}")
            if fresh["file_sha256"] != evidence.get("file_sha256") or fresh["payload_sha256"] != evidence.get("payload_sha256"):
                raise ValueError(f"{stage} prerequisite {label} was modified after planning")
            if fresh["payload"].get("source_tree_sha256") != plan.get("source_tree_sha256"):
                raise ValueError(f"{stage} prerequisite {label} source differs from plan")
        if stage == "water8":
            thresholds = plan.get("candidate_thresholds")
            if not isinstance(thresholds, list) or len(thresholds) != 2:
                raise ValueError("water8 plan must contain exactly two candidate thresholds")
            if len(jobs) != 8:
                raise ValueError("water8 plan must contain four timing and four CP jobs")
            selected = prerequisite.get("selected_candidates")
            if not isinstance(selected, list) or [item.get("fno_thresh") for item in selected if isinstance(item, dict)] != thresholds:
                raise ValueError("water8 prerequisite candidates do not match the plan")
    if stage in {"water2", "water4"}:
        if len(jobs) != 22:
            raise ValueError(f"{stage} plan must contain exactly 22 jobs")
        expected_phases = {f"{stage}-timing", f"{stage}-counterpoise"}
        if {job.get("phase") for job in jobs} != expected_phases:
            raise ValueError(f"{stage} plan has invalid phase labels")
    if stage == "water8":
        if [job.get("phase") for job in jobs].count("water8-timing") != 4 or [job.get("phase") for job in jobs].count("water8-counterpoise") != 4:
            raise ValueError("water8 plan must contain four timing and four counterpoise jobs")
    if stage in {"water2", "water4", "water8"}:
        regenerated = _regenerate_plan(plan)
        if _canonical_json(regenerated) != _canonical_json(plan):
            raise ValueError("stage plan differs from canonical regeneration; refusing submission")
    return plan


def make_water2_plan(
    *, task_root: Path, source_root: Path, run_id: str, node: str,
) -> dict[str, Any]:
    """Create the immutable seven-threshold water2 A/B plus CP DAG."""
    plan = make_plan(task_root=task_root, source_root=source_root, run_id=run_id, node=node)
    task = task_root.expanduser().resolve()
    # The water4-compatible builder above is intentionally reused so all
    # Slurm/export safety checks stay identical.  Rewrite only case-scoped
    # fields; preserve the 14 timing + 8 CP topology exactly.
    serialized = json.loads(json.dumps(plan))
    serialized["stage"] = "water2"
    serialized["case_id"] = "water2-tz"
    serialized["result_root"] = str(task / "results" / "fno-g2" / run_id / "water2")
    serialized["timing_result_root"] = str(Path(serialized["result_root"]) / "timing")
    serialized["counterpoise_result_root"] = str(Path(serialized["result_root"]) / "counterpoise")
    for job in serialized["jobs"]:
        if job["phase"] == "water4-timing":
            job["phase"] = "water2-timing"
        elif job["phase"] == "water4-counterpoise":
            job["phase"] = "water2-counterpoise"
        exports = job["exports"]
        exports["CASE"] = "water2-tz"
        if "FNO_STAGE" in exports:
            exports["FNO_STAGE"] = "water2"
        for key, value in list(exports.items()):
            if isinstance(value, str):
                exports[key] = value.replace("water4-tz", "water2-tz").replace(
                    str(task / "results" / "fno-g2" / run_id), serialized["result_root"]
                )
        job["expected_outputs"] = [
            str(item).replace("water4-tz", "water2-tz").replace(
                str(task / "results" / "fno-g2" / run_id), serialized["result_root"]
            ) for item in job["expected_outputs"]
        ]
    serialized["dimensions"] = CASE_DIMENSIONS["water2-tz"]
    serialized["threshold_order"] = list(FNO_THRESHOLDS)
    serialized["policies"]["case_contract"] = "WATER27 H2O2, spherical cc-pVTZ, exact frozen dimensions"
    return serialized


def make_water4_plan(
    *, task_root: Path, source_root: Path, run_id: str, node: str,
    water2_analysis: Path, water2_receipt: Path,
) -> dict[str, Any]:
    """Create water4 only after binding passing water2 evidence."""
    task = task_root.expanduser().resolve()
    digest = _source_digest(source_root, task)
    analysis = _require_analysis_evidence(water2_analysis, stage="water2", source_digest=digest)
    receipt = _require_receipt_evidence(water2_receipt, source_digest=digest, expected_stage="water2")
    if analysis["payload"].get("case_id") != "water2-tz":
        raise ValueError("water2 prerequisite analysis case differs")
    if not _analysis_receipt_identity_matches(analysis["payload"], receipt):
        raise ValueError("water2 analysis receipt identity does not match the supplied receipt")
    plan = make_plan(
        task_root=task, source_root=source_root, run_id=run_id, node=node,
        prior_analysis=water2_analysis, prior_receipt=water2_receipt,
    )
    plan["stage"] = "water4"
    plan["dimensions"] = CASE_DIMENSIONS["water4-tz"]
    plan["prerequisite"]["analysis"]["payload_case_id"] = analysis["payload"].get("case_id")
    plan["prerequisite"]["receipt"]["payload_case_id"] = receipt["payload"].get("case_id")
    return plan


def _analysis_candidate_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(analysis, dict) or analysis.get("schema") != SCHEMA:
        raise ValueError("water4 analysis has an unsupported schema")
    if analysis.get("stage") != "water4" or analysis.get("case_id") != "water4-tz":
        raise ValueError("water8 requires a complete water4 analysis")
    selection = analysis.get("selection")
    if not isinstance(selection, dict) or selection.get("ready_for_water8") is not True:
        raise ValueError("water4 analysis does not authorize water8")
    candidates = selection.get("water8_candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("water4 analysis must select exactly two water8 candidates")
    thresholds = [
        _number(item.get("fno_thresh")) for item in candidates
        if isinstance(item, dict)
    ]
    expected = list(FNO_THRESHOLDS)
    if len(thresholds) != 2 or any(value is None for value in thresholds):
        raise ValueError("water4 candidate thresholds are malformed")
    first = expected.index(thresholds[0]) if thresholds[0] in expected else -1
    second = expected.index(thresholds[1]) if thresholds[1] in expected else -1
    if first < 0 or second != first + 1:
        raise ValueError("water8 candidates must be the first pass and immediately tighter grid point")
    passing = [row for row in analysis.get("rows", []) if isinstance(row, dict) and row.get("qualifies_for_water8")]
    if not passing or passing[0].get("threshold") != thresholds[0]:
        raise ValueError("water4 candidate is not the first passing threshold")
    if len(analysis.get("rows", [])) != len(FNO_THRESHOLDS) or not analysis.get("grid_complete"):
        raise ValueError("water4 analysis is not a complete seven-threshold grid")
    return candidates


def make_water8_plan(
    *, task_root: Path, source_root: Path, run_id: str, node: str,
    water4_analysis: Path, water4_receipt: Path,
) -> dict[str, Any]:
    """Create the exactly-two-candidate water8 timing/CP DAG."""
    task = task_root.expanduser().resolve()
    if node != FIXED_NODE:
        raise ValueError(f"formal FNO G2 plans require the fixed node {FIXED_NODE}")
    source = source_root.expanduser().resolve()
    digest = _source_digest(source, task)
    analysis = _require_analysis_evidence(water4_analysis, stage="water4", source_digest=digest)
    receipt = _require_receipt_evidence(water4_receipt, source_digest=digest, expected_stage="water4")
    if not _analysis_receipt_identity_matches(analysis["payload"], receipt):
        raise ValueError("water4 analysis receipt identity does not match the supplied receipt")
    candidates = _analysis_candidate_rows(analysis["payload"])
    root = task / "results" / "fno-g2" / run_id / "water8"
    timing_root, cp_root = root / "timing", root / "counterpoise"
    benchmark = source / "benchmarks" / "cc" / "a100_water8" / "run_mtu_benchmark.sbatch"
    cp_launcher = source / "benchmarks" / "cc" / "a100_water8" / "run_mtu_counterpoise.sbatch"
    common = {
        "CCSD_TASK_ROOT": str(task), "CCSD_SOURCE_ROOT": str(source),
        "CCSD_EXPECTED_DEPLOYMENT_PROFILE": "candidate", "CCSD_REQUIRE_CANONICAL_PRISTINE": "0",
    }
    jobs: list[dict[str, Any]] = []
    previous: str | None = None
    for index, candidate in enumerate(candidates):
        cutoff = float(candidate["fno_thresh"])
        token = _cutoff_token(cutoff)
        artifact = root / "orbitals" / f"water8-tz-{token}.npz"
        canonical_id, fno_id = f"water8-{token}-canonical", f"water8-{token}-fno"
        jobs.append({
            "id": canonical_id, "phase": "water8-timing", "ordinal": len(jobs),
            "launcher": str(benchmark), "slurm_options": [f"--nodelist={node}"],
            "depends_on": [] if previous is None else [previous],
            "exports": {**common, "CASE": "water8-tz", "METHOD": "canonical",
                "REPEAT": f"{run_id}-{token}-canonical-a", "BENCHMARK_OUTPUT_DIR": str(timing_root),
                "BENCHMARK_ORBITAL_ARTIFACT_OUT": str(artifact)},
            "expected_outputs": [str(timing_root / f"water8-tz__canonical__r{run_id}-{token}-canonical-a.json"), str(artifact)],
            "role": "A: canonical water8 orbital producer and timing oracle",
        })
        jobs.append({
            "id": fno_id, "phase": "water8-timing", "ordinal": len(jobs),
            "launcher": str(benchmark), "slurm_options": [f"--nodelist={node}"],
            "depends_on": [canonical_id],
            "exports": {**common, "CASE": "water8-tz", "METHOD": "fno",
                "REPEAT": f"{run_id}-{token}-fno-b", "BENCHMARK_OUTPUT_DIR": str(timing_root),
                "BENCHMARK_ORBITAL_ARTIFACT_IN": str(artifact), "FNO_THRESH": repr(cutoff)},
            "expected_outputs": [str(timing_root / f"water8-tz__fno__r{run_id}-{token}-fno-b.json")],
            "role": "B: FNO water8 timing candidate", "fno_thresh": cutoff,
        })
        previous = fno_id
    cp_previous = previous
    for candidate in candidates:
        cutoff = float(candidate["fno_thresh"])
        token = _cutoff_token(cutoff)
        bundle = cp_root / f"water8-tz-{token}-canonical-orbitals.npz"
        canonical_id, fno_id = f"water8-{token}-cp-canonical", f"water8-{token}-cp-fno"
        jobs.append({
            "id": canonical_id, "phase": "water8-counterpoise", "ordinal": len(jobs),
            "launcher": str(cp_launcher), "slurm_options": [f"--nodelist={node}"],
            "depends_on": [cp_previous] if cp_previous else [],
            "exports": {**common, "CASE": "water8-tz", "METHOD": "canonical",
                "CP_RUN_ID": f"{run_id}-{token}-canonical", "CP_OUTPUT_ROOT": str(cp_root),
                "CP_ORBITAL_BUNDLE_OUT": str(bundle), "FNO_LOGICAL_JOB_ID": canonical_id,
                "FNO_STAGE": "water8"},
            "expected_outputs": [str(cp_root / f"water8-tz__canonical__{run_id}-{token}-canonical.json"), str(bundle)],
            "role": "A: canonical water8 counterpoise oracle", "fno_thresh": cutoff,
            "orchestration": {"logical_job_id": canonical_id},
        })
        jobs.append({
            "id": fno_id, "phase": "water8-counterpoise", "ordinal": len(jobs),
            "launcher": str(cp_launcher), "slurm_options": [f"--nodelist={node}"],
            "depends_on": [canonical_id],
            "exports": {**common, "CASE": "water8-tz", "METHOD": "fno",
                "CP_RUN_ID": f"{run_id}-{token}-fno", "CP_OUTPUT_ROOT": str(cp_root),
                "CP_ORBITAL_BUNDLE_IN": str(bundle), "FNO_THRESH": repr(cutoff),
                "FNO_LOGICAL_JOB_ID": fno_id, "FNO_STAGE": "water8"},
            "expected_outputs": [str(cp_root / f"water8-tz__fno__{run_id}-{token}-fno.json")],
            "role": "B: FNO water8 counterpoise candidate", "fno_thresh": cutoff,
            "orchestration": {"logical_job_id": fno_id},
        })
        cp_previous = fno_id
    return {
        "schema": PLAN_SCHEMA, "stage": "water8", "case_id": "water8-tz", "run_id": run_id,
        "task_root": str(task), "source_root": str(source), "source_tree_sha256": digest,
        "node": node, "dimensions": CASE_DIMENSIONS["water8-tz"],
        "threshold_order": [float(item["fno_thresh"]) for item in candidates],
        "candidate_thresholds": [float(item["fno_thresh"]) for item in candidates],
        "result_root": str(root), "timing_result_root": str(timing_root),
        "counterpoise_result_root": str(cp_root), "jobs": jobs,
        "prerequisite": {"stage": "water4", "analysis": _evidence_without_payload(analysis),
                          "receipt": _evidence_without_payload(receipt),
                          "selected_candidates": candidates},
        "policies": {"exactly_two_candidates": True, "first_pass_and_immediate_tighter": True,
                     "timing_and_counterpoise_are_fresh_process_ab_pairs": True},
    }


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
    plan: dict[str, Any], *, execute: bool, plan_path: Path | None = None
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
        "stage": plan.get("stage"),
        "case_id": plan.get("case_id"),
        "source_tree_sha256": plan.get("source_tree_sha256"),
        "plan_path": None if plan_path is None else str(plan_path.expanduser().resolve()),
        "plan_payload_sha256": _payload_sha256(plan),
        "prerequisite": plan.get("prerequisite"),
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
    plan.add_argument("--stage", choices=("water2", "water4", "water8"), default="water4")
    plan.add_argument("--prior-analysis", type=Path)
    plan.add_argument("--prior-receipt", type=Path)
    plan.add_argument("--output", type=Path, default=None)

    analyze = subparsers.add_parser(
        "analyze", help="audit synchronized water4 timing and CP JSON"
    )
    analyze.add_argument("--records", type=Path, nargs="+", required=True)
    analyze.add_argument("--stage", choices=("water2", "water4"), default="water4")
    analyze.add_argument("--receipt", type=Path, required=False,
                         help="executed receipt that must contain every result Slurm ID")
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
        if args.stage == "water2":
            payload = make_water2_plan(task_root=args.task_root, source_root=args.source_root,
                                       run_id=args.run_id, node=args.node)
        elif args.stage == "water4":
            if args.prior_analysis is None or args.prior_receipt is None:
                raise ValueError("water4 plan requires --prior-analysis and --prior-receipt")
            payload = make_water4_plan(task_root=args.task_root, source_root=args.source_root,
                                       run_id=args.run_id, node=args.node,
                                       water2_analysis=args.prior_analysis,
                                       water2_receipt=args.prior_receipt)
        else:
            if args.prior_analysis is None or args.prior_receipt is None:
                raise ValueError("water8 plan requires --prior-analysis and --prior-receipt")
            payload = make_water8_plan(task_root=args.task_root, source_root=args.source_root,
                                       run_id=args.run_id, node=args.node,
                                       water4_analysis=args.prior_analysis,
                                       water4_receipt=args.prior_receipt)
        if args.output is None:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _json_write_once(args.output, payload)
            print(args.output)
        return 0
    if args.command == "analyze":
        loaded = _load_records(args.records)
        payload = (analyze_water2(loaded, args.receipt)
                   if args.stage == "water2" else analyze_water4(loaded, args.receipt))
        _json_write_once(args.output, payload)
        print(json.dumps(payload["selection"], indent=2, sort_keys=True))
        ready = payload["selection"].get("ready_for_water8", payload["selection"].get("ready_for_next_stage", False))
        return 0 if ready else 1
    plan_payload = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.execute and args.receipt is None:
        raise ValueError("--execute requires --receipt")
    if not args.execute and args.receipt is not None:
        raise ValueError("--receipt is used only with --execute")
    if args.execute and args.receipt is not None and args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite submission receipt {args.receipt}")
    payload = submit_plan(plan_payload, execute=args.execute, plan_path=args.plan)
    if args.receipt is not None:
        _json_write_once(args.receipt, payload)
        print(args.receipt)
    else:
        for item in payload["jobs"]:
            print(item["shell_preview"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
