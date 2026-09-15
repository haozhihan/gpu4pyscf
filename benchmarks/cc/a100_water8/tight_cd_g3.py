#!/usr/bin/env python3
"""Plan, submit, and audit the receipt-bound tight-CD G3 experiment.

G3 compares two complete RR pipelines on byte-identical RHF orbitals:
``rr_canonical`` (A) and ``rr_cd`` (B).  Both use the same explicit RR
eigenvalue cutoff, but each pipeline constructs its own MP2 projector.  The
experiment therefore measures the incremental effect of CD on the final RR
pipeline; it is neither a shared-projector experiment nor a full-space
canonical/CD equivalence proof.

The water2 and water4 stages run the exact ``1e-4, 1e-6, 1e-8`` grid.  A
water4 analysis chooses the loosest passing tolerance, preferring any tighter
passing point whose whole-stage time is within three percent of that loose
anchor.  That analysis alone never authorizes water8.  A water8 plan also
requires a passing, receipt-bound G4 WATER4 gate produced at the selected
``eri_tol``.  Every candidate timing and counterpoise job consumes the exact
selected-GINT v2 release receipt pinned into the immutable source snapshot.

``plan`` writes an immutable JSON DAG.  ``submit`` only previews commands
unless ``--execute`` is supplied, and executed submissions write a receipt
once.  ``analyze`` rejects missing, duplicated, moved, source-mismatched, or
receipt-unbound records.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load_sibling(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_analysis = _load_sibling("tight_cd_g3_analysis", "analyze_results.py")
_g4 = _load_sibling("tight_cd_g3_rr_g4", "rr_g4_continue.py")


SCHEMA = "gpu4pyscf.water8.tight-cd-g3-analysis.v1"
PLAN_SCHEMA = "gpu4pyscf.water8.tight-cd-g3-plan.v1"
SUBMISSION_SCHEMA = "gpu4pyscf.water8.tight-cd-g3-submission.v1"
ERI_TOLERANCES = (1e-4, 1e-6, 1e-8)
RR_EIG_CUTOFF = 1e-11
ECORR_LIMIT_EH = 1e-4
CP_LIMIT_KCAL_MOL = 0.02
PROJECTED_RESIDUAL_LIMIT = 1e-6
PROJECTOR_ORTHOGONALITY_LIMIT = 1e-10
HBM_LIMIT_GIB = 72.0
RSS_LIMIT_GIB = 110.0
TIMING_TIE_FRACTION = 0.03
CASE_DIMENSIONS = {
    "water2-tz": {"nao": 116, "nocc": 10, "nvir": 106, "fragments": 2},
    "water4-tz": {"nao": 232, "nocc": 20, "nvir": 212, "fragments": 4},
    "water8-tz": {"nao": 464, "nocc": 40, "nvir": 424, "fragments": 8},
}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """Raised when G3 evidence fails closed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ContractError(f"{label} contains non-finite JSON number {value}")

    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain one JSON object")
    return value


def _json_write_once(path: Path, payload: dict[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True,
                   allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _file_evidence(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    payload = _json_object(resolved, label)
    return {
        "path": str(resolved),
        "file_sha256": _sha256(resolved),
        "payload_sha256": _payload_sha256(payload),
        "payload": payload,
    }


def _without_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "payload"}


def _number(value: Any, *, nonnegative: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        return None
    return result


def _same_float(left: Any, right: Any) -> bool:
    a, b = _number(left), _number(right)
    return bool(
        a is not None and b is not None
        and math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
    )


def _at(value: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _stage_case(stage: str) -> str:
    if stage not in {"water2", "water4", "water8"}:
        raise ContractError(f"unsupported G3 stage {stage!r}")
    return f"{stage}-tz"


def _source_digest(source_root: Path, task_root: Path) -> str:
    source = source_root.expanduser().resolve(strict=True)
    task = task_root.expanduser().resolve(strict=True)
    try:
        relative = source.relative_to(task / "snapshots")
    except ValueError as exc:
        raise ContractError(
            "source must equal <task>/snapshots/<sha256>/source"
        ) from exc
    if len(relative.parts) != 2 or relative.parts[1] != "source":
        raise ContractError("source must equal <task>/snapshots/<sha256>/source")
    digest = relative.parts[0]
    if SHA256_RE.fullmatch(digest) is None:
        raise ContractError("source snapshot directory is not a SHA-256")
    return digest


def _gint_contract(
    source_root: Path, receipt_path: Path, acceptance_path: Path,
    task_root: Path,
) -> dict[str, Any]:
    """Validate receipt/pin lineage and the completed release acceptance."""

    try:
        contract = _g4._accepted_receipt_contract(
            source_root, receipt_path, acceptance_path, task_root=task_root,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise ContractError(f"invalid selected-GINT release receipt: {exc}") from exc
    digest = _source_digest(source_root, task_root)
    pin = _json_object(Path(contract["release_pin_path"]), "GINT release pin")
    lineage = pin.get("lineage")
    if not isinstance(lineage, dict):
        raise ContractError("GINT release pin lacks qualification/release lineage")
    qualification_digest = lineage.get("qualification_source_tree_sha256")
    release_without_pin = lineage.get("release_source_without_pin_sha256")
    if (
        SHA256_RE.fullmatch(str(qualification_digest or "")) is None
        or qualification_digest != release_without_pin
        or qualification_digest == digest
    ):
        raise ContractError(
            "GINT release pin does not prove qualification-A to release-B lineage"
        )
    return {**contract, "release_source_tree_sha256": digest,
            "qualification_source_tree_sha256": qualification_digest}


def _threshold_token(value: float) -> str:
    if value not in ERI_TOLERANCES:
        raise ContractError(f"unsupported G3 eri_tol {value!r}")
    return {1e-4: "1e-4", 1e-6: "1e-6", 1e-8: "1e-8"}[value]


def _evidence_matches(expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = expected.get("path")
    if not isinstance(path, str):
        raise ContractError(f"{label} evidence path is missing")
    fresh = _file_evidence(Path(path), label)
    if (
        fresh["file_sha256"] != expected.get("file_sha256")
        or fresh["payload_sha256"] != expected.get("payload_sha256")
    ):
        raise ContractError(f"{label} changed after planning")
    return fresh


def _analysis_receipt_identity_matches(
    analysis: Mapping[str, Any], receipt_evidence: Mapping[str, Any],
) -> bool:
    binding = analysis.get("submission_binding")
    payload = receipt_evidence.get("payload")
    if not isinstance(binding, dict) or not isinstance(payload, dict):
        return False
    return all((
        binding.get("receipt_path") == receipt_evidence.get("path"),
        binding.get("receipt_file_sha256") == receipt_evidence.get("file_sha256"),
        binding.get("receipt_payload_sha256") == receipt_evidence.get("payload_sha256"),
        binding.get("plan_path") == payload.get("plan_path"),
        binding.get("plan_payload_sha256") == payload.get("plan_payload_sha256"),
        binding.get("release_gate_acceptance_sha256")
        == payload.get("gint_release_gate_acceptance", {}).get("sha256"),
    ))


def _require_prior(
    analysis_path: Path, receipt_path: Path, *, stage: str,
    source_digest: str, gint_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = _file_evidence(analysis_path, f"{stage} G3 analysis")
    payload = analysis["payload"]
    if (
        payload.get("schema") != SCHEMA
        or payload.get("stage") != stage
        or payload.get("case_id") != _stage_case(stage)
        or payload.get("source_tree_sha256") != source_digest
        or payload.get("gint_release_receipt", {}).get("sha256")
        != gint_contract.get("sha256")
        or payload.get("gint_release_gate_acceptance", {}).get("sha256")
        != gint_contract.get("release_gate_acceptance", {}).get("sha256")
    ):
        raise ContractError(f"{stage} analysis identity is invalid")
    ready_key = "ready_for_water4" if stage == "water2" else "ready_for_g4"
    if payload.get("selection", {}).get(ready_key) is not True:
        raise ContractError(f"{stage} analysis does not pass its G3 gate")
    receipt = _file_evidence(receipt_path, f"{stage} G3 submission receipt")
    _validate_submission_receipt(receipt["payload"])
    if (
        receipt["payload"].get("stage") != stage
        or receipt["payload"].get("source_tree_sha256") != source_digest
        or receipt["payload"].get("gint_release_receipt", {}).get("sha256")
        != gint_contract.get("sha256")
        or receipt["payload"].get("gint_release_gate_acceptance", {}).get("sha256")
        != gint_contract.get("release_gate_acceptance", {}).get("sha256")
        or not _analysis_receipt_identity_matches(payload, receipt)
    ):
        raise ContractError(f"{stage} analysis and receipt are not one chain")
    return analysis, receipt


def _require_g4_authorization(
    path: Path, *, task_root: Path, source_digest: str,
    selected_tol: float, gint_contract: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _file_evidence(path, "G4 WATER4 authorization")
    gate = evidence["payload"]
    if (
        gate.get("schema") != _g4.GATE_SCHEMA
        or gate.get("case_id") != "water4-tz"
        or gate.get("source_tree_sha256") != source_digest
        or gate.get("gint_release_gate_acceptance", {}).get("sha256")
        != gint_contract.get("release_gate_acceptance", {}).get("sha256")
        or gate.get("sequence_complete") is not True
        or gate.get("advance_to_water8") is not True
        or not gate.get("promotable_cutoffs")
    ):
        raise ContractError("G4 gate does not authorize WATER8")
    result_evidence = gate.get("result_evidence")
    if not isinstance(result_evidence, dict):
        raise ContractError("G4 gate lacks result evidence")
    result_path = result_evidence.get("path")
    if not isinstance(result_path, str):
        raise ContractError("G4 gate result path is missing")
    result = Path(result_path).expanduser().resolve(strict=True)
    try:
        result.relative_to(task_root.expanduser().resolve(strict=True) / "results")
    except ValueError as exc:
        raise ContractError("G4 result is outside task results") from exc
    if _sha256(result) != result_evidence.get("sha256"):
        raise ContractError("G4 result content differs from its gate")
    record = _json_object(result, "G4 WATER4 result")
    if (
        record.get("method") != "rr_cd"
        or not _same_float(_at(record, "plan.settings.eri_tol"), selected_tol)
        or _analysis._source_digest(record) != source_digest
    ):
        raise ContractError("G4 result did not use the selected G3 eri_tol")
    gate_receipt = _at(record, "provider_runtime_gate.receipt")
    if _at(record, "provider_runtime_gate.validated") is not True or not isinstance(gate_receipt, dict) or (
        gate_receipt.get("sha256") != gint_contract.get("sha256")
        or gate_receipt.get("payload_sha256") != gint_contract.get("payload_sha256")
    ):
        raise ContractError("G4 result did not consume this GINT receipt")
    return evidence


def _job_exports(
    *, task: Path, source: Path, case_id: str, method: str,
    run_id: str, output_root: Path, eri_tol: float,
    artifact_in: Path | None = None, artifact_out: Path | None = None,
    bundle_in: Path | None = None, bundle_out: Path | None = None,
    gint_receipt: Path | None = None, logical_job_id: str | None = None,
    stage: str,
) -> dict[str, str]:
    values = {
        "CCSD_TASK_ROOT": str(task),
        "CCSD_SOURCE_ROOT": str(source),
        "CCSD_EXPECTED_DEPLOYMENT_PROFILE": "candidate",
        "CCSD_REQUIRE_CANONICAL_PRISTINE": "0",
        "CASE": case_id,
        "METHOD": method,
        "ERI_TOL": repr(eri_tol),
        "RR_EIG_CUTOFF": repr(RR_EIG_CUTOFF),
        "RR_RING_KERNEL": "reference",
        "GINT_COLUMN_BACKEND": "selected",
        "GINT_COLUMN_KERNEL": "reference",
    }
    if bundle_in is not None or bundle_out is not None:
        values.update({"CP_RUN_ID": run_id, "CP_OUTPUT_ROOT": str(output_root)})
    else:
        values.update({"REPEAT": run_id, "BENCHMARK_OUTPUT_DIR": str(output_root)})
    if artifact_in is not None:
        values["BENCHMARK_ORBITAL_ARTIFACT_IN"] = str(artifact_in)
    if artifact_out is not None:
        values["BENCHMARK_ORBITAL_ARTIFACT_OUT"] = str(artifact_out)
    if bundle_in is not None:
        values["CP_ORBITAL_BUNDLE_IN"] = str(bundle_in)
    if bundle_out is not None:
        values["CP_ORBITAL_BUNDLE_OUT"] = str(bundle_out)
    if gint_receipt is not None:
        values["GINT_RUNTIME_GATE_RECEIPT"] = str(gint_receipt)
        if bundle_in is None and bundle_out is None:
            values["RUN_FULL_SPACE_DIAGNOSTIC"] = "1"
    if logical_job_id is not None:
        values["FNO_LOGICAL_JOB_ID"] = logical_job_id
        values["FNO_STAGE"] = f"tight-cd-g3-{stage}"
    return values


def _stage_jobs(
    *, task: Path, source: Path, run_id: str, stage: str,
    node: str, tolerances: Sequence[float], gint_receipt: Path,
) -> tuple[list[dict[str, Any]], Path]:
    case_id = _stage_case(stage)
    root = task / "results" / "tight-cd-g3" / run_id / stage
    timing, orbitals, cp = root / "timing", root / "orbitals", root / "counterpoise"
    benchmark = source / "benchmarks/cc/a100_water8/run_mtu_benchmark.sbatch"
    counterpoise = source / "benchmarks/cc/a100_water8/run_mtu_counterpoise.sbatch"
    jobs: list[dict[str, Any]] = []
    previous: str | None = None
    for tolerance in tolerances:
        token = _threshold_token(float(tolerance))
        artifact = orbitals / f"{case_id}-{token}.npz"
        a_id = f"{stage}-{token}-rr-canonical"
        b_id = f"{stage}-{token}-rr-cd"
        a_run = f"{run_id}-{token}-rr-canonical-a"
        b_run = f"{run_id}-{token}-rr-cd-b"
        jobs.append({
            "id": a_id, "phase": f"{stage}-timing", "ordinal": len(jobs),
            "role": "A: independently projected rr_canonical pipeline",
            "launcher": str(benchmark), "slurm_options": [f"--nodelist={node}"],
            "depends_on": [] if previous is None else [previous],
            "eri_tol": float(tolerance), "method": "rr_canonical",
            "exports": _job_exports(
                task=task, source=source, case_id=case_id,
                method="rr_canonical", run_id=a_run, output_root=timing,
                eri_tol=float(tolerance), artifact_out=artifact, stage=stage,
            ),
            "expected_outputs": [
                str(timing / f"{case_id}__rr_canonical__r{a_run}.json"),
                str(artifact),
            ],
        })
        jobs.append({
            "id": b_id, "phase": f"{stage}-timing", "ordinal": len(jobs),
            "role": "B: independently projected receipt-bound rr_cd pipeline",
            "launcher": str(benchmark), "slurm_options": [f"--nodelist={node}"],
            "depends_on": [a_id], "eri_tol": float(tolerance), "method": "rr_cd",
            "exports": _job_exports(
                task=task, source=source, case_id=case_id, method="rr_cd",
                run_id=b_run, output_root=timing, eri_tol=float(tolerance),
                artifact_in=artifact, gint_receipt=gint_receipt, stage=stage,
            ),
            "expected_outputs": [
                str(timing / f"{case_id}__rr_cd__r{b_run}.json"),
            ],
        })
        previous = b_id
    for tolerance in tolerances:
        token = _threshold_token(float(tolerance))
        bundle = cp / f"{case_id}-{token}-orbitals.npz"
        a_id = f"{stage}-{token}-cp-rr-canonical"
        b_id = f"{stage}-{token}-cp-rr-cd"
        a_run = f"{run_id}-{token}-rr-canonical"
        b_run = f"{run_id}-{token}-rr-cd"
        jobs.append({
            "id": a_id, "phase": f"{stage}-counterpoise", "ordinal": len(jobs),
            "role": "A: complete cluster plus ghost-monomer RR-canonical CP",
            "launcher": str(counterpoise), "slurm_options": [f"--nodelist={node}"],
            "depends_on": [] if previous is None else [previous],
            "eri_tol": float(tolerance), "method": "rr_canonical",
            "exports": _job_exports(
                task=task, source=source, case_id=case_id,
                method="rr_canonical", run_id=a_run, output_root=cp,
                eri_tol=float(tolerance), bundle_out=bundle,
                logical_job_id=a_id, stage=stage,
            ),
            "expected_outputs": [
                str(cp / f"{case_id}__rr_canonical__{a_run}.json"), str(bundle),
            ],
        })
        jobs.append({
            "id": b_id, "phase": f"{stage}-counterpoise", "ordinal": len(jobs),
            "role": "B: complete receipt-bound RR-CD CP on the same bundle",
            "launcher": str(counterpoise), "slurm_options": [f"--nodelist={node}"],
            "depends_on": [a_id], "eri_tol": float(tolerance), "method": "rr_cd",
            "exports": _job_exports(
                task=task, source=source, case_id=case_id, method="rr_cd",
                run_id=b_run, output_root=cp, eri_tol=float(tolerance),
                bundle_in=bundle, gint_receipt=gint_receipt,
                logical_job_id=b_id, stage=stage,
            ),
            "expected_outputs": [
                str(cp / f"{case_id}__rr_cd__{b_run}.json"),
            ],
        })
        previous = b_id
    return jobs, root


def _base_plan(
    *, task_root: Path, source_root: Path, gint_runtime_gate_receipt: Path,
    gint_release_gate_acceptance: Path,
    run_id: str, stage: str, tolerances: Sequence[float],
    prerequisite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ContractError("invalid G3 run id")
    task = task_root.expanduser().resolve(strict=True)
    source = source_root.expanduser().resolve(strict=True)
    digest = _source_digest(source, task)
    gint = _gint_contract(
        source, gint_runtime_gate_receipt, gint_release_gate_acceptance, task,
    )
    receipt = Path(gint["path"])
    jobs, result_root = _stage_jobs(
        task=task, source=source, run_id=run_id, stage=stage,
        node=str(gint["node"]), tolerances=tolerances, gint_receipt=receipt,
    )
    return {
        "schema": PLAN_SCHEMA,
        "stage": stage,
        "case_id": _stage_case(stage),
        "run_id": run_id,
        "task_root": str(task),
        "source_root": str(source),
        "source_tree_sha256": digest,
        "node": gint["node"],
        "dimensions": CASE_DIMENSIONS[_stage_case(stage)],
        "eri_tolerances": [float(item) for item in tolerances],
        "rr_eig_cutoff": RR_EIG_CUTOFF,
        "result_root": str(result_root),
        "gint_release_receipt": gint,
        "gint_release_gate_acceptance": gint["release_gate_acceptance"],
        "jobs": jobs,
        "prerequisite": prerequisite,
        "scientific_boundary": {
            "comparison": "rr_canonical versus rr_cd complete pipelines",
            "same_rr_eig_cutoff": True,
            "same_serialized_rhf_orbitals": True,
            "projector_artifact_shared": False,
            "projectors_built_independently": True,
            "measured_effect": "CD increment in the final RR pipeline",
            "full_space_canonical_cd_equivalence": False,
            "acceptance_table": "A",
            "integral_subtrack": "CD increment diagnostic inside A-track RR",
            "performance_claim_eligible": False,
        },
        "policies": {
            "exact_grid_on_water2_and_water4": stage != "water8",
            "loosest_passing_tolerance": True,
            "prefer_tighter_within_fraction": TIMING_TIE_FRACTION,
            "water8_requires_independent_g4_authorization": True,
            "counterpoise_complete_fragment_count": CASE_DIMENSIONS[
                _stage_case(stage)
            ]["fragments"],
        },
    }


def make_water2_plan(
    *, task_root: Path, source_root: Path,
    gint_runtime_gate_receipt: Path, gint_release_gate_acceptance: Path,
    run_id: str,
) -> dict[str, Any]:
    return _base_plan(
        task_root=task_root, source_root=source_root,
        gint_runtime_gate_receipt=gint_runtime_gate_receipt,
        gint_release_gate_acceptance=gint_release_gate_acceptance,
        run_id=run_id, stage="water2", tolerances=ERI_TOLERANCES,
    )


def make_water4_plan(
    *, task_root: Path, source_root: Path,
    gint_runtime_gate_receipt: Path, gint_release_gate_acceptance: Path,
    run_id: str,
    water2_analysis: Path, water2_receipt: Path,
) -> dict[str, Any]:
    task = task_root.expanduser().resolve(strict=True)
    source = source_root.expanduser().resolve(strict=True)
    digest = _source_digest(source, task)
    gint = _gint_contract(
        source, gint_runtime_gate_receipt, gint_release_gate_acceptance, task,
    )
    analysis, receipt = _require_prior(
        water2_analysis, water2_receipt, stage="water2",
        source_digest=digest, gint_contract=gint,
    )
    prerequisite = {
        "stage": "water2",
        "analysis": _without_payload(analysis),
        "receipt": _without_payload(receipt),
    }
    return _base_plan(
        task_root=task, source_root=source,
        gint_runtime_gate_receipt=Path(gint["path"]),
        gint_release_gate_acceptance=Path(
            gint["release_gate_acceptance"]["path"]
        ),
        run_id=run_id, stage="water4", tolerances=ERI_TOLERANCES,
        prerequisite=prerequisite,
    )


def make_water8_plan(
    *, task_root: Path, source_root: Path,
    gint_runtime_gate_receipt: Path, gint_release_gate_acceptance: Path,
    run_id: str,
    water4_analysis: Path, water4_receipt: Path,
    g4_authorization: Path,
) -> dict[str, Any]:
    task = task_root.expanduser().resolve(strict=True)
    source = source_root.expanduser().resolve(strict=True)
    digest = _source_digest(source, task)
    gint = _gint_contract(
        source, gint_runtime_gate_receipt, gint_release_gate_acceptance, task,
    )
    analysis, receipt = _require_prior(
        water4_analysis, water4_receipt, stage="water4",
        source_digest=digest, gint_contract=gint,
    )
    selected = _number(_at(analysis, "payload.selection.selected_eri_tol"))
    if selected not in ERI_TOLERANCES:
        raise ContractError("water4 G3 analysis lacks one selected eri_tol")
    authorization = _require_g4_authorization(
        g4_authorization, task_root=task, source_digest=digest,
        selected_tol=float(selected), gint_contract=gint,
    )
    prerequisite = {
        "stage": "water4-plus-g4",
        "analysis": _without_payload(analysis),
        "receipt": _without_payload(receipt),
        "g4_authorization": _without_payload(authorization),
        "selected_eri_tol": float(selected),
    }
    return _base_plan(
        task_root=task, source_root=source,
        gint_runtime_gate_receipt=Path(gint["path"]),
        gint_release_gate_acceptance=Path(
            gint["release_gate_acceptance"]["path"]
        ),
        run_id=run_id, stage="water8", tolerances=(float(selected),),
        prerequisite=prerequisite,
    )


def _regenerate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    common = {
        "task_root": Path(str(plan["task_root"])),
        "source_root": Path(str(plan["source_root"])),
        "gint_runtime_gate_receipt": Path(
            str(plan["gint_release_receipt"]["path"])
        ),
        "gint_release_gate_acceptance": Path(
            str(plan["gint_release_gate_acceptance"]["path"])
        ),
        "run_id": str(plan["run_id"]),
    }
    stage = plan.get("stage")
    if stage == "water2":
        return make_water2_plan(**common)
    prerequisite = plan.get("prerequisite")
    if not isinstance(prerequisite, dict):
        raise ContractError("stage plan prerequisite is missing")
    if stage == "water4":
        return make_water4_plan(
            **common,
            water2_analysis=Path(prerequisite["analysis"]["path"]),
            water2_receipt=Path(prerequisite["receipt"]["path"]),
        )
    if stage == "water8":
        return make_water8_plan(
            **common,
            water4_analysis=Path(prerequisite["analysis"]["path"]),
            water4_receipt=Path(prerequisite["receipt"]["path"]),
            g4_authorization=Path(prerequisite["g4_authorization"]["path"]),
        )
    raise ContractError(f"unsupported G3 stage {stage!r}")


def _validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ContractError("unsupported G3 plan schema")
    stage = plan.get("stage")
    if stage not in {"water2", "water4", "water8"}:
        raise ContractError("G3 plan stage is invalid")
    prerequisite = plan.get("prerequisite")
    expected_tolerances = (
        list(ERI_TOLERANCES) if stage in {"water2", "water4"}
        else [
            prerequisite.get("selected_eri_tol")
            if isinstance(prerequisite, dict) else None
        ]
    )
    if plan.get("eri_tolerances") != expected_tolerances:
        raise ContractError("G3 plan tolerance grid is invalid")
    boundary = plan.get("scientific_boundary")
    if not isinstance(boundary, dict) or not all((
        boundary.get("projector_artifact_shared") is False,
        boundary.get("projectors_built_independently") is True,
        boundary.get("full_space_canonical_cd_equivalence") is False,
        boundary.get("performance_claim_eligible") is False,
    )):
        raise ContractError("G3 scientific boundary was weakened")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 4 * len(expected_tolerances):
        raise ContractError("G3 plan has an incomplete logical job set")
    ids: list[str] = []
    known: set[str] = set()
    for ordinal, job in enumerate(jobs):
        if not isinstance(job, dict) or job.get("ordinal") != ordinal:
            raise ContractError("G3 job ordinal is invalid")
        job_id = job.get("id")
        if not isinstance(job_id, str) or job_id in known:
            raise ContractError("G3 job IDs are missing or duplicated")
        dependencies = job.get("depends_on")
        if not isinstance(dependencies, list) or any(
            item not in known for item in dependencies
        ):
            raise ContractError("G3 job has a missing or forward dependency")
        if job.get("slurm_options") != [f"--nodelist={plan.get('node')}"]:
            raise ContractError("G3 job is not fixed to the receipt node")
        exports = job.get("exports")
        if not isinstance(exports, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            or any(character in value for character in ",\r\n")
            for key, value in exports.items()
        ):
            raise ContractError("G3 job has unsafe sbatch exports")
        if exports.get("RR_EIG_CUTOFF") != repr(RR_EIG_CUTOFF):
            raise ContractError("G3 RR cutoff was changed")
        if job.get("method") == "rr_cd":
            if (
                exports.get("GINT_RUNTIME_GATE_RECEIPT")
                != plan.get("gint_release_receipt", {}).get("path")
                or exports.get("GINT_COLUMN_BACKEND") != "selected"
                or exports.get("GINT_COLUMN_KERNEL") != "reference"
            ):
                raise ContractError("RR-CD job does not consume released selected GINT")
        elif job.get("method") != "rr_canonical":
            raise ContractError("G3 plan contains an unexpected method")
        ids.append(job_id)
        known.add(job_id)
    if len(set(ids)) != len(ids):
        raise ContractError("G3 logical jobs are duplicated")
    regenerated = _regenerate_plan(plan)
    if _canonical_json(regenerated) != _canonical_json(plan):
        raise ContractError("G3 plan differs from canonical regeneration")
    return plan


def _sbatch_argv(job: Mapping[str, Any], submitted: Mapping[str, str]) -> list[str]:
    dependencies = [submitted[item] for item in job["depends_on"]]
    exports = ["ALL"] + [
        f"{key}={value}" for key, value in sorted(job["exports"].items())
    ]
    argv = ["sbatch", "--parsable", f"--job-name={job['id']}"]
    argv.extend(job["slurm_options"])
    if dependencies:
        argv.append("--dependency=afterok:" + ":".join(dependencies))
    argv.extend(["--export=" + ",".join(exports), str(job["launcher"])])
    return argv


def submit_plan(
    plan: dict[str, Any], *, execute: bool, plan_path: Path | None = None,
) -> dict[str, Any]:
    plan = _validate_plan(plan)
    submitted: dict[str, str] = {}
    commands: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        argv = _sbatch_argv(job, submitted)
        if execute:
            raw = subprocess.check_output(argv, text=True).strip()
            job_id = raw.split(";", 1)[0]
            if not re.fullmatch(r"[1-9][0-9]*", job_id):
                raise ContractError(f"sbatch returned invalid job id {raw!r}")
        else:
            job_id = f"<{job['id']}>"
        submitted[job["id"]] = job_id
        commands.append({
            "job": job["id"], "job_id": job_id,
            "argv": argv, "shell_preview": shlex.join(argv),
        })
    return {
        "schema": SUBMISSION_SCHEMA,
        "executed": execute,
        "stage": plan["stage"],
        "case_id": plan["case_id"],
        "run_id": plan["run_id"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "gint_release_receipt": plan["gint_release_receipt"],
        "gint_release_gate_acceptance": plan["gint_release_gate_acceptance"],
        "plan_path": None if plan_path is None else str(plan_path.resolve()),
        "plan_payload_sha256": _payload_sha256(plan),
        "jobs": commands,
    }


def _validate_submission_receipt(receipt: Any) -> tuple[dict[str, Any], dict[str, str]]:
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != SUBMISSION_SCHEMA
        or receipt.get("executed") is not True
    ):
        raise ContractError("G3 submission receipt is not executed")
    plan_path = receipt.get("plan_path")
    if not isinstance(plan_path, str):
        raise ContractError("G3 submission receipt lacks its immutable plan")
    plan = _json_object(Path(plan_path), "G3 immutable plan")
    _validate_plan(plan)
    if (
        receipt.get("plan_payload_sha256") != _payload_sha256(plan)
        or receipt.get("stage") != plan.get("stage")
        or receipt.get("source_tree_sha256") != plan.get("source_tree_sha256")
        or receipt.get("gint_release_receipt", {}).get("sha256")
        != plan.get("gint_release_receipt", {}).get("sha256")
        or receipt.get("gint_release_gate_acceptance", {}).get("sha256")
        != plan.get("gint_release_gate_acceptance", {}).get("sha256")
    ):
        raise ContractError("G3 submission receipt differs from its plan")
    expected = [job["id"] for job in plan["jobs"]]
    items = receipt.get("jobs")
    if not isinstance(items, list) or len(items) != len(expected):
        raise ContractError("G3 receipt does not contain the exact job set")
    logical: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ContractError("G3 receipt contains a malformed job")
        name, job_id = item.get("job"), item.get("job_id")
        if (
            name not in expected or name in logical
            or not isinstance(job_id, str)
            or re.fullmatch(r"[1-9][0-9]*", job_id) is None
        ):
            raise ContractError("G3 receipt contains an extra, duplicate, or invalid job")
        logical[str(name)] = job_id
    if set(logical) != set(expected) or len(set(logical.values())) != len(logical):
        raise ContractError("G3 receipt job identity is not one-to-one")
    return plan, logical


def _record_path(record: Mapping[str, Any]) -> str | None:
    value = record.get("_tight_cd_g3_input_path")
    return None if value is None else str(value)


def _load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for raw in paths:
        path = raw.expanduser()
        if path.is_dir():
            files.extend(path.rglob("*.json"))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for path in sorted(set(item.resolve() for item in files)):
        value = _json_object(path, "G3 result record")
        if (
            value.get("schema") == "gpu4pyscf.water8.counterpoise.v1"
            or (
                value.get("method") in {"rr_canonical", "rr_cd"}
                and "post_hf_seconds" in value
            )
        ):
            value = dict(value)
            value["_tight_cd_g3_input_path"] = str(path)
            records.append(value)
    return records


def _bind_records(
    records: Sequence[dict[str, Any]], plan: Mapping[str, Any],
    logical: Mapping[str, str],
) -> list[str]:
    reasons: list[str] = []
    jobs = {job["id"]: job for job in plan["jobs"]}
    seen: set[str] = set()
    for record in records:
        slurm = record.get("slurm")
        name = slurm.get("SLURM_JOB_NAME") if isinstance(slurm, dict) else None
        job_id = slurm.get("SLURM_JOB_ID") if isinstance(slurm, dict) else None
        if name not in logical or logical.get(name) != str(job_id):
            reasons.append("result Slurm identity is outside the exact G3 receipt")
            continue
        if name in seen:
            reasons.append("more than one result claims the same G3 logical job")
            continue
        job = jobs[str(name)]
        if _record_path(record) != job["expected_outputs"][0]:
            reasons.append("result path differs from the immutable G3 plan")
        if not isinstance(slurm, dict) or slurm.get("SLURM_JOB_NODELIST") != plan["node"]:
            reasons.append("result node differs from the receipt-bound G3 node")
        if record.get("method") != job["method"]:
            reasons.append("result method differs from its G3 logical job")
        if record.get("case_id", _at(record, "case.id")) != plan["case_id"]:
            reasons.append("result case differs from the G3 plan")
        if not _same_float(_record_eri_tol(record), job["eri_tol"]):
            reasons.append("result eri_tol differs from its immutable G3 logical job")
        if not _same_float(_record_rr_cutoff(record), RR_EIG_CUTOFF):
            reasons.append("result RR cutoff differs from its immutable G3 logical job")
        orchestration = record.get("orchestration")
        if record.get("schema") == "gpu4pyscf.water8.counterpoise.v1":
            if not isinstance(orchestration, dict) or any((
                orchestration.get("logical_job_id") != name,
                orchestration.get("run_id") != job["exports"].get("CP_RUN_ID"),
                orchestration.get("stage") != f"tight-cd-g3-{plan['stage']}",
            )):
                reasons.append("counterpoise orchestration differs from the G3 plan")
        seen.add(str(name))
    if seen != set(logical):
        reasons.append("G3 receipt has jobs without exactly one synchronized result")
    return list(dict.fromkeys(reasons))


def _record_eri_tol(record: Mapping[str, Any]) -> float | None:
    for path in (
        "plan.settings.eri_tol", "protocol.options.eri_tol",
        "approximation.eri_tol", "eri_tol",
    ):
        value = _number(_at(record, path))
        if value is not None:
            return value
    return None


def _record_rr_cutoff(record: Mapping[str, Any]) -> float | None:
    for path in (
        "plan.settings.rr_eig_cutoff", "protocol.options.rr_eig_cutoff",
        "approximation.rr_eig_cutoff", "rr_eig_cutoff",
    ):
        value = _number(_at(record, path))
        if value is not None:
            return value
    return None


def _projector(record: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    rank = _at(record, "ranks.rr_rank")
    dimension = _at(record, "ranks.rr_full_dimension")
    orthogonality = _number(
        _at(record, "method_metadata.rr_projector_build.projector.orthogonality_error"),
        nonnegative=True,
    )
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        reasons.append("RR projector rank is missing or invalid")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        reasons.append("RR full pair dimension is missing or invalid")
    if orthogonality is None or orthogonality > PROJECTOR_ORTHOGONALITY_LIMIT:
        reasons.append("RR projector orthogonality exceeds 1e-10")
    return {
        "rank": rank,
        "full_dimension": dimension,
        "orthogonality_error": orthogonality,
        "built_independently": True,
        "shared_artifact": False,
    }, reasons


def _projected_residual(record: Mapping[str, Any]) -> float | None:
    for path in (
        "residual.projected_equation",
        "residual.measurements.projected_equation.norm",
        "projected_equation_residual",
    ):
        value = _number(_at(record, path), nonnegative=True)
        if value is not None:
            return value
    return None


def _orbital_pair(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any],
    expected_dimensions: Mapping[str, int],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    left, right = baseline.get("canonical_orbitals"), candidate.get("canonical_orbitals")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {}, ["canonical orbital evidence is missing"]
    if left.get("mode") != "written-and-reloaded":
        reasons.append("RR-canonical A did not write and reload orbitals")
    if right.get("mode") != "loaded":
        reasons.append("RR-CD B did not load A's orbitals")
    for label, payload in (("A", left), ("B", right)):
        if payload.get("applied_to_mean_field") is not True:
            reasons.append(f"orbital artifact was not applied in {label}")
        if payload.get("included_in_post_hf") is not False:
            reasons.append(f"orbital setup timing boundary is ambiguous in {label}")
        identity = payload.get("identity")
        if not isinstance(identity, dict) or any(
            identity.get(key) != expected_dimensions[key]
            for key in ("nao", "nocc", "nvir")
        ):
            reasons.append(f"orbital dimensions violate the case contract in {label}")
    for key in (
        "schema", "artifact_path", "artifact_sha256", "orbital_fingerprint",
        "identity", "producer",
    ):
        if left.get(key) != right.get(key):
            reasons.append(f"A/B orbital artifact mismatch: {key}")
    artifact_path = left.get("artifact_path")
    artifact_sha = left.get("artifact_sha256")
    if not isinstance(artifact_path, str) or not Path(artifact_path).is_file():
        reasons.append("serialized orbital artifact is missing")
    elif not isinstance(artifact_sha, str) or _sha256(Path(artifact_path)) != artifact_sha:
        reasons.append("serialized orbital artifact hash differs")
    for label, record, payload in (("A", baseline, left), ("B", candidate, right)):
        checkpoint = record.get("checkpoint")
        if not isinstance(checkpoint, dict) or checkpoint.get("included_in_post_hf") is not True:
            reasons.append(f"normal checkpoint is absent from {label} timing")
        elif (
            checkpoint.get("orbital_artifact_sha256") != payload.get("artifact_sha256")
            or checkpoint.get("orbital_fingerprint") != payload.get("orbital_fingerprint")
        ):
            reasons.append(f"checkpoint/orbital identity differs in {label}")
    return {
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha,
        "orbital_fingerprint": left.get("orbital_fingerprint"),
        "identity": left.get("identity"),
    }, reasons


def _pipeline_compatibility(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any],
    source_digest: str, node: str,
) -> list[str]:
    reasons: list[str] = []
    for label, record in (("A", baseline), ("B", candidate)):
        if _analysis._source_digest(record) != source_digest:
            reasons.append(f"{label} source digest differs from the plan")
        if not _analysis._source_stable(record):
            reasons.append(f"{label} source was not stable during the run")
        snapshot_reasons = _analysis._timed_snapshot_evidence_reasons(dict(record))
        reasons.extend(f"{label} {reason}" for reason in snapshot_reasons)
        if _at(record, "slurm.SLURM_JOB_NODELIST") != node:
            reasons.append(f"{label} did not execute on the receipt node")
        if record.get("status") != "completed" or record.get("cc_converged") is not True:
            reasons.append(f"{label} CCSD did not converge")
    for getter, label in (
        (_analysis._scientific_signature, "scientific protocol"),
        (_analysis._hardware_fingerprint, "hardware"),
        (_analysis._timing_boundary, "timing boundary"),
    ):
        left, right = getter(dict(baseline)), getter(dict(candidate))
        if left is None or right is None or left != right:
            reasons.append(f"A/B {label} is missing or differs")
    return reasons


def _gint_timing_gate(
    record: Mapping[str, Any], *, contract: Mapping[str, Any],
    source_digest: str,
) -> list[str]:
    gate = record.get("provider_runtime_gate")
    receipt = gate.get("receipt") if isinstance(gate, dict) else None
    current = receipt.get("current_runtime") if isinstance(receipt, dict) else None
    environment = current.get("environment") if isinstance(current, dict) else None
    job_id = _at(record, "slurm.SLURM_JOB_ID")
    if not all((
        isinstance(gate, dict), gate.get("validated") is True,
        gate.get("execution_mode") == "consumer-benchmark",
        isinstance(receipt, dict), receipt.get("path") == contract.get("path"),
        receipt.get("sha256") == contract.get("sha256"),
        receipt.get("payload_sha256") == contract.get("payload_sha256"),
        receipt.get("release_source_sha256") == source_digest,
        isinstance(environment, dict), environment.get("SLURM_JOB_ID") == job_id,
    )):
        return ["RR-CD timing record did not validate the pinned GINT receipt"]
    return []


def _gint_cp_gates(
    record: Mapping[str, Any], *, contract: Mapping[str, Any],
    source_digest: str, fragments: int,
) -> list[str]:
    container = record.get("provider_runtime_gates")
    cluster = container.get("cluster") if isinstance(container, dict) else None
    items = container.get("fragments") if isinstance(container, dict) else None
    if not isinstance(items, list) or len(items) != fragments:
        return ["RR-CD counterpoise lacks one GINT gate per cluster/fragment"]
    reasons: list[str] = []
    job_id = _at(record, "slurm.SLURM_JOB_ID")
    for label, gate in [("cluster", cluster), *[
        (f"fragment-{index}", value) for index, value in enumerate(items)
    ]]:
        receipt = gate.get("receipt") if isinstance(gate, dict) else None
        current = receipt.get("current_runtime") if isinstance(receipt, dict) else None
        environment = current.get("environment") if isinstance(current, dict) else None
        if not all((
            isinstance(gate, dict), gate.get("validated") is True,
            gate.get("execution_mode") == "consumer-counterpoise",
            isinstance(receipt, dict), receipt.get("path") == contract.get("path"),
            receipt.get("sha256") == contract.get("sha256"),
            receipt.get("payload_sha256") == contract.get("payload_sha256"),
            receipt.get("release_source_sha256") == source_digest,
            isinstance(environment, dict),
            environment.get("SLURM_JOB_ID") == job_id,
        )):
            reasons.append(f"RR-CD counterpoise {label} did not validate GINT receipt")
    return reasons


def _cp_pair(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], *,
    source_digest: str, fragments: int, contract: Mapping[str, Any],
) -> tuple[float | None, dict[str, Any], list[str]]:
    reasons: list[str] = []
    for label, record in (("A", baseline), ("B", candidate)):
        passed, record_reasons = _analysis._cp_eligibility(dict(record), source_digest)
        if not passed:
            reasons.extend(f"CP {label} {reason}" for reason in record_reasons)
        if len(record.get("fragments", [])) != fragments:
            reasons.append(f"CP {label} does not contain exactly {fragments} ghost monomers")
    left_signature = _analysis._cp_signature(dict(baseline))
    right_signature = _analysis._cp_signature(dict(candidate))
    if left_signature is None or left_signature != right_signature:
        reasons.append("CP A/B scientific protocols differ")
    try:
        bundle = _analysis.shared_bundle_proof(dict(baseline), dict(candidate))
    except (KeyError, TypeError, ValueError) as exc:
        bundle = {}
        reasons.append(f"CP A/B shared orbital bundle proof failed: {exc}")
    bundle_path = _at(candidate, "orbital_bundle.path")
    bundle_sha = _at(candidate, "orbital_bundle.sha256")
    if not isinstance(bundle_path, str) or not Path(bundle_path).is_file():
        reasons.append("CP orbital bundle file is missing")
    elif not isinstance(bundle_sha, str) or _sha256(Path(bundle_path)) != bundle_sha:
        reasons.append("CP orbital bundle hash differs from its record")
    reasons.extend(_gint_cp_gates(
        candidate, contract=contract, source_digest=source_digest,
        fragments=fragments,
    ))
    left = _number(baseline.get("interaction_energy_cp_kcal_mol"))
    right = _number(candidate.get("interaction_energy_cp_kcal_mol"))
    error = None if left is None or right is None else abs(right - left)
    if error is None or error > CP_LIMIT_KCAL_MOL:
        reasons.append("RR-CD counterpoise error exceeds 0.02 kcal/mol")
    return error, bundle, list(dict.fromkeys(reasons))


def _candidate_row(
    tolerance: float, records: Sequence[dict[str, Any]], *,
    plan: Mapping[str, Any], contract: Mapping[str, Any],
) -> dict[str, Any]:
    case_id = str(plan["case_id"])
    timed = [
        item for item in records
        if item.get("schema") != "gpu4pyscf.water8.counterpoise.v1"
        and item.get("case_id") == case_id
        and _same_float(_record_eri_tol(item), tolerance)
    ]
    cp = [
        item for item in records
        if item.get("schema") == "gpu4pyscf.water8.counterpoise.v1"
        and _at(item, "case.id") == case_id
        and _same_float(_record_eri_tol(item), tolerance)
    ]
    baselines = [item for item in timed if item.get("method") == "rr_canonical"]
    candidates = [item for item in timed if item.get("method") == "rr_cd"]
    cp_a = [item for item in cp if item.get("method") == "rr_canonical"]
    cp_b = [item for item in cp if item.get("method") == "rr_cd"]
    reasons: list[str] = []
    integrity_reasons: list[str] = []
    for label, values in (
        ("RR-canonical timing", baselines), ("RR-CD timing", candidates),
        ("RR-canonical CP", cp_a), ("RR-CD CP", cp_b),
    ):
        if len(values) != 1:
            message = f"expected exactly one {label} record; found {len(values)}"
            reasons.append(message)
            integrity_reasons.append(message)
    baseline = baselines[0] if len(baselines) == 1 else None
    candidate = candidates[0] if len(candidates) == 1 else None
    projector_a: dict[str, Any] = {}
    projector_b: dict[str, Any] = {}
    orbitals: dict[str, Any] = {}
    cp_bundle: dict[str, Any] = {}
    ecorr_error = cp_error = residual_a = residual_b = None
    seconds_a = seconds_b = speedup = None
    resources: dict[str, Any] = {}
    full_space_diagnostic: dict[str, Any] = {}
    if baseline is not None and candidate is not None:
        messages = _pipeline_compatibility(
            baseline, candidate, str(plan["source_tree_sha256"]), str(plan["node"]),
        )
        reasons.extend(messages)
        integrity_reasons.extend(messages)
        if not _same_float(_record_rr_cutoff(baseline), RR_EIG_CUTOFF) or not _same_float(
            _record_rr_cutoff(candidate), RR_EIG_CUTOFF
        ):
            message = "A/B RR eigenvalue cutoff differs from the G3 contract"
            reasons.append(message)
            integrity_reasons.append(message)
        orbitals, orbital_reasons = _orbital_pair(
            baseline, candidate, CASE_DIMENSIONS[case_id],
        )
        reasons.extend(orbital_reasons)
        integrity_reasons.extend(orbital_reasons)
        projector_a, projector_reasons = _projector(baseline)
        reasons.extend(f"A {reason}" for reason in projector_reasons)
        integrity_reasons.extend(f"A {reason}" for reason in projector_reasons)
        projector_b, projector_reasons = _projector(candidate)
        reasons.extend(f"B {reason}" for reason in projector_reasons)
        integrity_reasons.extend(f"B {reason}" for reason in projector_reasons)
        if projector_a.get("full_dimension") != projector_b.get("full_dimension"):
            message = "A/B RR full pair dimensions differ"
            reasons.append(message)
            integrity_reasons.append(message)
        expected_pair_dimension = (
            int(CASE_DIMENSIONS[case_id]["nocc"])
            * int(CASE_DIMENSIONS[case_id]["nvir"])
        )
        if (
            projector_a.get("full_dimension") != expected_pair_dimension
            or projector_b.get("full_dimension") != expected_pair_dimension
        ):
            message = "A/B RR pair dimensions violate the frozen case contract"
            reasons.append(message)
            integrity_reasons.append(message)
        left_energy, right_energy = _number(baseline.get("e_corr")), _number(candidate.get("e_corr"))
        ecorr_error = (
            None if left_energy is None or right_energy is None
            else abs(right_energy - left_energy)
        )
        if ecorr_error is None or ecorr_error > ECORR_LIMIT_EH:
            reasons.append("RR-CD correlation-energy increment exceeds 1e-4 Eh")
        residual_a, residual_b = _projected_residual(baseline), _projected_residual(candidate)
        if residual_a is None or residual_a > PROJECTED_RESIDUAL_LIMIT:
            reasons.append("RR-canonical projected residual exceeds 1e-6")
        if residual_b is None or residual_b > PROJECTED_RESIDUAL_LIMIT:
            reasons.append("RR-CD projected residual exceeds 1e-6")
        seconds_a = _number(baseline.get("post_hf_seconds"))
        seconds_b = _number(candidate.get("post_hf_seconds"))
        if seconds_a is None or seconds_b is None or seconds_a <= 0 or seconds_b <= 0:
            message = "A/B whole-stage timing is missing or invalid"
            reasons.append(message)
            integrity_reasons.append(message)
        else:
            speedup = seconds_a / seconds_b
        resources = {
            "rr_canonical": _analysis._resources(dict(baseline)),
            "rr_cd": _analysis._resources(dict(candidate)),
            "acceptance_gate": "rr_cd",
        }
        if resources["rr_cd"].get("passed") is not True:
            reasons.append("RR-CD hardware, HBM, RSS, or checkpoint resource gate failed")
        messages = _gint_timing_gate(
            candidate, contract=contract,
            source_digest=str(plan["source_tree_sha256"]),
        )
        reasons.extend(messages)
        integrity_reasons.extend(messages)
        diagnostic = candidate.get("full_space_residual_diagnostic")
        diagnostic_norm = _number(
            diagnostic.get("residual_norm") if isinstance(diagnostic, dict) else None,
            nonnegative=True,
        )
        full_space_diagnostic = {
            "available": isinstance(diagnostic, dict)
            and diagnostic.get("available") is True
            and diagnostic.get("execution_status") == "completed"
            and diagnostic_norm is not None,
            "residual_norm": diagnostic_norm,
            "acceptance_limit": None,
            "role": "reported approximation-bias diagnostic, not a convergence gate",
        }
        if full_space_diagnostic["available"] is not True:
            message = "RR-CD reconstructed full-space residual diagnostic is missing"
            reasons.append(message)
            integrity_reasons.append(message)
    if len(cp_a) == 1 and len(cp_b) == 1:
        cp_error, cp_bundle, cp_reasons = _cp_pair(
            cp_a[0], cp_b[0], source_digest=str(plan["source_tree_sha256"]),
            fragments=int(CASE_DIMENSIONS[case_id]["fragments"]),
            contract=contract,
        )
        reasons.extend(cp_reasons)
        integrity_reasons.extend(
            reason for reason in cp_reasons
            if "counterpoise error exceeds" not in reason
        )
    reasons = list(dict.fromkeys(reasons))
    integrity_reasons = list(dict.fromkeys(integrity_reasons))
    return {
        "eri_tol": tolerance,
        "threshold_token": _threshold_token(tolerance),
        "records": {
            "rr_canonical": None if baseline is None else _record_path(baseline),
            "rr_cd": None if candidate is None else _record_path(candidate),
            "rr_canonical_cp": None if len(cp_a) != 1 else _record_path(cp_a[0]),
            "rr_cd_cp": None if len(cp_b) != 1 else _record_path(cp_b[0]),
        },
        "orbitals": orbitals,
        "projectors": {
            "rr_canonical": projector_a, "rr_cd": projector_b,
            "same_artifact_claimed": False,
            "rank_delta": (
                None if not isinstance(projector_a.get("rank"), int)
                or not isinstance(projector_b.get("rank"), int)
                else projector_b["rank"] - projector_a["rank"]
            ),
        },
        "accuracy": {
            "abs_delta_e_corr_eh": ecorr_error,
            "e_corr_limit_eh": ECORR_LIMIT_EH,
            "rr_canonical_projected_residual": residual_a,
            "rr_cd_projected_residual": residual_b,
            "projected_residual_limit": PROJECTED_RESIDUAL_LIMIT,
            "counterpoise_error_kcal_mol": cp_error,
            "counterpoise_limit_kcal_mol": CP_LIMIT_KCAL_MOL,
            "full_space_residual_diagnostic": full_space_diagnostic,
        },
        "counterpoise_bundle": cp_bundle,
        "timing": {
            "rr_canonical_post_hf_s": seconds_a,
            "rr_cd_post_hf_s": seconds_b,
            "pipeline_speedup": speedup,
            "selection_metric": "rr_cd complete post-HF seconds",
        },
        "resources": resources,
        "qualifies": not reasons,
        "evidence_integrity_passed": not integrity_reasons,
        "evidence_integrity_reasons": integrity_reasons,
        "performance_claim_eligible": False,
        "acceptance_table": "A",
        "integral_subtrack": "CD increment diagnostic inside A-track RR",
        "reasons": reasons,
    }


def select_eri_tolerance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the loose-first, 3%-tie-prefers-tighter selection rule."""

    eligible = [row for row in rows if row.get("qualifies") is True]
    if not eligible:
        return {
            "selected_eri_tol": None, "anchor_eri_tol": None,
            "eligible_eri_tolerances": [], "reason": "no tolerance passed every G3 gate",
        }
    anchor = eligible[0]
    anchor_seconds = _number(_at(anchor, "timing.rr_cd_post_hf_s"))
    if anchor_seconds is None or anchor_seconds <= 0:
        return {
            "selected_eri_tol": None, "anchor_eri_tol": anchor.get("eri_tol"),
            "eligible_eri_tolerances": [row.get("eri_tol") for row in eligible],
            "reason": "loose passing anchor has no valid whole-stage timing",
        }
    selected = anchor
    tied = [anchor.get("eri_tol")]
    ceiling = anchor_seconds * (1.0 + TIMING_TIE_FRACTION)
    for row in eligible[1:]:
        seconds = _number(_at(row, "timing.rr_cd_post_hf_s"))
        if seconds is not None and 0.0 < seconds <= ceiling:
            selected = row
            tied.append(row.get("eri_tol"))
    return {
        "selected_eri_tol": selected.get("eri_tol"),
        "anchor_eri_tol": anchor.get("eri_tol"),
        "eligible_eri_tolerances": [row.get("eri_tol") for row in eligible],
        "tie_group": tied,
        "tie_ceiling_seconds": ceiling,
        "selected_rr_cd_post_hf_s": _at(selected, "timing.rr_cd_post_hf_s"),
        "reason": (
            "loosest passing tolerance, with the tightest passing point no more "
            "than 3% slower than that loose anchor"
        ),
    }


def analyze_stage(
    records: Sequence[dict[str, Any]], submission_receipt: Path,
) -> dict[str, Any]:
    receipt_evidence = _file_evidence(submission_receipt, "G3 submission receipt")
    plan, logical = _validate_submission_receipt(receipt_evidence["payload"])
    contract = plan["gint_release_receipt"]
    reasons = _bind_records(records, plan, logical)
    rows = [
        _candidate_row(float(tolerance), records, plan=plan, contract=contract)
        for tolerance in plan["eri_tolerances"]
    ]
    expected_count = 4 * len(plan["eri_tolerances"])
    if len(records) != expected_count:
        reasons.append(
            f"expected exactly {expected_count} receipt-bound result records; found {len(records)}"
        )
    evidence_complete = not reasons and all(
        row["evidence_integrity_passed"] for row in rows
    )
    selection = select_eri_tolerance(rows)
    if selection["selected_eri_tol"] is None:
        reasons.append(str(selection["reason"]))
    stage = plan["stage"]
    stage_gate_passed = bool(
        evidence_complete and selection["selected_eri_tol"] is not None
    )
    selection.update({
        "ready_for_water4": bool(stage == "water2" and stage_gate_passed),
        "ready_for_g4": bool(stage == "water4" and stage_gate_passed),
        "ready_for_water8": False,
        "water8_requires_independent_g4_authorization": True,
        "water8_verified": bool(stage == "water8" and stage_gate_passed),
        "reasons": list(dict.fromkeys(reasons)),
    })
    return {
        "schema": SCHEMA,
        "stage": stage,
        "case_id": plan["case_id"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "node": plan["node"],
        "rr_eig_cutoff": RR_EIG_CUTOFF,
        "gint_release_receipt": plan["gint_release_receipt"],
        "gint_release_gate_acceptance": plan["gint_release_gate_acceptance"],
        "grid_complete": evidence_complete,
        "records_loaded": len(records),
        "rows": rows,
        "selection": selection,
        "submission_binding": {
            "receipt_path": receipt_evidence["path"],
            "receipt_file_sha256": receipt_evidence["file_sha256"],
            "receipt_payload_sha256": receipt_evidence["payload_sha256"],
            "plan_path": receipt_evidence["payload"]["plan_path"],
            "plan_payload_sha256": receipt_evidence["payload"]["plan_payload_sha256"],
            "logical_job_ids": logical,
            "release_gate_acceptance_sha256": plan[
                "gint_release_gate_acceptance"
            ]["sha256"],
        },
        "scientific_boundary": plan["scientific_boundary"],
        "claim_state": {
            "stage_gate_passed": stage_gate_passed,
            "g3_complete_claim_eligible": False,
            "full_space_canonical_cd_equivalence_proven": False,
            "projector_identity_proven": False,
            "water8_unlock_from_g3_alone": False,
            "speedup_claim_eligible": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--task-root", type=Path, required=True)
    plan.add_argument("--source-root", type=Path, required=True)
    plan.add_argument("--gint-runtime-gate-receipt", type=Path, required=True)
    plan.add_argument("--gint-release-gate-acceptance", type=Path, required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--stage", choices=("water2", "water4", "water8"), required=True)
    plan.add_argument("--prior-analysis", type=Path)
    plan.add_argument("--prior-receipt", type=Path)
    plan.add_argument("--g4-authorization", type=Path)
    plan.add_argument("--output", type=Path)
    submit = commands.add_parser("submit")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--execute", action="store_true")
    submit.add_argument("--receipt", type=Path)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--records", type=Path, nargs="+", required=True)
    analyze.add_argument("--submission-receipt", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        common = {
            "task_root": args.task_root, "source_root": args.source_root,
            "gint_runtime_gate_receipt": args.gint_runtime_gate_receipt,
            "gint_release_gate_acceptance": args.gint_release_gate_acceptance,
            "run_id": args.run_id,
        }
        if args.stage == "water2":
            if any((args.prior_analysis, args.prior_receipt, args.g4_authorization)):
                raise ContractError("water2 plan accepts no prerequisite artifacts")
            payload = make_water2_plan(**common)
        elif args.stage == "water4":
            if args.prior_analysis is None or args.prior_receipt is None:
                raise ContractError("water4 requires water2 analysis and receipt")
            if args.g4_authorization is not None:
                raise ContractError("water4 does not consume G4 authorization")
            payload = make_water4_plan(
                **common, water2_analysis=args.prior_analysis,
                water2_receipt=args.prior_receipt,
            )
        else:
            if any(item is None for item in (
                args.prior_analysis, args.prior_receipt, args.g4_authorization,
            )):
                raise ContractError("water8 requires water4 analysis, receipt, and G4 authorization")
            payload = make_water8_plan(
                **common, water4_analysis=args.prior_analysis,
                water4_receipt=args.prior_receipt,
                g4_authorization=args.g4_authorization,
            )
        if args.output is None:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            _json_write_once(args.output, payload)
            print(args.output.resolve())
        return 0
    if args.command == "submit":
        plan = _json_object(args.plan, "G3 plan")
        if args.execute and args.receipt is None:
            raise ContractError("--execute requires --receipt")
        if not args.execute and args.receipt is not None:
            raise ContractError("--receipt is valid only with --execute")
        receipt = submit_plan(plan, execute=args.execute, plan_path=args.plan)
        if args.receipt is not None:
            _json_write_once(args.receipt, receipt)
            print(args.receipt.resolve())
        else:
            for job in receipt["jobs"]:
                print(job["shell_preview"])
        return 0
    records = _load_records(args.records)
    payload = analyze_stage(records, args.submission_receipt)
    _json_write_once(args.output, payload)
    print(json.dumps(payload["selection"], indent=2, sort_keys=True))
    stage = payload["stage"]
    ready = (
        payload["selection"]["ready_for_water4"] if stage == "water2"
        else payload["selection"]["ready_for_g4"] if stage == "water4"
        else payload["selection"]["water8_verified"]
    )
    return 0 if ready else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"tight-CD G3 contract error: {exc}", file=sys.stderr)
        raise SystemExit(2)
