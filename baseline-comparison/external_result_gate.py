#!/usr/bin/env python3
"""Audit external WATER2 CCSD results and plan a dry-run WATER4 promotion.

This module is intentionally read-only with respect to benchmark results,
scheduler evidence, case contracts, and oracle records.  An audit records the
SHA-256 of every input and fails closed when it cannot bind a completed Slurm
job to one result, one frozen runner, one physical A100, and one canonical
numerical oracle.  The optional output files are created write-once.

``audit`` understands the v2 ByteQC and GANSU result documents emitted by the
existing MTU harnesses.  ``plan-water4`` never calls ``sbatch``: it emits a
manifest and includes a WATER4 runner argv only when the complete WATER2 gate
passed and the receipt-bound runner capability explicitly supports WATER4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


AUDIT_SCHEMA = "agent-qc.external-baseline-result-audit.v1"
PLAN_SCHEMA = "agent-qc.external-baseline-water4-plan.v1"
CAPABILITY_SCHEMA = "agent-qc.external-baseline-runner-capabilities.v1"
ORACLE_PROTOCOL_SCHEMA = "gpu4pyscf.water8-ccsd-a100.v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
JOB_ID_RE = re.compile(r"^[1-9][0-9]*$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)

CASE_ID = "water2-tz"
PROMOTION_CASE_ID = "water4-tz"
EXPECTED_SOURCE = "WATER27:H2O2"
EXPECTED_BASIS = "cc-pVTZ"
EXPECTED_GPU = "NVIDIA A100-SXM4-80GB"
EXPECTED_GPU_MEMORY_MIB = 81920
EXPECTED_CPU_FRAGMENT = "AMD EPYC 7513 32-Core Processor"
EXPECTED_CPU_AFFINITY = list(range(24, 32))
EXPECTED_NUMA = 3
ALLOWED_NODES = frozenset({"compute-1-0", "compute-1-2", "compute-1-3"})
HBM_LIMIT_MIB = 72 * 1024
RSS_LIMIT_GIB = 110.0
MAX_ENERGY_TOLERANCE_EH = 1e-4
GANSU_DIGEST_ENV = {
    "harness": "GANSU_EXPECTED_HARNESS_SHA256",
    "cases": "GANSU_EXPECTED_CASES_SHA256",
    "basis": "GANSU_EXPECTED_BASIS_SHA256",
    "auxiliary_basis": "GANSU_EXPECTED_AUX_BASIS_SHA256",
    "native_library": "GANSU_EXPECTED_NATIVE_SHA256",
    "python_wrapper": "GANSU_EXPECTED_WRAPPER_SHA256",
    "native_metadata": "GANSU_EXPECTED_NATIVE_META_SHA256",
    "libgfortran": "GANSU_EXPECTED_LIBGFORTRAN_SHA256",
    "libquadmath": "GANSU_EXPECTED_LIBQUADMATH_SHA256",
    "distribution_metadata": "GANSU_EXPECTED_DIST_METADATA_SHA256",
    "distribution_record": "GANSU_EXPECTED_DIST_RECORD_SHA256",
    "distribution_wheel": "GANSU_EXPECTED_DIST_WHEEL_SHA256",
    "cuda_runtime": "GANSU_EXPECTED_CUDART_SHA256",
}

SOFTWARE = {
    "byteqc": {
        "result_schema": "agent-qc.single-a100-byteqc-rccsd.v2",
        "release": "02af25481261ff29fa91ff5ba807728d6416a91b",
        "acceptance_table": "S",
        "evidence_level": "L1",
        "role": "protocol-adjacent canonical candidate",
        "formal_speedup_role": "candidate only; formal V1 timing requires missing capabilities",
    },
    "gansu": {
        "result_schema": "agent-qc.single-a100-gansu-ri-rccsd.v2",
        "release": "0b000a6cbcc5f5a630940ac1c2ba7f7ac2b8f132",
        "acceptance_table": "I",
        "evidence_level": "L2",
        "role": "RI B-native diagnostic",
        "formal_speedup_role": "never a V1 post-HF formal speedup baseline",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {token!r} is forbidden")

    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {resolved}")
    return value


def _evidence(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    payload = _read_json(resolved, label)
    return payload, {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved),
        "payload_sha256": _payload_sha256(payload),
    }


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {resolved}") from exc


def _at(value: Any, path: str) -> Any:
    current = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first(value: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        candidate = _at(value, path)
        if candidate is not None:
            return candidate
    return None


def _number(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        return None
    return result


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _sha(value: Any) -> str | None:
    return value if isinstance(value, str) and SHA256_RE.fullmatch(value) else None


def _geometry_sha256(case: Mapping[str, Any]) -> str:
    geometry = case.get("geometry_angstrom")
    errors = _geometry_contract_errors(geometry, "geometry")
    if errors:
        raise ValueError("; ".join(errors))
    encoded = json.dumps(
        geometry,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _geometry_contract_errors(geometry: Any, label: str) -> list[str]:
    if not isinstance(geometry, list) or not geometry:
        return [f"{label} must be a non-empty list of atom rows"]
    reasons: list[str] = []
    for index, row in enumerate(geometry):
        if not isinstance(row, list) or len(row) != 4:
            reasons.append(f"{label} row {index} must contain one atom and three coordinates")
            continue
        atom = row[0]
        if not isinstance(atom, str) or re.fullmatch(r"[A-Z][a-z]?", atom) is None:
            reasons.append(f"{label} row {index} has an invalid atom symbol")
        for axis, coordinate in zip("xyz", row[1:]):
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(float(coordinate))
            ):
                reasons.append(
                    f"{label} row {index} coordinate {axis} is not a finite JSON number"
                )
    return reasons


def _case_contract(cases: Mapping[str, Any]) -> dict[str, Any] | None:
    records = cases.get("cases")
    if not isinstance(records, list):
        return None
    matches = [item for item in records if isinstance(item, dict) and item.get("id") == CASE_ID]
    return matches[0] if len(matches) == 1 else None


def _gate(passed: bool, reasons: Sequence[str], **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "reasons": list(dict.fromkeys(reasons)), **details}


def _scheduler_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize explicit scheduler JSON, including the existing job audits."""

    scontrol = _first(payload, "terminal_evidence.scontrol", "job.scontrol", "scontrol")
    text = scontrol if isinstance(scontrol, str) else ""

    def from_scontrol(pattern: str) -> str | None:
        match = re.search(pattern, text)
        return match.group(1) if match else None

    job_id = _first(
        payload,
        "terminal_evidence.job_id",
        "sacct.JobID",
        "job_id",
        "job.id",
    )
    state = _first(
        payload,
        "terminal_evidence.state",
        "sacct.State",
        "status.state",
        "job.state",
        "state",
    )
    exit_code = _first(
        payload,
        "terminal_evidence.exit_code",
        "sacct.ExitCode",
        "status.exit_code",
        "job.exit_code",
        "exit_code",
    )
    node = _first(
        payload,
        "terminal_evidence.node",
        "sacct.NodeList",
        "status.node",
        "job.node",
        "node",
    )
    return {
        "job_id": str(job_id) if job_id is not None else from_scontrol(r"\bJobId=([0-9]+)"),
        "state": str(state).upper() if state is not None else from_scontrol(r"\bJobState=([^ ]+)"),
        "exit_code": str(exit_code) if exit_code is not None else from_scontrol(r"\bExitCode=([^ ]+)"),
        "node": str(node) if node is not None else from_scontrol(r"\bNodeList=([^ ]+)"),
    }


def _scheduler_source_sha(payload: Mapping[str, Any]) -> dict[str, Any]:
    direct = payload.get("source_sha256")
    if isinstance(direct, dict):
        return direct
    assets = _at(payload, "files.runtime_assets")
    return dict(assets) if isinstance(assets, Mapping) else {}


def _terminal_scheduler_gate(
    scheduler: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = _scheduler_record(scheduler)
    reasons: list[str] = []
    if normalized["state"] != "COMPLETED":
        reasons.append(f"scheduler state is {normalized['state']!r}, expected 'COMPLETED'")
    if normalized["exit_code"] != "0:0":
        reasons.append(f"scheduler exit code is {normalized['exit_code']!r}, expected '0:0'")
    job_id = normalized["job_id"]
    if not isinstance(job_id, str) or JOB_ID_RE.fullmatch(job_id) is None:
        reasons.append("terminal scheduler evidence lacks a numeric Slurm job ID")
    node = normalized["node"]
    if node not in ALLOWED_NODES:
        reasons.append(f"scheduler node is {node!r}, expected one of {sorted(ALLOWED_NODES)}")

    result_job = _first(result, "bound_topology.slurm.SLURM_JOB_ID")
    result_node = _first(
        result,
        "bound_topology.hostname",
        "bound_topology.slurm.SLURM_JOB_NODELIST",
    )
    if result_job is None or str(result_job) != job_id:
        reasons.append("result Slurm job ID does not match terminal scheduler evidence")
    if result_node != node:
        reasons.append("result host does not match terminal scheduler evidence")
    return _gate(not reasons, reasons, observed=normalized)


def _schema_status_gate(software: str, result: Mapping[str, Any]) -> dict[str, Any]:
    expected = SOFTWARE[software]["result_schema"]
    good_status = "completed" if software == "byteqc" else "complete"
    reasons: list[str] = []
    if result.get("schema") != expected:
        reasons.append(f"result schema is {result.get('schema')!r}, expected {expected!r}")
    if result.get("status") != good_status:
        reasons.append(f"result status is {result.get('status')!r}, expected {good_status!r}")
    if result.get("error") is not None:
        reasons.append("result retains an error payload")
    return _gate(not reasons, reasons, expected_schema=expected, expected_status=good_status)


def _case_gate(
    result: Mapping[str, Any], cases: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    trusted = _case_contract(cases)
    reasons: list[str] = []
    if trusted is None:
        return _gate(False, ["trusted cases file has no unique water2-tz record"]), None
    if trusted.get("source") != EXPECTED_SOURCE:
        reasons.append("trusted water2 source is not WATER27:H2O2")
    if trusted.get("basis") != EXPECTED_BASIS:
        reasons.append("trusted water2 basis is not cc-pVTZ")
    dimensions = trusted.get("expected_dimensions")
    if dimensions != {"nao": 116, "nocc": 10, "nvir": 106}:
        reasons.append("trusted water2 dimensions differ from nao=116,nocc=10,nvir=106")
    trusted_geometry_errors = _geometry_contract_errors(
        trusted.get("geometry_angstrom"), "trusted water2 geometry"
    )
    reasons.extend(trusted_geometry_errors)
    expected_geometry_sha = (
        None if trusted_geometry_errors else _geometry_sha256(trusted)
    )
    candidate_case = result.get("case")
    if not isinstance(candidate_case, Mapping):
        reasons.append("result case object is missing")
    else:
        reasons.extend(
            _geometry_contract_errors(
                candidate_case.get("geometry_angstrom"), "result water2 geometry"
            )
        )
        for field in ("id", "source", "basis", "expected_dimensions", "geometry_angstrom"):
            if candidate_case.get(field) != trusted.get(field):
                reasons.append(f"result case field {field} differs from the trusted contract")
    if expected_geometry_sha is None or result.get("geometry_sha256") != expected_geometry_sha:
        reasons.append("result geometry SHA-256 differs from the trusted geometry")
    observed_dimensions = result.get("dimensions")
    if not isinstance(observed_dimensions, Mapping):
        reasons.append("observed orbital dimensions are missing")
    else:
        projected = {key: observed_dimensions.get(key) for key in ("nao", "nocc", "nvir")}
        if projected != dimensions:
            reasons.append("observed orbital dimensions differ from the trusted contract")
    return _gate(
        not reasons,
        reasons,
        case_id=CASE_ID,
        geometry_sha256=expected_geometry_sha,
        expected_dimensions=dimensions,
    ), dict(trusted)


def _method_gate(software: str, result: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if software == "byteqc":
        contract = result.get("comparison_contract")
        classifier = result.get("runtime_method_classifier")
        if not isinstance(contract, Mapping):
            reasons.append("ByteQC comparison contract is missing")
            contract = {}
        expected = {
            "software": "ByteQC",
            "release_commit": SOFTWARE[software]["release"],
            "acceptance_table": "S-candidate-pending-oracle-validation",
            "density_fitting": False,
            "precision": "FP64",
            "basis": EXPECTED_BASIS,
            "spherical": True,
            "charge": 0,
            "spin": 0,
            "frozen_occupied": 0,
            "scf_excluded_from_timer": True,
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                reasons.append(f"ByteQC method field {key} differs from {value!r}")
        if contract.get("eri_representation") != "exact four-index ERIs":
            reasons.append("ByteQC result is not the canonical four-index path")
        if not isinstance(classifier, Mapping) or classifier.get("verified") is not True:
            reasons.append("ByteQC runtime method classifier did not verify the requested solver")
        elif classifier.get("actual_class") != "byteqc.cucc.ccsd.RCCSD":
            reasons.append("ByteQC runtime class is not byteqc.cucc.ccsd.RCCSD")
    else:
        method = result.get("method")
        backend = result.get("backend_verification")
        if not isinstance(method, Mapping):
            reasons.append("GANSU method contract is missing")
            method = {}
        expected = {
            "software": "GANSU",
            "version": "2026.8.10",
            "acceptance_table": "I",
            "canonical_four_index": False,
            "density_fitting": True,
            "thc": False,
        }
        for key, value in expected.items():
            if method.get(key) != value:
                reasons.append(f"GANSU method field {key} differs from {value!r}")
        if not isinstance(backend, Mapping):
            reasons.append("GANSU backend verification is missing")
        else:
            for key in (
                "requested_bnative_environment",
                "single_gpu_gate_passed",
                "ccsd_callback_shape_consistent_with_bnative",
            ):
                if backend.get(key) is not True:
                    reasons.append(f"GANSU backend capability {key} was not verified")
            if backend.get("requested_eri_method") != "ri":
                reasons.append("GANSU did not report the RI backend")
    return _gate(not reasons, reasons)


def _source_gate(
    software: str,
    result: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    capability: Mapping[str, Any],
    cases_file_sha256: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    scheduler_sha = _scheduler_source_sha(scheduler)
    runner_sha = capability.get("runner_sha256")
    if _sha(runner_sha) is None:
        reasons.append("runner capability does not contain a valid runner SHA-256")
    if capability.get("schema") != CAPABILITY_SCHEMA:
        reasons.append("runner capability schema is unsupported")
    if capability.get("software") != software:
        reasons.append("runner capability names a different software package")
    if capability.get("result_schema") != SOFTWARE[software]["result_schema"]:
        reasons.append("runner capability names a different result schema")

    if software == "byteqc":
        provenance = result.get("provenance")
        if not isinstance(provenance, Mapping):
            reasons.append("ByteQC provenance is missing")
            provenance = {}
        if provenance.get("head_commit") != SOFTWARE[software]["release"]:
            reasons.append("ByteQC source commit differs from the pinned release")
        if provenance.get("expected_commit") != SOFTWARE[software]["release"]:
            reasons.append("ByteQC expected commit differs from the pinned release")
        if provenance.get("tracked_source_clean") is not True:
            reasons.append("ByteQC tracked source was not clean")
        if not isinstance(provenance.get("source_root"), str) or not provenance.get("source_root"):
            reasons.append("ByteQC frozen source root is missing")
        if not isinstance(_at(provenance, "cases.path"), str) or not _at(provenance, "cases.path"):
            reasons.append("ByteQC executed cases path is missing")
        libraries = (
            provenance.get("native_libraries")
            if isinstance(provenance.get("native_libraries"), Mapping)
            else {}
        )
        bindings = {
            "byteqc_benchmark.py": _at(provenance, "harness.sha256"),
            "cases": _at(provenance, "cases.sha256"),
            "libgint": _at(libraries.get("libgint.so"), "sha256"),
            "libgvhf": _at(libraries.get("libgvhf.so"), "sha256"),
        }
    else:
        provenance = result.get("provenance")
        if not isinstance(provenance, Mapping):
            reasons.append("GANSU provenance is missing")
            provenance = {}
        release = provenance.get("release") if isinstance(provenance.get("release"), Mapping) else {}
        if release.get("official_tag_commit") != SOFTWARE[software]["release"]:
            reasons.append("GANSU tag commit differs from the pinned release")
        if not isinstance(provenance.get("runtime_root"), str) or not provenance.get("runtime_root"):
            reasons.append("GANSU frozen runtime root is missing")
        files = provenance.get("files") if isinstance(provenance.get("files"), Mapping) else {}
        for key in (
            "cases",
            "basis",
            "auxiliary_basis",
            "native_library",
        ):
            if not isinstance(_at(files.get(key), "path"), str) or not _at(files.get(key), "path"):
                reasons.append(f"GANSU {key} path is missing")
        bindings = {
            key: _at(provenance, f"files.{key}.sha256")
            for key in (
                "harness",
                "cases",
                "basis",
                "auxiliary_basis",
                "native_library",
                "python_wrapper",
                "native_metadata",
                "libgfortran",
                "libquadmath",
                "distribution_metadata",
                "distribution_record",
                "distribution_wheel",
                "cuda_runtime",
            )
        }
    for key, actual in bindings.items():
        expected = scheduler_sha.get(key)
        if _sha(actual) is None:
            reasons.append(f"result provenance lacks a valid {key} SHA-256")
        if _sha(expected) is None:
            reasons.append(f"scheduler evidence lacks a valid {key} SHA-256")
        if actual != expected:
            reasons.append(f"result and scheduler {key} SHA-256 values differ")
    harness_key = "byteqc_benchmark.py" if software == "byteqc" else "harness"
    if bindings.get(harness_key) != runner_sha:
        reasons.append("runner capability SHA-256 differs from the executed harness")
    if bindings.get("cases") != cases_file_sha256:
        reasons.append("executed cases SHA-256 differs from the trusted cases input")
    return _gate(not reasons, reasons, bound_sha256=bindings)


def _topology_gate(software: str, result: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    topology = result.get("bound_topology")
    if not isinstance(topology, Mapping):
        return _gate(False, ["bound topology evidence is missing"])
    node = topology.get("hostname")
    if node not in ALLOWED_NODES:
        reasons.append(f"result node is {node!r}, expected one of {sorted(ALLOWED_NODES)}")
    if EXPECTED_CPU_FRAGMENT not in str(topology.get("cpu_model", "")):
        reasons.append("CPU model is not AMD EPYC 7513")
    if topology.get("cpu_affinity") != EXPECTED_CPU_AFFINITY:
        reasons.append("runtime CPU affinity is not physical cores 24-31")
    if topology.get("unique_physical_cores") != 8:
        reasons.append("runtime did not demonstrate eight unique physical CPU cores")
    gpus = topology.get("visible_gpus")
    if not isinstance(gpus, list) or len(gpus) != 1 or not isinstance(gpus[0], Mapping):
        return _gate(False, reasons + ["topology does not contain exactly one physical GPU"])
    gpu = gpus[0]
    if gpu.get("name") != EXPECTED_GPU:
        reasons.append("GPU model is not NVIDIA A100-SXM4-80GB")
    if gpu.get("memory_total_mib") != EXPECTED_GPU_MEMORY_MIB:
        reasons.append("GPU memory inventory is not 81920 MiB")
    if str(gpu.get("numa_node")) != str(EXPECTED_NUMA):
        reasons.append("GPU is not attached to NUMA node 3")
    if str(gpu.get("current_link_width")) != "16":
        reasons.append("GPU PCIe link width is not x16")
    if "16.0 GT/s" not in str(gpu.get("current_link_speed")):
        reasons.append("GPU PCIe link speed is not Gen4 16.0 GT/s")
    uuid = gpu.get("uuid")
    if not isinstance(uuid, str) or GPU_UUID_RE.fullmatch(uuid) is None:
        reasons.append("allocated GPU UUID is missing or malformed")
    if software == "byteqc":
        identity = result.get("execution_gpu_identity")
        monitor = result.get("gpu_telemetry")
        if not isinstance(identity, Mapping) or identity.get("verified") is not True:
            reasons.append("ByteQC CUDA execution identity was not verified")
        if not isinstance(monitor, Mapping) or monitor.get("identity_verified") is not True:
            reasons.append("ByteQC GPU monitor identity was not verified")
        elif monitor.get("allocated_gpu_uuid") != uuid:
            reasons.append("ByteQC GPU monitor UUID differs from allocated GPU")
    else:
        identity = result.get("cuda_execution_identity")
        monitor = result.get("gpu_monitor")
        if not isinstance(identity, Mapping):
            reasons.append("GANSU CUDA execution identity is missing")
        else:
            if identity.get("verified_before_gansu_init") is not True:
                reasons.append("GANSU CUDA identity was not verified before initialization")
            if identity.get("verified_before_public_run") is not True:
                reasons.append("GANSU CUDA identity was not verified before the public run")
            if identity.get("validation_errors") not in ([], None):
                reasons.append("GANSU CUDA identity contains validation errors")
        if not isinstance(monitor, Mapping) or monitor.get("identity_verified") is not True:
            reasons.append("GANSU GPU monitor identity was not verified")
        elif monitor.get("expected_uuid") != uuid:
            reasons.append("GANSU GPU monitor UUID differs from allocated GPU")
    return _gate(not reasons, reasons, node=node, gpu_uuid=uuid)


def _timing_gate(software: str, result: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if software == "byteqc":
        seconds = _number(_at(result, "timing_seconds.post_hf_contract"), positive=True)
        if seconds is None:
            reasons.append("ByteQC post-HF contract time is missing, non-finite, or non-positive")
        timing = result.get("timing_definition")
        if not isinstance(timing, Mapping) or not isinstance(timing.get("post_hf_contract"), str):
            reasons.append("ByteQC timing boundary definition is missing")
    else:
        seconds = _number(
            _at(result, "timing_seconds.whole_rhf_ri_rccsd_public_run_wall"), positive=True
        )
        primary = _number(_at(result, "timing_seconds.primary_reportable_seconds"), positive=True)
        if seconds is None or primary is None or not math.isclose(seconds, primary, rel_tol=0, abs_tol=1e-12):
            reasons.append("GANSU public RHF-to-RI-RCCSD timing is missing or internally inconsistent")
        if _at(result, "timing_comparability.primary_public_whole_run_reportable") is not True:
            reasons.append("GANSU public timing was not classified as reportable")
    return _gate(not reasons, reasons, primary_seconds=seconds)


def _convergence_gate(software: str, result: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    convergence = result.get("convergence")
    if not isinstance(convergence, Mapping):
        return _gate(False, ["convergence evidence is missing"])
    if software == "byteqc":
        scf = result.get("scf")
        if not isinstance(scf, Mapping) or scf.get("converged") is not True:
            reasons.append("ByteQC did not demonstrate RHF convergence")
        elif _finite_number(scf.get("energy_eh")) is None:
            reasons.append("ByteQC converged RHF energy is missing or non-finite")
        if convergence.get("converged") is not True:
            reasons.append("ByteQC did not report CCSD convergence")
        conv_tol = _number(convergence.get("conv_tol"), positive=True)
        norm_tol = _number(convergence.get("conv_tol_normt"), positive=True)
        cycles = convergence.get("cycles_observed")
        maximum = convergence.get("max_cycle")
        if conv_tol is None or conv_tol > 1e-8:
            reasons.append("ByteQC energy convergence tolerance is missing or looser than 1e-8")
        if norm_tol is None or norm_tol > 1e-6:
            reasons.append("ByteQC update convergence tolerance is missing or looser than 1e-6")
        if (
            isinstance(cycles, bool)
            or not isinstance(cycles, int)
            or cycles < 1
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or cycles > maximum
        ):
            reasons.append("ByteQC convergence cycle count is invalid")
        history = convergence.get("history")
        last_norm = None
        if isinstance(history, list) and history and isinstance(history[-1], Mapping):
            last_norm = _number(history[-1].get("update_norm"))
        if last_norm is None or norm_tol is None or last_norm > norm_tol:
            reasons.append("ByteQC final observed update norm does not meet its tolerance")
    else:
        for key in (
            "scf_converged_inferred",
            "ccsd_converged_inferred",
            "ccsd_callback_shape_consistent_with_bnative",
        ):
            if convergence.get(key) is not True:
                reasons.append(f"GANSU convergence capability {key} was not demonstrated")
        threshold = _number(convergence.get("ccsd_threshold_eh"), positive=True)
        if threshold is None or threshold > 1e-10:
            reasons.append("GANSU CCSD callback threshold is missing or looser than 1e-10")
        count = convergence.get("ccsd_callback_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            reasons.append("GANSU CCSD callback history is empty")
        reasons.append(
            "GANSU v2 exposes no final observed CCSD residual or update norm; "
            "callback-inferred convergence is diagnostic and cannot authorize WATER4"
        )
    return _gate(not reasons, reasons)


def _resource_gate(software: str, result: Mapping[str, Any]) -> dict[str, Any]:
    if software == "byteqc":
        hbm = _number(_at(result, "gpu_telemetry.peak_memory_used_mib"))
        rss = _number(result.get("host_max_rss_gib"))
        samples = _at(result, "gpu_telemetry.sample_count")
        violations = result.get("resource_budget_violations", [])
    else:
        hbm = _number(_at(result, "resources.peak_hbm_mib"))
        rss = _number(_at(result, "resources.host_max_rss_gib"))
        samples = _at(result, "gpu_monitor.sample_count")
        violations = result.get("resource_budget_violations")
    reasons: list[str] = []
    if hbm is None:
        reasons.append("peak HBM is missing, non-finite, or negative")
    elif hbm > HBM_LIMIT_MIB:
        reasons.append(f"peak HBM {hbm} MiB exceeds {HBM_LIMIT_MIB} MiB")
    if rss is None:
        reasons.append("host peak RSS is missing, non-finite, or negative")
    elif rss > RSS_LIMIT_GIB:
        reasons.append(f"host peak RSS {rss} GiB exceeds {RSS_LIMIT_GIB} GiB")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        reasons.append("GPU resource monitor has no verified sample")
    if violations not in (None, []):
        reasons.append("result reports resource-budget violations")
    return _gate(not reasons, reasons, peak_hbm_mib=hbm, peak_host_rss_gib=rss)


def _oracle_gate(
    software: str,
    result: Mapping[str, Any],
    trusted_case: Mapping[str, Any] | None,
    oracle: Mapping[str, Any] | None,
    energy_tolerance: float | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    tolerance = _number(energy_tolerance, positive=True)
    if tolerance is None:
        reasons.append("an explicit finite positive correlation-energy tolerance is required")
    elif tolerance > MAX_ENERGY_TOLERANCE_EH:
        reasons.append(f"energy tolerance exceeds the {MAX_ENERGY_TOLERANCE_EH} Eh project limit")
    if oracle is None:
        reasons.append("a canonical GPU4PySCF oracle record is required")
        return _gate(False, reasons, tolerance_eh=tolerance)
    if trusted_case is None:
        reasons.append("trusted case contract is unavailable")
    if oracle.get("protocol_schema") != ORACLE_PROTOCOL_SCHEMA:
        reasons.append("oracle protocol schema is unsupported")
    if oracle.get("status") != "completed" or oracle.get("method") != "canonical":
        reasons.append("oracle is not a completed canonical GPU4PySCF result")
    expected_geometry = None
    if trusted_case is not None:
        geometry_errors = _geometry_contract_errors(
            trusted_case.get("geometry_angstrom"), "trusted oracle geometry"
        )
        reasons.extend(geometry_errors)
        if not geometry_errors:
            expected_geometry = _geometry_sha256(trusted_case)
    if oracle.get("case_id") != CASE_ID or oracle.get("geometry_sha256") != expected_geometry:
        reasons.append("oracle case or geometry differs from the trusted WATER2 contract")
    if _at(oracle, "case.basis") != EXPECTED_BASIS:
        reasons.append("oracle basis is not cc-pVTZ")
    expected_dimensions = trusted_case.get("expected_dimensions") if trusted_case else None
    if oracle.get("orbital_dimensions") != expected_dimensions:
        reasons.append("oracle orbital dimensions differ from the trusted contract")
    if oracle.get("cc_converged") is not True:
        reasons.append("oracle CCSD did not converge")
    source = oracle.get("source")
    if not isinstance(source, Mapping):
        reasons.append("oracle immutable source evidence is missing")
    else:
        tree = _at(source, "snapshot.tree_sha256")
        if _sha(tree) is None or source.get("revision") != f"tree-sha256:{tree}":
            reasons.append("oracle source tree SHA-256 is missing or inconsistent")
        for path in (
            "formal_performance_eligible",
            "snapshot.valid",
            "snapshot.valid_at_start",
            "snapshot.valid_at_end",
            "snapshot.stable_during_run",
            "snapshot.read_only_at_start",
            "snapshot.read_only_at_end",
        ):
            if _at(source, path) is not True:
                reasons.append(f"oracle source capability {path} was not verified")
    checkpoint = oracle.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("included_in_post_hf") is not True:
        reasons.append("oracle normal checkpoint is missing from its post-HF boundary")
    elif _sha(checkpoint.get("sha256")) is None:
        reasons.append("oracle checkpoint SHA-256 is missing")
    residual = oracle.get("residual")
    oracle_residual = None
    if not isinstance(residual, Mapping):
        reasons.append("oracle final full-space residual is missing")
    else:
        oracle_residual = _number(residual.get("equation_norm"))
        if (
            residual.get("available") is not True
            or residual.get("is_full_space") is not True
            or residual.get("included_in_post_hf") is not True
            or oracle_residual is None
            or oracle_residual > 1e-6
        ):
            reasons.append("oracle final full-space residual does not meet the 1e-6 contract")
    candidate_energy_path = "energies_eh.e_corr" if software == "byteqc" else "energies_eh.ccsd_correlation"
    candidate_energy = _finite_number(_at(result, candidate_energy_path))
    oracle_energy = _finite_number(oracle.get("e_corr"))
    error = None
    if candidate_energy is None:
        reasons.append("candidate correlation energy is missing or non-finite")
    if oracle_energy is None:
        reasons.append("oracle correlation energy is missing or non-finite")
    if candidate_energy is not None and oracle_energy is not None:
        error = abs(candidate_energy - oracle_energy)
        if tolerance is not None and error > tolerance:
            reasons.append(f"correlation-energy error {error} Eh exceeds {tolerance} Eh")
    return _gate(
        not reasons,
        reasons,
        tolerance_eh=tolerance,
        candidate_e_corr_eh=candidate_energy,
        oracle_e_corr_eh=oracle_energy,
        absolute_error_eh=error,
        oracle_full_space_residual=oracle_residual,
    )


def _formal_comparability(software: str, result: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if software == "gansu":
        reasons.append("GANSU is an I/L2 RI B-native diagnostic, not a V1 formal speedup baseline")
    contract = result.get("comparison_contract") if software == "byteqc" else result.get("method")
    contract = contract if isinstance(contract, Mapping) else {}
    if contract.get("normal_checkpoint") is not True:
        reasons.append("candidate does not provide a normal checkpoint inside the timed boundary")
    if contract.get("separate_final_full_space_residual") is not True:
        reasons.append("candidate does not provide a final full-space residual inside the timed boundary")
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("included_in_post_hf") is not True:
        reasons.append("candidate result has no timed normal checkpoint evidence")
    residual = result.get("residual")
    if not isinstance(residual, Mapping):
        reasons.append("candidate result has no final full-space residual evidence")
    else:
        norm = _number(residual.get("equation_norm"))
        if (
            residual.get("available") is not True
            or residual.get("is_full_space") is not True
            or residual.get("included_in_post_hf") is not True
            or norm is None
            or norm > 1e-6
        ):
            reasons.append("candidate final full-space residual is not a timed <=1e-6 check")
    if software == "byteqc":
        source_tree_sha = _at(result, "provenance.source_tree_sha256")
        if _sha(source_tree_sha) is None:
            reasons.append(
                "ByteQC result lacks an immutable source-tree content SHA-256; "
                "commit and selected-file hashes support L1 only"
            )
        oracle_orbits = _at(result, "canonical_orbitals.artifact_sha256")
        if _sha(oracle_orbits) is None:
            reasons.append("candidate does not prove the exact V1 canonical orbital artifact")
    return {
        "eligible": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "policy": "missing final residual or normal checkpoint always forbids a formal V1 post-HF speedup claim",
    }


def _runner_capability_gate(
    software: str, capability: Mapping[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    supported = capability.get("supported_cases")
    if not isinstance(supported, list) or CASE_ID not in supported:
        reasons.append("runner capability does not include the audited WATER2 case")
    if not isinstance(supported, list) or PROMOTION_CASE_ID not in supported:
        reasons.append("runner capability explicitly refuses WATER4")
    if capability.get("water4_argv_supported") is not True:
        reasons.append("runner capability does not authorize a WATER4 argv")
    if capability.get("software") != software:
        reasons.append("runner capability names a different software package")
    return _gate(not reasons, reasons, supported_cases=supported)


def audit_external_result(
    *,
    software: str,
    result_path: Path,
    scheduler_path: Path,
    cases_path: Path,
    capabilities_path: Path,
    oracle_path: Path | None,
    energy_tolerance: float | None,
) -> dict[str, Any]:
    if software not in SOFTWARE:
        raise ValueError(f"unsupported software {software!r}")
    if energy_tolerance is not None and _finite_number(energy_tolerance) is None:
        raise ValueError("energy tolerance must be a finite JSON number")
    result, result_ev = _evidence(result_path, "external result")
    scheduler, scheduler_ev = _evidence(scheduler_path, "scheduler evidence")
    cases, cases_ev = _evidence(cases_path, "trusted case contract")
    capability, capability_ev = _evidence(capabilities_path, "runner capability")
    oracle: dict[str, Any] | None = None
    oracle_ev: dict[str, Any] | None = None
    if oracle_path is not None:
        oracle, oracle_ev = _evidence(oracle_path, "canonical oracle")

    case_gate, trusted_case = _case_gate(result, cases)
    gates = {
        "scheduler_terminal": _terminal_scheduler_gate(scheduler, result),
        "result_schema_status": _schema_status_gate(software, result),
        "case_contract": case_gate,
        "method_identity": _method_gate(software, result),
        "source_receipt_binding": _source_gate(
            software,
            result,
            scheduler,
            capability,
            cases_ev["file_sha256"],
        ),
        "topology": _topology_gate(software, result),
        "timing": _timing_gate(software, result),
        "convergence": _convergence_gate(software, result),
        "resource_budget": _resource_gate(software, result),
        "numerical_oracle": _oracle_gate(
            software, result, trusted_case, oracle, energy_tolerance
        ),
        "water4_runner_capability": _runner_capability_gate(software, capability),
    }
    water2_gate_names = [
        name for name in gates if name != "water4_runner_capability"
    ]
    water2_blocking = [
        name for name in water2_gate_names if gates[name]["passed"] is not True
    ]
    blocking = [name for name, gate in gates.items() if gate["passed"] is not True]
    reasons = [f"{name}: {reason}" for name in blocking for reason in gates[name]["reasons"]]
    evidence = {
        "result": result_ev,
        "scheduler": scheduler_ev,
        "cases": cases_ev,
        "runner_capabilities": capability_ev,
        "oracle": oracle_ev,
    }
    classification = {
        "acceptance_table": SOFTWARE[software]["acceptance_table"],
        "evidence_level": SOFTWARE[software]["evidence_level"],
        "role": SOFTWARE[software]["role"],
        "formal_speedup_role": SOFTWARE[software]["formal_speedup_role"],
        "formal_v1_post_hf_comparability": _formal_comparability(software, result),
    }
    return {
        "schema": AUDIT_SCHEMA,
        "auditor": {
            "implementation_sha256": _file_sha256(Path(__file__).resolve()),
        },
        "audit_parameters": {
            "software": software,
            "energy_tolerance_eh": energy_tolerance,
        },
        "software": software,
        "case_id": CASE_ID,
        "classification": classification,
        "gates": gates,
        "water2_gate_passed": not water2_blocking,
        "water2_blocking_gates": water2_blocking,
        "w4_eligible": not blocking,
        "w4_blocking_gates": blocking,
        "w4_reasons": reasons,
        "evidence": evidence,
        "input_payload_sha256": {
            key: item["payload_sha256"] if item is not None else None
            for key, item in evidence.items()
        },
    }


def _verify_decision_evidence(decision: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    evidence = decision.get("evidence")
    if not isinstance(evidence, Mapping):
        return ["audit decision has no evidence ledger"]
    for label in ("result", "scheduler", "cases", "runner_capabilities", "oracle"):
        item = evidence.get(label)
        if label == "oracle" and item is None:
            continue
        if not isinstance(item, Mapping):
            reasons.append(f"audit decision lacks {label} evidence")
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str):
            reasons.append(f"audit decision {label} evidence has no path")
            continue
        path = Path(raw_path)
        if not path.is_file():
            reasons.append(f"audit decision {label} evidence file is missing")
            continue
        if _file_sha256(path) != item.get("file_sha256"):
            reasons.append(f"audit decision {label} evidence changed after audit")
    return reasons


def _recompute_bound_decision(
    decision: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Re-run the exact audit recorded by a decision.

    No eligibility boolean, gate, classification, or reason from the serialized
    decision is authoritative.  The planner uses only this recomputed object.
    """

    reasons: list[str] = []
    parameters = decision.get("audit_parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != {
        "software",
        "energy_tolerance_eh",
    }:
        return None, ["audit decision lacks the exact audit parameter set"]
    software = parameters.get("software")
    if software not in SOFTWARE:
        return None, ["recorded audit software parameter is unsupported"]
    tolerance = parameters.get("energy_tolerance_eh")
    if tolerance is not None and _finite_number(tolerance) is None:
        return None, ["recorded audit energy tolerance is not a finite JSON number"]
    evidence = decision.get("evidence")
    if not isinstance(evidence, Mapping):
        return None, ["audit decision has no evidence ledger"]

    def evidence_path(label: str, *, optional: bool = False) -> Path | None:
        item = evidence.get(label)
        if item is None and optional:
            return None
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            reasons.append(f"audit decision lacks the {label} evidence path")
            return None
        return Path(str(item["path"]))

    result_path = evidence_path("result")
    scheduler_path = evidence_path("scheduler")
    cases_path = evidence_path("cases")
    capabilities_path = evidence_path("runner_capabilities")
    oracle_path = evidence_path("oracle", optional=True)
    if any(
        path is None
        for path in (result_path, scheduler_path, cases_path, capabilities_path)
    ):
        return None, reasons
    try:
        recomputed = audit_external_result(
            software=str(software),
            result_path=result_path,
            scheduler_path=scheduler_path,
            cases_path=cases_path,
            capabilities_path=capabilities_path,
            oracle_path=oracle_path,
            energy_tolerance=tolerance,
        )
    except (OSError, ValueError) as exc:
        return None, [*reasons, f"bound audit recomputation failed: {exc}"]
    return recomputed, reasons


def make_water4_plan(
    *,
    decision_path: Path,
    runner_path: Path,
    python_executable: Path,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    decision, decision_ev = _evidence(decision_path, "external audit decision")
    reasons: list[str] = []
    if decision.get("schema") != AUDIT_SCHEMA:
        reasons.append("audit decision schema is unsupported")
    reasons.extend(_verify_decision_evidence(decision))
    recomputed, recompute_reasons = _recompute_bound_decision(decision)
    reasons.extend(recompute_reasons)
    original_digest = _payload_sha256(decision)
    recomputed_digest = _payload_sha256(recomputed) if recomputed is not None else None
    semantic_fields = (
        "software",
        "case_id",
        "classification",
        "gates",
        "water2_gate_passed",
        "water2_blocking_gates",
        "w4_eligible",
        "w4_blocking_gates",
        "w4_reasons",
        "evidence",
        "input_payload_sha256",
        "audit_parameters",
        "auditor",
    )
    semantic_match = bool(
        recomputed is not None
        and all(decision.get(field) == recomputed.get(field) for field in semantic_fields)
    )
    if recomputed is not None and original_digest != recomputed_digest:
        reasons.append("serialized audit decision differs from a fresh bound-audit recomputation")
    if recomputed is not None and not semantic_match:
        reasons.append("serialized audit semantic fields differ from the recomputed audit")
    if recomputed is not None and (
        recomputed.get("water2_gate_passed") is not True
        or recomputed.get("w4_eligible") is not True
    ):
        reasons.append("recomputed WATER2 audit did not pass every WATER4 promotion gate")
        reasons.extend(str(item) for item in recomputed.get("w4_reasons", []))
    software = recomputed.get("software") if recomputed is not None else None
    if software not in SOFTWARE:
        reasons.append("recomputed audit software is unsupported")
    if recomputed is not None and recomputed.get("case_id") != CASE_ID:
        reasons.append("recomputed audit is not for water2-tz")
    if RUN_ID_RE.fullmatch(run_id) is None:
        reasons.append("run id must be a safe 1-80 character identifier")

    runner = runner_path.expanduser().resolve()
    interpreter = python_executable.expanduser().resolve()
    if not runner.is_file():
        reasons.append("runner file is missing")
    if not interpreter.is_file():
        reasons.append("Python executable is missing")

    evidence = (
        recomputed.get("evidence")
        if recomputed is not None and isinstance(recomputed.get("evidence"), Mapping)
        else {}
    )
    capability_ev = evidence.get("runner_capabilities")
    capability = None
    if isinstance(capability_ev, Mapping) and isinstance(capability_ev.get("path"), str):
        try:
            capability = _read_json(Path(str(capability_ev["path"])), "runner capability")
        except (OSError, ValueError) as exc:
            reasons.append(str(exc))
    if capability is None:
        reasons.append("runner capability cannot be reloaded")
    elif runner.is_file() and _file_sha256(runner) != capability.get("runner_sha256"):
        reasons.append("planned runner SHA-256 differs from the audited runner capability")

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "dry_run": True,
        "would_submit": False,
        "submission_argv": None,
        "case_id": PROMOTION_CASE_ID,
        "software": software,
        "run_id": run_id,
        "w4_eligible": not reasons,
        "refusal_reasons": list(dict.fromkeys(reasons)),
        "audit_decision": decision_ev,
        "audit_recomputation": {
            "performed": recomputed is not None,
            "original_payload_sha256": original_digest,
            "recomputed_payload_sha256": recomputed_digest,
            "exact_payload_match": recomputed_digest == original_digest,
            "semantic_fields_match": semantic_match,
        },
        "policy": "manifest-only dry run; this tool has no sbatch execution path",
    }
    if reasons or capability is None or software not in SOFTWARE:
        return plan

    result_ev = evidence["result"]
    result = _read_json(Path(str(result_ev["path"])), "external result")
    output_root = output_dir.expanduser().resolve()
    if software == "byteqc":
        source_root = _at(result, "provenance.source_root")
        cases_path = str(_at(result, "provenance.cases.path"))
        output = output_root / f"water4-tz__byteqc-canonical-four-index__r{run_id}.json"
        scratch = output.with_suffix(".byteqc-buffer.h5")
        argv = [
            str(interpreter),
            str(runner),
            "--cases",
            cases_path,
            "--case",
            PROMOTION_CASE_ID,
            "--basis",
            EXPECTED_BASIS,
            "--repeat",
            run_id,
            "--source-root",
            str(source_root),
            "--eri-representation",
            "canonical-four-index",
            "--scratch-file",
            str(scratch),
            "--output",
            str(output),
        ]
        bound = recomputed["gates"]["source_receipt_binding"]["bound_sha256"]
        environment = {
            "BYTEQC_EXPECTED_HARNESS_SHA256": bound["byteqc_benchmark.py"],
            "BYTEQC_EXPECTED_CASES_SHA256": bound["cases"],
            "BYTEQC_EXPECTED_LIBGINT_SHA256": bound["libgint"],
            "BYTEQC_EXPECTED_LIBGVHF_SHA256": bound["libgvhf"],
        }
    else:
        provenance = result["provenance"]
        files = provenance["files"]
        cases_path = str(files["cases"]["path"])
        output = output_root / f"water4-tz__gansu-ri-bnative__r{run_id}.json"
        argv = [
            str(interpreter),
            str(runner),
            "--cases",
            cases_path,
            "--case",
            PROMOTION_CASE_ID,
            "--basis",
            str(files["basis"]["path"]),
            "--aux-basis",
            str(files["auxiliary_basis"]["path"]),
            "--runtime-root",
            str(provenance["runtime_root"]),
            "--native-library",
            str(files["native_library"]["path"]),
            "--repeat",
            run_id,
            "--output",
            str(output),
        ]
        environment = {
            "GANSU_CCSD_RI_BNATIVE": "1",
            "GANSU_CCSD_RI_LADDER_TILE": "0",
            "GANSU_LIB": str(files["native_library"]["path"]),
        }
        for key, digest in recomputed["gates"]["source_receipt_binding"]["bound_sha256"].items():
            environment[GANSU_DIGEST_ENV[key]] = digest
    plan["job"] = {
        "benchmark_argv": argv,
        "environment": environment,
        "expected_output": str(output),
        "resources": {
            "nodes": 1,
            "tasks": 1,
            "gpus": 1,
            "gpu_model": EXPECTED_GPU,
            "cpu_model": EXPECTED_CPU_FRAGMENT,
            "cpu_affinity": EXPECTED_CPU_AFFINITY,
            "preferred_numa": EXPECTED_NUMA,
            "hbm_limit_mib": HBM_LIMIT_MIB,
            "host_rss_limit_gib": RSS_LIMIT_GIB,
        },
    }
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="audit one completed external WATER2 result")
    audit.add_argument("--software", choices=sorted(SOFTWARE), required=True)
    audit.add_argument("--result", type=Path, required=True)
    audit.add_argument("--scheduler-evidence", type=Path, required=True)
    audit.add_argument("--cases", type=Path, required=True)
    audit.add_argument("--runner-capabilities", type=Path, required=True)
    audit.add_argument("--oracle", type=Path)
    audit.add_argument("--energy-tolerance", type=float)
    audit.add_argument("--output", type=Path)

    plan = sub.add_parser("plan-water4", help="emit a WATER4 manifest; never submit")
    plan.add_argument("--decision", type=Path, required=True)
    plan.add_argument("--runner", type=Path, required=True)
    plan.add_argument("--python", type=Path, default=Path(sys.executable))
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        decision = audit_external_result(
            software=args.software,
            result_path=args.result,
            scheduler_path=args.scheduler_evidence,
            cases_path=args.cases,
            capabilities_path=args.runner_capabilities,
            oracle_path=args.oracle,
            energy_tolerance=args.energy_tolerance,
        )
        if args.output is not None:
            _write_json_once(args.output, decision)
        print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))
        return 0 if decision["w4_eligible"] else 2
    plan = make_water4_plan(
        decision_path=args.decision,
        runner_path=args.runner,
        python_executable=args.python,
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    if args.manifest is not None:
        _write_json_once(args.manifest, plan)
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
    return 0 if plan["w4_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
