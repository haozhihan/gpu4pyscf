#!/usr/bin/env python3
"""Conservative analyzer for the frozen WATER27 CCSD contract.

The analyzer consumes raw benchmark JSON records and writes a new summary.  It
never replaces an existing output and never converts missing accuracy or
hardware evidence into a passing result.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

try:
    from .claim_record import make_claim_record, validate_claim_record
    from .cp_orbital_bundle import shared_bundle_proof
    from .snapshot_manifest import (
        LEGACY_SOURCE_SNAPSHOT_SCHEMA,
        SOURCE_SNAPSHOT_SCHEMA,
        captured_runtime_bundle_evidence_errors,
    )
except ImportError:  # direct ``python analyze_results.py`` execution
    from claim_record import make_claim_record, validate_claim_record
    from cp_orbital_bundle import shared_bundle_proof
    from snapshot_manifest import (
        LEGACY_SOURCE_SNAPSHOT_SCHEMA,
        SOURCE_SNAPSHOT_SCHEMA,
        captured_runtime_bundle_evidence_errors,
    )


FROZEN_BASELINE_SECONDS = 1330.347
FROZEN_BASE_COMMIT = "a89b3ae018d4e82968323ef95645d91adde294aa"
CANDIDATE_DEPLOYMENT_PROFILE = "candidate"
G0_DEPLOYMENT_PROFILE = "g0-canonical-pristine"
TARGET_SECONDS = 133.04
CV_LIMIT = 0.05
ECORR_LIMIT_EH = 1.0e-4
CANONICAL_ENERGY_LIMIT_EH = 1.0e-8
PROJECTED_RESIDUAL_LIMIT = 1.0e-6
CP_LIMIT_KCAL_MOL = 0.02
HBM_LIMIT_GIB = 72.0
RSS_LIMIT_GIB = 110.0
TARGET_CASE_ID = "water8-tz"
HOLDOUT_CASE_ID = "water8s4-tz"
# Result analysis keeps an explicit compatibility allowance for historical v2
# records. Runtime consumers require SOURCE_SNAPSHOT_SCHEMA (v3) separately.
LEGACY_RESULT_SOURCE_SNAPSHOT_SCHEMAS = frozenset({
    LEGACY_SOURCE_SNAPSHOT_SCHEMA,
})
SUPPORTED_RESULT_SOURCE_SNAPSHOT_SCHEMAS = frozenset({
    *LEGACY_RESULT_SOURCE_SNAPSHOT_SCHEMAS,
    SOURCE_SNAPSHOT_SCHEMA,
})
LOW_RANK_RESIDUAL_SCHEMA = "gpu4pyscf.cc.residual.v1"
LOW_RANK_PROJECTED_SPACE = "rr-projected-active-pair"
LOW_RANK_METHODS = frozenset({
    "rr_cd",
    "thc_cd",
    "rr_canonical",
    "thc_canonical",
    "rr",
    "thc_rr",
})
CASE_ROLES = {
    "water2-tz": "sanity",
    "water4-tz": "development",
    TARGET_CASE_ID: "target",
    HOLDOUT_CASE_ID: "holdout",
}


def _first(mapping: dict[str, Any], paths: Iterable[str]) -> Any:
    for path in paths:
        value: Any = mapping
        ok = True
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                ok = False
                break
            value = value[part]
        if ok and value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _record_id(record: dict[str, Any], index: int) -> str:
    value = _first(record, ("record_id", "run_id", "job_id", "pid"))
    return str(value) if value is not None else f"input-{index + 1}"


def load_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load raw JSON records from files or directories without reading summaries."""

    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)
    records: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text())
        candidates = payload if isinstance(payload, list) else payload.get("records", [payload]) if isinstance(payload, dict) else []
        for value in candidates:
            if not isinstance(value, dict):
                continue
            if value.get("schema") == "gpu4pyscf.water8.counterpoise.v1":
                copy = dict(value)
                copy["case_id"] = _first(value, ("case.id",))
                copy["_record_kind"] = "counterpoise"
                copy["_source_path"] = str(path)
                records.append(copy)
                continue
            # Analyzer outputs and claim records are not experiment records.
            if "case_id" not in value or "method" not in value:
                continue
            copy = dict(value)
            copy["_record_kind"] = "benchmark"
            copy["_source_path"] = str(path)
            records.append(copy)
    return records


def _case_id(record: dict[str, Any]) -> str | None:
    return _first(record, ("case_id", "case.id", "plan.case"))


def _method(record: dict[str, Any]) -> str | None:
    return _first(record, ("method", "plan.method"))


def _approximation_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Recover the effective mathematical controls from new or legacy data."""

    explicit = _first(record, ("approximation", "plan.approximation"))
    if isinstance(explicit, dict):
        return explicit
    method = str(_method(record) or "unknown")
    scientific_method = (
        "canonical" if method == "canonical_legacy" else method
    )
    payload: dict[str, Any] = {
        "schema": "gpu4pyscf.cc.approximation.v1",
        "method": scientific_method,
    }
    if method == "fno":
        selection_mode = _first(record, (
            "fno.selection_mode",
            "cluster.energy.metadata.fno.selection_mode",
        ))
        selection_value = _first(record, (
            "fno.selection_value",
            "cluster.energy.metadata.fno.selection_value",
        ))
        if selection_mode is None:
            nvir_act = _first(record, (
                "plan.settings.fno_nvir_act", "protocol.options.fno_nvir_act"
            ))
            pct_occ = _first(record, (
                "plan.settings.fno_pct_occ", "protocol.options.fno_pct_occ"
            ))
            threshold = _first(record, (
                "plan.settings.fno_thresh", "protocol.options.fno_thresh"
            ))
            if nvir_act is not None:
                selection_mode, selection_value = "nvir_act", int(nvir_act)
            elif pct_occ is not None:
                selection_mode, selection_value = "pct_occ", float(pct_occ)
            elif threshold is not None:
                selection_mode, selection_value = (
                    "occupation_threshold", float(threshold)
                )
            else:
                selection_mode, selection_value = "unspecified", None
        payload.update({
            "selection": {"mode": selection_mode, "value": selection_value},
            "correction": "delta-mp2",
        })
    elif method in {
        "rr_cd", "thc_cd", "rr_canonical", "thc_canonical", "rr", "thc_rr"
    }:
        payload.update({
            "eri_backend": _first(record, (
                "plan.settings.eri_backend", "eri_backend"
            )) or (
                "cd" if method in {"rr_cd", "thc_cd"}
                else "canonical" if method.endswith("_canonical")
                else "unspecified"
            ),
            "eri_tol": _first(record, ("plan.settings.eri_tol", "eri_tol")),
            "rr_eig_cutoff": _first(record, (
                "plan.settings.rr_eig_cutoff", "rr_eig_cutoff"
            )),
            "rr_max_rank": _first(record, (
                "plan.settings.rr_max_rank", "rr_max_rank"
            )),
            "precision": _first(record, (
                "plan.settings.precision", "protocol.options.precision", "precision"
            )) or "fp64",
        })
        if method in {"rr_cd", "thc_cd"}:
            payload.update({
                name: _first(record, (
                    f"plan.settings.{name}", f"protocol.options.{name}", name
                ))
                for name in (
                    "direct_scf_tol", "cd_max_rank", "denominator_tolerance",
                    "denominator_max_rank", "rr_solver_tolerance",
                    "rr_solver_maxiter", "rr_dense_fallback_dimension",
                    "rr_ritz_residual_tolerance",
                )
            })
        if method in {"thc_cd", "thc_canonical", "thc_rr"}:
            payload.update({
                name: _first(record, (
                    f"plan.settings.{name}",
                    f"protocol.options.{name}",
                    f"method_metadata.{name}",
                    name,
                ))
                for name in (
                    "thc_fit_tol",
                    "thc_rank",
                    "thc_initial_rank",
                    "thc_max_rank",
                    "thc_rank_growth",
                    "thc_orthogonality_cutoff",
                    "thc_orthogonality_tolerance",
                    "thc_max_iterations",
                    "thc_als_convergence_tolerance",
                    "thc_ridge",
                    "thc_seed",
                    "thc_replacement_validation_atol",
                    "thc_replacement_validation_rtol",
                )
            })
    return payload


def _approximation_signature(record: dict[str, Any]) -> str:
    explicit = _first(record, (
        "approximation_signature",
        "plan.approximation_signature",
        "protocol.approximation_signature",
    ))
    if explicit is not None:
        return str(explicit)
    payload = _approximation_payload(record)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{payload.get('method', 'unknown')}:v1:{digest}"


def _seconds(record: dict[str, Any]) -> float | None:
    return _number(_first(record, ("post_hf_seconds", "timing.post_hf_seconds", "seconds")))


def _geometry_signature(record: dict[str, Any]) -> Any:
    geometry_hash = _first(record, ("geometry_sha256", "case.geometry_sha256", "plan.geometry_sha256"))
    if geometry_hash is not None:
        return geometry_hash
    geometry = _first(record, ("case.geometry_angstrom", "geometry_angstrom"))
    if geometry is None:
        return None
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _scientific_signature(record: dict[str, Any]) -> dict[str, Any] | None:
    case = record.get("case") if isinstance(record.get("case"), dict) else {}
    plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    basis = _first(record, ("basis", "case.basis", "plan.basis"))
    sig = {
        "case_id": _case_id(record),
        "geometry_sha256": _geometry_signature(record),
        "basis": basis,
        "spherical": _first(record, ("spherical", "case.spherical", "plan.spherical")),
        "charge": _first(record, ("charge", "case.charge", "plan.charge")),
        "spin": _first(record, ("spin", "case.spin", "plan.spin")),
        "scf_conv_tol": _first(record, ("scf_conv_tol", "plan.settings.scf_conv_tol")),
        "cc_conv_tol": _first(record, ("cc_conv_tol", "plan.settings.cc_conv_tol")),
        "cc_conv_tol_normt": _first(record, ("cc_conv_tol_normt", "plan.settings.cc_conv_tol_normt")),
        "max_cycle": _first(record, (
            "cc_max_cycle", "max_cycle", "plan.settings.cc_max_cycle",
            "plan.settings.max_cycle",
        )),
        "case_source": case.get("source") or plan.get("source"),
    }
    required = (
        "case_id", "geometry_sha256", "basis", "spherical", "charge", "spin",
        "scf_conv_tol", "cc_conv_tol", "cc_conv_tol_normt", "max_cycle",
    )
    if any(sig[name] is None for name in required):
        return None
    return sig


def _hardware_fingerprint(record: dict[str, Any]) -> str | None:
    explicit = _first(record, ("hardware_fingerprint", "hardware.fingerprint"))
    if explicit is not None:
        return str(explicit)
    gpu = record.get("gpu") if isinstance(record.get("gpu"), dict) else {}
    device = gpu.get("device_0") if isinstance(gpu.get("device_0"), dict) else {}
    topology = record.get("topology")
    affinity = record.get("affinity")
    slurm = record.get("slurm") if isinstance(record.get("slurm"), dict) else {}
    thread_env = record.get("thread_environment") if isinstance(record.get("thread_environment"), dict) else {}
    cpu = _first(record, ("cpu_model", "hardware.cpu_model"))
    parts = {"gpu": device, "topology": topology, "affinity": affinity, "cpus": cpu,
             "numa": _first(record, ("numa_node", "hardware.numa_node", "plan.numa_node")),
             "partition": slurm.get("SLURM_JOB_PARTITION"),
             "threads": slurm.get("SLURM_CPUS_PER_TASK") or thread_env.get("OMP_NUM_THREADS")}
    if not any(value not in (None, {}, [], "") for value in parts.values()):
        return None
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _timing_boundary(record: dict[str, Any]) -> str | None:
    value = _first(record, ("timing_boundary", "timing_definition", "plan.timing_boundary"))
    return str(value) if value is not None else None


def _source_digest(record: dict[str, Any]) -> str | None:
    value = _first(record, (
        "source.tree_sha256",
        "source.tree_sha256_at_start",
    ))
    return str(value) if value is not None else None


def _source_stable(record: dict[str, Any]) -> bool:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    digest = source.get("tree_sha256")
    start = source.get("tree_sha256_at_start")
    end = source.get("tree_sha256_at_end")
    return bool(
        source.get("stable_during_run") is True
        and digest is not None
        and start is not None
        and end is not None
        and start == digest
        and end == digest
    )


def _captured_runtime_evidence_reasons(
    *,
    manifest: dict[str, Any],
    source: dict[str, Any],
    snapshot: dict[str, Any],
) -> list[str]:
    """Validate candidate-v3 runtime evidence without filesystem access."""

    if manifest.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
        return []
    provenance = manifest.get("local_provenance")
    profile = (
        provenance.get("deployment_profile")
        if isinstance(provenance, dict)
        else None
    )
    # G0 snapshots intentionally retain the explicit legacy runtime-root
    # contract. Candidate v3 snapshots must carry source-bound v2 bundle
    # evidence; historical v2 records remain analyzable above this boundary.
    if profile != CANDIDATE_DEPLOYMENT_PROFILE:
        return []
    source_count = _first(source, (
        "files_hashed",
        "files_hashed_at_start",
        "snapshot.files_hashed",
    ))
    snapshot_root = snapshot.get("snapshot_root")
    task_root = snapshot.get("expected_task_root")
    if not isinstance(task_root, str) and isinstance(snapshot_root, str):
        try:
            task_root = str(Path(snapshot_root).parent.parent)
        except (TypeError, ValueError):
            task_root = None
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count <= 0
        or not isinstance(task_root, str)
    ):
        return [
            "candidate v3 runtime evidence lacks captured source count or task root"
        ]
    return captured_runtime_bundle_evidence_errors(
        manifest.get("runtime_binaries"),
        deployment_profile=profile,
        source_tree_sha256=str(source.get("tree_sha256", "")),
        source_file_count=source_count,
        task_root=task_root,
        gint_release_lineage=manifest.get("gint_release_lineage"),
    )


def _timed_snapshot_evidence_reasons(record: dict[str, Any]) -> list[str]:
    """Independently validate captured timed-run snapshot provenance."""

    reasons: list[str] = []
    source = record.get("source")
    if not isinstance(source, dict):
        return ["formal benchmark source evidence is missing"]
    digest = source.get("tree_sha256")
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    ):
        reasons.append("benchmark source digest is not a lowercase SHA-256")
    if source.get("base_revision") != FROZEN_BASE_COMMIT:
        reasons.append("benchmark frozen base revision is missing or mismatched")
    if source.get("formal_performance_eligible") is not True:
        reasons.append("benchmark source is not explicitly formal-performance eligible")
    snapshot = source.get("snapshot")
    if not isinstance(snapshot, dict):
        reasons.append("benchmark immutable snapshot evidence is missing")
        return reasons
    if snapshot.get("schema") != "gpu4pyscf.snapshot-evidence.v1":
        reasons.append("benchmark snapshot evidence schema mismatch")
    if (
        snapshot.get("valid") is not True
        or snapshot.get("valid_at_start") is not True
        or snapshot.get("valid_at_end") is not True
    ):
        reasons.append("benchmark snapshot validation is missing or failed")
    if (
        snapshot.get("read_only") is not True
        or snapshot.get("read_only_at_start") is not True
        or snapshot.get("read_only_at_end") is not True
        or snapshot.get("writable_paths") != []
    ):
        reasons.append("benchmark source snapshot was not read-only throughout the run")
    if snapshot.get("reasons") != [] or snapshot.get("reasons_at_end") != []:
        reasons.append("benchmark snapshot validation recorded unresolved reasons")
    if snapshot.get("stable_during_run") is not True:
        reasons.append("benchmark snapshot stability is missing or failed")
    if (
        snapshot.get("runtime_binaries_valid_at_start") is not True
        or snapshot.get("runtime_binaries_valid_at_end") is not True
    ):
        reasons.append("benchmark runtime-binary validation is missing or failed")
    if snapshot.get("tree_sha256") != digest:
        reasons.append("benchmark snapshot digest does not match source")
    if snapshot.get("tree_sha256_at_end") != digest:
        reasons.append("benchmark ending snapshot digest does not match source")
    manifest_start = snapshot.get("manifest_sha256_at_start")
    if not (
        isinstance(manifest_start, str)
        and len(manifest_start) == 64
        and all(character in "0123456789abcdef" for character in manifest_start)
        and manifest_start == snapshot.get("manifest_sha256_at_end")
    ):
        reasons.append("benchmark snapshot manifest hash is missing, invalid, or changed")
    manifest = snapshot.get("manifest")
    if not isinstance(manifest, dict):
        reasons.append("benchmark source snapshot manifest is missing")
        return reasons
    if manifest.get("schema") not in SUPPORTED_RESULT_SOURCE_SNAPSHOT_SCHEMAS:
        reasons.append("benchmark source snapshot manifest schema mismatch")
    if manifest.get("tree_sha256") != digest:
        reasons.append("benchmark source snapshot manifest digest mismatch")
    if manifest.get("base_revision") != FROZEN_BASE_COMMIT:
        reasons.append("benchmark source snapshot manifest base revision mismatch")
    if manifest.get("immutable") is not True:
        reasons.append("benchmark source snapshot manifest does not declare immutability")
    expected_profile = snapshot.get("expected_deployment_profile")
    require_canonical = snapshot.get("canonical_pristine_required")
    if expected_profile not in {
        CANDIDATE_DEPLOYMENT_PROFILE, G0_DEPLOYMENT_PROFILE
    }:
        reasons.append("benchmark expected deployment profile is missing or invalid")
    if not isinstance(require_canonical, bool):
        reasons.append("benchmark canonical-pristine requirement is missing")
    provenance = manifest.get("local_provenance")
    if not isinstance(provenance, dict):
        reasons.append("benchmark local deployment provenance is missing")
    else:
        profile = provenance.get("deployment_profile")
        head = provenance.get("head_revision")
        tracked_pristine = provenance.get("tracked_pristine")
        untracked_allowed = provenance.get("untracked_paths_allowed")
        canonical_pristine = provenance.get("canonical_pristine")
        expected_canonical = bool(
            profile == G0_DEPLOYMENT_PROFILE
            and head == FROZEN_BASE_COMMIT
            and tracked_pristine is True
            and untracked_allowed is True
        )
        if provenance.get("schema") != "gpu4pyscf.local-provenance.v1":
            reasons.append("benchmark local provenance schema mismatch")
        if profile != expected_profile:
            reasons.append("benchmark deployment profile evidence is inconsistent")
        if provenance.get("frozen_base_revision") != FROZEN_BASE_COMMIT:
            reasons.append("benchmark provenance frozen base revision mismatch")
        if not isinstance(head, str) or len(head) != 40:
            reasons.append("benchmark provenance HEAD is invalid")
        if provenance.get("head_matches_frozen_base") is not (
            head == FROZEN_BASE_COMMIT
        ):
            reasons.append("benchmark provenance frozen-HEAD claim is inconsistent")
        if canonical_pristine is not expected_canonical:
            reasons.append("benchmark canonical-pristine claim is inconsistent")
        if require_canonical and canonical_pristine is not True:
            reasons.append("benchmark canonical-pristine deployment is required")
        allowed = provenance.get("allowed_untracked")
        expected_allowed = {
            "prefixes": ["benchmarks/cc/a100_water8/"],
            "files": ["gpu4pyscf/cc/device_runtime.py"],
        }
        status = provenance.get("git_status")
        if allowed != expected_allowed or not isinstance(status, dict):
            reasons.append("benchmark local provenance policy is missing or invalid")
        else:
            entries = status.get("entries")
            if not isinstance(entries, list) or not all(
                isinstance(entry, str) for entry in entries
            ) or status.get("entry_count") != len(entries):
                reasons.append("benchmark local provenance status is invalid")
            else:
                status_bytes = (
                    "\n".join(entries) + ("\n" if entries else "")
                ).encode("utf-8")
                policy_bytes = json.dumps(
                    {"allowed_untracked": allowed, "status": entries},
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")
                if status.get("sha256") != hashlib.sha256(
                    status_bytes
                ).hexdigest() or status.get(
                    "allowed_paths_status_sha256"
                ) != hashlib.sha256(policy_bytes).hexdigest():
                    reasons.append("benchmark local provenance status hash mismatch")
    runtime = manifest.get("runtime_binaries")
    entries = runtime.get("files") if isinstance(runtime, dict) else None
    runtime_valid = bool(
        isinstance(runtime, dict)
        and runtime.get("complete") is True
        and isinstance(entries, list)
        and entries
    )
    listed: set[str] = set()
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                runtime_valid = False
                continue
            relative = Path(str(entry.get("relative_path", "")))
            sha256 = entry.get("sha256")
            size = entry.get("bytes")
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix != ".so"
                or relative.as_posix() in listed
                or not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                runtime_valid = False
            listed.add(relative.as_posix())
    if not runtime_valid:
        reasons.append("benchmark runtime-binary manifest is missing or invalid")
    reasons.extend(
        f"benchmark {reason}"
        for reason in _captured_runtime_evidence_reasons(
            manifest=manifest,
            source=source,
            snapshot=snapshot,
        )
    )
    repository = source.get("repository")
    snapshot_root = snapshot.get("snapshot_root")
    if manifest.get("source") != repository or snapshot.get("source") != repository:
        reasons.append("benchmark source snapshot path evidence is inconsistent")
    try:
        path_layout_valid = bool(
            isinstance(repository, str)
            and isinstance(snapshot_root, str)
            and Path(repository).name == "source"
            and Path(repository).parent == Path(snapshot_root)
            and Path(snapshot_root).name == digest
            and Path(snapshot_root).parent.name == "snapshots"
            and snapshot.get("manifest_path")
            == str(Path(snapshot_root) / "manifest.json")
        )
    except (TypeError, ValueError):
        path_layout_valid = False
    if not path_layout_valid:
        reasons.append("benchmark source is not in a content-addressed snapshot path")
    return reasons


def compatibility(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare all frozen scientific, hardware and timing identifiers."""

    reasons: list[str] = []
    left = _scientific_signature(baseline)
    right = _scientific_signature(candidate)
    if left is None or right is None:
        reasons.append("missing scientific input signature")
    elif left != right:
        for key in left:
            if left.get(key) != right.get(key):
                reasons.append(f"scientific mismatch: {key}")
    left_hw, right_hw = _hardware_fingerprint(baseline), _hardware_fingerprint(candidate)
    if left_hw is None or right_hw is None:
        reasons.append("missing hardware fingerprint")
    elif left_hw != right_hw:
        reasons.append("hardware mismatch")
    left_t, right_t = _timing_boundary(baseline), _timing_boundary(candidate)
    if left_t is None or right_t is None:
        reasons.append("missing timing boundary")
    elif left_t != right_t:
        reasons.append("timing boundary mismatch")
    left_source, right_source = _source_digest(baseline), _source_digest(candidate)
    if not _source_stable(baseline):
        reasons.append("baseline source stability is missing or failed")
    if not _source_stable(candidate):
        reasons.append("candidate source stability is missing or failed")
    if _timed_snapshot_evidence_reasons(baseline):
        reasons.append("baseline immutable source snapshot evidence is missing or failed")
    if _timed_snapshot_evidence_reasons(candidate):
        reasons.append("candidate immutable source snapshot evidence is missing or failed")
    left_profile = _first(baseline, (
        "source.snapshot.expected_deployment_profile",
    ))
    right_profile = _first(candidate, (
        "source.snapshot.expected_deployment_profile",
    ))
    if left_profile != CANDIDATE_DEPLOYMENT_PROFILE:
        reasons.append("matched canonical record is not from the candidate deployment profile")
    if right_profile != CANDIDATE_DEPLOYMENT_PROFILE:
        reasons.append("candidate record is not from the candidate deployment profile")
    if left_source is None or right_source is None:
        reasons.append("missing source tree digest")
    elif left_source != right_source:
        reasons.append("source tree digest mismatch")
    if _first(baseline, ("performance_eligible",)) is False:
        reasons.append("baseline record is explicitly performance-ineligible")
    if _first(candidate, ("performance_eligible",)) is False:
        reasons.append("candidate record is explicitly performance-ineligible")
    left_memory = _first(baseline, (
        "max_memory_mb", "plan.settings.max_memory_mb"
    ))
    right_memory = _first(candidate, (
        "max_memory_mb", "plan.settings.max_memory_mb"
    ))
    return {"compatible": not reasons, "reasons": reasons, "scientific_signature": left,
            "baseline_hardware": left_hw, "candidate_hardware": right_hw,
            "timing_boundary": left_t, "source_tree_sha256": left_source,
            "memory_provenance": {
                "baseline_max_memory_mb": left_memory,
                "candidate_max_memory_mb": right_memory,
                "match": left_memory is not None and left_memory == right_memory,
            }}


def _ordered(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[int, str, int]:
        value = _first(item, ("run_order", "sequence", "order"))
        if value is not None:
            try:
                return (0, f"{float(value):020.9f}", records.index(item))
            except (TypeError, ValueError):
                pass
        stamp = str(_first(item, ("started_utc", "start_time", "timestamp")) or "")
        return (1, stamp, records.index(item))
    # Python's stable sort preserves directory/input order as the final tie-breaker.
    return sorted(records, key=key)


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [value for value in (_seconds(item) for item in records) if value is not None and value >= 0]
    if not values:
        return {"count": 0, "seconds": [], "median_seconds": None, "min_seconds": None,
                "max_seconds": None, "cv": None, "increase_required": False}
    mean = statistics.mean(values)
    cv = statistics.stdev(values) / mean if len(values) >= 2 and mean else None
    return {"count": len(values), "seconds": values, "median_seconds": statistics.median(values),
            "min_seconds": min(values), "max_seconds": max(values), "cv": cv,
            "increase_required": bool(cv is not None and cv > CV_LIMIT)}


def _value(record: dict[str, Any], paths: Iterable[str]) -> float | None:
    return _number(_first(record, paths))


def _low_rank_projected_residual(record: dict[str, Any]) -> dict[str, Any]:
    """Validate the only residual measurement admissible for RR/THC gates.

    Legacy scalar fields remain useful when inspecting old runs, but their
    numerical value is deliberately separated from the authoritative value.
    This prevents a Jacobi update or an unlabelled norm from silently closing
    the projected equation-residual gate.
    """

    legacy_display_value = _value(record, (
        "accuracy.projected_residual",
        "residual.projected_equation",
        "projected_equation_residual",
        "projected_residual",
    ))
    reasons: list[str] = []
    residual = record.get("residual")
    if not isinstance(residual, dict):
        reasons.append("record.residual is missing or is not an object")
        measurement = None
    else:
        if residual.get("schema") != LOW_RANK_RESIDUAL_SCHEMA:
            reasons.append("record.residual.schema is missing or unsupported")
        if residual.get("acceptance_measurement") != "projected_equation":
            reasons.append(
                "record.residual.acceptance_measurement is not projected_equation"
            )
        measurements = residual.get("measurements")
        if not isinstance(measurements, dict):
            reasons.append("record.residual.measurements is missing or invalid")
            measurement = None
        else:
            measurement = measurements.get("projected_equation")
            if not isinstance(measurement, dict):
                reasons.append(
                    "projected_equation residual measurement is missing or invalid"
                )
                measurement = None

    observed_norm = None
    if measurement is not None:
        if measurement.get("available") is not True:
            reasons.append("projected equation residual is not available")
        if measurement.get("kind") != "equation-residual":
            reasons.append("projected residual kind is not equation-residual")
        if measurement.get("space") != LOW_RANK_PROJECTED_SPACE:
            reasons.append(
                f"projected residual space is not {LOW_RANK_PROJECTED_SPACE}"
            )
        if measurement.get("is_full_space") is not False:
            reasons.append("projected residual must declare is_full_space=false")
        observed_norm = _number(measurement.get("norm"))
        if observed_norm is None or observed_norm < 0.0:
            reasons.append("projected equation residual norm is not finite and non-negative")

    valid = not reasons
    return {
        "valid": valid,
        "value": observed_norm if valid else None,
        "observed_measurement_norm": observed_norm,
        "legacy_display_value": legacy_display_value,
        "reasons": reasons,
    }


def _full_space_residual_diagnostic(record: dict[str, Any]) -> dict[str, Any]:
    """Validate the separately timed RR/THC approximation-bias diagnostic."""

    reasons: list[str] = []
    diagnostic = record.get("full_space_residual_diagnostic")
    if not isinstance(diagnostic, dict):
        return {
            "valid": False,
            "value": None,
            "reasons": ["full_space_residual_diagnostic is missing or invalid"],
        }

    if diagnostic.get("requested") is not True:
        reasons.append("full-space residual diagnostic was not explicitly requested")
    # Current benchmark records completion as a closed enumeration.  An
    # optional boolean is accepted only when it agrees with that enumeration.
    if diagnostic.get("execution_status") != "completed":
        reasons.append("full-space residual diagnostic did not complete")
    if "completed" in diagnostic and diagnostic.get("completed") is not True:
        reasons.append("full-space residual diagnostic completed flag is not true")
    if diagnostic.get("available") is not True:
        reasons.append("full-space residual diagnostic is not available")
    if diagnostic.get("kind") != "equation-residual":
        reasons.append("full-space residual diagnostic kind is not equation-residual")
    if diagnostic.get("space") != "full-active-pair":
        reasons.append("full-space residual diagnostic space is not full-active-pair")
    if diagnostic.get("is_full_space") is not True:
        reasons.append("full-space residual diagnostic must declare is_full_space=true")
    if diagnostic.get("included_in_normal_timed_path") is not False:
        reasons.append(
            "full-space residual diagnostic must be outside the normal timed path"
        )
    if diagnostic.get("included_in_post_hf") is not False:
        reasons.append("full-space residual diagnostic must be outside post_hf_seconds")
    residual_norm = _number(diagnostic.get("residual_norm"))
    if residual_norm is None or residual_norm < 0.0:
        reasons.append("full-space residual norm is not finite and non-negative")

    valid = not reasons
    return {
        "valid": valid,
        "value": residual_norm if valid else None,
        "observed_residual_norm": residual_norm,
        "wall_seconds": _number(diagnostic.get("wall_seconds")),
        "reasons": reasons,
    }


def _accuracy(
    record: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    *,
    cp_error: float | None = None,
    require_projected_residual: bool = True,
    require_cp: bool = True,
    route_table: str = "A",
) -> dict[str, Any]:
    direct_ecorr = _value(record, ("accuracy.ecorr_error_eh", "accuracy.e_corr_error_eh", "accuracy.e_corr_error",
                                   "errors.ecorr_error_eh", "e_corr_error_eh", "delta_e_corr"))
    if route_table == "S" and baseline is not None:
        candidate_energy = _value(record, ("e_corr", "energies.e_corr"))
        baseline_energy = _value(baseline, ("e_corr", "energies.e_corr"))
        ecorr_error = (
            abs(candidate_energy - baseline_energy)
            if candidate_energy is not None and baseline_energy is not None
            else None
        )
        ecorr_source = (
            "paired canonical e_corr" if ecorr_error is not None else None
        )
    elif route_table == "S":
        # Equation-preserving evidence is meaningful only against the exact
        # alternating oracle record.  A candidate-authored direct error is not
        # structurally linked to that oracle and cannot close this gate.
        ecorr_error, ecorr_source = None, None
    elif direct_ecorr is not None:
        ecorr_error = abs(direct_ecorr)
        ecorr_source = "record"
    elif baseline is not None:
        candidate_energy = _value(record, ("e_corr", "energies.e_corr"))
        baseline_energy = _value(baseline, ("e_corr", "energies.e_corr"))
        ecorr_error = abs(candidate_energy - baseline_energy) if candidate_energy is not None and baseline_energy is not None else None
        ecorr_source = "paired canonical e_corr" if ecorr_error is not None else None
    else:
        ecorr_error, ecorr_source = None, None
    energy_limit = (
        CANONICAL_ENERGY_LIMIT_EH if route_table == "S" else ECORR_LIMIT_EH
    )
    checks = {
        "ecorr": {"value": ecorr_error, "threshold": energy_limit, "unit": "Eh",
                   "passed": ecorr_error is not None and ecorr_error <= energy_limit, "source": ecorr_source},
    }
    if route_table == "S":
        direct_etot = _value(record, (
            "accuracy.etot_error_eh", "accuracy.e_tot_error_eh",
            "accuracy.e_total_error_eh", "errors.etot_error_eh",
            "e_tot_error_eh", "delta_e_tot",
        ))
        if baseline is not None:
            candidate_total = _value(record, ("e_tot", "energies.e_tot"))
            baseline_total = _value(baseline, ("e_tot", "energies.e_tot"))
            etot_error = (
                abs(candidate_total - baseline_total)
                if candidate_total is not None and baseline_total is not None
                else None
            )
            etot_source = "paired canonical e_tot" if etot_error is not None else None
        else:
            etot_error, etot_source = None, None
        checks["etot"] = {
            "value": etot_error,
            "threshold": CANONICAL_ENERGY_LIMIT_EH,
            "unit": "Eh",
            "passed": (
                etot_error is not None
                and etot_error <= CANONICAL_ENERGY_LIMIT_EH
            ),
            "source": etot_source,
        }
    if require_projected_residual:
        projected = _low_rank_projected_residual(record)
        checks["projected_residual"] = {
            "value": projected["value"],
            "threshold": PROJECTED_RESIDUAL_LIMIT,
            "unit": "norm",
            "passed": (
                projected["valid"]
                and projected["value"] <= PROJECTED_RESIDUAL_LIMIT
            ),
            "evidence_valid": projected["valid"],
            "evidence_reasons": projected["reasons"],
            "observed_measurement_norm": projected["observed_measurement_norm"],
            "legacy_display_value": projected["legacy_display_value"],
        }
        full_space = _full_space_residual_diagnostic(record)
        checks["full_space_residual_diagnostic"] = {
            "value": full_space["value"],
            "threshold": None,
            "unit": "norm",
            "passed": full_space["valid"],
            "criterion": (
                "finite labelled full-space equation residual, separately timed"
            ),
            "evidence_valid": full_space["valid"],
            "evidence_reasons": full_space["reasons"],
            "observed_residual_norm": full_space.get(
                "observed_residual_norm"
            ),
            "wall_seconds": full_space.get("wall_seconds"),
        }
    if require_cp:
        checks["cp_interaction"] = {
            "value": cp_error,
            "threshold": CP_LIMIT_KCAL_MOL,
            "unit": "kcal/mol",
            "passed": cp_error is not None and cp_error <= CP_LIMIT_KCAL_MOL,
        }
    return {
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
        "missing": [
            name for name, item in checks.items() if item["value"] is None
        ],
        "invalid_evidence": [
            name
            for name, item in checks.items()
            if item.get("evidence_valid") is False
        ],
    }


def _cp_signature(record: dict[str, Any]) -> dict[str, Any] | None:
    signature = {
        "case_id": _case_id(record),
        "geometry_sha256": _first(record, ("case.geometry_sha256",)),
        "basis": _first(record, ("case.basis",)),
        "spherical": _first(record, ("case.spherical",)),
        "scf_conv_tol": _first(record, ("protocol.scf_conv_tol",)),
        "cc_conv_tol": _first(record, ("protocol.cc_conv_tol",)),
        "cc_conv_tol_normt": _first(record, ("protocol.cc_conv_tol_normt",)),
        "cc_max_cycle": _first(record, (
            "protocol.cc_max_cycle", "protocol.options.cc_max_cycle"
        )),
    }
    if any(value is None for value in signature.values()):
        return None
    return signature


def _cp_eligibility(
    record: dict[str, Any], expected_source_digest: str
) -> tuple[bool, list[str]]:
    """Validate CP convergence and immutable-source evidence independently."""

    reasons: list[str] = []
    if record.get("accuracy_eligible") is not True:
        reasons.append("counterpoise record is not explicitly accuracy-eligible")
    if record.get("all_calculations_converged") is not True:
        reasons.append("counterpoise record lacks the all-converged declaration")
    if not _source_stable(record):
        reasons.append("counterpoise source stability is missing or failed")
    digest = _source_digest(record)
    if digest is None:
        reasons.append("counterpoise source digest is missing")
    elif digest != expected_source_digest:
        reasons.append("counterpoise source digest does not match timed candidate")
    for reason in _cp_snapshot_evidence_reasons(record, expected_source_digest):
        if reason not in reasons:
            reasons.append(reason)
    if _first(record, ("cluster.energy.metadata.scf_converged",)) is not True:
        reasons.append("counterpoise cluster SCF is missing or unconverged")
    if _first(record, ("cluster.energy.converged",)) is not True:
        reasons.append("counterpoise cluster CCSD is missing or unconverged")
    fragments = record.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        reasons.append("counterpoise fragment records are missing")
    else:
        for index, fragment in enumerate(fragments):
            if _first(fragment, ("energy.metadata.scf_converged",)) is not True:
                reasons.append(f"counterpoise fragment {index} SCF is missing or unconverged")
            if _first(fragment, ("energy.converged",)) is not True:
                reasons.append(f"counterpoise fragment {index} CCSD is missing or unconverged")
    return not reasons, reasons


def _cp_snapshot_evidence_reasons(
    record: dict[str, Any], expected_source_digest: str
) -> list[str]:
    """Independently validate captured formal-CP snapshot provenance."""

    reasons: list[str] = []
    if not (
        isinstance(expected_source_digest, str)
        and len(expected_source_digest) == 64
        and all(
            character in "0123456789abcdef"
            for character in expected_source_digest
        )
    ):
        reasons.append("timed candidate source digest is not a lowercase SHA-256")
    source = record.get("source")
    if not isinstance(source, dict):
        return ["counterpoise formal source evidence is missing"]
    if source.get("formal_counterpoise_eligible") is not True:
        reasons.append("counterpoise source is not explicitly formal-CP eligible")
    if source.get("base_revision") != FROZEN_BASE_COMMIT:
        reasons.append("counterpoise frozen base revision is missing or mismatched")
    snapshot = source.get("snapshot")
    if not isinstance(snapshot, dict):
        reasons.append("counterpoise immutable snapshot evidence is missing")
        return reasons
    if snapshot.get("schema") != "gpu4pyscf.snapshot-evidence.v1":
        reasons.append("counterpoise snapshot evidence schema mismatch")
    expected_profile = snapshot.get("expected_deployment_profile")
    if expected_profile not in {
        CANDIDATE_DEPLOYMENT_PROFILE, G0_DEPLOYMENT_PROFILE
    }:
        reasons.append(
            "counterpoise snapshot expected deployment profile is missing or unknown"
        )
    if (
        snapshot.get("valid") is not True
        or snapshot.get("valid_at_start") is not True
        or snapshot.get("valid_at_end") is not True
    ):
        reasons.append("counterpoise snapshot validation is missing or failed")
    if (
        snapshot.get("read_only") is not True
        or snapshot.get("read_only_at_start") is not True
        or snapshot.get("read_only_at_end") is not True
        or snapshot.get("writable_paths") != []
    ):
        reasons.append("counterpoise source snapshot was not read-only throughout the run")
    if snapshot.get("reasons") != [] or snapshot.get("reasons_at_end") != []:
        reasons.append("counterpoise snapshot validation recorded unresolved reasons")
    if snapshot.get("stable_during_run") is not True:
        reasons.append("counterpoise snapshot manifest stability is missing or failed")
    if (
        snapshot.get("runtime_binaries_valid_at_start") is not True
        or snapshot.get("runtime_binaries_valid_at_end") is not True
    ):
        reasons.append("counterpoise runtime-binary validation is missing or failed")
    if snapshot.get("tree_sha256") != expected_source_digest:
        reasons.append("counterpoise snapshot digest does not match timed candidate")
    if snapshot.get("tree_sha256_at_end") != expected_source_digest:
        reasons.append("counterpoise ending snapshot digest does not match timed candidate")
    manifest_start = snapshot.get("manifest_sha256_at_start")
    manifest_end = snapshot.get("manifest_sha256_at_end")
    if not (
        isinstance(manifest_start, str)
        and len(manifest_start) == 64
        and all(character in "0123456789abcdef" for character in manifest_start)
        and manifest_start == manifest_end
    ):
        reasons.append("counterpoise snapshot manifest content hash is missing, invalid, or changed")
    manifest = snapshot.get("manifest")
    if not isinstance(manifest, dict):
        reasons.append("counterpoise source snapshot manifest is missing")
        return reasons
    if manifest.get("schema") not in SUPPORTED_RESULT_SOURCE_SNAPSHOT_SCHEMAS:
        reasons.append("counterpoise source snapshot manifest schema mismatch")
    if manifest.get("tree_sha256") != expected_source_digest:
        reasons.append("counterpoise source snapshot manifest digest mismatch")
    if manifest.get("base_revision") != FROZEN_BASE_COMMIT:
        reasons.append("counterpoise source snapshot manifest base revision mismatch")
    if manifest.get("immutable") is not True:
        reasons.append("counterpoise source snapshot manifest does not declare immutability")
    provenance = manifest.get("local_provenance")
    if not isinstance(provenance, dict):
        reasons.append("counterpoise snapshot local provenance is missing")
    else:
        actual_profile = provenance.get("deployment_profile")
        if actual_profile != expected_profile:
            reasons.append(
                "counterpoise snapshot deployment profile does not match manifest provenance"
            )
        canonical_required = snapshot.get("canonical_pristine_required")
        expected_canonical = expected_profile == G0_DEPLOYMENT_PROFILE
        if canonical_required is not expected_canonical:
            reasons.append(
                "counterpoise canonical-pristine requirement is inconsistent"
            )
        if (
            expected_profile == G0_DEPLOYMENT_PROFILE
            and provenance.get("canonical_pristine") is not True
        ):
            reasons.append(
                "counterpoise G0 snapshot is not canonically pristine"
            )
    runtime = manifest.get("runtime_binaries")
    runtime_valid = bool(
        isinstance(runtime, dict) and runtime.get("complete") is True
    )
    entries = runtime.get("files") if isinstance(runtime, dict) else None
    if not isinstance(entries, list) or not entries:
        runtime_valid = False
    else:
        listed: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                runtime_valid = False
                continue
            relative = Path(str(entry.get("relative_path", "")))
            sha256 = entry.get("sha256")
            size = entry.get("bytes")
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix != ".so"
                or relative.as_posix() in listed
                or not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                runtime_valid = False
            listed.add(relative.as_posix())
    if not runtime_valid:
        reasons.append("counterpoise runtime-binary manifest is missing or invalid")
    reasons.extend(
        f"counterpoise {reason}"
        for reason in _captured_runtime_evidence_reasons(
            manifest=manifest,
            source=source,
            snapshot=snapshot,
        )
    )
    repository = source.get("repository")
    snapshot_root = snapshot.get("snapshot_root")
    if manifest.get("source") != repository or snapshot.get("source") != repository:
        reasons.append("counterpoise source snapshot path evidence is inconsistent")
    try:
        expected_task_root = snapshot.get("expected_task_root")
        path_layout_valid = bool(
            isinstance(repository, str)
            and isinstance(snapshot_root, str)
            and isinstance(expected_task_root, str)
            and Path(repository).name == "source"
            and Path(repository).parent == Path(snapshot_root)
            and Path(snapshot_root).name == expected_source_digest
            and Path(snapshot_root).parent
            == Path(expected_task_root) / "snapshots"
            and snapshot.get("manifest_path")
            == str(Path(snapshot_root) / "manifest.json")
        )
    except (TypeError, ValueError):
        path_layout_valid = False
    if not path_layout_valid:
        reasons.append("counterpoise source is not in a content-addressed snapshot path")
    return reasons


def _counterpoise_error(
    records: list[dict[str, Any]],
    case_id: str,
    baseline_method: str,
    candidate_method: str,
    candidate_signature: str,
    source_digest: str,
) -> tuple[float | None, list[str]]:
    baseline_signature_payload = {
        "schema": "gpu4pyscf.cc.approximation.v1",
        "method": baseline_method,
    }
    baseline_digest = hashlib.sha256(json.dumps(
        baseline_signature_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    baseline_signature = f"{baseline_method}:v1:{baseline_digest}"
    baseline_all = [
        item for item in records
        if _case_id(item) == case_id and _method(item) == baseline_method
        and _approximation_signature(item) == baseline_signature
    ]
    candidate_all = [
        item for item in records
        if _case_id(item) == case_id and _method(item) == candidate_method
        and _approximation_signature(item) == candidate_signature
    ]
    baseline = [
        item for item in baseline_all if _source_digest(item) == source_digest
    ]
    candidate = [
        item for item in candidate_all if _source_digest(item) == source_digest
    ]
    if not baseline or not candidate:
        if baseline_all and candidate_all:
            return None, [
                "canonical/candidate counterpoise source digest does not match timed candidate"
            ]
        return None, ["missing canonical or candidate counterpoise record"]
    eligibility = [
        _cp_eligibility(item, source_digest) for item in baseline + candidate
    ]
    if not all(passed for passed, _ in eligibility):
        reasons: list[str] = []
        for _, item_reasons in eligibility:
            for reason in item_reasons:
                if reason not in reasons:
                    reasons.append(reason)
        return None, reasons
    signatures = [_cp_signature(item) for item in baseline + candidate]
    if any(item is None for item in signatures):
        return None, ["counterpoise record is missing frozen protocol fields"]
    if any(item != signatures[0] for item in signatures[1:]):
        return None, ["counterpoise scientific protocol mismatch"]
    # Every CP value entering either median must come from one exact orbital
    # bundle.  Per-record bundle completeness is insufficient: independently
    # produced canonical and candidate bundles can both be internally valid
    # while defining different MO gauges.  Requiring every selected record to
    # prove identity with the first baseline record also prevents silently
    # cherry-picking a compatible subset from mixed inputs.
    reference = baseline[0]
    try:
        proofs = [
            shared_bundle_proof(reference, item)
            for item in baseline + candidate
        ]
    except (KeyError, TypeError, ValueError) as exc:
        return None, [
            "counterpoise canonical/candidate shared orbital bundle proof "
            f"failed: {exc}"
        ]
    match_keys = {proof.get("record_match_key") for proof in proofs}
    if len(match_keys) != 1 or None in match_keys:
        return None, [
            "counterpoise canonical/candidate records do not use one exact "
            "orbital bundle"
        ]
    baseline_values = [
        _value(item, ("interaction_energy_cp_kcal_mol",)) for item in baseline
    ]
    candidate_values = [
        _value(item, ("interaction_energy_cp_kcal_mol",)) for item in candidate
    ]
    if any(value is None for value in baseline_values + candidate_values):
        return None, ["counterpoise interaction energy is missing"]
    error = abs(statistics.median(candidate_values) - statistics.median(baseline_values))
    return float(error), []


def _counterpoise_memory_provenance(
    records: list[dict[str, Any]],
    case_id: str,
    baseline_method: str,
    candidate_method: str,
    candidate_signature: str,
    source_digest: str,
) -> dict[str, Any]:
    """Compare CP memory limits as provenance without making them a gate."""

    baseline_payload = {
        "schema": "gpu4pyscf.cc.approximation.v1",
        "method": baseline_method,
    }
    digest = hashlib.sha256(json.dumps(
        baseline_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    baseline_signature = f"{baseline_method}:v1:{digest}"

    def values(method: str, signature: str) -> list[int | float]:
        result: set[int | float] = set()
        for item in records:
            if (
                _case_id(item) != case_id
                or _method(item) != method
                or _approximation_signature(item) != signature
                or _source_digest(item) != source_digest
            ):
                continue
            value = _number(_first(item, (
                "protocol.max_memory_mb", "protocol.options.max_memory_mb",
                "max_memory_mb",
            )))
            if value is not None:
                result.add(int(value) if value.is_integer() else value)
        return sorted(result)

    baseline_values = values(baseline_method, baseline_signature)
    candidate_values = values(candidate_method, candidate_signature)
    return {
        "baseline_max_memory_mb": baseline_values,
        "candidate_max_memory_mb": candidate_values,
        "match": bool(
            baseline_values
            and candidate_values
            and baseline_values == candidate_values
        ),
        "acceptance_gate": False,
    }


def _resources(record: dict[str, Any]) -> dict[str, Any]:
    hbm_mib = _value(record, ("hbm.peak_process_MiB", "hbm.peak_process_mib"))
    hbm = _value(record, ("peak_hbm_GiB", "peak_hbm_gib"))
    if hbm is None and hbm_mib is not None:
        hbm = hbm_mib / 1024.0
    rss = _value(record, ("peak_host_RSS_GiB", "peak_host_rss_gib", "host_rss_gib"))
    hardware = _fixed_hardware(record)
    checkpoint_passed = _first(
        record, ("checkpoint.included_in_post_hf",)
    ) is True
    return {"hbm_gib": hbm, "hbm_limit_gib": HBM_LIMIT_GIB, "hbm_passed": hbm is not None and hbm <= HBM_LIMIT_GIB,
            "rss_gib": rss, "rss_limit_gib": RSS_LIMIT_GIB, "rss_passed": rss is not None and rss <= RSS_LIMIT_GIB,
            "hardware": hardware,
            "checkpoint_included_in_post_hf": checkpoint_passed,
            "passed": hbm is not None and hbm <= HBM_LIMIT_GIB and rss is not None and rss <= RSS_LIMIT_GIB and hardware["passed"] and checkpoint_passed}


def _fixed_hardware(record: dict[str, Any]) -> dict[str, Any]:
    gpu = record.get("gpu") if isinstance(record.get("gpu"), dict) else {}
    device = gpu.get("device_0") if isinstance(gpu.get("device_0"), dict) else {}
    name = str(device.get("name", ""))
    total_bytes = _number(device.get("totalGlobalMem"))
    pcie = record.get("pcie") if isinstance(record.get("pcie"), dict) else {}
    visible = pcie.get("visible_gpus") if isinstance(pcie.get("visible_gpus"), list) else []
    pcie_device = visible[0] if len(visible) == 1 and isinstance(visible[0], dict) else {}
    affinity = record.get("affinity")
    allowed_affinities = (
        list(range(24, 32)),
        list(range(24, 32)) + list(range(88, 96)),
    )
    cpu_model = str(_first(record, ("cpu_model", "hardware.cpu_model")) or "")
    numa = str(_first(record, ("thread_environment.GPU4PYSCF_NUMA", "numa_node")) or "")
    pcie_gen = pcie_device.get("pcie_gen_max")
    if pcie_gen is None:
        try:
            rate = float(str(pcie_device.get("max_link_speed", "")).split()[0])
        except (TypeError, ValueError, IndexError):
            rate = None
        pcie_gen = {
            2.5: "1", 5.0: "2", 8.0: "3", 16.0: "4",
            32.0: "5", 64.0: "6",
        }.get(rate)
    pcie_width = pcie_device.get(
        "pcie_width_max", pcie_device.get("max_link_width")
    )
    checks = {
        "single_visible_gpu": gpu.get("device_count") == 1 and len(visible) == 1,
        "gpu_model": "A100-SXM4-80GB" in name,
        "gpu_memory": total_bytes is not None and 79 * 1024**3 <= total_bytes <= 82 * 1024**3,
        "cpu_model": "AMD EPYC 7513" in cpu_model,
        "cpu_affinity": affinity in allowed_affinities,
        "numa_node": numa == "3",
        "pcie_gen": str(pcie_gen or "") == "4",
        "pcie_width": str(pcie_width or "") == "16",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "gpu_name": name or None,
            "gpu_total_bytes": total_bytes,
            "cpu_model": cpu_model or None,
            "affinity": affinity,
            "numa": numa or None,
            "pcie": pcie_device or None,
        },
    }


def _route_table(record: dict[str, Any]) -> str:
    method = (_method(record) or "").lower()
    cls = _first(record, ("method_class", "plan.method_class", "method_metadata.method_class"))
    if cls == "equation-preserving" or method in {
        "canonical", "canonical_legacy"
    }:
        return "S"
    if method in {"df", "ri", "cd", "cholesky"} or _first(record, ("eri_backend", "plan.settings.eri_backend")) in {"df", "ri", "cd"} and method not in {"fno", "rr_cd", "thc_cd", "rr_canonical", "thc_canonical"}:
        return "I"
    return "A"


def _eri_representation(record: dict[str, Any]) -> str:
    value = _first(record, ("eri_backend", "plan.settings.eri_backend", "integral_representation", "plan.integral_representation"))
    if value is not None:
        return str(value).lower()
    return "canonical" if (_method(record) or "").lower() in {
        "canonical", "canonical_legacy"
    } else "unspecified"


def _successful(record: dict[str, Any]) -> bool:
    return (
        _first(record, ("status",)) == "completed"
        and _first(record, ("cc_converged", "converged")) is True
        and _first(record, ("performance_eligible",)) is not False
        and _source_stable(record)
        and not _timed_snapshot_evidence_reasons(record)
    )


def _pair_records(
    records: list[dict[str, Any]],
    baseline_method: str,
    candidate_method: str,
    candidate_signature: str,
    candidate_source_digest: str,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    # Keep other signatures in the ordered stream so a B/C(cutoff-2) attempt
    # cannot be treated as an alternating B/C(cutoff-1) observation after
    # filtering.  Only truly adjacent records for this signature form a pair.
    selected = [
        item for item in _ordered(records)
        if _method(item) in {baseline_method, candidate_method}
    ]
    pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    # Consume pairs rather than counting overlapping transitions.  In the
    # sequence B,C,B,C there are two A/B observations, not three transitions.
    index = 0
    while index + 1 < len(selected):
        left, right = selected[index], selected[index + 1]
        if {_method(left), _method(right)} != {baseline_method, candidate_method}:
            index += 1
            continue
        baseline = left if _method(left) == baseline_method else right
        candidate = right if baseline is left else left
        if _approximation_signature(candidate) != candidate_signature:
            index += 1
            continue
        if _source_digest(candidate) != candidate_source_digest:
            index += 1
            continue
        comp = compatibility(baseline, candidate)
        if comp["compatible"]:
            pairs.append((baseline, candidate, comp))
        index += 2
    return pairs


def _compatibility_reasons(
    records: list[dict[str, Any]],
    baseline_method: str,
    candidate_method: str,
    candidate_signature: str,
    candidate_source_digest: str,
) -> list[str]:
    """Return mismatches observed in adjacent baseline/candidate attempts."""

    selected = [item for item in _ordered(records) if _method(item) in {baseline_method, candidate_method}]
    reasons: list[str] = []
    for left, right in zip(selected, selected[1:]):
        if {_method(left), _method(right)} != {baseline_method, candidate_method}:
            continue
        baseline = left if _method(left) == baseline_method else right
        candidate = right if baseline is left else left
        if _approximation_signature(candidate) != candidate_signature:
            continue
        if _source_digest(candidate) != candidate_source_digest:
            continue
        for reason in compatibility(baseline, candidate)["reasons"]:
            if reason not in reasons:
                reasons.append(reason)
    return reasons


def _run_ref(record: dict[str, Any], index: int) -> dict[str, Any]:
    return {"seconds": float(_seconds(record) or 0.0), "hardware_fingerprint": _hardware_fingerprint(record) or "unknown",
            "software_revision": str(_first(record, (
                "software_revision", "git_commit", "source.revision",
                "source.git_revision", "plan.software_revision",
            )) or "unknown"),
            "raw_record_ids": [_record_id(record, index)]}


def analyze_records(records: list[dict[str, Any]], *, baseline_method: str = "canonical",
                    candidate_methods: Iterable[str] | None = None,
                    frozen_baseline_seconds: float = FROZEN_BASELINE_SECONDS) -> dict[str, Any]:
    """Analyze all cases and methods; return a JSON-serializable summary."""

    cp_records = [
        item for item in records
        if item.get("_record_kind") == "counterpoise"
        or _value(item, ("interaction_energy_cp_kcal_mol",)) is not None
    ]
    performance_records = [item for item in records if _seconds(item) is not None]
    if candidate_methods is None:
        candidate_methods = sorted({
            _method(item) for item in performance_records
            if _method(item) and _method(item) != baseline_method
        })
    candidates = [str(item) for item in candidate_methods]
    cases = sorted({_case_id(item) for item in performance_records if _case_id(item)})
    groups: list[dict[str, Any]] = []
    tables: dict[str, list[dict[str, Any]]] = {"S": [], "I": [], "A": []}
    claim_records: list[dict[str, Any]] = []
    for case in cases:
        case_records = [item for item in performance_records if _case_id(item) == case]
        baseline_records = [item for item in case_records if _method(item) == baseline_method]
        baseline_stats = _summary(baseline_records)
        candidate_groups = sorted({
            (
                candidate_method,
                _approximation_signature(item),
                _source_digest(item) or "<missing-source-digest>",
            )
            for candidate_method in candidates
            for item in case_records
            if _method(item) == candidate_method
        })
        for candidate_method, candidate_signature, candidate_source_digest in candidate_groups:
            candidate_records = [
                item for item in case_records
                if _method(item) == candidate_method
                and _approximation_signature(item) == candidate_signature
                and (_source_digest(item) or "<missing-source-digest>") == candidate_source_digest
            ]
            if not candidate_records:
                continue
            pairs = _pair_records(
                case_records,
                baseline_method,
                candidate_method,
                candidate_signature,
                candidate_source_digest,
            )
            pair_count = len(pairs)
            paired_baselines = [base for base, _, _ in pairs]
            paired_candidates = [candidate for _, candidate, _ in pairs]
            acceptance_candidates = paired_candidates or candidate_records
            acceptance_baselines = paired_baselines or baseline_records
            route = _route_table(candidate_records[0])
            cp_error, cp_limitations = _counterpoise_error(
                cp_records,
                case,
                baseline_method,
                candidate_method,
                candidate_signature,
                candidate_source_digest,
            )
            cp_memory_provenance = _counterpoise_memory_provenance(
                cp_records,
                case,
                baseline_method,
                candidate_method,
                candidate_signature,
                candidate_source_digest,
            )
            require_projected = candidate_method in LOW_RANK_METHODS
            require_cp = route in {"I", "A"}
            displayed_cp_error = cp_error
            if not require_cp:
                cp_limitations = []
            pair_acc = [
                _accuracy(
                    cand,
                    base,
                    cp_error=cp_error,
                    require_projected_residual=require_projected,
                    require_cp=require_cp,
                    route_table=route,
                )
                for base, cand, _ in pairs
            ]
            # If no alternating pair exists, do not silently use a same-file or
            # unpaired energy as an accuracy oracle.
            per_record_acc = [
                _accuracy(
                    item,
                    cp_error=cp_error,
                    require_projected_residual=require_projected,
                    require_cp=require_cp,
                    route_table=route,
                )
                for item in candidate_records
            ]
            acc_records = pair_acc if pair_acc else per_record_acc
            accuracy_passed = bool(acc_records) and all(item["passed"] for item in acc_records)
            resources = [_resources(item) for item in acceptance_candidates]
            resources_passed = bool(resources) and all(item["passed"] for item in resources)
            converged = bool(acceptance_candidates) and all(
                _successful(item) for item in acceptance_candidates
            )
            candidate_stats = _summary(acceptance_candidates)
            baseline_stats = _summary(acceptance_baselines)
            matched_baseline_values = [_seconds(base) for base, _, _ in pairs if _seconds(base) is not None]
            matched_candidate_values = [_seconds(cand) for _, cand, _ in pairs if _seconds(cand) is not None]
            matched_speedup = (statistics.median(matched_baseline_values) / statistics.median(matched_candidate_values)
                               if matched_baseline_values and matched_candidate_values and statistics.median(matched_candidate_values) > 0 else None)
            frozen_speedup = (frozen_baseline_seconds / candidate_stats["median_seconds"]
                              if candidate_stats["median_seconds"] and candidate_stats["median_seconds"] > 0 else None)
            required_pair_count = 1 if case == HOLDOUT_CASE_ID else 3
            timing_sufficient = pair_count >= required_pair_count
            absolute_target_passed = bool(
                candidate_stats["median_seconds"] is not None
                and candidate_stats["median_seconds"] <= TARGET_SECONDS
            ) if case == TARGET_CASE_ID else True
            matched_10x_passed = bool(
                matched_speedup is not None and matched_speedup >= 10.0
            ) if case == TARGET_CASE_ID else True
            speed_passed = absolute_target_passed and matched_10x_passed
            validation_status = str(_first(candidate_records[0], (
                "validation_status",
                "plan.validation_status",
                "method_metadata.validation_status",
            )) or "")
            acceptance_label = str(_first(candidate_records[0], (
                "acceptance_table", "plan.acceptance_table"
            )) or "")
            source_performance_eligible = all(
                _first(item, ("performance_eligible",)) is not False
                and _source_stable(item)
                and not _timed_snapshot_evidence_reasons(item)
                for item in acceptance_candidates
            )
            method_implementation_eligible = (
                candidate_method not in {
                    "rr_canonical", "thc_cd", "thc_canonical"
                }
                and
                acceptance_label != "not_yet_accepted"
                and validation_status not in {
                    "dense-projected-validation",
                    "dense-two-level-validation-surrogate",
                }
            )
            production_eligible = (
                source_performance_eligible and method_implementation_eligible
            )
            development_gate = (
                converged
                and accuracy_passed
                and resources_passed
                and not candidate_stats["increase_required"]
                and production_eligible
                and (
                    case != "water4-tz"
                    or matched_speedup is not None and matched_speedup > 1.0
                )
            )
            gate_passed = (
                timing_sufficient
                and development_gate
                and speed_passed
            )
            compatibility_reasons = _compatibility_reasons(
                case_records,
                baseline_method,
                candidate_method,
                candidate_signature,
                candidate_source_digest,
            )
            row = {
                "case_id": case, "case_role": CASE_ROLES.get(case, "unknown"), "method": candidate_method,
                "approximation_signature": candidate_signature,
                "source_tree_sha256": candidate_source_digest,
                "approximation": _approximation_payload(candidate_records[0]),
                "route_table": route, "baseline": baseline_stats, "candidate": candidate_stats,
                "alternating_pairs": pair_count, "timing_evidence_sufficient": timing_sufficient,
                "required_pair_count": required_pair_count,
                "increase_required": candidate_stats["increase_required"], "speedup_frozen": frozen_speedup,
                "speedup_matched_canonical": matched_speedup, "target_speed_passed": speed_passed,
                "absolute_target_passed": absolute_target_passed,
                "matched_10x_passed": matched_10x_passed,
                "converged": converged, "accuracy": acc_records, "accuracy_passed": accuracy_passed,
                "resources": resources, "resources_passed": resources_passed, "accepted": gate_passed,
                "development_gate_passed": development_gate,
                "production_eligible": production_eligible,
                "validation_status": validation_status or None,
                "counterpoise_error_kcal_mol": displayed_cp_error,
                "memory_provenance": {
                    "timed_pairs": [
                        comp["memory_provenance"] for _, _, comp in pairs
                    ],
                    "counterpoise": cp_memory_provenance,
                },
                "compatibility_reasons": compatibility_reasons,
                "limitations": (
                    cp_limitations
                    + ([] if source_performance_eligible else [
                        "one or more timed records are explicitly performance-ineligible"
                    ])
                    + ([] if method_implementation_eligible else [
                        "validation-only implementation is ineligible for a performance claim"
                    ])
                    + ([] if gate_passed else ["one or more required gates are incomplete or failed"])
                ),
            }
            groups.append(row)
            tables[route].append(row)
            if pairs and candidate_stats["median_seconds"] is not None:
                obs: list[dict[str, Any]] = []
                # A claim record is emitted only when numeric accuracy evidence exists.
                for name, check in acc_records[0]["checks"].items():
                    if (
                        check["value"] is not None
                        and _number(check.get("threshold")) is not None
                    ):
                        obs.append({"name": name, "error": float(check["value"]), "threshold": float(check["threshold"]),
                                    "unit": check["unit"], "passed": bool(all(item["checks"].get(name, {}).get("passed", False) for item in acc_records))})
                if obs:
                    base0, cand0, comp0 = pairs[0]
                    cls = _first(cand0, ("method_class", "plan.method_class", "method_metadata.method_class")) or ("equation-preserving" if route == "S" else "controlled-approximation")
                    acceptance = route if gate_passed else "not_yet_accepted"
                    claim = make_claim_record(
                        claim_id=f"{case}:{candidate_method}:{candidate_signature}:{candidate_source_digest}", method_class=str(cls), acceptance_table=acceptance,
                        timing_scope="post_hf", same_scientific_input=comp0["compatible"], same_timing_boundary=comp0["compatible"],
                        baseline=_run_ref(base0, performance_records.index(base0)), candidate=_run_ref(cand0, performance_records.index(cand0)),
                        speedup=float(matched_speedup or frozen_speedup or 1.0),
                        formula="median(baseline post_hf seconds) / median(candidate post_hf seconds)",
                        contract_id="water8-ccsd-10x-v1", accuracy_observables=obs, accuracy_passed=accuracy_passed,
                        same_eri_representation=_eri_representation(base0) == _eri_representation(cand0),
                        approximation_parameters={
                            "method": candidate_method,
                            "approximation_signature": candidate_signature,
                            "source_tree_sha256": candidate_source_digest,
                            "effective_controls": _approximation_payload(cand0),
                        },
                        limitations=row["limitations"],
                    )
                    claim_records.append({"record": claim, "validation": validate_claim_record(claim)})
    # A target result cannot bypass the two staged gates or the independent
    # H2O8s4 holdout.  The holdout carries no speed target, but it must pass the
    # same numerical and resource contract with the selected method.
    for row in groups:
        if row["case_id"] != TARGET_CASE_ID:
            continue
        prior = {
            case_id: next((
                candidate for candidate in groups
                if candidate["case_id"] == case_id
                and candidate["method"] == row["method"]
                and candidate["approximation_signature"] == row["approximation_signature"]
                and candidate["source_tree_sha256"] == row["source_tree_sha256"]
            ), None)
            for case_id in ("water2-tz", "water4-tz")
        }
        row["prior_gates"] = {
            case_id: bool(value and value["development_gate_passed"])
            for case_id, value in prior.items()
        }
        row["prior_gates_passed"] = all(row["prior_gates"].values())
        if not row["prior_gates_passed"]:
            row["accepted"] = False
            row["limitations"].append("water2/water4 prerequisite gate is missing or failed")
        holdout = next((
            candidate for candidate in groups
            if candidate["case_id"] == HOLDOUT_CASE_ID
            and candidate["method"] == row["method"]
            and candidate["approximation_signature"] == row["approximation_signature"]
            and candidate["source_tree_sha256"] == row["source_tree_sha256"]
        ), None)
        row["holdout_gate_passed"] = bool(
            holdout and holdout["development_gate_passed"]
            and holdout["timing_evidence_sufficient"]
        )
        if not row["holdout_gate_passed"]:
            row["accepted"] = False
            row["limitations"].append(
                "H2O8s4 accuracy/resource holdout gate is missing or failed"
            )

    # Keep claim-record acceptance in sync with the prerequisite decision.
    for item in claim_records:
        claim = item["record"]
        row = next((
            candidate for candidate in groups
            if (
                f"{candidate['case_id']}:{candidate['method']}:"
                f"{candidate['approximation_signature']}:"
                f"{candidate['source_tree_sha256']}"
            ) == claim["claim_id"]
        ), None)
        if row is not None and not row["accepted"]:
            claim["acceptance_table"] = "not_yet_accepted"
            claim["limitations"] = list(row["limitations"])
            item["validation"] = validate_claim_record(claim)

    target_rows = [row for row in groups if row["case_id"] == TARGET_CASE_ID]
    # TARGET_SECONDS is the project's rounded operational definition of 10x.
    target_10x = any(row["accepted"] for row in target_rows)
    return {"schema_version": "a100-water8-analysis-v1", "generated_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "contract": {"frozen_baseline_seconds": frozen_baseline_seconds, "target_seconds": TARGET_SECONDS,
                         "cv_limit": CV_LIMIT, "ecorr_limit_eh": ECORR_LIMIT_EH,
                         "canonical_energy_limit_eh": CANONICAL_ENERGY_LIMIT_EH,
                         "projected_residual_limit": PROJECTED_RESIDUAL_LIMIT,
                         "cp_limit_kcal_mol": CP_LIMIT_KCAL_MOL, "hbm_limit_gib": HBM_LIMIT_GIB, "rss_limit_gib": RSS_LIMIT_GIB},
            "records_loaded": len(records), "performance_records_loaded": len(performance_records),
            "counterpoise_records_loaded": len(cp_records),
            "baseline_method": baseline_method, "cases": cases,
            "groups": groups, "tables": tables, "claim_records": claim_records, "target_10x_verified": target_10x,
            "limitations": [] if target_10x else ["10x is unverified: all required water8 timing, accuracy, resource and repetition gates are not passing"]}


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Case | Method | Pairs | Median s | Frozen speedup | Matched speedup | CV | Accuracy | Resources | Accepted |", "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for row in rows:
        candidate = row["candidate"]
        lines.append("| {case_id} | {method} | {alternating_pairs} | {median} | {frozen} | {matched} | {cv} | {acc} | {res} | {accepted} |".format(
            case_id=row["case_id"], method=row["method"], alternating_pairs=row["alternating_pairs"],
            median=_fmt(candidate["median_seconds"]), frozen=_fmt(row["speedup_frozen"]), matched=_fmt(row["speedup_matched_canonical"]),
            cv=_fmt(candidate["cv"]), acc=row["accuracy_passed"], res=row["resources_passed"], accepted=row["accepted"]))
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "missing" if value is None else f"{float(value):.6g}"


def write_outputs(summary: dict[str, Any], output_dir: str | Path) -> list[Path]:
    """Write summary and separate S/I/A JSON/Markdown files; refuse overwrite."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = [directory / "summary.json"]
    outputs.extend(directory / f"{name}.{suffix}" for name in ("S", "I", "A") for suffix in ("json", "md"))
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(str(path) for path in existing))
    directory.joinpath("summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for table_name in ("S", "I", "A"):
        rows = summary["tables"][table_name]
        directory.joinpath(f"{table_name}.json").write_text(json.dumps({"table": table_name, "rows": rows}, indent=2, sort_keys=True) + "\n")
        directory.joinpath(f"{table_name}.md").write_text(f"# Acceptance table {table_name}\n\n" + _markdown_table(rows))
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-method", default="canonical")
    parser.add_argument("--candidate-method", action="append", dest="candidate_methods")
    parser.add_argument("--frozen-baseline-seconds", type=float, default=FROZEN_BASELINE_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = load_records(args.inputs)
    summary = analyze_records(records, baseline_method=args.baseline_method,
                              candidate_methods=args.candidate_methods,
                              frozen_baseline_seconds=args.frozen_baseline_seconds)
    write_outputs(summary, args.output_dir)
    print(json.dumps({"records_loaded": summary["records_loaded"], "target_10x_verified": summary["target_10x_verified"],
                      "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
