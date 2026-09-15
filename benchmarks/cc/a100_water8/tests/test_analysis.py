"""CPU-only regression tests for the A100 result analyzer."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from analyze_results import (  # noqa: E402
    LOW_RANK_METHODS,
    _accuracy,
    _approximation_payload,
    _approximation_signature,
    _cp_eligibility,
    _counterpoise_error,
    _counterpoise_memory_provenance,
    _route_table,
    analyze_records,
    compatibility,
    write_outputs,
)
from claim_record import validate_claim_record  # noqa: E402


FROZEN_BASE_COMMIT = "a89b3ae018d4e82968323ef95645d91adde294aa"
SOURCE_DIGEST = "a" * 64
MANIFEST_DIGEST = "b" * 64


def _labelled_low_rank_residual(value: float) -> dict:
    """Match the record emitted by benchmark._labeled_low_rank_residual()."""

    return {
        "schema": "gpu4pyscf.cc.residual.v1",
        "available": True,
        "acceptance_measurement": "projected_equation",
        "projected_equation": value,
        "measurements": {
            "projected_equation": {
                "available": True,
                "norm": value,
                "kind": "equation-residual",
                "space": "rr-projected-active-pair",
                "is_full_space": False,
            },
        },
    }


def _completed_full_space_diagnostic(value: float = 2.0e-7) -> dict:
    """Match benchmark._run_post_hf_full_space_diagnostic() metadata."""

    return {
        "requested": True,
        "available": True,
        "execution_status": "completed",
        "kind": "equation-residual",
        "space": "full-active-pair",
        "is_full_space": True,
        "residual_norm": value,
        "wall_seconds": 1.25,
        "included_in_normal_timed_path": False,
        "included_in_post_hf": False,
    }


def _local_provenance(profile: str = "candidate") -> dict:
    allowed = {
        "prefixes": ["benchmarks/cc/a100_water8/"],
        "files": ["gpu4pyscf/cc/device_runtime.py"],
    }
    entries: list[str] = []
    status_bytes = b""
    policy_bytes = json.dumps(
        {"allowed_untracked": allowed, "status": entries},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    canonical = profile == "g0-canonical-pristine"
    return {
        "schema": "gpu4pyscf.local-provenance.v1",
        "deployment_profile": profile,
        "repository_root": "/local/repository",
        "head_revision": FROZEN_BASE_COMMIT,
        "frozen_base_revision": FROZEN_BASE_COMMIT,
        "head_matches_frozen_base": True,
        "tracked_pristine": True,
        "untracked_paths_allowed": True,
        "canonical_pristine": canonical,
        "allowed_untracked": allowed,
        "git_status": {
            "format": "porcelain-v1-lines",
            "sha256": hashlib.sha256(status_bytes).hexdigest(),
            "allowed_paths_status_sha256": hashlib.sha256(
                policy_bytes
            ).hexdigest(),
            "entry_count": 0,
            "entries": entries,
        },
    }


def _timed_source_evidence(
    *, profile: str = "candidate", revision: str = "test-revision",
    digest: str = SOURCE_DIGEST,
) -> dict:
    source = f"/task/snapshots/{digest}/source"
    require_canonical = profile == "g0-canonical-pristine"
    return {
        "repository": source,
        "revision": revision,
        "base_revision": FROZEN_BASE_COMMIT,
        "tree_sha256": digest,
        "tree_sha256_at_start": digest,
        "tree_sha256_at_end": digest,
        "stable_during_run": True,
        "formal_performance_eligible": True,
        "snapshot": {
            "schema": "gpu4pyscf.snapshot-evidence.v1",
            "valid": True,
            "valid_at_start": True,
            "valid_at_end": True,
            "source": source,
            "snapshot_root": f"/task/snapshots/{digest}",
            "manifest_path": f"/task/snapshots/{digest}/manifest.json",
            "tree_sha256": digest,
            "tree_sha256_at_end": digest,
            "read_only": True,
            "read_only_at_start": True,
            "read_only_at_end": True,
            "runtime_binaries_valid_at_start": True,
            "runtime_binaries_valid_at_end": True,
            "writable_paths": [],
            "reasons": [],
            "reasons_at_end": [],
            "stable_during_run": True,
            "manifest_sha256_at_start": MANIFEST_DIGEST,
            "manifest_sha256_at_end": MANIFEST_DIGEST,
            "expected_deployment_profile": profile,
            "canonical_pristine_required": require_canonical,
            "manifest": {
                "schema": "gpu4pyscf.source-snapshot.v2",
                "tree_sha256": digest,
                "base_revision": FROZEN_BASE_COMMIT,
                "source": source,
                "immutable": True,
                "local_provenance": _local_provenance(profile),
                "runtime_binaries": {
                    "complete": True,
                    "source_root": "/pinned/gpu4pyscf/lib",
                    "files": [{
                        "relative_path": "gpu4pyscf/lib/libtest.so",
                        "sha256": "c" * 64,
                        "bytes": 123,
                    }],
                },
            },
        },
    }


def _record(method: str, order: int, seconds: float, *, case: str = "water8-tz", error: float = 1e-7, residual: float = 1e-8) -> dict:
    source_digest = SOURCE_DIGEST
    record = {
        "record_id": f"{method}-{order}",
        "run_order": order,
        "case_id": case,
        "method": method,
        "method_class": "equation-preserving" if method == "canonical" else "controlled-approximation",
        "basis": "cc-pVTZ",
        "spherical": True,
        "charge": 0,
        "spin": 0,
        "scf_conv_tol": 1e-10,
        "cc_conv_tol": 1e-8,
        "cc_conv_tol_normt": 1e-6,
        "max_cycle": 50,
        "max_memory_mb": 110000,
        "geometry_sha256": "geometry-water8",
        "case": {"id": case, "basis": "cc-pVTZ", "source": "WATER27:H2O8d2d"},
        "status": "completed",
        "cc_converged": True,
        "post_hf_seconds": seconds,
        "e_corr": -10.0 if method == "canonical" else -10.0 + error,
        "acceptance_table": "S" if method == "canonical" else "A",
        "validation_status": "production-canonical" if method == "canonical" else "production-thc-rr",
        "accuracy": {
            "ecorr_error_eh": error,
            "projected_residual": residual,
            "cp_interaction_error_kcal_mol": 0.001,
        },
        "residual": {"projected_equation": residual},
        "peak_hbm_GiB": 40.0,
        "peak_host_RSS_GiB": 60.0,
        "checkpoint": {"included_in_post_hf": True},
        "gpu": {
            "device_count": 1,
            "device_0": {
                "name": "NVIDIA A100-SXM4-80GB",
                "totalGlobalMem": 80 * 1024**3,
            },
        },
        "pcie": {"visible_gpus": [{"pcie_gen_max": "4", "pcie_width_max": "16"}]},
        "cpu_model": "AMD EPYC 7513 32-Core Processor",
        "affinity": list(range(24, 32)),
        "thread_environment": {"GPU4PYSCF_NUMA": "3", "OMP_NUM_THREADS": "8"},
        "hardware_fingerprint": "a100-numa3",
        "timing_boundary": "post_hf includes setup and iterations",
        "software_revision": "test-revision",
        "source": _timed_source_evidence(),
    }
    if method in LOW_RANK_METHODS:
        record["residual"] = _labelled_low_rank_residual(residual)
        record["full_space_residual_diagnostic"] = (
            _completed_full_space_diagnostic()
        )
    return record


def test_hardware_gate_accepts_linux_sysfs_pcie_fields() -> None:
    record = _record("canonical", 0, 1000.0)
    record["pcie"] = {
        "visible_gpus": [{
            "max_link_speed": "16.0 GT/s PCIe",
            "max_link_width": "16",
        }]
    }

    hardware = analyze_records([record])["groups"]
    assert hardware == []  # no candidate group; exercise helper directly below

    from analyze_results import _fixed_hardware

    assert _fixed_hardware(record)["passed"] is True


def _alternating(*, candidate_seconds: list[float], error: float = 1e-7,
                 method: str = "thc_rr", case: str = "water8-tz") -> list[dict]:
    records: list[dict] = []
    for index, seconds in enumerate(candidate_seconds):
        order = index * 2
        records.append(_record("canonical", order, 1000.0 + index * 2, case=case))
        records.append(_record(method, order + 1, seconds, error=error, case=case))
    return records


def _all_gate_records(candidate_seconds: list[float]) -> list[dict]:
    records = []
    for case in ("water2-tz", "water4-tz", "water8-tz", "water8s4-tz"):
        timed = _alternating(candidate_seconds=candidate_seconds, case=case)
        records.extend(timed)
        baseline = next(item for item in timed if item["method"] == "canonical")
        candidate = next(item for item in timed if item["method"] != "canonical")
        source_digest = candidate["source"]["tree_sha256"]
        records.extend((
            _cp_record(
                "canonical",
                _approximation_signature(baseline),
                -10.0,
                case=case,
                source_digest=source_digest,
            ),
            _cp_record(
                candidate["method"],
                _approximation_signature(candidate),
                -9.999,
                case=case,
                source_digest=source_digest,
            ),
        ))
    return records


def test_three_alternating_pairs_verify_10x_and_claim_schema(tmp_path: Path) -> None:
    summary = analyze_records(_all_gate_records([99.0, 100.0, 101.0]))
    row = next(item for item in summary["groups"] if item["case_id"] == "water8-tz")
    assert row["alternating_pairs"] == 3
    assert row["timing_evidence_sufficient"] is True
    assert row["increase_required"] is False
    assert row["speedup_frozen"] > 10.0
    assert row["speedup_matched_canonical"] >= 10.0
    assert row["accepted"] is True
    assert row["holdout_gate_passed"] is True
    assert not any("counterpoise" in item for item in row["limitations"])
    assert summary["target_10x_verified"] is True
    claim = next(
        item for item in summary["claim_records"]
        if item["record"]["claim_id"].startswith("water8-tz:thc_rr:")
    )
    assert claim["validation"]["valid"] is True

    outputs = write_outputs(summary, tmp_path / "analysis")
    assert all(path.exists() for path in outputs)
    try:
        write_outputs(summary, tmp_path / "analysis")
    except FileExistsError:
        pass
    else:
        raise AssertionError("analyzer overwrote existing output")


def test_missing_accuracy_never_passes_and_insufficient_pairs_is_reported() -> None:
    records = _alternating(candidate_seconds=[100.0, 101.0])
    for record in records:
        if record["method"] != "canonical":
            record.pop("accuracy", None)
            record.pop("residual", None)
    summary = analyze_records(records)
    row = summary["groups"][0]
    assert row["alternating_pairs"] == 2
    assert row["timing_evidence_sufficient"] is False
    assert row["accuracy_passed"] is False
    assert row["accepted"] is False
    assert summary["target_10x_verified"] is False


def test_cv_over_five_percent_requires_more_runs() -> None:
    summary = analyze_records(_alternating(candidate_seconds=[50.0, 100.0, 200.0]))
    row = summary["groups"][0]
    assert row["candidate"]["cv"] > 0.05
    assert row["increase_required"] is True
    assert row["accepted"] is False


def test_water2_and_water4_have_accuracy_gates_without_target_speed_requirement() -> None:
    records: list[dict] = []
    for case in ("water2-tz", "water4-tz"):
        for index in range(3):
            order = len(records)
            records.append(_record("canonical", order, 100.0, case=case))
            records.append(_record("fno", order + 1, 100.0, case=case, error=2e-4))
    summary = analyze_records(records, candidate_methods=["fno"])
    assert len(summary["groups"]) == 2
    assert all(row["target_speed_passed"] for row in summary["groups"])
    assert all(not row["accuracy_passed"] for row in summary["groups"])
    assert all(not row["accepted"] for row in summary["groups"])


def test_equation_preserving_route_enforces_both_1e8_energy_gates() -> None:
    records: list[dict] = []
    for index, delta in enumerate((5.0e-9, 5.0e-9, 2.0e-8)):
        baseline = _record(
            "canonical_legacy", index * 2, 100.0, case="water4-tz"
        )
        candidate = _record(
            "canonical", index * 2 + 1, 90.0, case="water4-tz"
        )
        # Candidate-authored direct errors must not override the linked
        # alternating oracle values on the equation-preserving route.
        candidate["accuracy"]["ecorr_error_eh"] = 0.0
        candidate["accuracy"]["etot_error_eh"] = 0.0
        baseline["e_corr"] = -1.0
        baseline["e_tot"] = -10.0
        candidate["e_corr"] = baseline["e_corr"] + delta
        candidate["e_tot"] = baseline["e_tot"] + delta
        records.extend((baseline, candidate))

    summary = analyze_records(
        records,
        baseline_method="canonical_legacy",
        candidate_methods=["canonical"],
    )
    row = summary["groups"][0]
    assert row["route_table"] == "S"
    assert row["accuracy"][0]["checks"]["ecorr"]["threshold"] == 1.0e-8
    assert row["accuracy"][0]["checks"]["etot"]["threshold"] == 1.0e-8
    assert row["accuracy"][0]["passed"] is True
    assert row["accuracy"][2]["passed"] is False
    assert row["accuracy_passed"] is False


def test_unpaired_equation_preserving_direct_error_is_incomplete() -> None:
    candidate = _record(
        "canonical", 0, 90.0, case="water4-tz", error=0.0
    )
    candidate["e_tot"] = -10.0
    candidate["accuracy"]["etot_error_eh"] = 0.0

    summary = analyze_records(
        [candidate],
        baseline_method="canonical_legacy",
        candidate_methods=["canonical"],
    )
    row = summary["groups"][0]
    assert row["accuracy_passed"] is False
    assert set(row["accuracy"][0]["missing"]) == {"ecorr", "etot"}


def test_validation_only_rr_cannot_become_a_speed_claim() -> None:
    records = []
    for case in ("water2-tz", "water4-tz", "water8-tz"):
        records.extend(_alternating(
            candidate_seconds=[10.0, 10.1, 9.9],
            method="rr_canonical",
            case=case,
        ))
    summary = analyze_records(records)
    assert all(not row["production_eligible"] for row in summary["groups"])
    assert all(not row["accepted"] for row in summary["groups"])
    assert summary["target_10x_verified"] is False


def test_incompatible_fast_run_is_excluded_from_acceptance_statistics() -> None:
    records = _all_gate_records([99.0, 100.0, 101.0])
    incompatible = _record("thc_rr", 99, 1.0)
    incompatible["hardware_fingerprint"] = "different-hardware"
    records.append(incompatible)
    summary = analyze_records(records)
    target = next(item for item in summary["groups"] if item["case_id"] == "water8-tz")
    assert target["candidate"]["median_seconds"] == 100.0


def _set_source(record: dict, digest: str, *, stable: bool = True) -> None:
    source = _timed_source_evidence(revision=digest, digest=digest)
    if not stable:
        source["tree_sha256_at_end"] = "f" * 64
        source["stable_during_run"] = False
        source["formal_performance_eligible"] = False
        source["snapshot"]["stable_during_run"] = False
    record["source"] = source


def test_implementation_digests_split_timing_groups_and_pair_counts() -> None:
    records: list[dict] = []
    order = 0
    for digest, count in (("d" * 64, 2), ("e" * 64, 1)):
        for index in range(count):
            baseline = _record("canonical", order, 1000.0 + index)
            candidate = _record("thc_rr", order + 1, 100.0 + index)
            _set_source(baseline, digest)
            _set_source(candidate, digest)
            records.extend((baseline, candidate))
            order += 2
    summary = analyze_records(records)
    assert len(summary["groups"]) == 2
    assert {
        row["source_tree_sha256"]: row["alternating_pairs"]
        for row in summary["groups"]
    } == {"d" * 64: 2, "e" * 64: 1}
    assert all(not row["timing_evidence_sufficient"] for row in summary["groups"])


def test_timed_source_stability_is_a_formal_compatibility_gate() -> None:
    baseline = _record("canonical", 0, 1000.0)
    candidate = _record("thc_rr", 1, 100.0)
    _set_source(candidate, SOURCE_DIGEST, stable=False)
    result = compatibility(baseline, candidate)
    assert result["compatible"] is False
    assert "candidate source stability is missing or failed" in result["reasons"]

    legacy = _record("thc_rr", 1, 100.0)
    legacy["source"].pop("tree_sha256_at_start")
    assert compatibility(baseline, legacy)["compatible"] is False


def test_changed_manifest_hash_is_excluded_from_formal_timing() -> None:
    baseline = _record("canonical", 0, 1000.0)
    candidate = _record("thc_rr", 1, 100.0)
    candidate["source"]["snapshot"]["manifest_sha256_at_end"] = "0" * 64

    result = compatibility(baseline, candidate)

    assert result["compatible"] is False
    assert any("immutable v2 snapshot" in reason for reason in result["reasons"])


def test_g0_profile_cannot_masquerade_as_matched_canonical() -> None:
    baseline = _record("canonical", 0, 1000.0)
    candidate = _record("thc_rr", 1, 100.0)
    baseline["source"] = _timed_source_evidence(
        profile="g0-canonical-pristine"
    )

    result = compatibility(baseline, candidate)

    assert result["compatible"] is False
    assert any("matched canonical" in reason for reason in result["reasons"])


def test_max_cycle_is_protocol_gate_and_memory_is_compared_as_provenance() -> None:
    baseline = _record("canonical", 0, 1000.0)
    candidate = _record("thc_rr", 1, 100.0)
    candidate["max_memory_mb"] = 90000
    provenance_only = compatibility(baseline, candidate)
    assert provenance_only["compatible"] is True
    assert provenance_only["memory_provenance"]["match"] is False
    candidate["max_cycle"] = 51
    gated = compatibility(baseline, candidate)
    assert gated["compatible"] is False
    assert "scientific mismatch: max_cycle" in gated["reasons"]


def test_source_mutated_candidate_is_never_accepted() -> None:
    records = _all_gate_records([99.0, 100.0, 101.0])
    for record in records:
        if record["method"] != "canonical":
            record["performance_eligible"] = False
    summary = analyze_records(records)
    assert summary["target_10x_verified"] is False
    assert all(
        not row["accepted"] and not row["production_eligible"]
        for row in summary["groups"]
    )
    assert all(
        any("performance-ineligible" in item for item in row["limitations"])
        for row in summary["groups"]
    )


def test_target_requires_h2o8s4_holdout() -> None:
    records: list[dict] = []
    for case in ("water2-tz", "water4-tz", "water8-tz"):
        records.extend(_alternating(candidate_seconds=[99.0, 100.0, 101.0], case=case))
    summary = analyze_records(records)
    target = next(item for item in summary["groups"] if item["case_id"] == "water8-tz")
    assert target["holdout_gate_passed"] is False
    assert target["accepted"] is False
    assert summary["target_10x_verified"] is False


def test_holdout_needs_accuracy_and_resources_but_not_10x() -> None:
    records = _alternating(
        candidate_seconds=[900.0], case="water8s4-tz"
    )
    baseline = next(item for item in records if item["method"] == "canonical")
    candidate = next(item for item in records if item["method"] != "canonical")
    source_digest = candidate["source"]["tree_sha256"]
    records.extend((
        _cp_record(
            "canonical",
            _approximation_signature(baseline),
            -10.0,
            case="water8s4-tz",
            source_digest=source_digest,
        ),
        _cp_record(
            candidate["method"],
            _approximation_signature(candidate),
            -9.999,
            case="water8s4-tz",
            source_digest=source_digest,
        ),
    ))
    summary = analyze_records(records)
    holdout = summary["groups"][0]
    assert holdout["case_role"] == "holdout"
    assert holdout["required_pair_count"] == 1
    assert holdout["timing_evidence_sufficient"] is True
    assert holdout["target_speed_passed"] is True
    assert holdout["development_gate_passed"] is True
    assert holdout["accepted"] is True
    assert summary["target_10x_verified"] is False


def test_claim_reads_benchmark_source_revision() -> None:
    records = _all_gate_records([99.0, 100.0, 101.0])
    for record in records:
        if "software_revision" in record:
            record.pop("software_revision")
            record["source"]["revision"] = "source-revision"
    summary = analyze_records(records)
    claim = next(
        item["record"] for item in summary["claim_records"]
        if item["record"]["claim_id"].startswith("water8-tz:thc_rr:")
    )
    assert claim["candidate"]["software_revision"] == "source-revision"


def _set_fno_cutoff(record: dict, cutoff: float) -> str:
    record["method"] = "fno"
    record["method_class"] = "controlled-approximation"
    record["acceptance_table"] = "A"
    record["validation_status"] = "production-fno"
    record["plan"] = {
        "method": "fno",
        "settings": {"fno_thresh": cutoff},
    }
    signature = _approximation_signature(record)
    record["approximation_signature"] = signature
    return signature


def test_fno_cutoffs_form_separate_performance_groups_and_pairs() -> None:
    records: list[dict] = []
    signatures: dict[float, str] = {}
    order = 0
    for index in range(3):
        baseline1 = _record("canonical", order, 1000.0 + index, case="water2-tz")
        candidate1 = _record("fno", order + 1, 100.0 + index, case="water2-tz")
        signatures[1e-5] = _set_fno_cutoff(candidate1, 1e-5)
        baseline2 = _record("canonical", order + 2, 1000.0 + index, case="water2-tz")
        candidate2 = _record("fno", order + 3, 200.0 + index, case="water2-tz")
        signatures[1e-6] = _set_fno_cutoff(candidate2, 1e-6)
        records.extend((baseline1, candidate1, baseline2, candidate2))
        order += 4

    summary = analyze_records(records, candidate_methods=["fno"])
    assert len(summary["groups"]) == 2
    by_signature = {
        row["approximation_signature"]: row for row in summary["groups"]
    }
    assert by_signature[signatures[1e-5]]["candidate"]["seconds"] == [
        100.0, 101.0, 102.0
    ]
    assert by_signature[signatures[1e-6]]["candidate"]["seconds"] == [
        200.0, 201.0, 202.0
    ]
    assert all(row["alternating_pairs"] == 3 for row in summary["groups"])


def _cp_record(
    method: str,
    signature: str,
    energy: float,
    *,
    case: str = "water4-tz",
    source_digest: str = SOURCE_DIGEST,
    cluster_scf_converged: bool = True,
    cluster_cc_converged: bool = True,
    fragment_scf_converged: bool = True,
    fragment_cc_converged: bool = True,
    bundle_id: str = "shared-test-bundle",
) -> dict:
    cluster_payload = [["O", 0.0, 0.0, 0.0]]
    fragment_payload = [["ghost-O", 0.0, 0.0, 0.0]]
    cluster_atom_hash = hashlib.sha256(json.dumps(
        cluster_payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")).hexdigest()
    fragment_atom_hash = hashlib.sha256(json.dumps(
        fragment_payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")).hexdigest()
    record = {
        "schema": "gpu4pyscf.water8.counterpoise.v1",
        "_record_kind": "counterpoise",
        "case_id": case,
        "case": {
            "id": case,
            "source": None,
            "unit": "Angstrom",
            "geometry_angstrom": cluster_payload,
            "geometry_sha256": cluster_atom_hash,
            "basis": "cc-pVTZ",
            "spherical": True,
        },
        "method": method,
        "approximation_signature": signature,
        "accuracy_eligible": True,
        "all_calculations_converged": True,
        "protocol": {
            "scf_conv_tol": 1e-10,
            "cc_conv_tol": 1e-8,
            "cc_conv_tol_normt": 1e-6,
            "cc_max_cycle": 50,
            "max_memory_mb": 110000,
        },
        "cluster": {
            "atom_payload_sha256": cluster_atom_hash,
            "energy": {
                "converged": cluster_cc_converged,
                "metadata": {"scf_converged": cluster_scf_converged},
            }
        },
        "fragments": [{
            "fragment_id": 0,
            "atom_payload_sha256": fragment_atom_hash,
            "energy": {
                "converged": fragment_cc_converged,
                "metadata": {"scf_converged": fragment_scf_converged},
            }
        }],
        "source": {
            "repository": f"/task/snapshots/{source_digest}/source",
            "revision": "test-revision",
            "base_revision": FROZEN_BASE_COMMIT,
            "formal_counterpoise_eligible": True,
            "tree_sha256": source_digest,
            "tree_sha256_at_start": source_digest,
            "tree_sha256_at_end": source_digest,
            "stable_during_run": True,
            "snapshot": {
                "schema": "gpu4pyscf.snapshot-evidence.v1",
                "valid": True,
                "valid_at_start": True,
                "valid_at_end": True,
                "source": f"/task/snapshots/{source_digest}/source",
                "snapshot_root": f"/task/snapshots/{source_digest}",
                "manifest_path": f"/task/snapshots/{source_digest}/manifest.json",
                "expected_task_root": "/task",
                "tree_sha256": source_digest,
                "tree_sha256_at_end": source_digest,
                "read_only": True,
                "read_only_at_start": True,
                "read_only_at_end": True,
                "stable_during_run": True,
                "runtime_binaries_valid_at_start": True,
                "runtime_binaries_valid_at_end": True,
                "writable_paths": [],
                "manifest_sha256_at_start": MANIFEST_DIGEST,
                "manifest_sha256_at_end": MANIFEST_DIGEST,
                "expected_deployment_profile": "candidate",
                "canonical_pristine_required": False,
                "reasons": [],
                "reasons_at_end": [],
                "manifest": {
                    "schema": "gpu4pyscf.source-snapshot.v2",
                    "tree_sha256": source_digest,
                    "base_revision": FROZEN_BASE_COMMIT,
                    "source": f"/task/snapshots/{source_digest}/source",
                    "immutable": True,
                    "local_provenance": _local_provenance(),
                    "runtime_binaries": {
                        "complete": True,
                        "source_root": "/pinned/gpu4pyscf/lib",
                        "files": [{
                            "relative_path": "gpu4pyscf/lib/libtest.so",
                            "sha256": "c" * 64,
                            "bytes": 123,
                        }],
                    },
                },
            },
        },
        "interaction_energy_cp_kcal_mol": energy,
    }
    bundle_sha = hashlib.sha256(
        f"bundle-file:{bundle_id}".encode("ascii")
    ).hexdigest()
    bundle_fingerprint = hashlib.sha256(
        f"bundle-manifest:{bundle_id}".encode("ascii")
    ).hexdigest()
    entry_fingerprints = {
        key: hashlib.sha256(
            f"orbital:{bundle_id}:{key}".encode("ascii")
        ).hexdigest()
        for key in ("cluster", "fragment-000")
    }
    match_payload = {
        "bundle_sha256": bundle_sha,
        "bundle_fingerprint": bundle_fingerprint,
        "entry_fingerprints": entry_fingerprints,
    }
    match_key = hashlib.sha256(json.dumps(
        match_payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")).hexdigest()
    record["orbital_bundle"] = {
        "schema": "gpu4pyscf.water8.cp-canonical-orbitals.v1",
        "complete": True,
        "sha256": bundle_sha,
        "bundle_fingerprint": bundle_fingerprint,
        "entry_count": len(entry_fingerprints),
        "entry_fingerprints": entry_fingerprints,
        "record_match_key": match_key,
        "case_contract": {
            "case_id": case,
            "case_source": None,
            "basis": "cc-pVTZ",
            "unit": "Angstrom",
            "reference": "RHF",
            "spherical": True,
            "charge": 0,
            "spin": 0,
            "cluster_atom_payload": cluster_payload,
            "cluster_atom_payload_sha256": cluster_atom_hash,
        },
        "source_contract": {
            "base_revision": FROZEN_BASE_COMMIT,
            "revision": "test-revision",
            "tree_sha256": source_digest,
        },
    }

    def evidence(entry_key: str, atom_hash: str) -> dict:
        return {
            "entry_key": entry_key,
            "bundle_complete": True,
            "applied_before_solver_construction": True,
            "bundle_sha256": bundle_sha,
            "bundle_fingerprint": bundle_fingerprint,
            "orbital_fingerprint": entry_fingerprints[entry_key],
            "identity": {
                "entry_key": entry_key,
                "atom_payload_sha256": atom_hash,
            },
        }

    record["cluster"]["energy"]["metadata"]["canonical_orbitals"] = (
        evidence("cluster", cluster_atom_hash)
    )
    record["fragments"][0]["energy"]["metadata"][
        "canonical_orbitals"
    ] = evidence("fragment-000", fragment_atom_hash)
    return record


def test_counterpoise_pairing_uses_exact_candidate_signature() -> None:
    canonical_template = _record("canonical", 0, 1.0, case="water4-tz")
    canonical_signature = _approximation_signature(canonical_template)
    fno1 = _record("fno", 1, 1.0, case="water4-tz")
    signature1 = _set_fno_cutoff(fno1, 1e-5)
    fno2 = _record("fno", 2, 1.0, case="water4-tz")
    signature2 = _set_fno_cutoff(fno2, 1e-6)
    records = [
        _cp_record("canonical", canonical_signature, -10.0),
        _cp_record("fno", signature1, -9.99),
        _cp_record("fno", signature2, -9.90),
    ]
    error1, limitations1 = _counterpoise_error(
        records, "water4-tz", "canonical", "fno", signature1,
        SOURCE_DIGEST,
    )
    error2, limitations2 = _counterpoise_error(
        records, "water4-tz", "canonical", "fno", signature2,
        SOURCE_DIGEST,
    )
    assert np.isclose(error1, 0.01)
    assert np.isclose(error2, 0.10)
    assert limitations1 == limitations2 == []
    missing_error, missing_limitations = _counterpoise_error(
        [records[0], records[2]],
        "water4-tz",
        "canonical",
        "fno",
        signature1,
        SOURCE_DIGEST,
    )
    assert missing_error is None
    assert missing_limitations


def test_counterpoise_accepts_canonical_and_candidate_with_same_bundle() -> None:
    canonical_template = _record("canonical", 0, 1.0, case="water4-tz")
    canonical_signature = _approximation_signature(canonical_template)
    fno = _record("fno", 1, 1.0, case="water4-tz")
    fno_signature = _set_fno_cutoff(fno, 1e-6)
    records = [
        _cp_record(
            "canonical", canonical_signature, -10.0,
            bundle_id="canonical-produced",
        ),
        _cp_record(
            "fno", fno_signature, -9.995,
            bundle_id="canonical-produced",
        ),
    ]
    error, reasons = _counterpoise_error(
        records, "water4-tz", "canonical", "fno", fno_signature,
        SOURCE_DIGEST,
    )
    assert np.isclose(error, 0.005)
    assert reasons == []


def test_counterpoise_rejects_independently_valid_different_bundles() -> None:
    canonical_template = _record("canonical", 0, 1.0, case="water4-tz")
    canonical_signature = _approximation_signature(canonical_template)
    fno = _record("fno", 1, 1.0, case="water4-tz")
    fno_signature = _set_fno_cutoff(fno, 1e-6)
    records = [
        _cp_record(
            "canonical", canonical_signature, -10.0,
            bundle_id="canonical-bundle",
        ),
        _cp_record(
            "fno", fno_signature, -9.995,
            bundle_id="independent-candidate-bundle",
        ),
    ]
    error, reasons = _counterpoise_error(
        records, "water4-tz", "canonical", "fno", fno_signature,
        SOURCE_DIGEST,
    )
    assert error is None
    assert any("shared orbital bundle proof failed" in item for item in reasons)


def test_counterpoise_rejects_tampered_realized_entry_evidence() -> None:
    canonical_template = _record("canonical", 0, 1.0, case="water4-tz")
    canonical_signature = _approximation_signature(canonical_template)
    fno = _record("fno", 1, 1.0, case="water4-tz")
    fno_signature = _set_fno_cutoff(fno, 1e-6)
    baseline = _cp_record("canonical", canonical_signature, -10.0)
    candidate = _cp_record("fno", fno_signature, -9.995)
    candidate["fragments"][0]["energy"]["metadata"][
        "canonical_orbitals"
    ]["orbital_fingerprint"] = "f" * 64
    error, reasons = _counterpoise_error(
        [baseline, candidate], "water4-tz", "canonical", "fno",
        fno_signature, SOURCE_DIGEST,
    )
    assert error is None
    assert any("orbital evidence does not match" in item for item in reasons)


def test_counterpoise_rejects_pair_relabelled_away_from_bundle_case() -> None:
    canonical_template = _record("canonical", 0, 1.0, case="water4-tz")
    canonical_signature = _approximation_signature(canonical_template)
    fno = _record("fno", 1, 1.0, case="water4-tz")
    fno_signature = _set_fno_cutoff(fno, 1e-6)
    baseline = _cp_record("canonical", canonical_signature, -10.0)
    candidate = _cp_record("fno", fno_signature, -9.995)
    relabelled_payload = [["O", 1.0, 0.0, 0.0]]
    relabelled_hash = hashlib.sha256(json.dumps(
        relabelled_payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")).hexdigest()
    for record in (baseline, candidate):
        record["case"]["geometry_angstrom"] = relabelled_payload
        record["case"]["geometry_sha256"] = relabelled_hash
    error, reasons = _counterpoise_error(
        [baseline, candidate], "water4-tz", "canonical", "fno",
        fno_signature, SOURCE_DIGEST,
    )
    assert error is None
    assert any("does not match the record" in item for item in reasons)


def test_counterpoise_eligibility_rejects_legacy_and_unconverged_records() -> None:
    record = _cp_record("canonical", "canonical-signature", -10.0)
    record["fragments"][0]["energy"]["converged"] = False
    passed, reasons = _cp_eligibility(record, SOURCE_DIGEST)
    assert passed is False
    assert any("fragment 0 CCSD" in reason for reason in reasons)

    legacy = _cp_record("canonical", "canonical-signature", -10.0)
    legacy.pop("accuracy_eligible")
    legacy.pop("source")
    passed, reasons = _cp_eligibility(legacy, SOURCE_DIGEST)
    assert passed is False
    assert any("not explicitly accuracy-eligible" in reason for reason in reasons)
    assert any("source stability" in reason for reason in reasons)


def test_counterpoise_formal_snapshot_evidence_is_fail_closed() -> None:
    valid = _cp_record("canonical", "canonical-signature", -10.0)
    passed, reasons = _cp_eligibility(valid, SOURCE_DIGEST)
    assert passed is True
    assert reasons == []

    missing = copy.deepcopy(valid)
    missing["source"].pop("snapshot")
    passed, reasons = _cp_eligibility(missing, SOURCE_DIGEST)
    assert passed is False
    assert "counterpoise immutable snapshot evidence is missing" in reasons

    mutable = copy.deepcopy(valid)
    mutable["source"]["snapshot"]["read_only_at_end"] = False
    passed, reasons = _cp_eligibility(mutable, SOURCE_DIGEST)
    assert passed is False
    assert any("not read-only" in reason for reason in reasons)

    invalid = copy.deepcopy(valid)
    invalid["source"]["snapshot"]["manifest"]["immutable"] = False
    passed, reasons = _cp_eligibility(invalid, SOURCE_DIGEST)
    assert passed is False
    assert any("does not declare immutability" in reason for reason in reasons)

    changed = copy.deepcopy(valid)
    changed["source"]["snapshot"]["manifest_sha256_at_end"] = "c" * 64
    passed, reasons = _cp_eligibility(changed, SOURCE_DIGEST)
    assert passed is False
    assert any("content hash" in reason for reason in reasons)

    runtime_invalid = copy.deepcopy(valid)
    runtime_invalid["source"]["snapshot"]["manifest"]["runtime_binaries"][
        "files"
    ][0]["sha256"] = "invalid"
    passed, reasons = _cp_eligibility(runtime_invalid, SOURCE_DIGEST)
    assert passed is False
    assert any("runtime-binary manifest" in reason for reason in reasons)


def test_counterpoise_snapshot_profile_matches_manifest_and_g0_pristine() -> None:
    valid_candidate = _cp_record(
        "canonical", "canonical-signature", -10.0
    )
    passed, reasons = _cp_eligibility(valid_candidate, SOURCE_DIGEST)
    assert passed is True
    assert reasons == []

    mismatched = copy.deepcopy(valid_candidate)
    mismatched["source"]["snapshot"]["manifest"]["local_provenance"][
        "deployment_profile"
    ] = "g0-canonical-pristine"
    passed, reasons = _cp_eligibility(mismatched, SOURCE_DIGEST)
    assert passed is False
    assert any("does not match manifest provenance" in item for item in reasons)

    unknown = copy.deepcopy(valid_candidate)
    unknown["source"]["snapshot"]["expected_deployment_profile"] = "mystery"
    unknown["source"]["snapshot"]["manifest"]["local_provenance"][
        "deployment_profile"
    ] = "mystery"
    passed, reasons = _cp_eligibility(unknown, SOURCE_DIGEST)
    assert passed is False
    assert any("missing or unknown" in item for item in reasons)

    valid_g0 = _cp_record("canonical", "canonical-signature", -10.0)
    valid_g0["source"] = _timed_source_evidence(
        profile="g0-canonical-pristine"
    )
    valid_g0["source"]["formal_counterpoise_eligible"] = True
    valid_g0["source"]["snapshot"]["expected_task_root"] = "/task"
    passed, reasons = _cp_eligibility(valid_g0, SOURCE_DIGEST)
    assert passed is True
    assert reasons == []

    nonpristine_g0 = copy.deepcopy(valid_g0)
    nonpristine_g0["source"]["snapshot"]["manifest"]["local_provenance"][
        "canonical_pristine"
    ] = False
    passed, reasons = _cp_eligibility(nonpristine_g0, SOURCE_DIGEST)
    assert passed is False
    assert any("not canonically pristine" in item for item in reasons)


def test_counterpoise_requires_matching_source_and_max_cycle() -> None:
    canonical_template = _record("canonical", 0, 1.0, case="water4-tz")
    canonical_signature = _approximation_signature(canonical_template)
    fno = _record("fno", 1, 1.0, case="water4-tz")
    fno_signature = _set_fno_cutoff(fno, 1e-6)
    baseline_cp = _cp_record("canonical", canonical_signature, -10.0)
    candidate_cp = _cp_record("fno", fno_signature, -9.999)
    candidate_cp["protocol"]["cc_max_cycle"] = 51
    error, reasons = _counterpoise_error(
        [baseline_cp, candidate_cp],
        "water4-tz",
        "canonical",
        "fno",
        fno_signature,
        SOURCE_DIGEST,
    )
    assert error is None
    assert reasons == ["counterpoise scientific protocol mismatch"]

    candidate_cp["protocol"]["cc_max_cycle"] = 50
    _set_source(candidate_cp, "different-source")
    error, reasons = _counterpoise_error(
        [baseline_cp, candidate_cp],
        "water4-tz",
        "canonical",
        "fno",
        fno_signature,
        SOURCE_DIGEST,
    )
    assert error is None
    assert any("source digest" in reason for reason in reasons)


def test_counterpoise_memory_is_compared_as_non_gating_provenance() -> None:
    canonical_template = _record("canonical", 0, 1.0, case="water4-tz")
    canonical_signature = _approximation_signature(canonical_template)
    fno = _record("fno", 1, 1.0, case="water4-tz")
    fno_signature = _set_fno_cutoff(fno, 1e-6)
    baseline_cp = _cp_record("canonical", canonical_signature, -10.0)
    candidate_cp = _cp_record("fno", fno_signature, -9.999)
    candidate_cp["protocol"]["max_memory_mb"] = 90000

    error, reasons = _counterpoise_error(
        [baseline_cp, candidate_cp],
        "water4-tz",
        "canonical",
        "fno",
        fno_signature,
        SOURCE_DIGEST,
    )
    provenance = _counterpoise_memory_provenance(
        [baseline_cp, candidate_cp],
        "water4-tz",
        "canonical",
        "fno",
        fno_signature,
        SOURCE_DIGEST,
    )
    assert np.isclose(error, 0.001)
    assert reasons == []
    assert provenance == {
        "baseline_max_memory_mb": [110000],
        "candidate_max_memory_mb": [90000],
        "match": False,
        "acceptance_gate": False,
    }


def test_thc_cd_fallback_payload_and_route_preserve_every_control() -> None:
    record = _record("thc_cd", 1, 10.0, case="water2-tz")
    record["eri_backend"] = "cd"
    record["plan"] = {
        "settings": {
            "eri_backend": "cd",
            "eri_tol": 1e-8,
            "direct_scf_tol": 1e-13,
            "cd_max_rank": 301,
            "rr_eig_cutoff": 1e-6,
            "rr_max_rank": 79,
            "denominator_tolerance": 1e-10,
            "denominator_max_rank": 83,
            "rr_solver_tolerance": 2e-10,
            "rr_solver_maxiter": 107,
            "rr_dense_fallback_dimension": 64,
            "rr_ritz_residual_tolerance": 3e-9,
            "precision": "fp64",
            "thc_fit_tol": 1e-6,
            "thc_rank": 1060,
            "thc_initial_rank": None,
            "thc_max_rank": 1060,
            "thc_rank_growth": 1.5,
            "thc_orthogonality_cutoff": 1e-12,
            "thc_orthogonality_tolerance": 1e-10,
            "thc_max_iterations": 500,
            "thc_als_convergence_tolerance": 1e-10,
            "thc_ridge": 1e-12,
            "thc_seed": 0,
            "thc_replacement_validation_atol": 1e-11,
            "thc_replacement_validation_rtol": 2e-11,
        }
    }

    payload = _approximation_payload(record)

    assert payload["eri_backend"] == "cd"
    assert payload["direct_scf_tol"] == 1e-13
    assert payload["denominator_tolerance"] == 1e-10
    assert payload["thc_rank"] == 1060
    assert payload["thc_replacement_validation_atol"] == 1e-11
    assert payload["thc_replacement_validation_rtol"] == 2e-11
    assert _route_table(record) == "A"


def test_thc_cd_analyzer_requires_projected_residual_and_fails_closed() -> None:
    baseline = _record("canonical", 0, 100.0, case="water2-tz")
    candidate = _record("thc_cd", 1, 10.0, case="water2-tz")
    candidate["performance_eligible"] = True
    candidate["acceptance_table"] = "A"
    candidate["validation_status"] = "production-thc-cd"
    candidate["residual"] = {}
    candidate["accuracy"].pop("projected_residual")

    summary = analyze_records(
        [baseline, candidate],
        baseline_method="canonical",
        candidate_methods=["thc_cd"],
    )
    row = summary["groups"][0]

    assert row["route_table"] == "A"
    assert row["production_eligible"] is False
    assert row["accepted"] is False
    assert "projected_residual" in row["accuracy"][0]["checks"]
    assert "projected_residual" in row["accuracy"][0]["missing"]
    assert any(
        "validation-only implementation" in limitation
        for limitation in row["limitations"]
    )


@pytest.mark.parametrize("method", sorted(LOW_RANK_METHODS))
def test_current_benchmark_rr_thc_residual_product_passes_strict_semantic_gates(
    method: str,
) -> None:
    candidate = _record(method, 1, 10.0, case="water2-tz")

    accuracy = _accuracy(
        candidate,
        require_projected_residual=True,
        require_cp=False,
    )

    assert accuracy["passed"] is True
    projected = accuracy["checks"]["projected_residual"]
    assert projected["evidence_valid"] is True
    assert projected["value"] == 1.0e-8
    diagnostic = accuracy["checks"]["full_space_residual_diagnostic"]
    assert diagnostic["evidence_valid"] is True
    assert diagnostic["value"] == 2.0e-7
    assert diagnostic["threshold"] is None


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    [
        ("missing", "record.residual is missing"),
        ("wrong-schema", "schema is missing or unsupported"),
        ("wrong-kind", "kind is not equation-residual"),
        ("wrong-space", "space is not rr-projected-active-pair"),
        ("unavailable", "residual is not available"),
        ("legacy-only", "schema is missing or unsupported"),
    ],
)
def test_rr_projected_residual_gate_rejects_ambiguous_or_mislabeled_evidence(
    mutation: str, reason_fragment: str
) -> None:
    candidate = _record("thc_rr", 1, 10.0, case="water2-tz")
    if mutation == "missing":
        candidate.pop("residual")
    elif mutation == "wrong-schema":
        candidate["residual"]["schema"] = "legacy.residual.v0"
    elif mutation == "wrong-kind":
        candidate["residual"]["measurements"]["projected_equation"][
            "kind"
        ] = "jacobi-update"
    elif mutation == "wrong-space":
        candidate["residual"]["measurements"]["projected_equation"][
            "space"
        ] = "full-active-pair"
    elif mutation == "unavailable":
        measurement = candidate["residual"]["measurements"][
            "projected_equation"
        ]
        measurement["available"] = False
        measurement["norm"] = None
    elif mutation == "legacy-only":
        candidate["residual"] = {"projected_equation": 1.0e-8}
    else:  # pragma: no cover - guards the parametrization itself
        raise AssertionError(mutation)

    accuracy = _accuracy(
        candidate,
        require_projected_residual=True,
        require_cp=False,
    )
    check = accuracy["checks"]["projected_residual"]

    assert accuracy["passed"] is False
    assert check["passed"] is False
    assert check["evidence_valid"] is False
    assert check["value"] is None
    assert check["legacy_display_value"] == 1.0e-8
    assert reason_fragment in " | ".join(check["evidence_reasons"])


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    [
        ("missing", "diagnostic is missing or invalid"),
        ("not-requested", "was not explicitly requested"),
        ("not-completed", "did not complete"),
        ("wrong-kind", "kind is not equation-residual"),
        ("wrong-space", "space is not full-active-pair"),
        ("unavailable", "diagnostic is not available"),
        ("inside-normal-timer", "outside the normal timed path"),
        ("inside-post-hf", "outside post_hf_seconds"),
    ],
)
def test_rr_full_space_diagnostic_gate_rejects_incomplete_or_mislabeled_evidence(
    mutation: str, reason_fragment: str
) -> None:
    candidate = _record("rr_cd", 1, 10.0, case="water2-tz")
    if mutation == "missing":
        candidate.pop("full_space_residual_diagnostic")
    else:
        diagnostic = candidate["full_space_residual_diagnostic"]
        if mutation == "not-requested":
            diagnostic["requested"] = False
        elif mutation == "not-completed":
            diagnostic["execution_status"] = "not-requested"
        elif mutation == "wrong-kind":
            diagnostic["kind"] = "jacobi-update"
        elif mutation == "wrong-space":
            diagnostic["space"] = "rr-projected-active-pair"
        elif mutation == "unavailable":
            diagnostic["available"] = False
            diagnostic["residual_norm"] = None
        elif mutation == "inside-normal-timer":
            diagnostic["included_in_normal_timed_path"] = True
        elif mutation == "inside-post-hf":
            diagnostic["included_in_post_hf"] = True
        else:  # pragma: no cover - guards the parametrization itself
            raise AssertionError(mutation)

    accuracy = _accuracy(
        candidate,
        require_projected_residual=True,
        require_cp=False,
    )
    check = accuracy["checks"]["full_space_residual_diagnostic"]

    assert accuracy["passed"] is False
    assert check["passed"] is False
    assert check["evidence_valid"] is False
    assert check["value"] is None
    assert reason_fragment in " | ".join(check["evidence_reasons"])


def test_canonical_legacy_can_pair_with_resident_canonical_same_source() -> None:
    records = []
    for index in range(3):
        legacy = _record(
            "canonical_legacy", 2 * index, 1000.0, case="water8-tz"
        )
        legacy.update({
            "method_class": "equation-preserving",
            "acceptance_table": "S",
            "validation_status": "production-canonical-legacy-execution",
            "e_corr": -10.0,
        })
        resident = _record(
            "canonical", 2 * index + 1, 100.0, case="water8-tz"
        )
        resident["e_corr"] = -10.0
        records.extend((legacy, resident))

    summary = analyze_records(
        records,
        baseline_method="canonical_legacy",
        candidate_methods=["canonical"],
    )
    row = summary["groups"][0]

    assert _approximation_signature(records[0]) == (
        _approximation_signature(records[1])
    )
    assert row["route_table"] == "S"
    assert row["alternating_pairs"] == 3
    assert row["speedup_matched_canonical"] == 10.0
    assert row["compatibility_reasons"] == []
