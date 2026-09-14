"""Tests for the fail-closed canonical checkpoint parity tool."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import compare_canonical_checkpoints as COMPARATOR  # noqa: E402


compare_checkpoints = COMPARATOR.compare_checkpoints


def _record(case_id: str = "water2-tz") -> dict:
    dims = {"nao": 4, "nocc": 2, "nvir": 2}
    case = {
        "id": case_id,
        "source": "WATER27:H2O2",
        "role": "sanity",
        "basis": "cc-pVTZ",
        "expected_dimensions": dims,
        "geometry_angstrom": [["H", 0.0, 0.0, 0.0]],
    }
    settings = {"cc_conv_tol": 1.0e-8, "cc_conv_tol_normt": 1.0e-6,
                "cc_max_cycle": 50, "eri_backend": "canonical",
                "eri_tol": 1.0e-8, "precision": "fp64",
                "scf_conv_tol": 1.0e-10}
    return {
        "status": "completed", "cc_converged": True, "case_id": case_id,
        "case": case, "geometry_sha256": "geometry",
        "method": "canonical", "approximation": {"method": "canonical"},
        "approximation_signature": "canonical:test", "e_corr": -0.5,
        "cc_iterations": 3, "orbital_dimensions": dims,
        "plan": {"basis": "cc-pVTZ", "reference": "RHF", "spherical": True,
                  "charge": 0, "spin": 0, "method_class": "equation-preserving",
                  "protocol_schema": "gpu4pyscf.water8-ccsd-a100.v2",
                  "settings": settings},
        "source_protocol": {"frozen_base_commit": "a89"},
        "source": {"base_revision": "a89", "revision": "tree-sha256:tree",
                    "tree_sha256": "tree", "tree_sha256_at_end": "tree",
                    "files_hashed": 10, "files_hashed_at_end": 10,
                    "source_kind": "content-manifest", "stable_during_run": True,
                    "formal_performance_eligible": True,
                    "snapshot": {"tree_sha256": "tree", "tree_sha256_at_end": "tree",
                                  "manifest_sha256_at_end": "manifest",
                                  "stable_during_run": True, "read_only_at_end": True,
                                  "runtime_binaries_valid_at_end": True,
                                  "manifest": {"local_provenance": {"deployment_profile": "candidate"}}}},
        "host": "compute-1", "platform": "Linux", "cpu_model": "AMD EPYC 7513",
        "affinity": [24, 25],
        "thread_environment": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                                "MKL_NUM_THREADS": "2", "GPU4PYSCF_NUMA": "3"},
        "gpu": {"device_count": 1, "device_0": {"name": "NVIDIA A100-SXM4-80GB",
                                                   "major": 8, "minor": 0,
                                                   "totalGlobalMem": 80}},
        "pcie": {"visible_gpus": [{"name": "NVIDIA A100-SXM4-80GB", "numa_node": "3",
                                     "pci_bus_id": "01:00.0", "max_link_speed": "16.0",
                                     "max_link_width": "16", "current_link_speed": "16.0",
                                     "current_link_width": "16", "pcie_gen_current": "4",
                                     "pcie_width_current": "16"}]},
        "topology": "GPU0 NUMA3", "timing_definition": "complete post-HF",
        "residual": {"available": True, "equation_norm": 2.0e-7},
        "checkpoint": {"included_in_post_hf": True},
    }


def _write_pair(tmp_path: Path, *, conflict: bool = False) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    left_t1 = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    signs_o = np.asarray([1.0, -1.0])
    signs_v = np.asarray([1.0, -1.0])
    right_t1 = left_t1 * signs_o[:, None] * signs_v[None, :]
    if conflict:
        right_t1 = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    left_t2 = np.arange(16.0).reshape(2, 2, 2, 2) / 10.0
    right_t2 = left_t2 * signs_o[:, None, None, None] * signs_o[None, :, None, None] * signs_v[None, None, :, None] * signs_v[None, None, None, :]
    paths = []
    for index, (method, t1, t2) in enumerate((("canonical", left_t1, left_t2), ("canonical_legacy", right_t1, right_t2))):
        cp = tmp_path / f"cp{index}.npz"
        np.savez(cp, checkpoint_schema=np.asarray(["gpu4pyscf.cc.restart.v1"]),
                 e_corr=np.asarray([-0.5]), method=np.asarray([method]),
                 representation=np.asarray(["dense-t2"]), t1=t1, t2=t2)
        record = _record()
        record["method"] = method
        record["plan"]["settings"] = dict(record["plan"]["settings"])
        record["checkpoint"] = {"included_in_post_hf": True, "path": str(cp),
                                 "sha256": hashlib.sha256(cp.read_bytes()).hexdigest()}
        path = tmp_path / f"run{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1]


def _write_v2_pair(
    tmp_path: Path,
    *,
    rotate_right: bool = False,
    corrupt_fingerprint: bool = False,
    artifact_sha256: str = "a" * 64,
    origin: str = "loaded",
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": COMPARATOR.ORBITAL_IDENTITY_SCHEMA,
        "case_id": "water2-tz",
        "geometry_sha256": "geometry",
        "basis": "cc-pVTZ",
        "reference": "RHF",
        "spherical": True,
        "charge": 0,
        "spin": 0,
        "nao": 4,
        "nmo": 4,
        "nocc": 2,
        "nvir": 2,
        "nelectron": 4,
    }
    t1 = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    t2 = np.arange(16.0, dtype=np.float64).reshape(2, 2, 2, 2) / 10.0
    paths = []
    for index, method in enumerate(("canonical", "canonical_legacy")):
        coeff = np.eye(4, dtype=np.float64)
        if index == 1 and rotate_right:
            angle = 0.2
            coeff[:2, :2] = np.asarray([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ])
        orbitals = {
            "mo_coeff": coeff,
            "mo_occ": np.asarray([2.0, 2.0, 0.0, 0.0]),
            "mo_energy": np.asarray([-1.0, -0.5, 0.5, 1.0]),
        }
        fingerprint = COMPARATOR._orbital_fingerprint(identity, orbitals)
        if index == 1 and corrupt_fingerprint:
            fingerprint = "0" * 64
        cp = tmp_path / f"v2-cp{index}.npz"
        np.savez(
            cp,
            checkpoint_schema=np.asarray([COMPARATOR.RESTART_SCHEMA_V2]),
            e_corr=np.asarray([-0.5]),
            method=np.asarray([method]),
            representation=np.asarray(["dense-t2"]),
            t1=t1,
            t2=t2,
            mo_coeff=orbitals["mo_coeff"],
            mo_occ=orbitals["mo_occ"],
            mo_energy=orbitals["mo_energy"],
            orbital_identity_json=np.asarray([
                json.dumps(identity, sort_keys=True, separators=(",", ":"))
            ]),
            orbital_fingerprint=np.asarray([fingerprint]),
            orbital_artifact_sha256=np.asarray([artifact_sha256]),
            orbital_origin=np.asarray([origin]),
        )
        record = _record()
        record["method"] = method
        record["canonical_orbitals"] = {
            "orbital_fingerprint": fingerprint,
            "artifact_sha256": artifact_sha256 or None,
            "mode": origin,
        }
        record["checkpoint"] = {
            "included_in_post_hf": True,
            "path": str(cp),
            "sha256": hashlib.sha256(cp.read_bytes()).hexdigest(),
            "checkpoint_schema": COMPARATOR.RESTART_SCHEMA_V2,
            "orbital_fingerprint": fingerprint,
            "orbital_artifact_sha256": artifact_sha256 or None,
        }
        path = tmp_path / f"v2-run{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1]


def test_sign_gauge_alignment_reports_components_and_passes(tmp_path: Path) -> None:
    left, right = _write_pair(tmp_path)
    result = compare_checkpoints(left, right)
    assert result["passed"] is True
    assert result["phase_gauge"]["component_count"] == 1
    assert result["checks"]["amplitudes"]["t1_raw"]["max_abs"] > 0
    assert result["checks"]["amplitudes"]["t1_gauge_aligned"]["max_abs"] == 0
    assert result["checks"]["amplitudes"]["t2_gauge_aligned"]["l2"] == 0
    assert "orbital rotations" in result["scope_note"]


def test_inconsistent_bipartite_constraints_fail_closed(tmp_path: Path) -> None:
    left, right = _write_pair(tmp_path, conflict=True)
    with pytest.raises(ValueError, match="inconsistent MO sign constraints"):
        compare_checkpoints(left, right)


def test_hash_mismatch_and_provenance_mismatch_fail_closed(tmp_path: Path) -> None:
    left, right = _write_pair(tmp_path)
    payload = json.loads(right.read_text())
    payload["checkpoint"]["sha256"] = "0" * 64
    right.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        compare_checkpoints(left, right)
    right, _ = _write_pair(tmp_path / "other")
    payload = json.loads(right.read_text())
    payload["host"] = "other-host"
    right.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hardware provenance mismatch"):
        compare_checkpoints(left, right)


def test_invalid_checkpoint_keys_fail_closed(tmp_path: Path) -> None:
    left, right = _write_pair(tmp_path)
    payload = json.loads(left.read_text())
    cp = Path(payload["checkpoint"]["path"])
    np.savez(cp, checkpoint_schema=np.asarray(["gpu4pyscf.cc.restart.v1"]),
             e_corr=np.asarray([-0.5]), method=np.asarray(["canonical"]),
             representation=np.asarray(["dense-t2"]), t1=np.zeros((2, 2)),
             t2=np.zeros((2, 2, 2, 2)), extra=np.asarray([1]))
    payload["checkpoint"]["sha256"] = hashlib.sha256(cp.read_bytes()).hexdigest()
    left.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checkpoint keys"):
        compare_checkpoints(left, right)


def test_v2_exact_shared_orbitals_use_strict_element_comparison(
    tmp_path: Path,
) -> None:
    left, right = _write_v2_pair(tmp_path)
    result = compare_checkpoints(left, right)

    assert result["passed"] is True
    assert result["restart_schema"] == COMPARATOR.RESTART_SCHEMA_V2
    assert result["phase_gauge"] is None
    assert result["orbital_alignment"] == {
        "mode": "exact-canonical-mo",
        "orbital_fingerprint": result["orbital_alignment"][
            "orbital_fingerprint"
        ],
        "shared_artifact_sha256": "a" * 64,
        "shared_artifact": True,
        "left_origin": "loaded",
        "right_origin": "loaded",
        "arrays_byte_identical": True,
        "rotation_or_sign_alignment_applied": False,
    }
    amplitudes = result["checks"]["amplitudes"]
    assert amplitudes["t1_exact_orbitals"]["max_abs"] == 0.0
    assert amplitudes["t2_exact_orbitals"]["l2"] == 0.0
    assert "byte-identical" in result["scope_note"]


def test_v2_different_orbitals_fail_closed_without_rotation_alignment(
    tmp_path: Path,
) -> None:
    left, right = _write_v2_pair(tmp_path, rotate_right=True)
    with pytest.raises(ValueError, match="canonical MO orbitals differ"):
        compare_checkpoints(left, right)


def test_v2_corrupt_embedded_orbital_fingerprint_fails_closed(
    tmp_path: Path,
) -> None:
    left, right = _write_v2_pair(tmp_path, corrupt_fingerprint=True)
    with pytest.raises(ValueError, match="orbital fingerprint mismatch"):
        compare_checkpoints(left, right)


@pytest.mark.parametrize(
    ("artifact_sha256", "origin"),
    (("not-a-sha", "loaded"), ("", "loaded"), ("a" * 64, "fresh-scf-memory")),
)
def test_v2_artifact_hash_and_origin_relation_is_fail_closed(
    tmp_path: Path, artifact_sha256: str, origin: str
) -> None:
    left, right = _write_v2_pair(
        tmp_path, artifact_sha256=artifact_sha256, origin=origin
    )
    with pytest.raises(ValueError, match="artifact SHA-256"):
        compare_checkpoints(left, right)


def test_v2_cycle_count_is_diagnostic_and_not_a_parity_gate(
    tmp_path: Path,
) -> None:
    left, right = _write_v2_pair(tmp_path)
    record = json.loads(right.read_text())
    record["cc_iterations"] = 4
    right.write_text(json.dumps(record))

    result = compare_checkpoints(left, right)

    assert result["passed"] is True
    assert result["checks"]["cycles"] == {
        "left": 3,
        "right": 4,
        "matched": False,
        "acceptance_gate": False,
    }


def test_default_tolerances_match_the_g1_acceptance_contract() -> None:
    assert COMPARATOR.DEFAULT_ENERGY_TOL == 1.0e-8
    assert COMPARATOR.DEFAULT_RESIDUAL_TOL == 1.0e-6
    assert COMPARATOR.DEFAULT_AMPLITUDE_MAX_TOL == 1.0e-6
    assert COMPARATOR.DEFAULT_AMPLITUDE_L2_TOL == 1.0e-6


def test_convergence_threshold_mismatch_fails_scientific_provenance(
    tmp_path: Path,
) -> None:
    left, right = _write_v2_pair(tmp_path)
    record = json.loads(right.read_text())
    record["plan"]["settings"]["cc_conv_tol"] = 2.0e-8
    right.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="scientific provenance mismatch"):
        compare_checkpoints(left, right)
