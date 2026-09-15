#!/usr/bin/env python3
"""Reproducible A100 WATER27 CCSD benchmark and holdout harness.

The driver records one JSON object per run and refuses to overwrite an
existing result.  ``rr_cd`` and ``thc_cd`` exercise the direct-CD, compressed
RR lifecycle; ``rr_canonical`` and ``thc_canonical`` intentionally retain
their dense validation labels and are not production reduced-scaling claims.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[2]
CASE_FILE = ROOT / "cases.json"

try:
    from .snapshot_manifest import (
        CANDIDATE_DEPLOYMENT_PROFILE,
        FROZEN_BASE_COMMIT,
        G0_DEPLOYMENT_PROFILE,
        source_tree_digest as _snapshot_source_tree_digest,
        validate_snapshot,
    )
except ImportError:  # direct execution and importlib-based unit tests
    _snapshot_spec = importlib.util.spec_from_file_location(
        "water8_benchmark_snapshot_manifest", ROOT / "snapshot_manifest.py"
    )
    if _snapshot_spec is None or _snapshot_spec.loader is None:
        raise RuntimeError("cannot load snapshot_manifest.py")
    _snapshot_module = importlib.util.module_from_spec(_snapshot_spec)
    _snapshot_spec.loader.exec_module(_snapshot_module)
    CANDIDATE_DEPLOYMENT_PROFILE = (
        _snapshot_module.CANDIDATE_DEPLOYMENT_PROFILE
    )
    FROZEN_BASE_COMMIT = _snapshot_module.FROZEN_BASE_COMMIT
    G0_DEPLOYMENT_PROFILE = _snapshot_module.G0_DEPLOYMENT_PROFILE
    _snapshot_source_tree_digest = _snapshot_module.source_tree_digest
    validate_snapshot = _snapshot_module.validate_snapshot

# Load the backend-neutral evaluator directly from the frozen source tree.  A
# normal ``gpu4pyscf.cc`` package import initializes CUDA-facing modules, which
# would make the benchmark's CPU-only serialization tests depend on CuPy.
_residual_spec = importlib.util.spec_from_file_location(
    "water8_shared_residual_evaluator",
    REPOSITORY_ROOT / "gpu4pyscf" / "cc" / "residual_evaluator.py",
)
if _residual_spec is None or _residual_spec.loader is None:
    raise RuntimeError("cannot load gpu4pyscf/cc/residual_evaluator.py")
_residual_module = importlib.util.module_from_spec(_residual_spec)
sys.modules[_residual_spec.name] = _residual_module
_residual_spec.loader.exec_module(_residual_module)
LOW_RANK_RESIDUAL_SCHEMA = _residual_module.RESIDUAL_SCHEMA
_evaluate_dense_ccsd_residual = _residual_module.evaluate_dense_ccsd_residual

PROTOCOL_SCHEMA = "gpu4pyscf.water8-ccsd-a100.v2"
ORBITAL_ARTIFACT_SCHEMA = "gpu4pyscf.cc.canonical-orbitals.v1"
ORBITAL_IDENTITY_SCHEMA = "gpu4pyscf.cc.canonical-orbital-identity.v1"
RESTART_SCHEMA_V1 = "gpu4pyscf.cc.restart.v1"
RESTART_SCHEMA_V2 = "gpu4pyscf.cc.restart.v2"
ORBITAL_ORTHONORMALITY_TOLERANCE = 1.0e-8
METHODS = (
    "canonical",
    "canonical_legacy",
    "fno",
    "rr_cd",
    "thc_cd",
    "rr_canonical",
    "thc_canonical",
)
DIRECT_CD_METHODS = frozenset({"rr_cd", "thc_cd"})
FULL_SPACE_DIAGNOSTIC_METHODS = DIRECT_CD_METHODS
REQUIRED_EXPLICIT_APPROXIMATION_CONTROLS = {
    "rr_cd": ("eri_tol", "rr_eig_cutoff"),
    "rr_canonical": ("eri_tol", "rr_eig_cutoff"),
    "thc_cd": ("eri_tol", "rr_eig_cutoff", "thc_fit_tol"),
    "thc_canonical": ("eri_tol", "rr_eig_cutoff", "thc_fit_tol"),
}
EXPLICIT_APPROXIMATION_CONTROL_NAMES = frozenset(
    name
    for names in REQUIRED_EXPLICIT_APPROXIMATION_CONTROLS.values()
    for name in names
)
CASES = ("water2-tz", "water4-tz", "water8-tz", "water8s4-tz")
METHOD_INFO = {
    "canonical": {
        "method_class": "equation-preserving",
        "acceptance_table": "S",
        "validation_status": "production-canonical",
        "resident_iterations_policy": "solver-default",
    },
    "canonical_legacy": {
        "method_class": "equation-preserving",
        "acceptance_table": "S",
        "validation_status": "production-canonical-legacy-execution",
        "resident_iterations_policy": "forced-disabled",
    },
    "fno": {
        "method_class": "controlled-approximation",
        "acceptance_table": "A",
        "validation_status": "production-fno",
    },
    "rr_cd": {
        "method_class": "controlled-approximation",
        "acceptance_table": "not_yet_accepted",
        "validation_status": "direct-cd-compressed-rr-reference",
        "correctness_evidence_status": "recordable-pending-oracle-gates",
        "performance_eligible": False,
        "performance_gates": {
            "gint_restricted_columns": "pending-water2-timing-gate",
            "rr_ring_contraction": "bounded-memory-reference-kernel",
        },
        "performance_limitations": [
            "GINT restricted-column water2 timing gate pending",
            "RR ring contraction is the bounded-memory reference kernel",
        ],
    },
    "thc_cd": {
        "method_class": "controlled-approximation",
        "acceptance_table": "not_yet_accepted",
        "validation_status": (
            "direct-cd-full-pair-amplitude-thc-validation-endpoint"
        ),
        "correctness_evidence_status": "recordable-pending-oracle-gates",
        "performance_eligible": False,
        "performance_gates": {
            "inexact_amplitude_thc": "fail-closed",
            "shared_rr_paper_residual_decomposition": "not-implemented",
            "rr_ring_contraction": "bounded-memory-reference-kernel",
        },
        "performance_limitations": [
            "inexact amplitude THC is fail-closed",
            "shared RR/paper residual decomposition is pending",
            "remaining RR ring kernel is a bounded-memory reference",
        ],
    },
    "rr_canonical": {
        "method_class": "controlled-approximation",
        "acceptance_table": "not_yet_accepted",
        "validation_status": "dense-projected-validation",
    },
    "thc_canonical": {
        "method_class": "controlled-approximation",
        "acceptance_table": "not_yet_accepted",
        "validation_status": "dense-two-level-validation-surrogate",
    },
}


class _TrackExplicitApproximationValue(argparse.Action):
    """Store a value and remember that it came from the command line."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        provided = set(getattr(
            namespace, "_explicit_approximation_controls", ()
        ))
        provided.add(self.dest)
        setattr(
            namespace,
            "_explicit_approximation_controls",
            frozenset(provided),
        )
        sources = dict(getattr(
            namespace, "_approximation_control_sources", {}
        ))
        sources[self.dest] = "cli"
        setattr(namespace, "_approximation_control_sources", sources)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--repeat", default="1")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--max-memory-mb", type=int, default=110000)
    p.add_argument("--scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--cc-conv-tol", type=float, default=1e-8)
    p.add_argument("--cc-conv-tol-normt", type=float, default=1e-6)
    p.add_argument(
        "--max-cycle", "--cc-max-cycle", dest="max_cycle", type=int, default=50
    )
    p.add_argument(
        "--eri-tol", type=float, default=1e-8,
        action=_TrackExplicitApproximationValue,
    )
    p.add_argument("--direct-scf-tol", type=float, default=1e-13)
    p.add_argument("--cd-max-rank", type=int, default=None)
    p.add_argument(
        "--gint-column-backend",
        choices=("selected", "restricted-reference"),
        default="selected",
    )
    p.add_argument("--gint-group-size", type=int, default=16)
    p.add_argument("--gint-max-block-bytes", type=int, default=None)
    p.add_argument("--gint-max-batch-size", type=int, default=32)
    p.add_argument("--cd-mo-block-size", type=int, default=32)
    p.add_argument(
        "--rr-eig-cutoff", type=float, default=1e-6,
        action=_TrackExplicitApproximationValue,
    )
    p.add_argument("--rr-max-rank", type=int, default=None)
    p.add_argument("--denominator-tolerance", type=float, default=1e-10)
    p.add_argument("--denominator-max-rank", type=int, default=None)
    p.add_argument("--rr-initial-rank", type=int, default=32)
    p.add_argument("--rr-solver-tolerance", type=float, default=1e-10)
    p.add_argument("--rr-solver-maxiter", type=int, default=None)
    p.add_argument("--rr-dense-fallback-dimension", type=int, default=64)
    p.add_argument("--rr-ritz-residual-tolerance", type=float, default=None)
    p.add_argument("--rr-auxiliary-block-size", type=int, default=1)
    p.add_argument("--rr-virtual-block-size", type=int, default=8)
    p.add_argument("--precision", choices=("fp64",), default="fp64")
    p.add_argument(
        "--thc-fit-tol", type=float, default=1e-6,
        action=_TrackExplicitApproximationValue,
    )
    p.add_argument("--thc-rank", type=int, default=None)
    p.add_argument("--thc-initial-rank", type=int, default=None)
    p.add_argument("--thc-max-rank", type=int, default=None)
    p.add_argument("--thc-rank-growth", type=float, default=1.5)
    p.add_argument("--thc-orthogonality-cutoff", type=float, default=1e-12)
    p.add_argument(
        "--thc-orthogonality-tolerance", type=float, default=1e-10
    )
    p.add_argument("--thc-max-iterations", type=int, default=500)
    p.add_argument(
        "--thc-als-convergence-tolerance", type=float, default=1e-10
    )
    p.add_argument("--thc-ridge", type=float, default=1e-12)
    p.add_argument("--thc-seed", type=int, default=0)
    p.add_argument(
        "--thc-replacement-validation-atol", type=float, default=None
    )
    p.add_argument(
        "--thc-replacement-validation-rtol", type=float, default=None
    )
    p.add_argument("--fno-thresh", type=float, default=1e-6)
    p.add_argument("--fno-nvir-act", type=int, default=None)
    p.add_argument("--fno-pct-occ", type=float, default=None)
    p.add_argument(
        "--retain-cupy-cache",
        action="store_true",
        help="retain the CuPy memory pool between instrumented CC phases",
    )
    p.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="debug-only: omit the checkpoint (the run cannot pass acceptance)",
    )
    p.add_argument(
        "--orbital-artifact-in",
        type=Path,
        default=None,
        help=(
            "load and apply a validated canonical-MO artifact after the fresh "
            "SCF, before the post-HF timer starts"
        ),
    )
    p.add_argument(
        "--orbital-artifact-out",
        type=Path,
        default=None,
        help=(
            "write the converged canonical MOs once, validate/reload them, and "
            "apply them before the post-HF timer starts"
        ),
    )
    p.add_argument(
        "--run-full-space-diagnostic",
        action="store_true",
        help=(
            "after post-HF timing, reconstruct and evaluate the full pair-space "
            "RR/CD residual; supported only by rr_cd and thc_cd"
        ),
    )
    p.add_argument("--config", type=Path, default=None, help="JSON config whose keys match CLI names")
    p.add_argument("--dry-run", action="store_true", help="validate and print the planned run")
    return p


def load_cases() -> dict[str, dict[str, Any]]:
    payload = json.loads(CASE_FILE.read_text())
    cases = payload.get("cases", payload)
    if not isinstance(cases, list):
        raise ValueError("cases.json must contain a cases list")
    out = {item["id"]: item for item in cases}
    if set(out) != set(CASES):
        raise ValueError(f"cases.json must contain exactly {list(CASES)}")
    for item in out.values():
        if item["basis"] != "cc-pVTZ" or not item["geometry_angstrom"]:
            raise ValueError(f"invalid basis or geometry for {item['id']}")
    return out


def geometry_hash(case: dict[str, Any]) -> str:
    blob = json.dumps(case["geometry_angstrom"], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _canonical_orbital_identity(
    case_id: str,
    case: dict[str, Any],
    mol: Any,
) -> dict[str, Any]:
    """Return the scientific identity to which an MO artifact is bound."""

    nao = int(mol.nao_nr())
    nelectron = int(mol.nelectron)
    nocc = nelectron // 2
    return {
        "schema": ORBITAL_IDENTITY_SCHEMA,
        "case_id": str(case_id),
        "geometry_sha256": geometry_hash(case),
        "basis": str(case["basis"]),
        "reference": "RHF",
        "spherical": True,
        "charge": int(getattr(mol, "charge", 0)),
        "spin": int(getattr(mol, "spin", 0)),
        "nao": nao,
        "nmo": nao,
        "nocc": nocc,
        "nvir": nao - nocc,
        "nelectron": nelectron,
    }


def _host_float64(value: Any, np_module: Any) -> Any:
    if hasattr(value, "get"):
        value = value.get()
    array = np_module.asarray(value)
    if not np_module.issubdtype(array.dtype, np_module.number):
        raise ValueError("canonical orbital array must be numeric")
    if np_module.iscomplexobj(array):
        raise ValueError("canonical orbital array must be real")
    return np_module.asarray(array, dtype=np_module.float64, order="C")


def _orbital_fingerprint(
    identity: dict[str, Any],
    arrays: dict[str, Any],
) -> str:
    """Hash the exact FP64 MO arrays together with their scientific identity."""

    digest = hashlib.sha256()
    digest.update(_canonical_json(identity).encode("ascii"))
    for name in ("mo_coeff", "mo_occ", "mo_energy"):
        array = arrays[name]
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_canonical_orbitals(
    identity: dict[str, Any],
    arrays: dict[str, Any],
    mol: Any,
    np_module: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate dimensions, occupations, energies, and AO-metric orthogonality."""

    if identity.get("schema") != ORBITAL_IDENTITY_SCHEMA:
        raise ValueError(f"{label}: unsupported orbital identity schema")
    nao = int(identity["nao"])
    nmo = int(identity["nmo"])
    nocc = int(identity["nocc"])
    nelectron = int(identity["nelectron"])
    coeff = _host_float64(arrays["mo_coeff"], np_module)
    occupation = _host_float64(arrays["mo_occ"], np_module)
    energy = _host_float64(arrays["mo_energy"], np_module)
    arrays.update({
        "mo_coeff": coeff,
        "mo_occ": occupation,
        "mo_energy": energy,
    })
    if coeff.shape != (nao, nmo):
        raise ValueError(
            f"{label}: mo_coeff shape {coeff.shape} != {(nao, nmo)}"
        )
    if occupation.shape != (nmo,) or energy.shape != (nmo,):
        raise ValueError(
            f"{label}: mo_occ/mo_energy shapes must both equal {(nmo,)}"
        )
    for name, array in arrays.items():
        if not np_module.all(np_module.isfinite(array)):
            raise ValueError(f"{label}: {name} contains NaN or infinity")
    expected_occ = np_module.zeros(nmo, dtype=np_module.float64)
    expected_occ[:nocc] = 2.0
    if not np_module.array_equal(occupation, expected_occ):
        raise ValueError(
            f"{label}: mo_occ is not the contiguous closed-shell RHF occupation"
        )
    if not math.isclose(
        float(occupation.sum()), float(nelectron), rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(f"{label}: mo_occ electron count is inconsistent")
    if np_module.any(np_module.diff(energy) < -1.0e-8):
        raise ValueError(f"{label}: mo_energy is not nondecreasing")
    overlap = _host_float64(mol.intor_symmetric("int1e_ovlp"), np_module)
    if overlap.shape != (nao, nao):
        raise ValueError(f"{label}: AO overlap shape is inconsistent")
    gram = coeff.T @ overlap @ coeff
    orthogonality_error = float(
        np_module.max(np_module.abs(gram - np_module.eye(nmo)))
    )
    if orthogonality_error > ORBITAL_ORTHONORMALITY_TOLERANCE:
        raise ValueError(
            f"{label}: canonical MOs are not AO-metric orthonormal; "
            f"max error {orthogonality_error:.3e}"
        )
    return {
        "orthonormality_max_abs": orthogonality_error,
        "orthonormality_tolerance": ORBITAL_ORTHONORMALITY_TOLERANCE,
        "occupation": "closed-shell-rhf-contiguous",
    }


def _capture_canonical_orbitals(mf: Any, np_module: Any) -> dict[str, Any]:
    return {
        name: _host_float64(getattr(mf, name), np_module)
        for name in ("mo_coeff", "mo_occ", "mo_energy")
    }


def _orbital_producer(
    source: dict[str, Any],
    *,
    hf_energy: float,
    scf_conv_tol: float,
) -> dict[str, Any]:
    return {
        "protocol_schema": PROTOCOL_SCHEMA,
        "base_revision": source.get("base_revision"),
        "revision": source.get("revision"),
        "tree_sha256": source.get("tree_sha256"),
        "source_kind": source.get("source_kind"),
        "hf_converged": True,
        "hf_energy": float(hf_energy),
        "scf_conv_tol": float(scf_conv_tol),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def _validate_orbital_producer(producer: Any, *, label: str) -> None:
    if not isinstance(producer, dict):
        raise ValueError(f"{label}: orbital artifact producer is malformed")
    required_text = ("revision", "source_kind", "created_utc")
    if any(
        not isinstance(producer.get(name), str) or not producer[name]
        for name in required_text
    ):
        raise ValueError(f"{label}: orbital artifact source provenance is incomplete")
    tree_sha256 = producer.get("tree_sha256")
    if (
        not isinstance(tree_sha256, str)
        or len(tree_sha256) != 64
        or any(character not in "0123456789abcdef" for character in tree_sha256)
    ):
        raise ValueError(f"{label}: orbital artifact source tree hash is invalid")
    if producer.get("base_revision") != FROZEN_BASE_COMMIT:
        raise ValueError(
            f"{label}: orbital artifact frozen base revision is invalid"
        )
    if producer.get("protocol_schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"{label}: orbital artifact protocol schema is invalid")
    if producer.get("hf_converged") is not True:
        raise ValueError(f"{label}: orbital artifact producer HF did not converge")
    for name in ("hf_energy", "scf_conv_tol"):
        raw = producer.get(name)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise ValueError(
                f"{label}: orbital artifact producer {name} is invalid"
            )
    if float(producer["scf_conv_tol"]) <= 0.0:
        raise ValueError(
            f"{label}: orbital artifact producer scf_conv_tol is invalid"
        )


def _write_orbital_artifact_once(
    path: Path,
    identity: dict[str, Any],
    arrays: dict[str, Any],
    producer: dict[str, Any],
    mol: Any,
    np_module: Any,
) -> None:
    """Write one self-describing canonical-MO artifact without overwrite."""

    _validate_orbital_producer(producer, label="orbital artifact output")
    validation = _validate_canonical_orbitals(
        identity, arrays, mol, np_module, label="orbital artifact output"
    )
    fingerprint = _orbital_fingerprint(identity, arrays)
    payload = {
        "artifact_schema": np_module.asarray([ORBITAL_ARTIFACT_SCHEMA]),
        "identity_json": np_module.asarray([_canonical_json(identity)]),
        "producer_json": np_module.asarray([_canonical_json(producer)]),
        "validation_json": np_module.asarray([_canonical_json(validation)]),
        "orbital_fingerprint": np_module.asarray([fingerprint]),
        "mo_coeff": arrays["mo_coeff"],
        "mo_occ": arrays["mo_occ"],
        "mo_energy": arrays["mo_energy"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np_module.savez(stream, **payload)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite orbital artifact: {path}"
        ) from exc


def _load_orbital_artifact(
    path: Path,
    expected_identity: dict[str, Any],
    mol: Any,
    np_module: Any,
) -> dict[str, Any]:
    """Load an MO artifact and fail closed on content or case mismatch."""

    try:
        payload = np_module.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"cannot load orbital artifact {path}: {exc}") from exc
    required = {
        "artifact_schema",
        "identity_json",
        "producer_json",
        "validation_json",
        "orbital_fingerprint",
        "mo_coeff",
        "mo_occ",
        "mo_energy",
    }
    try:
        if set(payload.files) != required:
            raise ValueError(
                "orbital artifact keys do not equal the canonical v1 schema"
            )

        def scalar(name: str) -> str:
            value = np_module.asarray(payload[name])
            if value.shape != (1,):
                raise ValueError(
                    f"orbital artifact key {name!r} must be a scalar array"
                )
            return str(value[0])

        if scalar("artifact_schema") != ORBITAL_ARTIFACT_SCHEMA:
            raise ValueError("unsupported orbital artifact schema")
        try:
            identity = json.loads(scalar("identity_json"))
            producer = json.loads(scalar("producer_json"))
            stored_validation = json.loads(scalar("validation_json"))
        except json.JSONDecodeError as exc:
            raise ValueError("orbital artifact JSON metadata is malformed") from exc
        if identity != expected_identity:
            raise ValueError(
                "orbital artifact scientific identity does not match the "
                "requested case/basis/geometry"
            )
        if not isinstance(stored_validation, dict):
            raise ValueError("orbital artifact provenance is malformed")
        _validate_orbital_producer(producer, label="orbital artifact input")
        arrays = {
            name: np_module.array(payload[name], copy=True)
            for name in ("mo_coeff", "mo_occ", "mo_energy")
        }
        validation = _validate_canonical_orbitals(
            identity, arrays, mol, np_module, label="orbital artifact input"
        )
        fingerprint = _orbital_fingerprint(identity, arrays)
        if scalar("orbital_fingerprint") != fingerprint:
            raise ValueError("orbital artifact fingerprint mismatch")
    finally:
        payload.close()
    return {
        "schema": ORBITAL_ARTIFACT_SCHEMA,
        "identity": identity,
        "producer": producer,
        "stored_validation": stored_validation,
        "validation": validation,
        "orbital_fingerprint": fingerprint,
        "artifact_path": str(path.resolve()),
        "artifact_sha256": _file_sha256(path),
        "artifact_bytes": int(path.stat().st_size),
        "_arrays": arrays,
    }


def _apply_canonical_orbitals(
    mf: Any,
    arrays: dict[str, Any],
    np_module: Any,
    cp_module: Any,
) -> None:
    backend = cp_module if cp_module is not None else np_module
    for name in ("mo_coeff", "mo_occ", "mo_energy"):
        setattr(mf, name, backend.asarray(arrays[name]))


def _prepare_canonical_orbitals(
    args: argparse.Namespace,
    mf: Any,
    mol: Any,
    case: dict[str, Any],
    source: dict[str, Any],
    np_module: Any,
    cp_module: Any,
) -> dict[str, Any]:
    """Capture or load exact MOs and apply the serialized FP64 values."""

    identity = _canonical_orbital_identity(args.case, case, mol)
    current_hf_energy = float(mf.e_tot)
    if args.orbital_artifact_in is not None:
        context = _load_orbital_artifact(
            args.orbital_artifact_in, identity, mol, np_module
        )
        mode = "loaded"
    else:
        arrays = _capture_canonical_orbitals(mf, np_module)
        validation = _validate_canonical_orbitals(
            identity, arrays, mol, np_module, label="fresh SCF orbitals"
        )
        producer = _orbital_producer(
            source,
            hf_energy=current_hf_energy,
            scf_conv_tol=args.scf_conv_tol,
        )
        if args.orbital_artifact_out is not None:
            _write_orbital_artifact_once(
                args.orbital_artifact_out,
                identity,
                arrays,
                producer,
                mol,
                np_module,
            )
            # Use the exact serialized values in the producing run as well.
            context = _load_orbital_artifact(
                args.orbital_artifact_out, identity, mol, np_module
            )
            mode = "written-and-reloaded"
        else:
            context = {
                "schema": ORBITAL_ARTIFACT_SCHEMA,
                "identity": identity,
                "producer": producer,
                "stored_validation": validation,
                "validation": validation,
                "orbital_fingerprint": _orbital_fingerprint(identity, arrays),
                "artifact_path": None,
                "artifact_sha256": None,
                "artifact_bytes": None,
                "_arrays": arrays,
            }
            mode = "fresh-scf-memory"
    producer = context.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("canonical orbital producer provenance is missing")
    for field in ("base_revision", "revision", "tree_sha256", "source_kind"):
        if producer.get(field) != source.get(field):
            raise ValueError(
                "orbital artifact source provenance does not match the "
                f"current benchmark source: {field}"
            )
    producer_hf_energy = context["producer"].get("hf_energy")
    if not isinstance(producer_hf_energy, (int, float)) or not math.isfinite(
        float(producer_hf_energy)
    ):
        raise ValueError("orbital artifact producer HF energy is missing")
    hf_energy_delta = abs(current_hf_energy - float(producer_hf_energy))
    hf_energy_tolerance = max(1.0e-7, 100.0 * float(args.scf_conv_tol))
    if hf_energy_delta > hf_energy_tolerance:
        raise ValueError(
            "fresh SCF energy disagrees with orbital artifact producer: "
            f"delta {hf_energy_delta:.3e} Eh"
        )
    _apply_canonical_orbitals(
        mf, context["_arrays"], np_module, cp_module
    )
    # The serialized orbitals and their converged RHF reference energy form
    # one scientific artifact.  Applying both prevents a fresh process's tiny
    # SCF reduction-order difference from leaking into the reported CC total
    # energy or counterpoise protocol.
    mf.e_tot = float(producer_hf_energy)
    context.update({
        "mode": mode,
        "applied_to_mean_field": True,
        "included_in_post_hf": False,
        "fresh_hf_energy": current_hf_energy,
        "producer_hf_energy": float(producer_hf_energy),
        "applied_hf_energy": float(mf.e_tot),
        "hf_energy_delta": hf_energy_delta,
        "hf_energy_tolerance": hf_energy_tolerance,
    })
    return context


def _orbital_record(context: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in context.items() if key != "_arrays"}


def _setting(settings: Any, name: str, default: Any = None) -> Any:
    if isinstance(settings, dict):
        return settings.get(name, default)
    return getattr(settings, name, default)


def approximation_identity(method: str, settings: Any) -> tuple[dict[str, Any], str]:
    """Return the effective approximation controls and a stable signature.

    Only controls that change the mathematical approximation enter the hash.
    Runtime settings such as thread count and convergence tolerance remain in
    the scientific/hardware compatibility records instead.  The effective FNO
    selector follows PySCF precedence: ``nvir_act``, then ``pct_occ``, then the
    occupation threshold.
    """

    scientific_method = (
        "canonical" if method == "canonical_legacy" else str(method)
    )
    payload: dict[str, Any] = {
        "schema": "gpu4pyscf.cc.approximation.v1",
        "method": scientific_method,
    }
    if method == "fno":
        nvir_act = _setting(settings, "fno_nvir_act")
        pct_occ = _setting(settings, "fno_pct_occ")
        threshold = float(_setting(settings, "fno_thresh", 1.0e-6))
        if nvir_act is not None:
            selection = {"mode": "nvir_act", "value": int(nvir_act)}
        elif pct_occ is not None:
            selection = {"mode": "pct_occ", "value": float(pct_occ)}
        else:
            selection = {"mode": "occupation_threshold", "value": threshold}
        payload.update({"selection": selection, "correction": "delta-mp2"})
    elif method in {
        "rr_cd", "thc_cd", "rr_canonical", "thc_canonical"
    }:
        payload.update({
            "eri_backend": (
                "cd" if method in DIRECT_CD_METHODS else "canonical"
            ),
            "eri_tol": float(_setting(settings, "eri_tol", 1.0e-8)),
            "rr_eig_cutoff": float(
                _setting(settings, "rr_eig_cutoff", 1.0e-6)
            ),
            "rr_max_rank": _setting(settings, "rr_max_rank"),
            "precision": str(_setting(settings, "precision", "fp64")),
        })
        if method in DIRECT_CD_METHODS:
            # These controls can change the direct integral or projector
            # approximation.  Keep them distinct in the scientific identity;
            # layout-only block sizes remain part of the execution protocol.
            payload.update({
                "direct_scf_tol": float(
                    _setting(settings, "direct_scf_tol", 1.0e-13)
                ),
                "cd_max_rank": _setting(settings, "cd_max_rank"),
                "denominator_tolerance": float(
                    _setting(settings, "denominator_tolerance", 1.0e-10)
                ),
                "denominator_max_rank": _setting(
                    settings, "denominator_max_rank"
                ),
                "rr_solver_tolerance": float(
                    _setting(settings, "rr_solver_tolerance", 1.0e-10)
                ),
                "rr_solver_maxiter": _setting(settings, "rr_solver_maxiter"),
                "rr_dense_fallback_dimension": int(
                    _setting(settings, "rr_dense_fallback_dimension", 64)
                ),
                "rr_ritz_residual_tolerance": _setting(
                    settings, "rr_ritz_residual_tolerance"
                ),
            })
        if method in {"thc_cd", "thc_canonical"}:
            payload.update({
                "thc_fit_tol": float(
                    _setting(settings, "thc_fit_tol", 1.0e-6)
                ),
                "thc_rank": _setting(settings, "thc_rank"),
                "thc_initial_rank": _setting(settings, "thc_initial_rank"),
                "thc_max_rank": _setting(settings, "thc_max_rank"),
                "thc_rank_growth": float(
                    _setting(settings, "thc_rank_growth", 1.5)
                ),
                "thc_orthogonality_cutoff": float(_setting(
                    settings, "thc_orthogonality_cutoff", 1.0e-12
                )),
                "thc_orthogonality_tolerance": float(_setting(
                    settings, "thc_orthogonality_tolerance", 1.0e-10
                )),
                "thc_max_iterations": int(
                    _setting(settings, "thc_max_iterations", 500)
                ),
                "thc_als_convergence_tolerance": float(_setting(
                    settings, "thc_als_convergence_tolerance", 1.0e-10
                )),
                "thc_ridge": float(
                    _setting(settings, "thc_ridge", 1.0e-12)
                ),
                "thc_seed": int(_setting(settings, "thc_seed", 0)),
                "thc_replacement_validation_atol": _setting(
                    settings, "thc_replacement_validation_atol"
                ),
                "thc_replacement_validation_rtol": _setting(
                    settings, "thc_replacement_validation_rtol"
                ),
            })
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return payload, f"{scientific_method}:v1:{digest}"


def _version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _topology() -> str | None:
    try:
        return subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, timeout=10)
    except Exception:
        return None


def _command_text(arguments: list[str]) -> str | None:
    try:
        return subprocess.check_output(arguments, text=True, timeout=10).strip()
    except Exception:
        return None


def _cpu_model() -> str | None:
    text = _command_text(["lscpu", "-J"])
    if text:
        try:
            payload = json.loads(text)
            for item in payload.get("lscpu", []):
                if str(item.get("field", "")).rstrip(":") == "Model name":
                    return str(item.get("data"))
        except Exception:
            pass
    return platform.processor() or None


def _pcie_info() -> dict[str, Any]:
    # The MTU driver does not expose pci.link.* through nvidia-smi.  Query the
    # stable identity fields there and obtain negotiated/max link properties
    # from the Linux PCI sysfs entries instead.
    fields = "name,uuid,memory.total,driver_version,pci.bus_id"
    text = _command_text([
        "nvidia-smi", f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ])
    if not text:
        return {"available": False}
    rows = []
    for line in text.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 5:
            row = dict(zip(
                ("name", "uuid", "memory_total_mib", "driver_version",
                 "pci_bus_id"),
                values,
            ))
            bus_id = row["pci_bus_id"].lower()
            if bus_id.startswith("00000000:"):
                bus_id = "0000:" + bus_id.split(":", 1)[1]
            device = Path("/sys/bus/pci/devices") / bus_id
            row["sysfs_bus_id"] = bus_id
            for name in (
                "current_link_speed",
                "current_link_width",
                "max_link_speed",
                "max_link_width",
                "numa_node",
            ):
                try:
                    row[name] = (device / name).read_text().strip()
                except OSError:
                    row[name] = None
            row["pcie_gen_current"] = _pcie_generation(
                row["current_link_speed"]
            )
            row["pcie_gen_max"] = _pcie_generation(row["max_link_speed"])
            row["pcie_width_current"] = row["current_link_width"]
            row["pcie_width_max"] = row["max_link_width"]
            rows.append(row)
    return {"available": bool(rows), "visible_gpus": rows}


def _pcie_generation(speed: Any) -> str | None:
    """Translate the Linux PCI link rate into a PCIe generation label."""

    try:
        rate = float(str(speed).split()[0])
    except (TypeError, ValueError, IndexError):
        return None
    rates = ((2.5, "1"), (5.0, "2"), (8.0, "3"), (16.0, "4"),
             (32.0, "5"), (64.0, "6"))
    for expected, generation in rates:
        if abs(rate - expected) <= max(1.0e-6, expected * 1.0e-3):
            return generation
    return None


def _source_tree_digest(root: Path) -> tuple[str, int]:
    """Use the shared deployment snapshot source digest protocol."""

    return _snapshot_source_tree_digest(Path(root))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_task_root() -> Path | None:
    value = os.getenv("CCSD_TASK_ROOT")
    return None if not value else Path(value).expanduser().resolve()


def _expected_deployment_profile() -> str | None:
    return os.getenv("CCSD_EXPECTED_DEPLOYMENT_PROFILE") or None


def _require_canonical_pristine() -> bool:
    return os.getenv(
        "CCSD_REQUIRE_CANONICAL_PRISTINE", ""
    ).strip().lower() in {"1", "true", "yes"}


def _runtime_manifest_valid(manifest: Any) -> bool:
    if not isinstance(manifest, dict):
        return False
    runtime = manifest.get("runtime_binaries")
    if not isinstance(runtime, dict) or runtime.get("complete") is not True:
        return False
    files = runtime.get("files")
    if not isinstance(files, list) or not files:
        return False
    listed: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            return False
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
            return False
        listed.add(relative.as_posix())
    return True


def _runtime_inventory_valid(evidence: dict[str, Any]) -> bool:
    reasons = evidence.get("reasons")
    return bool(
        _runtime_manifest_valid(evidence.get("manifest"))
        and isinstance(reasons, list)
        and not any("runtime binary" in reason or "runtime-binary" in reason
                    for reason in reasons)
    )


def _snapshot_at_start(root: Path) -> dict[str, Any]:
    expected_profile = _expected_deployment_profile()
    require_canonical = _require_canonical_pristine()
    evidence = validate_snapshot(
        root,
        expected_base_revision=FROZEN_BASE_COMMIT,
        expected_task_root=_expected_task_root(),
        expected_deployment_profile=expected_profile,
        require_canonical_pristine=require_canonical,
    )
    if expected_profile is None:
        evidence["valid"] = False
        evidence.setdefault("reasons", []).append(
            "expected deployment profile is not configured"
        )
    manifest_path = evidence.get("manifest_path")
    try:
        manifest_sha256 = _file_sha256(str(manifest_path))
    except Exception as exc:
        manifest_sha256 = None
        evidence["manifest_hash_error_at_start"] = repr(exc)
    evidence["valid_at_start"] = evidence.get("valid") is True
    evidence["read_only_at_start"] = evidence.get("read_only") is True
    evidence["runtime_binaries_valid_at_start"] = _runtime_inventory_valid(
        evidence
    )
    evidence["manifest_sha256_at_start"] = manifest_sha256
    task_root = _expected_task_root()
    evidence["expected_task_root"] = None if task_root is None else str(task_root)
    return evidence


def _formal_source_evidence_valid(source: dict[str, Any]) -> bool:
    digest = source.get("tree_sha256")
    snapshot = source.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    manifest = snapshot.get("manifest")
    if not isinstance(manifest, dict):
        return False
    snapshot_root = snapshot.get("snapshot_root")
    repository = source.get("repository")
    manifest_start = snapshot.get("manifest_sha256_at_start")
    return bool(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        and source.get("tree_sha256_at_start") == digest
        and source.get("tree_sha256_at_end") == digest
        and source.get("stable_during_run") is True
        and source.get("base_revision") == FROZEN_BASE_COMMIT
        and snapshot.get("schema") == "gpu4pyscf.snapshot-evidence.v1"
        and snapshot.get("valid") is True
        and snapshot.get("valid_at_start") is True
        and snapshot.get("valid_at_end") is True
        and snapshot.get("read_only") is True
        and snapshot.get("read_only_at_start") is True
        and snapshot.get("read_only_at_end") is True
        and snapshot.get("runtime_binaries_valid_at_start") is True
        and snapshot.get("runtime_binaries_valid_at_end") is True
        and snapshot.get("writable_paths") == []
        and snapshot.get("reasons") == []
        and snapshot.get("reasons_at_end") == []
        and snapshot.get("stable_during_run") is True
        and snapshot.get("tree_sha256") == digest
        and snapshot.get("tree_sha256_at_end") == digest
        and isinstance(manifest_start, str)
        and len(manifest_start) == 64
        and manifest_start == snapshot.get("manifest_sha256_at_end")
        and manifest.get("schema") == "gpu4pyscf.source-snapshot.v2"
        and manifest.get("tree_sha256") == digest
        and manifest.get("base_revision") == FROZEN_BASE_COMMIT
        and manifest.get("immutable") is True
        and _runtime_manifest_valid(manifest)
        and snapshot.get("expected_deployment_profile")
        in {CANDIDATE_DEPLOYMENT_PROFILE, G0_DEPLOYMENT_PROFILE}
        and isinstance(snapshot.get("canonical_pristine_required"), bool)
        and isinstance(manifest.get("local_provenance"), dict)
        and manifest["local_provenance"].get("deployment_profile")
        == snapshot.get("expected_deployment_profile")
        and (
            snapshot.get("expected_deployment_profile")
            != G0_DEPLOYMENT_PROFILE
            or snapshot.get("canonical_pristine_required") is True
            and manifest["local_provenance"].get("canonical_pristine") is True
        )
        and manifest.get("source") == repository
        and snapshot.get("source") == repository
        and isinstance(snapshot_root, str)
        and Path(snapshot_root).parent.name == "snapshots"
        and Path(snapshot_root).name == digest
        and snapshot.get("manifest_path")
        == str(Path(snapshot_root) / "manifest.json")
        and Path(str(repository)).name == "source"
        and Path(str(repository)).parent == Path(snapshot_root)
    )


def _git_state() -> dict[str, Any]:
    def run_git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()

    tree_sha256, files_hashed = _source_tree_digest(REPOSITORY_ROOT)
    snapshot = _snapshot_at_start(REPOSITORY_ROOT)
    base_revision = os.getenv("FROZEN_GPU4PYSCF_COMMIT", FROZEN_BASE_COMMIT)
    try:
        revision = run_git("rev-parse", "HEAD")
        status = run_git("status", "--porcelain=v1", "--untracked-files=all")
        return {
            "repository": str(REPOSITORY_ROOT),
            "revision": revision,
            "base_revision": base_revision,
            "dirty": bool(status),
            "dirty_paths": [line[3:] for line in status.splitlines()],
            "tree_sha256": tree_sha256,
            "tree_sha256_at_start": tree_sha256,
            "files_hashed": files_hashed,
            "source_kind": "git-working-tree",
            "snapshot": snapshot,
            "formal_performance_eligible": False,
        }
    except Exception as exc:
        return {
            "repository": str(REPOSITORY_ROOT),
            "revision": f"tree-sha256:{tree_sha256}",
            "base_revision": base_revision,
            "dirty": True,
            "dirty_paths": None,
            "tree_sha256": tree_sha256,
            "tree_sha256_at_start": tree_sha256,
            "files_hashed": files_hashed,
            "source_kind": "content-manifest",
            "git_error": repr(exc),
            "snapshot": snapshot,
            "formal_performance_eligible": False,
        }


def _record_source_stability(record: dict[str, Any]) -> bool | None:
    """Revalidate the source snapshot and fail performance closed."""

    source = record.setdefault("source", {})
    start_digest = source.get("tree_sha256")
    source["tree_sha256_at_start"] = start_digest
    snapshot = source.setdefault("snapshot", {})
    try:
        end_digest, end_count = _source_tree_digest(REPOSITORY_ROOT)
    except Exception as exc:
        source["tree_sha256_at_end"] = None
        source["stable_during_run"] = None
        source["stability_check_error"] = repr(exc)
        stable: bool | None = None
    else:
        source["tree_sha256_at_end"] = end_digest
        source["files_hashed_at_end"] = end_count
        tree_stable = bool(start_digest is not None and start_digest == end_digest)
        try:
            end_evidence = validate_snapshot(
                REPOSITORY_ROOT,
                expected_base_revision=FROZEN_BASE_COMMIT,
                expected_task_root=_expected_task_root(),
                expected_deployment_profile=snapshot.get(
                    "expected_deployment_profile"
                ),
                require_canonical_pristine=bool(
                    snapshot.get("canonical_pristine_required")
                ),
            )
        except Exception as exc:
            end_evidence = {
                "valid": False,
                "read_only": False,
                "tree_sha256": None,
                "files_hashed": None,
                "reasons": [f"snapshot validation at end failed: {exc!r}"],
            }
        try:
            manifest_end = _file_sha256(str(snapshot.get("manifest_path")))
        except Exception as exc:
            manifest_end = None
            snapshot["manifest_hash_error_at_end"] = repr(exc)
        manifest_stable = bool(
            snapshot.get("manifest_sha256_at_start") is not None
            and snapshot.get("manifest_sha256_at_start") == manifest_end
        )
        stable = bool(
            tree_stable
            and manifest_stable
            and snapshot.get("valid_at_start") is True
            and end_evidence.get("valid") is True
        )
        source["stable_during_run"] = stable
        snapshot.update({
            "valid_at_end": end_evidence.get("valid") is True,
            "read_only_at_end": end_evidence.get("read_only") is True,
            "runtime_binaries_valid_at_end": _runtime_inventory_valid(
                end_evidence
            ),
            "tree_sha256_at_end": end_evidence.get("tree_sha256"),
            "files_hashed_at_end": end_evidence.get("files_hashed"),
            "manifest_sha256_at_end": manifest_end,
            "stable_during_run": stable,
            "reasons_at_end": end_evidence.get("reasons", []),
        })
    formal = _formal_source_evidence_valid(source)
    source["formal_performance_eligible"] = formal
    if not formal:
        record["performance_eligible"] = False
        if stable is False:
            reason = "source snapshot or manifest changed during the timed run"
        elif stable is None:
            reason = "source stability could not be verified at run end"
        else:
            reason = "source is not a validated immutable v2 snapshot"
        limitations = record.setdefault("limitations", [])
        if reason not in limitations:
            limitations.append(reason)
    return stable


def _apply_source_start_eligibility(record: dict[str, Any]) -> None:
    snapshot = (record.get("source") or {}).get("snapshot") or {}
    if snapshot.get("valid_at_start") is True:
        return
    record["performance_eligible"] = False
    limitation = "source is not a validated immutable v2 snapshot"
    limitations = record.setdefault("limitations", [])
    if limitation not in limitations:
        limitations.append(limitation)


def _gpu_info(cp: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        info["available"] = count > 0
        info["device_count"] = count
        if count:
            props = cp.cuda.runtime.getDeviceProperties(0)
            info["device_0"] = {
                key: (value.decode() if isinstance(value, bytes) else int(value) if isinstance(value, int) else str(value))
                for key, value in props.items()
                if key in {"name", "totalGlobalMem", "major", "minor"}
            }
            info["runtime_version"] = int(cp.cuda.runtime.runtimeGetVersion())
    except Exception as exc:
        info["error"] = repr(exc)
    return info


class _HBMMonitor:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.telemetry_samples: list[dict[str, Any]] = []
        self.post_hf_started_at: float | None = None
        self.post_hf_finished_at: float | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        pid = str(os.getpid())
        poll = 0
        while not self.stop.is_set():
            try:
                text = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory,gpu_uuid", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=4,
                )
                for line in text.splitlines():
                    fields = [value.strip() for value in line.split(",")]
                    if len(fields) == 3 and fields[0] == pid and fields[1].isdigit():
                        self.samples.append({"time": time.time(), "MiB": int(fields[1]), "gpu_uuid": fields[2]})
            except Exception:
                pass
            # Clocks and utilisation change more slowly than process HBM.
            # Sample them every fourth memory poll so the existing 0.5 s HBM
            # cadence does not launch two nvidia-smi processes on every pass.
            if poll % 4 == 0:
                try:
                    text = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=uuid,utilization.gpu,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,pstate",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=4,
                    )
                    self.telemetry_samples.extend(
                        _parse_gpu_telemetry(text, sampled_at=time.time())
                    )
                except Exception:
                    pass
            poll += 1
            self.stop.wait(0.5)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)


def _optional_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_gpu_telemetry(
    text: str, *, sampled_at: float
) -> list[dict[str, Any]]:
    """Parse stable, unit-free nvidia-smi device telemetry rows."""

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 7 or not values[0]:
            continue
        rows.append({
            "time": float(sampled_at),
            "gpu_uuid": values[0],
            "gpu_utilization_percent": _optional_float(values[1]),
            "sm_clock_mhz": _optional_float(values[2]),
            "memory_clock_mhz": _optional_float(values[3]),
            "power_draw_w": _optional_float(values[4]),
            "temperature_c": _optional_float(values[5]),
            "pstate": values[6] or None,
        })
    return rows


def _telemetry_metric(
    samples: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    values = sorted(
        float(sample[key])
        for sample in samples
        if sample.get(key) is not None
    )
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    midpoint = len(values) // 2
    median = (
        values[midpoint]
        if len(values) % 2
        else 0.5 * (values[midpoint - 1] + values[midpoint])
    )
    return {
        "count": len(values),
        "min": values[0],
        "median": median,
        "mean": sum(values) / len(values),
        "max": values[-1],
    }


def _summarize_gpu_telemetry(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample.get("gpu_uuid")), []).append(sample)
    devices: dict[str, Any] = {}
    for gpu_uuid, device_samples in sorted(grouped.items()):
        utilization = _telemetry_metric(
            device_samples, "gpu_utilization_percent"
        )
        active_count = sum(
            1
            for item in device_samples
            if item.get("gpu_utilization_percent") is not None
            and float(item["gpu_utilization_percent"]) > 0.0
        )
        pstates: dict[str, int] = {}
        for item in device_samples:
            if item.get("pstate") is not None:
                state = str(item["pstate"])
                pstates[state] = pstates.get(state, 0) + 1
        devices[gpu_uuid] = {
            "sample_count": len(device_samples),
            "gpu_utilization_percent": utilization,
            "active_sample_fraction": (
                None
                if utilization["count"] == 0
                else active_count / int(utilization["count"])
            ),
            "sm_clock_mhz": _telemetry_metric(
                device_samples, "sm_clock_mhz"
            ),
            "memory_clock_mhz": _telemetry_metric(
                device_samples, "memory_clock_mhz"
            ),
            "power_draw_w": _telemetry_metric(
                device_samples, "power_draw_w"
            ),
            "temperature_c": _telemetry_metric(
                device_samples, "temperature_c"
            ),
            "pstate_counts": pstates,
        }
    return {"sample_count": len(samples), "devices": devices}


def _gpu_telemetry_payload(monitor: _HBMMonitor) -> dict[str, Any]:
    process_gpu_uuids = sorted({
        str(sample["gpu_uuid"])
        for sample in monitor.samples
        if sample.get("gpu_uuid")
    })
    attributed_samples = [
        sample
        for sample in monitor.telemetry_samples
        if sample.get("gpu_uuid") in process_gpu_uuids
    ]
    unattributed_device_uuids = sorted({
        str(sample["gpu_uuid"])
        for sample in monitor.telemetry_samples
        if sample.get("gpu_uuid") not in process_gpu_uuids
    })
    post_hf_samples = [
        sample
        for sample in attributed_samples
        if monitor.post_hf_started_at is not None
        and monitor.post_hf_finished_at is not None
        and monitor.post_hf_started_at
        <= float(sample["time"])
        <= monitor.post_hf_finished_at
    ]
    return {
        "sampling_interval_s": 2.0,
        "source": "nvidia-smi device sampling",
        "attribution": {
            "eligible": bool(process_gpu_uuids),
            "process_gpu_uuids": process_gpu_uuids,
            "unattributed_device_uuids": unattributed_device_uuids,
            "policy": "process-compute-app-uuid-match",
        },
        "overall": _summarize_gpu_telemetry(attributed_samples),
        "post_hf": _summarize_gpu_telemetry(post_hf_samples),
    }


def _sync(cp: Any) -> None:
    if cp is not None:
        cp.cuda.get_current_stream().synchronize()


def _array(value: Any, np_module: Any) -> Any:
    return value.get() if hasattr(value, "get") else np_module.asarray(value)


def _checkpoint_array(
    value: Any,
    np_module: Any,
    transfer_counter: Any,
    *,
    operation: str,
) -> Any:
    """Copy one checkpoint array to the host with an explicit ledger label."""

    if hasattr(value, "get"):
        if transfer_counter is None:
            raise RuntimeError(
                "device checkpoint copies require an explicit transfer ledger"
            )
        host = value.get()
        transfer_counter.record_d2h(
            int(getattr(value, "nbytes", host.nbytes)),
            operation=operation,
        )
        return np_module.asarray(host)
    return np_module.asarray(value)


def _energy_payload(method: str, solver: Any, cc_obj: Any) -> dict[str, Any]:
    """Return the declared method energy, retaining raw FNO values."""

    raw_corr = float(cc_obj.e_corr)
    raw_total = float(cc_obj.e_tot)
    if method == "fno":
        return {
            "e_corr": float(solver.corrected_e_corr),
            "e_tot": float(solver.corrected_e_tot),
            "energy_definition": (
                "FNO-CCSD + Delta-MP2 (full-space MP2 minus FNO-space MP2)"
            ),
            "corrected_e_corr": float(solver.corrected_e_corr),
            "corrected_e_tot": float(solver.corrected_e_tot),
            "raw_fno_e_corr": raw_corr,
            "raw_fno_e_tot": raw_total,
            "delta_mp2": float(solver.delta_mp2),
        }
    return {"e_corr": raw_corr, "e_tot": raw_total}


def _norm_without_bulk_host_copy(
    value: Any,
    np_module: Any,
    transfer_counter: Any,
    *,
    operation: str,
) -> float:
    """Evaluate a norm on its resident device and download only the scalar."""

    if not hasattr(value, "get"):
        return float(np_module.linalg.norm(np_module.asarray(value)))
    if transfer_counter is None:
        raise RuntimeError(
            "device amplitude summaries require an explicit transfer ledger"
        )
    module_name = type(value).__module__.split(".", 1)[0]
    if module_name != "cupy":
        raise TypeError(
            f"unsupported device array module for amplitude summary: {module_name}"
        )
    import cupy

    scalar = cupy.linalg.norm(value)
    host_scalar = scalar.get()
    transfer_counter.record_d2h(
        int(scalar.nbytes), operation=operation
    )
    return float(np_module.asarray(host_scalar))


def _compressed_metadata_without_host_copy(
    doubles: Any, *, eigenvalues_in_checkpoint: bool = False
) -> dict[str, Any]:
    """Describe compressed doubles using shape/scalar attributes only."""

    projector = doubles.projector
    representation = (
        "thc-rr-doubles" if hasattr(doubles, "collocation") else "rr-doubles"
    )
    payload = {
        "representation": representation,
        "rank": int(projector.vectors.shape[1]),
        "nocc": int(doubles.nocc),
        "nvir": int(doubles.nvir),
        "storage_nbytes": int(getattr(doubles, "storage_nbytes", 0)),
        "projector": {
            "source": getattr(projector, "source", None),
            "cutoff": getattr(projector, "cutoff", None),
            "full_dimension": int(projector.vectors.shape[0]),
            "rank": int(projector.vectors.shape[1]),
            "eigenvalues_in_checkpoint": bool(eigenvalues_in_checkpoint),
        },
    }
    if representation == "thc-rr-doubles":
        payload.update({
            "thc_rank": int(doubles.collocation.shape[1]),
            "fit_tolerance": getattr(doubles, "fit_tolerance", None),
            "fit_residual": getattr(doubles, "fit_residual", None),
            "factorization_method": getattr(
                doubles, "factorization_method", None
            ),
            "paper_exact": getattr(doubles, "paper_exact", None),
        })
    return payload


def _amplitude_payload(
    kernel_result: Any,
    cc_obj: Any,
    np_module: Any,
    *,
    cached_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize amplitudes without reconstructing compressed doubles."""

    if cached_payload is not None:
        return dict(cached_payload)
    t1 = kernel_result[1] if isinstance(kernel_result, tuple) else cc_obj.t1
    doubles = kernel_result[2] if isinstance(kernel_result, tuple) else cc_obj.t2
    run_metrics = getattr(cc_obj, "run_metrics", None)
    transfer_counter = getattr(run_metrics, "transfers", None)
    t1_norm = _norm_without_bulk_host_copy(
        t1,
        np_module,
        transfer_counter,
        operation="amplitude_summary_t1_norm",
    )
    if hasattr(doubles, "reconstruct_t2") and hasattr(doubles, "metadata"):
        if hasattr(doubles, "reconstruct_rr_core"):
            effective_core = doubles.reconstruct_rr_core()
        else:
            effective_core = doubles.core
        t2_norm = _norm_without_bulk_host_copy(
            effective_core,
            np_module,
            transfer_counter,
            operation="amplitude_summary_rr_core_norm",
        )
        return {
            "t1_shape": list(t1.shape),
            "t2_shape": [doubles.nocc, doubles.nocc, doubles.nvir, doubles.nvir],
            "t1_norm": t1_norm,
            "t2_norm": t2_norm,
            "dense_t2_materialized_for_summary": False,
            "summary_transfer_mode": "device-norm-scalars-only",
            "doubles_metadata": _compressed_metadata_without_host_copy(doubles),
        }
    t2_norm = _norm_without_bulk_host_copy(
        doubles,
        np_module,
        transfer_counter,
        operation="amplitude_summary_t2_norm",
    )
    return {
        "t1_shape": list(t1.shape),
        "t2_shape": list(doubles.shape),
        "t1_norm": t1_norm,
        "t2_norm": t2_norm,
        "dense_t2_materialized_for_summary": False,
        "summary_transfer_mode": "device-norm-scalars-only",
    }


def _write_checkpoint_once(
    path: Path,
    method: str,
    kernel_result: Any,
    cc_obj: Any,
    np_module: Any,
    orbital_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a restart payload without reconstructing compressed doubles."""

    transfer_counter = getattr(
        getattr(cc_obj, "run_metrics", None), "transfers", None
    )
    t1 = _checkpoint_array(
        kernel_result[1] if isinstance(kernel_result, tuple) else cc_obj.t1,
        np_module,
        transfer_counter,
        operation="checkpoint_t1",
    )
    doubles = kernel_result[2] if isinstance(kernel_result, tuple) else cc_obj.t2
    checkpoint_schema = (
        RESTART_SCHEMA_V2 if orbital_context is not None else RESTART_SCHEMA_V1
    )
    arrays: dict[str, Any] = {
        "checkpoint_schema": np_module.asarray([checkpoint_schema]),
        "e_corr": np_module.asarray([float(cc_obj.e_corr)]),
        "t1": t1,
        "method": np_module.asarray([method]),
    }
    if orbital_context is not None:
        orbital_arrays = orbital_context.get("_arrays")
        identity = orbital_context.get("identity")
        fingerprint = orbital_context.get("orbital_fingerprint")
        if (
            not isinstance(orbital_arrays, dict)
            or not isinstance(identity, dict)
            or not isinstance(fingerprint, str)
            or _orbital_fingerprint(identity, orbital_arrays) != fingerprint
        ):
            raise ValueError("invalid canonical orbital context for checkpoint")
        arrays.update({
            "mo_coeff": np_module.asarray(orbital_arrays["mo_coeff"]),
            "mo_occ": np_module.asarray(orbital_arrays["mo_occ"]),
            "mo_energy": np_module.asarray(orbital_arrays["mo_energy"]),
            "orbital_identity_json": np_module.asarray([
                _canonical_json(identity)
            ]),
            "orbital_fingerprint": np_module.asarray([fingerprint]),
            "orbital_artifact_sha256": np_module.asarray([
                orbital_context.get("artifact_sha256") or ""
            ]),
            "orbital_origin": np_module.asarray([
                str(orbital_context.get("mode", "unknown"))
            ]),
        })
    representation = "dense-t2"
    doubles_metadata = None
    if hasattr(doubles, "projector") and hasattr(doubles, "core"):
        representation = (
            "thc-rr-doubles" if hasattr(doubles, "collocation") else "rr-doubles"
        )
        arrays.update({
            "rr_projector": _checkpoint_array(
                doubles.projector.vectors,
                np_module,
                transfer_counter,
                operation="checkpoint_rr_projector",
            ),
            "rr_eigenvalues": _checkpoint_array(
                doubles.projector.eigenvalues,
                np_module,
                transfer_counter,
                operation="checkpoint_rr_eigenvalues",
            ),
            "compressed_core": _checkpoint_array(
                doubles.core,
                np_module,
                transfer_counter,
                operation="checkpoint_rr_core",
            ),
            "nocc": np_module.asarray([int(doubles.nocc)]),
            "nvir": np_module.asarray([int(doubles.nvir)]),
        })
        if hasattr(doubles, "collocation"):
            arrays["thc_collocation"] = _checkpoint_array(
                doubles.collocation,
                np_module,
                transfer_counter,
                operation="checkpoint_thc_collocation",
            )
            effective_core = (
                arrays["thc_collocation"]
                @ arrays["compressed_core"]
                @ arrays["thc_collocation"].T.conj()
            )
        else:
            effective_core = arrays["compressed_core"]
        doubles_metadata = _compressed_metadata_without_host_copy(
            doubles, eigenvalues_in_checkpoint=True
        )
        amplitude_payload = {
            "t1_shape": list(t1.shape),
            "t2_shape": [
                int(doubles.nocc),
                int(doubles.nocc),
                int(doubles.nvir),
                int(doubles.nvir),
            ],
            "t1_norm": float(np_module.linalg.norm(t1)),
            "t2_norm": float(np_module.linalg.norm(effective_core)),
            "dense_t2_materialized_for_summary": False,
            "summary_transfer_mode": (
                "reused-counted-checkpoint-host-buffers"
            ),
            "doubles_metadata": doubles_metadata,
        }
    else:
        arrays["t2"] = _checkpoint_array(
            doubles,
            np_module,
            transfer_counter,
            operation="checkpoint_dense_t2",
        )
        amplitude_payload = {
            "t1_shape": list(t1.shape),
            "t2_shape": list(arrays["t2"].shape),
            "t1_norm": float(np_module.linalg.norm(t1)),
            "t2_norm": float(np_module.linalg.norm(arrays["t2"])),
            "dense_t2_materialized_for_summary": False,
            "summary_transfer_mode": (
                "reused-counted-checkpoint-host-buffers"
            ),
        }
    arrays["representation"] = np_module.asarray([representation])
    with path.open("xb") as stream:
        np_module.savez(stream, **arrays)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "checkpoint_schema": checkpoint_schema,
        "representation": representation,
        "array_keys": sorted(arrays),
        "included_in_post_hf": True,
        "dense_t2_reconstructed": False,
        "orbital_fingerprint": (
            None if orbital_context is None
            else orbital_context["orbital_fingerprint"]
        ),
        "orbital_artifact_sha256": (
            None if orbital_context is None
            else orbital_context.get("artifact_sha256")
        ),
        "_cached_amplitude_payload": amplitude_payload,
    }


def _rr_cd_constructor_options(args: argparse.Namespace) -> dict[str, Any]:
    """Map the benchmark protocol one-to-one onto the public RRCCSD API."""

    return {
        "eri_backend": "cd",
        "eri_tol": args.eri_tol,
        "direct_scf_tol": args.direct_scf_tol,
        "cd_max_rank": args.cd_max_rank,
        "gint_column_backend": args.gint_column_backend,
        "gint_group_size": args.gint_group_size,
        "gint_max_block_bytes": args.gint_max_block_bytes,
        "gint_max_batch_size": args.gint_max_batch_size,
        "cd_mo_block_size": args.cd_mo_block_size,
        "rr_eig_cutoff": args.rr_eig_cutoff,
        "rr_max_rank": args.rr_max_rank,
        "denominator_tolerance": args.denominator_tolerance,
        "denominator_max_rank": args.denominator_max_rank,
        "rr_initial_rank": args.rr_initial_rank,
        "rr_solver_tolerance": args.rr_solver_tolerance,
        "rr_solver_maxiter": args.rr_solver_maxiter,
        "rr_dense_fallback_dimension": args.rr_dense_fallback_dimension,
        "rr_ritz_residual_tolerance": args.rr_ritz_residual_tolerance,
        "rr_auxiliary_block_size": args.rr_auxiliary_block_size,
        "rr_virtual_block_size": args.rr_virtual_block_size,
        "precision": args.precision,
    }


def _thc_cd_constructor_options(args: argparse.Namespace) -> dict[str, Any]:
    """Map every direct-CD and THC endpoint control to ``THCRRCCSD``."""

    return {
        **_rr_cd_constructor_options(args),
        "thc_fit_tol": args.thc_fit_tol,
        "thc_rank": args.thc_rank,
        "thc_initial_rank": args.thc_initial_rank,
        "thc_max_rank": args.thc_max_rank,
        "thc_rank_growth": args.thc_rank_growth,
        "thc_orthogonality_cutoff": args.thc_orthogonality_cutoff,
        "thc_orthogonality_tolerance": args.thc_orthogonality_tolerance,
        "thc_max_iterations": args.thc_max_iterations,
        "thc_als_convergence_tolerance": (
            args.thc_als_convergence_tolerance
        ),
        "thc_ridge": args.thc_ridge,
        "thc_seed": args.thc_seed,
        "thc_replacement_validation_atol": (
            args.thc_replacement_validation_atol
        ),
        "thc_replacement_validation_rtol": (
            args.thc_replacement_validation_rtol
        ),
    }


def _make_solver(args: argparse.Namespace, mf: Any) -> tuple[Any, dict[str, Any]]:
    from gpu4pyscf.cc import ccsd_incore

    if args.method in {"canonical", "canonical_legacy"}:
        solver = ccsd_incore.CCSD(mf)
        _configure_solver(solver, args)
        if args.method == "canonical_legacy":
            solver.resident_iterations = False
        solver.frozen = 0
        metadata = {
            **METHOD_INFO[args.method],
            "resident_iterations": bool(
                getattr(solver, "resident_iterations", False)
            ),
        }
        return solver, metadata
    if args.method == "fno":
        from gpu4pyscf.cc.addons import FNOCCSD

        wrapper = FNOCCSD(
            mf,
            thresh=args.fno_thresh,
            pct_occ=args.fno_pct_occ,
            nvir_act=args.fno_nvir_act,
            use_gpu=args.device == "gpu",
        )
        _configure_solver(wrapper.build(), args)
        return wrapper, {**METHOD_INFO[args.method], "fno_wrapper": True}
    from gpu4pyscf.cc.rrccsd import RRCCSD

    rr_common = {
        "eri_tol": args.eri_tol,
        "rr_eig_cutoff": args.rr_eig_cutoff,
        "rr_max_rank": args.rr_max_rank,
        "precision": args.precision,
    }
    if args.method == "rr_cd":
        solver = RRCCSD(mf, **_rr_cd_constructor_options(args))
        _configure_solver(solver, args)
        solver.frozen = 0
        return solver, METHOD_INFO[args.method]
    if args.method == "thc_cd":
        from gpu4pyscf.cc.thc_rrccsd import THCRRCCSD

        solver = THCRRCCSD(mf, **_thc_cd_constructor_options(args))
        _configure_solver(solver, args)
        solver.frozen = 0
        return solver, METHOD_INFO[args.method]
    if args.method == "rr_canonical":
        solver = RRCCSD(mf, eri_backend="canonical", **rr_common)
        _configure_solver(solver, args)
        solver.frozen = 0
        return solver, METHOD_INFO[args.method]
    from gpu4pyscf.cc.thc_rrccsd import THCRRCCSD

    solver = THCRRCCSD(
        mf,
        thc_fit_tol=args.thc_fit_tol,
        thc_max_rank=args.thc_max_rank,
        eri_backend="canonical",
        **rr_common,
    )
    _configure_solver(solver, args)
    solver.frozen = 0
    return solver, METHOD_INFO[args.method]


def _configure_solver(solver: Any, args: argparse.Namespace) -> Any:
    """Apply PySCF runtime controls after constructing a CC object.

    PySCF CC constructors accept structural arguments such as ``frozen`` but
    do not accept convergence settings as keyword arguments.  Keeping these
    assignments explicit also guarantees identical controls for canonical,
    FNO, RR, and THC benchmark entries.
    """

    solver.conv_tol = float(args.cc_conv_tol)
    solver.conv_tol_normt = float(args.cc_conv_tol_normt)
    solver.max_cycle = int(args.max_cycle)
    solver.max_memory = int(args.max_memory_mb)
    return solver


def _execute_solver_kernel(
    method: str, solver: Any, cc_obj: Any
) -> tuple[Any, Any]:
    """Run one solver lifecycle and return ``(kernel_result, eris)``.

    The direct-CD RR entry owns its integral lifecycle.  Calling ``kernel``
    without a prebuilt ERI object prevents the benchmark harness from taking
    the dense canonical ``ao2mo`` path.  The returned ``None`` also makes it
    impossible to accidentally feed its three-index provider to the dense
    final-residual helper.
    """

    if method in DIRECT_CD_METHODS:
        return cc_obj.kernel(), None
    eris = cc_obj.ao2mo(cc_obj.mo_coeff)
    if method == "fno":
        return solver.kernel(eris=eris), eris
    return cc_obj.kernel(eris=eris), eris


def _timed_resident_diagnostic_scope(method: str, cc_obj: Any) -> Any:
    """Keep canonical resident state alive through its timed residual check."""

    if method != "canonical":
        return contextlib.nullcontext(False)
    scope = getattr(cc_obj, "_resident_post_hf_state_scope", None)
    if not callable(scope):
        raise RuntimeError(
            "canonical resident solver does not expose its diagnostic scope"
        )
    return scope()


def _merge_method_metadata(
    static_metadata: dict[str, Any], runtime: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge labels with one already-accounted solver metadata snapshot."""

    combined = dict(static_metadata)
    static_ineligible = static_metadata.get("performance_eligible") is False
    static_limitations = list(
        static_metadata.get("performance_limitations", [])
    )
    if runtime is not None:
        if not isinstance(runtime, dict):
            raise TypeError("runtime method metadata must be a dictionary")
        combined.update(runtime)
        for reason in runtime.get("performance_limitations", []):
            if reason not in static_limitations:
                static_limitations.append(reason)
    if static_ineligible:
        combined["performance_eligible"] = False
    if static_limitations:
        combined["performance_limitations"] = static_limitations
    return combined


def _runtime_method_metadata(
    solver: Any, static_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Collect and merge auditable solver metadata when no record exists."""

    getter = getattr(solver, "method_metadata", None)
    runtime = None
    if callable(getter):
        runtime = getter()
    return _merge_method_metadata(static_metadata, runtime)


def _apply_method_eligibility(
    record: dict[str, Any], method_metadata: dict[str, Any]
) -> None:
    """Propagate only negative method gates to the run-level decision."""

    if method_metadata.get("performance_eligible") is not False:
        return
    record["performance_eligible"] = False
    limitations = record.setdefault("limitations", [])
    for reason in method_metadata.get("performance_limitations", []):
        if reason not in limitations:
            limitations.append(reason)


def _dense_final_residual(
    cc_obj: Any,
    eris: Any,
    np_module: Any,
    *,
    orbital_space: str = "full",
) -> dict[str, Any]:
    """Evaluate an FP64 equation residual in the solver's active MO space."""

    if orbital_space not in {"full", "fno-active"}:
        raise ValueError(f"unsupported residual orbital space {orbital_space!r}")

    resident_operands = None
    resident_builder = getattr(
        cc_obj, "_resident_final_residual_operands", None
    )
    if callable(resident_builder):
        resident_operands = resident_builder(eris)
    if resident_operands is None:
        current_t1 = _array(cc_obj.t1, np_module)
        current_t2 = _array(cc_obj.t2, np_module)
        check_t1, check_t2 = cc_obj.update_amps(
            cc_obj.t1, cc_obj.t2, eris
        )
        check_t1 = _array(check_t1, np_module)
        check_t2 = _array(check_t2, np_module)
        nocc, nvir = current_t1.shape
        occupied = np_module.asarray(eris.mo_energy[:nocc])
        virtual = np_module.asarray(eris.mo_energy[nocc:nocc + nvir])
        scalar_to_float = None
    else:
        current_t1 = resident_operands["current_t1"]
        current_t2 = resident_operands["current_t2"]
        check_t1 = resident_operands["jacobi_t1"]
        check_t2 = resident_operands["jacobi_t2"]
        scalar_to_float = getattr(
            cc_obj, "_resident_residual_scalar_to_float", None
        )
        if not callable(scalar_to_float):
            raise RuntimeError(
                "resident residual operands require an accounted scalar "
                "transfer callback"
            )
        occupied = resident_operands["occupied_energies"]
        virtual = resident_operands["virtual_energies"]
    doubles_block_size = (
        None if resident_operands is None
        else int(resident_operands["doubles_block_size"])
    )
    nocc, nvir = current_t1.shape
    block_hbm_checkpoints = []
    if resident_operands is None:
        block_observer = None
    else:
        checkpoint = getattr(
            cc_obj, "_record_synchronized_hbm_checkpoint", None
        )
        if not callable(checkpoint):
            raise RuntimeError(
                "resident residual requires synchronized HBM checkpoints"
            )

        def block_observer(a0: int, a1: int) -> None:
            block_hbm_checkpoints.append(checkpoint(
                f"final-residual-block-{a0}-{a1}-live"
            ))

    residual = _evaluate_dense_ccsd_residual(
        current_t1,
        current_t2,
        check_t1,
        check_t2,
        occupied,
        virtual,
        level_shift=float(cc_obj.level_shift),
        scalar_to_float=scalar_to_float,
        doubles_block_size=doubles_block_size,
        block_observer=block_observer,
    )
    if resident_operands is not None:
        if not block_hbm_checkpoints:
            raise RuntimeError(
                "resident residual did not record a block HBM checkpoint"
            )
        synchronized_hbm_checkpoints = [
            *resident_operands["update_hbm_checkpoints"],
            *block_hbm_checkpoints,
        ]
        final_residual_hbm_peak = max(
            int(item["driver_used_bytes"])
            for item in synchronized_hbm_checkpoints
        )
        metrics = getattr(cc_obj, "run_metrics", None)
        counters = getattr(metrics, "counters", {})
        whole_stage_hbm_peak = int(counters.get(
            "synchronized_driver_hbm_peak_bytes",
            final_residual_hbm_peak,
        ))
    else:
        final_residual_hbm_peak = None
        whole_stage_hbm_peak = None
        synchronized_hbm_checkpoints = None
    return {
        "available": True,
        "definition": (
            "full-space canonical FP64 equation residual"
            if orbital_space == "full"
            else "FNO active-space FP64 equation residual"
        ),
        "orbital_space": orbital_space,
        "is_full_space": orbital_space == "full",
        "active_nocc": int(nocc),
        "active_nvir": int(nvir),
        "equation_norm": residual.full_space_equation_norm,
        "jacobi_update_norm": residual.full_space_jacobi_update_norm,
        "dense_t2_inputs_present": True,
        "full_size_residual_temporaries_materialized": (
            resident_operands is None
        ),
        "evaluation_backend": (
            "host-dense" if resident_operands is None else "gpu-blockwise-fp64"
        ),
        "doubles_block_size": doubles_block_size,
        "temporary_workspace_bound_bytes": (
            None if resident_operands is None
            else int(resident_operands["temporary_bound_bytes"])
        ),
        "released_staging_bytes": (
            None if resident_operands is None
            else int(resident_operands["released_staging_bytes"])
        ),
        "final_residual_synchronized_driver_hbm_peak_bytes": (
            final_residual_hbm_peak
        ),
        "whole_stage_synchronized_driver_hbm_peak_bytes": (
            whole_stage_hbm_peak
        ),
        "synchronized_hbm_checkpoints": (
            synchronized_hbm_checkpoints
        ),
        "included_in_post_hf": True,
    }


def _labeled_low_rank_residual(cc_obj: Any) -> dict[str, Any]:
    """Read and validate the mandatory RR/THC residual schema.

    Missing or inconsistent labels raise instead of falling back to an
    ambiguous ``residual.norm`` value.  This makes benchmark serialization
    fail closed if an older or partially initialized solver is used.
    """

    getter = getattr(cc_obj, "residual_metadata", None)
    if not callable(getter):
        raise TypeError(
            "RR/THC solver must expose residual_metadata() with labelled norms"
        )
    payload = getter()
    if not isinstance(payload, dict):
        raise TypeError("residual_metadata() must return a dictionary")
    if payload.get("schema") != LOW_RANK_RESIDUAL_SCHEMA:
        raise ValueError("RR/THC residual schema is missing or unsupported")
    if payload.get("acceptance_measurement") != "projected_equation":
        raise ValueError("RR/THC residual acceptance measurement is invalid")
    if "norm" in payload:
        raise ValueError("generic top-level residual.norm is forbidden")

    expected = {
        "projected_equation": (
            "equation-residual", "rr-projected-active-pair", False
        ),
        "full_space_equation": (
            "equation-residual", "full-active-pair", True
        ),
        "projected_jacobi_update": (
            "jacobi-update", "rr-projected-active-pair", False
        ),
        "full_space_jacobi_update": (
            "jacobi-update", "full-active-pair", True
        ),
        "representation_error": (
            "relative-representation-error", "full-active-pair", True
        ),
    }
    measurements = payload.get("measurements")
    if not isinstance(measurements, dict):
        raise ValueError("RR/THC residual measurements are missing")
    for name, (kind, space, is_full_space) in expected.items():
        measurement = measurements.get(name)
        if not isinstance(measurement, dict):
            raise ValueError(f"RR/THC residual measurement {name!r} is missing")
        if (
            measurement.get("kind") != kind
            or measurement.get("space") != space
            or measurement.get("is_full_space") is not is_full_space
        ):
            raise ValueError(
                f"RR/THC residual measurement {name!r} has invalid semantics"
            )
        available = measurement.get("available")
        norm = measurement.get("norm")
        if not isinstance(available, bool):
            raise ValueError(
                f"RR/THC residual measurement {name!r} lacks availability"
            )
        if available:
            if isinstance(norm, bool):
                raise ValueError(
                    f"RR/THC residual measurement {name!r} is not numeric"
                )
            try:
                norm_value = float(norm)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"RR/THC residual measurement {name!r} is not numeric"
                ) from exc
            if not math.isfinite(norm_value) or norm_value < 0.0:
                raise ValueError(
                    f"RR/THC residual measurement {name!r} is invalid"
                )
        elif norm is not None:
            raise ValueError(
                f"unavailable RR/THC residual measurement {name!r} has a norm"
            )

    projected = measurements["projected_equation"]
    if payload.get("available") is not projected["available"]:
        raise ValueError("RR/THC residual availability is inconsistent")
    if payload.get("projected_equation") != projected["norm"]:
        raise ValueError("RR/THC projected residual compatibility value differs")
    return payload


def _full_space_residual_diagnostic(method: str) -> dict[str, Any] | None:
    """Describe whether this driver can form a scientific full-space residual."""

    if method in FULL_SPACE_DIAGNOSTIC_METHODS:
        return {
            "available": True,
            "execution_status": "not-requested",
            "required_for_final_acceptance": True,
            "included_in_normal_timed_path": False,
            "reason": (
                "the direct-CD benchmark preserves compressed RRDoubles during "
                "normal timing; pass --run-full-space-diagnostic to run the "
                "separately timed dense pair-space check"
            ),
        }
    if method != "fno":
        return None
    return {
        "available": False,
        "required_for_this_probe": False,
        "reason": (
            "FNO amplitudes live only in the selected active virtual space; "
            "no validated lifting of the discarded-virtual amplitudes is "
            "defined, so an FNO full-space CCSD residual is not reported"
        ),
    }


def _run_post_hf_full_space_diagnostic(
    method: str,
    cc_obj: Any,
    cp: Any,
    *,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    """Run one explicitly requested diagnostic outside ``post_hf_seconds``."""

    if method not in FULL_SPACE_DIAGNOSTIC_METHODS:
        raise ValueError(
            "--run-full-space-diagnostic is supported only for rr_cd and thc_cd"
        )
    if getattr(cc_obj, "converged", None) is not True:
        raise RuntimeError(
            "the requested full-space diagnostic requires a converged solver"
        )
    callback = getattr(cc_obj, "run_full_space_residual_diagnostic", None)
    if not callable(callback):
        raise TypeError(
            f"{method} solver does not expose run_full_space_residual_diagnostic()"
        )
    _sync(cp)
    started = clock()
    diagnostic = callback()
    _sync(cp)
    elapsed = max(0.0, float(clock() - started))
    metadata_getter = getattr(diagnostic, "metadata", None)
    if not callable(metadata_getter):
        raise TypeError("full-space diagnostic result must expose metadata()")
    metadata = metadata_getter()
    if not isinstance(metadata, dict):
        raise TypeError("full-space diagnostic metadata() must return a dictionary")
    solver_metadata = getattr(cc_obj, "full_space_diagnostic_metadata", None)
    if solver_metadata is not None:
        if not isinstance(solver_metadata, dict):
            raise TypeError(
                "solver full_space_diagnostic_metadata must be a dictionary"
            )
        metadata.update(solver_metadata)
    metadata.update({
        "available": True,
        "execution_status": "completed",
        "requested": True,
        "method": method,
        "kind": "equation-residual",
        "space": "full-active-pair",
        "is_full_space": True,
        "wall_seconds": elapsed,
        "included_in_post_hf": False,
        "included_in_normal_timed_path": False,
    })
    return metadata


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    cfg = json.loads(args.config.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("--config must contain a JSON object")
    valid = vars(args)
    provided = set(getattr(
        args, "_explicit_approximation_controls", ()
    ))
    sources = dict(getattr(args, "_approximation_control_sources", {}))
    for key, value in cfg.items():
        if (
            key not in valid
            or key in {"config", "dry_run"}
            or key.startswith("_")
        ):
            raise ValueError(f"unknown or protected config key: {key}")
        if key in {
            "output_dir", "orbital_artifact_in", "orbital_artifact_out"
        } and value is not None:
            value = Path(value)
        setattr(args, key, value)
        if key in EXPLICIT_APPROXIMATION_CONTROL_NAMES:
            provided.add(key)
            sources[key] = "config"
    args._explicit_approximation_controls = frozenset(provided)
    args._approximation_control_sources = sources
    return args


def _validate_run_configuration(args: argparse.Namespace) -> None:
    if (
        args.orbital_artifact_in is not None
        and args.orbital_artifact_out is not None
    ):
        raise ValueError(
            "--orbital-artifact-in and --orbital-artifact-out are mutually "
            "exclusive"
        )
    if args.orbital_artifact_in is not None:
        if not args.orbital_artifact_in.is_file():
            raise ValueError(
                f"orbital artifact input does not exist: "
                f"{args.orbital_artifact_in}"
            )
    if (
        args.orbital_artifact_out is not None
        and args.orbital_artifact_out.exists()
    ):
        raise ValueError(
            f"refusing to overwrite orbital artifact: "
            f"{args.orbital_artifact_out}"
        )
    if (
        args.run_full_space_diagnostic
        and args.method not in FULL_SPACE_DIAGNOSTIC_METHODS
    ):
        raise ValueError(
            "--run-full-space-diagnostic is supported only for rr_cd and thc_cd"
        )
    if args.method == "fno":
        threshold = args.fno_thresh
        if (
            isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or float(threshold) < 0.0
        ):
            raise ValueError("fno_thresh must be finite and non-negative")
        if args.fno_pct_occ is not None:
            pct_occ = args.fno_pct_occ
            if (
                isinstance(pct_occ, bool)
                or not math.isfinite(float(pct_occ))
                or not 0.0 <= float(pct_occ) <= 1.0
            ):
                raise ValueError("fno_pct_occ must lie in [0, 1]")
        if args.fno_nvir_act is not None and (
            isinstance(args.fno_nvir_act, bool)
            or not isinstance(args.fno_nvir_act, int)
            or args.fno_nvir_act < 1
        ):
            raise ValueError("fno_nvir_act must be a positive integer")
    required_controls = REQUIRED_EXPLICIT_APPROXIMATION_CONTROLS.get(
        args.method, ()
    )
    provided_controls = set(getattr(
        args, "_explicit_approximation_controls", ()
    ))
    missing_controls = [
        name for name in required_controls if name not in provided_controls
    ]
    if missing_controls:
        flags = ", ".join(
            "--" + name.replace("_", "-") for name in missing_controls
        )
        raise ValueError(
            f"{args.method} requires explicit approximation controls via CLI "
            f"or config: {flags}"
        )
    for name in required_controls:
        raw = getattr(args, name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = math.nan
        if isinstance(raw, bool) or not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"explicit approximation control {name} must be finite and "
                "non-negative"
            )
    if args.method in DIRECT_CD_METHODS and args.device != "gpu":
        raise ValueError(
            f"{args.method} requires --device gpu for the GINT/CD provider"
        )
    if args.method not in DIRECT_CD_METHODS:
        return
    if str(args.precision).lower() != "fp64":
        raise ValueError(f"{args.method} currently requires precision=fp64")
    nonnegative = (
        "eri_tol",
        "rr_eig_cutoff",
        "denominator_tolerance",
    )
    optional_nonnegative = ("rr_ritz_residual_tolerance",)
    positive = ("direct_scf_tol", "rr_solver_tolerance")
    for name in nonnegative:
        raw = getattr(args, name)
        if isinstance(raw, bool):
            raise TypeError(f"{name} must be a real scalar")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name in optional_nonnegative:
        raw = getattr(args, name)
        if raw is not None:
            if isinstance(raw, bool):
                raise TypeError(f"{name} must be a real scalar or null")
            if not math.isfinite(float(raw)) or float(raw) < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative or null"
                )
    for name in positive:
        raw = getattr(args, name)
        if isinstance(raw, bool):
            raise TypeError(f"{name} must be a real scalar")
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if float(args.direct_scf_tol) > 1.0:
        raise ValueError("direct_scf_tol must lie in (0, 1]")
    for name in (
        "cd_max_rank",
        "gint_max_block_bytes",
        "rr_max_rank",
        "denominator_max_rank",
        "rr_solver_maxiter",
    ):
        raw = getattr(args, name)
        if raw is not None:
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise TypeError(f"{name} must be an integer or null")
            if raw < 1:
                raise ValueError(f"{name} must be positive or null")
    for name in (
        "gint_group_size",
        "gint_max_batch_size",
        "cd_mo_block_size",
        "rr_initial_rank",
        "rr_auxiliary_block_size",
        "rr_virtual_block_size",
    ):
        raw = getattr(args, name)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(f"{name} must be an integer")
        if raw < 1:
            raise ValueError(f"{name} must be positive")
    if isinstance(args.rr_dense_fallback_dimension, bool) or not isinstance(
        args.rr_dense_fallback_dimension, int
    ):
        raise TypeError("rr_dense_fallback_dimension must be an integer")
    if args.rr_dense_fallback_dimension < 0:
        raise ValueError("rr_dense_fallback_dimension must be non-negative")
    if args.method != "thc_cd":
        return

    if args.thc_rank is None:
        raise ValueError("thc_cd requires an explicit --thc-rank")
    if args.thc_replacement_validation_atol is None:
        raise ValueError(
            "thc_cd requires --thc-replacement-validation-atol"
        )
    if args.thc_replacement_validation_rtol is None:
        raise ValueError(
            "thc_cd requires --thc-replacement-validation-rtol"
        )
    for name in ("thc_rank", "thc_initial_rank", "thc_max_rank"):
        raw = getattr(args, name)
        if raw is not None:
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise TypeError(f"{name} must be an integer or null")
            if raw < 1:
                raise ValueError(f"{name} must be positive or null")
    if args.thc_initial_rank is not None:
        raise ValueError(
            "thc_initial_rank is invalid for the fixed-rank thc_cd endpoint"
        )
    if args.thc_max_rank is not None and args.thc_rank > args.thc_max_rank:
        raise ValueError("thc_rank cannot exceed thc_max_rank")
    for name in (
        "thc_fit_tol",
        "thc_orthogonality_tolerance",
        "thc_als_convergence_tolerance",
        "thc_ridge",
        "thc_replacement_validation_atol",
        "thc_replacement_validation_rtol",
    ):
        raw = getattr(args, name)
        if isinstance(raw, bool) or not math.isfinite(float(raw)):
            raise ValueError(f"{name} must be a finite non-negative scalar")
        if float(raw) < 0.0:
            raise ValueError(f"{name} must be a finite non-negative scalar")
    for name in ("thc_rank_growth", "thc_orthogonality_cutoff"):
        raw = getattr(args, name)
        if (
            isinstance(raw, bool)
            or not math.isfinite(float(raw))
            or float(raw) <= 0.0
        ):
            raise ValueError(f"{name} must be a finite positive scalar")
    if float(args.thc_rank_growth) <= 1.0:
        raise ValueError("thc_rank_growth must be greater than one")
    if (
        isinstance(args.thc_max_iterations, bool)
        or not isinstance(args.thc_max_iterations, int)
        or args.thc_max_iterations < 1
    ):
        raise ValueError("thc_max_iterations must be a positive integer")
    if isinstance(args.thc_seed, bool) or not isinstance(args.thc_seed, int):
        raise TypeError("thc_seed must be an integer")

    expected = load_cases()[args.case].get("expected_dimensions", {})
    if expected:
        full_pair_rank = int(expected["nocc"]) * int(expected["nvir"])
        if int(args.thc_rank) != full_pair_rank:
            raise ValueError(
                "inexact thc_cd is fail-closed: --thc-rank must equal the "
                f"full pair dimension {full_pair_rank} for {args.case}"
            )


def _plan(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    approximation, signature = approximation_identity(args.method, args)
    settings_names = (
        "eri_tol",
        "direct_scf_tol",
        "cd_max_rank",
        "gint_column_backend",
        "gint_group_size",
        "gint_max_block_bytes",
        "gint_max_batch_size",
        "cd_mo_block_size",
        "rr_eig_cutoff",
        "rr_max_rank",
        "denominator_tolerance",
        "denominator_max_rank",
        "rr_initial_rank",
        "rr_solver_tolerance",
        "rr_solver_maxiter",
        "rr_dense_fallback_dimension",
        "rr_ritz_residual_tolerance",
        "rr_auxiliary_block_size",
        "rr_virtual_block_size",
        "precision",
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
        "fno_thresh",
        "fno_nvir_act",
        "fno_pct_occ",
        "scf_conv_tol",
        "cc_conv_tol",
        "cc_conv_tol_normt",
        "max_cycle",
        "max_memory_mb",
        "retain_cupy_cache",
        "skip_checkpoint",
        "run_full_space_diagnostic",
    )
    method_info = METHOD_INFO[args.method]
    plan = {
        "protocol_schema": PROTOCOL_SCHEMA,
        "case": args.case,
        "case_role": case["role"],
        "method": args.method,
        "device": args.device,
        "repeat": str(args.repeat),
        "basis": case["basis"],
        "geometry_sha256": geometry_hash(case),
        "method_class": method_info["method_class"],
        "acceptance_table": method_info["acceptance_table"],
        "validation_status": method_info["validation_status"],
        "charge": 0,
        "spin": 0,
        "spherical": True,
        "reference": "RHF",
        "frozen_occupied": 0,
        "approximation": approximation,
        "approximation_signature": signature,
        "source_protocol": {
            "frozen_base_commit": FROZEN_BASE_COMMIT,
            "content_manifest_at_start": True,
            "content_manifest_at_end": True,
            "stable_source_required_for_performance": True,
        },
        "orbital_protocol": {
            "schema": ORBITAL_ARTIFACT_SCHEMA,
            "checkpoint_schema": RESTART_SCHEMA_V2,
            "input_requested": args.orbital_artifact_in is not None,
            "output_requested": args.orbital_artifact_out is not None,
            "fresh_scf_always_run": True,
            "artifact_validation_before_post_hf": True,
            "exact_mo_fingerprint_in_checkpoint": True,
        },
        "solver_lifecycle": {
            "benchmark_prebuilds_eris": args.method not in DIRECT_CD_METHODS,
            "dense_ao2mo_in_normal_path": args.method not in DIRECT_CD_METHODS,
            "dense_t2_in_normal_path": args.method not in DIRECT_CD_METHODS,
            "final_full_space_residual_in_timed_path": args.method
            in {"canonical", "canonical_legacy"},
            "projected_residual_recorded": args.method
            in {"rr_cd", "thc_cd", "rr_canonical", "thc_canonical"},
            "full_space_diagnostic_supported": (
                args.method in FULL_SPACE_DIAGNOSTIC_METHODS
            ),
            "full_space_diagnostic_requested": bool(
                args.run_full_space_diagnostic
            ),
            "requested_full_space_diagnostic_outside_post_hf": bool(
                args.run_full_space_diagnostic
            ),
            "resident_iterations": (
                False if args.method == "canonical_legacy" else None
            ),
            "resident_iterations_policy": METHOD_INFO[args.method].get(
                "resident_iterations_policy"
            ),
        },
        "timing_boundary": (
            "post_hf includes integral transformation, CD/FNO/projector/THC "
            "preprocessing, all iterations, normal-path final residual checks "
            "and normal checkpoint; SCF, counterpoise jobs, and the optional "
            "reconstructed full-space RR/CD diagnostic are excluded"
        ),
        "settings": {key: getattr(args, key) for key in settings_names},
    }
    required_controls = REQUIRED_EXPLICIT_APPROXIMATION_CONTROLS.get(
        args.method, ()
    )
    if required_controls:
        provided_controls = set(getattr(
            args, "_explicit_approximation_controls", ()
        ))
        plan["approximation_control_provenance"] = {
            "required_explicit": list(required_controls),
            "provided_via_cli_or_config": sorted(
                provided_controls.intersection(required_controls)
            ),
            "all_required_explicit": all(
                name in provided_controls for name in required_controls
            ),
            "source_by_control": {
                name: getattr(
                    args, "_approximation_control_sources", {}
                ).get(name)
                for name in required_controls
            },
        }
    if method_info.get("performance_eligible") is False:
        plan["performance_eligible"] = False
        plan["performance_gates"] = method_info.get("performance_gates", {})
        plan["performance_limitations"] = method_info.get(
            "performance_limitations", []
        )
    plan["settings"]["cc_max_cycle"] = int(args.max_cycle)
    plan["settings"]["eri_backend"] = (
        "cd" if args.method in DIRECT_CD_METHODS else "canonical"
    )
    return plan


def run(args: argparse.Namespace) -> int:
    args = _apply_config(args)
    _validate_run_configuration(args)
    cases = load_cases()
    case = cases[args.case]
    plan = _plan(args, case)
    if args.dry_run:
        print(json.dumps(
            {"dry_run": True, "plan": plan, "source": _git_state()},
            indent=2,
            sort_keys=True,
        ))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.case}__{args.method}__r{args.repeat}"
    result_path = args.output_dir / f"{stem}.json"
    log_path = args.output_dir / f"{stem}.log"
    checkpoint_path = args.output_dir / f"{stem}.checkpoint.npz"
    if result_path.exists() or log_path.exists() or checkpoint_path.exists():
        raise RuntimeError(f"refusing to overwrite existing run: {result_path}")

    started = datetime.now(timezone.utc).isoformat()
    record: dict[str, Any] = {
        "status": "starting",
        "started_utc": started,
        "case_id": args.case,
        "method": args.method,
        "device": args.device,
        "repeat": str(args.repeat),
        "plan": plan,
        "case": case,
        "geometry_sha256": geometry_hash(case),
        "cc_max_cycle": int(args.max_cycle),
        "max_memory_mb": int(args.max_memory_mb),
        "approximation": plan["approximation"],
        "approximation_signature": plan["approximation_signature"],
        "eri_backend": plan["settings"]["eri_backend"],
        "protocol_schema": plan["protocol_schema"],
        "source_protocol": plan["source_protocol"],
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "pid": os.getpid(),
        "slurm": {key: os.getenv(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION")},
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {key: os.getenv(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GPU4PYSCF_NUMA")},
        "topology": _topology(),
        "pcie": _pcie_info(),
        "versions": {key: _version(key) for key in ("numpy", "scipy", "pyscf", "cupy-cuda12x", "gpu4pyscf-cuda12x")},
        "source": _git_state(),
        "output": str(result_path),
    }
    _apply_source_start_eligibility(record)
    _apply_method_eligibility(record, METHOD_INFO[args.method])
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    monitor = _HBMMonitor()
    cp = None
    metrics = None
    try:
        import numpy as np
        import pyscf
        from pyscf import gto, lib, scf
        from gpu4pyscf.cc.device_runtime import RunMetrics

        metrics = RunMetrics(args.method, metadata=plan)
        if args.threads is not None:
            lib.num_threads(args.threads)
        with metrics.phase("molecule_setup"):
            mol = gto.M(
                atom=[(item[0], tuple(item[1:])) for item in case["geometry_angstrom"]],
                unit="Angstrom", basis=case["basis"], charge=0, spin=0, cart=False,
                verbose=4, output=str(log_path), max_memory=args.max_memory_mb,
            )
        if args.device == "gpu":
            import cupy as cp
            from gpu4pyscf.scf import hf as gpu_hf
            mf = gpu_hf.RHF(mol)
            record["gpu"] = _gpu_info(cp)
            if not record["gpu"].get("available"):
                raise RuntimeError("GPU requested but no usable CUDA device was found")
            monitor.start()
        else:
            mf = scf.RHF(mol)
        mf.conv_tol = args.scf_conv_tol
        mf.max_cycle = 100
        with metrics.phase("hf"):
            _sync(cp)
            hf_start = time.perf_counter()
            mf.kernel()
            _sync(cp)
            hf_seconds = time.perf_counter() - hf_start
        record.update({"nao": int(mol.nao_nr()), "nao_cart": int(mol.nao_nr(cart=True)), "nelectron": int(mol.nelectron), "hf_seconds": hf_seconds, "hf_converged": bool(mf.converged), "e_hf": float(mf.e_tot)})
        expected = case.get("expected_dimensions", {})
        observed = {
            "nao": int(mol.nao_nr()),
            "nocc": int(mol.nelectron // 2),
            "nvir": int(mol.nao_nr() - mol.nelectron // 2),
        }
        if expected and observed != expected:
            raise RuntimeError(
                f"frozen benchmark dimensions changed: expected {expected}, "
                f"observed {observed}"
            )
        record["orbital_dimensions"] = observed
        if not mf.converged:
            raise RuntimeError("HF failed to converge; CCSD was not started")
        with metrics.phase("canonical_orbital_setup"):
            orbital_context = _prepare_canonical_orbitals(
                args,
                mf,
                mol,
                case,
                record["source"],
                np,
                cp,
            )
        record["canonical_orbitals"] = _orbital_record(orbital_context)
        record["e_hf_for_post_hf"] = float(mf.e_tot)
        with metrics.phase("post_hf"):
            _sync(cp)
            post_start = time.perf_counter()
            monitor.post_hf_started_at = time.time()
            with metrics.phase("method_setup"):
                solver, method_meta = _make_solver(args, mf)
            if args.method == "fno":
                cc_obj = solver.build()
                cc_obj.run_metrics = metrics
                if hasattr(cc_obj, "free_cupy_cache"):
                    cc_obj.free_cupy_cache = not args.retain_cupy_cache
            else:
                cc_obj = solver
                # RR/THC maintain their own algorithm metrics.  Canonical CCSD
                # uses the outer run metrics so its instrumented phases are
                # included in the same result record.
                if not hasattr(cc_obj, "run_metrics"):
                    cc_obj.run_metrics = metrics
                if hasattr(cc_obj, "free_cupy_cache"):
                    cc_obj.free_cupy_cache = not args.retain_cupy_cache
            # Canonical/FNO retain the exact ERI object for their dense final
            # residual.  The resident canonical entry also retains its GPU ERI
            # cache and direct-integral workspace until that check completes.
            # rr_cd owns a three-index direct-CD lifecycle and must never pass
            # through this harness's dense ao2mo seam.
            with _timed_resident_diagnostic_scope(args.method, cc_obj):
                kernel_result, eris = _execute_solver_kernel(
                    args.method, solver, cc_obj
                )
                canonical_final_residual = None
                if args.method in {"canonical", "canonical_legacy", "fno"}:
                    with metrics.phase("final_full_space_residual"):
                        _sync(cp)
                        canonical_final_residual = _dense_final_residual(
                            cc_obj,
                            eris,
                            np,
                            orbital_space=(
                                "fno-active"
                                if args.method == "fno" else "full"
                            ),
                        )
                        _sync(cp)
            checkpoint = None
            cached_amplitude_payload = None
            if not args.skip_checkpoint:
                with metrics.phase("checkpoint"):
                    _sync(cp)
                    checkpoint = _write_checkpoint_once(
                        checkpoint_path,
                        args.method,
                        kernel_result,
                        cc_obj,
                        np,
                        orbital_context=orbital_context,
                    )
                    cached_amplitude_payload = checkpoint.pop(
                        "_cached_amplitude_payload"
                    )
            _sync(cp)
            monitor.post_hf_finished_at = time.time()
            post_hf_seconds = time.perf_counter() - post_start
        explicit_full_space_diagnostic = None
        if args.run_full_space_diagnostic:
            explicit_full_space_diagnostic = (
                _run_post_hf_full_space_diagnostic(
                    args.method, cc_obj, cp
                )
            )
        amplitude_payload = _amplitude_payload(
            kernel_result,
            cc_obj,
            np,
            cached_payload=cached_amplitude_payload,
        )
        experiment_record = None
        if hasattr(cc_obj, "experiment_record"):
            experiment_record = cc_obj.experiment_record()
            if not isinstance(experiment_record, dict):
                raise TypeError("experiment_record() must return a dictionary")
            method_meta = _merge_method_metadata(
                method_meta, experiment_record.get("method")
            )
        else:
            method_meta = _runtime_method_metadata(cc_obj, method_meta)
        record.update({"method_metadata": method_meta, "post_hf_seconds": post_hf_seconds, "hf_plus_post_hf_seconds": hf_seconds + post_hf_seconds, "cc_class": f"{cc_obj.__class__.__module__}.{cc_obj.__class__.__name__}", "cc_converged": bool(getattr(cc_obj, "converged", False)), "cc_iterations": int(getattr(cc_obj, "cycles", 0))})
        _apply_method_eligibility(record, method_meta)
        record.update(_energy_payload(args.method, solver, cc_obj))
        record["checkpoint"] = checkpoint or {
            "included_in_post_hf": False,
            "reason": "--skip-checkpoint",
        }
        record["amplitudes"] = amplitude_payload
        if canonical_final_residual is not None:
            record["residual"] = canonical_final_residual
        else:
            record["residual"] = _labeled_low_rank_residual(cc_obj)
        full_space_diagnostic = (
            explicit_full_space_diagnostic
            if explicit_full_space_diagnostic is not None
            else _full_space_residual_diagnostic(args.method)
        )
        if full_space_diagnostic is not None:
            record["full_space_residual_diagnostic"] = full_space_diagnostic
        projector = getattr(cc_obj, "rr_projector", None)
        thc_doubles = getattr(cc_obj, "thc_doubles", None)
        thc_factors = getattr(cc_obj, "thc_projector_factors", None)
        actual_thc_rank = None
        if thc_doubles is not None:
            actual_thc_rank = int(thc_doubles.thc_rank)
        elif thc_factors is not None:
            actual_thc_rank = int(thc_factors.thc_rank)
        record["ranks"] = {
            "rr_rank": None if projector is None else int(projector.rank),
            "rr_full_dimension": None if projector is None else int(projector.full_dimension),
            "thc_rank": actual_thc_rank,
        }
        if args.method == "fno":
            record["fno"] = solver.metadata.audit_dict()
            record["fno"]["ccsd_energy"] = solver.energy_metadata
        if experiment_record is not None:
            record["experiment_record"] = experiment_record
        record["status"] = "completed" if record["cc_converged"] else "not_converged"
    except Exception as exc:
        import traceback
        record.update({"status": "error", "error": repr(exc), "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        monitor.close()
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["peak_host_RSS_GiB"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2))
        record["hbm"] = {"sampling_interval_s": 0.5, "sample_count": len(monitor.samples), "peak_process_MiB": max((x["MiB"] for x in monitor.samples), default=None), "gpu_uuids": sorted({x["gpu_uuid"] for x in monitor.samples})}
        record["gpu_telemetry"] = _gpu_telemetry_payload(monitor)
        record["run_metrics"] = metrics.to_dict() if metrics is not None else None
        record["timing_definition"] = plan["timing_boundary"]
        _record_source_stability(record)
        result_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({key: record.get(key) for key in ("case", "status", "method", "hf_seconds", "post_hf_seconds", "e_tot", "e_corr", "error")}, sort_keys=True))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(run(_parser().parse_args()))
    except Exception as exc:
        print(f"benchmark configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
