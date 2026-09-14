#!/usr/bin/env python3
"""Auditable Boys--Bernardi counterpoise calculations for WATER27 cases.

The counterpoise calculation is deliberately a separate driver from the
timing harness.  It builds every monomer in the complete cluster basis (the
other atoms are represented by PySCF ``ghost-*`` atoms), evaluates

    E_int^CP = E_cluster - sum_i E_monomer_i^ghost,

and writes an immutable JSON record.  The fragment records include geometry
and atom-list hashes so that a result cannot be mistaken for a calculation in
an incomplete monomer basis.  CP timings are diagnostic only and are not part
of the post-HF speed claim.

The production RR/THC implementations are still validation paths in this
branch.  Their status is recorded in the output rather than hidden behind the
short method names.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[2]
CASE_FILE = ROOT / "cases.json"
KCAL_PER_EH = 627.5094740631

try:
    from .snapshot_manifest import (
        CANDIDATE_DEPLOYMENT_PROFILE,
        FROZEN_BASE_COMMIT,
        G0_DEPLOYMENT_PROFILE,
        SOURCE_SNAPSHOT_SCHEMA,
        captured_runtime_bundle_evidence_errors,
        source_tree_digest as _snapshot_source_tree_digest,
        validate_snapshot,
    )
except ImportError:  # direct execution and importlib-based unit tests
    _snapshot_spec = importlib.util.spec_from_file_location(
        "water8_cp_snapshot_manifest", ROOT / "snapshot_manifest.py"
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
    SOURCE_SNAPSHOT_SCHEMA = _snapshot_module.SOURCE_SNAPSHOT_SCHEMA
    captured_runtime_bundle_evidence_errors = (
        _snapshot_module.captured_runtime_bundle_evidence_errors
    )
    _snapshot_source_tree_digest = _snapshot_module.source_tree_digest
    validate_snapshot = _snapshot_module.validate_snapshot

try:
    from .cp_orbital_bundle import CPOrbitalBundle, shared_bundle_proof
except ImportError:  # direct execution and importlib-based unit tests
    _orbital_spec = importlib.util.spec_from_file_location(
        "water8_cp_orbital_bundle", ROOT / "cp_orbital_bundle.py"
    )
    if _orbital_spec is None or _orbital_spec.loader is None:
        raise RuntimeError("cannot load cp_orbital_bundle.py")
    _orbital_module = importlib.util.module_from_spec(_orbital_spec)
    _orbital_spec.loader.exec_module(_orbital_module)
    CPOrbitalBundle = _orbital_module.CPOrbitalBundle
    shared_bundle_proof = _orbital_module.shared_bundle_proof
METHOD_STATUS = {
    "canonical": ("equation-preserving", "production-canonical"),
    "fno": ("controlled-approximation", "production-fno"),
    "rr_cd": (
        "controlled-approximation",
        "direct-cd-compressed-rr-reference",
    ),
    "thc_cd": (
        "controlled-approximation",
        "direct-cd-r123-complement-full-pair-thc-endpoint",
    ),
    "rr_canonical": ("controlled-approximation", "dense-projected-validation"),
    "thc_canonical": (
        "controlled-approximation",
        "dense-two-level-validation-surrogate",
    ),
}


def _validate_rr_ring_kernel(method: str, selector: Any) -> str:
    """Fail closed when an RR-only execution selector is applied to THC."""

    if not isinstance(selector, str):
        raise TypeError("rr_ring_kernel must be a string")
    selector = selector.strip().lower()
    if selector not in {"reference", "gemm"}:
        raise ValueError(f"unsupported rr_ring_kernel {selector!r}")
    if method in {"thc_cd", "thc_canonical"} and selector != "reference":
        raise ValueError(
            "rr_ring_kernel applies only to RRCCSD; the THC residual engine "
            "requires reference (not applicable)"
        )
    return selector


def _effective_method_status(
    method: str, options: dict[str, Any]
) -> dict[str, Any]:
    """Describe the solver path selected by this CP invocation."""

    method_class, validation_status = METHOD_STATUS[method]
    selector = _validate_rr_ring_kernel(
        method, options.get("rr_ring_kernel", "reference")
    )
    if method in {"rr_cd", "rr_canonical"} and selector == "gemm":
        validation_status = (
            "direct-cd-compressed-rr-gemm-gate-pending"
            if method == "rr_cd"
            else "dense-projected-validation-gemm-ring"
        )
        ring_status = "gemm-awaiting-a100-numerical-allocation-gate"
    elif method in {"rr_cd", "rr_canonical"}:
        ring_status = "bounded-memory-reference-kernel"
    elif method in {"thc_cd", "thc_canonical"}:
        ring_status = "not-applicable-to-thc-engine"
    else:
        ring_status = None
    return {
        "method_class": method_class,
        "validation_status": validation_status,
        "rr_ring_kernel": selector if ring_status is not None else None,
        "rr_ring_kernel_status": ring_status,
    }


@dataclass(frozen=True)
class FragmentSpec:
    """One water monomer as atom indices into the complete cluster."""

    fragment_id: int
    atom_indices: tuple[int, ...]

    @property
    def active_atom_indices(self) -> tuple[int, ...]:
        return self.atom_indices


@dataclass(frozen=True)
class EnergyRecord:
    """Energy and provenance for one cluster or ghost-basis fragment."""

    total_energy_eh: float
    hf_energy_eh: float | None
    correlation_energy_eh: float | None
    converged: bool | None
    wall_time_s: float
    metadata: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_json(value: Any) -> str:
    """Return a stable hash for JSON-compatible values."""

    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert solver audit metadata to JSON without lossy string fallbacks."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("solver metadata dictionaries require string keys")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    raise TypeError(
        f"solver metadata contains a non-JSON value of type {type(value).__name__}"
    )


def approximation_identity(
    method: str, options: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Return the same effective approximation signature as the benchmark."""

    payload: dict[str, Any] = {
        "schema": "gpu4pyscf.cc.approximation.v1",
        "method": str(method),
    }
    if method == "fno":
        nvir_act = options.get("fno_nvir_act")
        pct_occ = options.get("fno_pct_occ")
        threshold = float(options.get("fno_thresh", 1.0e-6))
        if nvir_act is not None:
            selection = {"mode": "nvir_act", "value": int(nvir_act)}
        elif pct_occ is not None:
            selection = {"mode": "pct_occ", "value": float(pct_occ)}
        else:
            selection = {"mode": "occupation_threshold", "value": threshold}
        payload.update({"selection": selection, "correction": "delta-mp2"})
    elif method in {"rr_cd", "thc_cd", "rr_canonical", "thc_canonical"}:
        payload.update({
            "eri_backend": (
                "cd" if method in {"rr_cd", "thc_cd"} else "canonical"
            ),
            "eri_tol": float(options.get("eri_tol", 1.0e-8)),
            "rr_eig_cutoff": float(options.get("rr_eig_cutoff", 1.0e-6)),
            "rr_max_rank": options.get("rr_max_rank"),
            "precision": str(options.get("precision", "fp64")),
        })
        if method in {"rr_cd", "thc_cd"}:
            payload.update({
                "direct_scf_tol": float(options.get("direct_scf_tol", 1.0e-13)),
                "cd_max_rank": options.get("cd_max_rank"),
                "denominator_tolerance": float(
                    options.get("denominator_tolerance", 1.0e-10)
                ),
                "denominator_max_rank": options.get("denominator_max_rank"),
                "rr_solver_tolerance": float(
                    options.get("rr_solver_tolerance", 1.0e-10)
                ),
                "rr_solver_maxiter": options.get("rr_solver_maxiter"),
                "rr_dense_fallback_dimension": int(
                    options.get("rr_dense_fallback_dimension", 64)
                ),
                "rr_ritz_residual_tolerance": options.get(
                    "rr_ritz_residual_tolerance"
                ),
            })
        if method in {"thc_cd", "thc_canonical"}:
            payload.update({
                "thc_fit_tol": float(options.get("thc_fit_tol", 1.0e-6)),
                "thc_rank": options.get("thc_rank"),
                "thc_initial_rank": options.get("thc_initial_rank"),
                "thc_max_rank": options.get("thc_max_rank"),
                "thc_rank_growth": float(options.get("thc_rank_growth", 1.5)),
                "thc_orthogonality_cutoff": float(
                    options.get("thc_orthogonality_cutoff", 1.0e-12)
                ),
                "thc_orthogonality_tolerance": float(
                    options.get("thc_orthogonality_tolerance", 1.0e-10)
                ),
                "thc_max_iterations": int(
                    options.get("thc_max_iterations", 500)
                ),
                "thc_als_convergence_tolerance": float(
                    options.get("thc_als_convergence_tolerance", 1.0e-10)
                ),
                "thc_ridge": float(options.get("thc_ridge", 1.0e-12)),
                "thc_seed": int(options.get("thc_seed", 0)),
                "thc_replacement_validation_atol": options.get(
                    "thc_replacement_validation_atol"
                ),
                "thc_replacement_validation_rtol": options.get(
                    "thc_replacement_validation_rtol"
                ),
            })
    digest = hashlib.sha256(_json_bytes(payload)).hexdigest()
    return payload, f"{method}:v1:{digest}"


def load_cases(case_file: Path = CASE_FILE) -> dict[str, dict[str, Any]]:
    """Load and minimally validate the local WATER27 case manifest."""

    payload = json.loads(case_file.read_text())
    records = payload.get("cases", payload)
    if not isinstance(records, list):
        raise ValueError("cases.json must contain a cases list")
    cases = {record["id"]: record for record in records}
    expected = {"water2-tz", "water4-tz", "water8-tz", "water8s4-tz"}
    if set(cases) != expected:
        raise ValueError(f"cases.json must contain exactly {sorted(expected)}")
    for case_id, case in cases.items():
        if case.get("basis") != "cc-pVTZ":
            raise ValueError(f"{case_id}: CP driver requires spherical cc-pVTZ")
        geometry = case.get("geometry_angstrom")
        if not isinstance(geometry, list) or not geometry:
            raise ValueError(f"{case_id}: missing geometry_angstrom")
        if any(len(atom) != 4 for atom in geometry):
            raise ValueError(f"{case_id}: each atom must be [symbol, x, y, z]")
    return cases


def _distance(a: Sequence[Any], b: Sequence[Any]) -> float:
    return math.sqrt(sum((float(a[k]) - float(b[k])) ** 2 for k in (1, 2, 3)))


def infer_water_fragments(
    geometry_angstrom: Sequence[Sequence[Any]],
    *,
    bond_cutoff_angstrom: float = 1.35,
) -> tuple[FragmentSpec, ...]:
    """Infer disjoint O-H-H monomers from a water-cluster geometry.

    The WATER27 coordinates have intramolecular O-H distances near 1 Å and
    intermolecular hydrogen bonds substantially longer than the cutoff.  A
    global distance-sorted matching makes this inference auditable and avoids
    relying on the order in which atoms happen to occur in the manifest.
    """

    oxygens = [i for i, atom in enumerate(geometry_angstrom) if str(atom[0]).upper() == "O"]
    hydrogens = [i for i, atom in enumerate(geometry_angstrom) if str(atom[0]).upper() == "H"]
    if len(hydrogens) != 2 * len(oxygens) or not oxygens:
        raise ValueError("geometry is not a neutral water cluster with two H per O")
    edges = sorted(
        (_distance(geometry_angstrom[oi], geometry_angstrom[hi]), oi, hi)
        for oi in oxygens
        for hi in hydrogens
    )
    assigned_o: dict[int, list[int]] = {oi: [] for oi in oxygens}
    assigned_h: set[int] = set()
    for distance, oi, hi in edges:
        if distance > bond_cutoff_angstrom:
            break
        if len(assigned_o[oi]) < 2 and hi not in assigned_h:
            assigned_o[oi].append(hi)
            assigned_h.add(hi)
    if any(len(hydrogen_ids) != 2 for hydrogen_ids in assigned_o.values()) or len(assigned_h) != len(hydrogens):
        raise ValueError(
            "could not infer eight (or fewer) O-H-H fragments; "
            f"increase bond_cutoff_angstrom above {bond_cutoff_angstrom:g} only with an audit"
        )
    fragments = tuple(
        FragmentSpec(fragment_id=fragment_id, atom_indices=tuple([oi, *sorted(hydrogen_ids)]))
        for fragment_id, (oi, hydrogen_ids) in enumerate(sorted(assigned_o.items()))
    )
    if sorted(i for fragment in fragments for i in fragment.atom_indices) != list(range(len(geometry_angstrom))):
        raise ValueError("fragment assignment is not a disjoint cover of the cluster atoms")
    return fragments


def ghost_symbol(symbol: str) -> str:
    """Return the PySCF ghost-atom spelling for an elemental symbol."""

    clean = str(symbol).strip()
    if clean.lower().startswith("ghost-"):
        return clean
    return f"ghost-{clean}"


def fragment_atoms(
    geometry_angstrom: Sequence[Sequence[Any]],
    fragment: FragmentSpec,
) -> list[list[Any]]:
    """Build complete-basis atoms, keeping active atoms unghosted."""

    active = set(fragment.atom_indices)
    result: list[list[Any]] = []
    for index, atom in enumerate(geometry_angstrom):
        symbol, x, y, z = atom
        result.append([str(symbol) if index in active else ghost_symbol(str(symbol)), float(x), float(y), float(z)])
    return result


def molecule_atom_payload(geometry_angstrom: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [[str(symbol), float(x), float(y), float(z)] for symbol, x, y, z in geometry_angstrom]


def build_molecule(
    atoms: Sequence[Sequence[Any]],
    *,
    basis: str = "cc-pVTZ",
    charge: int = 0,
    spin: int = 0,
    verbose: int = 0,
) -> Any:
    """Construct a spherical PySCF molecule with an explicit basis protocol."""

    try:
        from pyscf import gto
    except ImportError as exc:  # pragma: no cover - exercised on GPU hosts
        raise RuntimeError("PySCF is required to build CP molecules") from exc
    return gto.M(
        atom=[list(atom) for atom in atoms],
        basis=basis,
        unit="Angstrom",
        charge=charge,
        spin=spin,
        cart=False,
        symmetry=False,
        verbose=verbose,
    )


def counterpoise_energy(cluster_energy_eh: float, fragment_energies_eh: Iterable[float]) -> float:
    """Compute Boys--Bernardi interaction energy in hartree."""

    fragments = tuple(float(value) for value in fragment_energies_eh)
    if not fragments:
        raise ValueError("at least one fragment energy is required")
    return float(cluster_energy_eh) - sum(fragments)


def _version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_revision(repository: Path = ROOT.parents[2]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _source_tree_digest(root: Path = REPOSITORY_ROOT) -> tuple[str, int]:
    """Hash the deterministic deployable-source manifest used by benchmarks."""

    return _snapshot_source_tree_digest(Path(root))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_manifest_valid(
    manifest: Any,
    *,
    source_tree_sha256: Any = None,
    source_file_count: Any = None,
    task_root: Any = None,
) -> bool:
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
    provenance = manifest.get("local_provenance")
    profile = (
        provenance.get("deployment_profile")
        if isinstance(provenance, dict)
        else None
    )
    if (
        manifest.get("schema") == SOURCE_SNAPSHOT_SCHEMA
        and profile == CANDIDATE_DEPLOYMENT_PROFILE
    ):
        if (
            not isinstance(source_tree_sha256, str)
            or isinstance(source_file_count, bool)
            or not isinstance(source_file_count, int)
            or source_file_count <= 0
            or not isinstance(task_root, (str, Path))
        ):
            return False
        if captured_runtime_bundle_evidence_errors(
            runtime,
            deployment_profile=profile,
            source_tree_sha256=source_tree_sha256,
            source_file_count=source_file_count,
            task_root=task_root,
            gint_release_lineage=manifest.get("gint_release_lineage"),
        ):
            return False
    return True


def _captured_task_root(evidence: dict[str, Any]) -> str | None:
    task_root = evidence.get("expected_task_root")
    if isinstance(task_root, str):
        return task_root
    snapshot_root = evidence.get("snapshot_root")
    if not isinstance(snapshot_root, str):
        return None
    try:
        return str(Path(snapshot_root).parent.parent)
    except (TypeError, ValueError):
        return None


def _expected_task_root() -> Path | None:
    value = os.getenv("CCSD_TASK_ROOT")
    return None if not value else Path(value).expanduser().resolve()


def _expected_deployment_profile() -> str | None:
    return os.getenv("CCSD_EXPECTED_DEPLOYMENT_PROFILE") or None


def _require_canonical_pristine() -> bool:
    return os.getenv(
        "CCSD_REQUIRE_CANONICAL_PRISTINE", ""
    ).strip().lower() in {"1", "true", "yes"}


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
    manifest = evidence.get("manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SOURCE_SNAPSHOT_SCHEMA
    ):
        evidence["valid"] = False
        evidence.setdefault("reasons", []).append(
            "counterpoise runtime requires a v3 source snapshot manifest"
        )
    manifest_path = evidence.get("manifest_path")
    try:
        manifest_sha256 = _file_sha256(str(manifest_path))
    except Exception as exc:
        manifest_sha256 = None
        evidence["manifest_hash_error_at_start"] = repr(exc)
    evidence["valid_at_start"] = evidence.get("valid") is True
    evidence["read_only_at_start"] = evidence.get("read_only") is True
    evidence["runtime_binaries_valid_at_start"] = _runtime_manifest_valid(
        evidence.get("manifest"),
        source_tree_sha256=evidence.get("tree_sha256"),
        source_file_count=evidence.get("files_hashed"),
        task_root=_captured_task_root(evidence),
    )
    evidence["manifest_sha256_at_start"] = manifest_sha256
    task_root = _expected_task_root()
    evidence["expected_task_root"] = None if task_root is None else str(task_root)
    return evidence


def _formal_source_evidence_valid(source: dict[str, Any]) -> bool:
    """Check the captured immutable-snapshot evidence without filesystem I/O."""

    digest = source.get("tree_sha256")
    start = source.get("tree_sha256_at_start")
    end = source.get("tree_sha256_at_end")
    snapshot = source.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    manifest = snapshot.get("manifest")
    if not isinstance(manifest, dict):
        return False
    manifest_start = snapshot.get("manifest_sha256_at_start")
    manifest_end = snapshot.get("manifest_sha256_at_end")
    repository = source.get("repository")
    snapshot_root = snapshot.get("snapshot_root")
    return bool(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        and start == digest == end
        and source.get("stable_during_run") is True
        and source.get("base_revision") == FROZEN_BASE_COMMIT
        and snapshot.get("schema") == "gpu4pyscf.snapshot-evidence.v1"
        and snapshot.get("valid") is True
        and snapshot.get("valid_at_start") is True
        and snapshot.get("valid_at_end") is True
        and snapshot.get("read_only") is True
        and snapshot.get("read_only_at_start") is True
        and snapshot.get("read_only_at_end") is True
        and snapshot.get("writable_paths") == []
        and snapshot.get("reasons") == []
        and snapshot.get("reasons_at_end") == []
        and snapshot.get("stable_during_run") is True
        and snapshot.get("tree_sha256") == digest
        and snapshot.get("tree_sha256_at_end") == digest
        and isinstance(manifest_start, str)
        and len(manifest_start) == 64
        and manifest_start == manifest_end
        and manifest.get("schema") == SOURCE_SNAPSHOT_SCHEMA
        and isinstance(manifest.get("source_normalization"), dict)
        and manifest.get("tree_sha256") == digest
        and manifest.get("base_revision") == FROZEN_BASE_COMMIT
        and manifest.get("immutable") is True
        and snapshot.get("runtime_binaries_valid_at_start") is True
        and snapshot.get("runtime_binaries_valid_at_end") is True
        and _runtime_manifest_valid(
            manifest,
            source_tree_sha256=digest,
            source_file_count=source.get("files_hashed_at_start"),
            task_root=_captured_task_root(snapshot),
        )
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
        and isinstance(snapshot.get("expected_task_root"), str)
        and Path(snapshot_root).parent
        == Path(snapshot["expected_task_root"]) / "snapshots"
        and snapshot.get("manifest_path")
        == str(Path(snapshot_root) / "manifest.json")
        and Path(snapshot_root).name == digest
        and Path(str(repository)).name == "source"
        and Path(str(repository)).parent == Path(snapshot_root)
    )


def _source_state() -> dict[str, Any]:
    digest, count = _source_tree_digest(REPOSITORY_ROOT)
    snapshot = _snapshot_at_start(REPOSITORY_ROOT)
    observed_revision = _git_revision(REPOSITORY_ROOT)
    return {
        "repository": str(REPOSITORY_ROOT.resolve()),
        "revision": observed_revision or f"tree-sha256:{digest}",
        "observed_git_revision": observed_revision,
        "base_revision": FROZEN_BASE_COMMIT,
        "tree_sha256": digest,
        "tree_sha256_at_start": digest,
        "files_hashed_at_start": count,
        "snapshot": snapshot,
        "formal_counterpoise_eligible": False,
    }


def _recorded_gint_runtime_options(
    source_state: dict[str, Any], receipt: str | Path | None
) -> dict[str, Any]:
    """Bind CP selected-GINT solvers to the source state captured in-process."""

    repository = Path(str(source_state.get("repository", ""))).resolve()
    if repository != REPOSITORY_ROOT.resolve():
        raise RuntimeError("recorded source state does not identify this CP driver")
    recorded_digest = source_state.get("tree_sha256_at_start")
    if recorded_digest != source_state.get("tree_sha256"):
        raise RuntimeError("recorded CP source digest is internally inconsistent")
    current_digest, _ = _source_tree_digest(REPOSITORY_ROOT)
    if current_digest != recorded_digest:
        raise RuntimeError("source changed before selected-GINT CP construction")
    snapshot = source_state.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    expected_manifest = REPOSITORY_ROOT.parent / "manifest.json"
    recorded_manifest = snapshot.get("manifest_path")
    if recorded_manifest is not None and Path(recorded_manifest).resolve() != (
        expected_manifest.resolve()
    ):
        raise RuntimeError("recorded CP manifest path is outside this snapshot")
    return {
        "gint_runtime_gate_receipt": (
            None if receipt is None
            else str(Path(receipt).expanduser().resolve())
        ),
        "gint_runtime_execution_mode": "consumer-counterpoise",
        "gint_runtime_source_digest": recorded_digest,
        "gint_runtime_source_root": str(REPOSITORY_ROOT.resolve()),
        "gint_runtime_manifest_path": str(expected_manifest.resolve()),
        "gint_runtime_topology_path": os.getenv("CCSD_TOPOLOGY_RECORD"),
    }


def _complete_source_stability(source: dict[str, Any]) -> bool:
    snapshot = source.setdefault("snapshot", {})
    try:
        end_digest, end_count = _source_tree_digest(REPOSITORY_ROOT)
    except Exception as exc:
        source.update({
            "tree_sha256_at_end": None,
            "files_hashed_at_end": None,
            "stable_during_run": None,
            "stability_error": repr(exc),
            "formal_counterpoise_eligible": False,
        })
        return False
    source["tree_sha256_at_end"] = end_digest
    source["files_hashed_at_end"] = end_count
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
            "reasons": [f"snapshot validation at end failed: {exc!r}"],
        }
    manifest_path = snapshot.get("manifest_path")
    try:
        manifest_end = _file_sha256(str(manifest_path))
    except Exception as exc:
        manifest_end = None
        snapshot["manifest_hash_error_at_end"] = repr(exc)
    manifest_stable = bool(
        snapshot.get("manifest_sha256_at_start") is not None
        and snapshot.get("manifest_sha256_at_start") == manifest_end
    )
    tree_stable = bool(
        source.get("tree_sha256_at_start")
        and source["tree_sha256_at_start"] == end_digest
    )
    snapshot.update({
        "valid_at_end": end_evidence.get("valid") is True,
        "read_only_at_end": end_evidence.get("read_only") is True,
        "runtime_binaries_valid_at_end": _runtime_manifest_valid(
            end_evidence.get("manifest"),
            source_tree_sha256=end_evidence.get("tree_sha256"),
            source_file_count=end_evidence.get("files_hashed"),
            task_root=_captured_task_root(end_evidence),
        ),
        "tree_sha256_at_end": end_evidence.get("tree_sha256"),
        "files_hashed_at_end": end_evidence.get("files_hashed"),
        "manifest_sha256_at_end": manifest_end,
        "stable_during_run": manifest_stable,
        "reasons_at_end": end_evidence.get("reasons", []),
    })
    source["stable_during_run"] = bool(tree_stable and manifest_stable)
    formal = _formal_source_evidence_valid(source)
    source["formal_counterpoise_eligible"] = formal
    return formal


def _make_solver(mol: Any, method: str, device: str, options: dict[str, Any]) -> Any:
    """Create a solver while keeping the CP protocol shared by all methods."""

    if method not in METHOD_STATUS:
        raise ValueError(f"unknown method {method!r}; choices: {sorted(METHOD_STATUS)}")
    ring_selector = _validate_rr_ring_kernel(
        method, options.get("rr_ring_kernel", "reference")
    )
    if device == "gpu":
        try:
            from gpu4pyscf import scf as scf_module
        except ImportError as exc:
            raise RuntimeError("GPU CP requested but gpu4pyscf is unavailable") from exc
    else:
        from pyscf import scf as scf_module
    mf = scf_module.RHF(mol)
    mf.conv_tol = float(options.get("scf_conv_tol", 1e-10))
    mf.max_cycle = int(options.get("scf_max_cycle", 100))
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for method={method} device={device}")
    orbital_request = options.get("_cp_orbital_request")
    if orbital_request is not None:
        if not isinstance(orbital_request, dict):
            raise TypeError("internal CP orbital request must be a dictionary")
        manager = orbital_request.get("manager")
        if not isinstance(manager, CPOrbitalBundle):
            raise TypeError("internal CP orbital request has no bundle manager")
        evidence = manager.prepare(
            entry_key=orbital_request["entry_key"],
            role=orbital_request["role"],
            fragment_id=orbital_request["fragment_id"],
            active_atom_indices=orbital_request["active_atom_indices"],
            atoms=orbital_request["atoms"],
            mol=mol,
            mf=mf,
        )
        # evaluate_energy samples this exact object after the solver completes.
        # prepare() has already replaced mf.mo_* and e_tot, so every solver
        # constructor below consumes the serialized canonical orbitals.
        mf._cp_canonical_orbitals = evidence
    def configure(solver: Any) -> Any:
        # PySCF CC constructors accept structural arguments only.  Assign
        # convergence and resource controls after construction for every path.
        solver.conv_tol = float(options.get("cc_conv_tol", 1e-8))
        solver.conv_tol_normt = float(options.get("cc_conv_tol_normt", 1e-6))
        solver.max_cycle = int(options.get("cc_max_cycle", 50))
        solver.max_memory = int(options.get("max_memory_mb", 110000))
        return solver
    if method == "canonical":
        if device == "gpu":
            from gpu4pyscf.cc.ccsd_incore import CCSD
        else:
            from pyscf.cc import CCSD
        return mf, configure(CCSD(mf))
    if method == "fno":
        from gpu4pyscf.cc.addons import FNOCCSD

        wrapper = FNOCCSD(
            mf,
            thresh=float(options.get("fno_thresh", 1e-6)),
            pct_occ=options.get("fno_pct_occ"),
            nvir_act=options.get("fno_nvir_act"),
            use_gpu=device == "gpu",
        )
        configure(wrapper.build())
        return mf, wrapper
    if device != "gpu":
        raise ValueError("RR/THC CP validation paths currently require gpu4pyscf")
    rr_options = {
        "eri_tol": float(options.get("eri_tol", 1e-8)),
        "rr_eig_cutoff": float(options.get("rr_eig_cutoff", 1e-6)),
        "rr_max_rank": options.get("rr_max_rank"),
        "rr_ring_kernel": ring_selector,
        "precision": str(options.get("precision", "fp64")),
    }
    from gpu4pyscf.cc.rrccsd import RRCCSD

    if method in {"rr_cd", "thc_cd"}:
        direct_options = {
            "eri_backend": "cd",
            "direct_scf_tol": float(options.get("direct_scf_tol", 1e-13)),
            "cd_max_rank": options.get("cd_max_rank"),
            "gint_column_backend": str(
                options.get("gint_column_backend", "selected")
            ),
            "gint_column_kernel": str(
                options.get("gint_column_kernel", "reference")
            ),
            "gint_group_size": int(options.get("gint_group_size", 16)),
            "gint_max_block_bytes": options.get("gint_max_block_bytes"),
            "gint_max_batch_size": int(
                options.get("gint_max_batch_size", 32)
            ),
            "cd_mo_block_size": int(options.get("cd_mo_block_size", 32)),
            "denominator_tolerance": float(
                options.get("denominator_tolerance", 1e-10)
            ),
            "denominator_max_rank": options.get("denominator_max_rank"),
            "rr_initial_rank": int(options.get("rr_initial_rank", 32)),
            "rr_solver_tolerance": float(
                options.get("rr_solver_tolerance", 1e-10)
            ),
            "rr_solver_maxiter": options.get("rr_solver_maxiter"),
            "rr_dense_fallback_dimension": int(
                options.get("rr_dense_fallback_dimension", 64)
            ),
            "rr_ritz_residual_tolerance": options.get(
                "rr_ritz_residual_tolerance"
            ),
            "rr_auxiliary_block_size": int(
                options.get("rr_auxiliary_block_size", 1)
            ),
            "rr_virtual_block_size": int(
                options.get("rr_virtual_block_size", 8)
            ),
        }
        if direct_options["gint_column_backend"] == "selected":
            runtime_binding = options.get("_gint_runtime_binding")
            if runtime_binding is not None:
                if not isinstance(runtime_binding, dict):
                    raise TypeError("internal GINT runtime binding must be a dictionary")
                direct_options.update(runtime_binding)
        if method == "rr_cd":
            solver = configure(RRCCSD(mf, **rr_options, **direct_options))
        else:
            from gpu4pyscf.cc.thc_rrccsd import THCRRCCSD

            solver = configure(THCRRCCSD(
                mf,
                thc_fit_tol=float(options.get("thc_fit_tol", 1e-6)),
                thc_rank=options.get("thc_rank"),
                thc_initial_rank=options.get("thc_initial_rank"),
                thc_max_rank=options.get("thc_max_rank"),
                thc_rank_growth=float(options.get("thc_rank_growth", 1.5)),
                thc_orthogonality_cutoff=float(
                    options.get("thc_orthogonality_cutoff", 1e-12)
                ),
                thc_orthogonality_tolerance=float(
                    options.get("thc_orthogonality_tolerance", 1e-10)
                ),
                thc_max_iterations=int(options.get("thc_max_iterations", 500)),
                thc_als_convergence_tolerance=float(
                    options.get("thc_als_convergence_tolerance", 1e-10)
                ),
                thc_ridge=float(options.get("thc_ridge", 1e-12)),
                thc_seed=int(options.get("thc_seed", 0)),
                thc_replacement_validation_atol=options.get(
                    "thc_replacement_validation_atol"
                ),
                thc_replacement_validation_rtol=options.get(
                    "thc_replacement_validation_rtol"
                ),
                **rr_options,
                **direct_options,
            ))
        solver.frozen = 0
        return mf, solver
    if method == "rr_canonical":
        solver = configure(RRCCSD(mf, eri_backend="canonical", **rr_options))
        solver.frozen = 0
        return mf, solver
    from gpu4pyscf.cc.thc_rrccsd import THCRRCCSD

    solver = configure(THCRRCCSD(
        mf,
        thc_fit_tol=float(options.get("thc_fit_tol", 1e-6)),
        thc_rank=options.get("thc_rank"),
        thc_initial_rank=options.get("thc_initial_rank"),
        thc_max_rank=options.get("thc_max_rank"),
        thc_rank_growth=float(options.get("thc_rank_growth", 1.5)),
        thc_orthogonality_cutoff=float(
            options.get("thc_orthogonality_cutoff", 1e-12)
        ),
        thc_orthogonality_tolerance=float(
            options.get("thc_orthogonality_tolerance", 1e-10)
        ),
        thc_max_iterations=int(options.get("thc_max_iterations", 500)),
        thc_als_convergence_tolerance=float(
            options.get("thc_als_convergence_tolerance", 1e-10)
        ),
        thc_ridge=float(options.get("thc_ridge", 1e-12)),
        thc_seed=int(options.get("thc_seed", 0)),
        thc_replacement_validation_atol=options.get(
            "thc_replacement_validation_atol"
        ),
        thc_replacement_validation_rtol=options.get(
            "thc_replacement_validation_rtol"
        ),
        eri_backend="canonical",
        **rr_options,
    ))
    solver.frozen = 0
    return mf, solver


def evaluate_energy(
    mol: Any,
    *,
    method: str,
    device: str,
    options: dict[str, Any] | None = None,
    solver_factory: Callable[[Any, str, str, dict[str, Any]], Any] | None = None,
    orbital_request: dict[str, Any] | None = None,
    runtime_source_state: dict[str, Any] | None = None,
) -> EnergyRecord:
    """Run one energy calculation, with an injectable CPU test factory."""

    options = dict(options or {})
    if "_gint_runtime_binding" in options:
        raise ValueError("internal GINT runtime binding cannot be caller supplied")
    if (
        solver_factory is None
        and runtime_source_state is not None
        and method in {"rr_cd", "thc_cd"}
        and str(options.get("gint_column_backend", "selected")) == "selected"
    ):
        options["_gint_runtime_binding"] = _recorded_gint_runtime_options(
            runtime_source_state, options.get("gint_runtime_gate_receipt")
        )
    if orbital_request is not None:
        options["_cp_orbital_request"] = orbital_request
    factory = solver_factory or _make_solver
    started = time.perf_counter()
    mf, solver = factory(mol, method, device, options)
    orbital_evidence = getattr(mf, "_cp_canonical_orbitals", None)
    if orbital_request is not None and not isinstance(orbital_evidence, dict):
        raise RuntimeError(
            "solver_factory did not apply the requested CP orbital bundle "
            "before constructing the solver"
        )
    output = solver.kernel()
    if method == "fno" and hasattr(solver, "corrected_e_corr"):
        # The CP observable must use the declared FNO-CCSD + Delta-MP2
        # method, rather than silently dropping the additive correction.
        e_corr = float(solver.corrected_e_corr)
    elif isinstance(output, tuple):
        e_corr = float(output[0])
    else:
        e_corr = float(output)
    hf_energy = float(mf.e_tot)
    total = hf_energy + e_corr
    scf_converged = bool(getattr(mf, "converged", False))
    converged = bool(getattr(solver, "converged", False))
    status = _effective_method_status(method, options)
    metadata = {
        "solver_type": type(solver).__name__,
        "scf_converged": scf_converged,
        **status,
        "canonical_orbitals": orbital_evidence,
    }
    method_metadata = getattr(solver, "method_metadata", None)
    if callable(method_metadata):
        realized = method_metadata()
        if not isinstance(realized, dict):
            raise TypeError("solver method_metadata() must return a dictionary")
        metadata["solver_method_metadata"] = _json_safe(realized)
        if method in {"rr_cd", "thc_cd"}:
            metadata["provider_runtime_gate"] = _json_safe(realized.get(
                "provider_runtime_gate",
                {
                    "required": (
                        str(options.get("gint_column_backend", "selected"))
                        == "selected"
                    ),
                    "validated": False,
                    "status": (
                        "not-recorded"
                        if str(options.get("gint_column_backend", "selected"))
                        == "selected" else "not-applicable"
                    ),
                    "validation_errors": [
                        "selected provider did not expose runtime-gate metadata"
                    ] if str(options.get(
                        "gint_column_backend", "selected"
                    )) == "selected" else [],
                },
            ))
    elif method in {"rr_cd", "thc_cd", "rr_canonical", "thc_canonical"}:
        raise TypeError(
            f"counterpoise {method} solver must expose method_metadata()"
        )
    if method == "fno" and hasattr(solver, "metadata"):
        metadata["fno"] = solver.metadata.audit_dict()
        metadata["fno_ccsd_energy"] = solver.energy_metadata
    if not scf_converged:
        raise RuntimeError(f"counterpoise SCF is unconverged for method={method}")
    if not converged:
        raise RuntimeError(f"counterpoise CCSD is unconverged for method={method}")
    return EnergyRecord(total, hf_energy, e_corr, converged, time.perf_counter() - started, metadata)


def _record_dict(record: EnergyRecord) -> dict[str, Any]:
    return {
        "total_energy_eh": record.total_energy_eh,
        "hf_energy_eh": record.hf_energy_eh,
        "correlation_energy_eh": record.correlation_energy_eh,
        "converged": record.converged,
        "wall_time_s": record.wall_time_s,
        "metadata": record.metadata,
    }


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    """Write exactly once, refusing to overwrite a previous CP result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing CP result: {path}") from exc


def run_counterpoise(
    case: dict[str, Any],
    *,
    method: str,
    device: str = "gpu",
    options: dict[str, Any] | None = None,
    solver_factory: Callable[[Any, str, str, dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run cluster and ghost-basis fragment energies and return a JSON-ready record."""

    options = dict(options or {})
    status = _effective_method_status(method, options)
    bundle_in = options.pop("orbital_bundle_in", None)
    bundle_out = options.pop("orbital_bundle_out", None)
    if bundle_in is not None and bundle_out is not None:
        raise ValueError(
            "orbital_bundle_in and orbital_bundle_out are mutually exclusive"
        )
    approximation, approximation_signature = approximation_identity(method, options)
    source = _source_state()
    geometry = molecule_atom_payload(case["geometry_angstrom"])
    fragments = infer_water_fragments(geometry)
    basis = str(case.get("basis", "cc-pVTZ"))
    cluster_atoms_hash = sha256_json(geometry)
    bundle_manager = None
    if bundle_in is not None or bundle_out is not None:
        bundle_manager = CPOrbitalBundle(
            mode="input" if bundle_in is not None else "output",
            path=Path(bundle_in if bundle_in is not None else bundle_out),
            case=case,
            cluster_atoms=geometry,
            source=source,
            scf_conv_tol=float(options.get("scf_conv_tol", 1e-10)),
        )

    def orbital_request(
        *,
        entry_key: str,
        role: str,
        fragment_id: int | None,
        active_atom_indices: Sequence[int],
        atoms: list[list[Any]],
    ) -> dict[str, Any] | None:
        if bundle_manager is None:
            return None
        return {
            "manager": bundle_manager,
            "entry_key": entry_key,
            "role": role,
            "fragment_id": fragment_id,
            "active_atom_indices": list(active_atom_indices),
            "atoms": atoms,
        }

    started = time.perf_counter()
    cluster_mol = build_molecule(geometry, basis=basis, verbose=0)
    cluster = evaluate_energy(
        cluster_mol,
        method=method,
        device=device,
        options=options,
        solver_factory=solver_factory,
        runtime_source_state=source,
        orbital_request=orbital_request(
            entry_key="cluster",
            role="cluster",
            fragment_id=None,
            active_atom_indices=tuple(range(len(geometry))),
            atoms=geometry,
        ),
    )
    fragment_records: list[dict[str, Any]] = []
    for fragment in fragments:
        atoms = fragment_atoms(geometry, fragment)
        fragment_mol = build_molecule(atoms, basis=basis, verbose=0)
        energy = evaluate_energy(
            fragment_mol,
            method=method,
            device=device,
            options=options,
            solver_factory=solver_factory,
            runtime_source_state=source,
            orbital_request=orbital_request(
                entry_key=f"fragment-{fragment.fragment_id:03d}",
                role="ghost-monomer",
                fragment_id=fragment.fragment_id,
                active_atom_indices=fragment.active_atom_indices,
                atoms=atoms,
            ),
        )
        fragment_records.append(
            {
                "fragment_id": fragment.fragment_id,
                "active_atom_indices": list(fragment.active_atom_indices),
                "atom_payload": atoms,
                "atom_payload_sha256": sha256_json(atoms),
                "energy": _record_dict(energy),
            }
        )
    cp_eh = counterpoise_energy(
        cluster.total_energy_eh,
        (record["energy"]["total_energy_eh"] for record in fragment_records),
    )
    expected_bundle_keys = {"cluster"} | {
        f"fragment-{fragment.fragment_id:03d}" for fragment in fragments
    }
    orbital_bundle = (
        None
        if bundle_manager is None
        else bundle_manager.finalize(expected_bundle_keys)
    )
    _complete_source_stability(source)
    source_eligible = bool(
        source.get("formal_counterpoise_eligible") is True
        and _formal_source_evidence_valid(source)
    )
    limitations: list[str] = []
    if source.get("stable_during_run") is not True:
        limitations.append("source tree or snapshot manifest changed during the counterpoise calculation")
    if not source_eligible:
        limitations.append(
            "formal counterpoise requires a stable, content-addressed, read-only source snapshot"
        )
        snapshot = source.get("snapshot")
        if isinstance(snapshot, dict):
            for reason in (*snapshot.get("reasons", []), *snapshot.get("reasons_at_end", [])):
                detail = f"snapshot evidence: {reason}"
                if detail not in limitations:
                    limitations.append(detail)
    bundle_eligible = bool(
        isinstance(orbital_bundle, dict)
        and orbital_bundle.get("complete") is True
        and orbital_bundle.get("entry_count") == len(fragments) + 1
    )
    if not bundle_eligible:
        limitations.append(
            "counterpoise accuracy acceptance requires an explicit complete "
            "cluster-plus-ghost-fragment canonical orbital bundle"
        )
    return {
        "schema": "gpu4pyscf.water8.counterpoise.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": {
            "id": case["id"],
            "source": case.get("source"),
            "basis": basis,
            "unit": "Angstrom",
            "spherical": True,
            "geometry_angstrom": geometry,
            "geometry_sha256": cluster_atoms_hash,
        },
        "method": method,
        **status,
        "approximation": approximation,
        "approximation_signature": approximation_signature,
        "device": device,
        "cc_max_cycle": int(options.get("cc_max_cycle", 50)),
        "max_memory_mb": int(options.get("max_memory_mb", 110000)),
        "performance_eligible": False,
        "provider_runtime_gates": {
            "cluster": cluster.metadata.get("provider_runtime_gate"),
            "fragments": [
                record["energy"]["metadata"].get("provider_runtime_gate")
                for record in fragment_records
            ],
            "method_performance_eligible": False,
            "reason": (
                "provider qualification is recorded independently; RR/THC "
                "production kernel and FP64 numerical gates remain pending"
            ),
        } if method in {"rr_cd", "thc_cd"} else None,
        "accuracy_eligible": bool(source_eligible and bundle_eligible),
        "all_calculations_converged": True,
        "limitations": limitations,
        "protocol": {
            "same_geometry_and_basis_for_cluster_and_fragments": True,
            "scf_conv_tol": options.get("scf_conv_tol", 1e-10),
            "cc_conv_tol": options.get("cc_conv_tol", 1e-8),
            "cc_conv_tol_normt": options.get("cc_conv_tol_normt", 1e-6),
            "cc_max_cycle": int(options.get("cc_max_cycle", 50)),
            "max_memory_mb": int(options.get("max_memory_mb", 110000)),
            "options": {
                key: value for key, value in options.items()
                if not key.startswith("_")
            },
            "approximation_signature": approximation_signature,
            "post_kernel_dense_t2_reconstruction": False,
            "fresh_scf_for_every_cluster_and_fragment": True,
            "canonical_orbitals_applied_before_solver_construction": (
                bundle_eligible
            ),
            "shared_orbital_bundle_required_for_accuracy": True,
        },
        "orbital_bundle": orbital_bundle,
        "cluster": {"atom_payload_sha256": cluster_atoms_hash, "energy": _record_dict(cluster)},
        "fragments": fragment_records,
        "interaction_energy_cp_eh": cp_eh,
        "interaction_energy_cp_kcal_mol": cp_eh * KCAL_PER_EH,
        "timing": {
            "counterpoise_wall_time_s": time.perf_counter() - started,
            "excluded_from_post_hf_speed_claim": True,
        },
        "software": {
            "python": platform.python_version(),
            "pyscf": _version("pyscf"),
            "gpu4pyscf": _version("gpu4pyscf") or _version("gpu4pyscf-cuda12x"),
            "git_revision": _git_revision(),
        },
        "source": source,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("water2-tz", "water4-tz", "water8-tz", "water8s4-tz"),
        required=True,
    )
    parser.add_argument("--method", choices=tuple(METHOD_STATUS), default="canonical")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--orbital-bundle-in",
        type=Path,
        default=None,
        help=(
            "load the exact cluster and ghost-monomer canonical orbitals "
            "created by an earlier CP run"
        ),
    )
    parser.add_argument(
        "--orbital-bundle-out",
        type=Path,
        default=None,
        help=(
            "write one complete cluster and ghost-monomer canonical orbital "
            "bundle and reapply every entry before its solver is constructed"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="validate fragments without importing PySCF")
    parser.add_argument("--scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--cc-conv-tol", type=float, default=1e-8)
    parser.add_argument("--cc-conv-tol-normt", type=float, default=1e-6)
    parser.add_argument("--cc-max-cycle", type=int, default=50)
    parser.add_argument("--max-memory-mb", type=int, default=110000)
    parser.add_argument("--eri-tol", type=float, default=1e-8)
    parser.add_argument("--direct-scf-tol", type=float, default=1e-13)
    parser.add_argument("--cd-max-rank", type=int, default=None)
    parser.add_argument(
        "--gint-column-backend",
        choices=("selected", "restricted-reference"),
        default="selected",
    )
    parser.add_argument(
        "--gint-column-kernel",
        choices=("reference", "grouped"),
        default="reference",
    )
    parser.add_argument("--gint-group-size", type=int, default=16)
    parser.add_argument("--gint-max-block-bytes", type=int, default=None)
    parser.add_argument("--gint-max-batch-size", type=int, default=32)
    parser.add_argument(
        "--gint-runtime-gate-receipt",
        type=Path,
        default=None,
        help=(
            "fixed external GINT qualification receipt; selected direct-CD CP "
            "runs remain runtime-gate pending when omitted"
        ),
    )
    parser.add_argument("--cd-mo-block-size", type=int, default=32)
    parser.add_argument("--rr-eig-cutoff", type=float, default=1e-6)
    parser.add_argument("--rr-max-rank", type=int, default=None)
    parser.add_argument("--denominator-tolerance", type=float, default=1e-10)
    parser.add_argument("--denominator-max-rank", type=int, default=None)
    parser.add_argument("--rr-initial-rank", type=int, default=32)
    parser.add_argument("--rr-solver-tolerance", type=float, default=1e-10)
    parser.add_argument("--rr-solver-maxiter", type=int, default=None)
    parser.add_argument("--rr-dense-fallback-dimension", type=int, default=64)
    parser.add_argument("--rr-ritz-residual-tolerance", type=float, default=None)
    parser.add_argument("--rr-auxiliary-block-size", type=int, default=1)
    parser.add_argument("--rr-virtual-block-size", type=int, default=8)
    parser.add_argument(
        "--rr-ring-kernel",
        choices=("reference", "gemm"),
        default="reference",
    )
    parser.add_argument("--precision", choices=("fp64",), default="fp64")
    parser.add_argument("--thc-fit-tol", type=float, default=1e-6)
    parser.add_argument("--thc-rank", type=int, default=None)
    parser.add_argument("--thc-initial-rank", type=int, default=None)
    parser.add_argument("--thc-max-rank", type=int, default=None)
    parser.add_argument("--thc-rank-growth", type=float, default=1.5)
    parser.add_argument("--thc-orthogonality-cutoff", type=float, default=1e-12)
    parser.add_argument(
        "--thc-orthogonality-tolerance", type=float, default=1e-10
    )
    parser.add_argument("--thc-max-iterations", type=int, default=500)
    parser.add_argument(
        "--thc-als-convergence-tolerance", type=float, default=1e-10
    )
    parser.add_argument("--thc-ridge", type=float, default=1e-12)
    parser.add_argument("--thc-seed", type=int, default=0)
    parser.add_argument(
        "--thc-replacement-validation-atol", type=float, default=None
    )
    parser.add_argument(
        "--thc-replacement-validation-rtol", type=float, default=None
    )
    parser.add_argument("--fno-thresh", type=float, default=1e-6)
    parser.add_argument("--fno-nvir-act", type=int, default=None)
    parser.add_argument("--fno-pct-occ", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    status = _effective_method_status(
        args.method, {"rr_ring_kernel": args.rr_ring_kernel}
    )
    if args.orbital_bundle_in is not None and args.orbital_bundle_out is not None:
        raise ValueError(
            "--orbital-bundle-in and --orbital-bundle-out are mutually exclusive"
        )
    if args.orbital_bundle_in is not None and not args.orbital_bundle_in.is_file():
        raise ValueError(
            f"CP orbital bundle input does not exist: {args.orbital_bundle_in}"
        )
    if args.orbital_bundle_out is not None and args.orbital_bundle_out.exists():
        raise ValueError(
            f"refusing to overwrite CP orbital bundle: {args.orbital_bundle_out}"
        )
    if args.method == "fno":
        if (
            isinstance(args.fno_thresh, bool)
            or not math.isfinite(float(args.fno_thresh))
            or float(args.fno_thresh) < 0.0
        ):
            raise ValueError("fno_thresh must be finite and non-negative")
        if args.fno_pct_occ is not None and (
            isinstance(args.fno_pct_occ, bool)
            or not math.isfinite(float(args.fno_pct_occ))
            or not 0.0 <= float(args.fno_pct_occ) <= 1.0
        ):
            raise ValueError("fno_pct_occ must lie in [0, 1]")
        if args.fno_nvir_act is not None and args.fno_nvir_act < 1:
            raise ValueError("fno_nvir_act must be a positive integer")
    if args.gint_runtime_gate_receipt is not None:
        if (
            args.method not in {"rr_cd", "thc_cd"}
            or args.gint_column_backend != "selected"
        ):
            raise ValueError(
                "--gint-runtime-gate-receipt applies only to selected direct-CD "
                "RR/THC counterpoise"
            )
        if not args.gint_runtime_gate_receipt.is_file():
            raise ValueError(
                "GINT runtime gate receipt does not exist: "
                f"{args.gint_runtime_gate_receipt}"
            )
    if (
        args.gint_column_backend != "selected"
        and args.gint_column_kernel != "reference"
    ):
        raise ValueError(
            "--gint-column-kernel applies only to --gint-column-backend selected"
        )
    if (
        args.gint_column_kernel == "grouped"
        and args.gint_runtime_gate_receipt is not None
    ):
        raise ValueError(
            "grouped selected-column scheduling cannot consume a release receipt"
        )
    cases = load_cases()
    case = cases[args.case]
    geometry = molecule_atom_payload(case["geometry_angstrom"])
    fragments = infer_water_fragments(geometry)
    if args.dry_run:
        plan = {
            "case": args.case,
            "method": args.method,
            "device": args.device,
            **status,
            "basis": case["basis"],
            "geometry_sha256": sha256_json(geometry),
            "fragment_atom_indices": [list(fragment.atom_indices) for fragment in fragments],
            "output": str(args.output),
            "cc_max_cycle": args.cc_max_cycle,
            "max_memory_mb": args.max_memory_mb,
            "accuracy_eligible": False,
            "orbital_bundle": {
                "input": (
                    None if args.orbital_bundle_in is None
                    else str(args.orbital_bundle_in)
                ),
                "output": (
                    None if args.orbital_bundle_out is None
                    else str(args.orbital_bundle_out)
                ),
                "required_for_accuracy": True,
            },
            "limitations": [
                "dry-run validates inputs only and never produces counterpoise accuracy evidence"
            ],
            "timing_excluded_from_post_hf_speed_claim": True,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    options = {
        "scf_conv_tol": args.scf_conv_tol,
        "cc_conv_tol": args.cc_conv_tol,
        "cc_conv_tol_normt": args.cc_conv_tol_normt,
        "cc_max_cycle": args.cc_max_cycle,
        "max_memory_mb": args.max_memory_mb,
        "eri_tol": args.eri_tol,
        "direct_scf_tol": args.direct_scf_tol,
        "cd_max_rank": args.cd_max_rank,
        "gint_column_backend": args.gint_column_backend,
        "gint_column_kernel": args.gint_column_kernel,
        "gint_group_size": args.gint_group_size,
        "gint_max_block_bytes": args.gint_max_block_bytes,
        "gint_max_batch_size": args.gint_max_batch_size,
        "gint_runtime_gate_receipt": (
            None if args.gint_runtime_gate_receipt is None
            else str(args.gint_runtime_gate_receipt.expanduser().resolve())
        ),
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
        "rr_ring_kernel": args.rr_ring_kernel,
        "precision": args.precision,
        "thc_fit_tol": args.thc_fit_tol,
        "thc_rank": args.thc_rank,
        "thc_initial_rank": args.thc_initial_rank,
        "thc_max_rank": args.thc_max_rank,
        "thc_rank_growth": args.thc_rank_growth,
        "thc_orthogonality_cutoff": args.thc_orthogonality_cutoff,
        "thc_orthogonality_tolerance": args.thc_orthogonality_tolerance,
        "thc_max_iterations": args.thc_max_iterations,
        "thc_als_convergence_tolerance": args.thc_als_convergence_tolerance,
        "thc_ridge": args.thc_ridge,
        "thc_seed": args.thc_seed,
        "thc_replacement_validation_atol": (
            args.thc_replacement_validation_atol
        ),
        "thc_replacement_validation_rtol": (
            args.thc_replacement_validation_rtol
        ),
        "fno_thresh": args.fno_thresh,
        "fno_nvir_act": args.fno_nvir_act,
        "fno_pct_occ": args.fno_pct_occ,
        "orbital_bundle_in": args.orbital_bundle_in,
        "orbital_bundle_out": args.orbital_bundle_out,
    }
    result = run_counterpoise(case, method=args.method, device=args.device, options=options)
    # The staged FNO gate binds every CP result to its submitted logical job.
    # Keep this metadata in the result itself so a copied path cannot stand in
    # for a different Slurm process or threshold.
    logical_job_id = os.environ.get("FNO_LOGICAL_JOB_ID")
    if logical_job_id:
        result["slurm"] = {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_JOB_NAME": os.environ.get("SLURM_JOB_NAME"),
        }
        result["orchestration"] = {
            "logical_job_id": logical_job_id,
            "run_id": os.environ.get("CP_RUN_ID"),
            "stage": os.environ.get("FNO_STAGE", "fno-g2"),
        }
    _write_json_once(args.output, result)
    print(json.dumps({"output": str(args.output), "interaction_energy_cp_eh": result["interaction_energy_cp_eh"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
