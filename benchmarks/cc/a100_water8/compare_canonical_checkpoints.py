#!/usr/bin/env python3
"""Compare two fresh-process canonical CC checkpoints.

Restart-v2 checkpoints carry their exact canonical MOs and are compared only
when those MOs are byte-identical.  Legacy restart-v1 checkpoints retain the
older diagonal-sign-gauge comparison for backward-compatible evidence reads.
No arbitrary occupied/virtual rotation is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "gpu4pyscf.water8.canonical-checkpoint-parity.v2"
RESTART_SCHEMA_V1 = "gpu4pyscf.cc.restart.v1"
RESTART_SCHEMA_V2 = "gpu4pyscf.cc.restart.v2"
ORBITAL_IDENTITY_SCHEMA = "gpu4pyscf.cc.canonical-orbital-identity.v1"
DEFAULT_NONZERO_TOL = 1.0e-12
DEFAULT_ENERGY_TOL = 1.0e-8
DEFAULT_RESIDUAL_TOL = 1.0e-6
DEFAULT_AMPLITUDE_MAX_TOL = 1.0e-6
DEFAULT_AMPLITUDE_L2_TOL = 1.0e-6


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _orbital_fingerprint(
    identity: dict[str, Any], arrays: dict[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(identity).encode("ascii"))
    for name in ("mo_coeff", "mo_occ", "mo_energy"):
        array = arrays[name]
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: benchmark JSON must contain one object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash checkpoint {path}: {exc}") from exc
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Return a stable, compact representation for provenance comparison."""
    if isinstance(value, dict):
        return {str(k): _jsonable(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in benchmark provenance")
        return value
    return value


def _method_family(method: Any) -> Any:
    # canonical_legacy and canonical are intentionally paired for the G1 A/B
    # test.  Their plan settings and checkpoints must still agree otherwise.
    if method in {"canonical", "canonical_legacy"}:
        return "canonical"
    return method


def _scientific_signature(record: dict[str, Any]) -> dict[str, Any]:
    plan = record.get("plan")
    case = record.get("case")
    settings = plan.get("settings") if isinstance(plan, dict) else None
    expected = case.get("expected_dimensions") if isinstance(case, dict) else None
    if not isinstance(plan, dict) or not isinstance(case, dict) or not isinstance(settings, dict):
        raise ValueError("benchmark is missing case, plan, or plan.settings provenance")
    # Run labels and the implementation's lifecycle bookkeeping are excluded;
    # every scientific input and numerical setting is retained.
    return _jsonable({
        "case_id": record.get("case_id"),
        "case": case,
        "geometry_sha256": record.get("geometry_sha256"),
        "basis": plan.get("basis"),
        "reference": plan.get("reference"),
        "spherical": plan.get("spherical"),
        "charge": plan.get("charge"),
        "spin": plan.get("spin"),
        "expected_dimensions": expected,
        "orbital_dimensions": record.get("orbital_dimensions"),
        "method_family": _method_family(record.get("method")),
        "approximation": record.get("approximation"),
        "approximation_signature": record.get("approximation_signature"),
        "method_class": plan.get("method_class"),
        "protocol_schema": plan.get("protocol_schema"),
        "settings": settings,
    })


def _source_signature(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("source")
    protocol = record.get("source_protocol")
    if not isinstance(source, dict) or not isinstance(protocol, dict):
        raise ValueError("benchmark is missing source or source_protocol provenance")
    snapshot = source.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("benchmark is missing source snapshot provenance")
    # Paths, host-local repository names, and timestamps are not source
    # identity.  Content/tree and immutable-run evidence are.
    fields = {
        "base_revision": source.get("base_revision"),
        "revision": source.get("revision"),
        "tree_sha256": source.get("tree_sha256"),
        "tree_sha256_at_end": source.get("tree_sha256_at_end"),
        "files_hashed": source.get("files_hashed"),
        "files_hashed_at_end": source.get("files_hashed_at_end"),
        "source_kind": source.get("source_kind"),
        "stable_during_run": source.get("stable_during_run"),
        "formal_performance_eligible": source.get("formal_performance_eligible"),
        "frozen_base_commit": protocol.get("frozen_base_commit"),
        "snapshot_tree_sha256": snapshot.get("tree_sha256"),
        "snapshot_tree_sha256_at_end": snapshot.get("tree_sha256_at_end"),
        "snapshot_manifest_sha256_at_end": snapshot.get("manifest_sha256_at_end"),
        "snapshot_stable_during_run": snapshot.get("stable_during_run"),
        "snapshot_read_only_at_end": snapshot.get("read_only_at_end"),
        "snapshot_runtime_binaries_valid_at_end": snapshot.get("runtime_binaries_valid_at_end"),
        "snapshot_profile": (
            snapshot.get("manifest", {}).get("local_provenance", {}).get("deployment_profile")
            if isinstance(snapshot.get("manifest"), dict) else None
        ),
    }
    if not fields["revision"] or not fields["tree_sha256"]:
        raise ValueError("benchmark source identity is incomplete")
    return _jsonable(fields)


def _hardware_signature(record: dict[str, Any]) -> dict[str, Any]:
    gpu = record.get("gpu")
    pcie = record.get("pcie")
    visible = pcie.get("visible_gpus") if isinstance(pcie, dict) else None
    device = gpu.get("device_0") if isinstance(gpu, dict) else None
    env = record.get("thread_environment")
    if not isinstance(gpu, dict) or not isinstance(device, dict) or not isinstance(pcie, dict) or not isinstance(visible, list) or len(visible) != 1:
        raise ValueError("benchmark hardware provenance is incomplete")
    if not isinstance(env, dict):
        raise ValueError("benchmark thread environment provenance is missing")
    link = visible[0]
    if not isinstance(link, dict):
        raise ValueError("benchmark PCIe provenance is malformed")
    # UUID/driver version are intentionally omitted: they identify a device
    # instance/driver build, while the contract is the architecture/topology.
    return _jsonable({
        "host": record.get("host"),
        "platform": record.get("platform"),
        "cpu_model": record.get("cpu_model"),
        "affinity": record.get("affinity"),
        "thread_environment": env,
        "gpu": {
            "device_count": gpu.get("device_count"),
            "name": device.get("name"),
            "major": device.get("major"),
            "minor": device.get("minor"),
            "totalGlobalMem": device.get("totalGlobalMem"),
        },
        "pcie": {
            "name": link.get("name"),
            "numa_node": link.get("numa_node"),
            "pci_bus_id": link.get("pci_bus_id"),
            "max_link_speed": link.get("max_link_speed"),
            "max_link_width": link.get("max_link_width"),
            "current_link_speed": link.get("current_link_speed"),
            "current_link_width": link.get("current_link_width"),
            "pcie_gen_current": link.get("pcie_gen_current"),
            "pcie_width_current": link.get("pcie_width_current"),
        },
        "topology": record.get("topology"),
    })


def _timing_signature(record: dict[str, Any]) -> str:
    timing = record.get("timing_definition")
    if not isinstance(timing, str) or not timing:
        raise ValueError("benchmark timing_definition is missing")
    return timing


def _validate_run(record: dict[str, Any], json_path: Path) -> dict[str, Any]:
    reasons = []
    if record.get("status") != "completed":
        reasons.append("status is not completed")
    if record.get("cc_converged") is not True:
        reasons.append("cc_converged is not true")
    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, dict):
        reasons.append("checkpoint metadata is missing")
    elif checkpoint.get("included_in_post_hf") is not True:
        reasons.append("checkpoint was not included in timed post-HF")
    residual = record.get("residual")
    if not isinstance(residual, dict) or residual.get("available") is not True:
        reasons.append("final residual evidence is missing")
    if reasons:
        raise ValueError(f"{json_path}: " + "; ".join(reasons))
    return {
        "scientific": _scientific_signature(record),
        "source": _source_signature(record),
        "hardware": _hardware_signature(record),
        "timing_definition": _timing_signature(record),
        "e_corr": record.get("e_corr"),
        "equation_residual": residual.get("equation_norm"),
        "cycles": record.get("cc_iterations"),
        "checkpoint": checkpoint,
    }


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} is missing or non-finite")
    return float(value)


def _checkpoint_path(record: dict[str, Any], override: Path | None, label: str) -> Path:
    metadata = record.get("checkpoint")
    if not isinstance(metadata, dict):
        raise ValueError(f"{label}: checkpoint metadata is missing")
    path = override if override is not None else Path(str(metadata.get("path", "")))
    if not str(path) or not path.is_file():
        raise ValueError(f"{label}: checkpoint is missing: {path}")
    declared = metadata.get("sha256")
    if not isinstance(declared, str) or len(declared) != 64 or any(c not in "0123456789abcdef" for c in declared.lower()):
        raise ValueError(f"{label}: checkpoint sha256 metadata is missing or malformed")
    actual = _sha256(path)
    if actual != declared.lower():
        raise ValueError(f"{label}: checkpoint sha256 mismatch (declared {declared}, actual {actual})")
    return path


def _scalar_npz(payload: Any, key: str, label: str) -> Any:
    if key not in payload:
        raise ValueError(f"{label}: checkpoint key {key!r} is missing")
    value = np.asarray(payload[key])
    if value.shape != (1,):
        raise ValueError(f"{label}: checkpoint key {key!r} has invalid shape {value.shape}")
    return value[0]


def _load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception as exc:  # numpy raises several format-specific errors
        raise ValueError(f"{label}: cannot load checkpoint {path}: {exc}") from exc
    try:
        keys = set(payload.files)
        common = {
            "checkpoint_schema", "e_corr", "method", "representation", "t1",
            "t2",
        }
        missing = common.difference(keys)
        if missing:
            raise ValueError(
                f"{label}: checkpoint keys are missing {sorted(missing)}"
            )
        schema = str(_scalar_npz(payload, "checkpoint_schema", label))
        if schema == RESTART_SCHEMA_V1:
            required = common
        elif schema == RESTART_SCHEMA_V2:
            required = common | {
                "mo_coeff",
                "mo_occ",
                "mo_energy",
                "orbital_identity_json",
                "orbital_fingerprint",
                "orbital_artifact_sha256",
                "orbital_origin",
            }
        else:
            raise ValueError(f"{label}: unsupported checkpoint schema {schema!r}")
        if keys != required:
            raise ValueError(
                f"{label}: checkpoint keys {sorted(keys)} do not equal "
                f"{sorted(required)}"
            )
        method = str(_scalar_npz(payload, "method", label))
        representation = str(_scalar_npz(payload, "representation", label))
        if representation != "dense-t2":
            raise ValueError(f"{label}: only dense-t2 checkpoints can be compared")
        e_corr = _finite_scalar(_scalar_npz(payload, "e_corr", label), f"{label} checkpoint e_corr")
        t1 = np.asarray(payload["t1"])
        t2 = np.asarray(payload["t2"])
        if t1.ndim != 2 or t2.ndim != 4:
            raise ValueError(f"{label}: t1/t2 must be rank 2/4, got {t1.ndim}/{t2.ndim}")
        if t2.shape[:2] != (t1.shape[0], t1.shape[0]) or t2.shape[2:] != (t1.shape[1], t1.shape[1]):
            raise ValueError(f"{label}: t1/t2 shapes are inconsistent: {t1.shape}/{t2.shape}")
        for key, array in (("t1", t1), ("t2", t2)):
            if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
                raise ValueError(f"{label}: {key} must be a real numeric array")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{label}: {key} contains NaN or infinity")
        checkpoint = {
            "schema": schema,
            "method": method,
            "representation": representation,
            "e_corr": e_corr,
            "t1": t1,
            "t2": t2,
            "orbitals": None,
        }
        if schema == RESTART_SCHEMA_V2:
            try:
                identity = json.loads(str(_scalar_npz(
                    payload, "orbital_identity_json", label
                )))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{label}: orbital identity JSON is malformed"
                ) from exc
            if (
                not isinstance(identity, dict)
                or identity.get("schema") != ORBITAL_IDENTITY_SCHEMA
            ):
                raise ValueError(f"{label}: orbital identity is malformed")
            orbital_arrays = {
                name: np.asarray(payload[name])
                for name in ("mo_coeff", "mo_occ", "mo_energy")
            }
            for key, array in orbital_arrays.items():
                if array.dtype != np.dtype(np.float64):
                    raise ValueError(
                        f"{label}: {key} must use exact FP64 storage"
                    )
                if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"{label}: {key} must be finite and real"
                    )
            nao = identity.get("nao")
            nmo = identity.get("nmo")
            nocc = identity.get("nocc")
            nvir = identity.get("nvir")
            if (
                not all(
                    isinstance(item, int) and not isinstance(item, bool)
                    for item in (nao, nmo, nocc, nvir)
                )
                or orbital_arrays["mo_coeff"].shape != (nao, nmo)
                or orbital_arrays["mo_occ"].shape != (nmo,)
                or orbital_arrays["mo_energy"].shape != (nmo,)
                or t1.shape != (nocc, nvir)
            ):
                raise ValueError(
                    f"{label}: orbital identity, MO arrays, and amplitudes "
                    "have inconsistent dimensions"
                )
            fingerprint = str(_scalar_npz(
                payload, "orbital_fingerprint", label
            ))
            actual_fingerprint = _orbital_fingerprint(
                identity, orbital_arrays
            )
            if fingerprint != actual_fingerprint:
                raise ValueError(
                    f"{label}: checkpoint orbital fingerprint mismatch"
                )
            artifact_sha256 = str(_scalar_npz(
                payload, "orbital_artifact_sha256", label
            ))
            origin = str(_scalar_npz(payload, "orbital_origin", label))
            artifact_modes = {"loaded", "written-and-reloaded"}
            if origin in artifact_modes and not _is_sha256(artifact_sha256):
                raise ValueError(
                    f"{label}: orbital origin {origin!r} requires a valid "
                    "artifact SHA-256"
                )
            if origin == "fresh-scf-memory" and artifact_sha256:
                raise ValueError(
                    f"{label}: fresh-scf-memory must not claim an artifact "
                    "SHA-256"
                )
            if origin not in artifact_modes | {"fresh-scf-memory"}:
                raise ValueError(f"{label}: unsupported orbital origin {origin!r}")
            checkpoint["orbitals"] = {
                "identity": identity,
                "fingerprint": fingerprint,
                "artifact_sha256": artifact_sha256,
                "origin": origin,
                "arrays": orbital_arrays,
            }
        return checkpoint
    finally:
        payload.close()


def _phase_graph(left: np.ndarray, right: np.ndarray, nonzero_tol: float) -> dict[str, Any]:
    nocc, nvir = left.shape
    if right.shape != left.shape:
        raise ValueError(f"t1 shape mismatch: {left.shape} versus {right.shape}")
    left_nonzero = np.abs(left) > nonzero_tol
    right_nonzero = np.abs(right) > nonzero_tol
    nodes = nocc + nvir
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(nodes)]
    edge_count = 0
    for i, a in zip(*np.where(left_nonzero & right_nonzero)):
        relation = 1 if float(left[i, a]) * float(right[i, a]) >= 0.0 else -1
        v = nocc + int(a)
        adjacency[int(i)].append((v, relation))
        adjacency[v].append((int(i), relation))
        edge_count += 1
    signs = np.zeros(nodes, dtype=np.int8)
    components: list[dict[str, Any]] = []
    for root in range(nodes):
        if signs[root] != 0:
            continue
        signs[root] = 1  # deterministic representative for each component
        stack = [root]
        members: list[int] = []
        edges = 0
        while stack:
            vertex = stack.pop()
            members.append(vertex)
            for neighbor, relation in adjacency[vertex]:
                edges += 1
                expected = int(signs[vertex]) * relation
                if signs[neighbor] == 0:
                    signs[neighbor] = expected
                    stack.append(neighbor)
                elif int(signs[neighbor]) != expected:
                    raise ValueError(
                        "inconsistent MO sign constraints in nonzero t1 graph "
                        f"at nodes {vertex} and {neighbor}"
                    )
        components.append({
            "root": root,
            "occupied": sorted(v for v in members if v < nocc),
            "virtual": sorted(v - nocc for v in members if v >= nocc),
            "nodes": len(members),
            "edges": edges // 2,
        })
    return {
        "nonzero_tolerance": nonzero_tol,
        "edge_count": edge_count,
        "component_count": len(components),
        "components": components,
        "occupied_signs": signs[:nocc].astype(int).tolist(),
        "virtual_signs": signs[nocc:].astype(int).tolist(),
        "scope": "diagonal MO sign gauge inferred from bipartite nonzero t1 graph; orbital rotations are not handled",
    }


def _errors(
    left: np.ndarray,
    right: np.ndarray,
    factors: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    max_abs = 0.0
    sum_sq = 0.0
    if left.ndim == 4:
        for i in range(left.shape[0]):
            right_block = right[i]
            if isinstance(factors, tuple):
                occupied, virtual = factors
                right_block = right_block * (
                    occupied[i]
                    * occupied[:, None, None]
                    * virtual[None, :, None]
                    * virtual[None, None, :]
                )
            elif factors is not None:
                right_block = right_block * factors[i]
            delta = np.asarray(left[i] - right_block, dtype=np.float64)
            max_abs = max(max_abs, float(np.max(np.abs(delta))))
            sum_sq += float(np.sum(delta * delta))
    else:
        aligned = right if factors is None else right * factors
        delta = np.asarray(left - aligned, dtype=np.float64)
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        sum_sq = float(np.sum(delta * delta))
    return {"max_abs": max_abs, "l2": math.sqrt(sum_sq)}


def _same(name: str, left: Any, right: Any) -> None:
    if left != right:
        raise ValueError(f"{name} provenance mismatch")


def _validate_v2_orbital_provenance(
    checkpoint: dict[str, Any],
    record: dict[str, Any],
    label: str,
) -> None:
    orbital = checkpoint.get("orbitals")
    metadata = record.get("canonical_orbitals")
    if not isinstance(orbital, dict) or not isinstance(metadata, dict):
        raise ValueError(
            f"{label}: restart-v2 requires canonical_orbitals benchmark "
            "provenance"
        )
    if metadata.get("orbital_fingerprint") != orbital["fingerprint"]:
        raise ValueError(
            f"{label}: checkpoint orbital fingerprint disagrees with "
            "benchmark JSON"
        )
    record_artifact_sha = metadata.get("artifact_sha256") or ""
    record_mode = metadata.get("mode")
    if record_mode != orbital["origin"]:
        raise ValueError(
            f"{label}: checkpoint orbital origin disagrees with benchmark JSON"
        )
    if record_mode in {"loaded", "written-and-reloaded"} and not _is_sha256(
        record_artifact_sha
    ):
        raise ValueError(
            f"{label}: benchmark orbital artifact SHA-256 is invalid"
        )
    if record_mode == "fresh-scf-memory" and record_artifact_sha:
        raise ValueError(
            f"{label}: fresh-SCF benchmark metadata must not claim an "
            "orbital artifact SHA-256"
        )
    if record_artifact_sha != orbital["artifact_sha256"]:
        raise ValueError(
            f"{label}: checkpoint orbital artifact hash disagrees with "
            "benchmark JSON"
        )
    identity = orbital["identity"]
    plan = record.get("plan", {})
    expected = {
        "case_id": record.get("case_id"),
        "geometry_sha256": record.get("geometry_sha256"),
        "basis": plan.get("basis"),
        "reference": plan.get("reference"),
        "spherical": plan.get("spherical"),
        "charge": plan.get("charge"),
        "spin": plan.get("spin"),
    }
    for name, value in expected.items():
        if identity.get(name) != value:
            raise ValueError(
                f"{label}: checkpoint orbital identity field {name!r} "
                "disagrees with benchmark JSON"
            )
    dimensions = record.get("orbital_dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError(f"{label}: orbital dimensions are missing")
    for name in ("nao", "nocc", "nvir"):
        if identity.get(name) != dimensions.get(name):
            raise ValueError(
                f"{label}: checkpoint orbital {name} disagrees with "
                "benchmark JSON"
            )
    checkpoint_metadata = record.get("checkpoint")
    if (
        isinstance(checkpoint_metadata, dict)
        and checkpoint_metadata.get("checkpoint_schema") is not None
        and checkpoint_metadata.get("checkpoint_schema") != RESTART_SCHEMA_V2
    ):
        raise ValueError(
            f"{label}: checkpoint schema disagrees with benchmark JSON"
        )


def _require_exact_same_orbitals(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    left_orbitals = left["orbitals"]
    right_orbitals = right["orbitals"]
    if left_orbitals["identity"] != right_orbitals["identity"]:
        raise ValueError(
            "canonical MO scientific identities differ; exact amplitude "
            "comparison is unsafe"
        )
    differing = [
        name
        for name in ("mo_coeff", "mo_occ", "mo_energy")
        if not np.array_equal(
            left_orbitals["arrays"][name], right_orbitals["arrays"][name]
        )
    ]
    if (
        left_orbitals["fingerprint"] != right_orbitals["fingerprint"]
        or differing
    ):
        detail = ", ".join(differing) if differing else "fingerprint"
        raise ValueError(
            "canonical MO orbitals differ "
            f"({detail}); exact amplitude comparison is unsafe. Re-run both "
            "fresh processes with the same --orbital-artifact-in file"
        )
    left_artifact = left_orbitals["artifact_sha256"]
    right_artifact = right_orbitals["artifact_sha256"]
    if left_artifact != right_artifact:
        raise ValueError("canonical orbital artifact hashes differ")
    return {
        "mode": "exact-canonical-mo",
        "orbital_fingerprint": left_orbitals["fingerprint"],
        "shared_artifact_sha256": left_artifact or None,
        "shared_artifact": bool(left_artifact),
        "left_origin": left_orbitals["origin"],
        "right_origin": right_orbitals["origin"],
        "arrays_byte_identical": True,
        "rotation_or_sign_alignment_applied": False,
    }


def compare_checkpoints(
    left_json: Path,
    right_json: Path,
    *,
    left_checkpoint: Path | None = None,
    right_checkpoint: Path | None = None,
    nonzero_tol: float = DEFAULT_NONZERO_TOL,
    energy_tol: float = DEFAULT_ENERGY_TOL,
    residual_tol: float = DEFAULT_RESIDUAL_TOL,
    amplitude_max_tol: float = DEFAULT_AMPLITUDE_MAX_TOL,
    amplitude_l2_tol: float = DEFAULT_AMPLITUDE_L2_TOL,
) -> dict[str, Any]:
    if nonzero_tol < 0 or energy_tol < 0 or residual_tol < 0 or amplitude_max_tol < 0 or amplitude_l2_tol < 0:
        raise ValueError("all tolerances must be non-negative")
    left_record = _load_json(left_json)
    right_record = _load_json(right_json)
    left_meta = _validate_run(left_record, left_json)
    right_meta = _validate_run(right_record, right_json)
    for name in ("scientific", "source", "hardware", "timing_definition"):
        _same(name, left_meta[name], right_meta[name])
    left_path = _checkpoint_path(left_record, left_checkpoint, "left")
    right_path = _checkpoint_path(right_record, right_checkpoint, "right")
    left = _load_checkpoint(left_path, "left")
    right = _load_checkpoint(right_path, "right")
    if _method_family(left["method"]) != _method_family(right["method"]):
        raise ValueError("checkpoint method provenance mismatch")
    if _method_family(left["method"]) != _method_family(left_record.get("method")):
        raise ValueError("left checkpoint method disagrees with benchmark JSON")
    if _method_family(right["method"]) != _method_family(right_record.get("method")):
        raise ValueError("right checkpoint method disagrees with benchmark JSON")
    if left["representation"] != right["representation"]:
        raise ValueError("checkpoint representation mismatch")
    if left["schema"] != right["schema"]:
        raise ValueError("checkpoint restart schema mismatch")
    if left["t1"].shape != right["t1"].shape or left["t2"].shape != right["t2"].shape:
        raise ValueError("checkpoint amplitude shapes mismatch")
    dimensions = left_meta["scientific"].get("orbital_dimensions")
    if isinstance(dimensions, dict):
        expected_shape = (dimensions.get("nocc"), dimensions.get("nvir"))
        if expected_shape != left["t1"].shape:
            raise ValueError("checkpoint t1 shape disagrees with orbital dimensions")
    left_energy = _finite_scalar(left_meta["e_corr"], "left benchmark e_corr")
    right_energy = _finite_scalar(right_meta["e_corr"], "right benchmark e_corr")
    left_residual = _finite_scalar(left_meta["equation_residual"], "left equation residual")
    right_residual = _finite_scalar(right_meta["equation_residual"], "right equation residual")
    left_cycles = left_meta["cycles"]
    right_cycles = right_meta["cycles"]
    if isinstance(left_cycles, bool) or not isinstance(left_cycles, int) or left_cycles <= 0 or isinstance(right_cycles, bool) or not isinstance(right_cycles, int) or right_cycles <= 0:
        raise ValueError("benchmark cc_iterations is missing or invalid")
    raw_t1 = _errors(left["t1"], right["t1"])
    raw_t2 = _errors(left["t2"], right["t2"])
    if left["schema"] == RESTART_SCHEMA_V2:
        _validate_v2_orbital_provenance(left, left_record, "left")
        _validate_v2_orbital_provenance(right, right_record, "right")
        orbital_alignment = _require_exact_same_orbitals(left, right)
        phase = None
        aligned_t1 = raw_t1
        aligned_t2 = raw_t2
        scope_note = (
            "Exact byte-identical canonical MOs; no sign or rotation "
            "alignment was applied."
        )
        amplitude_labels = {
            "t1_exact_orbitals": aligned_t1,
            "t2_exact_orbitals": aligned_t2,
        }
    else:
        phase = _phase_graph(left["t1"], right["t1"], nonzero_tol)
        so = np.asarray(phase["occupied_signs"], dtype=np.float64)
        sv = np.asarray(phase["virtual_signs"], dtype=np.float64)
        t1_factors = so[:, None] * sv[None, :]
        # Build only the small per-i factor block so water8 t2 comparison does
        # not materialize another full (nocc,nocc,nvir,nvir) array.
        aligned_t1 = _errors(left["t1"], right["t1"], t1_factors)
        aligned_t2 = _errors(left["t2"], right["t2"], (so, sv))
        orbital_alignment = {
            "mode": "legacy-diagonal-sign-gauge",
            "rotation_or_sign_alignment_applied": True,
            "warning": "arbitrary occupied/virtual rotations are unsupported",
        }
        scope_note = "MO sign gauge only; orbital rotations are not handled."
        amplitude_labels = {}
    energy_delta = abs(left_energy - right_energy)
    checkpoint_energy_delta = abs(left["e_corr"] - right["e_corr"])
    residual_delta = abs(left_residual - right_residual)
    checks = {
        "energy": {"left": left_energy, "right": right_energy, "benchmark_delta": energy_delta, "checkpoint_delta": checkpoint_energy_delta, "tolerance": energy_tol, "passed": energy_delta <= energy_tol and checkpoint_energy_delta <= energy_tol},
        "residual": {"left": left_residual, "right": right_residual, "delta": residual_delta, "tolerance": residual_tol, "passed": residual_delta <= residual_tol},
        "cycles": {
            "left": left_cycles,
            "right": right_cycles,
            "matched": left_cycles == right_cycles,
            "acceptance_gate": False,
        },
        "amplitudes": {
            "t1_raw": raw_t1, "t1_gauge_aligned": aligned_t1,
            "t2_raw": raw_t2, "t2_gauge_aligned": aligned_t2,
            **amplitude_labels,
            "max_tolerance": amplitude_max_tol, "l2_tolerance": amplitude_l2_tol,
            "passed": all(
                item[metric] <= (amplitude_max_tol if metric == "max_abs" else amplitude_l2_tol)
                for item in (aligned_t1, aligned_t2)
                for metric in ("max_abs", "l2")
            ),
        },
    }
    passed = all(
        checks[name]["passed"]
        for name in ("energy", "residual", "amplitudes")
    )
    return {
        "schema": SCHEMA,
        "passed": passed,
        "scope_note": scope_note,
        "restart_schema": left["schema"],
        "left": {"benchmark_json": str(left_json), "checkpoint": str(left_path), "checkpoint_sha256": _sha256(left_path)},
        "right": {"benchmark_json": str(right_json), "checkpoint": str(right_path), "checkpoint_sha256": _sha256(right_path)},
        "provenance": {"scientific": left_meta["scientific"], "source": left_meta["source"], "hardware": left_meta["hardware"], "timing_definition": left_meta["timing_definition"], "matched": True},
        "phase_gauge": phase,
        "orbital_alignment": orbital_alignment,
        "checks": checks,
        "tolerances": {"nonzero_t1": nonzero_tol, "energy": energy_tol, "residual": residual_tol, "amplitude_max": amplitude_max_tol, "amplitude_l2": amplitude_l2_tol},
    }


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {path}") from exc


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_json", type=Path)
    parser.add_argument("right_json", type=Path)
    parser.add_argument("--left-checkpoint", type=Path)
    parser.add_argument("--right-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nonzero-tol", type=float, default=DEFAULT_NONZERO_TOL)
    parser.add_argument("--energy-tol", type=float, default=DEFAULT_ENERGY_TOL)
    parser.add_argument("--residual-tol", type=float, default=DEFAULT_RESIDUAL_TOL)
    parser.add_argument("--amplitude-max-tol", type=float, default=DEFAULT_AMPLITUDE_MAX_TOL)
    parser.add_argument("--amplitude-l2-tol", type=float, default=DEFAULT_AMPLITUDE_L2_TOL)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = compare_checkpoints(
            args.left_json, args.right_json,
            left_checkpoint=args.left_checkpoint, right_checkpoint=args.right_checkpoint,
            nonzero_tol=args.nonzero_tol, energy_tol=args.energy_tol,
            residual_tol=args.residual_tol, amplitude_max_tol=args.amplitude_max_tol,
            amplitude_l2_tol=args.amplitude_l2_tol,
        )
        exit_code = 0 if payload["passed"] else 1
    except (ValueError, OSError) as exc:
        payload = {
            "schema": SCHEMA,
            "passed": False,
            "error": str(exc),
            "scope_note": (
                "Restart-v2 requires exact canonical MOs; restart-v1 supports "
                "only a diagonal MO sign gauge. Orbital rotations are not "
                "inferred."
            ),
        }
        exit_code = 2
    if args.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _write_once(args.output, payload)
        print(args.output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
