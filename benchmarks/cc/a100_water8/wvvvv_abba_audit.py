#!/usr/bin/env python3
"""Audit fused versus two-ladder Wvvvv from one resident RR/CD state.

This is deliberately a microbenchmark, not a production performance gate.  A
single RHF, direct-CD, projector, denominator, and initial RR-amplitude
lifecycle is built once.  Both Wvvvv implementations then receive the exact
same CuPy objects in a warmed ABBA sequence at three scopes: the Wvvvv term,
the complete doubles numerator plus denominator solve, and one full Jacobi
step.  CUDA events measure device elapsed time; all comparisons, hashes, and
serialization happen outside the event interval.

The output is fail closed and write once.  It is never performance eligible or
production qualified, even when every numerical and provenance gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[2]
SCRIPT_PATH = Path(__file__).resolve()
SCHEMA = "gpu4pyscf.water8.wvvvv-abba-audit.v2"
QUALIFICATION_SCOPE = "audit-only"
LAYERS = ("wvvvv_term", "doubles_numerator_solve", "jacobi")
ABBA = ("two-ladder", "fused", "fused", "two-ladder")
EXPECTED_OUTPUTS = {
    "wvvvv_term": ({"wvvvv_core"}, set()),
    "doubles_numerator_solve": (
        {"doubles_numerator_core", "doubles_solved_core"},
        set(),
    ),
    "jacobi": (
        {
            "t1",
            "doubles_core",
            "error_t1",
            "error_core",
            "equation_residual_t1",
            "equation_residual_core",
        },
        {"energy", "residual_norm", "update_norm"},
    ),
}
KERNEL_METADATA = {
    "two-ladder": "rr-wvvvv-two-ladder",
    "fused": "rr-wvvvv-fused",
}
EXPECTED_STATE_ARRAYS = frozenset({
    "cd_factors",
    "projector_vectors",
    "projector_eigenvalues",
    "t1",
    "doubles_core",
    "fock",
    "orbital_energies",
    "singles_denominator",
    "projected_denominator",
})
DEFAULT_EXPECTED_HOST = "compute-1-6"
REQUIRED_CPU_SET = frozenset(range(24, 32))
REQUIRED_CPU_MODEL_TOKEN = "AMD EPYC 7513"
MIN_HBM_BYTES = 72 << 30


def _load_benchmark_module():
    name = "water8_wvvvv_abba_shared_benchmark"
    specification = importlib.util.spec_from_file_location(
        name, ROOT / "benchmark.py"
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the shared WATER27 benchmark module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


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


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write_once(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite output {path}") from exc


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return int(value)


def _finite_float(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("water2-tz", "water4-tz", "water8-tz"),
        default="water4-tz",
    )
    parser.add_argument("--orbital-artifact-in", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eri-tol", type=float, required=True)
    parser.add_argument("--rr-eig-cutoff", type=float, required=True)
    parser.add_argument("--rr-max-rank", type=int, default=None)
    parser.add_argument("--cd-max-rank", type=int, default=None)
    parser.add_argument("--direct-scf-tol", type=float, default=1.0e-13)
    parser.add_argument(
        "--gint-column-backend",
        choices=("selected", "restricted-reference"),
        default="selected",
    )
    parser.add_argument(
        "--gint-runtime-gate-receipt", type=Path, default=None
    )
    parser.add_argument("--gint-group-size", type=int, default=16)
    parser.add_argument("--gint-max-batch-size", type=int, default=32)
    parser.add_argument("--gint-max-block-bytes", type=int, default=None)
    parser.add_argument("--cd-mo-block-size", type=int, default=32)
    parser.add_argument("--denominator-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--denominator-max-rank", type=int, default=None)
    parser.add_argument("--rr-initial-rank", type=int, default=32)
    parser.add_argument("--rr-solver-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--rr-solver-maxiter", type=int, default=None)
    parser.add_argument("--rr-dense-fallback-dimension", type=int, default=64)
    parser.add_argument(
        "--rr-ritz-residual-tolerance", type=float, default=None
    )
    parser.add_argument("--rr-auxiliary-block-size", type=int, default=1)
    parser.add_argument("--rr-virtual-block-size", type=int, default=8)
    parser.add_argument(
        "--rr-ring-kernel",
        choices=("reference", "gemm"),
        default="reference",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--atol", type=float, default=5.0e-10)
    parser.add_argument("--rtol", type=float, default=5.0e-11)
    parser.add_argument("--scf-conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--max-memory-mb", type=int, default=110000)
    parser.add_argument("--threads", type=int, default=8)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    args.output = args.output.expanduser().resolve()
    args.orbital_artifact_in = args.orbital_artifact_in.expanduser().resolve()
    if args.output.exists() or os.path.lexists(args.output):
        raise FileExistsError(f"refusing to overwrite output {args.output}")
    if args.output.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("audit output must be outside the immutable source tree")
    if not args.orbital_artifact_in.is_file():
        raise ValueError(
            "canonical orbital artifact does not exist: "
            f"{args.orbital_artifact_in}"
        )
    args.eri_tol = _finite_float(
        "eri_tol", args.eri_tol, nonnegative=True
    )
    args.rr_eig_cutoff = _finite_float(
        "rr_eig_cutoff", args.rr_eig_cutoff, nonnegative=True
    )
    args.direct_scf_tol = _finite_float(
        "direct_scf_tol", args.direct_scf_tol, positive=True
    )
    args.denominator_tolerance = _finite_float(
        "denominator_tolerance",
        args.denominator_tolerance,
        nonnegative=True,
    )
    args.rr_solver_tolerance = _finite_float(
        "rr_solver_tolerance", args.rr_solver_tolerance, positive=True
    )
    if args.rr_ritz_residual_tolerance is not None:
        args.rr_ritz_residual_tolerance = _finite_float(
            "rr_ritz_residual_tolerance",
            args.rr_ritz_residual_tolerance,
            nonnegative=True,
        )
    args.atol = _finite_float("atol", args.atol, nonnegative=True)
    args.rtol = _finite_float("rtol", args.rtol, nonnegative=True)
    args.scf_conv_tol = _finite_float(
        "scf_conv_tol", args.scf_conv_tol, positive=True
    )
    for name in (
        "gint_group_size",
        "gint_max_batch_size",
        "cd_mo_block_size",
        "rr_initial_rank",
        "rr_auxiliary_block_size",
        "rr_virtual_block_size",
        "max_memory_mb",
        "threads",
    ):
        setattr(args, name, _positive_int(name, getattr(args, name)))
    if args.threads != 8:
        raise ValueError("the fixed Wvvvv audit requires exactly 8 CPU threads")
    args.warmup = _positive_int("warmup", args.warmup, allow_zero=True)
    args.rr_dense_fallback_dimension = _positive_int(
        "rr_dense_fallback_dimension",
        args.rr_dense_fallback_dimension,
        allow_zero=True,
    )
    for name in (
        "rr_max_rank",
        "cd_max_rank",
        "gint_max_block_bytes",
        "denominator_max_rank",
        "rr_solver_maxiter",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, _positive_int(name, value))
    if args.gint_runtime_gate_receipt is not None:
        args.gint_runtime_gate_receipt = (
            args.gint_runtime_gate_receipt.expanduser().resolve()
        )
        if args.gint_column_backend != "selected":
            raise ValueError(
                "a GINT runtime receipt applies only to the selected backend"
            )
        if not args.gint_runtime_gate_receipt.is_file():
            raise ValueError("GINT runtime gate receipt does not exist")


def _host_name() -> str:
    return platform.node().split(".", 1)[0]


def _gpu_identity(cp: Any) -> dict[str, Any]:
    count = int(cp.cuda.runtime.getDeviceCount())
    if count != 1:
        raise RuntimeError(
            "the Wvvvv audit requires exactly one CUDA-visible GPU; "
            f"observed {count}"
        )
    device_id = int(cp.cuda.Device().id)
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name", "")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    identity = {
        "visible_device_count": count,
        "device_id": device_id,
        "name": str(name),
        "compute_capability": (
            f"{int(properties.get('major', -1))}."
            f"{int(properties.get('minor', -1))}"
        ),
        "total_global_memory_bytes": int(
            properties.get("totalGlobalMem", 0)
        ),
        "pci_bus_id": str(cp.cuda.Device(device_id).pci_bus_id),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
    }
    if "A100-SXM4" not in identity["name"].upper():
        raise RuntimeError(
            "the Wvvvv audit is fixed to an NVIDIA A100-SXM4; observed "
            f"{identity['name']!r}"
        )
    if identity["compute_capability"] != "8.0":
        raise RuntimeError(
            "the Wvvvv audit requires A100 compute capability 8.0"
        )
    if identity["total_global_memory_bytes"] < MIN_HBM_BYTES:
        raise RuntimeError(
            "the Wvvvv audit requires at least 72 GiB of device memory; "
            f"observed {identity['total_global_memory_bytes']} bytes"
        )
    return identity


def _process_affinity() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("the Wvvvv audit requires sched_getaffinity")
    affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    if set(affinity) != set(REQUIRED_CPU_SET):
        raise RuntimeError(
            "the Wvvvv audit requires exact process affinity CPU 24-31; "
            f"observed {affinity}"
        )
    return affinity


def _cpu_model_identity(
    model_reader: Callable[[], Any] | None = None,
) -> str:
    """Validate the fixed EPYC model without trusting a caller-supplied CLI."""

    reader = BENCHMARK._cpu_model if model_reader is None else model_reader
    model = reader()
    if not isinstance(model, str) or REQUIRED_CPU_MODEL_TOKEN not in model:
        raise RuntimeError(
            "the Wvvvv audit requires AMD EPYC 7513 CPUs; "
            f"observed {model!r}"
        )
    return model


def _array_pointer(array: Any) -> int:
    data = getattr(array, "data", None)
    pointer = getattr(data, "ptr", None)
    if pointer is None:
        raise TypeError("audit state contains a non-CuPy array")
    return int(pointer)


def _array_identity(array: Any) -> dict[str, Any]:
    device = getattr(array, "device", None)
    return {
        "python_object_id": id(array),
        "device_pointer": _array_pointer(array),
        "device_id": int(getattr(device, "id", -1)),
        "shape": [int(value) for value in array.shape],
        "strides": [int(value) for value in array.strides],
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
    }


def _array_sha256(array: Any, cp: Any, *, chunk_bytes: int = 64 << 20) -> str:
    """Hash a CuPy array deterministically with bounded host staging."""

    identity = {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "order": "C",
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("ascii"))
    if array.ndim == 0:
        host = cp.asnumpy(array)
        digest.update(host.tobytes(order="C"))
        return digest.hexdigest()
    tail_elements = math.prod(int(value) for value in array.shape[1:]) or 1
    row_bytes = max(1, tail_elements * int(array.dtype.itemsize))
    rows = max(1, int(chunk_bytes) // row_bytes)
    for start in range(0, int(array.shape[0]), rows):
        stop = min(start + rows, int(array.shape[0]))
        host = cp.asnumpy(array[start:stop])
        digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def _state_arrays(engine: Any, t1: Any, doubles: Any) -> dict[str, Any]:
    return {
        "cd_factors": engine.integrals.factors,
        "projector_vectors": engine.projector.vectors,
        "projector_eigenvalues": engine.projector.eigenvalues,
        "t1": t1,
        "doubles_core": doubles.core,
        "fock": engine.fock,
        "orbital_energies": engine.orbital_energies,
        "singles_denominator": engine.eia,
        "projected_denominator": engine.denominator.matrix,
    }


def _state_manifest(
    engine: Any,
    t1: Any,
    doubles: Any,
    cp: Any,
) -> dict[str, Any]:
    arrays = _state_arrays(engine, t1, doubles)
    if set(arrays) != set(EXPECTED_STATE_ARRAYS):
        raise RuntimeError("the fixed Wvvvv state-array inventory changed")
    array_records = {
        name: {
            **_array_identity(array),
            "value_sha256": _array_sha256(array, cp),
        }
        for name, array in arrays.items()
    }
    pointers = _state_pointer_proof(engine, t1, doubles)
    proof = {"arrays": array_records, "pointers": pointers}
    # This digest intentionally excludes object ids and device pointers.  It
    # is the content witness for the resident inputs; the full identity digest
    # below additionally binds the exact resident objects and views.
    proof["content_sha256"] = _payload_sha256(
        {
            name: {
                key: record[key]
                for key in ("shape", "dtype", "nbytes", "value_sha256")
            }
            for name, record in array_records.items()
        }
    )
    proof["identity_sha256"] = _payload_sha256(proof)
    return proof


def _state_pointer_proof(
    engine: Any,
    t1: Any,
    doubles: Any,
) -> dict[str, Any]:
    arrays = {
        name: _array_identity(array)
        for name, array in _state_arrays(engine, t1, doubles).items()
    }
    views = {
        name: _array_identity(array)
        for name, array in {
            "L_oo": engine.L_oo,
            "L_ov": engine.L_ov,
            "L_vv": engine.L_vv,
            "fock_oo": engine.fock_oo,
            "fock_ov": engine.fock_ov,
            "fock_vv": engine.fock_vv,
            "occupied_energies": engine.occupied_energies,
            "virtual_energies": engine.virtual_energies,
        }.items()
    }
    objects = {
        "integral_provider": id(engine.integrals),
        "projector": id(engine.projector),
        "doubles": id(doubles),
        "denominator": id(engine.denominator),
        "engine": id(engine),
    }
    proof = {
        "arrays": arrays,
        "views": views,
        "objects": objects,
        # Selector state is deliberately part of the non-numerical lifecycle
        # witness.  A callback that leaks "fused" changes the resident state
        # even when every numerical output happens to compare equal.
        "non_numeric_state": {
            "wvvvv_kernel": getattr(engine, "wvvvv_kernel", None),
        },
    }
    proof["identity_sha256"] = _payload_sha256(proof)
    return proof


def _compare_arrays(
    reference: Any,
    candidate: Any,
    cp: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "passed": False,
            "reason": "shape-mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    if reference.dtype != candidate.dtype:
        return {
            "passed": False,
            "reason": "dtype-mismatch",
            "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
        }
    finite = bool(
        (cp.all(cp.isfinite(reference)) & cp.all(cp.isfinite(candidate))).item()
    )
    delta = cp.abs(candidate - reference)
    tolerance = atol + rtol * cp.abs(reference)
    violation_value = cp.count_nonzero(delta > tolerance)
    violations = int(
        violation_value.item()
        if hasattr(violation_value, "item") else violation_value
    )
    maximum_absolute = float(cp.max(delta).item()) if delta.size else 0.0
    reference_scale = (
        float(cp.max(cp.abs(reference)).item()) if reference.size else 0.0
    )
    delta_l2 = float(cp.linalg.norm(delta).item()) if delta.size else 0.0
    return {
        "passed": bool(finite and violations == 0),
        "finite": finite,
        "shape": [int(value) for value in reference.shape],
        "dtype": str(reference.dtype),
        "atol": float(atol),
        "rtol": float(rtol),
        "maximum_absolute_difference": maximum_absolute,
        "reference_maximum_absolute": reference_scale,
        "difference_l2_norm": delta_l2,
        "violation_count": violations,
    }


def _compare_scalars(
    reference: float,
    candidate: float,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    reference = float(reference)
    candidate = float(candidate)
    finite = math.isfinite(reference) and math.isfinite(candidate)
    delta = abs(candidate - reference) if finite else math.inf
    tolerance = atol + rtol * abs(reference) if finite else math.nan
    return {
        "passed": bool(finite and delta <= tolerance),
        "finite": finite,
        "reference": reference,
        "candidate": candidate,
        "absolute_difference": delta,
        "tolerance": tolerance,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _output_payload(layer: str, result: Any) -> dict[str, Any]:
    if layer == "wvvvv_term":
        return {
            "arrays": {"wvvvv_core": result.core},
            "scalars": {},
            "kernel_metadata": result.kernel_name,
        }
    if layer == "doubles_numerator_solve":
        equation, solved = result
        wvvvv = next(
            item
            for item in equation.term_metadata
            if item.get("term") == "complete-cc-Wvvvv-ladder"
        )
        return {
            "arrays": {
                "doubles_numerator_core": equation.core,
                "doubles_solved_core": solved,
            },
            "scalars": {},
            "kernel_metadata": wvvvv.get("kernel_name"),
            "selector_metadata": equation.wvvvv_kernel,
        }
    if layer == "jacobi":
        wvvvv = next(
            item
            for item in result.doubles_equation.term_metadata
            if item.get("term") == "complete-cc-Wvvvv-ladder"
        )
        return {
            "arrays": {
                "t1": result.t1,
                "doubles_core": result.doubles.core,
                "error_t1": result.error_t1,
                "error_core": result.error_core,
                "equation_residual_t1": result.equation_residual_t1,
                "equation_residual_core": result.equation_residual_core,
            },
            "scalars": {
                "energy": float(result.energy.item()),
                "residual_norm": float(result.residual_norm.item()),
                "update_norm": float(result.update_norm.item()),
            },
            "kernel_metadata": wvvvv.get("kernel_name"),
            "selector_metadata": result.doubles_equation.wvvvv_kernel,
        }
    raise ValueError(f"unsupported audit layer {layer!r}")


def _compare_payloads(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    cp: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if set(reference["arrays"]) != set(candidate["arrays"]):
        raise ValueError("audit output array names changed between A/B arms")
    if set(reference["scalars"]) != set(candidate["scalars"]):
        raise ValueError("audit output scalar names changed between A/B arms")
    arrays = {
        name: _compare_arrays(
            reference["arrays"][name],
            candidate["arrays"][name],
            cp,
            atol=atol,
            rtol=rtol,
        )
        for name in reference["arrays"]
    }
    scalars = {
        name: _compare_scalars(
            reference["scalars"][name],
            candidate["scalars"][name],
            atol=atol,
            rtol=rtol,
        )
        for name in reference["scalars"]
    }
    return {
        "passed": bool(
            all(item["passed"] for item in arrays.values())
            and all(item["passed"] for item in scalars.values())
        ),
        "arrays": arrays,
        "scalars": scalars,
    }


def _cuda_timed_call(
    callback: Callable[[], Any], cp: Any
) -> tuple[Any, float]:
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record(stream)
    result = callback()
    stop.record(stream)
    stop.synchronize()
    elapsed = float(cp.cuda.get_elapsed_time(start, stop)) * 1.0e-3
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("CUDA event returned a non-positive elapsed time")
    return result, elapsed


def _warm(callback: Callable[[], Any], cp: Any, count: int) -> None:
    for _ in range(count):
        callback()
        cp.cuda.get_current_stream().synchronize()


def _layer_callbacks(
    engine: Any,
    t1: Any,
    doubles: Any,
) -> dict[str, Callable[[str], Any]]:
    from gpu4pyscf.cc.rr_residual import (
        build_projected_ccsd_doubles_numerator,
        projected_cc_wvvvv,
        projected_cc_wvvvv_two_ladder,
    )

    def wvvvv(selector: str):
        callback = (
            projected_cc_wvvvv_two_ladder
            if selector == "two-ladder"
            else projected_cc_wvvvv
        )
        return callback(
            doubles,
            t1,
            engine.L_ov,
            engine.L_vv,
            auxiliary_block_size=engine.auxiliary_block_size,
        )

    def doubles_scope(selector: str):
        equation = build_projected_ccsd_doubles_numerator(
            doubles,
            t1,
            engine.fock_oo,
            engine.fock_ov,
            engine.fock_vv,
            engine.occupied_energies,
            engine.virtual_energies,
            engine.L_oo,
            engine.L_ov,
            engine.L_vv,
            level_shift=engine.level_shift,
            auxiliary_block_size=engine.auxiliary_block_size,
            virtual_block_size=engine.virtual_block_size,
            ring_kernel=engine.rr_ring_kernel,
            wvvvv_kernel=selector,
        )
        return equation, equation.solve(engine.denominator)

    def jacobi(selector: str):
        return _jacobi_with_wvvvv_selector(
            engine, t1, doubles, selector=selector
        )

    return {
        "wvvvv_term": wvvvv,
        "doubles_numerator_solve": doubles_scope,
        "jacobi": jacobi,
    }


def _jacobi_with_wvvvv_selector(
    engine: Any,
    t1: Any,
    doubles: Any,
    *,
    selector: str,
) -> Any:
    """Run one selected Jacobi arm and restore lifecycle configuration."""

    if selector not in KERNEL_METADATA:
        raise ValueError(f"unknown Wvvvv selector {selector!r}")
    original_selector = engine.wvvvv_kernel
    try:
        engine.wvvvv_kernel = selector
        return engine.jacobi(t1, doubles)
    finally:
        # A failed arm must not contaminate any later observation or the
        # failure record derived from this resident lifecycle.
        engine.wvvvv_kernel = original_selector


def _measure_layers(
    engine: Any,
    t1: Any,
    doubles: Any,
    cp: Any,
    *,
    warmup: int,
    atol: float,
    rtol: float,
    state_identity_sha256: str,
    state_pointer_sha256: str,
) -> list[dict[str, Any]]:
    callbacks = _layer_callbacks(engine, t1, doubles)
    rows: list[dict[str, Any]] = []
    sequence_index = 0
    for layer in LAYERS:
        callback = callbacks[layer]
        for selector in ("two-ladder", "fused"):
            _warm(
                lambda selector=selector, callback=callback: callback(selector),
                cp,
                warmup,
            )
        oracle: dict[str, Any] | None = None
        for layer_index, selector in enumerate(ABBA):
            result, elapsed = _cuda_timed_call(
                lambda selector=selector, callback=callback: callback(selector),
                cp,
            )
            # Content hashing is deliberately outside the CUDA event interval.
            # Each row therefore carries the actual post-callback witness for
            # all resident inputs, rather than repeating only the expected
            # value supplied by the caller.  The D2H staging done by
            # _array_sha256 is recorded explicitly in the output contract.
            observed_state_manifest = _state_manifest(engine, t1, doubles, cp)
            observed_state_sha256 = observed_state_manifest["identity_sha256"]
            observed_content_sha256 = observed_state_manifest["content_sha256"]
            observed_pointer_sha256 = observed_state_manifest["pointers"][
                "identity_sha256"
            ]
            if observed_state_sha256 != state_identity_sha256:
                raise RuntimeError(
                    "resident RR input content or non-numerical state changed"
                )
            if observed_pointer_sha256 != state_pointer_sha256:
                raise RuntimeError(
                    "resident RR object or device-pointer identity changed"
                )
            payload = _output_payload(layer, result)
            expected_kernel = KERNEL_METADATA[selector]
            selector_metadata = payload.get("selector_metadata", selector)
            if payload["kernel_metadata"] != expected_kernel:
                raise RuntimeError(
                    f"{layer} {selector} reported the wrong Wvvvv kernel"
                )
            if selector_metadata != selector:
                raise RuntimeError(
                    f"{layer} {selector} reported the wrong selector"
                )
            if oracle is None:
                if selector != "two-ladder":
                    raise RuntimeError("the ABBA oracle must be two-ladder")
                oracle = payload
            comparison = _compare_payloads(
                oracle,
                payload,
                cp,
                atol=atol,
                rtol=rtol,
            )
            row = {
                "sequence_index": sequence_index,
                "layer_sequence_index": layer_index,
                "layer": layer,
                "selector": selector,
                "status": "passed" if comparison["passed"] else "failed",
                "error": None,
                "cuda_event_seconds": elapsed,
                "timing_domain": "cuda-event-device-elapsed",
                "synchronization": "before-start-and-stop-event",
                "rank": int(doubles.rank),
                "state_identity_sha256": observed_state_sha256,
                "state_content_sha256": observed_content_sha256,
                "state_pointer_sha256": observed_pointer_sha256,
                "state_manifest": observed_state_manifest,
                "state_observation": {
                    "computed_outside_cuda_event": True,
                    "host_staging": "bounded-D2H-for-content-hash",
                    "included_arrays": sorted(observed_state_manifest["arrays"]),
                },
                "kernel_metadata": payload["kernel_metadata"],
                "selector_metadata": selector_metadata,
                "scientific_scalars": dict(payload["scalars"]),
                "comparison_to_first_two_ladder": comparison,
            }
            rows.append(row)
            sequence_index += 1
            if not comparison["passed"]:
                raise RuntimeError(
                    f"numerical mismatch in {layer} {selector} A/B row"
                )
    return rows


def _required_row_errors(
    rows: Any,
    *,
    expected_rank: int,
    expected_state_identity: str,
    expected_state_pointer: str,
    expected_state_content: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(rows, list):
        return ["measurements must be a list"]
    if len(rows) != len(LAYERS) * len(ABBA):
        errors.append("measurement row count does not equal three ABBA blocks")
        return errors
    for layer_offset, layer in enumerate(LAYERS):
        block = rows[layer_offset * len(ABBA):(layer_offset + 1) * len(ABBA)]
        if [row.get("layer") for row in block] != [layer] * len(ABBA):
            errors.append(f"{layer} rows are missing or out of order")
        if [row.get("selector") for row in block] != list(ABBA):
            errors.append(f"{layer} selector order is not exact ABBA")
        counts = {
            selector: sum(row.get("selector") == selector for row in block)
            for selector in KERNEL_METADATA
        }
        if counts != {"two-ladder": 2, "fused": 2}:
            errors.append(f"{layer} does not contain two rows per selector")
        for row in block:
            required = {
                "sequence_index",
                "layer_sequence_index",
                "layer",
                "selector",
                "status",
                "error",
                "cuda_event_seconds",
                "rank",
                "state_identity_sha256",
                "state_content_sha256",
                "state_pointer_sha256",
                "state_manifest",
                "state_observation",
                "kernel_metadata",
                "selector_metadata",
                "scientific_scalars",
                "comparison_to_first_two_ladder",
            }
            missing = sorted(required - set(row))
            if missing:
                errors.append(f"measurement row is missing fields {missing}")
                continue
            if row["status"] != "passed" or row["error"] is not None:
                errors.append(f"{layer} contains a failed or error row")
            layer_sequence_index = row.get("layer_sequence_index")
            if type(layer_sequence_index) is not int or (
                layer_sequence_index not in range(len(ABBA))
            ):
                errors.append(f"{layer} layer sequence index is invalid")
            else:
                expected_sequence_index = (
                    layer_offset * len(ABBA) + layer_sequence_index
                )
                if row.get("sequence_index") != expected_sequence_index:
                    errors.append(f"{layer} sequence index is inconsistent")
            elapsed = row["cuda_event_seconds"]
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) <= 0.0
            ):
                errors.append(f"{layer} contains invalid CUDA timing")
            if row["rank"] != expected_rank:
                errors.append(f"{layer} rank changed within the lifecycle")
            if row["state_identity_sha256"] != expected_state_identity:
                errors.append(f"{layer} did not use the fixed resident state")
            if (
                expected_state_content is not None
                and row["state_content_sha256"] != expected_state_content
            ):
                errors.append(f"{layer} resident input content changed")
            if row["state_pointer_sha256"] != expected_state_pointer:
                errors.append(
                    f"{layer} resident object or device pointer changed"
                )
            manifest = row["state_manifest"]
            manifest_arrays = (
                manifest.get("arrays") if isinstance(manifest, dict) else None
            )
            manifest_pointers = (
                manifest.get("pointers")
                if isinstance(manifest, dict)
                else None
            )
            if (
                not isinstance(manifest, dict)
                or manifest.get("identity_sha256")
                != row["state_identity_sha256"]
                or manifest.get("content_sha256")
                != row["state_content_sha256"]
                or not isinstance(manifest_arrays, dict)
                or set(manifest_arrays) != set(EXPECTED_STATE_ARRAYS)
                or not isinstance(manifest_pointers, dict)
                or manifest_pointers.get("identity_sha256")
                != row["state_pointer_sha256"]
            ):
                errors.append(f"{layer} state manifest is incomplete")
            elif any(
                not isinstance(record, dict)
                or not _is_sha256(record.get("value_sha256"))
                or not isinstance(record.get("shape"), list)
                or not isinstance(record.get("dtype"), str)
                or type(record.get("nbytes")) is not int
                or record["nbytes"] < 0
                for record in manifest_arrays.values()
            ):
                errors.append(f"{layer} state array witness is malformed")
            observation = row["state_observation"]
            if (
                not isinstance(observation, dict)
                or observation.get("computed_outside_cuda_event") is not True
                or observation.get("host_staging")
                != "bounded-D2H-for-content-hash"
                or observation.get("included_arrays")
                != sorted(EXPECTED_STATE_ARRAYS)
            ):
                errors.append(f"{layer} state hash timing boundary is missing")
            selector = row.get("selector")
            if row["kernel_metadata"] != KERNEL_METADATA.get(selector):
                errors.append(f"{layer} kernel metadata disagrees with selector")
            if row["selector_metadata"] != selector:
                errors.append(f"{layer} selector metadata is inconsistent")
            comparison = row["comparison_to_first_two_ladder"]
            if not isinstance(comparison, dict) or comparison.get("passed") is not True:
                errors.append(f"{layer} numerical comparison did not pass")
            else:
                array_checks = comparison.get("arrays")
                scalar_checks = comparison.get("scalars")
                expected_arrays, expected_scalars = EXPECTED_OUTPUTS[layer]
                if (
                    not isinstance(array_checks, dict)
                    or set(array_checks) != expected_arrays
                    or any(
                        not isinstance(item, dict)
                        or item.get("passed") is not True
                        or item.get("finite") is not True
                        for item in array_checks.values()
                    )
                ):
                    errors.append(f"{layer} array comparisons are incomplete")
                if (
                    not isinstance(scalar_checks, dict)
                    or set(scalar_checks) != expected_scalars
                    or any(
                        not isinstance(item, dict)
                        or item.get("passed") is not True
                        or item.get("finite") is not True
                        for item in scalar_checks.values()
                    )
                ):
                    errors.append(f"{layer} scalar comparisons are incomplete")
            scalars = row["scientific_scalars"]
            if not isinstance(scalars, dict) or set(scalars) != EXPECTED_OUTPUTS[layer][1]:
                errors.append(f"{layer} scientific scalars are malformed")
            elif any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in scalars.values()
            ):
                errors.append(f"{layer} scientific scalars are not finite")
            elif isinstance(comparison, dict) and isinstance(
                comparison.get("scalars"), dict
            ):
                for name, value in scalars.items():
                    check = comparison["scalars"].get(name)
                    if (
                        not isinstance(check, dict)
                        or check.get("candidate") != value
                    ):
                        errors.append(
                            f"{layer} {name} scalar evidence is inconsistent"
                        )
    return errors


def _summarize_measurements(
    rows: list[dict[str, Any]],
    *,
    expected_rank: int,
    expected_state_identity: str,
    expected_state_pointer: str,
    expected_state_content: str | None = None,
) -> dict[str, Any]:
    errors = _required_row_errors(
        rows,
        expected_rank=expected_rank,
        expected_state_identity=expected_state_identity,
        expected_state_pointer=expected_state_pointer,
        expected_state_content=expected_state_content,
    )
    if errors:
        raise ValueError("; ".join(errors))
    layers: dict[str, Any] = {}
    for layer in LAYERS:
        layer_rows = [row for row in rows if row["layer"] == layer]
        timings = {
            selector: [
                float(row["cuda_event_seconds"])
                for row in layer_rows
                if row["selector"] == selector
            ]
            for selector in KERNEL_METADATA
        }
        medians = {
            selector: float(statistics.median(values))
            for selector, values in timings.items()
        }
        layers[layer] = {
            "samples_seconds": timings,
            "median_seconds": medians,
            "observed_two_ladder_over_fused_ratio": (
                medians["two-ladder"] / medians["fused"]
            ),
            "sample_count_per_selector": 2,
        }
    return {
        "status": "passed",
        "qualification_scope": QUALIFICATION_SCOPE,
        "performance_eligible": False,
        "production_qualified": False,
        "state_kind": "initial-mp2-amplitudes",
        "ccsd_converged": False,
        "diis_iterations": 0,
        "sample_interpretation": (
            "three initial-state microbenchmark layers; exactly two samples "
            "per selector; not a statistically stable estimate"
        ),
        "timing_interpretation": (
            "same-process CUDA-event observation from initial MP2 amplitudes; "
            "two samples per selector are not a statistically stable estimate; "
            "no production speedup claim"
        ),
        "layers": layers,
    }


def _source_stability(
    source_start: dict[str, Any],
    source_end: dict[str, Any],
    script_sha256_start: str,
) -> dict[str, Any]:
    script_sha256_end = _file_sha256(SCRIPT_PATH)
    stable = bool(
        source_start.get("tree_sha256") == source_end.get("tree_sha256")
        and source_start.get("files_hashed") == source_end.get("files_hashed")
        and source_start.get("revision") == source_end.get("revision")
        and script_sha256_start == script_sha256_end
    )
    return {
        "passed": stable,
        "tree_sha256_start": source_start.get("tree_sha256"),
        "tree_sha256_end": source_end.get("tree_sha256"),
        "files_hashed_start": source_start.get("files_hashed"),
        "files_hashed_end": source_end.get("files_hashed"),
        "revision_start": source_start.get("revision"),
        "revision_end": source_end.get("revision"),
        "script_sha256_start": script_sha256_start,
        "script_sha256_end": script_sha256_end,
    }


def _args_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import cupy as cp
    import numpy as np
    from pyscf import gto, lib

    from gpu4pyscf.cc.rr_engine import RRCCSDIterationEngine
    from gpu4pyscf.cc.rrccsd import RRCCSD
    from gpu4pyscf.scf import hf as gpu_hf

    observed_host = _host_name()
    if observed_host != DEFAULT_EXPECTED_HOST:
        raise RuntimeError(
            f"host gate requires {DEFAULT_EXPECTED_HOST!r}; observed {observed_host!r}"
        )
    gpu = _gpu_identity(cp)
    process_affinity = _process_affinity()
    cpu_model = _cpu_model_identity()
    lib.num_threads(args.threads)
    source_start = BENCHMARK._git_state()
    script_sha256_start = _file_sha256(SCRIPT_PATH)
    case = BENCHMARK.load_cases()[args.case]
    expected_dimensions = case["expected_dimensions"]
    molecule = gto.M(
        atom=[
            (item[0], tuple(item[1:]))
            for item in case["geometry_angstrom"]
        ],
        unit="Angstrom",
        basis=case["basis"],
        charge=0,
        spin=0,
        cart=False,
        verbose=4,
        max_memory=args.max_memory_mb,
    )
    mean_field = gpu_hf.RHF(molecule)
    mean_field.conv_tol = args.scf_conv_tol
    mean_field.max_cycle = 100
    mean_field.kernel()
    cp.cuda.get_current_stream().synchronize()
    if mean_field.converged is not True:
        raise RuntimeError("the single fresh RHF calculation did not converge")
    orbital_args = argparse.Namespace(
        case=args.case,
        orbital_artifact_in=args.orbital_artifact_in,
        orbital_artifact_out=None,
        scf_conv_tol=args.scf_conv_tol,
    )
    orbital_context = BENCHMARK._prepare_canonical_orbitals(
        orbital_args,
        mean_field,
        molecule,
        case,
        source_start,
        np,
        cp,
    )
    orbital_record = BENCHMARK._orbital_record(orbital_context)

    runtime_options: dict[str, Any] = {}
    if args.gint_column_backend == "selected":
        runtime_args = argparse.Namespace(
            gint_column_backend=args.gint_column_backend,
            gint_runtime_gate_receipt=args.gint_runtime_gate_receipt,
        )
        runtime_options = BENCHMARK._recorded_gint_runtime_options(
            runtime_args,
            source_start,
            execution_mode="consumer-benchmark",
        )
    solver = RRCCSD(
        mean_field,
        eri_backend="cd",
        eri_tol=args.eri_tol,
        direct_scf_tol=args.direct_scf_tol,
        cd_max_rank=args.cd_max_rank,
        gint_column_backend=args.gint_column_backend,
        gint_column_kernel="reference",
        gint_group_size=args.gint_group_size,
        gint_max_block_bytes=args.gint_max_block_bytes,
        gint_max_batch_size=args.gint_max_batch_size,
        cd_mo_block_size=args.cd_mo_block_size,
        rr_eig_cutoff=args.rr_eig_cutoff,
        rr_max_rank=args.rr_max_rank,
        denominator_tolerance=args.denominator_tolerance,
        denominator_max_rank=args.denominator_max_rank,
        rr_initial_rank=args.rr_initial_rank,
        rr_solver_tolerance=args.rr_solver_tolerance,
        rr_solver_maxiter=args.rr_solver_maxiter,
        rr_dense_fallback_dimension=args.rr_dense_fallback_dimension,
        rr_ritz_residual_tolerance=args.rr_ritz_residual_tolerance,
        rr_auxiliary_block_size=args.rr_auxiliary_block_size,
        rr_virtual_block_size=args.rr_virtual_block_size,
        rr_ring_kernel=args.rr_ring_kernel,
        precision="fp64",
        **runtime_options,
    )
    solver.frozen = 0
    solver.max_memory = args.max_memory_mb
    integrals = solver.ao2mo(solver.mo_coeff)
    emp2, t1, doubles = solver.init_amps(integrals)
    cp.cuda.get_current_stream().synchronize()
    engine = RRCCSDIterationEngine(
        integrals,
        solver.rr_projector,
        solver._rr_fock,
        solver._rr_orbital_energies,
        level_shift=float(solver.level_shift),
        auxiliary_block_size=min(
            args.rr_auxiliary_block_size, integrals.naux
        ),
        virtual_block_size=min(args.rr_virtual_block_size, integrals.nvir),
        rr_ring_kernel=args.rr_ring_kernel,
        wvvvv_kernel="two-ladder",
    )
    dimensions = {
        "nao": int(molecule.nao_nr()),
        "nocc": int(integrals.nocc),
        "nvir": int(integrals.nvir),
        "naux": int(integrals.naux),
        "pair_dimension": int(doubles.projector.full_dimension),
        "rr_rank": int(doubles.rank),
    }
    for name in ("nao", "nocc", "nvir"):
        if dimensions[name] != int(expected_dimensions[name]):
            raise RuntimeError(
                f"{name} differs from the fixed case contract: "
                f"{dimensions[name]} != {expected_dimensions[name]}"
            )
    if not math.isfinite(float(emp2)):
        raise RuntimeError("initial compressed MP2 energy is not finite")

    state_before = _state_manifest(engine, t1, doubles, cp)
    rows = _measure_layers(
        engine,
        t1,
        doubles,
        cp,
        warmup=args.warmup,
        atol=args.atol,
        rtol=args.rtol,
        state_identity_sha256=state_before["identity_sha256"],
        state_pointer_sha256=state_before["pointers"]["identity_sha256"],
    )
    cp.cuda.get_current_stream().synchronize()
    state_after = _state_manifest(engine, t1, doubles, cp)
    state_unchanged = bool(state_before == state_after)
    if not state_unchanged:
        raise RuntimeError("resident RR input state changed during the ABBA audit")
    source_end = BENCHMARK._git_state()
    source_stability = _source_stability(
        source_start, source_end, script_sha256_start
    )
    if not source_stability["passed"]:
        raise RuntimeError("source or audit script changed during execution")
    summary = _summarize_measurements(
        rows,
        expected_rank=doubles.rank,
        expected_state_identity=state_before["identity_sha256"],
        expected_state_pointer=state_before["pointers"]["identity_sha256"],
        expected_state_content=state_before["content_sha256"],
    )
    return {
        "schema": SCHEMA,
        "status": "passed",
        "qualification_scope": QUALIFICATION_SCOPE,
        "performance_eligible": False,
        "production_qualified": False,
        "state_kind": "initial-mp2-amplitudes",
        "ccsd_converged": False,
        "diis_iterations": 0,
        "measurement_interpretation": (
            "The three layers are microbenchmarks from the initial MP2 "
            "amplitudes. Each selector has exactly two samples; these are "
            "not a statistically stable performance estimate."
        ),
        "errors": [],
        "contract": {
            "case": args.case,
            "host": DEFAULT_EXPECTED_HOST,
            "gpu": "NVIDIA A100-SXM4-80GB compute capability 8.0",
            "precision": "fp64",
            "lifecycle_build_count": 1,
            "fresh_scf_count": 1,
            "measured_sequence_per_layer": list(ABBA),
            "layers": list(LAYERS),
            "timing": "synchronized CUDA events",
            "comparisons_outside_timing": True,
            "state_hashing_outside_timing": True,
            "state_hash_host_staging": "bounded-D2H-for-content-hash",
            "normal_production_path": False,
        },
        "arguments": _args_record(args),
        "source_start": source_start,
        "source_end": source_end,
        "source_stability": source_stability,
        "script": {
            "path": str(SCRIPT_PATH),
            "sha256": script_sha256_start,
        },
        "case_file": {
            "path": str(BENCHMARK.CASE_FILE.resolve()),
            "sha256": _file_sha256(BENCHMARK.CASE_FILE),
        },
        "orbital_artifact": orbital_record,
        "hardware": {
            "host": observed_host,
            "gpu": gpu,
            "cpu_model": cpu_model,
            "required_cpu_model_token": REQUIRED_CPU_MODEL_TOKEN,
            "platform": platform.platform(),
            "affinity": process_affinity,
            "required_affinity": sorted(REQUIRED_CPU_SET),
            "affinity_gate": "exactly-cpu-24-31",
            "numa": {
                "status": "not-proven-by-this-audit",
                "recorded": False,
            },
            "pcie": {
                "status": "not-proven-by-this-audit",
                "recorded_gpu_bus_id": gpu["pci_bus_id"],
            },
            "topology_proven": False,
            "threads": args.threads,
            "slurm": {
                name: os.getenv(name)
                for name in (
                    "SLURM_JOB_ID",
                    "SLURM_JOB_NAME",
                    "SLURM_JOB_NODELIST",
                    "SLURM_CPUS_PER_TASK",
                )
            },
        },
        "lifecycle": {
            "dimensions": dimensions,
            "initial_compressed_mp2_energy": float(emp2),
            "integrals": integrals.metadata(),
            "projector": solver._rr_projector_build_metadata,
            "engine": engine.metadata(),
            "same_python_objects_and_device_pointers": True,
            "state_unchanged": state_unchanged,
            "state_before": state_before,
            "state_after": state_after,
        },
        "warmup_calls_per_selector_per_layer": args.warmup,
        "measurements": rows,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
    except Exception as exc:  # noqa: BLE001 - CLI validation fails closed
        parser.error(str(exc))
    try:
        record = _run(args)
    except Exception as exc:  # noqa: BLE001 - write an audit failure record
        record = {
            "schema": SCHEMA,
            "status": "error",
            "qualification_scope": QUALIFICATION_SCOPE,
            "performance_eligible": False,
            "production_qualified": False,
            "state_kind": "initial-mp2-amplitudes",
            "ccsd_converged": False,
            "diis_iterations": 0,
            "arguments": _args_record(args),
            "errors": [f"{type(exc).__name__}: {exc}"],
            "traceback": traceback.format_exc(),
            "summary": None,
        }
        try:
            _json_write_once(args.output, record)
        except Exception as write_exc:  # noqa: BLE001 - preserve original failure
            print(f"failed to write audit error record: {write_exc}", file=sys.stderr)
        print(record["errors"][0], file=sys.stderr)
        return 1
    _json_write_once(args.output, record)
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
