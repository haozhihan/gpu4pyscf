#!/usr/bin/env python3
"""Run one auditable GPU RR-CCSD Jacobi equation on a real water system.

This is an integration smoke gate, not a performance result.  The factors are
from an explicit auxiliary-basis DF build (``cc-pvtz-ri`` by default), and the
ring contraction is selected explicitly as the reference or GEMM validation
kernel.  Neither selector has passed the complete direct-CD/A100 performance
gate, so the driver records ``performance_eligible=false`` in every result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
CASE_FILE = ROOT / "cases.json"
CASES = ("water1-sto3g", "water2-tz")
FROZEN_BASE_COMMIT = "a89b3ae018d4e82968323ef95645d91adde294aa"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="result JSON (default: results/rr-smoke-CASE-JOB.json)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--auxbasis", default="cc-pvtz-ri")
    parser.add_argument("--max-memory-mb", type=int, default=110000)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-10)
    parser.add_argument("--df-block-size", type=int, default=32)
    parser.add_argument("--auxiliary-block-size", type=int, default=8)
    parser.add_argument("--virtual-block-size", type=int, default=8)
    parser.add_argument(
        "--rr-ring-kernel",
        choices=("reference", "gemm"),
        default="reference",
    )
    parser.add_argument("--denominator-tolerance", type=float, default=1e-10)
    parser.add_argument("--denominator-max-rank", type=int, default=64)
    parser.add_argument("--rr-eig-cutoff", type=float, default=1e-6)
    parser.add_argument("--rr-max-rank", type=int, default=32)
    parser.add_argument("--lanczos-tolerance", type=float, default=1e-8)
    parser.add_argument("--ritz-residual-tolerance", type=float, default=1e-6)
    parser.add_argument("--oracle-tensor-atol", type=float, default=1e-10)
    parser.add_argument("--oracle-energy-atol", type=float, default=1e-10)
    parser.add_argument(
        "--water1-convergence",
        action="store_true",
        help=(
            "also converge the full-rank water1 RR equations and compare "
            "with an independent dense PySCF RCCSD solve"
        ),
    )
    parser.add_argument("--convergence-max-cycle", type=int, default=50)
    parser.add_argument("--convergence-energy-tol", type=float, default=1e-10)
    parser.add_argument("--convergence-residual-tol", type=float, default=1e-8)
    parser.add_argument(
        "--convergence-oracle-energy-atol", type=float, default=1e-8
    )
    parser.add_argument(
        "--convergence-oracle-amplitude-atol", type=float, default=1e-7
    )
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _command(arguments: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            arguments,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except Exception:
        return None


def _version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _load_case(case_id: str) -> dict[str, Any]:
    payload = json.loads(CASE_FILE.read_text())
    frozen = {case["id"]: case for case in payload["cases"]}
    water2 = frozen["water2-tz"]
    if case_id == "water2-tz":
        return dict(water2)
    geometry = water2["geometry_angstrom"][:3]
    return {
        "id": "water1-sto3g",
        "role": "minimal-real-molecule-smoke",
        "source": "WATER27:H2O2:first-three-atoms",
        "basis": "sto-3g",
        "expected_dimensions": {"nao": 7, "nocc": 5, "nvir": 2},
        "geometry_angstrom": geometry,
        "derived_from": "water2-tz frozen geometry without coordinate edits",
    }


def _geometry_hash(case: dict[str, Any]) -> str:
    blob = json.dumps(
        case["geometry_angstrom"], separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _gpu_hardware() -> dict[str, Any]:
    fields = "name,uuid,memory.total,driver_version,pci.bus_id"
    text = _command(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not text:
        return {"available": False}
    rows = []
    keys = (
        "name",
        "uuid",
        "memory_total_mib",
        "driver_version",
        "pci_bus_id",
    )
    for line in text.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == len(keys):
            rows.append(dict(zip(keys, values)))
    topology = _command(["nvidia-smi", "topo", "-m"])
    return {
        "available": bool(rows),
        "visible_gpus": rows,
        "topology": {
            "available": topology is not None,
            "nvidia_smi_topo_m": topology,
        },
    }


def _host_rss_mib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


class _HBMMonitor:
    """Poll process-specific HBM via nvidia-smi during the smoke run."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self.samples_mib: list[int] = []
        self.error: str | None = None

    def _run(self) -> None:
        pid = str(os.getpid())
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=4,
                )
                for line in output.splitlines():
                    fields = [item.strip() for item in line.split(",")]
                    if len(fields) == 2 and fields[0] == pid:
                        self.samples_mib.append(int(fields[1]))
            except Exception as exc:
                self.error = repr(exc)
            self._stop.wait(0.25)

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def close(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._thread.join(timeout=5)

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "nvidia-smi process polling",
            "sample_period_s": 0.25,
            "sample_count": len(self.samples_mib),
            "peak_mib": max(self.samples_mib, default=None),
            "last_poll_error": self.error,
        }


class _PhaseTimer:
    def __init__(self, cp: Any) -> None:
        self.cp = cp
        self.timings: dict[str, float] = {}
        self.memory_samples: list[dict[str, Any]] = []

    def _sync(self) -> None:
        self.cp.cuda.get_current_stream().synchronize()

    def sample_memory(self, label: str) -> None:
        free_bytes, total_bytes = self.cp.cuda.runtime.memGetInfo()
        pool = self.cp.get_default_memory_pool()
        self.memory_samples.append(
            {
                "label": label,
                "device_used_bytes": int(total_bytes - free_bytes),
                "device_total_bytes": int(total_bytes),
                "cupy_pool_used_bytes": int(pool.used_bytes()),
                "cupy_pool_total_bytes": int(pool.total_bytes()),
            }
        )

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.timings[name] = time.perf_counter() - started
            self.sample_memory(name)

    def memory_metadata(self) -> dict[str, Any]:
        peak = max(
            (row["device_used_bytes"] for row in self.memory_samples),
            default=None,
        )
        return {
            "source": "CUDA memGetInfo sampled at synchronized phase boundaries",
            "sample_count": len(self.memory_samples),
            "sampled_peak_device_used_bytes": peak,
            "samples": self.memory_samples,
        }


def _is_cupy_array(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "cupy"


def _device_id(value: Any) -> int | None:
    if not _is_cupy_array(value):
        return None
    return int(value.device.id)


def _to_device(value: Any, cp: Any, transfer_counter: Any) -> Any:
    if _is_cupy_array(value):
        return value
    nbytes = int(getattr(value, "nbytes", 0))
    result = cp.asarray(value)
    transfer_counter.record_h2d(nbytes)
    return result


def _read_device_scalar(value: Any, transfer_counter: Any) -> float:
    if _is_cupy_array(value):
        transfer_counter.record_d2h(int(value.nbytes))
    return float(value.item())


def _copy_to_host(value: Any, np: Any, transfer_counter: Any) -> Any:
    if _is_cupy_array(value):
        result = value.get()
        transfer_counter.record_d2h(int(value.nbytes))
        return result
    return np.asarray(value)


def _max_abs_error(left: Any, right: Any, np: Any) -> float:
    if left.shape != right.shape:
        raise ValueError(f"oracle comparison shape mismatch: {left.shape}, {right.shape}")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _water1_dense_pyscf_oracle(
    *,
    mol: Any,
    integrals: Any,
    fock: Any,
    orbital_energies: Any,
    result: Any,
    nocc: int,
    nvir: int,
    auxiliary_block_size: int,
    tensor_atol: float,
    energy_atol: float,
    transfer_counter: Any,
) -> dict[str, Any]:
    """Compare one full-rank RR update with dense PySCF RCCSD equations."""

    import numpy as np
    from pyscf.cc import rccsd
    from gpu4pyscf.cc.lowrank import t2_from_pair_matrix
    from gpu4pyscf.cc.rr_residual import rr_ccsd_energy

    if result.doubles.rank != nocc * nvir:
        raise ValueError("the dense oracle requires a full-rank RR projector")
    if tensor_atol <= 0 or energy_atol <= 0:
        raise ValueError("oracle tolerances must be positive")

    candidate_energy_device = rr_ccsd_energy(
        result.doubles,
        result.t1,
        fock[:nocc, nocc:],
        integrals.L_ov,
        auxiliary_block_size=auxiliary_block_size,
    ).value
    candidate_t2_device = result.doubles.reconstruct_t2()
    candidate_numerator_t2_device = t2_from_pair_matrix(
        result.doubles_equation.core, nocc, nvir
    )

    factors = _copy_to_host(integrals.factors, np, transfer_counter)
    fock_host = _copy_to_host(fock, np, transfer_counter)
    energy_host = _copy_to_host(
        orbital_energies, np, transfer_counter
    )
    candidate_t1 = _copy_to_host(result.t1, np, transfer_counter)
    candidate_t2 = _copy_to_host(
        candidate_t2_device, np, transfer_counter
    )
    candidate_numerator_t1 = _copy_to_host(
        result.singles_equation.numerator, np, transfer_counter
    )
    candidate_numerator_t2 = _copy_to_host(
        candidate_numerator_t2_device, np, transfer_counter
    )
    candidate_energy = _read_device_scalar(
        candidate_energy_device, transfer_counter
    )

    occupied = slice(0, nocc)
    virtual = slice(nocc, nocc + nvir)
    loo = factors[:, occupied, occupied]
    lov = factors[:, occupied, virtual]
    lvo = factors[:, virtual, occupied]
    lvv = factors[:, virtual, virtual]

    def eri_block(left: Any, right: Any) -> Any:
        return np.einsum(
            "Qpq,Qrs->pqrs", left, right.conj(), optimize=True
        )

    dense_blocks = {
        "oooo": eri_block(loo, loo),
        "ovoo": eri_block(lov, loo),
        "oovv": eri_block(loo, lvv),
        "ovvo": eri_block(lov, lvo),
        "ovov": eri_block(lov, lov),
        "ovvv": eri_block(lov, lvv),
        "vvvv": eri_block(lvv, lvv),
    }
    eris = rccsd._ChemistsERIs(mol)
    eris.nocc = nocc
    eris.fock = fock_host
    eris.mo_energy = energy_host
    eris.mo_coeff = None
    for name, block in dense_blocks.items():
        setattr(eris, name, block)

    oracle_driver = SimpleNamespace(level_shift=0.0, cc2=False)
    input_t1 = np.zeros((nocc, nvir), dtype=factors.dtype)
    input_t2 = np.zeros((nocc, nocc, nvir, nvir), dtype=factors.dtype)
    oracle_t1, oracle_t2 = rccsd.update_amps(
        oracle_driver, input_t1, input_t2, eris
    )
    eia = energy_host[:nocc, None] - energy_host[None, nocc:]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    oracle_numerator_t1 = oracle_t1 * eia
    oracle_numerator_t2 = oracle_t2 * eijab
    oracle_energy = float(
        rccsd.energy(oracle_driver, oracle_t1, oracle_t2, eris)
    )

    errors = {
        "t1_max_abs": _max_abs_error(candidate_t1, oracle_t1, np),
        "t2_max_abs": _max_abs_error(candidate_t2, oracle_t2, np),
        "singles_numerator_max_abs": _max_abs_error(
            candidate_numerator_t1, oracle_numerator_t1, np
        ),
        "doubles_numerator_max_abs": _max_abs_error(
            candidate_numerator_t2, oracle_numerator_t2, np
        ),
        "post_update_correlation_energy_abs": abs(
            candidate_energy - oracle_energy
        ),
    }
    tensor_keys = tuple(key for key in errors if not key.endswith("energy_abs"))
    passed = all(errors[key] <= tensor_atol for key in tensor_keys)
    passed = passed and errors["post_update_correlation_energy_abs"] <= energy_atol
    dense_block_bytes = {
        name: int(block.nbytes) for name, block in dense_blocks.items()
    }
    payload = {
        "executed": True,
        "case_scope": "water1-sto3g only",
        "passed": passed,
        "driver": "pyscf.cc.rccsd.update_amps",
        "integral_source": "same resident DF MO factors as RR candidate",
        "initial_amplitudes": "t1=0,t2=0",
        "tolerances": {
            "tensor_max_abs": float(tensor_atol),
            "correlation_energy_abs": float(energy_atol),
        },
        "errors": errors,
        "energies": {
            "rr_candidate_post_update_correlation": candidate_energy,
            "dense_pyscf_post_update_correlation": oracle_energy,
        },
        "dense_materialization": {
            "scope": "independent water1 validation oracle after RR equation",
            "oracle_mo_eri_blocks_materialized": True,
            "oracle_dense_t1_materialized": True,
            "oracle_dense_t2_materialized": True,
            "candidate_dense_t2_materialized_for_comparison": True,
            "candidate_dense_doubles_numerator_materialized_for_comparison": True,
            "production_rr_equation_materialized_dense_t2": False,
            "production_rr_equation_materialized_four_index_eri": False,
            "mo_eri_block_shapes": {
                name: list(block.shape) for name, block in dense_blocks.items()
            },
            "mo_eri_block_nbytes": dense_block_bytes,
            "mo_eri_blocks_total_nbytes": sum(dense_block_bytes.values()),
        },
        "transfers": transfer_counter.to_dict(),
    }
    return payload


def _water1_dense_pyscf_convergence_oracle(
    *,
    mol: Any,
    mf: Any,
    integrals: Any,
    fock: Any,
    orbital_energies: Any,
    engine: Any,
    projector: Any,
    nocc: int,
    nvir: int,
    max_cycle: int,
    energy_tol: float,
    residual_tol: float,
    oracle_energy_atol: float,
    oracle_amplitude_atol: float,
    transfer_counter: Any,
) -> dict[str, Any]:
    """Converge the full-rank RR endpoint and a dense PySCF oracle.

    Dense four-index blocks and dense doubles are created only inside this
    water-monomer validation oracle.  The RR iteration retains its compressed
    resident representation and consumes the same three-index factors.
    """

    import numpy as np
    from pyscf import scf
    from pyscf.cc import rccsd
    from gpu4pyscf.cc.lowrank import RRDoubles

    if projector.rank != nocc * nvir:
        raise ValueError("the convergence oracle requires a full-rank projector")
    if max_cycle < 1:
        raise ValueError("convergence-max-cycle must be positive")
    tolerances = (
        energy_tol,
        residual_tol,
        oracle_energy_atol,
        oracle_amplitude_atol,
    )
    if any(not math.isfinite(value) or value <= 0 for value in tolerances):
        raise ValueError("all convergence tolerances must be positive and finite")

    factors = _copy_to_host(integrals.factors, np, transfer_counter)
    fock_host = _copy_to_host(fock, np, transfer_counter)
    energy_host = _copy_to_host(
        orbital_energies, np, transfer_counter
    )
    mo_coeff_host = _copy_to_host(mf.mo_coeff, np, transfer_counter)
    occupied = slice(0, nocc)
    virtual = slice(nocc, nocc + nvir)
    loo = factors[:, occupied, occupied]
    lov = factors[:, occupied, virtual]
    lvo = factors[:, virtual, occupied]
    lvv = factors[:, virtual, virtual]

    def eri_block(left: Any, right: Any) -> Any:
        return np.einsum(
            "Qpq,Qrs->pqrs", left, right.conj(), optimize=True
        )

    dense_blocks = {
        "oooo": eri_block(loo, loo),
        "ovoo": eri_block(lov, loo),
        "oovv": eri_block(loo, lvv),
        "ovvo": eri_block(lov, lvo),
        "ovov": eri_block(lov, lov),
        "ovvv": eri_block(lov, lvv),
        "vvvv": eri_block(lvv, lvv),
    }
    eris = rccsd._ChemistsERIs(mol)
    eris.nocc = nocc
    eris.fock = fock_host
    eris.mo_energy = energy_host
    eris.mo_coeff = None
    for name, block in dense_blocks.items():
        setattr(eris, name, block)

    cpu_mf = scf.RHF(mol)
    cpu_mf.mo_coeff = mo_coeff_host
    cpu_mf.mo_energy = energy_host
    cpu_mf.mo_occ = np.concatenate(
        (np.full(nocc, 2.0), np.zeros(nvir))
    )
    cpu_mf.e_tot = float(mf.e_tot)
    cpu_mf.converged = True
    oracle = rccsd.RCCSD(cpu_mf)
    oracle.max_cycle = int(max_cycle)
    oracle.conv_tol = float(energy_tol)
    oracle.conv_tol_normt = float(residual_tol)
    _emp2, initial_t1_host, initial_t2_host = oracle.init_amps(eris)

    xp = type(integrals.factors).__module__.split(".", 1)[0]
    if xp != "cupy":
        raise TypeError("the MTU convergence gate requires resident CuPy factors")
    import cupy

    initial_t1 = cupy.asarray(initial_t1_host)
    initial_t2 = cupy.asarray(initial_t2_host)
    transfer_counter.record_h2d(
        int(initial_t1_host.nbytes + initial_t2_host.nbytes), count=2
    )
    initial_doubles = RRDoubles.from_t2(initial_t2, projector)

    cupy.cuda.get_current_stream().synchronize()
    rr_started = time.perf_counter()
    rr = engine.kernel(
        initial_t1,
        initial_doubles,
        max_cycle=max_cycle,
        conv_tol=energy_tol,
        conv_tol_normt=residual_tol,
        diis_space=6,
        diis_start_cycle=1,
    )
    cupy.cuda.get_current_stream().synchronize()
    rr_seconds = time.perf_counter() - rr_started

    oracle_started = time.perf_counter()
    oracle_energy, oracle_t1, oracle_t2 = oracle.kernel(
        t1=initial_t1_host,
        t2=initial_t2_host,
        eris=eris,
    )
    oracle_seconds = time.perf_counter() - oracle_started

    rr_t1 = _copy_to_host(rr.t1, np, transfer_counter)
    rr_t2_device = rr.doubles.reconstruct_t2()
    rr_t2 = _copy_to_host(rr_t2_device, np, transfer_counter)
    errors = {
        "correlation_energy_abs": abs(float(rr.energy) - float(oracle_energy)),
        "t1_max_abs": _max_abs_error(rr_t1, oracle_t1, np),
        "t2_max_abs": _max_abs_error(rr_t2, oracle_t2, np),
    }
    passed = bool(
        rr.converged
        and oracle.converged
        and rr.residual_norm <= residual_tol
        and errors["correlation_energy_abs"] <= oracle_energy_atol
        and errors["t1_max_abs"] <= oracle_amplitude_atol
        and errors["t2_max_abs"] <= oracle_amplitude_atol
    )
    history = []
    for row in rr.history:
        history.append({
            key: (None if not math.isfinite(value) else float(value))
            for key, value in row.items()
        })
    dense_block_bytes = {
        name: int(block.nbytes) for name, block in dense_blocks.items()
    }
    return {
        "executed": True,
        "case_scope": "water1-sto3g full-rank endpoint only",
        "passed": passed,
        "initial_amplitudes": "MP2 from the shared dense oracle ERIs",
        "candidate": {
            "converged": bool(rr.converged),
            "cycles": int(rr.cycles),
            "correlation_energy": float(rr.energy),
            "equation_residual_norm": float(rr.residual_norm),
            "update_norm": float(rr.update_norm),
            "seconds": rr_seconds,
            "history": history,
            "dense_t2_in_iteration": False,
            "four_index_eri_in_iteration": False,
        },
        "oracle": {
            "driver": "pyscf.cc.rccsd.RCCSD.kernel",
            "converged": bool(oracle.converged),
            "cycles": int(oracle.cycles),
            "correlation_energy": float(oracle_energy),
            "seconds": oracle_seconds,
            "dense_t2_materialized": True,
            "four_index_eri_materialized": True,
        },
        "tolerances": {
            "candidate_energy_change": float(energy_tol),
            "candidate_equation_residual": float(residual_tol),
            "oracle_energy_abs": float(oracle_energy_atol),
            "oracle_amplitude_max_abs": float(oracle_amplitude_atol),
        },
        "errors": errors,
        "dense_materialization": {
            "scope": "independent water1 validation oracle after RR convergence",
            "mo_eri_block_nbytes": dense_block_bytes,
            "mo_eri_blocks_total_nbytes": sum(dense_block_bytes.values()),
            "production_rr_iteration_dense_t2": False,
            "production_rr_iteration_four_index_eri": False,
        },
        "transfers": transfer_counter.to_dict(),
        "performance_eligible": False,
    }


def _default_output(case_id: str) -> Path:
    run_id = os.getenv("SLURM_JOB_ID", str(os.getpid()))
    return ROOT / "results" / f"rr-smoke-{case_id}-{run_id}.json"


def _initial_record(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": _utc_now(),
        "case": case,
        "geometry_sha256": _geometry_hash(case),
        "contract": {
            "purpose": "real-molecule RR-engine one-Jacobi integration gate",
            "device": "single CUDA GPU",
            "scf": "all-electron GPU RHF",
            "integral_factorization": "auxiliary-basis density fitting",
            "auxbasis": args.auxbasis,
            "one_jacobi_equation_evaluation": True,
            "performance_eligible": False,
            "production_path_dense_t2_materialized": False,
            "production_path_four_index_eri_materialized": False,
            "mp2_pair_matrix_materialized": False,
        },
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output", "overwrite"}
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "slurm": {
                key: os.getenv(key)
                for key in (
                    "SLURM_JOB_ID",
                    "SLURM_JOB_NAME",
                    "SLURM_JOB_NODELIST",
                    "SLURM_JOB_PARTITION",
                    "SLURM_CPUS_PER_TASK",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
            "thread_environment": {
                key: os.getenv(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "GPU4PYSCF_NUMA",
                )
            },
        },
        "source": {
            "frozen_baseline_commit": os.getenv(
                "FROZEN_GPU4PYSCF_COMMIT", FROZEN_BASE_COMMIT
            ),
            "git_head": _command(
                ["git", "-C", str(ROOT.parents[2]), "rev-parse", "HEAD"]
            ),
        },
        "software": {
            name: _version(name)
            for name in (
                "numpy",
                "scipy",
                "pyscf",
                "cupy-cuda12x",
                "gpu4pyscf-cuda12x",
            )
        },
        "gpu": _gpu_hardware(),
    }


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    case = _load_case(args.case)
    output = args.output or _default_output(args.case)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    record = _initial_record(args, case)
    _json_write(output, record)
    monitor = _HBMMonitor()
    timer = None
    started = time.perf_counter()
    try:
        import cupy as cp
        from pyscf import gto, lib
        from gpu4pyscf.cc.device_runtime import RunMetrics, TransferCounter
        from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
        from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
        from gpu4pyscf.cc.rr_engine import RRCCSDIterationEngine
        from gpu4pyscf.cc.rr_projector import (
            MP2PairOperator,
            build_rr_projector_lanczos,
        )
        from gpu4pyscf.df.df import DF
        from gpu4pyscf.scf.hf import RHF

        if int(cp.cuda.runtime.getDeviceCount()) != 1:
            raise RuntimeError("this smoke gate requires exactly one visible GPU")
        if args.max_memory_mb <= 0:
            raise ValueError("max-memory-mb must be positive")
        for name in (
            "df_block_size",
            "auxiliary_block_size",
            "virtual_block_size",
            "denominator_max_rank",
            "rr_max_rank",
            "convergence_max_cycle",
        ):
            if int(getattr(args, name)) < 1:
                raise ValueError(f"{name.replace('_', '-')} must be positive")

        lib.num_threads(int(os.getenv("SLURM_CPUS_PER_TASK", "8")))
        monitor.start()
        timer = _PhaseTimer(cp)
        timer.sample_memory("start")
        with timer.phase("molecule_setup"):
            mol = gto.M(
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
        observed = {
            "nao": int(mol.nao_nr()),
            "nocc": int(mol.nelectron // 2),
            "nvir": int(mol.nao_nr() - mol.nelectron // 2),
        }
        if observed != case["expected_dimensions"]:
            raise RuntimeError(
                "molecular dimensions differ from the frozen case: "
                f"expected {case['expected_dimensions']}, observed {observed}"
            )
        nocc, nvir = observed["nocc"], observed["nvir"]
        pair_dimension = nocc * nvir

        with timer.phase("gpu_rhf"):
            mf = RHF(mol)
            mf.conv_tol = args.scf_conv_tol
            mf.max_cycle = 100
            mf.kernel()
        if not mf.converged:
            raise RuntimeError("GPU RHF did not converge")

        metrics = RunMetrics(
            "real-molecule-rr-jacobi-smoke",
            metadata={
                "case": args.case,
                "factorization": "df",
                "performance_eligible": False,
            },
        )
        with timer.phase("canonical_mo_fock"):
            mo_coeff = _to_device(mf.mo_coeff, cp, metrics.transfers)
            orbital_energies = _to_device(
                mf.mo_energy, cp, metrics.transfers
            )
            ao_fock = _to_device(mf.get_fock(), cp, metrics.transfers)
            fock = mo_coeff.T.conj() @ ao_fock @ mo_coeff
            off_diagonal = fock - cp.diag(cp.diag(fock))
            maximum_fock_off_diagonal = _read_device_scalar(
                cp.max(cp.abs(off_diagonal)), metrics.transfers
            )

        with timer.phase("df_build"):
            dfobj = DF(mol, auxbasis=args.auxbasis)
            dfobj.use_gpu_memory = True
            dfobj.build()
        raw_naux = int(dfobj.naux)
        with timer.phase("df_ao_to_mo"):
            integrals = MOThreeIndexIntegralProvider.from_gpu4pyscf_df(
                dfobj,
                mo_coeff,
                nocc,
                auxiliary_block_size=args.df_block_size,
                transfer_counter=metrics.transfers,
                require_gpu_resident=True,
            )
        if integrals.factorization != "df" or integrals.threshold is not None:
            raise RuntimeError("the molecular smoke adapter lost its DF label")

        denominator_rank_cap = min(
            args.denominator_max_rank, pair_dimension
        )
        with timer.phase("mp2_pair_operator"):
            operator = MP2PairOperator(
                integrals.L_ov,
                orbital_energies[:nocc],
                orbital_energies[nocc:],
                denominator_tolerance=args.denominator_tolerance,
                denominator_max_rank=denominator_rank_cap,
                transfer_counter=metrics.transfers,
            )

        projector_build_metadata: dict[str, Any]
        with timer.phase("rr_projector"):
            if args.case == "water1-sto3g":
                vectors = cp.eye(pair_dimension, dtype=integrals.dtype)
                eigenvalues = cp.zeros(pair_dimension, dtype=integrals.dtype)
                projector = RRProjector(
                    vectors=vectors,
                    eigenvalues=eigenvalues,
                    cutoff=0.0,
                    full_dimension=pair_dimension,
                    source="full-rank-identity-real-molecule-smoke",
                )
                projector_build_metadata = {
                    "solver": "identity",
                    "requested_subspace": pair_dimension,
                    "cutoff_bracketed": True,
                    "capped": False,
                    "max_ritz_residual": None,
                    "diagnostic_only": True,
                }
            else:
                build = build_rr_projector_lanczos(
                    operator,
                    rr_eig_cutoff=args.rr_eig_cutoff,
                    initial_rank=min(args.rr_max_rank, pair_dimension - 1),
                    max_rank=min(args.rr_max_rank, pair_dimension - 1),
                    solver_tolerance=args.lanczos_tolerance,
                    dense_fallback_dimension=0,
                    ritz_residual_tolerance=args.ritz_residual_tolerance,
                    allow_incomplete=True,
                )
                projector = build.projector
                projector_build_metadata = {
                    **build.metadata(transfer_counter=metrics.transfers),
                    "diagnostic_only": True,
                }

        with timer.phase("rr_engine_setup"):
            engine = RRCCSDIterationEngine(
                integrals,
                projector,
                fock,
                orbital_energies,
                auxiliary_block_size=min(
                    args.auxiliary_block_size, integrals.naux
                ),
                virtual_block_size=min(args.virtual_block_size, nvir),
                rr_ring_kernel=args.rr_ring_kernel,
                metrics=metrics,
            )
            t1 = cp.zeros((nocc, nvir), dtype=integrals.dtype)
            initial_core = cp.zeros(
                (projector.rank, projector.rank), dtype=integrals.dtype
            )
            doubles = RRDoubles(projector, initial_core, nocc, nvir)

        with timer.phase("one_rr_jacobi_equation"):
            result = engine.jacobi(t1, doubles)

        if args.case == "water1-sto3g":
            oracle_transfers = TransferCounter()
            with timer.phase("water1_dense_pyscf_oracle"):
                dense_oracle = _water1_dense_pyscf_oracle(
                    mol=mol,
                    integrals=integrals,
                    fock=fock,
                    orbital_energies=orbital_energies,
                    result=result,
                    nocc=nocc,
                    nvir=nvir,
                    auxiliary_block_size=min(
                        args.auxiliary_block_size, integrals.naux
                    ),
                    tensor_atol=args.oracle_tensor_atol,
                    energy_atol=args.oracle_energy_atol,
                    transfer_counter=oracle_transfers,
                )
            if args.water1_convergence:
                convergence_transfers = TransferCounter()
                with timer.phase("water1_full_rank_convergence_oracle"):
                    convergence_oracle = (
                        _water1_dense_pyscf_convergence_oracle(
                            mol=mol,
                            mf=mf,
                            integrals=integrals,
                            fock=fock,
                            orbital_energies=orbital_energies,
                            engine=engine,
                            projector=projector,
                            nocc=nocc,
                            nvir=nvir,
                            max_cycle=args.convergence_max_cycle,
                            energy_tol=args.convergence_energy_tol,
                            residual_tol=args.convergence_residual_tol,
                            oracle_energy_atol=(
                                args.convergence_oracle_energy_atol
                            ),
                            oracle_amplitude_atol=(
                                args.convergence_oracle_amplitude_atol
                            ),
                            transfer_counter=convergence_transfers,
                        )
                    )
            else:
                convergence_oracle = {
                    "executed": False,
                    "case_scope": "water1-sto3g full-rank endpoint only",
                    "reason": "--water1-convergence was not supplied",
                    "performance_eligible": False,
                }
        else:
            dense_oracle = {
                "executed": False,
                "case_scope": "water1-sto3g only",
                "reason": "dense validation oracle is prohibited for water2",
                "performance_eligible": False,
            }
            convergence_oracle = {
                "executed": False,
                "case_scope": "water1-sto3g full-rank endpoint only",
                "reason": "convergence oracle is prohibited for water2",
                "performance_eligible": False,
            }

        result_scalars = {
            "input_state_energy": _read_device_scalar(
                result.energy, metrics.transfers
            ),
            "equation_residual_norm": _read_device_scalar(
                result.residual_norm, metrics.transfers
            ),
            "jacobi_update_norm": _read_device_scalar(
                result.update_norm, metrics.transfers
            ),
            "new_t1_norm": _read_device_scalar(
                cp.linalg.norm(result.t1), metrics.transfers
            ),
            "new_rr_core_norm": _read_device_scalar(
                cp.linalg.norm(result.doubles.core), metrics.transfers
            ),
            "maximum_canonical_fock_off_diagonal": (
                maximum_fock_off_diagonal
            ),
        }
        if not all(math.isfinite(value) for value in result_scalars.values()):
            raise FloatingPointError("RR equation produced a non-finite diagnostic")
        arrays = {
            "mo_coeff": mo_coeff,
            "orbital_energies": orbital_energies,
            "fock_mo": fock,
            "mo_three_index_factors": integrals.factors,
            "rr_projector": projector.vectors,
            "rr_core_input": doubles.core,
            "t1_output": result.t1,
            "rr_core_output": result.doubles.core,
            "singles_numerator": result.singles_equation.numerator,
            "doubles_numerator": result.doubles_equation.core,
        }
        residency = {
            name: {
                "cupy_array": _is_cupy_array(value),
                "device_id": _device_id(value),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in arrays.items()
        }
        if not all(item["cupy_array"] for item in residency.values()):
            raise RuntimeError("one or more RR smoke arrays left the GPU")
        if {item["device_id"] for item in residency.values()} != {0}:
            raise RuntimeError("RR smoke arrays are not all on visible GPU 0")

        # Metadata calls that validate orthogonality perform explicit and
        # counted device-to-host reads.  Take the transfer snapshot afterwards.
        projector_metadata = projector.metadata(
            transfer_counter=metrics.transfers
        )
        equations = {
            "singles": result.singles_equation.metadata(),
            "doubles": result.doubles_equation.metadata(),
        }
        dense_t2 = any(
            bool(item.get("materialized_dense_t2"))
            for item in equations.values()
        )
        dense_eri = any(
            bool(item.get("materialized_four_index_eri"))
            for item in equations.values()
        )
        if dense_t2 or dense_eri:
            raise RuntimeError("the RR equation metadata reports dense materialization")

        record.update(
            {
                "status": "passed",
                "finished_at_utc": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "dimensions": {
                    **observed,
                    "pair_dimension": pair_dimension,
                    "raw_auxiliary_naux": raw_naux,
                    "effective_df_rank": integrals.naux,
                    "rr_rank": projector.rank,
                },
                "rhf": {
                    "converged": bool(mf.converged),
                    "energy": float(mf.e_tot),
                    "convergence_tolerance": args.scf_conv_tol,
                },
                "integrals": integrals.metadata(),
                "mp2_pair_operator": operator.metadata(),
                "rr_projector_build": projector_build_metadata,
                "rr_projector": projector_metadata,
                "rr_engine": engine.metadata(),
                "equations": equations,
                "dense_pyscf_one_step_oracle": dense_oracle,
                "dense_pyscf_convergence_oracle": convergence_oracle,
                "result": result_scalars,
                "device_residency": residency,
                "timings_seconds": timer.timings,
                "run_metrics": metrics.to_dict(),
                "claims": {
                    "one_jacobi_equation_evaluation_completed": True,
                    "factorization_is_auxiliary_basis_df": True,
                    "tight_direct_ao_pair_cd_implemented": False,
                    "production_path": {
                        "dense_t2_materialized": dense_t2,
                        "four_index_eri_materialized": dense_eri,
                        "mp2_pair_matrix_materialized": False,
                        "performance_eligible": False,
                    },
                    "validation_oracle_executed": bool(
                        dense_oracle["executed"]
                    ),
                    "convergence_oracle_executed": bool(
                        convergence_oracle["executed"]
                    ),
                    "mp2_pair_matrix_materialized": False,
                    "all_audited_arrays_gpu_resident": True,
                    "performance_eligible": False,
                },
            }
        )
        if dense_oracle.get("executed") and not dense_oracle.get("passed"):
            raise AssertionError(
                "water1 dense PySCF RCCSD oracle parity failed: "
                + json.dumps(dense_oracle["errors"], sort_keys=True)
            )
        if (
            convergence_oracle.get("executed")
            and not convergence_oracle.get("passed")
        ):
            raise AssertionError(
                "water1 full-rank convergence parity failed: "
                + json.dumps(convergence_oracle["errors"], sort_keys=True)
            )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        raise
    finally:
        monitor.close()
        record["hbm_nvidia_smi"] = monitor.metadata()
        if timer is not None:
            record["hbm_cuda_samples"] = timer.memory_metadata()
        record["host_peak_rss_mib"] = _host_rss_mib()
        _json_write(output, record)
    return output, record


def main() -> int:
    args = _parser().parse_args()
    output, record = run(args)
    print(
        json.dumps(
            {
                "status": record["status"],
                "case": args.case,
                "output": str(output),
                "wall_seconds": record.get("wall_seconds"),
                "dimensions": record.get("dimensions"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
