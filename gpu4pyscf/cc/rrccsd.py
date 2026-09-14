# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rank-reduced RCCSD drivers for canonical validation and direct AO CD.

``eri_backend="canonical"`` retains the original dense projected validation
driver.  ``eri_backend="cd"`` is the resident, production-shaped path:

``GINT selected AO-pair columns -> pivoted CD -> MO three-index factors``
``-> matrix-free MP2 projector -> compressed RR-CCSD equations``.

The CD path never constructs a square AO-pair matrix, four-index MO ERIs, an
MP2 pair matrix (apart from the explicitly labelled small dense validation
endpoint), or a dense doubles tensor during an ordinary run.  Its ring
contraction defaults to the bounded-memory reference kernel; the opt-in GEMM
selector remains performance ineligible until the ring and GINT providers
pass their water2 timing gates.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import reduce
import math
from typing import Any, Dict, Optional

import numpy as np

from gpu4pyscf.cc import ccsd_incore
from gpu4pyscf.cc.device_runtime import RunMetrics, nvtx_range
from gpu4pyscf.cc.direct_cd import pivoted_cholesky_from_columns
from gpu4pyscf.cc.full_space_residual import (
    FullSpaceResidualDiagnostic,
    diagnose_reconstructed_full_space_residual,
)
from gpu4pyscf.cc.gint_pair_columns import GINTAOPairColumnProvider
from gpu4pyscf.cc.gint_selected_columns import GINTSelectedAOPairColumnProvider
from gpu4pyscf.cc.gint_transfer_audit import RUNTIME_GATE_EXECUTION_MODES
from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.lowrank import (
    RRDoubles,
    RRProjector,
    build_rr_projector,
    pair_matrix_from_t2,
)
from gpu4pyscf.cc.residual_evaluator import (
    evaluate_dense_ccsd_residual,
    labeled_low_rank_residuals,
)
from gpu4pyscf.cc.rr_engine import RRCCSDIterationEngine
from gpu4pyscf.cc.rr_projector import MP2PairOperator, build_rr_projector_lanczos
from gpu4pyscf.cc.rr_residual import ProjectedPairDenominator, rr_ccsd_energy


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def _array_backend_device_identity(array: Any) -> tuple[str, Optional[int]]:
    """Return a stable cache identity for a NumPy or CuPy array location."""

    xp = _array_module(array)
    if xp is np:
        return "numpy", None
    device = getattr(array, "device", None)
    device_id = getattr(device, "id", None)
    if device_id is None:
        raise TypeError("a CuPy array must expose its CUDA device identity")
    return "cupy", int(device_id)


def _array_device_scope(array: Any):
    """Select an array's CUDA device while preserving the caller's device."""

    backend, device_id = _array_backend_device_identity(array)
    if backend == "numpy":
        return nullcontext()
    xp = _array_module(array)
    return xp.cuda.Device(device_id)


def _finite_float(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_positive_integer(name: str, value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer or None")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(name: str, value: Any) -> int:
    result = _optional_positive_integer(name, value)
    if result is None:  # pragma: no cover - private caller invariant
        raise TypeError(f"{name} must be an integer")
    return result


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _active_restricted_orbitals(
    mo_coeff: Any,
    mo_occ: Any,
    frozen_mask: Any,
    *,
    nocc: int,
    nmo: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select and validate a real closed-shell active MO space on the host."""

    try:
        coefficient = np.asarray(mo_coeff)
        occupations = np.asarray(mo_occ)
        mask = np.asarray(frozen_mask, dtype=bool)
    except Exception as exc:
        raise TypeError(
            "RRCCSD reference orbitals must be host arrays after the mean-field "
            "object is converted to CPU"
        ) from exc
    if coefficient.ndim != 2:
        raise ValueError("restricted mo_coeff must be a rank-two array")
    if occupations.shape != (coefficient.shape[1],):
        raise ValueError("mo_occ does not match mo_coeff")
    if mask.shape != occupations.shape:
        raise ValueError("get_frozen_mask() does not match the MO space")
    if np.iscomplexobj(coefficient):
        raise NotImplementedError("the direct-CD RR engine supports real RHF only")
    if not np.issubdtype(coefficient.dtype, np.floating):
        raise TypeError("mo_coeff must use a real floating-point dtype")
    if coefficient.dtype != np.dtype(np.float64):
        raise TypeError("the fp64 RRCCSD path requires float64 MO coefficients")
    if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(occupations)):
        raise ValueError("MO coefficients and occupations must be finite")
    nocc = int(nocc)
    nmo = int(nmo)
    if not 0 < nocc < nmo:
        raise ValueError("the active space must contain occupied and virtual MOs")
    active_coefficient = coefficient[:, mask]
    active_occupations = occupations[mask]
    if active_coefficient.shape[1] != nmo:
        raise ValueError(
            "get_frozen_mask() active count disagrees with CCSD nmo: "
            f"{active_coefficient.shape[1]} != {nmo}"
        )
    if not np.allclose(active_occupations[:nocc], 2.0, rtol=0.0, atol=1e-12):
        raise NotImplementedError(
            "the direct-CD RR engine requires doubly occupied active RHF orbitals"
        )
    if not np.allclose(active_occupations[nocc:], 0.0, rtol=0.0, atol=1e-12):
        raise NotImplementedError(
            "the direct-CD RR engine requires unoccupied active virtual orbitals"
        )
    return active_coefficient, active_occupations, mask


def _resident_array(
    value: Any,
    xp: Any,
    transfer_counter: Any,
    *,
    operation: str,
):
    """Place an array on ``xp`` and account for every real host upload."""

    if _array_module(value) is xp:
        return value
    if xp is np:
        raise TypeError("refusing an implicit device-to-host array transfer")
    host = np.asarray(value)
    resident = xp.asarray(host)
    transfer_counter.record_h2d(int(host.nbytes), operation=operation)
    return resident


def _synchronize(array: Any) -> None:
    xp = _array_module(array)
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()


def _diagnostic_memory_snapshot(reference: Any) -> dict[str, Any]:
    """Return a non-allocating memory snapshot for diagnostic provenance."""

    xp = _array_module(reference)
    if xp is np:
        return {"array_backend": "numpy", "device_memory_available": False}
    free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
    pool = xp.get_default_memory_pool()
    return {
        "array_backend": "cupy",
        "device_memory_available": True,
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
        "cupy_pool_used_bytes": int(pool.used_bytes()),
        "cupy_pool_total_bytes": int(pool.total_bytes()),
    }


def _transfer_ledger_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Subtract two :class:`TransferCounter` snapshots without mutation."""

    kinds = set(before.get("by_kind", {})) | set(after.get("by_kind", {}))
    by_kind = {}
    for kind in sorted(kinds):
        left = before.get("by_kind", {}).get(kind, {})
        right = after.get("by_kind", {}).get(kind, {})
        by_kind[kind] = {
            "bytes": int(right.get("bytes", 0)) - int(left.get("bytes", 0)),
            "count": int(right.get("count", 0)) - int(left.get("count", 0)),
        }
    operations = (
        set(before.get("by_operation", {}))
        | set(after.get("by_operation", {}))
    )
    by_operation: dict[str, dict[str, dict[str, int]]] = {}
    for kind in sorted(operations):
        left_kind = before.get("by_operation", {}).get(kind, {})
        right_kind = after.get("by_operation", {}).get(kind, {})
        entries = {}
        for operation in sorted(set(left_kind) | set(right_kind)):
            left = left_kind.get(operation, {})
            right = right_kind.get(operation, {})
            byte_delta = int(right.get("bytes", 0)) - int(
                left.get("bytes", 0)
            )
            count_delta = int(right.get("count", 0)) - int(
                left.get("count", 0)
            )
            if byte_delta or count_delta:
                entries[operation] = {
                    "bytes": byte_delta,
                    "count": count_delta,
                }
        if entries:
            by_operation[kind] = entries
    return {
        "total_bytes": int(after.get("total_bytes", 0))
        - int(before.get("total_bytes", 0)),
        "total_transfers": int(after.get("total_transfers", 0))
        - int(before.get("total_transfers", 0)),
        "by_kind": by_kind,
        "by_operation": by_operation,
    }


def _canonical_rr_galerkin_step(
    current_t2: Any,
    jacobi_t2: Any,
    projector: RRProjector,
    orbital_energy_differences: Any,
    *,
    denominator: Optional[ProjectedPairDenominator] = None,
    transfer_counter: Any = None,
) -> tuple[RRDoubles, ProjectedPairDenominator]:
    """Apply one dense validation update for the RR Galerkin equation.

    A canonical CCSD update supplies the Jacobi candidate ``J2(T)``.  The raw
    doubles equation residual at the current state is

    ``R2(T) = D2 * (J2(T) - T2)``,

    where ``D2[(ia),(jb)] = eia[ia] + eia[jb]``.  The RR update solves

    ``K dC + dC K = U.H R2(T) U`` with ``K = U.H diag(eia) U``

    and returns ``C + dC``.  Its fixed point is therefore the Galerkin
    equation ``U.H R2(T) U = 0``.  Projecting ``J2`` directly would instead
    impose a denominator-weighted Petrov condition when the pair denominators
    are non-uniform.

    This helper deliberately accepts dense tensors: it is the canonical
    validation path, not the reduced-scaling direct-CD implementation.
    """

    if getattr(current_t2, "ndim", None) != 4:
        raise ValueError("current_t2 must have shape (nocc,nocc,nvir,nvir)")
    nocc, nocc_j, nvir, nvir_b = (
        int(value) for value in current_t2.shape
    )
    if nocc != nocc_j or nvir != nvir_b:
        raise ValueError("current_t2 occupied and virtual dimensions must be square")
    if getattr(jacobi_t2, "shape", None) != current_t2.shape:
        raise ValueError("jacobi_t2 must match current_t2")
    if getattr(orbital_energy_differences, "shape", None) != (nocc, nvir):
        raise ValueError(
            "orbital_energy_differences must have shape (nocc,nvir)"
        )
    if projector.full_dimension != nocc * nvir:
        raise ValueError("projector pair dimension does not match current_t2")

    vectors = projector.vectors
    xp = _array_module(current_t2)
    location = _array_backend_device_identity(current_t2)
    if any(
        _array_module(value) is not xp
        or _array_backend_device_identity(value) != location
        for value in (jacobi_t2, orbital_energy_differences, vectors)
    ):
        raise TypeError(
            "canonical RR tensors, projector, and denominators need one "
            "backend and device"
        )
    dtypes = tuple(
        np.dtype(value.dtype)
        for value in (
            current_t2,
            jacobi_t2,
            orbital_energy_differences,
            vectors,
        )
    )
    if any(np.issubdtype(dtype, np.complexfloating) for dtype in dtypes):
        raise NotImplementedError(
            "the canonical RR Galerkin validation path supports real RHF only"
        )
    if any(not np.issubdtype(dtype, np.floating) for dtype in dtypes):
        raise TypeError("canonical RR Galerkin tensors must be floating point")

    with _array_device_scope(current_t2):
        if denominator is None:
            denominator = ProjectedPairDenominator.build(
                projector,
                orbital_energy_differences,
                nocc,
                nvir,
                transfer_counter=transfer_counter,
            )
        elif (
            denominator.nocc != nocc
            or denominator.nvir != nvir
            or denominator.rank != projector.rank
            or _array_module(denominator.matrix) is not xp
            or _array_backend_device_identity(denominator.matrix) != location
        ):
            raise ValueError("cached projected denominator does not match RR space")

        pair_update = pair_matrix_from_t2(jacobi_t2 - current_t2)
        pair_denominator = orbital_energy_differences.reshape(-1)
        raw_residual = (
            pair_denominator[:, None] + pair_denominator[None, :]
        ) * pair_update
        projected_residual = vectors.T.conj() @ raw_residual @ vectors
        delta_core = denominator.solve(
            projected_residual,
            transfer_counter=transfer_counter,
        )
        current_core = RRDoubles.from_t2(current_t2, projector).core
        updated_core = current_core + delta_core
        updated_core = (updated_core + updated_core.T.conj()) * 0.5
        return (
            RRDoubles(projector, updated_core, nocc, nvir),
            denominator,
        )


def _canonical_rr_convergence_gate(
    iterative_converged: bool,
    projected_equation_residual: Any,
    tolerance: Any,
) -> tuple[bool, dict[str, Any]]:
    """Fail closed when the final raw Galerkin residual misses its contract."""

    threshold = _finite_float("conv_tol_normt", tolerance, positive=True)
    try:
        residual = _finite_float(
            "projected_equation_residual", projected_equation_residual
        )
    except (TypeError, ValueError, OverflowError):
        return False, {
            "measurement": "raw-projected-equation-residual",
            "residual_norm": None,
            "threshold": threshold,
            "available": False,
            "passed": False,
            "failure_reason": "missing-or-invalid-residual",
            "iterative_solver_converged_before_gate": bool(
                iterative_converged
            ),
            "reported_converged": False,
        }
    passed = residual <= threshold
    reported = bool(iterative_converged) and passed
    return reported, {
        "measurement": "raw-projected-equation-residual",
        "residual_norm": residual,
        "threshold": threshold,
        "available": True,
        "passed": passed,
        "iterative_solver_converged_before_gate": bool(iterative_converged),
        "reported_converged": reported,
    }


class RRCCSD(ccsd_incore.CCSD):
    """RCCSD constrained to a fixed MP2 doubles eigenspace.

    Parameters
    ----------
    eri_tol
        Absolute residual-diagonal threshold for direct AO-pair pivoted CD.
    rr_eig_cutoff
        Absolute magnitude cutoff for the symmetric MP2 pair operator.
    eri_backend
        ``"cd"`` selects the resident direct-CD RR equations. ``"canonical"``
        retains the dense projected validation oracle.
    direct_scf_tol
        Independent GINT Schwarz-screening tolerance.  It is deliberately not
        inferred from ``eri_tol`` and both values are recorded.
    gint_column_backend
        ``"selected"`` uses the bounded batched selected-shell C ABI.
        ``"restricted-reference"`` retains the earlier one-column reference
        provider for numerical comparisons.
    gint_column_kernel
        ``"reference"`` retains the qualified one-pivot-per-GOUT kernel.
        ``"grouped"`` reuses GOUT across pivots with the same ket shell task;
        it remains performance-ineligible until the fixed A100 A/B gate.
    gint_max_batch_size
        Maximum number of selected AO-pair columns evaluated in one blocked
        pivoted-Cholesky request.
    denominator_tolerance
        Absolute residual-diagonal threshold for the Cauchy factorization used
        only to construct the MP2 projector and compressed MP2 guess.
    precision
        Only ``"fp64"`` is accepted until the FP64 numerical gates pass.
    rr_ring_kernel
        Ring residual selector: ``"reference"`` by default, or ``"gemm"``
        for the bounded tiled GEMM path.  The latter remains performance
        ineligible until the A100 gates pass.
    """

    method_class = "controlled-approximation"
    acceptance_table = "not_yet_accepted"
    implementation_stage = "dense-projected-validation"
    _keys = set(ccsd_incore.CCSD._keys) | {
        "eri_tol",
        "rr_eig_cutoff",
        "eri_backend",
        "precision",
        "rr_max_rank",
        "direct_scf_tol",
        "cd_max_rank",
        "gint_column_backend",
        "gint_column_kernel",
        "gint_group_size",
        "gint_max_block_bytes",
        "gint_max_batch_size",
        "gint_runtime_gate_receipt",
        "gint_runtime_execution_mode",
        "gint_runtime_source_digest",
        "gint_runtime_source_root",
        "gint_runtime_manifest_path",
        "gint_runtime_topology_path",
        "cd_mo_block_size",
        "denominator_tolerance",
        "denominator_max_rank",
        "rr_initial_rank",
        "rr_solver_tolerance",
        "rr_solver_maxiter",
        "rr_dense_fallback_dimension",
        "rr_ritz_residual_tolerance",
        "rr_auxiliary_block_size",
        "rr_virtual_block_size",
        "rr_ring_kernel",
        "rr_projector",
        "doubles",
        "projected_equation_residual",
        "full_space_residual",
        "projected_jacobi_update_norm",
        "full_space_jacobi_update_norm",
        "representation_residual",
        "rr_history",
        "full_space_diagnostic_metadata",
        "run_metrics",
    }

    def __init__(
        self,
        mf: Any,
        *args: Any,
        eri_tol: Optional[float] = None,
        rr_eig_cutoff: Optional[float] = None,
        eri_backend: str = "cd",
        precision: str = "fp64",
        rr_max_rank: Optional[int] = None,
        direct_scf_tol: float = 1e-13,
        cd_max_rank: Optional[int] = None,
        gint_column_backend: str = "selected",
        gint_column_kernel: str = "reference",
        gint_group_size: int = 16,
        gint_max_block_bytes: Optional[int] = None,
        gint_max_batch_size: int = 32,
        gint_runtime_gate_receipt: Optional[str] = None,
        gint_runtime_execution_mode: str = "release-gate",
        gint_runtime_source_digest: Optional[str] = None,
        gint_runtime_source_root: Optional[str] = None,
        gint_runtime_manifest_path: Optional[str] = None,
        gint_runtime_topology_path: Optional[str] = None,
        cd_mo_block_size: int = 32,
        denominator_tolerance: float = 1e-10,
        denominator_max_rank: Optional[int] = None,
        rr_initial_rank: int = 32,
        rr_solver_tolerance: float = 1e-10,
        rr_solver_maxiter: Optional[int] = None,
        rr_dense_fallback_dimension: int = 64,
        rr_ritz_residual_tolerance: Optional[float] = None,
        rr_auxiliary_block_size: int = 1,
        rr_virtual_block_size: int = 8,
        rr_ring_kernel: str = "reference",
        **kwargs: Any,
    ) -> None:
        if eri_tol is None:
            raise TypeError("eri_tol must be supplied explicitly")
        if rr_eig_cutoff is None:
            raise TypeError("rr_eig_cutoff must be supplied explicitly")
        eri_tol = _finite_float("eri_tol", eri_tol)
        rr_eig_cutoff = _finite_float("rr_eig_cutoff", rr_eig_cutoff)
        direct_scf_tol = _finite_float(
            "direct_scf_tol", direct_scf_tol, positive=True
        )
        if direct_scf_tol > 1.0:
            raise ValueError("direct_scf_tol must lie in (0, 1]")
        denominator_tolerance = _finite_float(
            "denominator_tolerance", denominator_tolerance
        )
        rr_solver_tolerance = _finite_float(
            "rr_solver_tolerance", rr_solver_tolerance, positive=True
        )
        if rr_ritz_residual_tolerance is not None:
            rr_ritz_residual_tolerance = _finite_float(
                "rr_ritz_residual_tolerance", rr_ritz_residual_tolerance
            )
        eri_backend = str(eri_backend).lower()
        if eri_backend not in {"canonical", "cd"}:
            raise ValueError("eri_backend must be 'canonical' or 'cd'")
        gint_column_backend = str(gint_column_backend).lower()
        if gint_column_backend not in {"selected", "restricted-reference"}:
            raise ValueError(
                "gint_column_backend must be 'selected' or "
                "'restricted-reference'"
            )
        gint_column_kernel = str(gint_column_kernel).strip().lower()
        if gint_column_kernel not in {"reference", "grouped"}:
            raise ValueError(
                "gint_column_kernel must be 'reference' or 'grouped'"
            )
        if (
            gint_column_backend != "selected"
            and gint_column_kernel != "reference"
        ):
            raise ValueError(
                "gint_column_kernel applies only to the selected GINT backend"
            )
        gint_runtime_execution_mode = str(
            gint_runtime_execution_mode
        ).strip().lower()
        if gint_runtime_execution_mode not in RUNTIME_GATE_EXECUTION_MODES:
            raise ValueError(
                "gint_runtime_execution_mode must be release-gate, "
                "consumer-benchmark, or consumer-counterpoise"
            )
        runtime_values = (
            gint_runtime_gate_receipt,
            gint_runtime_source_digest,
            gint_runtime_source_root,
            gint_runtime_manifest_path,
            gint_runtime_topology_path,
        )
        if (eri_backend != "cd" or gint_column_backend != "selected") and (
            any(value is not None for value in runtime_values)
            or gint_runtime_execution_mode != "release-gate"
        ):
            raise ValueError(
                "GINT runtime-gate fields apply only to the selected direct-CD "
                "RR/THC path"
            )
        precision = str(precision).lower()
        if precision != "fp64":
            raise NotImplementedError(
                "RRCCSD supports only fp64 until the FP64 projected residual "
                "and accuracy gates pass"
            )

        rr_max_rank = _optional_positive_integer("rr_max_rank", rr_max_rank)
        cd_max_rank = _optional_positive_integer("cd_max_rank", cd_max_rank)
        denominator_max_rank = _optional_positive_integer(
            "denominator_max_rank", denominator_max_rank
        )
        rr_solver_maxiter = _optional_positive_integer(
            "rr_solver_maxiter", rr_solver_maxiter
        )
        gint_max_block_bytes = _optional_positive_integer(
            "gint_max_block_bytes", gint_max_block_bytes
        )
        gint_group_size = _positive_integer("gint_group_size", gint_group_size)
        gint_max_batch_size = _positive_integer(
            "gint_max_batch_size", gint_max_batch_size
        )
        cd_mo_block_size = _positive_integer("cd_mo_block_size", cd_mo_block_size)
        rr_initial_rank = _positive_integer("rr_initial_rank", rr_initial_rank)
        rr_dense_fallback_dimension = _nonnegative_integer(
            "rr_dense_fallback_dimension", rr_dense_fallback_dimension
        )
        rr_auxiliary_block_size = _positive_integer(
            "rr_auxiliary_block_size", rr_auxiliary_block_size
        )
        rr_virtual_block_size = _positive_integer(
            "rr_virtual_block_size", rr_virtual_block_size
        )
        rr_ring_kernel = str(rr_ring_kernel).lower()
        if rr_ring_kernel not in {"reference", "gemm"}:
            raise ValueError("rr_ring_kernel must be 'reference' or 'gemm'")

        super().__init__(mf, *args, **kwargs)
        self.eri_tol = eri_tol
        self.rr_eig_cutoff = rr_eig_cutoff
        self.eri_backend = eri_backend
        self.precision = precision
        self.rr_max_rank = rr_max_rank
        self.direct_scf_tol = direct_scf_tol
        self.cd_max_rank = cd_max_rank
        self.gint_column_backend = gint_column_backend
        self.gint_column_kernel = gint_column_kernel
        self.gint_group_size = gint_group_size
        self.gint_max_block_bytes = gint_max_block_bytes
        self.gint_max_batch_size = gint_max_batch_size
        self.gint_runtime_gate_receipt = gint_runtime_gate_receipt
        self.gint_runtime_execution_mode = gint_runtime_execution_mode
        self.gint_runtime_source_digest = gint_runtime_source_digest
        self.gint_runtime_source_root = gint_runtime_source_root
        self.gint_runtime_manifest_path = gint_runtime_manifest_path
        self.gint_runtime_topology_path = gint_runtime_topology_path
        self.cd_mo_block_size = cd_mo_block_size
        self.denominator_tolerance = denominator_tolerance
        self.denominator_max_rank = denominator_max_rank
        self.rr_initial_rank = rr_initial_rank
        self.rr_solver_tolerance = rr_solver_tolerance
        self.rr_solver_maxiter = rr_solver_maxiter
        self.rr_dense_fallback_dimension = rr_dense_fallback_dimension
        self.rr_ritz_residual_tolerance = rr_ritz_residual_tolerance
        self.rr_auxiliary_block_size = rr_auxiliary_block_size
        self.rr_virtual_block_size = rr_virtual_block_size
        self.rr_ring_kernel = rr_ring_kernel

        self.rr_projector: Optional[RRProjector] = None
        self.doubles: Optional[RRDoubles] = None
        self.projected_equation_residual: Optional[float] = None
        self.full_space_residual: Optional[float] = None
        self.projected_jacobi_update_norm: Optional[float] = None
        self.full_space_jacobi_update_norm: Optional[float] = None
        self.representation_residual: Optional[float] = None
        self.full_space_diagnostic_metadata: Optional[dict[str, Any]] = None
        self.rr_history: tuple[dict[str, float], ...] = ()
        self._dense_t2_reconstructed = False
        self._cd_integrals: Optional[MOThreeIndexIntegralProvider] = None
        self._rr_fock: Any = None
        self._rr_orbital_energies: Any = None
        self._mp2_operator: Optional[MP2PairOperator] = None
        self._rr_engine: Optional[RRCCSDIterationEngine] = None
        self._cd_initial_t1: Any = None
        self._cd_initial_doubles: Optional[RRDoubles] = None
        self._active_space_metadata: Optional[dict[str, Any]] = None
        self._ao_pair_provider_metadata: Optional[dict[str, Any]] = None
        self._direct_cholesky_metadata: Optional[dict[str, Any]] = None
        self._rr_projector_build_metadata: Optional[dict[str, Any]] = None
        self._mp2_initial_metadata: Optional[dict[str, Any]] = None
        self._canonical_projected_denominator: Optional[
            ProjectedPairDenominator
        ] = None
        self._canonical_projected_denominator_key: Optional[
            tuple[int, int, int, int, float, str, Optional[int]]
        ] = None
        self._canonical_orbital_energy_differences: Any = None
        self._canonical_galerkin_final_gate: Optional[dict[str, Any]] = None
        self.run_metrics = RunMetrics(
            self.__class__.__name__, metadata=self.method_metadata()
        )

    def _clear_cd_runtime(self) -> None:
        self._cd_integrals = None
        self._rr_fock = None
        self._rr_orbital_energies = None
        self._mp2_operator = None
        self._rr_engine = None
        self._cd_initial_t1 = None
        self._cd_initial_doubles = None
        self.rr_projector = None
        self.doubles = None
        self.rr_history = ()
        self._active_space_metadata = None
        self._ao_pair_provider_metadata = None
        self._direct_cholesky_metadata = None
        self._rr_projector_build_metadata = None
        self._mp2_initial_metadata = None
        self._canonical_projected_denominator = None
        self._canonical_projected_denominator_key = None
        self._canonical_orbital_energy_differences = None
        self._canonical_galerkin_final_gate = None
        self.projected_equation_residual = None
        self.full_space_residual = None
        self.projected_jacobi_update_norm = None
        self.full_space_jacobi_update_norm = None
        self.representation_residual = None
        self.full_space_diagnostic_metadata = None
        self._dense_t2_reconstructed = False

    def reset(self, mol: Any = None):
        result = super().reset(mol)
        self._clear_cd_runtime()
        return result

    def residual_metadata(self) -> Dict[str, Any]:
        """Return residual evidence with mandatory space and kind labels."""

        return labeled_low_rank_residuals(
            projected_equation=getattr(
                self, "projected_equation_residual", None
            ),
            full_space_equation=getattr(self, "full_space_residual", None),
            projected_jacobi_update=getattr(
                self, "projected_jacobi_update_norm", None
            ),
            full_space_jacobi_update=getattr(
                self, "full_space_jacobi_update_norm", None
            ),
            representation_error=getattr(
                self, "representation_residual", None
            ),
        )

    def method_metadata(self) -> Dict[str, Any]:
        is_cd = self.eri_backend == "cd"
        residuals = self.residual_metadata()
        ring_stage = (
            "direct-cd-compressed-rr-reference"
            if self.rr_ring_kernel == "reference"
            else "direct-cd-compressed-rr-gemm"
        )
        metadata: Dict[str, Any] = {
            "method": self.__class__.__name__,
            "method_class": self.method_class,
            "acceptance_table": self.acceptance_table,
            "implementation_stage": (
                ring_stage if is_cd else self.implementation_stage
            ),
            "eri_backend": self.eri_backend,
            "eri_tol": self.eri_tol,
            "direct_scf_tol": self.direct_scf_tol,
            "rr_eig_cutoff": self.rr_eig_cutoff,
            "rr_max_rank": self.rr_max_rank,
            "rr_ring_kernel": self.rr_ring_kernel,
            "ring_kernel": self.rr_ring_kernel,
            "precision": self.precision,
            "reduced_scaling_residual": is_cd,
            "dense_t2_in_normal_cd_path": False if is_cd else None,
            "dense_t2_reconstructed": bool(self._dense_t2_reconstructed),
            "performance_eligible": False,
            "residual_schema": residuals["schema"],
            "residuals": residuals,
            "projected_equation_residual_status": (
                "computed"
                if residuals["measurements"]["projected_equation"]["available"]
                else "not-computed"
            ),
            "projected_jacobi_update_status": (
                "computed"
                if residuals["measurements"]["projected_jacobi_update"][
                    "available"
                ]
                else "not-computed"
            ),
            "denominator_approximation": {
                "purpose": "MP2-projector-and-initial-guess-only",
                "tolerance": self.denominator_tolerance,
                "max_rank": self.denominator_max_rank,
            },
        }
        if is_cd:
            metadata.update({
                "direct_cd": {
                    "threshold": self.eri_tol,
                    "max_rank": self.cd_max_rank,
                    "gint_column_backend": self.gint_column_backend,
                    "gint_column_kernel": self.gint_column_kernel,
                    "gint_group_size": self.gint_group_size,
                    "gint_max_block_bytes": self.gint_max_block_bytes,
                    "gint_max_batch_size": self.gint_max_batch_size,
                    "mo_transform_auxiliary_block_size": self.cd_mo_block_size,
                    "gint_runtime_gate_requested": {
                        "execution_mode": self.gint_runtime_execution_mode,
                        "receipt_supplied": (
                            self.gint_runtime_gate_receipt is not None
                        ),
                    },
                },
                "rr_engine_blocks": {
                    "auxiliary": self.rr_auxiliary_block_size,
                    "virtual": self.rr_virtual_block_size,
                    "ring_kernel": self.rr_ring_kernel,
                },
                "lanczos": {
                    "initial_rank": self.rr_initial_rank,
                    "solver_tolerance": self.rr_solver_tolerance,
                    "maxiter": self.rr_solver_maxiter,
                    "dense_fallback_dimension": self.rr_dense_fallback_dimension,
                    "ritz_residual_tolerance": self.rr_ritz_residual_tolerance,
                },
                "full_space_residual_status": (
                    "computed" if self.full_space_residual is not None else
                    "requires-separate-explicit-dense-diagnostic"
                ),
                "performance_limitations": [
                    "complete RR water2/water4 performance gate pending",
                    (
                        "RR ring reference path awaits A100 allocation and "
                        "numerical gates"
                        if self.rr_ring_kernel == "reference"
                        else "RR ring GEMM path awaits A100 allocation and "
                        "numerical gates"
                    ),
                ],
            })
            if self.full_space_diagnostic_metadata is not None:
                metadata["full_space_residual_diagnostic"] = dict(
                    self.full_space_diagnostic_metadata
                )
            if self._ao_pair_provider_metadata is not None:
                provider_gate = self._ao_pair_provider_metadata.get(
                    "runtime_gate"
                )
                if provider_gate is not None:
                    metadata["provider_runtime_gate"] = dict(provider_gate)
        else:
            galerkin = {
                "role": "dense-canonical-validation",
                "doubles_residual": "R2(T)=D2*(J2(T)-T2)",
                "update_equation": (
                    "K*delta_C+delta_C*K=U.H*R2(T)*U; "
                    "K=U.H*diag(e_i-e_a-level_shift)*U"
                ),
                "fixed_point": "U.H*R2(T)*U=0",
                "singles_update": "canonical-Jacobi",
                "dense_t2_materialized": True,
                "projected_denominator_cached": (
                    self._canonical_projected_denominator is not None
                ),
                "final_raw_projected_residual_gate": "conv_tol_normt",
            }
            if self._canonical_projected_denominator is not None:
                galerkin["projected_denominator"] = (
                    self._canonical_projected_denominator.metadata()
                )
                cache_key = self._canonical_projected_denominator_key
                if cache_key is not None:
                    galerkin["projected_denominator_location"] = {
                        "backend": cache_key[-2],
                        "device_id": cache_key[-1],
                    }
            if self._canonical_galerkin_final_gate is not None:
                galerkin["final_gate"] = dict(
                    self._canonical_galerkin_final_gate
                )
            metadata["canonical_rr_galerkin"] = galerkin
        for key, value in (
            ("active_space", self._active_space_metadata),
            ("ao_pair_provider", self._ao_pair_provider_metadata),
            ("direct_cholesky", self._direct_cholesky_metadata),
            (
                "mo_three_index_integrals",
                None if self._cd_integrals is None else self._cd_integrals.metadata(),
            ),
            (
                "mp2_pair_operator",
                None if self._mp2_operator is None else self._mp2_operator.metadata(),
            ),
            ("rr_projector_build", self._rr_projector_build_metadata),
            ("mp2_initial_guess", self._mp2_initial_metadata),
            (
                "rr_engine",
                None if self._rr_engine is None else self._rr_engine.metadata(),
            ),
        ):
            if value is not None:
                metadata[key] = value
        if self.rr_projector is not None:
            cached = self._rr_projector_build_metadata
            if cached is not None and "projector" in cached:
                metadata["rr_projector"] = cached["projector"]
            else:
                metadata["rr_projector"] = self.rr_projector.metadata(
                    transfer_counter=self.run_metrics.transfers
                )
        for name in (
            "projected_equation_residual",
            "full_space_residual",
            "projected_jacobi_update_norm",
            "full_space_jacobi_update_norm",
            "representation_residual",
        ):
            value = getattr(self, name)
            if value is not None:
                metadata[name] = value
        return metadata

    def _active_reference(self) -> tuple[np.ndarray, np.ndarray]:
        nocc, nmo = int(self.nocc), int(self.nmo)
        if getattr(self._scf, "xc", None):
            raise NotImplementedError(
                "the direct-CD RR engine requires an RHF reference, not KS-DFT"
            )
        active_coeff, _active_occ, mask = _active_restricted_orbitals(
            self.mo_coeff,
            self.mo_occ,
            self.get_frozen_mask(),
            nocc=nocc,
            nmo=nmo,
        )
        if float(getattr(self.mol, "omega", 0.0) or 0.0) != 0.0:
            raise NotImplementedError(
                "range-separated two-electron integrals are not yet part of "
                "the direct-CD RR contract"
            )
        with nvtx_range("rr/reference-fock"):
            with self.run_metrics.phase("rr_reference_fock"):
                dm = self._scf.make_rdm1(self.mo_coeff, self.mo_occ)
                vhf = self._scf.get_veff(self.mol, dm)
                fock_ao = np.asarray(self._scf.get_fock(vhf=vhf, dm=dm))
                if np.iscomplexobj(fock_ao):
                    raise NotImplementedError(
                        "the direct-CD RR engine supports a real RHF Fock matrix"
                    )
                if fock_ao.dtype != np.dtype(np.float64):
                    raise TypeError("the fp64 RRCCSD path requires a float64 Fock matrix")
                if fock_ao.shape != (active_coeff.shape[0],) * 2:
                    raise ValueError("AO Fock matrix does not match mo_coeff")
                if not np.all(np.isfinite(fock_ao)):
                    raise ValueError("AO Fock matrix contains NaN or infinite values")
                fock = reduce(
                    np.dot, (active_coeff.T, fock_ao, active_coeff)
                )
                orbital_energies = np.diag(fock).real.copy()
        nvir = nmo - nocc
        gaps = (
            orbital_energies[nocc:][None, :]
            - orbital_energies[:nocc, None]
        )
        if not np.all(np.isfinite(gaps)) or float(np.min(gaps)) <= 0.0:
            raise ValueError(
                "the active occupied-virtual Fock gaps must all be positive"
            )
        off_diagonal = fock - np.diag(np.diag(fock))
        self._active_space_metadata = {
            "nao": int(active_coeff.shape[0]),
            "nmo": nmo,
            "nocc": nocc,
            "nvir": nvir,
            "active_indices": np.flatnonzero(mask).tolist(),
            "frozen_indices": np.flatnonzero(~mask).tolist(),
            "maximum_fock_off_diagonal": float(np.max(np.abs(off_diagonal))),
            "minimum_occupied_virtual_gap": float(np.min(gaps)),
            "reference": "recomputed-PySCF-RHF-Fock",
        }
        return active_coeff, np.asarray(fock)

    def _build_cd_integrals(self) -> MOThreeIndexIntegralProvider:
        self._clear_cd_runtime()
        active_coeff, fock_host = self._active_reference()
        with nvtx_range("rr/gint-pair-provider"):
            with self.run_metrics.phase("gint_pair_provider"):
                if self.gint_column_backend == "selected":
                    provider = GINTSelectedAOPairColumnProvider(
                        self.mol,
                        direct_scf_tol=self.direct_scf_tol,
                        group_size=self.gint_group_size,
                        max_batch_size=self.gint_max_batch_size,
                        column_kernel=self.gint_column_kernel,
                        transfer_counter=self.run_metrics.transfers,
                        runtime_gate_receipt=self.gint_runtime_gate_receipt,
                        runtime_execution_mode=self.gint_runtime_execution_mode,
                        runtime_source_digest=self.gint_runtime_source_digest,
                        runtime_source_root=self.gint_runtime_source_root,
                        runtime_manifest_path=self.gint_runtime_manifest_path,
                        runtime_release_topology_path=(
                            self.gint_runtime_topology_path
                        ),
                    )
                    column_batch_size = self.gint_max_batch_size
                    integral_source = "gint-selected-pair-direct-cd"
                else:
                    provider = GINTAOPairColumnProvider(
                        self.mol,
                        direct_scf_tol=self.direct_scf_tol,
                        group_size=self.gint_group_size,
                        max_block_bytes=self.gint_max_block_bytes,
                        transfer_counter=self.run_metrics.transfers,
                    )
                    column_batch_size = 1
                    integral_source = "gint-restricted-pair-direct-cd-reference"
        if provider.materializes_pair_matrix:
            raise RuntimeError("the direct-CD path refuses a dense AO-pair provider")
        rank_cap = (
            None if self.cd_max_rank is None
            else min(self.cd_max_rank, int(provider.dimension))
        )
        with nvtx_range("rr/direct-pair-cd"):
            with self.run_metrics.phase("direct_pair_cholesky"):
                cholesky = pivoted_cholesky_from_columns(
                    provider,
                    threshold=self.eri_tol,
                    max_rank=rank_cap,
                    transfer_counter=self.run_metrics.transfers,
                    column_batch_size=column_batch_size,
                )
                _synchronize(cholesky.pair_factors)
        if cholesky.capped:
            raise RuntimeError(
                "direct AO-pair Cholesky reached cd_max_rank before eri_tol: "
                f"residual {cholesky.residual_diagonal:.3e} > "
                f"{self.eri_tol:.3e}"
            )
        if cholesky.rank < 1:
            raise RuntimeError("direct AO-pair Cholesky produced zero factors")
        self._direct_cholesky_metadata = {
            **cholesky.metadata(),
            "requested_max_rank": self.cd_max_rank,
            "effective_max_rank": rank_cap,
        }
        self._ao_pair_provider_metadata = provider.metadata()

        xp = _array_module(cholesky.pair_factors)
        pair_i, pair_j = provider.pair_indices_device
        with nvtx_range("rr/unpack-cd-pairs"):
            with self.run_metrics.phase("unpack_cd_pairs"):
                ao_factors = cholesky.unpack_symmetric(
                    int(active_coeff.shape[0]), pair_i, pair_j
                )
                _synchronize(ao_factors)
        coefficient = _resident_array(
            active_coeff,
            xp,
            self.run_metrics.transfers,
            operation="rr_active_mo_coeff",
        )
        with nvtx_range("rr/cd-ao-to-mo"):
            with self.run_metrics.phase("cd_ao_to_mo"):
                integrals = MOThreeIndexIntegralProvider.from_ao_factors(
                    ao_factors,
                    coefficient,
                    int(self.nocc),
                    factorization="cd",
                    threshold=self.eri_tol,
                    source=integral_source,
                    auxiliary_block_size=min(
                        self.cd_mo_block_size, cholesky.rank
                    ),
                )
                _synchronize(integrals.factors)
        if integrals.xp is np:
            raise RuntimeError("the GINT direct-CD factors unexpectedly left the GPU")
        if integrals.dtype != np.dtype(np.float64):
            raise TypeError("the fp64 RRCCSD path requires float64 CD factors")
        self._rr_fock = _resident_array(
            fock_host,
            xp,
            self.run_metrics.transfers,
            operation="rr_active_fock",
        )
        self._rr_orbital_energies = _resident_array(
            np.diag(fock_host).real.copy(),
            xp,
            self.run_metrics.transfers,
            operation="rr_active_orbital_energies",
        )
        self._cd_integrals = integrals
        return integrals

    def ao2mo(self, mo_coeff: Any = None):
        if self.eri_backend != "cd":
            return super().ao2mo(mo_coeff)
        if mo_coeff is not None and mo_coeff is not self.mo_coeff:
            raise NotImplementedError(
                "the direct-CD path does not accept an alternate mo_coeff; "
                "set RRCCSD.mo_coeff before building the active-space factors"
            )
        return self._build_cd_integrals()

    def _build_projector(self, mp2_t2: Any) -> RRProjector:
        """Dense canonical validation projector."""

        with nvtx_range("rr/projector"):
            with self.run_metrics.phase("rr_projector"):
                projector = build_rr_projector(
                    mp2_t2,
                    self.rr_eig_cutoff,
                    max_rank=self.rr_max_rank,
                )
        if projector.rank == 0:
            raise ValueError(
                "rr_eig_cutoff removed the complete MP2 doubles space; "
                "use a tighter cutoff"
            )
        error = projector.orthogonality_error(
            transfer_counter=self.run_metrics.transfers
        )
        if error > 1e-10:
            raise RuntimeError(
                f"RR projector orthogonality error {error:.3e} exceeds 1e-10"
            )
        self.rr_projector = projector
        self._canonical_projected_denominator = None
        self._canonical_projected_denominator_key = None
        self._canonical_orbital_energy_differences = None
        self._canonical_galerkin_final_gate = None
        return projector

    def _build_cd_projector(
        self, integrals: MOThreeIndexIntegralProvider
    ) -> tuple[MP2PairOperator, RRProjector]:
        if integrals is not self._cd_integrals:
            raise ValueError("CD integrals do not belong to this RRCCSD lifecycle")
        if self._rr_fock is None or self._rr_orbital_energies is None:
            raise RuntimeError("the active Fock reference has not been prepared")
        nocc, nvir = integrals.nocc, integrals.nvir
        pair_dimension = nocc * nvir
        denominator_cap = (
            None if self.denominator_max_rank is None
            else min(self.denominator_max_rank, pair_dimension)
        )
        with nvtx_range("rr/mp2-pair-operator"):
            with self.run_metrics.phase("mp2_pair_operator"):
                operator = MP2PairOperator(
                    integrals.L_ov,
                    self._rr_orbital_energies[:nocc],
                    self._rr_orbital_energies[nocc:],
                    denominator_tolerance=self.denominator_tolerance,
                    denominator_max_rank=denominator_cap,
                    transfer_counter=self.run_metrics.transfers,
                )
        if operator.denominator.capped:
            raise RuntimeError(
                "the Cauchy denominator factorization reached "
                "denominator_max_rank before denominator_tolerance: residual "
                f"{operator.denominator.residual_diagonal:.3e} > "
                f"{self.denominator_tolerance:.3e}"
            )
        projector_cap = (
            None if self.rr_max_rank is None
            else min(self.rr_max_rank, pair_dimension)
        )
        if self.rr_eig_cutoff == 0.0 and (
            pair_dimension > self.rr_dense_fallback_dimension
            or (projector_cap is not None and projector_cap < pair_dimension)
        ):
            raise NotImplementedError(
                "rr_eig_cutoff=0 requires the explicit small-system full-rank "
                "endpoint; use a nonzero cutoff for matrix-free Lanczos"
            )
        with nvtx_range("rr/projector"):
            with self.run_metrics.phase("rr_projector"):
                build = build_rr_projector_lanczos(
                    operator,
                    rr_eig_cutoff=self.rr_eig_cutoff,
                    initial_rank=min(self.rr_initial_rank, pair_dimension),
                    max_rank=projector_cap,
                    solver_tolerance=self.rr_solver_tolerance,
                    maxiter=self.rr_solver_maxiter,
                    dense_fallback_dimension=self.rr_dense_fallback_dimension,
                    ritz_residual_tolerance=self.rr_ritz_residual_tolerance,
                    allow_incomplete=False,
                    allow_unconverged=False,
                )
        projector = build.projector
        if projector.rank == 0:
            raise ValueError(
                "rr_eig_cutoff removed the complete MP2 doubles space; "
                "use a tighter cutoff"
            )
        build_metadata = build.metadata(
            transfer_counter=self.run_metrics.transfers
        )
        orthogonality_error = float(
            build_metadata["projector"]["orthogonality_error"]
        )
        if orthogonality_error > 1e-10:
            raise RuntimeError(
                "RR projector orthogonality error "
                f"{orthogonality_error:.3e} exceeds 1e-10"
            )
        build_metadata["requested_max_rank"] = self.rr_max_rank
        build_metadata["effective_max_rank"] = projector_cap
        build_metadata["denominator_requested_max_rank"] = (
            self.denominator_max_rank
        )
        build_metadata["denominator_effective_max_rank"] = denominator_cap
        self._mp2_operator = operator
        self.rr_projector = projector
        self._canonical_projected_denominator = None
        self._canonical_projected_denominator_key = None
        self._canonical_orbital_energy_differences = None
        self._canonical_galerkin_final_gate = None
        self._rr_projector_build_metadata = build_metadata
        return operator, projector

    def _cd_initial_amplitudes(
        self, integrals: MOThreeIndexIntegralProvider
    ) -> tuple[float, Any, RRDoubles]:
        if self._cd_initial_t1 is not None and self._cd_initial_doubles is not None:
            return self.emp2, self._cd_initial_t1, self._cd_initial_doubles
        if self._mp2_operator is None or self.rr_projector is None:
            operator, projector = self._build_cd_projector(integrals)
        else:
            operator, projector = self._mp2_operator, self.rr_projector
        xp = integrals.xp
        nocc, nvir = integrals.nocc, integrals.nvir
        with nvtx_range("rr/compressed-mp2-initial-guess"):
            with self.run_metrics.phase("mp2_initial_amplitudes"):
                applied = operator.matmat(projector.vectors)
                core = projector.vectors.T.conj() @ applied
                core = (core + core.T.conj()) * 0.5
                eia = (
                    self._rr_orbital_energies[:nocc, None]
                    - self._rr_orbital_energies[None, nocc:]
                )
                t1 = self._rr_fock[:nocc, nocc:] / eia
                doubles = RRDoubles(projector, core, nocc, nvir)
                zeros = xp.zeros_like(t1)
                emp2_device = rr_ccsd_energy(
                    doubles,
                    zeros,
                    self._rr_fock[:nocc, nocc:],
                    integrals.L_ov,
                    auxiliary_block_size=min(
                        self.rr_auxiliary_block_size, integrals.naux
                    ),
                ).value
                _synchronize(emp2_device)
        if xp is not np:
            self.run_metrics.transfers.record_d2h(
                int(emp2_device.nbytes), operation="rr_mp2_energy"
            )
        emp2 = float(emp2_device.item())
        if not math.isfinite(emp2):
            raise FloatingPointError("compressed MP2 initial energy is not finite")
        self.emp2 = emp2
        self._cd_initial_t1 = t1
        self._cd_initial_doubles = doubles
        self._mp2_initial_metadata = {
            "representation": "rr-doubles",
            "source": "projected-direct-mp2-pair-operator",
            "rank": projector.rank,
            "pair_dimension": projector.full_dimension,
            "emp2": emp2,
            "dense_t2_materialized": False,
            "mp2_pair_matrix_materialized": (
                self._rr_projector_build_metadata["solver"].endswith(
                    "dense-validation"
                )
            ),
            "denominator": operator.denominator.metadata(),
        }
        return emp2, t1, doubles

    def init_amps(self, eris: Any = None):
        if self.eri_backend != "cd":
            return super().init_amps(eris)
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        if eris is not self._cd_integrals:
            raise ValueError(
                "the direct-CD initial guess requires factors built by this "
                "RRCCSD instance"
            )
        return self._cd_initial_amplitudes(eris)

    def _compress_doubles(self, t2: Any) -> RRDoubles:
        if self.rr_projector is None:
            raise RuntimeError("RR projector has not been initialized")
        return RRDoubles.from_t2(t2, self.rr_projector)

    def _store_doubles(self, doubles: RRDoubles) -> None:
        self.doubles = doubles

    def _project_doubles(self, t2: Any):
        with nvtx_range("rr/project-doubles"):
            with self.run_metrics.phase("rr_project_doubles"):
                doubles = self._compress_doubles(t2)
                dense = doubles.reconstruct_t2()
        self._store_doubles(doubles)
        return dense

    def _dense_update(self, t1: Any, t2: Any, eris: Any):
        """Canonical validation seam; never called by the direct-CD engine."""

        return ccsd_incore.update_amps(self, t1, t2, eris)

    def _canonical_galerkin_denominator(
        self,
        t2: Any,
        eris: Any,
    ) -> tuple[Any, ProjectedPairDenominator]:
        """Return the fixed RR Lyapunov operator for a canonical run."""

        if self.rr_projector is None:
            raise RuntimeError("RR projector has not been initialized")
        nocc, nvir = int(t2.shape[0]), int(t2.shape[2])
        xp = _array_module(t2)
        location = _array_backend_device_identity(t2)
        if _array_backend_device_identity(self.rr_projector.vectors) != location:
            raise TypeError(
                "canonical RR amplitudes and projector need one backend and device"
            )
        cache_key = (
            id(self.rr_projector),
            id(eris),
            nocc,
            nvir,
            float(self.level_shift),
            location[0],
            location[1],
        )
        if (
            self._canonical_projected_denominator is not None
            and self._canonical_orbital_energy_differences is not None
            and self._canonical_projected_denominator_key == cache_key
        ):
            cached_locations = (
                _array_backend_device_identity(
                    self._canonical_projected_denominator.matrix
                ),
                _array_backend_device_identity(
                    self._canonical_orbital_energy_differences
                ),
            )
            if any(cached != location for cached in cached_locations):
                raise RuntimeError(
                    "canonical projected denominator cache changed array location"
                )
            return (
                self._canonical_orbital_energy_differences,
                self._canonical_projected_denominator,
            )

        with _array_device_scope(t2):
            mo_energy = eris.mo_energy
            if xp is np:
                if _array_module(mo_energy) is not np:
                    raise TypeError(
                        "refusing an implicit device-to-host orbital-energy transfer"
                    )
                mo_energy = np.asarray(mo_energy)
            elif _array_module(mo_energy) is xp:
                if _array_backend_device_identity(mo_energy) != location:
                    raise TypeError(
                        "canonical RR orbital energies are on another CUDA device"
                    )
            else:
                mo_energy = _resident_array(
                    mo_energy,
                    xp,
                    self.run_metrics.transfers,
                    operation="canonical_rr_orbital_energies",
                )
            if getattr(mo_energy, "shape", None) is None or len(mo_energy) < (
                nocc + nvir
            ):
                raise ValueError(
                    "canonical ERI orbital energies do not match amplitudes"
                )
            occupied = mo_energy[:nocc]
            virtual = mo_energy[nocc:nocc + nvir]
            eia = (
                occupied[:, None]
                - virtual[None, :]
                - float(self.level_shift)
            )
            self._canonical_projected_denominator = (
                ProjectedPairDenominator.build(
                    self.rr_projector,
                    eia,
                    nocc,
                    nvir,
                    transfer_counter=self.run_metrics.transfers,
                )
            )
            self._canonical_projected_denominator_key = cache_key
            self._canonical_orbital_energy_differences = eia
            self.run_metrics.increment(
                "canonical_rr_projected_denominator_builds"
            )
            return eia, self._canonical_projected_denominator

    def update_amps(self, t1: Any, t2: Any, eris: Any):
        if self.eri_backend == "cd":
            if eris is not self._cd_integrals or self._rr_engine is None:
                raise RuntimeError(
                    "direct-CD update_amps requires the initialized RR engine"
                )
            if not isinstance(t2, RRDoubles):
                raise TypeError("direct-CD update_amps requires RRDoubles")
            result = self._rr_engine.jacobi(t1, t2)
            return result.t1, result.doubles
        with nvtx_range("rr/dense-residual-validation"):
            with self.run_metrics.phase("dense_residual_validation"):
                t1new, t2new = self._dense_update(t1, t2, eris)
        with _array_device_scope(t2):
            eia, denominator = self._canonical_galerkin_denominator(t2, eris)
            with nvtx_range("rr/canonical-galerkin-solve"):
                with self.run_metrics.phase("canonical_rr_galerkin_solve"):
                    updated, cached = _canonical_rr_galerkin_step(
                        t2,
                        t2new,
                        self.rr_projector,
                        eia,
                        denominator=denominator,
                        transfer_counter=self.run_metrics.transfers,
                    )
            if cached is not denominator:  # pragma: no cover - private invariant
                raise AssertionError(
                    "canonical projected denominator cache was replaced"
                )
            self.run_metrics.increment("canonical_rr_galerkin_updates")
            # Retain the representation hook used by the canonical THC
            # validation subclass.  For RRCCSD this is idempotent.
            projected = self._project_doubles(updated.reconstruct_t2())
        return t1new, projected

    def energy(self, t1: Any = None, t2: Any = None, eris: Any = None):
        if self.eri_backend != "cd":
            return super().energy(t1, t2, eris)
        if eris is None:
            eris = self._cd_integrals
        if eris is None or eris is not self._cd_integrals:
            raise RuntimeError("direct-CD energy requires this solver's factors")
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        if not isinstance(t2, RRDoubles):
            raise TypeError("direct-CD energy requires RRDoubles")
        result = rr_ccsd_energy(
            t2,
            t1,
            self._rr_fock[:eris.nocc, eris.nocc:],
            eris.L_ov,
            auxiliary_block_size=min(
                self.rr_auxiliary_block_size, eris.naux
            ),
        ).value
        _synchronize(result)
        if eris.xp is not np:
            self.run_metrics.transfers.record_d2h(
                int(result.nbytes), operation="rr_energy"
            )
        return float(result.item())

    def _validate_cd_lifecycle(self) -> None:
        if (
            type(self)._compress_doubles is not RRCCSD._compress_doubles
            or type(self)._store_doubles is not RRCCSD._store_doubles
        ):
            raise NotImplementedError(
                "the direct-CD engine currently supports the RR doubles "
                "representation only"
            )
        if bool(getattr(self, "cc2", False)):
            raise NotImplementedError("CC2 is not implemented by the RR/CD engine")
        if getattr(self, "callback", None) is not None:
            raise NotImplementedError(
                "per-iteration callbacks are not yet implemented by the RR/CD engine"
            )
        if float(getattr(self, "iterative_damping", 1.0)) != 1.0:
            raise NotImplementedError(
                "iterative damping is not yet implemented by the RR/CD engine"
            )
        if getattr(self, "diis_file", None) is not None:
            raise NotImplementedError(
                "file-backed DIIS is not supported; compressed DIIS stays resident"
            )
        if not isinstance(getattr(self, "diis", True), (bool, np.bool_)):
            raise NotImplementedError(
                "an external DIIS object cannot consume compressed RR amplitudes"
            )

    def _ccsd_cd(self, t1: Any = None, t2: Any = None, eris: Any = None):
        self._validate_cd_lifecycle()
        if hasattr(self, "check_sanity"):
            self.check_sanity()
        self.dump_flags()
        self.e_hf = self.get_e_hf()
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        if eris is not self._cd_integrals:
            raise ValueError(
                "the direct-CD kernel requires factors built by this RRCCSD instance"
            )

        if self.rr_projector is None or self._mp2_operator is None:
            _emp2, default_t1, default_t2 = self.init_amps(eris)
        else:
            default_t1 = default_t2 = None
        if t1 is None:
            if default_t1 is None:
                _emp2, default_t1, rebuilt_t2 = self._cd_initial_amplitudes(eris)
                if t2 is None:
                    default_t2 = rebuilt_t2
            t1 = default_t1
        else:
            t1 = _resident_array(
                t1,
                eris.xp,
                self.run_metrics.transfers,
                operation="rr_user_initial_t1",
            )
        if t2 is None:
            if default_t2 is None:
                _emp2, rebuilt_t1, default_t2 = self._cd_initial_amplitudes(eris)
                if t1 is None:
                    t1 = rebuilt_t1
            t2 = default_t2
        if not isinstance(t2, RRDoubles):
            raise TypeError(
                "the direct-CD kernel accepts only compressed RRDoubles; "
                "use init_amps() to obtain the lifecycle-compatible guess"
            )
        if t2.projector is not self.rr_projector:
            raise ValueError("initial RRDoubles use a different RR projector")

        with nvtx_range("rr/engine-setup"):
            with self.run_metrics.phase("rr_engine_setup"):
                engine = RRCCSDIterationEngine(
                    eris,
                    self.rr_projector,
                    self._rr_fock,
                    self._rr_orbital_energies,
                    level_shift=float(self.level_shift),
                    auxiliary_block_size=min(
                        self.rr_auxiliary_block_size, eris.naux
                    ),
                    virtual_block_size=min(
                        self.rr_virtual_block_size, eris.nvir
                    ),
                    rr_ring_kernel=self.rr_ring_kernel,
                    metrics=self.run_metrics,
                )
                _synchronize(eris.factors)
        self._rr_engine = engine
        diis_enabled = bool(self.diis)
        diis_start_cycle = (
            int(self.diis_start_cycle)
            if diis_enabled else int(self.max_cycle) + 1
        )
        with nvtx_range("rr/ccsd-iterations"):
            with self.run_metrics.phase("ccsd_iterations"):
                _synchronize(eris.factors)
                result = engine.kernel(
                    t1,
                    t2,
                    max_cycle=int(self.max_cycle),
                    conv_tol=float(self.conv_tol),
                    conv_tol_normt=float(self.conv_tol_normt),
                    diis_space=int(self.diis_space),
                    diis_start_cycle=diis_start_cycle,
                )
                _synchronize(result.doubles.core)

        self.converged = bool(result.converged)
        self.cycles = int(result.cycles)
        self.e_corr = float(result.energy)
        self.t1 = result.t1
        self.t2 = result.doubles
        self.projected_equation_residual = float(result.residual_norm)
        self.projected_jacobi_update_norm = float(result.update_norm)
        self.full_space_residual = None
        self.full_space_jacobi_update_norm = None
        self.full_space_diagnostic_metadata = None
        self.representation_residual = None
        self.rr_history = result.history
        self._store_doubles(result.doubles)
        self.run_metrics.metadata = self.method_metadata()
        self._finalize()
        return self.e_corr, self.t1, result.doubles

    def run_full_space_residual_diagnostic(
        self,
    ) -> FullSpaceResidualDiagnostic:
        """Reconstruct and check the complete RR/CD equation once.

        This is an explicit post-convergence diagnostic.  ``kernel()`` never
        calls it, because it materializes identity and amplitude matrices in
        the complete ``OV`` pair space.  Callers that report performance must
        therefore time it separately from the normal post-HF interval.
        """

        if self.eri_backend != "cd":
            raise NotImplementedError(
                "the explicit RR full-space diagnostic is for direct CD"
            )
        if getattr(self, "converged", None) is not True:
            raise RuntimeError(
                "run a converged RRCCSD.kernel() before the full-space diagnostic"
            )
        if (
            self._cd_integrals is None
            or self._rr_fock is None
            or self._rr_orbital_energies is None
            or self.t1 is None
            or not isinstance(self.doubles, RRDoubles)
        ):
            raise RuntimeError(
                "run RRCCSD.kernel() before the full-space diagnostic"
            )

        transfer_before = self.run_metrics.transfers.to_dict()
        memory_before = _diagnostic_memory_snapshot(self._cd_integrals.factors)
        with nvtx_range("rr/post-convergence-full-space-residual"):
            with self.run_metrics.phase(
                "post_convergence_full_space_residual_diagnostic",
                metadata={"included_in_normal_iteration_timing": False},
            ):
                diagnostic = diagnose_reconstructed_full_space_residual(
                    self._cd_integrals,
                    self.t1,
                    self.doubles,
                    self._rr_fock,
                    self._rr_orbital_energies,
                    level_shift=float(self.level_shift),
                    auxiliary_block_size=min(
                        self.rr_auxiliary_block_size,
                        self._cd_integrals.naux,
                    ),
                    virtual_block_size=min(
                        self.rr_virtual_block_size,
                        self._cd_integrals.nvir,
                    ),
                    transfer_counter=self.run_metrics.transfers,
                )
                _synchronize(self._cd_integrals.factors)
        memory_after = _diagnostic_memory_snapshot(self._cd_integrals.factors)
        transfer_after = self.run_metrics.transfers.to_dict()

        self.full_space_residual = float(diagnostic.residual_norm)
        self._dense_t2_reconstructed = True
        metadata = diagnostic.metadata()
        pair_matrix_nbytes = (
            int(diagnostic.pair_dimension) ** 2
            * int(np.dtype(self.doubles.core.dtype).itemsize)
        )
        metadata.update({
            "kind": "equation-residual",
            "space": "full-active-pair",
            "is_full_space": True,
            "normal_kernel_invokes_diagnostic": False,
            "phase_name": (
                "post_convergence_full_space_residual_diagnostic"
            ),
            "transfer_accounting": {
                "before": transfer_before,
                "after": transfer_after,
                "delta": _transfer_ledger_delta(
                    transfer_before, transfer_after
                ),
            },
            "resource_accounting": {
                "pair_matrix_nbytes": pair_matrix_nbytes,
                "largest_single_materialized_array_nbytes": int(
                    diagnostic.largest_materialized_nbytes
                ),
                "device_memory_before": memory_before,
                "device_memory_after": memory_after,
                "process_peak_memory_not_measured_at_solver_level": True,
            },
        })
        self.full_space_diagnostic_metadata = metadata
        self.run_metrics.metadata = self.method_metadata()
        return diagnostic

    def ccsd(self, t1: Any = None, t2: Any = None, eris: Any = None):
        if self.eri_backend == "cd":
            return self._ccsd_cd(t1, t2, eris)
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        with self.run_metrics.phase("mp2_initial_amplitudes"):
            _emp2, mp2_t1, mp2_t2 = super().init_amps(eris)
        self._build_projector(mp2_t2)

        if t1 is None:
            t1 = mp2_t1
        if t2 is None:
            t2 = mp2_t2
        t2 = self._project_doubles(t2)

        with self.run_metrics.phase("ccsd_iterations"):
            e_corr, final_t1, final_t2 = super().ccsd(t1, t2, eris)
        final_doubles = self._compress_doubles(final_t2)
        reconstructed = final_doubles.reconstruct_t2()
        self._dense_t2_reconstructed = True
        norm = np.linalg.norm(final_t2)
        error = np.linalg.norm(reconstructed - final_t2)
        self.representation_residual = (
            float(error) if norm == 0.0 else float(error / norm)
        )

        with nvtx_range("rr/final-full-space-residual"):
            with self.run_metrics.phase("final_full_space_residual"):
                check_t1, check_t2 = self._dense_update(
                    final_t1, reconstructed, eris
                )
                nocc, nvir = final_t1.shape
                residual = evaluate_dense_ccsd_residual(
                    final_t1,
                    reconstructed,
                    check_t1,
                    check_t2,
                    eris.mo_energy[:nocc],
                    eris.mo_energy[nocc:nocc + nvir],
                    level_shift=float(self.level_shift),
                    projector_vectors=self.rr_projector.vectors,
                )
                self.projected_equation_residual = (
                    residual.projected_equation_norm
                )
                self.full_space_residual = residual.full_space_equation_norm
                self.projected_jacobi_update_norm = (
                    residual.projected_jacobi_update_norm
                )
                self.full_space_jacobi_update_norm = (
                    residual.full_space_jacobi_update_norm
                )
        self.converged, self._canonical_galerkin_final_gate = (
            _canonical_rr_convergence_gate(
                bool(getattr(self, "converged", False)),
                self.projected_equation_residual,
                self.conv_tol_normt,
            )
        )
        self._store_doubles(final_doubles)
        self.run_metrics.metadata = self.method_metadata()
        return e_corr, final_t1, final_doubles

    def reconstruct_t2(self):
        if self.doubles is None:
            raise RuntimeError("RRCCSD has not produced doubles amplitudes")
        self._dense_t2_reconstructed = True
        return self.doubles.reconstruct_t2()

    def experiment_record(self) -> Dict[str, Any]:
        # Build metadata first because projector/doubles metadata performs
        # explicit, counted scalar/vector downloads on a GPU.  Snapshotting the
        # ledger before those calls would make the experiment record internally
        # inconsistent.
        method = self.method_metadata()
        doubles_metadata = None
        if self.doubles is not None:
            doubles_metadata = self.doubles.metadata(
                transfer_counter=self.run_metrics.transfers
            )
        record = self.run_metrics.to_dict()
        record["method"] = method
        if doubles_metadata is not None:
            record["doubles"] = doubles_metadata
        record["converged"] = bool(getattr(self, "converged", False))
        record["e_corr"] = (
            None if getattr(self, "e_corr", None) is None else float(self.e_corr)
        )
        record["cycles"] = int(getattr(self, "cycles", 0))
        record["history"] = [dict(row) for row in self.rr_history]
        return record

    @property
    def full_space_projection_residual(self) -> Optional[float]:
        """Compatibility alias for the full-space equation diagnostic."""

        return self.full_space_residual


__all__ = ["RRCCSD", "RRDoubles", "RRProjector"]
