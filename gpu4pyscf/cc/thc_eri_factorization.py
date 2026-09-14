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

"""Weighted CP factorization of occupied-virtual Cholesky factors.

Hohenstein *et al.*, J. Chem. Phys. **156**, 054102 (2022), Eqs. 17--21,
obtain the only ERI-THC class needed by their quartic THC-CCSD equations from

``L[A,i,a] ~= sum(I) xi[A,I] x_occ[i,I] x_vir[a,I]``.

The ERI core is then ``Z = xi.T @ xi``.  This module implements that
preprocessing endpoint without connecting it to the iterative solver.  The
orbital weights are explicit caller inputs; production callers are responsible
for deriving them from MP2 natural-orbital occupation changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gpu4pyscf.cc.thc_eri import ERITHCFactors


def _array_module(value: Any):
    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy" and isinstance(value, np.ndarray):
        return np
    if module == "cupy":
        import cupy

        if isinstance(value, cupy.ndarray):
            return cupy
    raise TypeError("ERI-THC fit tensors must be NumPy or CuPy arrays")


def _same_backend(reference: Any, *values: Any) -> Any:
    xp = _array_module(reference)
    if any(_array_module(value) is not xp for value in values):
        raise TypeError("ERI-THC fit tensors must use one array backend")
    if xp is not np:
        device_id = int(reference.device.id)
        if any(int(value.device.id) != device_id for value in values):
            raise TypeError("ERI-THC fit CuPy tensors must use one device")
    return xp


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _read_scalar(
    value: Any,
    *,
    xp: Any,
    transfer_counter: Any,
    operation: str,
) -> float:
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU ERI-THC fitting requires an explicit transfer_counter"
            )
        if not callable(getattr(transfer_counter, "record_d2h", None)):
            raise TypeError("transfer_counter must provide record_d2h")
        transfer_counter.record_d2h(int(value.nbytes), operation=operation)
    return float(value.item()) if hasattr(value, "item") else float(value)


def _right_solve(mttkrp: Any, gram: Any, ridge: float, xp: Any):
    regularized = gram + ridge * xp.eye(gram.shape[0], dtype=gram.dtype)
    return xp.linalg.solve(regularized.T, mttkrp.T).T


def _normalize_and_absorb(factor: Any, target: Any, xp: Any) -> None:
    tiny = np.finfo(np.dtype(factor.dtype)).tiny
    norms = xp.maximum(xp.linalg.norm(factor, axis=0), tiny)
    factor /= norms[None, :]
    target *= norms[None, :]


def _cp_cholesky(x_occ: Any, x_vir: Any, xi: Any, xp: Any):
    return xp.einsum("iI,aI,AI->Aia", x_occ, x_vir, xi)


def _blocked_residual_norm(
    lov: Any,
    *,
    x_occ: Any = None,
    x_vir: Any = None,
    xi: Any = None,
    weight_product: Any = None,
    auxiliary_block_size: int,
    xp: Any,
):
    """Return a norm without materializing a second full ``(A,O,V)`` array."""

    total = xp.zeros((), dtype=lov.dtype)
    for start in range(0, int(lov.shape[0]), auxiliary_block_size):
        stop = min(start + auxiliary_block_size, int(lov.shape[0]))
        if xi is None:
            block = lov[start:stop].copy()
        else:
            block = _cp_cholesky(x_occ, x_vir, xi[start:stop], xp)
            block -= lov[start:stop]
        if weight_product is not None:
            block *= weight_product
        total += xp.vdot(block, block).real
    return xp.sqrt(total)


def _validate_inputs(
    lov: Any,
    occupied_weights: Any,
    virtual_weights: Any,
    *,
    transfer_counter: Any,
) -> tuple[Any, int, int, int]:
    xp = _same_backend(lov, occupied_weights, virtual_weights)
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU ERI-THC fitting requires an explicit transfer_counter"
            )
        if not callable(getattr(transfer_counter, "record_d2h", None)) or not (
            callable(getattr(transfer_counter, "record_h2d", None))
        ):
            raise TypeError(
                "transfer_counter must provide record_d2h and record_h2d"
            )
    if getattr(lov, "ndim", None) != 3:
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    naux, nocc, nvir = map(int, lov.shape)
    if min(naux, nocc, nvir) < 1:
        raise ValueError("lov dimensions must be positive")
    if occupied_weights.shape != (nocc,):
        raise ValueError("occupied_weights must have shape (nocc,)")
    if virtual_weights.shape != (nvir,):
        raise ValueError("virtual_weights must have shape (nvir,)")
    dtypes = tuple(
        np.dtype(value.dtype)
        for value in (lov, occupied_weights, virtual_weights)
    )
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError("lov and orbital weights must have the same dtype")
    if np.issubdtype(dtypes[0], np.complexfloating):
        raise NotImplementedError("ERI-THC fitting currently requires real RHF")
    if not np.issubdtype(dtypes[0], np.floating):
        raise TypeError("ERI-THC fitting requires floating-point tensors")
    finite = (
        xp.all(xp.isfinite(lov))
        & xp.all(xp.isfinite(occupied_weights))
        & xp.all(xp.isfinite(virtual_weights))
    )
    minimum_weight = xp.minimum(
        xp.min(occupied_weights), xp.min(virtual_weights)
    )
    if not bool(_read_scalar(
        finite,
        xp=xp,
        transfer_counter=transfer_counter,
        operation="eri_thc_fit_input_finite",
    )):
        raise ValueError("ERI-THC fit inputs must be finite")
    if _read_scalar(
        minimum_weight,
        xp=xp,
        transfer_counter=transfer_counter,
        operation="eri_thc_fit_minimum_weight",
    ) <= 0.0:
        raise ValueError(
            "orbital weights must be positive; apply and record an explicit "
            "weight floor before fitting"
        )
    return xp, naux, nocc, nvir


@dataclass(frozen=True)
class ERITHCFitResult:
    """Auditable result of the Eq. 21 weighted CP fit."""

    factors: ERITHCFactors
    xi: Any
    occupied_weights: Any
    virtual_weights: Any
    fit_tolerance: float
    weighted_fit_residual: float
    unweighted_fit_residual: float
    iterations: int
    converged: bool
    ridge: float
    seed: int
    host_scalar_reads: int
    auxiliary_block_size: int
    largest_intermediate_nbytes: int
    allow_unconverged: bool
    exact_full_pair_endpoint: bool = False

    @property
    def rank(self) -> int:
        return self.factors.rank

    @property
    def storage_nbytes(self) -> int:
        return int(self.factors.storage_nbytes + self.xi.nbytes)

    def reconstruct_cholesky(self):
        xp = _array_module(self.xi)
        return _cp_cholesky(
            self.factors.x_occ, self.factors.x_vir, self.xi, xp
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "factorization": "weighted-cholesky-cp-als",
            "paper": "Hohenstein-2022",
            "paper_source": "arXiv:2111.11473v1",
            "eri_equation": 17,
            "least_squares_equations": [19, 20, 21],
            "orbital_weights": "explicit-caller-supplied",
            "production_weight_definition": (
                "sqrt(abs(MP2-natural-occupation-minus-SCF-occupation))"
            ),
            "rank": self.rank,
            "fit_tolerance": float(self.fit_tolerance),
            "weighted_fit_residual": float(self.weighted_fit_residual),
            "unweighted_fit_residual": float(self.unweighted_fit_residual),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "ridge": float(self.ridge),
            "seed": int(self.seed),
            "host_scalar_reads": int(self.host_scalar_reads),
            "auxiliary_block_size": int(self.auxiliary_block_size),
            "largest_intermediate_nbytes": int(
                self.largest_intermediate_nbytes
            ),
            "largest_intermediate_scope": "largest-single-logical-array",
            "actual_peak_memory_requires_profiling": True,
            "materializes_full_cholesky_during_fit": False,
            "explicit_reconstruction_materializes_full_cholesky": True,
            "storage_nbytes": self.storage_nbytes,
            "storage_includes_cholesky_cp_factor": True,
            "weights_borrowed_not_counted_in_storage": True,
            "exact_full_pair_endpoint": bool(self.exact_full_pair_endpoint),
            "allow_unconverged": bool(self.allow_unconverged),
            "audit_only": True,
            "production_enabled": False,
            "fit_time_included_in_benchmark_contract": True,
        }


def exact_eri_thc_from_cholesky(
    lov: Any,
    occupied_weights: Any,
    virtual_weights: Any,
    *,
    transfer_counter: Any = None,
) -> ERITHCFitResult:
    """Return the analytic pair-basis Eq. 17 endpoint for small systems."""

    xp, naux, nocc, nvir = _validate_inputs(
        lov,
        occupied_weights,
        virtual_weights,
        transfer_counter=transfer_counter,
    )
    rank = nocc * nvir
    dtype = lov.dtype
    pair = xp.arange(rank)
    x_occ = (
        xp.arange(nocc)[:, None] == pair[None, :] // nvir
    ).astype(dtype)
    x_vir = (
        xp.arange(nvir)[:, None] == pair[None, :] % nvir
    ).astype(dtype)
    xi = lov.reshape(naux, rank).copy()
    core = xi.T @ xi
    factors = ERITHCFactors(
        x_occ,
        x_vir,
        core,
        transfer_counter=transfer_counter,
    )
    return ERITHCFitResult(
        factors=factors,
        xi=xi,
        occupied_weights=occupied_weights,
        virtual_weights=virtual_weights,
        fit_tolerance=0.0,
        weighted_fit_residual=0.0,
        unweighted_fit_residual=0.0,
        iterations=0,
        converged=True,
        ridge=0.0,
        seed=0,
        host_scalar_reads=3 if xp is not np else 0,
        auxiliary_block_size=naux,
        largest_intermediate_nbytes=max(
            int(xi.nbytes), int(core.nbytes)
        ),
        allow_unconverged=False,
        exact_full_pair_endpoint=True,
    )


def fit_weighted_eri_thc(
    lov: Any,
    occupied_weights: Any,
    virtual_weights: Any,
    *,
    eri_thc_rank: int,
    fit_tolerance: float,
    max_iterations: int = 500,
    als_convergence_tolerance: float = 1e-10,
    ridge: float = 1e-12,
    seed: int = 0,
    allow_unconverged: bool = False,
    auxiliary_block_size: int = 64,
    transfer_counter: Any = None,
) -> ERITHCFitResult:
    """Fit Eq. 21 and form the Eq. 20 ERI core.

    ``occupied_weights`` and ``virtual_weights`` multiply the residual before
    its Frobenius norm is taken.  They must be strictly positive.  A production
    caller that needs a floor for tiny MP2 occupation-change weights must apply
    that floor explicitly and include it in the approximation identity.
    """

    xp, naux, nocc, nvir = _validate_inputs(
        lov,
        occupied_weights,
        virtual_weights,
        transfer_counter=transfer_counter,
    )
    if isinstance(eri_thc_rank, (bool, np.bool_)) or not isinstance(
        eri_thc_rank, (int, np.integer)
    ):
        raise TypeError("eri_thc_rank must be an integer")
    rank = int(eri_thc_rank)
    if not 1 <= rank <= nocc * nvir:
        raise ValueError("eri_thc_rank must lie in [1,nocc*nvir]")
    controls = (
        ("fit_tolerance", fit_tolerance, True),
        ("als_convergence_tolerance", als_convergence_tolerance, True),
        ("ridge", ridge, True),
    )
    normalized: dict[str, float] = {}
    for name, value, nonnegative in controls:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a real scalar")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{name} must be finite")
        if nonnegative and number < 0.0:
            raise ValueError(f"{name} must be non-negative")
        normalized[name] = number
    max_iterations = _positive_integer("max_iterations", max_iterations)
    auxiliary_block_size = min(
        _positive_integer("auxiliary_block_size", auxiliary_block_size),
        naux,
    )
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    if not isinstance(allow_unconverged, (bool, np.bool_)):
        raise TypeError("allow_unconverged must be a boolean")
    allow_unconverged = bool(allow_unconverged)

    fit_tolerance = normalized["fit_tolerance"]
    als_convergence_tolerance = normalized["als_convergence_tolerance"]
    ridge = normalized["ridge"]
    host_scalar_reads = 2 if xp is not np else 0

    weight_product = (
        occupied_weights[None, :, None]
        * virtual_weights[None, None, :]
    )
    weighted_target_norm_device = _blocked_residual_norm(
        lov,
        weight_product=weight_product,
        auxiliary_block_size=auxiliary_block_size,
        xp=xp,
    )
    weighted_target_norm = _read_scalar(
        weighted_target_norm_device,
        xp=xp,
        transfer_counter=transfer_counter,
        operation="eri_thc_fit_weighted_target_norm",
    )
    host_scalar_reads += int(xp is not np)
    if weighted_target_norm == 0.0:
        raise ValueError("weighted Cholesky target norm is zero")
    target_norm_device = _blocked_residual_norm(
        lov,
        auxiliary_block_size=auxiliary_block_size,
        xp=xp,
    )
    target_norm = _read_scalar(
        target_norm_device,
        xp=xp,
        transfer_counter=transfer_counter,
        operation="eri_thc_fit_target_norm",
    )
    host_scalar_reads += int(xp is not np)
    if target_norm == 0.0:
        raise ValueError("Cholesky target norm is zero")

    host_random = np.random.RandomState(int(seed))
    host_dtype = np.dtype(lov.dtype)
    host_x_occ = host_random.standard_normal((nocc, rank)).astype(host_dtype)
    host_x_vir = host_random.standard_normal((nvir, rank)).astype(host_dtype)
    if xp is np:
        x_occ = host_x_occ
        x_vir = host_x_vir
    else:
        x_occ = xp.asarray(host_x_occ)
        x_vir = xp.asarray(host_x_vir)
        transfer_counter.record_h2d(
            int(host_x_occ.nbytes + host_x_vir.nbytes),
            2,
            operation="eri_thc_fit_initial_factors",
        )
    x_occ /= xp.maximum(
        xp.linalg.norm(x_occ, axis=0, keepdims=True),
        np.finfo(host_dtype).tiny,
    )
    x_vir /= xp.maximum(
        xp.linalg.norm(x_vir, axis=0, keepdims=True),
        np.finfo(host_dtype).tiny,
    )
    xi = xp.zeros((naux, rank), dtype=lov.dtype)
    occupied_weight_sq = occupied_weights * occupied_weights
    virtual_weight_sq = virtual_weights * virtual_weights
    previous = float("inf")
    weighted_residual = float("inf")
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        if iteration > 1:
            mttkrp = xp.einsum(
                "Aia,aI,AI,a->iI",
                lov,
                x_vir,
                xi,
                virtual_weight_sq,
            )
            gram = (
                x_vir.T @ (virtual_weight_sq[:, None] * x_vir)
            ) * (xi.T @ xi)
            x_occ = _right_solve(mttkrp, gram, ridge, xp)
            _normalize_and_absorb(x_occ, xi, xp)

            mttkrp = xp.einsum(
                "Aia,iI,AI,i->aI",
                lov,
                x_occ,
                xi,
                occupied_weight_sq,
            )
            gram = (
                x_occ.T @ (occupied_weight_sq[:, None] * x_occ)
            ) * (xi.T @ xi)
            x_vir = _right_solve(mttkrp, gram, ridge, xp)
            _normalize_and_absorb(x_vir, xi, xp)

        mttkrp = xp.einsum(
            "Aia,iI,aI,i,a->AI",
            lov,
            x_occ,
            x_vir,
            occupied_weight_sq,
            virtual_weight_sq,
        )
        gram = (
            x_occ.T @ (occupied_weight_sq[:, None] * x_occ)
        ) * (x_vir.T @ (virtual_weight_sq[:, None] * x_vir))
        xi = _right_solve(mttkrp, gram, ridge, xp)
        residual_device = _blocked_residual_norm(
            lov,
            x_occ=x_occ,
            x_vir=x_vir,
            xi=xi,
            weight_product=weight_product,
            auxiliary_block_size=auxiliary_block_size,
            xp=xp,
        ) / weighted_target_norm_device
        weighted_residual = _read_scalar(
            residual_device,
            xp=xp,
            transfer_counter=transfer_counter,
            operation="eri_thc_fit_iteration_residual",
        )
        host_scalar_reads += int(xp is not np)
        iterations = iteration
        if weighted_residual <= fit_tolerance:
            converged = True
            break
        change = abs(previous - weighted_residual)
        if np.isfinite(previous) and change <= (
            als_convergence_tolerance * max(1.0, previous)
        ):
            break
        previous = weighted_residual

    if not converged and not allow_unconverged:
        raise RuntimeError(
            "weighted ERI CP/ALS did not reach fit_tolerance; increase "
            "eri_thc_rank or max_iterations, change the explicit seed, or "
            "set allow_unconverged=True for diagnostics only"
        )
    unweighted_device = _blocked_residual_norm(
        lov,
        x_occ=x_occ,
        x_vir=x_vir,
        xi=xi,
        auxiliary_block_size=auxiliary_block_size,
        xp=xp,
    ) / target_norm_device
    unweighted_residual = _read_scalar(
        unweighted_device,
        xp=xp,
        transfer_counter=transfer_counter,
        operation="eri_thc_fit_unweighted_residual",
    )
    host_scalar_reads += int(xp is not np)
    core = xi.T @ xi
    factors = ERITHCFactors(
        x_occ,
        x_vir,
        core,
        transfer_counter=transfer_counter,
    )
    host_scalar_reads += int(xp is not np)
    return ERITHCFitResult(
        factors=factors,
        xi=xi,
        occupied_weights=occupied_weights,
        virtual_weights=virtual_weights,
        fit_tolerance=fit_tolerance,
        weighted_fit_residual=weighted_residual,
        unweighted_fit_residual=unweighted_residual,
        iterations=iterations,
        converged=converged,
        ridge=ridge,
        seed=int(seed),
        host_scalar_reads=host_scalar_reads,
        auxiliary_block_size=auxiliary_block_size,
        largest_intermediate_nbytes=max(
            auxiliary_block_size * nocc * nvir * np.dtype(lov.dtype).itemsize,
            int(xi.nbytes),
            int(core.nbytes),
            int(x_occ.nbytes),
            int(x_vir.nbytes),
        ),
        allow_unconverged=allow_unconverged,
        exact_full_pair_endpoint=False,
    )


__all__ = [
    "ERITHCFitResult",
    "exact_eri_thc_from_cholesky",
    "fit_weighted_eri_thc",
]
