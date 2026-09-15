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

"""Weighted CP/ALS factorization of RR doubles projectors.

This implements Eqs. 8--16 of Hohenstein *et al.*, J. Chem. Phys. 156,
054102 (2022), DOI 10.1063/5.0077770. The fit minimizes

``sum(P) || epsilon[P] * (U[:,:,P] - CP[:,:,P]) ||_F**2``

so each residual contribution is weighted by the square of the corresponding
MP2 eigenvalue. Symmetric orthogonalization is then applied to the ``tau``
factor, restoring the semi-unitary RR projector contract without pinning the
occupied and virtual factors together.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Optional

import numpy as np

from gpu4pyscf.cc.lowrank import RRProjector


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def _scalar(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def _exact_pair_roundoff_bound(
    dtype: Any, *, pair_dimension: int, rr_rank: int
) -> float:
    """Return a dimension-aware FP roundoff bound for the analytic endpoint."""

    real_dtype = np.empty((), dtype=np.dtype(dtype)).real.dtype
    scale = math.sqrt(max(1, int(pair_dimension), int(rr_rank)))
    return float(128.0 * np.finfo(real_dtype).eps * scale)


def _right_solve(mttkrp: Any, gram: Any, ridge: float, xp: Any):
    regularized = gram + ridge * xp.eye(gram.shape[0], dtype=gram.dtype)
    return xp.linalg.solve(regularized.T, mttkrp.T).T


def _normalize_and_absorb(factor: Any, target: Any, xp: Any) -> None:
    norms = xp.linalg.norm(factor, axis=0)
    tiny = np.finfo(np.dtype(factor.dtype)).tiny
    norms = xp.maximum(norms, tiny)
    factor /= norms[None, :]
    target *= norms[None, :]


def _cp_tensor(y_occ: Any, y_vir: Any, tau: Any, xp: Any):
    return xp.einsum("iW,aW,PW->iaP", y_occ, y_vir, tau)


@dataclass(frozen=True)
class THCProjectorFactors:
    """Paper-equation CP factors for a semi-unitary RR projector."""

    y_occ: Any
    y_vir: Any
    tau: Any
    raw_tau: Any
    eigenvalues: Any
    fit_tolerance: float
    als_weighted_fit_residual: float
    als_unweighted_fit_residual: float
    weighted_fit_residual: float
    unweighted_fit_residual: float
    orthogonalized_projector_distance: float
    orthogonality_error: float
    overlap_min_eigenvalue: float
    orthogonality_cutoff: float
    iterations: int
    converged: bool
    ridge: float
    seed: int
    host_scalar_reads: int
    rank_attempts: tuple[int, ...] = ()

    @property
    def nocc(self) -> int:
        return int(self.y_occ.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.y_vir.shape[0])

    @property
    def rr_rank(self) -> int:
        return int(self.tau.shape[0])

    @property
    def thc_rank(self) -> int:
        return int(self.tau.shape[1])

    @property
    def analytic_full_pair_endpoint(self) -> bool:
        """Whether this is the deterministic exact pair-basis construction."""

        return bool(
            self.thc_rank == self.nocc * self.nvir
            and self.iterations == 0
            and bool(self.rank_attempts)
            and self.rank_attempts[-1] == self.nocc * self.nvir
        )

    @property
    def exact_pair_endpoint(self) -> bool:
        """Alias naming the analytic endpoint used by audit consumers."""

        return self.analytic_full_pair_endpoint

    @property
    def exact_pair_roundoff_bound(self) -> float:
        """Strict reconstruction bound for the analytic pair endpoint."""

        return _exact_pair_roundoff_bound(
            self.y_occ.dtype,
            pair_dimension=self.nocc * self.nvir,
            rr_rank=self.rr_rank,
        )

    @property
    def weighted_fit_gate_limit(self) -> float:
        """Return the accepted weighted-residual limit for this route."""

        if self.exact_pair_endpoint:
            return self.exact_pair_roundoff_bound
        return float(self.fit_tolerance)

    @property
    def weighted_fit_gate_passed(self) -> bool:
        """Apply the endpoint-aware weighted-fit contract to recorded data."""

        if self.exact_pair_endpoint:
            return self.exact_pair_roundoff_gate_passed
        residual = float(self.weighted_fit_residual)
        return bool(
            math.isfinite(residual) and residual <= self.weighted_fit_gate_limit
        )

    @property
    def exact_pair_roundoff_error(self) -> float:
        """Largest reconstruction/orthogonality error at the exact endpoint."""

        diagnostics = (
            self.als_weighted_fit_residual,
            self.als_unweighted_fit_residual,
            self.weighted_fit_residual,
            self.unweighted_fit_residual,
            self.orthogonalized_projector_distance,
            self.orthogonality_error,
        )
        if not all(math.isfinite(float(value)) for value in diagnostics):
            return float("inf")
        return float(max(abs(float(value)) for value in diagnostics))

    @property
    def exact_pair_roundoff_gate_passed(self) -> bool:
        """Require strict roundoff reconstruction at the analytic endpoint."""

        return bool(
            not self.exact_pair_endpoint
            or self.exact_pair_roundoff_error <= self.exact_pair_roundoff_bound
        )

    @property
    def storage_nbytes(self) -> int:
        return int(
            self.y_occ.nbytes
            + self.y_vir.nbytes
            + self.tau.nbytes
            + self.raw_tau.nbytes
        )

    def khatri_rao(self):
        xp = _array_module(self.y_occ)
        return xp.einsum("iW,aW->iaW", self.y_occ, self.y_vir).reshape(
            self.nocc * self.nvir, self.thc_rank
        )

    def reconstruct_fitted_projector(self):
        return self.khatri_rao() @ self.raw_tau.T

    def reconstruct_projector(self):
        return self.khatri_rao() @ self.tau.T

    def amplitude_core(self, rr_core: Any):
        """Transform an RR core to the THC auxiliary basis (Eq. 7)."""

        if rr_core.shape != (self.rr_rank, self.rr_rank):
            raise ValueError("RR core shape does not match projector factors")
        return self.tau.T @ rr_core @ self.tau

    def metadata(self) -> dict[str, Any]:
        return {
            "factorization": "weighted-projector-cp-als",
            "paper_equations": "Hohenstein-2022-Eqs-8-to-16",
            "paper_exact_weighting": True,
            "objective_weights": "mp2-eigenvalue-squared",
            "nocc": self.nocc,
            "nvir": self.nvir,
            "rr_rank": self.rr_rank,
            "thc_rank": self.thc_rank,
            "fit_tolerance": float(self.fit_tolerance),
            "als_weighted_fit_residual": float(
                self.als_weighted_fit_residual
            ),
            "als_unweighted_fit_residual": float(
                self.als_unweighted_fit_residual
            ),
            "weighted_fit_residual": float(self.weighted_fit_residual),
            "unweighted_fit_residual": float(self.unweighted_fit_residual),
            "orthogonalized_projector_distance": float(
                self.orthogonalized_projector_distance
            ),
            "orthogonality_error": float(self.orthogonality_error),
            "overlap_min_eigenvalue": float(self.overlap_min_eigenvalue),
            "orthogonality_cutoff": float(self.orthogonality_cutoff),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "ridge": float(self.ridge),
            "seed": int(self.seed),
            "host_scalar_reads": int(self.host_scalar_reads),
            "storage_nbytes": self.storage_nbytes,
            "storage_includes_raw_tau": True,
            "borrowed_eigenvalues_nbytes": int(self.eigenvalues.nbytes),
            "rank_attempts": [int(rank) for rank in self.rank_attempts],
            "analytic_full_pair_endpoint": self.analytic_full_pair_endpoint,
            "exact_pair_endpoint": self.exact_pair_endpoint,
            "exact_pair_roundoff_bound": self.exact_pair_roundoff_bound,
            "exact_pair_roundoff_error": self.exact_pair_roundoff_error,
            "weighted_fit_gate_limit": self.weighted_fit_gate_limit,
            "weighted_fit_gate_passed": self.weighted_fit_gate_passed,
            "exact_pair_roundoff_gate_passed": self.exact_pair_roundoff_gate_passed,
        }


def fit_weighted_thc_projector(
    projector: RRProjector,
    nocc: int,
    nvir: int,
    *,
    thc_rank: int,
    fit_tolerance: float,
    orthogonality_cutoff: float,
    max_iterations: int = 500,
    als_convergence_tolerance: float = 1e-10,
    ridge: float = 1e-12,
    seed: int = 0,
    orthogonality_tolerance: float = 1e-10,
    allow_unconverged: bool = False,
    transfer_counter: Any = None,
) -> THCProjectorFactors:
    """Fit weighted CP factors and symmetrically orthogonalize ``tau``."""

    if nocc * nvir != projector.full_dimension:
        raise ValueError("nocc*nvir does not match projector pair dimension")
    if int(thc_rank) != thc_rank or thc_rank < 1:
        raise ValueError("thc_rank must be a positive integer")
    if thc_rank < projector.rank:
        raise ValueError("thc_rank cannot be smaller than the RR rank")
    if thc_rank > projector.full_dimension:
        raise ValueError("thc_rank cannot exceed the occupied-virtual pair dimension")
    if fit_tolerance < 0 or als_convergence_tolerance < 0:
        raise ValueError("fit tolerances must be non-negative")
    if orthogonality_cutoff <= 0:
        raise ValueError("orthogonality_cutoff must be positive")
    if orthogonality_tolerance < 0:
        raise ValueError("orthogonality_tolerance must be non-negative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    vectors = projector.vectors
    eigenvalues = projector.eigenvalues
    xp = _array_module(vectors)
    if xp is not np and transfer_counter is None:
        raise ValueError(
            "GPU weighted CP/ALS requires an explicit transfer_counter"
        )
    host_scalar_reads = 0

    def read_scalar(value: Any) -> float:
        nonlocal host_scalar_reads
        if xp is not np:
            transfer_counter.record_d2h(
                int(value.nbytes), operation="thc_fit_control_scalar"
            )
            host_scalar_reads += 1
        return _scalar(value)

    if _array_module(eigenvalues) is not xp:
        raise TypeError("projector vectors and eigenvalues need one backend")
    if any(
        np.issubdtype(np.dtype(value.dtype), np.complexfloating)
        for value in (vectors, eigenvalues)
    ):
        raise NotImplementedError("weighted CP/ALS currently supports real RHF")
    if read_scalar(xp.min(xp.abs(eigenvalues))) == 0:
        raise ValueError("zero-eigenvalue projector columns have no fit weight")

    rr_rank = projector.rank
    target = vectors.reshape(nocc, nvir, rr_rank)
    weighted_target = target * eigenvalues[None, None, :]
    weighted_norm = xp.linalg.norm(weighted_target)
    target_norm = xp.linalg.norm(target)
    dtype = xp.result_type(vectors.dtype, eigenvalues.dtype)
    if thc_rank == projector.full_dimension:
        # Analytic full-pair endpoint.  A column W=(i,a) of the Khatri--Rao
        # factor is the corresponding pair-space unit vector, and tau.T is U.
        # This is an exact CP representation of every RR projector and gives a
        # deterministic endpoint for equation-level tests without relying on a
        # non-convex ALS solve.
        pair_index = xp.arange(projector.full_dimension)
        y_occ = (
            xp.arange(nocc)[:, None] == pair_index[None, :] // nvir
        ).astype(dtype)
        y_vir = (
            xp.arange(nvir)[:, None] == pair_index[None, :] % nvir
        ).astype(dtype)
        weighted_tau = eigenvalues[:, None] * vectors.T
        weighted_residual = 0.0
        converged = True
        iterations = 0
    else:
        # NumPy and CuPy use different pseudo-random algorithms for the same
        # seed. CP/ALS is non-convex, so that difference can change whether a
        # published seed reaches the requested fit gate. Generate the two
        # startup factors once on the host and account for the explicit
        # transfer. All ALS iterations and large tensors remain resident.
        host_random = np.random.RandomState(int(seed))
        host_dtype = np.dtype(dtype)
        host_y_occ = host_random.standard_normal((nocc, thc_rank)).astype(
            host_dtype
        )
        host_y_vir = host_random.standard_normal((nvir, thc_rank)).astype(
            host_dtype
        )
        if xp is np:
            y_occ = host_y_occ
            y_vir = host_y_vir
        else:
            y_occ = xp.asarray(host_y_occ)
            y_vir = xp.asarray(host_y_vir)
            transfer_counter.record_h2d(
                int(host_y_occ.nbytes + host_y_vir.nbytes),
                2,
                operation="thc_fit_initial_factors",
            )
        y_occ /= xp.maximum(
            xp.linalg.norm(y_occ, axis=0, keepdims=True),
            np.finfo(np.dtype(dtype)).tiny,
        )
        y_vir /= xp.maximum(
            xp.linalg.norm(y_vir, axis=0, keepdims=True),
            np.finfo(np.dtype(dtype)).tiny,
        )
        # The third CP factor is epsilon[P] * tau[P,W].
        weighted_tau = xp.zeros((rr_rank, thc_rank), dtype=dtype)
        previous = float("inf")
        weighted_residual = float("inf")
        converged = False
        iterations = 0
        for iteration in range(1, int(max_iterations) + 1):
            if iteration > 1:
                mttkrp = xp.einsum(
                    "iaP,aW,PW->iW", weighted_target, y_vir, weighted_tau
                )
                gram = (y_vir.T @ y_vir) * (weighted_tau.T @ weighted_tau)
                y_occ = _right_solve(mttkrp, gram, ridge, xp)
                _normalize_and_absorb(y_occ, weighted_tau, xp)

                mttkrp = xp.einsum(
                    "iaP,iW,PW->aW", weighted_target, y_occ, weighted_tau
                )
                gram = (y_occ.T @ y_occ) * (weighted_tau.T @ weighted_tau)
                y_vir = _right_solve(mttkrp, gram, ridge, xp)
                _normalize_and_absorb(y_vir, weighted_tau, xp)

            mttkrp = xp.einsum(
                "iaP,iW,aW->PW", weighted_target, y_occ, y_vir
            )
            gram = (y_occ.T @ y_occ) * (y_vir.T @ y_vir)
            weighted_tau = _right_solve(mttkrp, gram, ridge, xp)
            fitted_weighted = _cp_tensor(y_occ, y_vir, weighted_tau, xp)
            weighted_residual = read_scalar(
                xp.linalg.norm(weighted_target - fitted_weighted)
                / weighted_norm
            )
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
            "weighted CP/ALS did not reach fit_tolerance; increase thc_rank "
            "or max_iterations, change the explicit seed, or set "
            "allow_unconverged=True for diagnostics only"
        )

    raw_tau = weighted_tau / eigenvalues[:, None]
    fitted = _cp_tensor(y_occ, y_vir, raw_tau, xp)
    als_unweighted_residual = read_scalar(
        xp.linalg.norm(target - fitted) / target_norm
    )

    y_gram = (y_occ.T @ y_occ) * (y_vir.T @ y_vir)
    overlap = raw_tau @ y_gram @ raw_tau.T
    overlap = (overlap + overlap.T) * 0.5
    overlap_values, overlap_vectors = xp.linalg.eigh(overlap)
    minimum = read_scalar(xp.min(overlap_values))
    if minimum <= orthogonality_cutoff:
        raise np.linalg.LinAlgError(
            "CP projector overlap is singular at orthogonality_cutoff"
        )
    inverse_sqrt = (
        overlap_vectors * (1.0 / xp.sqrt(overlap_values))[None, :]
    ) @ overlap_vectors.T
    tau = inverse_sqrt @ raw_tau
    orthogonalized = _cp_tensor(y_occ, y_vir, tau, xp)
    orthogonality_error = read_scalar(
        xp.linalg.norm(
            orthogonalized.reshape(-1, rr_rank).T
            @ orthogonalized.reshape(-1, rr_rank)
            - xp.eye(rr_rank, dtype=dtype)
        )
    )
    orthogonalized_distance = read_scalar(
        xp.linalg.norm(target - orthogonalized) / target_norm
    )
    final_weighted_residual = read_scalar(
        xp.linalg.norm(
            weighted_target - orthogonalized * eigenvalues[None, None, :]
        ) / weighted_norm
    )
    diagnostics = (
        weighted_residual,
        als_unweighted_residual,
        minimum,
        orthogonality_error,
        orthogonalized_distance,
        final_weighted_residual,
    )
    if not all(math.isfinite(value) for value in diagnostics):
        raise FloatingPointError(
            "THC projector fit produced a non-finite diagnostic"
        )
    exact_pair_endpoint = thc_rank == projector.full_dimension
    exact_roundoff_bound = _exact_pair_roundoff_bound(
        dtype,
        pair_dimension=projector.full_dimension,
        rr_rank=rr_rank,
    )
    if (
        not allow_unconverged
        and not exact_pair_endpoint
        and final_weighted_residual > fit_tolerance
    ):
        raise RuntimeError(
            "orthogonalized THC projector exceeds fit_tolerance: "
            f"{final_weighted_residual:.3e} > {fit_tolerance:.3e}"
        )
    if not allow_unconverged and orthogonality_error > orthogonality_tolerance:
        raise RuntimeError(
            "orthogonalized THC projector exceeds the orthogonality gate: "
            f"{orthogonality_error:.3e} > {orthogonality_tolerance:.3e}"
        )
    result = THCProjectorFactors(
        y_occ=y_occ,
        y_vir=y_vir,
        tau=tau,
        raw_tau=raw_tau,
        eigenvalues=eigenvalues,
        fit_tolerance=float(fit_tolerance),
        als_weighted_fit_residual=weighted_residual,
        als_unweighted_fit_residual=als_unweighted_residual,
        weighted_fit_residual=final_weighted_residual,
        unweighted_fit_residual=orthogonalized_distance,
        orthogonalized_projector_distance=orthogonalized_distance,
        orthogonality_error=orthogonality_error,
        overlap_min_eigenvalue=minimum,
        orthogonality_cutoff=float(orthogonality_cutoff),
        iterations=iterations,
        converged=converged,
        ridge=float(ridge),
        seed=int(seed),
        host_scalar_reads=host_scalar_reads,
        rank_attempts=(int(thc_rank),),
    )
    if exact_pair_endpoint and not result.exact_pair_roundoff_gate_passed:
        raise RuntimeError(
            "analytic full-pair THC projector exceeds its roundoff bound: "
            f"{result.exact_pair_roundoff_error:.3e} > "
            f"{exact_roundoff_bound:.3e}"
        )
    return result


def fit_weighted_thc_projector_adaptive(
    projector: RRProjector,
    nocc: int,
    nvir: int,
    *,
    fit_tolerance: float,
    orthogonality_cutoff: float,
    initial_rank: Optional[int] = None,
    max_rank: Optional[int] = None,
    rank_growth: float = 1.5,
    max_iterations: int = 500,
    als_convergence_tolerance: float = 1e-10,
    ridge: float = 1e-12,
    seed: int = 0,
    orthogonality_tolerance: float = 1e-10,
    transfer_counter: Any = None,
) -> THCProjectorFactors:
    """Increase the CP rank deterministically until the fit gate passes.

    The approximation is controlled only by ``fit_tolerance``.  Rank is an
    observed resource selected by a declared deterministic schedule and is
    recorded in ``rank_attempts``.  If no lower rank passes, the analytic full
    pair endpoint provides a final auditable identity check.
    """

    dimension = int(projector.full_dimension)
    if nocc * nvir != dimension:
        raise ValueError("nocc*nvir does not match projector pair dimension")
    if initial_rank is not None and (
        isinstance(initial_rank, (bool, np.bool_))
        or not isinstance(initial_rank, (int, np.integer))
    ):
        raise TypeError("initial_rank must be an integer or None")
    if max_rank is not None and (
        isinstance(max_rank, (bool, np.bool_))
        or not isinstance(max_rank, (int, np.integer))
    ):
        raise TypeError("max_rank must be an integer or None")
    cap = dimension if max_rank is None else int(max_rank)
    if not 1 <= cap <= dimension:
        raise ValueError("max_rank must lie in [1, pair_dimension]")
    if cap < projector.rank:
        raise ValueError("max_rank cannot be smaller than the RR rank")
    start = projector.rank if initial_rank is None else int(initial_rank)
    if start < 1:
        raise ValueError("initial_rank must be positive")
    start = min(max(start, projector.rank), cap)
    growth = float(rank_growth)
    if not math.isfinite(growth) or growth <= 1.0:
        raise ValueError("rank_growth must be finite and greater than one")

    attempts: list[int] = []
    last_error: Optional[BaseException] = None
    rank = start
    while True:
        attempts.append(rank)
        try:
            result = fit_weighted_thc_projector(
                projector,
                nocc,
                nvir,
                thc_rank=rank,
                fit_tolerance=fit_tolerance,
                orthogonality_cutoff=orthogonality_cutoff,
                max_iterations=max_iterations,
                als_convergence_tolerance=als_convergence_tolerance,
                ridge=ridge,
                seed=seed,
                orthogonality_tolerance=orthogonality_tolerance,
                allow_unconverged=False,
                transfer_counter=transfer_counter,
            )
            return replace(result, rank_attempts=tuple(attempts))
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            last_error = exc
        if rank == cap:
            break
        rank = min(cap, max(rank + 1, int(math.ceil(rank * growth))))
    raise RuntimeError(
        "no THC projector rank reached fit_tolerance up to max_rank; "
        f"attempted ranks {attempts}"
    ) from last_error


__all__ = [
    "THCProjectorFactors",
    "fit_weighted_thc_projector",
    "fit_weighted_thc_projector_adaptive",
]
