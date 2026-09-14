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

"""Matrix-free MP2 pair operators and RR projector construction.

For a closed-shell reference the direct MP2 doubles matrix in the composite
pair index ``p=(i,a)`` is

``T[p,q] = -(B[:,p].H @ B[:,q]) / (gap[p] + gap[q])``.

The Cauchy denominator kernel is positive semidefinite. A pivoted Cholesky
factorization separates it as ``1/(gap[p]+gap[q]) ~= D[p,k]D[q,k]`` with an
explicit max-diagonal residual bound. Applying the MP2 matrix then requires
only GEMMs with resident three-index ``L_ov`` factors; neither the four-index
amplitudes nor the square pair matrix is formed.

The device projector builder uses SciPy Lanczos on NumPy arrays and CuPy's
``cupyx.scipy.sparse.linalg.eigsh`` on CuPy arrays. There is no GPU-to-CPU
fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from gpu4pyscf.cc.lowrank import RRProjector, build_rr_projector


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def _scalar(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def _same_backend(left: Any, right: Any) -> bool:
    return _array_module(left) is _array_module(right)


@dataclass(frozen=True)
class CauchyDenominatorFactor:
    """Pivoted-Cholesky representation of ``1/(gap[p]+gap[q])``."""

    factors: Any
    gaps: Any
    tolerance: float
    residual_diagonal: float
    pivots: tuple[int, ...]
    capped: bool
    construction_backend: str = "numpy"
    preprocessing_d2h_bytes: int = 0
    preprocessing_h2d_bytes: int = 0

    @property
    def rank(self) -> int:
        return int(self.factors.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.factors.shape[0])

    @property
    def max_element_error_bound(self) -> float:
        # For a positive-semidefinite residual,
        # |R[p,q]| <= sqrt(R[p,p] R[q,q]).
        return float(self.residual_diagonal)

    def metadata(self, transfer_counter: Any = None) -> dict[str, Any]:
        return {
            "kernel": "1/(gap[p]+gap[q])",
            "algorithm": "matrix-free-pivoted-cholesky",
            "dimension": self.dimension,
            "rank": self.rank,
            "tolerance": float(self.tolerance),
            "residual_diagonal": float(self.residual_diagonal),
            "max_element_error_bound": self.max_element_error_bound,
            "pivots": list(self.pivots),
            "capped": bool(self.capped),
            "construction_backend": self.construction_backend,
            "preprocessing_d2h_bytes": int(self.preprocessing_d2h_bytes),
            "preprocessing_h2d_bytes": int(self.preprocessing_h2d_bytes),
        }


def factor_cauchy_denominator(
    gaps: Any,
    *,
    tolerance: float,
    max_rank: Optional[int] = None,
    transfer_counter: Any = None,
) -> CauchyDenominatorFactor:
    """Factor the positive Cauchy kernel without materializing it.

    ``tolerance`` is an absolute threshold on the largest remaining diagonal.
    Because the residual is positive semidefinite, it also bounds every
    elementwise denominator-kernel error.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if getattr(gaps, "ndim", None) != 1 or int(gaps.size) == 0:
        raise ValueError("gaps must be a non-empty one-dimensional array")
    xp = _array_module(gaps)
    if not np.issubdtype(np.dtype(gaps.dtype), np.floating):
        raise TypeError("gaps must use a real floating-point dtype")
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU Cauchy factorization requires an explicit "
                "transfer_counter because its setup runs on the CPU"
            )
        host_gaps = gaps.get()
        transfer_counter.record_d2h(int(gaps.nbytes))
        host_result = factor_cauchy_denominator(
            host_gaps,
            tolerance=tolerance,
            max_rank=max_rank,
        )
        device_factors = xp.asarray(host_result.factors)
        transfer_counter.record_h2d(int(device_factors.nbytes))
        return CauchyDenominatorFactor(
            factors=device_factors,
            gaps=gaps,
            tolerance=host_result.tolerance,
            residual_diagonal=host_result.residual_diagonal,
            pivots=host_result.pivots,
            capped=host_result.capped,
            construction_backend="numpy-explicit-device-roundtrip",
            preprocessing_d2h_bytes=int(gaps.nbytes),
            preprocessing_h2d_bytes=int(device_factors.nbytes),
        )
    if not bool(xp.isfinite(gaps).all().item()):
        raise ValueError("gaps contain NaN or infinite values")
    if _scalar(xp.min(gaps)) <= 0:
        raise ValueError("every occupied-virtual energy gap must be positive")
    dimension = int(gaps.size)
    if max_rank is not None and (
        int(max_rank) != max_rank or not 1 <= int(max_rank) <= dimension
    ):
        raise ValueError("max_rank must lie in [1, len(gaps)]")
    cap = dimension if max_rank is None else int(max_rank)
    # The Cauchy kernel normally converges in a few dozen columns. Grow the
    # workspace geometrically so a zero-cutoff diagnostic does not eagerly
    # allocate an OV-by-OV matrix on the host.
    capacity = min(cap, 32)
    factors = xp.zeros((dimension, capacity), dtype=gaps.dtype)
    residual = 0.5 / gaps
    pivots: list[int] = []
    rank = 0
    while rank < cap:
        if rank == capacity:
            new_capacity = min(cap, max(capacity + 1, capacity * 2))
            grown = xp.zeros((dimension, new_capacity), dtype=gaps.dtype)
            grown[:, :rank] = factors[:, :rank]
            factors = grown
            capacity = new_capacity
        pivot = int(xp.argmax(residual).item())
        pivot_value = _scalar(residual[pivot])
        if pivot_value <= tolerance:
            break
        column = 1.0 / (gaps + gaps[pivot])
        if rank:
            column = column - factors[:, :rank] @ factors[pivot, :rank].conj()
        corrected_pivot = _scalar(xp.real(column[pivot]))
        roundoff = 100.0 * np.finfo(np.dtype(gaps.dtype)).eps * max(
            1.0, pivot_value
        )
        if corrected_pivot < -roundoff:
            raise RuntimeError(
                "Cauchy Cholesky residual lost positive semidefiniteness"
            )
        if corrected_pivot <= tolerance:
            residual[pivot] = 0
            continue
        column = column / xp.sqrt(corrected_pivot)
        factors[:, rank] = column
        residual = residual - xp.abs(column) ** 2
        residual = xp.maximum(residual, 0)
        pivots.append(pivot)
        rank += 1
    residual_max = _scalar(xp.max(residual))
    return CauchyDenominatorFactor(
        factors=factors[:, :rank],
        gaps=gaps,
        tolerance=float(tolerance),
        residual_diagonal=residual_max,
        pivots=tuple(pivots),
        capped=bool(rank == cap and residual_max > tolerance),
    )


class MP2PairOperator:
    """Matrix-free direct MP2 doubles operator backed by ``L_ov`` factors."""

    def __init__(
        self,
        lov: Any,
        occupied_energies: Any,
        virtual_energies: Any,
        *,
        denominator_tolerance: float,
        denominator_max_rank: Optional[int] = None,
        aux_axis: str = "first",
        transfer_counter: Any = None,
    ) -> None:
        if getattr(lov, "ndim", None) != 3:
            raise ValueError("lov must be a rank-three array")
        if aux_axis == "last":
            lov = lov.transpose(2, 0, 1)
        elif aux_axis != "first":
            raise ValueError("aux_axis must be 'first' or 'last'")
        self.xp = _array_module(lov)
        self.lov = lov
        self.naux, self.nocc, self.nvir = map(int, lov.shape)
        self.transfer_counter = transfer_counter
        if self.xp is not np and transfer_counter is None:
            raise ValueError(
                "GPU MP2PairOperator requires an explicit transfer_counter"
            )
        self.orbital_energies_uploaded = bool(
            self.xp is not np
            and (
                _array_module(occupied_energies) is not self.xp
                or _array_module(virtual_energies) is not self.xp
            )
        )
        if self.xp is np and (
            _array_module(occupied_energies) is not np
            or _array_module(virtual_energies) is not np
        ):
            raise TypeError(
                "a NumPy operator cannot implicitly download device energies"
            )
        if self.xp is not np and _array_module(occupied_energies) is not self.xp:
            host = np.asarray(occupied_energies)
            occupied_energies = self.xp.asarray(host)
            transfer_counter.record_h2d(int(host.nbytes))
        else:
            occupied_energies = self.xp.asarray(occupied_energies)
        if self.xp is not np and _array_module(virtual_energies) is not self.xp:
            host = np.asarray(virtual_energies)
            virtual_energies = self.xp.asarray(host)
            transfer_counter.record_h2d(int(host.nbytes))
        else:
            virtual_energies = self.xp.asarray(virtual_energies)
        if occupied_energies.shape != (self.nocc,):
            raise ValueError("occupied energies do not match lov")
        if virtual_energies.shape != (self.nvir,):
            raise ValueError("virtual energies do not match lov")
        if not _same_backend(self.lov, occupied_energies):
            raise TypeError("lov and orbital energies must use the same array backend")
        gaps = (
            virtual_energies[None, :] - occupied_energies[:, None]
        ).reshape(-1)
        self.denominator = factor_cauchy_denominator(
            gaps,
            tolerance=denominator_tolerance,
            max_rank=denominator_max_rank,
            transfer_counter=transfer_counter,
        )
        self.dimension = self.nocc * self.nvir
        self.dtype = np.dtype(lov.dtype)
        self._pair_factors = lov.reshape(self.naux, self.dimension)
        self.matvec_calls = 0
        self.matmat_calls = 0
        self.host_scalar_reads = 0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

    def _check_vector_backend(self, vector: Any) -> None:
        if self.xp is not np and _array_module(vector) is not self.xp:
            raise TypeError("GPU MP2PairOperator requires a device-resident vector")

    def matmat(self, vectors: Any):
        """Apply the pair matrix to one or more resident vectors."""

        self._check_vector_backend(vectors)
        vectors = self.xp.asarray(vectors)
        one_vector = vectors.ndim == 1
        if one_vector:
            vectors = vectors[:, None]
        if vectors.ndim != 2 or vectors.shape[0] != self.dimension:
            raise ValueError(
                f"vectors must have shape ({self.dimension},) or "
                f"({self.dimension}, nvec)"
            )
        output = self.xp.zeros(
            (self.dimension, vectors.shape[1]),
            dtype=self.xp.result_type(self.lov.dtype, vectors.dtype),
        )
        pair_factors = self._pair_factors
        for index in range(self.denominator.rank):
            scale = self.denominator.factors[:, index:index + 1]
            auxiliary = pair_factors @ (scale * vectors)
            output -= scale * (pair_factors.conj().T @ auxiliary)
        self.matmat_calls += 1
        if one_vector:
            self.matvec_calls += 1
            return output[:, 0]
        return output

    def matvec(self, vector: Any):
        return self.matmat(vector)

    def read_control_scalar(self, value: Any) -> float:
        """Read and account for one scalar used by host convergence control."""

        if self.xp is not np:
            self.transfer_counter.record_d2h(int(value.nbytes))
            self.host_scalar_reads += 1
        return _scalar(value)

    def materialize(self, *, exact_denominator: bool = False):
        """Materialize the small validation matrix; never used by Lanczos."""

        gram = self._pair_factors.conj().T @ self._pair_factors
        if exact_denominator:
            gaps = self.denominator.gaps
            kernel = 1.0 / (gaps[:, None] + gaps[None, :])
        else:
            factors = self.denominator.factors
            kernel = factors @ factors.conj().T
        return -gram * kernel

    def metadata(self) -> dict[str, Any]:
        return {
            "operator": "direct-mp2-pair-matrix",
            "storage": "L_ov-plus-cauchy-denominator",
            "materializes_pair_matrix": False,
            "naux": self.naux,
            "nocc": self.nocc,
            "nvir": self.nvir,
            "pair_dimension": self.dimension,
            "dtype": str(self.dtype),
            "orbital_energies_uploaded": self.orbital_energies_uploaded,
            "denominator": self.denominator.metadata(),
            "matvec_calls": int(self.matvec_calls),
            "matmat_calls": int(self.matmat_calls),
            "host_scalar_reads": int(self.host_scalar_reads),
        }


@dataclass(frozen=True)
class RRProjectorBuildResult:
    projector: RRProjector
    solver: str
    requested_subspace: int
    cutoff_bracketed: bool
    capped: bool
    max_ritz_residual: float
    operator_metadata: dict[str, Any]

    def metadata(self, transfer_counter: Any = None) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "requested_subspace": int(self.requested_subspace),
            "cutoff_bracketed": bool(self.cutoff_bracketed),
            "capped": bool(self.capped),
            "max_ritz_residual": float(self.max_ritz_residual),
            "projector": self.projector.metadata(
                transfer_counter=transfer_counter
            ),
            "operator": self.operator_metadata,
        }


def build_rr_projector_lanczos(
    operator: MP2PairOperator,
    *,
    rr_eig_cutoff: float,
    initial_rank: int = 32,
    max_rank: Optional[int] = None,
    solver_tolerance: float = 1e-10,
    maxiter: Optional[int] = None,
    dense_fallback_dimension: int = 64,
    ritz_residual_tolerance: Optional[float] = None,
    allow_incomplete: bool = False,
    allow_unconverged: bool = False,
) -> RRProjectorBuildResult:
    """Build a dominant MP2 eigenspace on the operator's current backend."""

    if rr_eig_cutoff < 0:
        raise ValueError("rr_eig_cutoff must be non-negative")
    if initial_rank < 1:
        raise ValueError("initial_rank must be positive")
    if solver_tolerance <= 0:
        raise ValueError("solver_tolerance must be positive")
    if dense_fallback_dimension < 0:
        raise ValueError("dense_fallback_dimension must be non-negative")
    if ritz_residual_tolerance is None:
        ritz_residual_tolerance = max(10.0 * solver_tolerance, 1e-12)
    if ritz_residual_tolerance < 0:
        raise ValueError("ritz_residual_tolerance must be non-negative")
    dimension = operator.dimension
    if max_rank is not None and (
        int(max_rank) != max_rank or not 1 <= int(max_rank) <= dimension
    ):
        raise ValueError("max_rank must lie in [1, pair_dimension]")
    cap = dimension if max_rank is None else int(max_rank)
    xp = operator.xp

    if dimension <= dense_fallback_dimension and cap == dimension:
        dense = operator.materialize()
        projector = build_rr_projector(
            dense, rr_eig_cutoff, max_rank=max_rank
        )
        residual = operator.matmat(projector.vectors) - (
            projector.vectors * projector.eigenvalues[None, :]
        )
        max_residual = (
            0.0 if projector.rank == 0
            else operator.read_control_scalar(
                xp.max(xp.linalg.norm(residual, axis=0))
            )
        )
        if max_residual > ritz_residual_tolerance and not allow_unconverged:
            raise RuntimeError(
                "dense RR projector failed the Ritz residual gate: "
                f"{max_residual:.3e} > {ritz_residual_tolerance:.3e}"
            )
        return RRProjectorBuildResult(
            projector=projector,
            solver=f"{xp.__name__}-dense-validation",
            requested_subspace=dimension,
            cutoff_bracketed=True,
            capped=False,
            max_ritz_residual=max_residual,
            operator_metadata=operator.metadata(),
        )

    spectral_cap = min(cap, dimension - 1)
    if spectral_cap < 1:
        raise ValueError("Lanczos requires pair dimension greater than one")
    requested = min(max(1, int(initial_rank)), spectral_cap)
    if xp is np:
        from scipy.sparse.linalg import LinearOperator, eigsh

        linear_operator = LinearOperator(
            operator.shape,
            matvec=operator.matvec,
            matmat=operator.matmat,
            dtype=operator.dtype,
        )
        solver_name = "scipy-eigsh-lanczos"
    else:
        try:
            from cupyx.scipy.sparse.linalg import LinearOperator, eigsh
        except Exception as exc:  # pragma: no cover - requires a CUDA host
            raise RuntimeError(
                "CuPy Lanczos is unavailable; refusing a CPU fallback"
            ) from exc
        linear_operator = LinearOperator(
            operator.shape,
            matvec=operator.matvec,
            matmat=operator.matmat,
            dtype=operator.dtype,
        )
        solver_name = "cupyx-eigsh-lanczos"

    values = vectors = None
    cutoff_bracketed = False
    while True:
        values, vectors = eigsh(
            linear_operator,
            k=requested,
            which="LM",
            tol=solver_tolerance,
            maxiter=maxiter,
        )
        order = xp.argsort(xp.abs(values))[::-1]
        values = values[order]
        vectors = vectors[:, order]
        cutoff_bracketed = (
            operator.read_control_scalar(xp.abs(values[-1])) < rr_eig_cutoff
        )
        if cutoff_bracketed or requested >= spectral_cap:
            break
        requested = min(requested * 2, spectral_cap)

    keep = xp.abs(values) >= rr_eig_cutoff
    kept_values = values[keep]
    kept_vectors = vectors[:, keep]
    projector = RRProjector(
        vectors=kept_vectors,
        eigenvalues=kept_values,
        cutoff=float(rr_eig_cutoff),
        full_dimension=dimension,
        source="mp2-Lov-cauchy-device-lanczos",
    )
    if projector.rank:
        residual = operator.matmat(projector.vectors) - (
            projector.vectors * projector.eigenvalues[None, :]
        )
        max_residual = operator.read_control_scalar(
            xp.max(xp.linalg.norm(residual, axis=0))
        )
    else:
        max_residual = 0.0
    capped = bool(not cutoff_bracketed and requested >= spectral_cap)
    if capped and not allow_incomplete:
        raise RuntimeError(
            "Lanczos reached its rank limit before bracketing rr_eig_cutoff; "
            "the returned subspace would be incomplete. Increase max_rank, "
            "use a looser nonzero cutoff, or set allow_incomplete=True for "
            "diagnostics only"
        )
    if max_residual > ritz_residual_tolerance and not allow_unconverged:
        raise RuntimeError(
            "Lanczos RR projector failed the Ritz residual gate: "
            f"{max_residual:.3e} > {ritz_residual_tolerance:.3e}"
        )
    return RRProjectorBuildResult(
        projector=projector,
        solver=solver_name,
        requested_subspace=requested,
        cutoff_bracketed=cutoff_bracketed,
        capped=capped,
        max_ritz_residual=max_residual,
        operator_metadata=operator.metadata(),
    )


__all__ = [
    "CauchyDenominatorFactor",
    "factor_cauchy_denominator",
    "MP2PairOperator",
    "RRProjectorBuildResult",
    "build_rr_projector_lanczos",
]
