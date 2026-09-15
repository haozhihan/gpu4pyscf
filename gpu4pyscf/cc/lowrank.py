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

"""Low-rank doubles-amplitude representations used by experimental CCSD.

The composite pair index is ordered as ``(i, a)``.  A conventional doubles
tensor is therefore viewed as the symmetric matrix

``T[(i, a), (j, b)] = t2[i, j, a, b]``.

This module deliberately contains no PySCF or GPU4PySCF imports.  The routines
work with NumPy arrays and with CuPy arrays when they are supplied by a caller.
That makes the numerical representation independently testable and prevents a
low-rank data type from silently transferring tensors between host and device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def _array_module(array: Any):
    """Return NumPy or CuPy without importing CuPy on CPU-only installations."""
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy  # pylint: disable=import-outside-toplevel
        return cupy
    return np


def _as_float(value: Any) -> float:
    try:
        return float(value.item())
    except AttributeError:
        return float(value)


def pair_matrix_from_t2(t2: Any):
    """Return the ``(ia,jb)`` matrix view of an ``(ijab)`` doubles tensor."""
    if getattr(t2, "ndim", None) != 4:
        raise ValueError("t2 must have shape (nocc, nocc, nvir, nvir)")
    nocc, nocc_j, nvir, nvir_b = t2.shape
    if nocc != nocc_j or nvir != nvir_b:
        raise ValueError("t2 occupied and virtual dimensions must be square")
    return t2.transpose(0, 2, 1, 3).reshape(nocc * nvir, nocc * nvir)


def t2_from_pair_matrix(matrix: Any, nocc: int, nvir: int):
    """Return an ``(ijab)`` doubles tensor from a composite-pair matrix."""
    dimension = nocc * nvir
    if getattr(matrix, "shape", None) != (dimension, dimension):
        raise ValueError(
            f"pair matrix must have shape {(dimension, dimension)}, "
            f"got {getattr(matrix, 'shape', None)}"
        )
    return matrix.reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)


@dataclass(frozen=True)
class RRProjector:
    """Dominant eigenspace of a symmetric MP2 pair-amplitude matrix."""

    vectors: Any
    eigenvalues: Any
    cutoff: float
    full_dimension: int
    source: str = "mp2-doubles"

    def __post_init__(self):
        if self.cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        if getattr(self.vectors, "ndim", None) != 2:
            raise ValueError("projector vectors must be a rank-2 array")
        if self.vectors.shape[0] != self.full_dimension:
            raise ValueError("projector row dimension does not match pair space")
        if self.vectors.shape[1] != len(self.eigenvalues):
            raise ValueError("one eigenvalue is required for each projector column")

    @property
    def rank(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def compression_fraction(self) -> float:
        if self.full_dimension == 0:
            return 0.0
        return self.rank / self.full_dimension

    def orthogonality_error(self, transfer_counter: Any = None) -> float:
        xp = _array_module(self.vectors)
        identity = xp.eye(self.rank, dtype=self.vectors.dtype)
        error = xp.linalg.norm(self.vectors.T.conj() @ self.vectors - identity)
        if xp is not np:
            if transfer_counter is None:
                raise ValueError(
                    "GPU projector validation requires an explicit "
                    "transfer_counter"
                )
            transfer_counter.record_d2h(int(error.nbytes))
        return _as_float(error)

    def metadata(self, transfer_counter: Any = None) -> Dict[str, Any]:
        xp = _array_module(self.eigenvalues)
        eigenvalues = self.eigenvalues
        if xp is not np:
            if transfer_counter is None:
                raise ValueError(
                    "GPU projector metadata requires an explicit "
                    "transfer_counter"
                )
            eigenvalues = eigenvalues.get()
            transfer_counter.record_d2h(int(self.eigenvalues.nbytes))
        return {
            "source": self.source,
            "cutoff": float(self.cutoff),
            "full_dimension": int(self.full_dimension),
            "rank": self.rank,
            "compression_fraction": self.compression_fraction,
            "orthogonality_error": self.orthogonality_error(
                transfer_counter=transfer_counter
            ),
            "eigenvalues": np.asarray(eigenvalues).tolist(),
        }


def build_rr_projector(
    mp2_doubles: Any,
    eig_cutoff: float,
    *,
    max_rank: Optional[int] = None,
) -> RRProjector:
    """Build the dominant symmetric eigenspace of MP2 doubles.

    ``eig_cutoff`` is an absolute threshold on the magnitude of the pair-space
    eigenvalues.  The retained eigenpairs are ordered by decreasing magnitude.
    The input can be an ``(ijab)`` tensor or an already flattened square matrix.
    """
    if eig_cutoff < 0:
        raise ValueError("eig_cutoff must be non-negative")
    if getattr(mp2_doubles, "ndim", None) == 4:
        matrix = pair_matrix_from_t2(mp2_doubles)
    elif getattr(mp2_doubles, "ndim", None) == 2:
        matrix = mp2_doubles
    else:
        raise ValueError("mp2_doubles must be a rank-4 tensor or square matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("MP2 pair-amplitude matrix must be square")
    dimension = int(matrix.shape[0])
    if max_rank is not None and not 0 < max_rank <= dimension:
        raise ValueError("max_rank must lie in [1, pair_dimension]")

    xp = _array_module(matrix)
    symmetric = (matrix + matrix.T.conj()) * 0.5
    eigenvalues, eigenvectors = xp.linalg.eigh(symmetric)
    order = xp.argsort(xp.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    keep = xp.abs(eigenvalues) >= eig_cutoff
    indices = xp.nonzero(keep)[0]
    if max_rank is not None:
        indices = indices[:max_rank]
    vectors = eigenvectors[:, indices]
    values = eigenvalues[indices]
    return RRProjector(vectors, values, eig_cutoff, dimension)


def build_rr_projector_from_operator(
    matvec: Callable[[np.ndarray], np.ndarray],
    dimension: int,
    eig_cutoff: float,
    *,
    initial_rank: int = 32,
    max_rank: Optional[int] = None,
    solver_tol: float = 1e-10,
    maxiter: Optional[int] = None,
) -> RRProjector:
    """Build an RR projector from a CPU matrix-vector callback.

    The adaptive ``eigsh`` loop doubles the requested subspace until its
    smallest returned eigenvalue falls below ``eig_cutoff``.  It avoids forming
    the full MP2 pair matrix, while keeping a deterministic validation path.
    GPU production code should provide the same contract with a device-native
    block eigensolver; this routine never performs an implicit device transfer.
    """
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if eig_cutoff < 0:
        raise ValueError("eig_cutoff must be non-negative")
    cap = dimension if max_rank is None else min(int(max_rank), dimension)
    if cap < 1:
        raise ValueError("max_rank must be positive")

    # eigsh requires k < N.  Materializing dimensions up to two is exact and
    # also provides the full-rank endpoint for unit tests.
    if dimension <= 2 or cap == dimension and dimension <= initial_rank:
        identity = np.eye(dimension)
        dense = np.column_stack([np.asarray(matvec(identity[:, i]))
                                 for i in range(dimension)])
        return build_rr_projector(dense, eig_cutoff, max_rank=cap)

    from scipy.sparse.linalg import LinearOperator, eigsh

    operator = LinearOperator(
        (dimension, dimension),
        matvec=lambda x: np.asarray(matvec(x)),
        dtype=np.float64,
    )
    requested = min(max(1, int(initial_rank)), cap, dimension - 1)
    last_values = None
    last_vectors = None
    while True:
        values, vectors = eigsh(
            operator,
            k=requested,
            which="LM",
            tol=solver_tol,
            maxiter=maxiter,
        )
        order = np.argsort(np.abs(values))[::-1]
        values = values[order]
        vectors = vectors[:, order]
        last_values, last_vectors = values, vectors
        boundary_below_cutoff = abs(values[-1]) < eig_cutoff
        reached_cap = requested >= min(cap, dimension - 1)
        if boundary_below_cutoff or reached_cap:
            break
        requested = min(requested * 2, cap, dimension - 1)

    keep = np.abs(last_values) >= eig_cutoff
    values = last_values[keep]
    vectors = last_vectors[:, keep]
    if values.size > cap:
        values = values[:cap]
        vectors = vectors[:, :cap]
    return RRProjector(vectors, values, eig_cutoff, dimension,
                       source="mp2-doubles-linear-operator")


@dataclass(frozen=True)
class RRDoubles:
    """Doubles amplitudes stored as ``U @ core @ U.T`` in pair space."""

    projector: RRProjector
    core: Any
    nocc: int
    nvir: int

    def __post_init__(self):
        rank = self.projector.rank
        if getattr(self.core, "shape", None) != (rank, rank):
            raise ValueError(f"RR core must have shape {(rank, rank)}")
        if self.nocc * self.nvir != self.projector.full_dimension:
            raise ValueError("nocc*nvir does not match projector pair dimension")

    @classmethod
    def from_t2(cls, t2: Any, projector: RRProjector) -> "RRDoubles":
        nocc, nocc_j, nvir, nvir_b = t2.shape
        if nocc != nocc_j or nvir != nvir_b:
            raise ValueError("t2 occupied and virtual dimensions must be square")
        matrix = pair_matrix_from_t2(t2)
        vectors = projector.vectors
        core = vectors.T.conj() @ matrix @ vectors
        core = (core + core.T.conj()) * 0.5
        return cls(projector, core, nocc, nvir)

    @property
    def rank(self) -> int:
        return self.projector.rank

    def reconstruct_pair_matrix(self):
        vectors = self.projector.vectors
        return vectors @ self.core @ vectors.T.conj()

    def reconstruct_t2(self):
        return t2_from_pair_matrix(
            self.reconstruct_pair_matrix(), self.nocc, self.nvir
        )

    def reconstruction_error(self, reference_t2: Any) -> float:
        xp = _array_module(reference_t2)
        denominator = xp.linalg.norm(reference_t2)
        numerator = xp.linalg.norm(self.reconstruct_t2() - reference_t2)
        denominator_value = _as_float(denominator)
        if denominator_value == 0.0:
            return _as_float(numerator)
        return _as_float(numerator) / denominator_value

    @property
    def storage_nbytes(self) -> int:
        return int(self.projector.vectors.nbytes + self.core.nbytes)

    def metadata(self, transfer_counter: Any = None) -> Dict[str, Any]:
        return {
            "representation": "rr-doubles",
            "rank": self.rank,
            "nocc": int(self.nocc),
            "nvir": int(self.nvir),
            "storage_nbytes": self.storage_nbytes,
            "projector": self.projector.metadata(
                transfer_counter=transfer_counter
            ),
        }


@dataclass(frozen=True)
class THCDoubles:
    """Two-level factorization ``U Y Z Y.T U.T`` of doubles amplitudes.

    ``symmetric-core-eigh`` is a deterministic validation factorization of the
    RR core.  It exercises the exact storage/API contract required by the
    forthcoming paper-equation THC residual engine, but its metadata explicitly
    records that it is not the paper's CP/LS-THC construction.
    """

    projector: RRProjector
    collocation: Any
    core: Any
    nocc: int
    nvir: int
    fit_tolerance: float
    fit_residual: float
    factorization_method: str = "symmetric-core-eigh"
    paper_exact: bool = False

    def __post_init__(self):
        rr_rank = self.projector.rank
        if getattr(self.collocation, "ndim", None) != 2:
            raise ValueError("THC collocation factor must be a matrix")
        if self.collocation.shape[0] != rr_rank:
            raise ValueError("THC collocation rows must equal the RR rank")
        thc_rank = self.collocation.shape[1]
        if getattr(self.core, "shape", None) != (thc_rank, thc_rank):
            raise ValueError("THC core shape must match the THC rank")
        if self.fit_tolerance < 0:
            raise ValueError("fit_tolerance must be non-negative")
        if self.nocc * self.nvir != self.projector.full_dimension:
            raise ValueError("nocc*nvir does not match projector pair dimension")

    @classmethod
    def from_rr(
        cls,
        doubles: RRDoubles,
        fit_tolerance: float,
        *,
        max_rank: Optional[int] = None,
    ) -> "THCDoubles":
        if fit_tolerance < 0:
            raise ValueError("fit_tolerance must be non-negative")
        rr_core = doubles.core
        xp = _array_module(rr_core)
        symmetric = (rr_core + rr_core.T.conj()) * 0.5
        values, vectors = xp.linalg.eigh(symmetric)
        order = xp.argsort(xp.abs(values))[::-1]
        values, vectors = values[order], vectors[:, order]
        indices = xp.nonzero(xp.abs(values) >= fit_tolerance)[0]
        if max_rank is not None:
            if max_rank < 1:
                raise ValueError("max_rank must be positive")
            indices = indices[:max_rank]
        values, vectors = values[indices], vectors[:, indices]
        small_core = xp.diag(values)
        fitted = vectors @ small_core @ vectors.T.conj()
        norm = xp.linalg.norm(symmetric)
        error = xp.linalg.norm(fitted - symmetric)
        norm_value = _as_float(norm)
        residual = _as_float(error) if norm_value == 0.0 else _as_float(error) / norm_value
        return cls(
            doubles.projector,
            vectors,
            small_core,
            doubles.nocc,
            doubles.nvir,
            fit_tolerance,
            residual,
        )

    @property
    def rr_rank(self) -> int:
        return self.projector.rank

    @property
    def thc_rank(self) -> int:
        return int(self.collocation.shape[1])

    def reconstruct_rr_core(self):
        return self.collocation @ self.core @ self.collocation.T.conj()

    def reconstruct_pair_matrix(self):
        vectors = self.projector.vectors
        return vectors @ self.reconstruct_rr_core() @ vectors.T.conj()

    def reconstruct_t2(self):
        return t2_from_pair_matrix(
            self.reconstruct_pair_matrix(), self.nocc, self.nvir
        )

    @property
    def storage_nbytes(self) -> int:
        return int(
            self.projector.vectors.nbytes
            + self.collocation.nbytes
            + self.core.nbytes
        )

    def metadata(self, transfer_counter: Any = None) -> Dict[str, Any]:
        return {
            "representation": "thc-rr-doubles",
            "factorization_method": self.factorization_method,
            "paper_exact": bool(self.paper_exact),
            "rr_rank": self.rr_rank,
            "thc_rank": self.thc_rank,
            "fit_tolerance": float(self.fit_tolerance),
            "fit_residual": float(self.fit_residual),
            "nocc": int(self.nocc),
            "nvir": int(self.nvir),
            "storage_nbytes": self.storage_nbytes,
            "projector": self.projector.metadata(
                transfer_counter=transfer_counter
            ),
        }


def project_t2(t2: Any, projector: RRProjector):
    """Project a dense doubles tensor into and back out of an RR subspace."""
    return RRDoubles.from_t2(t2, projector).reconstruct_t2()


def factorize_t2(
    t2: Any,
    projector: RRProjector,
    fit_tolerance: float,
) -> Tuple[THCDoubles, Any]:
    """Build the validation THC representation and return its dense image."""
    rr = RRDoubles.from_t2(t2, projector)
    thc = THCDoubles.from_rr(rr, fit_tolerance)
    return thc, thc.reconstruct_t2()
