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

"""Backend-neutral and device-resident integral providers for CCSD.

The production GPU CCSD implementation has its own integral transformation
and storage code.  The classes in this module deliberately provide a small
common interface around either a dense four-index ERI tensor or a factorized
three-index tensor.  They are useful for validation and for algorithms that
only need an ERI provider; they do not claim to be an out-of-core integral
engine. :class:`MOThreeIndexIntegralProvider` is the production-facing
resident interface: it preserves NumPy or CuPy storage and exposes the
``L_oo``, ``L_ov`` and ``L_vv`` blocks used by RR/THC contractions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]
Index = Union[int, slice, Sequence[int], np.ndarray]


@runtime_checkable
class IntegralProvider(Protocol):
    """Protocol for a small ERI provider.

    ``shape`` is always ``(n, n, n, n)`` in chemists' notation.  ``get``
    accepts either four indices or a tuple of four indices and returns the
    corresponding ERI block.  ``to_dense`` may be expensive for a factorized
    provider, so callers that only need a small block should use ``get``.
    """

    @property
    def shape(self) -> Tuple[int, int, int, int]: ...

    @property
    def dtype(self) -> np.dtype: ...

    def get(self, *indices: Index) -> np.ndarray: ...

    def to_dense(self) -> np.ndarray: ...


@dataclass(frozen=True)
class CholeskyResult:
    """Result and diagnostics from :func:`pivoted_cholesky`.

    ``factor`` has shape ``(n, rank)`` and reconstructs the matrix as
    ``factor @ factor.conj().T``.  ``max_error`` is the largest absolute
    element of the final reconstruction residual, while ``residual_diagonal``
    is the largest remaining diagonal used by the stopping criterion.
    """

    factor: np.ndarray
    threshold: float
    rank: int
    max_error: float
    frobenius_error: float
    residual_diagonal: float
    pivots: np.ndarray

    @property
    def error(self) -> float:
        """Alias for ``max_error`` used by validation code."""

        return self.max_error

    @property
    def reconstruction(self) -> np.ndarray:
        """Reconstructed matrix represented by the factor."""

        return self.factor @ self.factor.conj().T

    @property
    def metadata(self) -> dict:
        """JSON-friendly scalar diagnostics (the factor is kept separate)."""

        return {
            "threshold": self.threshold,
            "rank": self.rank,
            "max_error": self.max_error,
            "frobenius_error": self.frobenius_error,
            "residual_diagonal": self.residual_diagonal,
            "pivots": self.pivots.tolist(),
        }

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return (self.factor, self)[int(key)]
        return self.metadata[key]

    def __iter__(self):
        """Allow ``factor, metadata = pivoted_cholesky(...)`` unpacking."""

        yield self.factor
        yield self


def pivoted_cholesky(
    matrix: ArrayLike,
    threshold: float = 1e-10,
    max_rank: Optional[int] = None,
    *,
    tol: Optional[float] = None,
) -> CholeskyResult:
    """Compute a numerically stable pivoted Cholesky factorization.

    Parameters
    ----------
    matrix
        A real symmetric or complex Hermitian positive semidefinite matrix.
    threshold
        Absolute threshold on the largest residual diagonal.  The iteration
        stops once every residual diagonal is no larger than this value.
    max_rank
        Optional cap on the number of pivots.  ``None`` means no cap.
    tol
        Alias for ``threshold`` retained for callers using common Cholesky
        terminology.  Supplying both values is an error.

    Notes
    -----
    This routine is intentionally a NumPy reference implementation for small
    validation problems.  It does not silently repair an indefinite matrix:
    a materially negative pivot raises ``ValueError``.  Tiny negative values
    caused by roundoff are clipped to zero.
    """

    if tol is not None:
        if threshold != 1e-10:
            raise ValueError("supply either threshold or tol, not both")
        threshold = tol
    a = np.asarray(matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array")
    if not np.issubdtype(a.dtype, np.number):
        raise TypeError("matrix must have a numeric dtype")
    if not np.isfinite(a).all():
        raise ValueError("matrix contains NaN or infinite values")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if max_rank is not None and (max_rank < 0 or int(max_rank) != max_rank):
        raise ValueError("max_rank must be a non-negative integer or None")

    # Reject a materially non-Hermitian input before removing harmless
    # roundoff asymmetry. Silent Hermitization would factor a different matrix
    # and make the reconstruction diagnostics describe the wrong object.
    real_dtype = np.result_type(np.real(a).dtype, np.float64)
    matrix_scale = max(1.0, float(np.max(np.abs(a))) if a.size else 1.0)
    hermitian_tolerance = 100 * np.finfo(real_dtype).eps * matrix_scale
    hermitian_error = float(np.max(np.abs(a - a.conj().T))) if a.size else 0.0
    if hermitian_error > hermitian_tolerance:
        raise ValueError(
            "matrix must be Hermitian within floating-point roundoff; "
            f"max anti-Hermitian element is {hermitian_error:.3e}"
        )
    # Hermitize only to remove the accepted roundoff-level asymmetry. A complex
    # input is kept complex so that conjugate products remain correct.
    a = (a + a.conj().T) * 0.5
    n = a.shape[0]
    scale = max(1.0, float(np.max(np.abs(np.diag(a)))) if n else 1.0)
    neg_tol = 100 * np.finfo(real_dtype).eps * scale
    residual = np.real(np.diag(a)).astype(real_dtype, copy=True)
    if np.any(residual < -neg_tol):
        raise ValueError("matrix is not positive semidefinite (negative diagonal)")
    residual[residual < 0] = 0

    cap = n if max_rank is None else min(n, int(max_rank))
    work_dtype = np.result_type(a.dtype, np.float64)
    factor = np.zeros((n, cap), dtype=work_dtype)
    pivots = []
    rank = 0
    while rank < cap:
        pivot = int(np.argmax(residual)) if n else 0
        pivot_value = float(residual[pivot]) if n else 0.0
        if pivot_value <= threshold:
            break
        col = a[:, pivot].astype(work_dtype, copy=True)
        if rank:
            # A[:,p] - L L[p,:]^H.  This is the residual column before
            # normalization and is the crucial conjugate operation for
            # complex Hermitian matrices.
            col -= factor[:, :rank] @ factor[pivot, :rank].conj()
        pivot_value = float(np.real(col[pivot]))
        if pivot_value < -neg_tol:
            raise ValueError("matrix is not positive semidefinite")
        if pivot_value <= threshold:
            residual[pivot] = 0.0
            continue
        col /= np.sqrt(pivot_value)
        factor[:, rank] = col
        residual -= np.abs(col) ** 2
        # Small negative residuals are roundoff; a substantially negative
        # value indicates an inconsistent/non-PSD input.
        if residual.size and np.min(residual) < -100 * neg_tol:
            raise ValueError("matrix is not positive semidefinite")
        residual[residual < 0] = 0.0
        pivots.append(pivot)
        rank += 1

    factor = factor[:, :rank]
    remainder = a - factor @ factor.conj().T
    max_error = float(np.max(np.abs(remainder))) if n else 0.0
    frob_error = float(np.linalg.norm(remainder))
    residual_max = float(np.max(residual)) if n else 0.0
    return CholeskyResult(
        factor=factor,
        threshold=float(threshold),
        rank=rank,
        max_error=max_error,
        frobenius_error=frob_error,
        residual_diagonal=residual_max,
        pivots=np.asarray(pivots, dtype=int),
    )


def _normalize_indices(indices: Tuple[Index, ...], ndim: int) -> Tuple[Index, ...]:
    if len(indices) == 1 and isinstance(indices[0], tuple):
        indices = indices[0]
    if len(indices) == 0:
        return (slice(None),) * ndim
    if len(indices) != ndim:
        raise IndexError(f"expected {ndim} indices, got {len(indices)}")
    return indices


class DenseIntegralProvider:
    """ERI provider backed by a dense ``(n,n,n,n)`` NumPy array."""

    def __init__(self, eri: ArrayLike):
        arr = np.asarray(eri)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            n2 = arr.shape[0]
            n = int(np.sqrt(n2))
            if n * n != n2:
                raise ValueError("a two-dimensional ERI matrix must have n**2 rows")
            arr = arr.reshape(n, n, n, n)
        if arr.ndim != 4 or len(set(arr.shape)) != 1:
            raise ValueError("dense ERIs must have shape (n,n,n,n)")
        self._eri = np.asarray(arr)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self._eri.shape

    @property
    def dtype(self) -> np.dtype:
        return self._eri.dtype

    @property
    def n_orbitals(self) -> int:
        return self._eri.shape[0]

    def get(self, *indices: Index) -> np.ndarray:
        return self._eri[_normalize_indices(indices, 4)]

    def to_dense(self) -> np.ndarray:
        return self._eri.copy()

    reconstruct = to_dense
    get_eri = get


class ThreeIndexIntegralProvider:
    """ERI provider from a factor ``L[p,q,Q]``.

    The represented ERI is ``(pq|rs) = sum_Q L[p,q,Q] * conj(L[r,s,Q])``.
    The constructor accepts the natural three-dimensional shape ``(n,n,naux)``
    and the common two-dimensional full-pair shape ``(n*n,naux)`` or
    ``(naux,n*n)``.  Packed symmetric pairs can be supplied with
    ``pair_indices=(p, q)`` and ``n_orbitals``.
    """

    def __init__(
        self,
        factors: ArrayLike,
        n_orbitals: Optional[int] = None,
        *,
        axis: Optional[str] = None,
        pair_indices: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    ):
        arr = np.asarray(factors)
        if arr.ndim not in (2, 3):
            raise ValueError("three-index factors must be two- or three-dimensional")
        self._pair_indices = None
        if arr.ndim == 3:
            if axis in ("aux_first", "q_first"):
                arr = arr.transpose(1, 2, 0)
            elif axis in ("aux_last", "q_last"):
                pass
            elif axis is None:
                if arr.shape[0] == arr.shape[1]:
                    # Natural L[p,q,Q] convention.
                    pass
                elif arr.shape[1] == arr.shape[2]:
                    # Common DF convention L[Q,p,q].
                    arr = arr.transpose(1, 2, 0)
                else:
                    raise ValueError("cannot infer the auxiliary-index axis")
            else:
                raise ValueError("axis must be 'aux_first' or 'aux_last'")
            if arr.shape[0] != arr.shape[1]:
                raise ValueError("full three-index factors must have shape (n,n,naux)")
            n = arr.shape[0]
        else:
            if pair_indices is not None:
                if n_orbitals is None:
                    raise ValueError("n_orbitals is required with packed pair_indices")
                p, q = (np.asarray(pair_indices[0], dtype=int), np.asarray(pair_indices[1], dtype=int))
                if (
                    p.shape != q.shape
                    or p.ndim != 1
                    or (len(p) != arr.shape[0] and len(p) != arr.shape[1])
                ):
                    raise ValueError("pair_indices must contain one index per pair")
                npair = len(p)
                if arr.shape[0] == npair:
                    pair_factors = arr
                elif arr.shape[1] == npair:
                    pair_factors = arr.T
                else:
                    raise ValueError("factor dimensions do not match pair_indices")
                n = int(n_orbitals)
                if np.any(p < 0) or np.any(q < 0) or np.any(p >= n) or np.any(q >= n):
                    raise ValueError("pair_indices contain an orbital outside n_orbitals")
                full = np.zeros(
                    (n, n, pair_factors.shape[1]), dtype=pair_factors.dtype
                )
                full[p, q] = pair_factors
                off_diagonal = p != q
                full[q[off_diagonal], p[off_diagonal]] = pair_factors[
                    off_diagonal
                ].conj()
                if np.issubdtype(pair_factors.dtype, np.complexfloating):
                    diagonal = pair_factors[~off_diagonal]
                    diagonal_scale = max(
                        1.0,
                        float(np.max(np.abs(diagonal))) if diagonal.size else 1.0,
                    )
                    diagonal_tolerance = 100 * np.finfo(
                        np.real(pair_factors).dtype
                    ).eps * diagonal_scale
                    if (
                        diagonal.size
                        and np.max(np.abs(diagonal.imag)) > diagonal_tolerance
                    ):
                        raise ValueError(
                            "diagonal packed factors must be real for a "
                            "Hermitian orbital-pair factor"
                        )
                self._factors = full
                self._pair_indices = (p, q)
                return
            if n_orbitals is None:
                # Full pair factors have n**2 rows in the usual (pair,aux)
                # convention; infer n whenever possible.
                npair = arr.shape[0] if int(np.sqrt(arr.shape[0])) ** 2 == arr.shape[0] else arr.shape[1]
                n_orbitals = int(np.sqrt(npair))
            n = int(n_orbitals)
            if arr.shape[0] == n * n:
                arr = arr.reshape(n, n, arr.shape[1])
            elif arr.shape[1] == n * n:
                arr = arr.T.reshape(n, n, arr.shape[0])
            else:
                raise ValueError("full pair factors must contain n**2 pair rows")
        self._factors = np.asarray(arr)

    @property
    def factors(self) -> np.ndarray:
        return self._factors

    @property
    def three_index(self) -> np.ndarray:
        return self._factors

    @property
    def n_orbitals(self) -> int:
        return self._factors.shape[0]

    @property
    def naux(self) -> int:
        return self._factors.shape[2]

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        n = self.n_orbitals
        return (n, n, n, n)

    @property
    def dtype(self) -> np.dtype:
        return self._factors.dtype

    def get(self, *indices: Index) -> np.ndarray:
        normalized = _normalize_indices(indices, 4)
        orbital_indices = np.arange(self.n_orbitals)
        values = []
        scalar_axes = []
        for axis, index in enumerate(normalized):
            selected = np.asarray(orbital_indices[index])
            values.append(np.atleast_1d(selected).astype(int, copy=False))
            if selected.ndim == 0:
                scalar_axes.append(axis)

        p, q, r, s = values
        left = self._factors[np.ix_(p, q)]
        right = self._factors[np.ix_(r, s)]
        block = np.einsum("pqk,rsk->pqrs", left, right.conj())
        for axis in reversed(scalar_axes):
            block = np.squeeze(block, axis=axis)
        return block

    def to_dense(self) -> np.ndarray:
        return np.einsum("pqk,rsk->pqrs", self._factors, self._factors.conj())

    reconstruct = to_dense
    get_eri = get


def _array_module(array):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


class MOThreeIndexIntegralProvider:
    """Resident molecular-orbital factors in ``L[Q,p,q]`` layout.

    Unlike :class:`ThreeIndexIntegralProvider`, this class never converts the
    supplied factors to NumPy. A CuPy input remains on its current device. The
    factorization kind and threshold are mandatory audit metadata rather than
    inferred labels.
    """

    def __init__(
        self,
        factors,
        nocc: int,
        *,
        aux_axis: str = "first",
        factorization: str,
        threshold: Optional[float],
        source: str = "external",
    ):
        if getattr(factors, "ndim", None) != 3:
            raise ValueError("MO three-index factors must be rank three")
        if aux_axis == "last":
            factors = factors.transpose(2, 0, 1)
        elif aux_axis != "first":
            raise ValueError("aux_axis must be 'first' or 'last'")
        if factors.shape[1] != factors.shape[2]:
            raise ValueError("MO factors must have shape (naux,nmo,nmo)")
        nmo = int(factors.shape[1])
        if int(nocc) != nocc or not 0 < int(nocc) < nmo:
            raise ValueError("nocc must lie strictly between zero and nmo")
        factorization = str(factorization).lower()
        if factorization not in {"cd", "df", "custom"}:
            raise ValueError("factorization must be 'cd', 'df', or 'custom'")
        if threshold is not None and threshold < 0:
            raise ValueError("threshold must be non-negative")
        if factorization == "cd" and threshold is None:
            raise ValueError("a CD provider requires an explicit threshold")
        self._factors = factors
        self._nocc = int(nocc)
        self.factorization = factorization
        self.threshold = None if threshold is None else float(threshold)
        self.source = str(source)
        self.xp = _array_module(factors)

    @classmethod
    def from_ao_factors(
        cls,
        ao_factors,
        mo_coeff,
        nocc: int,
        *,
        aux_axis: str = "first",
        factorization: str,
        threshold: Optional[float],
        source: str = "ao-three-index",
        auxiliary_block_size: Optional[int] = None,
    ) -> "MOThreeIndexIntegralProvider":
        """Transform resident AO factors to MO factors in auxiliary blocks."""

        if getattr(ao_factors, "ndim", None) != 3:
            raise ValueError("AO factors must be rank three")
        if aux_axis == "last":
            ao_factors = ao_factors.transpose(2, 0, 1)
        elif aux_axis != "first":
            raise ValueError("aux_axis must be 'first' or 'last'")
        if ao_factors.shape[1] != ao_factors.shape[2]:
            raise ValueError("AO factors must have shape (naux,nao,nao)")
        xp = _array_module(ao_factors)
        mo_coeff = xp.asarray(mo_coeff)
        if mo_coeff.ndim != 2 or mo_coeff.shape[0] != ao_factors.shape[1]:
            raise ValueError("mo_coeff must have shape (nao,nmo)")
        naux = int(ao_factors.shape[0])
        nmo = int(mo_coeff.shape[1])
        block_size = naux if auxiliary_block_size is None else int(
            auxiliary_block_size
        )
        if block_size < 1:
            raise ValueError("auxiliary_block_size must be positive")
        result = xp.empty(
            (naux, nmo, nmo),
            dtype=xp.result_type(ao_factors.dtype, mo_coeff.dtype),
        )
        for start in range(0, naux, block_size):
            stop = min(start + block_size, naux)
            # L_Qpq = C_mp^* L_Qmn C_nq, split into two batched GEMMs.
            intermediate = xp.matmul(ao_factors[start:stop], mo_coeff)
            result[start:stop] = xp.matmul(mo_coeff.T.conj(), intermediate)
        return cls(
            result,
            nocc,
            factorization=factorization,
            threshold=threshold,
            source=source,
        )

    @classmethod
    def from_gpu4pyscf_df(
        cls,
        dfobj: Any,
        mo_coeff: Any,
        nocc: int,
        *,
        auxiliary_block_size: Optional[int] = None,
        source: str = "gpu4pyscf-df-cderi",
        transfer_counter: Any = None,
        require_gpu_resident: bool = True,
    ) -> "MOThreeIndexIntegralProvider":
        """Transform a single-GPU GPU4PySCF DF stream into resident MO factors.

        This adapter consumes the unpacked first value from ``DF.loop``.  It is
        an auxiliary-basis density-fitting path, not a direct AO-pair Cholesky
        implementation, and is therefore always labelled ``factorization=df``.
        The private ``_cderi`` shape is used deliberately because ``DF.naux``
        reports the raw auxiliary size even when metric linear dependence
        reduces the effective factor rank.
        """

        if getattr(dfobj, "_cderi", None) is None:
            dfobj.build()
        cderi = getattr(dfobj, "_cderi", None)
        if cderi is None:
            raise RuntimeError("GPU4PySCF DF.build() did not create _cderi")
        if hasattr(cderi, "items"):
            slices = list(cderi.items())
        else:
            slices = list(enumerate(cderi))
        nonempty = [
            (device, value)
            for device, value in slices
            if value is not None and int(value.shape[0]) > 0
        ]
        if len(nonempty) != 1:
            raise NotImplementedError(
                "the DF adapter currently requires exactly one non-empty GPU slice"
            )
        _, cderi_slice = nonempty[0]
        storage_xp = _array_module(cderi_slice)
        if require_gpu_resident and storage_xp is np:
            raise MemoryError(
                "GPU4PySCF stored CDERI on the host; set DF.use_gpu_memory=True "
                "or reduce the validation case"
            )
        effective_naux = int(cderi_slice.shape[0])
        block_size = (
            None if auxiliary_block_size is None else int(auxiliary_block_size)
        )
        if block_size is not None and block_size < 1:
            raise ValueError("auxiliary_block_size must be positive")

        result = None
        coefficient = None
        offset = 0
        for ao_block, _packed_block in dfobj.loop(
            blksize=block_size, unpack=True
        ):
            if ao_block is None or getattr(ao_block, "ndim", None) != 3:
                raise RuntimeError("DF.loop(unpack=True) did not yield AO factors")
            xp = _array_module(ao_block)
            if require_gpu_resident and xp is np:
                raise RuntimeError("DF.loop returned a host AO-factor block")
            if xp is not storage_xp:
                raise TypeError("DF.loop backend does not match _cderi storage")
            if coefficient is None:
                coefficient = mo_coeff
                if _array_module(coefficient) is not xp:
                    if xp is not np:
                        host_coefficient = np.asarray(coefficient)
                        original_nbytes = int(host_coefficient.nbytes)
                        if transfer_counter is None:
                            raise ValueError(
                                "uploading MO coefficients requires an explicit "
                                "transfer_counter"
                            )
                        coefficient = xp.asarray(host_coefficient)
                        transfer_counter.record_h2d(original_nbytes)
                    else:
                        coefficient = xp.asarray(coefficient)
                if coefficient.ndim != 2 or coefficient.shape[0] != ao_block.shape[1]:
                    raise ValueError("mo_coeff must have shape (nao,nmo)")
                nmo = int(coefficient.shape[1])
                result = xp.empty(
                    (effective_naux, nmo, nmo),
                    dtype=xp.result_type(ao_block.dtype, coefficient.dtype),
                )
            elif _array_module(result) is not xp:
                raise TypeError("DF.loop changed array backend between blocks")
            count = int(ao_block.shape[0])
            if offset + count > effective_naux:
                raise RuntimeError("DF.loop yielded more factors than _cderi stores")
            intermediate = xp.matmul(ao_block, coefficient)
            result[offset:offset + count] = xp.matmul(
                coefficient.T.conj(), intermediate
            )
            offset += count
        if result is None or offset != effective_naux:
            raise RuntimeError(
                "DF.loop factor count does not match the effective _cderi rank"
            )
        return cls(
            result,
            nocc,
            factorization="df",
            threshold=None,
            source=source,
        )

    @property
    def factors(self):
        return self._factors

    @property
    def naux(self) -> int:
        return int(self._factors.shape[0])

    @property
    def nocc(self) -> int:
        return self._nocc

    @property
    def nmo(self) -> int:
        return int(self._factors.shape[1])

    @property
    def nvir(self) -> int:
        return self.nmo - self.nocc

    @property
    def dtype(self):
        return self._factors.dtype

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.nmo,) * 4

    @property
    def L_oo(self):
        return self._factors[:, :self.nocc, :self.nocc]

    @property
    def L_ov(self):
        return self._factors[:, :self.nocc, self.nocc:]

    @property
    def L_vv(self):
        return self._factors[:, self.nocc:, self.nocc:]

    # Lower-case aliases make contraction call sites less noisy.
    loo = L_oo
    lov = L_ov
    lvv = L_vv

    @property
    def storage_nbytes(self) -> int:
        return int(self._factors.nbytes)

    def block(self, name: str, auxiliary_slice: slice = slice(None)):
        blocks = {"oo": self.L_oo, "ov": self.L_ov, "vv": self.L_vv}
        try:
            return blocks[name.lower()][auxiliary_slice]
        except KeyError as exc:
            raise ValueError("block name must be 'oo', 'ov', or 'vv'") from exc

    def iter_auxiliary(self, block_size: int, *, block: str = "ov"):
        if int(block_size) != block_size or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        for start in range(0, self.naux, int(block_size)):
            stop = min(start + int(block_size), self.naux)
            yield slice(start, stop), self.block(block, slice(start, stop))

    def get(self, *indices: Index):
        normalized = _normalize_indices(indices, 4)
        orbital_indices = self.xp.arange(self.nmo)
        selections = []
        scalar_axes = []
        for axis, index in enumerate(normalized):
            selected = orbital_indices[index]
            selections.append(self.xp.atleast_1d(selected).astype(int))
            if selected.ndim == 0:
                scalar_axes.append(axis)
        p, q, r, s = selections
        left = self._factors[:, p[:, None], q[None, :]]
        right = self._factors[:, r[:, None], s[None, :]]
        result = self.xp.einsum("Qpq,Qrs->pqrs", left, right.conj())
        for axis in reversed(scalar_axes):
            result = self.xp.squeeze(result, axis=axis)
        return result

    def to_dense(self):
        return self.xp.einsum(
            "Qpq,Qrs->pqrs", self._factors, self._factors.conj()
        )

    reconstruct = to_dense
    get_eri = get

    def metadata(self) -> dict:
        return {
            "representation": "three-index-mo",
            "factorization": self.factorization,
            "threshold": self.threshold,
            "source": self.source,
            "resident_backend": self.xp.__name__,
            "naux": self.naux,
            "nocc": self.nocc,
            "nvir": self.nvir,
            "nmo": self.nmo,
            "dtype": str(self.dtype),
            "storage_nbytes": self.storage_nbytes,
        }


# Names commonly used by callers of integral backends.
DenseERIProvider = DenseIntegralProvider
ThreeIndexERIProvider = ThreeIndexIntegralProvider


__all__ = [
    "IntegralProvider",
    "CholeskyResult",
    "pivoted_cholesky",
    "DenseIntegralProvider",
    "ThreeIndexIntegralProvider",
    "MOThreeIndexIntegralProvider",
    "DenseERIProvider",
    "ThreeIndexERIProvider",
]
