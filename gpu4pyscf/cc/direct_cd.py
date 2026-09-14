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

"""Matrix-free pivoted Cholesky for direct AO-pair integral columns.

The production integral engine must provide the diagonal and selected columns
of the positive-semidefinite AO-pair ERI matrix. This module performs the
Cholesky algebra without ever requesting or allocating the full pair matrix.
The small dense provider is a test oracle only.  The companion GINT adapter
supplies a bounded-memory correctness path; its selected-pair production
kernel remains a separate performance gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


@runtime_checkable
class AOPairColumnProvider(Protocol):
    """Source of one AO-pair ERI diagonal and arbitrary matrix columns."""

    @property
    def dimension(self) -> int: ...

    @property
    def materializes_pair_matrix(self) -> bool: ...

    def diagonal(self) -> Any: ...

    def column(self, pivot: int) -> Any: ...


class DenseAOPairColumnProvider:
    """Dense validation oracle; forbidden for production benchmark records."""

    def __init__(self, pair_matrix: Any) -> None:
        if getattr(pair_matrix, "ndim", None) != 2:
            raise ValueError("pair_matrix must be rank two")
        if pair_matrix.shape[0] != pair_matrix.shape[1]:
            raise ValueError("pair_matrix must be square")
        self.matrix = pair_matrix
        self.column_calls = 0
        self.batch_calls = 0

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def materializes_pair_matrix(self) -> bool:
        return True

    def diagonal(self):
        return self.matrix.diagonal().real.copy()

    def column(self, pivot: int):
        self.column_calls += 1
        return self.matrix[:, int(pivot)].copy()

    def columns(self, pivots: Any):
        indices = np.asarray(tuple(pivots), dtype=np.int64)
        if indices.ndim != 1 or indices.size < 1:
            raise ValueError("pivots must be a non-empty vector")
        self.batch_calls += 1
        self.column_calls += int(indices.size)
        xp = _array_module(self.matrix)
        backend_indices = indices if xp is np else xp.asarray(indices)
        return self.matrix[:, backend_indices].T.copy()


@dataclass(frozen=True)
class DirectCholeskyResult:
    """Direct AO-pair Cholesky factors and a rigorous PSD error bound."""

    pair_factors: Any
    threshold: float
    residual_diagonal: float
    pivots: tuple[int, ...]
    capped: bool
    provider_materializes_pair_matrix: bool
    provider_column_calls: int
    host_scalar_reads: int
    provider_batch_calls: int = 0
    column_batch_size: int = 1
    requested_column_batch_size: int = 1

    @property
    def rank(self) -> int:
        return int(self.pair_factors.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.pair_factors.shape[0])

    @property
    def max_element_error_bound(self) -> float:
        return float(self.residual_diagonal)

    def unpack_symmetric(
        self,
        n_orbitals: int,
        pair_i: Any,
        pair_j: Any,
    ):
        """Return resident ``L[Q,p,q]`` factors for real symmetric AO pairs."""

        xp = _array_module(self.pair_factors)
        pair_i = xp.asarray(pair_i)
        pair_j = xp.asarray(pair_j)
        if pair_i.shape != (self.dimension,) or pair_j.shape != (self.dimension,):
            raise ValueError("pair indices must contain one entry per pair")
        factors = xp.zeros(
            (self.rank, int(n_orbitals), int(n_orbitals)),
            dtype=self.pair_factors.dtype,
        )
        values = self.pair_factors.T
        factors[:, pair_i, pair_j] = values
        factors[:, pair_j, pair_i] = values
        return factors

    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": "matrix-free-direct-pair-pivoted-cholesky",
            "dimension": self.dimension,
            "rank": self.rank,
            "threshold": float(self.threshold),
            "residual_diagonal": float(self.residual_diagonal),
            "max_element_error_bound": self.max_element_error_bound,
            "pivots": list(self.pivots),
            "capped": bool(self.capped),
            "provider_materializes_pair_matrix": bool(
                self.provider_materializes_pair_matrix
            ),
            "provider_column_calls": int(self.provider_column_calls),
            "provider_batch_calls": int(self.provider_batch_calls),
            "column_batch_size": int(self.column_batch_size),
            "requested_column_batch_size": int(
                self.requested_column_batch_size
            ),
            "host_scalar_reads": int(self.host_scalar_reads),
        }


def pivoted_cholesky_from_columns(
    provider: AOPairColumnProvider,
    *,
    threshold: float,
    max_rank: Optional[int] = None,
    transfer_counter: Any = None,
    column_batch_size: int = 1,
) -> DirectCholeskyResult:
    """Factor a PSD AO-pair matrix by requesting only selected columns."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    dimension = int(provider.dimension)
    if dimension < 1:
        raise ValueError("provider dimension must be positive")
    if max_rank is not None and (
        int(max_rank) != max_rank or not 1 <= int(max_rank) <= dimension
    ):
        raise ValueError("max_rank must lie in [1, provider.dimension]")
    if (
        isinstance(column_batch_size, (bool, np.bool_))
        or not isinstance(column_batch_size, (int, np.integer))
        or int(column_batch_size) < 1
    ):
        raise ValueError("column_batch_size must be a positive integer")
    requested_column_batch_size = int(column_batch_size)
    provider_batch_limit = getattr(provider, "max_batch_size", None)
    if provider_batch_limit is not None:
        if (
            isinstance(provider_batch_limit, (bool, np.bool_))
            or not isinstance(provider_batch_limit, (int, np.integer))
            or int(provider_batch_limit) < 1
        ):
            raise ValueError("provider.max_batch_size must be a positive integer")
        column_batch_size = min(
            requested_column_batch_size, int(provider_batch_limit)
        )
    else:
        column_batch_size = requested_column_batch_size
    if column_batch_size > 1 and not callable(getattr(provider, "columns", None)):
        raise TypeError(
            "column_batch_size > 1 requires provider.columns(pivots)"
        )
    cap = dimension if max_rank is None else int(max_rank)
    diagonal = provider.diagonal()
    if getattr(diagonal, "shape", None) != (dimension,):
        raise ValueError("provider diagonal has the wrong shape")
    xp = _array_module(diagonal)
    if not np.issubdtype(np.dtype(diagonal.dtype), np.floating):
        raise TypeError("provider diagonal must use a real floating-point dtype")
    if xp is not np and transfer_counter is None:
        raise ValueError(
            "GPU direct Cholesky requires an explicit transfer_counter for "
            "host pivot control"
        )
    finite_value = xp.all(xp.isfinite(diagonal))
    minimum_value = xp.min(diagonal)
    scale_value = xp.max(xp.abs(diagonal))
    if xp is not np:
        transfer_counter.record_d2h(
            int(finite_value.nbytes)
            + int(minimum_value.nbytes)
            + int(scale_value.nbytes),
            count=3,
            operation="direct_cd_diagonal_validation_scalars",
        )
    if not bool(finite_value.item()):
        raise ValueError("provider diagonal contains NaN or infinite values")
    if float(minimum_value.item()) < 0:
        raise ValueError("AO-pair ERI diagonal contains a negative value")
    diagonal_scale = float(scale_value.item())

    if column_batch_size > 1:
        return _pivoted_cholesky_batched(
            provider,
            diagonal=diagonal,
            xp=xp,
            dimension=dimension,
            cap=cap,
            threshold=float(threshold),
            diagonal_scale=diagonal_scale,
            transfer_counter=transfer_counter,
            initial_host_scalar_reads=3 if xp is not np else 0,
            column_batch_size=column_batch_size,
            requested_column_batch_size=requested_column_batch_size,
        )

    capacity = min(cap, 32)
    factors = xp.zeros((dimension, capacity), dtype=diagonal.dtype)
    residual = diagonal.copy()
    pivots: list[int] = []
    rank = 0
    host_scalar_reads = 3 if xp is not np else 0
    residual_max = float("inf")
    while rank < cap:
        device_pivot = xp.argmax(residual)
        device_max = xp.max(residual)
        if xp is not np:
            transfer_counter.record_d2h(
                int(device_pivot.nbytes) + int(device_max.nbytes),
                count=2,
                operation="direct_cd_pivot_control",
            )
            host_scalar_reads += 2
        pivot = int(device_pivot.item())
        residual_max = float(device_max.item())
        if not np.isfinite(residual_max):
            raise ValueError("provider diagonal or residual contains NaN/Inf")
        if residual_max < 0:
            raise ValueError("AO-pair ERI matrix is not positive semidefinite")
        if residual_max <= threshold:
            break
        column = provider.column(pivot)
        if getattr(column, "shape", None) != (dimension,):
            raise ValueError("provider column has the wrong shape")
        if _array_module(column) is not xp:
            raise TypeError("provider diagonal and columns must use one backend")
        if not np.issubdtype(np.dtype(column.dtype), np.floating):
            raise TypeError("provider columns must use a real floating-point dtype")
        finite_column = xp.all(xp.isfinite(column))
        if xp is not np:
            transfer_counter.record_d2h(
                int(finite_column.nbytes),
                operation="direct_cd_column_finite_check",
            )
            host_scalar_reads += 1
        if not bool(finite_column.item()):
            raise ValueError("provider column contains NaN or infinite values")
        if rank:
            column = column - factors[:, :rank] @ factors[pivot, :rank].conj()
        corrected_pivot = column[pivot]
        if xp is not np:
            transfer_counter.record_d2h(
                int(corrected_pivot.nbytes),
                operation="direct_cd_corrected_pivot",
            )
            host_scalar_reads += 1
        corrected_pivot_value = float(corrected_pivot.item())
        consistency_tolerance = (
            1000.0
            * np.finfo(np.dtype(diagonal.dtype)).eps
            * max(1.0, diagonal_scale, residual_max)
        )
        if abs(corrected_pivot_value - residual_max) > consistency_tolerance:
            raise ValueError(
                "provider column is inconsistent with the AO-pair diagonal "
                f"at pivot {pivot}: {corrected_pivot_value:.16e} != "
                f"{residual_max:.16e}"
            )
        pivot_scale = xp.sqrt(residual[pivot])
        factors[:, rank] = column / pivot_scale
        next_residual = residual - xp.abs(factors[:, rank]) ** 2
        minimum_residual = xp.min(next_residual)
        if xp is not np:
            transfer_counter.record_d2h(
                int(minimum_residual.nbytes),
                operation="direct_cd_residual_minimum",
            )
            host_scalar_reads += 1
        minimum_residual_value = float(minimum_residual.item())
        if minimum_residual_value < -consistency_tolerance:
            raise ValueError(
                "provider columns do not define a positive-semidefinite "
                "AO-pair ERI matrix: residual diagonal reached "
                f"{minimum_residual_value:.3e}"
            )
        residual = xp.maximum(next_residual, 0)
        pivots.append(pivot)
        rank += 1
        if rank == capacity and rank < cap:
            new_capacity = min(cap, max(capacity + 1, capacity * 2))
            grown = xp.zeros((dimension, new_capacity), dtype=factors.dtype)
            grown[:, :rank] = factors[:, :rank]
            factors = grown
            capacity = new_capacity

    if rank == cap:
        device_max = xp.max(residual)
        if xp is not np:
            transfer_counter.record_d2h(
                int(device_max.nbytes),
                operation="direct_cd_final_residual_maximum",
            )
            host_scalar_reads += 1
        residual_max = float(device_max.item())
    capped = bool(rank == cap and residual_max > threshold)
    return DirectCholeskyResult(
        pair_factors=factors[:, :rank],
        threshold=float(threshold),
        residual_diagonal=residual_max,
        pivots=tuple(pivots),
        capped=capped,
        provider_materializes_pair_matrix=bool(
            provider.materializes_pair_matrix
        ),
        provider_column_calls=int(getattr(provider, "column_calls", rank)),
        host_scalar_reads=host_scalar_reads,
        provider_batch_calls=int(getattr(provider, "batch_calls", rank)),
        column_batch_size=1,
        requested_column_batch_size=requested_column_batch_size,
    )


def _pivoted_cholesky_batched(
    provider: AOPairColumnProvider,
    *,
    diagonal: Any,
    xp: Any,
    dimension: int,
    cap: int,
    threshold: float,
    diagonal_scale: float,
    transfer_counter: Any,
    initial_host_scalar_reads: int,
    column_batch_size: int,
    requested_column_batch_size: int,
) -> DirectCholeskyResult:
    """Blocked pivoted Cholesky using independently evaluated ERI columns.

    A batch contains the largest residual-diagonal candidates at the start of
    an outer step.  Their exact matrix columns are independent of earlier
    pivots and can therefore be evaluated together.  Candidates are accepted
    sequentially against the updated residual; a redundant candidate is
    skipped.  The global residual maximum is re-evaluated before every batch,
    preserving the same rigorous diagonal stopping condition as the scalar
    path.
    """

    capacity = min(cap, 32)
    factors = xp.zeros((dimension, capacity), dtype=diagonal.dtype)
    residual = diagonal.copy()
    pivots: list[int] = []
    rank = 0
    host_scalar_reads = int(initial_host_scalar_reads)
    residual_max = float("inf")
    while rank < cap:
        device_max = xp.max(residual)
        if xp is not np:
            transfer_counter.record_d2h(
                int(device_max.nbytes),
                operation="direct_cd_residual_maximum",
            )
            host_scalar_reads += 1
        residual_max = float(device_max.item())
        if not np.isfinite(residual_max):
            raise ValueError("provider diagonal or residual contains NaN/Inf")
        if residual_max < 0:
            raise ValueError("AO-pair ERI matrix is not positive semidefinite")
        if residual_max <= threshold:
            break

        request_count = min(column_batch_size, cap - rank, dimension)
        # The selected GINT provider deliberately refuses an allocation shaped
        # like the full pair matrix.  A one-dimensional endpoint is handled by
        # the scalar implementation before this helper is reached in practice.
        if dimension > 1:
            request_count = min(request_count, dimension - 1)
        if request_count < 1:
            raise RuntimeError("blocked Cholesky could not select a pivot")
        split = dimension - request_count
        candidates_device = xp.argpartition(residual, split)[split:]
        candidates_device = candidates_device[
            xp.argsort(residual[candidates_device])[::-1]
        ]
        if xp is np:
            candidates = np.asarray(candidates_device, dtype=np.int64)
        else:
            transfer_counter.record_d2h(
                int(candidates_device.nbytes),
                operation="direct_cd_candidate_indices",
            )
            host_scalar_reads += 1
            candidates = np.asarray(candidates_device.get(), dtype=np.int64)
        batch = provider.columns(tuple(int(value) for value in candidates))
        if getattr(batch, "shape", None) != (request_count, dimension):
            raise ValueError("provider column batch has the wrong shape")
        if _array_module(batch) is not xp:
            raise TypeError("provider diagonal and columns must use one backend")
        if not np.issubdtype(np.dtype(batch.dtype), np.floating):
            raise TypeError("provider columns must use a real floating-point dtype")
        finite_batch = xp.all(xp.isfinite(batch))
        if xp is not np:
            transfer_counter.record_d2h(
                int(finite_batch.nbytes),
                operation="direct_cd_batch_finite_check",
            )
            host_scalar_reads += 1
        if not bool(finite_batch.item()):
            raise ValueError("provider column contains NaN or infinite values")

        for offset, raw_pivot in enumerate(candidates):
            if rank >= cap:
                break
            pivot = int(raw_pivot)
            pivot_residual = residual[pivot]
            if xp is not np:
                transfer_counter.record_d2h(
                    int(pivot_residual.nbytes),
                    operation="direct_cd_candidate_residual",
                )
                host_scalar_reads += 1
            pivot_residual_value = float(pivot_residual.item())
            if pivot_residual_value <= threshold:
                continue
            column = batch[offset]
            if rank:
                column = column - factors[:, :rank] @ factors[pivot, :rank].conj()
            corrected_pivot = column[pivot]
            if xp is not np:
                transfer_counter.record_d2h(
                    int(corrected_pivot.nbytes),
                    operation="direct_cd_corrected_pivot",
                )
                host_scalar_reads += 1
            corrected_pivot_value = float(corrected_pivot.item())
            consistency_tolerance = (
                1000.0
                * np.finfo(np.dtype(diagonal.dtype)).eps
                * max(1.0, diagonal_scale, pivot_residual_value)
            )
            if (
                abs(corrected_pivot_value - pivot_residual_value)
                > consistency_tolerance
            ):
                raise ValueError(
                    "provider column is inconsistent with the AO-pair diagonal "
                    f"at pivot {pivot}: {corrected_pivot_value:.16e} != "
                    f"{pivot_residual_value:.16e}"
                )
            if rank == capacity:
                new_capacity = min(cap, max(capacity + 1, capacity * 2))
                grown = xp.zeros((dimension, new_capacity), dtype=factors.dtype)
                grown[:, :rank] = factors[:, :rank]
                factors = grown
                capacity = new_capacity
            factors[:, rank] = column / xp.sqrt(residual[pivot])
            next_residual = residual - xp.abs(factors[:, rank]) ** 2
            minimum_residual = xp.min(next_residual)
            if xp is not np:
                transfer_counter.record_d2h(
                    int(minimum_residual.nbytes),
                    operation="direct_cd_residual_minimum",
                )
                host_scalar_reads += 1
            minimum_residual_value = float(minimum_residual.item())
            if minimum_residual_value < -consistency_tolerance:
                raise ValueError(
                    "provider columns do not define a positive-semidefinite "
                    "AO-pair ERI matrix: residual diagonal reached "
                    f"{minimum_residual_value:.3e}"
                )
            residual = xp.maximum(next_residual, 0)
            pivots.append(pivot)
            rank += 1

    if rank == cap:
        device_max = xp.max(residual)
        if xp is not np:
            transfer_counter.record_d2h(
                int(device_max.nbytes),
                operation="direct_cd_final_residual_maximum",
            )
            host_scalar_reads += 1
        residual_max = float(device_max.item())
    capped = bool(rank == cap and residual_max > threshold)
    return DirectCholeskyResult(
        pair_factors=factors[:, :rank],
        threshold=float(threshold),
        residual_diagonal=residual_max,
        pivots=tuple(pivots),
        capped=capped,
        provider_materializes_pair_matrix=bool(provider.materializes_pair_matrix),
        provider_column_calls=int(getattr(provider, "column_calls", rank)),
        host_scalar_reads=host_scalar_reads,
        provider_batch_calls=int(getattr(provider, "batch_calls", 0)),
        column_batch_size=column_batch_size,
        requested_column_batch_size=requested_column_batch_size,
    )


__all__ = [
    "AOPairColumnProvider",
    "DenseAOPairColumnProvider",
    "DirectCholeskyResult",
    "pivoted_cholesky_from_columns",
]
