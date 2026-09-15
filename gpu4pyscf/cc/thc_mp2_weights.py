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

"""Audit-only MP2 natural-occupation weights from THC amplitudes.

For real restricted-reference amplitudes, :mod:`gpu4pyscf.mp.mp2` defines the
correlation parts of the occupied and virtual one-particle density matrices as

``delta_occ[i,j] = -(2 t[k,i,a,b] t[k,j,a,b]``
``                         - t[k,i,a,b] t[k,j,b,a])``

``delta_vir[b,a] = 2 t[i,j,c,a] t[i,j,c,b]``
``                         - t[i,j,c,a] t[i,j,b,c]``.

This module evaluates those expressions for the amplitude-THC representation

``t[i,j,a,b] = y_occ[i,X] y_vir[a,X] T[X,Y]``
``                         * y_occ[j,Y] y_vir[b,Y]``

without constructing dense doubles.  The exchange contractions use an outer
THC-index loop, making the formal leading work ``O(N_thc**4)`` while every
non-output logical intermediate is at most ``(N_thc, N_thc)``.  This is a
preprocessing audit endpoint and is deliberately not connected to the
THC-RRCCSD production driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any

import numpy as np

from gpu4pyscf.cc.lowrank import RRDoubles
from gpu4pyscf.cc.thc_factorization import THCProjectorFactors


def _array_module(value: Any):
    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy" and isinstance(value, np.ndarray):
        return np
    if module == "cupy":
        import cupy

        if isinstance(value, cupy.ndarray):
            return cupy
    raise TypeError("MP2 weight tensors must be NumPy or CuPy arrays")


def _same_backend(reference: Any, *values: Any):
    xp = _array_module(reference)
    if any(_array_module(value) is not xp for value in values):
        raise TypeError("MP2 weight tensors must use one array backend")
    if xp is not np:
        device_id = int(reference.device.id)
        if any(int(value.device.id) != device_id for value in values):
            raise TypeError("MP2 weight CuPy tensors must use one device")
    return xp


def _one_real_dtype(*values: Any) -> np.dtype:
    dtypes = tuple(np.dtype(value.dtype) for value in values)
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError("MP2 weight tensors must have the same dtype")
    dtype = dtypes[0]
    if np.issubdtype(dtype, np.complexfloating):
        raise NotImplementedError(
            "factorized MP2 occupation weights currently require real RHF"
        )
    if not np.issubdtype(dtype, np.floating):
        raise TypeError("MP2 weight tensors must have a floating-point dtype")
    return dtype


def _validate_host_metric(name: str, value: Any, *, nonnegative: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"THC fit {name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"THC fit {name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"THC fit {name} must be non-negative")
    return result


def _validate_floor(value: Any, dtype: np.dtype) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("occupation_change_floor must be a real scalar")
    floor = float(value)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("occupation_change_floor must be finite and positive")
    with np.errstate(over="ignore", under="ignore"):
        represented = np.asarray(floor, dtype=dtype)
    if not bool(np.isfinite(represented)) or float(represented) <= 0.0:
        raise ValueError(
            "occupation_change_floor must remain finite and positive in the "
            "input dtype"
        )
    return floor


def _quadratic_form_rows(factor: Any, metric: Any, xp: Any):
    """Return ``factor @ metric @ factor.T`` with only a vector workspace."""

    size = int(factor.shape[0])
    result = xp.empty((size, size), dtype=factor.dtype)
    for row in range(size):
        projected = factor[row] @ metric
        result[row] = projected @ factor.T
    return result


def _add_quadratic_form_rows(
    target: Any,
    factor: Any,
    metric: Any,
    coefficient: float,
) -> None:
    """Accumulate a quadratic form without allocating another orbital block."""

    for row in range(int(factor.shape[0])):
        projected = factor[row] @ metric
        target[row] += coefficient * (projected @ factor.T)


def _symmetrize_in_place(matrix: Any) -> None:
    """Clean roundoff asymmetry without allocating another orbital matrix."""

    for row in range(int(matrix.shape[0]) - 1):
        average = (matrix[row, row + 1 :] + matrix[row + 1 :, row]) * 0.5
        matrix[row, row + 1 :] = average
        matrix[row + 1 :, row] = average


def _occupied_exchange_metric(
    core: Any,
    occupied_gram: Any,
    virtual_gram: Any,
    xp: Any,
):
    """Return the occupied exchange metric with ``O(X**2)`` workspace."""

    rank = int(core.shape[0])
    exchange = xp.empty_like(core)
    for z_index in range(rank):
        # scaled[X,W] = So[X,W] Sv[X,Z] T[W,Z]
        scaled = (
            occupied_gram
            * virtual_gram[:, z_index, None]
            * core[:, z_index][None, :]
        )
        mixed = core.T @ scaled
        exchange[:, z_index] = xp.sum(mixed * virtual_gram, axis=1)
    return exchange


def _virtual_exchange_metric(
    core: Any,
    occupied_gram: Any,
    virtual_gram: Any,
    xp: Any,
):
    """Return the virtual exchange metric with ``O(X**2)`` workspace."""

    rank = int(core.shape[0])
    exchange = xp.empty_like(core)
    for y_index in range(rank):
        # scaled[X,Z] = Sv[X,Z] T[X,Y] So[Y,Z]
        scaled = (
            virtual_gram
            * core[:, y_index, None]
            * occupied_gram[y_index, :][None, :]
        )
        mixed = occupied_gram.T @ scaled
        exchange[:, y_index] = xp.sum(mixed * core, axis=1)
    return exchange


@dataclass(frozen=True)
class FactorizedMP2NaturalOccupationWeights:
    """Natural-orbital data derived from factorized MP2 amplitudes.

    The columns of ``occupied_rotation`` and ``virtual_rotation`` are natural
    orbitals expressed in the input working-MO bases.  The occupation-change
    vectors and weights use the same column order.  Thus an occupied-virtual
    Cholesky tensor is transformed for a later weighted ERI-THC fit as
    ``einsum('ip,Aia,aq->Apq', occupied_rotation, lov, virtual_rotation)``.
    """

    delta_occ: Any
    delta_vir: Any
    occupied_occupation_changes: Any
    virtual_occupation_changes: Any
    occupied_rotation: Any
    virtual_rotation: Any
    occupied_weights: Any
    virtual_weights: Any
    occupation_change_floor: float
    particle_trace_error: float
    particle_trace_tolerance: float
    rr_core_symmetry_ratio: float
    factor_orthogonality_error: float
    verified_thc_weighted_fit_residual: float
    host_scalar_reads: int
    largest_intermediate_nbytes: int
    _source_mp2_doubles: RRDoubles = field(
        repr=False, compare=False
    )
    _source_amplitude_factors: THCProjectorFactors = field(
        repr=False, compare=False
    )

    @property
    def nocc(self) -> int:
        return int(self.delta_occ.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.delta_vir.shape[0])

    @property
    def rr_rank(self) -> int:
        return int(self._source_mp2_doubles.rank)

    @property
    def thc_rank(self) -> int:
        return int(self._source_amplitude_factors.thc_rank)

    @property
    def occupied_natural_occupations(self):
        """Return RHF spatial-orbital occupations, including the SCF value 2."""

        return self.occupied_occupation_changes + 2.0

    @property
    def virtual_natural_occupations(self):
        """Return RHF virtual occupations (equal to their MP2 changes)."""

        return self.virtual_occupation_changes

    @property
    def storage_nbytes(self) -> int:
        return int(
            self.delta_occ.nbytes
            + self.delta_vir.nbytes
            + self.occupied_occupation_changes.nbytes
            + self.virtual_occupation_changes.nbytes
            + self.occupied_rotation.nbytes
            + self.virtual_rotation.nbytes
            + self.occupied_weights.nbytes
            + self.virtual_weights.nbytes
        )

    def is_bound_to(
        self,
        mp2_doubles: RRDoubles,
        amplitude_factors: THCProjectorFactors,
    ) -> bool:
        """Return whether the two exact runtime source objects produced this result."""

        return (
            mp2_doubles is self._source_mp2_doubles
            and amplitude_factors is self._source_amplitude_factors
            and amplitude_factors.eigenvalues
            is mp2_doubles.projector.eigenvalues
        )

    def validate_binding(
        self,
        mp2_doubles: RRDoubles,
        amplitude_factors: THCProjectorFactors,
    ) -> None:
        """Fail if callers try to reuse weights with another RR/THC lifecycle."""

        if not self.is_bound_to(mp2_doubles, amplitude_factors):
            raise ValueError(
                "MP2 natural-occupation weights do not match the supplied "
                "RR doubles and THC fit identity"
            )

    def metadata(self) -> dict[str, Any]:
        factors = self._source_amplitude_factors
        doubles = self._source_mp2_doubles
        projector = doubles.projector
        xp = _array_module(self.delta_occ)
        return {
            "representation": "factorized-mp2-natural-occupation-weights",
            "gamma1_reference": "gpu4pyscf.mp.mp2._gamma1_intermediates",
            "occupation_density_source": (
                "amplitude-THC reconstruction of RR MP2 doubles"
            ),
            "exact_full_mp2_density": False,
            "spin_convention": "real-restricted-spatial-orbital",
            "nocc": self.nocc,
            "nvir": self.nvir,
            "rr_rank": self.rr_rank,
            "amplitude_thc_rank": self.thc_rank,
            "backend": xp.__name__,
            "dtype": np.dtype(self.delta_occ.dtype).name,
            "occupation_change_floor": float(self.occupation_change_floor),
            "weight_definition": "sqrt(max(abs(occupation_change),floor))",
            "weight_basis": "returned-mp2-natural-orbital-rotations",
            "rotation_convention": (
                "delta=rotation@diag(occupation_change)@rotation.T"
            ),
            "occupation_change_order": "descending-absolute-magnitude",
            "particle_trace_error": float(self.particle_trace_error),
            "particle_trace_tolerance": float(self.particle_trace_tolerance),
            "particle_trace_conserved": True,
            "rr_core_symmetry_ratio": float(self.rr_core_symmetry_ratio),
            "factor_orthogonality_error": float(
                self.factor_orthogonality_error
            ),
            "verified_thc_weighted_fit_residual": float(
                self.verified_thc_weighted_fit_residual
            ),
            "rr_cutoff": float(projector.cutoff),
            "rr_source": str(projector.source),
            "rr_full_dimension": int(projector.full_dimension),
            "thc_fit_tolerance": float(factors.fit_tolerance),
            "thc_weighted_fit_residual": float(
                factors.weighted_fit_residual
            ),
            "thc_unweighted_fit_residual": float(
                factors.unweighted_fit_residual
            ),
            "thc_fit_converged": bool(factors.converged),
            "thc_fit_rank_attempts": [
                int(rank) for rank in factors.rank_attempts
            ],
            "thc_fit_analytic_full_pair_endpoint": bool(
                factors.analytic_full_pair_endpoint
            ),
            "thc_fit_exact_pair_endpoint": bool(factors.exact_pair_endpoint),
            "thc_fit_exact_pair_roundoff_bound": float(
                factors.exact_pair_roundoff_bound
            ),
            "thc_fit_exact_pair_roundoff_error": float(
                factors.exact_pair_roundoff_error
            ),
            "thc_fit_exact_pair_roundoff_gate_passed": bool(
                factors.exact_pair_roundoff_gate_passed
            ),
            "provenance_binding": "runtime-object-identity",
            "rr_doubles_runtime_identity": id(doubles),
            "rr_projector_runtime_identity": id(projector),
            "thc_fit_runtime_identity": id(factors),
            "projector_eigenvalue_identity_verified": True,
            "depends_on_rr_cutoff": True,
            "depends_on_thc_fit": True,
            "formal_arithmetic_scaling": "O(N_thc^4)",
            "fit_identity_validation_scaling": (
                "O(nocc*nvir*rr_rank*N_thc)"
            ),
            "largest_logical_intermediate_shape": [
                self.thc_rank,
                self.thc_rank,
            ],
            "largest_logical_intermediate_scaling": "O(N_thc^2)",
            "largest_intermediate_nbytes": int(
                self.largest_intermediate_nbytes
            ),
            "largest_intermediate_scope": "non-output-logical-array",
            "actual_peak_memory_requires_profiling": True,
            "required_output_shapes": {
                "delta_occ": [self.nocc, self.nocc],
                "delta_vir": [self.nvir, self.nvir],
            },
            "dense_mp2_t2_materialized": False,
            "mp2_pair_matrix_materialized": False,
            "implicit_host_tensor_transfer": False,
            "gpu_validation_requires_transfer_counter": True,
            "host_scalar_reads": int(self.host_scalar_reads),
            "storage_nbytes": self.storage_nbytes,
            "source_arrays_borrowed_not_counted_in_storage": True,
            "audit_only": True,
            "production_enabled": False,
            "performance_eligible": False,
        }


def _validate_sources(
    mp2_doubles: RRDoubles,
    amplitude_factors: THCProjectorFactors,
    *,
    occupation_change_floor: Any,
    transfer_counter: Any,
):
    if not isinstance(mp2_doubles, RRDoubles):
        raise TypeError("mp2_doubles must be an RRDoubles instance")
    if not isinstance(amplitude_factors, THCProjectorFactors):
        raise TypeError(
            "amplitude_factors must be a THCProjectorFactors instance"
        )
    projector = mp2_doubles.projector
    arrays = (
        mp2_doubles.core,
        projector.vectors,
        projector.eigenvalues,
        amplitude_factors.y_occ,
        amplitude_factors.y_vir,
        amplitude_factors.tau,
        amplitude_factors.raw_tau,
        amplitude_factors.eigenvalues,
    )
    xp = _same_backend(arrays[0], *arrays[1:])
    dtype = _one_real_dtype(*arrays)
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU MP2 occupation weights require an explicit "
                "transfer_counter"
            )
        if not callable(getattr(transfer_counter, "record_d2h", None)):
            raise TypeError("transfer_counter must provide record_d2h")

    nocc, nvir = int(mp2_doubles.nocc), int(mp2_doubles.nvir)
    rr_rank = int(mp2_doubles.rank)
    thc_rank = int(amplitude_factors.thc_rank)
    if min(nocc, nvir, rr_rank, thc_rank) < 1:
        raise ValueError("MP2 RR and amplitude-THC dimensions must be positive")
    if amplitude_factors.nocc != nocc or amplitude_factors.nvir != nvir:
        raise ValueError(
            "RR doubles and amplitude factors have different orbital dimensions"
        )
    if amplitude_factors.rr_rank != rr_rank:
        raise ValueError(
            "RR doubles and amplitude factors have different RR ranks"
        )
    if rr_rank > thc_rank:
        raise ValueError("amplitude THC rank cannot be smaller than RR rank")
    expected_shapes = {
        "RR core": (rr_rank, rr_rank),
        "projector vectors": (nocc * nvir, rr_rank),
        "projector eigenvalues": (rr_rank,),
        "y_occ": (nocc, thc_rank),
        "y_vir": (nvir, thc_rank),
        "tau": (rr_rank, thc_rank),
        "raw_tau": (rr_rank, thc_rank),
        "factor eigenvalues": (rr_rank,),
    }
    for (name, shape), array in zip(expected_shapes.items(), arrays):
        if getattr(array, "shape", None) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if amplitude_factors.eigenvalues is not projector.eigenvalues:
        raise ValueError(
            "amplitude factors are not bound to the RR projector eigenvalue "
            "identity"
        )
    if not bool(amplitude_factors.converged):
        raise ValueError("amplitude THC factors must have a converged fit")

    fit_tolerance = _validate_host_metric(
        "fit_tolerance", amplitude_factors.fit_tolerance, nonnegative=True
    )
    weighted_residual = _validate_host_metric(
        "weighted_fit_residual",
        amplitude_factors.weighted_fit_residual,
        nonnegative=True,
    )
    _validate_host_metric(
        "unweighted_fit_residual",
        amplitude_factors.unweighted_fit_residual,
        nonnegative=True,
    )
    recorded_orthogonality = _validate_host_metric(
        "orthogonality_error",
        amplitude_factors.orthogonality_error,
        nonnegative=True,
    )
    exact_pair_endpoint = amplitude_factors.exact_pair_endpoint
    if weighted_residual > amplitude_factors.weighted_fit_gate_limit:
        raise ValueError(
            "amplitude THC weighted fit residual exceeds its recorded gate"
        )
    cutoff = float(projector.cutoff)
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("RR cutoff must be finite and non-negative")
    if not isinstance(projector.source, str) or not projector.source:
        raise ValueError("RR projector source must be a non-empty string")
    floor = _validate_floor(occupation_change_floor, dtype)

    host_scalar_reads = [0]

    def read_scalar(value: Any, operation: str) -> float:
        if xp is not np:
            transfer_counter.record_d2h(int(value.nbytes), operation=operation)
            host_scalar_reads[0] += 1
        return float(value.item()) if hasattr(value, "item") else float(value)

    finite = xp.asarray(True)
    for array in arrays:
        finite = xp.logical_and(finite, xp.all(xp.isfinite(array)))
    if not bool(read_scalar(finite, "thc_mp2_weights_input_finite")):
        raise ValueError("MP2 RR and amplitude-THC inputs must be finite")
    minimum_eigenvalue_magnitude = read_scalar(
        xp.min(xp.abs(projector.eigenvalues)),
        "thc_mp2_weights_minimum_projector_eigenvalue",
    )
    if minimum_eigenvalue_magnitude <= 0.0:
        raise ValueError(
            "amplitude THC provenance requires nonzero RR projector eigenvalues"
        )

    # Recompute the weighted fit residual against the exact projector object.
    # Row-wise accumulation avoids an (O*V,X) Khatri-Rao tensor while detecting
    # stale or mutated factor arrays that scalar fit metadata alone cannot bind.
    fit_numerator = xp.zeros((), dtype=dtype)
    fit_denominator = xp.zeros((), dtype=dtype)
    for i_index in range(nocc):
        for a_index in range(nvir):
            pair_factor = (
                amplitude_factors.y_occ[i_index]
                * amplitude_factors.y_vir[a_index]
            )
            fitted = pair_factor @ amplitude_factors.tau.T
            fitted *= projector.eigenvalues
            reference = (
                projector.vectors[i_index * nvir + a_index]
                * projector.eigenvalues
            )
            difference = fitted - reference
            fit_numerator += xp.vdot(difference, difference).real
            fit_denominator += xp.vdot(reference, reference).real
    verified_fit_residual = read_scalar(
        xp.sqrt(fit_numerator / fit_denominator),
        "thc_mp2_weights_fit_identity_residual",
    )
    fit_roundoff = (
        256.0
        * np.finfo(dtype).eps
        * max(1, nocc * nvir, rr_rank, thc_rank)
    )
    verified_fit_limit = (
        amplitude_factors.weighted_fit_gate_limit
        if exact_pair_endpoint
        else fit_tolerance + fit_roundoff
    )
    if verified_fit_residual > verified_fit_limit:
        raise ValueError(
            "amplitude THC factors do not satisfy their fit tolerance for "
            "the supplied RR projector identity"
        )

    core = mp2_doubles.core
    core_scale = xp.maximum(
        xp.max(xp.abs(core)), xp.asarray(1.0, dtype=dtype)
    )
    core_symmetry_ratio = read_scalar(
        xp.max(xp.abs(core - core.T)) / core_scale,
        "thc_mp2_weights_core_symmetry",
    )
    symmetry_tolerance = (
        256.0 * np.finfo(dtype).eps * max(1, rr_rank)
    )
    if core_symmetry_ratio > symmetry_tolerance:
        raise ValueError("MP2 RR core must be symmetric")

    occupied_gram = amplitude_factors.y_occ.T @ amplitude_factors.y_occ
    virtual_gram = amplitude_factors.y_vir.T @ amplitude_factors.y_vir
    pair_gram = occupied_gram * virtual_gram
    overlap = amplitude_factors.tau @ pair_gram @ amplitude_factors.tau.T
    orthogonality_error = read_scalar(
        xp.linalg.norm(overlap - xp.eye(rr_rank, dtype=dtype)),
        "thc_mp2_weights_factor_orthogonality",
    )
    orthogonality_tolerance = max(
        1.0e-8,
        4096.0 * np.finfo(dtype).eps * max(1, rr_rank, thc_rank),
    )
    if (
        recorded_orthogonality > orthogonality_tolerance
        or orthogonality_error > orthogonality_tolerance
    ):
        raise ValueError(
            "amplitude THC factors fail the semi-unitary projector gate"
        )
    if (
        exact_pair_endpoint
        and (
            not amplitude_factors.exact_pair_roundoff_gate_passed
            or orthogonality_error
            > amplitude_factors.exact_pair_roundoff_bound
        )
    ):
        raise ValueError(
            "analytic full-pair amplitude THC factors exceed their roundoff gate"
        )
    return (
        xp,
        dtype,
        nocc,
        nvir,
        rr_rank,
        thc_rank,
        floor,
        occupied_gram,
        virtual_gram,
        core_symmetry_ratio,
        orthogonality_error,
        verified_fit_residual,
        host_scalar_reads,
        read_scalar,
    )


def build_factorized_mp2_natural_occupation_weights(
    mp2_doubles: RRDoubles,
    amplitude_factors: THCProjectorFactors,
    *,
    occupation_change_floor: float,
    transfer_counter: Any = None,
) -> FactorizedMP2NaturalOccupationWeights:
    """Build MP2 natural-orbital rotations and strictly positive fit weights.

    ``mp2_doubles.core`` is transformed through ``amplitude_factors.tau`` to
    obtain the amplitude-THC core.  Exact object identity between the factor
    eigenvalue array and the RR projector eigenvalue array is required; this is
    the lifecycle token established by ``fit_weighted_thc_projector``.  The
    returned result additionally binds the exact doubles and factor objects so
    stale weights cannot be silently reused with a later fit.
    """

    (
        xp,
        dtype,
        nocc,
        nvir,
        _rr_rank,
        thc_rank,
        floor,
        occupied_gram,
        virtual_gram,
        core_symmetry_ratio,
        orthogonality_error,
        verified_fit_residual,
        host_scalar_reads,
        read_scalar,
    ) = _validate_sources(
        mp2_doubles,
        amplitude_factors,
        occupation_change_floor=occupation_change_floor,
        transfer_counter=transfer_counter,
    )

    symmetric_rr_core = (mp2_doubles.core + mp2_doubles.core.T) * 0.5
    core = (
        amplitude_factors.tau.T
        @ symmetric_rr_core
        @ amplitude_factors.tau
    )
    core = (core + core.T) * 0.5
    pair_gram = occupied_gram * virtual_gram
    direct = core.T @ pair_gram @ core

    delta_occ = _quadratic_form_rows(
        amplitude_factors.y_occ,
        direct * virtual_gram,
        xp,
    )
    delta_occ *= -2.0
    occupied_exchange = _occupied_exchange_metric(
        core, occupied_gram, virtual_gram, xp
    )
    _add_quadratic_form_rows(
        delta_occ,
        amplitude_factors.y_occ,
        occupied_exchange,
        1.0,
    )

    delta_vir = _quadratic_form_rows(
        amplitude_factors.y_vir,
        direct * occupied_gram,
        xp,
    )
    delta_vir *= 2.0
    virtual_exchange = _virtual_exchange_metric(
        core, occupied_gram, virtual_gram, xp
    )
    _add_quadratic_form_rows(
        delta_vir,
        amplitude_factors.y_vir,
        virtual_exchange,
        -1.0,
    )

    # The analytic blocks are symmetric.  Remove only floating-point contraction
    # noise before diagonalization and retain the cleaned matrices in the result.
    _symmetrize_in_place(delta_occ)
    _symmetrize_in_place(delta_vir)

    trace_occ = xp.trace(delta_occ)
    trace_vir = xp.trace(delta_vir)
    trace_error = read_scalar(
        xp.abs(trace_occ + trace_vir),
        "thc_mp2_weights_particle_trace_error",
    )
    trace_scale = read_scalar(
        xp.maximum(
            xp.maximum(xp.abs(trace_occ), xp.abs(trace_vir)),
            xp.asarray(1.0, dtype=dtype),
        ),
        "thc_mp2_weights_particle_trace_scale",
    )
    trace_tolerance = (
        512.0
        * np.finfo(dtype).eps
        * max(1, nocc, nvir, thc_rank)
        * trace_scale
    )
    if trace_error > trace_tolerance:
        raise FloatingPointError(
            "factorized MP2 density blocks violate particle-number trace "
            "conservation"
        )

    occupied_changes, occupied_rotation = xp.linalg.eigh(delta_occ)
    virtual_changes, virtual_rotation = xp.linalg.eigh(delta_vir)
    occupied_order = xp.argsort(xp.abs(occupied_changes))[::-1]
    virtual_order = xp.argsort(xp.abs(virtual_changes))[::-1]
    occupied_changes = occupied_changes[occupied_order]
    virtual_changes = virtual_changes[virtual_order]
    occupied_rotation = occupied_rotation[:, occupied_order]
    virtual_rotation = virtual_rotation[:, virtual_order]

    floor_value = xp.asarray(floor, dtype=dtype)
    occupied_weights = xp.sqrt(xp.maximum(xp.abs(occupied_changes), floor_value))
    virtual_weights = xp.sqrt(xp.maximum(xp.abs(virtual_changes), floor_value))
    finite = (
        xp.all(xp.isfinite(delta_occ))
        & xp.all(xp.isfinite(delta_vir))
        & xp.all(xp.isfinite(occupied_changes))
        & xp.all(xp.isfinite(virtual_changes))
        & xp.all(xp.isfinite(occupied_rotation))
        & xp.all(xp.isfinite(virtual_rotation))
        & xp.all(xp.isfinite(occupied_weights))
        & xp.all(xp.isfinite(virtual_weights))
    )
    if not bool(read_scalar(finite, "thc_mp2_weights_output_finite")):
        raise FloatingPointError(
            "factorized MP2 natural-occupation calculation produced "
            "non-finite values"
        )
    minimum_weight = xp.minimum(
        xp.min(occupied_weights), xp.min(virtual_weights)
    )
    if read_scalar(
        minimum_weight, "thc_mp2_weights_minimum_weight"
    ) <= 0.0:
        raise FloatingPointError(
            "occupation-change floor did not produce positive weights"
        )

    return FactorizedMP2NaturalOccupationWeights(
        delta_occ=delta_occ,
        delta_vir=delta_vir,
        occupied_occupation_changes=occupied_changes,
        virtual_occupation_changes=virtual_changes,
        occupied_rotation=occupied_rotation,
        virtual_rotation=virtual_rotation,
        occupied_weights=occupied_weights,
        virtual_weights=virtual_weights,
        occupation_change_floor=floor,
        particle_trace_error=trace_error,
        particle_trace_tolerance=trace_tolerance,
        rr_core_symmetry_ratio=core_symmetry_ratio,
        factor_orthogonality_error=orthogonality_error,
        verified_thc_weighted_fit_residual=verified_fit_residual,
        host_scalar_reads=host_scalar_reads[0],
        largest_intermediate_nbytes=int(thc_rank * thc_rank * dtype.itemsize),
        _source_mp2_doubles=mp2_doubles,
        _source_amplitude_factors=amplitude_factors,
    )


__all__ = [
    "FactorizedMP2NaturalOccupationWeights",
    "build_factorized_mp2_natural_occupation_weights",
]
