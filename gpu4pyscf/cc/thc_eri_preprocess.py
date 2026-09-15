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

"""Audit-only preprocessing from RR MP2 amplitudes to working-basis ERI-THC.

The preprocessing path is explicit about its three coordinate systems:

1. Factorized RR/THC MP2 amplitudes produce approximate MP2 natural-orbital
   occupation changes and rotations.
2. The direct-CD tensor ``L[A,i,a]`` is rotated from the working MO basis into
   those occupied and virtual natural-orbital bases and fitted with the
   occupation-change weights.
3. The fitted occupied and virtual ERI factors are rotated back to the working
   MO basis used by the amplitude factors and Algorithms 4--7.

No dense doubles or four-index ERI is constructed.  The transformed
three-index Cholesky tensor is materialized because it is the declared input to
the existing weighted ERI-THC fitter.  This module is an audit integration seam
and is not connected to the production THC-RRCCSD driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any

import numpy as np

from gpu4pyscf.cc.lowrank import RRDoubles
from gpu4pyscf.cc.thc_eri import ERITHCFactors
from gpu4pyscf.cc.thc_eri_factorization import (
    ERITHCFitResult,
    exact_eri_thc_from_cholesky,
    fit_weighted_eri_thc,
)
from gpu4pyscf.cc.thc_factorization import THCProjectorFactors
from gpu4pyscf.cc.thc_mp2_weights import (
    FactorizedMP2NaturalOccupationWeights,
    build_factorized_mp2_natural_occupation_weights,
)


def _array_module(value: Any):
    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy" and isinstance(value, np.ndarray):
        return np
    if module == "cupy":
        import cupy

        if isinstance(value, cupy.ndarray):
            return cupy
    raise TypeError("ERI-THC preprocessing tensors must be NumPy or CuPy arrays")


def _same_backend(reference: Any, *values: Any):
    xp = _array_module(reference)
    if any(_array_module(value) is not xp for value in values):
        raise TypeError("ERI-THC preprocessing tensors must use one array backend")
    if xp is not np:
        device_id = int(reference.device.id)
        if any(int(value.device.id) != device_id for value in values):
            raise TypeError("ERI-THC preprocessing CuPy tensors must use one device")
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


def _integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _nonnegative_real(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _transform_cholesky_to_natural_orbitals(
    lov: Any,
    occupied_rotation: Any,
    virtual_rotation: Any,
    *,
    auxiliary_block_size: int,
    xp: Any,
):
    transformed = xp.empty_like(lov)
    largest_block_nbytes = 0
    for start in range(0, int(lov.shape[0]), auxiliary_block_size):
        stop = min(start + auxiliary_block_size, int(lov.shape[0]))
        half = xp.einsum(
            "ip,Aia->Apa", occupied_rotation, lov[start:stop]
        )
        transformed[start:stop] = xp.einsum(
            "Apa,aq->Apq", half, virtual_rotation
        )
        largest_block_nbytes = max(largest_block_nbytes, int(half.nbytes))
    return transformed, largest_block_nbytes


@dataclass(frozen=True)
class WeightedERITHCPreprocessResult:
    """Auditable weighted ERI-THC preprocessing result.

    ``fit`` remains in the MP2 natural-orbital basis.  ``factors`` contains the
    same ERI core with its occupied and virtual factors rotated back to the
    working MO basis, so it can be supplied directly to Algorithms 4--7.
    """

    factors: ERITHCFactors
    fit: ERITHCFitResult
    weights: FactorizedMP2NaturalOccupationWeights
    exact_pair_endpoint: bool
    transform_auxiliary_block_size: int
    preprocess_host_scalar_reads: int
    host_scalar_reads: int
    largest_intermediate_nbytes: int
    _source_mp2_doubles: RRDoubles = field(repr=False, compare=False)
    _source_amplitude_factors: THCProjectorFactors = field(
        repr=False, compare=False
    )
    _source_lov: Any = field(repr=False, compare=False)

    @property
    def naux(self) -> int:
        return int(self.fit.xi.shape[0])

    @property
    def nocc(self) -> int:
        return int(self.factors.nocc)

    @property
    def nvir(self) -> int:
        return int(self.factors.nvir)

    @property
    def rank(self) -> int:
        return int(self.factors.rank)

    @property
    def natural_orbital_factors(self) -> ERITHCFactors:
        return self.fit.factors

    @property
    def xi(self):
        return self.fit.xi

    @property
    def storage_nbytes(self) -> int:
        # The working factors share the fit's core, so count only their two new
        # orbital factor arrays in addition to the fit and weight results.
        return int(
            self.weights.storage_nbytes
            + self.fit.storage_nbytes
            + self.factors.x_occ.nbytes
            + self.factors.x_vir.nbytes
        )

    def reconstruct_working_cholesky(self):
        """Materialize the fitted three-index tensor in the working MO basis."""

        xp = _array_module(self.fit.xi)
        return xp.einsum(
            "iI,aI,AI->Aia",
            self.factors.x_occ,
            self.factors.x_vir,
            self.fit.xi,
        )

    def is_bound_to(
        self,
        mp2_doubles: RRDoubles,
        amplitude_factors: THCProjectorFactors,
        lov: Any,
    ) -> bool:
        """Return whether these exact runtime inputs produced the result."""

        return (
            mp2_doubles is self._source_mp2_doubles
            and amplitude_factors is self._source_amplitude_factors
            and lov is self._source_lov
            and self.weights.is_bound_to(mp2_doubles, amplitude_factors)
            and self.fit.occupied_weights is self.weights.occupied_weights
            and self.fit.virtual_weights is self.weights.virtual_weights
            and self.factors.core is self.fit.factors.core
        )

    def validate_binding(
        self,
        mp2_doubles: RRDoubles,
        amplitude_factors: THCProjectorFactors,
        lov: Any,
    ) -> None:
        """Fail if the result is reused with a different RR/THC/CD lifecycle."""

        if not self.is_bound_to(mp2_doubles, amplitude_factors, lov):
            raise ValueError(
                "weighted ERI-THC preprocessing result does not match the "
                "supplied RR doubles, amplitude THC fit, and Cholesky identity"
            )

    def metadata(self) -> dict[str, Any]:
        doubles = self._source_mp2_doubles
        amplitude_factors = self._source_amplitude_factors
        xp = _array_module(self.factors.x_occ)
        full_pair_rank = self.rank == self.nocc * self.nvir
        return {
            "representation": "weighted-eri-thc-preprocess",
            "pipeline": [
                "factorized-amplitude-thc-mp2-gamma1",
                "working-mo-to-mp2-natural-orbital-rotation",
                "weighted-cholesky-cp-factorization",
                "eri-factors-back-rotation-to-working-mo",
            ],
            "density_source": "amplitude-thc-approximate-mp2-gamma1",
            "density_is_exact_full_mp2": False,
            "depends_on_rr_cutoff": True,
            "depends_on_amplitude_thc_fit": True,
            "naux": self.naux,
            "nocc": self.nocc,
            "nvir": self.nvir,
            "eri_thc_rank": self.rank,
            "backend": xp.__name__,
            "dtype": np.dtype(self.factors.x_occ.dtype).name,
            "basis_flow": "working-mo->mp2-natural-orbitals->working-mo",
            "forward_rotation": "L_no[A,p,q]=R_occ[i,p]*L[A,i,a]*R_vir[a,q]",
            "back_rotation": "x_work=R*x_natural-orbital",
            "fit_coordinate_space": "mp2-natural-orbital",
            "output_coordinate_space": "working-mo",
            "algorithm_consumers": [4, 5, 6, 7],
            "fit_method": (
                "analytic-exact-pair"
                if self.exact_pair_endpoint
                else "weighted-cp-als"
            ),
            "als_controls_applied": not self.exact_pair_endpoint,
            "analytic_exact_pair_endpoint": bool(self.exact_pair_endpoint),
            "full_pair_rank": bool(full_pair_rank),
            "full_rank_als": bool(full_pair_rank and not self.exact_pair_endpoint),
            "full_rank_als_claimed_as_analytic_exact": False,
            "occupation_change_floor": float(
                self.weights.occupation_change_floor
            ),
            "fit_tolerance": float(self.fit.fit_tolerance),
            "weighted_fit_residual": float(self.fit.weighted_fit_residual),
            "unweighted_fit_residual": float(self.fit.unweighted_fit_residual),
            "fit_converged": bool(self.fit.converged),
            "rr_cutoff": float(doubles.projector.cutoff),
            "provenance_binding": "runtime-object-identity",
            "rr_doubles_runtime_identity": id(doubles),
            "amplitude_thc_fit_runtime_identity": id(amplitude_factors),
            "cholesky_runtime_identity": id(self._source_lov),
            "weight_result_runtime_identity": id(self.weights),
            "natural_orbital_fit_runtime_identity": id(self.fit),
            "transform_auxiliary_block_size": int(
                self.transform_auxiliary_block_size
            ),
            "preprocess_host_scalar_reads": int(
                self.preprocess_host_scalar_reads
            ),
            "host_scalar_reads": int(self.host_scalar_reads),
            "largest_intermediate_nbytes": int(
                self.largest_intermediate_nbytes
            ),
            "largest_intermediate_scope": "aggregate-preprocessing-path",
            "actual_peak_memory_requires_profiling": True,
            "transformed_three_index_cholesky_materialized": True,
            "dense_mp2_t2_materialized": False,
            "mp2_pair_matrix_materialized": False,
            "four_index_eri_materialized": False,
            "explicit_working_cholesky_reconstruction_materializes_3index": True,
            "implicit_host_tensor_transfer": False,
            "gpu_validation_requires_transfer_counter": True,
            "storage_nbytes": self.storage_nbytes,
            "source_cholesky_borrowed_not_counted_in_storage": True,
            "weights": self.weights.metadata(),
            "natural_orbital_fit": self.fit.metadata(),
            "working_eri_factors": self.factors.metadata(),
            "audit_only": True,
            "production_enabled": False,
            "performance_eligible": False,
        }


def _validate_inputs_and_controls(
    mp2_doubles: RRDoubles,
    amplitude_factors: THCProjectorFactors,
    lov: Any,
    *,
    occupation_change_floor: Any,
    eri_thc_rank: Any,
    fit_tolerance: Any,
    exact_pair_endpoint: Any,
    max_iterations: Any,
    als_convergence_tolerance: Any,
    ridge: Any,
    seed: Any,
    auxiliary_block_size: Any,
    transfer_counter: Any,
):
    if not isinstance(mp2_doubles, RRDoubles):
        raise TypeError("mp2_doubles must be an RRDoubles instance")
    if not isinstance(amplitude_factors, THCProjectorFactors):
        raise TypeError(
            "amplitude_factors must be a THCProjectorFactors instance"
        )
    xp = _same_backend(
        lov,
        mp2_doubles.core,
        amplitude_factors.y_occ,
        amplitude_factors.y_vir,
        amplitude_factors.tau,
    )
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU ERI-THC preprocessing requires an explicit "
                "transfer_counter"
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
    if (nocc, nvir) != (mp2_doubles.nocc, mp2_doubles.nvir):
        raise ValueError("lov and RR doubles have different orbital dimensions")
    if (nocc, nvir) != (amplitude_factors.nocc, amplitude_factors.nvir):
        raise ValueError(
            "lov and amplitude THC factors have different orbital dimensions"
        )
    dtypes = tuple(
        np.dtype(value.dtype)
        for value in (
            lov,
            mp2_doubles.core,
            amplitude_factors.y_occ,
            amplitude_factors.y_vir,
            amplitude_factors.tau,
        )
    )
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError("ERI-THC preprocessing tensors must have the same dtype")
    dtype = dtypes[0]
    if np.issubdtype(dtype, np.complexfloating):
        raise NotImplementedError(
            "weighted ERI-THC preprocessing currently requires real RHF"
        )
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(
            "ERI-THC preprocessing tensors must have a floating-point dtype"
        )

    rank = _positive_integer("eri_thc_rank", eri_thc_rank)
    if rank > nocc * nvir:
        raise ValueError("eri_thc_rank must lie in [1,nocc*nvir]")
    tolerance = _nonnegative_real("fit_tolerance", fit_tolerance)
    convergence_tolerance = _nonnegative_real(
        "als_convergence_tolerance", als_convergence_tolerance
    )
    ridge_value = _nonnegative_real("ridge", ridge)
    iterations = _positive_integer("max_iterations", max_iterations)
    seed_value = _integer("seed", seed)
    block_size = min(
        _positive_integer("auxiliary_block_size", auxiliary_block_size), naux
    )
    if not isinstance(exact_pair_endpoint, (bool, np.bool_)):
        raise TypeError("exact_pair_endpoint must be a boolean")
    exact_pair_endpoint = bool(exact_pair_endpoint)
    if exact_pair_endpoint and rank != nocc * nvir:
        raise ValueError(
            "exact_pair_endpoint requires eri_thc_rank == nocc*nvir"
        )
    if exact_pair_endpoint and tolerance != 0.0:
        raise ValueError("exact_pair_endpoint requires fit_tolerance == 0")

    finite = xp.all(xp.isfinite(lov))
    local_reads = [0]

    def read_scalar(value: Any, operation: str) -> float:
        if xp is not np:
            transfer_counter.record_d2h(int(value.nbytes), operation=operation)
            local_reads[0] += 1
        return float(value.item()) if hasattr(value, "item") else float(value)

    if not bool(read_scalar(finite, "eri_thc_preprocess_input_finite")):
        raise ValueError("lov must contain only finite values")
    return (
        xp,
        dtype,
        naux,
        nocc,
        nvir,
        rank,
        tolerance,
        exact_pair_endpoint,
        iterations,
        convergence_tolerance,
        ridge_value,
        seed_value,
        block_size,
        local_reads,
        read_scalar,
    )


def build_weighted_eri_thc_preprocess(
    mp2_doubles: RRDoubles,
    amplitude_factors: THCProjectorFactors,
    lov: Any,
    *,
    occupation_change_floor: float,
    eri_thc_rank: int,
    fit_tolerance: float,
    exact_pair_endpoint: bool = False,
    max_iterations: int = 500,
    als_convergence_tolerance: float = 1e-10,
    ridge: float = 1e-12,
    seed: int = 0,
    auxiliary_block_size: int = 64,
    transfer_counter: Any = None,
) -> WeightedERITHCPreprocessResult:
    """Build working-basis ERI-THC factors through an MP2-NO weighted fit.

    The default path calls :func:`fit_weighted_eri_thc`.  Setting
    ``exact_pair_endpoint=True`` selects the deterministic analytic endpoint
    and strictly requires ``eri_thc_rank == nocc*nvir`` and
    ``fit_tolerance == 0``; ALS-only controls are validated but do not apply on
    that branch.  A full-rank ALS call remains an ALS fit and is not relabeled
    as the analytic exact endpoint.
    """

    (
        xp,
        dtype,
        _naux,
        _nocc,
        _nvir,
        rank,
        tolerance,
        exact_pair_endpoint,
        iterations,
        convergence_tolerance,
        ridge_value,
        seed_value,
        block_size,
        local_reads,
        read_scalar,
    ) = _validate_inputs_and_controls(
        mp2_doubles,
        amplitude_factors,
        lov,
        occupation_change_floor=occupation_change_floor,
        eri_thc_rank=eri_thc_rank,
        fit_tolerance=fit_tolerance,
        exact_pair_endpoint=exact_pair_endpoint,
        max_iterations=max_iterations,
        als_convergence_tolerance=als_convergence_tolerance,
        ridge=ridge,
        seed=seed,
        auxiliary_block_size=auxiliary_block_size,
        transfer_counter=transfer_counter,
    )
    weights = build_factorized_mp2_natural_occupation_weights(
        mp2_doubles,
        amplitude_factors,
        occupation_change_floor=occupation_change_floor,
        transfer_counter=transfer_counter,
    )
    weights.validate_binding(mp2_doubles, amplitude_factors)
    lov_natural, transform_block_nbytes = (
        _transform_cholesky_to_natural_orbitals(
            lov,
            weights.occupied_rotation,
            weights.virtual_rotation,
            auxiliary_block_size=block_size,
            xp=xp,
        )
    )
    transformed_finite = xp.all(xp.isfinite(lov_natural))
    if not bool(
        read_scalar(
            transformed_finite,
            "eri_thc_preprocess_rotated_cholesky_finite",
        )
    ):
        raise FloatingPointError(
            "natural-orbital Cholesky transformation produced non-finite values"
        )

    if exact_pair_endpoint:
        fit = exact_eri_thc_from_cholesky(
            lov_natural,
            weights.occupied_weights,
            weights.virtual_weights,
            transfer_counter=transfer_counter,
        )
    else:
        fit = fit_weighted_eri_thc(
            lov_natural,
            weights.occupied_weights,
            weights.virtual_weights,
            eri_thc_rank=rank,
            fit_tolerance=tolerance,
            max_iterations=iterations,
            als_convergence_tolerance=convergence_tolerance,
            ridge=ridge_value,
            seed=seed_value,
            allow_unconverged=False,
            auxiliary_block_size=block_size,
            transfer_counter=transfer_counter,
        )
    if not fit.converged or fit.weighted_fit_residual > fit.fit_tolerance:
        raise RuntimeError(
            "weighted ERI-THC fit did not satisfy its declared tolerance"
        )
    if (
        fit.occupied_weights is not weights.occupied_weights
        or fit.virtual_weights is not weights.virtual_weights
    ):
        raise RuntimeError("weighted ERI-THC fit did not retain weight identity")

    x_occ_working = weights.occupied_rotation @ fit.factors.x_occ
    x_vir_working = weights.virtual_rotation @ fit.factors.x_vir
    factors = ERITHCFactors(
        x_occ_working,
        x_vir_working,
        fit.factors.core,
        transfer_counter=transfer_counter,
    )
    if xp is not np:
        # ERITHCFactors performs one combined finite-value control read.
        local_reads[0] += 1
    largest_intermediate = max(
        int(lov_natural.nbytes),
        int(transform_block_nbytes),
        int(weights.largest_intermediate_nbytes),
        int(fit.largest_intermediate_nbytes),
    )
    return WeightedERITHCPreprocessResult(
        factors=factors,
        fit=fit,
        weights=weights,
        exact_pair_endpoint=exact_pair_endpoint,
        transform_auxiliary_block_size=block_size,
        preprocess_host_scalar_reads=local_reads[0],
        host_scalar_reads=(
            local_reads[0] + weights.host_scalar_reads + fit.host_scalar_reads
        ),
        largest_intermediate_nbytes=largest_intermediate,
        _source_mp2_doubles=mp2_doubles,
        _source_amplitude_factors=amplitude_factors,
        _source_lov=lov,
    )


__all__ = [
    "WeightedERITHCPreprocessResult",
    "build_weighted_eri_thc_preprocess",
]
