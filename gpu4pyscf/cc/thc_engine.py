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

"""Fail-closed amplitude-THC validation inside the direct-CD RR lifecycle.

Algorithms 1--3 of Hohenstein *et al.*, J. Chem. Phys. 156, 054102
(2022), cut across the coarse intermediates returned by the complete RR
residual builder.  This module exposes that boundary explicitly as

``R_complete = R_123[U] + R_complement``.

The replacement seam consumes only ``R_complement`` and the THC evaluation
of ``R_123``.  It therefore cannot accidentally add a THC residual to the
complete equation or replace an entire coarse ``Woooo``, ``Wvvvv``, or ring
group.  The public iteration lifecycle still admits only the analytic
full-pair THC endpoint.  Lower-rank THC remains fail closed until the
water2/water4 residual and accuracy gates authorize it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from gpu4pyscf.cc.device_runtime import RunMetrics, nvtx_range
from gpu4pyscf.cc.full_space_residual import (
    FullSpaceResidualDiagnostic,
    diagnose_reconstructed_full_space_residual,
)
from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.rr_engine import (
    RRCCSDIterationEngine,
    RRIterationResult,
    RRKernelResult,
)
from gpu4pyscf.cc.rr_residual import (
    ProjectedPairDenominator,
    build_ccsd_singles_numerator,
    build_projected_ccsd_doubles_components,
    build_projected_ccsd_doubles_numerator,
    rr_ccsd_energy,
)
from gpu4pyscf.cc.thc_factorization import THCProjectorFactors
from gpu4pyscf.cc.thc_residual import (
    PairFactorResidual123Result,
    THCResidual123Result,
    _pair_factor_algorithms_1_3_block,
    rr_residual_algorithms_1_3,
    thc_algorithm_1,
    thc_algorithm_2,
    thc_algorithm_3,
    transform_cholesky_t1,
)


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def _same_backend(reference: Any, *values: Any) -> bool:
    xp = _array_module(reference)
    return all(_array_module(value) is xp for value in values)


@dataclass(frozen=True)
class RRResidual123Decomposition:
    """Exact RR doubles split at the paper Algorithms 1--3 boundary.

    ``complement_core`` contains every projected doubles-numerator diagram
    outside Eqs. 28, 29, and 31.  It is deliberately the only complete-RR
    object accepted by :meth:`replace_r123`.  The completed RR core is not
    retained by this object, so the replacement formula cannot double count
    ``R_123``.

    The legacy oracle evaluates the complete RR equation separately.  The
    production constructor instead accumulates the same coarse terms from a
    signed ``-R_123`` offset and can share each transformed Cholesky block
    between the exact-RR and THC ``R_123`` evaluations.  The fields below
    preserve that distinction in machine-readable metadata.
    """

    complement_core: Any
    exact_r123: PairFactorResidual123Result
    fock_intermediates: Any
    lagrangian_one_body: Any
    complete_term_metadata: tuple[dict[str, Any], ...]
    complete_rr_metadata: dict[str, Any]
    construction: str = "complete-minus-exact-r123-oracle"
    fused_contraction_implementation: bool = False
    complete_rr_evaluated_separately: bool = True
    r123_application_count: int = 1
    shared_transformed_cholesky_block_count: int = 0
    materialized_dense_t2: bool = False
    materialized_four_index_eri: bool = False

    @property
    def exact_r123_core(self):
        return self.exact_r123.sigma_pair

    def replace_r123(self, replacement_core: Any):
        """Compose ``R_complement + replacement_core`` in RR coordinates."""

        if _array_module(replacement_core) is not _array_module(
            self.complement_core
        ):
            raise TypeError("R123 replacement and complement need one backend")
        if getattr(replacement_core, "shape", None) != self.complement_core.shape:
            raise ValueError("R123 replacement shape does not match complement")
        if not np.issubdtype(
            np.dtype(replacement_core.dtype), np.floating
        ):
            raise TypeError("R123 replacement must be a floating-point matrix")
        return self.complement_core + replacement_core

    def reconstruct_complete(self):
        """Recombine the exact split for an equation-identity audit."""

        return self.replace_r123(self.exact_r123_core)

    @property
    def term_ledger(self) -> tuple[dict[str, Any], ...]:
        return (
            {
                "term": "r123-algorithm-1",
                "paper_equation": 28,
                "paper_algorithm": 1,
                "coordinate_space": "rr-core",
            },
            {
                "term": "r123-algorithm-2",
                "paper_equation": 29,
                "paper_algorithm": 2,
                "coordinate_space": "rr-core",
            },
            {
                "term": "r123-algorithm-3",
                "paper_equation": 31,
                "paper_algorithm": 3,
                "coordinate_space": "rr-core",
            },
            {
                "term": "r123-complement",
                "definition": "complete_rr_core - exact_r123_core",
                "coordinate_space": "rr-core",
            },
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "equation_identity": "complete_rr = exact_r123 + complement",
            "construction": self.construction,
            "paper": "Hohenstein-2022",
            "paper_equations": [28, 29, 31],
            "paper_algorithms": [1, 2, 3],
            "replacement_seam": "complement + replacement_r123",
            "exact_r123": self.exact_r123.metadata(),
            "complete_rr": dict(self.complete_rr_metadata),
            "term_ledger": list(self.term_ledger),
            "materialized_dense_t2": bool(self.materialized_dense_t2),
            "materialized_four_index_eri": bool(
                self.materialized_four_index_eri
            ),
            "fused_contraction_implementation": bool(
                self.fused_contraction_implementation
            ),
            "fused_contraction_scope": (
                "exact-and-thc-r123-share-t1-transformed-cholesky-blocks"
                if self.fused_contraction_implementation
                else None
            ),
            "complete_rr_evaluated_separately": bool(
                self.complete_rr_evaluated_separately
            ),
            "r123_application_count": int(self.r123_application_count),
            "shared_transformed_cholesky_block_count": int(
                self.shared_transformed_cholesky_block_count
            ),
            "coarse_r123_diagram_partition_fused": False,
            "performance_eligible": False,
        }


def build_shared_cholesky_r123_decomposition(
    doubles: RRDoubles,
    t1: Any,
    fock_oo: Any,
    fock_ov: Any,
    fock_vv: Any,
    occupied_energies: Any,
    virtual_energies: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    *,
    level_shift: float = 0.0,
    auxiliary_block_size: int = 1,
    virtual_block_size: int = 8,
    transfer_counter: Any = None,
) -> RRResidual123Decomposition:
    """Small-system oracle for the exact ``R_123``/complement identity.

    This intentionally retains the original complete-minus-exact evaluation
    order.  The production hybrid path uses the fused constructor below; this
    routine remains independent so small systems can audit its equation split.
    """

    complete = build_projected_ccsd_doubles_numerator(
        doubles,
        t1,
        fock_oo,
        fock_ov,
        fock_vv,
        occupied_energies,
        virtual_energies,
        loo,
        lov,
        lvv,
        level_shift=level_shift,
        auxiliary_block_size=auxiliary_block_size,
        virtual_block_size=virtual_block_size,
        transfer_counter=transfer_counter,
    )
    exact_r123 = rr_residual_algorithms_1_3(
        doubles,
        t1,
        loo,
        lov,
        lvv,
        auxiliary_block_size=auxiliary_block_size,
    )
    complement = complete.core - exact_r123.sigma_pair
    return RRResidual123Decomposition(
        complement_core=complement,
        exact_r123=exact_r123,
        fock_intermediates=complete.fock_intermediates,
        lagrangian_one_body=complete.lagrangian_one_body,
        complete_term_metadata=complete.term_metadata,
        complete_rr_metadata=complete.metadata(),
        construction="complete-minus-exact-r123-oracle",
        fused_contraction_implementation=False,
        complete_rr_evaluated_separately=True,
        r123_application_count=1,
    )


@dataclass(frozen=True)
class THCHybridDoublesResult:
    """RR complement composed with an audited THC ``R_123`` replacement."""

    core: Any
    decomposition: RRResidual123Decomposition
    thc_approximation_1_3: THCResidual123Result
    thc_backprojected_core: Any
    replacement_delta_core: Any
    replacement_delta_norm: float
    replacement_reference_norm: float
    replacement_backprojected_norm: float
    replacement_validation_threshold: float
    replacement_validation_atol: float
    replacement_validation_rtol: float
    replacement_validated: bool = True
    replacement_applied: bool = True
    materialized_dense_t2: bool = False
    materialized_four_index_eri: bool = False
    fused_r123_construction: Optional["FusedR123Construction"] = None

    def solve(
        self,
        denominator: ProjectedPairDenominator,
        *,
        symmetry_tolerance: float = 1e-12,
        transfer_counter: Any = None,
    ):
        if denominator.rank != self.core.shape[0]:
            raise ValueError("doubles denominator rank does not match numerator")
        return denominator.solve(
            self.core,
            symmetry_tolerance=symmetry_tolerance,
            transfer_counter=transfer_counter,
        )

    @property
    def fock_intermediates(self):
        return self.decomposition.fock_intermediates

    @property
    def lagrangian_one_body(self):
        return self.decomposition.lagrangian_one_body

    @property
    def rr_reference_1_3(self):
        """Compatibility view of the exact ``R_123`` contribution."""

        return self.decomposition.exact_r123

    @property
    def term_metadata(self) -> tuple[dict[str, Any], ...]:
        return self.decomposition.complete_term_metadata + (
            {
                "term": "exact-r123-subtraction",
                "coefficient": -1,
                "paper_equations": [28, 29, 31],
                "paper_algorithms": [1, 2, 3],
                "application_count": 1,
                "coordinate_space": "rr-core",
                "materialized_dense_t2": False,
                "materialized_four_index_eri": False,
            },
            {
                "term": "amplitude-thc-algorithms-1-3-replacement",
                "coefficient": 1,
                "formula": "rr_complement + thc_backprojection_1_3",
                "paper_equations": [28, 29, 31],
                "paper_algorithms": [1, 2, 3],
                "application_count": 1,
                "replacement_applied": True,
                "materialized_dense_t2": False,
                "materialized_four_index_eri": False,
            },
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "equation": "rr-r123-complement-plus-thc-r123",
            "complete_equation_graph": True,
            "canonical_equation": False,
            "replacement_formula": (
                "rr_complement + backproject(thc_algorithms_1_3)"
            ),
            "replacement_applied": bool(self.replacement_applied),
            "r123_application_count": int(self.replacement_applied),
            "replacement_validated": bool(self.replacement_validated),
            "replacement_delta_norm": float(self.replacement_delta_norm),
            "replacement_reference_norm": float(
                self.replacement_reference_norm
            ),
            "replacement_backprojected_norm": float(
                self.replacement_backprojected_norm
            ),
            "replacement_validation_threshold": float(
                self.replacement_validation_threshold
            ),
            "replacement_validation_atol": float(
                self.replacement_validation_atol
            ),
            "replacement_validation_rtol": float(
                self.replacement_validation_rtol
            ),
            "replacement_validation_criterion": (
                "delta <= atol + rtol*max(reference,backprojected)"
            ),
            "inexact_thc_enabled": False,
            "rr_r123_decomposition": self.decomposition.metadata(),
            "fused_r123_construction": (
                None
                if self.fused_r123_construction is None
                else self.fused_r123_construction.metadata()
            ),
            "fused_contraction_implementation": bool(
                self.decomposition.fused_contraction_implementation
            ),
            "complete_rr_evaluated_separately": bool(
                self.decomposition.complete_rr_evaluated_separately
            ),
            "rr_complete": dict(self.decomposition.complete_rr_metadata),
            "rr_algorithms_1_3_reference": self.rr_reference_1_3.metadata(),
            "thc_algorithms_1_3_endpoint": self.thc_approximation_1_3.metadata(),
            "terms": list(self.term_metadata),
            "materialized_dense_t2": bool(self.materialized_dense_t2),
            "materialized_four_index_eri": bool(
                self.materialized_four_index_eri
            ),
            "performance_eligible": False,
            "performance_gate": (
                "inexact replacement requires water2/water4 residual and "
                "accuracy gates"
            ),
        }


def _validate_factor_lifecycle(
    doubles: RRDoubles,
    factors: THCProjectorFactors,
    loo: Any,
    lov: Any,
    lvv: Any,
) -> None:
    vectors = doubles.projector.vectors
    if not _same_backend(
        vectors,
        doubles.core,
        factors.y_occ,
        factors.y_vir,
        factors.tau,
        factors.raw_tau,
        factors.eigenvalues,
        loo,
        lov,
        lvv,
    ):
        raise TypeError("RR state, THC factors, and CD factors need one backend")
    if factors.rr_rank != doubles.rank:
        raise ValueError("THC factor RR rank does not match the doubles state")
    if factors.nocc != doubles.nocc or factors.nvir != doubles.nvir:
        raise ValueError("THC factor orbital dimensions do not match doubles")
    if not factors.converged:
        raise RuntimeError("unconverged THC projector factors cannot enter CCSD")
    if factors.weighted_fit_residual > factors.fit_tolerance:
        raise RuntimeError("THC projector factors failed their weighted fit gate")
    if factors.orthogonality_error > 1e-10:
        raise RuntimeError("THC projector factors failed the orthogonality gate")
    if not factors.analytic_full_pair_endpoint:
        raise NotImplementedError(
            "inexact direct-CD amplitude THC is disabled until the water2 "
            "and water4 residual and accuracy gates pass; use the analytic "
            "full-pair endpoint for validation"
        )


@dataclass(frozen=True)
class FusedR123Construction:
    """Exact-RR and THC ``R_123`` values from one transformed-CD stream."""

    exact_rr: PairFactorResidual123Result
    thc: THCResidual123Result
    transformed_cholesky_block_count: int
    transformed_cholesky_application_count: int
    fused_contraction_implementation: bool = True

    def metadata(self) -> dict[str, Any]:
        return {
            "fused_contraction_implementation": bool(
                self.fused_contraction_implementation
            ),
            "fused_contraction_scope": (
                "exact-and-thc-r123-share-t1-transformed-cholesky-blocks"
            ),
            "transformed_cholesky_block_count": int(
                self.transformed_cholesky_block_count
            ),
            "transformed_cholesky_application_count": int(
                self.transformed_cholesky_application_count
            ),
            "r123_evaluation_count": 2,
            "r123_evaluation_spaces": ["rr-core", "thc-core"],
            "coarse_r123_diagram_partition_fused": False,
            "performance_eligible": False,
        }


def build_fused_r123_construction(
    doubles: RRDoubles,
    factors: THCProjectorFactors,
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    *,
    auxiliary_block_size: int = 1,
) -> FusedR123Construction:
    """Evaluate exact-RR and THC Algorithms 1--3 in one CD-block loop.

    Each raw ``loo/lov/lvv`` slice is transformed exactly once and the
    resulting ``Hoo/Hov/Hvv/Hvo`` block feeds both coordinate-space kernels.
    The pair and THC contractions remain separate because their algebra and
    ranks differ; the shared transform and block lifetime are the fused unit.
    """

    _validate_factor_lifecycle(doubles, factors, loo, lov, lvv)
    if t1.shape != (doubles.nocc, doubles.nvir):
        raise ValueError("t1 shape does not match RR doubles")
    if not _same_backend(
        doubles.projector.vectors,
        doubles.core,
        factors.y_occ,
        factors.y_vir,
        t1,
        loo,
        lov,
        lvv,
    ):
        raise TypeError("fused R123 inputs need one array backend")
    naux = int(lov.shape[0])
    block_size = int(auxiliary_block_size)
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")
    xp = _array_module(doubles.core)
    rr_dtype = xp.result_type(
        doubles.projector.vectors, doubles.core, t1, loo, lov, lvv
    )
    thc_core = factors.amplitude_core(doubles.core)
    thc_dtype = xp.result_type(
        factors.y_occ, factors.y_vir, thc_core, t1, loo, lov, lvv
    )
    rr1 = xp.zeros((doubles.rank, doubles.rank), dtype=rr_dtype)
    rr2 = xp.zeros_like(rr1)
    rr3 = xp.zeros_like(rr1)
    thc1 = xp.zeros((factors.thc_rank, factors.thc_rank), dtype=thc_dtype)
    thc2 = xp.zeros_like(thc1)
    thc3 = xp.zeros_like(thc1)
    rr_largest = 0
    thc_largest = 0
    block_count = 0
    for start in range(0, naux, block_size):
        stop = min(start + block_size, naux)
        transformed = transform_cholesky_t1(
            t1, loo[start:stop], lov[start:stop], lvv[start:stop]
        )
        part_rr1, part_rr2, part_rr3, part_rr_largest = (
            _pair_factor_algorithms_1_3_block(
                doubles.projector.vectors,
                doubles.core,
                transformed,
                doubles.nocc,
                doubles.nvir,
            )
        )
        part_thc1 = thc_algorithm_1(
            factors.y_occ, factors.y_vir, thc_core, transformed
        )
        part_thc2 = thc_algorithm_2(
            factors.y_occ, factors.y_vir, thc_core, transformed
        )
        part_thc3 = thc_algorithm_3(
            factors.y_occ, factors.y_vir, thc_core, transformed
        )
        rr1 += part_rr1
        rr2 += part_rr2
        rr3 += part_rr3
        thc1 += part_thc1
        thc2 += part_thc2
        thc3 += part_thc3
        rr_largest = max(
            rr_largest,
            transformed.storage_nbytes,
            int(part_rr_largest),
        )
        thc_largest = max(
            thc_largest,
            transformed.storage_nbytes,
            int(part_thc1.nbytes),
            int(part_thc2.nbytes),
            int(part_thc3.nbytes),
        )
        block_count += 1
    exact = PairFactorResidual123Result(
        sigma_pair=rr1 + rr2 + rr3,
        algorithm_1=rr1,
        algorithm_2=rr2,
        algorithm_3=rr3,
        nocc=doubles.nocc,
        nvir=doubles.nvir,
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=rr_largest,
    )
    approximation = THCResidual123Result(
        sigma_thc=thc1 + thc2 + thc3,
        algorithm_1=thc1,
        algorithm_2=thc2,
        algorithm_3=thc3,
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=thc_largest,
    )
    return FusedR123Construction(
        exact_rr=exact,
        thc=approximation,
        transformed_cholesky_block_count=block_count,
        transformed_cholesky_application_count=block_count,
    )


def build_fused_cholesky_r123_decomposition(
    doubles: RRDoubles,
    factors: THCProjectorFactors,
    t1: Any,
    fock_oo: Any,
    fock_ov: Any,
    fock_vv: Any,
    occupied_energies: Any,
    virtual_energies: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    *,
    level_shift: float = 0.0,
    auxiliary_block_size: int = 1,
    virtual_block_size: int = 8,
    transfer_counter: Any = None,
) -> tuple[RRResidual123Decomposition, FusedR123Construction]:
    """Construct the complement and both R123 values without a complete RR call."""

    _validate_factor_lifecycle(doubles, factors, loo, lov, lvv)
    components = build_projected_ccsd_doubles_components(
        doubles,
        t1,
        fock_oo,
        fock_ov,
        fock_vv,
        occupied_energies,
        virtual_energies,
        loo,
        lov,
        lvv,
        level_shift=level_shift,
        auxiliary_block_size=auxiliary_block_size,
        virtual_block_size=virtual_block_size,
        transfer_counter=transfer_counter,
    )
    fused = build_fused_r123_construction(
        doubles,
        factors,
        t1,
        loo,
        lov,
        lvv,
        auxiliary_block_size=auxiliary_block_size,
    )
    # The first assembled value is the complement itself.  There is no
    # complete-RR core whose only purpose is a later exact-R123 subtraction.
    complement = components.assemble_core(
        initial_core=-fused.exact_rr.sigma_pair
    )
    complete_metadata = components.metadata()
    complete_metadata.update({
        "assembly": "component-offset-construction",
        "initial_offset": "-exact_r123",
        "complete_rr_evaluated_separately": False,
    })
    decomposition = RRResidual123Decomposition(
        complement_core=complement,
        exact_r123=fused.exact_rr,
        fock_intermediates=components.fock_intermediates,
        lagrangian_one_body=components.lagrangian_one_body,
        complete_term_metadata=components.term_metadata,
        complete_rr_metadata=complete_metadata,
        construction=(
            "component-offset-complement-with-shared-r123-cd-stream"
        ),
        fused_contraction_implementation=True,
        complete_rr_evaluated_separately=False,
        r123_application_count=1,
        shared_transformed_cholesky_block_count=(
            fused.transformed_cholesky_block_count
        ),
    )
    return decomposition, fused


def _read_control_scalar(
    value: Any,
    *,
    transfer_counter: Any,
    operation: str,
) -> float:
    xp = _array_module(value)
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU THC validation requires an explicit transfer_counter"
            )
        transfer_counter.record_d2h(
            int(value.nbytes), operation=operation
        )
    return float(value.item())


def build_hybrid_projected_ccsd_doubles_numerator(
    doubles: RRDoubles,
    factors: THCProjectorFactors,
    t1: Any,
    fock_oo: Any,
    fock_ov: Any,
    fock_vv: Any,
    occupied_energies: Any,
    virtual_energies: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    *,
    level_shift: float = 0.0,
    auxiliary_block_size: int = 1,
    virtual_block_size: int = 8,
    replacement_validation_atol: float,
    replacement_validation_rtol: float,
    transfer_counter: Any = None,
) -> THCHybridDoublesResult:
    """Validate and apply the full-pair THC replacement at the ``R_123`` seam.

    The function retains its original name as an explicit compatibility seam.
    An inexact replacement is still blocked by :func:`_validate_factor_lifecycle`
    until its residual and accuracy gates pass.  The accepted full-pair
    endpoint is composed as ``R_complement + R_123[THC]`` rather than returning
    the independently evaluated complete RR numerator.
    """

    if not isinstance(doubles, RRDoubles):
        raise TypeError("doubles must be RRDoubles")
    validation_atol = float(replacement_validation_atol)
    validation_rtol = float(replacement_validation_rtol)
    if (
        not np.isfinite(validation_atol)
        or not np.isfinite(validation_rtol)
        or validation_atol < 0
        or validation_rtol < 0
    ):
        raise ValueError(
            "replacement validation tolerances must be finite and non-negative"
        )
    _validate_factor_lifecycle(doubles, factors, loo, lov, lvv)
    decomposition, fused_r123 = build_fused_cholesky_r123_decomposition(
        doubles,
        factors,
        t1,
        fock_oo,
        fock_ov,
        fock_vv,
        occupied_energies,
        virtual_energies,
        loo,
        lov,
        lvv,
        level_shift=level_shift,
        auxiliary_block_size=auxiliary_block_size,
        virtual_block_size=virtual_block_size,
        transfer_counter=transfer_counter,
    )
    approximation = fused_r123.thc
    backprojected = approximation.to_rr(factors.tau)
    delta = backprojected - decomposition.exact_r123_core
    xp = _array_module(delta)
    delta_norm_device = xp.linalg.norm(delta)
    reference_norm_device = xp.linalg.norm(decomposition.exact_r123_core)
    backprojected_norm_device = xp.linalg.norm(backprojected)
    delta_norm = _read_control_scalar(
        delta_norm_device,
        transfer_counter=transfer_counter,
        operation="thc_full_pair_replacement_validation",
    )
    reference_norm = _read_control_scalar(
        reference_norm_device,
        transfer_counter=transfer_counter,
        operation="thc_full_pair_replacement_reference_norm",
    )
    backprojected_norm = _read_control_scalar(
        backprojected_norm_device,
        transfer_counter=transfer_counter,
        operation="thc_full_pair_replacement_backprojected_norm",
    )
    validation_threshold = (
        validation_atol
        + validation_rtol * max(reference_norm, backprojected_norm)
    )
    if delta_norm > validation_threshold:
        raise RuntimeError(
            "analytic full-pair THC endpoint failed the Algorithms 1-3 "
            "back-projection gate: "
            f"delta {delta_norm:.3e} > {validation_threshold:.3e} "
            f"(atol {validation_atol:.3e}, rtol "
            f"{validation_rtol:.3e}, reference {reference_norm:.3e}, "
            f"backprojected {backprojected_norm:.3e})"
        )
    # Apply exactly one replacement at the explicit decomposition seam.  No
    # complete-equation object enters this composition, which makes accidental
    # ``complete + THC`` double counting structurally impossible.
    core = decomposition.replace_r123(backprojected)
    return THCHybridDoublesResult(
        core=core,
        decomposition=decomposition,
        thc_approximation_1_3=approximation,
        fused_r123_construction=fused_r123,
        thc_backprojected_core=backprojected,
        replacement_delta_core=delta,
        replacement_delta_norm=delta_norm,
        replacement_reference_norm=reference_norm,
        replacement_backprojected_norm=backprojected_norm,
        replacement_validation_threshold=validation_threshold,
        replacement_validation_atol=validation_atol,
        replacement_validation_rtol=validation_rtol,
    )


class THCRRCCSDIterationEngine(RRCCSDIterationEngine):
    """Complete RR/CD iteration with an exact full-pair THC replacement."""

    def __init__(
        self,
        integrals: MOThreeIndexIntegralProvider,
        projector: RRProjector,
        factors: THCProjectorFactors,
        fock: Any,
        orbital_energies: Any,
        *,
        level_shift: float = 0.0,
        auxiliary_block_size: int = 1,
        virtual_block_size: int = 8,
        replacement_validation_atol: float,
        replacement_validation_rtol: float,
        metrics: Optional[RunMetrics] = None,
    ) -> None:
        fp64_arrays = (
            integrals.factors,
            projector.vectors,
            projector.eigenvalues,
            factors.y_occ,
            factors.y_vir,
            factors.tau,
            fock,
            orbital_energies,
        )
        if any(np.dtype(value.dtype) != np.dtype(np.float64)
               for value in fp64_arrays):
            raise TypeError("the THC/RR iteration engine requires FP64 tensors")
        validation_atol = float(replacement_validation_atol)
        validation_rtol = float(replacement_validation_rtol)
        if (
            not np.isfinite(validation_atol)
            or not np.isfinite(validation_rtol)
            or validation_atol < 0
            or validation_rtol < 0
        ):
            raise ValueError(
                "replacement validation tolerances must be finite and "
                "non-negative"
            )
        super().__init__(
            integrals,
            projector,
            fock,
            orbital_energies,
            level_shift=level_shift,
            auxiliary_block_size=auxiliary_block_size,
            virtual_block_size=virtual_block_size,
            metrics=metrics,
        )
        probe = RRDoubles(
            projector,
            self.xp.zeros((projector.rank, projector.rank), dtype=fock.dtype),
            self.nocc,
            self.nvir,
        )
        _validate_factor_lifecycle(
            probe,
            factors,
            integrals.L_oo,
            integrals.L_ov,
            integrals.L_vv,
        )
        # The fit routine deliberately retains the exact eigenvalue object.
        # Requiring identity prevents factors from a same-shaped projector or
        # another geometry from being silently reused in this lifecycle.
        if factors.eigenvalues is not projector.eigenvalues:
            raise ValueError("THC factors do not belong to this RR projector")
        self.thc_factors = factors
        self.replacement_validation_atol = validation_atol
        self.replacement_validation_rtol = validation_rtol
        self.replacement_application_count = 0

    def jacobi(self, t1: Any, doubles: RRDoubles) -> RRIterationResult:
        """Replace ``R_123`` at the exact THC endpoint without dense T2."""

        self._validate_state(t1, doubles)
        with nvtx_range("thc-rr/full-singles-equation"):
            with self.metrics.phase("thc_rr_singles_equation"):
                singles = build_ccsd_singles_numerator(
                    doubles,
                    t1,
                    self.fock_oo,
                    self.fock_ov,
                    self.fock_vv,
                    self.occupied_energies,
                    self.virtual_energies,
                    self.integrals.L_oo,
                    self.integrals.L_ov,
                    self.integrals.L_vv,
                    level_shift=self.level_shift,
                    auxiliary_block_size=self.auxiliary_block_size,
                )
                new_t1 = singles.amplitudes(self.eia)
        with nvtx_range("thc-rr/endpoint-audited-doubles-equation"):
            with self.metrics.phase("thc_rr_endpoint_doubles_equation"):
                doubles_equation = (
                    build_hybrid_projected_ccsd_doubles_numerator(
                        doubles,
                        self.thc_factors,
                        t1,
                        self.fock_oo,
                        self.fock_ov,
                        self.fock_vv,
                        self.occupied_energies,
                        self.virtual_energies,
                        self.integrals.L_oo,
                        self.integrals.L_ov,
                        self.integrals.L_vv,
                        level_shift=self.level_shift,
                        auxiliary_block_size=self.auxiliary_block_size,
                        virtual_block_size=self.virtual_block_size,
                        replacement_validation_atol=(
                            self.replacement_validation_atol
                        ),
                        replacement_validation_rtol=(
                            self.replacement_validation_rtol
                        ),
                        transfer_counter=self.metrics.transfers,
                    )
                )
                new_core = doubles_equation.solve(
                    self.denominator,
                    transfer_counter=self.metrics.transfers,
                )
        new_doubles = RRDoubles(
            projector=self.projector,
            core=new_core,
            nocc=self.nocc,
            nvir=self.nvir,
        )
        error_t1 = new_t1 - t1
        error_core = new_core - doubles.core
        update_norm = self.xp.sqrt(
            self.xp.vdot(error_t1, error_t1).real
            + self.xp.vdot(error_core, error_core).real
        )
        equation_residual_t1 = singles.numerator - self.eia * t1
        equation_residual_core = doubles_equation.core - (
            self.denominator.matrix @ doubles.core
            + doubles.core @ self.denominator.matrix
        )
        residual_norm = self.xp.sqrt(
            self.xp.vdot(equation_residual_t1, equation_residual_t1).real
            + self.xp.vdot(
                equation_residual_core, equation_residual_core
            ).real
        )
        energy = rr_ccsd_energy(
            doubles,
            t1,
            self.fock_ov,
            self.integrals.L_ov,
            auxiliary_block_size=self.auxiliary_block_size,
        ).value
        result = RRIterationResult(
            t1=new_t1,
            doubles=new_doubles,
            error_t1=error_t1,
            error_core=error_core,
            update_norm=update_norm,
            equation_residual_t1=equation_residual_t1,
            equation_residual_core=equation_residual_core,
            residual_norm=residual_norm,
            energy=energy,
            singles_equation=singles,
            doubles_equation=doubles_equation,
        )
        self.replacement_application_count += 1
        self.metrics.increment("iterations")
        return result

    def kernel(
        self,
        t1: Any,
        doubles: RRDoubles,
        **kwargs: Any,
    ) -> RRKernelResult:
        result = super().kernel(t1, doubles, **kwargs)
        self.metrics.metadata.update(self.metadata())
        self.metrics.metadata.update({
            "converged": bool(result.converged),
            "cycles": int(result.cycles),
        })
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "engine": "THCRRCCSDIterationEngine",
            "equation": "rr-r123-complement-plus-thc-r123",
            "complete_equation_graph": True,
            "canonical_equation": False,
            "integrals": self.integrals.metadata(),
            "projector_rank": self.projector.rank,
            "thc_rank": self.thc_factors.thc_rank,
            "thc_projector_factors": self.thc_factors.metadata(),
            "validated_doubles_groups": {
                "paper": "Hohenstein-2022",
                "equations": [28, 29, 31],
                "algorithms": [1, 2, 3],
            },
            "replacement_formula": (
                "rr_complement + backproject(thc_algorithms_1_3)"
            ),
            "replacement_applied": bool(
                self.replacement_application_count > 0
            ),
            "replacement_application_count": int(
                self.replacement_application_count
            ),
            "r123_application_count": int(
                self.replacement_application_count
            ),
            "r123_evaluation_count": int(
                2 * self.replacement_application_count
            ),
            "exact_r123_subtraction_count": int(
                self.replacement_application_count
            ),
            "fused_contraction_implementation": True,
            "fused_contraction_scope": (
                "exact-and-thc-r123-share-t1-transformed-cholesky-blocks"
            ),
            "complete_rr_evaluated_separately": False,
            "coarse_r123_diagram_partition_fused": False,
            "replacement_validation_atol": (
                self.replacement_validation_atol
            ),
            "replacement_validation_rtol": (
                self.replacement_validation_rtol
            ),
            "inexact_thc_enabled": False,
            "inexact_thc_blocker": (
                "water2/water4 residual and accuracy gates have not passed"
            ),
            "singles_equation": "complete-rr-cd",
            "doubles_equation": "rr-r123-complement-plus-thc-r123",
            "iteration_state": "rr-core",
            "reported_energy": "rr-state-rccsd-energy-functional",
            "projected_residual": "complete-rr-equation",
            "auxiliary_block_size": self.auxiliary_block_size,
            "virtual_block_size": self.virtual_block_size,
            "dense_t2_in_iteration": False,
            "four_index_eri_in_iteration": False,
            "convergence_norm": "complete-rr-equation-residual-frobenius",
            "performance_eligible": False,
            "performance_gate": (
                "inexact replacement requires water2/water4 residual and "
                "accuracy gates"
            ),
        }


__all__ = [
    "RRResidual123Decomposition",
    "FusedR123Construction",
    "THCHybridDoublesResult",
    "FullSpaceResidualDiagnostic",
    "build_shared_cholesky_r123_decomposition",
    "build_fused_r123_construction",
    "build_fused_cholesky_r123_decomposition",
    "build_hybrid_projected_ccsd_doubles_numerator",
    "diagnose_reconstructed_full_space_residual",
    "THCRRCCSDIterationEngine",
]
