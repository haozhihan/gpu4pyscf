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
(2022), cut across the coarse intermediates returned by the current complete
RR residual builder.  Consequently, subtracting an independently evaluated
``R_123`` from that complete result has not yet been proven to preserve every
remaining CCSD diagram.

This first G5 tranche therefore admits only the analytic full-pair THC
endpoint.  It evaluates the paper kernels and their RR back-projection, checks
that their difference vanishes within an explicit tolerance, and returns the
unaltered complete RR numerator.  Every inexact factorization fails closed.
The kernels remain useful audited components, but no approximate hybrid
equation or performance eligibility is claimed until the RR builder exposes a
shared paper-equation decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from gpu4pyscf.cc.device_runtime import RunMetrics, nvtx_range
from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.rr_engine import (
    RRCCSDIterationEngine,
    RRIterationResult,
    RRKernelResult,
)
from gpu4pyscf.cc.rr_residual import (
    ProjectedPairDenominator,
    RRCCSDDoublesResult,
    build_ccsd_singles_numerator,
    build_projected_ccsd_doubles_numerator,
    rr_ccsd_energy,
)
from gpu4pyscf.cc.thc_factorization import THCProjectorFactors
from gpu4pyscf.cc.thc_residual import (
    PairFactorResidual123Result,
    THCResidual123Result,
    rr_residual_algorithms_1_3,
    thc_residual_algorithms_1_3,
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
class THCHybridDoublesResult:
    """Complete RR numerator plus an audited, non-applied THC endpoint check."""

    core: Any
    complete_rr: RRCCSDDoublesResult
    rr_reference_1_3: PairFactorResidual123Result
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
    replacement_applied: bool = False
    materialized_dense_t2: bool = False
    materialized_four_index_eri: bool = False

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
        return self.complete_rr.fock_intermediates

    @property
    def lagrangian_one_body(self):
        return self.complete_rr.lagrangian_one_body

    @property
    def term_metadata(self) -> tuple[dict[str, Any], ...]:
        return self.complete_rr.term_metadata + ({
            "term": "amplitude-thc-algorithms-1-3-full-pair-validation",
            "formula": "validate(rr_reference_1_3 == thc_backprojection_1_3)",
            "paper_equations": [28, 29, 31],
            "paper_algorithms": [1, 2, 3],
            "replacement_applied": False,
            "materialized_dense_t2": False,
            "materialized_four_index_eri": False,
        },)

    def metadata(self) -> dict[str, Any]:
        return {
            "equation": "complete-rr-rccsd-with-thc-endpoint-validation",
            "complete_equation_graph": True,
            "canonical_equation": False,
            "replacement_formula": None,
            "replacement_applied": bool(self.replacement_applied),
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
            "rr_complete": self.complete_rr.metadata(),
            "rr_algorithms_1_3_reference": self.rr_reference_1_3.metadata(),
            "thc_algorithms_1_3_endpoint": self.thc_approximation_1_3.metadata(),
            "terms": list(self.term_metadata),
            "materialized_dense_t2": bool(self.materialized_dense_t2),
            "materialized_four_index_eri": bool(
                self.materialized_four_index_eri
            ),
            "performance_eligible": False,
            "performance_gate": (
                "shared RR/paper residual decomposition is not implemented"
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
            "inexact direct-CD amplitude THC is disabled until the complete "
            "RR residual and paper Algorithms 1-3 share one proven term "
            "decomposition; use the analytic full-pair endpoint for "
            "validation"
        )


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


@dataclass(frozen=True)
class FullSpaceResidualDiagnostic:
    """One explicit post-convergence full pair-space residual check."""

    residual_norm: float
    singles_residual_norm: float
    doubles_residual_norm: float
    pair_dimension: int
    largest_materialized_nbytes: int
    materialized_dense_t2_equivalent: bool = True
    included_in_normal_iteration_timing: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": (
                "reconstruct-final-RR-pair-matrix-and-evaluate-complete-"
                "RR/CD-equation-in-the-identity-pair-basis"
            ),
            "norm": "full-pair-equation-residual-frobenius",
            "residual_norm": float(self.residual_norm),
            "singles_residual_norm": float(self.singles_residual_norm),
            "doubles_residual_norm": float(self.doubles_residual_norm),
            "pair_dimension": int(self.pair_dimension),
            "largest_materialized_nbytes": int(
                self.largest_materialized_nbytes
            ),
            "materialized_dense_t2_equivalent": bool(
                self.materialized_dense_t2_equivalent
            ),
            "included_in_normal_iteration_timing": bool(
                self.included_in_normal_iteration_timing
            ),
        }


def diagnose_reconstructed_full_space_residual(
    integrals: MOThreeIndexIntegralProvider,
    t1: Any,
    doubles: RRDoubles,
    fock: Any,
    orbital_energies: Any,
    *,
    level_shift: float = 0.0,
    auxiliary_block_size: int = 1,
    virtual_block_size: int = 8,
    transfer_counter: Any = None,
) -> FullSpaceResidualDiagnostic:
    """Evaluate the complete RR/CD equation once in the identity pair basis.

    This is an explicit diagnostic and deliberately materializes an
    ``(OV,OV)`` amplitude matrix and identity projector.  It is never called
    by the normal iteration path.  The returned residual is the Frobenius norm
    of the singles and full pair-space numerator equations at the supplied
    final state.
    """

    if not isinstance(integrals, MOThreeIndexIntegralProvider):
        raise TypeError("integrals must be an MOThreeIndexIntegralProvider")
    if not isinstance(doubles, RRDoubles):
        raise TypeError("doubles must be RRDoubles")
    nocc, nvir, nmo = integrals.nocc, integrals.nvir, integrals.nmo
    if doubles.nocc != nocc or doubles.nvir != nvir:
        raise ValueError("RR doubles dimensions do not match the integrals")
    values = (
        t1,
        doubles.projector.vectors,
        doubles.core,
        fock,
        orbital_energies,
    )
    if not _same_backend(integrals.factors, *values):
        raise TypeError("diagnostic tensors need one array backend")
    if t1.shape != (nocc, nvir):
        raise ValueError("t1 shape does not match the integrals")
    if fock.shape != (nmo, nmo):
        raise ValueError("fock must have shape (nmo,nmo)")
    if orbital_energies.shape != (nmo,):
        raise ValueError("orbital_energies must have shape (nmo,)")
    if any(
        np.dtype(value.dtype) != np.dtype(np.float64)
        for value in (integrals.factors,) + values
    ):
        raise TypeError("the full-space diagnostic requires FP64 tensors")
    xp = _array_module(integrals.factors)
    dimension = nocc * nvir
    identity = xp.eye(dimension, dtype=doubles.core.dtype)
    full_projector = RRProjector(
        identity,
        xp.ones(dimension, dtype=doubles.core.dtype),
        cutoff=0.0,
        full_dimension=dimension,
        source="explicit-full-space-residual-diagnostic",
    )
    pair_matrix = doubles.reconstruct_pair_matrix()
    full_doubles = RRDoubles(
        full_projector, pair_matrix, nocc=nocc, nvir=nvir
    )
    fock_oo = fock[:nocc, :nocc]
    fock_ov = fock[:nocc, nocc:]
    fock_vv = fock[nocc:, nocc:]
    occupied = orbital_energies[:nocc]
    virtual = orbital_energies[nocc:]
    eia = occupied[:, None] - virtual[None, :] - float(level_shift)
    singles = build_ccsd_singles_numerator(
        full_doubles,
        t1,
        fock_oo,
        fock_ov,
        fock_vv,
        occupied,
        virtual,
        integrals.L_oo,
        integrals.L_ov,
        integrals.L_vv,
        level_shift=level_shift,
        auxiliary_block_size=auxiliary_block_size,
    )
    doubles_equation = build_projected_ccsd_doubles_numerator(
        full_doubles,
        t1,
        fock_oo,
        fock_ov,
        fock_vv,
        occupied,
        virtual,
        integrals.L_oo,
        integrals.L_ov,
        integrals.L_vv,
        level_shift=level_shift,
        auxiliary_block_size=auxiliary_block_size,
        virtual_block_size=virtual_block_size,
        transfer_counter=transfer_counter,
    )
    singles_residual = singles.numerator - eia * t1
    pair_denominator = eia.reshape(-1)
    doubles_residual = doubles_equation.core - (
        pair_denominator[:, None] * pair_matrix
        + pair_matrix * pair_denominator[None, :]
    )
    singles_norm_device = xp.linalg.norm(singles_residual)
    doubles_norm_device = xp.linalg.norm(doubles_residual)
    total_norm_device = xp.sqrt(
        singles_norm_device * singles_norm_device
        + doubles_norm_device * doubles_norm_device
    )
    singles_norm = _read_control_scalar(
        singles_norm_device,
        transfer_counter=transfer_counter,
        operation="thc_full_space_singles_residual",
    )
    doubles_norm = _read_control_scalar(
        doubles_norm_device,
        transfer_counter=transfer_counter,
        operation="thc_full_space_doubles_residual",
    )
    total_norm = _read_control_scalar(
        total_norm_device,
        transfer_counter=transfer_counter,
        operation="thc_full_space_total_residual",
    )
    largest = max(
        int(identity.nbytes),
        int(pair_matrix.nbytes),
        int(singles_residual.nbytes),
        int(doubles_residual.nbytes),
    )
    return FullSpaceResidualDiagnostic(
        residual_norm=total_norm,
        singles_residual_norm=singles_norm,
        doubles_residual_norm=doubles_norm,
        pair_dimension=dimension,
        largest_materialized_nbytes=largest,
    )


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
    """Validate the full-pair THC endpoint and return the complete RR result.

    The function retains its original name as an explicit compatibility seam.
    It does not apply an inexact replacement.  Such a replacement is blocked
    until Algorithms 1--3 and the complement are generated from the same
    decomposition of the complete RR residual.
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
    reference = rr_residual_algorithms_1_3(
        doubles,
        t1,
        loo,
        lov,
        lvv,
        auxiliary_block_size=auxiliary_block_size,
    )
    thc_core = factors.amplitude_core(doubles.core)
    approximation = thc_residual_algorithms_1_3(
        factors.y_occ,
        factors.y_vir,
        thc_core,
        t1,
        loo,
        lov,
        lvv,
        auxiliary_block_size=auxiliary_block_size,
    )
    backprojected = approximation.to_rr(factors.tau)
    delta = backprojected - reference.sigma_pair
    xp = _array_module(delta)
    delta_norm_device = xp.linalg.norm(delta)
    reference_norm_device = xp.linalg.norm(reference.sigma_pair)
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
    # Do not add even the roundoff-size delta.  The only scientifically safe
    # public equation in this tranche is the complete RR numerator itself.
    core = complete.core
    return THCHybridDoublesResult(
        core=core,
        complete_rr=complete,
        rr_reference_1_3=reference,
        thc_approximation_1_3=approximation,
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
    """Complete RR/CD iteration with an exact THC endpoint audit."""

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

    def jacobi(self, t1: Any, doubles: RRDoubles) -> RRIterationResult:
        """Evaluate complete RR and audit the exact THC endpoint without T2."""

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
        self.metrics.increment("iterations")
        return RRIterationResult(
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
            "equation": "complete-rr-rccsd-with-thc-endpoint-validation",
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
            "replacement_formula": None,
            "replacement_applied": False,
            "replacement_validation_atol": (
                self.replacement_validation_atol
            ),
            "replacement_validation_rtol": (
                self.replacement_validation_rtol
            ),
            "inexact_thc_enabled": False,
            "inexact_thc_blocker": (
                "shared RR/paper residual decomposition is not implemented"
            ),
            "singles_equation": "complete-rr-cd",
            "doubles_equation": "complete-rr-cd",
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
                "shared RR/paper residual decomposition is not implemented"
            ),
        }


__all__ = [
    "THCHybridDoublesResult",
    "FullSpaceResidualDiagnostic",
    "build_hybrid_projected_ccsd_doubles_numerator",
    "diagnose_reconstructed_full_space_residual",
    "THCRRCCSDIterationEngine",
]
