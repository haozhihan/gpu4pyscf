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

"""GPU-resident RR-CCSD iteration engine over three-index MO factors.

This module owns the production-shaped iteration path.  It consumes a fixed
semi-unitary RR projector and :class:`MOThreeIndexIntegralProvider`, evaluates
the singles and projected doubles equations without calling canonical
``update_amps``, solves both denominators, and keeps DIIS in compressed space.

The ring selector defaults to the exact bounded-memory reference kernel and
can opt into the numerically equivalent tiled GEMM path. Both variants remain
performance ineligible until the A100 gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from gpu4pyscf.cc.compressed_diis import CompressedDIIS
from gpu4pyscf.cc.device_runtime import RunMetrics, nvtx_range
from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.rr_residual import (
    ProjectedPairDenominator,
    RRCCSDDoublesResult,
    RRCCSDSinglesResult,
    build_ccsd_singles_numerator,
    build_projected_ccsd_doubles_numerator,
    rr_ccsd_energy,
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
class RRIterationResult:
    """One RR equation evaluation and Jacobi update on the resident backend."""

    t1: Any
    doubles: RRDoubles
    error_t1: Any
    error_core: Any
    update_norm: Any
    equation_residual_t1: Any
    equation_residual_core: Any
    residual_norm: Any
    energy: Any
    singles_equation: RRCCSDSinglesResult
    doubles_equation: RRCCSDDoublesResult


@dataclass(frozen=True)
class RRKernelResult:
    """Converged or exhausted compressed RR iteration state."""

    converged: bool
    cycles: int
    energy: float
    t1: Any
    doubles: RRDoubles
    residual_norm: float
    update_norm: float
    history: tuple[dict[str, float], ...]


class RRCCSDIterationEngine:
    """Standalone RR/CD residual, denominator, convergence, and DIIS driver."""

    def __init__(
        self,
        integrals: MOThreeIndexIntegralProvider,
        projector: RRProjector,
        fock: Any,
        orbital_energies: Any,
        *,
        level_shift: float = 0.0,
        auxiliary_block_size: int = 1,
        virtual_block_size: int = 8,
        rr_ring_kernel: str = "reference",
        metrics: Optional[RunMetrics] = None,
    ) -> None:
        if not isinstance(integrals, MOThreeIndexIntegralProvider):
            raise TypeError("integrals must be an MOThreeIndexIntegralProvider")
        if integrals.factorization not in {"cd", "df", "custom"}:
            raise ValueError("unsupported three-index factorization")
        factors = integrals.factors
        if not _same_backend(
            factors, projector.vectors, projector.eigenvalues, fock, orbital_energies
        ):
            raise TypeError("integrals, projector, Fock, and energies need one backend")
        nocc, nvir, nmo = integrals.nocc, integrals.nvir, integrals.nmo
        if projector.full_dimension != nocc * nvir:
            raise ValueError("projector pair dimension does not match integrals")
        if fock.shape != (nmo, nmo):
            raise ValueError("fock must have shape (nmo,nmo)")
        if orbital_energies.shape != (nmo,):
            raise ValueError("orbital_energies must have shape (nmo,)")
        if auxiliary_block_size < 1 or virtual_block_size < 1:
            raise ValueError("auxiliary and virtual block sizes must be positive")
        rr_ring_kernel = str(rr_ring_kernel).lower()
        if rr_ring_kernel not in {"reference", "gemm"}:
            raise ValueError("rr_ring_kernel must be 'reference' or 'gemm'")
        self.integrals = integrals
        self.projector = projector
        self.fock = fock
        self.orbital_energies = orbital_energies
        self.level_shift = float(level_shift)
        self.auxiliary_block_size = int(auxiliary_block_size)
        self.virtual_block_size = int(virtual_block_size)
        self.rr_ring_kernel = rr_ring_kernel
        self.metrics = metrics or RunMetrics("rr-ccsd-iteration-engine")
        self.xp = _array_module(factors)
        self.nocc = nocc
        self.nvir = nvir
        self.fock_oo = fock[:nocc, :nocc]
        self.fock_ov = fock[:nocc, nocc:]
        self.fock_vv = fock[nocc:, nocc:]
        self.occupied_energies = orbital_energies[:nocc]
        self.virtual_energies = orbital_energies[nocc:]
        self.eia = (
            self.occupied_energies[:, None]
            - self.virtual_energies[None, :]
            - self.level_shift
        )
        self.denominator = ProjectedPairDenominator.build(
            projector,
            self.eia,
            nocc,
            nvir,
            transfer_counter=self.metrics.transfers,
        )

    def _read_scalar(self, value: Any) -> float:
        if self.xp is not np:
            self.metrics.transfers.record_d2h(int(value.nbytes))
        return float(value.item())

    def _validate_state(self, t1: Any, doubles: RRDoubles) -> None:
        if t1.shape != (self.nocc, self.nvir):
            raise ValueError("t1 shape does not match the iteration engine")
        if doubles.projector is not self.projector:
            raise ValueError("RR doubles must use the engine's fixed projector")
        if not _same_backend(self.integrals.factors, t1, doubles.core):
            raise TypeError("iteration state and integral provider need one backend")

    def jacobi(self, t1: Any, doubles: RRDoubles) -> RRIterationResult:
        """Evaluate one full RR/CD Jacobi update without dense doubles."""

        self._validate_state(t1, doubles)
        with nvtx_range("rr/full-singles-equation"):
            with self.metrics.phase("rr_singles_equation"):
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
        with nvtx_range("rr/full-doubles-equation"):
            with self.metrics.phase("rr_doubles_equation"):
                doubles_equation = build_projected_ccsd_doubles_numerator(
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
                    virtual_block_size=self.virtual_block_size,
                    ring_kernel=self.rr_ring_kernel,
                    transfer_counter=self.metrics.transfers,
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
            + self.xp.vdot(equation_residual_core, equation_residual_core).real
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
        *,
        max_cycle: int = 50,
        conv_tol: float = 1e-8,
        conv_tol_normt: float = 1e-6,
        diis_space: int = 6,
        diis_start_cycle: int = 1,
    ) -> RRKernelResult:
        """Iterate with DIIS and gate on a state-consistent equation residual.

        Each history row describes the amplitudes that entered that equation
        evaluation.  The Jacobi/DIIS candidate becomes the next row's state,
        so its energy and projected residual are never mixed with its parent.
        """

        self._validate_state(t1, doubles)
        if max_cycle < 1:
            raise ValueError("max_cycle must be positive")
        if conv_tol <= 0 or conv_tol_normt <= 0:
            raise ValueError("convergence tolerances must be positive")
        if diis_start_cycle < 0:
            raise ValueError("diis_start_cycle must be non-negative")
        diis = CompressedDIIS(
            space=diis_space,
            minimum_history=2,
            regularization=1e-12,
        )
        history: list[dict[str, float]] = []
        state_t1 = t1
        state_doubles = doubles
        previous_energy: Optional[float] = None
        converged = False
        energy_value = float("nan")
        residual_value = float("inf")
        update_value = float("inf")
        cycles = 0
        for cycle in range(1, int(max_cycle) + 1):
            with nvtx_range(f"rr/iteration-{cycle}"):
                with self.metrics.phase("rr_iteration", cycle=cycle):
                    update = self.jacobi(state_t1, state_doubles)
                    energy_value = self._read_scalar(update.energy)
                    residual_value = self._read_scalar(update.residual_norm)
                    update_value = self._read_scalar(update.update_norm)
            delta_energy = (
                float("inf")
                if previous_energy is None
                else energy_value - previous_energy
            )
            history.append({
                "cycle": float(cycle),
                "energy": energy_value,
                "delta_energy": delta_energy,
                "residual_norm": residual_value,
                "update_norm": update_value,
            })
            cycles = cycle
            if (
                previous_energy is not None
                and abs(delta_energy) <= conv_tol
                and residual_value <= conv_tol_normt
            ):
                converged = True
                break
            previous_energy = energy_value
            if cycle == int(max_cycle):
                break
            candidate_t1 = update.t1
            candidate_core = update.doubles.core
            if cycle >= diis_start_cycle:
                extrapolated = diis.update(
                    candidate_t1,
                    candidate_core,
                    update.error_t1,
                    update.error_core,
                )
                candidate_t1 = extrapolated.t1
                candidate_core = extrapolated.core
            state_t1 = candidate_t1
            state_doubles = RRDoubles(
                self.projector,
                candidate_core,
                self.nocc,
                self.nvir,
            )
        self.metrics.metadata.update({
            "equation": "complete-projected-rccsd",
            "integrals": self.integrals.metadata(),
            "projector_rank": self.projector.rank,
            "ring_kernel": self.rr_ring_kernel,
            "performance_eligible": False,
            "diis": diis.metadata(),
            "converged": converged,
            "cycles": cycles,
        })
        return RRKernelResult(
            converged=converged,
            cycles=cycles,
            energy=energy_value,
            t1=state_t1,
            doubles=state_doubles,
            residual_norm=residual_value,
            update_norm=update_value,
            history=tuple(history),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "engine": "RRCCSDIterationEngine",
            "equation": "complete-projected-rccsd",
            "integrals": self.integrals.metadata(),
            "projector_rank": self.projector.rank,
            "auxiliary_block_size": self.auxiliary_block_size,
            "virtual_block_size": self.virtual_block_size,
            "rr_ring_kernel": self.rr_ring_kernel,
            "dense_t2_in_iteration": False,
            "four_index_eri_in_iteration": False,
            "ring_kernel": self.rr_ring_kernel,
            "convergence_norm": "projected-equation-residual-frobenius",
            "performance_eligible": False,
        }


__all__ = [
    "RRIterationResult",
    "RRKernelResult",
    "RRCCSDIterationEngine",
]
