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

"""Explicit full pair-space residual diagnostics for low-rank RCCSD.

The diagnostic is representation-neutral: it reconstructs an ``RRDoubles``
state in the identity pair basis and evaluates the complete RR/CD equations.
It is an opt-in, post-convergence operation and never participates in the
normal compressed iteration path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.rr_residual import (
    build_ccsd_singles_numerator,
    build_projected_ccsd_doubles_numerator,
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


__all__ = [
    "FullSpaceResidualDiagnostic",
    "diagnose_reconstructed_full_space_residual",
]
