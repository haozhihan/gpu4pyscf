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

"""Factorized building blocks for the projected RR-CCSD residual.

This module is deliberately separate from the canonical ``update_amps``. Each
term is introduced with a dense reference test before it is allowed into the
production RR driver. The first term implemented here is the Coulomb
particle-particle ladder

``R_ijab += 1/2 sum(A,c,d) L_Aab L_Acd tau_ijcd``.

It consumes an RR core and produces a projected RR-core residual without ever
constructing ``t2[i,j,a,b]``. Exchange/permutation terms and the remaining
CCSD diagrams are separate gates; this kernel alone is not a complete CCSD
update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


@dataclass(frozen=True)
class RRResidualTermResult:
    """One projected residual contribution plus allocation diagnostics."""

    core: Any
    term: str
    auxiliary_block_size: int
    largest_intermediate_nbytes: int
    factor_symmetry_error: Optional[float] = None
    materialized_dense_t2: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "auxiliary_block_size": int(self.auxiliary_block_size),
            "largest_intermediate_nbytes": int(self.largest_intermediate_nbytes),
            "factor_symmetry_error": (
                None
                if self.factor_symmetry_error is None
                else float(self.factor_symmetry_error)
            ),
            "materialized_dense_t2": bool(self.materialized_dense_t2),
        }


@dataclass(frozen=True)
class RRSingleTermResult:
    """One full-space singles residual contribution from compressed doubles."""

    amplitudes: Any
    term: str
    largest_intermediate_nbytes: int
    materialized_dense_t2: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "largest_intermediate_nbytes": int(
                self.largest_intermediate_nbytes
            ),
            "materialized_dense_t2": bool(self.materialized_dense_t2),
        }


@dataclass(frozen=True)
class CCFockIntermediates:
    """Restricted CC ``F_oo``, ``F_ov`` and ``F_vv`` built from RR doubles."""

    foo: Any
    fov: Any
    fvv: Any
    auxiliary_block_size: int
    largest_intermediate_nbytes: int
    materialized_dense_t2: bool = False
    materialized_dense_ovov: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "equations": "PySCF-rintermediates-cc_Foo-cc_Fov-cc_Fvv",
            "auxiliary_block_size": int(self.auxiliary_block_size),
            "largest_intermediate_nbytes": int(
                self.largest_intermediate_nbytes
            ),
            "materialized_dense_t2": bool(self.materialized_dense_t2),
            "materialized_dense_ovov": bool(self.materialized_dense_ovov),
        }


@dataclass(frozen=True)
class RRCCSDEnergyResult:
    """Resident restricted-CCSD correlation energy from RR amplitudes."""

    value: Any
    auxiliary_block_size: int
    largest_intermediate_nbytes: int
    materialized_dense_t2: bool = False
    materialized_dense_ovov: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "equation": "restricted-ccsd-correlation-energy",
            "auxiliary_block_size": int(self.auxiliary_block_size),
            "largest_intermediate_nbytes": int(
                self.largest_intermediate_nbytes
            ),
            "materialized_dense_t2": bool(self.materialized_dense_t2),
            "materialized_dense_ovov": bool(self.materialized_dense_ovov),
        }


@dataclass(frozen=True)
class RRCCSDSinglesResult:
    """Complete restricted-CCSD singles numerator from RR/CD tensors."""

    numerator: Any
    fock_intermediates: CCFockIntermediates
    term_metadata: tuple[dict[str, Any], ...]
    materialized_dense_t2: bool = False
    materialized_four_index_eri: bool = False

    def amplitudes(self, orbital_energy_differences: Any):
        if orbital_energy_differences.shape != self.numerator.shape:
            raise ValueError("singles denominator shape does not match numerator")
        if not _same_backend(self.numerator, orbital_energy_differences):
            raise TypeError("singles numerator and denominator need one backend")
        return self.numerator / orbital_energy_differences

    def metadata(self) -> dict[str, Any]:
        return {
            "equation": "Hirata-2004-restricted-CCSD-singles",
            "fock_intermediates": self.fock_intermediates.metadata(),
            "terms": list(self.term_metadata),
            "materialized_dense_t2": bool(self.materialized_dense_t2),
            "materialized_four_index_eri": bool(
                self.materialized_four_index_eri
            ),
        }


@dataclass(frozen=True)
class RRCCSDDoublesResult:
    """Complete projected restricted-CCSD doubles numerator."""

    core: Any
    fock_intermediates: CCFockIntermediates
    lagrangian_one_body: "CCLagrangianOneBody"
    term_metadata: tuple[dict[str, Any], ...]
    materialized_dense_t2: bool = False
    materialized_four_index_eri: bool = False

    def solve(
        self,
        denominator: "ProjectedPairDenominator",
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

    def metadata(self) -> dict[str, Any]:
        return {
            "equation": "Hirata-2004-restricted-CCSD-doubles-projected",
            "fock_intermediates": self.fock_intermediates.metadata(),
            "lagrangian_one_body": self.lagrangian_one_body.metadata(),
            "terms": list(self.term_metadata),
            "materialized_dense_t2": bool(self.materialized_dense_t2),
            "materialized_four_index_eri": bool(
                self.materialized_four_index_eri
            ),
        }


@dataclass(frozen=True)
class CCLagrangianOneBody:
    """Restricted-CC ``L_oo`` and ``L_vv`` intermediates."""

    loo: Any
    lvv: Any
    largest_intermediate_nbytes: int
    materialized_four_index_eri: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "equations": "PySCF-rintermediates-Loo-Lvv",
            "largest_intermediate_nbytes": int(
                self.largest_intermediate_nbytes
            ),
            "materialized_four_index_eri": bool(
                self.materialized_four_index_eri
            ),
        }


def _same_backend(reference: Any, *values: Any) -> bool:
    xp = _array_module(reference)
    return all(_array_module(value) is xp for value in values)


def _require_real_floating(*values: Any) -> None:
    dtypes = tuple(np.dtype(value.dtype) for value in values)
    if any(np.issubdtype(dtype, np.complexfloating) for dtype in dtypes):
        raise NotImplementedError("this RR residual kernel is restricted to real RHF")
    if any(not np.issubdtype(dtype, np.floating) for dtype in dtypes):
        raise TypeError("RR residual kernels require real floating-point tensors")


@dataclass(frozen=True)
class ProjectedPairDenominator:
    """RR-space representation of the orbital-energy Lyapunov operator."""

    matrix: Any
    eigenvalues: Any
    eigenvectors: Any
    nocc: int
    nvir: int
    singular_tolerance: float
    minimum_pair_denominator: float
    host_scalar_reads: int

    @classmethod
    def build(
        cls,
        projector: RRProjector,
        orbital_energy_differences: Any,
        nocc: int,
        nvir: int,
        *,
        singular_tolerance: float = 1e-12,
        transfer_counter: Any = None,
    ) -> "ProjectedPairDenominator":
        """Build ``K = U.H @ diag(e_i-e_a) @ U`` on U's backend."""

        vectors = projector.vectors
        xp = _array_module(vectors)
        if not _same_backend(vectors, orbital_energy_differences):
            raise TypeError("projector and orbital energy differences need one backend")
        _require_real_floating(vectors, orbital_energy_differences)
        if orbital_energy_differences.shape == (nocc, nvir):
            differences = orbital_energy_differences.reshape(-1)
        elif orbital_energy_differences.shape == (nocc * nvir,):
            differences = orbital_energy_differences
        else:
            raise ValueError(
                "orbital_energy_differences must have shape (nocc,nvir) "
                "or (nocc*nvir,)"
            )
        if nocc * nvir != projector.full_dimension:
            raise ValueError("nocc*nvir does not match projector pair dimension")
        if singular_tolerance <= 0:
            raise ValueError("singular_tolerance must be positive")
        matrix = vectors.T @ (differences[:, None] * vectors)
        matrix = (matrix + matrix.T) * 0.5
        eigenvalues, eigenvectors = xp.linalg.eigh(matrix)
        pair_denominators = eigenvalues[:, None] + eigenvalues[None, :]
        minimum_value = xp.min(xp.abs(pair_denominators))
        host_scalar_reads = 0
        if xp is not np:
            if transfer_counter is None:
                raise ValueError(
                    "GPU denominator validation requires an explicit "
                    "transfer_counter"
                )
            transfer_counter.record_d2h(int(minimum_value.nbytes))
            host_scalar_reads = 1
        minimum = float(minimum_value.item())
        if minimum <= singular_tolerance:
            raise np.linalg.LinAlgError(
                "projected pair denominator is singular at "
                f"{singular_tolerance:.3e}"
            )
        return cls(
            matrix=matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            nocc=int(nocc),
            nvir=int(nvir),
            singular_tolerance=float(singular_tolerance),
            minimum_pair_denominator=minimum,
            host_scalar_reads=host_scalar_reads,
        )

    @property
    def rank(self) -> int:
        return int(self.matrix.shape[0])

    def solve(
        self,
        projected_numerator: Any,
        *,
        symmetry_tolerance: float = 1e-12,
        transfer_counter: Any = None,
    ):
        """Solve ``K C + C K = B`` without a dense pair-space denominator."""

        if projected_numerator.shape != (self.rank, self.rank):
            raise ValueError("projected numerator shape does not match RR rank")
        if not _same_backend(self.matrix, projected_numerator):
            raise TypeError("denominator and projected numerator need one backend")
        _require_real_floating(projected_numerator)
        if symmetry_tolerance < 0:
            raise ValueError("symmetry_tolerance must be non-negative")
        xp = _array_module(projected_numerator)
        asymmetry = xp.max(
            xp.abs(projected_numerator - projected_numerator.T)
        )
        if xp is not np:
            if transfer_counter is None:
                raise ValueError(
                    "GPU numerator validation requires an explicit "
                    "transfer_counter"
                )
            transfer_counter.record_d2h(int(asymmetry.nbytes))
        asymmetry_value = float(asymmetry.item())
        if asymmetry_value > symmetry_tolerance:
            raise ValueError(
                "projected numerator is not symmetric: error "
                f"{asymmetry_value:.3e} exceeds {symmetry_tolerance:.3e}"
            )
        rotated = self.eigenvectors.T @ projected_numerator @ self.eigenvectors
        rotated /= self.eigenvalues[:, None] + self.eigenvalues[None, :]
        result = self.eigenvectors @ rotated @ self.eigenvectors.T
        return (result + result.T) * 0.5

    def equation_residual(self, core: Any, projected_numerator: Any):
        return self.matrix @ core + core @ self.matrix - projected_numerator

    def metadata(self) -> dict[str, Any]:
        return {
            "operator": "projected-orbital-energy-lyapunov",
            "rank": self.rank,
            "nocc": self.nocc,
            "nvir": self.nvir,
            "singular_tolerance": float(self.singular_tolerance),
            "minimum_pair_denominator": float(self.minimum_pair_denominator),
            "host_scalar_reads": int(self.host_scalar_reads),
            "materializes_pair_denominator": False,
        }


def projected_bare_ovov(
    projector: RRProjector,
    lov: Any,
    nocc: int,
    nvir: int,
) -> RRResidualTermResult:
    """Project the bare ``(ia|jb)`` term directly from three-index factors."""

    vectors = projector.vectors
    if not _same_backend(vectors, lov):
        raise TypeError("projector and L_ov must use one array backend")
    if getattr(lov, "ndim", None) != 3 or lov.shape[1:] != (nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    if nocc * nvir != projector.full_dimension:
        raise ValueError("nocc*nvir does not match projector pair dimension")
    xp = _array_module(vectors)
    transformed = lov.reshape(lov.shape[0], -1).conj() @ vectors
    core = transformed.T.conj() @ transformed
    core = (core + core.T.conj()) * 0.5
    return RRResidualTermResult(
        core=core,
        term="bare-ovov",
        auxiliary_block_size=int(lov.shape[0]),
        largest_intermediate_nbytes=int(transformed.nbytes),
    )


def projected_linear_t1_doubles(
    projector: RRProjector,
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
) -> RRResidualTermResult:
    """Project the two explicit linear-``t1`` T2-numerator groups.

    This is the factorized form of the two ``tmp2``/``tmp`` groups before
    PySCF's ``cc_W*`` intermediates are applied.  Their pair-transposed copies
    are accumulated in RR space.  The bare ``ovov`` term is intentionally a
    separate contribution.
    """

    vectors = projector.vectors
    if not _same_backend(vectors, t1, loo, lov, lvv):
        raise TypeError("projector, t1, and CD factors need one backend")
    _require_real_floating(vectors, t1, loo, lov, lvv)
    if getattr(t1, "ndim", None) != 2:
        raise ValueError("t1 must be a matrix")
    nocc, nvir = map(int, t1.shape)
    if projector.full_dimension != nocc * nvir:
        raise ValueError("projector dimension does not match t1")
    if getattr(lov, "ndim", None) != 3 or lov.shape[1:] != (nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    naux = int(lov.shape[0])
    if loo.shape != (naux, nocc, nocc):
        raise ValueError("loo must have shape (naux,nocc,nocc)")
    if lvv.shape != (naux, nvir, nvir):
        raise ValueError("lvv must have shape (naux,nvir,nvir)")
    block_size = naux if auxiliary_block_size is None else int(
        auxiliary_block_size
    )
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")
    xp = _array_module(vectors)
    rank = projector.rank
    u = vectors.reshape(nocc, nvir, rank)
    half_virtual = xp.zeros((rank, rank), dtype=xp.result_type(
        vectors, t1, loo, lov, lvv
    ))
    half_occupied = xp.zeros_like(half_virtual)
    largest = 0
    for start in range(0, naux, block_size):
        stop = min(start + block_size, naux)
        oo = loo[start:stop]
        ov = lov[start:stop]
        vv = lvv[start:stop]

        oo_singles = xp.einsum("Aki,ka->Aia", oo, t1)
        vv_singles = xp.einsum("Acb,jc->Ajb", vv, t1)
        projected_left = xp.einsum(
            "iaP,Aia->AP", u, ov - oo_singles
        )
        projected_right = xp.einsum("jbQ,Ajb->AQ", u, vv_singles)
        half_virtual += projected_left.T @ projected_right

        ov_singles = xp.einsum("Akc,jc->Akj", ov, t1)
        occupied_right = xp.einsum("Akj,kb->Ajb", ov_singles, t1)
        occupied_right += xp.einsum("Ajk,kb->Ajb", oo, t1)
        projected_ov = xp.einsum("iaP,Aia->AP", u, ov)
        projected_occupied = xp.einsum(
            "jbQ,Ajb->AQ", u, occupied_right
        )
        half_occupied += projected_ov.T @ projected_occupied
        largest = max(
            largest,
            *(int(value.nbytes) for value in (
                oo_singles,
                vv_singles,
                projected_left,
                projected_right,
                ov_singles,
                occupied_right,
                projected_ov,
                projected_occupied,
            )),
        )
    projected = (
        half_virtual
        + half_virtual.T
        - half_occupied
        - half_occupied.T
    )
    return RRResidualTermResult(
        core=projected,
        term="explicit-linear-t1-doubles",
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=largest,
    )


def build_cc_fock_intermediates(
    doubles: RRDoubles,
    t1: Any,
    lov: Any,
    fock_oo: Any,
    fock_ov: Any,
    fock_vv: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
) -> CCFockIntermediates:
    """Build the standard restricted-CC Fock intermediates from ``L_ov``.

    This is the factorized form of PySCF ``rintermediates.cc_Foo``,
    ``cc_Fov`` and ``cc_Fvv``.  It contracts a resident RR core and the
    ``t1*t1`` contribution without constructing either ``t2`` or ``ovov``.
    Blocking controls the largest ``(A,k,i,P)`` temporary.
    """

    vectors = doubles.projector.vectors
    core = doubles.core
    values = (core, t1, lov, fock_oo, fock_ov, fock_vv)
    if not _same_backend(vectors, *values):
        raise TypeError("RR amplitudes, L_ov, and Fock blocks need one backend")
    _require_real_floating(vectors, *values)
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if t1.shape != (nocc, nvir):
        raise ValueError("t1 shape does not match RR doubles")
    if getattr(lov, "ndim", None) != 3 or lov.shape[1:] != (nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    if fock_oo.shape != (nocc, nocc):
        raise ValueError("fock_oo shape does not match RR doubles")
    if fock_ov.shape != (nocc, nvir):
        raise ValueError("fock_ov shape does not match RR doubles")
    if fock_vv.shape != (nvir, nvir):
        raise ValueError("fock_vv shape does not match RR doubles")
    naux = int(lov.shape[0])
    block_size = naux if auxiliary_block_size is None else int(
        auxiliary_block_size
    )
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")
    xp = _array_module(vectors)
    u = vectors.reshape(nocc, nvir, rank)
    dtype = xp.result_type(vectors, core, t1, lov, fock_oo, fock_ov, fock_vv)
    foo = xp.array(fock_oo, dtype=dtype, copy=True)
    fov = xp.array(fock_ov, dtype=dtype, copy=True)
    fvv = xp.array(fock_vv, dtype=dtype, copy=True)
    largest = 0

    for start in range(0, naux, block_size):
        factors = lov[start:min(start + block_size, naux)]
        # x[A,k,i,P] = sum(c) L[A,k,c] U[i,c,P]
        x = xp.einsum("Akc,icP->AkiP", factors, u)
        projected_trace = xp.einsum("Ald,ldQ->AQ", factors, u)
        core_trace = projected_trace @ core.T

        direct_oo = xp.einsum("AkiP,AP->ki", x, core_trace)
        x_core = xp.einsum("AliP,PQ->AliQ", x, core)
        exchange_oo = xp.einsum("AliQ,AklQ->ki", x_core, x)
        foo += 2.0 * direct_oo - exchange_oo

        direct_vv = xp.einsum("Akc,kaP,AP->ac", factors, u, core_trace)
        # z[A,k,l,P] = sum(Q) x[A,k,l,Q] C[P,Q]
        z = xp.einsum("AklQ,PQ->AklP", x, core)
        exchange_vv = xp.einsum("kaP,AklP,Alc->ac", u, z, factors)
        fvv += -2.0 * direct_vv + exchange_vv

        t1_trace = xp.einsum("Ald,ld->A", factors, t1)
        x_t1 = xp.einsum("Akc,ic->Aki", factors, t1)
        foo += 2.0 * xp.einsum("Aki,A->ki", x_t1, t1_trace)
        foo -= xp.einsum("Ali,Akl->ki", x_t1, x_t1)

        fvv -= 2.0 * xp.einsum("Akc,ka,A->ac", factors, t1, t1_trace)
        crossed_t1 = xp.einsum("Akd,ld->Akl", factors, t1)
        fvv += xp.einsum("ka,Akl,Alc->ac", t1, crossed_t1, factors)

        fov += 2.0 * xp.einsum("Akc,A->kc", factors, t1_trace)
        fov -= xp.einsum("Akl,Alc->kc", crossed_t1, factors)

        largest = max(
            largest,
            *(int(value.nbytes) for value in (
                x,
                projected_trace,
                core_trace,
                x_core,
                z,
                x_t1,
                crossed_t1,
            )),
        )
    return CCFockIntermediates(
        foo=foo,
        fov=fov,
        fvv=fvv,
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=largest,
    )


def rr_ccsd_energy(
    doubles: RRDoubles,
    t1: Any,
    fock_ov: Any,
    lov: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
) -> RRCCSDEnergyResult:
    """Evaluate the RCCSD correlation energy without reconstructing ``t2``.

    The two-electron part evaluates both the Coulomb and exchange contractions
    of ``tau = t2 + t1*t1`` from ``L_ov``.  ``value`` remains a scalar on the
    input backend; convergence/reporting code owns any explicit device read.
    """

    vectors = doubles.projector.vectors
    core = doubles.core
    if not _same_backend(vectors, core, t1, fock_ov, lov):
        raise TypeError("RR amplitudes, F_ov, and L_ov need one backend")
    _require_real_floating(vectors, core, t1, fock_ov, lov)
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if t1.shape != (nocc, nvir) or fock_ov.shape != (nocc, nvir):
        raise ValueError("t1 and F_ov shapes must match RR doubles")
    if getattr(lov, "ndim", None) != 3 or lov.shape[1:] != (nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    naux = int(lov.shape[0])
    block_size = naux if auxiliary_block_size is None else int(
        auxiliary_block_size
    )
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")
    xp = _array_module(vectors)
    u = vectors.reshape(nocc, nvir, rank)
    value = 2.0 * xp.einsum("ia,ia->", fock_ov, t1)
    direct = xp.zeros((), dtype=xp.result_type(core, t1, lov))
    exchange = xp.zeros_like(direct)
    largest = 0
    for start in range(0, naux, block_size):
        factors = lov[start:min(start + block_size, naux)]
        flattened = factors.reshape(factors.shape[0], -1)
        projected = flattened @ vectors
        singles_trace = flattened @ t1.reshape(-1)
        direct += xp.einsum("AP,PQ,AQ->", projected, core, projected)
        direct += xp.einsum("A,A->", singles_trace, singles_trace)

        # X[A,P,i,j] = sum(a) U[i,a,P] L[A,j,a].  The trace of
        # X[A,P] X[A,Q] gives the exchange pair grouping.
        crossed = xp.einsum("iaP,Aja->APij", u, factors)
        exchange += xp.einsum(
            "APij,PQ,AQji->", crossed, core, crossed
        )
        crossed_singles = xp.einsum("ia,Aja->Aij", t1, factors)
        exchange += xp.einsum(
            "Aij,Aji->", crossed_singles, crossed_singles
        )
        largest = max(
            largest,
            int(projected.nbytes),
            int(crossed.nbytes),
            int(crossed_singles.nbytes),
        )
    value += 2.0 * direct - exchange
    return RRCCSDEnergyResult(
        value=value,
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=largest,
    )


def build_ccsd_singles_numerator(
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
    auxiliary_block_size: Optional[int] = None,
) -> RRCCSDSinglesResult:
    """Build the complete real-RHF CCSD singles numerator in RR/CD form.

    The expression follows Hirata *et al.* Eqs. (35)--(36), as implemented by
    PySCF ``rccsd.update_amps``.  Every doubles occurrence is contracted from
    ``U @ core @ U.T`` and every ERI occurrence from ``L``; neither dense
    object is formed.
    """

    vectors = doubles.projector.vectors
    core = doubles.core
    values = (
        core,
        t1,
        fock_oo,
        fock_ov,
        fock_vv,
        occupied_energies,
        virtual_energies,
        loo,
        lov,
        lvv,
    )
    if not _same_backend(vectors, *values):
        raise TypeError("all RR/CD singles inputs need one array backend")
    _require_real_floating(vectors, *values)
    nocc, nvir = doubles.nocc, doubles.nvir
    if occupied_energies.shape != (nocc,):
        raise ValueError("occupied_energies must have shape (nocc,)")
    if virtual_energies.shape != (nvir,):
        raise ValueError("virtual_energies must have shape (nvir,)")
    xp = _array_module(vectors)
    f_intermediates = build_cc_fock_intermediates(
        doubles,
        t1,
        lov,
        fock_oo,
        fock_ov,
        fock_vv,
        auxiliary_block_size=auxiliary_block_size,
    )
    foo = f_intermediates.foo.copy()
    fvv = f_intermediates.fvv.copy()
    occupied_indices = xp.arange(nocc)
    virtual_indices = xp.arange(nvir)
    foo[occupied_indices, occupied_indices] -= occupied_energies
    fvv[virtual_indices, virtual_indices] -= (
        virtual_energies + float(level_shift)
    )

    numerator = -2.0 * xp.einsum("kc,ka,ic->ia", fock_ov, t1, t1)
    numerator += xp.einsum("ac,ic->ia", fvv, t1)
    numerator -= xp.einsum("ki,ka->ia", foo, t1)

    fov_doubles = singles_from_fov_doubles(doubles, f_intermediates.fov)
    numerator += fov_doubles.amplitudes
    numerator += xp.einsum("kc,ic,ka->ia", f_intermediates.fov, t1, t1)
    numerator += fock_ov

    linear_integrals = singles_from_cd_ovvo_oovv(t1, loo, lov, lvv)
    ovvv = singles_from_cd_ovvv(
        doubles,
        t1,
        lov,
        lvv,
        auxiliary_block_size=auxiliary_block_size,
    )
    ovoo = singles_from_cd_ovoo(
        doubles,
        t1,
        lov,
        loo,
        auxiliary_block_size=auxiliary_block_size,
    )
    numerator += linear_integrals.amplitudes
    numerator += ovvv.amplitudes
    numerator += ovoo.amplitudes
    return RRCCSDSinglesResult(
        numerator=numerator,
        fock_intermediates=f_intermediates,
        term_metadata=tuple(
            term.metadata()
            for term in (fov_doubles, linear_integrals, ovvv, ovoo)
        ),
    )


def build_cc_lagrangian_one_body(
    fock_intermediates: CCFockIntermediates,
    t1: Any,
    fock_ov: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
) -> CCLagrangianOneBody:
    """Build PySCF ``rintermediates.Loo`` and ``Lvv`` from CD factors."""

    values = (
        fock_intermediates.foo,
        fock_intermediates.fvv,
        t1,
        fock_ov,
        loo,
        lov,
        lvv,
    )
    if not _same_backend(values[0], *values[1:]):
        raise TypeError("one-body intermediate inputs need one backend")
    _require_real_floating(*values)
    if getattr(t1, "ndim", None) != 2:
        raise ValueError("t1 must be a matrix")
    nocc, nvir = map(int, t1.shape)
    naux = int(lov.shape[0]) if getattr(lov, "ndim", None) == 3 else -1
    if fock_intermediates.foo.shape != (nocc, nocc):
        raise ValueError("F_oo shape does not match t1")
    if fock_intermediates.fvv.shape != (nvir, nvir):
        raise ValueError("F_vv shape does not match t1")
    if fock_ov.shape != (nocc, nvir):
        raise ValueError("fock_ov shape does not match t1")
    if lov.shape != (naux, nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    if loo.shape != (naux, nocc, nocc):
        raise ValueError("loo must have shape (naux,nocc,nocc)")
    if lvv.shape != (naux, nvir, nvir):
        raise ValueError("lvv must have shape (naux,nvir,nvir)")
    xp = _array_module(t1)
    lagrangian_oo = fock_intermediates.foo.copy()
    lagrangian_oo += xp.einsum("kc,ic->ki", fock_ov, t1)
    ov_trace = xp.einsum("Alc,lc->A", lov, t1)
    lagrangian_oo += 2.0 * xp.einsum("Aki,A->ki", loo, ov_trace)
    crossed_oo = xp.einsum("Akc,Ali,lc->ki", lov, loo, t1)
    lagrangian_oo -= crossed_oo

    lagrangian_vv = fock_intermediates.fvv.copy()
    lagrangian_vv -= xp.einsum("kc,ka->ac", fock_ov, t1)
    lagrangian_vv += 2.0 * xp.einsum("Aac,A->ac", lvv, ov_trace)
    crossed_vv = xp.einsum("Akc,Aad,kd->ac", lov, lvv, t1)
    lagrangian_vv -= crossed_vv
    return CCLagrangianOneBody(
        loo=lagrangian_oo,
        lvv=lagrangian_vv,
        largest_intermediate_nbytes=max(
            int(ov_trace.nbytes),
            int(crossed_oo.nbytes),
            int(crossed_vv.nbytes),
        ),
    )


def projected_one_body_dressing(
    doubles: RRDoubles,
    ft_ij: Any,
    ft_ab: Any,
) -> RRResidualTermResult:
    """Project the final occupied/virtual Fock dressing of the T2 numerator."""

    vectors = doubles.projector.vectors
    core = doubles.core
    if not _same_backend(vectors, core, ft_ij, ft_ab):
        raise TypeError("RR doubles and Fock intermediates need one backend")
    _require_real_floating(vectors, core, ft_ij, ft_ab)
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if ft_ij.shape != (nocc, nocc) or ft_ab.shape != (nvir, nvir):
        raise ValueError("Fock intermediate shapes do not match RR doubles")
    xp = _array_module(vectors)
    u = vectors.reshape(nocc, nvir, rank)
    occupied_image = -xp.einsum("ki,kaQ->iaQ", ft_ij, u)
    virtual_image = xp.einsum("bc,jcQ->jbQ", ft_ab, u)
    flat_u = u.reshape(-1, rank)
    occupied_rr = flat_u.T @ occupied_image.reshape(-1, rank)
    virtual_rr = flat_u.T @ virtual_image.reshape(-1, rank)
    half = occupied_rr @ core + core @ virtual_rr.T
    result = half + half.T
    return RRResidualTermResult(
        core=result,
        term="one-body-fock-dressing",
        auxiliary_block_size=0,
        largest_intermediate_nbytes=max(
            int(occupied_image.nbytes), int(virtual_image.nbytes)
        ),
    )


def singles_from_fov_doubles(
    doubles: RRDoubles,
    fov: Any,
) -> RRSingleTermResult:
    """Return ``2 f_jb t_ijab - f_jb t_ijba`` without dense doubles."""

    vectors = doubles.projector.vectors
    core = doubles.core
    if not _same_backend(vectors, core, fov):
        raise TypeError("RR doubles and F_ov must use one array backend")
    _require_real_floating(vectors, core, fov)
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if fov.shape != (nocc, nvir):
        raise ValueError("F_ov shape does not match RR doubles")
    xp = _array_module(vectors)
    u = vectors.reshape(nocc, nvir, rank)

    # Direct pair grouping: sum_jb f[j,b] T[(i,a),(j,b)].
    projected_fov = vectors.T @ fov.reshape(-1)
    direct = (vectors @ (core @ projected_fov)).reshape(nocc, nvir)

    # Exchange grouping: T[i,j,b,a] crosses the two virtual indices.  Looping
    # over i keeps the only temporary at (nocc, rank), rather than creating an
    # (nvir,nvir,rank) object.
    exchange = xp.empty((nocc, nvir), dtype=xp.result_type(core, fov))
    largest = max(int(projected_fov.nbytes), int(direct.nbytes))
    for i in range(nocc):
        left = u[i] @ core
        weighted = fov @ left
        exchange[i] = xp.einsum("jQ,jaQ->a", weighted, u)
        largest = max(largest, int(left.nbytes), int(weighted.nbytes))
    return RRSingleTermResult(
        amplitudes=2.0 * direct - exchange,
        term="fov-doubles-singles",
        largest_intermediate_nbytes=largest,
    )


def singles_from_ovoo_doubles(
    doubles: RRDoubles,
    ovoo: Any,
) -> RRSingleTermResult:
    """Return the antisymmetrized ``ovoo * t2`` singles contribution.

    ``ovoo`` has the canonical GPU4PySCF layout ``(j,b,k,i)``.  The returned
    term is

    ``-sum(j,k,b) [2 ovoo[j,b,k,i] - ovoo[k,b,j,i]] t2[j,k,b,a]``.
    """

    vectors = doubles.projector.vectors
    core = doubles.core
    if not _same_backend(vectors, core, ovoo):
        raise TypeError("RR doubles and ovoo must use one array backend")
    _require_real_floating(vectors, core, ovoo)
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if ovoo.shape != (nocc, nvir, nocc, nocc):
        raise ValueError("ovoo shape does not match RR doubles")
    xp = _array_module(vectors)
    u = vectors.reshape(nocc, nvir, rank)
    transformed = 2.0 * xp.einsum("jbki,jbP->kiP", ovoo, u)
    transformed -= xp.einsum("kbji,jbP->kiP", ovoo, u)
    dressed = transformed.reshape(-1, rank) @ core
    dressed = dressed.reshape(nocc, nocc, rank)
    amplitudes = -xp.einsum("kiQ,kaQ->ia", dressed, u)
    return RRSingleTermResult(
        amplitudes=amplitudes,
        term="antisymmetrized-ovoo-doubles-singles",
        largest_intermediate_nbytes=max(
            int(transformed.nbytes), int(dressed.nbytes)
        ),
    )


def singles_from_cd_ovvv(
    doubles: RRDoubles,
    t1: Any,
    lov: Any,
    lvv: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
) -> RRSingleTermResult:
    """Return all ``ovvv`` terms in the RCCSD singles numerator from CD."""

    vectors = doubles.projector.vectors
    core = doubles.core
    if not _same_backend(vectors, core, t1, lov, lvv):
        raise TypeError("RR amplitudes and CD factors need one backend")
    _require_real_floating(vectors, core, t1, lov, lvv)
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if t1.shape != (nocc, nvir):
        raise ValueError("t1 shape does not match RR doubles")
    if getattr(lov, "ndim", None) != 3 or lov.shape[1:] != (nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    if getattr(lvv, "ndim", None) != 3 or lvv.shape != (
        lov.shape[0], nvir, nvir
    ):
        raise ValueError("lvv must have shape (naux,nvir,nvir)")
    naux = int(lov.shape[0])
    block_size = naux if auxiliary_block_size is None else int(
        auxiliary_block_size
    )
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")
    xp = _array_module(vectors)
    u = vectors.reshape(nocc, nvir, rank)
    result = xp.zeros((nocc, nvir), dtype=xp.result_type(core, t1, lov, lvv))
    largest = 0
    for start in range(0, naux, block_size):
        stop = min(start + block_size, naux)
        ov = lov[start:stop]
        vv = lvv[start:stop]
        projected_trace = xp.einsum("Akd,kdQ->AQ", ov, u)
        core_trace = projected_trace @ core.T
        direct = xp.einsum("Aac,icP,AP->ia", vv, u, core_trace)

        crossed_ov = xp.einsum("Akc,icP->AkiP", ov, u)
        crossed_ov = xp.einsum("AkiP,PQ->AkiQ", crossed_ov, core)
        virtual_image = xp.einsum("Aad,kdQ->AakQ", vv, u)
        exchange = xp.einsum("AkiQ,AakQ->ia", crossed_ov, virtual_image)

        singles_trace = xp.einsum("Akd,kd->A", ov, t1)
        direct_singles = xp.einsum("Aac,ic,A->ia", vv, t1, singles_trace)
        crossed_singles = xp.einsum("Akc,kd->Acd", ov, t1)
        exchange_singles = xp.einsum(
            "ic,Acd,Aad->ia", t1, crossed_singles, vv
        )
        result += (
            2.0 * direct
            - exchange
            + 2.0 * direct_singles
            - exchange_singles
        )
        largest = max(
            largest,
            *(int(value.nbytes) for value in (
                projected_trace,
                core_trace,
                crossed_ov,
                virtual_image,
                crossed_singles,
            )),
        )
    return RRSingleTermResult(
        amplitudes=result,
        term="cd-ovvv-singles",
        largest_intermediate_nbytes=largest,
    )


def singles_from_cd_ovoo(
    doubles: RRDoubles,
    t1: Any,
    lov: Any,
    loo: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
) -> RRSingleTermResult:
    """Return all ``ovoo`` terms in the RCCSD singles numerator from CD."""

    vectors = doubles.projector.vectors
    core = doubles.core
    if not _same_backend(vectors, core, t1, lov, loo):
        raise TypeError("RR amplitudes and CD factors need one backend")
    _require_real_floating(vectors, core, t1, lov, loo)
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if t1.shape != (nocc, nvir):
        raise ValueError("t1 shape does not match RR doubles")
    if getattr(lov, "ndim", None) != 3 or lov.shape[1:] != (nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    if getattr(loo, "ndim", None) != 3 or loo.shape != (
        lov.shape[0], nocc, nocc
    ):
        raise ValueError("loo must have shape (naux,nocc,nocc)")
    naux = int(lov.shape[0])
    block_size = naux if auxiliary_block_size is None else int(
        auxiliary_block_size
    )
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")
    xp = _array_module(vectors)
    u = vectors.reshape(nocc, nvir, rank)
    result = xp.zeros((nocc, nvir), dtype=xp.result_type(core, t1, lov, loo))
    largest = 0
    for start in range(0, naux, block_size):
        stop = min(start + block_size, naux)
        ov = lov[start:stop]
        oo = loo[start:stop]
        projected_trace = xp.einsum("Ajb,jbP->AP", ov, u)
        core_trace = projected_trace @ core
        direct = xp.einsum("Aki,kaQ,AQ->ia", oo, u, core_trace)

        crossed = xp.einsum("Akb,jbP->AkjP", ov, u)
        crossed = crossed @ core
        exchange = xp.einsum("Aji,AkjQ,kaQ->ia", oo, crossed, u)

        singles_trace = xp.einsum("Alc,lc->A", ov, t1)
        direct_singles = xp.einsum("Aki,ka,A->ia", oo, t1, singles_trace)
        crossed_singles = xp.einsum("Akc,lc->Akl", ov, t1)
        exchange_singles = xp.einsum(
            "Ali,Akl,ka->ia", oo, crossed_singles, t1
        )
        result += (
            -2.0 * direct
            + exchange
            - 2.0 * direct_singles
            + exchange_singles
        )
        largest = max(
            largest,
            *(int(value.nbytes) for value in (
                projected_trace,
                core_trace,
                crossed,
                crossed_singles,
            )),
        )
    return RRSingleTermResult(
        amplitudes=result,
        term="cd-ovoo-singles",
        largest_intermediate_nbytes=largest,
    )


def singles_from_cd_ovvo_oovv(
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
) -> RRSingleTermResult:
    """Return the linear-``t1`` ``2*ovvo-oovv`` singles contribution."""

    if not _same_backend(t1, loo, lov, lvv):
        raise TypeError("t1 and CD factors need one backend")
    _require_real_floating(t1, loo, lov, lvv)
    if getattr(t1, "ndim", None) != 2:
        raise ValueError("t1 must be a matrix")
    nocc, nvir = map(int, t1.shape)
    naux = int(lov.shape[0]) if getattr(lov, "ndim", None) == 3 else -1
    if lov.shape != (naux, nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    if loo.shape != (naux, nocc, nocc):
        raise ValueError("loo must have shape (naux,nocc,nocc)")
    if lvv.shape != (naux, nvir, nvir):
        raise ValueError("lvv must have shape (naux,nvir,nvir)")
    xp = _array_module(t1)
    trace = xp.einsum("Akc,kc->A", lov, t1)
    direct = 2.0 * xp.einsum("A,Aia->ia", trace, lov)
    exchange = xp.einsum("Aki,Aac,kc->ia", loo, lvv, t1)
    return RRSingleTermResult(
        amplitudes=direct - exchange,
        term="cd-linear-t1-ovvo-oovv-singles",
        largest_intermediate_nbytes=int(trace.nbytes),
    )


def projected_coulomb_ppl(
    doubles: RRDoubles,
    t1: Any,
    lvv: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
    factor_symmetry_tolerance: float = 1e-12,
    transfer_counter: Any = None,
) -> RRResidualTermResult:
    """Return the projected Coulomb PPL residual in the RR core basis.

    ``lvv`` uses ``(naux,nvir,nvir)`` layout. The implementation currently
    accepts real RHF tensors, which is the frozen benchmark contract. Blocking
    is over the Cholesky index; the largest temporary has shape
    ``(aux_block, rr_rank, rr_rank)``.
    """

    vectors = doubles.projector.vectors
    core = doubles.core
    xp = _array_module(vectors)
    if any(_array_module(value) is not xp for value in (core, t1, lvv)):
        raise TypeError("RR doubles, t1, and L_vv must use one array backend")
    dtypes = tuple(np.dtype(value.dtype) for value in (vectors, core, t1, lvv))
    if any(np.issubdtype(dtype, np.complexfloating) for dtype in dtypes):
        raise NotImplementedError("the first PPL kernel is restricted to real RHF")
    if any(not np.issubdtype(dtype, np.floating) for dtype in dtypes):
        raise TypeError("the first PPL kernel requires real floating-point tensors")
    if factor_symmetry_tolerance < 0:
        raise ValueError("factor_symmetry_tolerance must be non-negative")
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if t1.shape != (nocc, nvir):
        raise ValueError("t1 shape does not match RR doubles")
    if getattr(lvv, "ndim", None) != 3 or lvv.shape[1:] != (nvir, nvir):
        raise ValueError("lvv must have shape (naux,nvir,nvir)")
    naux = int(lvv.shape[0])
    symmetry_value = xp.max(xp.abs(lvv - lvv.transpose(0, 2, 1)))
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU symmetry validation requires an explicit transfer_counter"
            )
        transfer_counter.record_d2h(int(symmetry_value.nbytes))
    symmetry_error = float(symmetry_value.item())
    if symmetry_error > factor_symmetry_tolerance:
        raise ValueError(
            "L_vv is not symmetric in its virtual indices: "
            f"error {symmetry_error:.3e} exceeds "
            f"{factor_symmetry_tolerance:.3e}"
        )
    block_size = naux if auxiliary_block_size is None else int(
        auxiliary_block_size
    )
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")
    projector = vectors.reshape(nocc, nvir, rank)
    result = xp.zeros((rank, rank), dtype=xp.result_type(core, t1, lvv))
    largest = 0
    for start in range(0, naux, block_size):
        stop = min(start + block_size, naux)
        factors = lvv[start:stop]
        for i in range(nocc):
            ui = projector[i]
            for j in range(nocc):
                uj = projector[j]
                # B[A,P,Q] = U[i,:,P]^T L[A,:,:] U[j,:,Q]
                right = xp.einsum("Aab,bQ->AaQ", factors, uj)
                transformed = xp.einsum("aP,AaQ->APQ", ui, right)
                tau_trace = xp.einsum("APQ,PQ->A", transformed, core)
                tau_trace += xp.einsum("a,Aab,b->A", t1[i], factors, t1[j])
                result += 0.5 * xp.einsum(
                    "APQ,A->PQ", transformed, tau_trace
                )
                largest = max(
                    largest,
                    int(right.nbytes),
                    int(transformed.nbytes),
                    int(tau_trace.nbytes),
                )
    # Roundoff may introduce a tiny asymmetry although the pair-space ladder
    # and the RR core are symmetric for the real closed-shell contract.
    result = (result + result.T) * 0.5
    return RRResidualTermResult(
        core=result,
        term="coulomb-particle-particle-ladder",
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=largest,
        factor_symmetry_error=symmetry_error,
    )


def _projected_symmetric_ladder(
    doubles: RRDoubles,
    t1: Any,
    factors: Any,
    *,
    sector: str,
    auxiliary_block_size: Optional[int],
) -> RRResidualTermResult:
    vectors = doubles.projector.vectors
    core = doubles.core
    if not _same_backend(vectors, core, t1, factors):
        raise TypeError("RR amplitudes and ladder factors need one backend")
    _require_real_floating(vectors, core, t1, factors)
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if t1.shape != (nocc, nvir):
        raise ValueError("t1 shape does not match RR doubles")
    if sector == "virtual":
        expected_shape = (nvir, nvir)
        matrix_expression = "iaP,Aac,icR->APR"
        singles_expression = "iaP,Aac,ic->AP"
        term = "pure-vvvv-ladder"
    elif sector == "occupied":
        expected_shape = (nocc, nocc)
        matrix_expression = "iaP,Aki,kaR->APR"
        singles_expression = "iaP,Aki,ka->AP"
        term = "pure-oooo-ladder"
    else:  # pragma: no cover - private caller invariant
        raise ValueError("sector must be 'virtual' or 'occupied'")
    if getattr(factors, "ndim", None) != 3 or factors.shape[1:] != expected_shape:
        raise ValueError(
            f"{sector} ladder factors must have shape "
            f"(naux,{expected_shape[0]},{expected_shape[1]})"
        )
    naux = int(factors.shape[0])
    block_size = naux if auxiliary_block_size is None else int(
        auxiliary_block_size
    )
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")
    xp = _array_module(vectors)
    u = vectors.reshape(nocc, nvir, rank)
    result = xp.zeros((rank, rank), dtype=xp.result_type(core, t1, factors))
    largest = 0
    for start in range(0, naux, block_size):
        block = factors[start:min(start + block_size, naux)]
        transformed = xp.einsum(matrix_expression, u, block, u)
        singles = xp.einsum(singles_expression, u, block, t1)
        result += xp.einsum(
            "APR,RS,AQS->PQ", transformed, core, transformed
        )
        result += xp.einsum("AP,AQ->PQ", singles, singles)
        largest = max(largest, int(transformed.nbytes), int(singles.nbytes))
    result = (result + result.T) * 0.5
    return RRResidualTermResult(
        core=result,
        term=term,
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=largest,
    )


def projected_vvvv_ladder(
    doubles: RRDoubles,
    t1: Any,
    lvv: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
) -> RRResidualTermResult:
    """Project ``(ac|bd) * (t2+t1*t1)`` directly from ``L_vv``."""

    return _projected_symmetric_ladder(
        doubles,
        t1,
        lvv,
        sector="virtual",
        auxiliary_block_size=auxiliary_block_size,
    )


def projected_oooo_ladder(
    doubles: RRDoubles,
    t1: Any,
    loo: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
) -> RRResidualTermResult:
    """Project ``(ki|lj) * (t2+t1*t1)`` directly from ``L_oo``."""

    return _projected_symmetric_ladder(
        doubles,
        t1,
        loo,
        sector="occupied",
        auxiliary_block_size=auxiliary_block_size,
    )


def projected_cc_woooo(
    doubles: RRDoubles,
    t1: Any,
    loo: Any,
    lov: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
    virtual_block_size: int = 16,
    core_symmetry_tolerance: float = 1e-12,
    transfer_counter: Any = None,
) -> RRResidualTermResult:
    """Project the complete PySCF ``cc_Woooo * tau`` contribution.

    For real RHF/CD tensors, the occupied intermediate is

    ``W[k,l,i,j] = sum(A) B[A,k,i] B[A,l,j]``
    ``             + sum(A,P,Q) X[A,k,i,P] C[P,Q] X[A,l,j,Q]``

    with ``B[A,k,i] = Loo[A,k,i] + Lov[A,k,c] t1[i,c]`` and
    ``X[A,k,i,P] = Lov[A,k,c] U[i,c,P]``.  The first part is evaluated as a
    factorized ladder.  The second forms only the ``(nocc**4)`` intermediate
    and projects virtual tiles immediately, so neither dense ``t2`` nor a
    dense doubles residual is allocated.
    """

    vectors = doubles.projector.vectors
    core = doubles.core
    if not _same_backend(vectors, core, t1, loo, lov):
        raise TypeError("RR amplitudes, t1, and CD factors need one backend")
    _require_real_floating(vectors, core, t1, loo, lov)
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if t1.shape != (nocc, nvir):
        raise ValueError("t1 shape does not match RR doubles")
    if getattr(lov, "ndim", None) != 3 or lov.shape[1:] != (nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    if getattr(loo, "ndim", None) != 3 or loo.shape != (
        lov.shape[0], nocc, nocc
    ):
        raise ValueError("loo must have shape (naux,nocc,nocc)")
    if virtual_block_size < 1:
        raise ValueError("virtual_block_size must be positive")
    if core_symmetry_tolerance < 0:
        raise ValueError("core_symmetry_tolerance must be non-negative")
    xp = _array_module(vectors)
    asymmetry = xp.max(xp.abs(core - core.T))
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU core validation requires an explicit transfer_counter"
            )
        transfer_counter.record_d2h(int(asymmetry.nbytes))
    asymmetry_value = float(asymmetry.item())
    if asymmetry_value > core_symmetry_tolerance:
        raise ValueError(
            "RR doubles core is not symmetric: error "
            f"{asymmetry_value:.3e} exceeds "
            f"{core_symmetry_tolerance:.3e}"
        )

    naux = int(lov.shape[0])
    if auxiliary_block_size is None:
        itemsize = np.dtype(xp.result_type(vectors, core, lov)).itemsize
        bytes_per_auxiliary = max(1, nocc * nocc * rank * itemsize)
        block_size = min(
            naux,
            max(1, (64 * 1024 * 1024) // bytes_per_auxiliary),
        )
    else:
        block_size = int(auxiliary_block_size)
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")

    u = vectors.reshape(nocc, nvir, rank)
    transformed_loo = loo + xp.einsum("Akc,ic->Aki", lov, t1)
    factorized = _projected_symmetric_ladder(
        doubles,
        t1,
        transformed_loo,
        sector="occupied",
        auxiliary_block_size=block_size,
    )
    dtype = xp.result_type(vectors, core, t1, loo, lov)
    t2_dependent_woooo = xp.zeros(
        (nocc, nocc, nocc, nocc), dtype=dtype
    )
    largest = max(
        int(transformed_loo.nbytes),
        int(factorized.largest_intermediate_nbytes),
        int(t2_dependent_woooo.nbytes),
    )
    for start in range(0, naux, block_size):
        block = lov[start:min(start + block_size, naux)]
        transformed = xp.einsum("Akc,icP->AkiP", block, u)
        transformed_core = xp.einsum("AkiP,PQ->AkiQ", transformed, core)
        t2_dependent_woooo += xp.einsum(
            "AkiQ,AljQ->klij", transformed_core, transformed
        )
        largest = max(
            largest,
            int(transformed.nbytes),
            int(transformed_core.nbytes),
        )

    projected = xp.array(factorized.core, copy=True)
    vblock = int(virtual_block_size)
    for a_start in range(0, nvir, vblock):
        a_stop = min(a_start + vblock, nvir)
        ua = u[:, a_start:a_stop]
        ta = t1[:, a_start:a_stop]
        for b_start in range(0, nvir, vblock):
            b_stop = min(b_start + vblock, nvir)
            ub = u[:, b_start:b_stop]
            tb = t1[:, b_start:b_stop]
            tau = xp.einsum("kaR,RS,lbS->klab", ua, core, ub)
            tau += xp.einsum("ka,lb->klab", ta, tb)
            residual = xp.einsum(
                "klij,klab->ijab", t2_dependent_woooo, tau
            )
            left = xp.einsum("iaP,ijab->Pjb", ua, residual)
            projected += xp.einsum("Pjb,jbQ->PQ", left, ub)
            largest = max(
                largest,
                int(tau.nbytes),
                int(residual.nbytes),
                int(left.nbytes),
            )
    return RRResidualTermResult(
        core=projected,
        term="complete-cc-Woooo-ladder",
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=largest,
    )


def projected_cc_wvvvv(
    doubles: RRDoubles,
    t1: Any,
    lov: Any,
    lvv: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
) -> RRResidualTermResult:
    """Project the complete PySCF ``cc_Wvvvv * tau`` contribution.

    With ``M[A,a,c] = sum(k) t1[k,a] Lov[A,k,c]`` the full intermediate
    factorizes exactly as

    ``W[a,b,c,d] = (Lvv-M)[A,a,c] (Lvv-M)[A,b,d]``
    ``               - M[A,a,c] M[A,b,d]``.

    Both ladders are accumulated a Cholesky block at a time.  This avoids the
    dense ``vvvv`` tensor and does not retain the two ``(naux,nvir,nvir)``
    transformed-factor copies.
    """

    vectors = doubles.projector.vectors
    core = doubles.core
    if not _same_backend(vectors, core, t1, lov, lvv):
        raise TypeError("RR amplitudes, t1, and CD factors need one backend")
    _require_real_floating(vectors, core, t1, lov, lvv)
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if t1.shape != (nocc, nvir):
        raise ValueError("t1 shape does not match RR doubles")
    if getattr(lov, "ndim", None) != 3 or lov.shape[1:] != (nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    if getattr(lvv, "ndim", None) != 3 or lvv.shape != (
        lov.shape[0], nvir, nvir
    ):
        raise ValueError("lvv must have shape (naux,nvir,nvir)")
    naux = int(lov.shape[0])
    block_size = naux if auxiliary_block_size is None else int(
        auxiliary_block_size
    )
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")
    xp = _array_module(vectors)
    projected = xp.zeros((rank, rank), dtype=xp.result_type(*(
        vectors, core, t1, lov, lvv
    )))
    largest = 0
    for start in range(0, naux, block_size):
        stop = min(start + block_size, naux)
        ov = lov[start:stop]
        vv = lvv[start:stop]
        singles_transform = xp.einsum("Akc,ka->Aac", ov, t1)
        transformed = vv - singles_transform
        transformed_ladder = _projected_symmetric_ladder(
            doubles,
            t1,
            transformed,
            sector="virtual",
            auxiliary_block_size=int(stop - start),
        )
        correction_ladder = _projected_symmetric_ladder(
            doubles,
            t1,
            singles_transform,
            sector="virtual",
            auxiliary_block_size=int(stop - start),
        )
        projected += transformed_ladder.core - correction_ladder.core
        largest = max(
            largest,
            int(singles_transform.nbytes),
            int(transformed.nbytes),
            int(transformed_ladder.largest_intermediate_nbytes),
            int(correction_ladder.largest_intermediate_nbytes),
        )
    projected = (projected + projected.T) * 0.5
    return RRResidualTermResult(
        core=projected,
        term="complete-cc-Wvvvv-ladder",
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=largest,
    )


def _rr_t2_virtual_block(
    doubles: RRDoubles,
    first: slice,
    second: slice,
):
    u = doubles.projector.vectors.reshape(
        doubles.nocc, doubles.nvir, doubles.rank
    )
    xp = _array_module(u)
    return xp.einsum(
        "iaP,PQ,jbQ->ijab",
        u[:, first],
        doubles.core,
        u[:, second],
    )


def _cc_wvoov_block(
    doubles: RRDoubles,
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    virtual_slice: slice,
    auxiliary_block_size: int,
):
    """Return ``Wvoov[a_slice,k,i,c]`` without a dense ``t2``."""

    xp = _array_module(t1)
    nocc, nvir = doubles.nocc, doubles.nvir
    width = int(virtual_slice.stop - virtual_slice.start)
    dtype = xp.result_type(
        doubles.projector.vectors, doubles.core, t1, loo, lov, lvv
    )
    result = xp.zeros((width, nocc, nocc, nvir), dtype=dtype)
    all_virtual = slice(0, nvir)
    t2_da = _rr_t2_virtual_block(doubles, all_virtual, virtual_slice)
    t2_ad = _rr_t2_virtual_block(doubles, virtual_slice, all_virtual)
    t1a = t1[:, virtual_slice]
    largest = max(int(t2_da.nbytes), int(t2_ad.nbytes))
    naux = int(lov.shape[0])
    for start in range(0, naux, auxiliary_block_size):
        stop = min(start + auxiliary_block_size, naux)
        oo = loo[start:stop]
        ov = lov[start:stop]
        vv = lvv[start:stop, virtual_slice]
        ova = ov[:, :, virtual_slice]

        result += xp.einsum("Akc,Aad,id->akic", ov, vv, t1)
        result -= xp.einsum("Akc,Ali,la->akic", ov, oo, t1a)
        result += xp.einsum("Akc,Aia->akic", ov, ova)

        contracted_da = xp.einsum("Ald,ilda->Aia", ov, t2_da)
        result -= 0.5 * xp.einsum(
            "Akc,Aia->akic", ov, contracted_da
        )
        crossed_ad = xp.einsum("Akd,ilad->Akila", ov, t2_ad)
        result -= 0.5 * xp.einsum(
            "Alc,Akila->akic", ov, crossed_ad
        )

        singles_ld = xp.einsum("Ald,id->Ali", ov, t1)
        singles_pair = xp.einsum("Ali,la->Aia", singles_ld, t1a)
        result -= xp.einsum("Akc,Aia->akic", ov, singles_pair)
        contracted_ad = xp.einsum("Ald,ilad->Aia", ov, t2_ad)
        result += xp.einsum("Akc,Aia->akic", ov, contracted_ad)
        largest = max(
            largest,
            *(int(value.nbytes) for value in (
                contracted_da,
                crossed_ad,
                singles_ld,
                singles_pair,
                contracted_ad,
            )),
        )
    return result, largest


def _cc_wvovo_block(
    doubles: RRDoubles,
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    virtual_slice: slice,
    auxiliary_block_size: int,
):
    """Return ``Wvovo[a_slice,k,c,i]`` without a dense ``t2``."""

    xp = _array_module(t1)
    nocc, nvir = doubles.nocc, doubles.nvir
    width = int(virtual_slice.stop - virtual_slice.start)
    dtype = xp.result_type(
        doubles.projector.vectors, doubles.core, t1, loo, lov, lvv
    )
    result = xp.zeros((width, nocc, nvir, nocc), dtype=dtype)
    all_virtual = slice(0, nvir)
    t2_da = _rr_t2_virtual_block(doubles, all_virtual, virtual_slice)
    t1a = t1[:, virtual_slice]
    largest = int(t2_da.nbytes)
    naux = int(lov.shape[0])
    for start in range(0, naux, auxiliary_block_size):
        stop = min(start + auxiliary_block_size, naux)
        oo = loo[start:stop]
        ov = lov[start:stop]
        vv = lvv[start:stop, virtual_slice]

        singles_ki = xp.einsum("Akd,id->Aki", ov, t1)
        result += xp.einsum("Aki,Aac->akci", singles_ki, vv)
        singles_ac = xp.einsum("Alc,la->Aac", ov, t1a)
        result -= xp.einsum("Aac,Aki->akci", singles_ac, oo)
        result += xp.einsum("Aki,Aac->akci", oo, vv)

        crossed_da = xp.einsum("Akd,ilda->Akila", ov, t2_da)
        result -= 0.5 * xp.einsum(
            "Alc,Akila->akci", ov, crossed_da
        )
        result -= xp.einsum("Aac,Aki->akci", singles_ac, singles_ki)
        largest = max(
            largest,
            *(int(value.nbytes) for value in (
                singles_ki,
                singles_ac,
                crossed_da,
            )),
        )
    return result, largest


def projected_cc_ring(
    doubles: RRDoubles,
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    *,
    auxiliary_block_size: int = 1,
    virtual_block_size: int = 8,
) -> RRResidualTermResult:
    """Project the complete PySCF ``Wvoov/Wvovo`` doubles group.

    The implementation is an exact bounded-memory RR reference kernel.  It
    reconstructs only one ``(all_virtual, virtual_tile)`` doubles slice and
    one virtual tile of either four-index CC intermediate at a time.  This is
    the equation-level oracle for replacing the ring group with the fused THC
    Algorithm 2 contraction; it is not yet the final performance kernel.
    """

    vectors = doubles.projector.vectors
    core = doubles.core
    if not _same_backend(vectors, core, t1, loo, lov, lvv):
        raise TypeError("RR amplitudes, t1, and CD factors need one backend")
    _require_real_floating(vectors, core, t1, loo, lov, lvv)
    nocc, nvir, rank = doubles.nocc, doubles.nvir, doubles.rank
    if t1.shape != (nocc, nvir):
        raise ValueError("t1 shape does not match RR doubles")
    if getattr(lov, "ndim", None) != 3 or lov.shape[1:] != (nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    naux = int(lov.shape[0])
    if loo.shape != (naux, nocc, nocc):
        raise ValueError("loo must have shape (naux,nocc,nocc)")
    if lvv.shape != (naux, nvir, nvir):
        raise ValueError("lvv must have shape (naux,nvir,nvir)")
    ablock = int(auxiliary_block_size)
    vblock = int(virtual_block_size)
    if ablock < 1 or vblock < 1:
        raise ValueError("auxiliary and virtual block sizes must be positive")
    xp = _array_module(vectors)
    u = vectors.reshape(nocc, nvir, rank)
    half = xp.zeros((rank, rank), dtype=xp.result_type(
        vectors, core, t1, loo, lov, lvv
    ))
    largest = 0
    all_virtual = slice(0, nvir)

    # The first three half-residual contractions share Wvoov[a] and
    # Wvovo[a].
    for a_start in range(0, nvir, vblock):
        a_slice = slice(a_start, min(a_start + vblock, nvir))
        ua = u[:, a_slice]
        wvoov, wvoov_largest = _cc_wvoov_block(
            doubles, t1, loo, lov, lvv, a_slice, ablock
        )
        wvovo, wvovo_largest = _cc_wvovo_block(
            doubles, t1, loo, lov, lvv, a_slice, ablock
        )
        largest = max(
            largest,
            int(wvoov.nbytes),
            int(wvovo.nbytes),
            wvoov_largest,
            wvovo_largest,
        )
        for b_start in range(0, nvir, vblock):
            b_slice = slice(b_start, min(b_start + vblock, nvir))
            ub = u[:, b_slice]
            t2_cb = _rr_t2_virtual_block(
                doubles, all_virtual, b_slice
            )
            t2_bc = _rr_t2_virtual_block(
                doubles, b_slice, all_virtual
            )
            residual = 2.0 * xp.einsum(
                "akic,kjcb->ijab", wvoov, t2_cb
            )
            residual -= xp.einsum(
                "akci,kjcb->ijab", wvovo, t2_cb
            )
            residual -= xp.einsum(
                "akic,kjbc->ijab", wvoov, t2_bc
            )
            half += xp.einsum(
                "iaP,ijab,jbQ->PQ", ua, residual, ub
            )
            largest = max(
                largest,
                int(t2_cb.nbytes),
                int(t2_bc.nbytes),
                int(residual.nbytes),
            )

    # The fourth half-residual has the free virtual index on Wvovo[b].
    for b_start in range(0, nvir, vblock):
        b_slice = slice(b_start, min(b_start + vblock, nvir))
        ub = u[:, b_slice]
        wvovo_b, block_largest = _cc_wvovo_block(
            doubles, t1, loo, lov, lvv, b_slice, ablock
        )
        largest = max(largest, int(wvovo_b.nbytes), block_largest)
        for a_start in range(0, nvir, vblock):
            a_slice = slice(a_start, min(a_start + vblock, nvir))
            ua = u[:, a_slice]
            t2_ac = _rr_t2_virtual_block(
                doubles, a_slice, all_virtual
            )
            residual = -xp.einsum(
                "bkci,kjac->ijab", wvovo_b, t2_ac
            )
            half += xp.einsum(
                "iaP,ijab,jbQ->PQ", ua, residual, ub
            )
            largest = max(
                largest, int(t2_ac.nbytes), int(residual.nbytes)
            )

    projected = half + half.T
    return RRResidualTermResult(
        core=projected,
        term="complete-cc-Wvoov-Wvovo-ring-reference",
        auxiliary_block_size=ablock,
        largest_intermediate_nbytes=largest,
    )


def build_projected_ccsd_doubles_numerator(
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
) -> RRCCSDDoublesResult:
    """Build every real-RHF RCCSD doubles-numerator term in RR/CD form.

    This combines the exact factorized terms above in the same order and with
    the same coefficients as PySCF ``rccsd.update_amps``.  The ring group is
    currently the bounded-memory equation-reference implementation; replacing
    it with fused RR/THC contractions is a performance step that does not
    change this public numerical contract.
    """

    vectors = doubles.projector.vectors
    values = (
        doubles.core,
        t1,
        fock_oo,
        fock_ov,
        fock_vv,
        occupied_energies,
        virtual_energies,
        loo,
        lov,
        lvv,
    )
    if not _same_backend(vectors, *values):
        raise TypeError("all RR/CD doubles inputs need one array backend")
    _require_real_floating(vectors, *values)
    nocc, nvir = doubles.nocc, doubles.nvir
    if occupied_energies.shape != (nocc,):
        raise ValueError("occupied_energies must have shape (nocc,)")
    if virtual_energies.shape != (nvir,):
        raise ValueError("virtual_energies must have shape (nvir,)")
    ablock = int(auxiliary_block_size)
    vblock = int(virtual_block_size)
    if ablock < 1 or vblock < 1:
        raise ValueError("auxiliary and virtual block sizes must be positive")
    xp = _array_module(vectors)

    f_intermediates = build_cc_fock_intermediates(
        doubles,
        t1,
        lov,
        fock_oo,
        fock_ov,
        fock_vv,
        auxiliary_block_size=ablock,
    )
    lagrangian = build_cc_lagrangian_one_body(
        f_intermediates, t1, fock_ov, loo, lov, lvv
    )
    occupied_dressing = lagrangian.loo.copy()
    virtual_dressing = lagrangian.lvv.copy()
    occupied_indices = xp.arange(nocc)
    virtual_indices = xp.arange(nvir)
    occupied_dressing[occupied_indices, occupied_indices] -= (
        occupied_energies
    )
    virtual_dressing[virtual_indices, virtual_indices] -= (
        virtual_energies + float(level_shift)
    )

    terms = (
        projected_linear_t1_doubles(
            doubles.projector,
            t1,
            loo,
            lov,
            lvv,
            auxiliary_block_size=ablock,
        ),
        projected_bare_ovov(doubles.projector, lov, nocc, nvir),
        projected_cc_woooo(
            doubles,
            t1,
            loo,
            lov,
            auxiliary_block_size=ablock,
            virtual_block_size=vblock,
            transfer_counter=transfer_counter,
        ),
        projected_cc_wvvvv(
            doubles,
            t1,
            lov,
            lvv,
            auxiliary_block_size=ablock,
        ),
        projected_one_body_dressing(
            doubles, occupied_dressing, virtual_dressing
        ),
        projected_cc_ring(
            doubles,
            t1,
            loo,
            lov,
            lvv,
            auxiliary_block_size=ablock,
            virtual_block_size=vblock,
        ),
    )
    core = xp.zeros_like(terms[0].core)
    for term in terms:
        core += term.core
    return RRCCSDDoublesResult(
        core=core,
        fock_intermediates=f_intermediates,
        lagrangian_one_body=lagrangian,
        term_metadata=tuple(term.metadata() for term in terms),
    )


__all__ = [
    "RRResidualTermResult",
    "RRSingleTermResult",
    "CCFockIntermediates",
    "RRCCSDEnergyResult",
    "RRCCSDSinglesResult",
    "RRCCSDDoublesResult",
    "CCLagrangianOneBody",
    "ProjectedPairDenominator",
    "build_cc_fock_intermediates",
    "rr_ccsd_energy",
    "build_ccsd_singles_numerator",
    "build_projected_ccsd_doubles_numerator",
    "build_cc_lagrangian_one_body",
    "projected_bare_ovov",
    "projected_linear_t1_doubles",
    "projected_one_body_dressing",
    "projected_coulomb_ppl",
    "projected_vvvv_ladder",
    "projected_oooo_ladder",
    "projected_cc_woooo",
    "projected_cc_wvvvv",
    "projected_cc_ring",
    "singles_from_fov_doubles",
    "singles_from_ovoo_doubles",
    "singles_from_cd_ovvv",
    "singles_from_cd_ovoo",
    "singles_from_cd_ovvo_oovv",
]
