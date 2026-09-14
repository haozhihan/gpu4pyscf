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

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Mapping, Optional

import numpy as np

from gpu4pyscf.cc.device_runtime import nvtx_range
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


RRDoublesProfilePhase = Callable[
    [str, Mapping[str, Any]], ContextManager[Any]
]


@contextlib.contextmanager
def _rr_doubles_profile_phase(
    name: str,
    metadata: Mapping[str, Any],
    profile_phase: Optional[RRDoublesProfilePhase],
):
    """Wrap one coarse RR doubles operation in NVTX and optional timing.

    The callback deliberately owns timing policy.  The residual builder only
    declares stable phase boundaries and metadata, so a CPU caller can use a
    wall clock while the resident GPU engine records deferred CUDA events.
    """

    with nvtx_range(name):
        if profile_phase is None:
            yield
        else:
            with profile_phase(name, metadata):
                yield


@dataclass(frozen=True)
class RRResidualTermResult:
    """One projected residual contribution plus shape/allocation diagnostics.

    ``largest_intermediate_nbytes`` describes one logical array observed by
    the kernel.  It is not a peak HBM or simultaneously-live allocation
    measurement; the latter requires a device profiler.
    """

    core: Any
    term: str
    auxiliary_block_size: int
    largest_intermediate_nbytes: int
    factor_symmetry_error: Optional[float] = None
    materialized_dense_t2: bool = False
    kernel_name: str = "rr-factorized-reference"
    tile_residual_nbytes: int = 0
    workspace_shape_upper_bound: Optional[dict[str, Any]] = None

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
            "kernel_name": self.kernel_name,
            "tile_residual_nbytes": int(self.tile_residual_nbytes),
            "largest_intermediate_scope": "single-logical-array",
            "workspace_shape_upper_bound": (
                None
                if self.workspace_shape_upper_bound is None
                else dict(self.workspace_shape_upper_bound)
            ),
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
class RRCCSDDoublesComponents:
    """Unassembled projected doubles terms and their shared intermediates.

    The complete RR driver normally assembles these terms from a zero core.
    A hybrid residual can instead start from ``-R_123`` and add every term
    directly.  That produces the complement without first materializing an
    independently assembled complete-RR numerator.
    """

    terms: tuple[RRResidualTermResult, ...]
    fock_intermediates: CCFockIntermediates
    lagrangian_one_body: "CCLagrangianOneBody"
    materialized_dense_t2: bool = False
    materialized_four_index_eri: bool = False

    @property
    def term_metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(term.metadata() for term in self.terms)

    def assemble_core(self, *, initial_core: Any = None):
        """Accumulate the terms, optionally from a caller-supplied offset."""

        if not self.terms:
            raise ValueError("at least one doubles residual term is required")
        reference = self.terms[0].core
        xp = _array_module(reference)
        if initial_core is None:
            core = xp.zeros_like(reference)
        else:
            if not _same_backend(reference, initial_core):
                raise TypeError("doubles terms and initial core need one backend")
            if getattr(initial_core, "shape", None) != reference.shape:
                raise ValueError("initial core shape does not match doubles terms")
            if not np.issubdtype(np.dtype(initial_core.dtype), np.floating):
                raise TypeError("initial core must be floating point")
            core = xp.array(initial_core, copy=True)
        for term in self.terms:
            core += term.core
        return core

    def metadata(self) -> dict[str, Any]:
        return {
            "equation": "Hirata-2004-restricted-CCSD-doubles-projected",
            "fock_intermediates": self.fock_intermediates.metadata(),
            "lagrangian_one_body": self.lagrangian_one_body.metadata(),
            "terms": list(self.term_metadata),
            "assembled_complete_core": False,
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


def estimate_projected_cc_wvvvv_workspace_nbytes(
    nocc: int,
    nvir: int,
    rank: int,
    naux: int,
    *,
    auxiliary_block_size: int,
    itemsize: int = 8,
) -> dict[str, Any]:
    """Return the logical workspace policy for fused ``Wvvvv`` ladders.

    A fused bilinear ladder has two transformed matrices live together.  Each
    requested auxiliary block is therefore processed in sub-blocks no wider
    than half of that *actual* outer block.  Their two rank-square transforms
    fit within the old path's one-transform allowance for the outer block,
    including a short tail.  A singleton outer block uses the sequential
    two-ladder path because no positive fused sub-block can meet that bound.

    The aggregate ledger covers Python-visible logical arrays owned by the
    kernel.  Backend library workspaces and caller-owned input tensors are
    outside its scope.
    """

    nocc = int(nocc)
    nvir = int(nvir)
    rank = int(rank)
    naux = int(naux)
    block_size = int(auxiliary_block_size)
    itemsize = int(itemsize)
    if nocc < 1 or nvir < 1 or rank < 1 or itemsize < 1:
        raise ValueError("Wvvvv workspace dimensions must be positive")
    if naux < 0:
        raise ValueError("Wvvvv naux must be non-negative")
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")

    outer_widths = [
        min(block_size, naux - start)
        for start in range(0, naux, block_size)
    ]
    maximum_outer_width = max(outer_widths, default=0)
    fused_inner_widths = [
        width // 2 for width in outer_widths if width >= 2
    ]
    maximum_fused_inner_width = max(fused_inner_widths, default=0)
    singleton_count = sum(width == 1 for width in outer_widths)
    fused_subblock_count = sum(
        (width + width // 2 - 1) // (width // 2)
        for width in outer_widths
        if width >= 2
    )

    rank2 = rank * rank * itemsize
    projector = nocc * nvir * rank * itemsize
    fused_width = maximum_fused_inner_width
    fused_temporaries = {
        "right_factor_nbytes": int(fused_width * nvir * nvir * itemsize),
        "two_live_rank_square_transforms_nbytes": int(
            2 * fused_width * rank2
        ),
        "two_live_singles_transforms_nbytes": int(
            2 * fused_width * rank * itemsize
        ),
        "projected_contraction_return_nbytes": int(
            rank2 if fused_width else 0
        ),
    }
    sequential_temporaries = {
        "correction_factor_nbytes": int(
            nvir * nvir * itemsize if singleton_count else 0
        ),
        "transformed_factor_nbytes": int(
            nvir * nvir * itemsize if singleton_count else 0
        ),
        "one_live_rank_square_transform_nbytes": int(
            rank2 if singleton_count else 0
        ),
        "one_live_singles_transform_nbytes": int(
            rank * itemsize if singleton_count else 0
        ),
        "ladder_core_nbytes": int(rank2 if singleton_count else 0),
        "projected_contraction_return_nbytes": int(
            rank2 if singleton_count else 0
        ),
    }
    final_symmetrization_temporaries = {
        "projected_core_return_nbytes": int(rank2),
    }
    persistent = {
        "projected_core_nbytes": int(rank2),
        "symmetric_core_nbytes": int(
            rank2 if maximum_fused_inner_width else 0
        ),
    }
    fused_temporary_sum = sum(fused_temporaries.values())
    sequential_temporary_sum = sum(sequential_temporaries.values())
    final_symmetrization_temporary_sum = sum(
        final_symmetrization_temporaries.values()
    )
    temporary_sum = max(
        fused_temporary_sum,
        sequential_temporary_sum,
        final_symmetrization_temporary_sum,
    )
    persistent_sum = sum(persistent.values())
    prior_peak = maximum_outer_width * rank2
    two_live_peak = 2 * maximum_fused_inner_width * rank2
    sequential_peak = rank2 if singleton_count else 0
    maximum_transform_peak = max(two_live_peak, sequential_peak)
    fused_individual_arrays = [
        fused_temporaries["right_factor_nbytes"],
        fused_width * rank2,
        fused_width * rank * itemsize,
        fused_temporaries["projected_contraction_return_nbytes"],
    ]
    sequential_individual_arrays = [
        *sequential_temporaries.values(),
    ]
    all_workspace_arrays = [
        *persistent.values(),
        *fused_individual_arrays,
        *sequential_individual_arrays,
        *final_symmetrization_temporaries.values(),
    ]
    return {
        "scope": (
            "kernel-owned-logical-arrays-excludes-inputs-and-backend-workspace"
        ),
        "requested_auxiliary_block_size": int(block_size),
        "maximum_actual_outer_block_size": int(maximum_outer_width),
        "maximum_fused_auxiliary_subblock_size": int(
            maximum_fused_inner_width
        ),
        "fused_auxiliary_subblock_count": int(fused_subblock_count),
        "singleton_sequential_block_count": int(singleton_count),
        "singleton_policy": "sequential-two-ladder",
        "rank_square_transform_nbytes_per_auxiliary": int(rank2),
        "prior_one_transform_outer_block_nbytes": int(prior_peak),
        "maximum_simultaneously_live_rank_square_transform_nbytes": int(
            maximum_transform_peak
        ),
        "rank_square_transform_contract_satisfied": bool(
            maximum_transform_peak <= prior_peak
        ),
        "per_outer_block_contract_satisfied": bool(all(
            (
                width == 1
                or 2 * (width // 2) <= width
            )
            for width in outer_widths
        )),
        "projector_view_nbytes_not_in_workspace_sum": int(projector),
        "persistent": persistent,
        "persistent_sum_nbytes": int(persistent_sum),
        "temporaries": {
            "fused_bilinear": fused_temporaries,
            "singleton_sequential": sequential_temporaries,
            "final_symmetrization": final_symmetrization_temporaries,
        },
        "fused_temporary_sum_nbytes": int(fused_temporary_sum),
        "sequential_temporary_sum_nbytes": int(
            sequential_temporary_sum
        ),
        "final_symmetrization_temporary_sum_nbytes": int(
            final_symmetrization_temporary_sum
        ),
        "temporary_sum_nbytes": int(temporary_sum),
        "largest_logical_workspace_array_nbytes": int(
            max(all_workspace_arrays, default=0)
        ),
        "simultaneously_live_shape_upper_bound_nbytes": int(
            persistent_sum + temporary_sum
        ),
    }


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

    The difference of the two symmetric ladders is evaluated as one bilinear
    ladder.  If ``X`` and ``Y`` denote the RR transforms of ``Lvv`` and ``M``,
    respectively, then

    ``sym(ladder(X-Y) - ladder(Y))``
    ``    = sym(X @ sym(core) @ (X-2Y).T)``.

    The same identity applies to the singles outer product.  Each requested
    Cholesky block is internally capped at half its actual width, so the two
    live rank-square transforms do not exceed the prior path's one-transform
    block allowance.  A singleton block falls back to the sequential identity.
    This avoids the dense ``vvvv`` tensor and removes one core-weighted ladder
    contraction for every fused sub-block.
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
    dtype = xp.result_type(*(
        vectors, core, t1, lov, lvv
    ))
    projected = xp.zeros((rank, rank), dtype=dtype)
    workspace = estimate_projected_cc_wvvvv_workspace_nbytes(
        nocc,
        nvir,
        rank,
        naux,
        auxiliary_block_size=block_size,
        itemsize=int(dtype.itemsize),
    )
    has_fused_blocks = bool(
        workspace["maximum_fused_auxiliary_subblock_size"]
    )
    if has_fused_blocks:
        symmetric_core = core + core.T
        symmetric_core *= 0.5
    else:
        symmetric_core = None
    u = vectors.reshape(nocc, nvir, rank)
    largest = 0 if symmetric_core is None else int(symmetric_core.nbytes)
    for outer_start in range(0, naux, block_size):
        outer_stop = min(outer_start + block_size, naux)
        outer_width = outer_stop - outer_start
        if outer_width == 1:
            ov = lov[outer_start:outer_stop]
            vv = lvv[outer_start:outer_stop]
            correction = xp.einsum("Akc,ka->Aac", ov, t1)
            transformed = vv - correction
            transformed_ladder = _projected_symmetric_ladder(
                doubles,
                t1,
                transformed,
                sector="virtual",
                auxiliary_block_size=1,
            )
            projected += transformed_ladder.core
            largest = max(
                largest,
                int(correction.nbytes),
                int(transformed.nbytes),
                int(transformed_ladder.core.nbytes),
                int(transformed_ladder.largest_intermediate_nbytes),
            )
            del transformed, transformed_ladder
            correction_ladder = _projected_symmetric_ladder(
                doubles,
                t1,
                correction,
                sector="virtual",
                auxiliary_block_size=1,
            )
            projected -= correction_ladder.core
            largest = max(
                largest,
                int(correction_ladder.core.nbytes),
                int(correction_ladder.largest_intermediate_nbytes),
            )
            del correction, correction_ladder
            continue

        inner_block_size = outer_width // 2
        for start in range(outer_start, outer_stop, inner_block_size):
            stop = min(start + inner_block_size, outer_stop)
            ov = lov[start:stop]
            vv = lvv[start:stop]
            right_factors = xp.einsum("Akc,ka->Aac", ov, t1)
            right_factors *= -2.0
            right_factors += vv

            left_matrix = xp.einsum("iaP,Aac,icR->APR", u, vv, u)
            left_singles = xp.einsum("iaP,Aac,ic->AP", u, vv, t1)
            weighted_left = xp.einsum(
                "APR,RS->APS", left_matrix, symmetric_core
            )
            largest = max(
                largest,
                int(right_factors.nbytes),
                int(left_matrix.nbytes),
                int(left_singles.nbytes),
                int(weighted_left.nbytes),
            )
            # The weighted left transform is all that the remaining
            # contraction needs.  Release the unweighted matrix before
            # allocating its right counterpart.
            del left_matrix
            right_matrix = xp.einsum(
                "iaP,Aac,icR->APR", u, right_factors, u
            )
            right_singles = xp.einsum(
                "iaP,Aac,ic->AP", u, right_factors, t1
            )
            projected += xp.einsum(
                "APS,AQS->PQ", weighted_left, right_matrix
            )
            projected += xp.einsum(
                "AP,AQ->PQ", left_singles, right_singles
            )
            largest = max(
                largest,
                int(right_matrix.nbytes),
                int(right_singles.nbytes),
            )
            del (
                left_singles,
                right_factors,
                right_matrix,
                right_singles,
                weighted_left,
            )
    projected = projected + projected.T
    projected *= 0.5
    return RRResidualTermResult(
        core=projected,
        term="complete-cc-Wvvvv-ladder",
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=largest,
        workspace_shape_upper_bound=workspace,
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


def _select_projected_ring_matrix_chain(
    nocc: int,
    nvir: int,
    wa: int,
    wb: int,
    rank: int,
    itemsize: int,
) -> dict[str, Any]:
    """Choose the ordinary crossed-ring matrix chain from actual tile sizes.

    With ``m=nocc*wa``, ``k=nocc*nvir``, ``n=nocc*wb`` and ``r=rank``, the
    contraction is ``L.T @ W @ T @ R``.  The residual-first candidate keeps
    the historical association ``((L.T @ (W @ T)) @ R)``.  The alternative
    projects the right side first as ``L.T @ (W @ (T @ R))``.

    A right-first chain is admitted only when it has strictly fewer scalar
    multiplies and does not exceed the residual-first logical temporary peak.
    The latter is the local resource contract: a selector optimization may
    not consume more HBM workspace than the already admitted fallback.
    """

    dimensions = tuple(int(value) for value in (
        nocc,
        nvir,
        wa,
        wb,
        rank,
        itemsize,
    ))
    if any(value < 1 for value in dimensions):
        raise ValueError("ring matrix-chain dimensions must be positive")
    nocc, nvir, wa, wb, rank, itemsize = dimensions
    m = nocc * wa
    k = nocc * nvir
    n = nocc * wb
    r = rank

    residual_multiply_count = (
        m * k * n
        + r * m * n
        + r * n * r
    )
    right_multiply_count = (
        k * n * r
        + m * k * r
        + r * m * r
    )
    # Returned r x r output and input transpose/reshape copies are common to
    # both candidates.  The residual path may hold the GEMM output and its
    # reordered copy together; the right-first path may hold projected_t and
    # both layouts of projected_w during the pair-ordering copy.
    residual_temporary_nbytes = (2 * m * n + r * n) * itemsize
    right_temporary_nbytes = (k * r + 2 * m * r) * itemsize
    right_compute_better = right_multiply_count < residual_multiply_count
    right_within_resource_contract = (
        right_temporary_nbytes <= residual_temporary_nbytes
    )
    use_right_first = (
        right_compute_better and right_within_resource_contract
    )
    selected_order = (
        "right-project-first" if use_right_first else "residual-first"
    )
    if use_right_first:
        selection_reason = (
            "right-project-first-has-fewer-multiplies-and-does-not-"
            "increase-logical-peak-temporary"
        )
    elif not right_compute_better:
        selection_reason = "residual-first-has-fewer-or-equal-multiplies"
    else:
        selection_reason = (
            "residual-first-enforces-logical-peak-temporary-contract"
        )

    return {
        "schema": "gpu4pyscf.rr-ring-matrix-chain.v1",
        "dimensions": {"m": m, "k": k, "n": n, "r": r},
        "selected_order": selected_order,
        "selection_reason": selection_reason,
        "resource_contract": {
            "policy": "right-first-must-not-exceed-residual-first-peak",
            "temporary_budget_nbytes": int(residual_temporary_nbytes),
            "right_project_first_within_budget": bool(
                right_within_resource_contract
            ),
        },
        "candidates": {
            "residual_first": {
                "order": [
                    "wmat@tmat",
                    "left_u.T@residual",
                    "projected_left@right_u",
                ],
                "scalar_multiply_count": int(residual_multiply_count),
                "estimated_gemm_flops": int(2 * residual_multiply_count),
                "logical_peak_temporary_nbytes": int(
                    residual_temporary_nbytes
                ),
                "materialized_ijab": True,
            },
            "right_project_first": {
                "order": [
                    "tmat@right_u",
                    "wmat@projected_t",
                    "left_u.T@projected_w",
                ],
                "scalar_multiply_count": int(right_multiply_count),
                "estimated_gemm_flops": int(2 * right_multiply_count),
                "logical_peak_temporary_nbytes": int(
                    right_temporary_nbytes
                ),
                "materialized_ijab": False,
            },
        },
    }


def _projected_ring_candidate_temporaries(
    nocc: int,
    nvir: int,
    wa: int,
    wb: int,
    rank: int,
    itemsize: int,
    selected_order: str,
) -> dict[str, int]:
    """Candidate-specific logical arrays for one ordinary ring tile.

    Input layout copies, projector views, and the common returned core are
    accounted once in the estimator's separate common-workspace ledger.
    """

    nocc, nvir, wa, wb, rank, itemsize = (
        int(value)
        for value in (nocc, nvir, wa, wb, rank, itemsize)
    )
    if any(
        value < 1
        for value in (nocc, nvir, wa, wb, rank, itemsize)
    ):
        raise ValueError("ring temporary dimensions must be positive")
    if selected_order not in {"residual-first", "right-project-first"}:
        raise ValueError("unknown ordinary ring matrix-chain order")

    m = nocc * wa
    k = nocc * nvir
    n = nocc * wb
    r = rank
    temporaries: dict[str, int] = {}
    if selected_order == "residual-first":
        residual_nbytes = m * n * itemsize
        temporaries.update({
            "ijab_residual_nbytes": int(residual_nbytes),
            "ijab_transpose_reshape_copy_nbytes": int(residual_nbytes),
            "projected_left_nbytes": int(r * n * itemsize),
        })
    else:
        projected_w_nbytes = m * r * itemsize
        temporaries.update({
            "ijab_residual_nbytes": 0,
            "ijab_transpose_reshape_copy_nbytes": 0,
            "projected_t_nbytes": int(k * r * itemsize),
            "projected_w_nbytes": int(projected_w_nbytes),
            "projected_w_pair_order_copy_nbytes": int(
                projected_w_nbytes
            ),
        })
    return temporaries


def estimate_rr_ring_workspace_nbytes(
    nocc: int,
    nvir: int,
    *,
    auxiliary_block_size: int = 1,
    virtual_block_size: int = 32,
    itemsize: int = 8,
    rank: Optional[int] = None,
) -> dict[str, Any]:
    """Return conservative logical shape upper bounds for the ring path.

    The largest auxiliary-dependent crossed intermediate in the Wvoov/Wvovo
    builders has shape ``(A,k,i,l,a)``.  The estimate deliberately uses the
    requested auxiliary block, never the full ``naux``.  The ordinary crossed
    GEMM chain is selected from residual-first and right-project-first for
    every actual full/tail ``(wa,wb)`` shape.  Its peak ledger comes from the
    largest selected candidate rather than assuming that the maximum square
    tile selects the same association as every tail.  The fourth Wvovo
    contraction always retains one bounded ``ijab`` tile and is modeled
    separately.
    Transpose/reshape copies receive conservative copy-sized allowances.
    These are logical shape bounds, not runtime allocation or HBM claims.
    """

    nocc = int(nocc)
    nvir = int(nvir)
    ablock = int(auxiliary_block_size)
    vblock = int(virtual_block_size)
    itemsize = int(itemsize)
    rank = nocc * nvir if rank is None else int(rank)
    if nocc < 1 or nvir < 1 or ablock < 1 or vblock < 1 or itemsize < 1:
        raise ValueError("ring workspace dimensions must be positive")
    if rank < 1:
        raise ValueError("ring rank must be positive")
    vtile = min(nvir, vblock)
    crossed = ablock * nocc * nocc * nocc * vtile * itemsize
    w_tile = vtile * nocc * nocc * nvir * itemsize
    t2_tile = nocc * nocc * nvir * vtile * itemsize
    ijab_tile = nocc * nocc * vtile * vtile * itemsize
    projector_tile = nocc * vtile * rank * itemsize
    projected_w_tile = vtile * nocc * rank * itemsize
    projected_t = nocc * nvir * rank * itemsize
    rank2 = rank * rank * itemsize
    chain_rank2 = 3 * rank2
    chain_rank_rect = rank * nocc * vtile * itemsize
    ordinary_chain_selection = _select_projected_ring_matrix_chain(
        nocc, nvir, vtile, vtile, rank, itemsize
    )
    tail_width = nvir % vblock
    tile_width_set = {vtile}
    if tail_width:
        tile_width_set.add(tail_width)
    tile_widths = sorted(tile_width_set)
    ordinary_tile_selections = [
        {
            "left_virtual_width": int(left_width),
            "right_virtual_width": int(right_width),
            **_select_projected_ring_matrix_chain(
                nocc,
                nvir,
                left_width,
                right_width,
                rank,
                itemsize,
            ),
        }
        for left_width in tile_widths
        for right_width in tile_widths
    ]
    for selection in ordinary_tile_selections:
        selected_temporaries = _projected_ring_candidate_temporaries(
            nocc,
            nvir,
            int(selection["left_virtual_width"]),
            int(selection["right_virtual_width"]),
            rank,
            itemsize,
            str(selection["selected_order"]),
        )
        selection["selected_candidate_temporaries"] = selected_temporaries
        selection["selected_candidate_temporary_sum_nbytes"] = int(
            sum(selected_temporaries.values())
        )
    ordinary_peak_tile = max(
        ordinary_tile_selections,
        key=lambda selection: (
            int(selection["selected_candidate_temporary_sum_nbytes"]),
            int(
                selection["candidates"][
                    str(selection["selected_order"]).replace("-", "_")
                ]["logical_peak_temporary_nbytes"]
            ),
            int(selection["left_virtual_width"]),
            int(selection["right_virtual_width"]),
        ),
    )
    ordinary_selected_orders = sorted({
        str(selection["selected_order"])
        for selection in ordinary_tile_selections
    })
    ordinary_residual_tiles = [
        selection
        for selection in ordinary_tile_selections
        if selection["selected_order"] == "residual-first"
    ]
    ordinary_ijab_tile = max(
        (
            nocc
            * nocc
            * int(selection["left_virtual_width"])
            * int(selection["right_virtual_width"])
            * itemsize
            for selection in ordinary_residual_tiles
        ),
        default=0,
    )

    persistent = {
        "core_nbytes": int(rank2),
        "half_nbytes": int(rank2),
        "pair_metric_nbytes": int(rank2),
        "flat_u_nbytes": int(nocc * nvir * rank * itemsize),
    }
    persistent_sum = sum(persistent.values())

    # These arrays can overlap with the inputs returned by the W builders.
    # The counts deliberately include both W tiles and any transpose/reshape
    # copies, even when a particular backend happens to avoid a copy.
    w_builder_temporaries = {
        "crossed_intermediate_nbytes": int(crossed),
        "wvoov_nbytes": int(w_tile),
        "wvovo_nbytes": int(w_tile),
        "t2_da_builder_tile_nbytes": int(t2_tile),
        "t2_ad_builder_tile_nbytes": int(t2_tile),
    }
    direct_temporaries = {
        "wvoov_transpose_copy_nbytes": int(w_tile),
        "wvovo_transpose_copy_nbytes": int(w_tile),
        "projected_wvoov_nbytes": int(projected_w_tile),
        "projected_wvovo_nbytes": int(projected_w_tile),
        "left_projector_copy_nbytes": int(projector_tile),
        "left_wvoov_nbytes": int(rank2),
        "left_wvovo_nbytes": int(rank2),
        "direct_chain_matmul_temporary_nbytes": int(chain_rank2),
    }
    crossed_common_temporaries = {
        "wmat_transpose_copy_nbytes": int(w_tile),
        "tmat_transpose_copy_nbytes": int(t2_tile),
        "left_projector_copy_nbytes": int(projector_tile),
        "right_projector_copy_nbytes": int(projector_tile),
        "projected_return_nbytes": int(rank2),
    }
    crossed_residual_first_temporaries = (
        _projected_ring_candidate_temporaries(
            nocc,
            nvir,
            vtile,
            vtile,
            rank,
            itemsize,
            "residual-first",
        )
    )
    crossed_right_project_first_temporaries = (
        _projected_ring_candidate_temporaries(
            nocc,
            nvir,
            vtile,
            vtile,
            rank,
            itemsize,
            "right-project-first",
        )
    )
    selected_candidate_key = (
        "right_project_first"
        if ordinary_chain_selection["selected_order"]
        == "right-project-first"
        else "residual_first"
    )
    crossed_temporaries = dict(
        ordinary_peak_tile["selected_candidate_temporaries"]
    )
    crossed_wvovo_temporaries = {
        "wmat_transpose_copy_nbytes": int(w_tile),
        "tmat_transpose_copy_nbytes": int(t2_tile),
        "ijab_residual_nbytes": int(ijab_tile),
        "ijab_transpose_reshape_copy_nbytes": int(ijab_tile),
        "left_projector_copy_nbytes": int(projector_tile),
        "right_projector_copy_nbytes": int(projector_tile),
        "projected_residual_nbytes": int(chain_rank_rect),
        "projected_return_nbytes": int(rank2),
    }
    w_builder_sum = sum(w_builder_temporaries.values())
    direct_sum = sum(direct_temporaries.values())
    crossed_chain_sum = sum(crossed_temporaries.values())
    crossed_common_sum = sum(crossed_common_temporaries.values())
    crossed_sum = crossed_common_sum + crossed_chain_sum
    crossed_wvovo_sum = sum(crossed_wvovo_temporaries.values())
    crossed_peak_sum = max(crossed_sum, crossed_wvovo_sum)
    temporary_sum = w_builder_sum + direct_sum + crossed_peak_sum
    all_arrays = [
        *persistent.values(),
        *w_builder_temporaries.values(),
        *direct_temporaries.values(),
        *crossed_common_temporaries.values(),
        *crossed_temporaries.values(),
        *crossed_wvovo_temporaries.values(),
    ]
    return {
        # Flat aliases keep the key useful for lightweight planning clients.
        "crossed_intermediate_nbytes": int(crossed),
        "w_tile_nbytes": int(w_tile),
        "t2_tile_nbytes": int(t2_tile),
        "ijab_tile_nbytes": int(ijab_tile),
        "ordinary_crossed_ijab_tile_nbytes": int(ordinary_ijab_tile),
        "ordinary_crossed_full_tile_ijab_tile_nbytes": int(
            0
            if selected_candidate_key == "right_project_first"
            else ijab_tile
        ),
        "wvovo_crossed_ijab_tile_nbytes": int(ijab_tile),
        "projector_tile_nbytes": int(projector_tile),
        "projected_w_tile_nbytes": int(projected_w_tile),
        "projected_t_nbytes": int(projected_t),
        "pair_metric_nbytes": int(rank2),
        "ordinary_crossed_materialized_ijab": bool(
            ordinary_residual_tiles
        ),
        "ordinary_crossed_full_tile_materialized_ijab": bool(
            selected_candidate_key == "residual_first"
        ),
        "wvovo_crossed_materialized_ijab": True,
        "ordinary_crossed_contraction_order": list(
            ordinary_chain_selection["candidates"][
                selected_candidate_key
            ]["order"]
        ),
        "ordinary_crossed_contraction_order_scope": (
            "maximum-square-virtual-tile"
        ),
        "ordinary_crossed_selected_orders": ordinary_selected_orders,
        "ordinary_crossed_chain_selection": ordinary_chain_selection,
        "ordinary_crossed_chain_selections_by_tile_shape": (
            ordinary_tile_selections
        ),
        "ordinary_crossed_peak_tile": dict(ordinary_peak_tile),
        "ordinary_crossed_peak_candidate": str(
            ordinary_peak_tile["selected_order"]
        ),
        "ordinary_crossed_peak_chain_temporary_nbytes": int(
            crossed_chain_sum
        ),
        "ordinary_crossed_common_temporary_nbytes": int(
            crossed_common_sum
        ),
        "ordinary_crossed_peak_sum_nbytes": int(crossed_sum),
        "ordinary_crossed_candidate_temporaries": {
            "residual_first": crossed_residual_first_temporaries,
            "right_project_first": (
                crossed_right_project_first_temporaries
            ),
        },
        "ordinary_crossed_candidate_temporaries_scope": (
            "maximum-square-virtual-tile-candidate-reference"
        ),
        "ordinary_crossed_peak_tile_scope": (
            "maximum-over-all-full-and-tail-runtime-tile-shapes"
        ),
        "largest_logical_array_nbytes": int(max(all_arrays)),
        "persistent": persistent,
        "persistent_sum_nbytes": int(persistent_sum),
        "temporaries": {
            "w_builder": w_builder_temporaries,
            "direct_gemm": direct_temporaries,
            "crossed_gemm_common": crossed_common_temporaries,
            "crossed_gemm": crossed_temporaries,
            "crossed_wvovo_gemm": crossed_wvovo_temporaries,
        },
        "crossed_gemm_peak_sum_nbytes": int(crossed_peak_sum),
        "temporary_sum_nbytes": int(temporary_sum),
        "simultaneously_live_shape_upper_bound_nbytes": int(
            persistent_sum + temporary_sum
        ),
    }


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
        kernel_name="rr-ring-reference",
    )


def _projected_ring_gemm_tile(
    left_intermediate: Any,
    right_t2: Any,
    left_u: Any,
    right_u: Any,
    *,
    right_is_wvovo: bool = False,
):
    """Project one ring contraction through an explicitly ordered GEMM chain.

    ``right_t2`` is a virtual tile only.  The ordinary crossed-ring branch
    selects between the historical bounded ``(i,a,j,b)`` residual and the
    ``T @ U_right`` direct projection using its actual matrix dimensions.
    ``right_is_wvovo`` selects the fourth ring term, whose free virtual index
    lives on the intermediate; that branch retains its original bounded
    residual formula and projects it with an explicit right association.

    The two integer diagnostics are, respectively, the materialized residual
    tile size and the largest logical intermediate observed by this helper;
    the final mapping records the selected association and both candidates.
    """

    no, wa, rank = left_u.shape
    left_projector = left_u.reshape(no * wa, rank)
    if right_is_wvovo:
        # W[b,k,c,i] T[k,j,a,c] -> R[i,a,j,b]
        wb = left_intermediate.shape[0]
        wmat = left_intermediate.transpose(0, 3, 1, 2).reshape(
            wb * no, -1
        )
        tmat = right_t2.transpose(0, 3, 1, 2).reshape(
            -1, no * wa
        )
        residual = (wmat @ tmat).reshape(wb, no, no, wa)
        residual = residual.transpose(1, 3, 2, 0)
        residual_matrix = residual.reshape(no * wa, -1)
        right_projector = right_u.reshape(-1, rank)
        projected_residual = residual_matrix @ right_projector
        projected = left_projector.T @ projected_residual
        largest = max(
            int(wmat.nbytes),
            int(tmat.nbytes),
            int(residual.nbytes),
            int(projected_residual.nbytes),
            int(projected.nbytes),
        )
        return projected, int(residual.nbytes), largest, {
            "schema": "gpu4pyscf.rr-ring-fixed-wvovo-chain.v1",
            "selected_order": "residual-first-right-project",
            "materialized_ijab": True,
        }

    # W[a,k,i,c] T[k,j,b,c] -> R[i,a,j,b], projected without R.
    wb = right_u.shape[1]
    wmat = left_intermediate.transpose(0, 2, 1, 3).reshape(
        wa * no, -1
    )
    tmat = right_t2.transpose(0, 3, 1, 2).reshape(
        -1, no * wb
    )
    right_projector = right_u.reshape(no * wb, rank)
    nvir = int(left_intermediate.shape[3])
    selection = _select_projected_ring_matrix_chain(
        no,
        nvir,
        wa,
        wb,
        rank,
        np.dtype(left_intermediate.dtype).itemsize,
    )
    if selection["selected_order"] == "residual-first":
        residual = (wmat @ tmat).reshape(wa, no, no, wb)
        residual = residual.transpose(1, 0, 2, 3)
        residual_matrix = residual.reshape(no * wa, no * wb)
        projected_left = left_projector.T @ residual_matrix
        projected = projected_left @ right_projector
        largest = max(
            int(wmat.nbytes),
            int(tmat.nbytes),
            int(residual.nbytes),
            int(projected_left.nbytes),
            int(projected.nbytes),
        )
        return projected, int(residual.nbytes), largest, selection

    projected_t = tmat @ right_projector
    projected_w = (wmat @ projected_t).reshape(wa, no, rank)
    projected_w = projected_w.transpose(1, 0, 2).reshape(
        no * wa, rank
    )
    projected = left_projector.T @ projected_w
    largest = max(
        int(wmat.nbytes),
        int(tmat.nbytes),
        int(projected_t.nbytes),
        int(projected_w.nbytes),
        int(projected.nbytes),
    )
    return projected, 0, largest, selection


def projected_cc_ring_gemm(
    doubles: RRDoubles,
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    *,
    auxiliary_block_size: int = 1,
    virtual_block_size: int = 32,
) -> RRResidualTermResult:
    """GEMM path for the projected ``Wvoov/Wvovo`` ring group.

    The existing :func:`projected_cc_ring` remains the equation-level oracle.
    The two direct contractions are evaluated through RR factorized GEMMs.
    The crossed contractions use bounded virtual ``T2`` tiles, each at most
    ``(nocc,nocc,nvir,virtual_block)``.  For every ordinary crossed tile, a
    matrix-chain selector compares residual-first with right-project-first
    FLOPs and logical temporary bytes.  Right-project-first is used only when
    it reduces FLOPs without spending more temporary memory.  The fourth
    ``Wvovo`` term retains a bounded residual tile; choosing
    ``virtual_block_size >= nvir`` makes that one tile span both virtual
    dimensions.  A full dense ``T2`` or four-index ERI is never formed.
    ``auxiliary`` and ``virtual`` block sizes bound the logical shapes
    independently.
    """

    vectors = doubles.projector.vectors
    core = doubles.core
    values = (core, t1, loo, lov, lvv)
    if not _same_backend(vectors, *values):
        raise TypeError("RR amplitudes and CD factors need one array backend")
    _require_real_floating(vectors, *values)
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
    flat_u = u.reshape(nocc * nvir, rank)
    dtype = xp.result_type(vectors, core, t1, loo, lov, lvv)
    half = xp.zeros((rank, rank), dtype=dtype)
    # This is the right projected factor for the first two ring terms:
    # sum(j,b) U[j,b,Q] U[j,b,S].  It is also cheap for a truncated RR rank.
    pair_metric = flat_u.T @ flat_u
    largest = int(pair_metric.nbytes)
    tile_residual_nbytes = 0
    workspace_shape_upper_bound = estimate_rr_ring_workspace_nbytes(
        nocc,
        nvir,
        auxiliary_block_size=ablock,
        virtual_block_size=vblock,
        itemsize=np.dtype(dtype).itemsize,
        rank=rank,
    )
    ordinary_order_counts: dict[str, int] = {}
    ordinary_shape_counts: dict[tuple[int, int], int] = {}
    ordinary_shape_selections: dict[tuple[int, int], dict[str, Any]] = {}
    ordinary_tile_residual_nbytes = 0

    # The direct Wvoov/Wvovo terms can be fully factorized.  This removes the
    # repeated T2 slice reconstruction from the dominant part of the ring.
    for a_start in range(0, nvir, vblock):
        a_slice = slice(a_start, min(a_start + vblock, nvir))
        ua = u[:, a_slice]
        wvoov, wvoov_largest = _cc_wvoov_block(
            doubles, t1, loo, lov, lvv, a_slice, ablock
        )
        wvovo, wvovo_largest = _cc_wvovo_block(
            doubles, t1, loo, lov, lvv, a_slice, ablock
        )
        wa = int(a_slice.stop - a_slice.start)
        # Wvoov[a,k,i,c] and Wvovo[a,k,c,i] are both reordered to (a,i,k,c).
        wvoov_matrix = wvoov.transpose(0, 2, 1, 3).reshape(
            wa * nocc, nocc * nvir
        )
        wvovo_matrix = wvovo.transpose(0, 3, 1, 2).reshape(
            wa * nocc, nocc * nvir
        )
        projected_wvoov = wvoov_matrix @ flat_u
        projected_wvovo = wvovo_matrix @ flat_u
        left_u = ua.reshape(nocc * wa, rank)
        # The GEMM rows are ordered (a,i), while the left projector is
        # ordered (i,a).  Restore the pair ordering before the projection.
        projected_wvoov = projected_wvoov.reshape(wa, nocc, rank).transpose(
            1, 0, 2
        ).reshape(nocc * wa, rank)
        projected_wvovo = projected_wvovo.reshape(wa, nocc, rank).transpose(
            1, 0, 2
        ).reshape(nocc * wa, rank)
        left_wvoov = left_u.T @ projected_wvoov
        left_wvovo = left_u.T @ projected_wvovo
        half += 2.0 * (left_wvoov @ core @ pair_metric)
        half -= left_wvovo @ core @ pair_metric
        largest = max(
            largest,
            int(wvoov.nbytes),
            int(wvovo.nbytes),
            int(projected_wvoov.nbytes),
            int(projected_wvovo.nbytes),
            wvoov_largest,
            wvovo_largest,
        )

        # Crossed terms retain only a virtual T2 tile.  The matrix products
        # below replace the old three-index einsums in the a x b loop.
        for b_start in range(0, nvir, vblock):
            b_slice = slice(b_start, min(b_start + vblock, nvir))
            ub = u[:, b_slice]
            t2_bc = _rr_t2_virtual_block(doubles, b_slice, slice(0, nvir))
            (
                crossed,
                crossed_nbytes,
                crossed_largest,
                chain_selection,
            ) = _projected_ring_gemm_tile(
                wvoov, t2_bc, ua, ub
            )
            selected_order = str(chain_selection["selected_order"])
            shape_key = (wa, int(b_slice.stop - b_slice.start))
            ordinary_order_counts[selected_order] = (
                ordinary_order_counts.get(selected_order, 0) + 1
            )
            ordinary_shape_counts[shape_key] = (
                ordinary_shape_counts.get(shape_key, 0) + 1
            )
            ordinary_shape_selections.setdefault(shape_key, chain_selection)
            half -= crossed
            ordinary_tile_residual_nbytes = max(
                ordinary_tile_residual_nbytes, crossed_nbytes
            )
            tile_residual_nbytes = max(tile_residual_nbytes, crossed_nbytes)
            largest = max(
                largest,
                int(t2_bc.nbytes),
                int(crossed.nbytes),
                crossed_nbytes,
                crossed_largest,
            )

    # The final Wvovo[b,k,c,i] * T2[k,j,a,c] term is evaluated in a second
    # tiled pass.  It is intentionally separate so its virtual T2 tile stays
    # bounded and the reference index order remains obvious.
    for b_start in range(0, nvir, vblock):
        b_slice = slice(b_start, min(b_start + vblock, nvir))
        ub = u[:, b_slice]
        wvovo_b, block_largest = _cc_wvovo_block(
            doubles, t1, loo, lov, lvv, b_slice, ablock
        )
        for a_start in range(0, nvir, vblock):
            a_slice = slice(a_start, min(a_start + vblock, nvir))
            ua = u[:, a_slice]
            t2_ac = _rr_t2_virtual_block(
                doubles, a_slice, slice(0, nvir)
            )
            (
                crossed,
                crossed_nbytes,
                crossed_largest,
                _fixed_selection,
            ) = _projected_ring_gemm_tile(
                wvovo_b,
                t2_ac,
                ua,
                ub,
                right_is_wvovo=True,
            )
            half -= crossed
            tile_residual_nbytes = max(tile_residual_nbytes, crossed_nbytes)
            largest = max(
                largest,
                int(t2_ac.nbytes),
                int(crossed.nbytes),
                crossed_nbytes,
                crossed_largest,
                block_largest,
            )

    workspace_shape_upper_bound[
        "ordinary_crossed_runtime_selected_order_counts"
    ] = {
        key: int(ordinary_order_counts[key])
        for key in sorted(ordinary_order_counts)
    }
    workspace_shape_upper_bound["ordinary_crossed_selected_orders"] = sorted(
        ordinary_order_counts
    )
    workspace_shape_upper_bound["ordinary_crossed_materialized_ijab"] = bool(
        ordinary_order_counts.get("residual-first", 0)
    )
    workspace_shape_upper_bound["ordinary_crossed_ijab_tile_nbytes"] = int(
        ordinary_tile_residual_nbytes
    )
    workspace_shape_upper_bound[
        "ordinary_crossed_runtime_tile_selections"
    ] = [
        {
            "left_virtual_width": int(shape_key[0]),
            "right_virtual_width": int(shape_key[1]),
            "application_count": int(ordinary_shape_counts[shape_key]),
            **ordinary_shape_selections[shape_key],
        }
        for shape_key in sorted(ordinary_shape_selections)
    ]
    projected = half + half.T
    return RRResidualTermResult(
        core=projected,
        term="complete-cc-Wvoov-Wvovo-ring-gemm",
        auxiliary_block_size=ablock,
        largest_intermediate_nbytes=largest,
        kernel_name="rr-ring-gemm",
        tile_residual_nbytes=tile_residual_nbytes,
        workspace_shape_upper_bound=workspace_shape_upper_bound,
    )


def build_projected_ccsd_doubles_components(
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
    ring_kernel: str = "reference",
    transfer_counter: Any = None,
    profile_phase: Optional[RRDoublesProfilePhase] = None,
) -> RRCCSDDoublesComponents:
    """Build every real-RHF RCCSD doubles term without assembling its core.

    Keeping the six coarse equation groups separate lets the ordinary RR
    path sum from zero while a hybrid path sums from a signed ``R_123``
    offset.  ``ring_kernel`` remains an explicit selector so the bounded
    reference is the default and ``"gemm"`` opts into the tiled GEMM path.
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
    ring_kernel = str(ring_kernel).lower()
    if ring_kernel not in {"reference", "gemm"}:
        raise ValueError("ring_kernel must be 'reference' or 'gemm'")
    ablock = int(auxiliary_block_size)
    vblock = int(virtual_block_size)
    if ablock < 1 or vblock < 1:
        raise ValueError("auxiliary and virtual block sizes must be positive")
    xp = _array_module(vectors)

    def phase_metadata(coarse_term: str) -> dict[str, Any]:
        return {
            "coarse_term": coarse_term,
            "ring_kernel": ring_kernel,
            "auxiliary_block_size": ablock,
            "virtual_block_size": vblock,
        }

    with _rr_doubles_profile_phase(
        "rr_doubles_fock_intermediates",
        phase_metadata("fock_intermediates"),
        profile_phase,
    ):
        f_intermediates = build_cc_fock_intermediates(
            doubles,
            t1,
            lov,
            fock_oo,
            fock_ov,
            fock_vv,
            auxiliary_block_size=ablock,
        )
    with _rr_doubles_profile_phase(
        "rr_doubles_lagrangian",
        phase_metadata("lagrangian"),
        profile_phase,
    ):
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

    terms: list[RRResidualTermResult] = []
    with _rr_doubles_profile_phase(
        "rr_doubles_term_linear_t1",
        phase_metadata("linear_t1"),
        profile_phase,
    ):
        terms.append(
            projected_linear_t1_doubles(
                doubles.projector,
                t1,
                loo,
                lov,
                lvv,
                auxiliary_block_size=ablock,
            )
        )
    with _rr_doubles_profile_phase(
        "rr_doubles_term_bare_ovov",
        phase_metadata("bare_ovov"),
        profile_phase,
    ):
        terms.append(
            projected_bare_ovov(doubles.projector, lov, nocc, nvir)
        )
    with _rr_doubles_profile_phase(
        "rr_doubles_term_woooo",
        phase_metadata("woooo"),
        profile_phase,
    ):
        terms.append(
            projected_cc_woooo(
                doubles,
                t1,
                loo,
                lov,
                auxiliary_block_size=ablock,
                virtual_block_size=vblock,
                transfer_counter=transfer_counter,
            )
        )
    with _rr_doubles_profile_phase(
        "rr_doubles_term_wvvvv",
        phase_metadata("wvvvv"),
        profile_phase,
    ):
        terms.append(
            projected_cc_wvvvv(
                doubles,
                t1,
                lov,
                lvv,
                auxiliary_block_size=ablock,
            )
        )
    with _rr_doubles_profile_phase(
        "rr_doubles_term_one_body",
        phase_metadata("one_body"),
        profile_phase,
    ):
        terms.append(
            projected_one_body_dressing(
                doubles, occupied_dressing, virtual_dressing
            )
        )
    with _rr_doubles_profile_phase(
        "rr_doubles_term_ring",
        phase_metadata("ring"),
        profile_phase,
    ):
        ring_builder = (
            projected_cc_ring_gemm
            if ring_kernel == "gemm"
            else projected_cc_ring
        )
        terms.append(
            ring_builder(
                doubles,
                t1,
                loo,
                lov,
                lvv,
                auxiliary_block_size=ablock,
                virtual_block_size=vblock,
            )
        )
    return RRCCSDDoublesComponents(
        terms=tuple(terms),
        fock_intermediates=f_intermediates,
        lagrangian_one_body=lagrangian,
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
    ring_kernel: str = "reference",
    transfer_counter: Any = None,
    profile_phase: Optional[RRDoublesProfilePhase] = None,
) -> RRCCSDDoublesResult:
    """Build every real-RHF RCCSD doubles-numerator term in RR/CD form."""

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
        ring_kernel=ring_kernel,
        transfer_counter=transfer_counter,
        profile_phase=profile_phase,
    )
    return RRCCSDDoublesResult(
        core=components.assemble_core(),
        fock_intermediates=components.fock_intermediates,
        lagrangian_one_body=components.lagrangian_one_body,
        term_metadata=components.term_metadata,
    )


__all__ = [
    "RRDoublesProfilePhase",
    "RRResidualTermResult",
    "RRSingleTermResult",
    "CCFockIntermediates",
    "RRCCSDEnergyResult",
    "RRCCSDSinglesResult",
    "RRCCSDDoublesResult",
    "RRCCSDDoublesComponents",
    "CCLagrangianOneBody",
    "ProjectedPairDenominator",
    "build_cc_fock_intermediates",
    "rr_ccsd_energy",
    "build_ccsd_singles_numerator",
    "build_projected_ccsd_doubles_components",
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
    "projected_cc_ring_gemm",
    "estimate_projected_cc_wvvvv_workspace_nbytes",
    "estimate_rr_ring_workspace_nbytes",
    "singles_from_fov_doubles",
    "singles_from_ovoo_doubles",
    "singles_from_cd_ovvv",
    "singles_from_cd_ovoo",
    "singles_from_cd_ovvo_oovv",
]
