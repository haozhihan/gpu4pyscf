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

"""Amplitude-THC residual kernels from Hohenstein et al. Algorithms 1--3.

The routines in this module consume the paper's doubles representation

``t2[i,j,a,b] = y_occ[i,X] y_vir[a,X] T[X,Y]``
``                y_occ[j,Y] y_vir[b,Y]``

and return a residual in the same THC auxiliary pair space.  They cover Eqs.
28, 29, and 31 of J. Chem. Phys. 156, 054102 (2022); they are not a complete
CCSD doubles residual.  A caller retaining a semi-unitary RR working basis
must transform ``T_rr`` into the THC basis before these kernels and transform
their result back to RR before applying the projected denominator.

Algorithm 1 follows Eq. 28 rather than a literal reading of Appendix line 13.
The occupied transformed intermediate needs a transpose because the
T1-transformed Cholesky blocks are generally nonsymmetric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def _same_backend(reference: Any, *values: Any) -> bool:
    xp = _array_module(reference)
    return all(_array_module(value) is xp for value in values)


def _require_real_floating(*values: Any) -> None:
    dtypes = tuple(np.dtype(value.dtype) for value in values)
    if any(np.issubdtype(dtype, np.complexfloating) for dtype in dtypes):
        raise NotImplementedError("THC Algorithms 1-3 currently require real RHF")
    if any(not np.issubdtype(dtype, np.floating) for dtype in dtypes):
        raise TypeError("THC residual kernels require floating-point tensors")


@dataclass(frozen=True)
class T1TransformedCholesky:
    """One block of nonsymmetric T1-transformed MO Cholesky factors."""

    hoo: Any
    hov: Any
    hvv: Any
    hvo: Any

    @property
    def naux(self) -> int:
        return int(self.hoo.shape[0])

    @property
    def storage_nbytes(self) -> int:
        return int(
            self.hoo.nbytes
            + self.hov.nbytes
            + self.hvv.nbytes
            + self.hvo.nbytes
        )


@dataclass(frozen=True)
class THCResidual123Result:
    """Compressed residual pieces for paper Algorithms 1--3."""

    sigma_thc: Any
    algorithm_1: Any
    algorithm_2: Any
    algorithm_3: Any
    auxiliary_block_size: int
    largest_intermediate_nbytes: int
    algorithm_1_provenance: Optional[dict[str, Any]] = None
    algorithm_3_provenance: Optional[dict[str, Any]] = None
    provenance_largest_intermediate_nbytes: int = 0
    coarse_ledger: Optional[dict[str, Any]] = None
    coarse_ledger_largest_intermediate_nbytes: int = 0
    coarse_ledger_retained_nbytes: int = 0

    def to_rr(self, tau: Any):
        """Return ``tau @ sigma_thc @ tau.T`` in the RR working basis."""

        return project_thc_residual_to_rr(self.sigma_thc, tau)

    def metadata(self) -> dict[str, Any]:
        available = (
            self.algorithm_1_provenance is not None
            and self.algorithm_3_provenance is not None
        )
        coarse_available = self.coarse_ledger is not None
        return {
            "paper": "Hohenstein-2022",
            "equations": [28, 29, 31],
            "algorithms": [1, 2, 3],
            "complete_ccsd_residual": False,
            "algorithm_1_uses_eq28_index_orientation": True,
            "auxiliary_block_size": int(self.auxiliary_block_size),
            "largest_intermediate_nbytes": int(self.largest_intermediate_nbytes),
            "materialized_dense_t2": False,
            "materialized_four_index_eri": False,
            "provenance_decomposition": available,
            "provenance_available": available,
            "provenance_largest_intermediate_nbytes": int(
                self.provenance_largest_intermediate_nbytes
            ),
            "largest_intermediate_nbytes_scope": "aggregate-residual-path",
            "provenance_audit_memory_scope": "reported-separately",
            "coarse_r123_audit_ledger": coarse_available,
            "coarse_ledger_available": coarse_available,
            "coarse_r123_ledger_kind": "audit-only",
            "coarse_r123_diagram_partition_fused": False,
            "coarse_ledger_performance_eligible": False,
            "coarse_ledger_largest_intermediate_nbytes": int(
                self.coarse_ledger_largest_intermediate_nbytes
            ),
            "coarse_ledger_largest_intermediate_scope": (
                "largest-single-logical-array"
            ),
            "coarse_ledger_retained_nbytes": int(
                self.coarse_ledger_retained_nbytes
            ),
            "coarse_ledger_memory_scope": "reported-separately",
            "coarse_ledger_peak_memory_requires_profiling": True,
            "coarse_ledger_buckets": (
                list(self.coarse_ledger)
                if self.coarse_ledger is not None
                else []
            ),
            "algorithm_1_provenance": (
                list(self.algorithm_1_provenance)
                if self.algorithm_1_provenance is not None
                else []
            ),
            "algorithm_3_provenance": (
                list(self.algorithm_3_provenance)
                if self.algorithm_3_provenance is not None
                else []
            ),
        }


@dataclass(frozen=True)
class PairFactorResidual123Result:
    """Algorithms 1--3 projected through an arbitrary pair factor.

    ``pair_factor[(i,a),P]`` need not be separable in ``i`` and ``a``.  This
    gives the exact RR-space contribution which a hybrid engine subtracts
    before adding the amplitude-THC approximation.  The explicit reference is
    crucial: Algorithms 1--3 cut across the usual PySCF ``Woooo``, ``Wvvvv``
    and ring groupings, so subtracting one of those coarse groups would double
    count or omit diagrams.
    """

    sigma_pair: Any
    algorithm_1: Any
    algorithm_2: Any
    algorithm_3: Any
    nocc: int
    nvir: int
    auxiliary_block_size: int
    largest_intermediate_nbytes: int
    algorithm_1_provenance: Optional[dict[str, Any]] = None
    algorithm_3_provenance: Optional[dict[str, Any]] = None
    provenance_largest_intermediate_nbytes: int = 0
    coarse_ledger: Optional[dict[str, Any]] = None
    coarse_ledger_largest_intermediate_nbytes: int = 0
    coarse_ledger_retained_nbytes: int = 0

    def metadata(self) -> dict[str, Any]:
        available = (
            self.algorithm_1_provenance is not None
            and self.algorithm_3_provenance is not None
        )
        coarse_available = self.coarse_ledger is not None
        return {
            "paper": "Hohenstein-2022",
            "equations": [28, 29, 31],
            "algorithms": [1, 2, 3],
            "coordinate_space": "arbitrary-pair-factor",
            "complete_ccsd_residual": False,
            "purpose": "exact-subtraction-reference-for-hybrid-thc",
            "nocc": int(self.nocc),
            "nvir": int(self.nvir),
            "auxiliary_block_size": int(self.auxiliary_block_size),
            "largest_intermediate_nbytes": int(
                self.largest_intermediate_nbytes
            ),
            "materialized_dense_t2": False,
            "materialized_four_index_eri": False,
            "provenance_decomposition": available,
            "provenance_available": available,
            "provenance_largest_intermediate_nbytes": int(
                self.provenance_largest_intermediate_nbytes
            ),
            "largest_intermediate_nbytes_scope": "aggregate-residual-path",
            "provenance_audit_memory_scope": "reported-separately",
            "coarse_r123_audit_ledger": coarse_available,
            "coarse_ledger_available": coarse_available,
            "coarse_r123_ledger_kind": "audit-only",
            "coarse_r123_diagram_partition_fused": False,
            "coarse_ledger_performance_eligible": False,
            "coarse_ledger_largest_intermediate_nbytes": int(
                self.coarse_ledger_largest_intermediate_nbytes
            ),
            "coarse_ledger_largest_intermediate_scope": (
                "largest-single-logical-array"
            ),
            "coarse_ledger_retained_nbytes": int(
                self.coarse_ledger_retained_nbytes
            ),
            "coarse_ledger_memory_scope": "reported-separately",
            "coarse_ledger_peak_memory_requires_profiling": True,
            "coarse_ledger_buckets": (
                list(self.coarse_ledger)
                if self.coarse_ledger is not None
                else []
            ),
            "algorithm_1_provenance": (
                list(self.algorithm_1_provenance)
                if self.algorithm_1_provenance is not None
                else []
            ),
            "algorithm_3_provenance": (
                list(self.algorithm_3_provenance)
                if self.algorithm_3_provenance is not None
                else []
            ),
        }


def transform_cholesky_t1(
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
) -> T1TransformedCholesky:
    """Apply the particle/hole coefficient transforms of Eqs. 25--27."""

    if not _same_backend(t1, loo, lov, lvv):
        raise TypeError("t1 and Cholesky factors need one array backend")
    _require_real_floating(t1, loo, lov, lvv)
    if getattr(t1, "ndim", None) != 2:
        raise ValueError("t1 must be a matrix")
    nocc, nvir = map(int, t1.shape)
    if getattr(lov, "ndim", None) != 3 or lov.shape[1:] != (nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    naux = int(lov.shape[0])
    if loo.shape != (naux, nocc, nocc):
        raise ValueError("loo must have shape (naux,nocc,nocc)")
    if lvv.shape != (naux, nvir, nvir):
        raise ValueError("lvv must have shape (naux,nvir,nvir)")
    xp = _array_module(t1)
    hoo = loo + xp.einsum("Aia,ja->Aij", lov, t1)
    hov = lov
    hvv = lvv - xp.einsum("ia,Aib->Aab", t1, lov)
    hvo = lov.transpose(0, 2, 1).copy()
    hvo += xp.einsum("Aab,ib->Aai", lvv, t1)
    hvo -= xp.einsum("ka,Aki->Aai", t1, loo)
    hvo -= xp.einsum("ka,Akb,ib->Aai", t1, lov, t1)
    return T1TransformedCholesky(hoo=hoo, hov=hov, hvv=hvv, hvo=hvo)


def _validate_thc_inputs(
    y_occ: Any,
    y_vir: Any,
    core: Any,
    transformed: T1TransformedCholesky,
):
    if not _same_backend(
        y_occ,
        y_vir,
        core,
        transformed.hoo,
        transformed.hov,
        transformed.hvv,
        transformed.hvo,
    ):
        raise TypeError("THC amplitudes and transformed factors need one backend")
    _require_real_floating(
        y_occ,
        y_vir,
        core,
        transformed.hoo,
        transformed.hov,
        transformed.hvv,
        transformed.hvo,
    )
    if getattr(y_occ, "ndim", None) != 2 or getattr(y_vir, "ndim", None) != 2:
        raise ValueError("THC occupied and virtual factors must be matrices")
    nocc, rank = map(int, y_occ.shape)
    nvir, virtual_rank = map(int, y_vir.shape)
    if virtual_rank != rank or core.shape != (rank, rank):
        raise ValueError("THC factor ranks and core shape do not match")
    naux = transformed.naux
    expected = (
        transformed.hoo.shape == (naux, nocc, nocc)
        and transformed.hov.shape == (naux, nocc, nvir)
        and transformed.hvv.shape == (naux, nvir, nvir)
        and transformed.hvo.shape == (naux, nvir, nocc)
    )
    if not expected:
        raise ValueError(
            "T1-transformed Cholesky block shapes do not match THC factors"
        )


def thc_algorithm_1(
    y_occ: Any,
    y_vir: Any,
    core: Any,
    transformed: T1TransformedCholesky,
):
    """Evaluate paper Eq. 28 / Algorithm 1 in THC residual space."""

    _validate_thc_inputs(y_occ, y_vir, core, transformed)
    xp = _array_module(core)
    soo = y_occ.T @ y_occ
    svv = y_vir.T @ y_vir
    boo = xp.einsum("kX,Aki,iY->AXY", y_occ, transformed.hoo, y_occ)
    bvv = xp.einsum("aX,Aac,cY->AXY", y_vir, transformed.hvv, y_vir)
    # Appendix Algorithm 1 line 13 omits this occupied-index orientation.
    # Eq. 28 fixes it; the difference is observable whenever t1 != 0.
    combined = svv[None] * boo.transpose(0, 2, 1)
    combined -= soo[None] * bvv
    return xp.einsum("AXY,YZ,AWZ->XW", combined, core, combined)


def thc_algorithm_2(
    y_occ: Any,
    y_vir: Any,
    core: Any,
    transformed: T1TransformedCholesky,
):
    """Evaluate paper Eq. 29 / Algorithm 2 in THC residual space."""

    _validate_thc_inputs(y_occ, y_vir, core, transformed)
    xp = _array_module(core)
    soo = y_occ.T @ y_occ
    svv = y_vir.T @ y_vir
    abar = xp.einsum("iX,aX,Aai->AX", y_occ, y_vir, transformed.hvo)
    atilde = xp.einsum("iX,aX,Aia->AX", y_occ, y_vir, transformed.hov)
    btilde = xp.einsum("YZ,AZ->AY", core, atilde)
    dtilde = xp.einsum("XY,XY,AY->AX", soo, svv, btilde)
    exchange = xp.einsum(
        "cY,kZ,Akc->AYZ", y_vir, y_occ, transformed.hov
    )
    hterm = xp.einsum("XY,XZ,YZ,AYZ->AX", soo, svv, core, exchange)
    sigma = abar.T @ abar
    sigma += 2.0 * (dtilde.T @ abar + abar.T @ dtilde)
    sigma -= hterm.T @ abar + abar.T @ hterm
    return sigma


def thc_algorithm_3(
    y_occ: Any,
    y_vir: Any,
    core: Any,
    transformed: T1TransformedCholesky,
):
    """Evaluate paper Eq. 31 / Algorithm 3 in THC residual space."""

    _validate_thc_inputs(y_occ, y_vir, core, transformed)
    xp = _array_module(core)
    soo = y_occ.T @ y_occ
    svv = y_vir.T @ y_vir
    avv = xp.einsum("bX,cY,Abc->AXY", y_vir, y_vir, transformed.hvv)
    boo_transposed = xp.einsum(
        "jX,kY,Akj->AXY", y_occ, y_occ, transformed.hoo
    )
    coupling = xp.sum(avv * boo_transposed, axis=0)
    dressed_core = xp.einsum("XY,XY,YZ->XZ", soo, svv, core)
    return -(dressed_core @ coupling.T + coupling @ dressed_core.T)


def _validate_pair_factor_inputs(
    pair_factor: Any,
    core: Any,
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    nocc: int,
    nvir: int,
) -> None:
    if not _same_backend(pair_factor, core, t1, loo, lov, lvv):
        raise TypeError("pair factor, amplitudes, and CD factors need one backend")
    _require_real_floating(pair_factor, core, t1, loo, lov, lvv)
    if getattr(pair_factor, "ndim", None) != 2:
        raise ValueError("pair_factor must be a matrix")
    rank = int(pair_factor.shape[1])
    if pair_factor.shape[0] != nocc * nvir:
        raise ValueError("pair_factor rows must equal nocc*nvir")
    if core.shape != (rank, rank):
        raise ValueError("core shape must match the pair-factor rank")
    if t1.shape != (nocc, nvir):
        raise ValueError("t1 shape does not match the pair factor")
    if getattr(lov, "ndim", None) != 3 or lov.shape[1:] != (nocc, nvir):
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    naux = int(lov.shape[0])
    if loo.shape != (naux, nocc, nocc):
        raise ValueError("loo must have shape (naux,nocc,nocc)")
    if lvv.shape != (naux, nvir, nvir):
        raise ValueError("lvv must have shape (naux,nvir,nvir)")


def _coarse_r123_ledger(
    occupied: Any,
    virtual: Any,
    core: Any,
    algorithm_2: Any,
    algorithm_3: Any,
):
    """Build audit-only algebraic buckets from projected Hoo/Hvv factors.

    With ``O`` and ``V`` the projected Hoo and Hvv factors, respectively,
    Algorithm 1 is ``(O-V) C (O-V)``.  The first two buckets retain its pure
    ``OCO`` and ``VCV`` pieces.  The third bucket contains the two Algorithm 1
    cross terms and the complete Algorithms 2 and 3 contributions.  This is
    an algebraic accounting ledger only; the bucket names do not identify
    production PySCF Woooo/Wvvvv/ring diagram groups.
    """

    xp = _array_module(occupied)
    occupied_quadratic = xp.einsum(
        "AXY,YZ,AWZ->XW", occupied, core, occupied
    )
    virtual_quadratic = xp.einsum(
        "AXY,YZ,AWZ->XW", virtual, core, virtual
    )
    cross_plus_2_plus_3 = -(
        xp.einsum("AXY,YZ,AWZ->XW", occupied, core, virtual)
        + xp.einsum("AXY,YZ,AWZ->XW", virtual, core, occupied)
    )
    cross_plus_2_plus_3 += algorithm_2 + algorithm_3
    ledger = {
        "algorithm_1_occupied_quadratic": occupied_quadratic,
        "algorithm_1_virtual_quadratic": virtual_quadratic,
        "algorithm_1_cross_plus_2_plus_3": cross_plus_2_plus_3,
    }
    largest = max(
        int(value.nbytes) for value in (occupied, virtual, *ledger.values())
    )
    return ledger, largest


def _pair_factor_algorithms_1_3_block(
    pair_factor: Any,
    core: Any,
    transformed: T1TransformedCholesky,
    nocc: int,
    nvir: int,
    *,
    loo: Any = None,
    lvv: Any = None,
    return_provenance: bool = False,
):
    """Return Eqs. 28, 29 and 31 for one Cholesky block.

    This is the arbitrary-pair-factor generalization of the paper algorithms.
    Setting ``pair_factor[i,a,X] = y_occ[i,X] * y_vir[a,X]`` reduces each
    expression exactly to :func:`thc_algorithm_1`, :func:`thc_algorithm_2`,
    and :func:`thc_algorithm_3`.
    """

    xp = _array_module(pair_factor)
    rank = int(pair_factor.shape[1])
    u = pair_factor.reshape(nocc, nvir, rank)

    # Equation 28.  For a separable pair factor these two matrices become
    # svv*boo.T and soo*bvv, respectively.
    occupied = xp.einsum("iaX,Aki,kaY->AXY", u, transformed.hoo, u)
    virtual = xp.einsum("iaX,Aac,icY->AXY", u, transformed.hvv, u)
    combined = occupied - virtual
    part1 = xp.einsum("AXY,YZ,AWZ->XW", combined, core, combined)

    # Equation 29.  Build (2*t_ik^ac - t_ik^ca) * Hov_Akc without a dense
    # doubles tensor.  The exchange path retains at most (A,i,k,rank).
    abar = xp.einsum("iaX,Aai->AX", u, transformed.hvo)
    direct_trace = xp.einsum("Akc,kcQ->AQ", transformed.hov, u)
    direct_coeff = direct_trace @ core.T
    direct_action = 2.0 * xp.einsum("iaP,AP->Aia", u, direct_coeff)
    crossed = xp.einsum("icP,Akc->AikP", u, transformed.hov)
    crossed_core = xp.einsum("AikP,PQ->AikQ", crossed, core)
    exchange_action = xp.einsum("AikQ,kaQ->Aia", crossed_core, u)
    spin_action = direct_action - exchange_action
    projected_action = xp.einsum("iaX,Aia->AX", u, spin_action)
    part2 = abar.T @ abar
    part2 += projected_action.T @ abar + abar.T @ projected_action

    # Equation 31.  The pair Gram is the Hadamard occupied/virtual Gram in
    # the separable endpoint.  Keeping it explicit also permits independent
    # testing with non-orthogonal pair factors.
    pair_gram = pair_factor.T @ pair_factor
    dressed_core = pair_gram @ core
    occupied_action = xp.einsum("Akj,jbX->AkbX", transformed.hoo, u)
    virtual_action = xp.einsum("Abc,kcY->AkbY", transformed.hvv, u)
    coupling = xp.einsum("AkbX,AkbY->XY", occupied_action, virtual_action)
    part3 = -(
        dressed_core @ coupling.T + coupling @ dressed_core.T
    )
    largest = max(
        int(value.nbytes)
        for value in (
            occupied,
            virtual,
            combined,
            part1,
            abar,
            direct_trace,
            direct_coeff,
            direct_action,
            crossed,
            crossed_core,
            exchange_action,
            spin_action,
            projected_action,
            pair_gram,
            dressed_core,
            occupied_action,
            virtual_action,
            coupling,
            part2,
            part3,
        )
    )
    provenance = None
    coarse_ledger = None
    coarse_largest = 0
    if return_provenance:
        if loo is None or lvv is None:
            raise ValueError(
                "loo and lvv are required when return_provenance is true"
            )
        provenance = _pair_factor_provenance_algorithms_1_3_block(
            pair_factor, core, transformed, loo, lvv, nocc, nvir
        )
        coarse_ledger, coarse_largest = _coarse_r123_ledger(
            occupied, virtual, core, part2, part3
        )
    if return_provenance:
        return part1, part2, part3, largest, (
            provenance[0],
            provenance[1],
            provenance[2],
            coarse_ledger,
            coarse_largest,
        )
    return part1, part2, part3, largest


def _pair_factor_provenance_algorithms_1_3_block(
    pair_factor: Any,
    core: Any,
    transformed: T1TransformedCholesky,
    loo: Any,
    lvv: Any,
    nocc: int,
    nvir: int,
):
    """Expand Algorithms 1 and 3 by their base-L/T1-dressed provenance.

    The decomposition is performed after projection into pair-factor space,
    so it works for a genuinely non-separable RR factor as well as for the
    separable THC endpoint.  ``L`` denotes the bare ``Loo``/``Lvv`` blocks;
    ``T1`` denotes the complete linear T1 piece, including the minus sign in
    ``Hvv = Lvv - t1*Lov``.  Algorithm 1 is quadratic in the combined
    ``Hoo-Hvv`` factor and therefore has four ordered subcores.  Algorithm 3
    is linear in the Hoo/Hvv coupling and has four source-pair subcores.
    """

    xp = _array_module(pair_factor)
    rank = int(pair_factor.shape[1])
    u = pair_factor.reshape(nocc, nvir, rank)
    if loo.shape != transformed.hoo.shape or lvv.shape != transformed.hvv.shape:
        raise ValueError("base Cholesky blocks do not match transformed blocks")

    # Recover the two pieces without retaining a second copy in the public
    # transformed-factor object.  This keeps the production memory contract
    # unchanged while making the provenance explicit at the residual call.
    hoo_l = loo
    hoo_t1 = transformed.hoo - loo
    hvv_l = lvv
    hvv_t1 = transformed.hvv - lvv

    oo_l = xp.einsum("iaX,Aki,kaY->AXY", u, hoo_l, u)
    oo_t1 = xp.einsum("iaX,Aki,kaY->AXY", u, hoo_t1, u)
    vv_l = xp.einsum("iaX,Aac,icY->AXY", u, hvv_l, u)
    vv_t1 = xp.einsum("iaX,Aac,icY->AXY", u, hvv_t1, u)
    c_l = oo_l - vv_l
    c_t1 = oo_t1 - vv_t1
    algorithm_1 = {
        "LL": xp.einsum("AXY,YZ,AWZ->XW", c_l, core, c_l),
        "LT1": xp.einsum("AXY,YZ,AWZ->XW", c_l, core, c_t1),
        "T1L": xp.einsum("AXY,YZ,AWZ->XW", c_t1, core, c_l),
        "T1T1": xp.einsum("AXY,YZ,AWZ->XW", c_t1, core, c_t1),
    }

    pair_gram = pair_factor.T @ pair_factor
    dressed_core = pair_gram @ core
    oo_action_l = xp.einsum("Akj,jbX->AkbX", hoo_l, u)
    oo_action_t1 = xp.einsum("Akj,jbX->AkbX", hoo_t1, u)
    vv_action_l = xp.einsum("Abc,kcY->AkbY", hvv_l, u)
    vv_action_t1 = xp.einsum("Abc,kcY->AkbY", hvv_t1, u)
    coupling = {
        "Loo_Lvv": xp.einsum(
            "AkbX,AkbY->XY", oo_action_l, vv_action_l
        ),
        "Loo_T1vv": xp.einsum(
            "AkbX,AkbY->XY", oo_action_l, vv_action_t1
        ),
        "T1oo_Lvv": xp.einsum(
            "AkbX,AkbY->XY", oo_action_t1, vv_action_l
        ),
        "T1oo_T1vv": xp.einsum(
            "AkbX,AkbY->XY", oo_action_t1, vv_action_t1
        ),
    }
    algorithm_3 = {
        key: -(dressed_core @ value.T + value @ dressed_core.T)
        for key, value in coupling.items()
    }
    audit_values = (
        hoo_t1,
        hvv_t1,
        oo_l,
        oo_t1,
        vv_l,
        vv_t1,
        c_l,
        c_t1,
        pair_gram,
        dressed_core,
        oo_action_l,
        oo_action_t1,
        vv_action_l,
        vv_action_t1,
        *algorithm_1.values(),
        *coupling.values(),
        *algorithm_3.values(),
    )
    audit_largest = max(int(value.nbytes) for value in audit_values)
    return algorithm_1, algorithm_3, audit_largest


def pair_factor_residual_algorithms_1_3(
    pair_factor: Any,
    core: Any,
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    *,
    nocc: int,
    nvir: int,
    auxiliary_block_size: Optional[int] = None,
    record_provenance: bool = False,
) -> PairFactorResidual123Result:
    """Evaluate the exact Algorithms 1--3 group for a general pair factor.

    Cholesky vectors are streamed and neither ``t2[i,j,a,b]`` nor a four-index
    ERI is constructed.  This routine is the subtraction side of the hybrid
    identity ``R_hybrid = R_RR - R_123[U] + R_123[U_THC]``.

    Set ``record_provenance=True`` for the diagnostic L/T1 subcores.  It is
    disabled by default so production residual calls do not allocate or
    evaluate the audit contractions.
    """

    nocc = int(nocc)
    nvir = int(nvir)
    if nocc < 1 or nvir < 1:
        raise ValueError("nocc and nvir must be positive")
    _validate_pair_factor_inputs(
        pair_factor, core, t1, loo, lov, lvv, nocc, nvir
    )
    naux = int(lov.shape[0])
    block_size = naux if auxiliary_block_size is None else int(
        auxiliary_block_size
    )
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")
    xp = _array_module(pair_factor)
    rank = int(pair_factor.shape[1])
    dtype = xp.result_type(pair_factor, core, t1, loo, lov, lvv)
    sigma1 = xp.zeros((rank, rank), dtype=dtype)
    sigma2 = xp.zeros_like(sigma1)
    sigma3 = xp.zeros_like(sigma1)
    provenance1 = None
    provenance3 = None
    provenance_largest = 0
    coarse_ledger = None
    coarse_largest = 0
    largest = 0
    for start in range(0, naux, block_size):
        stop = min(start + block_size, naux)
        transformed = transform_cholesky_t1(
            t1, loo[start:stop], lov[start:stop], lvv[start:stop]
        )
        block_result = _pair_factor_algorithms_1_3_block(
            pair_factor,
            core,
            transformed,
            nocc,
            nvir,
            loo=loo[start:stop] if record_provenance else None,
            lvv=lvv[start:stop] if record_provenance else None,
            return_provenance=record_provenance,
        )
        if record_provenance:
            (
                part1,
                part2,
                part3,
                block_largest,
                block_provenance,
            ) = block_result
        else:
            part1, part2, part3, block_largest = block_result
        sigma1 += part1
        sigma2 += part2
        sigma3 += part3
        if record_provenance and block_provenance is not None:
            (
                block1,
                block3,
                block_provenance_largest,
                block_coarse,
                block_coarse_largest,
            ) = block_provenance
            provenance_largest = max(
                provenance_largest, block_provenance_largest
            )
            coarse_largest = max(coarse_largest, block_coarse_largest)
            if provenance1 is None:
                provenance1 = {
                    key: xp.zeros_like(value)
                    for key, value in block1.items()
                }
                provenance3 = {
                    key: xp.zeros_like(value)
                    for key, value in block3.items()
                }
                coarse_ledger = {
                    key: xp.zeros_like(value)
                    for key, value in block_coarse.items()
                }
            for key, value in block1.items():
                provenance1[key] += value
            for key, value in block3.items():
                provenance3[key] += value
            for key, value in block_coarse.items():
                coarse_ledger[key] += value
        largest = max(
            largest,
            transformed.storage_nbytes,
            block_largest,
        )
    coarse_retained = (
        0
        if coarse_ledger is None
        else sum(int(value.nbytes) for value in coarse_ledger.values())
    )
    return PairFactorResidual123Result(
        sigma_pair=sigma1 + sigma2 + sigma3,
        algorithm_1=sigma1,
        algorithm_2=sigma2,
        algorithm_3=sigma3,
        nocc=nocc,
        nvir=nvir,
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=largest,
        algorithm_1_provenance=provenance1,
        algorithm_3_provenance=provenance3,
        provenance_largest_intermediate_nbytes=provenance_largest,
        coarse_ledger=coarse_ledger,
        coarse_ledger_largest_intermediate_nbytes=coarse_largest,
        coarse_ledger_retained_nbytes=coarse_retained,
    )


def rr_residual_algorithms_1_3(
    doubles: Any,
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
    record_provenance: bool = False,
) -> PairFactorResidual123Result:
    """Return the exact RR-space contribution of paper Algorithms 1--3.

    ``record_provenance`` is forwarded to the arbitrary-pair implementation
    and is disabled by default for the production path.
    """

    required = ("projector", "core", "nocc", "nvir")
    if any(not hasattr(doubles, name) for name in required):
        raise TypeError("doubles must provide the RRDoubles interface")
    return pair_factor_residual_algorithms_1_3(
        doubles.projector.vectors,
        doubles.core,
        t1,
        loo,
        lov,
        lvv,
        nocc=int(doubles.nocc),
        nvir=int(doubles.nvir),
        auxiliary_block_size=auxiliary_block_size,
        record_provenance=record_provenance,
    )


def thc_residual_algorithms_1_3(
    y_occ: Any,
    y_vir: Any,
    core: Any,
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
    record_provenance: bool = False,
) -> THCResidual123Result:
    """Stream CD blocks through Algorithms 1--3 and sum their residuals.

    Set ``record_provenance=True`` to retain the L/T1 audit subcores.  The
    default path does not build the pair-factor audit tensor or its extra
    contractions.
    """

    if not _same_backend(y_occ, y_vir, core, t1, loo, lov, lvv):
        raise TypeError("all THC/CD inputs need one array backend")
    if getattr(lov, "ndim", None) != 3:
        raise ValueError("lov must have shape (naux,nocc,nvir)")
    naux = int(lov.shape[0])
    block_size = naux if auxiliary_block_size is None else int(auxiliary_block_size)
    if block_size < 1:
        raise ValueError("auxiliary_block_size must be positive")
    xp = _array_module(core)
    rank = int(core.shape[0])
    dtype = xp.result_type(y_occ, y_vir, core, t1, loo, lov, lvv)
    sigma1 = xp.zeros((rank, rank), dtype=dtype)
    sigma2 = xp.zeros_like(sigma1)
    sigma3 = xp.zeros_like(sigma1)
    provenance1 = None
    provenance3 = None
    provenance_largest = 0
    coarse_ledger = None
    coarse_largest = 0
    pair_factor = None
    if record_provenance:
        pair_factor = xp.einsum("iX,aX->iaX", y_occ, y_vir).reshape(
            int(y_occ.shape[0]) * int(y_vir.shape[0]), rank
        )
    largest = 0
    for start in range(0, naux, block_size):
        stop = min(start + block_size, naux)
        transformed = transform_cholesky_t1(
            t1, loo[start:stop], lov[start:stop], lvv[start:stop]
        )
        part1 = thc_algorithm_1(y_occ, y_vir, core, transformed)
        part2 = thc_algorithm_2(y_occ, y_vir, core, transformed)
        part3 = thc_algorithm_3(y_occ, y_vir, core, transformed)
        if record_provenance:
            block_provenance = _pair_factor_provenance_algorithms_1_3_block(
                pair_factor,
                core,
                transformed,
                loo[start:stop],
                lvv[start:stop],
                int(y_occ.shape[0]),
                int(y_vir.shape[0]),
            )
            block1, block3, block_provenance_largest = block_provenance
            provenance_largest = max(
                provenance_largest, block_provenance_largest
            )
            if provenance1 is None:
                provenance1 = {
                    key: xp.zeros_like(value) for key, value in block1.items()
                }
                provenance3 = {
                    key: xp.zeros_like(value) for key, value in block3.items()
                }
            for key, value in block1.items():
                provenance1[key] += value
            for key, value in block3.items():
                provenance3[key] += value
            pair_view = pair_factor.reshape(
                int(y_occ.shape[0]), int(y_vir.shape[0]), rank
            )
            occupied = xp.einsum(
                "iaX,Aki,kaY->AXY", pair_view, transformed.hoo, pair_view
            )
            virtual = xp.einsum(
                "iaX,Aac,icY->AXY", pair_view, transformed.hvv, pair_view
            )
            block_coarse, block_coarse_largest = _coarse_r123_ledger(
                occupied, virtual, core, part2, part3
            )
            coarse_largest = max(coarse_largest, block_coarse_largest)
            if coarse_ledger is None:
                coarse_ledger = {
                    key: xp.zeros_like(value)
                    for key, value in block_coarse.items()
                }
            for key, value in block_coarse.items():
                coarse_ledger[key] += value
        sigma1 += part1
        sigma2 += part2
        sigma3 += part3
        largest = max(
            largest,
            transformed.storage_nbytes,
            int(part1.nbytes),
            int(part2.nbytes),
            int(part3.nbytes),
        )
    total = sigma1 + sigma2 + sigma3
    coarse_retained = (
        0
        if coarse_ledger is None
        else sum(int(value.nbytes) for value in coarse_ledger.values())
    )
    return THCResidual123Result(
        sigma_thc=total,
        algorithm_1=sigma1,
        algorithm_2=sigma2,
        algorithm_3=sigma3,
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=largest,
        algorithm_1_provenance=provenance1,
        algorithm_3_provenance=provenance3,
        provenance_largest_intermediate_nbytes=provenance_largest,
        coarse_ledger=coarse_ledger,
        coarse_ledger_largest_intermediate_nbytes=coarse_largest,
        coarse_ledger_retained_nbytes=coarse_retained,
    )


def project_thc_residual_to_rr(sigma_thc: Any, tau: Any):
    """Transform a THC residual into the semi-unitary RR core basis."""

    if not _same_backend(sigma_thc, tau):
        raise TypeError("THC residual and tau need one array backend")
    _require_real_floating(sigma_thc, tau)
    if (
        getattr(sigma_thc, "ndim", None) != 2
        or sigma_thc.shape[0] != sigma_thc.shape[1]
    ):
        raise ValueError("sigma_thc must be a square matrix")
    if getattr(tau, "ndim", None) != 2 or tau.shape[1] != sigma_thc.shape[0]:
        raise ValueError("tau must have shape (rr_rank,thc_rank)")
    return tau @ sigma_thc @ tau.T


__all__ = [
    "T1TransformedCholesky",
    "THCResidual123Result",
    "PairFactorResidual123Result",
    "transform_cholesky_t1",
    "thc_algorithm_1",
    "thc_algorithm_2",
    "thc_algorithm_3",
    "thc_residual_algorithms_1_3",
    "pair_factor_residual_algorithms_1_3",
    "rr_residual_algorithms_1_3",
    "project_thc_residual_to_rr",
]
