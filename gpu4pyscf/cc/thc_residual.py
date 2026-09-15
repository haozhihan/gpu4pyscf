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

    def to_rr(self, tau: Any):
        """Return ``tau @ sigma_thc @ tau.T`` in the RR working basis."""

        return project_thc_residual_to_rr(self.sigma_thc, tau)

    def metadata(self) -> dict[str, Any]:
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

    def metadata(self) -> dict[str, Any]:
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


def _pair_factor_algorithms_1_3_block(
    pair_factor: Any,
    core: Any,
    transformed: T1TransformedCholesky,
    nocc: int,
    nvir: int,
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
    return part1, part2, part3, largest


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
) -> PairFactorResidual123Result:
    """Evaluate the exact Algorithms 1--3 group for a general pair factor.

    Cholesky vectors are streamed and neither ``t2[i,j,a,b]`` nor a four-index
    ERI is constructed.  This routine is the subtraction side of the hybrid
    identity ``R_hybrid = R_RR - R_123[U] + R_123[U_THC]``.
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
    largest = 0
    for start in range(0, naux, block_size):
        stop = min(start + block_size, naux)
        transformed = transform_cholesky_t1(
            t1, loo[start:stop], lov[start:stop], lvv[start:stop]
        )
        part1, part2, part3, block_largest = (
            _pair_factor_algorithms_1_3_block(
                pair_factor, core, transformed, nocc, nvir
            )
        )
        sigma1 += part1
        sigma2 += part2
        sigma3 += part3
        largest = max(
            largest,
            transformed.storage_nbytes,
            block_largest,
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
    )


def rr_residual_algorithms_1_3(
    doubles: Any,
    t1: Any,
    loo: Any,
    lov: Any,
    lvv: Any,
    *,
    auxiliary_block_size: Optional[int] = None,
) -> PairFactorResidual123Result:
    """Return the exact RR-space contribution of paper Algorithms 1--3."""

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
) -> THCResidual123Result:
    """Stream CD blocks through Algorithms 1--3 and sum their residuals."""

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
    largest = 0
    for start in range(0, naux, block_size):
        stop = min(start + block_size, naux)
        transformed = transform_cholesky_t1(
            t1, loo[start:stop], lov[start:stop], lvv[start:stop]
        )
        part1 = thc_algorithm_1(y_occ, y_vir, core, transformed)
        part2 = thc_algorithm_2(y_occ, y_vir, core, transformed)
        part3 = thc_algorithm_3(y_occ, y_vir, core, transformed)
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
    return THCResidual123Result(
        sigma_thc=total,
        algorithm_1=sigma1,
        algorithm_2=sigma2,
        algorithm_3=sigma3,
        auxiliary_block_size=block_size,
        largest_intermediate_nbytes=largest,
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
