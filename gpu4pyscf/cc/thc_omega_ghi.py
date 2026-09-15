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

"""Audit-only THC contractions for published Algorithms 8--10.

The routines in this module reproduce Appendix Algorithms 8--10 and
Eqs. 39--43 of Hohenstein *et al.*, J. Chem. Phys. **156**, 054102
(2022), DOI 10.1063/5.0077770.  The corresponding equations in
arXiv:2111.11473v1 are numbered 38--42.  The routines cover the
``Omega-G/H`` singles terms, the ``xi_oo`` and ``xi_vv`` intermediates used
by ``Omega-E``, and the final ``Omega-I/J`` singles terms.

The Cholesky arrays and hatted Fock blocks are already T1 transformed when
they enter this module.  This endpoint deliberately does not manufacture
those transformations.  The final paper prints a Kronecker ``delta_XY`` in
Algorithm 8 line 8.  Expanding published Eq. 24 into Eq. 39 fixes that object
as the identity in the amplitude-THC auxiliary space even when the columns
of ``y`` are nonorthogonal.  The Eq. 10--14 metric instead orthogonalizes the
RR projector in its distinct ``P,Q`` space; it is not a replacement for this
``delta_XY``.  The hatted-Fock mapping to the complete residual remains a
separate composition gate.

Literal Algorithm 8 line 13 uses ``l_ij`` where published Eqs. 24 and 39
require ``l_ji`` once the T1 transform removes Cholesky-matrix symmetry.  The
published Eq. 39 result and the literal appendix result are therefore
retained separately.  Likewise, for a nonsymmetric amplitude core, literal
Algorithm 10 uses the core in the Coulomb-like term and its transpose in the
exchange-like term; it is published Eq. 43 only at the symmetric-core
endpoint.  These routines are algebraic audits only and are not wired to the
production THC-RR driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _array_module(value: Any):
    """Return the NumPy/CuPy namespace without converting ``value``."""

    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy" and isinstance(value, np.ndarray):
        return np
    if module == "cupy":
        import cupy

        if isinstance(value, cupy.ndarray):
            return cupy
    raise TypeError("THC Omega-G/H/E/I/J tensors must be NumPy or CuPy arrays")


def _one_backend_and_device(reference: Any, *values: Any):
    xp = _array_module(reference)
    if any(_array_module(value) is not xp for value in values):
        raise TypeError(
            "all THC Omega-G/H/E/I/J tensors must use one array backend"
        )
    if xp is not np:
        device_id = int(reference.device.id)
        if any(int(value.device.id) != device_id for value in values):
            raise TypeError(
                "all THC Omega-G/H/E/I/J CuPy tensors must use one device"
            )
    return xp


def _one_real_dtype(*values: Any) -> np.dtype:
    dtypes = tuple(np.dtype(value.dtype) for value in values)
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError(
            "all THC Omega-G/H/E/I/J tensors must have the same dtype"
        )
    dtype = dtypes[0]
    if np.issubdtype(dtype, np.complexfloating):
        raise NotImplementedError(
            "THC Algorithms 8-10 currently require real RHF"
        )
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(
            "THC Omega-G/H/E/I/J tensors must have a floating-point dtype"
        )
    return dtype


def _require_finite(
    xp: Any,
    *,
    transfer_counter: Any,
    operation: str,
    error_type: type[Exception] = ValueError,
    **values: Any,
) -> None:
    # A GPU fail-closed check reads one combined boolean.  Record that scalar
    # explicitly; no tensor payload is copied to the host.
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU THC Omega-G/H/E/I/J validation requires an explicit "
                "transfer_counter"
            )
        if not callable(getattr(transfer_counter, "record_d2h", None)):
            raise TypeError("transfer_counter must provide record_d2h")
    finite = xp.asarray(True)
    for value in values.values():
        finite = xp.logical_and(finite, xp.all(xp.isfinite(value)))
    if xp is not np:
        transfer_counter.record_d2h(int(finite.nbytes), operation=operation)
    if not bool(finite.item() if hasattr(finite, "item") else finite):
        names = ", ".join(values)
        raise error_type(f"non-finite values in one of: {names}")


def _positive_block_size(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("cholesky_block_size must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError("cholesky_block_size must be positive")
    return value


def _validate_amplitudes(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    *other_arrays: Any,
):
    xp = _one_backend_and_device(
        y_occ, y_vir, amplitude_core, *other_arrays
    )
    dtype = _one_real_dtype(y_occ, y_vir, amplitude_core, *other_arrays)
    if getattr(y_occ, "ndim", None) != 2:
        raise ValueError("y_occ must have shape (nocc,amplitude_rank)")
    if getattr(y_vir, "ndim", None) != 2:
        raise ValueError("y_vir must have shape (nvir,amplitude_rank)")
    if getattr(amplitude_core, "ndim", None) != 2:
        raise ValueError(
            "amplitude_core must have shape "
            "(amplitude_rank,amplitude_rank)"
        )
    nocc, rank = map(int, y_occ.shape)
    nvir, virtual_rank = map(int, y_vir.shape)
    if nocc < 1 or nvir < 1 or rank < 1:
        raise ValueError("THC Omega-G/H/E/I/J dimensions must be positive")
    if virtual_rank != rank:
        raise ValueError("y_occ and y_vir amplitude ranks do not match")
    if amplitude_core.shape != (rank, rank):
        raise ValueError("amplitude_core must be square with the factor rank")
    return xp, dtype, nocc, nvir, rank


def _audit_metadata(
    *,
    algorithm: int,
    equations: list[int],
    arxiv_v1_equations: list[int],
    output: str,
    backend: str,
    dtype: str,
) -> dict[str, Any]:
    return {
        "paper": "Hohenstein-2022",
        "paper_source": (
            "J. Chem. Phys. 156, 054102 (2022), "
            "DOI:10.1063/5.0077770"
        ),
        "paper_version": "version-of-record",
        "equation_numbering_authority": "published-version-of-record",
        "paper_algorithm": algorithm,
        "paper_equations": equations,
        "published_equations": equations,
        "arxiv_v1_source": "arXiv:2111.11473v1",
        "arxiv_v1_equations": arxiv_v1_equations,
        "output": output,
        "coordinate_space": (
            "orbital-singles" if algorithm != 9 else "amplitude-thc-auxiliary"
        ),
        "audit_only": True,
        "production_enabled": False,
        "performance_eligible": False,
        "gpu_performance_claim": False,
        "complete_ccsd_residual": False,
        "full_equation_ledger_uniquely_validated": False,
        "delta_xy_convention": (
            "published-algorithm8-line8-kronecker-identity"
        ),
        "delta_xy_convention_uniquely_validated": True,
        "delta_xy_primary_source_basis": (
            "version-of-record-Algorithm8-line8-and-Eqs24-39"
        ),
        "delta_xy_dense_oracle_scope": (
            "rank-at-least-3-nonorthogonal-y-symmetric-core-index-loop"
        ),
        "delta_xy_pair_gram_alternative_rejected": True,
        "delta_xy_tau_metric_scope": (
            "tau-pair_gram-tau_transpose-is-RR-projector-PQ-identity;"
            "it-does-not-replace-amplitude-THC-delta_XY"
        ),
        "delta_xy_dependency": {
            8: "direct-algorithm8-line8",
            9: "indirect-through-algorithm8-xi",
            10: "none",
        }[algorithm],
        "transformed_f_convention": (
            "caller-supplied-hatted-F-blocks-from-published-equations-42-43"
        ),
        "transformed_f_convention_uniquely_validated": False,
        "literal_equation_oracle_requires_symmetric_amplitude_core": True,
        "core_symmetry_enforced": False,
        "backend": backend,
        "dtype": dtype,
        "materializes_dense_t2": False,
        "materializes_four_index_eri": False,
        "implicit_host_tensor_transfer": False,
        "gpu_validation_requires_transfer_counter": True,
    }


@dataclass(frozen=True)
class T1TransformedCholeskyBlocks:
    """Caller-supplied T1-transformed Cholesky vectors for Algorithm 8.

    Arrays have shapes ``(nchol,nocc,nocc)``, ``(nchol,nvir,nvir)``, and
    ``(nchol,nocc,nvir)``.  Both orbital axes retain the particle/hole
    orientation used in Appendix Algorithm 8; no symmetry is assumed.
    """

    l_oo: Any
    l_vv: Any
    l_ov: Any
    transfer_counter: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        xp = _one_backend_and_device(self.l_oo, self.l_vv, self.l_ov)
        _one_real_dtype(self.l_oo, self.l_vv, self.l_ov)
        if getattr(self.l_oo, "ndim", None) != 3:
            raise ValueError("l_oo must have shape (nchol,nocc,nocc)")
        if getattr(self.l_vv, "ndim", None) != 3:
            raise ValueError("l_vv must have shape (nchol,nvir,nvir)")
        if getattr(self.l_ov, "ndim", None) != 3:
            raise ValueError("l_ov must have shape (nchol,nocc,nvir)")
        nchol, nocc, nocc_second = map(int, self.l_oo.shape)
        nvv_chol, nvir, nvir_second = map(int, self.l_vv.shape)
        nov_chol, nov_occ, nov_vir = map(int, self.l_ov.shape)
        if nchol < 1 or nocc < 1 or nvir < 1:
            raise ValueError("T1-transformed Cholesky dimensions must be positive")
        if nocc_second != nocc:
            raise ValueError("l_oo orbital dimensions must be square")
        if nvir_second != nvir:
            raise ValueError("l_vv orbital dimensions must be square")
        if (nvv_chol, nov_chol) != (nchol, nchol):
            raise ValueError("all Cholesky arrays must have the same nchol")
        if (nov_occ, nov_vir) != (nocc, nvir):
            raise ValueError("l_ov orbital dimensions do not match l_oo/l_vv")
        _require_finite(
            xp,
            transfer_counter=self.transfer_counter,
            operation="thc_omega_gh_cholesky_validation",
            l_oo=self.l_oo,
            l_vv=self.l_vv,
            l_ov=self.l_ov,
        )

    @property
    def nchol(self) -> int:
        return int(self.l_ov.shape[0])

    @property
    def nocc(self) -> int:
        return int(self.l_ov.shape[1])

    @property
    def nvir(self) -> int:
        return int(self.l_ov.shape[2])

    @property
    def storage_nbytes(self) -> int:
        return int(self.l_oo.nbytes + self.l_vv.nbytes + self.l_ov.nbytes)


@dataclass(frozen=True)
class THCOmegaGHAlgorithm8Result:
    """Published Eq. 39 singles and Eq. 40--41 intermediates."""

    omega_g: Any
    omega_h: Any
    singles_gh: Any
    appendix_line13_omega_h: Any
    appendix_line13_singles_gh: Any
    xi_oo: Any
    xi_vv: Any
    amplitude_rank: int
    nchol: int
    cholesky_block_size: int
    largest_intermediate_nbytes: int
    transfer_counter: Any = field(default=None, repr=False, compare=False)

    @property
    def sigma_singles(self):
        return self.singles_gh

    def metadata(self) -> dict[str, Any]:
        xp = _array_module(self.singles_gh)
        metadata = _audit_metadata(
            algorithm=8,
            equations=[39, 40, 41],
            arxiv_v1_equations=[38, 39, 40],
            output="Omega-G/H-singles-and-xi-oo-xi-vv",
            backend=xp.__name__,
            dtype=np.dtype(self.singles_gh.dtype).name,
        )
        metadata.update(
            {
                "nchol": int(self.nchol),
                "amplitude_rank": int(self.amplitude_rank),
                "cholesky_block_size": int(self.cholesky_block_size),
                "largest_intermediate_nbytes": int(
                    self.largest_intermediate_nbytes
                ),
                "cholesky_input_convention": (
                    "caller-supplied-T1-transformed-particle-hole-blocks"
                ),
                "cholesky_orbital_symmetry_enforced": False,
                "published_equation39_occupied_orientation": (
                    "l_ji-times-D_ja"
                ),
                "legacy_arxiv_v1_equation38_occupied_orientation": (
                    "l_ji-times-D_ja"
                ),
                "appendix_algorithm8_line13_orientation": "l_ij-times-D_ja",
                "algorithm8_line13_published_equation39_equivalence": False,
                "algorithm8_line13_status": "retained-separately-fail-closed",
                "algorithm8_line13_reason": (
                    "published Eqs. 24 and 39 require the transpose of the "
                    "literal line-13 occupied Cholesky block after the T1 "
                    "transform"
                ),
                "transformed_f_used": False,
                "actual_schedule": (
                    "blocked NumPy/CuPy published Eq. 39-41 implementation "
                    "plus a separate literal Appendix Algorithm 8 line-13 "
                    "audit"
                ),
            }
        )
        return metadata


@dataclass(frozen=True)
class THCOmegaEAlgorithm9Result:
    """Algorithm 9 raw amplitude-THC doubles-core contribution."""

    omega_e: Any
    amplitude_metric: Any
    one_body_metric: Any
    amplitude_rank: int
    largest_intermediate_nbytes: int
    transfer_counter: Any = field(default=None, repr=False, compare=False)

    @property
    def sigma_thc(self):
        return self.omega_e

    def metadata(self) -> dict[str, Any]:
        xp = _array_module(self.omega_e)
        metadata = _audit_metadata(
            algorithm=9,
            equations=[40, 41, 42],
            arxiv_v1_equations=[39, 40, 41],
            output="Omega-E-raw-amplitude-thc-core",
            backend=xp.__name__,
            dtype=np.dtype(self.omega_e.dtype).name,
        )
        metadata.update(
            {
                "amplitude_rank": int(self.amplitude_rank),
                "largest_intermediate_nbytes": int(
                    self.largest_intermediate_nbytes
                ),
                "consumes_algorithm8_xi_only": True,
                "consumes_algorithm8_singles": False,
                "fhat_block_shapes": ["oo", "vv"],
                "requires_rr_back_projection": True,
                "applies_published_equation42_core_symmetrization": True,
                "actual_schedule": (
                    "NumPy/CuPy matrix products in Appendix Algorithm 9 order"
                ),
            }
        )
        return metadata


@dataclass(frozen=True)
class THCOmegaIJAlgorithm10Result:
    """Separated Algorithm 10 schedule contributions to singles."""

    direct_fhat_vo: Any
    coulomb_like: Any
    exchange_like: Any
    singles_ij: Any
    amplitude_rank: int
    largest_intermediate_nbytes: int
    transfer_counter: Any = field(default=None, repr=False, compare=False)

    @property
    def sigma_singles(self):
        return self.singles_ij

    def metadata(self) -> dict[str, Any]:
        xp = _array_module(self.singles_ij)
        metadata = _audit_metadata(
            algorithm=10,
            equations=[43],
            arxiv_v1_equations=[42],
            output="Omega-I/J-singles",
            backend=xp.__name__,
            dtype=np.dtype(self.singles_ij.dtype).name,
        )
        metadata.update(
            {
                "amplitude_rank": int(self.amplitude_rank),
                "largest_intermediate_nbytes": int(
                    self.largest_intermediate_nbytes
                ),
                "fhat_block_shapes": ["ov", "vo"],
                "direct_fhat_vo_included": True,
                "published_equation43_equivalence_requires_symmetric_"
                "amplitude_core": True,
                "published_equation43_symmetric_core_endpoint_validated": True,
                "published_equation43_nonsymmetric_core_equivalence": False,
                "appendix_algorithm10_nonsymmetric_core_mapping": (
                    "published-eq43-coulomb-uses-T-exchange-uses-T-transpose"
                ),
                "nonsymmetric_amplitude_core_status": (
                    "appendix-literal-schedule-audit-only-not-production-eligible"
                ),
                "nonsymmetric_amplitude_core_production_eligible": False,
                "actual_schedule": (
                    "NumPy/CuPy matrix products in Appendix Algorithm 10 order"
                ),
            }
        )
        return metadata


def thc_omega_gh_algorithm8(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    cholesky: T1TransformedCholeskyBlocks,
    *,
    cholesky_block_size: int = 1,
    transfer_counter: Any = None,
) -> THCOmegaGHAlgorithm8Result:
    """Evaluate Appendix Algorithm 8 with T1-transformed Cholesky blocks."""

    if not isinstance(cholesky, T1TransformedCholeskyBlocks):
        raise TypeError(
            "cholesky must be a T1TransformedCholeskyBlocks instance"
        )
    cholesky_block_size = _positive_block_size(cholesky_block_size)
    if transfer_counter is None:
        transfer_counter = cholesky.transfer_counter
    xp, dtype, nocc, nvir, rank = _validate_amplitudes(
        y_occ,
        y_vir,
        amplitude_core,
        cholesky.l_oo,
        cholesky.l_vv,
        cholesky.l_ov,
    )
    if (nocc, nvir) != (cholesky.nocc, cholesky.nvir):
        raise ValueError(
            "amplitude and T1-transformed Cholesky orbital dimensions "
            "do not match"
        )
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="thc_omega_gh_input_validation",
        y_occ=y_occ,
        y_vir=y_vir,
        amplitude_core=amplitude_core,
        l_oo=cholesky.l_oo,
        l_vv=cholesky.l_vv,
        l_ov=cholesky.l_ov,
    )

    omega_g = xp.zeros((nocc, nvir), dtype=dtype)
    omega_h = xp.zeros_like(omega_g)
    appendix_line13_omega_h = xp.zeros_like(omega_g)
    xi_oo = xp.zeros((nocc, nocc), dtype=dtype)
    xi_vv = xp.zeros((nvir, nvir), dtype=dtype)
    delta_xy = xp.eye(rank, dtype=dtype)
    largest_intermediate_nbytes = max(
        int(omega_g.nbytes),
        int(xi_oo.nbytes),
        int(xi_vv.nbytes),
        int(delta_xy.nbytes),
    )
    two = dtype.type(2)

    for a0 in range(0, cholesky.nchol, cholesky_block_size):
        a1 = min(a0 + cholesky_block_size, cholesky.nchol)
        l_oo = cholesky.l_oo[a0:a1]
        l_vv = cholesky.l_vv[a0:a1]
        l_ov = cholesky.l_ov[a0:a1]

        # Algorithm 8, lines 6--9.  ``delta_xy`` is the Kronecker identity
        # printed on line 8.  Direct expansion of published Eq. 24 into
        # Eq. 39 retains this identity for nonorthogonal y columns.  The
        # Eq. 10--14 tau/pair metric acts in the distinct RR-projector P,Q
        # space and must not be substituted here.
        a_ax = xp.einsum("Aia,iX,aX->AX", l_ov, y_occ, y_vir)
        b_axy = xp.einsum("Aia,iX,aY->AXY", l_ov, y_occ, y_vir)
        diagonal_ax = a_ax @ amplitude_core.T
        c_axy = -(b_axy * amplitude_core[None, :, :])
        c_axy += (
            two * diagonal_ax[:, :, None] * delta_xy[None, :, :]
        )
        d_aia = xp.einsum("iY,aX,AXY->Aia", y_occ, y_vir, c_axy)

        # Lines 10--12 agree with published Eqs. 39--41 as printed.
        xi_oo += xp.einsum("Aia,Aja->ij", l_ov, d_aia)
        xi_vv -= xp.einsum("Aia,Aib->ab", d_aia, l_ov)
        omega_g += xp.einsum("Aib,Aab->ia", d_aia, l_vv)

        # Published Eq. 24 gives (l c-hat|k i) = Lhat_lc Lhat_ki.
        # Consequently the occupied part of published Eq. 39 requires l_ji
        # below.  Literal Appendix Algorithm 8 line 13 instead prints l_ij;
        # retain it separately so
        # nonsymmetric T1-transformed blocks expose, rather than hide, the
        # discrepancy.  This is the same kind of transpose boundary already
        # documented for Appendix Algorithm 1 in ``thc_residual.py``.
        omega_h -= xp.einsum("Aji,Aja->ia", l_oo, d_aia)
        appendix_line13_omega_h -= xp.einsum(
            "Aij,Aja->ia", l_oo, d_aia
        )
        largest_intermediate_nbytes = max(
            largest_intermediate_nbytes,
            int(a_ax.nbytes),
            int(b_axy.nbytes),
            int(diagonal_ax.nbytes),
            int(c_axy.nbytes),
            int(d_aia.nbytes),
        )

    singles_gh = omega_g + omega_h
    appendix_line13_singles_gh = omega_g + appendix_line13_omega_h
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="thc_omega_gh_output_validation",
        error_type=FloatingPointError,
        omega_g=omega_g,
        omega_h=omega_h,
        singles_gh=singles_gh,
        appendix_line13_omega_h=appendix_line13_omega_h,
        appendix_line13_singles_gh=appendix_line13_singles_gh,
        xi_oo=xi_oo,
        xi_vv=xi_vv,
    )
    return THCOmegaGHAlgorithm8Result(
        omega_g=omega_g,
        omega_h=omega_h,
        singles_gh=singles_gh,
        appendix_line13_omega_h=appendix_line13_omega_h,
        appendix_line13_singles_gh=appendix_line13_singles_gh,
        xi_oo=xi_oo,
        xi_vv=xi_vv,
        amplitude_rank=rank,
        nchol=cholesky.nchol,
        cholesky_block_size=cholesky_block_size,
        largest_intermediate_nbytes=largest_intermediate_nbytes,
        transfer_counter=transfer_counter,
    )


def thc_omega_e_algorithm9(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    fhat_oo: Any,
    fhat_vv: Any,
    algorithm8: THCOmegaGHAlgorithm8Result,
    *,
    transfer_counter: Any = None,
) -> THCOmegaEAlgorithm9Result:
    """Evaluate Algorithm 9 using only Algorithm 8's ``xi`` matrices."""

    if not isinstance(algorithm8, THCOmegaGHAlgorithm8Result):
        raise TypeError(
            "algorithm8 must be a THCOmegaGHAlgorithm8Result instance"
        )
    if transfer_counter is None:
        transfer_counter = algorithm8.transfer_counter
    xp, dtype, nocc, nvir, rank = _validate_amplitudes(
        y_occ,
        y_vir,
        amplitude_core,
        fhat_oo,
        fhat_vv,
        algorithm8.xi_oo,
        algorithm8.xi_vv,
    )
    if getattr(fhat_oo, "ndim", None) != 2 or fhat_oo.shape != (nocc, nocc):
        raise ValueError("fhat_oo must have shape (nocc,nocc)")
    if getattr(fhat_vv, "ndim", None) != 2 or fhat_vv.shape != (nvir, nvir):
        raise ValueError("fhat_vv must have shape (nvir,nvir)")
    if algorithm8.xi_oo.shape != (nocc, nocc):
        raise ValueError("Algorithm 8 xi_oo orbital dimensions do not match")
    if algorithm8.xi_vv.shape != (nvir, nvir):
        raise ValueError("Algorithm 8 xi_vv orbital dimensions do not match")
    if algorithm8.amplitude_rank != rank:
        raise ValueError("Algorithm 8 amplitude rank does not match")
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="thc_omega_e_input_validation",
        y_occ=y_occ,
        y_vir=y_vir,
        amplitude_core=amplitude_core,
        fhat_oo=fhat_oo,
        fhat_vv=fhat_vv,
        xi_oo=algorithm8.xi_oo,
        xi_vv=algorithm8.xi_vv,
    )

    # Algorithm 9, lines 1--4.
    overlap_occ = y_occ.T @ y_occ
    overlap_vir = y_vir.T @ y_vir
    pair_overlap = overlap_occ * overlap_vir
    amplitude_metric = pair_overlap @ amplitude_core

    # Lines 5--9.  The occupied transpose is literal: C_i^X contains
    # (F_ji + xi_ji) y_j^X.
    c_ix = (fhat_oo + algorithm8.xi_oo).T @ y_occ
    d_ax = (fhat_vv + algorithm8.xi_vv) @ y_vir
    dressed_occ = y_occ.T @ c_ix
    dressed_vir = y_vir.T @ d_ax
    one_body_metric = (
        overlap_occ * dressed_vir - dressed_occ * overlap_vir
    )

    # Line 10 is the symmetric projected published Eq. 42 contribution.
    omega_e = (
        one_body_metric @ amplitude_metric.T
        + amplitude_metric @ one_body_metric.T
    )
    largest_intermediate_nbytes = max(
        int(overlap_occ.nbytes),
        int(overlap_vir.nbytes),
        int(pair_overlap.nbytes),
        int(amplitude_metric.nbytes),
        int(c_ix.nbytes),
        int(d_ax.nbytes),
        int(dressed_occ.nbytes),
        int(dressed_vir.nbytes),
        int(one_body_metric.nbytes),
        int(omega_e.nbytes),
    )
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="thc_omega_e_output_validation",
        error_type=FloatingPointError,
        amplitude_metric=amplitude_metric,
        one_body_metric=one_body_metric,
        omega_e=omega_e,
    )
    return THCOmegaEAlgorithm9Result(
        omega_e=omega_e,
        amplitude_metric=amplitude_metric,
        one_body_metric=one_body_metric,
        amplitude_rank=rank,
        largest_intermediate_nbytes=largest_intermediate_nbytes,
        transfer_counter=transfer_counter,
    )


def thc_omega_ij_algorithm10(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    fhat_ov: Any,
    fhat_vo: Any,
    *,
    transfer_counter: Any = None,
) -> THCOmegaIJAlgorithm10Result:
    """Evaluate the Algorithm 10 ``Omega-I/J`` singles contribution.

    The result is formally identified with published Eq. 43 only for a
    symmetric amplitude core.  For a nonsymmetric core, the literal Appendix
    schedule uses ``T`` in the Coulomb-like term and ``T.T`` in the
    exchange-like term.  A nonsymmetric core remains useful as a transpose
    test, but that mixed expression is audit-only and is not production
    eligible.
    """

    xp, dtype, nocc, nvir, rank = _validate_amplitudes(
        y_occ, y_vir, amplitude_core, fhat_ov, fhat_vo
    )
    if getattr(fhat_ov, "ndim", None) != 2 or fhat_ov.shape != (nocc, nvir):
        raise ValueError("fhat_ov must have shape (nocc,nvir)")
    if getattr(fhat_vo, "ndim", None) != 2 or fhat_vo.shape != (nvir, nocc):
        raise ValueError("fhat_vo must have shape (nvir,nocc)")
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="thc_omega_ij_input_validation",
        y_occ=y_occ,
        y_vir=y_vir,
        amplitude_core=amplitude_core,
        fhat_ov=fhat_ov,
        fhat_vo=fhat_vo,
    )

    # Algorithm 10, lines 1--5.
    a_ix = fhat_ov @ y_vir
    b_xy = y_occ.T @ a_ix
    c_x = amplitude_core @ xp.diag(b_xy)
    d_ix = y_occ * c_x[None, :]
    two = dtype.type(2)
    coulomb_like = two * (d_ix @ y_vir.T)

    # Lines 6--8.
    e_xy = amplitude_core * b_xy
    f_ix = y_occ @ e_xy.T
    exchange_like = -(f_ix @ y_vir.T)

    # Line 10 is the explicit hatted F_ai addend.
    direct_fhat_vo = fhat_vo.T
    singles_ij = direct_fhat_vo + coulomb_like + exchange_like
    largest_intermediate_nbytes = max(
        int(a_ix.nbytes),
        int(b_xy.nbytes),
        int(c_x.nbytes),
        int(d_ix.nbytes),
        int(coulomb_like.nbytes),
        int(e_xy.nbytes),
        int(f_ix.nbytes),
        int(exchange_like.nbytes),
        int(singles_ij.nbytes),
    )
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="thc_omega_ij_output_validation",
        error_type=FloatingPointError,
        direct_fhat_vo=direct_fhat_vo,
        coulomb_like=coulomb_like,
        exchange_like=exchange_like,
        singles_ij=singles_ij,
    )
    return THCOmegaIJAlgorithm10Result(
        direct_fhat_vo=direct_fhat_vo,
        coulomb_like=coulomb_like,
        exchange_like=exchange_like,
        singles_ij=singles_ij,
        amplitude_rank=rank,
        largest_intermediate_nbytes=largest_intermediate_nbytes,
        transfer_counter=transfer_counter,
    )
