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

"""Audit-only ERI-THC contractions for paper Algorithms 4--6.

This module implements the raw projected contributions in Eqs. 32 and 33 of
Hohenstein *et al.*, J. Chem. Phys. **156**, 054102 (2022), DOI
10.1063/5.0077770 (arXiv:2111.11473v1).  The corresponding factorized
contraction orders are Appendix Algorithms 4--6.  Algorithm 6 evaluates the
sum of Eqs. 32 and 33 (Eq. 34) through shared three-index intermediates.

The amplitude representation is

``t[i,j,a,b] = sum(X,Y) y_occ[i,X] y_vir[a,X] T[X,Y]``
``                         * y_occ[j,Y] y_vir[b,Y]``

and the only ERI class needed here is the Eq. 17 representation

``(i a | j b) = sum(I,J) x_occ[i,I] x_vir[a,I] Z[I,J]``
``                         * x_occ[j,J] x_vir[b,J]``.

The routines deliberately do not apply a residual prefactor, permutation, or
symmetrization.  They are correctness/audit endpoints and are not connected to
the production THC-RR driver.  Their output is in the amplitude-THC auxiliary
coordinates, not the semi-unitary RR working coordinates.
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
    raise TypeError("ERI-THC tensors must be NumPy or CuPy arrays")


def _one_array_backend(reference: Any, *values: Any):
    xp = _array_module(reference)
    if any(_array_module(value) is not xp for value in values):
        raise TypeError("all ERI-THC tensors must use one array backend")
    if xp is not np:
        device_id = int(reference.device.id)
        if any(int(value.device.id) != device_id for value in values):
            raise TypeError("all ERI-THC CuPy tensors must use one device")
    return xp


def _require_one_real_dtype(*values: Any) -> np.dtype:
    dtypes = tuple(np.dtype(value.dtype) for value in values)
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError("all ERI-THC tensors must have the same dtype")
    dtype = dtypes[0]
    if np.issubdtype(dtype, np.complexfloating):
        raise NotImplementedError("ERI-THC Algorithms 4-6 currently require real RHF")
    if not np.issubdtype(dtype, np.floating):
        raise TypeError("ERI-THC tensors must have a floating-point dtype")
    return dtype


def _require_finite(
    xp: Any,
    *,
    transfer_counter: Any,
    operation: str,
    **values: Any,
) -> None:
    # On a GPU, fail-closed finite-value validation necessarily reads one
    # combined control scalar.  No tensor payload is copied to the host.
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU ERI-THC validation requires an explicit transfer_counter"
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
        raise ValueError(f"non-finite values in one of: {names}")


def _positive_block_size(
    value: Any, *, name: str = "outer_block_size"
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_eri_arrays(
    x_occ: Any,
    x_vir: Any,
    core: Any,
    *,
    transfer_counter: Any = None,
):
    xp = _one_array_backend(x_occ, x_vir, core)
    dtype = _require_one_real_dtype(x_occ, x_vir, core)
    if getattr(x_occ, "ndim", None) != 2:
        raise ValueError("x_occ must have shape (nocc,eri_rank)")
    if getattr(x_vir, "ndim", None) != 2:
        raise ValueError("x_vir must have shape (nvir,eri_rank)")
    if getattr(core, "ndim", None) != 2:
        raise ValueError("ERI core must have shape (eri_rank,eri_rank)")
    nocc, eri_rank = map(int, x_occ.shape)
    nvir, virtual_rank = map(int, x_vir.shape)
    if nocc < 1 or nvir < 1 or eri_rank < 1:
        raise ValueError("ERI-THC dimensions must be positive")
    if virtual_rank != eri_rank:
        raise ValueError("x_occ and x_vir ERI ranks do not match")
    if core.shape != (eri_rank, eri_rank):
        raise ValueError("ERI core must be square with the factor rank")
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="eri_thc_factor_validation",
        x_occ=x_occ,
        x_vir=x_vir,
        eri_core=core,
    )
    return xp, dtype, nocc, nvir, eri_rank


@dataclass(frozen=True)
class ERITHCFactors:
    """Eq. 17 THC factors for the ``ovov`` ERI class.

    ``x_occ`` and ``x_vir`` have shapes ``(O,Q)`` and ``(V,Q)``.  ``core``
    has shape ``(Q,Q)``.  The amplitude THC rank ``N_thc`` is independent of
    the ERI THC rank ``Q``.  CuPy factors require ``transfer_counter`` so the
    scalar reads used by fail-closed validation are explicit and recorded.
    """

    x_occ: Any
    x_vir: Any
    core: Any
    transfer_counter: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_eri_arrays(
            self.x_occ,
            self.x_vir,
            self.core,
            transfer_counter=self.transfer_counter,
        )

    @property
    def nocc(self) -> int:
        return int(self.x_occ.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.x_vir.shape[0])

    @property
    def rank(self) -> int:
        return int(self.x_occ.shape[1])

    @property
    def storage_nbytes(self) -> int:
        return int(self.x_occ.nbytes + self.x_vir.nbytes + self.core.nbytes)

    def reconstruct_ovov(self):
        """Materialize ``(i a | j b)`` for small correctness problems."""

        xp, *_ = _validate_eri_arrays(
            self.x_occ,
            self.x_vir,
            self.core,
            transfer_counter=self.transfer_counter,
        )
        return xp.einsum(
            "iI,aI,IJ,jJ,bJ->iajb",
            self.x_occ,
            self.x_vir,
            self.core,
            self.x_occ,
            self.x_vir,
        )

    def metadata(self) -> dict[str, Any]:
        xp = _array_module(self.x_occ)
        return {
            "representation": "eri-thc-ovov",
            "paper": "Hohenstein-2022",
            "paper_source": "arXiv:2111.11473v1",
            "eri_equation": 17,
            "residual_equations": [32, 33, 34],
            "algorithms": [4, 5, 6],
            "nocc": self.nocc,
            "nvir": self.nvir,
            "eri_rank": self.rank,
            "backend": xp.__name__,
            "dtype": np.dtype(self.x_occ.dtype).name,
            "storage_nbytes": self.storage_nbytes,
            "audit_only": True,
            "production_enabled": False,
            "performance_eligible": False,
            "complete_ccsd_residual": False,
            "output_coordinate_space": "amplitude-thc-auxiliary",
            "requires_rr_back_projection": True,
            "applies_prefactor": False,
            "applies_permutation_or_symmetrization": False,
            "contribution_convention": "raw-projected-paper-equation",
            "algorithm_materializes_dense_t2": False,
            "algorithm_materializes_four_index_eri": False,
            "dense_reference_reconstruction_available": True,
            "core_symmetry_enforced": False,
            "gpu_validation_requires_transfer_counter": True,
            "implicit_host_tensor_transfer": False,
            "algorithm6_custom_cuda_kernel": False,
            "algorithm6_python_gemm_audit_endpoint": True,
        }


def _validate_problem(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    eri_factors: ERITHCFactors,
    transfer_counter: Any,
):
    if not isinstance(eri_factors, ERITHCFactors):
        raise TypeError("eri_factors must be an ERITHCFactors instance")
    xp = _one_array_backend(
        y_occ,
        y_vir,
        amplitude_core,
        eri_factors.x_occ,
        eri_factors.x_vir,
        eri_factors.core,
    )
    dtype = _require_one_real_dtype(
        y_occ,
        y_vir,
        amplitude_core,
        eri_factors.x_occ,
        eri_factors.x_vir,
        eri_factors.core,
    )
    if getattr(y_occ, "ndim", None) != 2:
        raise ValueError("y_occ must have shape (nocc,amplitude_rank)")
    if getattr(y_vir, "ndim", None) != 2:
        raise ValueError("y_vir must have shape (nvir,amplitude_rank)")
    if getattr(amplitude_core, "ndim", None) != 2:
        raise ValueError(
            "amplitude_core must have shape (amplitude_rank,amplitude_rank)"
        )
    nocc, rank = map(int, y_occ.shape)
    nvir, virtual_rank = map(int, y_vir.shape)
    if nocc < 1 or nvir < 1 or rank < 1:
        raise ValueError("amplitude THC dimensions must be positive")
    if virtual_rank != rank:
        raise ValueError("y_occ and y_vir amplitude ranks do not match")
    if amplitude_core.shape != (rank, rank):
        raise ValueError("amplitude_core must be square with the factor rank")
    if (nocc, nvir) != (eri_factors.nocc, eri_factors.nvir):
        raise ValueError("amplitude and ERI orbital dimensions do not match")
    # Revalidate the factor arrays because a frozen dataclass does not make
    # the underlying mutable arrays immutable.
    _validate_eri_arrays(
        eri_factors.x_occ,
        eri_factors.x_vir,
        eri_factors.core,
        transfer_counter=transfer_counter,
    )
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="eri_thc_amplitude_validation",
        y_occ=y_occ,
        y_vir=y_vir,
        amplitude_core=amplitude_core,
    )
    return xp, dtype, nocc, nvir, rank


def _validate_output(
    xp: Any,
    result: Any,
    name: str,
    transfer_counter: Any,
) -> Any:
    finite = xp.all(xp.isfinite(result))
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU ERI-THC validation requires an explicit transfer_counter"
            )
        transfer_counter.record_d2h(
            int(finite.nbytes), operation="eri_thc_output_validation"
        )
    if not bool(finite.item() if hasattr(finite, "item") else finite):
        raise FloatingPointError(f"{name} produced non-finite values")
    return result


def thc_omega_a_algorithm4(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    eri_factors: ERITHCFactors,
    *,
    outer_block_size: int = 1,
    transfer_counter: Any = None,
):
    """Evaluate the raw projected Eq. 32 contribution (Algorithm 4).

    The result is an amplitude-THC core with shape ``(N_thc,N_thc)``; it is not
    an RR-space core.  Following the appendix pseudocode, the outer loop fixes
    ``(i,j)``.  Each named intermediate below is a matrix; neither dense
    doubles amplitudes nor four-index ERIs are materialized.
    ``outer_block_size`` groups outer ``i`` slices for future scheduling but
    does not change the paper contraction order.

    This raw Algorithm 4 contribution need not be symmetric for arbitrary
    factors.  A production residual assembler would have to apply every
    required coefficient/permutation and RR back-projection explicitly.
    """

    if not isinstance(eri_factors, ERITHCFactors):
        raise TypeError("eri_factors must be an ERITHCFactors instance")
    if transfer_counter is None:
        transfer_counter = eri_factors.transfer_counter
    xp, dtype, nocc, _nvir, rank = _validate_problem(
        y_occ, y_vir, amplitude_core, eri_factors, transfer_counter
    )
    outer_block_size = _positive_block_size(outer_block_size)
    x_occ = eri_factors.x_occ
    x_vir = eri_factors.x_vir
    eri_core = eri_factors.core
    sigma = xp.zeros((rank, rank), dtype=dtype)

    for i0 in range(0, nocc, outer_block_size):
        for i in range(i0, min(i0 + outer_block_size, nocc)):
            yi = y_occ[i]
            for j in range(nocc):
                # Appendix Algorithm 4, lines 3--5: reconstruct t_ij^(cd).
                a_zz = yi[:, None] * y_occ[j][None, :]  # (N_thc,N_thc)
                b_zz = amplitude_core * a_zz  # (N_thc,N_thc)
                c_cd = y_vir @ b_zz @ y_vir.T  # (V,V)

                # Lines 6--8: sum_cd t_ij^(cd) (k c | l d).
                d_ij = x_vir.T @ c_cd @ x_vir  # (Q,Q)
                e_ij = eri_core * d_ij  # (Q,Q)
                f_kl = x_occ @ e_ij @ x_occ.T  # (O,O)

                # Lines 9--11: contract t_kl^(ab).
                g_yy = y_occ.T @ f_kl @ y_occ  # (N_thc,N_thc)
                h_yy = amplitude_core * g_yy  # (N_thc,N_thc)
                i_ab = y_vir @ h_yy @ y_vir.T  # (V,V)

                # Lines 12--13: project on U_ia^X U_jb^X'.
                j_xx = y_vir.T @ i_ab @ y_vir  # (N_thc,N_thc)
                sigma += j_xx * a_zz

    return _validate_output(
        xp, sigma, "THC Algorithm 4", transfer_counter
    )


def thc_omega_c_algorithm5(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    eri_factors: ERITHCFactors,
    *,
    outer_block_size: int = 1,
    transfer_counter: Any = None,
):
    """Evaluate the raw projected Eq. 33 contribution (Algorithm 5).

    The result is an amplitude-THC core with shape ``(N_thc,N_thc)``; it is not
    an RR-space core.  The appendix contraction fixes ``(j,a)`` and forms only
    matrix slices.  In particular, ``F[k,c]`` uses ``x_vir[c,I] Z[I,J]
    x_occ[k,J]``; retaining that orientation is required when the ERI core is
    nonsymmetric.

    This raw Algorithm 5 contribution need not be symmetric for arbitrary
    factors.  A production residual assembler would have to apply every
    required coefficient/permutation and RR back-projection explicitly.
    """

    if not isinstance(eri_factors, ERITHCFactors):
        raise TypeError("eri_factors must be an ERITHCFactors instance")
    if transfer_counter is None:
        transfer_counter = eri_factors.transfer_counter
    xp, dtype, nocc, nvir, rank = _validate_problem(
        y_occ, y_vir, amplitude_core, eri_factors, transfer_counter
    )
    outer_block_size = _positive_block_size(outer_block_size)
    x_occ = eri_factors.x_occ
    x_vir = eri_factors.x_vir
    eri_core = eri_factors.core
    sigma = xp.zeros((rank, rank), dtype=dtype)

    for j0 in range(0, nocc, outer_block_size):
        for j in range(j0, min(j0 + outer_block_size, nocc)):
            yj = y_occ[j]
            for a in range(nvir):
                # Appendix Algorithm 5, lines 3--5: reconstruct t_lj^(ad).
                a_zz = y_vir[a][:, None] * yj[None, :]  # (N_thc,N_thc)
                b_zz = amplitude_core * a_zz  # (N_thc,N_thc)
                c_ld = y_occ @ b_zz @ y_vir.T  # (O,V)

                # Lines 6--8: contract t_lj^(ad) (l c | k d).
                d_ij = x_occ.T @ c_ld @ x_vir  # (Q,Q)
                e_ij = eri_core * d_ij  # (Q,Q)
                f_kc = x_occ @ e_ij.T @ x_vir.T  # (O,V)

                # Lines 9--11: contract t_ik^(cb).
                g_yy = y_vir.T @ f_kc.T @ y_occ  # (N_thc,N_thc)
                h_yy = amplitude_core * g_yy  # (N_thc,N_thc)
                i_ib = y_occ @ h_yy @ y_vir.T  # (O,V)

                # Lines 12--13: project on U_ia^X U_jb^X'.
                j_xx = y_occ.T @ i_ib @ y_vir  # (N_thc,N_thc)
                sigma += j_xx * a_zz

    return _validate_output(
        xp, sigma, "THC Algorithm 5", transfer_counter
    )


def thc_omega_ac_algorithm6_metadata(
    *, x_block_size: int = 1
) -> dict[str, Any]:
    """Describe the actual scheduling contract of the Algorithm 6 endpoint."""

    x_block_size = _positive_block_size(
        x_block_size, name="x_block_size"
    )
    return {
        "paper": "Hohenstein-2022",
        "paper_source": "arXiv:2111.11473v1",
        "paper_algorithm": 6,
        "paper_equations": [32, 33, 34],
        "output": "raw-amplitude-thc-core",
        "output_coordinate_space": "amplitude-thc-auxiliary",
        "requires_rr_back_projection": True,
        "complete_ccsd_residual": False,
        "contribution_convention": "raw-projected-paper-equation",
        "applies_prefactor": False,
        "applies_permutation_or_symmetrization": False,
        "audit_only": True,
        "production_enabled": False,
        "performance_eligible": False,
        "formal_scaling_with_linear_ranks": "O(N^4)",
        "actual_schedule": (
            "Python loops over X blocks and ERI-THC rank; NumPy/CuPy "
            "einsum, GEMM, and unfused elementwise Algorithm-6 line 11"
        ),
        "custom_cuda_line11": False,
        "gpu_performance_claim": False,
        "x_block_size": x_block_size,
        "x_blocked_intermediates": ["C", "D", "H", "I"],
        "full_three_index_A_materialized": True,
        "full_three_index_B_materialized": False,
        "eri_B_scheduling": "one I slice at a time",
        "materializes_dense_t2": False,
        "materializes_four_index_eri": False,
        "implicit_host_tensor_transfer": False,
        "gpu_inputs_must_be_same_device_resident": True,
        "gpu_validation_scalar_reads_recorded": True,
        "core_symmetry_enforced": False,
        "production_requirements": [
            "fused/custom CUDA kernel for rate-limiting line 11",
            "A/H workspace plan validated against the HBM budget",
            "qualified A100 end-to-end timing",
        ],
    }


def thc_omega_ac_algorithm6(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    eri_factors: ERITHCFactors,
    *,
    x_block_size: int = 1,
    transfer_counter: Any = None,
):
    """Evaluate the raw joint Eq. 34 contribution (Appendix Algorithm 6).

    This is a correctness endpoint for the paper's joint ``Omega-A`` and
    ``Omega-C`` contraction.  It returns an ``(N_thc,N_thc)`` raw core in the
    amplitude-THC coordinates.  No residual prefactor, permutation,
    symmetrization, or RR back-projection is applied.

    The implementation follows the appendix index directions exactly:

    ``A[X,i,a] = sum(Y) y_occ[i,Y] y_vir[a,Y] T[Y,X]``

    ``B[I,i,a] = sum(J) x_occ[i,J] x_vir[a,J] Z[I,J]``.

    In particular, neither core is assumed symmetric.  The first and second
    products accumulated into ``H[X,Y,Z]`` reproduce Eqs. 32 and 33,
    respectively, for nonsymmetric test cores.

    ``x_block_size`` bounds the first THC dimension of the ``C``, ``D``,
    ``H``, and ``I`` intermediates.  The complete three-index ``A`` bridge is
    materialized, while ``B`` is formed one ERI-THC slice at a time.  Dense
    doubles amplitudes and four-index ERIs are never formed.

    The paper uses a custom CUDA kernel for the rate-limiting line 11.  This
    audit implementation instead uses Python scheduling plus NumPy/CuPy
    GEMMs and unfused elementwise operations.  It therefore makes no GPU
    performance claim and is not eligible for production timing evidence.
    """

    if not isinstance(eri_factors, ERITHCFactors):
        raise TypeError("eri_factors must be an ERITHCFactors instance")
    if transfer_counter is None:
        transfer_counter = eri_factors.transfer_counter
    xp, dtype, nocc, nvir, rank = _validate_problem(
        y_occ, y_vir, amplitude_core, eri_factors, transfer_counter
    )
    x_block_size = _positive_block_size(
        x_block_size, name="x_block_size"
    )
    x_occ = eri_factors.x_occ
    x_vir = eri_factors.x_vir
    eri_core = eri_factors.core

    # Appendix Algorithm 6, line 1.  ``T[Y,X]`` is deliberate: using
    # ``T[X,Y]`` is only equivalent when the amplitude core is symmetric.
    pair_y = (y_occ[:, None, :] * y_vir[None, :, :]).reshape(
        nocc * nvir, rank
    )
    a_xia = (pair_y @ amplitude_core).T.reshape(rank, nocc, nvir)
    sigma = xp.zeros((rank, rank), dtype=dtype)

    for x0 in range(0, rank, x_block_size):
        x1 = min(x0 + x_block_size, rank)
        y_occ_x = y_occ[:, x0:x1]
        y_vir_x = y_vir[:, x0:x1]

        # Lines 4--5.  The first THC coordinate is blocked; the second spans
        # all columns of the three-index A bridge.
        c_ixy = xp.einsum("aX,Yia->iXY", y_vir_x, a_xia)
        d_axy = xp.einsum("iX,Yia->aXY", y_occ_x, a_xia)
        h_xyz = xp.zeros((x1 - x0, rank, rank), dtype=dtype)

        # Lines 6--12.  Line 2's B tensor is evaluated a slice at a time so
        # no Q*O*V allocation is required.  Z[I,J] is not transposed.
        for eri_index in range(eri_factors.rank):
            b_ia = (
                x_occ * eri_core[eri_index][None, :]
            ) @ x_vir.T
            e_xy = xp.einsum(
                "iXY,i->XY", c_ixy, x_occ[:, eri_index]
            )
            f_xz = xp.einsum(
                "aXZ,a->XZ", d_axy, x_vir[:, eri_index]
            )
            g_yz = (y_occ.T @ b_ia) @ y_vir

            # Algorithm 6, line 11.  The two summands are Omega-A (Eq. 32)
            # and Omega-C (Eq. 33), with coefficient +1 for each.
            h_xyz += (
                e_xy[:, :, None] * f_xz[:, None, :]
                + e_xy[:, None, :] * f_xz[:, :, None]
            ) * g_yz[None, :, :]

        # Lines 13--16.  Two GEMMs per X slice avoid a five-index einsum and
        # retain the paper's O(N^4) formal operation count for linear ranks.
        i_xia = xp.empty((x1 - x0, nocc, nvir), dtype=dtype)
        for local_x in range(x1 - x0):
            i_xia[local_x] = (
                y_occ @ h_xyz[local_x].T
            ) @ y_vir.T
        sigma[x0:x1] = i_xia.reshape(x1 - x0, nocc * nvir) @ pair_y

    return _validate_output(
        xp, sigma, "THC Algorithm 6", transfer_counter
    )
