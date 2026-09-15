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

"""Audit-only THC ``Omega-D`` contraction from published Algorithm 7.

This module follows Eqs. 37--38 and Appendix Algorithm 7 of the version of
record of Hohenstein *et al.*, J. Chem. Phys. **156**, 054102 (2022),
DOI 10.1063/5.0077770.  It returns the Eq. 38 main contribution, the signed
spurious-term removal in Algorithm 7 line 19, and their sum separately.

The version-of-record Eq. 36 contains ``(2 t_jl^bd - t_jl^db)`` and is
algebraically equivalent to that sum in the physical full-pair gauge.  The
submitted arXiv:2111.11473v1 Eq. 35 instead printed
``(t_jl^bd - t_jl^db)``.  That historical formula is retained as an explicit
legacy diagnostic in :mod:`gpu4pyscf.cc.thc_omega_cd_audit`; it is never used
as the correctness oracle here.  This endpoint still remains audit-only and
is not a production ``Omega-D`` replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gpu4pyscf.cc.thc_eri import ERITHCFactors


def _array_module(value: Any):
    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy" and isinstance(value, np.ndarray):
        return np
    if module == "cupy":
        import cupy

        if isinstance(value, cupy.ndarray):
            return cupy
    raise TypeError("THC Omega-D tensors must be NumPy or CuPy arrays")


def _one_backend_and_device(reference: Any, *values: Any):
    xp = _array_module(reference)
    if any(_array_module(value) is not xp for value in values):
        raise TypeError("all THC Omega-D tensors must use one array backend")
    if xp is not np:
        device_id = int(reference.device.id)
        if any(int(value.device.id) != device_id for value in values):
            raise TypeError("all THC Omega-D CuPy tensors must use one device")
    return xp


def _one_real_dtype(*values: Any) -> np.dtype:
    dtypes = tuple(np.dtype(value.dtype) for value in values)
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError("all THC Omega-D tensors must have the same dtype")
    dtype = dtypes[0]
    if np.issubdtype(dtype, np.complexfloating):
        raise NotImplementedError("THC Algorithm 7 currently requires real RHF")
    if not np.issubdtype(dtype, np.floating):
        raise TypeError("THC Omega-D tensors must have a floating-point dtype")
    return dtype


def _require_finite(
    xp: Any,
    *,
    transfer_counter: Any,
    operation: str,
    error_type: type[Exception] = ValueError,
    **values: Any,
) -> None:
    if xp is not np:
        if transfer_counter is None:
            raise ValueError(
                "GPU THC Omega-D validation requires an explicit "
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
        raise TypeError("x_block_size must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError("x_block_size must be positive")
    return value


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _validate_inputs(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    eri_factors: ERITHCFactors,
    *,
    transfer_counter: Any,
):
    if not isinstance(eri_factors, ERITHCFactors):
        raise TypeError("eri_factors must be an ERITHCFactors instance")
    xp = _one_backend_and_device(
        y_occ,
        y_vir,
        amplitude_core,
        eri_factors.x_occ,
        eri_factors.x_vir,
        eri_factors.core,
    )
    dtype = _one_real_dtype(
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
            "amplitude_core must have shape "
            "(amplitude_rank,amplitude_rank)"
        )
    nocc, rank = map(int, y_occ.shape)
    nvir, virtual_rank = map(int, y_vir.shape)
    if nocc < 1 or nvir < 1 or rank < 1:
        raise ValueError("THC Omega-D dimensions must be positive")
    if virtual_rank != rank:
        raise ValueError("y_occ and y_vir amplitude ranks do not match")
    if amplitude_core.shape != (rank, rank):
        raise ValueError("amplitude_core must be square with the factor rank")
    if getattr(eri_factors.x_occ, "ndim", None) != 2:
        raise ValueError("x_occ must have shape (nocc,eri_rank)")
    if getattr(eri_factors.x_vir, "ndim", None) != 2:
        raise ValueError("x_vir must have shape (nvir,eri_rank)")
    eri_nocc, eri_rank = map(int, eri_factors.x_occ.shape)
    eri_nvir, virtual_eri_rank = map(int, eri_factors.x_vir.shape)
    if eri_rank < 1 or virtual_eri_rank != eri_rank:
        raise ValueError("x_occ and x_vir ERI ranks do not match")
    if (nocc, nvir) != (eri_nocc, eri_nvir):
        raise ValueError("amplitude and ERI orbital dimensions do not match")
    if eri_factors.core.shape != (eri_rank, eri_rank):
        raise ValueError("ERI core must be square with the factor rank")
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="thc_omega_d_input_validation",
        y_occ=y_occ,
        y_vir=y_vir,
        amplitude_core=amplitude_core,
        x_occ=eri_factors.x_occ,
        x_vir=eri_factors.x_vir,
        eri_core=eri_factors.core,
    )
    return xp, dtype, nocc, nvir, rank


@dataclass(frozen=True)
class THCOmegaDAlgorithm7Result:
    """Separated audit result for Appendix Algorithm 7."""

    r_intermediate: Any
    s_exchange: Any
    main_eq38: Any
    spurious_removal: Any
    combined: Any
    x_block_size: int
    eri_rank: int
    largest_intermediate_nbytes: int

    @property
    def sigma_thc(self):
        """Return the signed published Algorithm 7 line-19 sum."""

        return self.combined

    @property
    def main_eq37(self):
        """Compatibility alias for the submitted-preprint equation number.

        The same expression is Eq. 38 in the version of record.  New code
        should use :attr:`main_eq38`.
        """

        return self.main_eq38

    def metadata(self) -> dict[str, Any]:
        xp = _array_module(self.combined)
        return {
            "paper": "Hohenstein-2022",
            "paper_source": (
                "J. Chem. Phys. 156, 054102 (2022), "
                "DOI:10.1063/5.0077770"
            ),
            "paper_source_version": "version-of-record",
            "paper_algorithm": 7,
            "paper_equations_referenced": [36, 37, 38],
            "implemented_equations": [37, 38],
            "dense_audit_equations": [36],
            "published_eq36_scope": (
                "literal-equivalence-verified-in-full-pair-gauge"
            ),
            "published_eq36_literal_equivalence_in_full_pair_gauge": True,
            "published_eq36_literal_equivalence_unconditional": False,
            "published_eq36_equivalence_requires_full_pair_gauge": True,
            "coordinate_space": "amplitude-thc-auxiliary",
            "main_convention": "version-of-record-equation-38",
            "spurious_removal_convention": (
                "signed-plus-one-quarter-tildeG-X-dot-S-Y"
            ),
            "spurious_removal_is_signed_addend": True,
            "combined_convention": (
                "version-of-record-appendix-algorithm-7-line-19"
            ),
            "legacy_preprint_source": "arXiv:2111.11473v1",
            "legacy_preprint_equation": 35,
            "legacy_preprint_eq35_second_amplitude_factor": (
                "t_jl^bd-t_jl^db"
            ),
            "legacy_preprint_eq35_is_correctness_oracle": False,
            "legacy_preprint_eq35_literal_equivalence": False,
            "legacy_preprint_eq35_status": "known-version-difference",
            # Backward-compatible keys are explicitly scoped to the legacy
            # submitted preprint; they must not be mistaken for Eq. 36 of the
            # version of record.
            "eq35_literal_equivalence": False,
            "eq35_literal_status": "legacy-preprint-version-difference",
            "eq35_literal_reason": (
                "arXiv:2111.11473v1 Eq. 35 omits the direct coefficient in "
                "the second amplitude factor; the version-of-record Eq. 36 "
                "prints 2*t_jl^bd-t_jl^db"
            ),
            "requires_rr_back_projection": True,
            "complete_ccsd_residual": False,
            "audit_only": True,
            "production_enabled": False,
            "performance_eligible": False,
            "gpu_performance_claim": False,
            "formal_scaling_with_linear_ranks": "O(N^4)",
            "actual_schedule": (
                "Python X-block loop with NumPy/CuPy GEMM and einsum; "
                "no production kernel integration"
            ),
            "backend": xp.__name__,
            "dtype": np.dtype(self.combined.dtype).name,
            "x_block_size": int(self.x_block_size),
            "eri_rank": int(self.eri_rank),
            "largest_intermediate_nbytes": int(
                self.largest_intermediate_nbytes
            ),
            "largest_intermediate_nbytes_scope": (
                "largest-explicit-logical-array"
            ),
            "peak_memory_requires_profiling": True,
            "materializes_dense_t2": False,
            "materializes_four_index_eri": False,
            "implicit_host_tensor_transfer": False,
            "gpu_validation_requires_transfer_counter": True,
            "core_symmetry_enforced": False,
        }


def thc_omega_d_algorithm7(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    eri_factors: ERITHCFactors,
    *,
    x_block_size: int = 1,
    transfer_counter: Any = None,
    require_eq35_literal_equivalence: bool = False,
) -> THCOmegaDAlgorithm7Result:
    """Evaluate Appendix Algorithm 7 using version-of-record numbering.

    ``main_eq38`` evaluates Eq. 38 using the Eq. 37 intermediate.
    ``spurious_removal`` is the signed final addend of Algorithm 7 line 19,
    and ``combined`` is their sum.  Dense doubles and four-index ERIs are not
    materialized.

    ``require_eq35_literal_equivalence`` is a compatibility switch referring
    only to the erroneous submitted-preprint Eq. 35.  It remains rejected so
    the historical expression cannot silently replace published Eq. 36.
    """

    require_eq35_literal_equivalence = _boolean(
        require_eq35_literal_equivalence,
        name="require_eq35_literal_equivalence",
    )
    if require_eq35_literal_equivalence:
        raise NotImplementedError(
            "arXiv:2111.11473v1 literal Eq. 35 is a legacy version "
            "difference and is not equivalent to published Algorithm 7; "
            "use the version-of-record Eq. 36 audit"
        )
    x_block_size = _positive_block_size(x_block_size)
    if not isinstance(eri_factors, ERITHCFactors):
        raise TypeError("eri_factors must be an ERITHCFactors instance")
    if transfer_counter is None:
        transfer_counter = eri_factors.transfer_counter
    xp, dtype, nocc, nvir, rank = _validate_inputs(
        y_occ,
        y_vir,
        amplitude_core,
        eri_factors,
        transfer_counter=transfer_counter,
    )

    x_occ = eri_factors.x_occ
    x_vir = eri_factors.x_vir
    eri_core = eri_factors.core
    pair_y = (y_occ[:, None, :] * y_vir[None, :, :]).reshape(
        nocc * nvir, rank
    )
    pair_x = (x_occ[:, None, :] * x_vir[None, :, :]).reshape(
        nocc * nvir, eri_factors.rank
    )

    # Algorithm 7, lines 1--3.  A[X,Y] uses T[Y,Z], not T[Z,Y].
    overlap_occ = y_occ.T @ y_occ
    overlap_vir = y_vir.T @ y_vir
    a_xy = (overlap_occ * overlap_vir) @ amplitude_core.T

    r_xia = xp.empty((rank, nocc, nvir), dtype=dtype)
    s_xia = xp.empty_like(r_xia)
    largest_intermediate_nbytes = max(
        int(pair_y.nbytes),
        int(pair_x.nbytes),
        int(overlap_occ.nbytes),
        int(overlap_vir.nbytes),
        int(a_xy.nbytes),
        int(r_xia.nbytes),
        int(s_xia.nbytes),
    )

    # Lines 4--8.  S is retained separately because line 19 reuses only its
    # exchange component for the explicit spurious-term removal.
    for x0 in range(0, rank, x_block_size):
        x1 = min(x0 + x_block_size, rank)
        direct = (
            2 * (a_xy[x0:x1] @ pair_y.T)
        ).reshape(x1 - x0, nocc, nvir)
        largest_intermediate_nbytes = max(
            largest_intermediate_nbytes, int(direct.nbytes)
        )
        for local_x, x_index in enumerate(range(x0, x1)):
            b_yz = (
                overlap_vir[:, x_index, None]
                * overlap_occ[x_index, None, :]
                * amplitude_core
            )
            s_xia[x_index] = -(y_occ @ b_yz) @ y_vir.T
            r_xia[x_index] = direct[local_x] + s_xia[x_index]
            largest_intermediate_nbytes = max(
                largest_intermediate_nbytes, int(b_yz.nbytes)
            )

    # Published Eq. 37 is now represented by R.  Lines 9--10 provide the
    # Coulomb-like half of Eq. 38.  D[X,I] = sum(J) C[X,J] Z[I,J].
    r_flat = r_xia.reshape(rank, nocc * nvir)
    s_flat = s_xia.reshape(rank, nocc * nvir)
    c_xi = r_flat @ pair_x
    d_xi = c_xi @ eri_core.T
    coulomb = 2 * (c_xi @ d_xi.T)
    exchange = xp.empty((rank, rank), dtype=dtype)
    removal = xp.empty((rank, rank), dtype=dtype)
    largest_intermediate_nbytes = max(
        largest_intermediate_nbytes,
        int(c_xi.nbytes),
        int(d_xi.nbytes),
        int(coulomb.nbytes),
        int(exchange.nbytes),
        int(removal.nbytes),
    )

    # Lines 11--18, blocked over the output X coordinate.  F[I,J] uses
    # Z[I,J] as printed; G then contracts x_occ[:,J] and x_vir[:,I].
    for x0 in range(0, rank, x_block_size):
        x1 = min(x0 + x_block_size, rank)
        g_xia = xp.empty((x1 - x0, nocc, nvir), dtype=dtype)
        g_tilde_xia = xp.empty_like(g_xia)
        for local_x, x_index in enumerate(range(x0, x1)):
            e_ij = x_occ.T @ r_xia[x_index] @ x_vir
            f_ij = e_ij * eri_core
            g_xia[local_x] = (x_occ @ f_ij.T) @ x_vir.T

            e_tilde_ij = x_occ.T @ s_xia[x_index] @ x_vir
            f_tilde_ij = e_tilde_ij * eri_core
            g_tilde_xia[local_x] = (
                x_occ @ f_tilde_ij.T
            ) @ x_vir.T
            largest_intermediate_nbytes = max(
                largest_intermediate_nbytes,
                int(e_ij.nbytes),
                int(f_ij.nbytes),
                int(e_tilde_ij.nbytes),
                int(f_tilde_ij.nbytes),
            )
        exchange[x0:x1] = g_xia.reshape(
            x1 - x0, nocc * nvir
        ) @ r_flat.T
        removal[x0:x1] = g_tilde_xia.reshape(
            x1 - x0, nocc * nvir
        ) @ s_flat.T
        largest_intermediate_nbytes = max(
            largest_intermediate_nbytes,
            int(g_xia.nbytes),
            int(g_tilde_xia.nbytes),
        )

    quarter = dtype.type(0.25)
    main_eq38 = quarter * (coulomb - exchange)
    spurious_removal = quarter * removal
    combined = main_eq38 + spurious_removal
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="thc_omega_d_output_validation",
        error_type=FloatingPointError,
        r_intermediate=r_xia,
        s_exchange=s_xia,
        main_eq38=main_eq38,
        spurious_removal=spurious_removal,
        combined=combined,
    )
    return THCOmegaDAlgorithm7Result(
        r_intermediate=r_xia,
        s_exchange=s_xia,
        main_eq38=main_eq38,
        spurious_removal=spurious_removal,
        combined=combined,
        x_block_size=x_block_size,
        eri_rank=eri_factors.rank,
        largest_intermediate_nbytes=largest_intermediate_nbytes,
    )


__all__ = ["THCOmegaDAlgorithm7Result", "thc_omega_d_algorithm7"]
