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

"""Dense, audit-only joint ``Omega-C``/``Omega-D`` oracle.

The printed equations and appendix algorithms in Hohenstein *et al.*,
J. Chem. Phys. **156**, 054102 (2022), DOI 10.1063/5.0077770
(arXiv:2111.11473v1), leave an important implementation boundary.  The text
states that Appendix Algorithm 7 removes a spurious contribution that is
normally cancelled while evaluating ``Omega-C``, but does not identify a
unique term-by-term equality between printed Eq. 35 and Algorithm 7 line 22.

This module keeps the two conventions separate.  For deliberately small
NumPy problems it reconstructs dense doubles amplitudes and ``ovov`` ERIs,
evaluates literal Eq. 35, and compares the following joint quantities::

    joint_algorithm = Eq. 33 + Eq. 37 + Algorithm-7 line 22
    joint_literal   = Eq. 33 + literal Eq. 35

No sign is adjusted to make that comparison pass.  Equality is attempted
only when the reconstructed real-RHF tensors obey the physical pair
symmetries ``t[i,j,a,b] = t[j,i,b,a]`` and
``(i a | j b) = (j b | i a)``.  Even a successful audit remains ineligible
for production use: this endpoint explicitly materializes fourth-order
tensors and does not establish the missing paper convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from gpu4pyscf.cc.thc_eri import ERITHCFactors, thc_omega_c_algorithm5
from gpu4pyscf.cc.thc_omega_d import thc_omega_d_algorithm7


_DEFAULT_MAX_DENSE_WORKING_BYTES = 512 * 1024**2


def _require_numpy_array(value: Any, *, name: str) -> np.ndarray:
    """Reject device arrays before any coercion or scalar read can occur."""

    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"{name} must be a NumPy ndarray; the dense joint audit rejects "
            "GPU and implicitly convertible arrays"
        )
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value)))


def _within_scaled_tolerance(
    difference: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float, float]:
    max_abs_difference = _max_abs(difference)
    scale = max(_max_abs(left), _max_abs(right))
    threshold = float(atol + rtol * scale)
    return max_abs_difference <= threshold, max_abs_difference, scale, threshold


def _validate_problem(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    eri_factors: ERITHCFactors,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(eri_factors, ERITHCFactors):
        raise TypeError("eri_factors must be an ERITHCFactors instance")

    arrays = (
        _require_numpy_array(y_occ, name="y_occ"),
        _require_numpy_array(y_vir, name="y_vir"),
        _require_numpy_array(amplitude_core, name="amplitude_core"),
        _require_numpy_array(eri_factors.x_occ, name="eri_factors.x_occ"),
        _require_numpy_array(eri_factors.x_vir, name="eri_factors.x_vir"),
        _require_numpy_array(eri_factors.core, name="eri_factors.core"),
    )
    y_occ, y_vir, amplitude_core, x_occ, x_vir, eri_core = arrays

    dtypes = tuple(np.dtype(value.dtype) for value in arrays)
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError("all dense joint-audit tensors must have the same dtype")
    dtype = dtypes[0]
    if np.issubdtype(dtype, np.complexfloating):
        raise NotImplementedError(
            "the dense joint audit currently requires real RHF tensors"
        )
    if not np.issubdtype(dtype, np.floating):
        raise TypeError("dense joint-audit tensors must have a floating-point dtype")

    if y_occ.ndim != 2:
        raise ValueError("y_occ must have shape (nocc,amplitude_rank)")
    if y_vir.ndim != 2:
        raise ValueError("y_vir must have shape (nvir,amplitude_rank)")
    if amplitude_core.ndim != 2:
        raise ValueError(
            "amplitude_core must have shape (amplitude_rank,amplitude_rank)"
        )
    nocc, amplitude_rank = map(int, y_occ.shape)
    nvir, virtual_amplitude_rank = map(int, y_vir.shape)
    if nocc < 1 or nvir < 1 or amplitude_rank < 1:
        raise ValueError("dense joint-audit dimensions must be positive")
    if virtual_amplitude_rank != amplitude_rank:
        raise ValueError("y_occ and y_vir amplitude ranks do not match")
    if amplitude_core.shape != (amplitude_rank, amplitude_rank):
        raise ValueError("amplitude_core must be square with the factor rank")

    if x_occ.ndim != 2:
        raise ValueError("eri_factors.x_occ must have shape (nocc,eri_rank)")
    if x_vir.ndim != 2:
        raise ValueError("eri_factors.x_vir must have shape (nvir,eri_rank)")
    if eri_core.ndim != 2:
        raise ValueError("eri_factors.core must have shape (eri_rank,eri_rank)")
    eri_nocc, eri_rank = map(int, x_occ.shape)
    eri_nvir, virtual_eri_rank = map(int, x_vir.shape)
    if eri_rank < 1 or virtual_eri_rank != eri_rank:
        raise ValueError("x_occ and x_vir ERI ranks do not match")
    if (eri_nocc, eri_nvir) != (nocc, nvir):
        raise ValueError("amplitude and ERI orbital dimensions do not match")
    if eri_core.shape != (eri_rank, eri_rank):
        raise ValueError("ERI core must be square with the factor rank")

    for name, value in zip(
        (
            "y_occ",
            "y_vir",
            "amplitude_core",
            "eri_factors.x_occ",
            "eri_factors.x_vir",
            "eri_factors.core",
        ),
        arrays,
    ):
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
    return arrays


def _dense_t2(
    y_occ: np.ndarray,
    y_vir: np.ndarray,
    amplitude_core: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "iX,aX,XY,jY,bY->ijab",
        y_occ,
        y_vir,
        amplitude_core,
        y_occ,
        y_vir,
        optimize=True,
    )


def _dense_ovov(
    x_occ: np.ndarray,
    x_vir: np.ndarray,
    eri_core: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "iI,aI,IJ,jJ,bJ->iajb",
        x_occ,
        x_vir,
        eri_core,
        x_occ,
        x_vir,
        optimize=True,
    )


def _project_dense(
    projector: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    return np.einsum(
        "Xia,Yjb,ijab->XY", projector, projector, residual, optimize=True
    )


def _dense_eq33(
    projector: np.ndarray,
    t2: np.ndarray,
    ovov: np.ndarray,
) -> np.ndarray:
    residual = np.einsum(
        "ikcb,ljad,lckd->ijab", t2, t2, ovov, optimize=True
    )
    return _project_dense(projector, residual)


def _dense_literal_eq35(
    projector: np.ndarray,
    t2: np.ndarray,
    ovov: np.ndarray,
) -> np.ndarray:
    # Printed Eq. 35, first product:
    # (2 t_ik^ac - t_ik^ca) [2(kc|ld) - (kd|lc)]
    #                         (t_jl^bd - t_jl^db).
    two_t_minus_virtual_swap = 2 * t2 - t2.swapaxes(2, 3)
    two_eri_minus_exchange = 2 * ovov - ovov.transpose(0, 3, 2, 1)
    virtual_antisymmetric_t = t2 - t2.swapaxes(2, 3)
    residual = np.einsum(
        "ikac,kcld,jlbd->ijab",
        two_t_minus_virtual_swap,
        two_eri_minus_exchange,
        virtual_antisymmetric_t,
        optimize=True,
    )

    # Printed Eq. 35, final product:
    # + t_ik^ca (kd|lc) t_jl^db.
    residual += np.einsum(
        "ikca,kdlc,jldb->ijab", t2, ovov, t2, optimize=True
    )
    return np.dtype(t2.dtype).type(0.25) * _project_dense(
        projector, residual
    )


def _dense_eq36(
    projector: np.ndarray, t2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    virtual_swap = t2.swapaxes(2, 3)
    r_intermediate = np.einsum(
        "Xjb,ijab->Xia",
        projector,
        2 * t2 - virtual_swap,
        optimize=True,
    )
    s_exchange = -np.einsum(
        "Xjb,ijab->Xia", projector, virtual_swap, optimize=True
    )
    return r_intermediate, s_exchange


def _dense_eq37(
    r_intermediate: np.ndarray, ovov: np.ndarray
) -> np.ndarray:
    two_eri_minus_exchange = 2 * ovov - ovov.transpose(0, 3, 2, 1)
    return np.dtype(ovov.dtype).type(0.25) * np.einsum(
        "Xia,Yjb,iajb->XY",
        r_intermediate,
        r_intermediate,
        two_eri_minus_exchange,
        optimize=True,
    )


def _dense_algorithm7_line22(
    s_exchange: np.ndarray, ovov: np.ndarray
) -> np.ndarray:
    exchange_ovov = ovov.transpose(0, 3, 2, 1)
    return np.dtype(ovov.dtype).type(0.25) * np.einsum(
        "Xia,Yjb,iajb->XY",
        s_exchange,
        s_exchange,
        exchange_ovov,
        optimize=True,
    )


@dataclass(frozen=True)
class THCOmegaCDSymmetryDiagnostics:
    """Pair-symmetry evidence computed from reconstructed dense tensors."""

    amplitude_core_max_abs_asymmetry: float
    eri_core_max_abs_asymmetry: float
    t2_pair_max_abs_asymmetry: float
    eri_pair_max_abs_asymmetry: float
    amplitude_core_scale: float
    eri_core_scale: float
    t2_scale: float
    eri_scale: float
    amplitude_core_symmetric: bool
    eri_core_symmetric: bool
    t2_pair_symmetric: bool
    eri_pair_symmetric: bool
    full_pair_gauge: bool
    symmetry_atol: float
    symmetry_rtol: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "amplitude_core_max_abs_asymmetry": float(
                self.amplitude_core_max_abs_asymmetry
            ),
            "eri_core_max_abs_asymmetry": float(
                self.eri_core_max_abs_asymmetry
            ),
            "t2_pair_max_abs_asymmetry": float(
                self.t2_pair_max_abs_asymmetry
            ),
            "eri_pair_max_abs_asymmetry": float(
                self.eri_pair_max_abs_asymmetry
            ),
            "amplitude_core_scale": float(self.amplitude_core_scale),
            "eri_core_scale": float(self.eri_core_scale),
            "t2_scale": float(self.t2_scale),
            "eri_scale": float(self.eri_scale),
            "amplitude_core_symmetric": bool(
                self.amplitude_core_symmetric
            ),
            "eri_core_symmetric": bool(self.eri_core_symmetric),
            "t2_pair_symmetric": bool(self.t2_pair_symmetric),
            "eri_pair_symmetric": bool(self.eri_pair_symmetric),
            "full_pair_gauge": bool(self.full_pair_gauge),
            "symmetry_atol": float(self.symmetry_atol),
            "symmetry_rtol": float(self.symmetry_rtol),
        }


@dataclass(frozen=True)
class THCOmegaCDJointAuditResult:
    """Separated dense-oracle and appendix-algorithm joint audit result."""

    omega_c_eq33: np.ndarray
    omega_d_eq35_literal: np.ndarray
    omega_d_eq37: np.ndarray
    algorithm7_line22_correction: np.ndarray
    joint_algorithm: np.ndarray
    joint_literal: np.ndarray
    joint_difference: np.ndarray
    omega_c_eq33_dense_reference: np.ndarray
    omega_d_eq37_dense_reference: np.ndarray
    algorithm7_line22_dense_reference: np.ndarray
    symmetry_diagnostics: THCOmegaCDSymmetryDiagnostics
    endpoint_consistent: bool
    equivalence_attempted: bool
    joint_equivalent: bool | None
    endpoint_max_abs_difference: float
    joint_max_abs_difference: float
    joint_scale: float
    joint_equivalence_threshold: float
    algorithm5_outer_block_size: int
    algorithm7_x_block_size: int
    estimated_dense_working_bytes: int
    max_dense_working_bytes: int
    equivalence_atol: float
    equivalence_rtol: float

    @property
    def full_pair_gauge(self) -> bool:
        return self.symmetry_diagnostics.full_pair_gauge

    @property
    def algorithm7_combined(self) -> np.ndarray:
        return self.omega_d_eq37 + self.algorithm7_line22_correction

    def metadata(self) -> dict[str, Any]:
        if not self.endpoint_consistent:
            equivalence_status = "not-attempted-endpoint-mismatch"
        elif not self.full_pair_gauge:
            equivalence_status = "not-attempted-outside-full-pair-gauge"
        elif self.joint_equivalent:
            equivalence_status = "equal-within-explicit-audit-tolerance"
        else:
            equivalence_status = "unequal-within-explicit-audit-tolerance"
        return {
            "paper": "Hohenstein-2022",
            "paper_source": "arXiv:2111.11473v1",
            "paper_equations": [33, 35, 36, 37],
            "paper_algorithms": [5, 7],
            "scope": "joint-omega-c-omega-d-dense-audit",
            "omega_c_convention": "printed-equation-33-algorithm-5",
            "omega_d_literal_convention": "printed-equation-35",
            "omega_d_algorithm_convention": (
                "printed-equation-37-plus-appendix-algorithm-7-line-22"
            ),
            "joint_algorithm_definition": "eq33+eq37+algorithm7-line22",
            "joint_literal_definition": "eq33+literal-eq35",
            "joint_difference_definition": "joint_algorithm-joint_literal",
            "line22_correction_sign": "as-printed-positive-addend",
            "line22_eq35_term_mapping": "unresolved-by-paper",
            "difference_is_diagnostic": True,
            "literal_eq35_is_production_oracle": False,
            "endpoint_consistent": bool(self.endpoint_consistent),
            "endpoint_max_abs_difference": float(
                self.endpoint_max_abs_difference
            ),
            "equivalence_attempted": bool(self.equivalence_attempted),
            "joint_equivalent": self.joint_equivalent,
            "joint_equivalence_status": equivalence_status,
            "joint_max_abs_difference": float(
                self.joint_max_abs_difference
            ),
            "joint_scale": float(self.joint_scale),
            "joint_equivalence_threshold": float(
                self.joint_equivalence_threshold
            ),
            "equivalence_atol": float(self.equivalence_atol),
            "equivalence_rtol": float(self.equivalence_rtol),
            "symmetry": self.symmetry_diagnostics.as_dict(),
            "full_pair_gauge_definition": [
                "t[i,j,a,b]=t[j,i,b,a]",
                "eri[i,a,j,b]=eri[j,b,i,a]",
            ],
            "input_backend": "numpy",
            "dtype": np.dtype(self.joint_difference.dtype).name,
            "algorithm5_outer_block_size": int(
                self.algorithm5_outer_block_size
            ),
            "algorithm7_x_block_size": int(self.algorithm7_x_block_size),
            "estimated_dense_working_bytes": int(
                self.estimated_dense_working_bytes
            ),
            "max_dense_working_bytes": int(self.max_dense_working_bytes),
            "materializes_dense_t2": True,
            "materializes_four_index_eri": True,
            "small_problem_only": True,
            "implicit_gpu_host_transfer": False,
            "gpu_inputs_accepted": False,
            "audit_only": True,
            "production_enabled": False,
            "performance_eligible": False,
            "complete_ccsd_residual": False,
            "requires_rr_back_projection": True,
        }


def _symmetry_diagnostics(
    amplitude_core: np.ndarray,
    eri_core: np.ndarray,
    t2: np.ndarray,
    ovov: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> THCOmegaCDSymmetryDiagnostics:
    amplitude_core_difference = amplitude_core - amplitude_core.T
    eri_core_difference = eri_core - eri_core.T
    t2_pair_difference = t2 - t2.transpose(1, 0, 3, 2)
    eri_pair_difference = ovov - ovov.transpose(2, 3, 0, 1)

    amplitude_core_symmetric, amplitude_core_defect, amplitude_core_scale, _ = (
        _within_scaled_tolerance(
            amplitude_core_difference,
            amplitude_core,
            amplitude_core.T,
            atol=atol,
            rtol=rtol,
        )
    )
    eri_core_symmetric, eri_core_defect, eri_core_scale, _ = (
        _within_scaled_tolerance(
            eri_core_difference,
            eri_core,
            eri_core.T,
            atol=atol,
            rtol=rtol,
        )
    )
    t2_pair_symmetric, t2_pair_defect, t2_scale, _ = (
        _within_scaled_tolerance(
            t2_pair_difference,
            t2,
            t2.transpose(1, 0, 3, 2),
            atol=atol,
            rtol=rtol,
        )
    )
    eri_pair_symmetric, eri_pair_defect, eri_scale, _ = (
        _within_scaled_tolerance(
            eri_pair_difference,
            ovov,
            ovov.transpose(2, 3, 0, 1),
            atol=atol,
            rtol=rtol,
        )
    )
    return THCOmegaCDSymmetryDiagnostics(
        amplitude_core_max_abs_asymmetry=amplitude_core_defect,
        eri_core_max_abs_asymmetry=eri_core_defect,
        t2_pair_max_abs_asymmetry=t2_pair_defect,
        eri_pair_max_abs_asymmetry=eri_pair_defect,
        amplitude_core_scale=amplitude_core_scale,
        eri_core_scale=eri_core_scale,
        t2_scale=t2_scale,
        eri_scale=eri_scale,
        amplitude_core_symmetric=amplitude_core_symmetric,
        eri_core_symmetric=eri_core_symmetric,
        t2_pair_symmetric=t2_pair_symmetric,
        eri_pair_symmetric=eri_pair_symmetric,
        full_pair_gauge=t2_pair_symmetric and eri_pair_symmetric,
        symmetry_atol=atol,
        symmetry_rtol=rtol,
    )


def thc_omega_cd_joint_audit(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    eri_factors: ERITHCFactors,
    *,
    algorithm5_outer_block_size: int = 1,
    algorithm7_x_block_size: int = 1,
    symmetry_atol: Real = 0.0,
    symmetry_rtol: Real | None = None,
    equivalence_atol: Real = 0.0,
    equivalence_rtol: Real | None = None,
    max_dense_working_bytes: int = _DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> THCOmegaCDJointAuditResult:
    """Audit Algorithms 5/7 against literal dense Eqs. 33 and 35--37.

    The return value always remains audit-only.  ``joint_equivalent`` is
    ``None`` outside the full physical pair gauge or if either reused
    algorithm endpoint fails its independent dense reference.  Inside that
    boundary it is the result of the explicitly recorded scaled-tolerance
    comparison; inequality is returned as ``False`` rather than hidden by a
    sign or permutation adjustment.

    Device arrays are rejected without coercion.  ``max_dense_working_bytes``
    is checked before any fourth-order tensor is allocated so this small-case
    oracle cannot accidentally be launched on the WATER8 production shape.
    """

    (
        y_occ,
        y_vir,
        amplitude_core,
        x_occ,
        x_vir,
        eri_core,
    ) = _validate_problem(y_occ, y_vir, amplitude_core, eri_factors)
    algorithm5_outer_block_size = _positive_integer(
        algorithm5_outer_block_size, name="algorithm5_outer_block_size"
    )
    algorithm7_x_block_size = _positive_integer(
        algorithm7_x_block_size, name="algorithm7_x_block_size"
    )
    max_dense_working_bytes = _positive_integer(
        max_dense_working_bytes, name="max_dense_working_bytes"
    )
    symmetry_atol = _nonnegative_real(symmetry_atol, name="symmetry_atol")
    equivalence_atol = _nonnegative_real(
        equivalence_atol, name="equivalence_atol"
    )
    dtype = np.dtype(y_occ.dtype)
    if symmetry_rtol is None:
        symmetry_rtol = 256.0 * float(np.finfo(dtype).eps)
    symmetry_rtol = _nonnegative_real(symmetry_rtol, name="symmetry_rtol")
    if equivalence_rtol is None:
        equivalence_rtol = 2048.0 * float(np.finfo(dtype).eps)
    equivalence_rtol = _nonnegative_real(
        equivalence_rtol, name="equivalence_rtol"
    )

    nocc, amplitude_rank = map(int, y_occ.shape)
    nvir = int(y_vir.shape[0])
    tensor_elements = nocc * nocc * nvir * nvir
    # The dense equations hold t2/ovov plus several Eq. 35 work arrays.  The
    # factor of ten is deliberately conservative and excludes BLAS scratch,
    # which is why the metadata calls this an estimate rather than a peak.
    estimated_dense_working_bytes = int(
        10 * tensor_elements * dtype.itemsize
        + 8 * amplitude_rank * nocc * nvir * dtype.itemsize
    )
    if estimated_dense_working_bytes > max_dense_working_bytes:
        raise MemoryError(
            "dense joint audit estimated working set "
            f"{estimated_dense_working_bytes} bytes exceeds explicit limit "
            f"{max_dense_working_bytes} bytes"
        )

    projector = np.einsum("iX,aX->Xia", y_occ, y_vir, optimize=True)
    t2 = _dense_t2(y_occ, y_vir, amplitude_core)
    ovov = _dense_ovov(x_occ, x_vir, eri_core)

    omega_c_eq33_dense_reference = _dense_eq33(projector, t2, ovov)
    omega_d_eq35_literal = _dense_literal_eq35(projector, t2, ovov)
    r_intermediate, s_exchange = _dense_eq36(projector, t2)
    omega_d_eq37_dense_reference = _dense_eq37(r_intermediate, ovov)
    algorithm7_line22_dense_reference = _dense_algorithm7_line22(
        s_exchange, ovov
    )

    omega_c_eq33 = thc_omega_c_algorithm5(
        y_occ,
        y_vir,
        amplitude_core,
        eri_factors,
        outer_block_size=algorithm5_outer_block_size,
    )
    omega_d_result = thc_omega_d_algorithm7(
        y_occ,
        y_vir,
        amplitude_core,
        eri_factors,
        x_block_size=algorithm7_x_block_size,
    )
    omega_d_eq37 = omega_d_result.main_eq37
    algorithm7_line22_correction = omega_d_result.spurious_removal

    endpoint_checks = []
    endpoint_defects = []
    for observed, expected in (
        (omega_c_eq33, omega_c_eq33_dense_reference),
        (omega_d_eq37, omega_d_eq37_dense_reference),
        (
            algorithm7_line22_correction,
            algorithm7_line22_dense_reference,
        ),
    ):
        check, defect, _, _ = _within_scaled_tolerance(
            observed - expected,
            observed,
            expected,
            atol=equivalence_atol,
            rtol=equivalence_rtol,
        )
        endpoint_checks.append(check)
        endpoint_defects.append(defect)
    endpoint_consistent = all(endpoint_checks)
    endpoint_max_abs_difference = max(endpoint_defects)

    joint_algorithm = (
        omega_c_eq33 + omega_d_eq37 + algorithm7_line22_correction
    )
    joint_literal = omega_c_eq33 + omega_d_eq35_literal
    joint_difference = joint_algorithm - joint_literal
    (
        joint_equal_within_tolerance,
        joint_max_abs_difference,
        joint_scale,
        joint_equivalence_threshold,
    ) = _within_scaled_tolerance(
        joint_difference,
        joint_algorithm,
        joint_literal,
        atol=equivalence_atol,
        rtol=equivalence_rtol,
    )

    symmetry_diagnostics = _symmetry_diagnostics(
        amplitude_core,
        eri_core,
        t2,
        ovov,
        atol=symmetry_atol,
        rtol=symmetry_rtol,
    )
    equivalence_attempted = bool(
        endpoint_consistent and symmetry_diagnostics.full_pair_gauge
    )
    joint_equivalent = (
        bool(joint_equal_within_tolerance) if equivalence_attempted else None
    )

    for name, value in (
        ("omega_c_eq33", omega_c_eq33),
        ("omega_d_eq35_literal", omega_d_eq35_literal),
        ("omega_d_eq37", omega_d_eq37),
        (
            "algorithm7_line22_correction",
            algorithm7_line22_correction,
        ),
        ("joint_algorithm", joint_algorithm),
        ("joint_literal", joint_literal),
        ("joint_difference", joint_difference),
    ):
        if not bool(np.all(np.isfinite(value))):
            raise FloatingPointError(f"{name} produced non-finite values")

    return THCOmegaCDJointAuditResult(
        omega_c_eq33=omega_c_eq33,
        omega_d_eq35_literal=omega_d_eq35_literal,
        omega_d_eq37=omega_d_eq37,
        algorithm7_line22_correction=algorithm7_line22_correction,
        joint_algorithm=joint_algorithm,
        joint_literal=joint_literal,
        joint_difference=joint_difference,
        omega_c_eq33_dense_reference=omega_c_eq33_dense_reference,
        omega_d_eq37_dense_reference=omega_d_eq37_dense_reference,
        algorithm7_line22_dense_reference=(
            algorithm7_line22_dense_reference
        ),
        symmetry_diagnostics=symmetry_diagnostics,
        endpoint_consistent=endpoint_consistent,
        equivalence_attempted=equivalence_attempted,
        joint_equivalent=joint_equivalent,
        endpoint_max_abs_difference=endpoint_max_abs_difference,
        joint_max_abs_difference=joint_max_abs_difference,
        joint_scale=joint_scale,
        joint_equivalence_threshold=joint_equivalence_threshold,
        algorithm5_outer_block_size=algorithm5_outer_block_size,
        algorithm7_x_block_size=algorithm7_x_block_size,
        estimated_dense_working_bytes=estimated_dense_working_bytes,
        max_dense_working_bytes=max_dense_working_bytes,
        equivalence_atol=equivalence_atol,
        equivalence_rtol=equivalence_rtol,
    )


__all__ = [
    "THCOmegaCDJointAuditResult",
    "THCOmegaCDSymmetryDiagnostics",
    "thc_omega_cd_joint_audit",
]
