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

"""Fail-closed assembly audit for THC CCSD paper Algorithms 1--10.

This module composes the independently audited endpoints without restating
their tensor formulae.  The doubles-core schedule is

``A1 + A2 + A3 + ((A4 + A5) XOR A6) + (A7 + A7.T) + A9``

in amplitude-THC coordinates, followed by exactly one shared RR
back-projection.  ``A7`` is the raw Appendix Algorithm 7 endpoint; its pair
transpose is added by the complete residual assembler and both are retained
separately for audit.  The singles schedule is
``A8.singles_gh + A10.singles_ij`` and is never back-projected.

The result is deliberately not a production residual.  The version-of-record
Eq. 36 / Algorithm 7 identity is resolved in the physical full-pair gauge,
while the submitted arXiv v1 Eq. 35 is retained only as a historical version
diagnostic.  Resolving that algebra boundary does not validate the complete or
inexact equations, so every acceptance and production flag stays false.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from typing import Any

import numpy as np

from gpu4pyscf.cc import thc_eri as _eri
from gpu4pyscf.cc import thc_omega_d as _omega_d
from gpu4pyscf.cc import thc_omega_ghi as _omega_ghi
from gpu4pyscf.cc import thc_residual as _residual
from gpu4pyscf.cc.thc_eri import ERITHCFactors
from gpu4pyscf.cc.thc_fhat import T1TransformedFHat
from gpu4pyscf.cc.thc_omega_d import THCOmegaDAlgorithm7Result
from gpu4pyscf.cc.thc_omega_ghi import (
    THCOmegaEAlgorithm9Result,
    THCOmegaGHAlgorithm8Result,
    THCOmegaIJAlgorithm10Result,
    T1TransformedCholeskyBlocks,
)
from gpu4pyscf.cc.thc_residual import T1TransformedCholesky


_PROVENANCE_KEYS = {
    "orbitals": ("orbital_identity", "orbital_fingerprint"),
    "integrals": ("integral_identity", "integral_fingerprint"),
    "hcore": ("hcore_identity", "hcore_fingerprint"),
}
_OMEGA_AC_PATHS = {"separate-4-plus-5", "joint-6"}


def _array_module(value: Any):
    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy" and isinstance(value, np.ndarray):
        return np
    if module == "cupy":
        import cupy

        if isinstance(value, cupy.ndarray):
            return cupy
    raise TypeError("complete THC audit tensors must be NumPy or CuPy arrays")


def _one_backend_and_device(reference: Any, *values: Any):
    xp = _array_module(reference)
    if any(_array_module(value) is not xp for value in values):
        raise TypeError("all complete THC audit tensors must use one backend")
    if xp is not np:
        device_id = int(reference.device.id)
        if any(int(value.device.id) != device_id for value in values):
            raise TypeError("all complete THC audit tensors must use one device")
        current_device_id = int(xp.cuda.runtime.getDevice())
        if current_device_id != device_id:
            raise RuntimeError(
                "current CuPy device does not match complete THC audit "
                f"inputs: current={current_device_id}, input={device_id}"
            )
    return xp


def _one_real_fp64_dtype(*values: Any) -> np.dtype:
    dtypes = tuple(np.dtype(value.dtype) for value in values)
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError("all complete THC audit tensors must have one dtype")
    if dtypes[0] != np.dtype(np.float64):
        raise TypeError("complete THC audit tensors must use real FP64")
    return dtypes[0]


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_counter(xp: Any, transfer_counter: Any) -> None:
    if transfer_counter is not None and not callable(
        getattr(transfer_counter, "record_d2h", None)
    ):
        raise TypeError("transfer_counter must provide record_d2h")
    if xp is not np and transfer_counter is None:
        raise ValueError(
            "GPU complete THC audit validation requires an explicit "
            "transfer_counter"
        )


def _read_control_scalar(
    xp: Any,
    value: Any,
    *,
    transfer_counter: Any,
    operation: str,
) -> Any:
    scalar = xp.asarray(value)
    if scalar.ndim != 0:
        raise ValueError("internal complete THC audit control value must be scalar")
    if xp is not np:
        transfer_counter.record_d2h(int(scalar.nbytes), operation=operation)
    return scalar.item() if hasattr(scalar, "item") else scalar


def _require_finite(
    xp: Any,
    *,
    transfer_counter: Any,
    operation: str,
    error_type: type[Exception] = ValueError,
    **values: Any,
) -> None:
    finite = xp.asarray(True)
    for value in values.values():
        finite = xp.logical_and(finite, xp.all(xp.isfinite(value)))
    observed = _read_control_scalar(
        xp,
        finite,
        transfer_counter=transfer_counter,
        operation=operation,
    )
    if not bool(observed):
        names = ", ".join(values)
        raise error_type(f"non-finite values in one of: {names}")


def _json_snapshot(value: Any, *, path: str, top_level: bool = True) -> Any:
    """Validate and copy a non-ambiguous JSON identity tree."""

    if value is None:
        if top_level:
            raise ValueError(f"{path} must not be null")
        return None
    if type(value) is bool:
        raise TypeError(f"{path} must not contain boolean identity values")
    if type(value) is str:
        if not value.strip():
            raise ValueError(f"{path} must not contain an empty string")
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite floats")
        return value
    if type(value) is list:
        if not value:
            raise ValueError(f"{path} must not contain an empty list")
        return [
            _json_snapshot(item, path=f"{path}[{index}]", top_level=False)
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if not value:
            raise ValueError(f"{path} must not contain an empty dictionary")
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            if not key.strip():
                raise ValueError(f"{path} keys must not be empty")
            result[key] = _json_snapshot(
                item, path=f"{path}.{key}", top_level=False
            )
        return result
    raise TypeError(
        f"{path} must be a JSON-safe host identity tree"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, init=False)
class IdentityAttestedERITHCFactors:
    """ERI-THC factors accompanied by immutable caller-attested tokens.

    ``ERITHCFactors`` intentionally carries only numerical arrays.  Complete
    assembly therefore requires this small wrapper to compare the caller's
    orbital/integral labels with F-hat provenance.  The wrapper freezes only
    those caller-attested identity trees as canonical JSON; it neither makes
    the numerical arrays immutable nor proves that their content originated
    from the labelled source.  The complete audit reports that limitation and
    remains ineligible for formal validation.
    """

    factors: ERITHCFactors
    _orbital_identity_json: str = field(repr=False)
    _integral_identity_json: str = field(repr=False)
    identity_token_digest: str

    def __init__(
        self,
        factors: ERITHCFactors,
        *,
        orbital_identity_token: Any,
        integral_identity_token: Any,
    ) -> None:
        if not isinstance(factors, ERITHCFactors):
            raise TypeError("factors must be an ERITHCFactors instance")
        orbital = _json_snapshot(
            orbital_identity_token, path="eri_factors.orbital_identity_token"
        )
        integral = _json_snapshot(
            integral_identity_token,
            path="eri_factors.integral_identity_token",
        )
        orbital_json = _canonical_json(orbital)
        integral_json = _canonical_json(integral)
        digest = _sha256_json(
            {"orbitals": orbital, "integrals": integral}
        )
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "_orbital_identity_json", orbital_json)
        object.__setattr__(self, "_integral_identity_json", integral_json)
        object.__setattr__(self, "identity_token_digest", digest)

    @property
    def orbital_identity_token(self) -> Any:
        return json.loads(self._orbital_identity_json)

    @property
    def integral_identity_token(self) -> Any:
        return json.loads(self._integral_identity_json)

    def metadata(self) -> dict[str, Any]:
        return {
            "binding": "caller-attested-identity-tokens-only",
            "orbital_identity_token": self.orbital_identity_token,
            "integral_identity_token": self.integral_identity_token,
            "identity_token_digest": self.identity_token_digest,
            "numerical_content_provenance_validated": False,
            "factor_arrays_immutable": False,
        }


def _match_fhat_identity(
    provenance: dict[str, Any],
    *,
    resource: str,
    supplied_token: Any,
) -> tuple[Any, str]:
    token = _json_snapshot(
        supplied_token, path=f"{resource}_identity_token"
    )
    candidates = []
    for key in _PROVENANCE_KEYS[resource]:
        if key in provenance and provenance[key] is not None:
            candidates.append((key, provenance[key]))
    if not candidates:
        raise ValueError(f"F-hat provenance does not bind {resource}")
    encoded = _canonical_json(token)
    matches = [key for key, value in candidates if _canonical_json(value) == encoded]
    if not matches:
        raise ValueError(
            f"{resource} identity token does not match F-hat provenance"
        )
    return token, matches[0]


def _validate_provenance(
    fhat: T1TransformedFHat,
    eri_factors: IdentityAttestedERITHCFactors,
    *,
    orbital_identity_token: Any,
    integral_identity_token: Any,
    hcore_identity_token: Any,
) -> tuple[dict[str, Any], Any, Any, Any, str, str, str, str, str]:
    if fhat.provenance_bound is not True:
        raise ValueError("F-hat provenance must be complete and bound")
    provenance = _json_snapshot(
        fhat.provenance_context, path="fhat.provenance_context"
    )
    for resource, keys in _PROVENANCE_KEYS.items():
        if not any(key in provenance and provenance[key] is not None for key in keys):
            raise ValueError(
                "F-hat provenance must bind orbitals, integrals, and hcore; "
                f"missing {resource}"
            )
    orbital_token, orbital_key = _match_fhat_identity(
        provenance,
        resource="orbitals",
        supplied_token=orbital_identity_token,
    )
    integral_token, integral_key = _match_fhat_identity(
        provenance,
        resource="integrals",
        supplied_token=integral_identity_token,
    )
    hcore_token, hcore_key = _match_fhat_identity(
        provenance,
        resource="hcore",
        supplied_token=hcore_identity_token,
    )
    if (
        _canonical_json(eri_factors.orbital_identity_token)
        != _canonical_json(orbital_token)
    ):
        raise ValueError(
            "ERI factors orbital identity token does not match F-hat and "
            "caller provenance"
        )
    if (
        _canonical_json(eri_factors.integral_identity_token)
        != _canonical_json(integral_token)
    ):
        raise ValueError(
            "ERI factors integral identity token does not match F-hat and "
            "caller provenance"
        )
    fhat_digest = _sha256_json(provenance)
    assembly_digest = _sha256_json(
        {
            "orbitals": orbital_token,
            "integrals": integral_token,
            "hcore": hcore_token,
            "fhat_provenance": provenance,
            "eri_identity_token_digest": eri_factors.identity_token_digest,
        }
    )
    return (
        provenance,
        orbital_token,
        integral_token,
        hcore_token,
        orbital_key,
        integral_key,
        hcore_key,
        fhat_digest,
        assembly_digest,
    )


def _validate_inputs(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    tau: Any,
    eri_factors: ERITHCFactors,
    fhat: T1TransformedFHat,
    *,
    transfer_counter: Any,
    amplitude_core_symmetry_tolerance: Any,
):
    if not isinstance(eri_factors, ERITHCFactors):
        raise TypeError("eri_factors must be an ERITHCFactors instance")
    if not isinstance(fhat, T1TransformedFHat):
        raise TypeError("fhat must be a frozen T1TransformedFHat instance")
    arrays = (
        y_vir,
        amplitude_core,
        tau,
        eri_factors.x_occ,
        eri_factors.x_vir,
        eri_factors.core,
        fhat.l_oo,
        fhat.l_vv,
        fhat.l_ov,
        fhat.l_vo,
        fhat.fhat_oo,
        fhat.fhat_vv,
        fhat.fhat_ov,
        fhat.fhat_vo,
    )
    xp = _one_backend_and_device(y_occ, *arrays)
    dtype = _one_real_fp64_dtype(y_occ, *arrays)
    _validate_counter(xp, transfer_counter)

    nested_counters = (
        ("eri_factors", eri_factors.transfer_counter),
        ("fhat", fhat.transfer_counter),
    )
    for name, nested in nested_counters:
        if nested is not None and nested is not transfer_counter:
            raise ValueError(
                f"{name} and complete THC audit must use the same "
                "transfer_counter"
            )

    if getattr(y_occ, "ndim", None) != 2:
        raise ValueError("y_occ must have shape (nocc,amplitude_rank)")
    if getattr(y_vir, "ndim", None) != 2:
        raise ValueError("y_vir must have shape (nvir,amplitude_rank)")
    if getattr(amplitude_core, "ndim", None) != 2:
        raise ValueError("amplitude_core must be a square matrix")
    nocc, rank = map(int, y_occ.shape)
    nvir, virtual_rank = map(int, y_vir.shape)
    if nocc < 1 or nvir < 1 or rank < 1:
        raise ValueError("complete THC audit dimensions must be positive")
    if virtual_rank != rank or amplitude_core.shape != (rank, rank):
        raise ValueError("amplitude THC factors and core shapes do not match")
    if getattr(tau, "ndim", None) != 2 or tau.shape[1] != rank:
        raise ValueError("tau must have shape (rr_rank,amplitude_rank)")
    if int(tau.shape[0]) < 1:
        raise ValueError("tau RR rank must be positive")

    if getattr(eri_factors.x_occ, "ndim", None) != 2:
        raise ValueError("eri_factors.x_occ must be a matrix")
    if getattr(eri_factors.x_vir, "ndim", None) != 2:
        raise ValueError("eri_factors.x_vir must be a matrix")
    if getattr(eri_factors.core, "ndim", None) != 2:
        raise ValueError("eri_factors.core must be a matrix")
    eri_rank = int(eri_factors.x_occ.shape[1])
    if eri_factors.x_occ.shape != (nocc, eri_rank):
        raise ValueError("ERI occupied factors do not match amplitude orbitals")
    if eri_factors.x_vir.shape != (nvir, eri_rank):
        raise ValueError("ERI virtual factors do not match amplitude orbitals")
    if eri_rank < 1 or eri_factors.core.shape != (eri_rank, eri_rank):
        raise ValueError("ERI THC core shape does not match its factors")

    if getattr(fhat.l_oo, "ndim", None) != 3:
        raise ValueError("fhat.l_oo must be a three-index tensor")
    naux = int(fhat.l_oo.shape[0])
    expected_shapes = {
        "l_oo": (naux, nocc, nocc),
        "l_vv": (naux, nvir, nvir),
        "l_ov": (naux, nocc, nvir),
        "l_vo": (naux, nvir, nocc),
        "fhat_oo": (nocc, nocc),
        "fhat_vv": (nvir, nvir),
        "fhat_ov": (nocc, nvir),
        "fhat_vo": (nvir, nocc),
    }
    if naux < 1:
        raise ValueError("F-hat auxiliary rank must be positive")
    for name, expected in expected_shapes.items():
        if getattr(fhat, name).shape != expected:
            raise ValueError(f"fhat.{name} must have shape {expected}")
    _positive_integer(
        fhat.auxiliary_block_size, name="fhat.auxiliary_block_size"
    )

    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="thc_complete_audit_input_finite",
        y_occ=y_occ,
        y_vir=y_vir,
        amplitude_core=amplitude_core,
        tau=tau,
        eri_x_occ=eri_factors.x_occ,
        eri_x_vir=eri_factors.x_vir,
        eri_core=eri_factors.core,
        fhat_l_oo=fhat.l_oo,
        fhat_l_vv=fhat.l_vv,
        fhat_l_ov=fhat.l_ov,
        fhat_l_vo=fhat.l_vo,
        fhat_oo=fhat.fhat_oo,
        fhat_vv=fhat.fhat_vv,
        fhat_ov=fhat.fhat_ov,
        fhat_vo=fhat.fhat_vo,
    )
    symmetry_tolerance = _nonnegative_real(
        amplitude_core_symmetry_tolerance,
        name="amplitude_core_symmetry_tolerance",
    )
    asymmetry = _read_control_scalar(
        xp,
        xp.max(xp.abs(amplitude_core - amplitude_core.T)),
        transfer_counter=transfer_counter,
        operation="thc_complete_audit_core_symmetry",
    )
    asymmetry = float(asymmetry)
    if asymmetry > symmetry_tolerance:
        raise ValueError(
            "amplitude_core must be symmetric within the explicit absolute "
            f"tolerance: max_asymmetry={asymmetry:.17g}, "
            f"tolerance={symmetry_tolerance:.17g}"
        )
    return xp, dtype, nocc, nvir, rank, asymmetry, symmetry_tolerance


def _validate_endpoint_outputs(
    reference: Any,
    *,
    rank: int,
    nocc: int,
    nvir: int,
    transfer_counter: Any,
    **values: Any,
) -> None:
    arrays = tuple(values.values())
    xp = _one_backend_and_device(reference, *arrays)
    _one_real_fp64_dtype(reference, *arrays)
    for name, value in values.items():
        expected = (nocc, nvir) if name.startswith("singles_") else (rank, rank)
        if getattr(value, "ndim", None) != 2 or value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="thc_complete_audit_endpoint_outputs",
        error_type=FloatingPointError,
        **values,
    )


def _validate_final_outputs(
    reference: Any,
    tau: Any,
    singles: Any,
    doubles_thc: Any,
    doubles_rr: Any,
    *,
    rank: int,
    nocc: int,
    nvir: int,
    transfer_counter: Any,
) -> None:
    xp = _one_backend_and_device(
        reference, singles, doubles_thc, doubles_rr
    )
    _one_real_fp64_dtype(reference, singles, doubles_thc, doubles_rr)
    expected = {
        "singles": (nocc, nvir),
        "doubles_thc": (rank, rank),
        "doubles_rr": (int(tau.shape[0]), int(tau.shape[0])),
    }
    values = {
        "singles": singles,
        "doubles_thc": doubles_thc,
        "doubles_rr": doubles_rr,
    }
    for name, shape in expected.items():
        value = values[name]
        if getattr(value, "ndim", None) != 2 or value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="thc_complete_audit_final_outputs",
        error_type=FloatingPointError,
        **values,
    )


@dataclass(frozen=True)
class THCCompleteAuditLedgerEntry:
    """One coefficient and coordinate-space record in the assembly ledger."""

    algorithm: int
    endpoint: str
    equations: tuple[int, ...]
    contribution: str
    coefficient: float
    selected: bool
    coordinate_space: str
    xor_group: str | None
    xor_branch: str | None
    individual_backprojection_count: int
    participates_in_shared_backprojection: bool
    pair_symmetrization: str | None
    pair_symmetrization_application_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": int(self.algorithm),
            "endpoint": self.endpoint,
            "equations": list(self.equations),
            "contribution": self.contribution,
            "coefficient": float(self.coefficient),
            "selected": bool(self.selected),
            "coordinate_space": self.coordinate_space,
            "xor_group": self.xor_group,
            "xor_branch": self.xor_branch,
            "backprojection_count": int(
                self.individual_backprojection_count
            ),
            "participates_in_shared_backprojection": bool(
                self.participates_in_shared_backprojection
            ),
            "pair_symmetrization": self.pair_symmetrization,
            "pair_symmetrization_application_count": int(
                self.pair_symmetrization_application_count
            ),
        }


@dataclass(frozen=True)
class THCCompleteAuditLedger:
    """Immutable term ledger for one complete audit assembly."""

    entries: tuple[THCCompleteAuditLedgerEntry, ...]
    omega_ac_selection: str
    rr_backprojection_count: int
    rr_backprojection_input_space: str
    rr_backprojection_output_space: str
    singles_backprojection_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.as_dict() for entry in self.entries],
            "equation_numbering_authority": (
                "J. Chem. Phys. 156, 054102 (2022), "
                "DOI:10.1063/5.0077770"
            ),
            "entry_equations_semantics": (
                "published equations whose tensors are directly emitted by "
                "each endpoint; dependencies consumed from earlier "
                "algorithms are not repeated"
            ),
            "legacy_arxiv_v1_direct_equation_numbers": {
                "algorithm8": [38, 39, 40],
                "algorithm9": [41],
                "algorithm10": [42],
            },
            "omega_ac_xor_group": "algorithms-4-plus-5-xor-algorithm-6",
            "omega_ac_selection": self.omega_ac_selection,
            "rr_backprojection_count": int(self.rr_backprojection_count),
            "rr_backprojection_input_space": self.rr_backprojection_input_space,
            "rr_backprojection_output_space": self.rr_backprojection_output_space,
            "singles_backprojection_count": int(
                self.singles_backprojection_count
            ),
        }


@dataclass(frozen=True)
class THCCompleteAuditResult:
    """Computed but unaccepted Algorithms 1--10 audit assembly."""

    singles: Any
    doubles_thc: Any
    doubles_rr: Any
    algorithm_1: Any
    algorithm_2: Any
    algorithm_3: Any
    algorithm_4: Any | None
    algorithm_5: Any | None
    algorithm_6: Any | None
    algorithm_7: THCOmegaDAlgorithm7Result
    algorithm_7_pair_symmetrized: Any
    algorithm_8: THCOmegaGHAlgorithm8Result
    algorithm_9: THCOmegaEAlgorithm9Result
    algorithm_10: THCOmegaIJAlgorithm10Result
    ledger: THCCompleteAuditLedger
    amplitude_core_max_asymmetry: float
    amplitude_core_symmetry_tolerance: float
    algorithm7_input_gauge_precondition_checked: bool
    _orbital_identity_json: str = field(repr=False, compare=False)
    _integral_identity_json: str = field(repr=False, compare=False)
    _hcore_identity_json: str = field(repr=False, compare=False)
    orbital_fhat_provenance_key: str
    integral_fhat_provenance_key: str
    hcore_fhat_provenance_key: str
    fhat_identity_context_digest: str
    eri_factors_identity_token_digest: str
    assembly_identity_token_digest: str
    _fhat_provenance_json: str = field(repr=False, compare=False)
    transfer_counter: Any = field(default=None, repr=False, compare=False)

    @property
    def accepted(self) -> bool:
        return False

    @property
    def production_enabled(self) -> bool:
        return False

    @property
    def complete_validated(self) -> bool:
        return False

    @property
    def complete_ccsd_residual(self) -> bool:
        return False

    @property
    def orbital_identity_token(self) -> Any:
        return json.loads(self._orbital_identity_json)

    @property
    def integral_identity_token(self) -> Any:
        return json.loads(self._integral_identity_json)

    @property
    def hcore_identity_token(self) -> Any:
        return json.loads(self._hcore_identity_json)

    @property
    def fhat_provenance_context(self) -> dict[str, Any]:
        return json.loads(self._fhat_provenance_json)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "gpu4pyscf.thc-complete-audit.v1",
            "paper": "Hohenstein-2022",
            "paper_source": (
                "J. Chem. Phys. 156, 054102 (2022), "
                "DOI:10.1063/5.0077770"
            ),
            "paper_source_version": "version-of-record",
            "covered_algorithms": list(range(1, 11)),
            "doubles_definition": (
                "A1+A2+A3+((A4+A5) XOR A6)+(A7+A7.T)+A9"
            ),
            "singles_definition": "A8.singles_gh+A10.singles_ij",
            "accepted": False,
            "production_enabled": False,
            "complete_validated": False,
            "complete_ccsd_residual": False,
            "formal_full_residual_eligible": False,
            "audit_only": True,
            "performance_eligible": False,
            "algorithm7_computed_value_retained": True,
            "algorithm7_raw_value_retained": True,
            "algorithm7_pair_symmetrized_value_retained": True,
            "algorithm7_pair_symmetrization_applied": True,
            "algorithm7_pair_symmetrization": "raw-plus-pair-transpose",
            "algorithm7_pair_symmetrization_scope": (
                "complete-residual-composition-only"
            ),
            "algorithm7_equation_version_contract_asserted": True,
            "algorithm7_input_gauge_precondition_checked": bool(
                self.algorithm7_input_gauge_precondition_checked
            ),
            "algorithm7_published_eq36_mapping": (
                "eq38-plus-algorithm7-line19-in-full-pair-gauge"
            ),
            "algorithm7_published_eq36_contract_scope": (
                "physical-full-pair-gauge"
            ),
            "algorithm7_published_eq36_status": (
                "version-of-record-contract-asserted"
            ),
            "algorithm7_instance_numerical_equivalence_audited": False,
            "algorithm7_instance_numerical_equivalence": None,
            "algorithm7_instance_numerical_equivalence_oracle": (
                "thc_omega_cd_joint_audit"
            ),
            "algorithm7_legacy_preprint_source": "arXiv:2111.11473v1",
            "algorithm7_legacy_preprint_eq35_equivalence": False,
            "algorithm7_legacy_preprint_eq35_status": (
                "known-version-difference"
            ),
            "require_eq35_equivalence_parameter_status": "deprecated-name",
            "require_eq35_equivalence_parameter_semantics": (
                "assert-version-of-record-eq36-contract-and-check-"
                "physical-full-pair-gauge-precondition"
            ),
            # Compatibility keys remain explicit about the old name.  The
            # legacy preprint expression itself is still not equivalent.
            "algorithm7_eq35_mapping": (
                "deprecated-name-mapped-to-version-of-record-eq36"
            ),
            "algorithm7_eq35_equivalence": False,
            "algorithm7_eq35_status": "legacy-preprint-version-difference",
            "algorithm7_sign_or_permutation_tuned": False,
            "amplitude_core_symmetry_required": True,
            "amplitude_core_max_asymmetry": float(
                self.amplitude_core_max_asymmetry
            ),
            "amplitude_core_symmetry_tolerance": float(
                self.amplitude_core_symmetry_tolerance
            ),
            "algorithm8_delta_xy_convention": (
                "published-algorithm8-line8-kronecker-identity"
            ),
            "algorithm8_delta_xy_convention_uniquely_validated": True,
            "algorithm8_delta_xy_pair_gram_alternative_rejected": True,
            "algorithm10_published_eq43_requires_symmetric_core": True,
            "algorithm10_symmetric_core_precondition_checked": True,
            "algorithm10_nonsymmetric_core_equivalence": False,
            "algorithm10_nonsymmetric_core_production_eligible": False,
            "fhat_identity_tokens_complete": True,
            "identity_binding": (
                "caller-fhat-and-eri-attestation-token-equivalence"
            ),
            "identity_binding_scope": "caller-attested-tokens-only",
            "numerical_content_provenance_validated": False,
            "factor_arrays_immutable": False,
            "orbital_identity_token": json.loads(
                json.dumps(self.orbital_identity_token, ensure_ascii=False)
            ),
            "integral_identity_token": json.loads(
                json.dumps(self.integral_identity_token, ensure_ascii=False)
            ),
            "hcore_identity_token": json.loads(
                json.dumps(self.hcore_identity_token, ensure_ascii=False)
            ),
            "orbital_fhat_provenance_key": self.orbital_fhat_provenance_key,
            "integral_fhat_provenance_key": self.integral_fhat_provenance_key,
            "hcore_fhat_provenance_key": self.hcore_fhat_provenance_key,
            "fhat_provenance_context": json.loads(
                json.dumps(self.fhat_provenance_context, ensure_ascii=False)
            ),
            "fhat_identity_context_digest": (
                self.fhat_identity_context_digest
            ),
            "eri_factors_identity_token_digest": (
                self.eri_factors_identity_token_digest
            ),
            "assembly_identity_token_digest": (
                self.assembly_identity_token_digest
            ),
            "implicit_host_tensor_transfer": False,
            "gpu_validation_requires_transfer_counter": True,
            "transfer_counter_provided": self.transfer_counter is not None,
            "ledger": self.ledger.as_dict(),
        }


def _ledger(omega_ac_path: str) -> THCCompleteAuditLedger:
    separate = omega_ac_path == "separate-4-plus-5"
    specifications = (
        (1, "thc_algorithm_1", (28,), "doubles", True, None),
        (2, "thc_algorithm_2", (29,), "doubles", True, None),
        (3, "thc_algorithm_3", (31,), "doubles", True, None),
        (4, "thc_omega_a_algorithm4", (32,), "doubles", separate, "separate-4-plus-5"),
        (5, "thc_omega_c_algorithm5", (33,), "doubles", separate, "separate-4-plus-5"),
        (6, "thc_omega_ac_algorithm6", (34,), "doubles", not separate, "joint-6"),
        (7, "thc_omega_d_algorithm7", (36, 37, 38), "doubles", True, None),
        (8, "thc_omega_gh_algorithm8", (39, 40, 41), "singles", True, None),
        (9, "thc_omega_e_algorithm9", (42,), "doubles", True, None),
        (10, "thc_omega_ij_algorithm10", (43,), "singles", True, None),
    )
    entries = []
    for (
        algorithm,
        endpoint,
        equations,
        contribution,
        selected,
        branch,
    ) in specifications:
        doubles = contribution == "doubles"
        entries.append(
            THCCompleteAuditLedgerEntry(
                algorithm=algorithm,
                endpoint=endpoint,
                equations=equations,
                contribution=contribution,
                coefficient=1.0,
                selected=selected,
                coordinate_space=(
                    "amplitude-thc-auxiliary"
                    if doubles
                    else "orbital-singles-ia"
                ),
                xor_group=(
                    "algorithms-4-plus-5-xor-algorithm-6"
                    if algorithm in (4, 5, 6)
                    else None
                ),
                xor_branch=branch,
                individual_backprojection_count=0,
                participates_in_shared_backprojection=doubles and selected,
                pair_symmetrization=(
                    "raw-plus-pair-transpose" if algorithm == 7 else None
                ),
                pair_symmetrization_application_count=(
                    1 if algorithm == 7 else 0
                ),
            )
        )
    return THCCompleteAuditLedger(
        entries=tuple(entries),
        omega_ac_selection=omega_ac_path,
        rr_backprojection_count=1,
        rr_backprojection_input_space="amplitude-thc-auxiliary",
        rr_backprojection_output_space="rr-core",
        singles_backprojection_count=0,
    )


def assemble_complete_thc_ccsd_audit(
    y_occ: Any,
    y_vir: Any,
    amplitude_core: Any,
    tau: Any,
    eri_factors: IdentityAttestedERITHCFactors,
    fhat: T1TransformedFHat,
    *,
    orbital_identity_token: Any,
    integral_identity_token: Any,
    hcore_identity_token: Any,
    omega_ac_path: str = "joint-6",
    algorithm4_5_outer_block_size: int = 1,
    algorithm6_x_block_size: int = 1,
    algorithm7_x_block_size: int = 1,
    algorithm8_cholesky_block_size: int = 1,
    amplitude_core_symmetry_tolerance: Real = 0.0,
    transfer_counter: Any = None,
    require_eq35_equivalence: bool = False,
) -> THCCompleteAuditResult:
    """Assemble the Algorithms 1--10 audit schedule from existing endpoints.

    ``omega_ac_path`` selects either separate Algorithms 4 and 5 or joint
    Algorithm 6.  It is an exclusive choice; no execution path can add both.

    ``require_eq35_equivalence`` is retained as a deprecated parameter name.
    When true it asserts the scoped version-of-record Eq. 36 contract
    (published Eq. 38 plus Algorithm 7 line 19) and checks that the factor
    cores satisfy the physical full-pair-gauge precondition.  It does not run
    the dense numerical equivalence audit for this instance; that evidence is
    provided only by ``thc_omega_cd_joint_audit``.  The deprecated name never
    requests the erroneous submitted-preprint Eq. 35 expression.

    Raw Algorithm 7 and its physical pair-symmetric composition are computed
    and retained, but the returned result can never be accepted or enabled
    for production by this audit endpoint.
    """

    require_eq35_equivalence = _boolean(
        require_eq35_equivalence, name="require_eq35_equivalence"
    )
    if type(omega_ac_path) is not str or omega_ac_path not in _OMEGA_AC_PATHS:
        raise ValueError(
            "omega_ac_path must be 'separate-4-plus-5' or 'joint-6'"
        )
    algorithm4_5_outer_block_size = _positive_integer(
        algorithm4_5_outer_block_size,
        name="algorithm4_5_outer_block_size",
    )
    algorithm6_x_block_size = _positive_integer(
        algorithm6_x_block_size, name="algorithm6_x_block_size"
    )
    algorithm7_x_block_size = _positive_integer(
        algorithm7_x_block_size, name="algorithm7_x_block_size"
    )
    algorithm8_cholesky_block_size = _positive_integer(
        algorithm8_cholesky_block_size,
        name="algorithm8_cholesky_block_size",
    )
    if not isinstance(eri_factors, IdentityAttestedERITHCFactors):
        raise TypeError(
            "eri_factors must be an IdentityAttestedERITHCFactors instance"
        )
    identity_attested_eri_factors = eri_factors
    eri_factors = identity_attested_eri_factors.factors

    (
        xp,
        dtype,
        nocc,
        nvir,
        rank,
        amplitude_core_max_asymmetry,
        amplitude_core_symmetry_tolerance,
    ) = _validate_inputs(
        y_occ,
        y_vir,
        amplitude_core,
        tau,
        eri_factors,
        fhat,
        transfer_counter=transfer_counter,
        amplitude_core_symmetry_tolerance=amplitude_core_symmetry_tolerance,
    )
    if require_eq35_equivalence:
        eri_core_asymmetry = _read_control_scalar(
            xp,
            xp.max(xp.abs(eri_factors.core - eri_factors.core.T)),
            transfer_counter=transfer_counter,
            operation="thc_complete_audit_eq36_eri_pair_symmetry",
        )
        eri_core_asymmetry = float(eri_core_asymmetry)
        if eri_core_asymmetry > amplitude_core_symmetry_tolerance:
            raise ValueError(
                "deprecated require_eq35_equivalence requests the "
                "version-of-record Eq. 36 identity in the physical "
                "full-pair gauge, but the ERI core is not symmetric within "
                "the explicit tolerance: "
                f"max_asymmetry={eri_core_asymmetry:.17g}, "
                f"tolerance={amplitude_core_symmetry_tolerance:.17g}"
            )
    (
        fhat_provenance_context,
        orbital_identity_token,
        integral_identity_token,
        hcore_identity_token,
        orbital_fhat_provenance_key,
        integral_fhat_provenance_key,
        hcore_fhat_provenance_key,
        fhat_identity_context_digest,
        assembly_identity_token_digest,
    ) = _validate_provenance(
        fhat,
        identity_attested_eri_factors,
        orbital_identity_token=orbital_identity_token,
        integral_identity_token=integral_identity_token,
        hcore_identity_token=hcore_identity_token,
    )

    transformed = T1TransformedCholesky(
        hoo=fhat.l_oo,
        hov=fhat.l_ov,
        hvv=fhat.l_vv,
        hvo=fhat.l_vo,
    )
    algorithm_1 = _residual.thc_algorithm_1(
        y_occ, y_vir, amplitude_core, transformed
    )
    algorithm_2 = _residual.thc_algorithm_2(
        y_occ, y_vir, amplitude_core, transformed
    )
    algorithm_3 = _residual.thc_algorithm_3(
        y_occ, y_vir, amplitude_core, transformed
    )

    algorithm_4 = None
    algorithm_5 = None
    algorithm_6 = None
    if omega_ac_path == "separate-4-plus-5":
        algorithm_4 = _eri.thc_omega_a_algorithm4(
            y_occ,
            y_vir,
            amplitude_core,
            eri_factors,
            outer_block_size=algorithm4_5_outer_block_size,
            transfer_counter=transfer_counter,
        )
        algorithm_5 = _eri.thc_omega_c_algorithm5(
            y_occ,
            y_vir,
            amplitude_core,
            eri_factors,
            outer_block_size=algorithm4_5_outer_block_size,
            transfer_counter=transfer_counter,
        )
        omega_ac = algorithm_4 + algorithm_5
    else:
        algorithm_6 = _eri.thc_omega_ac_algorithm6(
            y_occ,
            y_vir,
            amplitude_core,
            eri_factors,
            x_block_size=algorithm6_x_block_size,
            transfer_counter=transfer_counter,
        )
        omega_ac = algorithm_6

    algorithm_7 = _omega_d.thc_omega_d_algorithm7(
        y_occ,
        y_vir,
        amplitude_core,
        eri_factors,
        x_block_size=algorithm7_x_block_size,
        transfer_counter=transfer_counter,
        require_eq35_literal_equivalence=False,
    )
    if not isinstance(algorithm_7, THCOmegaDAlgorithm7Result):
        raise TypeError("Algorithm 7 endpoint returned an invalid result")
    algorithm7_metadata = algorithm_7.metadata()
    if (
        algorithm7_metadata.get(
            "published_eq36_literal_equivalence_in_full_pair_gauge"
        )
        is not True
        or algorithm7_metadata.get(
            "published_eq36_literal_equivalence_unconditional"
        )
        is not False
        or algorithm7_metadata.get("paper_source_version")
        != "version-of-record"
        or algorithm7_metadata.get("production_enabled") is not False
        or algorithm7_metadata.get("audit_only") is not True
    ):
        raise RuntimeError(
            "Algorithm 7 must provide the scoped version-of-record Eq. 36 "
            "algebra contract while remaining audit-only"
        )

    algorithm8_cholesky = T1TransformedCholeskyBlocks(
        l_oo=fhat.l_oo,
        l_vv=fhat.l_vv,
        l_ov=fhat.l_ov,
        transfer_counter=transfer_counter,
    )
    algorithm_8 = _omega_ghi.thc_omega_gh_algorithm8(
        y_occ,
        y_vir,
        amplitude_core,
        algorithm8_cholesky,
        cholesky_block_size=algorithm8_cholesky_block_size,
        transfer_counter=transfer_counter,
    )
    if not isinstance(algorithm_8, THCOmegaGHAlgorithm8Result):
        raise TypeError("Algorithm 8 endpoint returned an invalid result")
    algorithm8_metadata = algorithm_8.metadata()
    if (
        algorithm8_metadata.get("delta_xy_convention")
        != "published-algorithm8-line8-kronecker-identity"
        or algorithm8_metadata.get("delta_xy_convention_uniquely_validated")
        is not True
        or algorithm8_metadata.get("delta_xy_pair_gram_alternative_rejected")
        is not True
        or algorithm8_metadata.get("production_enabled") is not False
        or algorithm8_metadata.get("audit_only") is not True
    ):
        raise RuntimeError(
            "Algorithm 8 must provide the resolved published delta_XY "
            "contract while remaining audit-only"
        )
    algorithm_9 = _omega_ghi.thc_omega_e_algorithm9(
        y_occ,
        y_vir,
        amplitude_core,
        fhat.fhat_oo,
        fhat.fhat_vv,
        algorithm_8,
        transfer_counter=transfer_counter,
    )
    if not isinstance(algorithm_9, THCOmegaEAlgorithm9Result):
        raise TypeError("Algorithm 9 endpoint returned an invalid result")
    algorithm_10 = _omega_ghi.thc_omega_ij_algorithm10(
        y_occ,
        y_vir,
        amplitude_core,
        fhat.fhat_ov,
        fhat.fhat_vo,
        transfer_counter=transfer_counter,
    )
    if not isinstance(algorithm_10, THCOmegaIJAlgorithm10Result):
        raise TypeError("Algorithm 10 endpoint returned an invalid result")
    algorithm10_metadata = algorithm_10.metadata()
    if (
        algorithm10_metadata.get(
            "published_equation43_equivalence_requires_symmetric_amplitude_core"
        )
        is not True
        or algorithm10_metadata.get(
            "published_equation43_symmetric_core_endpoint_validated"
        )
        is not True
        or algorithm10_metadata.get(
            "published_equation43_nonsymmetric_core_equivalence"
        )
        is not False
        or algorithm10_metadata.get(
            "nonsymmetric_amplitude_core_production_eligible"
        )
        is not False
        or algorithm10_metadata.get(
            "appendix_algorithm10_nonsymmetric_core_mapping"
        )
        != "published-eq43-coulomb-uses-T-exchange-uses-T-transpose"
        or algorithm10_metadata.get("production_enabled") is not False
        or algorithm10_metadata.get("audit_only") is not True
    ):
        raise RuntimeError(
            "Algorithm 10 must provide the symmetric-core published Eq. 43 "
            "contract and reject nonsymmetric-core production promotion"
        )
    for name, result in (
        ("algorithm_8", algorithm_8),
        ("algorithm_9", algorithm_9),
        ("algorithm_10", algorithm_10),
    ):
        if result.transfer_counter is not transfer_counter:
            raise ValueError(
                f"{name} result did not retain the shared transfer_counter"
            )

    _validate_endpoint_outputs(
        y_occ,
        rank=rank,
        nocc=nocc,
        nvir=nvir,
        transfer_counter=transfer_counter,
        algorithm_1=algorithm_1,
        algorithm_2=algorithm_2,
        algorithm_3=algorithm_3,
        omega_ac=omega_ac,
        algorithm_7=algorithm_7.combined,
        algorithm_9=algorithm_9.omega_e,
        singles_algorithm_8=algorithm_8.singles_gh,
        singles_algorithm_10=algorithm_10.singles_ij,
    )

    # In the RR/CD complete-residual convention, the raw Algorithm 7 endpoint
    # needs its X<->X' counterpart.  Add that pair transpose explicitly at
    # composition.  The raw paper endpoint remains unchanged and is retained
    # alongside this value for the published Eq. 36 algebra audit.
    algorithm_7_pair_symmetrized = (
        algorithm_7.combined + algorithm_7.combined.T
    )
    _validate_endpoint_outputs(
        y_occ,
        rank=rank,
        nocc=nocc,
        nvir=nvir,
        transfer_counter=transfer_counter,
        algorithm_7_pair_symmetrized=algorithm_7_pair_symmetrized,
    )

    doubles_thc = xp.zeros((rank, rank), dtype=dtype)
    for contribution in (
        algorithm_1,
        algorithm_2,
        algorithm_3,
        omega_ac,
        algorithm_7_pair_symmetrized,
        algorithm_9.omega_e,
    ):
        doubles_thc += contribution

    # This is the only RR back-projection in the complete audit.  The child
    # endpoints all return amplitude-THC coordinates and are never projected
    # individually.
    doubles_rr = _residual.project_thc_residual_to_rr(doubles_thc, tau)
    singles = algorithm_8.singles_gh + algorithm_10.singles_ij
    _validate_final_outputs(
        y_occ,
        tau,
        singles,
        doubles_thc,
        doubles_rr,
        rank=rank,
        nocc=nocc,
        nvir=nvir,
        transfer_counter=transfer_counter,
    )

    ledger = _ledger(omega_ac_path)
    return THCCompleteAuditResult(
        singles=singles,
        doubles_thc=doubles_thc,
        doubles_rr=doubles_rr,
        algorithm_1=algorithm_1,
        algorithm_2=algorithm_2,
        algorithm_3=algorithm_3,
        algorithm_4=algorithm_4,
        algorithm_5=algorithm_5,
        algorithm_6=algorithm_6,
        algorithm_7=algorithm_7,
        algorithm_7_pair_symmetrized=algorithm_7_pair_symmetrized,
        algorithm_8=algorithm_8,
        algorithm_9=algorithm_9,
        algorithm_10=algorithm_10,
        ledger=ledger,
        amplitude_core_max_asymmetry=amplitude_core_max_asymmetry,
        amplitude_core_symmetry_tolerance=amplitude_core_symmetry_tolerance,
        algorithm7_input_gauge_precondition_checked=(
            require_eq35_equivalence
        ),
        _orbital_identity_json=_canonical_json(orbital_identity_token),
        _integral_identity_json=_canonical_json(integral_identity_token),
        _hcore_identity_json=_canonical_json(hcore_identity_token),
        orbital_fhat_provenance_key=orbital_fhat_provenance_key,
        integral_fhat_provenance_key=integral_fhat_provenance_key,
        hcore_fhat_provenance_key=hcore_fhat_provenance_key,
        fhat_identity_context_digest=fhat_identity_context_digest,
        eri_factors_identity_token_digest=(
            identity_attested_eri_factors.identity_token_digest
        ),
        assembly_identity_token_digest=assembly_identity_token_digest,
        _fhat_provenance_json=_canonical_json(fhat_provenance_context),
        transfer_counter=transfer_counter,
    )


__all__ = [
    "IdentityAttestedERITHCFactors",
    "THCCompleteAuditLedgerEntry",
    "THCCompleteAuditLedger",
    "THCCompleteAuditResult",
    "assemble_complete_thc_ccsd_audit",
]
