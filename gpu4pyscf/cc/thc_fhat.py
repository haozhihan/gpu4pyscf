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

"""Audit-only T1-transformed inactive F-hat construction.

This module evaluates the one-body object consumed by the audit endpoints for
Algorithms 8--10 of Hohenstein *et al.*, J. Chem. Phys. **156**, 054102
(2022), DOI 10.1063/5.0077770 (arXiv:2111.11473v1).  Following the inactive
T1 transformation used in Koch *et al.*, the left and right transformations
are

``P = [[I, -t1], [0, I]]`` and ``H = [[I, 0], [t1.T, I]]``.

For a raw symmetric Cholesky vector ``L[A]`` and one-electron matrix ``h``,

``hhat = P.T @ h @ H`` and ``Lhat[A] = P.T @ L[A] @ H``.

The inactive transformed Fock matrix is then

``Fhat[p,q] = hhat[p,q] + sum(k occ) (2*g[k,k,p,q] - g[k,q,p,k])``

with ``g[p,q,r,s] = sum(A) Lhat[A,p,q] * Lhat[A,r,s]``.  The implementation
contracts the three-index factors in auxiliary blocks and never materializes
``g``.  It is deliberately separate from the production THC-RR driver and
does not include any T2-dependent contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from numbers import Real
from typing import Any

import numpy as np


_PROVENANCE_RESOURCE_KEYS = {
    "orbitals": ("orbital_identity", "orbital_fingerprint"),
    "integrals": ("integral_identity", "integral_fingerprint"),
    "hcore": ("hcore_identity", "hcore_fingerprint"),
}


def _array_module(value: Any):
    """Return the NumPy/CuPy namespace without moving ``value``."""

    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy" and isinstance(value, np.ndarray):
        return np
    if module == "cupy":
        import cupy

        if isinstance(value, cupy.ndarray):
            return cupy
    raise TypeError("T1 F-hat tensors must be NumPy or CuPy arrays")


def _one_backend_and_device(reference: Any, *values: Any):
    xp = _array_module(reference)
    if any(_array_module(value) is not xp for value in values):
        raise TypeError("all T1 F-hat tensors must use one array backend")
    if xp is not np:
        device_id = int(reference.device.id)
        if any(int(value.device.id) != device_id for value in values):
            raise TypeError("all T1 F-hat CuPy tensors must use one device")
        current_device_id = int(xp.cuda.runtime.getDevice())
        if current_device_id != device_id:
            raise RuntimeError(
                "current CuPy device does not match the T1 F-hat input "
                f"device: current={current_device_id}, input={device_id}"
            )
    return xp


def _require_real_fp64(*values: Any) -> np.dtype:
    dtypes = tuple(np.dtype(value.dtype) for value in values)
    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise TypeError("all T1 F-hat tensors must have the same dtype")
    dtype = dtypes[0]
    if np.issubdtype(dtype, np.complexfloating):
        raise NotImplementedError("T1 F-hat construction currently requires real RHF")
    if dtype != np.dtype(np.float64):
        raise TypeError("T1 F-hat tensors must use real FP64")
    return dtype


def _validate_gpu_counter(xp: Any, transfer_counter: Any) -> None:
    if xp is np:
        return
    if transfer_counter is None:
        raise ValueError(
            "GPU T1 F-hat validation requires an explicit transfer_counter"
        )
    if not callable(getattr(transfer_counter, "record_d2h", None)):
        raise TypeError("transfer_counter must provide record_d2h")


def _read_control_scalar(
    xp: Any,
    value: Any,
    *,
    transfer_counter: Any,
    operation: str,
) -> Any:
    """Read and account for one validation scalar on a GPU."""

    scalar = xp.asarray(value)
    if scalar.ndim != 0:
        raise ValueError("internal T1 F-hat control value must be scalar")
    if xp is not np:
        transfer_counter.record_d2h(int(scalar.nbytes), operation=operation)
    return scalar.item() if hasattr(scalar, "item") else scalar


def _require_finite(
    xp: Any,
    *,
    transfer_counter: Any,
    operation: str,
    **values: Any,
) -> int:
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
        raise ValueError(f"non-finite values in one of: {names}")
    return int(xp is not np)


def _positive_block_size(value: Any, naux: int) -> int:
    if value is None:
        return int(naux)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("auxiliary_block_size must be an integer or None")
    value = int(value)
    if value < 1:
        raise ValueError("auxiliary_block_size must be positive")
    return min(value, int(naux))


def _has_provenance_value(value: Any) -> bool:
    return value is not None


def _validate_json_safe_host_tree(value: Any, *, path: str) -> None:
    """Reject objects whose identity cannot be represented as plain JSON."""

    if value is None:
        return
    if type(value) is bool:
        raise TypeError(f"{path} must not contain boolean identity values")
    if type(value) is str:
        if not value.strip():
            raise ValueError(f"{path} must not contain an empty string")
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite floats")
        return
    if type(value) is list:
        if not value:
            raise ValueError(f"{path} must not contain an empty list")
        for index, item in enumerate(value):
            _validate_json_safe_host_tree(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        if not value:
            raise ValueError(f"{path} must not contain an empty dictionary")
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            if not key.strip():
                raise ValueError(f"{path} keys must not be empty")
            _validate_json_safe_host_tree(item, path=f"{path}.{key}")
        return
    raise TypeError(
        f"{path} must be a JSON-safe host tree of dict/list/string/int/"
        "finite-float/null values"
    )


def _validate_provenance_context(value: Any) -> tuple[dict[str, Any], bool]:
    """Freeze a host provenance snapshot and require all three identities.

    Each resource may be represented by an ``*_identity`` value, an
    ``*_fingerprint`` value, or both.  A partially specified context is more
    dangerous than no context because it could appear to bind the result to a
    complete calculation, so partial contexts fail closed.
    """

    if value is None:
        return {}, False
    if type(value) is not dict:
        raise TypeError("provenance_context must be a JSON-safe host dictionary")
    _validate_json_safe_host_tree(value, path="provenance_context")
    try:
        snapshot = json.loads(
            json.dumps(value, allow_nan=False, ensure_ascii=False)
        )
    except (TypeError, ValueError) as exc:  # defensive parity with validator
        raise TypeError("provenance_context must round-trip through JSON") from exc

    missing = []
    for resource, keys in _PROVENANCE_RESOURCE_KEYS.items():
        if not any(
            key in snapshot and _has_provenance_value(snapshot[key])
            for key in keys
        ):
            missing.append(resource)
    if missing:
        raise ValueError(
            "provenance_context must bind orbitals, integrals, and hcore; "
            "missing " + ", ".join(missing)
        )
    return snapshot, True


def _validate_empirical_symmetry_tolerance(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(
            "empirical_raw_cholesky_symmetry_tolerance must be a real scalar"
        )
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "empirical_raw_cholesky_symmetry_tolerance must be finite and "
            "non-negative"
        )
    return tolerance


def _validate_inputs(
    h_mo: Any,
    l_mo: Any,
    t1: Any,
    *,
    auxiliary_block_size: Any,
    transfer_counter: Any,
    provenance_context: Any,
    empirical_raw_cholesky_symmetry_tolerance: Any,
):
    xp = _one_backend_and_device(h_mo, l_mo, t1)
    dtype = _require_real_fp64(h_mo, l_mo, t1)
    _validate_gpu_counter(xp, transfer_counter)

    if getattr(t1, "ndim", None) != 2:
        raise ValueError("t1 must have shape (nocc,nvir)")
    nocc, nvir = map(int, t1.shape)
    if nocc < 1 or nvir < 1:
        raise ValueError("T1 F-hat occupied and virtual dimensions must be positive")
    nmo = nocc + nvir
    if getattr(h_mo, "ndim", None) != 2 or h_mo.shape != (nmo, nmo):
        raise ValueError("h_mo must have shape (nocc+nvir,nocc+nvir)")
    if getattr(l_mo, "ndim", None) != 3:
        raise ValueError("l_mo must have shape (naux,nmo,nmo)")
    naux = int(l_mo.shape[0])
    if naux < 1 or l_mo.shape[1:] != (nmo, nmo):
        raise ValueError("l_mo must have shape (naux,nocc+nvir,nocc+nvir)")
    block_size = _positive_block_size(auxiliary_block_size, naux)
    provenance_snapshot, provenance_bound = _validate_provenance_context(
        provenance_context
    )
    raw_l_tolerance = _validate_empirical_symmetry_tolerance(
        empirical_raw_cholesky_symmetry_tolerance
    )

    host_scalar_reads = _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="t1_fhat_input_finite",
        h_mo=h_mo,
        l_mo=l_mo,
        t1=t1,
    )
    raw_l_scale = _read_control_scalar(
        xp,
        xp.max(xp.abs(l_mo)),
        transfer_counter=transfer_counter,
        operation="t1_fhat_raw_cholesky_scale",
    )
    host_scalar_reads += int(xp is not np)
    raw_l_max_asymmetry = _read_control_scalar(
        xp,
        xp.max(xp.abs(l_mo - l_mo.swapaxes(1, 2))),
        transfer_counter=transfer_counter,
        operation="t1_fhat_raw_cholesky_asymmetry",
    )
    host_scalar_reads += int(xp is not np)
    raw_l_scale = float(raw_l_scale)
    raw_l_max_asymmetry = float(raw_l_max_asymmetry)
    if raw_l_max_asymmetry > raw_l_tolerance:
        raise ValueError(
            "raw l_mo Cholesky symmetry error exceeds the user-supplied "
            "empirical absolute "
            f"tolerance: max_asymmetry={raw_l_max_asymmetry:.17g}, "
            f"tolerance={raw_l_tolerance:.17g}, scale={raw_l_scale:.17g}"
        )

    # Always construct the exact symmetric representative accepted by the
    # tolerance gate.  Downstream blocks therefore cannot depend on which
    # triangle inherited the harmless MO-transform roundoff.
    symmetric_l_mo = (l_mo + l_mo.swapaxes(1, 2)) * 0.5

    return (
        xp,
        dtype,
        nocc,
        nvir,
        naux,
        block_size,
        host_scalar_reads,
        symmetric_l_mo,
        raw_l_scale,
        raw_l_max_asymmetry,
        raw_l_tolerance,
        provenance_snapshot,
        provenance_bound,
    )


@dataclass(frozen=True)
class T1TransformedFHat:
    """T1-transformed Cholesky and inactive F-hat orbital blocks.

    ``l_oo``, ``l_vv``, ``l_ov``, and ``l_vo`` retain the auxiliary axis as
    their first dimension.  No symmetry is assumed after the non-unitary T1
    transformation.  ``fhat_*`` are the corresponding four orbital blocks.
    """

    l_oo: Any
    l_vv: Any
    l_ov: Any
    l_vo: Any
    fhat_oo: Any
    fhat_vv: Any
    fhat_ov: Any
    fhat_vo: Any
    auxiliary_block_size: int
    host_scalar_reads: int
    raw_cholesky_scale: float
    raw_cholesky_max_asymmetry: float
    raw_cholesky_empirical_absolute_tolerance: float
    raw_cholesky_resymmetrized: bool
    provenance_context: dict[str, Any] = field(repr=False, compare=False)
    provenance_bound: bool
    transfer_counter: Any = field(default=None, repr=False, compare=False)

    @property
    def naux(self) -> int:
        return int(self.l_oo.shape[0])

    @property
    def nocc(self) -> int:
        return int(self.l_oo.shape[1])

    @property
    def nvir(self) -> int:
        return int(self.l_vv.shape[1])

    @property
    def storage_nbytes(self) -> int:
        arrays = (
            self.l_oo,
            self.l_vv,
            self.l_ov,
            self.l_vo,
            self.fhat_oo,
            self.fhat_vv,
            self.fhat_ov,
            self.fhat_vo,
        )
        return sum(int(value.nbytes) for value in arrays)

    def metadata(self) -> dict[str, Any]:
        xp = _array_module(self.l_oo)
        return {
            "representation": "T1-transformed-inactive-F-hat",
            "paper": "Hohenstein-2022-and-Koch-1994",
            "paper_source": "arXiv:2111.11473v1",
            "left_transformation": "P.T with P=[[I,-t1],[0,I]]",
            "right_transformation": "H with H=[[I,0],[t1.T,I]]",
            "inactive_fock_definition": (
                "hhat_pq+sum_k_occ(2*g_kkpq-g_kqpk)"
            ),
            "naux": self.naux,
            "nocc": self.nocc,
            "nvir": self.nvir,
            "backend": xp.__name__,
            "dtype": np.dtype(self.l_oo.dtype).name,
            "auxiliary_block_size": int(self.auxiliary_block_size),
            "host_scalar_reads": int(self.host_scalar_reads),
            "storage_nbytes": int(self.storage_nbytes),
            "audit_only": True,
            "production_enabled": False,
            "performance_eligible": False,
            "gpu_performance_claim": False,
            "includes_t1": True,
            "includes_t2": False,
            "inactive_occupied_contraction_only": True,
            "raw_cholesky_symmetry_required": True,
            "raw_cholesky_symmetry_policy": (
                "exact-provider-output-by-default-or-user-supplied-"
                "empirical-absolute-tolerance"
            ),
            "raw_cholesky_scale": float(self.raw_cholesky_scale),
            "raw_cholesky_max_asymmetry": float(
                self.raw_cholesky_max_asymmetry
            ),
            "raw_cholesky_empirical_absolute_tolerance": float(
                self.raw_cholesky_empirical_absolute_tolerance
            ),
            "raw_cholesky_tolerance_source": (
                "default-exact"
                if self.raw_cholesky_empirical_absolute_tolerance == 0.0
                else "user-supplied-empirical"
            ),
            "raw_cholesky_tolerance_is_proven_error_bound": False,
            "raw_cholesky_resymmetrized": bool(
                self.raw_cholesky_resymmetrized
            ),
            "raw_one_electron_symmetry_enforced": False,
            "transformed_cholesky_symmetry_enforced": False,
            "algorithm_consumers": [8, 9, 10],
            "algorithm8_cholesky_blocks": ["oo", "vv", "ov"],
            "algorithm9_fhat_blocks": ["oo", "vv"],
            "algorithm10_fhat_blocks": ["ov", "vo"],
            "materializes_four_index_eri": False,
            "materializes_dense_t2": False,
            "auxiliary_blocked_contractions": True,
            "returned_cholesky_blocks_are_full": True,
            "implicit_host_tensor_transfer": False,
            "gpu_validation_requires_transfer_counter": True,
            "provenance_bound": bool(self.provenance_bound),
            "provenance_context": json.loads(
                json.dumps(self.provenance_context, ensure_ascii=False)
            ),
            "provenance_required_resources": [
                "orbitals",
                "integrals",
                "hcore",
            ],
            "formal_full_residual_validated": False,
            "formal_full_residual_eligible": False,
        }


def build_t1_transformed_fhat(
    h_mo: Any,
    l_mo: Any,
    t1: Any,
    *,
    auxiliary_block_size: int | None = None,
    transfer_counter: Any = None,
    provenance_context: dict[str, Any] | None = None,
    empirical_raw_cholesky_symmetry_tolerance: float = 0.0,
) -> T1TransformedFHat:
    """Build transformed Cholesky blocks and the inactive F-hat.

    Parameters
    ----------
    h_mo
        Real FP64 one-electron matrix with shape ``(nmo,nmo)``.
    l_mo
        Real FP64 symmetric raw Cholesky vectors with shape
        ``(naux,nmo,nmo)``.
    t1
        Real FP64 RHF singles amplitudes with shape ``(nocc,nvir)``.
    auxiliary_block_size
        Number of Cholesky vectors contracted at once.  ``None`` selects all
        vectors.  Blocking limits contraction workspaces; all transformed
        three-index blocks remain part of the returned audit result.
    transfer_counter
        Required for CuPy input.  The four fail-closed scalar reads are
        explicitly recorded through ``record_d2h``.
    provenance_context
        Optional host audit metadata.  It must bind each of orbitals,
        integrals, and hcore through an ``*_identity`` or ``*_fingerprint``
        entry.  The copied mapping is returned verbatim in metadata.  Omitting
        it leaves ``provenance_bound`` false; supplying it does not by itself
        qualify a formal full-residual result.
    empirical_raw_cholesky_symmetry_tolerance
        User-supplied absolute tolerance for raw-factor asymmetry.  The
        default, zero, requires exact symmetric provider output.  A nonzero
        value is an empirical acceptance policy, not a floating-point error
        bound.  Accepted factors are explicitly symmetrized before use.
    """

    (
        xp,
        dtype,
        nocc,
        nvir,
        naux,
        block_size,
        host_scalar_reads,
        l_mo,
        raw_l_scale,
        raw_l_max_asymmetry,
        raw_l_tolerance,
        provenance_snapshot,
        provenance_bound,
    ) = _validate_inputs(
        h_mo,
        l_mo,
        t1,
        auxiliary_block_size=auxiliary_block_size,
        transfer_counter=transfer_counter,
        provenance_context=provenance_context,
        empirical_raw_cholesky_symmetry_tolerance=(
            empirical_raw_cholesky_symmetry_tolerance
        ),
    )

    occupied = slice(0, nocc)
    virtual = slice(nocc, nocc + nvir)
    t1_transpose = t1.T

    # Apply P.T @ h @ H in blocks.  Keep the intermediate transformed oo
    # block because it appears in the lower-left block as a whole.
    hhat_oo = h_mo[occupied, occupied] + xp.matmul(
        h_mo[occupied, virtual], t1_transpose
    )
    hhat_ov = h_mo[occupied, virtual].copy()
    hhat_vo = (
        h_mo[virtual, occupied]
        + xp.matmul(h_mo[virtual, virtual], t1_transpose)
        - xp.matmul(t1_transpose, hhat_oo)
    )
    hhat_vv = h_mo[virtual, virtual] - xp.matmul(
        t1_transpose, h_mo[occupied, virtual]
    )

    l_oo = xp.empty((naux, nocc, nocc), dtype=dtype)
    l_vv = xp.empty((naux, nvir, nvir), dtype=dtype)
    l_ov = xp.empty((naux, nocc, nvir), dtype=dtype)
    l_vo = xp.empty((naux, nvir, nocc), dtype=dtype)

    fhat_oo = hhat_oo.copy()
    fhat_vv = hhat_vv.copy()
    fhat_ov = hhat_ov.copy()
    fhat_vo = hhat_vo.copy()

    for start in range(0, naux, block_size):
        stop = min(start + block_size, naux)
        raw = l_mo[start:stop]
        raw_oo = raw[:, occupied, occupied]
        raw_ov = raw[:, occupied, virtual]
        raw_vo = raw[:, virtual, occupied]
        raw_vv = raw[:, virtual, virtual]

        block_oo = raw_oo + xp.matmul(raw_ov, t1_transpose)
        block_ov = raw_ov.copy()
        block_vo = (
            raw_vo
            + xp.matmul(raw_vv, t1_transpose)
            - xp.matmul(t1_transpose, block_oo)
        )
        block_vv = raw_vv - xp.matmul(t1_transpose, raw_ov)

        l_oo[start:stop] = block_oo
        l_ov[start:stop] = block_ov
        l_vo[start:stop] = block_vo
        l_vv[start:stop] = block_vv

        occupied_trace = xp.einsum("Lii->L", block_oo)
        fhat_oo += 2.0 * xp.einsum(
            "L,Lij->ij", occupied_trace, block_oo
        ) - xp.einsum("Lkj,Lik->ij", block_oo, block_oo)
        fhat_ov += 2.0 * xp.einsum(
            "L,Lia->ia", occupied_trace, block_ov
        ) - xp.einsum("Lka,Lik->ia", block_ov, block_oo)
        fhat_vo += 2.0 * xp.einsum(
            "L,Laj->aj", occupied_trace, block_vo
        ) - xp.einsum("Lkj,Lak->aj", block_oo, block_vo)
        fhat_vv += 2.0 * xp.einsum(
            "L,Lab->ab", occupied_trace, block_vv
        ) - xp.einsum("Lkb,Lak->ab", block_ov, block_vo)

    host_scalar_reads += _require_finite(
        xp,
        transfer_counter=transfer_counter,
        operation="t1_fhat_output_finite",
        l_oo=l_oo,
        l_vv=l_vv,
        l_ov=l_ov,
        l_vo=l_vo,
        fhat_oo=fhat_oo,
        fhat_vv=fhat_vv,
        fhat_ov=fhat_ov,
        fhat_vo=fhat_vo,
    )

    return T1TransformedFHat(
        l_oo=l_oo,
        l_vv=l_vv,
        l_ov=l_ov,
        l_vo=l_vo,
        fhat_oo=fhat_oo,
        fhat_vv=fhat_vv,
        fhat_ov=fhat_ov,
        fhat_vo=fhat_vo,
        auxiliary_block_size=block_size,
        host_scalar_reads=host_scalar_reads,
        raw_cholesky_scale=raw_l_scale,
        raw_cholesky_max_asymmetry=raw_l_max_asymmetry,
        raw_cholesky_empirical_absolute_tolerance=raw_l_tolerance,
        raw_cholesky_resymmetrized=True,
        provenance_context=provenance_snapshot,
        provenance_bound=provenance_bound,
        transfer_counter=transfer_counter,
    )


__all__ = ["T1TransformedFHat", "build_t1_transformed_fhat"]
