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

"""Shared, explicitly labelled CCSD residual evaluation helpers.

``update_amps`` returns a Jacobi amplitude candidate.  Its distance from the
current amplitudes is useful for solver diagnostics, but it is not an equation
residual.  The latter is obtained by multiplying the amplitude difference by
the occupied--virtual orbital-energy denominators.  Keeping those two norms in
one evaluator prevents validation and benchmark paths from silently swapping
their meanings.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Optional

import numpy as np


RESIDUAL_SCHEMA = "gpu4pyscf.cc.residual.v1"


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy  # pylint: disable=import-outside-toplevel

        return cupy
    return np


def _as_float(
    value: Any,
    *,
    scalar_to_float: Optional[Callable[..., float]] = None,
    operation: Optional[str] = None,
) -> float:
    if scalar_to_float is None:
        try:
            result = float(value.item())
        except AttributeError:
            result = float(value)
    else:
        if not operation:
            raise ValueError("a scalar transfer operation label is required")
        result = float(scalar_to_float(value, operation=operation))
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("a residual norm must be finite and non-negative")
    return result


def _norm(
    array: Any,
    xp: Any,
    *,
    scalar_to_float: Optional[Callable[..., float]],
    operation: str,
) -> float:
    return _as_float(
        xp.sqrt(xp.vdot(array, array).real),
        scalar_to_float=scalar_to_float,
        operation=operation,
    )


def _combined_norm(
    first: Any,
    second: Any,
    xp: Any,
    *,
    scalar_to_float: Optional[Callable[..., float]],
    operation: str,
) -> float:
    return _as_float(
        xp.sqrt(xp.vdot(first, first).real + xp.vdot(second, second).real),
        scalar_to_float=scalar_to_float,
        operation=operation,
    )


def _pair_matrix(t2: Any, nocc: int, nvir: int):
    return t2.transpose(0, 2, 1, 3).reshape(nocc * nvir, nocc * nvir)


def _blockwise_doubles_squared_norms(
    current_t2: Any,
    jacobi_t2: Any,
    eia: Any,
    xp: Any,
    block_size: int,
    block_observer: Optional[Callable[[int, int], None]] = None,
):
    """Accumulate doubles norms without materializing three full tensors."""

    nvir = int(current_t2.shape[2])
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("doubles_block_size must be positive")
    block_size = min(block_size, nvir)
    # Seed device scalars without a host scalar conversion.  Each loop keeps
    # only one virtual slice of delta, denominator, and equation residual.
    doubles_update_squared = xp.vdot(
        current_t2[:, :, :1, :], current_t2[:, :, :1, :]
    ).real * 0
    doubles_equation_squared = doubles_update_squared * 0
    for a0 in range(0, nvir, block_size):
        a1 = min(nvir, a0 + block_size)
        update_block = (
            jacobi_t2[:, :, a0:a1, :] - current_t2[:, :, a0:a1, :]
        )
        denominator_block = (
            eia[:, None, a0:a1, None] + eia[None, :, None, :]
        )
        equation_block = update_block * denominator_block
        if block_observer is not None:
            block_observer(a0, a1)
        doubles_update_squared = (
            doubles_update_squared
            + xp.vdot(update_block, update_block).real
        )
        doubles_equation_squared = (
            doubles_equation_squared
            + xp.vdot(equation_block, equation_block).real
        )
        update_block = denominator_block = equation_block = None
    return doubles_update_squared, doubles_equation_squared


@dataclass(frozen=True)
class DenseCCSDResidualEvaluation:
    """Norms from one dense CCSD Jacobi candidate at a fixed state.

    ``full_*`` values use the complete active ``ijab`` tensor.  Projected
    values use ``U.conj().T @ pair_tensor @ U`` and are absent when no RR
    projector was supplied.
    """

    full_space_equation_norm: float
    full_space_jacobi_update_norm: float
    singles_equation_norm: float
    full_space_doubles_equation_norm: float
    singles_jacobi_update_norm: float
    full_space_doubles_jacobi_update_norm: float
    projected_equation_norm: Optional[float] = None
    projected_jacobi_update_norm: Optional[float] = None
    projected_doubles_equation_norm: Optional[float] = None
    projected_doubles_jacobi_update_norm: Optional[float] = None


def evaluate_dense_ccsd_residual(
    current_t1: Any,
    current_t2: Any,
    jacobi_t1: Any,
    jacobi_t2: Any,
    occupied_energies: Any,
    virtual_energies: Any,
    *,
    level_shift: float = 0.0,
    projector_vectors: Any = None,
    scalar_to_float: Optional[Callable[..., float]] = None,
    doubles_block_size: Optional[int] = None,
    block_observer: Optional[Callable[[int, int], None]] = None,
) -> DenseCCSDResidualEvaluation:
    """Evaluate equation and Jacobi-update norms without conflating them.

    The dense equation residual is

    ``R1 = (t1_jacobi - t1) * eia`` and
    ``R2 = (t2_jacobi - t2) * (eia[ia] + eia[jb])``.

    When ``projector_vectors`` is supplied, the doubles parts of both tensors
    are projected *after* forming the full tensors.  In particular, the
    projected equation residual is ``U^H R2 U``; multiplying a projected
    Jacobi update by a projected denominator is generally not equivalent.

    ``scalar_to_float`` is an optional accounted device-to-host boundary.  It
    receives each final norm scalar plus a stable ``operation`` label.  Tensor
    algebra stays in the array module selected by ``current_t1``.
    ``doubles_block_size`` performs the same FP64 norm reduction by virtual
    slices, avoiding full-size delta, denominator, and equation temporaries.
    ``block_observer`` runs while all three slice temporaries are live, which
    lets a benchmark synchronously sample allocator/driver high-water usage.
    """

    if getattr(current_t1, "ndim", None) != 2:
        raise ValueError("current_t1 must have shape (nocc,nvir)")
    if getattr(jacobi_t1, "shape", None) != current_t1.shape:
        raise ValueError("jacobi_t1 must match current_t1")
    nocc, nvir = (int(value) for value in current_t1.shape)
    expected_t2_shape = (nocc, nocc, nvir, nvir)
    if getattr(current_t2, "shape", None) != expected_t2_shape:
        raise ValueError(f"current_t2 must have shape {expected_t2_shape}")
    if getattr(jacobi_t2, "shape", None) != expected_t2_shape:
        raise ValueError("jacobi_t2 must match current_t2")
    if getattr(occupied_energies, "shape", None) != (nocc,):
        raise ValueError("occupied_energies must match nocc")
    if getattr(virtual_energies, "shape", None) != (nvir,):
        raise ValueError("virtual_energies must match nvir")
    level_shift = float(level_shift)
    if not math.isfinite(level_shift):
        raise ValueError("level_shift must be finite")

    xp = _array_module(current_t1)
    update_t1 = jacobi_t1 - current_t1
    eia = (
        occupied_energies[:, None]
        - virtual_energies[None, :]
        - level_shift
    )
    equation_t1 = update_t1 * eia

    if doubles_block_size is None:
        if block_observer is not None:
            raise ValueError(
                "block_observer requires a blockwise doubles reduction"
            )
        update_t2 = jacobi_t2 - current_t2
        eijab = eia[:, None, :, None] + eia[None, :, None, :]
        equation_t2 = update_t2 * eijab
        singles_equation_norm = _norm(
            equation_t1, xp, scalar_to_float=scalar_to_float,
            operation="singles-equation-norm",
        )
        full_doubles_equation_norm = _norm(
            equation_t2, xp, scalar_to_float=scalar_to_float,
            operation="doubles-equation-norm",
        )
        singles_update_norm = _norm(
            update_t1, xp, scalar_to_float=scalar_to_float,
            operation="singles-jacobi-update-norm",
        )
        full_doubles_update_norm = _norm(
            update_t2, xp, scalar_to_float=scalar_to_float,
            operation="doubles-jacobi-update-norm",
        )
        full_equation_norm = _combined_norm(
            equation_t1, equation_t2, xp,
            scalar_to_float=scalar_to_float,
            operation="full-equation-norm",
        )
        full_update_norm = _combined_norm(
            update_t1, update_t2, xp,
            scalar_to_float=scalar_to_float,
            operation="full-jacobi-update-norm",
        )
    else:
        if projector_vectors is not None:
            raise ValueError(
                "blockwise doubles reduction does not support a projector"
            )
        singles_equation_squared = xp.vdot(
            equation_t1, equation_t1
        ).real
        singles_update_squared = xp.vdot(update_t1, update_t1).real
        (
            doubles_update_squared,
            doubles_equation_squared,
        ) = _blockwise_doubles_squared_norms(
            current_t2,
            jacobi_t2,
            eia,
            xp,
            doubles_block_size,
            block_observer,
        )

        def read_squared(value: Any, operation: str) -> float:
            return _as_float(
                xp.sqrt(value),
                scalar_to_float=scalar_to_float,
                operation=operation,
            )

        singles_equation_norm = read_squared(
            singles_equation_squared, "singles-equation-norm"
        )
        full_doubles_equation_norm = read_squared(
            doubles_equation_squared, "doubles-equation-norm"
        )
        singles_update_norm = read_squared(
            singles_update_squared, "singles-jacobi-update-norm"
        )
        full_doubles_update_norm = read_squared(
            doubles_update_squared, "doubles-jacobi-update-norm"
        )
        full_equation_norm = read_squared(
            singles_equation_squared + doubles_equation_squared,
            "full-equation-norm",
        )
        full_update_norm = read_squared(
            singles_update_squared + doubles_update_squared,
            "full-jacobi-update-norm",
        )
        update_t2 = equation_t2 = None

    projected_equation_norm = None
    projected_update_norm = None
    projected_doubles_equation_norm = None
    projected_doubles_update_norm = None
    if projector_vectors is not None:
        pair_dimension = nocc * nvir
        if getattr(projector_vectors, "ndim", None) != 2:
            raise ValueError("projector_vectors must be a rank-two array")
        if int(projector_vectors.shape[0]) != pair_dimension:
            raise ValueError("projector row dimension does not match nocc*nvir")
        projected_equation = (
            projector_vectors.T.conj()
            @ _pair_matrix(equation_t2, nocc, nvir)
            @ projector_vectors
        )
        projected_update = (
            projector_vectors.T.conj()
            @ _pair_matrix(update_t2, nocc, nvir)
            @ projector_vectors
        )
        projected_doubles_equation_norm = _norm(
            projected_equation, xp, scalar_to_float=scalar_to_float,
            operation="projected-doubles-equation-norm",
        )
        projected_doubles_update_norm = _norm(
            projected_update, xp, scalar_to_float=scalar_to_float,
            operation="projected-doubles-jacobi-update-norm",
        )
        projected_equation_norm = _combined_norm(
            equation_t1, projected_equation, xp,
            scalar_to_float=scalar_to_float,
            operation="projected-equation-norm",
        )
        projected_update_norm = _combined_norm(
            update_t1, projected_update, xp,
            scalar_to_float=scalar_to_float,
            operation="projected-jacobi-update-norm",
        )

    return DenseCCSDResidualEvaluation(
        full_space_equation_norm=full_equation_norm,
        full_space_jacobi_update_norm=full_update_norm,
        singles_equation_norm=singles_equation_norm,
        full_space_doubles_equation_norm=full_doubles_equation_norm,
        singles_jacobi_update_norm=singles_update_norm,
        full_space_doubles_jacobi_update_norm=full_doubles_update_norm,
        projected_equation_norm=projected_equation_norm,
        projected_jacobi_update_norm=projected_update_norm,
        projected_doubles_equation_norm=projected_doubles_equation_norm,
        projected_doubles_jacobi_update_norm=projected_doubles_update_norm,
    )


_MEASUREMENT_SEMANTICS = {
    "projected_equation": {
        "kind": "equation-residual",
        "space": "rr-projected-active-pair",
        "is_full_space": False,
    },
    "full_space_equation": {
        "kind": "equation-residual",
        "space": "full-active-pair",
        "is_full_space": True,
    },
    "projected_jacobi_update": {
        "kind": "jacobi-update",
        "space": "rr-projected-active-pair",
        "is_full_space": False,
    },
    "full_space_jacobi_update": {
        "kind": "jacobi-update",
        "space": "full-active-pair",
        "is_full_space": True,
    },
    "representation_error": {
        "kind": "relative-representation-error",
        "space": "full-active-pair",
        "is_full_space": True,
    },
}


def _measurement(name: str, value: Any) -> dict[str, Any]:
    if name not in _MEASUREMENT_SEMANTICS:  # pragma: no cover - private invariant
        raise KeyError(name)
    norm = None if value is None else _as_float(value)
    return {
        "available": norm is not None,
        "norm": norm,
        **_MEASUREMENT_SEMANTICS[name],
    }


def labeled_low_rank_residuals(
    *,
    projected_equation: Any = None,
    full_space_equation: Any = None,
    projected_jacobi_update: Any = None,
    full_space_jacobi_update: Any = None,
    representation_error: Any = None,
) -> dict[str, Any]:
    """Build the fail-closed residual record used by RR/THC metadata.

    The top-level ``projected_equation`` scalar is retained for benchmark
    analyzer compatibility.  Every authoritative value also appears under
    ``measurements`` with explicit space and kind labels.  There is deliberately
    no generic top-level ``norm`` field.
    """

    values = {
        "projected_equation": projected_equation,
        "full_space_equation": full_space_equation,
        "projected_jacobi_update": projected_jacobi_update,
        "full_space_jacobi_update": full_space_jacobi_update,
        "representation_error": representation_error,
    }
    measurements = {
        name: _measurement(name, value) for name, value in values.items()
    }
    acceptance = measurements["projected_equation"]
    return {
        "schema": RESIDUAL_SCHEMA,
        "available": bool(acceptance["available"]),
        "acceptance_measurement": "projected_equation",
        "projected_equation": acceptance["norm"],
        "measurements": measurements,
    }


__all__ = [
    "RESIDUAL_SCHEMA",
    "DenseCCSDResidualEvaluation",
    "evaluate_dense_ccsd_residual",
    "labeled_low_rank_residuals",
]
