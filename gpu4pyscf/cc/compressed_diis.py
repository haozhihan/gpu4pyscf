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

"""Device-resident DIIS for full singles and an RR/THC doubles core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


@dataclass(frozen=True)
class CompressedDIISResult:
    t1: Any
    core: Any
    coefficients: Any
    history_size: int
    extrapolated: bool


class CompressedDIIS:
    """Pulay DIIS that never reconstructs or stores dense doubles amplitudes."""

    def __init__(
        self,
        *,
        space: int = 6,
        minimum_history: int = 2,
        regularization: float = 0.0,
    ) -> None:
        if int(space) != space or space < 2:
            raise ValueError("DIIS space must be an integer of at least two")
        if int(minimum_history) != minimum_history or not 2 <= minimum_history <= space:
            raise ValueError("minimum_history must lie in [2, space]")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        self.space = int(space)
        self.minimum_history = int(minimum_history)
        self.regularization = float(regularization)
        self._states: list[tuple[Any, Any]] = []
        self._errors: list[tuple[Any, Any]] = []
        self._shape: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._xp: Any = None
        self.extrapolations = 0

    @property
    def history_size(self) -> int:
        return len(self._states)

    @property
    def storage_nbytes(self) -> int:
        return int(
            sum(
                t1.nbytes + core.nbytes + err1.nbytes + errc.nbytes
                for (t1, core), (err1, errc) in zip(
                    self._states, self._errors
                )
            )
        )

    def reset(self) -> None:
        self._states.clear()
        self._errors.clear()
        self._shape = None
        self._xp = None
        self.extrapolations = 0

    def _validate(self, t1: Any, core: Any, error_t1: Any, error_core: Any) -> Any:
        xp = _array_module(t1)
        if any(_array_module(value) is not xp for value in (core, error_t1, error_core)):
            raise TypeError("DIIS state and error tensors must use one backend")
        if t1.shape != error_t1.shape or core.shape != error_core.shape:
            raise ValueError("DIIS state and error shapes must match")
        if core.ndim != 2 or core.shape[0] != core.shape[1]:
            raise ValueError("compressed doubles core must be square")
        shape = (tuple(t1.shape), tuple(core.shape))
        if self._shape is not None and shape != self._shape:
            raise ValueError("DIIS tensor shapes changed within one history")
        if self._xp is not None and xp is not self._xp:
            raise TypeError("DIIS backend changed within one history")
        self._shape = shape
        self._xp = xp
        return xp

    def update(
        self,
        t1: Any,
        core: Any,
        error_t1: Any,
        error_core: Any,
    ) -> CompressedDIISResult:
        """Append one resident state/error pair and extrapolate when ready."""

        xp = self._validate(t1, core, error_t1, error_core)
        self._states.append((t1.copy(), core.copy()))
        self._errors.append((error_t1.copy(), error_core.copy()))
        if self.history_size > self.space:
            del self._states[0]
            del self._errors[0]
        size = self.history_size
        if size < self.minimum_history:
            return CompressedDIISResult(
                t1=t1.copy(),
                core=core.copy(),
                coefficients=xp.ones((1,), dtype=t1.dtype),
                history_size=size,
                extrapolated=False,
            )

        dtype = xp.result_type(t1.dtype, core.dtype, np.float64)
        pulay = xp.empty((size + 1, size + 1), dtype=dtype)
        for row, (left_t1, left_core) in enumerate(self._errors):
            for column in range(row + 1):
                right_t1, right_core = self._errors[column]
                value = xp.vdot(left_t1, right_t1) + xp.vdot(
                    left_core, right_core
                )
                pulay[row, column] = value
                pulay[column, row] = value.conj()
        if self.regularization:
            diagonal = xp.arange(size)
            pulay[diagonal, diagonal] += self.regularization
        pulay[:size, size] = -1
        pulay[size, :size] = -1
        pulay[size, size] = 0
        rhs = xp.zeros((size + 1,), dtype=dtype)
        rhs[size] = -1
        coefficients = xp.linalg.solve(pulay, rhs)[:size]
        out_t1 = xp.zeros_like(t1, dtype=dtype)
        out_core = xp.zeros_like(core, dtype=dtype)
        for coefficient, (state_t1, state_core) in zip(
            coefficients, self._states
        ):
            out_t1 += coefficient * state_t1
            out_core += coefficient * state_core
        self.extrapolations += 1
        return CompressedDIISResult(
            t1=out_t1,
            core=out_core,
            coefficients=coefficients,
            history_size=size,
            extrapolated=True,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "representation": "full-singles-plus-compressed-doubles-core",
            "space": self.space,
            "minimum_history": self.minimum_history,
            "regularization": self.regularization,
            "history_size": self.history_size,
            "storage_nbytes": self.storage_nbytes,
            "backend": None if self._xp is None else self._xp.__name__,
            "extrapolations": int(self.extrapolations),
            "materializes_dense_t2": False,
        }


__all__ = ["CompressedDIIS", "CompressedDIISResult"]
