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

"""Direct GPU GINT columns for selected packed AO pairs.

This module is the bounded-memory successor to the restricted-task adapter in
``gint_pair_columns``.  A host-side schedule maps every original spherical AO
pair to exactly one screened GINT shell-pair task.  The CUDA entry points then
evaluate only selected right-hand shell pairs and transform each shell quartet
directly into packed FP64 columns or a packed diagonal.

The first release deliberately fails closed unless every original AO is
supported by exactly one segmented Cartesian shell.  It never materializes an
AO four-index tensor, the square AO-pair matrix, or a per-pivot AO matrix.
Performance eligibility remains false until the water2 numerical, transfer,
and timing gates have run against a rebuilt CUDA library.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, fields
from typing import Any, Iterable, Sequence

import numpy as np

from gpu4pyscf.cc.gint_pair_columns import (
    SortedAOPairMap,
    original_ao_shell_support,
)


_COLUMNS_SYMBOL = "GINTfill_selected_int2e_columns"
_DIAGONAL_SYMBOL = "GINTfill_selected_int2e_diagonal"
_RELEASE_SCOPE = "spherical-original-aos-with-single-shell-support"
_PERFORMANCE_GATE = "pending-water2-cuda-parity-transfer-and-timing-validation"


def _readonly(array: Any, dtype: Any, *, name: str, ndim: int = 1) -> np.ndarray:
    raw = np.asarray(array)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    target = np.dtype(dtype)
    if np.issubdtype(target, np.integer):
        if raw.dtype.kind not in "iu":
            raise TypeError(f"{name} must contain integers")
        limits = np.iinfo(target)
        if raw.size and (int(raw.min()) < limits.min or int(raw.max()) > limits.max):
            raise OverflowError(f"{name} exceeds the {target.name} ABI")
    result = np.asarray(raw, dtype=target)
    result = np.ascontiguousarray(result)
    result.flags.writeable = False
    return result


def _checked_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def _normalise_selected_pairs(
    pivots: Iterable[int], *, dimension: int, max_batch_size: int
) -> np.ndarray:
    """Return an int32 pivot vector while enforcing bounded column batches."""

    if isinstance(pivots, np.ndarray):
        raw = pivots
    else:
        try:
            raw = np.asarray(tuple(pivots))
        except TypeError as exc:
            raise TypeError("pivots must be an iterable of integers") from exc
    if raw.ndim != 1:
        raise ValueError("pivots must be one-dimensional")
    if raw.size == 0:
        raise ValueError("pivots must contain at least one pair index")
    if raw.dtype.kind not in "iu":
        raise TypeError("pivots must contain integers")
    if raw.dtype.kind == "u" and raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        raise OverflowError("pivot index exceeds the supported integer range")
    as_int64 = raw.astype(np.int64, copy=False)
    if np.any(as_int64 < 0) or np.any(as_int64 >= dimension):
        raise IndexError(f"pivot indices must lie in [0, {dimension})")
    if raw.size > max_batch_size:
        raise ValueError(
            f"selected-column batch has {raw.size} pivots; maximum is "
            f"{max_batch_size}"
        )
    # Allocating dimension columns is itself a square pair matrix, even when
    # duplicate pivots happen to be present.  Reject by allocation shape.
    if dimension > 1 and raw.size >= dimension:
        raise MemoryError(
            "a selected-column batch may not allocate the full AO-pair matrix"
        )
    result = np.ascontiguousarray(as_int64, dtype=np.int32)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class SelectedPairSchedule:
    """Immutable host schedule mirrored by compact device arrays."""

    nao_original: int
    nbas: int
    npair: int
    ncp: int
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_cp_id: np.ndarray
    pair_task_id: np.ndarray
    pair_symmetrize: np.ndarray
    cp_task_offsets: np.ndarray
    task_log_q: np.ndarray
    shell_original_offsets: np.ndarray
    shell_original_aos: np.ndarray

    def __post_init__(self) -> None:
        expected_pair = (self.npair,)
        for name in (
            "pair_i",
            "pair_j",
            "pair_cp_id",
            "pair_task_id",
            "pair_symmetrize",
        ):
            value = getattr(self, name)
            if value.shape != expected_pair:
                raise ValueError(f"{name} has an inconsistent pair dimension")
        if self.cp_task_offsets.shape != (self.ncp + 1,):
            raise ValueError("cp_task_offsets must contain ncp + 1 entries")
        if self.shell_original_offsets.shape != (self.nbas + 1,):
            raise ValueError("shell_original_offsets must contain nbas + 1 entries")
        if int(self.cp_task_offsets[-1]) != self.task_log_q.size:
            raise ValueError("task_log_q does not cover the cp task schedule")
        if int(self.shell_original_offsets[-1]) != self.nao_original:
            raise ValueError("shell AO support must cover each original AO once")

    @property
    def scheduled_pair_count(self) -> int:
        return int(np.count_nonzero(self.pair_task_id >= 0))

    @property
    def screened_pair_count(self) -> int:
        return int(self.npair - self.scheduled_pair_count)

    @property
    def host_nbytes(self) -> int:
        return int(
            sum(
                getattr(self, item.name).nbytes
                for item in fields(self)
                if isinstance(getattr(self, item.name), np.ndarray)
            )
        )

    def pairs_for_cp(self, cp_id: int) -> np.ndarray:
        cp_id = _checked_int(
            cp_id, name="cp id", minimum=0, maximum=self.ncp - 1
        )
        result = np.flatnonzero(
            (self.pair_cp_id == cp_id) & (self.pair_task_id >= 0)
        ).astype(np.int32, copy=False)
        result.flags.writeable = False
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "nao_original": int(self.nao_original),
            "nbas": int(self.nbas),
            "npair": int(self.npair),
            "ncp": int(self.ncp),
            "task_count": int(self.task_log_q.size),
            "scheduled_pair_count": self.scheduled_pair_count,
            "screened_pair_count": self.screened_pair_count,
            "single_shell_support": True,
            "host_schedule_bytes": self.host_nbytes,
        }


def build_selected_pair_schedule(
    coeff: Any,
    shell_ao_locs: Sequence[int],
    shell_to_group: Sequence[int],
    group_i: Sequence[int],
    group_j: Sequence[int],
    pair2bra: Sequence[Sequence[int]],
    pair2ket: Sequence[Sequence[int]],
    log_qs: Sequence[Sequence[float]],
    q_cond: Any,
    *,
    direct_scf_tol: float,
) -> SelectedPairSchedule:
    """Build the selected-pair schedule without importing CuPy.

    This function is intentionally pure NumPy so the release-scope and task
    inventory invariants are testable on a CPU-only development host.
    """

    coeff = np.asarray(coeff, dtype=np.float64)
    if coeff.ndim != 2 or coeff.shape[0] < 1 or coeff.shape[1] < 1:
        raise ValueError("coefficient map must be a non-empty matrix")
    if not np.all(np.isfinite(coeff)):
        raise ValueError("coefficient map must be finite")
    if not np.isfinite(direct_scf_tol) or not 0 < direct_scf_tol <= 1:
        raise ValueError("direct_scf_tol must be finite and lie in (0, 1]")

    shell_ao_locs = _readonly(
        shell_ao_locs, np.int64, name="shell AO locations"
    )
    if (
        shell_ao_locs.size < 2
        or shell_ao_locs[0] != 0
        or shell_ao_locs[-1] != coeff.shape[0]
        or np.any(np.diff(shell_ao_locs) <= 0)
    ):
        raise ValueError("shell AO locations must partition coefficient rows")
    nbas = int(shell_ao_locs.size - 1)
    shell_to_group = _readonly(
        shell_to_group, np.int32, name="shell-to-group map"
    )
    if shell_to_group.shape != (nbas,) or np.any(shell_to_group < 0):
        raise ValueError("shell-to-group map has an inconsistent shape")

    group_i = _readonly(group_i, np.int32, name="cp first groups")
    group_j = _readonly(group_j, np.int32, name="cp second groups")
    ncp = int(group_i.size)
    if ncp < 1 or group_j.shape != group_i.shape:
        raise ValueError("cp group arrays must be non-empty and aligned")
    if np.any(group_i < group_j):
        raise ValueError("GINT contraction groups must use lower-triangle order")
    if not (
        len(pair2bra) == len(pair2ket) == len(log_qs) == ncp
    ):
        raise ValueError("GINT cp task metadata lengths disagree")

    q_cond = np.asarray(q_cond, dtype=np.float64)
    if q_cond.shape != (nbas, nbas) or not np.all(np.isfinite(q_cond)):
        raise ValueError("q_cond must be a finite nbas by nbas matrix")
    if np.any(q_cond < 0):
        raise ValueError("q_cond must be non-negative")

    supports = original_ao_shell_support(coeff, shell_ao_locs)
    if any(len(shells) != 1 for shells in supports):
        raise NotImplementedError(
            "selected GINT columns require each original AO to have exactly "
            "one segmented-shell support"
        )

    shell_original_offsets = np.zeros(nbas + 1, dtype=np.int32)
    shell_lists: list[list[int]] = [[] for _ in range(nbas)]
    for original_ao, support in enumerate(supports):
        shell_lists[int(support[0])].append(original_ao)
    shell_original_aos_list: list[int] = []
    for shell, original_aos in enumerate(shell_lists):
        shell_original_aos_list.extend(original_aos)
        shell_original_offsets[shell + 1] = len(shell_original_aos_list)
    shell_original_aos = np.asarray(shell_original_aos_list, dtype=np.int32)

    cp_task_offsets = np.zeros(ncp + 1, dtype=np.int32)
    flat_logs: list[np.ndarray] = []
    task_lookup: dict[tuple[int, int], tuple[int, int, float]] = {}
    for cp_id in range(ncp):
        bra = np.asarray(pair2bra[cp_id])
        ket = np.asarray(pair2ket[cp_id])
        logs = np.asarray(log_qs[cp_id], dtype=np.float64)
        if bra.ndim != 1 or ket.shape != bra.shape or logs.shape != bra.shape:
            raise ValueError(f"cp {cp_id} shell-pair task arrays disagree")
        if bra.dtype.kind not in "iu" or ket.dtype.kind not in "iu":
            raise TypeError("GINT shell-pair tasks must contain integers")
        if not np.all(np.isfinite(logs)):
            raise ValueError("GINT task log_q values must be finite")
        bra = bra.astype(np.int64, copy=False)
        ket = ket.astype(np.int64, copy=False)
        if (
            np.any(bra < 0)
            or np.any(bra >= nbas)
            or np.any(ket < 0)
            or np.any(ket >= nbas)
        ):
            raise ValueError("GINT task names an out-of-range shell")
        for task_id, (ish, jsh, log_q) in enumerate(zip(bra, ket, logs)):
            ish, jsh = int(ish), int(jsh)
            if (
                int(shell_to_group[ish]) != int(group_i[cp_id])
                or int(shell_to_group[jsh]) != int(group_j[cp_id])
            ):
                raise ValueError("GINT task shell does not match its cp group")
            key = (ish, jsh)
            if key in task_lookup:
                raise ValueError("GINT shell-pair task is not unique")
            expected = min(float(np.log(q_cond[ish, jsh])), 0.0)
            if not np.isclose(float(log_q), expected, rtol=1e-12, atol=1e-14):
                raise ValueError("GINT task log_q does not match q_cond")
            task_lookup[key] = (cp_id, task_id, float(log_q))
        flat_logs.append(np.ascontiguousarray(logs, dtype=np.float64))
        task_count = int(cp_task_offsets[cp_id]) + int(logs.size)
        if task_count > np.iinfo(np.int32).max:
            raise OverflowError("GINT task schedule exceeds the int32 ABI")
        cp_task_offsets[cp_id + 1] = task_count

    pair_map = SortedAOPairMap(int(coeff.shape[1]))
    if pair_map.dimension > np.iinfo(np.int32).max:
        raise OverflowError("packed AO-pair dimension exceeds the int32 ABI")
    pair_i = np.asarray(pair_map.pair_i, dtype=np.int32)
    pair_j = np.asarray(pair_map.pair_j, dtype=np.int32)
    pair_cp_id = np.full(pair_map.dimension, -1, dtype=np.int32)
    pair_task_id = np.full(pair_map.dimension, -1, dtype=np.int32)
    pair_symmetrize = np.zeros(pair_map.dimension, dtype=np.uint8)

    for pair_id, (first, second) in enumerate(zip(pair_i, pair_j)):
        first_shell = int(supports[int(first)][0])
        second_shell = int(supports[int(second)][0])
        first_group = int(shell_to_group[first_shell])
        second_group = int(shell_to_group[second_shell])
        if first_group < second_group:
            key = (second_shell, first_shell)
        else:
            key = (first_shell, second_shell)
        task = task_lookup.get(key)
        if task is None:
            if float(q_cond[first_shell, second_shell]) > direct_scf_tol:
                raise RuntimeError(
                    "an unscreened original-AO support shell pair has no "
                    "GINT task"
                )
            continue
        cp_id, task_id, _ = task
        pair_cp_id[pair_id] = cp_id
        pair_task_id[pair_id] = task_id
        pair_symmetrize[pair_id] = np.uint8(first_group != second_group)

    arrays = {
        "pair_i": pair_i,
        "pair_j": pair_j,
        "pair_cp_id": pair_cp_id,
        "pair_task_id": pair_task_id,
        "pair_symmetrize": pair_symmetrize,
        "cp_task_offsets": cp_task_offsets,
        "task_log_q": (
            np.concatenate(flat_logs)
            if flat_logs
            else np.empty(0, dtype=np.float64)
        ),
        "shell_original_offsets": shell_original_offsets,
        "shell_original_aos": shell_original_aos,
    }
    for value in arrays.values():
        value.flags.writeable = False
    return SelectedPairSchedule(
        nao_original=pair_map.nao,
        nbas=nbas,
        npair=pair_map.dimension,
        ncp=ncp,
        **arrays,
    )


class _CSelectedPairData(ctypes.Structure):
    _fields_ = [
        ("npair", ctypes.c_int),
        ("nao_original", ctypes.c_int),
        ("nbas", ctypes.c_int),
        ("spherical", ctypes.c_int),
        ("single_shell_support", ctypes.c_int),
        ("abi_version", ctypes.c_int),
        ("coeff_rows", ctypes.c_int),
        ("shell_original_aos_count", ctypes.c_int),
        ("task_count", ctypes.c_int),
        ("coeff", ctypes.c_void_p),
        ("shell_original_offsets", ctypes.c_void_p),
        ("shell_original_aos", ctypes.c_void_p),
        ("pair_i", ctypes.c_void_p),
        ("pair_j", ctypes.c_void_p),
        ("pair_cp_id", ctypes.c_void_p),
        ("pair_task_id", ctypes.c_void_p),
        ("pair_symmetrize", ctypes.c_void_p),
        ("task_log_q", ctypes.c_void_p),
    ]


class _BasisProdCacheConstantLayout(ctypes.Structure):
    _fields_ = [
        ("nbas", ctypes.c_int),
        ("ncptype", ctypes.c_int),
        *[(f"pointer_{index}", ctypes.c_void_p) for index in range(17)],
    ]


_BPCACHE_CONSTANT_BYTES = ctypes.sizeof(_BasisProdCacheConstantLayout)


class GINTSelectedAOPairColumnProvider:
    """Selected-shell-pair C-ABI provider returning resident GPU FP64 arrays."""

    factorization = "gint-selected-shell-pair-columns"
    backend = "cupy-gint-selected-cabi"
    performance_eligible = False
    materializes_pair_matrix = False
    materializes_four_index_tensor = False
    materializes_per_pivot_ao_matrix = False

    def __init__(
        self,
        mol: Any,
        *,
        direct_scf_tol: float = 1e-13,
        group_size: int = 16,
        max_batch_size: int = 32,
        transfer_counter: Any = None,
        omega: float = 0.0,
    ) -> None:
        try:
            import cupy
            from gpu4pyscf.scf import int4c2e
        except (ImportError, OSError) as exc:  # pragma: no cover - GPU host only
            raise RuntimeError(
                "selected GINT columns require CuPy and GPU4PySCF GINT"
            ) from exc
        if getattr(mol, "nao_nr", None) is None:
            raise TypeError("mol must be a built PySCF Mole-like object")
        if bool(getattr(mol, "cart", False)):
            raise NotImplementedError(
                "selected GINT columns support spherical original AOs only"
            )
        if not np.isfinite(direct_scf_tol) or not 0 < direct_scf_tol <= 1:
            raise ValueError("direct_scf_tol must be finite and lie in (0, 1]")
        group_size = _checked_int(
            group_size,
            name="group_size",
            minimum=1,
            maximum=np.iinfo(np.int32).max,
        )
        max_batch_size = _checked_int(
            max_batch_size,
            name="max_batch_size",
            minimum=1,
            maximum=np.iinfo(np.int32).max,
        )
        if not np.isfinite(omega):
            raise ValueError("omega must be finite")
        if transfer_counter is None:
            from gpu4pyscf.cc.device_runtime import TransferCounter

            transfer_counter = TransferCounter()

        self._cupy = cupy
        self._int4c2e = int4c2e
        self.mol = mol
        self.direct_scf_tol = float(direct_scf_tol)
        self.group_size = group_size
        self.max_batch_size = max_batch_size
        self.omega = float(omega)
        self.transfer_counter = transfer_counter
        self._transfer_bytes = {"h2d": {}, "d2h": {}}
        self._transfer_counts = {"h2d": {}, "d2h": {}}

        vhfopt = int4c2e._VHFOpt(mol, "int2e")
        vhfopt.build(
            cutoff=self.direct_scf_tol,
            group_size=self.group_size,
            diag_block_with_triu=True,
        )
        maximum_l = int(vhfopt.uniq_l_ctr[:, 0].max())
        if maximum_l > int4c2e.LMAX_ON_GPU:
            raise NotImplementedError(
                "selected GINT columns do not mix CPU high-l integrals"
            )
        self._maximum_rys_order = 2 * maximum_l + 1
        if self._maximum_rys_order > 8:
            raise NotImplementedError(
                "selected GINT columns currently support Rys orders 1 through 8"
            )
        self.vhfopt = vhfopt
        self.diag_block_with_triu = True
        self.pair_map = SortedAOPairMap(int(mol.nao_nr()))
        self._coeff = cupy.ascontiguousarray(vhfopt.coeff, dtype=cupy.float64)
        if self._coeff.ndim != 2 or self._coeff.shape[1] != self.pair_map.nao:
            raise RuntimeError("_VHFOpt coefficient map has an unexpected shape")

        shell_offsets = np.asarray(vhfopt.l_ctr_offsets, dtype=np.int64)
        shell_to_group = np.empty(vhfopt.mol.nbas, dtype=np.int32)
        for group, (start, stop) in enumerate(
            zip(shell_offsets[:-1], shell_offsets[1:])
        ):
            shell_to_group[int(start) : int(stop)] = group
        group_i, group_j = np.tril_indices(len(vhfopt.uniq_l_ctr))
        coeff_host = cupy.asnumpy(self._coeff)
        self._record_transfer(
            "d2h", "selected_schedule_coeff_support", int(coeff_host.nbytes)
        )
        log_q_host = []
        log_q_bytes = 0
        for values in vhfopt.log_qs:
            host = cupy.asnumpy(values).astype(np.float64, copy=False)
            log_q_host.append(host)
            log_q_bytes += int(host.nbytes)
        self._record_transfer(
            "d2h",
            "selected_schedule_log_q",
            log_q_bytes,
            count=len(log_q_host),
        )
        q_cond = np.asarray(vhfopt.get_q_cond(), dtype=np.float64)
        self.schedule = build_selected_pair_schedule(
            coeff_host,
            np.asarray(vhfopt.mol.ao_loc_nr(cart=True), dtype=np.int64),
            shell_to_group,
            group_i,
            group_j,
            vhfopt.pair2bra,
            vhfopt.pair2ket,
            log_q_host,
            q_cond,
            direct_scf_tol=self.direct_scf_tol,
        )
        del coeff_host, log_q_host
        if self.schedule.npair != self.pair_map.dimension:
            raise RuntimeError("selected schedule has an inconsistent pair dimension")
        self._group_i = np.asarray(group_i, dtype=np.int32)
        self._group_j = np.asarray(group_j, dtype=np.int32)

        self._device_schedule: dict[str, Any] = {}
        for name in (
            "shell_original_offsets",
            "shell_original_aos",
            "pair_i",
            "pair_j",
            "pair_cp_id",
            "pair_task_id",
            "pair_symmetrize",
            "task_log_q",
        ):
            host = getattr(self.schedule, name)
            self._device_schedule[name] = cupy.asarray(host)
        self._scheduled_pairs_host, self._scheduled_cp_offsets = (
            self._build_diagonal_device_schedule()
        )
        self._scheduled_pairs_device = cupy.asarray(self._scheduled_pairs_host)
        schedule_h2d = sum(
            int(array.nbytes) for array in self._device_schedule.values()
        ) + int(self._scheduled_pairs_device.nbytes)
        self._record_transfer(
            "h2d",
            "selected_device_schedule",
            schedule_h2d,
            count=len(self._device_schedule) + 1,
        )

        device = self._device_schedule
        self._c_schedule = _CSelectedPairData(
            npair=self.schedule.npair,
            nao_original=self.schedule.nao_original,
            nbas=self.schedule.nbas,
            spherical=1,
            single_shell_support=1,
            abi_version=1,
            coeff_rows=int(self._coeff.shape[0]),
            shell_original_aos_count=int(self.schedule.shell_original_aos.size),
            task_count=int(self.schedule.task_log_q.size),
            coeff=ctypes.c_void_p(int(self._coeff.data.ptr)),
            shell_original_offsets=ctypes.c_void_p(
                int(device["shell_original_offsets"].data.ptr)
            ),
            shell_original_aos=ctypes.c_void_p(
                int(device["shell_original_aos"].data.ptr)
            ),
            pair_i=ctypes.c_void_p(int(device["pair_i"].data.ptr)),
            pair_j=ctypes.c_void_p(int(device["pair_j"].data.ptr)),
            pair_cp_id=ctypes.c_void_p(int(device["pair_cp_id"].data.ptr)),
            pair_task_id=ctypes.c_void_p(int(device["pair_task_id"].data.ptr)),
            pair_symmetrize=ctypes.c_void_p(
                int(device["pair_symmetrize"].data.ptr)
            ),
            task_log_q=ctypes.c_void_p(int(device["task_log_q"].data.ptr)),
        )
        self._columns_fn, self._diagonal_fn = self._bind_c_abi()

        self.batch_calls = 0
        self.column_calls = 0
        self.diagonal_calls = 0
        self.cabi_calls = 0
        self.kernel_launches = 0
        self._maximum_batch_observed = 0
        self._diagonal = None

    @property
    def dimension(self) -> int:
        return self.schedule.npair

    @property
    def pair_indices_device(self) -> tuple[Any, Any]:
        """Resident lower-triangle original-AO pair indices.

        The direct-CD driver consumes these arrays when it unpacks the packed
        Cholesky factors.  Returning the schedule-owned device arrays keeps
        that operation on the GPU and gives the selected C-ABI provider the
        same narrow contract as the restricted-task reference provider.
        """

        return (
            self._device_schedule["pair_i"],
            self._device_schedule["pair_j"],
        )

    def _record_transfer(
        self, direction: str, operation: str, nbytes: int, *, count: int = 1
    ) -> None:
        nbytes, count = int(nbytes), int(count)
        if direction not in ("h2d", "d2h") or nbytes < 0 or count < 0:
            raise ValueError("invalid transfer record")
        self._transfer_bytes[direction][operation] = (
            self._transfer_bytes[direction].get(operation, 0) + nbytes
        )
        self._transfer_counts[direction][operation] = (
            self._transfer_counts[direction].get(operation, 0) + count
        )
        self.transfer_counter.record(
            direction, nbytes, count=count, operation=operation
        )

    def _build_diagonal_device_schedule(self) -> tuple[np.ndarray, np.ndarray]:
        groups = [self.schedule.pairs_for_cp(cp) for cp in range(self.schedule.ncp)]
        offsets = np.zeros(self.schedule.ncp + 1, dtype=np.int32)
        for cp, values in enumerate(groups):
            offsets[cp + 1] = offsets[cp] + values.size
        flat = (
            np.concatenate(groups).astype(np.int32, copy=False)
            if groups
            else np.empty(0, dtype=np.int32)
        )
        flat = np.ascontiguousarray(flat)
        flat.flags.writeable = False
        offsets.flags.writeable = False
        return flat, offsets

    def _bind_c_abi(self):
        libgint = self._int4c2e.libgint
        try:
            columns_fn = getattr(libgint, _COLUMNS_SYMBOL)
            diagonal_fn = getattr(libgint, _DIAGONAL_SYMBOL)
        except AttributeError as exc:
            raise RuntimeError(
                "libgint lacks the selected-column C ABI; rebuild gpu4pyscf/lib"
            ) from exc
        void_p = ctypes.c_void_p
        columns_fn.restype = ctypes.c_int
        columns_fn.argtypes = [
            void_p,
            void_p,
            ctypes.POINTER(_CSelectedPairData),
            void_p,
            ctypes.c_int,
            void_p,
            void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
        ]
        diagonal_fn.restype = ctypes.c_int
        diagonal_fn.argtypes = [
            void_p,
            void_p,
            ctypes.POINTER(_CSelectedPairData),
            void_p,
            void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
        ]
        return columns_fn, diagonal_fn

    def _c_common(self) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
        stream = self._cupy.cuda.get_current_stream()
        bpcache = ctypes.cast(self.vhfopt.bpcache, ctypes.c_void_p)
        return ctypes.c_void_p(int(stream.ptr)), bpcache

    def columns(self, pivots: Iterable[int]):
        pivots = _normalise_selected_pairs(
            pivots,
            dimension=self.dimension,
            max_batch_size=self.max_batch_size,
        )
        cupy = self._cupy
        batch_size = int(pivots.size)
        output = cupy.zeros((batch_size, self.dimension), dtype=cupy.float64)
        rows = np.arange(batch_size, dtype=np.int32)
        cp_ids = self.schedule.pair_cp_id[pivots]
        task_ids = self.schedule.pair_task_id[pivots]
        stream, bpcache = self._c_common()
        for cp_kl_id in range(self.schedule.ncp):
            positions = np.flatnonzero(
                (cp_ids == cp_kl_id) & (task_ids >= 0)
            ).astype(np.int32, copy=False)
            if positions.size == 0:
                continue
            selected_host = np.ascontiguousarray(pivots[positions], dtype=np.int32)
            selected_rows_host = np.ascontiguousarray(rows[positions], dtype=np.int32)
            selected_device = cupy.asarray(selected_host)
            selected_rows_device = cupy.asarray(selected_rows_host)
            self._record_transfer(
                "h2d",
                "selected_column_request",
                int(selected_host.nbytes + selected_rows_host.nbytes),
                count=2,
            )
            for cp_ij_id in range(self.schedule.ncp):
                if int(self.schedule.cp_task_offsets[cp_ij_id + 1]) == int(
                    self.schedule.cp_task_offsets[cp_ij_id]
                ):
                    continue
                error = self._columns_fn(
                    stream,
                    bpcache,
                    ctypes.byref(self._c_schedule),
                    ctypes.c_void_p(int(output.data.ptr)),
                    ctypes.c_int(batch_size),
                    ctypes.c_void_p(int(selected_device.data.ptr)),
                    ctypes.c_void_p(int(selected_rows_device.data.ptr)),
                    ctypes.c_int(int(positions.size)),
                    ctypes.c_int(cp_ij_id),
                    ctypes.c_int(cp_kl_id),
                    ctypes.c_int(int(self._group_i[cp_ij_id] == self._group_j[cp_ij_id])),
                    ctypes.c_double(float(np.log(self.direct_scf_tol))),
                    ctypes.c_double(self.omega),
                )
                self.cabi_calls += 1
                self.kernel_launches += 1
                self._record_transfer(
                    "h2d", "selected_gint_constant_cache", _BPCACHE_CONSTANT_BYTES
                )
                if error != 0:
                    raise RuntimeError(
                        f"{_COLUMNS_SYMBOL} failed for cp groups "
                        f"({cp_ij_id}, {cp_kl_id}), error={error}"
                    )
        self.batch_calls += 1
        self.column_calls += batch_size
        self._maximum_batch_observed = max(
            self._maximum_batch_observed, batch_size
        )
        return output

    def column(self, pivot: int):
        pivot = _checked_int(
            pivot, name="pivot", minimum=0, maximum=self.dimension - 1
        )
        return self.columns((pivot,))[0].copy()

    def diagonal(self):
        self.diagonal_calls += 1
        if self._diagonal is None:
            cupy = self._cupy
            output = cupy.zeros(self.dimension, dtype=cupy.float64)
            stream, bpcache = self._c_common()
            itemsize = np.dtype(np.int32).itemsize
            for cp_id in range(self.schedule.ncp):
                start = int(self._scheduled_cp_offsets[cp_id])
                stop = int(self._scheduled_cp_offsets[cp_id + 1])
                if start == stop:
                    continue
                selected_ptr = int(self._scheduled_pairs_device.data.ptr) + start * itemsize
                error = self._diagonal_fn(
                    stream,
                    bpcache,
                    ctypes.byref(self._c_schedule),
                    ctypes.c_void_p(int(output.data.ptr)),
                    ctypes.c_void_p(selected_ptr),
                    ctypes.c_int(stop - start),
                    ctypes.c_int(cp_id),
                    ctypes.c_int(int(self._group_i[cp_id] == self._group_j[cp_id])),
                    ctypes.c_double(float(np.log(self.direct_scf_tol))),
                    ctypes.c_double(self.omega),
                )
                self.cabi_calls += 1
                self.kernel_launches += 1
                self._record_transfer(
                    "h2d", "selected_gint_constant_cache", _BPCACHE_CONSTANT_BYTES
                )
                if error != 0:
                    raise RuntimeError(
                        f"{_DIAGONAL_SYMBOL} failed for cp group {cp_id}, "
                        f"error={error}"
                    )
            self._diagonal = output
        return self._diagonal.copy()

    def explicit_transfer_ledger(self) -> dict[str, Any]:
        directions = {}
        for direction in ("h2d", "d2h"):
            operations = {
                name: {
                    "bytes": int(nbytes),
                    "count": int(self._transfer_counts[direction][name]),
                }
                for name, nbytes in self._transfer_bytes[direction].items()
            }
            directions[direction] = {
                "bytes": sum(item["bytes"] for item in operations.values()),
                "count": sum(item["count"] for item in operations.values()),
                "operations": operations,
            }
        return {
            "schema": "gpu4pyscf.gint-selected-transfer-ledger.v1",
            "directions": directions,
            "total_bytes": sum(value["bytes"] for value in directions.values()),
            "output_residency": "gpu",
            "known_provider_arrays_byte_counted": True,
            "complete_for_vhfopt_internal_setup": False,
            "unresolved_operations": [
                {
                    "operation": "vhfopt-internal-basis-cache-and-coeff-build",
                    "bytes": None,
                    "reason": "_VHFOpt does not expose its internal transfer ledger",
                },
                {
                    "operation": "cuda-runtime-kernel-argument-marshalling",
                    "bytes": None,
                    "reason": "runtime-managed launch traffic is not an array copy",
                },
            ],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "factorization": self.factorization,
            "backend": self.backend,
            "c_abi": {
                "columns": _COLUMNS_SYMBOL,
                "diagonal": _DIAGONAL_SYMBOL,
            },
            "dimension": self.dimension,
            "nao": self.schedule.nao_original,
            "sorted_cartesian_nao": int(self._coeff.shape[0]),
            "direct_scf_tol": self.direct_scf_tol,
            "group_size": self.group_size,
            "max_batch_size": self.max_batch_size,
            "maximum_batch_observed": self._maximum_batch_observed,
            "omega": self.omega,
            "precision": "fp64",
            "maximum_rys_order": self._maximum_rys_order,
            "output_residency": "gpu",
            "output_layout": "batch-by-packed-lower-triangle",
            "materializes_pair_matrix": self.materializes_pair_matrix,
            "materializes_four_index_tensor": self.materializes_four_index_tensor,
            "materializes_per_pivot_ao_matrix": self.materializes_per_pivot_ao_matrix,
            "device_schedule": True,
            "device_schedule_bytes": int(
                sum(array.nbytes for array in self._device_schedule.values())
                + self._scheduled_pairs_device.nbytes
            ),
            "release_scope": _RELEASE_SCOPE,
            "diag_block_with_triu": self.diag_block_with_triu,
            "performance_eligible": self.performance_eligible,
            "performance_gate": _PERFORMANCE_GATE,
            "batch_calls": self.batch_calls,
            "column_calls": self.column_calls,
            "diagonal_calls": self.diagonal_calls,
            "cabi_calls": self.cabi_calls,
            "kernel_launches": self.kernel_launches,
            "schedule": self.schedule.metadata(),
            "explicit_transfers": self.explicit_transfer_ledger(),
        }


# Short alias for call sites that already spell out the AO-pair contract.
GINTSelectedPairColumnProvider = GINTSelectedAOPairColumnProvider


__all__ = [
    "GINTSelectedAOPairColumnProvider",
    "GINTSelectedPairColumnProvider",
    "SelectedPairSchedule",
    "build_selected_pair_schedule",
]
