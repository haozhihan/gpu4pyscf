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
Performance eligibility additionally requires a qualification receipt anchored
by a content-addressed release snapshot and rebuilt CUDA library.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import socket
import threading
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from gpu4pyscf.cc.gint_pair_columns import (
    SortedAOPairMap,
    original_ao_shell_support,
)
from gpu4pyscf.cc.gint_transfer_audit import (
    GINTSetupTransferAudit,
    validate_runtime_performance_gate,
)


_COLUMNS_SYMBOL = "GINTfill_selected_int2e_columns"
_DIAGONAL_SYMBOL = "GINTfill_selected_int2e_diagonal"
_WORKSPACE_SIZE_SYMBOL = "GINTselected_workspace_size"
_BPCACHE_SIZE_SYMBOL = "GINTsizeof_basis_prod_cache"
_SELECTED_ABI_SIZE_SYMBOL = "GINTsizeof_selected_pair_data"
_SELECTED_ABI_OFFSET_SYMBOL = "GINToffsetof_selected_pair_data"
_SELECTED_ABI_VERSION_SYMBOL = "GINTselected_pair_data_abi_version"
_SELECTED_ABI_VERSION = 2
_RELEASE_SCOPE = "spherical-original-aos-with-single-shell-support"
_PERFORMANCE_GATE = "water2-write-once-qualification-receipt-v1"
_HIGH_RYS_MINIMUM_ORDER = 7
_HIGH_RYS_THREADS_PER_BLOCK = 32
_STREAM_SAFETY_STRATEGY = (
    "process-wide-per-device-single-stream-with-host-enqueue-lock"
)
_STREAM_SAFETY_POLICY_VERSION = 1
_COLUMN_KERNELS = ("reference", "grouped")
_GROUPED_COLUMN_FLAG = 2


@dataclass(frozen=True)
class _DeviceStreamBinding:
    """Strong process-lifetime reference to one selected-GINT CUDA stream."""

    process_id: int
    device_id: int
    stream_ptr: int
    stream: Any


# ``c_bpcache`` is CUDA constant memory shared by every provider in the
# process.  A provider's bounded high-Rys workspace is also reused by all of
# its launches.  Keeping every selected-GINT launch for a device on one stream
# makes the asynchronous constant copies, workspace users, and kernels
# strictly ordered without adding a device synchronization to the hot path.
# The lock serializes host enqueue sequences; it is deliberately held only
# until all work for one public provider call has been enqueued because CUDA
# stream order protects the work after the lock is released.
_STREAM_ENQUEUE_LOCK = threading.RLock()
_DEVICE_STREAM_BINDINGS: dict[int, _DeviceStreamBinding] = {}


def _device_multiprocessor_count(cupy: Any) -> int:
    """Return the active CUDA device's SM count across CuPy API versions."""

    device = int(cupy.cuda.runtime.getDevice())
    properties = cupy.cuda.runtime.getDeviceProperties(device)
    value = properties.get("multiProcessorCount")
    if value is None:
        value = properties.get(b"multiProcessorCount")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError("CUDA device properties omit multiProcessorCount")
    count = int(value)
    if count < 1:
        raise RuntimeError("CUDA device reports no multiprocessors")
    return count


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


def _normalise_column_kernel(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("column_kernel must be a string")
    result = value.strip().lower()
    if result not in _COLUMN_KERNELS:
        raise ValueError(
            "column_kernel must be one of: " + ", ".join(_COLUMN_KERNELS)
        )
    return result


def _order_selected_request(
    pivots: np.ndarray,
    rows: np.ndarray,
    pair_task_id: np.ndarray,
    *,
    column_kernel: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Order one cp-local request and report its ket shell-task groups.

    The grouped CUDA path recognizes adjacent pivots with the same
    ``pair_task_id``.  A stable sort preserves the input order inside a group;
    the separately carried output rows preserve the public batch layout.
    """

    column_kernel = _normalise_column_kernel(column_kernel)
    pivots = np.asarray(pivots, dtype=np.int32)
    rows = np.asarray(rows, dtype=np.int32)
    pair_task_id = np.asarray(pair_task_id, dtype=np.int32)
    if pivots.ndim != 1 or rows.shape != pivots.shape:
        raise ValueError("selected pivots and output rows must be aligned vectors")
    if pair_task_id.ndim != 1:
        raise ValueError("pair_task_id must be one-dimensional")
    if pivots.size and (
        np.any(pivots < 0) or np.any(pivots >= pair_task_id.size)
    ):
        raise IndexError("selected pivot is outside pair_task_id")
    tasks = pair_task_id[pivots]
    if np.any(tasks < 0):
        raise ValueError("selected request contains a screened shell-pair task")
    if column_kernel == "grouped" and pivots.size > 1:
        order = np.argsort(tasks, kind="stable")
        pivots = pivots[order]
        rows = rows[order]
        tasks = tasks[order]
    ordered_pivots = np.ascontiguousarray(pivots, dtype=np.int32)
    ordered_rows = np.ascontiguousarray(rows, dtype=np.int32)
    if column_kernel == "reference":
        group_count = int(tasks.size)
    else:
        group_count = (
            0
            if tasks.size == 0
            else 1 + int(np.count_nonzero(tasks[1:] != tasks[:-1]))
        )
    ordered_pivots.flags.writeable = False
    ordered_rows.flags.writeable = False
    return ordered_pivots, ordered_rows, group_count


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


@lru_cache(maxsize=None)
def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _basis_prod_cache_abi_identity(libgint: Any) -> dict[str, Any]:
    """Bind and verify the C ``BasisProdCache`` size oracle."""

    try:
        size_fn = getattr(libgint, _BPCACHE_SIZE_SYMBOL)
    except AttributeError as exc:
        raise RuntimeError(
            "libgint lacks the BasisProdCache size ABI; rebuild gpu4pyscf/lib"
        ) from exc
    size_fn.restype = ctypes.c_size_t
    size_fn.argtypes = []
    reported = int(size_fn())
    expected = int(_BPCACHE_CONSTANT_BYTES)
    if reported != expected:
        raise RuntimeError(
            "BasisProdCache ABI mismatch: "
            f"Python expects {expected} bytes but libgint reports {reported}"
        )
    library_name = getattr(libgint, "_name", None)
    library_path = None
    library_sha256 = None
    library_bytes = None
    if isinstance(library_name, str):
        candidate = Path(library_name).expanduser().resolve()
        if candidate.is_file():
            library_path = str(candidate)
            library_sha256 = _sha256_file(library_path)
            library_bytes = int(candidate.stat().st_size)
    return {
        "symbol": _BPCACHE_SIZE_SYMBOL,
        "python_ctypes_size_bytes": expected,
        "libgint_size_bytes": reported,
        "libgint_path": library_path,
        "libgint_sha256": library_sha256,
        "libgint_bytes": library_bytes,
        "verified": True,
    }


def _selected_pair_abi_identity(libgint: Any) -> dict[str, Any]:
    """Bind the selected-pair structure's C size, offsets, and version."""

    try:
        size_fn = getattr(libgint, _SELECTED_ABI_SIZE_SYMBOL)
        offset_fn = getattr(libgint, _SELECTED_ABI_OFFSET_SYMBOL)
        version_fn = getattr(libgint, _SELECTED_ABI_VERSION_SYMBOL)
    except AttributeError as exc:
        raise RuntimeError(
            "libgint lacks the selected-pair ABI identity; rebuild gpu4pyscf/lib"
        ) from exc
    size_fn.restype = ctypes.c_size_t
    size_fn.argtypes = []
    offset_fn.restype = ctypes.c_size_t
    offset_fn.argtypes = [ctypes.c_int]
    version_fn.restype = ctypes.c_int
    version_fn.argtypes = []
    python_size = int(ctypes.sizeof(_CSelectedPairData))
    library_size = int(size_fn())
    version = int(version_fn())
    fields_identity = []
    for index, (name, _) in enumerate(_CSelectedPairData._fields_):
        python_offset = int(getattr(_CSelectedPairData, name).offset)
        library_offset = int(offset_fn(index))
        fields_identity.append({
            "index": index,
            "name": name,
            "python_offset_bytes": python_offset,
            "libgint_offset_bytes": library_offset,
        })
    if library_size != python_size:
        raise RuntimeError(
            "selected-pair ABI size mismatch: "
            f"Python expects {python_size} bytes but libgint reports {library_size}"
        )
    mismatches = [
        item for item in fields_identity
        if item["python_offset_bytes"] != item["libgint_offset_bytes"]
    ]
    if mismatches:
        raise RuntimeError("selected-pair ABI field-offset mismatch")
    if version != _SELECTED_ABI_VERSION:
        raise RuntimeError(
            "selected-pair ABI version mismatch: "
            f"Python expects {_SELECTED_ABI_VERSION} but libgint reports {version}"
        )
    return {
        "schema": "gpu4pyscf.gint-selected-pair-abi.v1",
        "size_symbol": _SELECTED_ABI_SIZE_SYMBOL,
        "offset_symbol": _SELECTED_ABI_OFFSET_SYMBOL,
        "version_symbol": _SELECTED_ABI_VERSION_SYMBOL,
        "python_ctypes_size_bytes": python_size,
        "libgint_size_bytes": library_size,
        "abi_version": version,
        "fields": fields_identity,
        "verified": True,
    }


def _format_cpu_list(values: Iterable[int]) -> str:
    ordered = sorted({int(value) for value in values})
    ranges: list[str] = []
    start = previous = None
    for value in ordered:
        if start is None:
            start = previous = value
        elif value == previous + 1:
            previous = value
        else:
            ranges.append(
                str(start) if start == previous else f"{start}-{previous}"
            )
            start = previous = value
    if start is not None:
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _normalise_pci_bus_id(value: Any) -> Optional[str]:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="strict")
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip().lower()
    if value.startswith("00000000:"):
        value = "0000:" + value.split(":", 1)[1]
    return value


def _release_process_context(cupy: Any) -> dict[str, Any]:
    """Capture current process/allocation facts independently of the sidecar."""

    bus_id: Optional[str] = None
    numa_node: Optional[int] = None
    try:
        bus_id = _normalise_pci_bus_id(cupy.cuda.Device().pci_bus_id)
        if bus_id is not None:
            numa_node = int(
                (Path("/sys/bus/pci/devices") / bus_id / "numa_node")
                .read_text(encoding="utf-8")
                .strip()
            )
    except Exception:
        bus_id = None
        numa_node = None
    environment_names = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "CUDA_VISIBLE_DEVICES",
        "CCSD_PERFORMANCE_ELIGIBLE",
        "CCSD_BENCHMARK_TRACK",
        "CCSD_TOPOLOGY_RECORD",
        "CCSD_TOPOLOGY_SHA256",
        "CCSD_BOUND_CPUS",
        "GPU4PYSCF_NUMA",
    )
    return {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "affinity": _format_cpu_list(os.sched_getaffinity(0)),
        "gpu_pci_bus_id": bus_id,
        "gpu_numa_node": numa_node,
        "environment": {name: os.getenv(name) for name in environment_names},
    }


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
        column_kernel: str = "reference",
        transfer_counter: Any = None,
        omega: float = 0.0,
        runtime_gate_receipt: Optional[str | Path] = None,
        runtime_execution_mode: str = "release-gate",
        runtime_source_digest: Optional[str] = None,
        runtime_source_root: Optional[str | Path] = None,
        runtime_manifest_path: Optional[str | Path] = None,
        runtime_release_topology_path: Optional[str | Path] = None,
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
        column_kernel = _normalise_column_kernel(column_kernel)
        if not np.isfinite(omega):
            raise ValueError("omega must be finite")
        if transfer_counter is None:
            from gpu4pyscf.cc.device_runtime import TransferCounter

            transfer_counter = TransferCounter()

        self._cupy = cupy
        self._int4c2e = int4c2e
        self._construction_process_id = os.getpid()
        self._construction_device_id = int(cupy.cuda.runtime.getDevice())
        self._bound_stream_ptr: Optional[int] = None
        self._bound_stream: Any = None
        self._stream_contract_call_count = 0
        # Reserve the process-wide device stream before setup enqueues any
        # CuPy work.  This also makes all setup arrays and their first selected
        # GINT consumer share an explicit CUDA ordering domain.
        self._register_selected_stream()
        self.mol = mol
        self.direct_scf_tol = float(direct_scf_tol)
        self.group_size = group_size
        self.max_batch_size = max_batch_size
        self.column_kernel = column_kernel
        self.omega = float(omega)
        self.transfer_counter = transfer_counter
        self._transfer_bytes = {"h2d": {}, "d2h": {}}
        self._transfer_counts = {"h2d": {}, "d2h": {}}

        self._setup_transfer_audit = GINTSetupTransferAudit(
            transfer_counter=transfer_counter
        )
        self._setup_transfer_audit.record_exclusion(
            "cuda_runtime_kernel_argument_marshalling",
            classification="runtime-control-not-application-array-copy",
            reason=(
                "CUDA launch parameters are runtime-managed control traffic; "
                "no application array is copied between host and device"
            ),
        )
        vhfopt = int4c2e._VHFOpt(
            mol, "int2e", transfer_audit=self._setup_transfer_audit
        )
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
            abi_version=_SELECTED_ABI_VERSION,
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
        (
            self._columns_fn,
            self._diagonal_fn,
            self._workspace_size_fn,
        ) = self._bind_c_abi()
        self._multiprocessor_count = _device_multiprocessor_count(cupy)
        self._high_rys_workspace_nbytes = int(self._workspace_size_fn(
            ctypes.c_int(self._maximum_rys_order),
            ctypes.c_int(self._multiprocessor_count),
        ))
        if self._maximum_rys_order >= _HIGH_RYS_MINIMUM_ORDER:
            if self._high_rys_workspace_nbytes < 1:
                raise RuntimeError(
                    "libgint returned an invalid high-Rys workspace size"
                )
            self._high_rys_workspace = cupy.empty(
                self._high_rys_workspace_nbytes, dtype=cupy.uint8
            )
        else:
            if self._high_rys_workspace_nbytes != 0:
                raise RuntimeError(
                    "libgint requested high-Rys workspace for a low-order basis"
                )
            self._high_rys_workspace = None
        if runtime_gate_receipt is not None:
            module_source_root = Path(__file__).resolve().parents[2]
            module_manifest_path = module_source_root.parent / "manifest.json"
            if (
                runtime_source_root is not None
                and Path(runtime_source_root).expanduser().resolve()
                != module_source_root
            ):
                raise ValueError(
                    "runtime source root differs from the imported provider"
                )
            if (
                runtime_manifest_path is not None
                and Path(runtime_manifest_path).expanduser().resolve()
                != module_manifest_path
            ):
                raise ValueError(
                    "runtime manifest differs from the imported provider snapshot"
                )
            runtime_source_root = module_source_root
            runtime_manifest_path = module_manifest_path
        release_context = (
            _release_process_context(cupy)
            if runtime_gate_receipt is not None else None
        )
        self._runtime_gate = validate_runtime_performance_gate(
            runtime_gate_receipt,
            execution_mode=runtime_execution_mode,
            basis_prod_cache_size_bytes=self._basis_prod_cache_abi[
                "libgint_size_bytes"
            ],
            loaded_libgint_sha256=self._basis_prod_cache_abi[
                "libgint_sha256"
            ],
            loaded_libgint_path=self._basis_prod_cache_abi[
                "libgint_path"
            ],
            loaded_libgint_bytes=self._basis_prod_cache_abi[
                "libgint_bytes"
            ],
            selected_pair_abi=self._selected_pair_abi,
            selected_c_abi={
                "columns": _COLUMNS_SYMBOL,
                "diagonal": _DIAGONAL_SYMBOL,
                "workspace_size": _WORKSPACE_SIZE_SYMBOL,
            },
            loaded_source_digest=runtime_source_digest,
            loaded_source_root=runtime_source_root,
            loaded_manifest_path=runtime_manifest_path,
            release_topology_path=runtime_release_topology_path,
            loaded_release_job_id=os.getenv("SLURM_JOB_ID"),
            loaded_release_context=release_context,
        )
        transfer_complete = self._setup_transfer_audit.complete
        self.performance_eligible = bool(
            transfer_complete
            and self._basis_prod_cache_abi["verified"]
            and self._runtime_gate["validated"]
            and self.column_kernel == "reference"
        )

        self.batch_calls = 0
        self.column_calls = 0
        self.diagonal_calls = 0
        self.cabi_calls = 0
        self.kernel_launches = 0
        self.constant_cache_copies = 0
        self.column_gout_evaluations = 0
        self.reference_column_gout_evaluations = 0
        self.diagonal_gout_evaluations = 0
        self.precutoff_column_gout_attempts = 0
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
        self._basis_prod_cache_abi = _basis_prod_cache_abi_identity(libgint)
        self._selected_pair_abi = _selected_pair_abi_identity(libgint)
        try:
            columns_fn = getattr(libgint, _COLUMNS_SYMBOL)
            diagonal_fn = getattr(libgint, _DIAGONAL_SYMBOL)
            workspace_size_fn = getattr(libgint, _WORKSPACE_SIZE_SYMBOL)
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
            void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
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
            void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
        ]
        workspace_size_fn.restype = ctypes.c_size_t
        workspace_size_fn.argtypes = [ctypes.c_int, ctypes.c_int]
        return columns_fn, diagonal_fn, workspace_size_fn

    def _high_rys_workspace_c_args(self) -> tuple[ctypes.c_void_p, ctypes.c_size_t]:
        workspace = self._high_rys_workspace
        pointer = 0 if workspace is None else int(workspace.data.ptr)
        return (
            ctypes.c_void_p(pointer),
            ctypes.c_size_t(self._high_rys_workspace_nbytes),
        )

    def _validate_and_bind_stream_locked(self) -> tuple[Any, int, int]:
        """Return the one legal stream while holding ``_STREAM_ENQUEUE_LOCK``."""

        process_id = os.getpid()
        if process_id != self._construction_process_id:
            raise RuntimeError(
                "selected GINT providers cannot cross a process boundary"
            )
        device_id = int(self._cupy.cuda.runtime.getDevice())
        if device_id != self._construction_device_id:
            raise RuntimeError(
                "selected GINT provider was constructed on CUDA device "
                f"{self._construction_device_id}, but the current device is "
                f"{device_id}"
            )
        stream = self._cupy.cuda.get_current_stream()
        stream_ptr = int(stream.ptr)

        process_binding = _DEVICE_STREAM_BINDINGS.get(device_id)
        if process_binding is not None:
            if process_binding.process_id != process_id:
                raise RuntimeError(
                    "selected GINT stream binding belongs to another process"
                )
            if process_binding.stream_ptr != stream_ptr:
                raise RuntimeError(
                    "selected GINT CUDA device "
                    f"{device_id} is process-wide bound to stream "
                    f"{process_binding.stream_ptr}; current stream {stream_ptr} "
                    "is forbidden because libgint c_bpcache is shared constant "
                    "memory"
                )

        if (
            self._bound_stream_ptr is not None
            and self._bound_stream_ptr != stream_ptr
        ):
            raise RuntimeError(
                "selected GINT provider is bound to CUDA stream "
                f"{self._bound_stream_ptr}; current stream {stream_ptr} is "
                "forbidden because its workspace may still be in use"
            )

        if process_binding is None:
            process_binding = _DeviceStreamBinding(
                process_id=process_id,
                device_id=device_id,
                stream_ptr=stream_ptr,
                stream=stream,
            )
            _DEVICE_STREAM_BINDINGS[device_id] = process_binding
        if self._bound_stream_ptr is None:
            self._bound_stream_ptr = stream_ptr
            # Retain the canonical process binding's stream object so the
            # CUDA stream cannot be destroyed while asynchronous work exists.
            self._bound_stream = process_binding.stream
        return stream, stream_ptr, device_id

    def _register_selected_stream(self) -> None:
        """Bind this provider and its device before asynchronous setup work."""

        with _STREAM_ENQUEUE_LOCK:
            self._validate_and_bind_stream_locked()

    @contextmanager
    def _selected_enqueue_scope(self):
        """Serialize one complete host enqueue sequence on the bound stream."""

        with _STREAM_ENQUEUE_LOCK:
            _stream, stream_ptr, _device_id = (
                self._validate_and_bind_stream_locked()
            )
            self._stream_contract_call_count += 1
            bpcache = ctypes.cast(self.vhfopt.bpcache, ctypes.c_void_p)
            yield ctypes.c_void_p(stream_ptr), bpcache

    def _record_constant_cache_copy(self, performed: int) -> None:
        performed = int(performed)
        if performed not in (0, 1):
            raise RuntimeError("selected GINT returned an invalid constant-copy status")
        if performed:
            self.constant_cache_copies = (
                int(getattr(self, "constant_cache_copies", 0)) + 1
            )
            self._record_transfer(
                "h2d", "selected_gint_constant_cache", _BPCACHE_CONSTANT_BYTES
            )

    def _column_gout_counts(
        self,
        *,
        cp_ij_id: int,
        cp_kl_id: int,
        selected_pairs: np.ndarray,
    ) -> tuple[int, int]:
        """Count GOUT builds implied by one successful columns launch.

        This mirrors the kernel's Schwarz cutoff on the immutable host
        schedule.  The values are deterministic operation counts, not timing
        estimates.  ``reference`` counts every selected pivot; ``grouped``
        counts each distinct ket shell task once for every surviving row task.
        """

        row_start = int(self.schedule.cp_task_offsets[cp_ij_id])
        row_stop = int(self.schedule.cp_task_offsets[cp_ij_id + 1])
        ket_start = int(self.schedule.cp_task_offsets[cp_kl_id])
        row_log_q = self.schedule.task_log_q[row_start:row_stop]
        local_task_ids = self.schedule.pair_task_id[selected_pairs]
        ket_log_q = self.schedule.task_log_q[ket_start + local_task_ids]
        log_cutoff = float(np.log(self.direct_scf_tol))
        reference = int(np.count_nonzero(
            row_log_q[:, None] + ket_log_q[None, :] >= log_cutoff
        ))
        if self.column_kernel == "reference":
            return reference, reference
        unique_ket_log_q = ket_log_q[
            np.concatenate((
                np.ones(1, dtype=bool),
                local_task_ids[1:] != local_task_ids[:-1],
            ))
        ]
        grouped = int(np.count_nonzero(
            row_log_q[:, None] + unique_ket_log_q[None, :] >= log_cutoff
        ))
        return reference, grouped

    def _diagonal_gout_count(
        self,
        *,
        cp_id: int,
        selected_pairs: np.ndarray,
        group_diagonal: int,
    ) -> int:
        task_start = int(self.schedule.cp_task_offsets[cp_id])
        task_ids = self.schedule.pair_task_id[selected_pairs]
        log_q = self.schedule.task_log_q[task_start + task_ids]
        symmetrize = self.schedule.pair_symmetrize[selected_pairs]
        return int(np.count_nonzero(
            (2.0 * log_q >= float(np.log(self.direct_scf_tol)))
            & (symmetrize != int(group_diagonal))
        ))

    def columns(self, pivots: Iterable[int]):
        pivots = _normalise_selected_pairs(
            pivots,
            dimension=self.dimension,
            max_batch_size=self.max_batch_size,
        )
        with self._selected_enqueue_scope() as (stream, bpcache):
            cupy = self._cupy
            batch_size = int(pivots.size)
            output = cupy.zeros(
                (batch_size, self.dimension), dtype=cupy.float64
            )
            rows = np.arange(batch_size, dtype=np.int32)
            cp_ids = self.schedule.pair_cp_id[pivots]
            task_ids = self.schedule.pair_task_id[pivots]
            workspace, workspace_nbytes = self._high_rys_workspace_c_args()
            for cp_kl_id in range(self.schedule.ncp):
                positions = np.flatnonzero(
                    (cp_ids == cp_kl_id) & (task_ids >= 0)
                ).astype(np.int32, copy=False)
                if positions.size == 0:
                    continue
                selected_host, selected_rows_host, task_group_count = (
                    _order_selected_request(
                        pivots[positions],
                        rows[positions],
                        self.schedule.pair_task_id,
                        column_kernel=self.column_kernel,
                    )
                )
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
                    reference_gout, actual_gout = self._column_gout_counts(
                        cp_ij_id=cp_ij_id,
                        cp_kl_id=cp_kl_id,
                        selected_pairs=selected_host,
                    )
                    constant_copy_performed = ctypes.c_int(0)
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
                        ctypes.c_int(
                            int(
                                self._group_i[cp_ij_id]
                                == self._group_j[cp_ij_id]
                            )
                            | (
                                _GROUPED_COLUMN_FLAG
                                if self.column_kernel == "grouped" else 0
                            )
                        ),
                        ctypes.c_double(float(np.log(self.direct_scf_tol))),
                        ctypes.c_double(self.omega),
                        workspace,
                        workspace_nbytes,
                        ctypes.byref(constant_copy_performed),
                    )
                    self.cabi_calls += 1
                    self.kernel_launches += 1
                    self._record_constant_cache_copy(
                        constant_copy_performed.value
                    )
                    if error != 0:
                        raise RuntimeError(
                            f"{_COLUMNS_SYMBOL} failed for cp groups "
                            f"({cp_ij_id}, {cp_kl_id}), error={error}"
                        )
                    self.reference_column_gout_evaluations += reference_gout
                    self.column_gout_evaluations += actual_gout
                    row_task_count = int(
                        self.schedule.cp_task_offsets[cp_ij_id + 1]
                        - self.schedule.cp_task_offsets[cp_ij_id]
                    )
                    self.precutoff_column_gout_attempts += (
                        task_group_count * row_task_count
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
        with self._selected_enqueue_scope() as (stream, bpcache):
            self.diagonal_calls += 1
            if self._diagonal is None:
                cupy = self._cupy
                output = cupy.zeros(self.dimension, dtype=cupy.float64)
                workspace, workspace_nbytes = self._high_rys_workspace_c_args()
                itemsize = np.dtype(np.int32).itemsize
                for cp_id in range(self.schedule.ncp):
                    start = int(self._scheduled_cp_offsets[cp_id])
                    stop = int(self._scheduled_cp_offsets[cp_id + 1])
                    if start == stop:
                        continue
                    selected_ptr = (
                        int(self._scheduled_pairs_device.data.ptr)
                        + start * itemsize
                    )
                    group_diagonal = int(
                        self._group_i[cp_id] == self._group_j[cp_id]
                    )
                    diagonal_gout = self._diagonal_gout_count(
                        cp_id=cp_id,
                        selected_pairs=self._scheduled_pairs_host[start:stop],
                        group_diagonal=group_diagonal,
                    )
                    constant_copy_performed = ctypes.c_int(0)
                    error = self._diagonal_fn(
                        stream,
                        bpcache,
                        ctypes.byref(self._c_schedule),
                        ctypes.c_void_p(int(output.data.ptr)),
                        ctypes.c_void_p(selected_ptr),
                        ctypes.c_int(stop - start),
                        ctypes.c_int(cp_id),
                        ctypes.c_int(group_diagonal),
                        ctypes.c_double(float(np.log(self.direct_scf_tol))),
                        ctypes.c_double(self.omega),
                        workspace,
                        workspace_nbytes,
                        ctypes.byref(constant_copy_performed),
                    )
                    self.cabi_calls += 1
                    self.kernel_launches += 1
                    self._record_constant_cache_copy(
                        constant_copy_performed.value
                    )
                    if error != 0:
                        raise RuntimeError(
                            f"{_DIAGONAL_SYMBOL} failed for cp group {cp_id}, "
                            f"error={error}"
                        )
                    self.diagonal_gout_evaluations += diagonal_gout
                self._diagonal = output
            return self._diagonal.copy()

    def explicit_transfer_ledger(self) -> dict[str, Any]:
        setup = self._setup_transfer_audit.to_dict()
        directions = {}
        for direction in ("h2d", "d2h"):
            operations = dict(setup["directions"][direction]["operations"])
            provider_operations = {
                name: {
                    "bytes": int(nbytes),
                    "count": int(self._transfer_counts[direction][name]),
                    "logical_payload": name.replace("_", "-"),
                    "provenance": (
                        "gpu4pyscf.cc.gint_selected_columns."
                        "GINTSelectedAOPairColumnProvider"
                    ),
                }
                for name, nbytes in self._transfer_bytes[direction].items()
            }
            duplicate = set(operations) & set(provider_operations)
            if duplicate:
                raise RuntimeError(
                    "setup and provider transfer ledgers overlap: "
                    + ", ".join(sorted(duplicate))
                )
            operations.update(provider_operations)
            directions[direction] = {
                "bytes": sum(item["bytes"] for item in operations.values()),
                "count": sum(item["count"] for item in operations.values()),
                "operations": operations,
            }
        abi_verified = bool(self._basis_prod_cache_abi.get("verified", False))
        complete = bool(setup["complete_for_declared_payloads"] and abi_verified)
        return {
            "schema": "gpu4pyscf.gint-selected-transfer-ledger.v2",
            "directions": directions,
            "total_bytes": sum(value["bytes"] for value in directions.values()),
            "output_residency": "gpu",
            "logical_device_payloads": setup["logical_device_payloads"],
            "known_provider_arrays_byte_counted": True,
            "complete_for_vhfopt_internal_setup": setup[
                "complete_for_declared_payloads"
            ],
            "basis_prod_cache_abi": dict(self._basis_prod_cache_abi),
            "coverage": {
                "basis_seg_contraction": setup["complete_for_declared_payloads"],
                "vhfopt_ao_sort_index": (
                    "vhfopt_ao_sort_index"
                    in directions["h2d"]["operations"]
                ),
                "vhfopt_log_q": (
                    "vhfopt_log_q" in directions["h2d"]["operations"]
                ),
                "gint_basis_product_cache_device_init": all(
                    operation in directions["h2d"]["operations"]
                    for operation in (
                        "gint_basis_cache_ao_loc",
                        "gint_basis_cache_bas_coords",
                        "gint_basis_cache_bas_atm",
                        "gint_basis_cache_aexyz",
                        "gint_basis_cache_bas_pair2shls",
                    )
                ),
                "gint_per_call_constant_cache_copy": abi_verified,
                "selected_provider_array_transfers": True,
                "cuda_runtime_kernel_argument_marshalling": "excluded",
            },
            "excluded_operations": setup["excluded_operations"],
            "unresolved_operations": setup["unresolved_operations"],
            "complete_for_declared_payloads": complete,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "factorization": self.factorization,
            "backend": self.backend,
            "c_abi": {
                "columns": _COLUMNS_SYMBOL,
                "diagonal": _DIAGONAL_SYMBOL,
                "workspace_size": _WORKSPACE_SIZE_SYMBOL,
                "basis_prod_cache_size": _BPCACHE_SIZE_SYMBOL,
                "basis_prod_cache_identity": dict(
                    self._basis_prod_cache_abi
                ),
                "selected_pair_identity": dict(self._selected_pair_abi),
            },
            "dimension": self.dimension,
            "nao": self.schedule.nao_original,
            "sorted_cartesian_nao": int(self._coeff.shape[0]),
            "direct_scf_tol": self.direct_scf_tol,
            "group_size": self.group_size,
            "max_batch_size": self.max_batch_size,
            "column_kernel": self.column_kernel,
            "column_kernel_status": (
                "reference-qualified-by-existing-runtime-gate"
                if self.column_kernel == "reference"
                else "prototype-awaiting-mtu-a100-ab-gate"
            ),
            "gout_reuse_scope": (
                "none-one-gout-per-row-task-and-pivot"
                if self.column_kernel == "reference"
                else "single-thread-contiguous-ket-shell-task-group"
            ),
            "maximum_batch_observed": self._maximum_batch_observed,
            "omega": self.omega,
            "precision": "fp64",
            "maximum_rys_order": self._maximum_rys_order,
            "high_rys_workspace": {
                "strategy": "provider-owned-bounded-global-grid-stride",
                "minimum_rys_order": _HIGH_RYS_MINIMUM_ORDER,
                "maximum_rys_order": self._maximum_rys_order,
                "threads_per_block": _HIGH_RYS_THREADS_PER_BLOCK,
                "multiprocessor_count": self._multiprocessor_count,
                "slot_count": (
                    self._multiprocessor_count * _HIGH_RYS_THREADS_PER_BLOCK
                    if self._high_rys_workspace is not None else 0
                ),
                "nbytes": self._high_rys_workspace_nbytes,
                "resident": self._high_rys_workspace is not None,
                "static_thread_gout_eliminated_for_orders": [7, 8],
            },
            "stream_safety": {
                "strategy": _STREAM_SAFETY_STRATEGY,
                "policy_version": _STREAM_SAFETY_POLICY_VERSION,
                "binding_scope": "process-wide-per-cuda-device",
                "construction_process_id": self._construction_process_id,
                "construction_device_id": self._construction_device_id,
                "bound_stream_ptr": self._bound_stream_ptr,
                "strong_stream_lifetime": self._bound_stream is not None,
                "host_enqueue_lock_scope": "entire-public-provider-call",
                "shared_constant_cache_ordering": "bound-cuda-stream",
                "provider_workspace_ordering": "bound-cuda-stream",
                "cross_stream_behavior": "reject-before-device-enqueue",
                "per_call_device_synchronize": False,
                "successful_call_count": self._stream_contract_call_count,
            },
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
            "runtime_gate": dict(self._runtime_gate),
            "batch_calls": self.batch_calls,
            "column_calls": self.column_calls,
            "diagonal_calls": self.diagonal_calls,
            "cabi_calls": self.cabi_calls,
            "kernel_launches": self.kernel_launches,
            "constant_cache_copies": self.constant_cache_copies,
            "gout_evaluations": int(
                self.column_gout_evaluations + self.diagonal_gout_evaluations
            ),
            "column_gout_evaluations": self.column_gout_evaluations,
            "reference_column_gout_evaluations": (
                self.reference_column_gout_evaluations
            ),
            "diagonal_gout_evaluations": self.diagonal_gout_evaluations,
            "precutoff_column_gout_attempts": (
                self.precutoff_column_gout_attempts
            ),
            "operation_count_semantics": {
                "gout_evaluations": (
                    "post-Schwarz selected_build_gout calls implied by the "
                    "immutable host schedule"
                ),
                "reference_column_gout_evaluations": (
                    "post-Schwarz one-GOUT-per-row-task-and-pivot counterfactual"
                ),
                "precutoff_column_gout_attempts": (
                    "row-task by ket-task-group combinations submitted before "
                    "the kernel Schwarz cutoff"
                ),
                "constant_cache_copies": (
                    "copies confirmed by the C-ABI output flag"
                ),
            },
            "grouped_gout_evaluations_saved": int(
                self.reference_column_gout_evaluations
                - self.column_gout_evaluations
            ),
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
