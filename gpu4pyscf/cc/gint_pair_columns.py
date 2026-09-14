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

"""Restricted-task AO-pair ERI columns and diagonals on a GPU.

``GINTfill_int2e`` addresses shell-pair tasks through offsets into a
contraction-pair (``cp``) task list.  Passing ``[task, task + 1]`` as a
one-bin layout therefore evaluates one selected task without changing the
GINT ABI.  This module uses that property to evaluate every row task against
only the shell-pair tasks that support a requested *original AO* pair.

The coefficient map made by :class:`gpu4pyscf.scf.int4c2e._VHFOpt` can in
general map an original AO to more than one segmented Cartesian shell.  This
fixed spherical benchmark release fails closed unless every original AO has
exactly one supporting shell.  Its batched diagonal path evaluates each
shell-pair task once on both sides and supplies all original-AO pair diagonals
assigned to it.

Neither a four-index ERI tensor nor the square AO-pair matrix is allocated.
The implementation remains ``performance_eligible=False`` until its complete
water2 timing and transfer ledger have passed the project performance gate.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np


class _BasisProdCacheConstantLayout(ctypes.Structure):
    """Host ABI mirror for the per-GINT-call constant-memory copy."""

    _fields_ = [
        ("nbas", ctypes.c_int),
        ("ncptype", ctypes.c_int),
        *[(f"pointer_{index}", ctypes.c_void_p) for index in range(17)],
    ]


_GINT_BPCACHE_CONSTANT_BYTES = ctypes.sizeof(_BasisProdCacheConstantLayout)


def _as_index(value: Any, *, upper_bound: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result < upper_bound:
        raise IndexError(f"{name} must lie in [0, {upper_bound})")
    return result


@dataclass(frozen=True)
class SortedAOPairMap:
    """Canonical row-major lower-triangular map for original AO indices.

    Pair ``(p, q)`` is normalized to ``p >= q`` and stored at
    ``p * (p + 1) // 2 + q``.  Thus the order is ``(0,0), (1,0), (1,1), ...``.
    The host arrays are immutable control metadata; integral values never use
    host storage.
    """

    nao: int
    pair_i: np.ndarray = field(init=False, repr=False)
    pair_j: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.nao, (bool, np.bool_)) or not isinstance(
            self.nao, (int, np.integer)
        ):
            raise TypeError("nao must be an integer")
        nao = int(self.nao)
        if nao < 1:
            raise ValueError("nao must be positive")
        pair_i, pair_j = np.tril_indices(nao)
        pair_i = np.asarray(pair_i, dtype=np.int64)
        pair_j = np.asarray(pair_j, dtype=np.int64)
        pair_i.flags.writeable = False
        pair_j.flags.writeable = False
        object.__setattr__(self, "nao", nao)
        object.__setattr__(self, "pair_i", pair_i)
        object.__setattr__(self, "pair_j", pair_j)

    @property
    def dimension(self) -> int:
        return self.nao * (self.nao + 1) // 2

    def index(self, first: int, second: int) -> int:
        first = _as_index(first, upper_bound=self.nao, name="first AO")
        second = _as_index(second, upper_bound=self.nao, name="second AO")
        high, low = max(first, second), min(first, second)
        return high * (high + 1) // 2 + low

    def pair(self, index: int) -> tuple[int, int]:
        index = _as_index(index, upper_bound=self.dimension, name="pair index")
        return int(self.pair_i[index]), int(self.pair_j[index])


@dataclass(frozen=True)
class ColumnValidationResult:
    """Small-system dense-oracle result; never a performance measurement."""

    pivots: tuple[int, ...]
    max_abs_error: float
    errors: tuple[float, ...]
    tolerance: float

    @property
    def passed(self) -> bool:
        return bool(self.max_abs_error <= self.tolerance)


@dataclass(frozen=True)
class RestrictedTaskSpan:
    """One safe, contiguous task span inside one original screening bin."""

    cp_id: int
    task_start: int
    task_stop: int
    bin_index: int
    bin_floor: float

    @property
    def task_count(self) -> int:
        return self.task_stop - self.task_start


@dataclass(frozen=True)
class GINTTaskDescriptor:
    """Host control metadata for one shell-pair task in a GINT cp list."""

    cp_id: int
    task_id: int
    shell_i: int
    shell_j: int
    group_i: int
    group_j: int
    span: RestrictedTaskSpan
    log_q: float


def restricted_task_span(
    cp_id: int,
    task_start: int,
    task_stop: int,
    bins: Sequence[int],
    bins_floor: Sequence[float],
    log_q: Sequence[float],
) -> RestrictedTaskSpan:
    """Build and validate a GINT task restriction using its original floor.

    GINT's screening loop treats each element of ``bins_floor`` as an upper
    bound for all tasks in that bin.  Reusing the original floor is therefore
    conservative.  A span that crosses a bin boundary is rejected because no
    single original floor describes it.
    """

    cp_id = _as_index(cp_id, upper_bound=np.iinfo(np.int32).max, name="cp id")
    bins = np.asarray(bins)
    floors = np.asarray(bins_floor)
    values = np.asarray(log_q)
    if bins.ndim != 1 or floors.ndim != 1 or values.ndim != 1:
        raise ValueError("GINT bins, floors, and log_q must be one-dimensional")
    if bins.size < 2 or floors.size != bins.size - 1:
        raise ValueError("GINT bins/floors have inconsistent lengths")
    if not np.issubdtype(bins.dtype, np.integer):
        raise TypeError("GINT bins must contain integers")
    if bins.size and (
        int(bins.min()) < np.iinfo(np.int32).min
        or int(bins.max()) > np.iinfo(np.int32).max
    ):
        raise OverflowError("GINT bins exceed the int32 ABI")
    if bins[0] != 0 or bins[-1] != values.size or np.any(np.diff(bins) < 0):
        raise ValueError("GINT bins must be monotone and cover every task")
    if not np.all(np.isfinite(floors)) or not np.all(np.isfinite(values)):
        raise ValueError("GINT floors and log_q must be finite")
    if isinstance(task_start, (bool, np.bool_)) or not isinstance(
        task_start, (int, np.integer)
    ):
        raise TypeError("task_start must be an integer")
    if isinstance(task_stop, (bool, np.bool_)) or not isinstance(
        task_stop, (int, np.integer)
    ):
        raise TypeError("task_stop must be an integer")
    task_start, task_stop = int(task_start), int(task_stop)
    if not 0 <= task_start < task_stop <= values.size:
        raise IndexError("restricted task span is outside its cp task list")

    first_bin = int(np.searchsorted(bins, task_start, side="right") - 1)
    last_bin = int(np.searchsorted(bins, task_stop - 1, side="right") - 1)
    if first_bin != last_bin:
        raise ValueError("restricted task span crosses an original GINT bin")
    floor = float(floors[first_bin])
    # `_make_s_index` assigns the less-negative edge as the bin floor.  A
    # larger log_q would invalidate the screening upper-bound argument.
    rounding = 32 * np.finfo(np.float64).eps * max(1.0, abs(floor))
    if np.any(values[task_start:task_stop] > floor + rounding):
        raise ValueError("restricted task log_q exceeds its original bin floor")
    return RestrictedTaskSpan(
        cp_id=cp_id,
        task_start=task_start,
        task_stop=task_stop,
        bin_index=first_bin,
        bin_floor=floor,
    )


def original_ao_shell_support(
    coeff: np.ndarray, shell_ao_locs: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """Return all sorted segmented shells supporting each original AO."""

    coeff = np.asarray(coeff)
    locs = np.asarray(shell_ao_locs)
    if coeff.ndim != 2:
        raise ValueError("coefficient map must be a matrix")
    if not np.all(np.isfinite(coeff)):
        raise ValueError("coefficient map must be finite")
    if locs.ndim != 1 or locs.size < 2:
        raise ValueError("shell AO locations must be one-dimensional")
    if not np.issubdtype(locs.dtype, np.integer):
        raise TypeError("shell AO locations must contain integers")
    if locs[0] != 0 or locs[-1] != coeff.shape[0] or np.any(np.diff(locs) <= 0):
        raise ValueError("shell AO locations do not partition coefficient rows")

    supports: list[tuple[int, ...]] = []
    for original_ao in range(coeff.shape[1]):
        shells = tuple(
            shell
            for shell, (start, stop) in enumerate(zip(locs[:-1], locs[1:]))
            if np.any(coeff[int(start) : int(stop), original_ao] != 0.0)
        )
        if not shells:
            raise ValueError(
                f"original AO {original_ao} has no exact nonzero shell support"
            )
        supports.append(shells)
    return tuple(supports)


def _require_single_shell_support(
    supports: Sequence[Sequence[int]],
) -> None:
    """Fail closed outside the restricted benchmark release's AO mapping."""

    if any(len(shells) != 1 for shells in supports):
        raise NotImplementedError(
            "the restricted-task benchmark release requires each original "
            "AO to have exactly one segmented-shell support"
        )


class GINTAOPairColumnProvider:
    """Evaluate selected original-AO ERI columns with bounded GPU blocks.

    Parameters
    ----------
    mol
        A built PySCF molecule using spherical AOs.  Pair indices refer to its
        original AO ordering.  Cartesian molecules and coefficient maps with
        multi-shell original-AO support fail closed in this release.
    direct_scf_tol
        Schwarz screening threshold passed to ``_VHFOpt`` and GINT.
    group_size
        Target number of sorted Cartesian AOs in a contraction group.  The
        existing ``_VHFOpt`` splitter keeps at least two shells together, so a
        high-angular-momentum group can exceed this target.  The actual bound
        is reported by ``maximum_quartet_elements``.
    max_block_bytes
        Optional hard allocation guard, checked against the actual groups
        created by ``_VHFOpt`` before any quartet is evaluated.
    transfer_counter
        Optional :class:`gpu4pyscf.cc.device_runtime.TransferCounter`.  Host
        launch/pivot control is recorded as zero-byte events.  Known array
        transfers retain direction, operation, and byte counts.  The final
        device-generated ``vhfopt.coeff`` size is reported separately from
        unresolved setup-transfer bytes.

    Notes
    -----
    ``diag_block_with_triu=True`` is a correctness invariant.  It makes every
    ordered shell pair available inside a diagonal cp group; no implicit
    one-half scaling is then needed.  The provider verifies the resulting task
    inventory before it evaluates any integral.
    """

    factorization = "gint-four-center-restricted-task-columns"
    backend = "cupy"
    performance_eligible = False

    def __init__(
        self,
        mol: Any,
        *,
        direct_scf_tol: float = 1e-13,
        group_size: int = 16,
        max_block_bytes: Optional[int] = None,
        transfer_counter: Any = None,
        omega: float = 0.0,
    ) -> None:
        try:
            import cupy
            from gpu4pyscf.scf import int4c2e
        except Exception as exc:  # pragma: no cover - GPU environment only
            raise RuntimeError(
                "GINTAOPairColumnProvider requires CuPy and GPU4PySCF GINT"
            ) from exc

        if not np.isfinite(direct_scf_tol) or not 0 < direct_scf_tol <= 1:
            raise ValueError("direct_scf_tol must be finite and lie in (0, 1]")
        if isinstance(group_size, (bool, np.bool_)) or not isinstance(
            group_size, (int, np.integer)
        ):
            raise TypeError("group_size must be an integer")
        if int(group_size) < 1:
            raise ValueError("group_size must be positive")
        if max_block_bytes is not None:
            if isinstance(max_block_bytes, (bool, np.bool_)) or not isinstance(
                max_block_bytes, (int, np.integer)
            ):
                raise TypeError("max_block_bytes must be an integer")
            if int(max_block_bytes) < 8:
                raise ValueError("max_block_bytes must be at least 8")
        if not np.isfinite(omega):
            raise ValueError("omega must be finite")
        if getattr(mol, "nao_nr", None) is None:
            raise TypeError("mol must be a built PySCF Mole-like object")
        if bool(getattr(mol, "cart", False)):
            raise NotImplementedError(
                "the restricted-task benchmark release supports spherical "
                "original AOs only"
            )

        if transfer_counter is None:
            from gpu4pyscf.cc.device_runtime import TransferCounter

            transfer_counter = TransferCounter()

        self._cupy = cupy
        self._int4c2e = int4c2e
        self.mol = mol
        self.direct_scf_tol = float(direct_scf_tol)
        self.group_size = int(group_size)
        self.max_block_bytes = (
            None if max_block_bytes is None else int(max_block_bytes)
        )
        self.omega = float(omega)
        self.transfer_counter = transfer_counter
        self._explicit_transfer_bytes: dict[str, dict[str, int]] = {
            "h2d": {},
            "d2h": {},
        }
        self._explicit_transfer_counts: dict[str, dict[str, int]] = {
            "h2d": {},
            "d2h": {},
        }
        self._logical_device_payloads: dict[str, dict[str, Any]] = {}
        self.pair_map = SortedAOPairMap(int(mol.nao_nr()))

        vhfopt = int4c2e._VHFOpt(mol, "int2e")
        # ``basis_seg_contraction`` creates the final dense coefficient matrix
        # on the device from uploaded C2S/decontraction blocks.  Its resident
        # size is useful capacity evidence, but it is not the number of PCIe
        # bytes transferred.  Keep it out of the byte counter and expose the
        # unresolved internal setup operations in the audit below.
        self._record_logical_device_payload(
            "provider_setup_vhfopt_coeff",
            int(vhfopt.coeff.nbytes),
            provenance=(
                "gpu4pyscf.gto.mole.basis_seg_contraction device-generated "
                "block diagonal"
            ),
        )
        vhfopt.build(
            cutoff=self.direct_scf_tol,
            group_size=self.group_size,
            diag_block_with_triu=True,
        )
        self._record_data_transfer(
            "h2d",
            "provider_setup_vhfopt_log_q",
            sum(int(values.nbytes) for values in vhfopt.log_qs),
            count=len(vhfopt.log_qs),
        )
        if int(vhfopt.uniq_l_ctr[:, 0].max()) > int4c2e.LMAX_ON_GPU:
            raise NotImplementedError(
                "the selected-column adapter does not mix CPU high-l ERIs "
                "into a CuPy column"
            )
        self.vhfopt = vhfopt
        self.diag_block_with_triu = True
        self._coeff = cupy.asarray(vhfopt.coeff)
        if self._coeff.ndim != 2 or self._coeff.shape[1] != self.pair_map.nao:
            raise RuntimeError("_VHFOpt coefficient map has an unexpected shape")

        self._sorted_nao = int(self._coeff.shape[0])
        self._group_i, self._group_j = np.tril_indices(
            len(vhfopt.uniq_l_ctr)
        )
        self._record_data_transfer(
            "h2d",
            "provider_setup_gint_basis_product_cache",
            self._gint_basis_product_cache_h2d_bytes(),
            count=5,
        )
        shell_offsets = np.asarray(vhfopt.l_ctr_offsets, dtype=np.int64)
        cart_ao_loc = np.asarray(vhfopt.mol.ao_loc_nr(cart=True), dtype=np.int64)
        self._shell_ao_locs = cart_ao_loc
        self._group_ao_locs = cart_ao_loc[shell_offsets]
        coeff_host = cupy.asnumpy(self._coeff)
        self._record_data_transfer(
            "d2h", "provider_setup_coeff_support_metadata", int(coeff_host.nbytes)
        )
        self._ao_shell_support = original_ao_shell_support(
            coeff_host, self._shell_ao_locs
        )
        self._single_shell_support = all(
            len(shells) == 1 for shells in self._ao_shell_support
        )
        _require_single_shell_support(self._ao_shell_support)
        del coeff_host

        self._shell_to_group = np.empty(vhfopt.mol.nbas, dtype=np.int32)
        for group, (start, stop) in enumerate(
            zip(shell_offsets[:-1], shell_offsets[1:])
        ):
            self._shell_to_group[int(start) : int(stop)] = group
        self._q_cond = np.asarray(vhfopt.get_q_cond(), dtype=np.float64)
        if self._q_cond.shape != (vhfopt.mol.nbas, vhfopt.mol.nbas):
            raise RuntimeError("_VHFOpt q_cond has an unexpected shape")
        if not np.all(np.isfinite(self._q_cond)):
            raise RuntimeError("_VHFOpt q_cond must be finite")
        (
            self._task_descriptors,
            self._task_lookup,
            self._cp_bins,
            self._cp_floors,
            self._cp_log_q,
        ) = self._build_and_validate_task_inventory()
        self._pair_task_cache: dict[tuple[int, int], tuple[GINTTaskDescriptor, ...]] = {}

        if (
            self.max_block_bytes is not None
            and self.maximum_quartet_elements * np.dtype(np.float64).itemsize
            > self.max_block_bytes
        ):
            required = self.maximum_quartet_elements * np.dtype(np.float64).itemsize
            raise MemoryError(
                "largest GINT quartet block requires "
                f"{required} bytes, exceeding max_block_bytes="
                f"{self.max_block_bytes}"
            )
        self._pair_i_device = cupy.asarray(self.pair_map.pair_i)
        self._pair_j_device = cupy.asarray(self.pair_map.pair_j)
        self._record_data_transfer(
            "h2d",
            "provider_setup_pair_indices",
            int(self.pair_map.pair_i.nbytes + self.pair_map.pair_j.nbytes),
            count=2,
        )

        self.column_calls = 0
        self.diagonal_calls = 0
        self.gint_fill_calls = 0
        self.kernel_launches = 0
        self.host_control_transfers = 0
        self.restricted_fill_calls = 0
        self.double_restricted_fill_calls = 0
        self.requested_task_quartets = 0
        self.evaluated_task_quartets = 0
        self.full_task_quartet_equivalent = 0
        self.column_support_tasks = 0
        self.diagonal_task_batches = 0
        self.maximum_observed_block_elements = 0
        self.diagonal_mode = "not-run"
        self._diagonal = None

    @property
    def dimension(self) -> int:
        return self.pair_map.dimension

    @property
    def materializes_pair_matrix(self) -> bool:
        return False

    @property
    def pair_indices_device(self) -> tuple[Any, Any]:
        """Resident lower-triangle original-AO pair indices."""

        return self._pair_i_device, self._pair_j_device

    @property
    def maximum_quartet_elements(self) -> int:
        group_pair_areas = []
        for cp_id in range(len(self._group_i)):
            i0, i1, j0, j1 = self._group_bounds(cp_id)
            group_pair_areas.append((i1 - i0) * (j1 - j0))
        task_pair_areas = []
        for task in self._task_descriptors:
            i0, i1, j0, j1 = self._task_bounds(task)
            task_pair_areas.append((i1 - i0) * (j1 - j0))
        return int(max(group_pair_areas, default=0) * max(task_pair_areas, default=0))

    def _record_data_transfer(
        self,
        direction: str,
        operation: str,
        nbytes: int,
        *,
        count: int = 1,
    ) -> None:
        """Record one byte-counted transfer with an operation-level label."""

        if direction not in {"h2d", "d2h"}:
            raise ValueError("transfer direction must be h2d or d2h")
        if not operation or "/" in operation:
            raise ValueError("transfer operation must be a non-empty path segment")
        nbytes, count = int(nbytes), int(count)
        if nbytes < 0 or count < 0:
            raise ValueError("transfer bytes and count must be non-negative")
        byte_ledger = self._explicit_transfer_bytes[direction]
        count_ledger = self._explicit_transfer_counts[direction]
        byte_ledger[operation] = byte_ledger.get(operation, 0) + nbytes
        count_ledger[operation] = count_ledger.get(operation, 0) + count
        self.transfer_counter.record(
            direction, nbytes, count=count, operation=operation
        )

    def _record_logical_device_payload(
        self,
        operation: str,
        resident_bytes: int,
        *,
        provenance: str,
    ) -> None:
        """Record resident device capacity without calling it a transfer."""

        if not operation or "/" in operation:
            raise ValueError("payload operation must be a non-empty path segment")
        resident_bytes = int(resident_bytes)
        if resident_bytes < 0:
            raise ValueError("resident payload bytes must be non-negative")
        if operation in self._logical_device_payloads:
            raise ValueError(f"logical device payload {operation!r} is duplicated")
        self._logical_device_payloads[operation] = {
            "resident_bytes": resident_bytes,
            "actual_transfer_bytes": None,
            "classification": "logical-device-resident-payload",
            "provenance": str(provenance),
        }

    def _gint_basis_product_cache_h2d_bytes(self) -> int:
        """Exact payload of the five ``DEVICE_INIT`` calls in bpcache.cu."""

        vhfopt = self.vhfopt
        nbas = int(vhfopt.mol.nbas)
        n_bas_pairs = sum(len(tasks) for tasks in vhfopt.pair2bra)
        n_primitive_pairs = 0
        for cp_id, tasks in enumerate(vhfopt.pair2bra):
            group_i = int(self._group_i[cp_id])
            group_j = int(self._group_j[cp_id])
            nprim_i = int(vhfopt.uniq_l_ctr[group_i, 1])
            nprim_j = int(vhfopt.uniq_l_ctr[group_j, 1])
            n_primitive_pairs += len(tasks) * nprim_i * nprim_j
        return int(
            (nbas + 1) * np.dtype(np.int32).itemsize
            + 3 * nbas * np.dtype(np.float64).itemsize
            + nbas * np.dtype(np.int32).itemsize
            + 7 * n_primitive_pairs * np.dtype(np.float64).itemsize
            + 2 * n_bas_pairs * np.dtype(np.int32).itemsize
        )

    def explicit_transfer_ledger(self) -> dict[str, Any]:
        """Return a JSON-safe ledger without double-counting direction totals."""

        directions: dict[str, Any] = {}
        for direction in ("h2d", "d2h"):
            operations = {
                operation: {
                    "bytes": int(nbytes),
                    "count": int(
                        self._explicit_transfer_counts[direction].get(
                            operation, 0
                        )
                    ),
                }
                for operation, nbytes in self._explicit_transfer_bytes[
                    direction
                ].items()
            }
            directions[direction] = {
                "bytes": sum(item["bytes"] for item in operations.values()),
                "count": sum(item["count"] for item in operations.values()),
                "operations": operations,
            }
        unresolved = [
            {
                "direction": "h2d",
                "operation": "provider_setup_vhfopt_coeff_components",
                "bytes": None,
                "reason": (
                    "basis_seg_contraction uploads C2S/decontraction blocks "
                    "and block-diagonal metadata before generating the final "
                    "coefficient matrix on device; the _VHFOpt API does not "
                    "expose those cache- and basis-dependent transfer bytes"
                ),
            },
            {
                "direction": "h2d",
                "operation": "provider_setup_vhfopt_ao_sort_indices",
                "bytes": None,
                "reason": (
                    "_VHFOpt.build performs device advanced indexing from a "
                    "NumPy AO permutation without exposing the runtime copy"
                ),
            },
            {
                "direction": "h2d",
                "operation": "cuda_runtime_kernel_argument_marshalling",
                "bytes": None,
                "reason": (
                    "CUDA launch-parameter transport is runtime managed and "
                    "is not reported as an application array transfer"
                ),
            },
        ]
        return {
            "schema": "gpu4pyscf.gint-explicit-transfer-ledger.v1",
            "directions": directions,
            "total_bytes": sum(item["bytes"] for item in directions.values()),
            "logical_device_payloads": dict(self._logical_device_payloads),
            "coverage": {
                "vhfopt_coeff_resident_payload": (
                    "provider_setup_vhfopt_coeff"
                    in self._logical_device_payloads
                ),
                "vhfopt_coeff_h2d_exact": False,
                "vhfopt_log_q_uploads": True,
                "gint_basis_product_cache_device_init": True,
                "gint_per_call_constant_cache_copy": True,
                "provider_python_array_transfers": True,
                "cuda_runtime_kernel_argument_marshalling": False,
            },
            "audit": {
                "known_explicit_array_transfers_byte_counted": True,
                "provider_setup_h2d_complete": False,
                "unresolved_operations": unresolved,
            },
            "complete_for_declared_payloads": False,
        }

    def _build_and_validate_task_inventory(self):
        """Freeze and audit the host task metadata used for restrictions."""

        vhfopt = self.vhfopt
        group_count = len(vhfopt.uniq_l_ctr)
        expected_cp_count = group_count * (group_count + 1) // 2
        if expected_cp_count >= np.iinfo(np.int32).max:
            raise MemoryError("GINT cp count exceeds the int32 ABI")
        if len(vhfopt.log_qs) != expected_cp_count:
            raise RuntimeError("GINT cp inventory is not a full lower triangle")
        if not (
            len(vhfopt.pair2bra)
            == len(vhfopt.pair2ket)
            == len(vhfopt.bins)
            == len(vhfopt.bins_floor)
            == expected_cp_count
        ):
            raise RuntimeError("GINT cp metadata lengths disagree")

        bas = np.asarray(vhfopt.mol._bas)
        angular_slot = self._int4c2e.gto.ANG_OF
        nprim_slot = self._int4c2e.gto.NPRIM_OF
        offsets = np.asarray(vhfopt.l_ctr_offsets, dtype=np.int64)
        for group, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            expected = np.asarray(vhfopt.uniq_l_ctr[group], dtype=np.int64)
            actual = bas[int(start) : int(stop), [angular_slot, nprim_slot]]
            if actual.shape[0] < 1 or np.any(actual != expected):
                raise RuntimeError("a GINT group mixes angular/nprim shell types")

        descriptors: list[GINTTaskDescriptor] = []
        lookup: dict[tuple[int, int], GINTTaskDescriptor] = {}
        cp_bins: list[np.ndarray] = []
        cp_floors: list[np.ndarray] = []
        cp_log_q: list[np.ndarray] = []
        metadata_d2h_bytes = 0
        for cp_id in range(expected_cp_count):
            shell_i = np.asarray(vhfopt.pair2bra[cp_id], dtype=np.int64)
            shell_j = np.asarray(vhfopt.pair2ket[cp_id], dtype=np.int64)
            if shell_i.ndim != 1 or shell_j.shape != shell_i.shape:
                raise RuntimeError("GINT shell-pair task arrays disagree")
            raw_bins = np.asarray(vhfopt.bins[cp_id])
            if not np.issubdtype(raw_bins.dtype, np.integer):
                raise RuntimeError("GINT bins must be integer-valued")
            if raw_bins.size and (
                int(raw_bins.min()) < 0
                or int(raw_bins.max()) > np.iinfo(np.int32).max
            ):
                raise RuntimeError("GINT bins exceed the int32 ABI")
            bins = raw_bins.astype(np.int32, copy=True)
            floors = np.asarray(vhfopt.bins_floor[cp_id], dtype=np.float64).copy()
            logs = self._cupy.asnumpy(vhfopt.log_qs[cp_id]).astype(
                np.float64, copy=False
            )
            metadata_d2h_bytes += int(logs.nbytes)
            # This call audits all bin invariants even when there are no tasks.
            if bins.ndim != 1 or bins.size < 2:
                raise RuntimeError("GINT bins must contain at least one bin")
            if floors.size != bins.size - 1:
                raise RuntimeError("GINT bins/floors have inconsistent lengths")
            if bins[0] != 0 or bins[-1] != logs.size or np.any(np.diff(bins) < 0):
                raise RuntimeError("GINT bins do not cover the cp task list")
            if not np.all(np.isfinite(floors)) or not np.all(np.isfinite(logs)):
                raise RuntimeError("GINT screening metadata must be finite")
            if shell_i.size != logs.size:
                raise RuntimeError("GINT task and screening arrays disagree")

            group_i, group_j = int(self._group_i[cp_id]), int(self._group_j[cp_id])
            for task_id, (ish, jsh) in enumerate(zip(shell_i, shell_j)):
                ish, jsh = int(ish), int(jsh)
                if not 0 <= ish < vhfopt.mol.nbas or not 0 <= jsh < vhfopt.mol.nbas:
                    raise RuntimeError("GINT task names an out-of-range shell")
                if int(self._shell_to_group[ish]) != group_i or int(
                    self._shell_to_group[jsh]
                ) != group_j:
                    raise RuntimeError("GINT task shell does not match its cp group")
                span = restricted_task_span(
                    cp_id,
                    task_id,
                    task_id + 1,
                    bins,
                    floors,
                    logs,
                )
                expected_log_q = min(float(np.log(self._q_cond[ish, jsh])), 0.0)
                if not np.isclose(
                    logs[task_id], expected_log_q, rtol=1e-12, atol=1e-14
                ):
                    raise RuntimeError("GINT task log_q does not match q_cond")
                descriptor = GINTTaskDescriptor(
                    cp_id=cp_id,
                    task_id=task_id,
                    shell_i=ish,
                    shell_j=jsh,
                    group_i=group_i,
                    group_j=group_j,
                    span=span,
                    log_q=float(logs[task_id]),
                )
                key = (ish, jsh)
                if key in lookup:
                    raise RuntimeError("GINT shell-pair task is not unique")
                lookup[key] = descriptor
                descriptors.append(descriptor)
            bins.flags.writeable = False
            floors.flags.writeable = False
            logs.flags.writeable = False
            cp_bins.append(bins)
            cp_floors.append(floors)
            cp_log_q.append(logs)
        self._record_data_transfer(
            "d2h",
            "provider_setup_log_q_metadata",
            metadata_d2h_bytes,
            count=expected_cp_count,
        )

        # This proves that the build retained both orientations inside every
        # diagonal group (`diag_block_with_triu=True`).  Missing screened tasks
        # are permitted only when the same q_cond used by `_VHFOpt` is <= tol.
        for first_shell in range(vhfopt.mol.nbas):
            first_group = int(self._shell_to_group[first_shell])
            for second_shell in range(vhfopt.mol.nbas):
                second_group = int(self._shell_to_group[second_shell])
                if first_group < second_group:
                    continue
                key = (first_shell, second_shell)
                present = key in lookup
                required = bool(
                    self._q_cond[first_shell, second_shell] > self.direct_scf_tol
                )
                if present != required:
                    raise RuntimeError(
                        "GINT task inventory disagrees with q_cond screening; "
                        "diag_block_with_triu=True is required"
                    )
        return (
            tuple(descriptors),
            lookup,
            tuple(cp_bins),
            tuple(cp_floors),
            tuple(cp_log_q),
        )

    def _record_control(self, kind: str, *, count: int = 1) -> None:
        self.host_control_transfers += int(count)
        self.transfer_counter.record(kind, 0, count=count)

    def _group_bounds(self, cp_id: int) -> tuple[int, int, int, int]:
        cp_id = _as_index(cp_id, upper_bound=len(self._group_i), name="cp id")
        first = int(self._group_i[cp_id])
        second = int(self._group_j[cp_id])
        locs = self._group_ao_locs
        return (
            int(locs[first]),
            int(locs[first + 1]),
            int(locs[second]),
            int(locs[second + 1]),
        )

    def _task_bounds(
        self, task: GINTTaskDescriptor
    ) -> tuple[int, int, int, int]:
        if not isinstance(task, GINTTaskDescriptor):
            raise TypeError("task must be a GINTTaskDescriptor")
        locs = self._shell_ao_locs
        return (
            int(locs[task.shell_i]),
            int(locs[task.shell_i + 1]),
            int(locs[task.shell_j]),
            int(locs[task.shell_j + 1]),
        )

    def _task_layout(
        self, cp_id: int, task: Optional[GINTTaskDescriptor]
    ) -> tuple[np.ndarray, np.ndarray]:
        if task is None:
            return self._cp_bins[cp_id], self._cp_floors[cp_id]
        if task.cp_id != cp_id or task.span.cp_id != cp_id:
            raise RuntimeError("restricted task belongs to a different cp")
        if task.span.task_count != 1 or task.span.task_start != task.task_id:
            raise RuntimeError("the selected-column path requires [t, t+1]")
        bins = np.asarray(
            [task.span.task_start, task.span.task_stop], dtype=np.int32
        )
        floors = np.asarray([task.span.bin_floor], dtype=np.float64)
        return bins, floors

    def _tasks_for_original_pair(
        self, first: int, second: int
    ) -> tuple[GINTTaskDescriptor, ...]:
        first = _as_index(first, upper_bound=self.pair_map.nao, name="first AO")
        second = _as_index(second, upper_bound=self.pair_map.nao, name="second AO")
        cache_key = (first, second)
        cached = self._pair_task_cache.get(cache_key)
        if cached is not None:
            return cached

        selected: dict[tuple[int, int], GINTTaskDescriptor] = {}
        for first_shell in self._ao_shell_support[first]:
            for second_shell in self._ao_shell_support[second]:
                first_group = int(self._shell_to_group[first_shell])
                second_group = int(self._shell_to_group[second_shell])
                if first_group < second_group:
                    key = (second_shell, first_shell)
                else:
                    key = (first_shell, second_shell)
                task = self._task_lookup.get(key)
                if task is None:
                    if (
                        self._q_cond[first_shell, second_shell]
                        > self.direct_scf_tol
                    ):
                        raise RuntimeError(
                            "an unscreened original-AO support shell pair has "
                            "no GINT task"
                        )
                    continue
                selected[(task.cp_id, task.task_id)] = task
        result = tuple(selected[key] for key in sorted(selected))
        self._pair_task_cache[cache_key] = result
        return result

    def _fill_quartet_block(
        self,
        cp_ij_id: int,
        cp_kl_id: int,
        *,
        ij_task: Optional[GINTTaskDescriptor] = None,
        kl_task: Optional[GINTTaskDescriptor] = None,
    ):
        cupy = self._cupy
        cp_ij_id = _as_index(
            cp_ij_id, upper_bound=len(self._group_i), name="bra cp id"
        )
        cp_kl_id = _as_index(
            cp_kl_id, upper_bound=len(self._group_i), name="ket cp id"
        )
        i0, i1, j0, j1 = (
            self._group_bounds(cp_ij_id)
            if ij_task is None
            else self._task_bounds(ij_task)
        )
        k0, k1, l0, l1 = (
            self._group_bounds(cp_kl_id)
            if kl_task is None
            else self._task_bounds(kl_task)
        )
        if ij_task is not None and ij_task.cp_id != cp_ij_id:
            raise RuntimeError("bra task belongs to a different cp")
        if kl_task is not None and kl_task.cp_id != cp_kl_id:
            raise RuntimeError("ket task belongs to a different cp")
        ni, nj = i1 - i0, j1 - j0
        nk, nl = k1 - k0, l1 - l0
        dimensions = (nl, nk, nj, ni)
        if any(width <= 0 for width in dimensions):
            raise RuntimeError("GINT block has a non-positive AO extent")
        elements = int(np.prod(dimensions, dtype=object))
        block_bytes = elements * np.dtype(np.float64).itemsize
        if elements > np.iinfo(np.intp).max:
            raise MemoryError("GINT block element count exceeds platform index range")
        if self.max_block_bytes is not None and block_bytes > self.max_block_bytes:
            raise MemoryError(
                f"GINT quartet block requires {block_bytes} bytes, exceeding "
                f"max_block_bytes={self.max_block_bytes}"
            )
        self.maximum_observed_block_elements = max(
            self.maximum_observed_block_elements, elements
        )
        block = cupy.zeros((nl, nk, nj, ni), dtype=np.float64, order="C")
        stride_values = [1, ni, ni * nj, ni * nj * nk]
        if max(stride_values) > np.iinfo(np.int32).max:
            raise MemoryError("GINT block strides exceed the int32 ABI")
        if max(i1, j1, k1, l1, self._sorted_nao) > np.iinfo(np.int32).max:
            raise MemoryError("GINT AO offsets exceed the int32 ABI")
        strides = np.asarray(stride_values, dtype=np.int32)
        ao_offsets = np.asarray([i0, j0, k0, l0], dtype=np.int32)
        bins_locs_ij, bins_floor_ij = self._task_layout(cp_ij_id, ij_task)
        bins_locs_kl, bins_floor_kl = self._task_layout(cp_kl_id, kl_task)
        nbins_ij = len(bins_locs_ij) - 1
        nbins_kl = len(bins_locs_kl) - 1
        if nbins_ij < 1 or nbins_kl < 1:
            return block

        stream = cupy.cuda.get_current_stream()
        fn = self._int4c2e.libgint.GINTfill_int2e
        fn.restype = ctypes.c_int
        launch_count, evaluated_task_quartets = self._launch_statistics(
            bins_locs_ij,
            bins_locs_kl,
            bins_floor_ij,
            bins_floor_kl,
        )
        self._record_control("gint_fill_call_control")
        self._record_control("gint_kernel_launch_control", count=launch_count)
        self.gint_fill_calls += 1
        self.kernel_launches += launch_count
        requested_ij = int(bins_locs_ij[-1] - bins_locs_ij[0])
        requested_kl = int(bins_locs_kl[-1] - bins_locs_kl[0])
        self.requested_task_quartets += requested_ij * requested_kl
        self.evaluated_task_quartets += evaluated_task_quartets
        self.full_task_quartet_equivalent += int(
            int(self._cp_bins[cp_ij_id][-1])
            * int(self._cp_bins[cp_kl_id][-1])
        )
        if ij_task is not None or kl_task is not None:
            self.restricted_fill_calls += 1
        if ij_task is not None and kl_task is not None:
            self.double_restricted_fill_calls += 1
        # `GINTfill_int2e` begins with cudaMemcpyToSymbol(c_bpcache, ...).
        self._record_data_transfer(
            "h2d",
            "gint_fill_constant_cache",
            _GINT_BPCACHE_CONSTANT_BYTES,
        )
        error = fn(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            self.vhfopt.bpcache,
            ctypes.cast(block.data.ptr, ctypes.c_void_p),
            ctypes.c_int(self._sorted_nao),
            strides.ctypes.data_as(ctypes.c_void_p),
            ao_offsets.ctypes.data_as(ctypes.c_void_p),
            bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
            bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
            bins_floor_ij.ctypes.data_as(ctypes.c_void_p),
            bins_floor_kl.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbins_ij),
            ctypes.c_int(nbins_kl),
            ctypes.c_int(cp_ij_id),
            ctypes.c_int(cp_kl_id),
            ctypes.c_double(np.log(self.direct_scf_tol)),
            ctypes.c_double(self.omega),
        )
        if error != 0:
            raise RuntimeError(
                "GINTfill_int2e failed for contraction groups "
                f"({cp_ij_id}, {cp_kl_id}), error={error}"
            )
        return block

    def _launch_statistics(
        self,
        bins_ij: np.ndarray,
        bins_kl: np.ndarray,
        floors_ij: np.ndarray,
        floors_kl: np.ndarray,
    ) -> tuple[int, int]:
        """Mirror the public GINT driver's bin loop for exact bookkeeping."""

        log_cutoff = np.log(self.direct_scf_tol)
        launches = 0
        task_quartets = 0
        for kl_bin, floor_kl in enumerate(floors_kl):
            ntasks_kl = int(bins_kl[kl_bin + 1] - bins_kl[kl_bin])
            if ntasks_kl <= 0:
                continue
            ij_bin1 = 0
            for floor_ij in floors_ij:
                if floor_ij + floor_kl < log_cutoff:
                    break
                ij_bin1 += 1
            ntasks_ij = int(bins_ij[ij_bin1] - bins_ij[0])
            if ntasks_ij > 0:
                launches += 1
                task_quartets += ntasks_ij * ntasks_kl
        return launches, task_quartets

    def _task_pair_weight(
        self, task: GINTTaskDescriptor, first: int, second: int
    ):
        cupy = self._cupy
        k0, k1, l0, l1 = self._task_bounds(task)
        coeff_k_first = self._coeff[k0:k1, first]
        coeff_l_second = self._coeff[l0:l1, second]
        weight = cupy.outer(coeff_k_first, coeff_l_second)
        if task.group_i != task.group_j:
            weight = weight + cupy.outer(
                self._coeff[k0:k1, second],
                self._coeff[l0:l1, first],
            )
        return weight

    def _task_pair_weights(
        self, task: GINTTaskDescriptor, pivots: np.ndarray
    ):
        """Vectorized pair weights for one batched diagonal task."""

        cupy = self._cupy
        pivots = np.asarray(pivots, dtype=np.int64)
        pivot_device = cupy.asarray(pivots)
        self._record_data_transfer(
            "h2d", "diagonal_pivot_indices", int(pivots.nbytes)
        )
        first = self._pair_i_device[pivot_device]
        second = self._pair_j_device[pivot_device]
        i0, i1, j0, j1 = self._task_bounds(task)
        coeff_i_first = self._coeff[i0:i1][:, first].T
        coeff_j_second = self._coeff[j0:j1][:, second].T
        weights = coeff_i_first[:, :, None] * coeff_j_second[:, None, :]
        if task.group_i != task.group_j:
            coeff_i_second = self._coeff[i0:i1][:, second].T
            coeff_j_first = self._coeff[j0:j1][:, first].T
            weights = weights + (
                coeff_i_second[:, :, None] * coeff_j_first[:, None, :]
            )
        return pivot_device, weights

    def column(self, pivot: int):
        pivot = _as_index(pivot, upper_bound=self.dimension, name="pivot")
        first, second = self.pair_map.pair(pivot)
        self.column_calls += 1
        self._record_control("host_pivot_control")
        cupy = self._cupy
        cart_column = cupy.zeros(
            (self._sorted_nao, self._sorted_nao), dtype=np.float64
        )

        tasks = self._tasks_for_original_pair(first, second)
        self.column_support_tasks += len(tasks)
        for task in tasks:
            right_weight = self._task_pair_weight(task, first, second)
            for cp_ij_id in range(len(self.vhfopt.log_qs)):
                block = self._fill_quartet_block(
                    cp_ij_id, task.cp_id, kl_task=task
                )
                contribution = cupy.einsum(
                    "lkji,kl->ij", block, right_weight, optimize=True
                )
                i0, i1, j0, j1 = self._group_bounds(cp_ij_id)
                cart_column[i0:i1, j0:j1] += contribution
                if self._group_i[cp_ij_id] != self._group_j[cp_ij_id]:
                    cart_column[j0:j1, i0:i1] += contribution.T

        original_column = self._coeff.T @ cart_column @ self._coeff
        return original_column[
            self._pair_i_device, self._pair_j_device
        ].copy()

    def _batched_single_shell_diagonal(self):
        cupy = self._cupy
        batches: dict[GINTTaskDescriptor, list[int]] = {}
        for pivot, (first, second) in enumerate(
            zip(self.pair_map.pair_i, self.pair_map.pair_j)
        ):
            tasks = self._tasks_for_original_pair(int(first), int(second))
            if len(tasks) == 0:
                continue
            elif len(tasks) == 1:
                batches.setdefault(tasks[0], []).append(pivot)
            else:
                raise RuntimeError(
                    "single-shell original AO pair unexpectedly maps to "
                    "multiple GINT tasks"
                )

        diagonal = cupy.zeros(self.dimension, dtype=np.float64)
        for task, pivot_list in batches.items():
            pivots = np.asarray(pivot_list, dtype=np.int64)
            block = self._fill_quartet_block(
                task.cp_id,
                task.cp_id,
                ij_task=task,
                kl_task=task,
            )
            pivot_device, weights = self._task_pair_weights(task, pivots)
            values = cupy.einsum(
                "lkji,xij,xkl->x", block, weights, weights, optimize=True
            )
            diagonal[pivot_device] = values
            self.diagonal_task_batches += 1
        # ``diagonal`` was initialized to zero, which is the screened-source
        # value for pairs whose only shell quartet was removed by q_cond.
        self.diagonal_mode = "restricted-task-batched"
        return diagonal

    def diagonal(self):
        self.diagonal_calls += 1
        if self._diagonal is None:
            if not self._single_shell_support:  # defensive after init gate
                raise RuntimeError("multi-shell AO support passed the release gate")
            self._diagonal = self._batched_single_shell_diagonal()
        return self._diagonal.copy()

    def metadata(self) -> dict[str, Any]:
        return {
            "factorization": self.factorization,
            "backend": self.backend,
            "dimension": self.dimension,
            "nao": self.pair_map.nao,
            "sorted_cartesian_nao": self._sorted_nao,
            "direct_scf_tol": self.direct_scf_tol,
            "group_size": self.group_size,
            "max_block_bytes": self.max_block_bytes,
            "omega": self.omega,
            "materializes_pair_matrix": self.materializes_pair_matrix,
            "materializes_four_index_tensor": False,
            "performance_eligible": self.performance_eligible,
            "performance_gate": (
                "pending-water2-end-to-end-and-complete-transfer-audit"
            ),
            "release_scope": (
                "spherical-original-aos-with-single-segmented-shell-support"
            ),
            "diag_block_with_triu": self.diag_block_with_triu,
            "task_count": len(self._task_descriptors),
            "original_ao_single_shell_support": self._single_shell_support,
            "maximum_original_ao_support_shells": max(
                map(len, self._ao_shell_support), default=0
            ),
            "column_calls": self.column_calls,
            "column_support_tasks": self.column_support_tasks,
            "diagonal_calls": self.diagonal_calls,
            "diagonal_mode": self.diagonal_mode,
            "diagonal_task_batches": self.diagonal_task_batches,
            "gint_fill_calls": self.gint_fill_calls,
            "restricted_fill_calls": self.restricted_fill_calls,
            "double_restricted_fill_calls": self.double_restricted_fill_calls,
            "kernel_launches": self.kernel_launches,
            "requested_task_quartets": self.requested_task_quartets,
            "evaluated_task_quartets": self.evaluated_task_quartets,
            "full_task_quartet_equivalent": self.full_task_quartet_equivalent,
            "host_control_transfers": self.host_control_transfers,
            "maximum_quartet_elements": self.maximum_quartet_elements,
            "maximum_quartet_bytes": (
                self.maximum_quartet_elements
                * np.dtype(np.float64).itemsize
            ),
            "maximum_observed_block_elements": self.maximum_observed_block_elements,
            "maximum_observed_block_bytes": (
                self.maximum_observed_block_elements
                * np.dtype(np.float64).itemsize
            ),
            "explicit_transfers": self.explicit_transfer_ledger(),
        }


def _normalise_pivots(
    provider: GINTAOPairColumnProvider,
    pivots: Optional[Iterable[int]],
) -> tuple[int, ...]:
    if pivots is None:
        pivots = (0, provider.dimension // 2, provider.dimension - 1)
    result = tuple(
        _as_index(pivot, upper_bound=provider.dimension, name="pivot")
        for pivot in pivots
    )
    if not result:
        raise ValueError("pivots must contain at least one pair index")
    return result


def validate_water1_columns(
    provider: GINTAOPairColumnProvider,
    *,
    pivots: Optional[Iterable[int]] = None,
    tolerance: float = 2e-10,
    max_nao: int = 32,
) -> ColumnValidationResult:
    """Compare a few water-monomer columns with a dense PySCF oracle.

    This helper intentionally materializes a *small CPU validation oracle* and
    is guarded by molecular composition and AO-count checks.  It is excluded
    from benchmark timing and never used by ``column()`` or ``diagonal()``.
    """

    mol = provider.mol
    symbols = sorted(mol.atom_symbol(atom) for atom in range(mol.natm))
    if symbols != ["H", "H", "O"]:
        raise ValueError("validate_water1_columns requires one H2O molecule")
    if provider.pair_map.nao > int(max_nao):
        raise ValueError("water1 dense validation oracle exceeds max_nao")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    selected = _normalise_pivots(provider, pivots)
    intor = "int2e_cart" if mol.cart else "int2e_sph"
    reference = np.asarray(mol.intor(intor), dtype=np.float64).reshape(
        (provider.pair_map.nao,) * 4
    )
    errors = []
    cupy = provider._cupy
    for pivot in selected:
        first, second = provider.pair_map.pair(pivot)
        expected_host = reference[:, :, first, second][
            provider.pair_map.pair_i, provider.pair_map.pair_j
        ]
        expected = cupy.asarray(expected_host)
        provider._record_data_transfer(
            "h2d", "validation_oracle_column", int(expected_host.nbytes)
        )
        error = cupy.max(cupy.abs(provider.column(pivot) - expected))
        provider._record_data_transfer(
            "d2h", "validation_oracle_error", int(error.nbytes)
        )
        errors.append(float(error.item()))
    return ColumnValidationResult(
        pivots=selected,
        max_abs_error=max(errors),
        errors=tuple(errors),
        tolerance=float(tolerance),
    )


__all__ = [
    "ColumnValidationResult",
    "GINTAOPairColumnProvider",
    "GINTTaskDescriptor",
    "RestrictedTaskSpan",
    "SortedAOPairMap",
    "original_ao_shell_support",
    "restricted_task_span",
    "validate_water1_columns",
]
