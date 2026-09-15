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

"""Small, optional-CuPy runtime helpers for GPU CC calculations.

The CC kernels are intentionally kept separate from this module.  These
helpers provide the bookkeeping and allocation policy that kernels can share
without making a CPU-only import of :mod:`gpu4pyscf` depend on CuPy.  CuPy and
cuTensor are imported lazily; NumPy is always a supported execution backend.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import time
from collections import OrderedDict
from typing import Any, Iterator, Mapping, MutableMapping, Optional, Sequence

import numpy as np

try:  # CuPy is optional for importing and testing this module.
    import cupy as _cp
except Exception:  # pragma: no cover - exercised on systems without CuPy
    _cp = None


def _jsonable(value: Any) -> Any:
    """Convert common scientific Python values to JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    return str(value)


def contract_traced_ao_metric(
    staging: Any,
    coefficients: Any,
    *,
    array_module: Any = np,
    transpose_output: bool = False,
) -> Any:
    """Contract ``pqkk,pi,qj`` without a four-index outer product.

    CuPy's default path for the three-operand einsum may flatten the two
    coefficient operands into an ``O(nao**4)`` temporary.  Taking the occupied
    trace first reduces the operation to two matrix multiplications and keeps
    the largest intermediate at ``O(nao**2)``.
    """

    if getattr(staging, "ndim", None) != 4:
        raise ValueError("staging must be a rank-4 pqij tensor")
    if getattr(coefficients, "ndim", None) != 2:
        raise ValueError("coefficients must be a rank-2 matrix")
    if (
        staging.shape[0] != staging.shape[1]
        or staging.shape[2] != staging.shape[3]
        or staging.shape[0] != coefficients.shape[0]
    ):
        raise ValueError(
            "staging and coefficient dimensions are incompatible: "
            f"{staging.shape!r} and {coefficients.shape!r}"
        )
    traced = array_module.trace(staging, axis1=2, axis2=3)
    result = array_module.matmul(
        array_module.matmul(coefficients.T, traced), coefficients
    )
    return result.T if transpose_output else result


def make_block_lower_ao_layout(
    nao: int,
    blocks: Sequence[tuple[int, int, int, int]],
) -> dict[str, Any]:
    """Describe a lossless block-lower packing of a square AO matrix.

    Every lower off-diagonal shell block is stored once in C order and mirrored
    only during reconstruction.  Diagonal shell blocks are stored in full, so
    this layout does not assume element-wise symmetry within such a block.
    """

    nao = int(nao)
    if nao < 1:
        raise ValueError("nao must be positive")
    offsets: list[tuple[int, int]] = []
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    mirror_sources: list[np.ndarray] = []
    mirror_rows: list[np.ndarray] = []
    mirror_columns: list[np.ndarray] = []
    cursor = 0
    coverage = np.zeros((nao, nao), dtype=np.uint8)
    for raw in blocks:
        if len(raw) != 4:
            raise ValueError("each AO block must have four bounds")
        row0, row1, column0, column1 = (int(value) for value in raw)
        if not (0 <= column0 < column1 <= nao and 0 <= row0 < row1 <= nao):
            raise ValueError("AO block bounds are outside the matrix")
        if row0 < column0:
            raise ValueError("AO blocks must follow the lower-triangle shell schedule")
        if row0 == column0 and (row1 - row0) != (column1 - column0):
            raise ValueError("a diagonal AO shell block must be square")
        if row0 > column0 and row0 < column1:
            raise ValueError("off-diagonal AO shell blocks must not overlap")
        block_rows, block_columns = np.meshgrid(
            np.arange(row0, row1, dtype=np.int64),
            np.arange(column0, column1, dtype=np.int64),
            indexing="ij",
        )
        block_rows = block_rows.reshape(-1)
        block_columns = block_columns.reshape(-1)
        count = int(block_rows.size)
        offsets.append((cursor, cursor + count))
        rows.append(block_rows)
        columns.append(block_columns)
        coverage[block_rows, block_columns] += 1
        if row0 != column0:
            sources = np.arange(cursor, cursor + count, dtype=np.int64)
            mirror_sources.append(sources)
            mirror_rows.append(block_columns)
            mirror_columns.append(block_rows)
            coverage[block_columns, block_rows] += 1
        cursor += count
    if not np.all(coverage == 1):
        raise ValueError("AO shell blocks must cover the square matrix exactly once")

    def concatenate(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(parts)

    return {
        "nao": nao,
        "stored_pair_count": cursor,
        "block_offsets": tuple(offsets),
        "row_indices": concatenate(rows),
        "column_indices": concatenate(columns),
        "mirror_source_indices": concatenate(mirror_sources),
        "mirror_row_indices": concatenate(mirror_rows),
        "mirror_column_indices": concatenate(mirror_columns),
    }


def store_block_lower_ao_block(
    packed: Any,
    block: Any,
    offset: tuple[int, int],
    *,
    array_module: Any = np,
) -> None:
    """Store one AO shell block in a block-lower packed workspace."""

    if getattr(packed, "ndim", None) is None or packed.ndim < 1:
        raise ValueError("packed must have at least one dimension")
    if getattr(block, "ndim", None) != packed.ndim + 1:
        raise ValueError("block must have two AO axes followed by packed trailing axes")
    if tuple(block.shape[2:]) != tuple(packed.shape[1:]):
        raise ValueError("block and packed trailing dimensions do not match")
    start, stop = (int(value) for value in offset)
    if start < 0 or stop < start or stop > packed.shape[0]:
        raise ValueError("packed block offset is outside the workspace")
    if stop - start != int(block.shape[0]) * int(block.shape[1]):
        raise ValueError("packed block offset has the wrong size")
    packed[start:stop] = block.reshape((stop - start,) + block.shape[2:])


def _validate_block_lower_ao(
    packed: Any,
    coefficients: Any,
    layout: Mapping[str, Any],
) -> tuple[int, int]:
    if getattr(packed, "ndim", None) != 3:
        raise ValueError("packed must have shape (nao_pair,nocc,nocc)")
    if packed.shape[1] != packed.shape[2]:
        raise ValueError("packed occupied dimensions must be square")
    if getattr(coefficients, "ndim", None) != 2:
        raise ValueError("coefficients must be a rank-2 matrix")
    nao = int(coefficients.shape[0])
    if int(layout.get("nao", -1)) != nao:
        raise ValueError("packed AO layout does not match coefficients")
    if packed.shape[0] != int(layout.get("stored_pair_count", -1)):
        raise ValueError("packed AO-pair dimension does not match layout")
    return nao, int(packed.shape[1])


def _reconstruct_block_lower_ao(
    values: Any,
    layout: Mapping[str, Any],
    output: Any,
) -> Any:
    output[..., layout["row_indices"], layout["column_indices"]] = values
    output[
        ..., layout["mirror_row_indices"], layout["mirror_column_indices"]
    ] = values[..., layout["mirror_source_indices"]]
    return output


def contract_block_lower_traced_ao_metric(
    packed: Any,
    coefficients: Any,
    layout: Mapping[str, Any],
    *,
    array_module: Any = np,
    transpose_output: bool = False,
) -> Any:
    """Packed equivalent of :func:`contract_traced_ao_metric`."""

    nao, _nocc = _validate_block_lower_ao(
        packed, coefficients, layout
    )
    traced_packed = array_module.trace(packed, axis1=1, axis2=2)
    traced = array_module.empty((nao, nao), dtype=packed.dtype)
    _reconstruct_block_lower_ao(traced_packed, layout, traced)
    result = array_module.matmul(
        array_module.matmul(coefficients.T, traced), coefficients
    )
    return result.T if transpose_output else result


def contract_block_lower_ao_to_virtual(
    packed: Any,
    virtual_coefficients: Any,
    layout: Mapping[str, Any],
    *,
    occupied_block_size: int = 4,
    array_module: Any = np,
    scale: float = 1.0,
) -> Any:
    """Transform a packed symmetric ``pqji`` tensor to ``bjia``.

    Dense ``pqji`` reconstruction is limited to a caller-controlled block of
    the first occupied index.  The returned tensor has the same layout as
    ``einsum('aqji,qb->bjia', einsum('pqji,pa->aqji', W, C), C)``.
    """

    nao, nocc = _validate_block_lower_ao(
        packed, virtual_coefficients, layout
    )
    occupied_block_size = int(occupied_block_size)
    if occupied_block_size < 1:
        raise ValueError("occupied_block_size must be positive")
    nvir = int(virtual_coefficients.shape[1])
    output = array_module.empty(
        (nvir, nocc, nocc, nvir), dtype=packed.dtype
    )
    scaled_coefficients = virtual_coefficients * scale
    for j0 in range(0, nocc, occupied_block_size):
        j1 = min(nocc, j0 + occupied_block_size)
        count = (j1 - j0) * nocc
        # Keep ``values`` as a view.  Giving the reconstruction its explicit
        # (j,i,p,q) axes avoids a packed-size transpose copy.
        values = packed[:, j0:j1, :].transpose(1, 2, 0)
        dense_ji = array_module.empty(
            (j1 - j0, nocc, nao, nao), dtype=packed.dtype
        )
        _reconstruct_block_lower_ao(values, layout, dense_ji)
        dense = dense_ji.reshape(count, nao, nao)
        half = array_module.einsum(
            "xpq,pa->xaq", dense, scaled_coefficients
        )
        transformed = array_module.einsum(
            "xaq,qb->xab", half, virtual_coefficients
        )
        output[:, j0:j1, :, :] = transformed.reshape(
            j1 - j0, nocc, nvir, nvir
        ).transpose(3, 0, 1, 2)
        # Drop every block-local reference before allocating the next block.
        # CuPy can then reuse these exact-size pool blocks without temporarily
        # holding two reconstruction batches live.
        values = dense_ji = dense = half = transformed = None
    return output


class TransferCounter:
    """Count bytes moved between host and device (and between devices).

    ``record`` accepts arbitrary labels, while the convenience methods use
    the labels used by the CC profiling reports.  Counters are integers so
    that they remain exact and directly serializable.
    """

    def __init__(self) -> None:
        self._bytes: MutableMapping[str, int] = OrderedDict()
        self._counts: MutableMapping[str, int] = OrderedDict()
        self._operation_bytes: MutableMapping[tuple[str, str], int] = OrderedDict()
        self._operation_counts: MutableMapping[tuple[str, str], int] = OrderedDict()

    def record(
        self,
        kind: str,
        nbytes: int,
        count: int = 1,
        *,
        operation: Optional[str] = None,
    ) -> int:
        if not isinstance(kind, str) or not kind:
            raise ValueError("transfer kind must be a non-empty string")
        nbytes = int(nbytes)
        count = int(count)
        if nbytes < 0 or count < 0:
            raise ValueError("transfer bytes and count must be non-negative")
        self._bytes[kind] = self._bytes.get(kind, 0) + nbytes
        self._counts[kind] = self._counts.get(kind, 0) + count
        if operation is not None:
            operation = str(operation)
            if not operation:
                raise ValueError("transfer operation must be a non-empty string")
            key = (kind, operation)
            self._operation_bytes[key] = self._operation_bytes.get(key, 0) + nbytes
            self._operation_counts[key] = self._operation_counts.get(key, 0) + count
        return nbytes

    def add(self, other: "TransferCounter") -> "TransferCounter":
        for kind, nbytes in other._bytes.items():
            self.record(kind, nbytes, other._counts.get(kind, 0))
        for key, nbytes in other._operation_bytes.items():
            self._operation_bytes[key] = self._operation_bytes.get(key, 0) + nbytes
            self._operation_counts[key] = (
                self._operation_counts.get(key, 0)
                + other._operation_counts.get(key, 0)
            )
        return self

    def record_h2d(
        self, nbytes: int, count: int = 1, *, operation: Optional[str] = None
    ) -> int:
        return self.record("h2d", nbytes, count, operation=operation)

    def record_d2h(
        self, nbytes: int, count: int = 1, *, operation: Optional[str] = None
    ) -> int:
        return self.record("d2h", nbytes, count, operation=operation)

    def record_peer(
        self, nbytes: int, count: int = 1, *, operation: Optional[str] = None
    ) -> int:
        return self.record("peer", nbytes, count, operation=operation)

    # These aliases make call sites read naturally in transfer-heavy kernels.
    h2d = record_h2d
    d2h = record_d2h
    peer = record_peer

    @property
    def total_bytes(self) -> int:
        return sum(self._bytes.values())

    @property
    def total_transfers(self) -> int:
        return sum(self._counts.values())

    def to_dict(self) -> dict[str, Any]:
        by_kind = {
            kind: {
                "bytes": int(nbytes),
                "count": int(self._counts.get(kind, 0)),
            }
            for kind, nbytes in self._bytes.items()
        }
        by_operation: dict[str, dict[str, dict[str, int]]] = OrderedDict()
        for (kind, operation), nbytes in self._operation_bytes.items():
            by_operation.setdefault(kind, OrderedDict())[operation] = {
                "bytes": int(nbytes),
                "count": int(self._operation_counts.get((kind, operation), 0)),
            }
        return {
            "total_bytes": int(self.total_bytes),
            "total_transfers": int(self.total_transfers),
            "by_kind": by_kind,
            "by_operation": by_operation,
        }

    as_dict = to_dict

    def reset(self) -> None:
        self._bytes.clear()
        self._counts.clear()
        self._operation_bytes.clear()
        self._operation_counts.clear()


@dataclasses.dataclass
class _PhaseStackEntry:
    name: str
    start: float
    child_seconds: float = 0.0
    depth: int = 0


class RunMetrics:
    """Wall-clock phase metrics and counters for one CC run."""

    def __init__(
        self,
        name: str = "run",
        metadata: Optional[Mapping[str, Any]] = None,
        clock: Any = time.perf_counter,
    ) -> None:
        self.name = str(name)
        self.metadata = dict(metadata or {})
        self._clock = clock
        self.phases: list[dict[str, Any]] = []
        self.counters: MutableMapping[str, float] = OrderedDict()
        self.transfers = TransferCounter()
        self._stack: list[_PhaseStackEntry] = []

    @contextlib.contextmanager
    def phase(self, name: str, metadata: Optional[Mapping[str, Any]] = None, **fields: Any) -> Iterator["RunMetrics"]:
        """Record a phase, including exclusive time for nested phases."""
        entry = _PhaseStackEntry(str(name), self._clock(), depth=len(self._stack))
        self._stack.append(entry)
        try:
            yield self
        finally:
            elapsed = max(0.0, float(self._clock() - entry.start))
            self._stack.pop()
            exclusive = max(0.0, elapsed - entry.child_seconds)
            record = {
                "name": entry.name,
                "elapsed_s": elapsed,
                "exclusive_s": exclusive,
                "depth": entry.depth,
                "metadata": _jsonable(dict(metadata or {}, **fields)),
            }
            self.phases.append(record)
            if self._stack:
                self._stack[-1].child_seconds += elapsed

    measure = phase
    time_phase = phase

    def record_phase(
        self,
        name: str,
        elapsed_s: float,
        exclusive_s: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        depth: int = 0,
    ) -> None:
        elapsed_s = float(elapsed_s)
        depth = int(depth)
        if depth < 0:
            raise ValueError("phase depth must be non-negative")
        self.phases.append({
            "name": str(name),
            "elapsed_s": elapsed_s,
            "exclusive_s": elapsed_s if exclusive_s is None else float(exclusive_s),
            "depth": depth,
            "metadata": _jsonable(dict(metadata or {})),
        })

    def record_device_phase(
        self,
        name: str,
        elapsed_s: float,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        account_as_child: bool = False,
        depth: Optional[int] = None,
    ) -> None:
        """Record elapsed device work with an unambiguous timing semantic.

        GPU kernels are asynchronous with respect to Python.  Callers must
        obtain ``elapsed_s`` from device events after the event has completed;
        this method marks the resulting record so it cannot be confused with
        host enqueue time in machine-readable reports.  By default its depth
        is the current phase-stack depth, so a deferred device phase resolved
        while its parent is open retains the expected nesting.
        """

        fields = dict(metadata or {})
        semantic = fields.setdefault(
            "timing_semantics", "cuda-event-device-elapsed"
        )
        if semantic != "cuda-event-device-elapsed":
            raise ValueError(
                "device phases must use cuda-event-device-elapsed semantics"
            )
        elapsed_s = float(elapsed_s)
        if depth is None:
            depth = len(self._stack)
        self.record_phase(name, elapsed_s, metadata=fields, depth=depth)
        if account_as_child and self._stack:
            self._stack[-1].child_seconds += elapsed_s

    def increment(self, name: str, value: float = 1.0) -> float:
        self.counters[str(name)] = self.counters.get(str(name), 0.0) + float(value)
        return self.counters[str(name)]

    add_counter = increment

    def aggregate(self) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, float]] = OrderedDict()
        for phase in self.phases:
            item = totals.setdefault(phase["name"], {"calls": 0, "total_s": 0.0, "exclusive_s": 0.0})
            item["calls"] += 1
            item["total_s"] += float(phase["elapsed_s"])
            item["exclusive_s"] += float(phase["exclusive_s"])
        for item in totals.values():
            item["calls"] = int(item["calls"])
        return dict(totals)

    phase_totals = aggregate

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "name": self.name,
            "metadata": self.metadata,
            "phases": self.phases,
            "phase_totals": self.aggregate(),
            "counters": dict(self.counters),
            "transfers": self.transfers.to_dict(),
        })

    as_dict = to_dict

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


def _nvtx_api() -> Any:
    if _cp is not None:
        try:
            return _cp.cuda.nvtx
        except Exception:
            pass
    try:
        import nvtx  # type: ignore
        return nvtx
    except Exception:
        return None


class NVTXRange:
    """Best-effort NVTX range that is a no-op when NVTX is unavailable."""

    def __init__(self, name: str, color: Optional[str] = None, enabled: bool = True) -> None:
        self.name = str(name)
        self.color = color
        self.enabled = bool(enabled)
        self._api = None

    def __enter__(self) -> "NVTXRange":
        if self.enabled:
            self._api = _nvtx_api()
            if self._api is not None:
                try:
                    if hasattr(self._api, "RangePush"):
                        self._api.RangePush(self.name)
                    elif hasattr(self._api, "range_push"):
                        self._api.range_push(self.name)
                except Exception:
                    self._api = None
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._api is not None:
            try:
                if hasattr(self._api, "RangePop"):
                    self._api.RangePop()
                elif hasattr(self._api, "range_pop"):
                    self._api.range_pop()
            except Exception:
                pass


def nvtx_range(name: str, color: Optional[str] = None, enabled: bool = True) -> NVTXRange:
    return NVTXRange(name, color=color, enabled=enabled)


def canonical_ccsd_residual_update(
    t1: Any,
    t2: Any,
    *,
    fock: Any,
    mo_e_o: Any,
    mo_e_v: Any,
    orbo: Any,
    orbv: Any,
    wpq: Any,
    t1new: Any,
    t2new: Any,
    wVOov: Any,
    wVooV: Any,
    oooo: Any,
    ovoo: Any,
    oovv: Any,
    ovvo: Any,
    array_module: Any = np,
    phase: Any = None,
) -> tuple[Any, Any]:
    """Finish the canonical RCCSD update in one array namespace.

    The direct ``vvvv``/``ovvv`` driver supplies the five work arrays.  Every
    remaining contraction is accumulated in the namespace of ``array_module``
    and the function performs no host conversion.  This separation makes the
    transfer contract explicit: the caller uploads inputs before this function
    and downloads the completed amplitudes afterwards.

    ``phase`` may be a callable returning a context manager.  It is used by the
    benchmark harness to retain the established residual timing breakdown.
    Inputs named ``*new`` and ``wV*`` are scratch arrays and may be modified.
    """

    einsum = array_module.einsum
    diag = array_module.diag
    phase = phase or (lambda _name: contextlib.nullcontext())

    nocc = int(t1.shape[0])
    t2new *= 0.5  # the occupied-pair symmetrization is applied at the end

    with phase("residual_fock_intermediates"):
        fov = fock[:nocc, nocc:].copy()
        t1new += fock[:nocc, nocc:]

        foo = fock[:nocc, :nocc] - diag(mo_e_o)
        foo += 0.5 * einsum("ia,ja->ij", fock[:nocc, nocc:], t1)

        fvv = einsum("pa,qp,qb->ab", orbv, wpq, orbv)
        t1new -= einsum("ab,ib->ia", fvv, t1)

        fvv += fock[nocc:, nocc:] - diag(mo_e_v)
        fvv -= 0.5 * einsum("ia,ib->ab", t1, fock[:nocc, nocc:])

        foo += einsum("pi,qp,qj->ij", orbo, wpq, orbo)
        fov += einsum("pi,qp,qa->ia", orbo, wpq, orbv)

    with phase("residual_oooo_ladder"):
        tau = einsum("ia,jb->ijab", t1, t1)
        tau += t2
        woooo = einsum("ijab,kabl->ijkl", tau, ovvo)
        woooo += oooo.transpose(0, 2, 1, 3)
        tmp = einsum("la,jaik->lkji", t1, ovoo)
        woooo += tmp
        woooo += tmp.transpose(1, 0, 3, 2)
        t2new += 0.5 * einsum("ijkl,klab->ijab", woooo, tau)
        woooo = tau = tmp = None

    with phase("residual_linear_ov"):
        wVOov -= einsum("jbik,ka->bjia", ovoo, t1)
        t2new += wVOov.transpose(1, 2, 0, 3)

        wVooV += einsum("kbij,ka->bija", ovoo, t1)
        wVooV -= oovv.transpose(2, 0, 1, 3)
        wVOov += wVooV * 0.5
        wVOov += ovvo.transpose(2, 3, 0, 1)

        t2new += (ovvo * 0.5).transpose(0, 3, 1, 2)
        t1new += einsum("pi,pq,qa->ia", orbo, wpq, orbv)

        tmp = einsum("ic,kjbc->ikjb", t1, oovv)
        tmp += einsum("jbck,ic->jkib", ovvo, t1)
        t2new -= einsum("ka,jkib->jiba", t1, tmp)
        tmp = None

    with phase("residual_wVooV_ring"):
        tau = t2 * 0.5
        tau += einsum("ia,jb->ijab", t1, t1)
        wVooV += einsum("kbci,jkca->bija", ovvo, tau)
        tau = None

        tmp = einsum("jkca,ckib->jaib", t2, wVooV)
        t2new += tmp.transpose(2, 0, 1, 3)
        tmp *= 0.5
        t2new += tmp.transpose(0, 2, 1, 3)
        tmp = None

    with phase("residual_fock_dressing"):
        tau = einsum("ia,jb->iajb", t1 * 0.5, t1)
        tau += t2.transpose(0, 2, 1, 3)
        ovOV = ovvo.transpose(0, 1, 3, 2) * 2
        ovOV -= ovvo.transpose(3, 1, 0, 2)
        fvv -= einsum("jcia,jcib->ab", tau, ovOV)
        foo += einsum("iakb,jakb->ij", ovOV, tau)

    with phase("residual_wVOov_ring"):
        theta = t2.transpose(0, 2, 1, 3) * 2
        theta -= t2.transpose(1, 2, 0, 3)
        tau = theta * 0.25
        tau -= einsum("ia,jb->jaib", t1 * 0.5, t1)
        wVOov += einsum("kcia,kcjb->aijb", ovOV, tau)
        ovOV = tau = None

        t2new += einsum("kcia,ckjb->ijab", theta, wVOov)
        theta = wVOov = wVooV = None

    with phase("residual_singles_doubles_coupling"):
        t1new += einsum("jb,ijab->ia", fov, t2) * 2
        t1new -= einsum("jb,ijba->ia", fov, t2)
        ovoo_antisym = ovoo * 2
        ovoo_antisym -= ovoo.transpose(2, 1, 0, 3)
        t1new -= einsum("jbki,jkba->ia", ovoo_antisym, t2)
        ovoo_antisym = None

    with phase("residual_one_body_doubles"):
        ft_ij = foo + einsum("ja,ia->ij", 0.5 * t1, fov)
        ft_ab = fvv - einsum("ia,ib->ab", 0.5 * t1, fov)
        t2new += einsum("ijac,bc->ijab", t2, ft_ab)
        t2new -= einsum("ki,kjab->ijab", ft_ij, t2)

    with phase("residual_denominator_and_symmetry"):
        eia = mo_e_o[:, None] - mo_e_v
        t1new += einsum("ib,ab->ia", t1, fvv)
        t1new -= einsum("ja,ji->ia", t1, foo)
        t1new /= eia

        t2new = t2new + t2new.transpose(1, 0, 3, 2)
        t2new /= eia[:, None, :, None] + eia[:, None, :]

    return t1new, t2new


def rccsd_amplitude_vector_size(nocc: int, nvir: int) -> int:
    """Return the packed PySCF RCCSD amplitude-vector length.

    RCCSD doubles are symmetric in the compound occupied/virtual indices,
    ``t2[ia,jb] == t2[jb,ia]``.  PySCF's DIIS vector therefore stores the
    singles followed by the lower triangle of that compound-index matrix.
    Keeping this definition here lets the resident driver reproduce PySCF's
    convergence norm and DIIS ordering without constructing gigantic index
    arrays on the GPU.
    """

    nocc = int(nocc)
    nvir = int(nvir)
    if nocc < 0 or nvir < 0:
        raise ValueError("nocc and nvir must be non-negative")
    nov = nocc * nvir
    return nov + nov * (nov + 1) // 2


_PACK_RCCSD_KERNEL = None
_PACK_RCCSD_DELTA_KERNEL = None
_UNPACK_RCCSD_KERNEL = None


def _cupy_pack_rccsd_kernel() -> Any:
    global _PACK_RCCSD_KERNEL
    if _PACK_RCCSD_KERNEL is None:
        if _cp is None:  # pragma: no cover - guarded by callers
            raise RuntimeError("CuPy is unavailable")
        _PACK_RCCSD_KERNEL = _cp.ElementwiseKernel(
            "raw T t1, raw T t2, int64 nov, int64 nocc, int64 nvir",
            "T vector",
            r"""
            long long out_index = (long long)i;
            if (out_index < nov) {
                vector = t1[out_index];
            } else {
                long long k = out_index - nov;
                long long row = (long long)((sqrt(8.0 * (double)k + 1.0) - 1.0) * 0.5);
                while ((row + 1) * (row + 2) / 2 <= k) ++row;
                while (row * (row + 1) / 2 > k) --row;
                long long col = k - row * (row + 1) / 2;
                long long io = row / nvir;
                long long a = row - io * nvir;
                long long jo = col / nvir;
                long long b = col - jo * nvir;
                long long source = ((io * nocc + jo) * nvir + a) * nvir + b;
                vector = t2[source];
            }
            """,
            "gpu4pyscf_pack_rccsd_amplitudes",
        )
    return _PACK_RCCSD_KERNEL


def _cupy_pack_rccsd_delta_kernel() -> Any:
    global _PACK_RCCSD_DELTA_KERNEL
    if _PACK_RCCSD_DELTA_KERNEL is None:
        if _cp is None:  # pragma: no cover - guarded by callers
            raise RuntimeError("CuPy is unavailable")
        _PACK_RCCSD_DELTA_KERNEL = _cp.ElementwiseKernel(
            (
                "raw T t1new, raw T t2new, raw T t1old, raw T t2old, "
                "int64 nov, int64 nocc, int64 nvir"
            ),
            "T vector",
            r"""
            long long out_index = (long long)i;
            if (out_index < nov) {
                vector = t1new[out_index] - t1old[out_index];
            } else {
                long long k = out_index - nov;
                long long row = (long long)((sqrt(8.0 * (double)k + 1.0) - 1.0) * 0.5);
                while ((row + 1) * (row + 2) / 2 <= k) ++row;
                while (row * (row + 1) / 2 > k) --row;
                long long col = k - row * (row + 1) / 2;
                long long io = row / nvir;
                long long a = row - io * nvir;
                long long jo = col / nvir;
                long long b = col - jo * nvir;
                long long source = ((io * nocc + jo) * nvir + a) * nvir + b;
                vector = t2new[source] - t2old[source];
            }
            """,
            "gpu4pyscf_pack_rccsd_amplitude_delta",
        )
    return _PACK_RCCSD_DELTA_KERNEL


def _cupy_unpack_rccsd_kernel() -> Any:
    global _UNPACK_RCCSD_KERNEL
    if _UNPACK_RCCSD_KERNEL is None:
        if _cp is None:  # pragma: no cover - guarded by callers
            raise RuntimeError("CuPy is unavailable")
        _UNPACK_RCCSD_KERNEL = _cp.ElementwiseKernel(
            "raw T vector, int64 nov, int64 nocc, int64 nvir",
            "T t2",
            r"""
            long long source_index = (long long)i;
            long long b = source_index % nvir;
            source_index /= nvir;
            long long a = source_index % nvir;
            source_index /= nvir;
            long long jo = source_index % nocc;
            long long io = source_index / nocc;
            long long row = io * nvir + a;
            long long col = jo * nvir + b;
            long long hi = row >= col ? row : col;
            long long lo = row >= col ? col : row;
            long long packed = hi * (hi + 1) / 2 + lo;
            t2 = vector[nov + packed];
            """,
            "gpu4pyscf_unpack_rccsd_amplitudes",
        )
    return _UNPACK_RCCSD_KERNEL


def _validate_rccsd_amplitudes(t1: Any, t2: Any) -> tuple[int, int]:
    if getattr(t1, "ndim", None) != 2:
        raise ValueError("RCCSD singles must have shape (nocc, nvir)")
    nocc, nvir = (int(x) for x in t1.shape)
    if tuple(getattr(t2, "shape", ())) != (nocc, nocc, nvir, nvir):
        raise ValueError(
            "RCCSD doubles must have shape (nocc, nocc, nvir, nvir)"
        )
    if np.dtype(t1.dtype) != np.dtype(t2.dtype):
        raise ValueError("RCCSD singles and doubles must have the same dtype")
    return nocc, nvir


def pack_rccsd_amplitudes(
    t1: Any,
    t2: Any,
    *,
    array_module: Any = None,
) -> Any:
    """Pack RCCSD amplitudes in the exact ordering used by PySCF DIIS.

    The CuPy path uses a single elementwise kernel.  In particular it does not
    call ``cupy.tril_indices``: for WATER27/H2O8 that pair of index arrays alone
    would occupy more than two GiB of HBM.
    """

    nocc, nvir = _validate_rccsd_amplitudes(t1, t2)
    if array_module is None:
        array_module = _cp if _is_cupy_array(t1) else np
    if array_module is np:
        t1_array = np.asarray(t1)
        pair_matrix = np.asarray(t2).transpose(0, 2, 1, 3).reshape(
            nocc * nvir, nocc * nvir
        )
        lower = np.tril_indices(nocc * nvir)
        return np.concatenate((t1_array.ravel(), pair_matrix[lower]))
    if _cp is None or array_module is not _cp:
        raise TypeError("array_module must be NumPy or CuPy")
    if not (_is_cupy_array(t1) and _is_cupy_array(t2)):
        raise TypeError("the CuPy packer requires resident CuPy amplitudes")
    size = rccsd_amplitude_vector_size(nocc, nvir)
    return _cupy_pack_rccsd_kernel()(
        t1, t2, nocc * nvir, nocc, nvir, size=size
    )


def pack_rccsd_amplitude_difference(
    t1new: Any,
    t2new: Any,
    t1old: Any,
    t2old: Any,
    *,
    array_module: Any = None,
) -> Any:
    """Pack ``new - old`` without materializing a dense doubles difference."""

    nocc, nvir = _validate_rccsd_amplitudes(t1new, t2new)
    old_shape = _validate_rccsd_amplitudes(t1old, t2old)
    if old_shape != (nocc, nvir):
        raise ValueError("old and new RCCSD amplitude shapes do not match")
    if np.dtype(t1old.dtype) != np.dtype(t1new.dtype):
        raise ValueError("old and new RCCSD amplitude dtypes do not match")
    if array_module is None:
        array_module = _cp if _is_cupy_array(t1new) else np
    if array_module is np:
        return pack_rccsd_amplitudes(
            np.asarray(t1new) - np.asarray(t1old),
            np.asarray(t2new) - np.asarray(t2old),
            array_module=np,
        )
    if _cp is None or array_module is not _cp:
        raise TypeError("array_module must be NumPy or CuPy")
    if not all(
        _is_cupy_array(value) for value in (t1new, t2new, t1old, t2old)
    ):
        raise TypeError("the CuPy delta packer requires resident CuPy amplitudes")
    size = rccsd_amplitude_vector_size(nocc, nvir)
    return _cupy_pack_rccsd_delta_kernel()(
        t1new,
        t2new,
        t1old,
        t2old,
        nocc * nvir,
        nocc,
        nvir,
        size=size,
    )


def unpack_rccsd_amplitudes(
    vector: Any,
    nocc: int,
    nvir: int,
    *,
    array_module: Any = None,
) -> tuple[Any, Any]:
    """Inverse of :func:`pack_rccsd_amplitudes` in NumPy or CuPy."""

    nocc = int(nocc)
    nvir = int(nvir)
    expected = rccsd_amplitude_vector_size(nocc, nvir)
    if getattr(vector, "ndim", None) != 1 or int(vector.size) != expected:
        raise ValueError(
            f"packed RCCSD vector has size {getattr(vector, 'size', None)}; "
            f"expected {expected}"
        )
    nov = nocc * nvir
    if array_module is None:
        array_module = _cp if _is_cupy_array(vector) else np
    if array_module is np:
        source = np.asarray(vector)
        t1 = source[:nov].copy().reshape(nocc, nvir)
        matrix = np.empty((nov, nov), dtype=source.dtype)
        lower = np.tril_indices(nov)
        matrix[lower] = source[nov:]
        matrix[(lower[1], lower[0])] = source[nov:]
        t2 = np.ascontiguousarray(
            matrix.reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)
        )
        return t1, t2
    if _cp is None or array_module is not _cp:
        raise TypeError("array_module must be NumPy or CuPy")
    if not _is_cupy_array(vector):
        raise TypeError("the CuPy unpacker requires a resident CuPy vector")
    t1 = _cp.ascontiguousarray(vector[:nov].reshape(nocc, nvir))
    t2 = _cupy_unpack_rccsd_kernel()(
        vector,
        nov,
        nocc,
        nvir,
        size=nocc * nocc * nvir * nvir,
    ).reshape(nocc, nocc, nvir, nvir)
    return t1, t2


@dataclasses.dataclass(frozen=True)
class WorkspaceSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    order: str = "C"
    phase: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class PhaseArena:
    """Reusable allocations belonging to one named planner phase."""

    def __init__(self, planner: "WorkspacePlanner", phase: str) -> None:
        self.planner = planner
        self.phase = str(phase)

    def get(
        self,
        name: str,
        shape: Sequence[int] | int | WorkspaceSpec,
        dtype: Any = np.float64,
        order: str = "C",
    ) -> Any:
        if isinstance(shape, WorkspaceSpec):
            spec = shape
            shape, dtype, order = spec.shape, np.dtype(spec.dtype), spec.order
        elif isinstance(shape, int):
            shape = (shape,)
        return self.planner._get(self.phase, str(name), tuple(int(x) for x in shape), dtype, order)

    empty = get

    def allocate(self, name: str, shape_or_nbytes: Sequence[int] | int, dtype: Any = np.float64, order: str = "C") -> Any:
        if isinstance(shape_or_nbytes, int):
            return self.get(name, (shape_or_nbytes,), np.uint8, order)
        return self.get(name, shape_or_nbytes, dtype, order)

    def spec(self, name: str, shape: Sequence[int] | int, dtype: Any = np.float64, order: str = "C") -> WorkspaceSpec:
        return self.planner.reserve(name, shape, dtype=dtype, order=order, phase=self.phase)

    def clear(self) -> None:
        self.planner.release(self.phase)


class WorkspacePlanner:
    """Plan and reuse temporary arrays, preferring a configured array module."""

    def __init__(
        self,
        device: str = "auto",
        array_module: Any = None,
        metrics: Optional[RunMetrics] = None,
        allow_host_fallback: bool = False,
    ) -> None:
        self.device = str(device).lower()
        self.metrics = metrics
        self._requested_module = array_module
        self.allow_host_fallback = bool(allow_host_fallback)
        self._arrays: dict[tuple[str, str, tuple[int, ...], str, str], Any] = {}
        self._specs: dict[tuple[str, str, tuple[int, ...], str, str], WorkspaceSpec] = {}
        self._allocations = 0
        self._reuses = 0
        self._module_fallbacks = 0

    def _module(self) -> Any:
        if self._requested_module is not None:
            return self._requested_module
        if self.device in {"cpu", "numpy", "host"}:
            return np
        if _cp is not None:
            return _cp
        if self.device in {"gpu", "cuda", "cupy"}:
            raise RuntimeError(
                "a GPU workspace was requested but CuPy is unavailable"
            )
        if self.device == "auto":
            return np
        raise ValueError(f"unknown workspace device {self.device!r}")

    @property
    def array_module(self) -> Any:
        return self._module()

    def reserve(
        self,
        name: str,
        shape: Sequence[int] | int,
        dtype: Any = np.float64,
        order: str = "C",
        phase: str = "default",
    ) -> WorkspaceSpec:
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(int(x) for x in shape)
        if any(x < 0 for x in shape):
            raise ValueError("workspace shape dimensions must be non-negative")
        dtype_obj = np.dtype(dtype)
        return WorkspaceSpec(str(name), shape, dtype_obj.str, int(np.prod(shape, dtype=np.int64)) * dtype_obj.itemsize, order, str(phase))

    def _get(self, phase: str, name: str, shape: tuple[int, ...], dtype: Any, order: str) -> Any:
        spec = self.reserve(name, shape, dtype=dtype, order=order, phase=phase)
        key = (str(phase), str(name), spec.shape, spec.dtype, spec.order)
        if key in self._arrays:
            self._reuses += 1
            return self._arrays[key]
        module = self._module()
        try:
            array = module.empty(spec.shape, dtype=np.dtype(spec.dtype), order=spec.order)
        except Exception:
            if module is np or not self.allow_host_fallback:
                raise
            self._module_fallbacks += 1
            array = np.empty(spec.shape, dtype=np.dtype(spec.dtype), order=spec.order)
        self._arrays[key] = array
        self._specs[key] = spec
        self._allocations += 1
        if self.metrics is not None:
            self.metrics.increment("workspace_allocations")
        return array

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[PhaseArena]:
        yield PhaseArena(self, str(name))

    arena = phase

    def get(self, phase: str, name: str, shape: Sequence[int] | int, dtype: Any = np.float64, order: str = "C") -> Any:
        return self._get(str(phase), str(name), (shape,) if isinstance(shape, int) else tuple(shape), dtype, order)

    empty = get

    def release(self, phase: Optional[str] = None) -> None:
        if phase is None:
            self._arrays.clear()
            self._specs.clear()
            return
        phase = str(phase)
        for key in list(self._arrays):
            if key[0] == phase:
                del self._arrays[key]
                self._specs.pop(key, None)

    def specs(self) -> list[WorkspaceSpec]:
        return list(self._specs.values())

    def stats(self) -> dict[str, Any]:
        return {
            "allocations": int(self._allocations),
            "reuses": int(self._reuses),
            "active_arrays": len(self._arrays),
            "active_bytes": int(sum(spec.nbytes for spec in self._specs.values())),
            "module": "cupy" if self._module() is _cp and _cp is not None else "numpy",
            "module_fallbacks": int(self._module_fallbacks),
            "allow_host_fallback": self.allow_host_fallback,
        }


def _is_cupy_array(value: Any) -> bool:
    return _cp is not None and isinstance(value, _cp.ndarray)


class ContractionBackend:
    """Cached einsum/matmul dispatch with optional cuTensor acceleration."""

    def __init__(
        self,
        backend: str = "auto",
        use_cutensor: bool = True,
        metrics: Optional[RunMetrics] = None,
        transfers: Optional[TransferCounter] = None,
    ) -> None:
        self.requested_backend = str(backend).lower()
        valid_backends = {"auto", "numpy", "cpu", "cupy", "cuda", "gpu", "cutensor"}
        if self.requested_backend not in valid_backends:
            raise ValueError(f"unknown contraction backend {backend!r}")
        if self.requested_backend in {"cupy", "cuda", "gpu", "cutensor"} and _cp is None:
            raise RuntimeError("a GPU contraction backend was requested but CuPy is unavailable")
        self.use_cutensor = bool(use_cutensor)
        self.metrics = metrics
        self.transfers = transfers or (metrics.transfers if metrics is not None else TransferCounter())
        self._cache: dict[tuple[Any, ...], str] = {}
        self._cutensor_contract: Any = None
        self._cutensor_checked = False
        self._last_engine: Optional[str] = None
        self._tuning: dict[tuple[Any, ...], dict[str, Any]] = {}

    @property
    def backend(self) -> str:
        return self._last_engine or self.requested_backend

    @property
    def cache(self) -> Mapping[tuple[Any, ...], str]:
        return self._cache

    def _want_cupy(self, operands: Sequence[Any]) -> bool:
        if self.requested_backend in {"numpy", "cpu"}:
            return False
        if self.requested_backend in {"cupy", "cuda", "gpu", "cutensor"}:
            return _cp is not None
        return any(_is_cupy_array(x) for x in operands) and _cp is not None

    def _prepare(self, operands: Sequence[Any]) -> tuple[list[Any], bool]:
        want_cupy = self._want_cupy(operands)
        if want_cupy:
            converted = []
            for operand in operands:
                if _is_cupy_array(operand):
                    converted.append(operand)
                else:
                    arr = np.asarray(operand)
                    converted.append(_cp.asarray(arr))
                    self.transfers.record_h2d(arr.nbytes)
            return converted, True
        converted = []
        for operand in operands:
            if _is_cupy_array(operand):
                arr = _cp.asnumpy(operand)
                converted.append(arr)
                self.transfers.record_d2h(arr.nbytes)
            else:
                converted.append(np.asarray(operand))
        return converted, False

    def _cutensor(self) -> Any:
        if not self._cutensor_checked:
            self._cutensor_checked = True
            if _cp is not None and self.use_cutensor:
                try:
                    from gpu4pyscf.lib.cutensor import contract
                    self._cutensor_contract = contract
                except Exception:
                    self._cutensor_contract = None
        return self._cutensor_contract

    def _route(self, op: str, expression: Any, operands: Sequence[Any], is_cupy: bool) -> str:
        key = (op, str(expression), tuple((tuple(x.shape), str(x.dtype)) for x in operands), is_cupy)
        if key in self._cache:
            return self._cache[key]
        if not is_cupy:
            route = "numpy-einsum" if op == "einsum" else "numpy-matmul"
        elif op == "einsum" and len(operands) == 2 and self._cutensor() is not None:
            route = "cutensor"
        else:
            route = "cupy-einsum" if op == "einsum" else "cupy-matmul"
        self._cache[key] = route
        return route

    @staticmethod
    def _time_cuda(operation: Any, warmup: int, repeats: int) -> float:
        for _ in range(warmup):
            operation()
        start = _cp.cuda.Event()
        stop = _cp.cuda.Event()
        start.record()
        for _ in range(repeats):
            result = operation()
        stop.record()
        stop.synchronize()
        # Keep the last result alive until the timing event completes.
        _ = result
        return float(_cp.cuda.get_elapsed_time(start, stop)) / repeats

    def autotune_binary_gemm(
        self,
        expression: str,
        left: Any,
        right: Any,
        *,
        warmup: int = 2,
        repeats: int = 5,
        cutensor_advantage: float = 0.05,
        validation_rtol: float = 1e-10,
        validation_atol: float = 1e-12,
    ) -> str:
        """Choose cuBLAS GEMM or cuTENSOR for one fixed binary shape.

        The caller guarantees that ``expression`` is mathematically identical
        to ``left @ right``. cuTENSOR is selected only when its mean repeated
        CUDA-event time is at least ``cutensor_advantage`` faster. A tie
        therefore deterministically selects cuBLAS, as required by the water4
        tuning contract.
        """

        if int(warmup) != warmup or warmup < 0:
            raise ValueError("warmup must be a non-negative integer")
        if int(repeats) != repeats or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if not 0 <= cutensor_advantage < 1:
            raise ValueError("cutensor_advantage must lie in [0, 1)")
        arrays, is_cupy = self._prepare((left, right))
        if not is_cupy:
            raise RuntimeError("CUDA autotuning requires resident CuPy operands")
        key = (
            "einsum",
            str(expression),
            tuple((tuple(x.shape), str(x.dtype)) for x in arrays),
            True,
        )
        reference = _cp.matmul(arrays[0], arrays[1])
        cutensor = self._cutensor()
        if cutensor is None:
            self._cache[key] = "cupy-matmul"
            self._tuning[key] = {
                "selected": "cupy-matmul",
                "reason": "cutensor-unavailable",
                "tie_policy": "prefer-cublas-within-5-percent",
            }
            return "cupy-matmul"
        try:
            candidate = cutensor(expression, arrays[0], arrays[1])
            difference = _cp.max(_cp.abs(candidate - reference))
            scale = _cp.max(_cp.abs(reference))
            self.transfers.record_d2h(
                int(difference.nbytes) + int(scale.nbytes), count=2
            )
            error = float(difference.item())
            reference_scale = float(scale.item())
            if error > validation_atol + validation_rtol * reference_scale:
                raise RuntimeError(
                    "cuTENSOR and cuBLAS disagree during contraction tuning"
                )
            cublas_ms = self._time_cuda(
                lambda: _cp.matmul(arrays[0], arrays[1]),
                int(warmup),
                int(repeats),
            )
            cutensor_ms = self._time_cuda(
                lambda: cutensor(expression, arrays[0], arrays[1]),
                int(warmup),
                int(repeats),
            )
            selected = (
                "cutensor"
                if cutensor_ms < cublas_ms * (1.0 - cutensor_advantage)
                else "cupy-matmul"
            )
            reason = "measured"
        except Exception as exc:
            selected = "cupy-matmul"
            cublas_ms = self._time_cuda(
                lambda: _cp.matmul(arrays[0], arrays[1]),
                int(warmup),
                int(repeats),
            )
            cutensor_ms = None
            reason = f"cutensor-rejected:{type(exc).__name__}"
        self._cache[key] = selected
        self._tuning[key] = {
            "selected": selected,
            "reason": reason,
            "cublas_ms": cublas_ms,
            "cutensor_ms": cutensor_ms,
            "warmup": int(warmup),
            "repeats": int(repeats),
            "cutensor_advantage_required": float(cutensor_advantage),
            "tie_policy": "prefer-cublas-within-5-percent",
        }
        if self.metrics is not None:
            self.metrics.increment("contraction_autotunes")
            self.metrics.increment("autotune_cuda_synchronizations", 2)
        return selected

    @staticmethod
    def _write_result(result: Any, out: Any, alpha: float, beta: float) -> Any:
        if out is None:
            return result * alpha
        if beta == 0:
            out[...] = result * alpha
        else:
            out[...] = out * beta + result * alpha
        return out

    def einsum(self, expression: str, *operands: Any, out: Any = None, alpha: float = 1.0, beta: float = 0.0) -> Any:
        arrays, is_cupy = self._prepare(operands)
        route = self._route("einsum", expression, arrays, is_cupy)
        self._last_engine = route
        if route == "cutensor":
            try:
                return self._cutensor_contract(
                    expression,
                    arrays[0],
                    arrays[1],
                    alpha=alpha,
                    beta=beta,
                    out=out,
                )
            except Exception:
                # cuTensor support differs slightly between CuPy releases;
                # retain a reliable CuPy einsum fallback and cache it.
                key = ("einsum", str(expression), tuple((tuple(x.shape), str(x.dtype)) for x in arrays), is_cupy)
                self._cache[key] = "cupy-einsum"
                route = "cupy-einsum"
                self._last_engine = route
        if route == "cupy-matmul":
            result = _cp.matmul(arrays[0], arrays[1])
            return self._write_result(result, out, alpha, beta)
        engine = _cp if is_cupy else np
        result = engine.einsum(expression, *arrays)
        return self._write_result(result, out, alpha, beta)

    contract = einsum

    def matmul(self, left: Any, right: Any, out: Any = None, alpha: float = 1.0, beta: float = 0.0) -> Any:
        arrays, is_cupy = self._prepare((left, right))
        route = self._route("matmul", "matmul", arrays, is_cupy)
        self._last_engine = route
        engine = _cp if is_cupy else np
        result = engine.matmul(arrays[0], arrays[1])
        return self._write_result(result, out, alpha, beta)

    dot = matmul

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "use_cutensor": self.use_cutensor,
            "last_engine": self._last_engine,
            "cached_routes": len(self._cache),
            "tuned_routes": [
                {
                    "signature": str(key),
                    **_jsonable(value),
                }
                for key, value in self._tuning.items()
            ],
            "transfers": self.transfers.to_dict(),
        }


__all__ = [
    "ContractionBackend",
    "NVTXRange",
    "PhaseArena",
    "RunMetrics",
    "TransferCounter",
    "WorkspacePlanner",
    "WorkspaceSpec",
    "canonical_ccsd_residual_update",
    "contract_traced_ao_metric",
    "nvtx_range",
]
