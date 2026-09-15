"""Audit complete THC Algorithms 1--10 on one converged WATER2 RR state.

This driver is a development seam, not an iterative THC-CCSD implementation.
It runs one RR/CD lifecycle, captures the resulting runtime object and array
pointer identities, and then uses those same objects for an analytic full-pair
identity anchor followed by a staged inexact amplitude/ERI-THC search.  It does
not hash tensor contents or attest that in-place values are immutable.  All
acceptance, production, formal, and performance flags remain false regardless
of numerical results.

The two one-body conventions are kept distinct:

* ``physical-cd-hamiltonian`` uses the physical MO-basis hcore together with
  the direct-CD factors.  Its zero-T1 F-hat defines the internally consistent
  CD Fock oracle.
* ``direct-cd-effective-one-body`` adds ``F_RR - F_CD`` to the physical hcore.
  This makes the zero-T1 F-hat equal the exact Fock object used by the already
  converged RR lifecycle and permits an identity comparison at that state.

The complete THC audit already contains the full diagonal F-hat action.  The
ordinary RR builders expose numerators, so the orbital-energy denominator is
subtracted only on the RR-oracle side.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

# Keep the development driver directly executable from its benchmark path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gpu4pyscf.cc.device_runtime import TransferCounter
from gpu4pyscf.cc.lowrank import RRDoubles
from gpu4pyscf.cc.rr_residual import (
    ProjectedPairDenominator,
    build_ccsd_singles_numerator,
    build_projected_ccsd_doubles_numerator,
)
from gpu4pyscf.cc.thc_complete_audit import (
    IdentityAttestedERITHCFactors,
    assemble_complete_thc_ccsd_audit,
)
from gpu4pyscf.cc.thc_eri_preprocess import (
    build_weighted_eri_thc_preprocess,
)
from gpu4pyscf.cc.thc_factorization import (
    THCProjectorFactors,
    fit_weighted_thc_projector_adaptive,
)
from gpu4pyscf.cc.thc_fhat import build_t1_transformed_fhat

ROOT = Path(__file__).resolve().parent
CASE_FILE = ROOT / 'cases.json'
CASE_ID = 'water2-tz'
SCHEMA = 'gpu4pyscf.water2-thc-complete-audit.v2'
PAIR_DIMENSION = 10 * 106
GIB = 1024**3
HARD_HBM_LIMIT_BYTES = 72 * GIB
HARD_HOST_RSS_LIMIT_BYTES = 110 * GIB
FAIL_CLOSED_FLAGS = {
    'accepted': False,
    'production_enabled': False,
    'performance_eligible': False,
    'formal_validation_eligible': False,
    'complete_validated': False,
}


def _load_benchmark_module():
    name = 'water2_thc_complete_shared_benchmark'
    specification = importlib.util.spec_from_file_location(
        name, ROOT / 'benchmark.py'
    )
    if specification is None or specification.loader is None:
        raise RuntimeError('cannot load the shared WATER27 benchmark module')
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _array_module(value: Any):
    module = type(value).__module__.split('.', 1)[0]
    if module == 'numpy' and isinstance(value, np.ndarray):
        return np
    if module == 'cupy':
        import cupy

        if isinstance(value, cupy.ndarray):
            return cupy
    raise TypeError('audit tensors must be NumPy or CuPy arrays')


def _sync(value: Any) -> None:
    xp = _array_module(value)
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer')
    result = int(value)
    if result < 1:
        raise ValueError(f'{name} must be positive')
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer')
    result = int(value)
    if result < 0:
        raise ValueError(f'{name} must be non-negative')
    return result


def _nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be a real scalar')
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = _nonnegative_float(value, name=name)
    if result == 0.0:
        raise ValueError(f'{name} must be positive')
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(',', ':'),
        sort_keys=True,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode('ascii')).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f'refusing to overwrite audit result: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def _copy_flags() -> dict[str, bool]:
    return dict(FAIL_CLOSED_FLAGS)


@dataclass(frozen=True)
class ERIFitSpec:
    """One ERI-THC rank/tolerance pair, including the zero-tolerance anchor."""

    rank: int
    fit_tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, 'rank', _positive_int(self.rank, name='ERI rank'))
        object.__setattr__(
            self,
            'fit_tolerance',
            _nonnegative_float(self.fit_tolerance, name='ERI fit tolerance'),
        )

    @classmethod
    def parse(cls, text: str) -> ERIFitSpec:
        fields = str(text).split(':')
        if len(fields) != 2:
            raise argparse.ArgumentTypeError('ERI candidate must have the form RANK:TOLERANCE')
        try:
            return cls(rank=int(fields[0]), fit_tolerance=float(fields[1]))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    def metadata(self) -> dict[str, Any]:
        return {'rank': self.rank, 'fit_tolerance': self.fit_tolerance}


@dataclass(frozen=True)
class AuditTolerances:
    """Explicit equation gates for the audit seam."""

    exact_identity_atol: float
    candidate_delta_norm: float
    candidate_projected_residual_norm: float
    amplitude_core_symmetry_atol: float

    def __post_init__(self) -> None:
        for name in (
            'exact_identity_atol',
            'candidate_delta_norm',
            'candidate_projected_residual_norm',
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name=name))
        object.__setattr__(
            self,
            'amplitude_core_symmetry_atol',
            _nonnegative_float(
                self.amplitude_core_symmetry_atol,
                name='amplitude_core_symmetry_atol',
            ),
        )

    def metadata(self) -> dict[str, float]:
        return {
            'exact_identity_atol': self.exact_identity_atol,
            'candidate_delta_norm': self.candidate_delta_norm,
            'candidate_projected_residual_norm': (self.candidate_projected_residual_norm),
            'amplitude_core_symmetry_atol': (self.amplitude_core_symmetry_atol),
        }


@dataclass(frozen=True)
class ResourceBudget:
    """Hard process budgets from the WATER8 performance contract."""

    hbm_limit_bytes: int = HARD_HBM_LIMIT_BYTES
    host_rss_limit_bytes: int = HARD_HOST_RSS_LIMIT_BYTES
    reserve_bytes: int = 512 * 1024**2

    def __post_init__(self) -> None:
        for name in ('hbm_limit_bytes', 'host_rss_limit_bytes'):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        if self.hbm_limit_bytes > HARD_HBM_LIMIT_BYTES:
            raise ValueError('hbm_limit_bytes cannot exceed the fixed 72-GiB contract')
        if self.host_rss_limit_bytes > HARD_HOST_RSS_LIMIT_BYTES:
            raise ValueError('host_rss_limit_bytes cannot exceed the fixed 110-GiB contract')
        if isinstance(self.reserve_bytes, bool) or not isinstance(self.reserve_bytes, int):
            raise TypeError('reserve_bytes must be an integer')
        if self.reserve_bytes < 0:
            raise ValueError('reserve_bytes must be non-negative')

    def metadata(self) -> dict[str, Any]:
        return {
            'hbm_limit_bytes': self.hbm_limit_bytes,
            'hbm_limit_gib': self.hbm_limit_bytes / GIB,
            'host_rss_limit_bytes': self.host_rss_limit_bytes,
            'host_rss_limit_gib': self.host_rss_limit_bytes / GIB,
            'reserve_bytes': self.reserve_bytes,
        }


@dataclass(frozen=True)
class CandidateMemoryEstimate:
    """Conservative logical allocation estimate before a candidate runs."""

    components: dict[str, int]
    estimated_device_additional_bytes: int
    estimated_host_additional_bytes: int

    def metadata(self) -> dict[str, Any]:
        return {
            'components': dict(self.components),
            'estimated_device_additional_bytes': (self.estimated_device_additional_bytes),
            'estimated_device_additional_gib': (self.estimated_device_additional_bytes / GIB),
            'estimated_host_additional_bytes': self.estimated_host_additional_bytes,
            'estimated_host_additional_gib': (self.estimated_host_additional_bytes / GIB),
            'scope': 'conservative-logical-arrays-plus-reserve',
            'actual_peak_requires_measurement': True,
        }


@dataclass(frozen=True)
class MemoryPreflight:
    passed: bool
    estimate: CandidateMemoryEstimate
    hbm_current_used_bytes: int | None
    hbm_free_bytes: int | None
    hbm_total_bytes: int | None
    host_rss_current_bytes: int
    projected_hbm_bytes: int | None
    projected_host_rss_bytes: int
    reasons: tuple[str, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            'passed': self.passed,
            'estimate': self.estimate.metadata(),
            'hbm_current_used_bytes': self.hbm_current_used_bytes,
            'hbm_free_bytes': self.hbm_free_bytes,
            'hbm_total_bytes': self.hbm_total_bytes,
            'host_rss_current_bytes': self.host_rss_current_bytes,
            'projected_hbm_bytes': self.projected_hbm_bytes,
            'projected_host_rss_bytes': self.projected_host_rss_bytes,
            'reasons': list(self.reasons),
            'gate_kind': 'pre-allocation-fail-closed',
        }


def estimate_candidate_memory(
    *,
    naux: int,
    nocc: int,
    nvir: int,
    rr_rank: int,
    amplitude_rank: int,
    eri_rank: int,
    auxiliary_block_size: int,
    x_block_size: int,
    itemsize: int = 8,
) -> CandidateMemoryEstimate:
    """Estimate live logical arrays without claiming a measured CUDA peak.

    Multipliers intentionally cover simultaneous old/new ALS factors, solver
    workspaces, endpoint outputs, and the full transformed-Cholesky result.
    The estimate is conservative for the current Python/CuPy audit endpoints.
    """

    values = {
        'naux': naux,
        'nocc': nocc,
        'nvir': nvir,
        'rr_rank': rr_rank,
        'amplitude_rank': amplitude_rank,
        'eri_rank': eri_rank,
        'auxiliary_block_size': auxiliary_block_size,
        'x_block_size': x_block_size,
        'itemsize': itemsize,
    }
    normalized = {name: _positive_int(value, name=name) for name, value in values.items()}
    naux = normalized['naux']
    nocc = normalized['nocc']
    nvir = normalized['nvir']
    rr_rank = normalized['rr_rank']
    amplitude_rank = normalized['amplitude_rank']
    eri_rank = normalized['eri_rank']
    ablock = min(normalized['auxiliary_block_size'], naux)
    xblock = min(normalized['x_block_size'], amplitude_rank)
    itemsize = normalized['itemsize']
    pair = nocc * nvir
    nmo = nocc + nvir

    elements = {
        # Weighted projector CP/ALS: target, weighted target, fitted target,
        # factors, tau variants, and square normal equations.
        'amplitude_fit': (
            4 * pair * rr_rank + 4 * (nocc + nvir + 2 * rr_rank) * amplitude_rank + 8 * amplitude_rank**2
        ),
        # MP2-NO weights, transformed Lov, CP factors/xi/core, and blocked
        # residual evaluation.  Keep two xi/core copies for fit plus result.
        'eri_fit': (
            2 * naux * pair
            + 3 * naux * eri_rank
            + 3 * eri_rank**2
            + 4 * (nocc + nvir) * eri_rank
            + ablock * pair
            + 4 * (nocc**2 + nvir**2)
        ),
        # Two semantic results remain live.  While the second is built, its
        # symmetrized raw copy and transformed factors coexist with the first
        # result.  Five factor-sized arrays leave room for those objects and
        # the zero-T1 construction used to derive the CD Fock.
        'fhat': 5 * naux * nmo**2 + 8 * nmo**2,
        # Algorithms 1--3 and 8--10 retain several amplitude-rank cores.
        'algorithms_1_3_8_9_10': (
            22 * amplitude_rank**2 + 5 * ablock * amplitude_rank**2 + 5 * ablock * pair + 4 * (nocc**2 + nvir**2)
        ),
        # Algorithm 6 and 7 workspaces with their first THC coordinate blocked.
        'algorithms_6_7': (
            10 * amplitude_rank * pair
            + 8 * amplitude_rank**2
            + 4 * xblock * amplitude_rank**2
            + 4 * amplitude_rank * eri_rank
            + 2 * eri_rank**2
            + 4 * xblock * pair
        ),
        'rr_backprojection_and_comparison': (8 * rr_rank**2 + 4 * pair + 4 * nmo**2),
    }
    components = {name: int(count * itemsize) for name, count in elements.items()}
    device = int(sum(components.values()))
    host = int(max(64 * 1024**2, 8 * nmo**2 + 8 * pair))
    return CandidateMemoryEstimate(
        components=components,
        estimated_device_additional_bytes=device,
        estimated_host_additional_bytes=host,
    )


def check_memory_preflight(
    estimate: CandidateMemoryEstimate,
    budget: ResourceBudget,
    *,
    hbm_current_used_bytes: int | None,
    hbm_free_bytes: int | None,
    hbm_total_bytes: int | None,
    host_rss_current_bytes: int,
) -> MemoryPreflight:
    """Apply the HBM/RSS budget and current-device-capacity gates."""

    if not isinstance(estimate, CandidateMemoryEstimate):
        raise TypeError('estimate must be a CandidateMemoryEstimate')
    if not isinstance(budget, ResourceBudget):
        raise TypeError('budget must be a ResourceBudget')
    host_rss = _positive_int(max(1, host_rss_current_bytes), name='host_rss_current_bytes')
    reasons: list[str] = []
    projected_host = host_rss + estimate.estimated_host_additional_bytes
    if projected_host > budget.host_rss_limit_bytes:
        reasons.append('projected host RSS exceeds the configured hard limit')

    hbm_values = (hbm_current_used_bytes, hbm_free_bytes, hbm_total_bytes)
    if any(value is None for value in hbm_values):
        projected_hbm = None
        if not all(value is None for value in hbm_values):
            raise ValueError('HBM snapshot fields must be supplied together')
    else:
        current = int(hbm_current_used_bytes)
        free = int(hbm_free_bytes)
        total = int(hbm_total_bytes)
        if min(current, free, total) < 0 or current + free > total:
            raise ValueError('HBM snapshot is inconsistent')
        required = estimate.estimated_device_additional_bytes + budget.reserve_bytes
        projected_hbm = current + required
        if projected_hbm > budget.hbm_limit_bytes:
            reasons.append('projected HBM exceeds the configured 72-GiB contract')
        if required > free:
            reasons.append('candidate plus reserve exceeds currently free HBM')
    return MemoryPreflight(
        passed=not reasons,
        estimate=estimate,
        hbm_current_used_bytes=hbm_current_used_bytes,
        hbm_free_bytes=hbm_free_bytes,
        hbm_total_bytes=hbm_total_bytes,
        host_rss_current_bytes=host_rss,
        projected_hbm_bytes=projected_hbm,
        projected_host_rss_bytes=projected_host,
        reasons=tuple(reasons),
    )


def _host_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == 'darwin' else rss * 1024


def _memory_snapshot(reference: Any) -> tuple[int | None, int | None, int | None]:
    xp = _array_module(reference)
    if xp is np:
        return None, None, None
    xp.cuda.get_current_stream().synchronize()
    free, total = xp.cuda.runtime.memGetInfo()
    return int(total - free), int(free), int(total)


def _normalize_pci_bus_id(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        value = value.decode('ascii', errors='replace')
    result = str(value).strip().lower()
    if result in {'', 'none', 'n/a', 'unknown'}:
        return ''
    if result.startswith('00000000:'):
        result = '0000:' + result.split(':', 1)[1]
    return result


def _nvidia_smi_uuid_for_pci(pci_bus_id: str) -> str | None:
    """Resolve a UUID through an exact PCI-bus match, or return ``None``."""

    try:
        output = subprocess.check_output(
            (
                'nvidia-smi',
                '--query-gpu=uuid,pci.bus_id',
                '--format=csv,noheader,nounits',
            ),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001 - PCI identity is the declared fallback
        return None
    expected = _normalize_pci_bus_id(pci_bus_id)
    matches = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 2 and _normalize_pci_bus_id(fields[1]) == expected:
            matches.append(fields[0])
    if len(matches) != 1 or not matches[0]:
        return None
    return matches[0]


def capture_a100_gpu_identity(
    cupy_module: Any,
    *,
    uuid_lookup: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Fail closed unless exactly one visible CUDA device is an 80-GB A100."""

    runtime = cupy_module.cuda.runtime
    visible_count = int(runtime.getDeviceCount())
    if visible_count != 1:
        raise RuntimeError(f'the WATER2 THC audit requires exactly one CUDA-visible GPU; observed {visible_count}')
    device = cupy_module.cuda.Device()
    device_id = int(device.id)
    properties = runtime.getDeviceProperties(device_id)
    raw_name = properties.get('name', '')
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode('utf-8', errors='replace')
    name = str(raw_name)
    major = int(properties.get('major', -1))
    minor = int(properties.get('minor', -1))
    total_memory = int(properties.get('totalGlobalMem', 0))
    pci_bus_id = _normalize_pci_bus_id(getattr(device, 'pci_bus_id', ''))
    if 'A100' not in name.upper():
        raise RuntimeError(f'the WATER2 THC audit requires an NVIDIA A100; observed {name!r}')
    if (major, minor) != (8, 0):
        raise RuntimeError(f'the WATER2 THC audit requires compute capability 8.0; observed {major}.{minor}')
    if total_memory < HARD_HBM_LIMIT_BYTES:
        raise RuntimeError('the visible A100 does not expose enough HBM for the fixed 72-GiB contract')
    if not pci_bus_id:
        raise RuntimeError('the CUDA device does not expose a PCI bus identity')

    lookup = _nvidia_smi_uuid_for_pci if uuid_lookup is None else uuid_lookup
    uuid_lookup_error = None
    try:
        gpu_uuid = lookup(pci_bus_id)
    except Exception as exc:  # noqa: BLE001 - explicit PCI identity remains valid
        gpu_uuid = None
        uuid_lookup_error = repr(exc)
    if gpu_uuid is not None:
        gpu_uuid = str(gpu_uuid).strip()
        if not gpu_uuid.upper().startswith('GPU-'):
            gpu_uuid = None
    primary_kind = 'uuid' if gpu_uuid is not None else 'pci_bus_id'
    primary_value = gpu_uuid if gpu_uuid is not None else pci_bus_id
    return {
        'validated': True,
        'hardware_contract': 'one-visible-A100-cc8.0-at-least-72-GiB',
        'visible_device_count': visible_count,
        'device_id': device_id,
        'name': name,
        'compute_capability': f'{major}.{minor}',
        'total_global_memory_bytes': total_memory,
        'total_global_memory_gib': total_memory / GIB,
        'uuid': gpu_uuid,
        'pci_bus_id': pci_bus_id,
        'primary_identity_kind': primary_kind,
        'primary_identity': primary_value,
        'uuid_source': ('nvidia-smi-exact-pci-match' if gpu_uuid is not None else None),
        'uuid_lookup_error': uuid_lookup_error,
        'pci_bus_fallback_used': gpu_uuid is None,
        'runtime_version': int(runtime.runtimeGetVersion()),
        'driver_version': int(runtime.driverGetVersion()),
        'formal_performance_identity': False,
        **_copy_flags(),
    }


class SynchronizedHBMObserver:
    """Sample CUDA and allocator memory at synchronized stage boundaries."""

    def __init__(
        self,
        cupy_module: Any,
        budget: ResourceBudget,
        gpu_identity: dict[str, Any],
        *,
        host_rss_reader: Callable[[], int] = _host_rss_bytes,
    ) -> None:
        if gpu_identity.get('validated') is not True:
            raise ValueError('gpu_identity must be validated before HBM observation')
        self._cp = cupy_module
        self._budget = budget
        self._gpu_identity = dict(gpu_identity)
        self._host_rss_reader = host_rss_reader
        self._samples: list[dict[str, Any]] = []

    def sample(self, label: str) -> dict[str, Any]:
        if not isinstance(label, str) or not label.strip():
            raise ValueError('HBM stage label must be non-empty')
        device = self._cp.cuda.Device()
        device_id = int(device.id)
        if device_id != int(self._gpu_identity['device_id']):
            raise RuntimeError('the active CUDA device changed during the WATER2 audit')
        self._cp.cuda.get_current_stream().synchronize()
        free_bytes, total_bytes = self._cp.cuda.runtime.memGetInfo()
        free_bytes = int(free_bytes)
        total_bytes = int(total_bytes)
        if min(free_bytes, total_bytes) < 0 or free_bytes > total_bytes:
            raise RuntimeError('CUDA memGetInfo returned an inconsistent snapshot')
        identity_total_bytes = int(self._gpu_identity['total_global_memory_bytes'])
        if total_bytes > identity_total_bytes or total_bytes < HARD_HBM_LIMIT_BYTES:
            raise RuntimeError('CUDA context-visible total memory is inconsistent with the GPU contract')
        pool = self._cp.get_default_memory_pool()
        allocator_used = int(pool.used_bytes())
        allocator_reserved = int(pool.total_bytes())
        if min(allocator_used, allocator_reserved) < 0 or allocator_used > allocator_reserved:
            raise RuntimeError('the CuPy allocator returned an inconsistent snapshot')
        device_used = total_bytes - free_bytes
        host_rss = int(self._host_rss_reader())
        violations = []
        if device_used > self._budget.hbm_limit_bytes:
            violations.append('synchronized device-wide HBM observation exceeds its limit')
        if allocator_reserved > self._budget.hbm_limit_bytes:
            violations.append('CuPy allocator reserved-byte observation exceeds its HBM limit')
        if host_rss > self._budget.host_rss_limit_bytes:
            violations.append('host RSS observation exceeds its limit')
        record = {
            'label': label.strip(),
            'synchronized_before_observation': True,
            'device_id': device_id,
            'device_free_bytes': free_bytes,
            'device_total_bytes': total_bytes,
            'gpu_identity_total_global_memory_bytes': identity_total_bytes,
            'device_total_matches_gpu_identity': total_bytes == identity_total_bytes,
            'device_wide_used_bytes': device_used,
            'cupy_allocator_used_bytes': allocator_used,
            'cupy_allocator_reserved_bytes': allocator_reserved,
            'host_rss_high_water_bytes': host_rss,
            'hbm_limit_bytes': self._budget.hbm_limit_bytes,
            'host_rss_limit_bytes': self._budget.host_rss_limit_bytes,
            'violations': violations,
            'gate_passed': not violations,
        }
        self._samples.append(record)
        if violations:
            raise MemoryError('; '.join(violations))
        return dict(record)

    def metadata(self) -> dict[str, Any]:
        device_high_water = max(
            (sample['device_wide_used_bytes'] for sample in self._samples),
            default=None,
        )
        allocator_used_high_water = max(
            (sample['cupy_allocator_used_bytes'] for sample in self._samples),
            default=None,
        )
        allocator_reserved_high_water = max(
            (sample['cupy_allocator_reserved_bytes'] for sample in self._samples),
            default=None,
        )
        violations = [reason for sample in self._samples for reason in sample['violations']]
        return {
            'available': bool(self._samples),
            'measurement_scope': 'synchronized-stage-boundary-observations',
            'device_memory_scope': 'device-wide-CUDA-memGetInfo',
            'allocator_memory_scope': 'current-process-default-CuPy-pool',
            'sample_count': len(self._samples),
            'samples': list(self._samples),
            'sampled_device_used_high_water_bytes': device_high_water,
            'sampled_allocator_used_high_water_bytes': allocator_used_high_water,
            'sampled_allocator_reserved_high_water_bytes': allocator_reserved_high_water,
            'observed_peak_is_true_peak': False,
            'true_peak_limitation': (
                'stage-boundary samples can miss transient non-pool allocations; '
                'allocator reserved bytes are a retained sampled high-water proxy'
            ),
            'allocator_reports_native_high_water': False,
            'hbm_limit_bytes': self._budget.hbm_limit_bytes,
            'host_rss_limit_bytes': self._budget.host_rss_limit_bytes,
            'violations': violations,
            'gate_passed': bool(self._samples) and not violations,
            'formal_performance_measurement': False,
            **_copy_flags(),
        }


def _pending_gpu_identity() -> dict[str, Any]:
    return {'validated': False, 'status': 'pending', **_copy_flags()}


def _pending_hbm_observations() -> dict[str, Any]:
    return {
        'available': False,
        'status': 'pending',
        'gate_passed': False,
        'observed_peak_is_true_peak': False,
        'formal_performance_measurement': False,
        **_copy_flags(),
    }


def _array_pointer(value: Any) -> int:
    xp = _array_module(value)
    if xp is np:
        return int(value.__array_interface__['data'][0])
    pointer = getattr(getattr(value, 'data', None), 'ptr', None)
    if pointer is None:
        raise TypeError('CuPy audit array does not expose a device pointer')
    return int(pointer)


@dataclass(frozen=True)
class RRLifecycleBinding:
    """Runtime object/pointer identity snapshot of one RR/CD lifecycle."""

    solver: Any
    engine: Any
    integrals: Any
    projector: Any
    t1: Any
    doubles: RRDoubles
    mp2_doubles: RRDoubles
    fock: Any
    orbital_energies: Any
    denominator: ProjectedPairDenominator
    loo: Any
    lov: Any
    lvv: Any
    mo_coeff: Any
    transfer_counter: TransferCounter
    case_id: str
    array_pointer_snapshot: tuple[tuple[str, int], ...]

    @classmethod
    def capture(cls, solver: Any, *, case_id: str = CASE_ID) -> RRLifecycleBinding:
        if getattr(solver, 'converged', None) is not True:
            raise RuntimeError('a converged RRCCSD.kernel() lifecycle is required')
        engine = getattr(solver, '_rr_engine', None)
        integrals = getattr(solver, '_cd_integrals', None)
        projector = getattr(solver, 'rr_projector', None)
        doubles = getattr(solver, 'doubles', None)
        mp2_doubles = getattr(solver, '_cd_initial_doubles', None)
        denominator = getattr(engine, 'denominator', None)
        metrics = getattr(solver, 'run_metrics', None)
        counter = getattr(metrics, 'transfers', None)
        if engine is None or integrals is None or projector is None:
            raise RuntimeError('RR direct-CD runtime objects are incomplete')
        if not isinstance(doubles, RRDoubles) or not isinstance(mp2_doubles, RRDoubles):
            raise TypeError('RR lifecycle must retain converged and MP2 RRDoubles')
        if not isinstance(denominator, ProjectedPairDenominator):
            raise TypeError('RR lifecycle must retain its ProjectedPairDenominator')
        if not isinstance(counter, TransferCounter):
            raise TypeError('RR lifecycle must expose its TransferCounter')
        t1 = solver.t1
        fock = solver._rr_fock
        orbital_energies = solver._rr_orbital_energies
        loo = integrals.L_oo
        lov = integrals.L_ov
        lvv = integrals.L_vv
        mo_coeff = solver.mo_coeff
        pointer_arrays = {
            'integral_factors': integrals.factors,
            'projector_vectors': projector.vectors,
            'projector_eigenvalues': projector.eigenvalues,
            't1': t1,
            'doubles_core': doubles.core,
            'mp2_doubles_core': mp2_doubles.core,
            'fock': fock,
            'orbital_energies': orbital_energies,
            'denominator_matrix': denominator.matrix,
            'denominator_eigenvalues': denominator.eigenvalues,
            'denominator_eigenvectors': denominator.eigenvectors,
            'loo': loo,
            'lov': lov,
            'lvv': lvv,
            'mo_coeff': mo_coeff,
        }
        binding = cls(
            solver=solver,
            engine=engine,
            integrals=integrals,
            projector=projector,
            t1=t1,
            doubles=doubles,
            mp2_doubles=mp2_doubles,
            fock=fock,
            orbital_energies=orbital_energies,
            denominator=denominator,
            loo=loo,
            lov=lov,
            lvv=lvv,
            mo_coeff=mo_coeff,
            transfer_counter=counter,
            case_id=str(case_id),
            array_pointer_snapshot=tuple((name, _array_pointer(value)) for name, value in pointer_arrays.items()),
        )
        binding.validate()
        return binding

    @property
    def nocc(self) -> int:
        return int(self.integrals.nocc)

    @property
    def nvir(self) -> int:
        return int(self.integrals.nvir)

    @property
    def pair_dimension(self) -> int:
        return self.nocc * self.nvir

    def validate(self) -> None:
        checks = {
            'solver._rr_engine': getattr(self.solver, '_rr_engine', None) is self.engine,
            'solver._cd_integrals': (getattr(self.solver, '_cd_integrals', None) is self.integrals),
            'solver.rr_projector': (getattr(self.solver, 'rr_projector', None) is self.projector),
            'solver.t1': getattr(self.solver, 't1', None) is self.t1,
            'solver.t2': getattr(self.solver, 't2', None) is self.doubles,
            'solver.doubles': getattr(self.solver, 'doubles', None) is self.doubles,
            'solver._cd_initial_doubles': (getattr(self.solver, '_cd_initial_doubles', None) is self.mp2_doubles),
            'solver._rr_fock': getattr(self.solver, '_rr_fock', None) is self.fock,
            'solver._rr_orbital_energies': (
                getattr(self.solver, '_rr_orbital_energies', None) is self.orbital_energies
            ),
            'solver.mo_coeff': getattr(self.solver, 'mo_coeff', None) is self.mo_coeff,
            'engine.integrals': getattr(self.engine, 'integrals', None) is self.integrals,
            'engine.projector': getattr(self.engine, 'projector', None) is self.projector,
            'engine.fock': getattr(self.engine, 'fock', None) is self.fock,
            'engine.orbital_energies': (getattr(self.engine, 'orbital_energies', None) is self.orbital_energies),
            'engine.denominator': getattr(self.engine, 'denominator', None) is self.denominator,
            'doubles.projector': self.doubles.projector is self.projector,
            'mp2_doubles.projector': self.mp2_doubles.projector is self.projector,
            'integrals.factors': self.integrals.factors is self.integrals._factors,
            'transfer_counter': (getattr(self.solver.run_metrics, 'transfers', None) is self.transfer_counter),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError('RR lifecycle object identity changed: ' + ', '.join(failed))
        if self.pair_dimension != self.projector.full_dimension:
            raise ValueError('RR projector dimension does not match CD orbitals')
        if self.t1.shape != (self.nocc, self.nvir):
            raise ValueError('RR singles shape does not match CD orbitals')
        rr_rank = int(self.projector.rank)
        expected_shapes = {
            'projector vectors': (self.pair_dimension, rr_rank),
            'projector eigenvalues': (rr_rank,),
            'converged RR core': (rr_rank, rr_rank),
            'MP2 RR core': (rr_rank, rr_rank),
            'denominator matrix': (rr_rank, rr_rank),
            'denominator eigenvalues': (rr_rank,),
            'denominator eigenvectors': (rr_rank, rr_rank),
        }
        shape_arrays = (
            self.projector.vectors,
            self.projector.eigenvalues,
            self.doubles.core,
            self.mp2_doubles.core,
            self.denominator.matrix,
            self.denominator.eigenvalues,
            self.denominator.eigenvectors,
        )
        for (name, expected), value in zip(expected_shapes.items(), shape_arrays):
            if value.shape != expected:
                raise ValueError(f'{name} must have shape {expected}')
        if self.doubles.rank != rr_rank or self.mp2_doubles.rank != rr_rank:
            raise ValueError('RR doubles rank does not match the captured projector')
        if (self.denominator.nocc, self.denominator.nvir) != (self.nocc, self.nvir):
            raise ValueError('RR denominator orbital dimensions changed')
        backend = _array_module(self.integrals.factors)
        arrays = (
            self.projector.vectors,
            self.projector.eigenvalues,
            self.t1,
            self.doubles.core,
            self.mp2_doubles.core,
            self.fock,
            self.orbital_energies,
            self.denominator.matrix,
            self.denominator.eigenvalues,
            self.denominator.eigenvectors,
            self.loo,
            self.lov,
            self.lvv,
        )
        if any(_array_module(value) is not backend for value in arrays):
            raise TypeError('RR lifecycle audit tensors must use one backend')
        if any(np.dtype(value.dtype) != np.dtype(np.float64) for value in arrays):
            raise TypeError('RR lifecycle audit tensors must use real FP64')
        current_pointer_arrays = {
            'integral_factors': self.integrals.factors,
            'projector_vectors': self.projector.vectors,
            'projector_eigenvalues': self.projector.eigenvalues,
            't1': self.t1,
            'doubles_core': self.doubles.core,
            'mp2_doubles_core': self.mp2_doubles.core,
            'fock': self.fock,
            'orbital_energies': self.orbital_energies,
            'denominator_matrix': self.denominator.matrix,
            'denominator_eigenvalues': self.denominator.eigenvalues,
            'denominator_eigenvectors': self.denominator.eigenvectors,
            'loo': self.loo,
            'lov': self.lov,
            'lvv': self.lvv,
            'mo_coeff': self.mo_coeff,
        }
        current_pointers = tuple((name, _array_pointer(value)) for name, value in current_pointer_arrays.items())
        if current_pointers != self.array_pointer_snapshot:
            raise ValueError('RR lifecycle array pointer identity changed')

    def metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            'case_id': self.case_id,
            'solver_runtime_identity': id(self.solver),
            'engine_runtime_identity': id(self.engine),
            'integrals_runtime_identity': id(self.integrals),
            'integral_factors_runtime_identity': id(self.integrals.factors),
            'projector_runtime_identity': id(self.projector),
            't1_runtime_identity': id(self.t1),
            'doubles_runtime_identity': id(self.doubles),
            'mp2_doubles_runtime_identity': id(self.mp2_doubles),
            'fock_runtime_identity': id(self.fock),
            'orbital_energies_runtime_identity': id(self.orbital_energies),
            'denominator_runtime_identity': id(self.denominator),
            'lov_view_runtime_identity': id(self.lov),
            'mo_coeff_runtime_identity': id(self.mo_coeff),
            'nocc': self.nocc,
            'nvir': self.nvir,
            'pair_dimension': self.pair_dimension,
            'rr_rank': int(self.projector.rank),
            'array_pointer_snapshot': dict(self.array_pointer_snapshot),
            'identity_attestation_scope': 'python-object-and-array-pointer-identity',
            'object_and_pointer_identity_validated': True,
            'content_immutability_cryptographically_attested': False,
            'tensor_contents_hashed': False,
            'in_place_content_mutation_detection_supported': False,
            'identity_scope_limitation': (
                'object replacement and pointer relocation are detected; '
                'in-place tensor value mutation is outside this attestation'
            ),
        }


@dataclass(frozen=True)
class ProvenanceTokens:
    orbitals: dict[str, Any]
    integrals: dict[str, Any]
    hcore: dict[str, Any]
    digest: str

    def context(self) -> dict[str, Any]:
        return {
            'orbital_identity': self.orbitals,
            'integral_identity': self.integrals,
            'hcore_identity': self.hcore,
            'audit_provenance_digest': self.digest,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            'orbital_identity': self.orbitals,
            'integral_identity': self.integrals,
            'hcore_identity': self.hcore,
            'digest': self.digest,
            'binding': 'runtime-object-and-array-pointer-identity-plus-explicit-semantics',
            'content_immutability_cryptographically_attested': False,
            'tensor_contents_hashed': False,
        }


def make_provenance_tokens(
    binding: RRLifecycleBinding,
    *,
    hcore_mo: Any,
    semantics: str,
    orbital_fingerprint: str | None,
) -> ProvenanceTokens:
    """Bind every audit artifact to the captured lifecycle and hcore object."""

    binding.validate()
    if semantics not in {
        'physical-cd-hamiltonian',
        'direct-cd-effective-one-body',
    }:
        raise ValueError('unsupported F-hat one-body semantics')
    if _array_module(hcore_mo) is not _array_module(binding.integrals.factors):
        raise TypeError('hcore and RR lifecycle must use one backend')
    if hcore_mo.shape != binding.fock.shape:
        raise ValueError('hcore shape does not match the RR Fock matrix')
    if orbital_fingerprint is not None:
        orbital_fingerprint = str(orbital_fingerprint)
        if not orbital_fingerprint.strip():
            raise ValueError('orbital_fingerprint must be non-empty or None')
    orbitals = {
        'schema': 'gpu4pyscf.water2-thc-audit-orbitals.v1',
        'case_id': binding.case_id,
        'solver_runtime_identity': id(binding.solver),
        'mo_coeff_runtime_identity': id(binding.mo_coeff),
        'mo_coeff_array_pointer': _array_pointer(binding.mo_coeff),
        'projector_runtime_identity': id(binding.projector),
        'projector_vectors_array_pointer': _array_pointer(binding.projector.vectors),
        't1_runtime_identity': id(binding.t1),
        't1_array_pointer': _array_pointer(binding.t1),
        'doubles_runtime_identity': id(binding.doubles),
        'doubles_core_array_pointer': _array_pointer(binding.doubles.core),
        'orbital_energies_runtime_identity': id(binding.orbital_energies),
        'orbital_energies_array_pointer': _array_pointer(binding.orbital_energies),
        'denominator_runtime_identity': id(binding.denominator),
        'denominator_matrix_array_pointer': _array_pointer(binding.denominator.matrix),
        'orbital_fingerprint': orbital_fingerprint or 'fresh-scf-memory',
        'nocc': binding.nocc,
        'nvir': binding.nvir,
    }
    integrals = {
        'schema': 'gpu4pyscf.water2-thc-audit-integrals.v1',
        'provider_runtime_identity': id(binding.integrals),
        'factors_runtime_identity': id(binding.integrals.factors),
        'factors_array_pointer': _array_pointer(binding.integrals.factors),
        'lov_view_runtime_identity': id(binding.lov),
        'lov_view_array_pointer': _array_pointer(binding.lov),
        'mp2_doubles_runtime_identity': id(binding.mp2_doubles),
        'mp2_doubles_core_array_pointer': _array_pointer(binding.mp2_doubles.core),
        'factorization': str(binding.integrals.factorization),
        'threshold': binding.integrals.threshold,
        'naux': int(binding.integrals.naux),
    }
    hcore = {
        'schema': 'gpu4pyscf.water2-thc-audit-hcore.v1',
        'semantics': semantics,
        'hcore_runtime_identity': id(hcore_mo),
        'hcore_array_pointer': _array_pointer(hcore_mo),
        'rr_fock_runtime_identity': id(binding.fock),
        'rr_fock_array_pointer': _array_pointer(binding.fock),
    }
    digest = _sha256_json({'orbitals': orbitals, 'integrals': integrals, 'hcore': hcore})
    return ProvenanceTokens(orbitals, integrals, hcore, digest)


@dataclass(frozen=True)
class FHatSemantics:
    name: str
    hcore_mo: Any
    fhat: Any
    zero_t1_fock: Any
    tokens: ProvenanceTokens
    correction_max_abs: float

    def validate(self, binding: RRLifecycleBinding) -> None:
        binding.validate()
        if self.tokens.hcore['hcore_runtime_identity'] != id(self.hcore_mo):
            raise ValueError('F-hat hcore provenance object identity changed')
        if self.tokens.hcore['hcore_array_pointer'] != _array_pointer(self.hcore_mo):
            raise ValueError('F-hat hcore provenance array pointer changed')
        if self.tokens.hcore['semantics'] != self.name:
            raise ValueError('F-hat semantic label changed')
        if self.fhat.provenance_context != self.tokens.context():
            raise ValueError('F-hat provenance context changed')
        if self.fhat.transfer_counter is not binding.transfer_counter:
            raise ValueError('F-hat did not retain the RR transfer counter')

    def metadata(self) -> dict[str, Any]:
        result = self.fhat.metadata()
        result.update(
            {
                'one_body_semantics': self.name,
                'hcore_runtime_identity': id(self.hcore_mo),
                'zero_t1_fock_runtime_identity': id(self.zero_t1_fock),
                'effective_correction_max_abs': self.correction_max_abs,
                'provenance_tokens': self.tokens.metadata(),
                **_copy_flags(),
            }
        )
        return result


def _to_backend(value: np.ndarray, reference: Any, counter: TransferCounter, operation: str):
    xp = _array_module(reference)
    if xp is np:
        return np.asarray(value, dtype=np.float64)
    host = np.asarray(value, dtype=np.float64, order='C')
    result = xp.asarray(host)
    counter.record_h2d(int(host.nbytes), operation=operation)
    return result


def _join_fhat_blocks(fhat: Any):
    xp = _array_module(fhat.fhat_oo)
    return xp.concatenate(
        (
            xp.concatenate((fhat.fhat_oo, fhat.fhat_ov), axis=1),
            xp.concatenate((fhat.fhat_vo, fhat.fhat_vv), axis=1),
        ),
        axis=0,
    )


def _read_scalar(value: Any, counter: TransferCounter, *, operation: str) -> float:
    if isinstance(value, np.generic):
        xp = np
        scalar = np.asarray(value)
    else:
        xp = _array_module(value)
        scalar = xp.asarray(value)
    if scalar.ndim != 0:
        raise ValueError('audit control read must be scalar')
    if xp is not np:
        counter.record_d2h(int(scalar.nbytes), operation=operation)
    return float(scalar.item())


def build_fhat_semantics(
    binding: RRLifecycleBinding,
    *,
    auxiliary_block_size: int,
    orbital_fingerprint: str | None = None,
) -> tuple[FHatSemantics, FHatSemantics]:
    """Build physical-CD and RR-aligned effective one-body F-hat objects."""

    binding.validate()
    block_size = min(
        _positive_int(auxiliary_block_size, name='auxiliary_block_size'),
        int(binding.integrals.naux),
    )
    coefficient = np.asarray(binding.mo_coeff)
    mask = np.asarray(binding.solver.get_frozen_mask(), dtype=bool)
    if coefficient.ndim != 2 or mask.shape != (coefficient.shape[1],):
        raise ValueError('RR active MO coefficient/mask dimensions are invalid')
    active_coefficient = coefficient[:, mask]
    if active_coefficient.shape[1] != binding.nocc + binding.nvir:
        raise ValueError('RR active MO count changed after convergence')
    raw_hcore = np.asarray(binding.solver._scf.get_hcore(binding.solver.mol), dtype=np.float64)
    if raw_hcore.shape != (active_coefficient.shape[0],) * 2:
        raise ValueError('physical AO hcore does not match the active MOs')
    physical_host = active_coefficient.T @ raw_hcore @ active_coefficient
    physical_hcore = _to_backend(
        physical_host,
        binding.integrals.factors,
        binding.transfer_counter,
        'thc_water2_physical_hcore',
    )
    physical_tokens = make_provenance_tokens(
        binding,
        hcore_mo=physical_hcore,
        semantics='physical-cd-hamiltonian',
        orbital_fingerprint=orbital_fingerprint,
    )
    xp = _array_module(binding.integrals.factors)
    zeros = xp.zeros_like(binding.t1)
    zero_physical = build_t1_transformed_fhat(
        physical_hcore,
        binding.integrals.factors,
        zeros,
        auxiliary_block_size=block_size,
        transfer_counter=binding.transfer_counter,
        provenance_context=physical_tokens.context(),
    )
    fock_cd = _join_fhat_blocks(zero_physical)
    _sync(fock_cd)
    del zero_physical, zeros

    effective_correction = binding.fock - fock_cd
    effective_hcore = physical_hcore + effective_correction
    correction_max_abs = _read_scalar(
        xp.max(xp.abs(effective_correction)),
        binding.transfer_counter,
        operation='thc_water2_effective_hcore_correction',
    )
    effective_tokens = make_provenance_tokens(
        binding,
        hcore_mo=effective_hcore,
        semantics='direct-cd-effective-one-body',
        orbital_fingerprint=orbital_fingerprint,
    )
    physical_fhat = build_t1_transformed_fhat(
        physical_hcore,
        binding.integrals.factors,
        binding.t1,
        auxiliary_block_size=block_size,
        transfer_counter=binding.transfer_counter,
        provenance_context=physical_tokens.context(),
    )
    effective_fhat = build_t1_transformed_fhat(
        effective_hcore,
        binding.integrals.factors,
        binding.t1,
        auxiliary_block_size=block_size,
        transfer_counter=binding.transfer_counter,
        provenance_context=effective_tokens.context(),
    )
    physical = FHatSemantics(
        name='physical-cd-hamiltonian',
        hcore_mo=physical_hcore,
        fhat=physical_fhat,
        zero_t1_fock=fock_cd,
        tokens=physical_tokens,
        correction_max_abs=0.0,
    )
    effective = FHatSemantics(
        name='direct-cd-effective-one-body',
        hcore_mo=effective_hcore,
        fhat=effective_fhat,
        zero_t1_fock=binding.fock,
        tokens=effective_tokens,
        correction_max_abs=correction_max_abs,
    )
    physical.validate(binding)
    effective.validate(binding)
    zero_effective_t1 = xp.zeros_like(binding.t1)
    zero_effective = build_t1_transformed_fhat(
        effective_hcore,
        binding.integrals.factors,
        zero_effective_t1,
        auxiliary_block_size=block_size,
        transfer_counter=binding.transfer_counter,
        provenance_context=effective_tokens.context(),
    )
    zero_effective_fock = _join_fhat_blocks(zero_effective)
    _sync(zero_effective_fock)
    aligned = _read_scalar(
        xp.max(xp.abs(zero_effective_fock - binding.fock)),
        binding.transfer_counter,
        operation='thc_water2_effective_zero_t1_fock_identity',
    )
    del zero_effective_t1, zero_effective, zero_effective_fock
    if aligned > 5e-12:
        raise RuntimeError(f'direct-CD effective one-body failed to reproduce the RR Fock: {aligned:.3e}')
    return physical, effective


@dataclass(frozen=True)
class RREquationOracle:
    semantics: str
    singles: Any
    doubles_rr: Any
    singles_numerator: Any
    doubles_numerator: Any
    denominator: ProjectedPairDenominator
    denominator_matrix: Any
    eia: Any

    def metadata(self, counter: TransferCounter) -> dict[str, Any]:
        xp = _array_module(self.singles)
        residual = xp.sqrt(xp.vdot(self.singles, self.singles).real + xp.vdot(self.doubles_rr, self.doubles_rr).real)
        return {
            'semantics': self.semantics,
            'residual_norm': _read_scalar(
                residual,
                counter,
                operation=f'thc_water2_{self.semantics}_oracle_norm',
            ),
            'ordinary_rr_builder_output': 'numerator',
            'denominator_subtracted_on_oracle_side_once': True,
            'denominator_runtime_identity': id(self.denominator),
            'denominator_matrix_runtime_identity': id(self.denominator_matrix),
            'denominator_matrix_array_pointer': _array_pointer(self.denominator_matrix),
            'thc_audit_denominator_subtraction_count': 0,
            'singles_formula': 'numerator-eia*t1',
            'doubles_formula': 'core-(D@C+C@D)',
        }


def equation_residual_from_numerators(
    *,
    singles_numerator: Any,
    doubles_numerator: Any,
    t1: Any,
    core: Any,
    eia: Any,
    denominator_matrix: Any,
) -> tuple[Any, Any]:
    """Subtract the RR denominator action exactly once."""

    singles = singles_numerator - eia * t1
    doubles = doubles_numerator - (denominator_matrix @ core + core @ denominator_matrix)
    return singles, doubles


def build_rr_equation_oracle(
    binding: RRLifecycleBinding,
    *,
    semantics: str,
    fock: Any,
    orbital_energies: Any,
    auxiliary_block_size: int,
    virtual_block_size: int,
) -> RREquationOracle:
    """Evaluate the independent RR/CD numerator and form its equation residual."""

    binding.validate()
    nocc, nvir = binding.nocc, binding.nvir
    if fock.shape != binding.fock.shape:
        raise ValueError('oracle Fock shape changed')
    if orbital_energies.shape != binding.orbital_energies.shape:
        raise ValueError('oracle orbital-energy shape changed')
    xp = _array_module(binding.integrals.factors)
    if _array_module(fock) is not xp or _array_module(orbital_energies) is not xp:
        raise TypeError('oracle Fock/energies must remain on the RR backend')
    occupied = orbital_energies[:nocc]
    virtual = orbital_energies[nocc:]
    eia = occupied[:, None] - virtual[None, :] - float(binding.solver.level_shift)
    if semantics == 'direct-cd-effective-one-body':
        if fock is not binding.fock or orbital_energies is not binding.orbital_energies:
            raise ValueError('effective one-body oracle must use the captured RR Fock and energies')
        denominator = binding.denominator
    elif semantics == 'physical-cd-hamiltonian':
        denominator = ProjectedPairDenominator.build(
            binding.projector,
            eia,
            nocc,
            nvir,
            transfer_counter=binding.transfer_counter,
        )
    else:
        raise ValueError('unsupported RR equation oracle semantics')
    singles_result = build_ccsd_singles_numerator(
        binding.doubles,
        binding.t1,
        fock[:nocc, :nocc],
        fock[:nocc, nocc:],
        fock[nocc:, nocc:],
        occupied,
        virtual,
        binding.loo,
        binding.lov,
        binding.lvv,
        level_shift=float(binding.solver.level_shift),
        auxiliary_block_size=min(auxiliary_block_size, int(binding.integrals.naux)),
    )
    doubles_result = build_projected_ccsd_doubles_numerator(
        binding.doubles,
        binding.t1,
        fock[:nocc, :nocc],
        fock[:nocc, nocc:],
        fock[nocc:, nocc:],
        occupied,
        virtual,
        binding.loo,
        binding.lov,
        binding.lvv,
        level_shift=float(binding.solver.level_shift),
        auxiliary_block_size=min(auxiliary_block_size, int(binding.integrals.naux)),
        virtual_block_size=min(virtual_block_size, nvir),
        ring_kernel=str(binding.solver.rr_ring_kernel),
        transfer_counter=binding.transfer_counter,
    )
    singles, doubles = equation_residual_from_numerators(
        singles_numerator=singles_result.numerator,
        doubles_numerator=doubles_result.core,
        t1=binding.t1,
        core=binding.doubles.core,
        eia=eia,
        denominator_matrix=denominator.matrix,
    )
    return RREquationOracle(
        semantics=semantics,
        singles=singles,
        doubles_rr=doubles,
        singles_numerator=singles_result.numerator,
        doubles_numerator=doubles_result.core,
        denominator=denominator,
        denominator_matrix=denominator.matrix,
        eia=eia,
    )


def compare_audit_to_oracle(
    audit: Any,
    oracle: RREquationOracle,
    *,
    counter: TransferCounter,
    delta_tolerance: float,
    residual_tolerance: float,
) -> dict[str, Any]:
    """Compare equation residuals directly; never subtract audit denominators."""

    delta_tolerance = _positive_float(delta_tolerance, name='delta_tolerance')
    residual_tolerance = _positive_float(residual_tolerance, name='residual_tolerance')
    xp = _array_module(audit.singles)
    if _array_module(oracle.singles) is not xp:
        raise TypeError('audit and RR oracle must use one backend')
    delta_singles = audit.singles - oracle.singles
    delta_doubles = audit.doubles_rr - oracle.doubles_rr
    delta_norm = xp.sqrt(xp.vdot(delta_singles, delta_singles).real + xp.vdot(delta_doubles, delta_doubles).real)
    audit_norm = xp.sqrt(xp.vdot(audit.singles, audit.singles).real + xp.vdot(audit.doubles_rr, audit.doubles_rr).real)
    values = {
        'singles_max_abs_delta': xp.max(xp.abs(delta_singles)),
        'doubles_max_abs_delta': xp.max(xp.abs(delta_doubles)),
        'combined_delta_norm': delta_norm,
        'audit_projected_residual_norm': audit_norm,
    }
    observed = {
        name: _read_scalar(
            value,
            counter,
            operation=f'thc_water2_compare_{name}',
        )
        for name, value in values.items()
    }
    delta_passed = observed['combined_delta_norm'] <= delta_tolerance
    residual_passed = observed['audit_projected_residual_norm'] <= residual_tolerance
    return {
        **observed,
        'delta_tolerance': delta_tolerance,
        'projected_residual_tolerance': residual_tolerance,
        'delta_gate_passed': bool(delta_passed),
        'projected_residual_gate_passed': bool(residual_passed),
        'diagnostic_gate_passed': bool(delta_passed and residual_passed),
        'audit_residual_contains_full_fhat_diagonal_action': True,
        'audit_denominator_subtraction_count': 0,
        'oracle_denominator_subtraction_count': 1,
    }


def _assert_audit_fail_closed(audit: Any) -> None:
    metadata = audit.metadata()
    required_false = (
        'accepted',
        'production_enabled',
        'complete_validated',
        'complete_ccsd_residual',
        'formal_full_residual_eligible',
        'performance_eligible',
    )
    if any(metadata.get(name) is not False for name in required_false):
        raise RuntimeError('complete audit unexpectedly opened a production gate')
    if metadata.get('audit_only') is not True:
        raise RuntimeError('complete audit lost its audit-only marker')


def fit_amplitude_factors(
    binding: RRLifecycleBinding,
    *,
    fit_tolerance: float,
    initial_rank: int,
    max_rank: int,
    max_iterations: int,
    als_convergence_tolerance: float,
    ridge: float,
    seed: int,
    orthogonality_cutoff: float,
    orthogonality_tolerance: float,
) -> THCProjectorFactors:
    """Fit one amplitude projector while retaining the captured projector."""

    binding.validate()
    factors = fit_weighted_thc_projector_adaptive(
        binding.projector,
        binding.nocc,
        binding.nvir,
        fit_tolerance=fit_tolerance,
        orthogonality_cutoff=orthogonality_cutoff,
        initial_rank=initial_rank,
        max_rank=max_rank,
        max_iterations=max_iterations,
        als_convergence_tolerance=als_convergence_tolerance,
        ridge=ridge,
        seed=seed,
        orthogonality_tolerance=orthogonality_tolerance,
        transfer_counter=binding.transfer_counter,
    )
    binding.validate()
    if factors.eigenvalues is not binding.projector.eigenvalues:
        raise ValueError('amplitude fit did not retain projector eigenvalue identity')
    return factors


def evaluate_candidate(
    binding: RRLifecycleBinding,
    amplitude_factors: THCProjectorFactors,
    *,
    eri_spec: ERIFitSpec,
    exact_eri_endpoint: bool,
    semantic_oracles: Sequence[tuple[FHatSemantics, RREquationOracle]],
    tolerances: AuditTolerances,
    occupation_change_floor: float,
    fit_max_iterations: int,
    als_convergence_tolerance: float,
    ridge: float,
    seed: int,
    auxiliary_block_size: int,
    virtual_block_size: int,
    algorithm_x_block_size: int,
    preflight: MemoryPreflight,
    stage: str,
) -> dict[str, Any]:
    """Run one ERI fit and complete audit against one or more F-hat semantics."""

    del virtual_block_size  # recorded by the RR oracle, not used by THC endpoints
    binding.validate()
    if not preflight.passed:
        raise MemoryError('candidate memory preflight failed: ' + '; '.join(preflight.reasons))
    if amplitude_factors.eigenvalues is not binding.projector.eigenvalues:
        raise ValueError('amplitude factors belong to another RR projector')
    amplitude_core = amplitude_factors.amplitude_core(binding.doubles.core)
    preprocess_started = time.perf_counter()
    eri_preprocess = build_weighted_eri_thc_preprocess(
        binding.mp2_doubles,
        amplitude_factors,
        binding.lov,
        occupation_change_floor=occupation_change_floor,
        eri_thc_rank=eri_spec.rank,
        fit_tolerance=(0.0 if exact_eri_endpoint else eri_spec.fit_tolerance),
        exact_pair_endpoint=exact_eri_endpoint,
        max_iterations=fit_max_iterations,
        als_convergence_tolerance=als_convergence_tolerance,
        ridge=ridge,
        seed=seed,
        auxiliary_block_size=auxiliary_block_size,
        transfer_counter=binding.transfer_counter,
    )
    _sync(eri_preprocess.factors.core)
    preprocess_seconds = time.perf_counter() - preprocess_started
    eri_preprocess.validate_binding(binding.mp2_doubles, amplitude_factors, binding.lov)
    binding.validate()

    semantic_results = []
    for semantic, oracle in semantic_oracles:
        semantic.validate(binding)
        attested = IdentityAttestedERITHCFactors(
            eri_preprocess.factors,
            orbital_identity_token=semantic.tokens.orbitals,
            integral_identity_token=semantic.tokens.integrals,
        )
        started = time.perf_counter()
        audit = assemble_complete_thc_ccsd_audit(
            amplitude_factors.y_occ,
            amplitude_factors.y_vir,
            amplitude_core,
            amplitude_factors.tau,
            attested,
            semantic.fhat,
            orbital_identity_token=semantic.tokens.orbitals,
            integral_identity_token=semantic.tokens.integrals,
            hcore_identity_token=semantic.tokens.hcore,
            omega_ac_path='joint-6',
            algorithm4_5_outer_block_size=1,
            algorithm6_x_block_size=algorithm_x_block_size,
            algorithm7_x_block_size=algorithm_x_block_size,
            algorithm8_cholesky_block_size=auxiliary_block_size,
            amplitude_core_symmetry_tolerance=(tolerances.amplitude_core_symmetry_atol),
            transfer_counter=binding.transfer_counter,
            require_eq35_equivalence=False,
        )
        _sync(audit.doubles_rr)
        audit_seconds = time.perf_counter() - started
        _assert_audit_fail_closed(audit)
        identity_stage = stage == 'exact-anchor'
        oracle_metadata = oracle.metadata(binding.transfer_counter)
        comparison = compare_audit_to_oracle(
            audit,
            oracle,
            counter=binding.transfer_counter,
            delta_tolerance=(tolerances.exact_identity_atol if identity_stage else tolerances.candidate_delta_norm),
            residual_tolerance=(
                max(
                    tolerances.candidate_projected_residual_norm,
                    oracle_metadata['residual_norm'] + tolerances.exact_identity_atol,
                )
                if semantic.name == 'physical-cd-hamiltonian'
                else tolerances.candidate_projected_residual_norm
            ),
        )
        semantic_results.append(
            {
                'semantics': semantic.name,
                'audit_seconds': audit_seconds,
                'comparison': comparison,
                'audit': audit.metadata(),
                'fhat': semantic.metadata(),
                'oracle': oracle_metadata,
                **_copy_flags(),
            }
        )

    inexact = amplitude_factors.thc_rank < binding.pair_dimension or eri_spec.rank < binding.pair_dimension
    diagnostic_passed = all(row['comparison']['diagnostic_gate_passed'] for row in semantic_results)
    if stage != 'exact-anchor':
        diagnostic_passed = diagnostic_passed and inexact
    return {
        'stage': stage,
        'status': 'completed',
        'amplitude_fit': amplitude_factors.metadata(),
        'amplitude_core_shape': list(amplitude_core.shape),
        'amplitude_core_runtime_identity': id(amplitude_core),
        'eri_fit_spec': eri_spec.metadata(),
        'eri_exact_pair_endpoint': bool(exact_eri_endpoint),
        'eri_preprocess': eri_preprocess.metadata(),
        'eri_preprocess_seconds': preprocess_seconds,
        'preflight': preflight.metadata(),
        'semantics': semantic_results,
        'inexact_compression_achieved': bool(inexact),
        'diagnostic_gate_passed': bool(diagnostic_passed),
        'energy_gate_evaluated': False,
        'interaction_energy_gate_evaluated': False,
        'full_space_residual_gate_evaluated': False,
        'reason_not_accepted': (
            'audit-only residual seam; Algorithm 7 literal Eq. 35 mapping, '
            'iterative energy, CP interaction energy, and full-space gates '
            'remain unresolved'
        ),
        **_copy_flags(),
    }


def validate_scan_order(
    amplitude_tolerances: Sequence[float], eri_specs: Sequence[ERIFitSpec]
) -> tuple[tuple[float, ...], tuple[ERIFitSpec, ...]]:
    """Require deterministic loose-to-tight scans without a Cartesian grid."""

    amplitude = tuple(_positive_float(value, name='amplitude fit tolerance') for value in amplitude_tolerances)
    eri = tuple(eri_specs)
    if not amplitude:
        raise ValueError('at least one inexact amplitude tolerance is required')
    if len(eri) < 2:
        raise ValueError('at least two ordered ERI candidates are required')
    if any(spec.fit_tolerance == 0.0 for spec in eri):
        raise ValueError('inexact ERI candidate tolerances must be positive')
    if any(right >= left for left, right in pairwise(amplitude)):
        raise ValueError('amplitude tolerances must be strictly loose-to-tight')
    for left, right in pairwise(eri):
        no_tighter = (
            right.rank < left.rank
            or right.fit_tolerance > left.fit_tolerance
            or (right.rank == left.rank and right.fit_tolerance == left.fit_tolerance)
        )
        if no_tighter:
            raise ValueError(
                'ERI candidates must have nondecreasing rank, nonincreasing tolerance, and become strictly tighter'
            )
    return amplitude, eri


def select_first_and_next_tighter(
    records: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return the first diagnostic pass and its immediate tighter neighbor."""

    first = next(
        (index for index, row in enumerate(records) if row.get('diagnostic_gate_passed') is True),
        None,
    )
    if first is None:
        return ()
    stop = min(first + 2, len(records))
    return tuple(records[first:stop])


def _candidate_preflight(
    binding: RRLifecycleBinding,
    budget: ResourceBudget,
    *,
    amplitude_rank: int,
    eri_rank: int,
    auxiliary_block_size: int,
    x_block_size: int,
) -> MemoryPreflight:
    estimate = estimate_candidate_memory(
        naux=int(binding.integrals.naux),
        nocc=binding.nocc,
        nvir=binding.nvir,
        rr_rank=int(binding.projector.rank),
        amplitude_rank=amplitude_rank,
        eri_rank=eri_rank,
        auxiliary_block_size=auxiliary_block_size,
        x_block_size=x_block_size,
        itemsize=np.dtype(binding.integrals.factors.dtype).itemsize,
    )
    current, free, total = _memory_snapshot(binding.integrals.factors)
    return check_memory_preflight(
        estimate,
        budget,
        hbm_current_used_bytes=current,
        hbm_free_bytes=free,
        hbm_total_bytes=total,
        host_rss_current_bytes=_host_rss_bytes(),
    )


def _failed_candidate(*, stage: str, controls: dict[str, Any], error: BaseException) -> dict[str, Any]:
    return {
        'stage': stage,
        'status': 'failed',
        'controls': controls,
        'error_type': type(error).__name__,
        'error': str(error),
        'diagnostic_gate_passed': False,
        **_copy_flags(),
    }


def run_staged_audit(
    binding: RRLifecycleBinding,
    *,
    amplitude_tolerances: Sequence[float],
    amplitude_initial_rank: int,
    amplitude_max_rank: int,
    eri_specs: Sequence[ERIFitSpec],
    tolerances: AuditTolerances,
    budget: ResourceBudget,
    occupation_change_floor: float,
    fit_max_iterations: int,
    als_convergence_tolerance: float,
    ridge: float,
    amplitude_seed: int,
    eri_seed: int,
    orthogonality_cutoff: float,
    orthogonality_tolerance: float,
    auxiliary_block_size: int,
    virtual_block_size: int,
    algorithm_x_block_size: int,
    orbital_fingerprint: str | None = None,
    hbm_observer: SynchronizedHBMObserver | None = None,
) -> dict[str, Any]:
    """Run exact anchor, amplitude scan, then ERI scan on one RR lifecycle."""

    binding.validate()
    if hbm_observer is not None:
        hbm_observer.sample('audit-entry')
    amplitude_tolerances, eri_specs = validate_scan_order(amplitude_tolerances, eri_specs)
    if binding.case_id != CASE_ID or binding.pair_dimension != PAIR_DIMENSION:
        raise ValueError('this audit seam is fixed to WATER27 water2-tz with OV=1060')
    pair_dimension = binding.pair_dimension
    initial_rank = _positive_int(amplitude_initial_rank, name='amplitude_initial_rank')
    max_rank = _positive_int(amplitude_max_rank, name='amplitude_max_rank')
    if initial_rank < binding.projector.rank:
        raise ValueError('amplitude_initial_rank cannot be smaller than RR rank')
    if not initial_rank <= max_rank < pair_dimension:
        raise ValueError('inexact amplitude ranks must satisfy initial <= max < OV')
    if any(spec.rank >= pair_dimension for spec in eri_specs):
        raise ValueError('inexact ERI candidate rank must be smaller than OV')
    block_size = min(
        _positive_int(auxiliary_block_size, name='auxiliary_block_size'),
        int(binding.integrals.naux),
    )
    vblock = min(_positive_int(virtual_block_size, name='virtual_block_size'), binding.nvir)
    xblock = _positive_int(algorithm_x_block_size, name='algorithm_x_block_size')
    occupation_change_floor = _positive_float(occupation_change_floor, name='occupation_change_floor')

    # This estimate includes the two F-hat semantics and oracle workspaces, so
    # it must be checked before any of those large audit-only objects exist.
    exact_preflight = _candidate_preflight(
        binding,
        budget,
        amplitude_rank=pair_dimension,
        eri_rank=pair_dimension,
        auxiliary_block_size=block_size,
        x_block_size=xblock,
    )
    if not exact_preflight.passed:
        raise MemoryError('exact full-pair anchor failed preflight: ' + '; '.join(exact_preflight.reasons))
    physical, effective = build_fhat_semantics(
        binding,
        auxiliary_block_size=block_size,
        orbital_fingerprint=orbital_fingerprint,
    )
    if hbm_observer is not None:
        hbm_observer.sample('fhat-semantics-built')
    xp = _array_module(binding.fock)
    physical_energies = xp.diag(physical.zero_t1_fock).copy()
    physical_oracle = build_rr_equation_oracle(
        binding,
        semantics=physical.name,
        fock=physical.zero_t1_fock,
        orbital_energies=physical_energies,
        auxiliary_block_size=block_size,
        virtual_block_size=vblock,
    )
    effective_oracle = build_rr_equation_oracle(
        binding,
        semantics=effective.name,
        fock=binding.fock,
        orbital_energies=binding.orbital_energies,
        auxiliary_block_size=block_size,
        virtual_block_size=vblock,
    )
    if hbm_observer is not None:
        hbm_observer.sample('rr-oracles-built')
    exact_amplitude: THCProjectorFactors | None = None
    try:
        exact_amplitude = fit_amplitude_factors(
            binding,
            fit_tolerance=0.0,
            initial_rank=pair_dimension,
            max_rank=pair_dimension,
            max_iterations=fit_max_iterations,
            als_convergence_tolerance=als_convergence_tolerance,
            ridge=ridge,
            seed=amplitude_seed,
            orthogonality_cutoff=orthogonality_cutoff,
            orthogonality_tolerance=orthogonality_tolerance,
        )
        exact = evaluate_candidate(
            binding,
            exact_amplitude,
            eri_spec=ERIFitSpec(pair_dimension, 0.0),
            exact_eri_endpoint=True,
            semantic_oracles=(
                (physical, physical_oracle),
                (effective, effective_oracle),
            ),
            tolerances=tolerances,
            occupation_change_floor=occupation_change_floor,
            fit_max_iterations=fit_max_iterations,
            als_convergence_tolerance=als_convergence_tolerance,
            ridge=ridge,
            seed=eri_seed,
            auxiliary_block_size=block_size,
            virtual_block_size=vblock,
            algorithm_x_block_size=xblock,
            preflight=exact_preflight,
            stage='exact-anchor',
        )
    finally:
        del exact_amplitude
        if hbm_observer is not None:
            hbm_observer.sample('exact-anchor-exit')
    if not exact['diagnostic_gate_passed']:
        raise RuntimeError('analytic full-pair Algorithms 1--10 identity anchor failed')

    amplitude_records: list[dict[str, Any]] = []
    selected_amplitude: THCProjectorFactors | None = None
    selected_amplitude_index: int | None = None
    for index, fit_tolerance in enumerate(amplitude_tolerances):
        controls = {
            'fit_tolerance': fit_tolerance,
            'initial_rank': initial_rank,
            'max_rank': max_rank,
        }
        factors: THCProjectorFactors | None = None
        try:
            preflight = _candidate_preflight(
                binding,
                budget,
                amplitude_rank=max_rank,
                eri_rank=pair_dimension,
                auxiliary_block_size=block_size,
                x_block_size=xblock,
            )
            factors = fit_amplitude_factors(
                binding,
                fit_tolerance=fit_tolerance,
                initial_rank=initial_rank,
                max_rank=max_rank,
                max_iterations=fit_max_iterations,
                als_convergence_tolerance=als_convergence_tolerance,
                ridge=ridge,
                seed=amplitude_seed,
                orthogonality_cutoff=orthogonality_cutoff,
                orthogonality_tolerance=orthogonality_tolerance,
            )
            record = evaluate_candidate(
                binding,
                factors,
                eri_spec=ERIFitSpec(pair_dimension, 0.0),
                exact_eri_endpoint=True,
                semantic_oracles=((effective, effective_oracle),),
                tolerances=tolerances,
                occupation_change_floor=occupation_change_floor,
                fit_max_iterations=fit_max_iterations,
                als_convergence_tolerance=als_convergence_tolerance,
                ridge=ridge,
                seed=eri_seed,
                auxiliary_block_size=block_size,
                virtual_block_size=vblock,
                algorithm_x_block_size=xblock,
                preflight=preflight,
                stage='amplitude-scan-exact-eri',
            )
            record['scan_index'] = index
            amplitude_records.append(record)
            if selected_amplitude is None and record['diagnostic_gate_passed']:
                selected_amplitude = factors
                selected_amplitude_index = index
        except Exception as exc:  # noqa: BLE001 - persist a fail-closed scan row
            failed = _failed_candidate(
                stage='amplitude-scan-exact-eri',
                controls=controls,
                error=exc,
            )
            failed['scan_index'] = index
            amplitude_records.append(failed)
        finally:
            del factors
            if hbm_observer is not None:
                hbm_observer.sample(f'amplitude-scan-{index}-exit')

    eri_records: list[dict[str, Any]] = []
    if selected_amplitude is not None:
        for index, eri_spec in enumerate(eri_specs):
            try:
                preflight = _candidate_preflight(
                    binding,
                    budget,
                    amplitude_rank=selected_amplitude.thc_rank,
                    eri_rank=eri_spec.rank,
                    auxiliary_block_size=block_size,
                    x_block_size=xblock,
                )
                record = evaluate_candidate(
                    binding,
                    selected_amplitude,
                    eri_spec=eri_spec,
                    exact_eri_endpoint=False,
                    semantic_oracles=((effective, effective_oracle),),
                    tolerances=tolerances,
                    occupation_change_floor=occupation_change_floor,
                    fit_max_iterations=fit_max_iterations,
                    als_convergence_tolerance=als_convergence_tolerance,
                    ridge=ridge,
                    seed=eri_seed,
                    auxiliary_block_size=block_size,
                    virtual_block_size=vblock,
                    algorithm_x_block_size=xblock,
                    preflight=preflight,
                    stage='eri-scan-fixed-first-amplitude',
                )
                record['scan_index'] = index
                eri_records.append(record)
            except Exception as exc:  # noqa: BLE001 - persist a fail-closed scan row
                failed = _failed_candidate(
                    stage='eri-scan-fixed-first-amplitude',
                    controls=eri_spec.metadata(),
                    error=exc,
                )
                failed['scan_index'] = index
                eri_records.append(failed)
            finally:
                if hbm_observer is not None:
                    hbm_observer.sample(f'eri-scan-{index}-exit')
    final_candidates = select_first_and_next_tighter(eri_records)
    if hbm_observer is not None:
        hbm_observer.sample('audit-exit')
    return {
        'schema': SCHEMA,
        'case_id': binding.case_id,
        'strategy': (
            'exact-anchor -> exact-ERI amplitude scan -> fixed-first-amplitude ERI scan -> first-pass-plus-next-tighter'
        ),
        'cartesian_product_scan_performed': False,
        'lifecycle': binding.metadata(),
        'tolerances': tolerances.metadata(),
        'resource_budget': budget.metadata(),
        'one_body_semantics': {
            'physical': physical.metadata(),
            'effective': effective.metadata(),
        },
        'oracles': {
            'physical': physical_oracle.metadata(binding.transfer_counter),
            'effective': effective_oracle.metadata(binding.transfer_counter),
        },
        'exact_anchor': exact,
        'amplitude_scan': amplitude_records,
        'selected_amplitude_scan_index': selected_amplitude_index,
        'eri_scan': eri_records,
        'final_candidate_scan_indices': [row.get('scan_index') for row in final_candidates],
        'final_candidates': list(final_candidates),
        'staged_search_found_candidate': bool(final_candidates),
        'transfer_counter': binding.transfer_counter.to_dict(),
        'hbm_observations': (None if hbm_observer is None else hbm_observer.metadata()),
        'reason_not_accepted': (
            'development audit only; no iterative THC energy, CP interaction '
            'energy, full-space residual, WATER4 gate, or Eq. 35 resolution'
        ),
        **_copy_flags(),
    }


def _load_case() -> dict[str, Any]:
    payload = json.loads(CASE_FILE.read_text(encoding='utf-8'))
    cases = {case['id']: case for case in payload['cases']}
    case = dict(cases[CASE_ID])
    if case.get('expected_dimensions') != {'nao': 116, 'nocc': 10, 'nvir': 106}:
        raise ValueError('fixed WATER2 dimensions changed')
    return case


def _load_orbital_artifact(path: Path, mf: Any, case: dict[str, Any]) -> dict[str, Any]:
    """Use the benchmark's strict v1 artifact validator without GPU uploads."""

    identity = BENCHMARK._canonical_orbital_identity(CASE_ID, case, mf.mol)
    context = BENCHMARK._load_orbital_artifact(path, identity, mf.mol, np)
    for name, value in context['_arrays'].items():
        setattr(mf, name, np.array(value, copy=True))
    mf.e_tot = float(context['producer']['hf_energy'])
    return {key: value for key, value in context.items() if key != '_arrays'}


def _gint_runtime_options(
    args: argparse.Namespace,
    source_state: dict[str, Any],
) -> dict[str, Any]:
    if args.gint_column_backend != 'selected':
        return {}
    runtime_args = argparse.Namespace(
        gint_column_backend=args.gint_column_backend,
        gint_runtime_gate_receipt=args.gint_runtime_gate_receipt,
    )
    return BENCHMARK._recorded_gint_runtime_options(
        runtime_args,
        source_state,
        execution_mode='consumer-thc-complete-audit',
    )


def _build_rr_solver(
    args: argparse.Namespace,
    *,
    source_state: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from pyscf import gto, lib, scf

    from gpu4pyscf.cc.rrccsd import RRCCSD

    case = _load_case()
    if args.threads is not None:
        lib.num_threads(args.threads)
    mol = gto.M(
        atom=[(item[0], tuple(item[1:])) for item in case['geometry_angstrom']],
        unit='Angstrom',
        basis=case['basis'],
        charge=0,
        spin=0,
        cart=False,
        verbose=args.verbose,
        output=(None if args.log is None else str(args.log)),
        max_memory=args.max_memory_mb,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = args.scf_conv_tol
    mf.max_cycle = 100
    hf_started = time.perf_counter()
    mf.kernel()
    hf_seconds = time.perf_counter() - hf_started
    if not mf.converged:
        raise RuntimeError('WATER2 RHF did not converge')
    orbital = {
        'mode': 'fresh-scf-memory',
        'orbital_fingerprint': None,
        'artifact_sha256': None,
    }
    if args.orbital_artifact_in is not None:
        orbital = _load_orbital_artifact(args.orbital_artifact_in, mf, case)

    runtime_options = _gint_runtime_options(args, source_state)
    solver = RRCCSD(
        mf,
        eri_tol=args.eri_tol,
        rr_eig_cutoff=args.rr_eig_cutoff,
        eri_backend='cd',
        precision='fp64',
        direct_scf_tol=args.direct_scf_tol,
        cd_max_rank=args.cd_max_rank,
        gint_column_backend=args.gint_column_backend,
        gint_column_kernel=args.gint_column_kernel,
        gint_group_size=args.gint_group_size,
        gint_max_batch_size=args.gint_max_batch_size,
        cd_mo_block_size=args.cd_mo_block_size,
        denominator_tolerance=args.denominator_tolerance,
        denominator_max_rank=args.denominator_max_rank,
        rr_initial_rank=args.rr_initial_rank,
        rr_max_rank=args.rr_max_rank,
        rr_solver_tolerance=args.rr_solver_tolerance,
        rr_solver_maxiter=args.rr_solver_maxiter,
        rr_dense_fallback_dimension=args.rr_dense_fallback_dimension,
        rr_ritz_residual_tolerance=args.rr_ritz_residual_tolerance,
        rr_auxiliary_block_size=args.auxiliary_block_size,
        rr_virtual_block_size=args.virtual_block_size,
        rr_ring_kernel=args.rr_ring_kernel,
        **runtime_options,
    )
    solver.frozen = 0
    solver.conv_tol = args.cc_conv_tol
    solver.conv_tol_normt = args.cc_conv_tol_normt
    solver.max_cycle = args.cc_max_cycle
    solver.max_memory = args.max_memory_mb
    post_started = time.perf_counter()
    kernel = solver.kernel()
    _sync(kernel[2].core)
    rr_seconds = time.perf_counter() - post_started
    if kernel[1] is not solver.t1 or kernel[2] is not solver.doubles:
        raise ValueError('RR kernel return objects differ from retained lifecycle')
    method_metadata = solver.method_metadata()
    if not isinstance(method_metadata, dict):
        raise TypeError('RRCCSD method_metadata() must return a dictionary')
    provider_runtime_gate = method_metadata.get('provider_runtime_gate')
    if args.gint_runtime_gate_receipt is not None and not (
        isinstance(provider_runtime_gate, dict)
        and provider_runtime_gate.get('validated') is True
    ):
        raise RuntimeError(
            'selected GINT release receipt was not validated by the provider'
        )
    return solver, {
        'hf_seconds': hf_seconds,
        'rr_post_hf_seconds': rr_seconds,
        'hf_energy': float(mf.e_tot),
        'rr_correlation_energy': float(kernel[0]),
        'orbital': orbital,
        'solver_method_metadata': method_metadata,
        'provider_runtime_gate': provider_runtime_gate,
    }


def _git_state() -> dict[str, Any]:
    return BENCHMARK._git_state()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--orbital-artifact-in', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-memory-mb', type=int, default=110000)
    parser.add_argument('--verbose', type=int, default=4)
    parser.add_argument('--scf-conv-tol', type=float, default=1e-10)
    parser.add_argument('--cc-conv-tol', type=float, default=1e-8)
    parser.add_argument('--cc-conv-tol-normt', type=float, default=1e-6)
    parser.add_argument('--cc-max-cycle', type=int, default=50)

    parser.add_argument('--eri-tol', type=float, required=True)
    parser.add_argument('--rr-eig-cutoff', type=float, required=True)
    parser.add_argument('--direct-scf-tol', type=float, default=1e-13)
    parser.add_argument('--cd-max-rank', type=int)
    parser.add_argument(
        '--gint-column-backend',
        choices=('selected', 'restricted-reference'),
        default='selected',
    )
    parser.add_argument('--gint-runtime-gate-receipt', type=Path)
    parser.add_argument('--gint-column-kernel', choices=('reference', 'grouped'), default='reference')
    parser.add_argument('--gint-group-size', type=int, default=16)
    parser.add_argument('--gint-max-batch-size', type=int, default=32)
    parser.add_argument('--cd-mo-block-size', type=int, default=32)
    parser.add_argument('--denominator-tolerance', type=float, default=1e-10)
    parser.add_argument('--denominator-max-rank', type=int)
    parser.add_argument('--rr-initial-rank', type=int, default=32)
    parser.add_argument('--rr-max-rank', type=int)
    parser.add_argument('--rr-solver-tolerance', type=float, default=1e-10)
    parser.add_argument('--rr-solver-maxiter', type=int)
    parser.add_argument('--rr-dense-fallback-dimension', type=int, default=64)
    parser.add_argument('--rr-ritz-residual-tolerance', type=float)
    parser.add_argument('--rr-ring-kernel', choices=('reference', 'gemm'), default='reference')
    parser.add_argument('--auxiliary-block-size', type=int, default=32)
    parser.add_argument('--virtual-block-size', type=int, default=32)

    parser.add_argument(
        '--amplitude-fit-tolerances',
        type=float,
        nargs='+',
        required=True,
        help='strictly decreasing loose-to-tight tolerances',
    )
    parser.add_argument('--amplitude-initial-rank', type=int, required=True)
    parser.add_argument('--amplitude-max-rank', type=int, required=True)
    parser.add_argument(
        '--eri-candidate',
        type=ERIFitSpec.parse,
        action='append',
        required=True,
        help='ordered inexact RANK:TOLERANCE candidate; supply at least two',
    )
    parser.add_argument('--occupation-change-floor', type=float, required=True)
    parser.add_argument('--fit-max-iterations', type=int, default=500)
    parser.add_argument('--als-convergence-tolerance', type=float, default=1e-10)
    parser.add_argument('--fit-ridge', type=float, default=1e-12)
    parser.add_argument('--amplitude-seed', type=int, default=0)
    parser.add_argument('--eri-seed', type=int, default=0)
    parser.add_argument('--orthogonality-cutoff', type=float, default=1e-12)
    parser.add_argument('--orthogonality-tolerance', type=float, default=1e-10)
    parser.add_argument('--algorithm-x-block-size', type=int, default=1)
    parser.add_argument('--exact-identity-atol', type=float, default=1e-9)
    parser.add_argument('--candidate-delta-norm', type=float, default=1e-6)
    parser.add_argument('--candidate-projected-residual-norm', type=float, default=1e-6)
    parser.add_argument('--amplitude-core-symmetry-atol', type=float, default=1e-10)
    parser.add_argument('--hbm-limit-gib', type=float, default=72.0)
    parser.add_argument('--host-rss-limit-gib', type=float, default=110.0)
    parser.add_argument('--preflight-reserve-gib', type=float, default=0.5)
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'refusing to overwrite audit result: {args.output}')
    if args.orbital_artifact_in is not None and not args.orbital_artifact_in.is_file():
        raise FileNotFoundError(args.orbital_artifact_in)
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
    for name in ('threads', 'max_memory_mb', 'cc_max_cycle'):
        _positive_int(getattr(args, name), name=name)
    _nonnegative_int(args.verbose, name='verbose')
    for name in ('scf_conv_tol', 'cc_conv_tol', 'cc_conv_tol_normt'):
        _positive_float(getattr(args, name), name=name)
    for name in ('eri_tol', 'rr_eig_cutoff', 'denominator_tolerance'):
        _nonnegative_float(getattr(args, name), name=name)
    direct_scf_tol = _positive_float(args.direct_scf_tol, name='direct_scf_tol')
    if direct_scf_tol > 1.0:
        raise ValueError('direct_scf_tol must lie in (0, 1]')
    for name in (
        'gint_group_size',
        'gint_max_batch_size',
        'cd_mo_block_size',
        'rr_initial_rank',
        'auxiliary_block_size',
        'virtual_block_size',
        'amplitude_initial_rank',
        'amplitude_max_rank',
        'fit_max_iterations',
        'algorithm_x_block_size',
    ):
        _positive_int(getattr(args, name), name=name)
    for name in ('cd_max_rank', 'denominator_max_rank', 'rr_max_rank', 'rr_solver_maxiter'):
        value = getattr(args, name)
        if value is not None:
            _positive_int(value, name=name)
    _nonnegative_int(args.rr_dense_fallback_dimension, name='rr_dense_fallback_dimension')
    _positive_float(args.rr_solver_tolerance, name='rr_solver_tolerance')
    if args.rr_ritz_residual_tolerance is not None:
        _nonnegative_float(args.rr_ritz_residual_tolerance, name='rr_ritz_residual_tolerance')
    if args.gint_runtime_gate_receipt is not None:
        args.gint_runtime_gate_receipt = (
            args.gint_runtime_gate_receipt.expanduser().resolve()
        )
        if args.gint_column_backend != 'selected':
            raise ValueError(
                'a GINT runtime receipt applies only to the selected backend'
            )
        if args.gint_column_kernel != 'reference':
            raise ValueError(
                'grouped selected-column scheduling cannot consume a release receipt'
            )
        if not args.gint_runtime_gate_receipt.is_file():
            raise ValueError('GINT runtime gate receipt does not exist')
    if args.gint_column_backend != 'selected' and args.gint_column_kernel != 'reference':
        raise ValueError('gint_column_kernel applies only to the selected GINT backend')
    if not args.amplitude_initial_rank <= args.amplitude_max_rank < PAIR_DIMENSION:
        raise ValueError('inexact amplitude ranks must satisfy initial <= max < OV')
    if any(spec.rank >= PAIR_DIMENSION for spec in args.eri_candidate):
        raise ValueError('inexact ERI candidate rank must be smaller than OV')
    _positive_float(args.occupation_change_floor, name='occupation_change_floor')
    _nonnegative_float(args.als_convergence_tolerance, name='als_convergence_tolerance')
    _nonnegative_float(args.fit_ridge, name='fit_ridge')
    _nonnegative_int(args.amplitude_seed, name='amplitude_seed')
    _nonnegative_int(args.eri_seed, name='eri_seed')
    _positive_float(args.orthogonality_cutoff, name='orthogonality_cutoff')
    _nonnegative_float(args.orthogonality_tolerance, name='orthogonality_tolerance')
    validate_scan_order(args.amplitude_fit_tolerances, args.eri_candidate)
    AuditTolerances(
        exact_identity_atol=args.exact_identity_atol,
        candidate_delta_norm=args.candidate_delta_norm,
        candidate_projected_residual_norm=args.candidate_projected_residual_norm,
        amplitude_core_symmetry_atol=args.amplitude_core_symmetry_atol,
    )
    hbm_limit_gib = _positive_float(args.hbm_limit_gib, name='hbm_limit_gib')
    host_rss_limit_gib = _positive_float(args.host_rss_limit_gib, name='host_rss_limit_gib')
    reserve_gib = _nonnegative_float(args.preflight_reserve_gib, name='preflight_reserve_gib')
    ResourceBudget(
        hbm_limit_bytes=int(hbm_limit_gib * GIB),
        host_rss_limit_bytes=int(host_rss_limit_gib * GIB),
        reserve_bytes=int(reserve_gib * GIB),
    )


def run(args: argparse.Namespace) -> int:
    _validate_cli(args)
    started = _utc_now()
    source_start = _git_state()
    base: dict[str, Any] = {
        'schema': SCHEMA,
        'status': 'starting',
        'started_utc': started,
        'case_id': CASE_ID,
        'host': platform.node(),
        'pid': os.getpid(),
        'slurm': {
            name: os.getenv(name)
            for name in (
                'SLURM_JOB_ID',
                'SLURM_JOB_NAME',
                'SLURM_JOB_NODELIST',
                'SLURM_CPUS_PER_TASK',
            )
        },
        'source': source_start,
        'gpu_identity': _pending_gpu_identity(),
        'hbm_observations': _pending_hbm_observations(),
        'controls': {
            key: (
                [item.metadata() for item in value]
                if key == 'eri_candidate'
                else str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        **_copy_flags(),
    }
    _atomic_json(args.output, base, overwrite=args.overwrite)
    hbm_observer: SynchronizedHBMObserver | None = None
    try:
        tolerances = AuditTolerances(
            exact_identity_atol=args.exact_identity_atol,
            candidate_delta_norm=args.candidate_delta_norm,
            candidate_projected_residual_norm=(args.candidate_projected_residual_norm),
            amplitude_core_symmetry_atol=args.amplitude_core_symmetry_atol,
        )
        budget = ResourceBudget(
            hbm_limit_bytes=int(args.hbm_limit_gib * GIB),
            host_rss_limit_bytes=int(args.host_rss_limit_gib * GIB),
            reserve_bytes=int(args.preflight_reserve_gib * GIB),
        )
        import cupy

        gpu_identity = capture_a100_gpu_identity(cupy)
        hbm_observer = SynchronizedHBMObserver(cupy, budget, gpu_identity)
        hbm_observer.sample('process-entry')
        base.update(
            {
                'gpu_identity': gpu_identity,
                'hbm_observations': hbm_observer.metadata(),
            }
        )
        _atomic_json(args.output, base, overwrite=True)

        solver, rr_record = _build_rr_solver(args, source_state=source_start)
        binding = RRLifecycleBinding.capture(solver)
        hbm_observer.sample('rr-lifecycle-converged')
        base.update(
            {
                'rr_run': rr_record,
                'hbm_observations': hbm_observer.metadata(),
            }
        )
        _atomic_json(args.output, base, overwrite=True)
        audit = run_staged_audit(
            binding,
            amplitude_tolerances=args.amplitude_fit_tolerances,
            amplitude_initial_rank=args.amplitude_initial_rank,
            amplitude_max_rank=args.amplitude_max_rank,
            eri_specs=args.eri_candidate,
            tolerances=tolerances,
            budget=budget,
            occupation_change_floor=args.occupation_change_floor,
            fit_max_iterations=args.fit_max_iterations,
            als_convergence_tolerance=args.als_convergence_tolerance,
            ridge=args.fit_ridge,
            amplitude_seed=args.amplitude_seed,
            eri_seed=args.eri_seed,
            orthogonality_cutoff=args.orthogonality_cutoff,
            orthogonality_tolerance=args.orthogonality_tolerance,
            auxiliary_block_size=args.auxiliary_block_size,
            virtual_block_size=args.virtual_block_size,
            algorithm_x_block_size=args.algorithm_x_block_size,
            orbital_fingerprint=rr_record['orbital'].get('orbital_fingerprint'),
            hbm_observer=hbm_observer,
        )
        hbm_observer.sample('process-exit')
        source_end = _git_state()
        source_stability = BENCHMARK._record_source_stability(base)
        if (
            args.gint_runtime_gate_receipt is not None
            and source_stability is not True
        ):
            raise RuntimeError(
                'source snapshot or manifest changed during the receipt-bound audit'
            )
        base.update(
            {
                'status': 'completed',
                'finished_utc': _utc_now(),
                'rr_run': rr_record,
                'audit': audit,
                'hbm_observations': hbm_observer.metadata(),
                'host_rss_peak_bytes': _host_rss_bytes(),
                'source_at_end': source_end,
                'source_stability': source_stability,
                **_copy_flags(),
            }
        )
        _atomic_json(args.output, base, overwrite=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - write the fail-closed result record
        if hbm_observer is not None:
            base['hbm_observations'] = hbm_observer.metadata()
        base.update(
            {
                'status': 'failed',
                'finished_utc': _utc_now(),
                'error_type': type(exc).__name__,
                'error': str(exc),
                'traceback': traceback.format_exc(),
                'host_rss_peak_bytes': _host_rss_bytes(),
                'source_at_end': _git_state(),
                **_copy_flags(),
            }
        )
        _atomic_json(args.output, base, overwrite=True)
        return 2


def main(argv: Iterable[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
