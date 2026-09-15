# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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


'''
Rewrite the pyscf/cc/ccsd.py using cupy, and GPU for ERIs.
This implementation requires that the GPU memory is large enough to hold at
least two t2 tensors.
'''

import time
import ctypes
import contextlib
import functools
import cupy
import numpy as np
from pyscf import gto
from pyscf import lib
from pyscf.ao2mo.outcore import balance_partition
from pyscf.ao2mo import _ao2mo
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf import __config__
from gpu4pyscf.scf import int4c2e
from gpu4pyscf.lib.cupy_helper import load_library
from gpu4pyscf.lib import logger
from gpu4pyscf.cc.device_runtime import (
    WorkspacePlanner,
    canonical_ccsd_residual_update,
    contract_block_lower_ao_to_virtual,
    contract_block_lower_traced_ao_metric,
    contract_traced_ao_metric,
    make_block_lower_ao_layout,
    nvtx_range,
    pack_rccsd_amplitude_difference,
    pack_rccsd_amplitudes,
    unpack_rccsd_amplitudes,
    store_block_lower_ao_block,
)

FREE_CUPY_CACHE = True
RESIDENT_ITERATIONS = True
FINAL_RESIDUAL_BLOCK_TARGET_BYTES = 64 * 1024**2

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

libgint = load_library('libgint')
libgint.GINTfill_int2e.restype = ctypes.c_int

def _flush_cuda_profile_events(mycc):
    """Resolve deferred residual CUDA events after an outer stream sync."""

    pending = getattr(mycc, '_ccsd_pending_cuda_profile_events', None)
    if not pending:
        return
    delattr(mycc, '_ccsd_pending_cuda_profile_events')
    metrics = getattr(mycc, 'run_metrics', None)
    if metrics is None:
        return
    for phase_name, start, stop in pending:
        elapsed_s = float(cupy.cuda.get_elapsed_time(start, stop)) * 1e-3
        metadata = {
            'timing_semantics': 'cuda-event-device-elapsed',
            'synchronization': 'outer-instrumented-phase',
        }
        if hasattr(metrics, 'record_device_phase'):
            metrics.record_device_phase(
                phase_name, elapsed_s, metadata=metadata,
                account_as_child=True)
        elif hasattr(metrics, 'record_phase'):
            metrics.record_phase(phase_name, elapsed_s, metadata=metadata)

def _instrumented_phase(name):
    """Decorate a major CC phase while keeping the default path lightweight."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(mycc, *args, **kwargs):
            metrics = getattr(mycc, 'run_metrics', None)
            if metrics is None or not hasattr(metrics, 'phase'):
                return fn(mycc, *args, **kwargs)
            resident = getattr(mycc, '_ccsd_resident_lifecycle', False)
            metadata = ({
                'timing_semantics': 'cuda-synchronized-wall'
            } if resident else None)
            with metrics.phase(name, metadata=metadata), nvtx_range(name):
                result = fn(mycc, *args, **kwargs)
                if resident:
                    # CPU wall timers otherwise stop at asynchronous kernel
                    # submission and under-report the profiled device phase.
                    cupy.cuda.get_current_stream().synchronize()
                    _flush_cuda_profile_events(mycc)
                return result
        return wrapped
    return decorator

@contextlib.contextmanager
def _profile_phase(mycc, name):
    """Record a phase with explicit asynchronous-device timing semantics."""
    metrics = getattr(mycc, 'run_metrics', None)
    if metrics is None or not hasattr(metrics, 'phase'):
        yield
        return
    resident = getattr(mycc, '_ccsd_resident_lifecycle', False)
    if resident and name.startswith('residual_'):
        # Residual terms execute consecutively in the same CUDA stream.  Event
        # pairs preserve their true device durations without forcing nine
        # synchronization barriers into every CCSD iteration.  The enclosing
        # instrumented update performs one stream sync, then resolves them.
        start = cupy.cuda.Event()
        stop = cupy.cuda.Event()
        start.record()
        try:
            with nvtx_range(name):
                yield
        finally:
            stop.record()
            _maybe_record_hbm_checkpoint(
                mycc, f'final-residual-internal-{name}')
            pending = getattr(
                mycc, '_ccsd_pending_cuda_profile_events', None)
            if pending is None:
                pending = []
                mycc._ccsd_pending_cuda_profile_events = pending
            pending.append((name, start, stop))
        return

    metadata = ({
        'timing_semantics': 'cuda-synchronized-wall'
    } if resident else None)
    with metrics.phase(name, metadata=metadata), nvtx_range(name):
        try:
            yield
        finally:
            if resident:
                cupy.cuda.get_current_stream().synchronize()

def _device_array(mycc, array, *, operation, order=None):
    """Convert a host operand once and account for a real host upload."""
    is_device = isinstance(array, cupy.ndarray) or hasattr(array, '__cuda_array_interface__')
    result = cupy.asarray(array, order=order)
    if not is_device:
        metrics = getattr(mycc, 'run_metrics', None)
        if metrics is not None and hasattr(metrics, 'transfers'):
            metrics.transfers.record_h2d(
                np.asarray(array).nbytes, operation=operation)
    return result

def _host_array(mycc, array, *, operation):
    """Copy a device operand to host and account for the explicit download."""
    is_device = isinstance(array, cupy.ndarray) or hasattr(array, '__cuda_array_interface__')
    if not is_device:
        return np.asarray(array)
    result = array.get()
    metrics = getattr(mycc, 'run_metrics', None)
    if metrics is not None and hasattr(metrics, 'transfers'):
        metrics.transfers.record_d2h(array.nbytes, operation=operation)
    return result

def _free_cupy_cache(mycc, phase_name):
    # A resident CCSD lifecycle owns live static operands and phase arenas.
    # Flushing the allocator from a nested residual phase would discard the
    # arena's reusable free blocks and reintroduce per-iteration allocation
    # churn.  The lifecycle performs one explicit cleanup after the final
    # device-to-host result boundary.
    if getattr(mycc, '_ccsd_resident_lifecycle', False):
        return
    if not getattr(mycc, 'free_cupy_cache', FREE_CUPY_CACHE):
        return
    planner = getattr(mycc, '_ccsd_workspace_planner', None)
    if planner is not None:
        # A caller may toggle ``free_cupy_cache`` between runs.  In that case
        # release explicitly retained arena arrays before flushing the pool.
        planner.release()
        delattr(mycc, '_ccsd_workspace_planner')
    metrics = getattr(mycc, 'run_metrics', None)
    if metrics is None or not hasattr(metrics, 'phase'):
        cupy.get_default_memory_pool().free_all_blocks()
        return
    with metrics.phase(phase_name), nvtx_range(phase_name):
        cupy.get_default_memory_pool().free_all_blocks()

def _workspace_array(mycc, phase, name, shape, *, dtype=np.float64, zero=False):
    """Allocate one CuPy work array and reuse it only when cache retention is on."""
    retain = (
        getattr(mycc, '_ccsd_resident_lifecycle', False)
        or not getattr(mycc, 'free_cupy_cache', FREE_CUPY_CACHE)
    )
    if not retain:
        array = cupy.empty(shape, dtype=dtype)
    else:
        planner = getattr(mycc, '_ccsd_workspace_planner', None)
        if planner is None:
            planner = WorkspacePlanner(
                device='gpu', array_module=cupy,
                metrics=getattr(mycc, 'run_metrics', None),
            )
            mycc._ccsd_workspace_planner = planner
        array = planner.get(phase, name, shape, dtype=dtype)
    if zero:
        array.fill(0)
    return array


def _allocate_direct_resident_staging(
        mycc, packed_staging_shape, staging_shape):
    """Allocate both direct-CCSD staging arrays with one safe OOM retry.

    CuPy's pool can retain enough fragmented free blocks that the first
    simultaneous allocation of these two large arrays fails even though the
    same shapes fit after the unused blocks are returned to CUDA.  Release
    only this phase's references before flushing the pool so that immutable
    resident ERIs and workspaces owned by other phases remain live.

    Returns ``(wVVoo, wVvoO, True)`` on the device path and
    ``(None, None, False)`` only after both allocation attempts fail.  The
    caller retains the historical NumPy fallback for genuinely small GPUs.
    """
    phase = 'direct-resident-staging'
    metrics = getattr(mycc, 'run_metrics', None)
    oom_errors = (MemoryError, cupy.cuda.memory.OutOfMemoryError)

    for attempt in range(2):
        wVVoo = wVvoO = None
        try:
            # Allocate both arrays before zeroing either one.  This makes a
            # successful attempt an all-or-nothing phase reservation.
            wVVoo = _workspace_array(
                mycc, phase, 'wVVoo-packed', packed_staging_shape)
            wVvoO = _workspace_array(
                mycc, phase, 'wVvoO', staging_shape)
            wVVoo.fill(0)
            wVvoO.fill(0)
            return wVVoo, wVvoO, True
        except oom_errors:
            # Drop local references before releasing the planner's phase and
            # asking CuPy to return only unused blocks to CUDA.
            wVVoo = wVvoO = None
            planner = getattr(mycc, '_ccsd_workspace_planner', None)
            if planner is not None:
                planner.release(phase)
            cupy.get_default_memory_pool().free_all_blocks()

            if attempt == 0:
                if metrics is not None and hasattr(metrics, 'increment'):
                    metrics.increment(
                        'direct_resident_staging_allocation_retries')
                continue

            if metrics is not None and hasattr(metrics, 'increment'):
                metrics.increment('direct_host_staging_fallbacks')
            return None, None, False

    raise AssertionError('unreachable direct resident staging allocation path')


def _release_device_workspaces(mycc):
    planner = getattr(mycc, '_ccsd_workspace_planner', None)
    if planner is not None:
        planner.release()
        delattr(mycc, '_ccsd_workspace_planner')


def _release_resident_state(mycc):
    """Release lifecycle-only device and pinned-host references."""
    for name in (
            '_ccsd_resident_eri_cache', '_ccsd_pinned_output_cache',
            '_ccsd_pending_cuda_profile_events'):
        if hasattr(mycc, name):
            delattr(mycc, name)
    _release_device_workspaces(mycc)


@contextlib.contextmanager
def _resident_post_hf_state_scope(mycc):
    """Retain canonical resident state through one timed diagnostic.

    The normal public ``kernel`` remains self-cleaning.  The benchmark needs a
    slightly wider ownership boundary because its mandatory final equation
    residual is another call to ``update_amps`` on the just-converged
    amplitudes.  Holding this scope around ``kernel`` and that diagnostic lets
    the latter reuse the exact ERI cache, direct-integral optimizer, and GPU
    staging workspace built by the iteration loop.  The lifecycle flag itself
    is still disabled between the two calls and is restored only for the
    diagnostic update.

    Unsupported solvers yield ``False`` and retain their historical lifecycle.
    The outermost scope owns cleanup, including exceptional exits.
    """

    supported, _reason = _resident_lifecycle_supported(mycc)
    if not supported:
        yield False
        return

    previous_depth = int(getattr(
        mycc, '_ccsd_resident_state_scope_depth', 0))
    mycc._ccsd_resident_state_scope_depth = previous_depth + 1
    try:
        yield True
    finally:
        if previous_depth:
            mycc._ccsd_resident_state_scope_depth = previous_depth
        else:
            if hasattr(mycc, '_ccsd_resident_state_scope_depth'):
                delattr(mycc, '_ccsd_resident_state_scope_depth')
            _release_resident_state(mycc)
            _free_cupy_cache(mycc, 'resident_post_hf_scope_pool_flush')


def _resident_final_residual_operands(mycc, eris):
    """Build the timed final residual operands in the GPU namespace.

    ``None`` tells callers to use the historical host compatibility path.
    Inside :func:`_resident_post_hf_state_scope`, the converged device
    amplitudes and orbital energies are still live.  The diagnostic update
    therefore reuses them, the exact ERI cache, direct-integral optimizer, and
    staging workspace without any dense tensor transfer.  Only the residual
    norm scalars cross to the host later through
    :func:`_resident_residual_scalar_to_float`.
    """

    cache = getattr(mycc, '_ccsd_resident_eri_cache', None)
    retained = (
        int(getattr(mycc, '_ccsd_resident_state_scope_depth', 0)) > 0
        and cache is not None
        and cache.get('eris') is eris
        and cache.get('final_t1') is not None
        and cache.get('final_t2') is not None
    )
    if not retained:
        return None

    previous_lifecycle = getattr(mycc, '_ccsd_resident_lifecycle', False)
    previous_hbm_audit = getattr(mycc, '_ccsd_hbm_audit_active', False)
    mycc._ccsd_resident_lifecycle = True
    try:
        t1_device = cache['final_t1']
        t2_device = cache['final_t2']
        metrics = getattr(mycc, 'run_metrics', None)
        checkpoint_start = 0
        if metrics is not None and hasattr(metrics, 'metadata'):
            checkpoint_start = len(metrics.metadata.get(
                'synchronized_hbm_checkpoints', []))
        mycc._ccsd_hbm_audit_active = True
        try:
            check_t1, check_t2 = mycc.update_amps(
                t1_device, t2_device, eris)
        finally:
            if previous_hbm_audit:
                mycc._ccsd_hbm_audit_active = previous_hbm_audit
            elif hasattr(mycc, '_ccsd_hbm_audit_active'):
                delattr(mycc, '_ccsd_hbm_audit_active')
        update_hbm = _record_synchronized_hbm_checkpoint(
            mycc, 'final-residual-update-live')
        planner = getattr(mycc, '_ccsd_workspace_planner', None)
        released_staging_bytes = 0
        if planner is not None:
            released_staging_bytes = sum(
                int(spec.nbytes) for spec in planner.specs()
                if spec.phase == 'direct-resident-staging'
            )
            planner.release('direct-resident-staging')
        # The final update is complete, so cached direct-transform objects are
        # no longer needed.  Drop every such reference before returning unused
        # pool blocks to CUDA for the bounded norm reduction below.
        for name in tuple(cache):
            if name.startswith('direct_') or name.startswith('block_lower_'):
                cache.pop(name, None)
        cupy.get_default_memory_pool().free_all_blocks()

        if metrics is not None and hasattr(metrics, 'increment'):
            metrics.increment('resident_final_residual_updates')
            metrics.increment(
                'final_residual_released_staging_bytes',
                released_staging_bytes)
            metrics.increment('final_residual_staging_release_events')
        nocc, nvir = t1_device.shape
        block_size, temporary_bound_bytes = _final_residual_block_plan(
            nocc,
            nvir,
            t2_device.dtype.itemsize,
            target_bytes=FINAL_RESIDUAL_BLOCK_TARGET_BYTES,
        )
        if metrics is not None and hasattr(metrics, 'increment'):
            metrics.increment(
                'final_residual_temporary_bound_bytes',
                temporary_bound_bytes)
        if metrics is not None and hasattr(metrics, 'metadata'):
            metrics.metadata.update({
                'final_residual_backend': 'gpu-blockwise-fp64',
                'final_residual_doubles_block_size': block_size,
                'final_residual_temporary_bound_bytes': (
                    temporary_bound_bytes),
                'final_residual_released_staging_bytes': (
                    released_staging_bytes),
            })
            recorded_checkpoints = metrics.metadata.get(
                'synchronized_hbm_checkpoints', [])
            update_hbm_checkpoints = list(
                recorded_checkpoints[checkpoint_start:])
            if not update_hbm_checkpoints:
                update_hbm_checkpoints = [update_hbm]
        else:
            update_hbm_checkpoints = [update_hbm]
        return {
            'current_t1': t1_device,
            'current_t2': t2_device,
            'jacobi_t1': check_t1,
            'jacobi_t2': check_t2,
            'occupied_energies': cache['mo_energy'][:nocc],
            'virtual_energies': cache['mo_energy'][nocc:nocc + nvir],
            'doubles_block_size': block_size,
            'temporary_bound_bytes': temporary_bound_bytes,
            'released_staging_bytes': released_staging_bytes,
            'update_hbm_checkpoint': update_hbm,
            'update_hbm_checkpoints': update_hbm_checkpoints,
        }
    finally:
        if previous_lifecycle:
            mycc._ccsd_resident_lifecycle = previous_lifecycle
        elif hasattr(mycc, '_ccsd_resident_lifecycle'):
            delattr(mycc, '_ccsd_resident_lifecycle')


def _resident_residual_scalar_to_float(mycc, value, *, operation):
    """Download one GPU residual norm with an auditable ledger label."""

    return _device_scalar_to_float(
        mycc, value, operation=f'final-residual-{operation}-download')


def _record_synchronized_hbm_checkpoint(mycc, name):
    """Record driver and CuPy HBM use at an explicit synchronized boundary."""

    cupy.cuda.get_current_stream().synchronize()
    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    pool = cupy.get_default_memory_pool()
    checkpoint = {
        'name': str(name),
        'driver_used_bytes': int(total_bytes) - int(free_bytes),
        'driver_total_bytes': int(total_bytes),
        'cupy_pool_used_bytes': int(pool.used_bytes()),
        'cupy_pool_total_bytes': int(pool.total_bytes()),
        'timing_semantics': 'cuda-synchronized-instantaneous',
        'measurement_scope': 'exclusive-slurm-device-global',
        'source': 'cudaMemGetInfo-and-cupy-default-memory-pool',
    }
    metrics = getattr(mycc, 'run_metrics', None)
    if metrics is not None and hasattr(metrics, 'metadata'):
        checkpoints = metrics.metadata.setdefault(
            'synchronized_hbm_checkpoints', [])
        checkpoints.append(checkpoint)
    if metrics is not None and hasattr(metrics, 'counters'):
        for key, value in (
                ('synchronized_driver_hbm_peak_bytes',
                 checkpoint['driver_used_bytes']),
                ('synchronized_cupy_pool_used_peak_bytes',
                 checkpoint['cupy_pool_used_bytes']),
                ('synchronized_cupy_pool_total_peak_bytes',
                 checkpoint['cupy_pool_total_bytes'])):
            metrics.counters[key] = max(
                float(value), float(metrics.counters.get(key, 0.0)))
    return checkpoint


def _maybe_record_hbm_checkpoint(mycc, name):
    """Synchronously sample HBM only for the one audited final update."""

    if not getattr(mycc, '_ccsd_hbm_audit_active', False):
        return None
    return _record_synchronized_hbm_checkpoint(mycc, name)


def _record_resident_update_peak(mycc):
    """Capture the known direct-update live-set peak for every timed cycle."""

    metrics = getattr(mycc, 'run_metrics', None)
    if (
        not getattr(mycc, '_ccsd_resident_lifecycle', False)
        or metrics is None
        or not hasattr(metrics, 'metadata')
        or not hasattr(metrics, 'counters')
    ):
        return None
    checkpoint = _record_synchronized_hbm_checkpoint(
        mycc, 'resident-update-direct-output-live')
    if hasattr(metrics, 'increment'):
        metrics.increment('resident_update_hbm_checkpoint_count')
    return checkpoint


def _final_residual_block_plan(
        nocc, nvir, itemsize, *, target_bytes=FINAL_RESIDUAL_BLOCK_TARGET_BYTES):
    """Bound the three live doubles-slice temporaries in residual reduction."""

    nocc = int(nocc)
    nvir = int(nvir)
    itemsize = int(itemsize)
    target_bytes = int(target_bytes)
    if min(nocc, nvir, itemsize, target_bytes) < 1:
        raise ValueError('final residual block dimensions must be positive')
    bytes_per_virtual = nocc * nocc * nvir * itemsize
    block_size = max(
        1, min(nvir, target_bytes // max(1, bytes_per_virtual)))
    return block_size, 3 * block_size * bytes_per_virtual


def _reset_with_device_workspaces(mycc, mol=None):
    _release_resident_state(mycc)
    return ccsd.CCSDBase.reset(mycc, mol)

def _run_diis_instrumented(mycc, *args, **kwargs):
    """Expose the inherited CPU DIIS step in both timing and NVTX ledgers."""
    metrics = getattr(mycc, 'run_metrics', None)
    if metrics is None or not hasattr(metrics, 'phase'):
        return ccsd.CCSDBase.run_diis(mycc, *args, **kwargs)
    with metrics.phase('diis'), nvtx_range('diis'):
        return ccsd.CCSDBase.run_diis(mycc, *args, **kwargs)


def _prepare_resident_eris(mycc, eris):
    """Upload immutable ERI operands once for the resident iteration loop."""

    existing = getattr(mycc, '_ccsd_resident_eri_cache', None)
    if existing is not None and existing.get('eris') is eris:
        return existing
    if existing is not None:
        delattr(mycc, '_ccsd_resident_eri_cache')

    cache = {
        'eris': eris,
        'fock': _device_array(
            mycc, eris.fock, operation='resident-static-fock-upload'),
        'mo_energy': _device_array(
            mycc, eris.mo_energy,
            operation='resident-static-mo-energy-upload'),
        'mo_coeff': _device_array(
            mycc, eris.mo_coeff,
            operation='resident-static-mo-coeff-upload'),
        'oooo': _device_array(
            mycc, eris.oooo, operation='resident-static-oooo-upload'),
        'ovoo': _device_array(
            mycc, eris.ovoo, operation='resident-static-ovoo-upload'),
        'oovv': _device_array(
            mycc, eris.oovv, operation='resident-static-oovv-upload'),
        'ovvo': _device_array(
            mycc, eris.ovvo, operation='resident-static-ovvo-upload'),
    }
    mycc._ccsd_resident_eri_cache = cache
    metrics = getattr(mycc, 'run_metrics', None)
    if metrics is not None and hasattr(metrics, 'increment'):
        static_bytes = sum(
            int(value.nbytes) for name, value in cache.items()
            if name != 'eris'
        )
        metrics.increment('resident_static_cache_bytes', static_bytes)
    return cache


def _resident_eri_cache(mycc, eris):
    cache = getattr(mycc, '_ccsd_resident_eri_cache', None)
    if cache is None or cache.get('eris') is not eris:
        if not getattr(mycc, '_ccsd_resident_lifecycle', False):
            return None
        cache = _prepare_resident_eris(mycc, eris)
    return cache


def _device_scalar_to_float(mycc, value, *, operation):
    """Cross a labelled eight-byte scalar boundary for logging/control flow."""

    host = _host_array(mycc, cupy.asarray(value), operation=operation)
    return float(np.asarray(host).reshape(()))


def _allocate_pinned_array(shape, dtype):
    """Allocate a NumPy view over CUDA pinned memory."""

    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    owner = cupy.cuda.alloc_pinned_memory(nbytes)
    array = np.ndarray(shape, dtype=dtype, buffer=owner)
    return owner, array


def _download_pinned_vector(mycc, vector, *, operation):
    """Download one DIIS vector into fresh pinned storage.

    A fresh input buffer is intentional.  PySCF DIIS may retain the submitted
    array as an in-core history entry; reusing that storage would silently
    mutate its subspace on the next cycle.
    """

    try:
        _owner, host = _allocate_pinned_array(vector.shape, vector.dtype)
        vector.get(out=host, blocking=True)
    except Exception:
        host = vector.get()
        metrics = getattr(mycc, 'run_metrics', None)
        if metrics is not None and hasattr(metrics, 'increment'):
            metrics.increment('resident_pinned_download_fallbacks')
    metrics = getattr(mycc, 'run_metrics', None)
    if metrics is not None and hasattr(metrics, 'transfers'):
        metrics.transfers.record_d2h(vector.nbytes, operation=operation)
    return host


def _upload_pinned_vector(mycc, vector, *, operation):
    """Stage a CPU DIIS result through a reusable pinned output buffer."""

    source = np.asarray(vector)
    key = (tuple(source.shape), source.dtype.str)
    cache = getattr(mycc, '_ccsd_pinned_output_cache', None)
    if cache is None or cache.get('key') != key:
        try:
            owner, host = _allocate_pinned_array(source.shape, source.dtype)
            cache = {'key': key, 'owner': owner, 'array': host}
        except Exception:
            cache = {'key': key, 'owner': None, 'array': np.empty_like(source)}
            metrics = getattr(mycc, 'run_metrics', None)
            if metrics is not None and hasattr(metrics, 'increment'):
                metrics.increment('resident_pinned_upload_fallbacks')
        mycc._ccsd_pinned_output_cache = cache
    np.copyto(cache['array'], source)
    result = cupy.asarray(cache['array'])
    metrics = getattr(mycc, 'run_metrics', None)
    if metrics is not None and hasattr(metrics, 'transfers'):
        metrics.transfers.record_h2d(source.nbytes, operation=operation)
    return result

@_instrumented_phase('ccsd_update')
def update_amps(mycc, t1, t2, eris):
    """Build one canonical RCCSD update in a single GPU namespace.

    A direct public call retains the historical NumPy return type.  The private
    resident lifecycle sets ``_ccsd_resident_lifecycle`` and receives CuPy
    amplitudes, allowing one update to feed the next without a tensor-sized
    host round trip.
    """

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, _nvir = t1.shape

    t1 = _device_array(mycc, t1, operation='iteration-t1-upload')
    t2 = _device_array(mycc, t2, operation='iteration-t2-upload')

    wpq, t1new, t2new, wVOov, wVooV = _direct_ovvv_vvvv(mycc, t1, t2)

    # The resident driver uploads immutable ERI operands before the loop.
    # Standalone update_amps calls still use explicit one-call boundaries.
    cache = _resident_eri_cache(mycc, eris)
    if cache is None:
        fock = _device_array(
            mycc, eris.fock, operation='iteration-fock-upload')
        mo_energy = _device_array(
            mycc, eris.mo_energy, operation='iteration-mo-energy-upload')
        mo_coeff = _device_array(
            mycc, eris.mo_coeff, operation='iteration-mo-coeff-upload')
        oooo = _device_array(
            mycc, eris.oooo, operation='iteration-oooo-upload')
        ovoo = _device_array(
            mycc, eris.ovoo, operation='iteration-ovoo-upload')
        oovv = _device_array(
            mycc, eris.oovv, operation='iteration-oovv-upload')
        ovvo = _device_array(
            mycc, eris.ovvo, operation='iteration-ovvo-upload')
    else:
        fock = cache['fock']
        mo_energy = cache['mo_energy']
        mo_coeff = cache['mo_coeff']
        oooo = cache['oooo']
        ovoo = cache['ovoo']
        oovv = cache['oovv']
        ovvo = cache['ovvo']
    mo_e_o = mo_energy[:nocc]
    mo_e_v = mo_energy[nocc:] + mycc.level_shift
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]

    t1new, t2new = canonical_ccsd_residual_update(
        t1,
        t2,
        fock=fock,
        mo_e_o=mo_e_o,
        mo_e_v=mo_e_v,
        orbo=orbo,
        orbv=orbv,
        wpq=wpq,
        t1new=t1new,
        t2new=t2new,
        wVOov=wVOov,
        wVooV=wVooV,
        oooo=oooo,
        ovoo=ovoo,
        oovv=oovv,
        ovvo=ovvo,
        array_module=cupy,
        phase=lambda name: _profile_phase(mycc, name),
    )

    if getattr(mycc, '_ccsd_resident_lifecycle', False):
        time0 = log.timer_debug1('update t1 t2', *time0)
        return t1new, t2new

    # Compatibility boundary for direct public update_amps calls.
    t1_cpu = _host_array(
        mycc, t1new, operation='iteration-t1-result-download')
    t2_cpu = _host_array(
        mycc, t2new, operation='iteration-t2-result-download')

    time0 = log.timer_debug1('update t1 t2', *time0)
    return t1_cpu, t2_cpu

def _active_mo_coeff(mycc, nocc, nvir):
    """Return the CC active MO columns and verify amplitude dimensions."""

    frozen_mask = np.asarray(mycc.get_frozen_mask(), dtype=bool)
    mo_coeff = mycc.mo_coeff
    if frozen_mask.ndim != 1 or frozen_mask.size != mo_coeff.shape[1]:
        raise ValueError(
            'get_frozen_mask() does not match the CCSD MO coefficient dimensions')
    active_mo_coeff = mo_coeff[:, frozen_mask]
    if active_mo_coeff.shape[1] != nocc + nvir:
        raise ValueError(
            'active MO count does not match the amplitude dimensions: '
            f'{active_mo_coeff.shape[1]} != {nocc} + {nvir}')
    return active_mo_coeff


@_instrumented_phase('direct_vvvv_ovvv')
# Corresponds to the _add_vvvv_tril function in pyscf.cc.ccsd
def _direct_ovvv_vvvv(mycc, t1, t2):
    nocc, nvir = t1.shape
    resident_cache = getattr(mycc, '_ccsd_resident_eri_cache', None)
    if resident_cache is None:
        active_mo_coeff = _active_mo_coeff(mycc, nocc, nvir)
        active_mo_coeff = _device_array(
            mycc, active_mo_coeff,
            operation='direct-active-mo-coeff-upload')
    else:
        active_mo_coeff = resident_cache['mo_coeff']
        if active_mo_coeff.shape[1] != nocc + nvir:
            raise ValueError(
                'resident active MO count does not match amplitude dimensions')
    nocc2 = nocc*(nocc+1)//2
    nao_cart = mycc.mol.nao_nr(cart=True)
    Ht2_mem = nocc2*nao_cart**2 * 8 * 2  # x2 and Ht2
    if resident_cache is not None and 'direct_blksize' in resident_cache:
        # Free-memory readings include CuPy's retained arena blocks and would
        # shrink on every cycle.  Reuse the first-cycle memory plan instead of
        # lowering the pool limit beneath live resident allocations.
        blksize = resident_cache['direct_blksize']
        mem_avail = resident_cache['direct_mem_avail']
    else:
        max_memory = max(MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
        blksize = ((max_memory*.9e6-t2.size*4*8)/8/nao_cart**2/3.5)**.5
        mem_avail = int(cupy.cuda.runtime.memGetInfo()[0] * .75)
        if mem_avail * .9 < Ht2_mem:
            raise RuntimeError(
                f'Not enough GPU memory. Available {mem_avail*1e-6} MB, '
                f'required {Ht2_mem/.9e-6} MB')
        # Reserve some memory for ERIs?
        pool = cupy.get_default_memory_pool()
        pool_limit = mem_avail
        if resident_cache is not None:
            # ``mem_avail`` is an additional-workspace budget computed after
            # static ERIs and amplitudes became live in the pool.
            pool_limit += int(pool.used_bytes())
        pool.set_limit(pool_limit)
        blksize = max(BLKMIN, int(min(
            (nao_cart+3)/4,
            blksize,
            ((mem_avail-Ht2_mem)*.5/8/nao_cart**2)**.5,
        )))
        if resident_cache is not None:
            resident_cache['direct_blksize'] = blksize
            resident_cache['direct_mem_avail'] = mem_avail
    logger.debug1(mycc, 'blksize %d nao %d', blksize, nao_cart)

    if resident_cache is not None and 'direct_vhfopt' in resident_cache:
        vhfopt = resident_cache['direct_vhfopt']
        blksize = resident_cache['direct_blksize']
    else:
        vhfopt = int4c2e._VHFOpt(mycc.mol, 'int2e')
        vhfopt.build(group_size=blksize, diag_block_with_triu=True)
        if resident_cache is not None:
            resident_cache['direct_vhfopt'] = vhfopt
            metrics = getattr(mycc, 'run_metrics', None)
            if metrics is not None and hasattr(metrics, 'increment'):
                metrics.increment('resident_direct_vhfopt_builds')
    mol = vhfopt.mol

    _einsum = cupy.einsum

    # Frozen orbitals must never enter the direct vvvv/ovvv transform.  This is
    # essential for FNO-CCSD: slicing the first ``nocc+nvir`` columns of the
    # full coefficient matrix would reintroduce discarded virtual orbitals and
    # silently mismatch the active-space amplitudes.
    if resident_cache is not None and 'direct_orbo' in resident_cache:
        orbo = resident_cache['direct_orbo']
        orbv = resident_cache['direct_orbv']
        mo = resident_cache['direct_mo']
    else:
        mo = vhfopt.coeff.dot(active_mo_coeff)
        orbo = cupy.asarray(mo[:,:nocc])
        orbv = cupy.asarray(mo[:,nocc:])
        if resident_cache is not None:
            resident_cache['direct_mo'] = mo
            resident_cache['direct_orbo'] = orbo
            resident_cache['direct_orbv'] = orbv
    t1po = orbv.dot(t1.T)
    tau = make_tau_tril(t1, t2)
    x2 = _einsum('xab,pa->xpb', tau, orbv)
    x2 = _einsum('xpb,qb->xpq', x2, orbv)
    tau = None

    nao, nmo = mo.shape
    ao_loc = mol.ao_loc
    nao2 = nao * nao

    x2 = cupy.asarray(x2, order='C')
    Ht2ao = cupy.zeros_like(x2)
    _dgemm = cupy.cuda.cublas.dgemm
    handle = cupy.cuda.device.get_cublas_handle()
    N = cupy.cuda.cublas.CUBLAS_OP_N
    T = cupy.cuda.cublas.CUBLAS_OP_T
    one = np.ones(1)
    one_ptr = one.ctypes.data
    x2_ptr = np.int64(x2.data.ptr)
    Ht2ao_ptr = np.int64(Ht2ao.data.ptr)
    def contract_vvvv_(eri, i0, i1, j0, j1):
        ic = i1 - i0
        jc = j1 - j0
        eri = eri.reshape(-1,jc*nao)
        #:Ht2[:,j0:j1] += np.einsum('xef,efab->xab', x2[:,i0:i1], eri)
        _dgemm(handle, N, N, jc*nao, nocc2, ic*nao,
               one_ptr, eri.data.ptr, jc*nao, x2_ptr+i0*nao*8, nao2,
               one_ptr, Ht2ao_ptr+j0*nao*8, nao2)

        if i0 > j0:
            #:Ht2[:,i0:i1] += np.einsum('xef,abef->xab', x2[:,j0:j1], eri)
            _dgemm(handle, T, N, ic*nao, nocc2, jc*nao,
                   one_ptr, eri.data.ptr, jc*nao, x2_ptr+j0*nao*8, nao2,
                   one_ptr, Ht2ao_ptr+i0*nao*8, nao2)

    l_ctr_offsets = vhfopt.l_ctr_offsets
    log_qs = vhfopt.log_qs
    cp_idx, cp_jdx = np.tril_indices(len(vhfopt.uniq_l_ctr))

    if vhfopt.uniq_l_ctr[:,0].max() <= int4c2e.LMAX_ON_GPU:
        # Computing ERIs on GPU
        idx, idy = cupy.tril_indices(nao)
        #eribuf = cupy.empty(blksize**2*nao**2)
        def fint(ish0, ish1, jsh0, jsh1, group_id):
            i0, i1 = ao_loc[ish0], ao_loc[ish1]
            j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
            #eri = cupy.ndarray((i1-i0, nao, j1-j0, nao), memptr=eribuf.data)
            #eri.fill(0.)
            eri = cupy.zeros([i1-i0,nao,j1-j0,nao])

            # strides to ensure data order consistent with eri(k1-k0,nao,l1-l0,nao)
            strides = [1, (j1-j0)*nao, (j1-j0)*nao**2, nao]
            ao_offsets = [0, 0, i0, j0]
            _fill_eri_block(eri, strides, ao_offsets, vhfopt, group_id)
            # Fill lower triangular part
            eri[:,idx,:,idy] = eri[:,idy,:,idx]
            return eri
    else:
        intor = mol._add_suffix('int2e')
        ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                 'CVHFsetnr_direct_scf')
        eribuf = np.empty((blksize,blksize,nao,nao))
        loadbuf = np.empty((blksize,blksize,nao,nao))
        def fint(ish0, ish1, jsh0, jsh1, group_id):
            if ish0 != jsh0:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                eri = gto.moleintor.getints4c(
                    intor, mol._atm, mol._bas, mol._env,
                    shls_slice=(ish0,ish1,jsh0,jsh1), aosym='s2kl',
                    ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                aoblk = np.ndarray((i1-i0,nao,j1-j0,nao), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(aoblk.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, j0, j1),
                                       ctypes.c_int(nao))
            else:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                eri = gto.moleintor.getints4c(
                    intor, mol._atm, mol._bas, mol._env,
                    shls_slice=(ish0,ish1,ish0,ish1), aosym='s4',
                    ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                eri = lib.unpack_tril(eri, axis=0)
                aoblk = np.ndarray((i1-i0,nao,i1-i0,nao), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(aoblk.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, i0, i1),
                                       ctypes.c_int(nao))
            return _device_array(
                mycc, aoblk, operation='direct-cpu-eri-block-upload')

    # Keep both AO/occupied intermediates on the GPU when memory permits.  The
    # A100 contract has room for these two fixed-shape arrays and this removes
    # every per-shell contraction download.  Smaller devices retain the former
    # host-staged path as a numerical compatibility fallback.
    staging_shape = (nao, nao, nocc, nocc)
    packed_layout = None
    if resident_cache is not None:
        packed_layout = resident_cache.get('direct_wVVoo_block_lower_layout')
        if packed_layout is None:
            shell_blocks = []
            for cpi, cpj in zip(cp_idx, cp_jdx):
                shell_i0 = l_ctr_offsets[cpi]
                shell_j0 = l_ctr_offsets[cpj]
                shell_i1 = l_ctr_offsets[cpi + 1]
                shell_j1 = l_ctr_offsets[cpj + 1]
                shell_blocks.append((
                    int(ao_loc[shell_i0]), int(ao_loc[shell_i1]),
                    int(ao_loc[shell_j0]), int(ao_loc[shell_j1]),
                ))
            host_layout = make_block_lower_ao_layout(nao, shell_blocks)
            packed_layout = dict(host_layout)
            for key in (
                    'row_indices', 'column_indices',
                    'mirror_source_indices', 'mirror_row_indices',
                    'mirror_column_indices'):
                packed_layout[key] = _device_array(
                    mycc, host_layout[key],
                    operation=f'resident-packed-layout-{key}-upload')
            resident_cache['direct_wVVoo_block_lower_layout'] = packed_layout
    packed_staging_shape = (
        int(packed_layout['stored_pair_count']), nocc, nocc
    ) if packed_layout is not None else None
    # Preserve the historical host-staging memory lifecycle for direct public
    # calls and the canonical_legacy benchmark.  Only the explicit resident
    # driver may retain these intermediates in HBM.
    staging_on_device = resident_cache is not None
    block_lower_staging = staging_on_device
    if staging_on_device:
        wVVoo, wVvoO, staging_on_device = (
            _allocate_direct_resident_staging(
                mycc, packed_staging_shape, staging_shape))
        block_lower_staging = staging_on_device
    if not staging_on_device:
        wVVoo = np.zeros(staging_shape)
        wVvoO = np.zeros(staging_shape)

    metrics = getattr(mycc, 'run_metrics', None)
    if metrics is not None and hasattr(metrics, 'increment'):
        if staging_on_device:
            metrics.increment('direct_resident_staging_iterations')
        if block_lower_staging:
            metrics.increment('direct_resident_block_lower_staging_iterations')
            if not resident_cache.get('block_lower_wVVoo_bytes_accounted', False):
                dense_bytes = int(np.prod(staging_shape)) * 8
                packed_bytes = int(np.prod(packed_staging_shape)) * 8
                saved_bytes = dense_bytes - packed_bytes
                metrics.increment(
                    'resident_packed_wVVoo_live_bytes_saved',
                    saved_bytes)
                if hasattr(metrics, 'metadata'):
                    metrics.metadata['direct_wVVoo_block_lower_layout'] = {
                        'schema': 'gpu4pyscf.cc.block-lower-ao.v1',
                        'convention': (
                            'C-flattened lower off-diagonal shell blocks; '
                            'full C-flattened diagonal shell blocks; '
                            'off-diagonal reconstruction mirrors AO axes only '
                            'to reproduce the frozen canonical implementation'
                        ),
                        'nao_cart': int(nao),
                        'nocc': int(nocc),
                        'shell_pair_block_count': len(
                            packed_layout['block_offsets']),
                        'stored_ao_pair_count': int(
                            packed_layout['stored_pair_count']),
                        'dense_ao_pair_count': int(nao * nao),
                        'allocated_bytes': packed_bytes,
                        'dense_equivalent_bytes': dense_bytes,
                        'saved_live_bytes': saved_bytes,
                        'virtual_transform_occupied_block_size': 4,
                    }
                resident_cache['block_lower_wVVoo_bytes_accounted'] = True

    #mempool = cupy.get_default_memory_pool()
    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        li = vhfopt.uniq_l_ctr[cpi,0]
        lj = vhfopt.uniq_l_ctr[cpj,0]
        if li > int4c2e.LMAX_ON_GPU or lj > int4c2e.LMAX_ON_GPU or log_q_ij.size == 0:
            continue

        ish0 = l_ctr_offsets[cpi]
        jsh0 = l_ctr_offsets[cpj]
        ish1 = l_ctr_offsets[cpi+1]
        jsh1 = l_ctr_offsets[cpj+1]
        aoblk = fint(ish0, ish1, jsh0, jsh1, cp_ij_id)

        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
        contract_vvvv_(aoblk, i0, i1, j0, j1)

        #:fvv += 2*np.einsum('kc,kcab->ab', t1, eris_ovvv)
        #:fvv -= np.einsum('kc,kbca->ab', t1, eris_ovvv)
        pppo = _einsum('prqs,si->prqi', aoblk, orbo)
        wVvoO_block = _einsum('prqi,pj->qrij', pppo, t1po[i0:i1])
        wVVoo_block = _einsum('prqi,rj->pqij', pppo, t1po)
        if not staging_on_device:
            wVvoO_block = _host_array(
                mycc, wVvoO_block,
                operation='direct-host-staging-wVvoO-block-download')
            wVVoo_block = _host_array(
                mycc, wVVoo_block,
                operation='direct-host-staging-wVVoo-block-download')
        wVvoO[j0:j1] += wVvoO_block
        if block_lower_staging:
            store_block_lower_ao_block(
                wVVoo, wVVoo_block,
                packed_layout['block_offsets'][cp_ij_id],
                array_module=cupy)
        else:
            wVVoo[i0:i1,j0:j1] = wVVoo_block
        pppo = wVvoO_block = wVVoo_block = None

        if ish0 != jsh0:
            if not block_lower_staging:
                wVVoo[j0:j1,i0:i1] = wVVoo[i0:i1,j0:j1].transpose(1,0,2,3)
            #mempool.free_all_blocks()
            tmp = _einsum('prqs,ri->piqs', aoblk, orbo)
            mirror_block = _einsum('piqs,qj->psij', tmp, t1po[j0:j1])
            if not staging_on_device:
                mirror_block = _host_array(
                    mycc, mirror_block,
                    operation='direct-host-staging-wVvoO-mirror-download')
            wVvoO[i0:i1] += mirror_block
            tmp = mirror_block = None

        aoblk = None
    eribuf = loadbuf = x2 = None

    #:t1new += 2*lib.einsum('edac,ikcd->ikea', eris_ovvv, t2)
    #:t1new +=  -lib.einsum('edac,ikdc->ikea', eris_ovvv, t2)
    Ht2full = _unpack_t2_tril(Ht2ao, nocc, nao)
    t1tmp  = _einsum('ijpq,qj->ip', Ht2full, orbo) * 2
    t1tmp -= _einsum('ijqp,qj->ip', Ht2full, orbo)
    t1new = t1tmp.dot(orbv)

    # vvvv-t2 contractions back to MO repr.
    Ht2tril = _einsum('xpq,pa->xaq', Ht2ao, orbv)
    Ht2tril = _einsum('xaq,qb->xab', Ht2tril, orbv)

    # part of ovvv-t2 contractions back to MO repr.
    #: tmp = np.einsum('ijcd,ka,kdcb->ijba', tau, t1, eris.ovvv)
    #: t2new -= tmp + tmp.transpose(1,0,3,2)
    t1pv = orbo.dot(t1)
    tmp = _einsum('xpq,pa->xaq', Ht2ao, orbv)
    Ht2tril -= _einsum('xaq,qb->xab', tmp, t1pv)

    tmp = _einsum('xpq,pa->xaq', Ht2ao, t1pv)
    Ht2tril -= _einsum('xaq,qb->xab', tmp, orbv)#_einsum('xpq,pa,qb->xab', Ht2ao, t1pv, orbv)

    t2new = _unpack_t2_tril(Ht2tril, nocc, nvir)
    _maybe_record_hbm_checkpoint(
        mycc, 'final-residual-internal-direct-transform-live')
    Ht2ao = Ht2full = None

    if staging_on_device:
        c = vhfopt.coeff
        if block_lower_staging:
            wpq = 2 * contract_block_lower_traced_ao_metric(
                wVVoo, c, packed_layout, array_module=cupy)
        else:
            wpq = 2 * contract_traced_ao_metric(
                wVVoo, c, array_module=cupy)
        wpq -= contract_traced_ao_metric(
            wVvoO, c, array_module=cupy, transpose_output=True)

        tmp = _einsum('pqji,qb->pbji', wVvoO, orbv)
        wVOov = _einsum('pbji,pa->bjia', tmp, orbv)
        tmp = None

        if block_lower_staging:
            wVooV = contract_block_lower_ao_to_virtual(
                wVVoo, orbv, packed_layout, occupied_block_size=4,
                array_module=cupy, scale=-1.0)
        else:
            tmp = _einsum('pqji,pa->aqji', wVVoo, -orbv)
            wVooV = _einsum('aqji,qb->bjia', tmp, orbv)
    else:
        c = _host_array(
            mycc, vhfopt.coeff,
            operation='direct-host-staging-coefficient-map-download')
        wpq = 2 * contract_traced_ao_metric(
            wVVoo, c, array_module=np)
        wpq -= contract_traced_ao_metric(
            wVvoO, c, array_module=np, transpose_output=True)
        wpq = _device_array(
            mycc, wpq, operation='direct-host-staging-wpq-upload')

        wVvoO_device = _device_array(
            mycc, wVvoO, operation='direct-host-staging-wVvoO-upload')
        tmp = _einsum('pqji,qb->pbji', wVvoO_device, orbv)
        wVOov = _einsum('pbji,pa->bjia', tmp, orbv)
        tmp = wVvoO_device = None
        if getattr(mycc, 'free_cupy_cache', FREE_CUPY_CACHE):
            cupy.get_default_memory_pool().free_all_blocks()

        wVVoo_device = _device_array(
            mycc, wVVoo, operation='direct-host-staging-wVVoo-upload')
        tmp = _einsum('pqji,pa->aqji', wVVoo_device, -orbv)
        wVooV = _einsum('aqji,qb->bjia', tmp, orbv)
    _record_resident_update_peak(mycc)
    _maybe_record_hbm_checkpoint(
        mycc, 'final-residual-internal-direct-output-live')
    wVVoo = wVvoO = tmp = None

    _free_cupy_cache(mycc, 'direct_vvvv_ovvv_pool_flush')
    return wpq, t1new, t2new, wVOov, wVooV

def make_tau_tril(t1, t2):
    nocc, nvir = t1.shape
    t1 = cupy.asarray(t1)
    tau = cupy.einsum('ia,jb->ijab', t1, t1)
    tau += cupy.asarray(t2)
    return tau[cupy.tril_indices(nocc)]

def _unpack_t2_tril(t2tril, nocc, nvir):
    t2 = cupy.empty((nocc,nocc,nvir,nvir))
    idx,idy = cupy.tril_indices(nocc)
    t2[idy,idx] = t2tril.transpose(0,2,1)
    t2[idx,idy] = t2tril
    return t2

def _fill_eri_block(eri, strides, ao_offsets, vhfopt, group_id):
    log_qs = vhfopt.log_qs
    cp_kl_id = group_id
    log_q_kl = log_qs[cp_kl_id]
    if log_q_kl.size == 0:
        return eri

    cp_idx, cp_jdx = np.tril_indices(len(vhfopt.uniq_l_ctr))
    cpk = cp_idx[cp_kl_id]
    cpl = cp_jdx[cp_kl_id]
    lk = vhfopt.uniq_l_ctr[cpk,0]
    ll = vhfopt.uniq_l_ctr[cpl,0]
    if lk > int4c2e.LMAX_ON_GPU or ll > int4c2e.LMAX_ON_GPU:
        raise NotImplementedError

    stream = cupy.cuda.get_current_stream()
    log_cutoff = np.log(vhfopt.direct_scf_tol)
    omega = 0.

    l_symb = lib.param.ANGULAR
    nao = vhfopt.coeff.shape[0]
    bins_locs_kl = vhfopt.bins[cp_kl_id]
    bins_floor_kl = vhfopt.bins_floor[cp_kl_id]
    nbins_kl = len(bins_locs_kl) - 1

    fn = libgint.GINTfill_int2e
    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        li = vhfopt.uniq_l_ctr[cpi,0]
        lj = vhfopt.uniq_l_ctr[cpj,0]
        if li > int4c2e.LMAX_ON_GPU or lj > int4c2e.LMAX_ON_GPU or log_q_ij.size == 0:
            continue

        t0 = time.perf_counter()
        bins_locs_ij = vhfopt.bins[cp_ij_id]
        bins_floor_ij = vhfopt.bins_floor[cp_ij_id]
        nbins_ij = len(bins_locs_ij) - 1

        err = fn(ctypes.cast(stream.ptr, ctypes.c_void_p), vhfopt.bpcache,
                 ctypes.cast(eri.data.ptr, ctypes.c_void_p), ctypes.c_int(nao),
                 (ctypes.c_int*4)(*strides), (ctypes.c_int*4)(*ao_offsets),
                 bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                 bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                 bins_floor_ij.ctypes.data_as(ctypes.c_void_p),
                 bins_floor_kl.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(nbins_ij), ctypes.c_int(nbins_kl),
                 ctypes.c_int(cp_ij_id), ctypes.c_int(cp_kl_id),
                 ctypes.c_double(log_cutoff),
                 ctypes.c_double(omega))
        if err != 0:
            detail = f'CUDA Error for ({l_symb[li]}{l_symb[lj]}|{l_symb[lk]}{l_symb[ll]})'
            raise RuntimeError(detail)
        logger.debug1(vhfopt.mol, '(%s%s|%s%s) on GPU %.3fs',
                      l_symb[li], l_symb[lj], l_symb[lk], l_symb[ll],
                      time.perf_counter() - t0)
    return eri

@_instrumented_phase('integral_transform')
def _make_eris_incore(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = ccsd._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    # Cupy memory buffer may be created in previous SCF calculations.
    _free_cupy_cache(mycc, 'integral_transform_pool_flush')

    mol = mycc.mol
    mo_coeff = _device_array(
        mycc, eris.mo_coeff, operation='ao2mo-mo-coeff-upload', order='F')
    nocc = eris.nocc
    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc

    nao_cart = mycc.mol.nao_nr(cart=True)
    max_memory = max(MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
    blksize = ((max_memory*.9e6-nocc**2*nao_cart**2*2*8)/8/nao_cart**2/2.5)**.5
    mem_avail = int(cupy.cuda.runtime.memGetInfo()[0] * .75)
    cupy.get_default_memory_pool().set_limit(mem_avail)
    blksize = max(BLKMIN, int(min((nao_cart+3)/4, blksize,
                                  (mem_avail*.5/8/nao_cart**2)**.5)))
    logger.debug1(mycc, 'blksize %d nao %d', blksize, nao_cart)

    vhfopt = int4c2e._VHFOpt(mycc.mol, 'int2e')
    vhfopt.build(group_size=blksize, diag_block_with_triu=True)
    mol = vhfopt.mol
    mo = vhfopt.coeff.dot(mo_coeff)
    orbo = cupy.asarray(mo[:,:nocc])
    orbv = cupy.asarray(mo[:,nocc:])
    ao_loc = mol.ao_loc
    nao = mo.shape[0]

    l_ctr_offsets = vhfopt.l_ctr_offsets
    log_qs = vhfopt.log_qs
    cp_idx, cp_jdx = np.tril_indices(len(vhfopt.uniq_l_ctr))

    ppOO = np.empty((nao,nao,nocc,nocc))
    pPoO = np.zeros((nao,nao,nocc,nocc))
    eribuf = cupy.empty(blksize**2*nao**2)
    #mempool = cupy.get_default_memory_pool()
    idx, idy = cupy.tril_indices(nao)

    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        li = vhfopt.uniq_l_ctr[cpi,0]
        lj = vhfopt.uniq_l_ctr[cpj,0]
        if li > int4c2e.LMAX_ON_GPU or lj > int4c2e.LMAX_ON_GPU or log_q_ij.size == 0:
            continue

        ish0 = l_ctr_offsets[cpi]
        jsh0 = l_ctr_offsets[cpj]
        ish1 = l_ctr_offsets[cpi+1]
        jsh1 = l_ctr_offsets[cpj+1]
        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
        eri = cupy.ndarray((nao, i1-i0, j1-j0, nao), memptr=eribuf.data)
        eri.fill(0.)
        # strides to ensure data order consistent with eri(nao,k1-k0,l1-l0,nao)
        strides = [1, (i1-i0)*(j1-j0)*nao, (j1-j0)*nao, nao]
        ao_offsets = [0, 0, i0, j0]
        _fill_eri_block(eri, strides, ao_offsets, vhfopt, cp_ij_id)
        # Fill lower triangular part
        eri[idx,:,:,idy] = eri[idy,:,:,idx]

        pijo = cupy.dot(eri.reshape(-1,nao), orbo)
        ijoo = cupy.dot(pijo.reshape(nao,-1).T, orbo)
        ppOO[i0:i1,j0:j1] = _host_array(
            mycc, ijoo, operation='ao2mo-ppOO-block-download'
        ).reshape(i1-i0,j1-j0,nocc,nocc)
        ijoo = None

        jopi = cupy.asarray(pijo.reshape(nao*(i1-i0),(j1-j0)*nocc).T, order='C')
        jopo = cupy.dot(jopi.reshape(-1,i1-i0), orbo[i0:i1])
        pPoO[j0:j1] += _host_array(
            mycc, jopo, operation='ao2mo-pPoO-block-download'
        ).reshape(j1-j0,nocc,nao,nocc).transpose(0,2,1,3)
        pijo = jopo = None

        if ish0 != jsh0:
            ppOO[j0:j1,i0:i1] = ppOO[i0:i1,j0:j1].transpose(1,0,2,3)
            opio = cupy.dot(jopi.reshape(j1-j0,-1).T, orbo[j0:j1])
            pPoO[i0:i1] += _host_array(
                mycc, opio, operation='ao2mo-pPoO-mirror-block-download'
            ).reshape(nocc,nao,i1-i0,nocc).transpose(2,1,0,3)
            jopi = opio = None

    ppOO = _device_array(
        mycc, ppOO, operation='ao2mo-ppOO-upload')
    pooo = cupy.dot(ppOO.reshape(nao,-1).T, orbo)
    oooo = cupy.dot(pooo.reshape(nao,-1).T, orbo).reshape(nocc,nocc,nocc,nocc)
    ooov = cupy.dot(pooo.reshape(nao,-1).T, orbv).reshape(nocc,nocc,nocc,nvir)
    eris.oooo = _host_array(
        mycc, oooo, operation='ao2mo-oooo-download')
    eris.ovoo = lib.transpose(
        _host_array(
            mycc, ooov, operation='ao2mo-ooov-download'
        ).reshape(nocc*nocc,nocc*nvir)
    ).reshape(nocc,nvir,nocc,nocc)
    pooo = oooo = ooov = None

    poov = cupy.dot(ppOO.reshape(nao,-1).T, orbv)
    oovv = cupy.dot(poov.reshape(nao,-1).T, orbv).reshape(nocc,nocc,nvir,nvir)
    eris.oovv = _host_array(
        mycc, oovv, operation='ao2mo-oovv-download')
    ppOO = poov = oovv = None

    pPoO = _device_array(
        mycc, pPoO, operation='ao2mo-pPoO-upload')
    poov = cupy.dot(pPoO.reshape(nao,-1).T, orbv)
    voov = cupy.dot(orbv.T, poov.reshape(nao,-1))
    eris.ovvo = lib.transpose(
        _host_array(
            mycc, voov, operation='ao2mo-ovvo-download'
        ).reshape(nvir*nocc,nocc*nvir)
    ).reshape(nocc,nvir,nvir,nocc)
    eris.ovov = eris.ovvo.transpose(0,1,3,2)
    pPoO = poov = voov = None
    log.timer('CCSD integral transformation', *cput0)

    _free_cupy_cache(mycc, 'integral_transform_pool_flush_final')
    return eris


def _resident_energy_device(mycc, t1, t2, eris):
    """Evaluate canonical RCCSD energy without leaving the GPU."""

    cache = _resident_eri_cache(mycc, eris)
    if cache is None:
        raise RuntimeError('resident energy requires a prepared ERI cache')
    nocc, nvir = t1.shape
    fock = cache['fock']
    ovvo = cache['ovvo']
    energy = cupy.einsum('ia,ia', fock[:nocc, nocc:], t1) * 2

    # Bound the transient tau allocation.  The slice size is independent of
    # host RAM and remains deterministic for a fixed molecular case.
    target_bytes = 384 * 1024**2
    bytes_per_virtual = max(
        1, int(nocc) * int(nocc) * int(nvir) * int(t2.dtype.itemsize) * 2)
    block_size = max(1, min(int(nvir), target_bytes // bytes_per_virtual))
    for p0 in range(0, nvir, block_size):
        p1 = min(nvir, p0 + block_size)
        eri_ovvo = ovvo[:, p0:p1]
        tau = t2[:, :, p0:p1] + cupy.einsum(
            'ia,jb->ijab', t1[:, p0:p1], t1)
        energy += 2 * cupy.einsum('ijab,iabj', tau, eri_ovvo)
        energy -= cupy.einsum('jiab,iabj', tau, eri_ovvo)
        tau = None
    return energy.real


def _resident_update_norm(mycc, t1new, t2new, t1old, t2old):
    """Reproduce PySCF's packed-amplitude update norm on the GPU."""

    delta = pack_rccsd_amplitude_difference(
        t1new, t2new, t1old, t2old, array_module=cupy)
    norm_device = cupy.linalg.norm(delta)
    delta = None
    return _device_scalar_to_float(
        mycc, norm_device, operation='iteration-norm-download')


def _resident_diis(mycc, t1, t2, adiis):
    """Run one CPU DIIS extrapolation across two accounted vector transfers."""

    vector_device = pack_rccsd_amplitudes(
        t1, t2, array_module=cupy)
    vector_host = _download_pinned_vector(
        mycc, vector_device, operation='diis-vector-download')
    vector_device = None
    # Preserve PySCF 2.14 CCSD.run_diis semantics exactly: update receives the
    # current amplitude vector with xerr=None.  DIIS then forms its error from
    # this vector and the previous *returned* vector.  The separately packed
    # Jacobi delta is solely the CCSD convergence norm and must not be supplied
    # as an explicit DIIS error vector.
    vector_host = adiis.update(vector_host)
    vector_device = _upload_pinned_vector(
        mycc, vector_host, operation='diis-vector-upload')
    return unpack_rccsd_amplitudes(
        vector_device, t1.shape[0], t1.shape[1], array_module=cupy)


def _resident_lifecycle_supported(mycc, t1=None, t2=None):
    """Return whether canonical semantics can be preserved by this driver."""

    if not getattr(mycc, 'resident_iterations', RESIDENT_ITERATIONS):
        return False, 'disabled'
    # Subclasses can change amplitude representation, energy, or DIIS rules.
    # They retain PySCF's established host lifecycle until they opt into a
    # representation-specific resident driver of their own.
    if mycc.__class__ is not CCSD:
        return False, 'subclass'
    if any(name in mycc.__dict__ for name in (
            'update_amps', 'run_diis', 'amplitudes_to_vector',
            'vector_to_amplitudes', 'energy')):
        return False, 'custom-lifecycle-hook'
    if mycc.callback is not None:
        return False, 'callback'
    if np.iscomplexobj(mycc.mo_coeff):
        return False, 'complex-orbitals'
    if t1 is not None and np.iscomplexobj(t1):
        return False, 'complex-singles'
    if t2 is not None and np.iscomplexobj(t2):
        return False, 'complex-doubles'
    return True, None


def _record_resident_fallback(mycc, reason):
    metrics = getattr(mycc, 'run_metrics', None)
    if metrics is not None and hasattr(metrics, 'increment'):
        metrics.increment('resident_lifecycle_fallbacks')
    if metrics is not None and hasattr(metrics, 'metadata'):
        metrics.metadata['resident_lifecycle_fallback_reason'] = str(reason)


def _resident_iteration_kernel(
        mycc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8,
        tolnormt=1e-6, verbose=None):
    """Canonical PySCF iteration semantics with device-resident amplitudes."""

    log = logger.new_logger(mycc, verbose)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]
    if t1 is None or t2 is None:
        raise ValueError('both t1 and t2 are required after initialisation')
    if np.iscomplexobj(t1) or np.iscomplexobj(t2):
        raise TypeError('the resident canonical loop currently requires real amplitudes')

    name = mycc.__class__.__name__
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    previous_lifecycle = getattr(mycc, '_ccsd_resident_lifecycle', False)
    mycc._ccsd_resident_lifecycle = True
    try:
        _prepare_resident_eris(mycc, eris)
        t1 = _device_array(
            mycc, t1, operation='resident-initial-t1-upload')
        t2 = _device_array(
            mycc, t2, operation='resident-initial-t2-upload')

        with _profile_phase(mycc, 'ccsd_energy'):
            energy_device = _resident_energy_device(mycc, t1, t2, eris)
            eccsd = _device_scalar_to_float(
                mycc, energy_device, operation='initial-energy-download')
        log.info('Init E_corr(%s) = %.15g', name, eccsd)

        if isinstance(mycc.diis, lib.diis.DIIS):
            adiis = mycc.diis
        elif mycc.diis:
            adiis = lib.diis.DIIS(
                mycc, mycc.diis_file, incore=mycc.incore_complete)
            adiis.space = mycc.diis_space
        else:
            adiis = None

        eold = 0.0
        converged = False
        mycc.cycles = 0
        for istep in range(max_cycle):
            t1new, t2new = mycc.update_amps(t1, t2, eris)
            with _profile_phase(mycc, 'update_norm'):
                normt = _resident_update_norm(
                    mycc, t1new, t2new, t1, t2)

            if mycc.iterative_damping < 1.0:
                alpha = float(np.asarray(mycc.iterative_damping))
                t1new *= alpha
                t1new += (1.0 - alpha) * t1
                t2new *= alpha
                t2new += (1.0 - alpha) * t2

            t1, t2 = t1new, t2new
            t1new = t2new = None
            if (adiis is not None
                    and istep >= mycc.diis_start_cycle
                    and abs(eccsd - eold) < mycc.diis_start_energy_diff):
                with _profile_phase(mycc, 'diis'), nvtx_range('diis'):
                    t1, t2 = _resident_diis(mycc, t1, t2, adiis)
                metrics = getattr(mycc, 'run_metrics', None)
                if metrics is not None and hasattr(metrics, 'increment'):
                    metrics.increment('resident_diis_steps')
                logger.debug1(mycc, 'DIIS for step %d', istep)

            eold = eccsd
            with _profile_phase(mycc, 'ccsd_energy'):
                energy_device = _resident_energy_device(mycc, t1, t2, eris)
                eccsd = _device_scalar_to_float(
                    mycc, energy_device,
                    operation='iteration-energy-download')
            mycc.cycles = istep + 1
            metrics = getattr(mycc, 'run_metrics', None)
            if metrics is not None and hasattr(metrics, 'increment'):
                metrics.increment('resident_iterations')
            log.info(
                'cycle = %d  E_corr(%s) = %.15g  dE = %.9g  '
                'norm(t1,t2) = %.6g',
                istep + 1, name, eccsd, eccsd - eold, normt)
            cput1 = log.timer(f'{name} iter', *cput1)
            if abs(eccsd - eold) < tol and normt < tolnormt:
                converged = True
                break

        log.timer(name, *cput0)
        if getattr(mycc, '_ccsd_resident_state_scope_depth', 0):
            cache = _resident_eri_cache(mycc, eris)
            cache['final_t1'] = t1
            cache['final_t2'] = t2
        t1_host = _host_array(
            mycc, t1, operation='final-t1-download')
        t2_host = _host_array(
            mycc, t2, operation='final-t2-download')
        return converged, eccsd, t1_host, t2_host
    finally:
        outer_state_scope = bool(getattr(
            mycc, '_ccsd_resident_state_scope_depth', 0))
        if previous_lifecycle:
            mycc._ccsd_resident_lifecycle = previous_lifecycle
        elif hasattr(mycc, '_ccsd_resident_lifecycle'):
            delattr(mycc, '_ccsd_resident_lifecycle')
        if not outer_state_scope:
            _release_resident_state(mycc)
            _free_cupy_cache(mycc, 'resident_lifecycle_pool_flush')


def _resident_ccsd(mycc, t1=None, t2=None, eris=None):
    """Private CCSD lifecycle selected by the unchanged public ``kernel``."""

    supported, reason = _resident_lifecycle_supported(mycc, t1, t2)
    if not supported:
        _record_resident_fallback(mycc, reason)
        return ccsd.CCSDBase.ccsd(mycc, t1, t2, eris)

    assert mycc.mo_coeff is not None
    assert mycc.mo_occ is not None
    if mycc.verbose >= logger.WARN:
        mycc.check_sanity()
    mycc.dump_flags()
    mycc.e_hf = mycc.get_e_hf()
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if any(np.iscomplexobj(getattr(eris, name)) for name in (
            'fock', 'oooo', 'ovoo', 'oovv', 'ovvo')):
        _record_resident_fallback(mycc, 'complex-eris')
        return ccsd.CCSDBase.ccsd(mycc, t1, t2, eris)

    mycc.converged, mycc.e_corr, mycc.t1, mycc.t2 = \
        _resident_iteration_kernel(
            mycc, eris, t1, t2,
            max_cycle=mycc.max_cycle,
            tol=mycc.conv_tol,
            tolnormt=mycc.conv_tol_normt,
            verbose=mycc.verbose,
        )
    mycc._finalize()
    return mycc.e_corr, mycc.t1, mycc.t2

class CCSDBase(lib.StreamObject):
    # attributes
    # Retain the historical behavior while allowing a run to keep the CuPy
    # pool warm between phases (useful for profiling and repeated iterations).
    free_cupy_cache       = FREE_CUPY_CACHE
    resident_iterations   = RESIDENT_ITERATIONS
    _keys                  = set(ccsd.CCSDBase._keys) | {
        'free_cupy_cache', 'resident_iterations', 'run_metrics'
    }
    max_cycle              = ccsd.CCSDBase.max_cycle
    conv_tol               = ccsd.CCSDBase.conv_tol
    iterative_damping      = ccsd.CCSDBase.iterative_damping
    conv_tol_normt         = ccsd.CCSDBase.conv_tol_normt

    diis                   = ccsd.CCSDBase.diis
    diis_space             = ccsd.CCSDBase.diis_space
    diis_file              = None
    diis_start_cycle       = ccsd.CCSDBase.diis_start_cycle
    diis_start_energy_diff = ccsd.CCSDBase.diis_start_energy_diff

    direct                 = ccsd.CCSDBase.direct
    async_io               = None
    incore_complete        = ccsd.CCSDBase.incore_complete
    cc2                    = ccsd.CCSDBase.cc2
    callback               = None

    # functions
    __init__           = ccsd.CCSDBase.__init__
    ecc                = ccsd.CCSDBase.ecc
    e_tot              = ccsd.CCSDBase.e_tot
    nocc               = ccsd.CCSDBase.nocc
    nmo                = ccsd.CCSDBase.nmo
    reset              = _reset_with_device_workspaces
    get_nocc           = ccsd.CCSDBase.get_nocc
    get_nmo            = ccsd.CCSDBase.get_nmo
    get_frozen_mask    = ccsd.CCSDBase.get_frozen_mask
    get_e_hf           = ccsd.CCSDBase.get_e_hf
    set_frozen         = ccsd.CCSDBase.set_frozen
    dump_flags         = ccsd.CCSDBase.dump_flags
    get_init_guess     = ccsd.CCSDBase.get_init_guess
    init_amps          = ccsd.CCSDBase.init_amps
    energy             = ccsd.CCSDBase.energy
    _add_vvvv          = ccsd.CCSDBase._add_vvvv
    update_amps        = update_amps
    kernel             = ccsd.CCSDBase.kernel
    _finalize          = ccsd.CCSDBase._finalize
    as_scanner         = ccsd.CCSDBase.as_scanner
    restore_from_diis_ = ccsd.CCSDBase.restore_from_diis_

    solve_lambda         = NotImplemented
    ccsd_t               = NotImplemented
    ipccsd               = NotImplemented
    eaccsd               = NotImplemented
    eeccsd               = NotImplemented
    eomee_ccsd_singlet   = NotImplemented
    eomee_ccsd_triplet   = NotImplemented
    eomsf_ccsd           = NotImplemented
    eomip_method         = NotImplemented
    eomea_method         = NotImplemented
    eomee_method         = NotImplemented
    make_rdm1            = NotImplemented
    make_rdm2            = NotImplemented
    ao2mo                = _make_eris_incore
    run_diis             = _run_diis_instrumented
    _resident_post_hf_state_scope = _resident_post_hf_state_scope
    _resident_final_residual_operands = _resident_final_residual_operands
    _resident_residual_scalar_to_float = _resident_residual_scalar_to_float
    _record_synchronized_hbm_checkpoint = _record_synchronized_hbm_checkpoint
    amplitudes_to_vector = ccsd.CCSDBase.amplitudes_to_vector
    vector_to_amplitudes = ccsd.CCSDBase.vector_to_amplitudes
    dump_chk             = None
    density_fit          = NotImplemented
    nuc_grad_method      = NotImplemented

    # to_cpu can be reused only when __init__ still takes mf
    def to_cpu(self):
        mf = self._scf.to_cpu()
        from importlib import import_module
        mod = import_module(self.__module__.replace('gpu4pyscf', 'pyscf'))
        cls = getattr(mod, self.__class__.__name__)
        obj = cls(mf)
        return obj

CCSDBase.ccsd = _resident_ccsd

class CCSD(CCSDBase):
    from gpu4pyscf.lib.utils import to_gpu, device

    def __init__(self, mf, *args, **kwargs):
        if hasattr(mf, 'to_cpu'):
            mf = mf.to_cpu()
        if hasattr(mf, 'with_df') and mf.with_df:
            lib.logger.warn(mf.mol, 'DF-CCSD not available. Run the standard CCSD.')
        ccsd.CCSD.__init__(self, mf, *args, **kwargs)
