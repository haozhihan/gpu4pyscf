"""CPU-only coverage for the optional GPU CC runtime helpers."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).parents[1] / "device_runtime.py"
_SPEC = importlib.util.spec_from_file_location("gpu4pyscf_cc_device_runtime", _MODULE_PATH)
runtime = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = runtime
_SPEC.loader.exec_module(runtime)


def test_metrics_nested_phase_timing_and_json_serialization():
    ticks = iter([0.0, 1.0, 1.25, 2.0])
    metrics = runtime.RunMetrics("ccsd", metadata={"nocc": np.int64(4)}, clock=lambda: next(ticks))
    with metrics.phase("outer", iteration=np.int64(2)):
        with metrics.phase("inner"):
            pass
    metrics.increment("iterations")

    assert metrics.phases[0]["name"] == "inner"
    assert metrics.phases[0]["elapsed_s"] == 0.25
    assert metrics.phases[1]["name"] == "outer"
    assert metrics.phases[1]["elapsed_s"] == 2.0
    assert metrics.phases[1]["exclusive_s"] == 1.75
    payload = metrics.to_dict()
    json.dumps(payload)
    assert payload["phase_totals"]["inner"]["calls"] == 1
    assert payload["counters"]["iterations"] == 1.0


def test_device_phase_has_machine_readable_cuda_event_semantics():
    metrics = runtime.RunMetrics("resident-ccsd")
    metrics.record_device_phase(
        "residual_oooo_ladder",
        0.125,
        metadata={"synchronization": "outer-instrumented-phase"},
    )

    phase = metrics.to_dict()["phases"][0]
    assert phase["elapsed_s"] == 0.125
    assert phase["metadata"] == {
        "synchronization": "outer-instrumented-phase",
        "timing_semantics": "cuda-event-device-elapsed",
    }
    with pytest.raises(ValueError, match="cuda-event-device-elapsed"):
        metrics.record_device_phase(
            "invalid", 0.0,
            metadata={"timing_semantics": "host-enqueue"},
        )


def test_nested_device_phase_is_accounted_in_parent_exclusive_time():
    ticks = iter([0.0, 2.0])
    metrics = runtime.RunMetrics("resident", clock=lambda: next(ticks))
    with metrics.phase("ccsd_update"):
        metrics.record_device_phase(
            "residual_oooo_ladder", 0.75, account_as_child=True
        )
    outer = metrics.phases[-1]
    device = metrics.phases[0]
    assert device["depth"] == 1
    assert outer["elapsed_s"] == 2.0
    assert outer["exclusive_s"] == 1.25


def test_transfer_counter_accounts_bytes_by_kind():
    counter = runtime.TransferCounter()
    counter.record_h2d(128, operation="amplitudes")
    counter.record_h2d(32, count=2, operation="amplitudes")
    counter.record_d2h(64, operation="direct-block")
    counter.record_peer(16)

    result = counter.to_dict()
    assert result["total_bytes"] == 240
    assert result["total_transfers"] == 5
    assert result["by_kind"]["h2d"] == {"bytes": 160, "count": 3}
    assert result["by_operation"]["h2d"]["amplitudes"] == {
        "bytes": 160,
        "count": 3,
    }
    assert result["by_operation"]["d2h"]["direct-block"] == {
        "bytes": 64,
        "count": 1,
    }


def test_transfer_counter_add_and_reset_preserve_operation_breakdown():
    first = runtime.TransferCounter()
    second = runtime.TransferCounter()
    first.record_d2h(10, operation="block")
    second.record_d2h(20, count=2, operation="block")
    first.add(second)

    result = first.to_dict()
    assert result["by_kind"]["d2h"] == {"bytes": 30, "count": 3}
    assert result["by_operation"]["d2h"]["block"] == {
        "bytes": 30,
        "count": 3,
    }
    first.reset()
    assert first.to_dict()["by_operation"] == {}


def test_workspace_planner_reuses_cpu_phase_arrays():
    planner = runtime.WorkspacePlanner(device="cpu")
    with planner.phase("residual") as arena:
        first = arena.get("tmp", (3, 4), np.float64)
    with planner.arena("residual") as arena:
        second = arena.empty("tmp", (3, 4), np.float64)

    assert isinstance(first, np.ndarray)
    assert first is second
    assert planner.stats()["allocations"] == 1
    assert planner.stats()["reuses"] == 1
    assert planner.stats()["active_bytes"] == first.nbytes


def test_workspace_planner_releases_block_packed_staging_without_leak():
    planner = runtime.WorkspacePlanner(device="cpu")
    packed = planner.get(
        "direct-resident-staging", "wVVoo-packed", (37, 3, 3)
    )
    dense = planner.get(
        "direct-resident-staging", "wVvoO", (8, 8, 3, 3)
    )
    expected = packed.nbytes + dense.nbytes
    assert planner.stats()["active_arrays"] == 2
    assert planner.stats()["active_bytes"] == expected

    planner.release("direct-resident-staging")
    assert planner.stats()["active_arrays"] == 0
    assert planner.stats()["active_bytes"] == 0

    replacement = planner.get(
        "direct-resident-staging", "wVVoo-packed", (37, 3, 3)
    )
    assert replacement is not packed
    planner.release()
    assert planner.stats()["active_arrays"] == 0
    assert planner.stats()["active_bytes"] == 0


def test_block_lower_ao_workspace_matches_dense_contractions():
    rng = np.random.default_rng(1241)
    nao, nocc, nvir, nmo = 7, 5, 6, 9
    dense = rng.normal(size=(nao, nao, nocc, nocc))
    # The two diagonal shell blocks are deliberately non-symmetric.  Only the
    # off-diagonal block is mirrored, matching the direct-integral schedule.
    dense[0:3, 3:7] = dense[3:7, 0:3].transpose(1, 0, 2, 3)
    blocks = ((0, 3, 0, 3), (3, 7, 0, 3), (3, 7, 3, 7))
    layout = runtime.make_block_lower_ao_layout(nao, blocks)
    packed = np.empty((layout["stored_pair_count"], nocc, nocc))
    for bounds, offset in zip(blocks, layout["block_offsets"]):
        row0, row1, column0, column1 = bounds
        runtime.store_block_lower_ao_block(
            packed, dense[row0:row1, column0:column1], offset,
            array_module=np,
        )
    coefficients = rng.normal(size=(nao, nmo))
    virtual = rng.normal(size=(nao, nvir))

    expected_metric = runtime.contract_traced_ao_metric(
        dense, coefficients, array_module=np
    )
    observed_metric = runtime.contract_block_lower_traced_ao_metric(
        packed, coefficients, layout, array_module=np
    )
    np.testing.assert_allclose(
        observed_metric, expected_metric, rtol=3e-14, atol=3e-14
    )

    half = np.einsum("pqji,pa->aqji", dense, -virtual)
    expected_virtual = np.einsum("aqji,qb->bjia", half, virtual)
    observed_virtual = runtime.contract_block_lower_ao_to_virtual(
        packed, virtual, layout, occupied_block_size=2,
        array_module=np, scale=-1.0,
    )
    np.testing.assert_allclose(
        observed_virtual, expected_virtual, rtol=3e-14, atol=3e-14
    )


def test_store_block_lower_ao_block_reconstructs_baseline_layout():
    rng = np.random.default_rng(1242)
    nao, nocc = 8, 3
    dense = np.zeros((nao, nao, nocc, nocc))
    blocks = ((0, 3, 0, 3), (3, 6, 0, 3), (3, 6, 3, 6),
              (6, 8, 0, 3), (6, 8, 3, 6), (6, 8, 6, 8))
    layout = runtime.make_block_lower_ao_layout(nao, blocks)
    packed = np.zeros((layout["stored_pair_count"], nocc, nocc))
    for bounds, offset in zip(blocks, layout["block_offsets"]):
        row0, row1, column0, column1 = bounds
        block = rng.normal(size=(row1-row0, column1-column0, nocc, nocc))
        runtime.store_block_lower_ao_block(
            packed, block, offset, array_module=np
        )
        dense[row0:row1, column0:column1] = block
        if row0 != column0:
            dense[column0:column1, row0:row1] = block.transpose(1, 0, 2, 3)
    reconstructed = np.empty_like(dense)
    values = packed.transpose(1, 2, 0)
    runtime._reconstruct_block_lower_ao(values, layout, reconstructed.transpose(2, 3, 0, 1))
    np.testing.assert_array_equal(reconstructed, dense)


def test_block_lower_ao_workspace_rejects_invalid_shapes():
    layout = runtime.make_block_lower_ao_layout(
        3, ((0, 1, 0, 1), (1, 3, 0, 1), (1, 3, 1, 3))
    )
    with pytest.raises(ValueError, match="AO-pair dimension"):
        runtime.contract_block_lower_traced_ao_metric(
            np.empty((5, 2, 2)), np.empty((3, 4)), layout,
            array_module=np
        )
    with pytest.raises(ValueError, match="positive"):
        runtime.contract_block_lower_ao_to_virtual(
            np.empty((layout["stored_pair_count"], 2, 2)),
            np.empty((3, 4)), layout,
            occupied_block_size=0, array_module=np,
        )
    with pytest.raises(ValueError, match="lower-triangle"):
        runtime.make_block_lower_ao_layout(
            4, ((0, 2, 2, 4), (0, 2, 0, 2), (2, 4, 2, 4))
        )


def test_explicit_gpu_workspace_fails_when_cupy_is_unavailable(monkeypatch):
    monkeypatch.setattr(runtime, "_cp", None)
    planner = runtime.WorkspacePlanner(device="gpu")
    with pytest.raises(RuntimeError, match="CuPy is unavailable"):
        planner.get("test", "array", (2, 2))


def test_contraction_backend_cpu_fallback_and_out_beta():
    backend = runtime.ContractionBackend(backend="auto", use_cutensor=True)
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(12.0).reshape(3, 4)
    out = np.ones((2, 4))
    result = backend.contract("ik,kj->ij", left, right, out=out, alpha=2.0, beta=0.5)

    expected = 2.0 * np.einsum("ik,kj->ij", left, right) + 0.5
    np.testing.assert_allclose(result, expected)
    assert result is out
    assert backend.backend == "numpy-einsum"
    assert backend.to_dict()["cached_routes"] == 1


def test_nvtx_range_is_safe_without_nvtx_or_cupy():
    with runtime.nvtx_range("cpu-test", enabled=True):
        pass


def test_unknown_backend_fails_instead_of_silently_falling_back():
    with pytest.raises(ValueError):
        runtime.ContractionBackend(backend="mystery")


def test_cuda_binary_autotuner_preserves_gemm_result_and_records_decision():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    rng = np.random.default_rng(301)
    left = cp.asarray(rng.normal(size=(48, 32)))
    right = cp.asarray(rng.normal(size=(32, 40)))
    backend = runtime.ContractionBackend(backend="gpu", use_cutensor=True)
    route = backend.autotune_binary_gemm(
        "ik,kj->ij", left, right, warmup=1, repeats=2
    )
    observed = backend.einsum("ik,kj->ij", left, right)
    cp.testing.assert_allclose(observed, left @ right, rtol=1e-11, atol=1e-12)
    assert route in {"cupy-matmul", "cutensor"}
    assert backend.to_dict()["tuned_routes"][0]["selected"] == route
