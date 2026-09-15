"""Schedule, ABI, and optional CUDA parity tests for selected GINT columns."""

from __future__ import annotations

import ctypes
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gpu4pyscf.cc.gint_selected_columns as selected_module
from gpu4pyscf.cc.gint_selected_columns import (
    GINTSelectedAOPairColumnProvider,
    _BPCACHE_CONSTANT_BYTES,
    _BPCACHE_SIZE_SYMBOL,
    _COLUMNS_SYMBOL,
    _DIAGONAL_SYMBOL,
    _WORKSPACE_SIZE_SYMBOL,
    _SELECTED_ABI_OFFSET_SYMBOL,
    _SELECTED_ABI_SIZE_SYMBOL,
    _SELECTED_ABI_VERSION,
    _SELECTED_ABI_VERSION_SYMBOL,
    _CSelectedPairData,
    _basis_prod_cache_abi_identity,
    _device_multiprocessor_count,
    _normalise_column_kernel,
    _selected_pair_abi_identity,
    _normalise_selected_pairs,
    _order_selected_request,
    build_selected_pair_schedule,
)
from gpu4pyscf.cc.gint_transfer_audit import (
    GINTSetupTransferAudit,
    REQUIRED_DEVICE_PAYLOADS,
    REQUIRED_SETUP_TRANSFERS,
    validate_runtime_performance_gate,
)


def _identity_schedule_inputs():
    coeff = np.eye(3, dtype=np.float64)
    shell_ao_locs = np.arange(4, dtype=np.int32)
    shell_to_group = np.array([0, 0, 1], dtype=np.int32)
    group_i = np.array([0, 1, 1], dtype=np.int32)
    group_j = np.array([0, 0, 1], dtype=np.int32)
    pair2bra = (
        np.array([0, 0, 1, 1], dtype=np.int32),
        np.array([2, 2], dtype=np.int32),
        np.array([2], dtype=np.int32),
    )
    pair2ket = (
        np.array([0, 1, 0, 1], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        np.array([2], dtype=np.int32),
    )
    log_qs = tuple(np.zeros(values.size) for values in pair2bra)
    q_cond = np.ones((3, 3), dtype=np.float64)
    return {
        "coeff": coeff,
        "shell_ao_locs": shell_ao_locs,
        "shell_to_group": shell_to_group,
        "group_i": group_i,
        "group_j": group_j,
        "pair2bra": pair2bra,
        "pair2ket": pair2ket,
        "log_qs": log_qs,
        "q_cond": q_cond,
        "direct_scf_tol": 1e-12,
    }


def _build_identity_schedule(**updates):
    inputs = _identity_schedule_inputs()
    inputs.update(updates)
    return build_selected_pair_schedule(**inputs)


class _Counter:
    def __init__(self):
        self.records = []

    def record(self, direction, nbytes, count=1, *, operation=None):
        self.records.append((direction, operation, int(nbytes), int(count)))


def _complete_setup_audit(counter=None):
    audit = GINTSetupTransferAudit(transfer_counter=counter)
    for index, operation in enumerate(REQUIRED_SETUP_TRANSFERS, start=1):
        audit.record_transfer(
            "h2d",
            operation,
            8 * index,
            count=index,
            logical_payload=f"logical payload {index}",
            provenance=f"test provenance {index}",
        )
    for index, operation in enumerate(REQUIRED_DEVICE_PAYLOADS, start=1):
        audit.record_device_payload(
            operation,
            100 * index,
            logical_payload=f"resident payload {index}",
            provenance=f"device provenance {index}",
        )
    audit.record_exclusion(
        "cuda_runtime_kernel_argument_marshalling",
        classification="runtime-control-not-application-array-copy",
        reason="test exclusion",
    )
    return audit


def test_setup_transfer_audit_preserves_exact_bytes_counts_and_payloads():
    counter = _Counter()
    audit = _complete_setup_audit(counter)
    ledger = audit.to_dict()

    expected_bytes = sum(8 * index for index in range(1, 13))
    expected_count = sum(range(1, 13))
    assert ledger["directions"]["h2d"]["bytes"] == expected_bytes
    assert ledger["directions"]["h2d"]["count"] == expected_count
    assert ledger["total_bytes"] == expected_bytes
    assert len(counter.records) == len(REQUIRED_SETUP_TRANSFERS)
    assert ledger["complete_for_declared_payloads"] is True
    assert ledger["unresolved_operations"] == []
    assert ledger["logical_device_payloads"][
        "basis_block_diag_coeff"
    ]["origin"] == "device-generated"
    operation = ledger["directions"]["h2d"]["operations"][
        "gint_basis_cache_aexyz"
    ]
    assert operation["bytes"] == 11 * 8
    assert operation["count"] == 11
    assert operation["logical_payload"] == "logical payload 11"
    assert operation["provenance"] == "test provenance 11"


def test_setup_transfer_audit_fails_closed_on_missing_or_blank_provenance():
    audit = GINTSetupTransferAudit()
    with pytest.raises(ValueError, match="provenance"):
        audit.record_transfer(
            "h2d",
            "basis_c2s_blocks",
            8,
            logical_payload="C2S coefficients",
            provenance="",
        )
    audit.record_transfer(
        "h2d",
        "basis_c2s_blocks",
        8,
        logical_payload="C2S coefficients",
        provenance="unit test",
    )
    ledger = audit.to_dict()
    assert ledger["complete_for_declared_payloads"] is False
    assert "vhfopt_ao_sort_index" in ledger["coverage"][
        "missing_transfer_operations"
    ]
    assert ledger["unresolved_operations"]


class _FakeCFunction:
    def __init__(self, result):
        self.result = result

    def __call__(self, *args):
        return self.result(*args) if callable(self.result) else self.result


def test_basis_prod_cache_size_abi_match_and_mismatch_fail_closed():
    matching = SimpleNamespace(
        **{_BPCACHE_SIZE_SYMBOL: _FakeCFunction(_BPCACHE_CONSTANT_BYTES)}
    )
    identity = _basis_prod_cache_abi_identity(matching)
    assert identity["verified"] is True
    assert identity["libgint_size_bytes"] == _BPCACHE_CONSTANT_BYTES

    mismatched = SimpleNamespace(
        **{_BPCACHE_SIZE_SYMBOL: _FakeCFunction(_BPCACHE_CONSTANT_BYTES + 8)}
    )
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        _basis_prod_cache_abi_identity(mismatched)
    with pytest.raises(RuntimeError, match="size ABI"):
        _basis_prod_cache_abi_identity(SimpleNamespace())


def test_selected_pair_abi_binds_size_offsets_and_version():
    offsets = {
        index: getattr(_CSelectedPairData, name).offset
        for index, (name, _) in enumerate(_CSelectedPairData._fields_)
    }
    matching = SimpleNamespace(**{
        _SELECTED_ABI_SIZE_SYMBOL: _FakeCFunction(
            ctypes.sizeof(_CSelectedPairData)
        ),
        _SELECTED_ABI_OFFSET_SYMBOL: _FakeCFunction(offsets.__getitem__),
        _SELECTED_ABI_VERSION_SYMBOL: _FakeCFunction(_SELECTED_ABI_VERSION),
    })
    identity = _selected_pair_abi_identity(matching)
    assert identity["verified"] is True
    assert identity["abi_version"] == _SELECTED_ABI_VERSION == 2
    assert identity["fields"][-1]["name"] == "task_log_q"

    mismatched = SimpleNamespace(**{
        _SELECTED_ABI_SIZE_SYMBOL: _FakeCFunction(
            ctypes.sizeof(_CSelectedPairData)
        ),
        _SELECTED_ABI_OFFSET_SYMBOL: _FakeCFunction(
            lambda index: offsets[index] + (1 if index == 3 else 0)
        ),
        _SELECTED_ABI_VERSION_SYMBOL: _FakeCFunction(_SELECTED_ABI_VERSION),
    })
    with pytest.raises(RuntimeError, match="field-offset"):
        _selected_pair_abi_identity(mismatched)


def test_runtime_performance_gate_rejects_mapping_and_requires_receipt_path():
    pending = validate_runtime_performance_gate(
        None, basis_prod_cache_size_bytes=_BPCACHE_CONSTANT_BYTES
    )
    assert pending["validated"] is False
    assert pending["status"] == "pending"
    assert pending["binding_contract"]["loader_status"] == "implemented"
    with pytest.raises(TypeError, match="mappings are forbidden"):
        validate_runtime_performance_gate(
            {"status": "passed"},
            basis_prod_cache_size_bytes=_BPCACHE_CONSTANT_BYTES,
        )


def test_selected_schedule_maps_every_packed_pair_to_one_shell_task():
    schedule = _build_identity_schedule()
    np.testing.assert_array_equal(schedule.pair_i, [0, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(schedule.pair_j, [0, 0, 1, 0, 1, 2])
    np.testing.assert_array_equal(schedule.pair_cp_id, [0, 0, 0, 1, 1, 2])
    np.testing.assert_array_equal(schedule.pair_task_id, [0, 2, 3, 0, 1, 0])
    np.testing.assert_array_equal(schedule.pair_symmetrize, [0, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(schedule.cp_task_offsets, [0, 4, 6, 7])
    np.testing.assert_array_equal(schedule.shell_original_offsets, [0, 1, 2, 3])
    np.testing.assert_array_equal(schedule.shell_original_aos, [0, 1, 2])
    np.testing.assert_array_equal(schedule.pairs_for_cp(1), [3, 4])
    assert schedule.scheduled_pair_count == schedule.npair == 6
    assert schedule.screened_pair_count == 0
    assert schedule.metadata()["single_shell_support"] is True
    for value in (
        schedule.pair_i,
        schedule.pair_task_id,
        schedule.task_log_q,
        schedule.shell_original_aos,
    ):
        assert value.flags.writeable is False


def test_selected_schedule_fails_closed_for_multi_shell_original_ao_support():
    inputs = _identity_schedule_inputs()
    inputs.update(
        coeff=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0],
            ]
        ),
        shell_ao_locs=np.arange(5, dtype=np.int32),
        shell_to_group=np.array([0, 0, 1, 1], dtype=np.int32),
        q_cond=np.ones((4, 4)),
    )
    with pytest.raises(NotImplementedError, match="exactly one"):
        build_selected_pair_schedule(**inputs)


def test_selected_schedule_rejects_an_unscreened_missing_task():
    inputs = _identity_schedule_inputs()
    inputs["pair2bra"] = (
        inputs["pair2bra"][0],
        np.array([2], dtype=np.int32),
        inputs["pair2bra"][2],
    )
    inputs["pair2ket"] = (
        inputs["pair2ket"][0],
        np.array([0], dtype=np.int32),
        inputs["pair2ket"][2],
    )
    inputs["log_qs"] = tuple(np.zeros(values.size) for values in inputs["pair2bra"])
    with pytest.raises(RuntimeError, match="unscreened.*no GINT task"):
        build_selected_pair_schedule(**inputs)


def test_selected_schedule_encodes_a_screened_missing_task_as_zero_source():
    inputs = _identity_schedule_inputs()
    inputs["pair2bra"] = (
        inputs["pair2bra"][0],
        np.array([2], dtype=np.int32),
        inputs["pair2bra"][2],
    )
    inputs["pair2ket"] = (
        inputs["pair2ket"][0],
        np.array([0], dtype=np.int32),
        inputs["pair2ket"][2],
    )
    inputs["log_qs"] = tuple(np.zeros(values.size) for values in inputs["pair2bra"])
    q_cond = inputs["q_cond"].copy()
    q_cond[2, 1] = q_cond[1, 2] = 0.5e-12
    inputs["q_cond"] = q_cond
    schedule = build_selected_pair_schedule(**inputs)
    # Packed pair (2,1) is index 4.
    assert schedule.pair_cp_id[4] == -1
    assert schedule.pair_task_id[4] == -1
    assert schedule.screened_pair_count == 1


def test_selected_schedule_checks_log_q_against_q_cond():
    inputs = _identity_schedule_inputs()
    logs = list(inputs["log_qs"])
    logs[0] = logs[0].copy()
    logs[0][0] = -1.0
    inputs["log_qs"] = tuple(logs)
    with pytest.raises(ValueError, match="log_q.*q_cond"):
        build_selected_pair_schedule(**inputs)


def test_selected_schedule_rejects_noninteger_or_overflowed_abi_indices():
    inputs = _identity_schedule_inputs()
    inputs["shell_to_group"] = np.array([0.0, 0.0, 1.0])
    with pytest.raises(TypeError, match="integers"):
        build_selected_pair_schedule(**inputs)

    inputs = _identity_schedule_inputs()
    inputs["group_i"] = np.array([0, 1, 2**40], dtype=np.int64)
    with pytest.raises(OverflowError, match="int32 ABI"):
        build_selected_pair_schedule(**inputs)


def test_selected_column_batch_contract_accepts_b1_and_b_gt_1_only():
    np.testing.assert_array_equal(
        _normalise_selected_pairs([2], dimension=6, max_batch_size=2), [2]
    )
    np.testing.assert_array_equal(
        _normalise_selected_pairs([5, 0], dimension=6, max_batch_size=2), [5, 0]
    )
    with pytest.raises(ValueError, match="maximum"):
        _normalise_selected_pairs([0, 1, 2], dimension=6, max_batch_size=2)
    with pytest.raises(MemoryError, match="full AO-pair matrix"):
        _normalise_selected_pairs(range(6), dimension=6, max_batch_size=6)
    with pytest.raises(IndexError):
        _normalise_selected_pairs([6], dimension=6, max_batch_size=2)
    with pytest.raises(TypeError):
        _normalise_selected_pairs([True], dimension=6, max_batch_size=2)


@pytest.mark.parametrize("selected_count", [1, 2, 8, 32])
def test_grouped_selected_request_is_stable_and_preserves_output_rows(
    selected_count,
):
    pair_task_id = np.repeat(np.arange(8, dtype=np.int32), 8)
    pivots = np.arange(selected_count, dtype=np.int32)[::-1]
    rows = np.arange(selected_count, dtype=np.int32)

    ordered, ordered_rows, group_count = _order_selected_request(
        pivots,
        rows,
        pair_task_id,
        column_kernel="grouped",
    )

    tasks = pair_task_id[ordered]
    assert np.all(tasks[1:] >= tasks[:-1])
    assert group_count == np.unique(pair_task_id[pivots]).size
    reconstructed = np.empty_like(ordered)
    reconstructed[ordered_rows] = ordered
    np.testing.assert_array_equal(reconstructed, pivots)
    assert ordered.dtype == np.int32
    assert ordered_rows.dtype == np.int32
    assert ordered.flags.writeable is False
    assert ordered_rows.flags.writeable is False


def test_reference_selected_request_preserves_order_and_counts_each_pivot():
    pivots = np.array([5, 0, 3, 1], dtype=np.int32)
    rows = np.arange(pivots.size, dtype=np.int32)
    tasks = np.array([2, 0, 1, 0, 3, 2], dtype=np.int32)
    ordered, ordered_rows, group_count = _order_selected_request(
        pivots, rows, tasks, column_kernel="reference"
    )
    np.testing.assert_array_equal(ordered, pivots)
    np.testing.assert_array_equal(ordered_rows, rows)
    assert group_count == pivots.size


def test_column_kernel_selector_is_explicit_and_fail_closed():
    assert _normalise_column_kernel(" REFERENCE ") == "reference"
    assert _normalise_column_kernel("Grouped") == "grouped"
    with pytest.raises(TypeError, match="must be a string"):
        _normalise_column_kernel(None)
    with pytest.raises(ValueError, match="reference, grouped"):
        _normalise_column_kernel("automatic")


def test_grouped_gout_counter_matches_cutoff_and_distinct_ket_tasks():
    provider = object.__new__(GINTSelectedAOPairColumnProvider)
    provider.schedule = SimpleNamespace(
        cp_task_offsets=np.array([0, 3, 5], dtype=np.int32),
        task_log_q=np.array([0.0, -5.0, -10.0, 0.0, -20.0]),
        pair_task_id=np.array([0, 0, 1], dtype=np.int32),
    )
    provider.direct_scf_tol = float(np.exp(-6.0))
    selected = np.array([0, 1, 2], dtype=np.int32)

    provider.column_kernel = "grouped"
    reference, grouped = provider._column_gout_counts(
        cp_ij_id=0, cp_kl_id=1, selected_pairs=selected
    )
    assert reference == 4
    assert grouped == 2

    provider.column_kernel = "reference"
    reference, actual = provider._column_gout_counts(
        cp_ij_id=0, cp_kl_id=1, selected_pairs=selected
    )
    assert reference == actual == 4


def test_pre_cutoff_group_attempts_are_distinct_from_screened_gout_builds():
    provider = object.__new__(GINTSelectedAOPairColumnProvider)
    provider.schedule = SimpleNamespace(
        cp_task_offsets=np.array([0, 1, 3], dtype=np.int32),
        task_log_q=np.array([0.0, 0.0, -20.0]),
        pair_task_id=np.array([0, 1], dtype=np.int32),
    )
    provider.direct_scf_tol = float(np.exp(-6.0))
    provider.column_kernel = "grouped"
    selected = np.array([0, 1], dtype=np.int32)
    _ordered, _rows, group_count = _order_selected_request(
        selected,
        np.arange(2, dtype=np.int32),
        provider.schedule.pair_task_id,
        column_kernel="grouped",
    )
    _reference, screened_gout = provider._column_gout_counts(
        cp_ij_id=0, cp_kl_id=1, selected_pairs=selected
    )
    precutoff_attempts = group_count * int(
        provider.schedule.cp_task_offsets[1]
        - provider.schedule.cp_task_offsets[0]
    )
    assert precutoff_attempts == 2
    assert screened_gout == 1


def test_selected_provider_exposes_resident_pair_indices_without_copy():
    provider = object.__new__(GINTSelectedAOPairColumnProvider)
    pair_i = object()
    pair_j = object()
    provider._device_schedule = {"pair_i": pair_i, "pair_j": pair_j}

    actual_i, actual_j = provider.pair_indices_device

    assert actual_i is pair_i
    assert actual_j is pair_j


def test_device_multiprocessor_count_accepts_cupy_string_and_bytes_keys():
    def fake_cupy(properties):
        return SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
            getDevice=lambda: 0,
            getDeviceProperties=lambda _device: properties,
        )))

    assert _device_multiprocessor_count(
        fake_cupy({"multiProcessorCount": 108})
    ) == 108
    assert _device_multiprocessor_count(
        fake_cupy({b"multiProcessorCount": 108})
    ) == 108
    with pytest.raises(RuntimeError, match="multiProcessorCount"):
        _device_multiprocessor_count(fake_cupy({}))


def test_high_rys_workspace_c_args_are_exact_and_null_for_low_order():
    provider = object.__new__(GINTSelectedAOPairColumnProvider)
    provider._high_rys_workspace = None
    provider._high_rys_workspace_nbytes = 0
    pointer, size = provider._high_rys_workspace_c_args()
    assert pointer.value is None
    assert size.value == 0

    provider._high_rys_workspace = SimpleNamespace(
        data=SimpleNamespace(ptr=123456)
    )
    provider._high_rys_workspace_nbytes = 600_514_560
    pointer, size = provider._high_rys_workspace_c_args()
    assert pointer.value == 123456
    assert size.value == 600_514_560


def _mock_stream_provider(cupy):
    provider = object.__new__(GINTSelectedAOPairColumnProvider)
    provider._cupy = cupy
    provider._construction_process_id = selected_module.os.getpid()
    provider._construction_device_id = 0
    provider._bound_stream_ptr = None
    provider._bound_stream = None
    provider._stream_contract_call_count = 0
    provider.vhfopt = SimpleNamespace(bpcache=ctypes.c_void_p(1234))
    return provider


def test_stream_contract_rejects_a_second_stream_for_one_provider(
    monkeypatch,
):
    monkeypatch.setattr(selected_module, "_DEVICE_STREAM_BINDINGS", {})
    current_stream = [SimpleNamespace(ptr=101)]
    device_synchronizations = []
    cupy = SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: current_stream[0],
        runtime=SimpleNamespace(
            getDevice=lambda: 0,
            deviceSynchronize=lambda: device_synchronizations.append(True),
        ),
    ))
    provider = _mock_stream_provider(cupy)
    provider._register_selected_stream()

    with provider._selected_enqueue_scope() as (stream, bpcache):
        assert stream.value == 101
        assert bpcache.value == 1234
    current_stream[0] = SimpleNamespace(ptr=202)
    with pytest.raises(RuntimeError, match="process-wide bound to stream 101"):
        with provider._selected_enqueue_scope():
            pass

    assert provider._bound_stream_ptr == 101
    assert provider._stream_contract_call_count == 1
    assert device_synchronizations == []


def test_stream_contract_rejects_two_providers_crossing_streams(monkeypatch):
    monkeypatch.setattr(selected_module, "_DEVICE_STREAM_BINDINGS", {})
    current_stream = [SimpleNamespace(ptr=303)]
    cupy = SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: current_stream[0],
        runtime=SimpleNamespace(getDevice=lambda: 0),
    ))
    first = _mock_stream_provider(cupy)
    second = _mock_stream_provider(cupy)
    first._register_selected_stream()
    second._register_selected_stream()

    # Both providers may enqueue on the one process-wide device stream.
    with first._selected_enqueue_scope():
        pass
    with second._selected_enqueue_scope():
        pass

    # The process binding rejects another provider before it can enqueue a
    # constant-memory copy or a workspace-using kernel on a different stream.
    current_stream[0] = SimpleNamespace(ptr=404)
    with pytest.raises(RuntimeError, match="c_bpcache is shared constant memory"):
        with second._selected_enqueue_scope():
            pass
    assert first._stream_contract_call_count == 1
    assert second._stream_contract_call_count == 1


def test_cached_diagonal_adds_no_second_transfer_or_constant_copy():
    class _CachedArray:
        def copy(self):
            return self

    provider = object.__new__(GINTSelectedAOPairColumnProvider)
    provider._selected_enqueue_scope = lambda: nullcontext((None, None))
    provider.diagonal_calls = 1
    provider._diagonal = _CachedArray()
    provider._setup_transfer_audit = _complete_setup_audit()
    provider._basis_prod_cache_abi = {
        "symbol": _BPCACHE_SIZE_SYMBOL,
        "python_ctypes_size_bytes": _BPCACHE_CONSTANT_BYTES,
        "libgint_size_bytes": _BPCACHE_CONSTANT_BYTES,
        "verified": True,
    }
    provider._transfer_bytes = {
        "h2d": {"selected_gint_constant_cache": _BPCACHE_CONSTANT_BYTES},
        "d2h": {},
    }
    provider._transfer_counts = {
        "h2d": {"selected_gint_constant_cache": 1},
        "d2h": {},
    }
    before = provider.explicit_transfer_ledger()
    provider.diagonal()
    after = provider.explicit_transfer_ledger()
    assert after["directions"] == before["directions"]
    assert after["total_bytes"] == before["total_bytes"]
    assert provider.diagonal_calls == 2


def test_constant_cache_copy_status_counts_only_a_confirmed_c_abi_copy():
    provider = object.__new__(GINTSelectedAOPairColumnProvider)
    provider.transfer_counter = _Counter()
    provider._transfer_bytes = {"h2d": {}, "d2h": {}}
    provider._transfer_counts = {"h2d": {}, "d2h": {}}

    provider._record_constant_cache_copy(0)
    assert provider._transfer_bytes["h2d"] == {}
    assert provider.transfer_counter.records == []
    assert getattr(provider, "constant_cache_copies", 0) == 0

    provider._record_constant_cache_copy(1)
    assert provider._transfer_bytes["h2d"] == {
        "selected_gint_constant_cache": _BPCACHE_CONSTANT_BYTES
    }
    assert provider._transfer_counts["h2d"] == {
        "selected_gint_constant_cache": 1
    }
    assert provider.constant_cache_copies == 1
    with pytest.raises(RuntimeError, match="invalid constant-copy status"):
        provider._record_constant_cache_copy(2)


def test_selected_cuda_source_and_public_abi_are_bounded_and_fail_closed():
    root = Path(__file__).parents[2]
    source = (root / "lib" / "gint" / "selected_columns.cu").read_text()
    bpcache = (root / "lib" / "gint" / "bpcache.cu").read_text()
    header = (root / "lib" / "gint" / "gint.h").read_text()
    cmake = (root / "lib" / "gint" / "CMakeLists.txt").read_text()
    provider = (root / "cc" / "gint_selected_columns.py").read_text()

    assert _COLUMNS_SYMBOL in source and _COLUMNS_SYMBOL in header
    assert _DIAGONAL_SYMBOL in source and _DIAGONAL_SYMBOL in header
    assert _WORKSPACE_SIZE_SYMBOL in source and _WORKSPACE_SIZE_SYMBOL in header
    assert _BPCACHE_SIZE_SYMBOL in bpcache and _BPCACHE_SIZE_SYMBOL in header
    assert _SELECTED_ABI_SIZE_SYMBOL in source and _SELECTED_ABI_SIZE_SYMBOL in header
    assert _SELECTED_ABI_OFFSET_SYMBOL in source and _SELECTED_ABI_OFFSET_SYMBOL in header
    assert _SELECTED_ABI_VERSION_SYMBOL in source and _SELECTED_ABI_VERSION_SYMBOL in header
    assert "constant_copy_performed" in source
    assert "constant_copy_performed" in header
    assert "GINTg0_2e_2d4d" in source
    assert "GINTgout2e" in source
    assert "selected_columns.cu" in cmake
    assert "cudaMalloc" not in source
    assert "GINT_SELECTED_HIGH_RYS_THREADS = 32" in source
    assert "selected_columns_kernel_bounded_workspace" in source
    assert "selected_diagonal_kernel_bounded_workspace" in source
    assert "selected_columns_kernel_grouped_cutoff" in source
    assert "selected_columns_kernel_grouped_bounded_workspace" in source
    assert "selected_columns_grouped_task" in source
    assert "GINT_SELECTED_GROUPED_COLUMNS = 2" in source
    assert "data->abi_version != 2" in source
    assert "provider-owned-bounded-global-grid-stride" in provider
    assert "case 7: return launch_selected_columns_bounded_workspace" in source
    assert "case 8: return launch_selected_columns_bounded_workspace" in source
    assert "case 7: return launch_selected_diagonal_bounded_workspace" in source
    assert "case 8: return launch_selected_diagonal_bounded_workspace" in source
    assert "performance_eligible = False" in provider
    assert '_COLUMN_KERNELS = ("reference", "grouped")' in provider
    assert "materializes_per_pivot_ao_matrix = False" in provider
    assert "single_shell_support != 1" in source
    assert "spherical != 1" in source


def _require_selected_cuda_abi():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.scf import int4c2e

    if any(
        not hasattr(int4c2e.libgint, symbol)
        for symbol in (
            _COLUMNS_SYMBOL,
            _DIAGONAL_SYMBOL,
            _WORKSPACE_SIZE_SYMBOL,
            _BPCACHE_SIZE_SYMBOL,
            _SELECTED_ABI_SIZE_SYMBOL,
            _SELECTED_ABI_OFFSET_SYMBOL,
            _SELECTED_ABI_VERSION_SYMBOL,
        )
    ):
        pytest.skip("libgint has not been rebuilt with the selected-pair ABI")
    return cupy


def _require_grouped_cuda_abi():
    if os.getenv("GPU4PYSCF_TEST_GROUPED_GINT") != "1":
        pytest.skip(
            "set GPU4PYSCF_TEST_GROUPED_GINT=1 only with a grouped-enabled "
            "libgint rebuilt from this source tree"
        )
    return _require_selected_cuda_abi()


def test_selected_cuda_stream_contract_rejects_two_nondefault_streams(
    monkeypatch,
):
    cupy = _require_selected_cuda_abi()
    from pyscf import gto

    # Isolate this contract test from the process binding established by
    # earlier GPU tests, after first completing all of their asynchronous work.
    cupy.cuda.runtime.deviceSynchronize()
    monkeypatch.setattr(selected_module, "_DEVICE_STREAM_BINDINGS", {})
    first_stream = cupy.cuda.Stream(non_blocking=True)
    second_stream = cupy.cuda.Stream(non_blocking=True)
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )

    try:
        with first_stream:
            first = GINTSelectedAOPairColumnProvider(
                mol, direct_scf_tol=1e-14, group_size=8, max_batch_size=2
            )
            second = GINTSelectedAOPairColumnProvider(
                mol, direct_scf_tol=1e-14, group_size=8, max_batch_size=2
            )
            # Leave selected-GINT work pending.  Safety must come from stream
            # ordering, not a hidden per-call synchronization.
            first.columns([0])
            second.columns([0])

        with second_stream:
            with pytest.raises(RuntimeError, match="process-wide bound"):
                first.columns([0])
            with pytest.raises(RuntimeError, match="process-wide bound"):
                second.columns([0])

        assert first.metadata()["stream_safety"][
            "per_call_device_synchronize"
        ] is False
        assert second.metadata()["stream_safety"]["bound_stream_ptr"] == int(
            first_stream.ptr
        )
    finally:
        cupy.cuda.runtime.deviceSynchronize()


def test_selected_cuda_b1_b2_columns_and_direct_diagonal_match_reference():
    cupy = _require_selected_cuda_abi()
    from pyscf import gto

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    provider = GINTSelectedAOPairColumnProvider(
        mol, direct_scf_tol=1e-14, group_size=8, max_batch_size=2
    )
    eri = mol.intor("int2e_sph")
    pair_i = provider.pair_map.pair_i
    pair_j = provider.pair_map.pair_j

    one = provider.columns([0])
    expected_one = eri[pair_i, pair_j, 0, 0][None, :]
    cupy.testing.assert_allclose(
        one, cupy.asarray(expected_one), atol=2e-11, rtol=2e-11
    )

    pivots = [1, 2]
    many = provider.columns(pivots)
    expected_many = np.stack(
        [
            eri[pair_i, pair_j, pair_i[pivot], pair_j[pivot]]
            for pivot in pivots
        ]
    )
    cupy.testing.assert_allclose(
        many, cupy.asarray(expected_many), atol=2e-11, rtol=2e-11
    )

    expected_diagonal = eri[pair_i, pair_j, pair_i, pair_j]
    cupy.testing.assert_allclose(
        provider.diagonal(),
        cupy.asarray(expected_diagonal),
        atol=2e-11,
        rtol=2e-11,
    )
    metadata = provider.metadata()
    assert metadata["performance_eligible"] is False
    assert metadata["materializes_pair_matrix"] is False
    assert metadata["materializes_four_index_tensor"] is False
    assert metadata["materializes_per_pivot_ao_matrix"] is False
    assert metadata["device_schedule"] is True
    assert metadata["precision"] == "fp64"
    assert metadata["output_residency"] == "gpu"
    assert metadata["runtime_gate"]["status"] == "pending"
    workspace = metadata["high_rys_workspace"]
    assert workspace["minimum_rys_order"] == 7
    assert workspace["maximum_rys_order"] == metadata["maximum_rys_order"]
    assert workspace["threads_per_block"] == 32
    assert workspace["multiprocessor_count"] > 0
    assert workspace["nbytes"] == 0
    assert workspace["resident"] is False
    stream_safety = metadata["stream_safety"]
    assert stream_safety["strategy"] == selected_module._STREAM_SAFETY_STRATEGY
    assert stream_safety["policy_version"] == 1
    assert stream_safety["binding_scope"] == "process-wide-per-cuda-device"
    assert stream_safety["strong_stream_lifetime"] is True
    assert stream_safety["cross_stream_behavior"] == (
        "reject-before-device-enqueue"
    )
    assert stream_safety["per_call_device_synchronize"] is False
    ledger_before_cached_diagonal = metadata["explicit_transfers"]
    provider.diagonal()
    ledger_after_cached_diagonal = provider.explicit_transfer_ledger()
    assert ledger_after_cached_diagonal["directions"] == (
        ledger_before_cached_diagonal["directions"]
    )
    assert ledger_after_cached_diagonal["complete_for_vhfopt_internal_setup"] is True
    assert ledger_after_cached_diagonal["complete_for_declared_payloads"] is True
    assert ledger_after_cached_diagonal["unresolved_operations"] == []
    assert ledger_after_cached_diagonal["basis_prod_cache_abi"]["verified"] is True
    assert {
        item["operation"]
        for item in ledger_after_cached_diagonal["excluded_operations"]
    } == {"cuda_runtime_kernel_argument_marshalling"}
    assert type(one).__module__.split(".", 1)[0] == "cupy"


def test_selected_cuda_water_d_shells_and_offdiagonal_groups_match_reference():
    """Exercise C2S coefficients, symmetrized groups, and Rys order > 1."""

    cupy = _require_selected_cuda_abi()
    from pyscf import gto

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="cc-pvdz",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    provider = GINTSelectedAOPairColumnProvider(
        mol, direct_scf_tol=1e-14, group_size=8, max_batch_size=2
    )
    symmetrized = np.flatnonzero(provider.schedule.pair_symmetrize)
    assert symmetrized.size > 0
    assert provider._maximum_rys_order > 1
    pivots = [int(symmetrized[0]), int(symmetrized[-1])]
    eri = mol.intor("int2e_sph")
    pair_i = provider.pair_map.pair_i
    pair_j = provider.pair_map.pair_j
    expected = np.stack(
        [
            eri[pair_i, pair_j, pair_i[pivot], pair_j[pivot]]
            for pivot in pivots
        ]
    )
    actual = provider.columns(pivots)
    cupy.testing.assert_allclose(
        actual, cupy.asarray(expected), atol=2e-10, rtol=2e-11
    )

    expected_diagonal = eri[pair_i, pair_j, pair_i, pair_j]
    cupy.testing.assert_allclose(
        provider.diagonal(),
        cupy.asarray(expected_diagonal),
        atol=2e-10,
        rtol=2e-11,
    )


def test_grouped_cuda_selected_counts_1_2_8_32_match_reference():
    cupy = _require_grouped_cuda_abi()
    from pyscf import gto

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="cc-pvdz",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    reference = GINTSelectedAOPairColumnProvider(
        mol,
        direct_scf_tol=1e-14,
        group_size=8,
        max_batch_size=32,
        column_kernel="reference",
    )
    grouped = GINTSelectedAOPairColumnProvider(
        mol,
        direct_scf_tol=1e-14,
        group_size=8,
        max_batch_size=32,
        column_kernel="grouped",
    )
    scheduled = np.flatnonzero(reference.schedule.pair_task_id >= 0)
    order = np.lexsort((
        scheduled,
        reference.schedule.pair_task_id[scheduled],
        reference.schedule.pair_cp_id[scheduled],
    ))
    master = scheduled[order][:32]
    assert master.size == 32

    for selected_count in (1, 2, 8, 32):
        pivots = master[:selected_count]
        expected = reference.columns(pivots)
        actual = grouped.columns(pivots)
        assert float(cupy.max(cupy.abs(actual - expected)).item()) <= 2e-10

    reference_diagonal = reference.diagonal()
    grouped_diagonal = grouped.diagonal()
    assert float(cupy.max(
        cupy.abs(grouped_diagonal - reference_diagonal)
    ).item()) <= 2e-10
    metadata = grouped.metadata()
    assert metadata["column_kernel"] == "grouped"
    assert metadata["performance_eligible"] is False
    assert metadata["column_gout_evaluations"] > 0
    assert metadata["reference_column_gout_evaluations"] >= (
        metadata["column_gout_evaluations"]
    )
    assert metadata["grouped_gout_evaluations_saved"] > 0
    assert metadata["constant_cache_copies"] == metadata["cabi_calls"]


def test_grouped_cuda_keeps_direct_cd_pivots_rank_and_diagonal():
    cupy = _require_grouped_cuda_abi()
    from pyscf import gto

    from gpu4pyscf.cc.device_runtime import TransferCounter
    from gpu4pyscf.cc.direct_cd import pivoted_cholesky_from_columns

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    reference = GINTSelectedAOPairColumnProvider(
        mol,
        direct_scf_tol=1e-14,
        group_size=8,
        max_batch_size=2,
        column_kernel="reference",
    )
    grouped = GINTSelectedAOPairColumnProvider(
        mol,
        direct_scf_tol=1e-14,
        group_size=8,
        max_batch_size=2,
        column_kernel="grouped",
    )
    reference_diagonal = reference.diagonal()
    grouped_diagonal = grouped.diagonal()
    reference_cd = pivoted_cholesky_from_columns(
        reference,
        threshold=1e-10,
        column_batch_size=2,
        transfer_counter=TransferCounter(),
    )
    grouped_cd = pivoted_cholesky_from_columns(
        grouped,
        threshold=1e-10,
        column_batch_size=2,
        transfer_counter=TransferCounter(),
    )
    assert float(cupy.max(
        cupy.abs(grouped_diagonal - reference_diagonal)
    ).item()) <= 2e-10
    assert grouped_cd.pivots == reference_cd.pivots
    assert grouped_cd.rank == reference_cd.rank
    assert abs(
        grouped_cd.residual_diagonal - reference_cd.residual_diagonal
    ) <= 2e-10


def test_grouped_cuda_rys7_workspace_screening_and_interleaved_ab():
    """MTU-only coverage for the bounded Rys-7 grouped workspace path."""

    cupy = _require_grouped_cuda_abi()
    from pyscf import gto

    from gpu4pyscf.cc.device_runtime import TransferCounter
    from gpu4pyscf.cc.direct_cd import pivoted_cholesky_from_columns

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="cc-pvtz",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    providers = {
        kernel: GINTSelectedAOPairColumnProvider(
            mol,
            direct_scf_tol=1e-8,
            group_size=16,
            max_batch_size=32,
            column_kernel=kernel,
        )
        for kernel in ("reference", "grouped")
    }
    reference = providers["reference"]
    grouped = providers["grouped"]
    assert reference._maximum_rys_order == grouped._maximum_rys_order == 7
    assert reference._high_rys_workspace_nbytes > 0
    assert (
        grouped._high_rys_workspace_nbytes
        == reference._high_rys_workspace_nbytes
    )

    scheduled = np.flatnonzero(reference.schedule.pair_task_id >= 0)
    keys = np.stack((
        reference.schedule.pair_cp_id[scheduled],
        reference.schedule.pair_task_id[scheduled],
    ), axis=1)
    groups = []
    log_cutoff = float(np.log(reference.direct_scf_tol))
    for key in np.unique(keys, axis=0):
        members = scheduled[np.all(keys == key, axis=1)]
        cp_id, task_id = (int(value) for value in key)
        ket_log_q = reference.schedule.task_log_q[
            int(reference.schedule.cp_task_offsets[cp_id]) + task_id
        ]
        screened = any(
            np.any(
                reference.schedule.task_log_q[
                    int(reference.schedule.cp_task_offsets[row_cp_id])
                    : int(reference.schedule.cp_task_offsets[row_cp_id + 1])
                ] + ket_log_q < log_cutoff
            )
            for row_cp_id in range(reference.schedule.ncp)
        )
        groups.append((members, screened))
    screened_repeated = [
        members for members, screened in groups
        if screened and members.size > 1
    ]
    assert screened_repeated
    primary = min(screened_repeated, key=lambda values: int(values[0]))
    remainder = scheduled[~np.isin(scheduled, primary[:2])]
    master = np.concatenate((primary[:2], remainder))[:32]
    assert master.size == 32
    master_keys = np.stack((
        reference.schedule.pair_cp_id[master],
        reference.schedule.pair_task_id[master],
    ), axis=1)
    _, master_counts = np.unique(master_keys, axis=0, return_counts=True)
    assert np.any(master_counts > 1)

    reference_batches = {}
    grouped_batches = {}
    reference_diagonal = grouped_diagonal = None
    for selected_count in (1, 2, 8, 32):
        pivots = master[:selected_count]
        reference_batches[selected_count] = reference.columns(pivots)
        grouped_batches[selected_count] = grouped.columns(pivots)
        if selected_count == 2:
            reference_diagonal = reference.diagonal()
            grouped_diagonal = grouped.diagonal()
    cupy.cuda.runtime.deviceSynchronize()

    for selected_count in (1, 2, 8, 32):
        error = float(cupy.max(cupy.abs(
            grouped_batches[selected_count]
            - reference_batches[selected_count]
        )).item())
        assert error <= 2e-10
    assert reference_diagonal is not None and grouped_diagonal is not None
    assert float(cupy.max(cupy.abs(
        grouped_diagonal - reference_diagonal
    )).item()) <= 2e-10

    reference_cd = pivoted_cholesky_from_columns(
        reference,
        threshold=1e-8,
        max_rank=8,
        column_batch_size=8,
        transfer_counter=TransferCounter(),
    )
    grouped_cd = pivoted_cholesky_from_columns(
        grouped,
        threshold=1e-8,
        max_rank=8,
        column_batch_size=8,
        transfer_counter=TransferCounter(),
    )
    cupy.cuda.runtime.deviceSynchronize()
    assert grouped_cd.pivots == reference_cd.pivots
    assert grouped_cd.rank == reference_cd.rank
    assert abs(
        grouped_cd.residual_diagonal - reference_cd.residual_diagonal
    ) <= 2e-10

    metadata = grouped.metadata()
    assert metadata["high_rys_workspace"]["maximum_rys_order"] == 7
    assert metadata["high_rys_workspace"]["resident"] is True
    assert metadata["stream_safety"]["per_call_device_synchronize"] is False
    assert metadata["precutoff_column_gout_attempts"] > (
        metadata["column_gout_evaluations"]
    )
    assert metadata["grouped_gout_evaluations_saved"] > 0
    assert metadata["performance_eligible"] is False
