"""Schedule, ABI, and optional CUDA parity tests for selected GINT columns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpu4pyscf.cc.gint_selected_columns import (
    GINTSelectedAOPairColumnProvider,
    _COLUMNS_SYMBOL,
    _DIAGONAL_SYMBOL,
    _normalise_selected_pairs,
    build_selected_pair_schedule,
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


def test_selected_provider_exposes_resident_pair_indices_without_copy():
    provider = object.__new__(GINTSelectedAOPairColumnProvider)
    pair_i = object()
    pair_j = object()
    provider._device_schedule = {"pair_i": pair_i, "pair_j": pair_j}

    actual_i, actual_j = provider.pair_indices_device

    assert actual_i is pair_i
    assert actual_j is pair_j


def test_selected_cuda_source_and_public_abi_are_bounded_and_fail_closed():
    root = Path(__file__).parents[2]
    source = (root / "lib" / "gint" / "selected_columns.cu").read_text()
    header = (root / "lib" / "gint" / "gint.h").read_text()
    cmake = (root / "lib" / "gint" / "CMakeLists.txt").read_text()
    provider = (root / "cc" / "gint_selected_columns.py").read_text()

    assert _COLUMNS_SYMBOL in source and _COLUMNS_SYMBOL in header
    assert _DIAGONAL_SYMBOL in source and _DIAGONAL_SYMBOL in header
    assert "GINTg0_2e_2d4d" in source
    assert "GINTgout2e" in source
    assert "selected_columns.cu" in cmake
    assert "cudaMalloc" not in source
    assert "performance_eligible = False" in provider
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

    if not hasattr(int4c2e.libgint, _COLUMNS_SYMBOL) or not hasattr(
        int4c2e.libgint, _DIAGONAL_SYMBOL
    ):
        pytest.skip("libgint has not been rebuilt with the selected-pair ABI")
    return cupy


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
