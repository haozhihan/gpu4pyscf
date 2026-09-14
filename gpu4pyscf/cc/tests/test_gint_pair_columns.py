"""Index-contract and GPU oracle checks for GINT AO-pair columns."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from gpu4pyscf.cc.gint_pair_columns import (
    GINTAOPairColumnProvider,
    SortedAOPairMap,
    _require_single_shell_support,
    original_ao_shell_support,
    restricted_task_span,
    validate_water1_columns,
)
from gpu4pyscf.cc.direct_cd import pivoted_cholesky_from_columns


def _require_cuda():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cupy


def _shell_and_local_ao(ao_locs, ao):
    shell = int(np.searchsorted(ao_locs, ao, side="right") - 1)
    return shell, int(ao - ao_locs[shell])


def _shell_sliced_pair_column(mol, pair_map, pivot):
    """CPU oracle assembled from shell quartets, never an ``nao**4`` tensor."""

    first, second = pair_map.pair(pivot)
    ao_locs = np.asarray(mol.ao_loc_nr(), dtype=np.int64)
    shell_k, local_k = _shell_and_local_ao(ao_locs, first)
    shell_l, local_l = _shell_and_local_ao(ao_locs, second)
    intor = "int2e_cart" if mol.cart else "int2e_sph"
    result = np.empty(pair_map.dimension, dtype=np.float64)
    for shell_i in range(mol.nbas):
        for shell_j in range(shell_i + 1):
            block = mol.intor_by_shell(
                intor, (shell_i, shell_j, shell_k, shell_l)
            )
            for original_i in range(ao_locs[shell_i], ao_locs[shell_i + 1]):
                for original_j in range(
                    ao_locs[shell_j], ao_locs[shell_j + 1]
                ):
                    if original_i < original_j:
                        continue
                    result[pair_map.index(original_i, original_j)] = block[
                        original_i - ao_locs[shell_i],
                        original_j - ao_locs[shell_j],
                        local_k,
                        local_l,
                    ]
    return result


def _shell_sliced_pair_diagonal(mol, pair_map):
    """CPU pair diagonal assembled one shell quartet at a time."""

    ao_locs = np.asarray(mol.ao_loc_nr(), dtype=np.int64)
    intor = "int2e_cart" if mol.cart else "int2e_sph"
    result = np.empty(pair_map.dimension, dtype=np.float64)
    for shell_i in range(mol.nbas):
        for shell_j in range(shell_i + 1):
            block = mol.intor_by_shell(
                intor, (shell_i, shell_j, shell_i, shell_j)
            )
            for original_i in range(ao_locs[shell_i], ao_locs[shell_i + 1]):
                local_i = original_i - ao_locs[shell_i]
                for original_j in range(
                    ao_locs[shell_j], ao_locs[shell_j + 1]
                ):
                    if original_i < original_j:
                        continue
                    local_j = original_j - ao_locs[shell_j]
                    result[pair_map.index(original_i, original_j)] = block[
                        local_i, local_j, local_i, local_j
                    ]
    return result


def _legacy_full_task_column(provider, pivot):
    """Test-only reproduction of the superseded all-cp-by-all-cp path."""

    cupy = provider._cupy
    first, second = provider.pair_map.pair(pivot)
    cart_column = cupy.zeros(
        (provider._sorted_nao, provider._sorted_nao), dtype=np.float64
    )
    for cp_kl_id in range(len(provider.vhfopt.log_qs)):
        k0, k1, l0, l1 = provider._group_bounds(cp_kl_id)
        weight = cupy.outer(
            provider._coeff[k0:k1, first],
            provider._coeff[l0:l1, second],
        )
        if provider._group_i[cp_kl_id] != provider._group_j[cp_kl_id]:
            weight = weight + cupy.outer(
                provider._coeff[k0:k1, second],
                provider._coeff[l0:l1, first],
            )
        for cp_ij_id in range(len(provider.vhfopt.log_qs)):
            block = provider._fill_quartet_block(cp_ij_id, cp_kl_id)
            contribution = cupy.einsum(
                "lkji,kl->ij", block, weight, optimize=True
            )
            i0, i1, j0, j1 = provider._group_bounds(cp_ij_id)
            cart_column[i0:i1, j0:j1] += contribution
            if provider._group_i[cp_ij_id] != provider._group_j[cp_ij_id]:
                cart_column[j0:j1, i0:i1] += contribution.T
    original = provider._coeff.T @ cart_column @ provider._coeff
    return original[provider._pair_i_device, provider._pair_j_device].copy()


def _timed_cuda_call(cupy, function):
    cupy.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    result = function()
    cupy.cuda.get_current_stream().synchronize()
    return result, time.perf_counter() - started


def test_sorted_pair_map_is_row_major_lower_triangle():
    pair_map = SortedAOPairMap(4)
    np.testing.assert_array_equal(
        pair_map.pair_i,
        np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
    )
    np.testing.assert_array_equal(
        pair_map.pair_j,
        np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]),
    )
    assert pair_map.dimension == 10
    for index in range(pair_map.dimension):
        first, second = pair_map.pair(index)
        assert first >= second
        assert pair_map.index(first, second) == index
        assert pair_map.index(second, first) == index
    assert pair_map.pair_i.flags.writeable is False
    assert pair_map.pair_j.flags.writeable is False


@pytest.mark.parametrize("nao", [0, -1, 1.5, True])
def test_sorted_pair_map_rejects_invalid_nao(nao):
    expected = TypeError if isinstance(nao, (float, bool)) else ValueError
    with pytest.raises(expected):
        SortedAOPairMap(nao)


def test_sorted_pair_map_checks_ao_and_pair_bounds():
    pair_map = SortedAOPairMap(3)
    with pytest.raises(IndexError):
        pair_map.pair(6)
    with pytest.raises(IndexError):
        pair_map.index(3, 0)
    with pytest.raises(TypeError):
        pair_map.index(True, 0)


def test_restricted_task_span_reuses_the_original_nonempty_bin_floor():
    bins = np.array([0, 0, 2, 5, 5], dtype=np.int32)
    floors = np.array([0.0, -3.0, -6.0, -9.0])
    log_q = np.array([-3.2, -4.9, -6.1, -7.2, -8.8])
    first = restricted_task_span(3, 0, 1, bins, floors, log_q)
    last = restricted_task_span(3, 4, 5, bins, floors, log_q)
    assert (first.bin_index, first.bin_floor) == (1, -3.0)
    assert (last.bin_index, last.bin_floor) == (2, -6.0)
    assert first.task_count == 1
    assert last.task_count == 1


def test_restricted_task_span_fails_closed_on_invalid_layout_or_span():
    log_q = np.array([-0.1, -1.1, -2.1])
    with pytest.raises(ValueError, match="crosses"):
        restricted_task_span(
            0,
            0,
            2,
            np.array([0, 1, 3]),
            np.array([0.0, -1.0]),
            log_q,
        )
    with pytest.raises(ValueError, match="cover"):
        restricted_task_span(
            0,
            0,
            1,
            np.array([1, 3]),
            np.array([0.0]),
            log_q,
        )
    with pytest.raises(ValueError, match="exceeds"):
        restricted_task_span(
            0,
            0,
            1,
            np.array([0, 3]),
            np.array([-0.2]),
            log_q,
        )


def test_original_ao_shell_support_keeps_every_exact_nonzero_segment():
    coeff = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.5, 0.0, 0.0],
        ]
    )
    support = original_ao_shell_support(coeff, np.array([0, 2, 3, 4]))
    assert support == ((0, 2), (0,), (1,))


def test_original_ao_shell_support_rejects_missing_or_nonfinite_columns():
    with pytest.raises(ValueError, match="no exact nonzero"):
        original_ao_shell_support(np.array([[1.0, 0.0]]), [0, 1])
    with pytest.raises(ValueError, match="finite"):
        original_ao_shell_support(np.array([[np.nan]]), [0, 1])


def test_release_scope_fails_closed_for_multi_shell_original_ao_support():
    _require_single_shell_support(((0,), (2,), (5,)))
    with pytest.raises(NotImplementedError, match="exactly one"):
        _require_single_shell_support(((0,), (1, 4)))


def test_operation_transfer_ledger_counts_each_payload_once():
    from gpu4pyscf.cc.device_runtime import TransferCounter

    provider = object.__new__(GINTAOPairColumnProvider)
    provider.transfer_counter = TransferCounter()
    provider._explicit_transfer_bytes = {"h2d": {}, "d2h": {}}
    provider._explicit_transfer_counts = {"h2d": {}, "d2h": {}}
    provider._logical_device_payloads = {}
    provider._record_data_transfer("h2d", "setup_coeff", 128)
    provider._record_data_transfer("h2d", "setup_coeff", 64)
    provider._record_data_transfer("d2h", "support_metadata", 32, count=2)
    provider._record_logical_device_payload(
        "generated_coeff",
        4096,
        provenance="test-only device-generated matrix",
    )

    ledger = provider.explicit_transfer_ledger()
    assert ledger["directions"]["h2d"] == {
        "bytes": 192,
        "count": 2,
        "operations": {"setup_coeff": {"bytes": 192, "count": 2}},
    }
    assert ledger["directions"]["d2h"]["bytes"] == 32
    assert ledger["total_bytes"] == 224
    assert provider.transfer_counter.total_bytes == 224
    assert provider.transfer_counter.to_dict()["by_operation"]["h2d"] == {
        "setup_coeff": {"bytes": 192, "count": 2}
    }
    assert ledger["logical_device_payloads"]["generated_coeff"] == {
        "resident_bytes": 4096,
        "actual_transfer_bytes": None,
        "classification": "logical-device-resident-payload",
        "provenance": "test-only device-generated matrix",
    }
    assert provider.transfer_counter.total_bytes != 224 + 4096
    assert ledger["audit"]["provider_setup_h2d_complete"] is False


def test_gint_water1_selected_columns_match_pyscf_spherical_eri():
    cupy = _require_cuda()
    from pyscf import gto

    from gpu4pyscf.cc.device_runtime import TransferCounter

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="cc-pvdz",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    transfers = TransferCounter()
    provider = GINTAOPairColumnProvider(
        mol,
        direct_scf_tol=1e-14,
        group_size=8,
        transfer_counter=transfers,
    )
    pair_map = provider.pair_map
    pivots = (
        pair_map.index(0, 0),
        pair_map.index(5, 2),
        pair_map.index(mol.nao_nr() - 1, mol.nao_nr() - 2),
    )
    result = validate_water1_columns(
        provider, pivots=pivots, tolerance=2e-10
    )
    assert result.passed
    assert provider.materializes_pair_matrix is False
    assert provider.metadata()["materializes_four_index_tensor"] is False
    assert provider.metadata()["performance_eligible"] is False
    assert provider.maximum_quartet_elements > 0
    assert provider.metadata()["maximum_quartet_bytes"] == (
        provider.maximum_quartet_elements * 8
    )
    assert type(provider.column(pivots[0])).__module__.split(".", 1)[0] == "cupy"
    transfer_record = transfers.to_dict()
    operations = transfer_record["by_operation"]
    assert operations["h2d"]["provider_setup_vhfopt_log_q"]["bytes"] > 0
    assert operations["h2d"][
        "provider_setup_gint_basis_product_cache"
    ]["bytes"] > 0
    assert operations["h2d"]["provider_setup_pair_indices"]["bytes"] > 0
    assert operations["h2d"]["gint_fill_constant_cache"]["count"] == (
        provider.gint_fill_calls
    )
    assert operations["d2h"][
        "provider_setup_coeff_support_metadata"
    ]["bytes"] > 0
    assert operations["d2h"]["provider_setup_log_q_metadata"]["count"] == len(
        provider.vhfopt.log_qs
    )
    assert operations["d2h"]["validation_oracle_error"]["count"] == len(
        pivots
    )
    kinds = transfer_record["by_kind"]
    assert kinds["host_pivot_control"]["count"] >= len(pivots) + 1
    assert kinds["gint_fill_call_control"]["count"] == provider.gint_fill_calls
    assert kinds["gint_kernel_launch_control"]["count"] == (
        provider.kernel_launches
    )
    metadata = provider.metadata()
    assert metadata["restricted_fill_calls"] == metadata["gint_fill_calls"]
    assert metadata["requested_task_quartets"] < (
        metadata["full_task_quartet_equivalent"]
    )
    assert metadata["diag_block_with_triu"] is True
    assert metadata["performance_gate"] == (
        "pending-water2-end-to-end-and-complete-transfer-audit"
    )
    assert metadata["release_scope"] == (
        "spherical-original-aos-with-single-segmented-shell-support"
    )
    ledger = metadata["explicit_transfers"]
    assert ledger["directions"]["h2d"]["bytes"] > 0
    assert ledger["directions"]["d2h"]["bytes"] > 0
    coeff = ledger["logical_device_payloads"][
        "provider_setup_vhfopt_coeff"
    ]
    assert coeff["resident_bytes"] == provider._coeff.nbytes
    assert coeff["actual_transfer_bytes"] is None
    assert ledger["coverage"]["vhfopt_coeff_resident_payload"] is True
    assert ledger["coverage"]["vhfopt_coeff_h2d_exact"] is False
    assert ledger["coverage"]["gint_per_call_constant_cache_copy"] is True
    assert ledger["coverage"]["cuda_runtime_kernel_argument_marshalling"] is False
    assert ledger["audit"]["provider_setup_h2d_complete"] is False
    assert ledger["complete_for_declared_payloads"] is False
    cupy.get_default_memory_pool().free_all_blocks()


def test_gint_provider_fails_closed_for_cartesian_release_scope():
    _require_cuda()
    from pyscf import gto

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        verbose=0,
    )
    with pytest.raises(NotImplementedError, match="spherical"):
        GINTAOPairColumnProvider(mol)


def test_gint_water1_diagonal_matches_dense_pair_diagonal():
    cupy = _require_cuda()
    from pyscf import gto

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    provider = GINTAOPairColumnProvider(
        mol, direct_scf_tol=1e-14, group_size=8
    )
    eri = mol.intor("int2e_sph").reshape((mol.nao_nr(),) * 4)
    pair_i, pair_j = provider.pair_map.pair_i, provider.pair_map.pair_j
    expected = eri[pair_i, pair_j, pair_i, pair_j]
    cupy.testing.assert_allclose(
        provider.diagonal(), cupy.asarray(expected), atol=2e-10, rtol=1e-11
    )
    first_column_calls = provider.column_calls
    provider.diagonal()
    assert provider.column_calls == first_column_calls
    assert provider.diagonal_calls == 2
    assert provider.metadata()["diagonal_mode"] == "restricted-task-batched"
    assert provider.metadata()["diagonal_task_batches"] > 0
    assert provider.double_restricted_fill_calls == provider.diagonal_task_batches


def test_gint_water1_drives_matrix_free_pivoted_cholesky():
    cupy = _require_cuda()
    from pyscf import gto

    from gpu4pyscf.cc.device_runtime import TransferCounter

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    transfers = TransferCounter()
    provider = GINTAOPairColumnProvider(
        mol,
        direct_scf_tol=1e-14,
        group_size=8,
        transfer_counter=transfers,
    )
    threshold = 1e-10
    result = pivoted_cholesky_from_columns(
        provider,
        threshold=threshold,
        transfer_counter=transfers,
    )
    pair_i, pair_j = provider.pair_map.pair_i, provider.pair_map.pair_j
    eri = mol.intor("int2e_sph").reshape((mol.nao_nr(),) * 4)
    pair_matrix = eri[
        pair_i[:, None],
        pair_j[:, None],
        pair_i[None, :],
        pair_j[None, :],
    ]
    reconstruction = result.pair_factors @ result.pair_factors.T
    cupy.testing.assert_allclose(
        reconstruction,
        cupy.asarray(pair_matrix),
        atol=2e-10,
        rtol=2e-10,
    )
    assert result.capped is False
    assert result.residual_diagonal <= threshold
    assert result.provider_materializes_pair_matrix is False
    assert result.provider_column_calls == provider.column_calls


def test_gint_water2_ccpvtz_restricted_tasks_and_shell_sliced_oracles():
    """Water2 production-shape gate without constructing an ``nao**4`` ERI."""

    cupy = _require_cuda()
    from pyscf import gto

    mol = gto.M(
        atom="""
        O  -1.3305065   0.9990741   0.0000000
        H  -1.0635455   1.9210871   0.0000000
        H  -0.5051326   0.4919421   0.0000000
        O   1.0052144  -0.7512682   0.0000000
        H   0.9469851  -1.3304176  -0.7655752
        H   0.9469851  -1.3304176   0.7655752
        """,
        basis="cc-pvtz",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    provider = GINTAOPairColumnProvider(
        mol, direct_scf_tol=1e-14, group_size=16
    )
    metadata_before = provider.metadata()
    assert metadata_before["original_ao_single_shell_support"] is True
    assert metadata_before["maximum_original_ao_support_shells"] == 1
    assert metadata_before["materializes_pair_matrix"] is False
    assert metadata_before["materializes_four_index_tensor"] is False

    pair_map = provider.pair_map
    representative = (
        pair_map.index(0, 0),
        pair_map.index(mol.nao_nr() // 2, mol.nao_nr() // 3),
        pair_map.index(mol.nao_nr() - 1, mol.nao_nr() - 2),
    )
    column_errors = []
    for pivot in representative:
        expected = _shell_sliced_pair_column(mol, pair_map, pivot)
        actual = provider.column(pivot)
        column_errors.append(
            float(cupy.max(cupy.abs(actual - cupy.asarray(expected))).item())
        )
    assert max(column_errors) <= 2e-10

    expected_diagonal = _shell_sliced_pair_diagonal(mol, pair_map)
    diagonal, diagonal_seconds = _timed_cuda_call(cupy, provider.diagonal)
    diagonal_error = float(
        cupy.max(cupy.abs(diagonal - cupy.asarray(expected_diagonal))).item()
    )
    assert diagonal_error <= 2e-10
    assert provider.metadata()["diagonal_mode"] == "restricted-task-batched"

    timing_pivot = representative[0]
    before_restricted = provider.gint_fill_calls
    restricted_column, restricted_seconds = _timed_cuda_call(
        cupy, lambda: provider.column(timing_pivot)
    )
    restricted_calls = provider.gint_fill_calls - before_restricted
    restricted_maximum_observed_block_bytes = (
        provider.maximum_observed_block_elements * np.dtype(np.float64).itemsize
    )
    before_full = provider.gint_fill_calls
    full_column, full_seconds = _timed_cuda_call(
        cupy, lambda: _legacy_full_task_column(provider, timing_pivot)
    )
    full_calls = provider.gint_fill_calls - before_full
    full_difference = float(
        cupy.max(cupy.abs(restricted_column - full_column)).item()
    )
    assert full_difference <= 2e-10
    assert restricted_calls < full_calls

    report = {
        "case": "water2-cc-pvtz",
        "nao": mol.nao_nr(),
        "npair": pair_map.dimension,
        "task_count": provider.metadata()["task_count"],
        "single_shell_support": True,
        "representative_column_max_abs_errors": column_errors,
        "diagonal_max_abs_error": diagonal_error,
        "restricted_vs_full_max_abs_difference": full_difference,
        "restricted_column_gint_calls": restricted_calls,
        "legacy_full_column_gint_calls": full_calls,
        "restricted_column_seconds": restricted_seconds,
        "legacy_full_column_seconds": full_seconds,
        "batched_diagonal_seconds": diagonal_seconds,
        "diagonal_task_batches": provider.diagonal_task_batches,
        "restricted_maximum_observed_block_bytes": (
            restricted_maximum_observed_block_bytes
        ),
        "test_including_legacy_maximum_observed_block_bytes": provider.metadata()[
            "maximum_observed_block_bytes"
        ],
        "performance_eligible": provider.performance_eligible,
    }
    print("GINT_WATER2_RESTRICTED_TASK_REPORT=" + json.dumps(report, sort_keys=True))
    assert report["performance_eligible"] is False


def test_gint_provider_fails_before_quartet_exceeds_hard_block_limit():
    _require_cuda()
    from pyscf import gto

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    with pytest.raises(MemoryError, match="max_block_bytes"):
        GINTAOPairColumnProvider(
            mol,
            direct_scf_tol=1e-14,
            group_size=8,
            max_block_bytes=8,
        )
