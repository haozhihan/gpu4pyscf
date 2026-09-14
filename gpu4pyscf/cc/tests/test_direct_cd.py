"""Dense-oracle checks for matrix-free direct AO-pair Cholesky."""

import numpy as np
import pytest

from gpu4pyscf.cc.direct_cd import (
    DenseAOPairColumnProvider,
    pivoted_cholesky_from_columns,
)


def test_direct_column_cholesky_reconstructs_psd_pair_matrix():
    rng = np.random.default_rng(211)
    source = rng.normal(size=(18, 5))
    pair_matrix = source @ source.T
    provider = DenseAOPairColumnProvider(pair_matrix)
    result = pivoted_cholesky_from_columns(provider, threshold=1e-12)
    reconstruction = result.pair_factors @ result.pair_factors.T
    np.testing.assert_allclose(reconstruction, pair_matrix, atol=2e-10)
    assert result.rank <= source.shape[1]
    assert result.provider_column_calls == result.rank
    assert result.max_element_error_bound <= 1e-12
    assert result.metadata()["algorithm"].startswith("matrix-free-direct")


def test_direct_column_cholesky_reports_explicit_rank_cap():
    rng = np.random.default_rng(223)
    source = rng.normal(size=(16, 7))
    provider = DenseAOPairColumnProvider(source @ source.T)
    result = pivoted_cholesky_from_columns(
        provider, threshold=1e-14, max_rank=2
    )
    assert result.rank == 2
    assert result.capped is True
    assert result.residual_diagonal > result.threshold


def test_batched_direct_column_cholesky_reconstructs_psd_pair_matrix():
    rng = np.random.default_rng(225)
    source = rng.normal(size=(31, 9))
    pair_matrix = source @ source.T
    provider = DenseAOPairColumnProvider(pair_matrix)

    result = pivoted_cholesky_from_columns(
        provider,
        threshold=1e-12,
        column_batch_size=4,
    )

    np.testing.assert_allclose(
        result.pair_factors @ result.pair_factors.T,
        pair_matrix,
        atol=3e-10,
    )
    assert result.residual_diagonal <= result.threshold
    assert result.provider_batch_calls == provider.batch_calls
    assert result.provider_batch_calls < result.provider_column_calls
    assert result.provider_column_calls >= result.rank
    assert result.column_batch_size == 4
    assert result.requested_column_batch_size == 4


def test_batched_direct_column_cholesky_requires_batch_provider():
    class ScalarOnlyProvider:
        dimension = 2
        materializes_pair_matrix = False

        @staticmethod
        def diagonal():
            return np.ones(2)

        @staticmethod
        def column(pivot):
            return np.eye(2)[:, pivot]

    with pytest.raises(TypeError, match="provider.columns"):
        pivoted_cholesky_from_columns(
            ScalarOnlyProvider(),
            threshold=1e-12,
            column_batch_size=2,
        )


def test_batched_direct_column_cholesky_caps_to_provider_limit():
    rng = np.random.default_rng(226)
    source = rng.normal(size=(17, 5))

    class LimitedProvider(DenseAOPairColumnProvider):
        max_batch_size = 2

    provider = LimitedProvider(source @ source.T)
    result = pivoted_cholesky_from_columns(
        provider,
        threshold=1e-12,
        column_batch_size=8,
    )

    np.testing.assert_allclose(
        result.pair_factors @ result.pair_factors.T,
        source @ source.T,
        atol=2e-10,
    )
    assert result.column_batch_size == 2
    assert result.requested_column_batch_size == 8


def test_direct_pair_factors_unpack_to_symmetric_ao_layout():
    rng = np.random.default_rng(227)
    pair_i, pair_j = np.tril_indices(4)
    source = rng.normal(size=(len(pair_i), 3))
    provider = DenseAOPairColumnProvider(source @ source.T)
    result = pivoted_cholesky_from_columns(provider, threshold=1e-12)
    unpacked = result.unpack_symmetric(4, pair_i, pair_j)
    np.testing.assert_allclose(unpacked.transpose(0, 2, 1), unpacked)
    np.testing.assert_allclose(unpacked[:, pair_i, pair_j].T, result.pair_factors)


def test_direct_column_cholesky_rejects_negative_diagonal():
    provider = DenseAOPairColumnProvider(np.diag([1.0, -0.1]))
    with np.testing.assert_raises(ValueError):
        pivoted_cholesky_from_columns(provider, threshold=1e-12)


class _InconsistentColumnProvider:
    dimension = 2
    materializes_pair_matrix = False
    column_calls = 0

    @staticmethod
    def diagonal():
        return np.ones(2)

    def column(self, pivot):
        self.column_calls += 1
        return np.array([10.0, 1.0]) if pivot == 0 else np.array([1.0, 10.0])


def test_direct_column_cholesky_rejects_inconsistent_provider():
    with pytest.raises(ValueError, match="inconsistent"):
        pivoted_cholesky_from_columns(
            _InconsistentColumnProvider(), threshold=1e-12
        )


def test_gpu_direct_column_cholesky_counts_every_control_scalar():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    rng = np.random.default_rng(229)
    source = rng.normal(size=(12, 4))
    provider = DenseAOPairColumnProvider(cp.asarray(source @ source.T))
    transfers = TransferCounter()
    result = pivoted_cholesky_from_columns(
        provider, threshold=1e-12, transfer_counter=transfers
    )
    cp.testing.assert_allclose(
        result.pair_factors @ result.pair_factors.T,
        cp.asarray(source @ source.T),
        atol=2e-10,
    )
    d2h = transfers.to_dict()["by_kind"]["d2h"]
    assert d2h["count"] == result.host_scalar_reads
    assert d2h["bytes"] > 0


def test_gpu_batched_direct_column_cholesky_counts_every_control_read():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    rng = np.random.default_rng(231)
    source = rng.normal(size=(19, 6))
    pair_matrix = source @ source.T
    provider = DenseAOPairColumnProvider(cp.asarray(pair_matrix))
    transfers = TransferCounter()

    result = pivoted_cholesky_from_columns(
        provider,
        threshold=1e-12,
        transfer_counter=transfers,
        column_batch_size=4,
    )

    cp.testing.assert_allclose(
        result.pair_factors @ result.pair_factors.T,
        cp.asarray(pair_matrix),
        atol=2e-10,
    )
    d2h = transfers.to_dict()["by_kind"]["d2h"]
    assert d2h["count"] == result.host_scalar_reads
    assert d2h["bytes"] > 0
    assert result.provider_batch_calls < result.provider_column_calls
