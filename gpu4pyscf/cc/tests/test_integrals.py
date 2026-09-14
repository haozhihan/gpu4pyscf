import numpy as np
import pytest

try:
    from gpu4pyscf.cc.integrals import (
        DenseIntegralProvider,
        MOThreeIndexIntegralProvider,
        ThreeIndexIntegralProvider,
        pivoted_cholesky,
    )
except ImportError:  # Allows the NumPy-only checks without the GPU dependencies.
    import importlib.util
    from pathlib import Path
    import sys

    _path = Path(__file__).parents[1] / "integrals.py"
    _spec = importlib.util.spec_from_file_location("gpu4pyscf_cc_integrals", _path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
    DenseIntegralProvider = _module.DenseIntegralProvider
    MOThreeIndexIntegralProvider = _module.MOThreeIndexIntegralProvider
    ThreeIndexIntegralProvider = _module.ThreeIndexIntegralProvider
    pivoted_cholesky = _module.pivoted_cholesky

from gpu4pyscf.cc.device_runtime import TransferCounter


def test_three_index_reconstructs_dense_eri():
    rng = np.random.default_rng(22)
    factors = rng.normal(size=(4, 4, 6))
    provider = ThreeIndexIntegralProvider(factors)
    expected = np.einsum("pqk,rsk->pqrs", factors, factors)
    np.testing.assert_allclose(provider.to_dense(), expected)
    np.testing.assert_allclose(provider.get(slice(1, 3), 0, slice(None), 2), expected[1:3, 0, :, 2])

    dense = DenseIntegralProvider(expected)
    assert dense.shape == expected.shape
    np.testing.assert_allclose(dense.get(1, 0, slice(None), 2), expected[1, 0, :, 2])


def test_three_index_pair_matrix_layouts_agree():
    rng = np.random.default_rng(9)
    n, naux = 3, 5
    factors = rng.normal(size=(n, n, naux))
    provider_a = ThreeIndexIntegralProvider(factors.reshape(n * n, naux))
    provider_b = ThreeIndexIntegralProvider(factors.transpose(2, 0, 1))
    np.testing.assert_allclose(provider_a.to_dense(), provider_b.to_dense())


def test_pivoted_cholesky_threshold_and_metadata():
    rng = np.random.default_rng(5)
    source = rng.normal(size=(8, 3))
    matrix = source @ source.T
    result = pivoted_cholesky(matrix, threshold=1e-12)
    assert result.rank <= 3
    assert result.pivots.shape == (result.rank,)
    assert result.max_error < 1e-10
    assert result.residual_diagonal <= result.threshold + 1e-12
    np.testing.assert_allclose(result.factor @ result.factor.T, matrix, atol=1e-10)

    truncated = pivoted_cholesky(matrix, threshold=1.0, max_rank=1)
    assert truncated.rank == 1
    assert truncated.max_error > 1e-8
    assert truncated.error == truncated.max_error


def test_pivoted_cholesky_rejects_indefinite_input():
    with np.testing.assert_raises(ValueError):
        pivoted_cholesky(np.diag([1.0, -1e-2]))


def test_pivoted_cholesky_rejects_materially_nonhermitian_input():
    with np.testing.assert_raises(ValueError):
        pivoted_cholesky(np.array([[2.0, 0.2], [0.1, 1.0]]))
    with np.testing.assert_raises(ValueError):
        pivoted_cholesky(
            np.array([[2.0, 0.2 + 0.3j], [0.2 + 0.3j, 1.0]])
        )


def test_pivoted_cholesky_accepts_roundoff_asymmetry():
    source = np.array([[1.0, 0.3], [0.3, 0.8]])
    perturbed = source.copy()
    perturbed[0, 1] += 2e-15
    result = pivoted_cholesky(perturbed, threshold=1e-14)
    np.testing.assert_allclose(result.reconstruction, source, atol=3e-14)


def test_complex_packed_pair_factors_reconstruct_hermitian_full_layout():
    p, q = np.tril_indices(3)
    rng = np.random.default_rng(37)
    packed = rng.normal(size=(len(p), 4)) + 1j * rng.normal(size=(len(p), 4))
    packed[p == q] = packed[p == q].real
    provider = ThreeIndexIntegralProvider(
        packed,
        n_orbitals=3,
        pair_indices=(p, q),
    )
    np.testing.assert_allclose(
        provider.factors.transpose(1, 0, 2), provider.factors.conj()
    )
    transposed = ThreeIndexIntegralProvider(
        packed.T,
        n_orbitals=3,
        pair_indices=(p, q),
    )
    np.testing.assert_allclose(transposed.factors, provider.factors)


def test_resident_mo_provider_blocks_and_ao_transform():
    rng = np.random.default_rng(81)
    raw = rng.normal(size=(5, 4, 3))
    ao_factors = np.einsum("Qpk,Qqk->Qpq", raw, raw)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    expected = np.einsum("mp,Qmn,nq->Qpq", q, ao_factors, q)
    provider = MOThreeIndexIntegralProvider.from_ao_factors(
        ao_factors,
        q,
        2,
        factorization="cd",
        threshold=1e-8,
        auxiliary_block_size=2,
    )
    np.testing.assert_allclose(provider.factors, expected)
    assert provider.L_oo.shape == (5, 2, 2)
    assert provider.L_ov.shape == (5, 2, 2)
    assert provider.L_vv.shape == (5, 2, 2)
    dense = np.einsum("Qpq,Qrs->pqrs", expected, expected)
    np.testing.assert_allclose(provider.get(slice(1, 3), 0, 2, slice(None)), dense[1:3, 0, 2, :])
    pieces = list(provider.iter_auxiliary(2, block="ov"))
    assert [piece.shape[0] for _, piece in pieces] == [2, 2, 1]
    assert provider.metadata()["threshold"] == 1e-8


def test_resident_mo_provider_uses_conjugate_coefficients():
    rng = np.random.default_rng(91)
    raw = rng.normal(size=(3, 3, 2))
    ao_factors = np.einsum("Qpk,Qqk->Qpq", raw, raw)
    coefficient = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    coefficient, _ = np.linalg.qr(coefficient)
    expected = np.einsum(
        "mp,Qmn,nq->Qpq", coefficient.conj(), ao_factors, coefficient
    )
    provider = MOThreeIndexIntegralProvider.from_ao_factors(
        ao_factors,
        coefficient,
        1,
        factorization="custom",
        threshold=None,
    )
    np.testing.assert_allclose(provider.factors, expected)


def test_resident_cd_provider_requires_explicit_threshold():
    with np.testing.assert_raises(ValueError):
        MOThreeIndexIntegralProvider(
            np.ones((2, 3, 3)),
            1,
            factorization="cd",
            threshold=None,
        )


class _FakeGPU4PySCFDF:
    def __init__(self, factors, block_size=2):
        self.factors = factors
        self.xp = type(factors).__module__.split(".", 1)[0]
        if self.xp == "cupy":
            import cupy

            self.xp = cupy
        else:
            self.xp = np
        self.block_size = block_size
        self._cderi = {0: self.xp.empty((factors.shape[0], 1))}
        self.naux = factors.shape[0] + 7
        self.requested_block_size = None

    def loop(self, blksize=None, unpack=True):
        assert unpack is True
        self.requested_block_size = blksize
        size = self.block_size if blksize is None else blksize
        work = self.xp.empty((size,) + self.factors.shape[1:])
        for start in range(0, self.factors.shape[0], size):
            stop = min(start + size, self.factors.shape[0])
            count = stop - start
            work[:count] = self.factors[start:stop]
            yield work[:count], self.xp.empty((count, 1))


def test_resident_mo_provider_streams_gpu4pyscf_df_unpacked_blocks():
    rng = np.random.default_rng(131)
    raw = rng.normal(size=(5, 4, 4))
    ao_factors = (raw + raw.transpose(0, 2, 1)) * 0.5
    coefficient, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    dfobj = _FakeGPU4PySCFDF(ao_factors)
    provider = MOThreeIndexIntegralProvider.from_gpu4pyscf_df(
        dfobj,
        coefficient,
        2,
        auxiliary_block_size=3,
        source="unit-test-mp2fit",
        require_gpu_resident=False,
    )
    expected = np.einsum(
        "mp,Qmn,nq->Qpq", coefficient.conj(), ao_factors, coefficient
    )
    np.testing.assert_allclose(provider.factors, expected)
    assert provider.naux == 5
    assert provider.factorization == "df"
    assert provider.threshold is None
    assert provider.source == "unit-test-mp2fit"
    assert dfobj.requested_block_size == 3


def test_gpu4pyscf_df_adapter_rejects_multiple_device_slices():
    dfobj = _FakeGPU4PySCFDF(np.ones((2, 3, 3)))
    dfobj._cderi[1] = np.ones((1, 1))
    with np.testing.assert_raises(NotImplementedError):
        MOThreeIndexIntegralProvider.from_gpu4pyscf_df(
            dfobj,
            np.eye(3),
            1,
            require_gpu_resident=False,
        )


def test_gpu4pyscf_df_adapter_keeps_factors_resident_and_counts_mo_upload():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    rng = np.random.default_rng(141)
    raw = rng.normal(size=(4, 3, 3))
    ao_host = (raw + raw.transpose(0, 2, 1)) * 0.5
    coefficient, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    dfobj = _FakeGPU4PySCFDF(cp.asarray(ao_host))
    transfers = TransferCounter()
    provider = MOThreeIndexIntegralProvider.from_gpu4pyscf_df(
        dfobj,
        coefficient,
        1,
        auxiliary_block_size=2,
        transfer_counter=transfers,
    )
    expected = np.einsum(
        "mp,Qmn,nq->Qpq", coefficient.conj(), ao_host, coefficient
    )
    assert isinstance(provider.factors, cp.ndarray)
    cp.testing.assert_allclose(provider.factors, cp.asarray(expected))
    transfer_metadata = transfers.to_dict()
    assert transfer_metadata["by_kind"]["h2d"]["bytes"] == coefficient.nbytes
