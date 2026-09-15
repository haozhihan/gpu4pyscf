"""Tests for the resident compressed-amplitude DIIS implementation."""

import numpy as np
import pytest

from gpu4pyscf.cc.compressed_diis import CompressedDIIS


def test_compressed_diis_extrapolates_without_dense_doubles():
    diis = CompressedDIIS(space=3, minimum_history=2, regularization=1e-14)
    first = diis.update(
        np.array([[1.0, 2.0]]),
        np.eye(2),
        np.array([[1.0, 0.0]]),
        np.zeros((2, 2)),
    )
    assert first.extrapolated is False
    second = diis.update(
        np.array([[3.0, 4.0]]),
        2.0 * np.eye(2),
        np.array([[0.0, 1.0]]),
        np.zeros((2, 2)),
    )
    assert second.extrapolated is True
    np.testing.assert_allclose(second.coefficients.sum(), 1.0)
    np.testing.assert_allclose(second.t1, np.array([[2.0, 3.0]]), atol=1e-12)
    np.testing.assert_allclose(second.core, 1.5 * np.eye(2), atol=1e-12)
    assert diis.metadata()["materializes_dense_t2"] is False


def test_compressed_diis_enforces_fixed_shapes_and_backend():
    diis = CompressedDIIS()
    diis.update(
        np.zeros((2, 3)), np.eye(2), np.ones((2, 3)), np.eye(2)
    )
    with pytest.raises(ValueError, match="shapes changed"):
        diis.update(
            np.zeros((3, 3)), np.eye(2), np.ones((3, 3)), np.eye(2)
        )


def test_gpu_compressed_diis_keeps_history_and_solve_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    diis = CompressedDIIS(space=3, regularization=1e-12)
    diis.update(cp.ones((2, 3)), cp.eye(2), cp.ones((2, 3)), cp.eye(2))
    result = diis.update(
        2 * cp.ones((2, 3)),
        2 * cp.eye(2),
        cp.arange(6.0).reshape(2, 3),
        cp.eye(2) * 2,
    )
    assert isinstance(result.t1, cp.ndarray)
    assert isinstance(result.core, cp.ndarray)
    assert isinstance(result.coefficients, cp.ndarray)
    assert diis.metadata()["backend"] == "cupy"
