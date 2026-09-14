"""CPU tests for the matrix-free MP2 RR projector."""

import numpy as np
import pytest

from gpu4pyscf.cc.rr_projector import (
    MP2PairOperator,
    build_rr_projector_lanczos,
    factor_cauchy_denominator,
)
from gpu4pyscf.cc.device_runtime import TransferCounter


def test_cauchy_factor_has_auditable_elementwise_error_bound():
    gaps = np.geomspace(0.2, 30.0, 31)
    result = factor_cauchy_denominator(gaps, tolerance=1e-12)
    exact = 1.0 / (gaps[:, None] + gaps[None, :])
    error = np.max(np.abs(exact - result.factors @ result.factors.T))
    assert result.rank < gaps.size
    assert result.residual_diagonal <= 1e-12
    assert error <= result.max_element_error_bound * (1 + 1e-8)
    assert result.metadata()["capped"] is False


def test_cauchy_factor_reports_rank_cap():
    gaps = np.geomspace(0.1, 100.0, 40)
    result = factor_cauchy_denominator(
        gaps, tolerance=1e-15, max_rank=2
    )
    assert result.rank == 2
    assert result.capped is True
    assert result.residual_diagonal > result.tolerance


def test_mp2_operator_matches_explicit_pair_matrix():
    rng = np.random.default_rng(9)
    lov = rng.normal(size=(7, 3, 4))
    eps_occ = np.array([-1.1, -0.8, -0.5])
    eps_vir = np.array([0.1, 0.4, 0.7, 1.2])
    operator = MP2PairOperator(
        lov,
        eps_occ,
        eps_vir,
        denominator_tolerance=1e-13,
    )
    vectors = rng.normal(size=(12, 3))
    approximate = operator.matmat(vectors)
    exact = operator.materialize(exact_denominator=True) @ vectors
    np.testing.assert_allclose(approximate, exact, atol=2e-11, rtol=2e-11)
    assert operator.metadata()["materializes_pair_matrix"] is False


def test_lanczos_projector_matches_dense_dominant_subspace():
    rng = np.random.default_rng(14)
    lov = rng.normal(size=(5, 3, 5))
    operator = MP2PairOperator(
        lov,
        np.array([-1.0, -0.7, -0.4]),
        np.array([0.2, 0.5, 0.9, 1.4, 2.0]),
        denominator_tolerance=1e-14,
    )
    dense = operator.materialize()
    values = np.linalg.eigvalsh(dense)
    cutoff = np.sort(np.abs(values))[-4] * (1 - 1e-8)
    result = build_rr_projector_lanczos(
        operator,
        rr_eig_cutoff=cutoff,
        initial_rank=5,
        max_rank=8,
        dense_fallback_dimension=0,
        solver_tolerance=1e-12,
    )
    expected_values, expected_vectors = np.linalg.eigh(dense)
    order = np.argsort(np.abs(expected_values))[::-1][:result.projector.rank]
    expected_vectors = expected_vectors[:, order]
    overlap = np.linalg.svd(
        expected_vectors.T @ result.projector.vectors,
        compute_uv=False,
    )
    assert result.projector.rank == 4
    np.testing.assert_allclose(overlap, 1.0, atol=1e-8)
    assert result.max_ritz_residual < 1e-8
    assert result.projector.orthogonality_error() < 1e-10
    assert result.metadata()["projector"]["rank"] == 4


def test_operator_rejects_non_positive_gap():
    with pytest.raises(ValueError, match="gap"):
        MP2PairOperator(
            np.ones((2, 1, 2)),
            np.array([0.5]),
            np.array([0.2, 1.0]),
            denominator_tolerance=1e-10,
        )


def test_lanczos_rejects_incomplete_zero_cutoff_endpoint():
    rng = np.random.default_rng(41)
    operator = MP2PairOperator(
        rng.normal(size=(4, 3, 4)),
        np.array([-1.0, -0.7, -0.4]),
        np.array([0.2, 0.5, 0.9, 1.4]),
        denominator_tolerance=1e-13,
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        build_rr_projector_lanczos(
            operator,
            rr_eig_cutoff=0.0,
            initial_rank=3,
            dense_fallback_dimension=0,
        )


def test_gpu_operator_and_lanczos_stay_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    rng = np.random.default_rng(23)
    transfers = TransferCounter()
    operator = MP2PairOperator(
        cp.asarray(rng.normal(size=(4, 2, 4))),
        cp.asarray([-1.0, -0.6]),
        cp.asarray([0.2, 0.5, 0.9, 1.4]),
        denominator_tolerance=1e-13,
        transfer_counter=transfers,
    )
    vector = cp.asarray(rng.normal(size=8))
    observed = operator.matvec(vector)
    assert isinstance(observed, cp.ndarray)
    cp.testing.assert_allclose(
        observed,
        operator.materialize(exact_denominator=True) @ vector,
        atol=2e-10,
        rtol=2e-10,
    )
    result = build_rr_projector_lanczos(
        operator,
        rr_eig_cutoff=1e-8,
        initial_rank=3,
        max_rank=5,
        dense_fallback_dimension=0,
        solver_tolerance=1e-10,
        allow_incomplete=True,
    )
    assert isinstance(result.projector.vectors, cp.ndarray)
    assert result.solver == "cupyx-eigsh-lanczos"
    assert result.capped is True
    result_metadata = result.metadata(transfer_counter=transfers)
    assert result_metadata["projector"]["rank"] == result.projector.rank
    metadata = operator.metadata()
    assert metadata["denominator"]["preprocessing_d2h_bytes"] > 0
    assert metadata["denominator"]["preprocessing_h2d_bytes"] > 0
    assert metadata["host_scalar_reads"] > 0
