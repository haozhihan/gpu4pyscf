"""Reference tests for the paper-equation weighted projector CP fit."""

import numpy as np
import pytest

from gpu4pyscf.cc.lowrank import RRProjector
from gpu4pyscf.cc.thc_factorization import (
    fit_weighted_thc_projector,
    fit_weighted_thc_projector_adaptive,
)


def _exact_cp_projector(seed=4):
    rng = np.random.default_rng(seed)
    nocc, nvir, rr_rank, thc_rank = 3, 4, 3, 4
    y_occ = rng.normal(size=(nocc, thc_rank))
    y_vir = rng.normal(size=(nvir, thc_rank))
    tau = rng.normal(size=(rr_rank, thc_rank))
    pairs = np.einsum("iW,aW->iaW", y_occ, y_vir).reshape(
        nocc * nvir, thc_rank
    )
    raw = pairs @ tau.T
    overlap = raw.T @ raw
    values, vectors = np.linalg.eigh(overlap)
    inverse_sqrt = (vectors / np.sqrt(values)[None, :]) @ vectors.T
    exact = raw @ inverse_sqrt
    return nocc, nvir, exact


def test_weighted_cp_fit_recovers_semiunitary_cp_projector():
    nocc, nvir, vectors = _exact_cp_projector()
    eigenvalues = np.array([-2.0, -0.8, -0.2])
    projector = RRProjector(
        vectors, eigenvalues, cutoff=0.0, full_dimension=nocc * nvir
    )
    result = fit_weighted_thc_projector(
        projector,
        nocc,
        nvir,
        thc_rank=4,
        fit_tolerance=1e-4,
        orthogonality_cutoff=1e-10,
        max_iterations=1600,
        als_convergence_tolerance=1e-13,
        ridge=1e-13,
        seed=7,
    )
    assert result.converged
    assert result.weighted_fit_residual <= 1e-4
    assert result.als_weighted_fit_residual <= 1e-4
    assert result.orthogonality_error < 1e-9
    np.testing.assert_allclose(
        result.reconstruct_projector().T @ result.reconstruct_projector(),
        np.eye(3),
        atol=1e-9,
    )
    assert result.metadata()["objective_weights"] == "mp2-eigenvalue-squared"
    assert result.metadata()["paper_exact_weighting"] is True


def test_amplitude_core_matches_thc_pair_reconstruction():
    nocc, nvir, vectors = _exact_cp_projector(seed=8)
    projector = RRProjector(
        vectors,
        np.array([-1.0, -0.5, -0.1]),
        cutoff=0.0,
        full_dimension=nocc * nvir,
    )
    factors = fit_weighted_thc_projector(
        projector,
        nocc,
        nvir,
        thc_rank=4,
        fit_tolerance=1e-2,
        orthogonality_cutoff=1e-10,
        max_iterations=2000,
        als_convergence_tolerance=1e-13,
        ridge=1e-13,
        seed=2,
    )
    rr_core = np.arange(9, dtype=float).reshape(3, 3)
    rr_core = (rr_core + rr_core.T) * 0.5
    thc_core = factors.amplitude_core(rr_core)
    khatri_rao = factors.khatri_rao()
    observed = khatri_rao @ thc_core @ khatri_rao.T
    orthogonal_projector = factors.reconstruct_projector()
    expected = orthogonal_projector @ rr_core @ orthogonal_projector.T
    np.testing.assert_allclose(observed, expected, atol=1e-10)


def test_weighted_cp_fit_rejects_unconverged_factors_by_default():
    nocc, nvir, vectors = _exact_cp_projector(seed=13)
    projector = RRProjector(
        vectors,
        np.array([-1.0, -0.5, -0.1]),
        cutoff=0.0,
        full_dimension=nocc * nvir,
    )
    with pytest.raises(RuntimeError, match="did not reach"):
        fit_weighted_thc_projector(
            projector,
            nocc,
            nvir,
            thc_rank=4,
            fit_tolerance=1e-12,
            orthogonality_cutoff=1e-10,
            max_iterations=1,
            seed=3,
        )


def test_full_pair_rank_is_an_exact_deterministic_endpoint():
    rng = np.random.default_rng(19)
    nocc, nvir, rr_rank = 2, 3, 4
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rr_rank)))
    projector = RRProjector(
        vectors,
        -np.linspace(1.0, 0.2, rr_rank),
        cutoff=0.0,
        full_dimension=nocc * nvir,
    )
    result = fit_weighted_thc_projector(
        projector,
        nocc,
        nvir,
        thc_rank=nocc * nvir,
        fit_tolerance=1e-12,
        orthogonality_cutoff=1e-12,
        max_iterations=1,
        seed=999,
    )
    np.testing.assert_allclose(
        result.reconstruct_projector(), vectors, atol=2e-14
    )
    assert result.iterations == 0
    assert result.converged
    assert result.metadata()["analytic_full_pair_endpoint"] is True
    assert result.storage_nbytes == (
        result.y_occ.nbytes
        + result.y_vir.nbytes
        + result.tau.nbytes
        + result.raw_tau.nbytes
    )


def test_thc_rank_cannot_exceed_pair_dimension():
    nocc, nvir, vectors = _exact_cp_projector(seed=20)
    projector = RRProjector(
        vectors,
        np.array([-1.0, -0.5, -0.1]),
        cutoff=0.0,
        full_dimension=nocc * nvir,
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        fit_weighted_thc_projector(
            projector,
            nocc,
            nvir,
            thc_rank=nocc * nvir + 1,
            fit_tolerance=1e-5,
            orthogonality_cutoff=1e-12,
        )


def test_adaptive_rank_reaches_and_records_exact_full_pair_endpoint():
    rng = np.random.default_rng(21)
    nocc, nvir, rr_rank = 2, 3, 4
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rr_rank)))
    projector = RRProjector(
        vectors,
        -np.linspace(1.0, 0.2, rr_rank),
        cutoff=0.0,
        full_dimension=nocc * nvir,
    )
    result = fit_weighted_thc_projector_adaptive(
        projector,
        nocc,
        nvir,
        fit_tolerance=1e-12,
        orthogonality_cutoff=1e-12,
        initial_rank=1,
        rank_growth=1.5,
        max_iterations=1,
        seed=5,
    )
    assert result.thc_rank == nocc * nvir
    assert result.rank_attempts[0] == rr_rank
    assert result.rank_attempts[-1] == nocc * nvir
    assert result.analytic_full_pair_endpoint is True
    assert result.exact_pair_roundoff_gate_passed is True
    np.testing.assert_allclose(result.reconstruct_projector(), vectors, atol=2e-14)


def test_adaptive_rank_fails_closed_at_explicit_cap():
    rng = np.random.default_rng(22)
    nocc, nvir, rr_rank = 2, 3, 4
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rr_rank)))
    projector = RRProjector(
        vectors,
        -np.linspace(1.0, 0.2, rr_rank),
        cutoff=0.0,
        full_dimension=nocc * nvir,
    )
    with pytest.raises(ValueError, match="smaller than the RR rank"):
        fit_weighted_thc_projector_adaptive(
            projector,
            nocc,
            nvir,
            fit_tolerance=1e-14,
            orthogonality_cutoff=1e-12,
            initial_rank=1,
            max_rank=3,
            max_iterations=1,
            seed=5,
        )


def test_weighted_cp_diagnostic_mode_marks_unconverged_result():
    nocc, nvir, vectors = _exact_cp_projector(seed=14)
    projector = RRProjector(
        vectors,
        np.array([-1.0, -0.5, -0.1]),
        cutoff=0.0,
        full_dimension=nocc * nvir,
    )
    result = fit_weighted_thc_projector(
        projector,
        nocc,
        nvir,
        thc_rank=4,
        fit_tolerance=1e-12,
        orthogonality_cutoff=1e-10,
        max_iterations=1,
        seed=3,
        allow_unconverged=True,
    )
    assert result.converged is False
    assert result.weighted_fit_residual > result.fit_tolerance


def test_gpu_weighted_cp_fit_stays_on_device_and_counts_host_control():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    nocc, nvir, vectors = _exact_cp_projector(seed=4)
    projector = RRProjector(
        cp.asarray(vectors),
        cp.asarray([-2.0, -0.8, -0.2]),
        cutoff=0.0,
        full_dimension=nocc * nvir,
    )
    transfers = TransferCounter()
    result = fit_weighted_thc_projector(
        projector,
        nocc,
        nvir,
        thc_rank=4,
        fit_tolerance=1e-4,
        orthogonality_cutoff=1e-10,
        max_iterations=1600,
        als_convergence_tolerance=1e-13,
        ridge=1e-13,
        seed=7,
        transfer_counter=transfers,
    )
    assert isinstance(result.tau, cp.ndarray)
    assert result.host_scalar_reads > 0
    assert transfers.to_dict()["by_kind"]["d2h"]["count"] == (
        result.host_scalar_reads
    )
    assert transfers.to_dict()["by_kind"]["h2d"]["count"] == 2


def test_gpu_full_pair_endpoint_has_no_host_factor_upload():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    rng = np.random.default_rng(23)
    nocc, nvir, rr_rank = 2, 3, 4
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rr_rank)))
    projector = RRProjector(
        cp.asarray(vectors),
        cp.asarray(-np.linspace(1.0, 0.2, rr_rank)),
        cutoff=0.0,
        full_dimension=nocc * nvir,
    )
    transfers = TransferCounter()
    result = fit_weighted_thc_projector(
        projector,
        nocc,
        nvir,
        thc_rank=nocc * nvir,
        fit_tolerance=1e-12,
        orthogonality_cutoff=1e-12,
        max_iterations=1,
        transfer_counter=transfers,
    )
    assert isinstance(result.tau, cp.ndarray)
    cp.testing.assert_allclose(
        result.reconstruct_projector(), projector.vectors, atol=2e-13
    )
    assert transfers.to_dict()["by_kind"].get("h2d", {"count": 0})[
        "count"
    ] == 0
