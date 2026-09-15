"""Tests for the weighted occupied-virtual ERI-THC fit."""

import numpy as np
import pytest

from gpu4pyscf.cc.device_runtime import TransferCounter
from gpu4pyscf.cc.thc_eri_factorization import (
    exact_eri_thc_from_cholesky,
    fit_weighted_eri_thc,
)


def _positive_weights(nocc, nvir, dtype=np.float64):
    return (
        np.linspace(0.2, 1.0, nocc, dtype=dtype),
        np.linspace(0.1, 0.9, nvir, dtype=dtype),
    )


def test_exact_pair_endpoint_reconstructs_cholesky_and_ovov():
    rng = np.random.default_rng(811)
    lov = rng.normal(size=(5, 2, 3))
    occupied_weights, virtual_weights = _positive_weights(2, 3)

    result = exact_eri_thc_from_cholesky(
        lov, occupied_weights, virtual_weights
    )

    np.testing.assert_allclose(result.reconstruct_cholesky(), lov, atol=0.0)
    expected_ovov = np.einsum("Aia,Ajb->iajb", lov, lov)
    np.testing.assert_allclose(
        result.factors.reconstruct_ovov(), expected_ovov, atol=2e-13
    )
    assert result.rank == 6
    assert result.exact_full_pair_endpoint is True
    assert result.iterations == 0
    assert result.converged is True


def test_rank_one_weighted_als_recovers_synthetic_cholesky_tensor():
    rng = np.random.default_rng(812)
    x_occ = rng.normal(size=(3, 1))
    x_vir = rng.normal(size=(4, 1))
    xi = rng.normal(size=(6, 1))
    lov = np.einsum("iI,aI,AI->Aia", x_occ, x_vir, xi)
    occupied_weights, virtual_weights = _positive_weights(3, 4)

    result = fit_weighted_eri_thc(
        lov,
        occupied_weights,
        virtual_weights,
        eri_thc_rank=1,
        fit_tolerance=1e-11,
        max_iterations=100,
        als_convergence_tolerance=1e-14,
        ridge=0.0,
        seed=17,
    )

    assert result.converged is True
    assert result.weighted_fit_residual < 1e-11
    assert result.unweighted_fit_residual < 1e-11
    np.testing.assert_allclose(
        result.reconstruct_cholesky(), lov, atol=2e-11, rtol=2e-11
    )
    np.testing.assert_allclose(
        result.factors.core, result.xi.T @ result.xi, atol=1e-13
    )


def test_rank_two_weighted_als_exercises_orbital_normal_equations():
    rng = np.random.default_rng(900)
    nocc, nvir, naux, rank = 4, 5, 7, 2
    x_occ = rng.normal(size=(nocc, rank))
    x_vir = rng.normal(size=(nvir, rank))
    xi = rng.normal(size=(naux, rank))
    lov = np.einsum("iI,aI,AI->Aia", x_occ, x_vir, xi)
    occupied_weights, virtual_weights = _positive_weights(nocc, nvir)

    result = fit_weighted_eri_thc(
        lov,
        occupied_weights,
        virtual_weights,
        eri_thc_rank=rank,
        fit_tolerance=1e-8,
        max_iterations=200,
        als_convergence_tolerance=1e-14,
        ridge=1e-12,
        seed=0,
        auxiliary_block_size=2,
    )

    assert result.iterations > 1
    assert result.converged is True
    assert result.weighted_fit_residual < 1e-8
    assert result.unweighted_fit_residual < 2e-8
    np.testing.assert_allclose(
        result.reconstruct_cholesky(), lov, atol=4e-8, rtol=4e-8
    )


def test_fit_metadata_keeps_weights_and_production_state_explicit():
    rng = np.random.default_rng(813)
    lov = rng.normal(size=(4, 2, 3))
    occupied_weights, virtual_weights = _positive_weights(2, 3)
    result = fit_weighted_eri_thc(
        lov,
        occupied_weights,
        virtual_weights,
        eri_thc_rank=2,
        fit_tolerance=0.0,
        max_iterations=2,
        allow_unconverged=True,
        seed=9,
    )

    metadata = result.metadata()
    assert metadata["least_squares_equations"] == [19, 20, 21]
    assert metadata["orbital_weights"] == "explicit-caller-supplied"
    assert metadata["production_weight_definition"].startswith(
        "sqrt(abs(MP2-natural-occupation"
    )
    assert metadata["audit_only"] is True
    assert metadata["production_enabled"] is False
    assert metadata["fit_time_included_in_benchmark_contract"] is True
    assert metadata["exact_full_pair_endpoint"] is False
    assert metadata["materializes_full_cholesky_during_fit"] is False
    assert metadata["actual_peak_memory_requires_profiling"] is True
    assert metadata["auxiliary_block_size"] == 4
    assert metadata["largest_intermediate_nbytes"] > 0


@pytest.mark.parametrize(
    "field,value,exception,match",
    [
        ("occupied", np.array([1.0]), ValueError, "occupied_weights"),
        ("virtual", np.array([1.0]), ValueError, "virtual_weights"),
        ("rank", 0, ValueError, "eri_thc_rank"),
        ("rank", 1.5, TypeError, "eri_thc_rank"),
        ("iterations", 0, ValueError, "max_iterations"),
        ("tolerance", -1.0, ValueError, "fit_tolerance"),
    ],
)
def test_invalid_fit_controls_fail_closed(field, value, exception, match):
    lov = np.ones((3, 2, 2))
    occupied_weights = np.ones(2)
    virtual_weights = np.ones(2)
    kwargs = {
        "eri_thc_rank": 1,
        "fit_tolerance": 1e-4,
        "max_iterations": 3,
    }
    if field == "occupied":
        occupied_weights = value
    elif field == "virtual":
        virtual_weights = value
    elif field == "rank":
        kwargs["eri_thc_rank"] = value
    elif field == "iterations":
        kwargs["max_iterations"] = value
    elif field == "tolerance":
        kwargs["fit_tolerance"] = value
    with pytest.raises(exception, match=match):
        fit_weighted_eri_thc(
            lov, occupied_weights, virtual_weights, **kwargs
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_nonpositive_or_nonfinite_weights_fail_closed(bad):
    lov = np.ones((3, 2, 2))
    occupied_weights = np.ones(2)
    occupied_weights[0] = bad
    with pytest.raises(ValueError, match="finite|positive"):
        fit_weighted_eri_thc(
            lov,
            occupied_weights,
            np.ones(2),
            eri_thc_rank=1,
            fit_tolerance=1e-4,
        )


def test_zero_target_fails_closed():
    with pytest.raises(ValueError, match="target norm is zero"):
        fit_weighted_eri_thc(
            np.zeros((3, 2, 2)),
            np.ones(2),
            np.ones(2),
            eri_thc_rank=1,
            fit_tolerance=1e-4,
        )


def test_gpu_fit_requires_and_accounts_for_transfer_counter():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    rng = np.random.default_rng(814)
    x_occ = rng.normal(size=(2, 1))
    x_vir = rng.normal(size=(3, 1))
    xi = rng.normal(size=(4, 1))
    lov = np.einsum("iI,aI,AI->Aia", x_occ, x_vir, xi)
    occupied_weights, virtual_weights = _positive_weights(2, 3)
    device = tuple(
        cp.asarray(value)
        for value in (lov, occupied_weights, virtual_weights)
    )

    with pytest.raises(ValueError, match="transfer_counter"):
        fit_weighted_eri_thc(
            *device,
            eri_thc_rank=1,
            fit_tolerance=1e-9,
            max_iterations=30,
        )

    class IncompleteCounter:
        def record_d2h(self, *_args, **_kwargs):
            pass

    with pytest.raises(TypeError, match="record_d2h and record_h2d"):
        fit_weighted_eri_thc(
            *device,
            eri_thc_rank=1,
            fit_tolerance=1e-9,
            max_iterations=30,
            transfer_counter=IncompleteCounter(),
        )

    counter = TransferCounter()
    result = fit_weighted_eri_thc(
        *device,
        eri_thc_rank=1,
        fit_tolerance=1e-9,
        max_iterations=30,
        ridge=0.0,
        seed=23,
        transfer_counter=counter,
    )
    record = counter.to_dict()
    assert isinstance(result.factors.x_occ, cp.ndarray)
    assert record["by_kind"]["d2h"]["count"] == result.host_scalar_reads
    assert record["by_kind"]["h2d"]["count"] == 2


def test_gpu_fit_rejects_arrays_from_different_devices():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("requires two CUDA devices")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    with cp.cuda.Device(0):
        lov = cp.ones((2, 2, 2), dtype=cp.float64)
        occupied_weights = cp.ones(2, dtype=cp.float64)
    with cp.cuda.Device(1):
        virtual_weights = cp.ones(2, dtype=cp.float64)
    with pytest.raises(TypeError, match="one device"):
        fit_weighted_eri_thc(
            lov,
            occupied_weights,
            virtual_weights,
            eri_thc_rank=1,
            fit_tolerance=1e-6,
            transfer_counter=TransferCounter(),
        )
