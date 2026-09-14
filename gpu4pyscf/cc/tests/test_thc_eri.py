"""Dense-reference tests for ERI-THC paper Algorithms 4 and 5."""

import numpy as np
import pytest

from gpu4pyscf.cc.thc_eri import (
    ERITHCFactors,
    thc_omega_a_algorithm4,
    thc_omega_c_algorithm5,
)


def _problem(seed=301, dtype=np.float64):
    rng = np.random.default_rng(seed)
    nocc, nvir = 2, 3
    amplitude_rank, eri_rank = 3, 4
    y_occ = rng.normal(size=(nocc, amplitude_rank)).astype(dtype)
    y_vir = rng.normal(size=(nvir, amplitude_rank)).astype(dtype)
    amplitude_core = rng.normal(
        size=(amplitude_rank, amplitude_rank)
    ).astype(dtype)
    x_occ = rng.normal(size=(nocc, eri_rank)).astype(dtype)
    x_vir = rng.normal(size=(nvir, eri_rank)).astype(dtype)
    eri_core = rng.normal(size=(eri_rank, eri_rank)).astype(dtype)
    return y_occ, y_vir, amplitude_core, ERITHCFactors(
        x_occ, x_vir, eri_core
    )


def _dense_amplitudes(y_occ, y_vir, core):
    return np.einsum(
        "iX,aX,XY,jY,bY->ijab",
        y_occ,
        y_vir,
        core,
        y_occ,
        y_vir,
    )


def _project_dense(dense, y_occ, y_vir):
    return np.einsum(
        "iX,aX,ijab,jY,bY->XY",
        y_occ,
        y_vir,
        dense,
        y_occ,
        y_vir,
    )


def test_eri_thc_reconstruction_matches_eq17():
    y_occ, y_vir, _amplitude_core, eri = _problem(302)
    expected = np.einsum(
        "iI,aI,IJ,jJ,bJ->iajb",
        eri.x_occ,
        eri.x_vir,
        eri.core,
        eri.x_occ,
        eri.x_vir,
    )
    np.testing.assert_allclose(eri.reconstruct_ovov(), expected, atol=1e-13)
    assert expected.shape == (
        y_occ.shape[0],
        y_vir.shape[0],
        y_occ.shape[0],
        y_vir.shape[0],
    )


def test_algorithm4_matches_direct_eq32_with_distinct_ranks():
    y_occ, y_vir, amplitude_core, eri = _problem(303)
    t2 = _dense_amplitudes(y_occ, y_vir, amplitude_core)
    ovov = eri.reconstruct_ovov()
    dense = np.einsum("ijcd,klab,kcld->ijab", t2, t2, ovov)
    expected = _project_dense(dense, y_occ, y_vir)

    observed = thc_omega_a_algorithm4(
        y_occ, y_vir, amplitude_core, eri
    )
    np.testing.assert_allclose(observed, expected, atol=2e-10, rtol=2e-12)
    assert observed.shape == (amplitude_core.shape[0],) * 2


def test_algorithm5_matches_direct_eq33_with_nonsymmetric_cores():
    y_occ, y_vir, amplitude_core, eri = _problem(304)
    assert not np.allclose(amplitude_core, amplitude_core.T)
    assert not np.allclose(eri.core, eri.core.T)
    t2 = _dense_amplitudes(y_occ, y_vir, amplitude_core)
    ovov = eri.reconstruct_ovov()
    dense = np.einsum("ikcb,ljad,lckd->ijab", t2, t2, ovov)
    expected = _project_dense(dense, y_occ, y_vir)

    observed = thc_omega_c_algorithm5(
        y_occ, y_vir, amplitude_core, eri
    )
    np.testing.assert_allclose(observed, expected, atol=2e-10, rtol=2e-12)
    assert observed.shape == (amplitude_core.shape[0],) * 2


@pytest.mark.parametrize(
    "algorithm", [thc_omega_a_algorithm4, thc_omega_c_algorithm5]
)
def test_outer_blocking_does_not_change_contraction(algorithm):
    args = _problem(305)
    unblocked = algorithm(*args, outer_block_size=1)
    blocked = algorithm(*args, outer_block_size=2)
    np.testing.assert_array_equal(blocked, unblocked)


def test_metadata_is_explicitly_audit_only_and_production_disabled():
    *_amplitudes, eri = _problem(306)
    metadata = eri.metadata()
    assert metadata["eri_equation"] == 17
    assert metadata["residual_equations"] == [32, 33]
    assert metadata["algorithms"] == [4, 5]
    assert metadata["backend"] == "numpy"
    assert metadata["dtype"] == "float64"
    assert metadata["audit_only"] is True
    assert metadata["production_enabled"] is False
    assert metadata["complete_ccsd_residual"] is False
    assert metadata["output_coordinate_space"] == "amplitude-thc-auxiliary"
    assert metadata["requires_rr_back_projection"] is True
    assert metadata["applies_prefactor"] is False
    assert metadata["applies_permutation_or_symmetrization"] is False
    assert metadata["contribution_convention"] == (
        "raw-projected-paper-equation"
    )
    assert metadata["algorithm_materializes_dense_t2"] is False
    assert metadata["algorithm_materializes_four_index_eri"] is False
    assert metadata["dense_reference_reconstruction_available"] is True
    assert metadata["core_symmetry_enforced"] is False
    assert metadata["gpu_validation_requires_transfer_counter"] is True
    assert metadata["implicit_host_tensor_transfer"] is False


@pytest.mark.parametrize(
    "x_occ,x_vir,core,match",
    [
        (np.ones(3), np.ones((3, 2)), np.eye(2), "x_occ"),
        (np.ones((2, 2)), np.ones(3), np.eye(2), "x_vir"),
        (np.ones((2, 2)), np.ones((3, 2)), np.ones(2), "ERI core"),
        (
            np.ones((2, 2)),
            np.ones((3, 3)),
            np.eye(2),
            "ranks do not match",
        ),
        (
            np.ones((2, 2)),
            np.ones((3, 2)),
            np.ones((2, 3)),
            "square",
        ),
    ],
)
def test_eri_factor_shape_errors_fail_closed(x_occ, x_vir, core, match):
    with pytest.raises(ValueError, match=match):
        ERITHCFactors(x_occ, x_vir, core)


def test_eri_factor_dtype_and_nonfinite_errors_fail_closed():
    with pytest.raises(TypeError, match="NumPy or CuPy"):
        ERITHCFactors([[1.0]], np.ones((1, 1)), np.ones((1, 1)))
    with pytest.raises(TypeError, match="same dtype"):
        ERITHCFactors(
            np.ones((2, 2), dtype=np.float32),
            np.ones((3, 2), dtype=np.float64),
            np.eye(2, dtype=np.float64),
        )
    with pytest.raises(TypeError, match="floating-point"):
        ERITHCFactors(
            np.ones((2, 2), dtype=np.int64),
            np.ones((3, 2), dtype=np.int64),
            np.eye(2, dtype=np.int64),
        )
    with pytest.raises(NotImplementedError, match="real RHF"):
        ERITHCFactors(
            np.ones((2, 2), dtype=np.complex128),
            np.ones((3, 2), dtype=np.complex128),
            np.eye(2, dtype=np.complex128),
        )
    bad = np.eye(2)
    bad[0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ERITHCFactors(np.ones((2, 2)), np.ones((3, 2)), bad)


@pytest.mark.parametrize(
    "algorithm", [thc_omega_a_algorithm4, thc_omega_c_algorithm5]
)
def test_amplitude_contract_errors_fail_closed(algorithm):
    y_occ, y_vir, amplitude_core, eri = _problem(307)
    with pytest.raises(ValueError, match="amplitude_core must be square"):
        algorithm(y_occ, y_vir, amplitude_core[:, :2], eri)
    with pytest.raises(TypeError, match="same dtype"):
        algorithm(y_occ.astype(np.float32), y_vir, amplitude_core, eri)
    bad = y_vir.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        algorithm(y_occ, bad, amplitude_core, eri)
    with pytest.raises(ValueError, match="orbital dimensions"):
        other = ERITHCFactors(
            np.ones((3, eri.rank)), eri.x_vir.copy(), eri.core.copy()
        )
        algorithm(y_occ, y_vir, amplitude_core, other)
    with pytest.raises(TypeError, match="outer_block_size"):
        algorithm(y_occ, y_vir, amplitude_core, eri, outer_block_size=1.5)
    with pytest.raises(ValueError, match="positive"):
        algorithm(y_occ, y_vir, amplitude_core, eri, outer_block_size=0)
    with pytest.raises(TypeError, match="ERITHCFactors"):
        algorithm(y_occ, y_vir, amplitude_core, object())


def test_backend_mismatch_fails_closed_when_cupy_is_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    y_occ, y_vir, amplitude_core, eri = _problem(308)
    with pytest.raises(TypeError, match="one array backend"):
        ERITHCFactors(cp.asarray(eri.x_occ), eri.x_vir, eri.core)
    with pytest.raises(TypeError, match="one array backend"):
        thc_omega_a_algorithm4(
            cp.asarray(y_occ), y_vir, amplitude_core, eri
        )


@pytest.mark.parametrize(
    "algorithm,dense_expression",
    [
        (
            thc_omega_a_algorithm4,
            lambda t2, ovov: np.einsum(
                "ijcd,klab,kcld->ijab", t2, t2, ovov
            ),
        ),
        (
            thc_omega_c_algorithm5,
            lambda t2, ovov: np.einsum(
                "ikcb,ljad,lckd->ijab", t2, t2, ovov
            ),
        ),
    ],
)
def test_float32_contract_preserves_dtype_and_matches_dense_reference(
    algorithm, dense_expression
):
    y_occ, y_vir, amplitude_core, eri = _problem(309, np.float32)
    t2 = _dense_amplitudes(y_occ, y_vir, amplitude_core)
    expected = _project_dense(
        dense_expression(t2, eri.reconstruct_ovov()), y_occ, y_vir
    )
    observed = algorithm(y_occ, y_vir, amplitude_core, eri)
    assert observed.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(observed, expected, atol=5e-3, rtol=2e-5)


@pytest.mark.parametrize(
    "algorithm", [thc_omega_a_algorithm4, thc_omega_c_algorithm5]
)
def test_cupy_path_returns_device_array_and_matches_numpy(algorithm):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    y_occ, y_vir, amplitude_core, eri = _problem(310)
    expected = algorithm(y_occ, y_vir, amplitude_core, eri)
    from gpu4pyscf.cc.device_runtime import TransferCounter

    x_occ = cp.asarray(eri.x_occ)
    x_vir = cp.asarray(eri.x_vir)
    eri_core = cp.asarray(eri.core)
    with pytest.raises(ValueError, match="explicit transfer_counter"):
        ERITHCFactors(x_occ, x_vir, eri_core)
    transfers = TransferCounter()
    gpu_eri = ERITHCFactors(
        x_occ, x_vir, eri_core, transfer_counter=transfers
    )
    observed = algorithm(
        cp.asarray(y_occ),
        cp.asarray(y_vir),
        cp.asarray(amplitude_core),
        gpu_eri,
    )
    assert isinstance(observed, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(observed), expected, atol=2e-10)
    report = transfers.to_dict()
    assert report["total_transfers"] == 4
    assert report["by_operation"]["d2h"] == {
        "eri_thc_factor_validation": {"bytes": 2, "count": 2},
        "eri_thc_amplitude_validation": {"bytes": 1, "count": 1},
        "eri_thc_output_validation": {"bytes": 1, "count": 1},
    }
