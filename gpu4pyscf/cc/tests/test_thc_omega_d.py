"""Dense audit tests for THC Omega-D paper Algorithm 7."""

import numpy as np
import pytest

from gpu4pyscf.cc.thc_eri import ERITHCFactors
from gpu4pyscf.cc.thc_omega_d import thc_omega_d_algorithm7


def _problem(seed=701, dtype=np.float64):
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


def _pair_projector(y_occ, y_vir):
    return np.einsum("iX,aX->Xia", y_occ, y_vir)


def _dense_amplitudes(y_occ, y_vir, core):
    return np.einsum(
        "iX,aX,XY,jY,bY->ijab",
        y_occ,
        y_vir,
        core,
        y_occ,
        y_vir,
    )


def _dense_ovov(eri):
    return np.einsum(
        "iI,aI,IJ,jJ,bJ->iajb",
        eri.x_occ,
        eri.x_vir,
        eri.core,
        eri.x_occ,
        eri.x_vir,
    )


def _dense_eq36(projector, t2):
    exchange_t2 = t2.swapaxes(2, 3)
    return np.einsum(
        "Xjb,ijab->Xia", projector, 2 * t2 - exchange_t2
    )


def _dense_s_exchange(projector, t2):
    return -np.einsum(
        "Xjb,ijab->Xia", projector, t2.swapaxes(2, 3)
    )


def _dense_eq37(r_intermediate, ovov):
    exchange_ovov = ovov.transpose(0, 3, 2, 1)
    return 0.25 * np.einsum(
        "Xia,Yjb,iajb->XY",
        r_intermediate,
        r_intermediate,
        2 * ovov - exchange_ovov,
    )


def _dense_spurious_removal(s_exchange, ovov):
    exchange_ovov = ovov.transpose(0, 3, 2, 1)
    return 0.25 * np.einsum(
        "Xia,Yjb,iajb->XY",
        s_exchange,
        s_exchange,
        exchange_ovov,
    )


def _literal_dense_eq35(projector, t2, ovov):
    anti_t2 = 2 * t2 - t2.swapaxes(2, 3)
    anti_ovov = 2 * ovov - ovov.transpose(0, 3, 2, 1)
    difference_t2 = t2 - t2.swapaxes(2, 3)
    first = np.einsum(
        "ikac,kcld,jlbd->ijab",
        anti_t2,
        anti_ovov,
        difference_t2,
        optimize=True,
    )
    second = np.einsum(
        "ikca,kdlc,jldb->ijab", t2, ovov, t2, optimize=True
    )
    return 0.25 * np.einsum(
        "Xia,Yjb,ijab->XY", projector, projector, first + second
    )


def test_algorithm7_eq36_and_s_match_dense_nonsymmetric_oracles():
    y_occ, y_vir, amplitude_core, eri = _problem(702)
    assert not np.allclose(amplitude_core, amplitude_core.T)
    assert not np.allclose(eri.core, eri.core.T)
    projector = _pair_projector(y_occ, y_vir)
    t2 = _dense_amplitudes(y_occ, y_vir, amplitude_core)
    result = thc_omega_d_algorithm7(
        y_occ, y_vir, amplitude_core, eri, x_block_size=2
    )

    np.testing.assert_allclose(
        result.r_intermediate,
        _dense_eq36(projector, t2),
        atol=2e-11,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        result.s_exchange,
        _dense_s_exchange(projector, t2),
        atol=2e-11,
        rtol=2e-12,
    )


def test_algorithm7_main_matches_dense_eq37_with_nonsymmetric_cores():
    y_occ, y_vir, amplitude_core, eri = _problem(703)
    assert not np.allclose(amplitude_core, amplitude_core.T)
    assert not np.allclose(eri.core, eri.core.T)
    projector = _pair_projector(y_occ, y_vir)
    t2 = _dense_amplitudes(y_occ, y_vir, amplitude_core)
    ovov = _dense_ovov(eri)
    r_dense = _dense_eq36(projector, t2)

    result = thc_omega_d_algorithm7(y_occ, y_vir, amplitude_core, eri)
    np.testing.assert_allclose(
        result.main_eq37,
        _dense_eq37(r_dense, ovov),
        atol=2e-10,
        rtol=2e-12,
    )


def test_algorithm7_spurious_removal_matches_independent_dense_oracle():
    y_occ, y_vir, amplitude_core, eri = _problem(704)
    assert not np.allclose(amplitude_core, amplitude_core.T)
    assert not np.allclose(eri.core, eri.core.T)
    projector = _pair_projector(y_occ, y_vir)
    t2 = _dense_amplitudes(y_occ, y_vir, amplitude_core)
    ovov = _dense_ovov(eri)
    s_dense = _dense_s_exchange(projector, t2)

    result = thc_omega_d_algorithm7(y_occ, y_vir, amplitude_core, eri)
    np.testing.assert_allclose(
        result.spurious_removal,
        _dense_spurious_removal(s_dense, ovov),
        atol=2e-10,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        result.combined,
        result.main_eq37 + result.spurious_removal,
        atol=0,
        rtol=0,
    )
    assert result.sigma_thc is result.combined


def test_literal_eq35_one_dimensional_counterexample_is_fail_closed():
    rng = np.random.default_rng(705)
    y_occ = rng.normal(size=(1, 1))
    y_vir = rng.normal(size=(1, 1))
    amplitude_core = rng.normal(size=(1, 1))
    eri = ERITHCFactors(
        rng.normal(size=(1, 1)),
        rng.normal(size=(1, 1)),
        rng.normal(size=(1, 1)),
    )
    projector = _pair_projector(y_occ, y_vir)
    t2 = _dense_amplitudes(y_occ, y_vir, amplitude_core)
    literal_eq35 = _literal_dense_eq35(projector, t2, _dense_ovov(eri))
    result = thc_omega_d_algorithm7(y_occ, y_vir, amplitude_core, eri)

    assert not np.allclose(result.combined, literal_eq35, atol=1e-14, rtol=1e-12)
    assert result.metadata()["eq35_literal_equivalence"] is False
    assert result.metadata()["eq35_literal_status"] == "fail-closed"
    with pytest.raises(NotImplementedError, match="literal Eq. 35"):
        thc_omega_d_algorithm7(
            y_occ,
            y_vir,
            amplitude_core,
            eri,
            require_eq35_literal_equivalence=True,
        )


def test_algorithm7_x_blocking_does_not_change_any_output():
    args = _problem(706)
    single = thc_omega_d_algorithm7(*args, x_block_size=1)
    paired = thc_omega_d_algorithm7(*args, x_block_size=2)
    full = thc_omega_d_algorithm7(*args, x_block_size=3)
    oversized = thc_omega_d_algorithm7(*args, x_block_size=9)
    for name in (
        "r_intermediate",
        "s_exchange",
        "main_eq37",
        "spurious_removal",
        "combined",
    ):
        expected = getattr(single, name)
        # BLAS is free to accumulate different block shapes in a different
        # order, so compare at a tight FP64 roundoff tolerance.
        np.testing.assert_allclose(
            getattr(paired, name), expected, atol=5e-13, rtol=5e-14
        )
        np.testing.assert_allclose(
            getattr(full, name), expected, atol=5e-13, rtol=5e-14
        )
        np.testing.assert_allclose(
            getattr(oversized, name), expected, atol=5e-13, rtol=5e-14
        )


def test_algorithm7_metadata_is_explicitly_audit_only():
    result = thc_omega_d_algorithm7(*_problem(707), x_block_size=2)
    metadata = result.metadata()
    assert metadata["paper_algorithm"] == 7
    assert metadata["paper_equations_referenced"] == [35, 36, 37]
    assert metadata["implemented_equations"] == [36, 37]
    assert metadata["unimplemented_equations"] == [35]
    assert metadata["eq35_scope"] == "referenced_but_not_literal_equivalent"
    assert metadata["main_convention"] == "equation-37"
    assert metadata["spurious_removal_is_signed_addend"] is True
    assert metadata["combined_convention"] == "appendix-algorithm-7-line-22"
    assert metadata["eq35_literal_equivalence"] is False
    assert metadata["eq35_literal_status"] == "fail-closed"
    assert "one-dimensional counterexample" in metadata["eq35_literal_reason"]
    assert metadata["requires_rr_back_projection"] is True
    assert metadata["complete_ccsd_residual"] is False
    assert metadata["audit_only"] is True
    assert metadata["production_enabled"] is False
    assert metadata["performance_eligible"] is False
    assert metadata["gpu_performance_claim"] is False
    assert metadata["backend"] == "numpy"
    assert metadata["dtype"] == "float64"
    assert metadata["x_block_size"] == 2
    assert metadata["largest_intermediate_nbytes"] > 0
    assert metadata["largest_intermediate_nbytes_scope"] == (
        "largest-explicit-logical-array"
    )
    assert metadata["peak_memory_requires_profiling"] is True
    assert metadata["materializes_dense_t2"] is False
    assert metadata["materializes_four_index_eri"] is False
    assert metadata["implicit_host_tensor_transfer"] is False
    assert metadata["core_symmetry_enforced"] is False


@pytest.mark.parametrize(
    "mutator,exception,match",
    [
        (
            lambda yo, yv, core, eri: (yo.ravel(), yv, core, eri),
            ValueError,
            "y_occ",
        ),
        (
            lambda yo, yv, core, eri: (yo, yv.ravel(), core, eri),
            ValueError,
            "y_vir",
        ),
        (
            lambda yo, yv, core, eri: (yo, yv, core[:, :2], eri),
            ValueError,
            "amplitude_core must be square",
        ),
        (
            lambda yo, yv, core, eri: (
                yo.astype(np.float32),
                yv,
                core,
                eri,
            ),
            TypeError,
            "same dtype",
        ),
    ],
)
def test_algorithm7_shape_and_dtype_errors_fail_closed(
    mutator, exception, match
):
    with pytest.raises(exception, match=match):
        thc_omega_d_algorithm7(*mutator(*_problem(708)))


def test_algorithm7_scalar_option_and_nonfinite_errors_fail_closed():
    y_occ, y_vir, amplitude_core, eri = _problem(709)
    with pytest.raises(TypeError, match="x_block_size"):
        thc_omega_d_algorithm7(
            y_occ, y_vir, amplitude_core, eri, x_block_size=True
        )
    with pytest.raises(ValueError, match="positive"):
        thc_omega_d_algorithm7(
            y_occ, y_vir, amplitude_core, eri, x_block_size=0
        )
    with pytest.raises(TypeError, match="must be a boolean"):
        thc_omega_d_algorithm7(
            y_occ,
            y_vir,
            amplitude_core,
            eri,
            require_eq35_literal_equivalence=1,
        )
    bad = amplitude_core.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        thc_omega_d_algorithm7(y_occ, y_vir, bad, eri)
    with pytest.raises(TypeError, match="ERITHCFactors"):
        thc_omega_d_algorithm7(y_occ, y_vir, amplitude_core, object())


def test_algorithm7_complex_integer_and_backend_errors_fail_closed():
    y_occ, y_vir, amplitude_core, eri = _problem(710)
    with pytest.raises(NotImplementedError, match="real RHF"):
        complex_eri = ERITHCFactors(
            eri.x_occ.astype(np.complex128),
            eri.x_vir.astype(np.complex128),
            eri.core.astype(np.complex128),
        )
        thc_omega_d_algorithm7(
            y_occ.astype(np.complex128),
            y_vir.astype(np.complex128),
            amplitude_core.astype(np.complex128),
            complex_eri,
        )
    with pytest.raises(TypeError, match="floating-point"):
        integer_eri = ERITHCFactors(
            eri.x_occ.astype(np.int64),
            eri.x_vir.astype(np.int64),
            eri.core.astype(np.int64),
        )
        thc_omega_d_algorithm7(
            y_occ.astype(np.int64),
            y_vir.astype(np.int64),
            amplitude_core.astype(np.int64),
            integer_eri,
        )
    with pytest.raises(TypeError, match="NumPy or CuPy"):
        thc_omega_d_algorithm7(
            y_occ.tolist(), y_vir, amplitude_core, eri
        )


def test_algorithm7_orbital_dimension_error_fails_closed():
    y_occ, y_vir, amplitude_core, eri = _problem(711)
    other = ERITHCFactors(
        np.ones((3, eri.rank)), eri.x_vir.copy(), eri.core.copy()
    )
    with pytest.raises(ValueError, match="orbital dimensions"):
        thc_omega_d_algorithm7(y_occ, y_vir, amplitude_core, other)


def test_algorithm7_float32_preserves_dtype_and_matches_dense_oracles():
    y_occ, y_vir, amplitude_core, eri = _problem(712, np.float32)
    projector = _pair_projector(y_occ, y_vir)
    t2 = _dense_amplitudes(y_occ, y_vir, amplitude_core)
    ovov = _dense_ovov(eri)
    r_dense = _dense_eq36(projector, t2)
    s_dense = _dense_s_exchange(projector, t2)
    result = thc_omega_d_algorithm7(y_occ, y_vir, amplitude_core, eri)

    assert result.combined.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        result.r_intermediate, r_dense, atol=3e-4, rtol=4e-5
    )
    np.testing.assert_allclose(
        result.main_eq37,
        _dense_eq37(r_dense, ovov),
        atol=5e-3,
        rtol=6e-5,
    )
    np.testing.assert_allclose(
        result.spurious_removal,
        _dense_spurious_removal(s_dense, ovov),
        atol=5e-3,
        rtol=6e-5,
    )


def test_algorithm7_never_calls_four_index_eri_reconstruction(monkeypatch):
    y_occ, y_vir, amplitude_core, eri = _problem(713)

    def forbidden_reconstruction(_self):
        raise AssertionError("Algorithm 7 must not form a four-index ERI")

    monkeypatch.setattr(
        ERITHCFactors, "reconstruct_ovov", forbidden_reconstruction
    )
    result = thc_omega_d_algorithm7(y_occ, y_vir, amplitude_core, eri)
    assert result.combined.shape == (amplitude_core.shape[0],) * 2
    assert np.all(np.isfinite(result.combined))


def test_algorithm7_backend_mismatch_fails_closed_with_cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    y_occ, y_vir, amplitude_core, eri = _problem(714)
    with pytest.raises(TypeError, match="one array backend"):
        thc_omega_d_algorithm7(
            cp.asarray(y_occ), y_vir, amplitude_core, eri
        )


def test_algorithm7_cupy_path_matches_numpy_and_records_scalar_reads():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    y_occ, y_vir, amplitude_core, eri = _problem(715)
    expected = thc_omega_d_algorithm7(
        y_occ, y_vir, amplitude_core, eri, x_block_size=2
    )
    transfers = TransferCounter()
    gpu_eri = ERITHCFactors(
        cp.asarray(eri.x_occ),
        cp.asarray(eri.x_vir),
        cp.asarray(eri.core),
        transfer_counter=transfers,
    )
    observed = thc_omega_d_algorithm7(
        cp.asarray(y_occ),
        cp.asarray(y_vir),
        cp.asarray(amplitude_core),
        gpu_eri,
        x_block_size=2,
    )
    for name in (
        "r_intermediate",
        "s_exchange",
        "main_eq37",
        "spurious_removal",
        "combined",
    ):
        value = getattr(observed, name)
        assert isinstance(value, cp.ndarray)
        np.testing.assert_allclose(
            cp.asnumpy(value), getattr(expected, name), atol=2e-10, rtol=2e-12
        )
    report = transfers.to_dict()
    assert report["total_transfers"] == 3
    assert report["by_operation"]["d2h"] == {
        "eri_thc_factor_validation": {"bytes": 1, "count": 1},
        "thc_omega_d_input_validation": {"bytes": 1, "count": 1},
        "thc_omega_d_output_validation": {"bytes": 1, "count": 1},
    }


def test_algorithm7_device_mismatch_fails_closed_with_two_gpus():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("requires two CUDA devices")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    with cp.cuda.Device(0):
        transfers = TransferCounter()
        gpu_eri = ERITHCFactors(
            cp.ones((2, 2)),
            cp.ones((3, 2)),
            cp.eye(2),
            transfer_counter=transfers,
        )
        y_occ = cp.ones((2, 2))
        y_vir = cp.ones((3, 2))
    with cp.cuda.Device(1):
        core = cp.eye(2)
    with pytest.raises(TypeError, match="one device"):
        thc_omega_d_algorithm7(
            y_occ,
            y_vir,
            core,
            gpu_eri,
            transfer_counter=transfers,
        )
