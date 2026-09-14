"""Integration tests for MP2-NO weighted ERI-THC preprocessing."""

from dataclasses import replace

import numpy as np
import pytest

from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.thc_eri import (
    ERITHCFactors,
    thc_omega_a_algorithm4,
    thc_omega_ac_algorithm6,
    thc_omega_c_algorithm5,
)
from gpu4pyscf.cc.thc_eri_preprocess import (
    build_weighted_eri_thc_preprocess,
)
from gpu4pyscf.cc.thc_factorization import fit_weighted_thc_projector
from gpu4pyscf.cc.thc_omega_d import thc_omega_d_algorithm7


def _amplitude_problem(seed=901, dtype=np.float64):
    rng = np.random.default_rng(seed)
    nocc, nvir, rr_rank = 2, 3, 4
    pair_dimension = nocc * nvir
    vectors, _ = np.linalg.qr(
        rng.normal(size=(pair_dimension, rr_rank)).astype(dtype)
    )
    vectors = vectors.astype(dtype, copy=False)
    eigenvalues = -np.linspace(1.1, 0.2, rr_rank, dtype=dtype)
    projector = RRProjector(
        vectors,
        eigenvalues,
        cutoff=3e-5,
        full_dimension=pair_dimension,
        source="mp2-Lov-cauchy-device-lanczos",
    )
    core = rng.normal(size=(rr_rank, rr_rank)).astype(dtype)
    core = (core + core.T) * np.asarray(0.5, dtype=dtype)
    doubles = RRDoubles(projector, core, nocc, nvir)
    factor_tolerance = (
        1e-12 if np.dtype(dtype) == np.dtype(np.float64) else 2e-5
    )
    factors = fit_weighted_thc_projector(
        projector,
        nocc,
        nvir,
        thc_rank=pair_dimension,
        fit_tolerance=factor_tolerance,
        orthogonality_cutoff=1e-12,
        orthogonality_tolerance=factor_tolerance,
        max_iterations=1,
    )
    return rng, doubles, factors


def _working_ovov(lov):
    return np.einsum("Aia,Ajb->iajb", lov, lov)


def test_exact_pair_endpoint_back_rotates_nontrivial_natural_orbitals():
    rng, doubles, amplitude_factors = _amplitude_problem()
    lov = rng.normal(size=(5, doubles.nocc, doubles.nvir))

    result = build_weighted_eri_thc_preprocess(
        doubles,
        amplitude_factors,
        lov,
        occupation_change_floor=1e-9,
        eri_thc_rank=doubles.nocc * doubles.nvir,
        fit_tolerance=0.0,
        exact_pair_endpoint=True,
        auxiliary_block_size=2,
    )

    assert isinstance(result.factors, ERITHCFactors)
    assert result.exact_pair_endpoint is True
    assert result.fit.exact_full_pair_endpoint is True
    assert result.rank == doubles.nocc * doubles.nvir
    assert np.max(
        np.abs(result.weights.delta_occ - np.diag(np.diag(result.weights.delta_occ)))
    ) > 1e-4
    assert np.max(
        np.abs(result.weights.delta_vir - np.diag(np.diag(result.weights.delta_vir)))
    ) > 1e-4

    expected_natural = np.einsum(
        "ip,Aia,aq->Apq",
        result.weights.occupied_rotation,
        lov,
        result.weights.virtual_rotation,
    )
    np.testing.assert_allclose(
        result.fit.reconstruct_cholesky(), expected_natural, atol=2e-14
    )
    np.testing.assert_allclose(
        result.factors.x_occ,
        result.weights.occupied_rotation @ result.fit.factors.x_occ,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.factors.x_vir,
        result.weights.virtual_rotation @ result.fit.factors.x_vir,
        atol=0.0,
    )
    assert result.factors.core is result.fit.factors.core
    np.testing.assert_allclose(
        result.reconstruct_working_cholesky(), lov, atol=7e-15
    )
    np.testing.assert_allclose(
        result.factors.reconstruct_ovov(), _working_ovov(lov), atol=2e-13
    )


def test_default_weighted_als_path_recovers_rank_one_working_cholesky():
    rng, doubles, amplitude_factors = _amplitude_problem(seed=902)
    x_occ = rng.normal(size=(doubles.nocc, 1))
    x_vir = rng.normal(size=(doubles.nvir, 1))
    xi = rng.normal(size=(6, 1))
    lov = np.einsum("iI,aI,AI->Aia", x_occ, x_vir, xi)

    result = build_weighted_eri_thc_preprocess(
        doubles,
        amplitude_factors,
        lov,
        occupation_change_floor=2e-8,
        eri_thc_rank=1,
        fit_tolerance=1e-10,
        max_iterations=100,
        als_convergence_tolerance=1e-14,
        ridge=0.0,
        seed=17,
        auxiliary_block_size=2,
    )

    assert result.exact_pair_endpoint is False
    assert result.fit.exact_full_pair_endpoint is False
    assert result.fit.converged is True
    assert result.fit.weighted_fit_residual <= result.fit.fit_tolerance
    assert result.fit.occupied_weights is result.weights.occupied_weights
    assert result.fit.virtual_weights is result.weights.virtual_weights
    np.testing.assert_allclose(
        result.reconstruct_working_cholesky(), lov, atol=2e-10, rtol=2e-10
    )
    np.testing.assert_allclose(
        result.factors.reconstruct_ovov(), _working_ovov(lov), atol=3e-10
    )
    assert result.is_bound_to(doubles, amplitude_factors, lov)
    result.validate_binding(doubles, amplitude_factors, lov)


def test_working_basis_factors_are_direct_algorithm4_to_7_inputs():
    rng, doubles, amplitude_factors = _amplitude_problem(seed=915)
    lov = rng.normal(size=(4, doubles.nocc, doubles.nvir))
    result = build_weighted_eri_thc_preprocess(
        doubles,
        amplitude_factors,
        lov,
        occupation_change_floor=1e-8,
        eri_thc_rank=doubles.nocc * doubles.nvir,
        fit_tolerance=0.0,
        exact_pair_endpoint=True,
    )
    amplitude_core = amplitude_factors.amplitude_core(doubles.core)
    arguments = (
        amplitude_factors.y_occ,
        amplitude_factors.y_vir,
        amplitude_core,
        result.factors,
    )
    omega_a = thc_omega_a_algorithm4(*arguments)
    omega_c = thc_omega_c_algorithm5(*arguments)
    omega_ac = thc_omega_ac_algorithm6(*arguments)
    omega_d = thc_omega_d_algorithm7(*arguments)
    expected_shape = (amplitude_factors.thc_rank,) * 2
    assert omega_a.shape == expected_shape
    assert omega_c.shape == expected_shape
    assert omega_ac.shape == expected_shape
    assert omega_d.combined.shape == expected_shape
    assert np.all(np.isfinite(omega_a))
    assert np.all(np.isfinite(omega_c))
    assert np.all(np.isfinite(omega_ac))
    assert np.all(np.isfinite(omega_d.combined))


def test_full_rank_als_is_not_labeled_analytic_exact_endpoint():
    rng, doubles, amplitude_factors = _amplitude_problem(seed=903)
    lov = rng.normal(size=(4, doubles.nocc, doubles.nvir))
    result = build_weighted_eri_thc_preprocess(
        doubles,
        amplitude_factors,
        lov,
        occupation_change_floor=1e-8,
        eri_thc_rank=doubles.nocc * doubles.nvir,
        fit_tolerance=1e-9,
        exact_pair_endpoint=False,
        max_iterations=20,
        als_convergence_tolerance=1e-14,
        ridge=0.0,
        seed=3,
    )

    metadata = result.metadata()
    assert metadata["fit_method"] == "weighted-cp-als"
    assert metadata["als_controls_applied"] is True
    assert metadata["full_pair_rank"] is True
    assert metadata["full_rank_als"] is True
    assert metadata["analytic_exact_pair_endpoint"] is False
    assert metadata["full_rank_als_claimed_as_analytic_exact"] is False
    assert result.fit.exact_full_pair_endpoint is False
    np.testing.assert_allclose(
        result.reconstruct_working_cholesky(), lov, atol=2e-8, rtol=2e-8
    )


def test_metadata_records_basis_flow_approximation_and_disabled_state():
    rng, doubles, amplitude_factors = _amplitude_problem(seed=904)
    lov = rng.normal(size=(3, doubles.nocc, doubles.nvir))
    result = build_weighted_eri_thc_preprocess(
        doubles,
        amplitude_factors,
        lov,
        occupation_change_floor=4e-9,
        eri_thc_rank=doubles.nocc * doubles.nvir,
        fit_tolerance=0.0,
        exact_pair_endpoint=True,
        auxiliary_block_size=2,
    )
    metadata = result.metadata()

    assert metadata["pipeline"] == [
        "factorized-amplitude-thc-mp2-gamma1",
        "working-mo-to-mp2-natural-orbital-rotation",
        "weighted-cholesky-cp-factorization",
        "eri-factors-back-rotation-to-working-mo",
    ]
    assert metadata["density_source"] == (
        "amplitude-thc-approximate-mp2-gamma1"
    )
    assert metadata["density_is_exact_full_mp2"] is False
    assert metadata["depends_on_rr_cutoff"] is True
    assert metadata["depends_on_amplitude_thc_fit"] is True
    assert metadata["fit_coordinate_space"] == "mp2-natural-orbital"
    assert metadata["output_coordinate_space"] == "working-mo"
    assert metadata["algorithm_consumers"] == [4, 5, 6, 7]
    assert metadata["fit_method"] == "analytic-exact-pair"
    assert metadata["als_controls_applied"] is False
    assert metadata["occupation_change_floor"] == 4e-9
    assert metadata["rr_cutoff"] == doubles.projector.cutoff
    assert metadata["rr_doubles_runtime_identity"] == id(doubles)
    assert metadata["amplitude_thc_fit_runtime_identity"] == id(
        amplitude_factors
    )
    assert metadata["cholesky_runtime_identity"] == id(lov)
    assert metadata["transformed_three_index_cholesky_materialized"] is True
    assert metadata["dense_mp2_t2_materialized"] is False
    assert metadata["mp2_pair_matrix_materialized"] is False
    assert metadata["four_index_eri_materialized"] is False
    assert metadata["implicit_host_tensor_transfer"] is False
    assert metadata["audit_only"] is True
    assert metadata["production_enabled"] is False
    assert metadata["performance_eligible"] is False
    assert metadata["preprocess_host_scalar_reads"] == 0
    assert metadata["host_scalar_reads"] == 0
    assert metadata["weights"]["audit_only"] is True
    assert metadata["natural_orbital_fit"]["audit_only"] is True
    assert metadata["working_eri_factors"]["algorithms"] == [4, 5, 6]
    assert result.largest_intermediate_nbytes >= lov.nbytes


def test_runtime_binding_rejects_same_shape_replacements():
    rng, doubles, amplitude_factors = _amplitude_problem(seed=905)
    lov = rng.normal(size=(3, doubles.nocc, doubles.nvir))
    result = build_weighted_eri_thc_preprocess(
        doubles,
        amplitude_factors,
        lov,
        occupation_change_floor=1e-8,
        eri_thc_rank=doubles.nocc * doubles.nvir,
        fit_tolerance=0.0,
        exact_pair_endpoint=True,
    )
    replacement_doubles = RRDoubles(
        doubles.projector,
        doubles.core.copy(),
        doubles.nocc,
        doubles.nvir,
    )
    replacement_factors = replace(amplitude_factors)
    for sources in (
        (replacement_doubles, amplitude_factors, lov),
        (doubles, replacement_factors, lov),
        (doubles, amplitude_factors, lov.copy()),
    ):
        assert not result.is_bound_to(*sources)
        with pytest.raises(ValueError, match="does not match"):
            result.validate_binding(*sources)


@pytest.mark.parametrize(
    "overrides,exception,match",
    [
        ({"eri_thc_rank": 0}, ValueError, "eri_thc_rank"),
        ({"eri_thc_rank": 7}, ValueError, "eri_thc_rank"),
        ({"eri_thc_rank": 1.5}, TypeError, "eri_thc_rank"),
        ({"fit_tolerance": -1.0}, ValueError, "fit_tolerance"),
        ({"fit_tolerance": np.nan}, ValueError, "fit_tolerance"),
        ({"max_iterations": 0}, ValueError, "max_iterations"),
        ({"als_convergence_tolerance": -1.0}, ValueError, "convergence"),
        ({"ridge": -1.0}, ValueError, "ridge"),
        ({"seed": 1.5}, TypeError, "seed"),
        ({"auxiliary_block_size": 0}, ValueError, "auxiliary_block_size"),
        ({"exact_pair_endpoint": 1}, TypeError, "exact_pair_endpoint"),
    ],
)
def test_invalid_fit_controls_fail_closed(overrides, exception, match):
    rng, doubles, amplitude_factors = _amplitude_problem(seed=906)
    lov = rng.normal(size=(3, doubles.nocc, doubles.nvir))
    kwargs = {
        "occupation_change_floor": 1e-8,
        "eri_thc_rank": 1,
        "fit_tolerance": 1e-8,
    }
    kwargs.update(overrides)
    with pytest.raises(exception, match=match):
        build_weighted_eri_thc_preprocess(
            doubles, amplitude_factors, lov, **kwargs
        )


def test_exact_pair_endpoint_requires_full_rank_and_zero_tolerance():
    rng, doubles, amplitude_factors = _amplitude_problem(seed=907)
    lov = rng.normal(size=(3, doubles.nocc, doubles.nvir))
    with pytest.raises(ValueError, match="rank == nocc.nvir"):
        build_weighted_eri_thc_preprocess(
            doubles,
            amplitude_factors,
            lov,
            occupation_change_floor=1e-8,
            eri_thc_rank=1,
            fit_tolerance=0.0,
            exact_pair_endpoint=True,
        )
    with pytest.raises(ValueError, match="fit_tolerance == 0"):
        build_weighted_eri_thc_preprocess(
            doubles,
            amplitude_factors,
            lov,
            occupation_change_floor=1e-8,
            eri_thc_rank=doubles.nocc * doubles.nvir,
            fit_tolerance=1e-12,
            exact_pair_endpoint=True,
        )


def test_shape_dtype_finite_floor_and_factor_identity_fail_closed():
    rng, doubles, amplitude_factors = _amplitude_problem(seed=908)
    lov = rng.normal(size=(3, doubles.nocc, doubles.nvir))
    base = {
        "occupation_change_floor": 1e-8,
        "eri_thc_rank": 1,
        "fit_tolerance": 1e-8,
    }
    with pytest.raises(ValueError, match="lov must have shape"):
        build_weighted_eri_thc_preprocess(
            doubles, amplitude_factors, lov[0], **base
        )
    with pytest.raises(ValueError, match="orbital dimensions"):
        build_weighted_eri_thc_preprocess(
            doubles,
            amplitude_factors,
            np.ones((3, doubles.nocc + 1, doubles.nvir)),
            **base,
        )
    with pytest.raises(TypeError, match="same dtype"):
        build_weighted_eri_thc_preprocess(
            doubles, amplitude_factors, lov.astype(np.float32), **base
        )
    bad = lov.copy()
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        build_weighted_eri_thc_preprocess(
            doubles, amplitude_factors, bad, **base
        )
    with pytest.raises(ValueError, match="finite and positive"):
        build_weighted_eri_thc_preprocess(
            doubles,
            amplitude_factors,
            lov,
            **{**base, "occupation_change_floor": 0.0},
        )
    with pytest.raises(ValueError, match="eigenvalue identity"):
        build_weighted_eri_thc_preprocess(
            doubles,
            replace(
                amplitude_factors,
                eigenvalues=amplitude_factors.eigenvalues.copy(),
            ),
            lov,
            **base,
        )


def test_fit_tolerance_failure_is_not_returned_as_a_result():
    rng, doubles, amplitude_factors = _amplitude_problem(seed=909)
    lov = rng.normal(size=(4, doubles.nocc, doubles.nvir))
    with pytest.raises(RuntimeError, match="did not reach fit_tolerance"):
        build_weighted_eri_thc_preprocess(
            doubles,
            amplitude_factors,
            lov,
            occupation_change_floor=1e-8,
            eri_thc_rank=1,
            fit_tolerance=0.0,
            max_iterations=1,
        )


def test_nonfinite_natural_orbital_transform_fails_before_fit(monkeypatch):
    import gpu4pyscf.cc.thc_eri_preprocess as preprocess_module

    rng, doubles, amplitude_factors = _amplitude_problem(seed=916)
    lov = rng.normal(size=(3, doubles.nocc, doubles.nvir))

    def broken_transform(source, *_args, **_kwargs):
        transformed = source.copy()
        transformed[0, 0, 0] = np.inf
        return transformed, int(transformed.nbytes)

    monkeypatch.setattr(
        preprocess_module,
        "_transform_cholesky_to_natural_orbitals",
        broken_transform,
    )
    with pytest.raises(FloatingPointError, match="produced non-finite"):
        build_weighted_eri_thc_preprocess(
            doubles,
            amplitude_factors,
            lov,
            occupation_change_floor=1e-8,
            eri_thc_rank=1,
            fit_tolerance=1e-8,
        )


def test_float32_exact_endpoint_preserves_dtype():
    rng, doubles, amplitude_factors = _amplitude_problem(
        seed=910, dtype=np.float32
    )
    lov = rng.normal(size=(3, doubles.nocc, doubles.nvir)).astype(np.float32)
    result = build_weighted_eri_thc_preprocess(
        doubles,
        amplitude_factors,
        lov,
        occupation_change_floor=1e-6,
        eri_thc_rank=doubles.nocc * doubles.nvir,
        fit_tolerance=0.0,
        exact_pair_endpoint=True,
    )

    for array in (
        result.factors.x_occ,
        result.factors.x_vir,
        result.factors.core,
        result.fit.factors.x_occ,
        result.fit.factors.x_vir,
        result.fit.xi,
        result.weights.occupied_weights,
        result.weights.virtual_weights,
    ):
        assert array.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        result.reconstruct_working_cholesky(), lov, atol=2e-5, rtol=2e-5
    )


def test_preprocess_does_not_call_dense_t2_or_four_index_helpers(monkeypatch):
    rng, doubles, amplitude_factors = _amplitude_problem(seed=911)
    lov = rng.normal(size=(3, doubles.nocc, doubles.nvir))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense reconstruction helper was called")

    monkeypatch.setattr(RRDoubles, "reconstruct_t2", forbidden)
    monkeypatch.setattr(RRDoubles, "reconstruct_pair_matrix", forbidden)
    monkeypatch.setattr(ERITHCFactors, "reconstruct_ovov", forbidden)
    result = build_weighted_eri_thc_preprocess(
        doubles,
        amplitude_factors,
        lov,
        occupation_change_floor=1e-8,
        eri_thc_rank=doubles.nocc * doubles.nvir,
        fit_tolerance=0.0,
        exact_pair_endpoint=True,
    )
    assert result.factors.core.shape == (result.rank, result.rank)


def test_backend_mismatch_fails_closed_when_cupy_is_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    rng, doubles, amplitude_factors = _amplitude_problem(seed=912)
    lov = rng.normal(size=(3, doubles.nocc, doubles.nvir))
    with pytest.raises(TypeError, match="one array backend"):
        build_weighted_eri_thc_preprocess(
            doubles,
            amplitude_factors,
            cp.asarray(lov),
            occupation_change_floor=1e-8,
            eri_thc_rank=1,
            fit_tolerance=1e-8,
        )


def test_cupy_exact_endpoint_stays_on_device_and_accounts_control_reads():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    rng, cpu_doubles, cpu_amplitude_factors = _amplitude_problem(seed=913)
    cpu_lov = rng.normal(
        size=(3, cpu_doubles.nocc, cpu_doubles.nvir)
    )
    device_id = int(cp.cuda.Device().id)
    eigenvalues = cp.asarray(cpu_doubles.projector.eigenvalues)
    projector = RRProjector(
        cp.asarray(cpu_doubles.projector.vectors),
        eigenvalues,
        cutoff=cpu_doubles.projector.cutoff,
        full_dimension=cpu_doubles.projector.full_dimension,
        source=cpu_doubles.projector.source,
    )
    doubles = RRDoubles(
        projector,
        cp.asarray(cpu_doubles.core),
        cpu_doubles.nocc,
        cpu_doubles.nvir,
    )
    amplitude_factors = replace(
        cpu_amplitude_factors,
        y_occ=cp.asarray(cpu_amplitude_factors.y_occ),
        y_vir=cp.asarray(cpu_amplitude_factors.y_vir),
        tau=cp.asarray(cpu_amplitude_factors.tau),
        raw_tau=cp.asarray(cpu_amplitude_factors.raw_tau),
        eigenvalues=eigenvalues,
    )
    lov = cp.asarray(cpu_lov)
    with pytest.raises(ValueError, match="explicit transfer_counter"):
        build_weighted_eri_thc_preprocess(
            doubles,
            amplitude_factors,
            lov,
            occupation_change_floor=1e-8,
            eri_thc_rank=doubles.nocc * doubles.nvir,
            fit_tolerance=0.0,
            exact_pair_endpoint=True,
        )

    class IncompleteCounter:
        def record_d2h(self, *_args, **_kwargs):
            pass

    with pytest.raises(TypeError, match="record_d2h and record_h2d"):
        build_weighted_eri_thc_preprocess(
            doubles,
            amplitude_factors,
            lov,
            occupation_change_floor=1e-8,
            eri_thc_rank=doubles.nocc * doubles.nvir,
            fit_tolerance=0.0,
            exact_pair_endpoint=True,
            transfer_counter=IncompleteCounter(),
        )

    transfers = TransferCounter()
    result = build_weighted_eri_thc_preprocess(
        doubles,
        amplitude_factors,
        lov,
        occupation_change_floor=1e-8,
        eri_thc_rank=doubles.nocc * doubles.nvir,
        fit_tolerance=0.0,
        exact_pair_endpoint=True,
        auxiliary_block_size=2,
        transfer_counter=transfers,
    )
    for array in (
        result.factors.x_occ,
        result.factors.x_vir,
        result.factors.core,
        result.fit.factors.x_occ,
        result.fit.factors.x_vir,
        result.fit.xi,
        result.weights.occupied_rotation,
        result.weights.virtual_rotation,
        result.weights.occupied_weights,
        result.weights.virtual_weights,
    ):
        assert isinstance(array, cp.ndarray)
        assert int(array.device.id) == device_id

    assert result.preprocess_host_scalar_reads == 3
    assert result.weights.host_scalar_reads == 9
    assert result.fit.host_scalar_reads == 3
    assert result.host_scalar_reads == 15
    report = transfers.to_dict()
    assert report["total_transfers"] == result.host_scalar_reads
    assert report["by_kind"]["d2h"]["count"] == result.host_scalar_reads
    assert "h2d" not in report["by_kind"]
    assert report["by_operation"]["d2h"] == {
        "eri_thc_preprocess_input_finite": {"bytes": 1, "count": 1},
        "thc_mp2_weights_input_finite": {"bytes": 1, "count": 1},
        "thc_mp2_weights_minimum_projector_eigenvalue": {
            "bytes": 8,
            "count": 1,
        },
        "thc_mp2_weights_fit_identity_residual": {"bytes": 8, "count": 1},
        "thc_mp2_weights_core_symmetry": {"bytes": 8, "count": 1},
        "thc_mp2_weights_factor_orthogonality": {"bytes": 8, "count": 1},
        "thc_mp2_weights_particle_trace_error": {"bytes": 8, "count": 1},
        "thc_mp2_weights_particle_trace_scale": {"bytes": 8, "count": 1},
        "thc_mp2_weights_output_finite": {"bytes": 1, "count": 1},
        "thc_mp2_weights_minimum_weight": {"bytes": 8, "count": 1},
        "eri_thc_preprocess_rotated_cholesky_finite": {
            "bytes": 1,
            "count": 1,
        },
        "eri_thc_fit_input_finite": {"bytes": 1, "count": 1},
        "eri_thc_fit_minimum_weight": {"bytes": 8, "count": 1},
        "eri_thc_factor_validation": {"bytes": 2, "count": 2},
    }
    before = transfers.total_transfers
    metadata = result.metadata()
    assert transfers.total_transfers == before
    assert metadata["backend"] == "cupy"
    np.testing.assert_allclose(
        cp.asnumpy(result.reconstruct_working_cholesky()),
        cpu_lov,
        atol=2e-13,
    )


def test_cupy_mixed_devices_fail_closed_when_two_devices_are_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("two CUDA devices are required")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    rng, cpu_doubles, cpu_amplitude_factors = _amplitude_problem(seed=914)
    with cp.cuda.Device(0):
        eigenvalues = cp.asarray(cpu_doubles.projector.eigenvalues)
        projector = RRProjector(
            cp.asarray(cpu_doubles.projector.vectors),
            eigenvalues,
            cutoff=cpu_doubles.projector.cutoff,
            full_dimension=cpu_doubles.projector.full_dimension,
            source=cpu_doubles.projector.source,
        )
        doubles = RRDoubles(
            projector,
            cp.asarray(cpu_doubles.core),
            cpu_doubles.nocc,
            cpu_doubles.nvir,
        )
        amplitude_factors = replace(
            cpu_amplitude_factors,
            y_occ=cp.asarray(cpu_amplitude_factors.y_occ),
            y_vir=cp.asarray(cpu_amplitude_factors.y_vir),
            tau=cp.asarray(cpu_amplitude_factors.tau),
            raw_tau=cp.asarray(cpu_amplitude_factors.raw_tau),
            eigenvalues=eigenvalues,
        )
    with cp.cuda.Device(1):
        lov = cp.asarray(
            rng.normal(size=(3, doubles.nocc, doubles.nvir))
        )
    with pytest.raises(TypeError, match="one device"):
        build_weighted_eri_thc_preprocess(
            doubles,
            amplitude_factors,
            lov,
            occupation_change_floor=1e-8,
            eri_thc_rank=1,
            fit_tolerance=1e-8,
            transfer_counter=TransferCounter(),
        )
