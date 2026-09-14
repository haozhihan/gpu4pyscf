"""Tests for factorized MP2 natural-occupation weights."""

from dataclasses import replace

import numpy as np
import pytest

from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.thc_factorization import (
    THCProjectorFactors,
    fit_weighted_thc_projector,
)
from gpu4pyscf.cc.thc_mp2_weights import (
    build_factorized_mp2_natural_occupation_weights,
)


def _problem(seed=701, dtype=np.float64):
    rng = np.random.default_rng(seed)
    nocc, nvir, rr_rank = 2, 3, 4
    dimension = nocc * nvir
    vectors, _ = np.linalg.qr(
        rng.normal(size=(dimension, rr_rank)).astype(dtype)
    )
    vectors = vectors.astype(dtype, copy=False)
    eigenvalues = -np.linspace(1.2, 0.2, rr_rank, dtype=dtype)
    projector = RRProjector(
        vectors,
        eigenvalues,
        cutoff=2.5e-4,
        full_dimension=dimension,
        source="mp2-Lov-cauchy-device-lanczos",
    )
    core = rng.normal(size=(rr_rank, rr_rank)).astype(dtype)
    core = (core + core.T) * np.asarray(0.5, dtype=dtype)
    doubles = RRDoubles(projector, core, nocc, nvir)
    fit_tolerance = 1e-12 if np.dtype(dtype) == np.dtype(np.float64) else 2e-5
    factors = fit_weighted_thc_projector(
        projector,
        nocc,
        nvir,
        thc_rank=dimension,
        fit_tolerance=fit_tolerance,
        orthogonality_cutoff=1e-12,
        orthogonality_tolerance=fit_tolerance,
        max_iterations=1,
    )
    return doubles, factors


def _dense_gamma1_oracle(doubles, factors):
    core = factors.tau.T @ doubles.core @ factors.tau
    t2 = np.einsum(
        "iX,aX,XY,jY,bY->ijab",
        factors.y_occ,
        factors.y_vir,
        core,
        factors.y_occ,
        factors.y_vir,
    )
    nocc, _, nvir, _ = t2.shape
    dm1occ = np.zeros((nocc, nocc), dtype=t2.dtype)
    dm1vir = np.zeros((nvir, nvir), dtype=t2.dtype)
    # Mirror gpu4pyscf.mp.mp2._gamma1_intermediates exactly.  Dense t2 is
    # confined to this small-system oracle and is never used by the kernel.
    for i_index in range(nocc):
        t2i = t2[i_index]
        dm1vir += (
            2.0 * np.einsum("jca,jcb->ba", t2i, t2i)
            - np.einsum("jca,jbc->ba", t2i, t2i)
        )
        dm1occ += (
            2.0 * np.einsum("iab,jab->ij", t2i, t2i)
            - np.einsum("iab,jba->ij", t2i, t2i)
        )
    return -dm1occ, dm1vir


def _nontrivial_cp_problem(seed=714):
    """Build an exact non-pair-basis CP projector without running ALS."""

    rng = np.random.default_rng(seed)
    nocc, nvir, rr_rank, thc_rank = 3, 3, 3, 4
    y_occ = rng.normal(size=(nocc, thc_rank))
    y_vir = rng.normal(size=(nvir, thc_rank))
    trial_tau = rng.normal(size=(rr_rank, thc_rank))
    occupied_gram = y_occ.T @ y_occ
    virtual_gram = y_vir.T @ y_vir
    overlap = trial_tau @ (occupied_gram * virtual_gram) @ trial_tau.T
    values, vectors = np.linalg.eigh(overlap)
    inverse_sqrt = (vectors / np.sqrt(values)[None, :]) @ vectors.T
    tau = inverse_sqrt @ trial_tau
    pair_factors = np.einsum("iX,aX->iaX", y_occ, y_vir).reshape(
        nocc * nvir, thc_rank
    )
    projector_vectors = pair_factors @ tau.T
    eigenvalues = -np.linspace(1.1, 0.3, rr_rank)
    projector = RRProjector(
        projector_vectors,
        eigenvalues,
        cutoff=1e-5,
        full_dimension=nocc * nvir,
    )
    core = rng.normal(size=(rr_rank, rr_rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(projector, core, nocc, nvir)
    factors = THCProjectorFactors(
        y_occ=y_occ,
        y_vir=y_vir,
        tau=tau,
        raw_tau=tau.copy(),
        eigenvalues=eigenvalues,
        fit_tolerance=1e-12,
        als_weighted_fit_residual=0.0,
        als_unweighted_fit_residual=0.0,
        weighted_fit_residual=0.0,
        unweighted_fit_residual=0.0,
        orthogonalized_projector_distance=0.0,
        orthogonality_error=float(
            np.linalg.norm(projector_vectors.T @ projector_vectors - np.eye(rr_rank))
        ),
        overlap_min_eigenvalue=float(np.min(values)),
        orthogonality_cutoff=1e-12,
        iterations=0,
        converged=True,
        ridge=0.0,
        seed=seed,
        host_scalar_reads=0,
        rank_attempts=(thc_rank,),
    )
    return doubles, factors


def test_factorized_gamma1_matches_dense_t2_oracle_and_conserves_trace():
    doubles, factors = _problem()
    expected_occ, expected_vir = _dense_gamma1_oracle(doubles, factors)
    result = build_factorized_mp2_natural_occupation_weights(
        doubles,
        factors,
        occupation_change_floor=1e-10,
    )

    np.testing.assert_allclose(result.delta_occ, expected_occ, atol=3e-12)
    np.testing.assert_allclose(result.delta_vir, expected_vir, atol=3e-12)
    np.testing.assert_allclose(result.delta_occ, result.delta_occ.T, atol=0.0)
    np.testing.assert_allclose(result.delta_vir, result.delta_vir.T, atol=0.0)
    assert abs(np.trace(result.delta_occ) + np.trace(result.delta_vir)) < 2e-12
    assert result.particle_trace_error <= result.particle_trace_tolerance


def test_dense_oracle_with_nontrivial_collocation_factors():
    doubles, factors = _nontrivial_cp_problem()
    assert factors.thc_rank < doubles.projector.full_dimension
    expected_occ, expected_vir = _dense_gamma1_oracle(doubles, factors)
    result = build_factorized_mp2_natural_occupation_weights(
        doubles,
        factors,
        occupation_change_floor=1e-10,
    )
    np.testing.assert_allclose(result.delta_occ, expected_occ, atol=2e-11)
    np.testing.assert_allclose(result.delta_vir, expected_vir, atol=2e-11)


def test_eigh_rotations_and_strictly_positive_weights_share_column_order():
    doubles, factors = _problem(seed=702)
    floor = 2e-7
    result = build_factorized_mp2_natural_occupation_weights(
        doubles,
        factors,
        occupation_change_floor=floor,
    )

    reconstructed_occ = (
        result.occupied_rotation
        * result.occupied_occupation_changes[None, :]
    ) @ result.occupied_rotation.T
    reconstructed_vir = (
        result.virtual_rotation
        * result.virtual_occupation_changes[None, :]
    ) @ result.virtual_rotation.T
    np.testing.assert_allclose(reconstructed_occ, result.delta_occ, atol=2e-12)
    np.testing.assert_allclose(reconstructed_vir, result.delta_vir, atol=2e-12)
    np.testing.assert_allclose(
        result.occupied_rotation.T @ result.occupied_rotation,
        np.eye(result.nocc),
        atol=2e-13,
    )
    np.testing.assert_allclose(
        result.virtual_rotation.T @ result.virtual_rotation,
        np.eye(result.nvir),
        atol=2e-13,
    )
    np.testing.assert_allclose(
        result.occupied_weights,
        np.sqrt(np.maximum(np.abs(result.occupied_occupation_changes), floor)),
    )
    np.testing.assert_allclose(
        result.virtual_weights,
        np.sqrt(np.maximum(np.abs(result.virtual_occupation_changes), floor)),
    )
    assert np.all(result.occupied_weights > 0.0)
    assert np.all(result.virtual_weights > 0.0)
    assert np.all(
        np.diff(np.abs(result.occupied_occupation_changes)) <= 0.0
    )
    assert np.all(np.diff(np.abs(result.virtual_occupation_changes)) <= 0.0)
    np.testing.assert_allclose(
        result.occupied_natural_occupations,
        2.0 + result.occupied_occupation_changes,
    )
    assert result.virtual_natural_occupations is result.virtual_occupation_changes


def test_metadata_records_cost_approximation_and_runtime_provenance():
    doubles, factors = _problem(seed=703)
    result = build_factorized_mp2_natural_occupation_weights(
        doubles,
        factors,
        occupation_change_floor=1e-9,
    )
    metadata = result.metadata()

    assert metadata["gamma1_reference"] == (
        "gpu4pyscf.mp.mp2._gamma1_intermediates"
    )
    assert metadata["occupation_density_source"] == (
        "amplitude-THC reconstruction of RR MP2 doubles"
    )
    assert metadata["exact_full_mp2_density"] is False
    assert metadata["rr_cutoff"] == doubles.projector.cutoff
    assert metadata["rr_source"] == doubles.projector.source
    assert metadata["thc_fit_runtime_identity"] == id(factors)
    assert metadata["rr_doubles_runtime_identity"] == id(doubles)
    assert metadata["projector_eigenvalue_identity_verified"] is True
    assert metadata["verified_thc_weighted_fit_residual"] <= (
        factors.fit_tolerance
    )
    assert metadata["depends_on_rr_cutoff"] is True
    assert metadata["depends_on_thc_fit"] is True
    assert metadata["formal_arithmetic_scaling"] == "O(N_thc^4)"
    assert metadata["fit_identity_validation_scaling"] == (
        "O(nocc*nvir*rr_rank*N_thc)"
    )
    assert metadata["largest_logical_intermediate_shape"] == [
        factors.thc_rank,
        factors.thc_rank,
    ]
    assert metadata["largest_logical_intermediate_scaling"] == "O(N_thc^2)"
    assert result.largest_intermediate_nbytes == (
        factors.thc_rank**2 * doubles.core.dtype.itemsize
    )
    assert metadata["dense_mp2_t2_materialized"] is False
    assert metadata["mp2_pair_matrix_materialized"] is False
    assert metadata["implicit_host_tensor_transfer"] is False
    assert metadata["audit_only"] is True
    assert metadata["production_enabled"] is False
    assert metadata["performance_eligible"] is False
    assert metadata["host_scalar_reads"] == 0
    assert result.storage_nbytes == sum(
        array.nbytes
        for array in (
            result.delta_occ,
            result.delta_vir,
            result.occupied_occupation_changes,
            result.virtual_occupation_changes,
            result.occupied_rotation,
            result.virtual_rotation,
            result.occupied_weights,
            result.virtual_weights,
        )
    )
    assert result.is_bound_to(doubles, factors)
    result.validate_binding(doubles, factors)


def test_runtime_binding_rejects_same_shape_replacements():
    doubles, factors = _problem(seed=704)
    result = build_factorized_mp2_natural_occupation_weights(
        doubles,
        factors,
        occupation_change_floor=1e-10,
    )
    replacement_doubles = RRDoubles(
        doubles.projector,
        doubles.core.copy(),
        doubles.nocc,
        doubles.nvir,
    )
    replacement_factors = replace(factors)
    assert not result.is_bound_to(replacement_doubles, factors)
    assert not result.is_bound_to(doubles, replacement_factors)
    with pytest.raises(ValueError, match="do not match"):
        result.validate_binding(replacement_doubles, factors)
    with pytest.raises(ValueError, match="do not match"):
        result.validate_binding(doubles, replacement_factors)


def test_kernel_does_not_call_dense_reconstruction_helpers(monkeypatch):
    doubles, factors = _problem(seed=705)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense reconstruction helper was called")

    monkeypatch.setattr(RRDoubles, "reconstruct_t2", forbidden)
    monkeypatch.setattr(RRDoubles, "reconstruct_pair_matrix", forbidden)
    monkeypatch.setattr(type(factors), "khatri_rao", forbidden)
    monkeypatch.setattr(type(factors), "reconstruct_projector", forbidden)
    monkeypatch.setattr(type(factors), "reconstruct_fitted_projector", forbidden)
    monkeypatch.setattr(type(factors), "amplitude_core", forbidden)
    result = build_factorized_mp2_natural_occupation_weights(
        doubles,
        factors,
        occupation_change_floor=1e-10,
    )
    assert result.delta_occ.shape == (doubles.nocc, doubles.nocc)
    assert result.delta_vir.shape == (doubles.nvir, doubles.nvir)


@pytest.mark.parametrize("floor", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_occupation_change_floor_must_be_finite_and_positive(floor):
    doubles, factors = _problem(seed=706)
    with pytest.raises(ValueError, match="finite and positive"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            factors,
            occupation_change_floor=floor,
        )


@pytest.mark.parametrize("floor", [True, "1e-8", None])
def test_occupation_change_floor_must_be_real(floor):
    doubles, factors = _problem(seed=707)
    with pytest.raises(TypeError, match="real scalar"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            factors,
            occupation_change_floor=floor,
        )


def test_floor_must_be_representable_in_input_dtype():
    doubles, factors = _problem(seed=708, dtype=np.float32)
    with pytest.raises(ValueError, match="input dtype"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            factors,
            occupation_change_floor=1e-100,
        )


def test_shape_dtype_finite_symmetry_and_fit_gates_fail_closed():
    doubles, factors = _problem(seed=709)

    with pytest.raises(TypeError, match="RRDoubles"):
        build_factorized_mp2_natural_occupation_weights(
            object(), factors, occupation_change_floor=1e-8
        )
    with pytest.raises(TypeError, match="THCProjectorFactors"):
        build_factorized_mp2_natural_occupation_weights(
            doubles, object(), occupation_change_floor=1e-8
        )
    with pytest.raises(ValueError, match="orbital dimensions"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            replace(factors, y_occ=np.ones((3, factors.thc_rank))),
            occupation_change_floor=1e-8,
        )
    with pytest.raises(TypeError, match="same dtype"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            replace(factors, y_occ=factors.y_occ.astype(np.float32)),
            occupation_change_floor=1e-8,
        )

    bad_core = doubles.core.copy()
    bad_core[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        build_factorized_mp2_natural_occupation_weights(
            RRDoubles(
                doubles.projector, bad_core, doubles.nocc, doubles.nvir
            ),
            factors,
            occupation_change_floor=1e-8,
        )
    bad_factor = factors.y_vir.copy()
    bad_factor[0, 0] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            replace(factors, y_vir=bad_factor),
            occupation_change_floor=1e-8,
        )
    asymmetric_core = doubles.core.copy()
    asymmetric_core[0, 1] += 0.1
    with pytest.raises(ValueError, match="symmetric"):
        build_factorized_mp2_natural_occupation_weights(
            RRDoubles(
                doubles.projector,
                asymmetric_core,
                doubles.nocc,
                doubles.nvir,
            ),
            factors,
            occupation_change_floor=1e-8,
        )
    with pytest.raises(ValueError, match="converged fit"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            replace(factors, converged=False),
            occupation_change_floor=1e-8,
        )
    with pytest.raises(ValueError, match="residual exceeds"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            replace(
                factors,
                weighted_fit_residual=2.0 * factors.fit_tolerance,
            ),
            occupation_change_floor=1e-8,
        )
    with pytest.raises(ValueError, match="eigenvalue identity"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            replace(factors, eigenvalues=factors.eigenvalues.copy()),
            occupation_change_floor=1e-8,
        )
    wrong_vectors = doubles.projector.vectors.copy()
    wrong_vectors[:, [0, 1]] = wrong_vectors[:, [1, 0]]
    wrong_projector = RRProjector(
        wrong_vectors,
        doubles.projector.eigenvalues,
        cutoff=doubles.projector.cutoff,
        full_dimension=doubles.projector.full_dimension,
        source=doubles.projector.source,
    )
    with pytest.raises(ValueError, match="supplied RR projector identity"):
        build_factorized_mp2_natural_occupation_weights(
            RRDoubles(
                wrong_projector,
                doubles.core,
                doubles.nocc,
                doubles.nvir,
            ),
            factors,
            occupation_change_floor=1e-8,
        )
    with pytest.raises(ValueError, match="semi-unitary"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            replace(factors, orthogonality_error=1.0),
            occupation_change_floor=1e-8,
        )


def test_particle_trace_gate_detects_a_broken_exchange_kernel(monkeypatch):
    import gpu4pyscf.cc.thc_mp2_weights as weights_module

    doubles, factors = _problem(seed=710)
    original = weights_module._virtual_exchange_metric

    def broken_exchange(*args, **kwargs):
        exchange = original(*args, **kwargs)
        exchange[0, 0] += 1.0
        return exchange

    monkeypatch.setattr(weights_module, "_virtual_exchange_metric", broken_exchange)
    with pytest.raises(FloatingPointError, match="particle-number trace"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            factors,
            occupation_change_floor=1e-8,
        )


def test_float32_preserves_dtype_and_matches_dense_oracle():
    doubles, factors = _problem(seed=711, dtype=np.float32)
    expected_occ, expected_vir = _dense_gamma1_oracle(doubles, factors)
    result = build_factorized_mp2_natural_occupation_weights(
        doubles,
        factors,
        occupation_change_floor=1e-6,
    )

    for array in (
        result.delta_occ,
        result.delta_vir,
        result.occupied_occupation_changes,
        result.virtual_occupation_changes,
        result.occupied_rotation,
        result.virtual_rotation,
        result.occupied_weights,
        result.virtual_weights,
    ):
        assert array.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(result.delta_occ, expected_occ, atol=4e-4, rtol=2e-5)
    np.testing.assert_allclose(result.delta_vir, expected_vir, atol=4e-4, rtol=2e-5)


def test_backend_mismatch_fails_closed_when_cupy_is_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    doubles, factors = _problem(seed=712)
    with pytest.raises(TypeError, match="one array backend"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            replace(factors, y_occ=cp.asarray(factors.y_occ)),
            occupation_change_floor=1e-8,
        )


def test_cupy_outputs_stay_on_device_and_control_reads_are_counted():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    cpu_doubles, cpu_factors = _problem(seed=713)
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
    factors = replace(
        cpu_factors,
        y_occ=cp.asarray(cpu_factors.y_occ),
        y_vir=cp.asarray(cpu_factors.y_vir),
        tau=cp.asarray(cpu_factors.tau),
        raw_tau=cp.asarray(cpu_factors.raw_tau),
        eigenvalues=eigenvalues,
    )
    with pytest.raises(ValueError, match="explicit transfer_counter"):
        build_factorized_mp2_natural_occupation_weights(
            doubles, factors, occupation_change_floor=1e-8
        )

    transfers = TransferCounter()
    result = build_factorized_mp2_natural_occupation_weights(
        doubles,
        factors,
        occupation_change_floor=1e-8,
        transfer_counter=transfers,
    )
    for array in (
        result.delta_occ,
        result.delta_vir,
        result.occupied_occupation_changes,
        result.virtual_occupation_changes,
        result.occupied_rotation,
        result.virtual_rotation,
        result.occupied_weights,
        result.virtual_weights,
    ):
        assert isinstance(array, cp.ndarray)
        assert int(array.device.id) == device_id

    assert result.host_scalar_reads == 9
    report = transfers.to_dict()
    assert report["total_transfers"] == result.host_scalar_reads
    assert report["by_kind"]["d2h"]["count"] == result.host_scalar_reads
    assert "h2d" not in report["by_kind"]
    assert report["by_operation"]["d2h"] == {
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
    }
    before = transfers.total_transfers
    metadata = result.metadata()
    assert transfers.total_transfers == before
    assert metadata["backend"] == "cupy"
    expected_occ, expected_vir = _dense_gamma1_oracle(
        cpu_doubles, cpu_factors
    )
    np.testing.assert_allclose(cp.asnumpy(result.delta_occ), expected_occ, atol=3e-12)
    np.testing.assert_allclose(cp.asnumpy(result.delta_vir), expected_vir, atol=3e-12)


def test_cupy_mixed_devices_fail_closed_when_two_devices_are_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("two CUDA devices are required")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    cpu_doubles, cpu_factors = _problem(seed=715)
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
        factors = replace(
            cpu_factors,
            y_occ=cp.asarray(cpu_factors.y_occ),
            y_vir=cp.asarray(cpu_factors.y_vir),
            tau=cp.asarray(cpu_factors.tau),
            raw_tau=cp.asarray(cpu_factors.raw_tau),
            eigenvalues=eigenvalues,
        )
    with cp.cuda.Device(1):
        wrong_device_y_occ = cp.asarray(cpu_factors.y_occ)
    with pytest.raises(TypeError, match="one device"):
        build_factorized_mp2_natural_occupation_weights(
            doubles,
            replace(factors, y_occ=wrong_device_y_occ),
            occupation_change_floor=1e-8,
        )
