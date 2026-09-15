"""End-to-end tests for the resident RR/CD iteration engine."""

import numpy as np
import pytest

from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.rr_engine import RRCCSDIterationEngine


def _engine_problem(xp=np, seed=301, nocc=2, nvir=3, naux=4):
    rng = np.random.default_rng(seed)
    nmo = nocc + nvir
    raw = rng.normal(size=(naux, nmo, nmo))
    factors = (raw + raw.transpose(0, 2, 1)) * 0.5
    dimension = nocc * nvir
    vectors, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    core = rng.normal(size=(dimension, dimension))
    core = (core + core.T) * 0.5
    t1 = rng.normal(size=(nocc, nvir))
    energies = np.concatenate((
        -np.linspace(1.3, 0.6, nocc),
        np.linspace(0.2, 1.1, nvir),
    ))
    fock = rng.normal(size=(nmo, nmo))
    provider = MOThreeIndexIntegralProvider(
        xp.asarray(factors),
        nocc,
        factorization="cd",
        threshold=1e-8,
        source="unit-test",
    )
    projector = RRProjector(
        xp.asarray(vectors),
        xp.asarray(-np.ones(dimension)),
        0.0,
        dimension,
    )
    doubles = RRDoubles(
        projector, xp.asarray(core), nocc=nocc, nvir=nvir
    )
    engine = RRCCSDIterationEngine(
        provider,
        projector,
        xp.asarray(fock),
        xp.asarray(energies),
        level_shift=0.02,
        auxiliary_block_size=2,
        virtual_block_size=2,
    )
    return engine, xp.asarray(t1), doubles


def test_full_rank_engine_jacobi_solves_both_projected_equations():
    engine, t1, doubles = _engine_problem()
    result = engine.jacobi(t1, doubles)
    expected_t1 = result.singles_equation.amplitudes(engine.eia)
    expected_core = engine.denominator.solve(result.doubles_equation.core)
    np.testing.assert_allclose(result.t1, expected_t1)
    np.testing.assert_allclose(result.doubles.core, expected_core)
    expected_singles_residual = result.singles_equation.numerator - engine.eia * t1
    expected_doubles_residual = result.doubles_equation.core - (
        engine.denominator.matrix @ doubles.core
        + doubles.core @ engine.denominator.matrix
    )
    expected_residual_norm = np.sqrt(
        np.vdot(expected_singles_residual, expected_singles_residual).real
        + np.vdot(expected_doubles_residual, expected_doubles_residual).real
    )
    expected_update_norm = np.sqrt(
        np.vdot(result.t1 - t1, result.t1 - t1).real
        + np.vdot(
            result.doubles.core - doubles.core,
            result.doubles.core - doubles.core,
        ).real
    )
    np.testing.assert_allclose(
        result.equation_residual_t1, expected_singles_residual
    )
    np.testing.assert_allclose(
        result.equation_residual_core, expected_doubles_residual
    )
    np.testing.assert_allclose(result.residual_norm, expected_residual_norm)
    np.testing.assert_allclose(result.update_norm, expected_update_norm)
    assert not np.isclose(result.residual_norm, result.update_norm)
    assert result.singles_equation.materialized_dense_t2 is False
    assert result.doubles_equation.materialized_dense_t2 is False
    assert result.doubles_equation.materialized_four_index_eri is False
    assert engine.metadata()["dense_t2_in_iteration"] is False
    assert engine.metadata()["performance_eligible"] is False


def test_zero_problem_kernel_converges_without_dense_doubles():
    nocc, nvir, naux = 1, 2, 1
    nmo = nocc + nvir
    projector = RRProjector(
        np.eye(nocc * nvir),
        -np.ones(nocc * nvir),
        0.0,
        nocc * nvir,
    )
    provider = MOThreeIndexIntegralProvider(
        np.zeros((naux, nmo, nmo)),
        nocc,
        factorization="cd",
        threshold=1e-8,
        source="zero-problem",
    )
    energies = np.array([-1.0, 0.4, 0.9])
    engine = RRCCSDIterationEngine(
        provider,
        projector,
        np.diag(energies),
        energies,
        auxiliary_block_size=1,
        virtual_block_size=1,
    )
    initial = RRDoubles(projector, np.zeros((2, 2)), nocc, nvir)
    result = engine.kernel(
        np.zeros((nocc, nvir)),
        initial,
        max_cycle=3,
        conv_tol=1e-12,
        conv_tol_normt=1e-12,
    )
    assert result.converged
    assert result.cycles == 2
    assert result.energy == 0.0
    assert result.residual_norm == 0.0
    assert result.update_norm == 0.0
    assert engine.metrics.metadata["performance_eligible"] is False


def test_gpu_engine_one_step_stays_on_device_and_matches_cpu():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    cpu_engine, cpu_t1, cpu_doubles = _engine_problem(
        seed=302, nocc=1, nvir=3, naux=3
    )
    gpu_engine, gpu_t1, gpu_doubles = _engine_problem(
        xp=cp, seed=302, nocc=1, nvir=3, naux=3
    )
    cpu = cpu_engine.jacobi(cpu_t1, cpu_doubles)
    gpu = gpu_engine.jacobi(gpu_t1, gpu_doubles)
    assert isinstance(gpu.t1, cp.ndarray)
    assert isinstance(gpu.doubles.core, cp.ndarray)
    assert isinstance(gpu.residual_norm, cp.ndarray)
    assert isinstance(gpu.energy, cp.ndarray)
    cp.testing.assert_allclose(gpu.t1, cp.asarray(cpu.t1), atol=2e-9)
    cp.testing.assert_allclose(
        gpu.doubles.core, cp.asarray(cpu.doubles.core), atol=3e-8
    )
    assert gpu_engine.metrics.transfers.to_dict()["total_bytes"] > 0


def test_kernel_reports_energy_and_residual_for_returned_diis_state():
    engine, t1, doubles = _engine_problem(seed=303, nocc=1, nvir=2, naux=2)
    result = engine.kernel(
        t1,
        doubles,
        max_cycle=2,
        conv_tol=1e-30,
        conv_tol_normt=1e-30,
        diis_start_cycle=1,
    )
    checked = engine.jacobi(result.t1, result.doubles)
    np.testing.assert_allclose(result.energy, checked.energy)
    np.testing.assert_allclose(result.residual_norm, checked.residual_norm)
    np.testing.assert_allclose(result.update_norm, checked.update_norm)
    np.testing.assert_allclose(result.history[-1]["energy"], checked.energy)
    np.testing.assert_allclose(
        result.history[-1]["residual_norm"], checked.residual_norm
    )
