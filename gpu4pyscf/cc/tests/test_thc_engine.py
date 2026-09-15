"""Equation and lifecycle tests for fail-closed RR/amplitude-THC G5."""

from dataclasses import replace

import numpy as np
import pytest

from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.rr_engine import RRCCSDIterationEngine
from gpu4pyscf.cc.rr_residual import (
    build_projected_ccsd_doubles_numerator,
)
from gpu4pyscf.cc.thc_engine import (
    THCRRCCSDIterationEngine,
    build_hybrid_projected_ccsd_doubles_numerator,
    diagnose_reconstructed_full_space_residual,
)
from gpu4pyscf.cc.thc_factorization import fit_weighted_thc_projector


VALIDATION_ATOL = 1e-11
VALIDATION_RTOL = 1e-11


def _problem(seed=401, nocc=2, nvir=3, naux=3, rank=4):
    rng = np.random.default_rng(seed)
    nmo = nocc + nvir
    raw = rng.normal(size=(naux, nmo, nmo))
    factors = (raw + raw.transpose(0, 2, 1)) * 0.5
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    projector = RRProjector(
        vectors,
        -np.linspace(1.0, 0.2, rank),
        cutoff=0.0,
        full_dimension=nocc * nvir,
    )
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(projector, core, nocc, nvir)
    t1 = rng.normal(size=(nocc, nvir))
    orbital_energies = np.concatenate((
        -np.linspace(1.4, 0.7, nocc),
        np.linspace(0.2, 1.0, nvir),
    ))
    fock = rng.normal(size=(nmo, nmo))
    provider = MOThreeIndexIntegralProvider(
        factors,
        nocc,
        factorization="cd",
        threshold=1e-8,
        source="thc-engine-unit-test",
    )
    return provider, projector, fock, orbital_energies, t1, doubles


def _full_pair_factors(projector, nocc, nvir):
    return fit_weighted_thc_projector(
        projector,
        nocc,
        nvir,
        thc_rank=nocc * nvir,
        fit_tolerance=1e-12,
        orthogonality_cutoff=1e-12,
        max_iterations=1,
        seed=0,
    )


def _equation_inputs(problem):
    provider, _projector, fock, energies, t1, doubles = problem
    nocc = provider.nocc
    return (
        doubles,
        t1,
        fock[:nocc, :nocc],
        fock[:nocc, nocc:],
        fock[nocc:, nocc:],
        energies[:nocc],
        energies[nocc:],
        provider.L_oo,
        provider.L_ov,
        provider.L_vv,
    )


def test_full_pair_thc_endpoint_reproduces_complete_rr_numerator():
    problem = _problem(seed=402)
    provider, projector, *_ = problem
    factors = _full_pair_factors(
        projector, provider.nocc, provider.nvir
    )
    args = _equation_inputs(problem)
    complete = build_projected_ccsd_doubles_numerator(
        *args, auxiliary_block_size=2, virtual_block_size=2
    )
    hybrid = build_hybrid_projected_ccsd_doubles_numerator(
        args[0],
        factors,
        *args[1:],
        auxiliary_block_size=2,
        virtual_block_size=2,
        replacement_validation_atol=VALIDATION_ATOL,
        replacement_validation_rtol=VALIDATION_RTOL,
    )
    np.testing.assert_allclose(
        hybrid.thc_backprojected_core,
        hybrid.rr_reference_1_3.sigma_pair,
        atol=2e-10,
    )
    np.testing.assert_allclose(hybrid.replacement_delta_core, 0.0, atol=2e-10)
    np.testing.assert_allclose(hybrid.core, complete.core, atol=2e-10)
    metadata = hybrid.metadata()
    assert metadata["complete_equation_graph"] is True
    assert metadata["canonical_equation"] is False
    assert metadata["materialized_dense_t2"] is False
    assert metadata["replacement_applied"] is False
    assert metadata["replacement_validated"] is True
    assert metadata["inexact_thc_enabled"] is False


def test_full_pair_backprojection_gate_is_scale_aware():
    problem = _problem(seed=410)
    provider, projector, *_ = problem
    factors = _full_pair_factors(projector, provider.nocc, provider.nvir)
    args = list(_equation_inputs(problem))
    scale = 1e6
    args[-3:] = [value * scale for value in args[-3:]]
    result = build_hybrid_projected_ccsd_doubles_numerator(
        args[0],
        factors,
        *args[1:],
        auxiliary_block_size=2,
        virtual_block_size=2,
        replacement_validation_atol=VALIDATION_ATOL,
        replacement_validation_rtol=VALIDATION_RTOL,
    )
    # The two exact contraction paths accumulate in a different order. At
    # this scale their absolute roundoff exceeds the absolute tolerance while
    # the declared mixed absolute/relative gate correctly accepts the identity.
    assert result.replacement_delta_norm > VALIDATION_ATOL
    assert (
        result.replacement_delta_norm
        <= result.replacement_validation_threshold
    )
    metadata = result.metadata()
    assert metadata["replacement_validation_criterion"] == (
        "delta <= atol + rtol*max(reference,backprojected)"
    )


def test_inexact_fit_fails_closed_before_any_residual_replacement():
    problem = _problem(seed=403, rank=3)
    provider, projector, *_ = problem
    factors = fit_weighted_thc_projector(
        projector,
        provider.nocc,
        provider.nvir,
        thc_rank=3,
        fit_tolerance=10.0,
        orthogonality_cutoff=1e-12,
        max_iterations=10,
        seed=9,
    )
    args = _equation_inputs(problem)
    with pytest.raises(NotImplementedError, match="inexact direct-CD"):
        build_hybrid_projected_ccsd_doubles_numerator(
            args[0],
            factors,
            *args[1:],
            auxiliary_block_size=1,
            virtual_block_size=2,
            replacement_validation_atol=VALIDATION_ATOL,
            replacement_validation_rtol=VALIDATION_RTOL,
        )


def test_corrupted_full_pair_endpoint_fails_backprojection_gate():
    problem = _problem(seed=409)
    provider, projector, *_ = problem
    factors = _full_pair_factors(projector, provider.nocc, provider.nvir)
    corrupted_tau = factors.tau.copy()
    corrupted_tau[0, 0] += 1e-4
    corrupted = replace(factors, tau=corrupted_tau)
    args = _equation_inputs(problem)
    with pytest.raises(RuntimeError, match="back-projection gate"):
        build_hybrid_projected_ccsd_doubles_numerator(
            args[0],
            corrupted,
            *args[1:],
            auxiliary_block_size=2,
            virtual_block_size=2,
            replacement_validation_atol=1e-12,
            replacement_validation_rtol=1e-12,
        )


def test_full_pair_hybrid_jacobi_matches_rr_engine():
    problem = _problem(seed=404)
    provider, projector, fock, energies, t1, doubles = problem
    factors = _full_pair_factors(projector, provider.nocc, provider.nvir)
    rr = RRCCSDIterationEngine(
        provider,
        projector,
        fock,
        energies,
        level_shift=0.03,
        auxiliary_block_size=2,
        virtual_block_size=2,
    )
    thc = THCRRCCSDIterationEngine(
        provider,
        projector,
        factors,
        fock,
        energies,
        level_shift=0.03,
        auxiliary_block_size=2,
        virtual_block_size=2,
        replacement_validation_atol=VALIDATION_ATOL,
        replacement_validation_rtol=VALIDATION_RTOL,
    )
    expected = rr.jacobi(t1, doubles)
    observed = thc.jacobi(t1, doubles)
    np.testing.assert_allclose(observed.t1, expected.t1, atol=2e-10)
    np.testing.assert_allclose(
        observed.doubles.core, expected.doubles.core, atol=3e-10
    )
    np.testing.assert_allclose(observed.energy, expected.energy, atol=2e-11)
    np.testing.assert_allclose(
        observed.equation_residual_core,
        expected.equation_residual_core,
        atol=3e-10,
    )
    assert observed.doubles_equation.materialized_dense_t2 is False
    assert thc.metadata()["performance_eligible"] is False


def test_zero_hybrid_kernel_converges_and_reports_hybrid_residual():
    nocc, nvir, naux = 1, 2, 1
    projector = RRProjector(
        np.eye(nocc * nvir),
        -np.ones(nocc * nvir),
        0.0,
        nocc * nvir,
    )
    provider = MOThreeIndexIntegralProvider(
        np.zeros((naux, nocc + nvir, nocc + nvir)),
        nocc,
        factorization="cd",
        threshold=1e-8,
        source="zero-thc-problem",
    )
    energies = np.array([-1.0, 0.4, 0.9])
    factors = _full_pair_factors(projector, nocc, nvir)
    engine = THCRRCCSDIterationEngine(
        provider,
        projector,
        factors,
        np.diag(energies),
        energies,
        auxiliary_block_size=1,
        virtual_block_size=1,
        replacement_validation_atol=VALIDATION_ATOL,
        replacement_validation_rtol=VALIDATION_RTOL,
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
    assert engine.metrics.metadata["equation"] == (
        "complete-rr-rccsd-with-thc-endpoint-validation"
    )
    assert engine.metrics.metadata["performance_eligible"] is False


def test_full_space_residual_diagnostic_uses_identity_pair_basis_explicitly():
    provider, _projector, fock, energies, t1, doubles = _problem(seed=408)
    diagnostic = diagnose_reconstructed_full_space_residual(
        provider,
        t1,
        doubles,
        fock,
        energies,
        level_shift=0.02,
        auxiliary_block_size=2,
        virtual_block_size=2,
    )
    dimension = provider.nocc * provider.nvir
    identity_projector = RRProjector(
        np.eye(dimension),
        np.ones(dimension),
        cutoff=0.0,
        full_dimension=dimension,
    )
    full_doubles = RRDoubles(
        identity_projector,
        doubles.reconstruct_pair_matrix(),
        provider.nocc,
        provider.nvir,
    )
    reference_engine = RRCCSDIterationEngine(
        provider,
        identity_projector,
        fock,
        energies,
        level_shift=0.02,
        auxiliary_block_size=2,
        virtual_block_size=2,
    )
    reference = reference_engine.jacobi(t1, full_doubles)
    np.testing.assert_allclose(
        diagnostic.residual_norm, float(reference.residual_norm), atol=2e-10
    )
    assert diagnostic.pair_dimension == dimension
    assert diagnostic.materialized_dense_t2_equivalent is True
    assert diagnostic.included_in_normal_iteration_timing is False
    assert diagnostic.metadata()["norm"] == (
        "full-pair-equation-residual-frobenius"
    )


def test_engine_rejects_factors_from_another_projector():
    problem = _problem(seed=405)
    provider, projector, fock, energies, *_ = problem
    factors = _full_pair_factors(projector, provider.nocc, provider.nvir)
    copied_projector = RRProjector(
        projector.vectors.copy(),
        projector.eigenvalues.copy(),
        projector.cutoff,
        projector.full_dimension,
    )
    with pytest.raises(ValueError, match="do not belong"):
        THCRRCCSDIterationEngine(
            provider,
            copied_projector,
            factors,
            fock,
            energies,
            auxiliary_block_size=1,
            virtual_block_size=1,
            replacement_validation_atol=VALIDATION_ATOL,
            replacement_validation_rtol=VALIDATION_RTOL,
        )


def test_engine_rejects_non_fp64_lifecycle():
    problem = _problem(seed=406)
    provider, projector, fock, energies, *_ = problem
    factors = _full_pair_factors(projector, provider.nocc, provider.nvir)
    with pytest.raises(TypeError, match="FP64"):
        THCRRCCSDIterationEngine(
            provider,
            projector,
            factors,
            fock.astype(np.float32),
            energies,
            auxiliary_block_size=1,
            virtual_block_size=1,
            replacement_validation_atol=VALIDATION_ATOL,
            replacement_validation_rtol=VALIDATION_RTOL,
        )


def test_gpu_full_pair_hybrid_step_stays_resident_and_matches_cpu():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    cpu_problem = _problem(seed=407, nocc=1, nvir=2, naux=2, rank=2)
    provider, projector, fock, energies, t1, doubles = cpu_problem
    gpu_provider = MOThreeIndexIntegralProvider(
        cp.asarray(provider.factors),
        provider.nocc,
        factorization="cd",
        threshold=provider.threshold,
        source="gpu-thc-engine-unit-test",
    )
    gpu_projector = RRProjector(
        cp.asarray(projector.vectors),
        cp.asarray(projector.eigenvalues),
        projector.cutoff,
        projector.full_dimension,
    )
    from gpu4pyscf.cc.device_runtime import TransferCounter

    transfers = TransferCounter()
    cpu_factors = _full_pair_factors(
        projector, provider.nocc, provider.nvir
    )
    gpu_factors = fit_weighted_thc_projector(
        gpu_projector,
        provider.nocc,
        provider.nvir,
        thc_rank=provider.nocc * provider.nvir,
        fit_tolerance=1e-12,
        orthogonality_cutoff=1e-12,
        max_iterations=1,
        transfer_counter=transfers,
    )
    cpu_engine = THCRRCCSDIterationEngine(
        provider,
        projector,
        cpu_factors,
        fock,
        energies,
        auxiliary_block_size=1,
        virtual_block_size=1,
        replacement_validation_atol=VALIDATION_ATOL,
        replacement_validation_rtol=VALIDATION_RTOL,
    )
    gpu_engine = THCRRCCSDIterationEngine(
        gpu_provider,
        gpu_projector,
        gpu_factors,
        cp.asarray(fock),
        cp.asarray(energies),
        auxiliary_block_size=1,
        virtual_block_size=1,
        replacement_validation_atol=VALIDATION_ATOL,
        replacement_validation_rtol=VALIDATION_RTOL,
    )
    cpu = cpu_engine.jacobi(t1, doubles)
    gpu_doubles = RRDoubles(
        gpu_projector,
        cp.asarray(doubles.core),
        doubles.nocc,
        doubles.nvir,
    )
    gpu = gpu_engine.jacobi(cp.asarray(t1), gpu_doubles)
    assert isinstance(gpu.doubles.core, cp.ndarray)
    assert isinstance(gpu.residual_norm, cp.ndarray)
    cp.testing.assert_allclose(
        gpu.doubles.core, cp.asarray(cpu.doubles.core), atol=2e-8
    )
