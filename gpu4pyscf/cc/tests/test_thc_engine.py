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
    build_shared_cholesky_r123_decomposition,
)
from gpu4pyscf.cc.thc_factorization import fit_weighted_thc_projector
from gpu4pyscf.cc.thc_residual import (
    thc_residual_algorithms_1_3,
    transform_cholesky_t1,
)


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


def _dense_r123(doubles, t1, loo, lov, lvv):
    """Independent dense Eqs. 28, 29 and 31 in the RR core basis."""

    nocc, nvir = t1.shape
    u = doubles.projector.vectors.reshape(nocc, nvir, doubles.rank)
    t2 = np.einsum("iaX,XY,jbY->ijab", u, doubles.core, u)
    transformed = transform_cholesky_t1(t1, loo, lov, lvv)
    part1 = np.einsum(
        "ijcd,Aac,Abd->ijab", t2, transformed.hvv, transformed.hvv
    )
    part1 += np.einsum(
        "klab,Aki,Alj->ijab", t2, transformed.hoo, transformed.hoo
    )
    part1 -= np.einsum(
        "ilcb,Aac,Alj->ijab", t2, transformed.hvv, transformed.hoo
    )
    part1 -= np.einsum(
        "kjad,Aki,Abd->ijab", t2, transformed.hoo, transformed.hvv
    )
    spin_adapted = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    part2 = np.einsum("Aai,Abj->ijab", transformed.hvo, transformed.hvo)
    part2 += np.einsum(
        "ikac,Abj,Akc->ijab",
        spin_adapted,
        transformed.hvo,
        transformed.hov,
    )
    part2 += np.einsum(
        "jkbc,Aai,Akc->ijab",
        spin_adapted,
        transformed.hvo,
        transformed.hov,
    )
    part3 = -np.einsum(
        "ikac,Akj,Abc->ijab", t2, transformed.hoo, transformed.hvv
    )
    part3 -= np.einsum(
        "jkbc,Aki,Aac->ijab", t2, transformed.hoo, transformed.hvv
    )
    return tuple(
        np.einsum("iaX,ijab,jbY->XY", u, part, u)
        for part in (part1, part2, part3)
    )


def test_shared_decomposition_matches_dense_arbitrary_pair_and_recombines():
    problem = _problem(seed=411, nocc=2, nvir=3, naux=4, rank=4)
    args = _equation_inputs(problem)
    complete = build_projected_ccsd_doubles_numerator(
        *args, auxiliary_block_size=2, virtual_block_size=2
    )
    split = build_shared_cholesky_r123_decomposition(
        *args, auxiliary_block_size=2, virtual_block_size=2
    )
    dense = _dense_r123(args[0], args[1], *args[-3:])

    np.testing.assert_allclose(split.exact_r123.algorithm_1, dense[0], atol=3e-9)
    np.testing.assert_allclose(split.exact_r123.algorithm_2, dense[1], atol=3e-9)
    np.testing.assert_allclose(split.exact_r123.algorithm_3, dense[2], atol=3e-9)
    np.testing.assert_allclose(
        split.exact_r123_core,
        sum(dense),
        atol=5e-9,
    )
    np.testing.assert_allclose(
        split.reconstruct_complete(), complete.core, atol=3e-12
    )
    assert not hasattr(split, "complete_rr")
    assert [term["term"] for term in split.term_ledger] == [
        "r123-algorithm-1",
        "r123-algorithm-2",
        "r123-algorithm-3",
        "r123-complement",
    ]
    metadata = split.metadata()
    assert metadata["equation_identity"] == (
        "complete_rr = exact_r123 + complement"
    )
    assert metadata["replacement_seam"] == "complement + replacement_r123"
    assert metadata["materialized_dense_t2"] is False
    assert metadata["materialized_four_index_eri"] is False
    assert metadata["performance_eligible"] is False


def test_zero_t1_t2_isolates_bare_algorithm_2_and_preserves_identity():
    problem = _problem(seed=412, nocc=2, nvir=2, naux=3, rank=3)
    args = list(_equation_inputs(problem))
    doubles = args[0]
    args[0] = RRDoubles(
        doubles.projector,
        np.zeros_like(doubles.core),
        doubles.nocc,
        doubles.nvir,
    )
    args[1] = np.zeros_like(args[1])
    complete = build_projected_ccsd_doubles_numerator(
        *args, auxiliary_block_size=1, virtual_block_size=1
    )
    split = build_shared_cholesky_r123_decomposition(
        *args, auxiliary_block_size=1, virtual_block_size=1
    )

    np.testing.assert_allclose(split.exact_r123.algorithm_1, 0.0, atol=1e-13)
    np.testing.assert_allclose(split.exact_r123.algorithm_3, 0.0, atol=1e-13)
    np.testing.assert_allclose(
        split.exact_r123_core, split.exact_r123.algorithm_2, atol=1e-13
    )
    assert np.linalg.norm(split.exact_r123.algorithm_2) > 1e-8
    np.testing.assert_allclose(
        split.reconstruct_complete(), complete.core, atol=2e-13
    )


def test_truncated_thc_candidate_changes_only_r123_at_composition_seam():
    problem = _problem(seed=413, nocc=2, nvir=3, naux=3, rank=3)
    provider, projector, *_ = problem
    args = _equation_inputs(problem)
    complete = build_projected_ccsd_doubles_numerator(
        *args, auxiliary_block_size=2, virtual_block_size=2
    )
    split = build_shared_cholesky_r123_decomposition(
        *args, auxiliary_block_size=2, virtual_block_size=2
    )
    factors = fit_weighted_thc_projector(
        projector,
        provider.nocc,
        provider.nvir,
        thc_rank=projector.rank,
        fit_tolerance=10.0,
        orthogonality_cutoff=1e-12,
        max_iterations=30,
        seed=17,
    )
    assert not factors.analytic_full_pair_endpoint
    approximation = thc_residual_algorithms_1_3(
        factors.y_occ,
        factors.y_vir,
        factors.amplitude_core(args[0].core),
        args[1],
        *args[-3:],
        auxiliary_block_size=2,
    )
    replacement = approximation.to_rr(factors.tau)
    candidate = split.replace_r123(replacement)
    expected_delta = replacement - split.exact_r123_core

    np.testing.assert_allclose(
        candidate - complete.core, expected_delta, atol=5e-12
    )
    assert np.linalg.norm(expected_delta) > 1e-8
    perturbation = np.eye(projector.rank) * 0.125
    np.testing.assert_allclose(
        split.replace_r123(replacement + perturbation) - candidate,
        perturbation,
        atol=1e-13,
    )
    denominator_action = np.arange(projector.rank**2, dtype=float).reshape(
        projector.rank, projector.rank
    )
    complete_residual = complete.core - denominator_action
    candidate_residual = candidate - denominator_action
    np.testing.assert_allclose(
        candidate_residual - complete_residual, expected_delta, atol=5e-12
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
    assert metadata["replacement_applied"] is True
    assert metadata["replacement_validated"] is True
    assert metadata["inexact_thc_enabled"] is False
    assert metadata["replacement_formula"] == (
        "rr_complement + backproject(thc_algorithms_1_3)"
    )
    np.testing.assert_allclose(
        hybrid.core,
        hybrid.decomposition.complement_core
        + hybrid.thc_backprojected_core,
        atol=1e-13,
    )


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


def test_failed_first_engine_step_does_not_claim_replacement_application():
    problem = _problem(seed=414)
    provider, projector, fock, energies, t1, doubles = problem
    factors = _full_pair_factors(projector, provider.nocc, provider.nvir)
    corrupted_tau = factors.tau.copy()
    corrupted_tau[0, 0] += 1e-4
    corrupted = replace(factors, tau=corrupted_tau)
    engine = THCRRCCSDIterationEngine(
        provider,
        projector,
        corrupted,
        fock,
        energies,
        auxiliary_block_size=2,
        virtual_block_size=2,
        replacement_validation_atol=1e-12,
        replacement_validation_rtol=1e-12,
    )

    assert engine.metadata()["replacement_applied"] is False
    assert engine.metadata()["replacement_application_count"] == 0
    with pytest.raises(RuntimeError, match="back-projection gate"):
        engine.jacobi(t1, doubles)
    assert engine.metadata()["replacement_applied"] is False
    assert engine.metadata()["replacement_application_count"] == 0


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
    assert thc.metadata()["replacement_applied"] is False
    assert thc.metadata()["replacement_application_count"] == 0
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
    assert thc.metadata()["replacement_applied"] is True
    assert thc.metadata()["replacement_application_count"] == 1


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
        "rr-r123-complement-plus-thc-r123"
    )
    assert engine.metrics.metadata["replacement_applied"] is True
    assert engine.metrics.metadata["replacement_application_count"] == 2
    assert engine.metrics.metadata["performance_eligible"] is False


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
