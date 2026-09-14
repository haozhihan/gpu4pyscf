"""GPU lifecycle gates for the public direct-CD THC/RR driver."""

from dataclasses import replace

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
cp = pytest.importorskip("cupy")

from gpu4pyscf.cc.full_space_residual import (
    diagnose_reconstructed_full_space_residual,
)
from gpu4pyscf.cc.lowrank import RRDoubles
from gpu4pyscf.cc.rrccsd import RRCCSD
from gpu4pyscf.cc.thc_rrccsd import THCRRCCSD


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def water_sto3g():
    mol = pyscf.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        verbose=0,
    )
    return mol.RHF().run()


def _configure(solver):
    solver.max_cycle = 50
    solver.conv_tol = 1e-10
    solver.conv_tol_normt = 1e-8
    return solver


def _direct_options():
    return {
        "eri_tol": 1e-10,
        "rr_eig_cutoff": 0.0,
        "eri_backend": "cd",
        "rr_dense_fallback_dimension": 64,
        "rr_auxiliary_block_size": 2,
        "rr_virtual_block_size": 2,
    }


def _host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def test_full_pair_thc_endpoint_matches_direct_cd_rr_lifecycle(water_sto3g):
    nocc = int(np.count_nonzero(water_sto3g.mo_occ > 0))
    nvir = int(water_sto3g.mo_occ.size - nocc)
    rr = _configure(RRCCSD(water_sto3g, **_direct_options()))
    thc = _configure(THCRRCCSD(
        water_sto3g,
        **_direct_options(),
        thc_fit_tol=1e-10,
        thc_rank=nocc * nvir,
        thc_max_iterations=1,
        thc_replacement_validation_atol=1e-11,
        thc_replacement_validation_rtol=1e-11,
    ))
    assert thc.method_metadata()["replacement_applied"] is False
    assert thc.method_metadata()["replacement_application_count"] == 0
    e_rr, t1_rr, doubles_rr = rr.kernel()
    e_thc, t1_thc, doubles_thc = thc.kernel()

    assert rr.converged and thc.converged
    assert isinstance(doubles_thc, RRDoubles)
    assert abs(e_thc - e_rr) < 2e-9
    np.testing.assert_allclose(_host(t1_thc), _host(t1_rr), atol=2e-8)
    np.testing.assert_allclose(
        _host(doubles_thc.core), _host(doubles_rr.core), atol=3e-8
    )
    np.testing.assert_allclose(
        _host(thc.reconstruct_thc_t2()),
        _host(thc.reconstruct_t2()),
        atol=3e-8,
    )
    method = thc.experiment_record()["method"]
    assert method["complete_equation_graph"] is True
    assert method["paper_factor_construction"] is True
    assert method["paper_exact_thc"] is False
    assert method["performance_eligible"] is False
    assert method["rr_ring_kernel"] == "reference"
    assert method["rr_ring_kernel_not_applicable_to_thc_engine"] is True
    assert method["replacement_applied"] is True
    assert method["replacement_application_count"] > 0
    assert method["replacement_formula"] == (
        "rr_complement + backproject(thc_algorithms_1_3)"
    )
    assert method["inexact_thc_enabled"] is False
    assert method["residuals"]["measurements"][
        "projected_equation"
    ]["kind"] == "equation-residual"
    assert method["residuals"]["measurements"][
        "projected_jacobi_update"
    ]["kind"] == "jacobi-update"
    assert np.isfinite(thc.projected_jacobi_update_norm)
    assert "residual_norm" not in thc.__dict__
    assert method["thc_projector_factors"][
        "analytic_full_pair_endpoint"
    ] is True
    assert thc.full_space_residual is None
    assert method["full_space_residual_status"] == (
        "requires-separate-explicit-dense-diagnostic"
    )
    diagnostic = thc.run_full_space_residual_diagnostic()
    assert np.isfinite(diagnostic.residual_norm)
    assert diagnostic.included_in_normal_iteration_timing is False
    assert thc.full_space_residual == diagnostic.residual_norm
    assert thc.representation_residual < 1e-8
    diagnosed = thc.experiment_record()["method"]
    assert diagnosed["full_space_residual_status"] == "computed"
    assert diagnosed["dense_t2_in_normal_cd_path"] is False
    assert diagnosed["full_space_residual_diagnostic"][
        "normal_kernel_invokes_diagnostic"
    ] is False
    assert diagnosed["full_space_residual_diagnostic"]["kind"] == (
        "equation-residual"
    )
    assert diagnosed["full_space_residual_diagnostic"][
        "is_full_space"
    ] is True


def test_failed_first_solver_step_keeps_replacement_metadata_false(water_sto3g):
    nocc = int(np.count_nonzero(water_sto3g.mo_occ > 0))
    nvir = int(water_sto3g.mo_occ.size - nocc)
    solver = THCRRCCSD(
        water_sto3g,
        **_direct_options(),
        thc_fit_tol=1e-10,
        thc_rank=nocc * nvir,
        thc_max_iterations=1,
        thc_replacement_validation_atol=1e-12,
        thc_replacement_validation_rtol=1e-12,
    )
    original_fit = solver._fit_thc_projector

    def corrupted_fit():
        factors = original_fit()
        corrupted_tau = factors.tau.copy()
        corrupted_tau[0, 0] += 1e-4
        corrupted = replace(factors, tau=corrupted_tau)
        solver.thc_projector_factors = corrupted
        return corrupted

    solver._fit_thc_projector = corrupted_fit
    assert solver.method_metadata()["replacement_applied"] is False
    assert solver.method_metadata()["replacement_application_count"] == 0
    with pytest.raises(RuntimeError, match="back-projection gate"):
        solver.kernel()
    assert solver.method_metadata()["replacement_applied"] is False
    assert solver.method_metadata()["replacement_application_count"] == 0


def test_thc_rejects_rr_ring_gemm_selector(water_sto3g):
    with pytest.raises(NotImplementedError, match="does not consume"):
        THCRRCCSD(
            water_sto3g,
            **_direct_options(),
            rr_ring_kernel="gemm",
            thc_fit_tol=1e-5,
            thc_rank=1,
            thc_replacement_validation_atol=1e-11,
            thc_replacement_validation_rtol=1e-11,
        )


def test_direct_cd_requires_a_fixed_endpoint_rank(water_sto3g):
    with pytest.raises(NotImplementedError, match="fixed thc_rank"):
        THCRRCCSD(
            water_sto3g,
            eri_tol=1e-8,
            rr_eig_cutoff=1e-5,
            thc_fit_tol=1e-5,
            thc_replacement_validation_atol=1e-11,
            thc_replacement_validation_rtol=1e-11,
            eri_backend="cd",
        )


def test_direct_cd_inexact_rank_fails_closed(water_sto3g):
    solver = THCRRCCSD(
        water_sto3g,
        **_direct_options(),
        thc_fit_tol=1e-5,
        thc_rank=1,
        thc_replacement_validation_atol=1e-11,
        thc_replacement_validation_rtol=1e-11,
    )
    with pytest.raises(NotImplementedError, match="inexact direct-CD"):
        solver.kernel()


def test_full_space_diagnostic_matches_canonical_dense_update(water_sto3g):
    """Independently anchor the diagnostic to canonical dense RCCSD."""

    nocc = int(np.count_nonzero(water_sto3g.mo_occ > 0))
    nvir = int(water_sto3g.mo_occ.size - nocc)
    solver = THCRRCCSD(
        water_sto3g,
        **_direct_options(),
        thc_fit_tol=1e-10,
        thc_rank=nocc * nvir,
        thc_replacement_validation_atol=1e-11,
        thc_replacement_validation_rtol=1e-11,
    )
    provider = solver.ao2mo(solver.mo_coeff)
    _emp2, initial_t1, initial_doubles = solver.init_amps(provider)
    rng = np.random.default_rng(917)
    probe_t1 = initial_t1 + cp.asarray(
        0.02 * rng.normal(size=initial_t1.shape)
    )
    core_delta = rng.normal(size=initial_doubles.core.shape)
    core_delta = 0.01 * (core_delta + core_delta.T)
    probe_doubles = RRDoubles(
        initial_doubles.projector,
        initial_doubles.core + cp.asarray(core_delta),
        nocc,
        nvir,
    )
    fock_delta = rng.normal(size=solver._rr_fock.shape)
    fock_delta = 0.01 * (fock_delta + fock_delta.T)
    probe_fock = solver._rr_fock + cp.asarray(fock_delta)
    host_fock = _host(probe_fock)
    assert np.linalg.norm(host_fock - np.diag(np.diag(host_fock))) > 1e-4

    diagnostic = diagnose_reconstructed_full_space_residual(
        provider,
        probe_t1,
        probe_doubles,
        probe_fock,
        solver._rr_orbital_energies,
        level_shift=float(solver.level_shift),
        auxiliary_block_size=2,
        virtual_block_size=2,
        transfer_counter=solver.run_metrics.transfers,
    )

    from gpu4pyscf.cc import ccsd_incore

    canonical = ccsd_incore.CCSD(water_sto3g)
    canonical.level_shift = solver.level_shift
    canonical_eris = canonical.ao2mo(canonical.mo_coeff)
    canonical_eris.fock = host_fock
    dense_t1 = _host(probe_t1)
    dense_t2 = _host(probe_doubles.reconstruct_t2())
    updated_t1, updated_t2 = ccsd_incore.update_amps(
        canonical, dense_t1, dense_t2, canonical_eris
    )
    eia = (
        np.asarray(canonical_eris.mo_energy[:nocc])[:, None]
        - np.asarray(canonical_eris.mo_energy[nocc:])[None, :]
        - float(canonical.level_shift)
    )
    dense_singles_residual = eia * (updated_t1 - dense_t1)
    dense_doubles_denominator = (
        eia[:, None, :, None] + eia[None, :, None, :]
    )
    dense_doubles_residual = dense_doubles_denominator * (
        updated_t2 - dense_t2
    )
    expected_singles = np.linalg.norm(dense_singles_residual)
    expected_doubles = np.linalg.norm(dense_doubles_residual)
    expected_total = np.hypot(expected_singles, expected_doubles)
    np.testing.assert_allclose(
        diagnostic.singles_residual_norm,
        expected_singles,
        atol=2e-8,
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        diagnostic.doubles_residual_norm,
        expected_doubles,
        atol=3e-8,
        rtol=3e-7,
    )
    np.testing.assert_allclose(
        diagnostic.residual_norm,
        expected_total,
        atol=4e-8,
        rtol=3e-7,
    )
