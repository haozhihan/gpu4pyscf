"""Focused lifecycle tests for the public direct-CD RRCCSD bridge."""

from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pytest


# The developer laptop intentionally has no PySCF/CuPy installation.  The
# reference tensor modules are loaded by conftest.py; this minimal base lets us
# import rrccsd.py and test its backend-neutral active-space/projector bridge.
if importlib.util.find_spec("pyscf") is None:
    stub = types.ModuleType("gpu4pyscf.cc.ccsd_incore")

    class _CPUOnlyCCSDStub:
        _keys: set[str] = set()

    stub.CCSD = _CPUOnlyCCSDStub
    stub.update_amps = None
    sys.modules[stub.__name__] = stub

from gpu4pyscf.cc.device_runtime import RunMetrics
from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.lowrank import RRDoubles
from gpu4pyscf.cc.rrccsd import RRCCSD, _active_restricted_orbitals
from gpu4pyscf.cc.thc_rrccsd import THCRRCCSD


def test_rrccsd_selected_column_kernel_selector_fails_closed():
    assert "gint_column_kernel" in RRCCSD._keys
    with pytest.raises(ValueError, match="must be 'reference' or 'grouped'"):
        RRCCSD(
            None,
            eri_tol=1e-8,
            rr_eig_cutoff=1e-6,
            gint_column_kernel="automatic",
        )
    with pytest.raises(ValueError, match="applies only to the selected"):
        RRCCSD(
            None,
            eri_tol=1e-8,
            rr_eig_cutoff=1e-6,
            gint_column_backend="restricted-reference",
            gint_column_kernel="grouped",
        )


def _bare_rr_solver(
    integrals: MOThreeIndexIntegralProvider,
    energies: np.ndarray,
) -> RRCCSD:
    solver = RRCCSD.__new__(RRCCSD)
    solver._cd_integrals = integrals
    solver._rr_fock = np.diag(energies)
    solver._rr_orbital_energies = energies
    solver._mp2_operator = None
    solver._rr_engine = None
    solver._cd_initial_t1 = None
    solver._cd_initial_doubles = None
    solver.rr_projector = None
    solver.denominator_tolerance = 1e-14
    solver.denominator_max_rank = None
    solver.rr_eig_cutoff = 0.0
    solver.rr_max_rank = None
    solver.rr_initial_rank = 4
    solver.rr_solver_tolerance = 1e-12
    solver.rr_solver_maxiter = None
    solver.rr_dense_fallback_dimension = 64
    solver.rr_ritz_residual_tolerance = 1e-10
    solver.rr_auxiliary_block_size = 2
    solver.rr_ring_kernel = "reference"
    solver._rr_projector_build_metadata = None
    solver._mp2_initial_metadata = None
    solver.run_metrics = RunMetrics("rrccsd-cpu-bridge-test")
    return solver


def test_active_restricted_orbitals_applies_noncontiguous_frozen_mask():
    coefficient = np.arange(30.0).reshape(6, 5)
    occupations = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
    mask = np.array([False, True, True, False, True])

    active, active_occ, returned_mask = _active_restricted_orbitals(
        coefficient,
        occupations,
        mask,
        nocc=1,
        nmo=3,
    )

    np.testing.assert_array_equal(active, coefficient[:, [1, 2, 4]])
    np.testing.assert_array_equal(active_occ, [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(returned_mask, mask)


def test_rr_residual_metadata_separates_equation_and_jacobi_norms():
    solver = RRCCSD.__new__(RRCCSD)
    solver.projected_equation_residual = 2.0e-7
    solver.full_space_residual = None
    solver.projected_jacobi_update_norm = 4.0e-6
    solver.full_space_jacobi_update_norm = None
    solver.representation_residual = 3.0e-5

    residual = solver.residual_metadata()

    assert residual["available"] is True
    assert residual["projected_equation"] == 2.0e-7
    assert "norm" not in residual
    projected = residual["measurements"]["projected_equation"]
    jacobi = residual["measurements"]["projected_jacobi_update"]
    full = residual["measurements"]["full_space_equation"]
    assert projected["kind"] == "equation-residual"
    assert projected["space"] == "rr-projected-active-pair"
    assert projected["is_full_space"] is False
    assert jacobi["kind"] == "jacobi-update"
    assert jacobi["norm"] == 4.0e-6
    assert full["available"] is False
    assert full["norm"] is None


@pytest.mark.parametrize(
    "occupations, message",
    [
        (np.array([1.0, 0.0]), "doubly occupied"),
        (np.array([2.0, 0.5]), "unoccupied active virtual"),
    ],
)
def test_active_restricted_orbitals_rejects_non_rhf_occupations(
    occupations, message
):
    with pytest.raises(NotImplementedError, match=message):
        _active_restricted_orbitals(
            np.eye(2), occupations, np.ones(2, dtype=bool), nocc=1, nmo=2
        )


def test_full_rank_compressed_mp2_guess_matches_exact_pair_operator():
    rng = np.random.default_rng(904)
    nocc, nvir, naux = 1, 2, 4
    raw = rng.normal(size=(naux, nocc + nvir, nocc + nvir))
    factors = (raw + raw.transpose(0, 2, 1)) * 0.5
    integrals = MOThreeIndexIntegralProvider(
        factors,
        nocc,
        factorization="cd",
        threshold=0.0,
        source="rrccsd-cpu-test",
    )
    energies = np.array([-1.1, 0.3, 0.8])
    solver = _bare_rr_solver(integrals, energies)

    emp2, t1, doubles = solver._cd_initial_amplitudes(integrals)

    assert isinstance(doubles, RRDoubles)
    assert doubles.rank == nocc * nvir
    np.testing.assert_allclose(t1, 0.0, atol=0.0)
    expected_pair = solver._mp2_operator.materialize(exact_denominator=True)
    np.testing.assert_allclose(
        doubles.reconstruct_pair_matrix(), expected_pair, atol=2e-13, rtol=2e-13
    )
    assert np.isfinite(emp2)
    assert solver._mp2_initial_metadata["dense_t2_materialized"] is False
    assert solver._mp2_initial_metadata["denominator"]["tolerance"] == 1e-14
    assert solver._rr_projector_build_metadata["solver"].endswith(
        "dense-validation"
    )


def test_zero_cutoff_large_pair_space_fails_before_incomplete_lanczos():
    rng = np.random.default_rng(905)
    nocc, nvir, naux = 2, 2, 3
    raw = rng.normal(size=(naux, nocc + nvir, nocc + nvir))
    factors = (raw + raw.transpose(0, 2, 1)) * 0.5
    integrals = MOThreeIndexIntegralProvider(
        factors,
        nocc,
        factorization="cd",
        threshold=0.0,
        source="rrccsd-cpu-test",
    )
    solver = _bare_rr_solver(integrals, np.array([-1.3, -0.7, 0.2, 0.9]))
    solver.rr_dense_fallback_dimension = 2

    with pytest.raises(NotImplementedError, match="nonzero cutoff"):
        solver._build_cd_projector(integrals)


def test_capped_denominator_factorization_fails_closed():
    rng = np.random.default_rng(906)
    nocc, nvir, naux = 1, 3, 2
    raw = rng.normal(size=(naux, nocc + nvir, nocc + nvir))
    factors = (raw + raw.transpose(0, 2, 1)) * 0.5
    integrals = MOThreeIndexIntegralProvider(
        factors,
        nocc,
        factorization="cd",
        threshold=0.0,
        source="rrccsd-cpu-test",
    )
    solver = _bare_rr_solver(integrals, np.array([-1.0, 0.1, 0.8, 2.0]))
    solver.denominator_max_rank = 1

    with pytest.raises(RuntimeError, match="denominator_max_rank"):
        solver._build_cd_projector(integrals)


def test_explicit_full_space_diagnostic_records_resources_and_transfer_delta():
    rng = np.random.default_rng(907)
    nocc, nvir, naux = 1, 2, 4
    raw = rng.normal(size=(naux, nocc + nvir, nocc + nvir))
    factors = (raw + raw.transpose(0, 2, 1)) * 0.5
    integrals = MOThreeIndexIntegralProvider(
        factors,
        nocc,
        factorization="cd",
        threshold=0.0,
        source="rrccsd-full-space-diagnostic-cpu-test",
    )
    energies = np.array([-1.1, 0.3, 0.8])
    solver = _bare_rr_solver(integrals, energies)
    _emp2, t1, doubles = solver._cd_initial_amplitudes(integrals)
    solver.eri_backend = "cd"
    solver.t1 = t1
    solver.t2 = doubles
    solver.doubles = doubles
    solver.level_shift = 0.0
    solver.converged = True
    solver.rr_virtual_block_size = 2
    solver.full_space_residual = None
    solver.full_space_diagnostic_metadata = None
    solver._dense_t2_reconstructed = False
    solver.method_metadata = lambda: {
        "full_space_residual": solver.full_space_residual,
        "full_space_residual_diagnostic": (
            solver.full_space_diagnostic_metadata
        ),
    }

    diagnostic = solver.run_full_space_residual_diagnostic()

    assert np.isfinite(diagnostic.residual_norm)
    assert diagnostic.included_in_normal_iteration_timing is False
    assert solver.full_space_residual == diagnostic.residual_norm
    assert solver._dense_t2_reconstructed is True
    metadata = solver.full_space_diagnostic_metadata
    assert metadata["normal_kernel_invokes_diagnostic"] is False
    assert metadata["kind"] == "equation-residual"
    assert metadata["space"] == "full-active-pair"
    assert metadata["is_full_space"] is True
    assert metadata["phase_name"] == (
        "post_convergence_full_space_residual_diagnostic"
    )
    assert metadata["resource_accounting"]["pair_matrix_nbytes"] == 32
    assert metadata["resource_accounting"][
        "largest_single_materialized_array_nbytes"
    ] >= 32
    assert metadata["resource_accounting"]["device_memory_before"] == {
        "array_backend": "numpy",
        "device_memory_available": False,
    }
    delta = metadata["transfer_accounting"]["delta"]
    assert delta["total_bytes"] == 0
    assert delta["total_transfers"] == 0
    phases = solver.run_metrics.phase_totals()
    assert phases["post_convergence_full_space_residual_diagnostic"][
        "calls"
    ] == 1


def test_thc_full_space_diagnostic_requires_converged_state():
    solver = THCRRCCSD.__new__(THCRRCCSD)
    solver.eri_backend = "cd"
    solver.converged = False

    with pytest.raises(RuntimeError, match="converged THCRRCCSD.kernel"):
        solver.run_full_space_residual_diagnostic()


@pytest.mark.slow
def test_water_sto3g_direct_cd_full_rank_matches_canonical_gpu():
    pyscf = pytest.importorskip("pyscf")
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc import ccsd_incore

    mol = pyscf.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        cart=False,
        verbose=0,
    )
    mf = mol.RHF().run(conv_tol=1e-12)
    reference = ccsd_incore.CCSD(mf)
    reference.max_cycle = 50
    reference.conv_tol = 1e-10
    reference.conv_tol_normt = 1e-8
    reference.kernel()

    solver = RRCCSD(
        mf,
        eri_tol=1e-12,
        direct_scf_tol=1e-14,
        rr_eig_cutoff=0.0,
        denominator_tolerance=1e-13,
        rr_auxiliary_block_size=4,
        rr_virtual_block_size=2,
    )
    solver.max_cycle = 50
    solver.conv_tol = 1e-10
    solver.conv_tol_normt = 1e-8
    e_corr, _t1, doubles = solver.kernel()

    assert solver.converged
    assert isinstance(doubles, RRDoubles)
    assert doubles.rank == solver.nocc * (solver.nmo - solver.nocc)
    assert abs(e_corr - reference.e_corr) < 2e-8
    assert solver.projected_equation_residual <= 1e-8
    assert np.isfinite(solver.projected_jacobi_update_norm)
    assert "residual_norm" not in solver.__dict__
    metadata = solver.method_metadata()
    assert metadata["eri_tol"] == 1e-12
    assert metadata["direct_scf_tol"] == 1e-14
    assert metadata["reduced_scaling_residual"] is True
    assert metadata["dense_t2_reconstructed"] is False
    assert metadata["projected_equation_residual_status"] == "computed"
    assert metadata["projected_jacobi_update_status"] == "computed"
    assert metadata["residuals"]["measurements"][
        "projected_equation"
    ]["kind"] == "equation-residual"
    assert metadata["full_space_residual_status"] == (
        "requires-separate-explicit-dense-diagnostic"
    )
    candidate_t2 = solver.reconstruct_t2().get()
    np.testing.assert_allclose(candidate_t2, reference.t2, atol=2e-6, rtol=2e-6)
    assert solver.method_metadata()["dense_t2_reconstructed"] is True
