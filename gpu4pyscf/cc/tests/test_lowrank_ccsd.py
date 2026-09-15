"""GPU integration checks for the experimental RR/THC validation drivers."""

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("cupy")
from gpu4pyscf.cc import ccsd_incore
from gpu4pyscf.cc.lowrank import RRDoubles, THCDoubles
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


def _host(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)


def test_required_thresholds_and_direct_cd_defaults_are_explicit(water_sto3g):
    with pytest.raises(TypeError):
        RRCCSD(water_sto3g, rr_eig_cutoff=1e-5)
    solver = RRCCSD(
        water_sto3g,
        eri_tol=1e-8,
        rr_eig_cutoff=1e-5,
    )
    metadata = solver.method_metadata()
    assert metadata["eri_backend"] == "cd"
    assert metadata["eri_tol"] == 1e-8
    assert metadata["rr_eig_cutoff"] == 1e-5
    assert metadata["reduced_scaling_residual"] is True
    assert metadata["dense_t2_in_normal_cd_path"] is False
    assert metadata["performance_eligible"] is False


def test_full_rank_rr_matches_gpu_canonical(water_sto3g):
    reference = _configure(ccsd_incore.CCSD(water_sto3g)).run()
    solver = _configure(RRCCSD(
        water_sto3g,
        eri_tol=0.0,
        rr_eig_cutoff=0.0,
        eri_backend="canonical",
    ))
    e_corr, _t1, doubles = solver.kernel()

    assert solver.converged
    assert isinstance(doubles, RRDoubles)
    # PySCF 2.14's inherited kernel delegates to ccsd(); it does not assign the
    # returned compressed object back to self.t2.  Keep this compatibility
    # invariant explicit because ordinary energy/DIIS consumers expect ijab.
    assert not isinstance(solver.t2, RRDoubles)
    assert tuple(solver.t2.shape) == tuple(reference.t2.shape)
    assert doubles.projector.orthogonality_error() <= 1e-10
    assert abs(e_corr - reference.e_corr) < 1e-9
    np.testing.assert_allclose(
        _host(solver.reconstruct_t2()), _host(solver.t2), atol=1e-7
    )
    np.testing.assert_allclose(
        _host(solver.reconstruct_t2()), _host(reference.t2), atol=1e-7
    )
    assert np.isfinite(solver.projected_equation_residual)
    assert np.isfinite(solver.full_space_residual)
    assert np.isfinite(solver.projected_jacobi_update_norm)
    assert np.isfinite(solver.full_space_jacobi_update_norm)
    np.testing.assert_allclose(
        solver.projected_equation_residual,
        solver.full_space_residual,
        atol=1e-12,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        solver.projected_jacobi_update_norm,
        solver.full_space_jacobi_update_norm,
        atol=1e-12,
        rtol=1e-9,
    )
    method = solver.experiment_record()["method"]
    assert method["reduced_scaling_residual"] is False
    assert method["residuals"]["measurements"][
        "projected_equation"
    ]["kind"] == "equation-residual"
    assert method["residuals"]["measurements"][
        "projected_jacobi_update"
    ]["kind"] == "jacobi-update"


def test_full_rank_two_level_factorization_matches_rr(water_sto3g):
    solver = _configure(THCRRCCSD(
        water_sto3g,
        eri_tol=0.0,
        rr_eig_cutoff=0.0,
        thc_fit_tol=0.0,
        eri_backend="canonical",
    ))
    _e_corr, _t1, doubles = solver.kernel()

    assert solver.converged
    assert isinstance(doubles, THCDoubles)
    assert not isinstance(solver.t2, THCDoubles)
    assert getattr(solver.t2, "ndim", None) == 4
    np.testing.assert_allclose(
        _host(solver.reconstruct_t2()), _host(solver.t2), atol=1e-7
    )
    assert doubles.fit_residual < 1e-12
    assert doubles.paper_exact is False
    assert np.isfinite(solver.projected_equation_residual)
    assert np.isfinite(solver.full_space_residual)
    assert np.isfinite(solver.projected_jacobi_update_norm)
    assert np.isfinite(solver.full_space_jacobi_update_norm)
    np.testing.assert_allclose(
        solver.projected_equation_residual,
        solver.full_space_residual,
        atol=1e-12,
        rtol=1e-9,
    )
    method = solver.experiment_record()["method"]
    assert method["paper_exact_thc"] is False
    assert method["residuals"]["measurements"][
        "full_space_equation"
    ]["is_full_space"] is True
