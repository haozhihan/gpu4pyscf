"""CPU checks for the dense canonical RR Galerkin validation update."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from gpu4pyscf.cc.device_runtime import RunMetrics
from gpu4pyscf.cc.lowrank import (
    RRDoubles,
    RRProjector,
    t2_from_pair_matrix,
)


def _load_rrccsd_with_cpu_stubs(monkeypatch):
    """Load rrccsd without importing the optional CUDA/PySCF runtime."""

    ccsd_incore = types.ModuleType("gpu4pyscf.cc.ccsd_incore")

    class CCSD:
        _keys = set()

    ccsd_incore.CCSD = CCSD
    ccsd_incore.update_amps = None
    monkeypatch.setitem(
        sys.modules, "gpu4pyscf.cc.ccsd_incore", ccsd_incore
    )

    for module_name, class_name in (
        ("gint_pair_columns", "GINTAOPairColumnProvider"),
        ("gint_selected_columns", "GINTSelectedAOPairColumnProvider"),
    ):
        module = types.ModuleType(f"gpu4pyscf.cc.{module_name}")
        setattr(module, class_name, type(class_name, (), {}))
        monkeypatch.setitem(sys.modules, module.__name__, module)

    transfer_audit = types.ModuleType("gpu4pyscf.cc.gint_transfer_audit")
    transfer_audit.RUNTIME_GATE_EXECUTION_MODES = {
        "release-gate",
        "consumer-benchmark",
        "consumer-counterpoise",
    }
    monkeypatch.setitem(
        sys.modules, "gpu4pyscf.cc.gint_transfer_audit", transfer_audit
    )

    path = Path(__file__).parents[1] / "rrccsd.py"
    spec = importlib.util.spec_from_file_location(
        "_gpu4pyscf_rrccsd_galerkin_cpu_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mixed_projector() -> RRProjector:
    raw = np.array(
        [
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 3.0],
        ]
    )
    vectors, _ = np.linalg.qr(raw)
    return RRProjector(
        vectors=vectors,
        eigenvalues=np.array([2.0, 1.0]),
        cutoff=0.0,
        full_dimension=3,
        source="deterministic-test",
    )


def test_galerkin_step_projects_raw_residual_before_lyapunov_solve(
    monkeypatch,
):
    module = _load_rrccsd_with_cpu_stubs(monkeypatch)
    projector = _mixed_projector()
    vectors = projector.vectors
    eia = np.array([[-1.0, -2.5, -6.0]])
    pair_denominator = eia.ravel()
    d2 = pair_denominator[:, None] + pair_denominator[None, :]

    current_core = np.array([[0.20, -0.07], [-0.07, -0.11]])
    current_pair = vectors @ current_core @ vectors.T
    raw_residual = np.array(
        [
            [0.40, -0.13, 0.21],
            [-0.13, -0.32, 0.18],
            [0.21, 0.18, 0.27],
        ]
    )
    jacobi_pair = current_pair + raw_residual / d2
    current_t2 = t2_from_pair_matrix(current_pair, 1, 3)
    jacobi_t2 = t2_from_pair_matrix(jacobi_pair, 1, 3)

    updated, denominator = module._canonical_rr_galerkin_step(
        current_t2,
        jacobi_t2,
        projector,
        eia,
    )

    projected_raw = vectors.T @ raw_residual @ vectors
    expected_delta = denominator.solve(projected_raw)
    np.testing.assert_allclose(
        updated.core, current_core + expected_delta, atol=2e-15, rtol=2e-15
    )

    # The old project-then-return path applies D2^-1 in full pair space before
    # projection.  Non-uniform denominators make that a different update.
    old_projected_jacobi_delta = (
        vectors.T @ (raw_residual / d2) @ vectors
    )
    assert np.linalg.norm(expected_delta - old_projected_jacobi_delta) > 1e-3

    delta_pair = vectors @ (updated.core - current_core) @ vectors.T
    linearized_residual_after_step = raw_residual - d2 * delta_pair
    np.testing.assert_allclose(
        vectors.T @ linearized_residual_after_step @ vectors,
        np.zeros((2, 2)),
        atol=2e-15,
        rtol=0.0,
    )


def test_full_rank_galerkin_endpoint_is_canonical_jacobi_and_reuses_cache(
    monkeypatch,
):
    module = _load_rrccsd_with_cpu_stubs(monkeypatch)
    vectors, _ = np.linalg.qr(
        np.array(
            [
                [1.0, 2.0, -1.0],
                [3.0, -1.0, 2.0],
                [2.0, 1.0, 4.0],
            ]
        )
    )
    projector = RRProjector(
        vectors=vectors,
        eigenvalues=np.ones(3),
        cutoff=0.0,
        full_dimension=3,
        source="identity-test",
    )
    eia = np.array([[-1.0, -2.5, -6.0]])
    current_pair = np.array(
        [
            [0.2, -0.1, 0.3],
            [-0.1, 0.4, -0.2],
            [0.3, -0.2, -0.5],
        ]
    )
    jacobi_pair = np.array(
        [
            [-0.3, 0.2, 0.1],
            [0.2, -0.6, 0.4],
            [0.1, 0.4, 0.7],
        ]
    )
    current_t2 = t2_from_pair_matrix(current_pair, 1, 3)
    jacobi_t2 = t2_from_pair_matrix(jacobi_pair, 1, 3)

    first, denominator = module._canonical_rr_galerkin_step(
        current_t2, jacobi_t2, projector, eia
    )
    second, reused = module._canonical_rr_galerkin_step(
        current_t2,
        jacobi_t2,
        projector,
        eia,
        denominator=denominator,
    )

    assert reused is denominator
    np.testing.assert_allclose(first.reconstruct_t2(), jacobi_t2, atol=1e-15)
    np.testing.assert_allclose(second.core, first.core, atol=0.0, rtol=0.0)


def test_canonical_update_keeps_singles_jacobi_and_caches_denominator(
    monkeypatch,
):
    module = _load_rrccsd_with_cpu_stubs(monkeypatch)
    projector = _mixed_projector()
    vectors = projector.vectors
    current_core = np.array([[0.20, -0.07], [-0.07, -0.11]])
    current_pair = vectors @ current_core @ vectors.T
    jacobi_pair = current_pair + np.array(
        [
            [0.03, -0.01, 0.02],
            [-0.01, -0.04, 0.01],
            [0.02, 0.01, 0.05],
        ]
    )
    current_t2 = t2_from_pair_matrix(current_pair, 1, 3)
    jacobi_t2 = t2_from_pair_matrix(jacobi_pair, 1, 3)
    current_t1 = np.array([[0.1, -0.2, 0.3]])
    jacobi_t1 = np.array([[-0.4, 0.5, -0.6]])

    solver = object.__new__(module.RRCCSD)
    solver.eri_backend = "canonical"
    solver.rr_projector = projector
    solver.level_shift = 0.25
    solver.run_metrics = RunMetrics("canonical-rr-unit")
    solver.doubles = None
    solver._canonical_projected_denominator = None
    solver._canonical_projected_denominator_key = None
    solver._canonical_orbital_energy_differences = None
    solver._dense_update = types.MethodType(
        lambda _self, _t1, _t2, _eris: (jacobi_t1, jacobi_t2),
        solver,
    )
    eris = types.SimpleNamespace(
        mo_energy=np.array([-1.0, 0.5, 2.0, 5.5])
    )
    eia = (
        eris.mo_energy[:1, None]
        - eris.mo_energy[None, 1:]
        - solver.level_shift
    )
    expected, _ = module._canonical_rr_galerkin_step(
        current_t2,
        jacobi_t2,
        projector,
        eia,
    )
    old_project_jacobi = RRDoubles.from_t2(
        jacobi_t2, projector
    ).reconstruct_t2()

    returned_t1, first_t2 = solver.update_amps(
        current_t1, current_t2, eris
    )
    cached = solver._canonical_projected_denominator
    returned_t1_again, second_t2 = solver.update_amps(
        current_t1, current_t2, eris
    )

    assert returned_t1 is jacobi_t1
    assert returned_t1_again is jacobi_t1
    assert solver._canonical_projected_denominator is cached
    assert solver._canonical_projected_denominator_key[-2:] == (
        "numpy",
        None,
    )
    assert solver.run_metrics.counters[
        "canonical_rr_projected_denominator_builds"
    ] == 1.0
    assert solver.run_metrics.counters["canonical_rr_galerkin_updates"] == 2.0
    np.testing.assert_allclose(
        first_t2, expected.reconstruct_t2(), atol=2e-15, rtol=2e-15
    )
    assert np.linalg.norm(first_t2 - old_project_jacobi) > 1e-4
    np.testing.assert_allclose(second_t2, first_t2, atol=0.0, rtol=0.0)

    first_key = solver._canonical_projected_denominator_key
    solver.level_shift = 0.5
    solver.update_amps(current_t1, current_t2, eris)
    assert solver._canonical_projected_denominator is not cached
    assert solver._canonical_projected_denominator_key != first_key
    assert solver._canonical_projected_denominator_key[-3:] == (
        0.5,
        "numpy",
        None,
    )
    assert solver.run_metrics.counters[
        "canonical_rr_projected_denominator_builds"
    ] == 2.0


def test_numpy_array_location_has_explicit_cpu_cache_identity(monkeypatch):
    module = _load_rrccsd_with_cpu_stubs(monkeypatch)

    assert module._array_backend_device_identity(np.zeros(1)) == (
        "numpy",
        None,
    )


def test_gpu_cache_stays_on_input_device_and_reuses_resident_eia(monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - depends on CUDA host state
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if device_count < 1:
        pytest.skip("no CUDA device")

    module = _load_rrccsd_with_cpu_stubs(monkeypatch)
    target_device = 0
    with cp.cuda.Device(target_device):
        host_projector = _mixed_projector()
        projector = RRProjector(
            vectors=cp.asarray(host_projector.vectors),
            eigenvalues=cp.asarray(host_projector.eigenvalues),
            cutoff=0.0,
            full_dimension=3,
            source="gpu-residency-test",
        )
        vectors = projector.vectors
        current_core = cp.asarray(
            np.array([[0.20, -0.07], [-0.07, -0.11]])
        )
        current_pair = vectors @ current_core @ vectors.T
        jacobi_pair = current_pair + cp.asarray(
            np.array(
                [
                    [0.03, -0.01, 0.02],
                    [-0.01, -0.04, 0.01],
                    [0.02, 0.01, 0.05],
                ]
            )
        )
        current_t2 = t2_from_pair_matrix(current_pair, 1, 3)
        jacobi_t2 = t2_from_pair_matrix(jacobi_pair, 1, 3)
        current_t1 = cp.asarray(np.array([[0.1, -0.2, 0.3]]))
        jacobi_t1 = cp.asarray(np.array([[-0.4, 0.5, -0.6]]))

    solver = object.__new__(module.RRCCSD)
    solver.eri_backend = "canonical"
    solver.rr_projector = projector
    solver.level_shift = 0.25
    solver.run_metrics = RunMetrics("canonical-rr-gpu-residency")
    solver.doubles = None
    solver._canonical_projected_denominator = None
    solver._canonical_projected_denominator_key = None
    solver._canonical_orbital_energy_differences = None
    solver._dense_update = types.MethodType(
        lambda _self, _t1, _t2, _eris: (jacobi_t1, jacobi_t2),
        solver,
    )
    eris = types.SimpleNamespace(
        mo_energy=np.array([-1.0, 0.5, 2.0, 5.5])
    )

    caller_device = 1 if device_count > 1 else target_device
    with cp.cuda.Device(caller_device):
        _returned_t1, first_t2 = solver.update_amps(
            current_t1, current_t2, eris
        )
        cached = solver._canonical_projected_denominator
        _returned_t1, second_t2 = solver.update_amps(
            current_t1, current_t2, eris
        )
        assert int(cp.cuda.runtime.getDevice()) == caller_device

    assert first_t2.device.id == target_device
    assert second_t2.device.id == target_device
    assert solver._canonical_projected_denominator is cached
    assert cached.matrix.device.id == target_device
    assert solver._canonical_orbital_energy_differences.device.id == target_device
    assert solver._canonical_projected_denominator_key[-2:] == (
        "cupy",
        target_device,
    )
    transfer = solver.run_metrics.transfers.to_dict()["by_operation"]
    energy_upload = transfer["h2d"]["canonical_rr_orbital_energies"]
    assert energy_upload == {"bytes": eris.mo_energy.nbytes, "count": 1}

    if device_count > 1:
        with cp.cuda.Device(1):
            other_vectors = cp.asarray(host_projector.vectors)
            other_values = cp.asarray(host_projector.eigenvalues)
        wrong_device_projector = RRProjector(
            vectors=other_vectors,
            eigenvalues=other_values,
            cutoff=0.0,
            full_dimension=3,
            source="wrong-device-test",
        )
        solver.rr_projector = wrong_device_projector
        with pytest.raises(TypeError, match="one backend and device"):
            solver._canonical_galerkin_denominator(current_t2, eris)


def test_final_raw_projected_residual_gate_is_fail_closed(monkeypatch):
    module = _load_rrccsd_with_cpu_stubs(monkeypatch)

    converged, record = module._canonical_rr_convergence_gate(
        True, 1.01e-6, 1.0e-6
    )
    assert converged is False
    assert record["passed"] is False
    assert record["reported_converged"] is False

    converged, record = module._canonical_rr_convergence_gate(
        True, 1.0e-7, 1.0e-6
    )
    assert converged is True
    assert record["passed"] is True

    converged, record = module._canonical_rr_convergence_gate(
        False, 1.0e-7, 1.0e-6
    )
    assert converged is False
    assert record["iterative_solver_converged_before_gate"] is False

    for invalid in (None, np.nan, np.inf, -1.0):
        converged, record = module._canonical_rr_convergence_gate(
            True, invalid, 1.0e-6
        )
        assert converged is False
        assert record["available"] is False
        assert record["passed"] is False
        assert record["failure_reason"] == "missing-or-invalid-residual"
