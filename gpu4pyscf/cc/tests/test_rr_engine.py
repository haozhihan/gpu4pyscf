"""End-to-end tests for the resident RR/CD iteration engine."""

import numpy as np
import pytest

import gpu4pyscf.cc.rr_engine as rr_engine_module
from gpu4pyscf.cc.device_runtime import RunMetrics
from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.rr_engine import (
    RRCCSDIterationEngine,
    _RRDoublesPhaseTiming,
)


class _FakeStream:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


class _FakeEvent:
    def __init__(self, owner, serial):
        self.owner = owner
        self.serial = serial

    def record(self, stream):
        self.owner.recorded_streams.append(stream)

    def synchronize(self):  # pragma: no cover - called only by a regression
        raise AssertionError("term profiling must not synchronize each event")


class _FakeCuda:
    def __init__(self):
        self.stream = _FakeStream()
        self.events = []
        self.recorded_streams = []

    def get_current_stream(self):
        return self.stream

    def Event(self):
        event = _FakeEvent(self, len(self.events))
        self.events.append(event)
        return event

    @staticmethod
    def get_elapsed_time(start, stop):
        return float(stop.serial - start.serial)


class _FakeArrayModule:
    def __init__(self):
        self.cuda = _FakeCuda()


def _engine_problem(
    xp=np,
    seed=301,
    nocc=2,
    nvir=3,
    naux=4,
    rr_ring_kernel="reference",
    wvvvv_kernel="fused",
    rank=None,
):
    rng = np.random.default_rng(seed)
    nmo = nocc + nvir
    raw = rng.normal(size=(naux, nmo, nmo))
    factors = (raw + raw.transpose(0, 2, 1)) * 0.5
    dimension = nocc * nvir
    rank = dimension if rank is None else int(rank)
    vectors, _ = np.linalg.qr(rng.normal(size=(dimension, rank)))
    core = rng.normal(size=(rank, rank))
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
        xp.asarray(-np.ones(rank)),
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
        rr_ring_kernel=rr_ring_kernel,
        wvvvv_kernel=wvvvv_kernel,
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

    expected_phases = [
        "rr_doubles_fock_intermediates",
        "rr_doubles_lagrangian",
        "rr_doubles_term_linear_t1",
        "rr_doubles_term_bare_ovov",
        "rr_doubles_term_woooo",
        "rr_doubles_term_wvvvv",
        "rr_doubles_term_one_body",
        "rr_doubles_term_ring",
    ]
    phases = [
        phase for phase in engine.metrics.phases
        if phase["name"] in expected_phases
    ]
    assert [phase["name"] for phase in phases] == expected_phases
    assert all(phase["depth"] == 1 for phase in phases)
    assert all(
        phase["metadata"]["timing_semantics"] == "host-wall"
        for phase in phases
    )
    assert all(
        phase["metadata"]["accounted_as_child"] is True
        for phase in phases
    )
    doubles_outer = next(
        phase for phase in engine.metrics.phases
        if phase["name"] == "rr_doubles_equation"
    )
    assert doubles_outer["metadata"]["timing_semantics"] == "host-wall"
    profile = engine.metadata()["rr_doubles_term_profiling"]
    assert profile == {
        "timing_semantics": "host-wall",
        "synchronization": "none-cpu-synchronous",
        "separate_device_timing_domain": False,
        "accounted_as_child": True,
        "coverage": "component-build-only",
        "excluded_operations": [
            "core_assembly",
            "projected_denominator_solve",
        ],
        "term_sum_is_complete_doubles_time": False,
        "ring_kernel": "reference",
        "wvvvv_kernel": "fused",
        "auxiliary_block_size": 2,
        "virtual_block_size": 2,
    }


def test_deferred_device_term_timing_uses_one_stream_sync_and_no_event_sync():
    fake_xp = _FakeArrayModule()
    ticks = iter([0.0, 2.0])
    metrics = RunMetrics("rr-device-timing", clock=lambda: next(ticks))
    timing = _RRDoublesPhaseTiming(fake_xp, metrics)
    names = [
        "rr_doubles_fock_intermediates",
        "rr_doubles_lagrangian",
        "rr_doubles_term_linear_t1",
        "rr_doubles_term_bare_ovov",
        "rr_doubles_term_woooo",
        "rr_doubles_term_wvvvv",
        "rr_doubles_term_one_body",
        "rr_doubles_term_ring",
    ]
    with metrics.phase("rr_doubles_equation"):
        for index, name in enumerate(names):
            with timing.phase(name, {
                "coarse_term": str(index),
                "ring_kernel": "gemm",
                "auxiliary_block_size": 7,
                "virtual_block_size": 11,
            }):
                pass

    assert timing.pending_count == len(names)
    assert fake_xp.cuda.stream.synchronize_calls == 0
    assert not [phase for phase in metrics.phases if phase["name"] in names]

    timing.synchronize_and_record()

    assert timing.pending_count == 0
    assert fake_xp.cuda.stream.synchronize_calls == 1
    assert fake_xp.cuda.recorded_streams == [fake_xp.cuda.stream] * 16
    phases = [phase for phase in metrics.phases if phase["name"] in names]
    assert [phase["name"] for phase in phases] == names
    assert all(phase["depth"] == 1 for phase in phases)
    assert all(phase["elapsed_s"] == 0.001 for phase in phases)
    assert all(
        phase["metadata"]["timing_semantics"]
        == "cuda-event-device-elapsed"
        for phase in phases
    )
    assert all(
        phase["metadata"]["separate_device_timing_domain"] is True
        and phase["metadata"]["accounted_as_child"] is False
        for phase in phases
    )
    assert all(
        phase["metadata"]["coverage"] == "component-build-only"
        and phase["metadata"]["term_sum_is_complete_doubles_time"] is False
        for phase in phases
    )
    outer = next(
        phase for phase in metrics.phases
        if phase["name"] == "rr_doubles_equation"
    )
    assert outer["elapsed_s"] == 2.0
    assert outer["exclusive_s"] == 2.0


def test_deferred_device_term_timing_rolls_back_partial_metric_writes():
    class FailingMetrics(RunMetrics):
        def __init__(self):
            super().__init__("failing-device-metrics")
            self.record_calls = 0

        def record_device_phase(self, *args, **kwargs):
            self.record_calls += 1
            if self.record_calls == 2:
                raise RuntimeError("injected metrics failure")
            return super().record_device_phase(*args, **kwargs)

    fake_xp = _FakeArrayModule()
    metrics = FailingMetrics()
    timing = _RRDoublesPhaseTiming(fake_xp, metrics)
    for name in ("rr_doubles_term_linear_t1", "rr_doubles_term_ring"):
        with timing.phase(name, {
            "coarse_term": name,
            "ring_kernel": "reference",
            "auxiliary_block_size": 2,
            "virtual_block_size": 3,
        }):
            pass

    with pytest.raises(RuntimeError, match="injected metrics failure"):
        timing.synchronize_and_record()

    assert timing.pending_count == 0
    assert fake_xp.cuda.stream.synchronize_calls == 1
    assert metrics.phases == []


def test_jacobi_discards_pending_device_events_when_doubles_build_fails(
    monkeypatch,
):
    engine, t1, doubles = _engine_problem(seed=399, nocc=1, nvir=2, naux=2)
    fake_xp = _FakeArrayModule()
    engine.xp = fake_xp

    def fail_after_one_phase(*_args, profile_phase=None, **_kwargs):
        assert profile_phase is not None
        with profile_phase("rr_doubles_term_ring", {
            "coarse_term": "ring",
            "ring_kernel": "reference",
            "auxiliary_block_size": 2,
            "virtual_block_size": 2,
        }):
            pass
        raise RuntimeError("injected doubles failure")

    monkeypatch.setattr(
        rr_engine_module,
        "build_projected_ccsd_doubles_numerator",
        fail_after_one_phase,
    )
    with pytest.raises(RuntimeError, match="injected doubles failure"):
        engine.jacobi(t1, doubles)

    assert fake_xp.cuda.stream.synchronize_calls == 0
    assert not [
        phase for phase in engine.metrics.phases
        if phase["name"].startswith("rr_doubles_term_")
    ]


@pytest.mark.parametrize("rank_full", [False, True])
def test_engine_gemm_ring_matches_reference_on_full_and_truncated_rr(rank_full):
    # The helper's projector is full rank for the endpoint and truncated for
    # the reduced case; both paths must produce the same complete numerator.
    nocc, nvir = 2, 3
    dimension = nocc * nvir
    seed = 311 if rank_full else 312
    ref_engine, ref_t1, ref_doubles = _engine_problem(
        seed=seed, nocc=nocc, nvir=nvir, rr_ring_kernel="reference"
    )
    gemm_engine, gemm_t1, gemm_doubles = _engine_problem(
        seed=seed, nocc=nocc, nvir=nvir, rr_ring_kernel="gemm"
    )
    if not rank_full:
        # Recreate both engines with the same explicitly truncated projector.
        rank = dimension - 1
        ref_engine, ref_t1, ref_doubles = _engine_problem(
            seed=seed,
            nocc=nocc,
            nvir=nvir,
            rr_ring_kernel="reference",
            rank=rank,
        )
        gemm_engine, gemm_t1, gemm_doubles = _engine_problem(
            seed=seed,
            nocc=nocc,
            nvir=nvir,
            rr_ring_kernel="gemm",
            rank=rank,
        )
    ref = ref_engine.jacobi(ref_t1, ref_doubles)
    gemm = gemm_engine.jacobi(gemm_t1, gemm_doubles)
    np.testing.assert_allclose(gemm.t1, ref.t1, atol=5e-9)
    np.testing.assert_allclose(gemm.doubles.core, ref.doubles.core, atol=5e-8)
    assert gemm_engine.metadata()["rr_ring_kernel"] == "gemm"
    assert gemm_engine.metadata()["performance_eligible"] is False


def test_engine_rejects_unknown_ring_kernel():
    with pytest.raises(ValueError, match="rr_ring_kernel"):
        _engine_problem(rr_ring_kernel="unknown")


@pytest.mark.parametrize("rank_full", [False, True])
def test_engine_wvvvv_selector_matches_two_ladder_oracle(rank_full):
    nocc, nvir = 2, 3
    rank = nocc * nvir if rank_full else nocc * nvir - 1
    reference_engine, reference_t1, reference_doubles = _engine_problem(
        seed=411,
        nocc=nocc,
        nvir=nvir,
        rank=rank,
        wvvvv_kernel="two-ladder",
    )
    fused_engine, fused_t1, fused_doubles = _engine_problem(
        seed=411,
        nocc=nocc,
        nvir=nvir,
        rank=rank,
        wvvvv_kernel="fused",
    )

    reference = reference_engine.jacobi(reference_t1, reference_doubles)
    fused = fused_engine.jacobi(fused_t1, fused_doubles)

    np.testing.assert_allclose(fused.t1, reference.t1, atol=3e-10)
    np.testing.assert_allclose(
        fused.doubles_equation.core,
        reference.doubles_equation.core,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        fused.doubles.core, reference.doubles.core, atol=3e-9
    )
    np.testing.assert_allclose(fused.energy, reference.energy, atol=3e-10)
    np.testing.assert_allclose(
        fused.residual_norm, reference.residual_norm, atol=3e-9
    )
    assert reference_engine.metadata()["wvvvv_kernel"] == "two-ladder"
    assert fused_engine.metadata()["wvvvv_kernel"] == "fused"
    reference_wvvvv = next(
        term for term in reference.doubles_equation.term_metadata
        if term["term"] == "complete-cc-Wvvvv-ladder"
    )
    fused_wvvvv = next(
        term for term in fused.doubles_equation.term_metadata
        if term["term"] == "complete-cc-Wvvvv-ladder"
    )
    assert reference_wvvvv["kernel_name"] == "rr-wvvvv-two-ladder"
    assert fused_wvvvv["kernel_name"] == "rr-wvvvv-fused"


def test_engine_rejects_unknown_wvvvv_kernel():
    with pytest.raises(ValueError, match="wvvvv_kernel"):
        _engine_problem(wvvvv_kernel="unknown")


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
