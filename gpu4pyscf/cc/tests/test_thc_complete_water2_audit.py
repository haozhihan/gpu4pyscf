"""CPU contracts for the fixed-state WATER2 complete-THC audit seam."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpu4pyscf.cc.device_runtime import TransferCounter
from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.rr_residual import ProjectedPairDenominator
from gpu4pyscf.cc.thc_engine import _validate_factor_lifecycle
from gpu4pyscf.cc.thc_fhat import build_t1_transformed_fhat
from gpu4pyscf.cc.thc_mp2_weights import build_factorized_mp2_natural_occupation_weights

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DRIVER_PATH = REPOSITORY_ROOT / 'benchmarks' / 'cc' / 'a100_water8' / 'thc_complete_water2_audit.py'
SPEC = importlib.util.spec_from_file_location('_test_thc_complete_water2_audit_driver', DRIVER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - repository corruption
    raise RuntimeError('cannot load the WATER2 complete-THC audit driver')
audit_driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_driver
SPEC.loader.exec_module(audit_driver)


def _join_fhat(result):
    return np.block(
        [
            [result.fhat_oo, result.fhat_ov],
            [result.fhat_vo, result.fhat_vv],
        ]
    )


def _tiny_binding(seed: int = 2101):
    """Build a real NumPy RR/CD object graph behind a minimal solver shell."""

    rng = np.random.default_rng(seed)
    nocc, nvir, naux, rr_rank = 2, 3, 5, 3
    pair_dimension = nocc * nvir
    nmo = nocc + nvir

    vectors, _ = np.linalg.qr(rng.normal(size=(pair_dimension, rr_rank)))
    eigenvalues = np.asarray([-0.8, -0.45, -0.2], dtype=np.float64)
    projector = RRProjector(
        vectors,
        eigenvalues,
        cutoff=1e-8,
        full_dimension=pair_dimension,
        source='unit-fixed-state-mp2',
    )
    converged_core = rng.normal(size=(rr_rank, rr_rank)) * 0.025
    converged_core = (converged_core + converged_core.T) * 0.5
    mp2_core = np.diag(eigenvalues)
    doubles = RRDoubles(projector, converged_core, nocc=nocc, nvir=nvir)
    mp2_doubles = RRDoubles(projector, mp2_core, nocc=nocc, nvir=nvir)

    raw_factors = rng.normal(size=(naux, nmo, nmo)) * 0.02
    raw_factors = (raw_factors + raw_factors.swapaxes(1, 2)) * 0.5
    integrals = MOThreeIndexIntegralProvider(
        raw_factors,
        nocc,
        factorization='cd',
        threshold=1e-8,
        source='unit-direct-cd',
    )
    hcore = np.diag(np.asarray([-2.1, -1.4, 0.45, 0.8, 1.15]))
    hcore_noise = rng.normal(size=(nmo, nmo)) * 0.003
    hcore += (hcore_noise + hcore_noise.T) * 0.5
    zero_physical = build_t1_transformed_fhat(
        hcore,
        integrals.factors,
        np.zeros((nocc, nvir), dtype=np.float64),
        auxiliary_block_size=2,
    )
    fock_cd = _join_fhat(zero_physical)
    correction = rng.normal(size=(nmo, nmo)) * 0.004
    correction = (correction + correction.T) * 0.5
    fock = fock_cd + correction
    orbital_energies = np.diag(fock).copy()
    eia = orbital_energies[:nocc, None] - orbital_energies[None, nocc:]
    counter = TransferCounter()
    denominator = ProjectedPairDenominator.build(
        projector,
        eia,
        nocc,
        nvir,
        transfer_counter=counter,
    )
    engine = SimpleNamespace(
        integrals=integrals,
        projector=projector,
        fock=fock,
        orbital_energies=orbital_energies,
        denominator=denominator,
    )
    mol = object()
    scf = SimpleNamespace(get_hcore=lambda supplied_mol: hcore)
    t1 = rng.normal(size=(nocc, nvir)) * 0.015
    solver = SimpleNamespace(
        converged=True,
        _rr_engine=engine,
        _cd_integrals=integrals,
        rr_projector=projector,
        t1=t1,
        t2=doubles,
        doubles=doubles,
        _cd_initial_doubles=mp2_doubles,
        _rr_fock=fock,
        _rr_orbital_energies=orbital_energies,
        mo_coeff=np.eye(nmo, dtype=np.float64),
        run_metrics=SimpleNamespace(transfers=counter),
        get_frozen_mask=lambda: np.ones(nmo, dtype=bool),
        _scf=scf,
        mol=mol,
        level_shift=0.0,
        rr_ring_kernel='reference',
    )
    binding = audit_driver.RRLifecycleBinding.capture(solver, case_id='unit')
    return binding, hcore, fock_cd


def _assert_flags_false(value):
    flag_names = {
        'accepted',
        'production_enabled',
        'performance_eligible',
        'formal_validation_eligible',
        'complete_validated',
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if key in flag_names:
                assert item is False
            _assert_flags_false(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_flags_false(item)


class _FakeMemoryPool:
    def __init__(self):
        self.used = 2 * audit_driver.GIB
        self.reserved = 3 * audit_driver.GIB

    def used_bytes(self):
        return self.used

    def total_bytes(self):
        return self.reserved


class _FakeCudaRuntime:
    def __init__(self):
        self.visible_count = 1
        self.total = 80 * audit_driver.GIB
        self.free = 76 * audit_driver.GIB

    def getDeviceCount(self):
        return self.visible_count

    def getDeviceProperties(self, device_id):
        assert device_id == 0
        return {
            'name': b'NVIDIA A100-SXM4-80GB',
            'major': 8,
            'minor': 0,
            'totalGlobalMem': self.total,
        }

    def memGetInfo(self):
        return self.free, self.total

    def runtimeGetVersion(self):
        return 12020

    def driverGetVersion(self):
        return 12020


class _FakeCuda:
    def __init__(self):
        self.runtime = _FakeCudaRuntime()
        self.sync_count = 0

    def Device(self):
        return SimpleNamespace(id=0, pci_bus_id='00000000:41:00.0')

    def get_current_stream(self):
        return SimpleNamespace(synchronize=self._synchronize)

    def _synchronize(self):
        self.sync_count += 1


class _FakeCuPy:
    def __init__(self):
        self.cuda = _FakeCuda()
        self.pool = _FakeMemoryPool()

    def get_default_memory_pool(self):
        return self.pool


def test_equation_oracle_subtracts_denominators_once():
    singles_numerator = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    doubles_numerator = np.asarray([[3.0, 0.4], [0.4, 2.0]])
    t1 = np.asarray([[0.2, -0.1], [0.05, 0.3]])
    core = np.asarray([[0.7, 0.2], [0.2, -0.4]])
    eia = np.asarray([[-1.0, -1.5], [-0.7, -1.2]])
    denominator = np.asarray([[-0.8, 0.1], [0.1, -1.1]])

    singles, doubles = audit_driver.equation_residual_from_numerators(
        singles_numerator=singles_numerator,
        doubles_numerator=doubles_numerator,
        t1=t1,
        core=core,
        eia=eia,
        denominator_matrix=denominator,
    )

    np.testing.assert_array_equal(singles, singles_numerator - eia * t1)
    expected_doubles = doubles_numerator - (denominator @ core + core @ denominator)
    np.testing.assert_array_equal(doubles, expected_doubles)
    assert not np.array_equal(singles, singles - eia * t1)
    assert not np.array_equal(doubles, doubles - (denominator @ core + core @ denominator))


def test_preflight_and_staged_scan_helpers_are_fail_closed():
    estimate = audit_driver.estimate_candidate_memory(
        naux=5,
        nocc=2,
        nvir=3,
        rr_rank=3,
        amplitude_rank=6,
        eri_rank=6,
        auxiliary_block_size=2,
        x_block_size=1,
    )
    budget = audit_driver.ResourceBudget(
        hbm_limit_bytes=4 * audit_driver.GIB,
        host_rss_limit_bytes=4 * audit_driver.GIB,
        reserve_bytes=128 * 1024**2,
    )
    passed = audit_driver.check_memory_preflight(
        estimate,
        budget,
        hbm_current_used_bytes=1 * audit_driver.GIB,
        hbm_free_bytes=3 * audit_driver.GIB,
        hbm_total_bytes=4 * audit_driver.GIB,
        host_rss_current_bytes=1 * audit_driver.GIB,
    )
    assert passed.passed is True
    failed = audit_driver.check_memory_preflight(
        estimate,
        budget,
        hbm_current_used_bytes=4 * audit_driver.GIB - 64 * 1024**2,
        hbm_free_bytes=64 * 1024**2,
        hbm_total_bytes=4 * audit_driver.GIB,
        host_rss_current_bytes=4 * audit_driver.GIB,
    )
    assert failed.passed is False
    assert len(failed.reasons) >= 2
    assert estimate.components['fhat'] == (5 * 5 * 5**2 + 8 * 5**2) * 8

    specs = (
        audit_driver.ERIFitSpec(3, 1e-3),
        audit_driver.ERIFitSpec(4, 1e-4),
        audit_driver.ERIFitSpec(5, 1e-5),
    )
    assert audit_driver.validate_scan_order((1e-2, 1e-3), specs) == (
        (1e-2, 1e-3),
        specs,
    )
    with pytest.raises(ValueError, match='positive'):
        audit_driver.validate_scan_order(
            (1e-2,),
            (audit_driver.ERIFitSpec(3, 0.0), specs[1]),
        )
    with pytest.raises(ValueError, match='strictly tighter'):
        audit_driver.validate_scan_order(
            (1e-2,),
            (specs[1], audit_driver.ERIFitSpec(3, 1e-5)),
        )
    records = [
        {'scan_index': 0, 'diagnostic_gate_passed': False},
        {'scan_index': 1, 'diagnostic_gate_passed': True},
        {'scan_index': 2, 'diagnostic_gate_passed': False},
    ]
    assert audit_driver.select_first_and_next_tighter(records) == tuple(records[1:])
    failure = audit_driver._failed_candidate(stage='unit', controls={'rank': 3}, error=RuntimeError('expected'))
    _assert_flags_false(failure)


def test_gpu_identity_and_synchronized_hbm_observations_fail_closed():
    fake = _FakeCuPy()
    identity = audit_driver.capture_a100_gpu_identity(
        fake,
        uuid_lookup=lambda pci: 'GPU-unit-a100' if pci == '0000:41:00.0' else None,
    )
    assert identity['validated'] is True
    assert identity['visible_device_count'] == 1
    assert identity['device_id'] == 0
    assert identity['compute_capability'] == '8.0'
    assert identity['primary_identity_kind'] == 'uuid'
    assert identity['uuid'] == 'GPU-unit-a100'
    assert identity['pci_bus_id'] == '0000:41:00.0'
    _assert_flags_false(identity)

    fallback = audit_driver.capture_a100_gpu_identity(
        fake,
        uuid_lookup=lambda _pci: None,
    )
    assert fallback['primary_identity_kind'] == 'pci_bus_id'
    assert fallback['primary_identity'] == '0000:41:00.0'
    assert fallback['pci_bus_fallback_used'] is True

    with pytest.raises(ValueError, match='72-GiB'):
        audit_driver.ResourceBudget(hbm_limit_bytes=72 * audit_driver.GIB + 1)

    budget = audit_driver.ResourceBudget(reserve_bytes=0)
    observer = audit_driver.SynchronizedHBMObserver(
        fake,
        budget,
        identity,
        host_rss_reader=lambda: 2 * audit_driver.GIB,
    )
    observer.sample('start')
    fake.cuda.runtime.free = 10 * audit_driver.GIB
    fake.pool.used = 60 * audit_driver.GIB
    fake.pool.reserved = 70 * audit_driver.GIB
    observer.sample('after-stage')
    metadata = observer.metadata()
    assert fake.cuda.sync_count == 2
    assert metadata['sampled_device_used_high_water_bytes'] == 70 * audit_driver.GIB
    assert metadata['sampled_allocator_used_high_water_bytes'] == 60 * audit_driver.GIB
    assert metadata['sampled_allocator_reserved_high_water_bytes'] == 70 * audit_driver.GIB
    assert metadata['observed_peak_is_true_peak'] is False
    assert metadata['allocator_reports_native_high_water'] is False
    assert metadata['gate_passed'] is True
    _assert_flags_false(metadata)

    fake.cuda.runtime.free = 7 * audit_driver.GIB
    with pytest.raises(MemoryError, match='exceeds'):
        observer.sample('over-72-gib')
    assert observer.metadata()['gate_passed'] is False

    fake.cuda.runtime.visible_count = 2
    with pytest.raises(RuntimeError, match='exactly one'):
        audit_driver.capture_a100_gpu_identity(fake, uuid_lookup=lambda _pci: None)


def test_cli_validates_the_fixed_inexact_search_before_running_rr(tmp_path):
    arguments = [
        '--output',
        str(tmp_path / 'audit.json'),
        '--eri-tol',
        '1e-8',
        '--rr-eig-cutoff',
        '1e-6',
        '--amplitude-fit-tolerances',
        '1e-3',
        '1e-4',
        '--amplitude-initial-rank',
        '32',
        '--amplitude-max-rank',
        '128',
        '--eri-candidate',
        '256:1e-4',
        '--eri-candidate',
        '384:1e-5',
        '--occupation-change-floor',
        '1e-12',
    ]
    args = audit_driver._parser().parse_args(arguments)
    audit_driver._validate_cli(args)

    args.gint_column_backend = 'restricted-reference'
    args.gint_column_kernel = 'grouped'
    with pytest.raises(ValueError, match='selected GINT'):
        audit_driver._validate_cli(args)
    args.gint_column_backend = 'selected'
    args.gint_column_kernel = 'reference'
    args.amplitude_max_rank = audit_driver.PAIR_DIMENSION
    with pytest.raises(ValueError, match='initial <= max < OV'):
        audit_driver._validate_cli(args)


def test_cli_receipt_is_bound_only_to_selected_reference_gint(tmp_path):
    receipt = tmp_path / 'receipt.json'
    receipt.write_text('{}\n', encoding='utf-8')
    arguments = [
        '--output',
        str(tmp_path / 'audit.json'),
        '--eri-tol',
        '1e-8',
        '--rr-eig-cutoff',
        '1e-6',
        '--gint-runtime-gate-receipt',
        str(receipt),
        '--amplitude-fit-tolerances',
        '1e-3',
        '--amplitude-initial-rank',
        '32',
        '--amplitude-max-rank',
        '128',
        '--eri-candidate',
        '256:1e-4',
        '--eri-candidate',
        '384:1e-5',
        '--occupation-change-floor',
        '1e-12',
    ]
    args = audit_driver._parser().parse_args(arguments)
    audit_driver._validate_cli(args)
    assert args.gint_runtime_gate_receipt == receipt.resolve()

    args.gint_column_backend = 'restricted-reference'
    with pytest.raises(ValueError, match='selected backend'):
        audit_driver._validate_cli(args)
    args.gint_column_backend = 'selected'
    args.gint_column_kernel = 'grouped'
    with pytest.raises(ValueError, match='cannot consume'):
        audit_driver._validate_cli(args)
    args.gint_column_kernel = 'reference'
    args.gint_runtime_gate_receipt = tmp_path / 'missing.json'
    with pytest.raises(ValueError, match='does not exist'):
        audit_driver._validate_cli(args)


def test_selected_gint_runtime_options_reuse_recorded_source_contract(
    tmp_path,
    monkeypatch,
):
    receipt = tmp_path / 'receipt.json'
    receipt.write_text('{}\n', encoding='utf-8')
    args = SimpleNamespace(
        gint_column_backend='selected',
        gint_runtime_gate_receipt=receipt,
    )
    source_state = {'tree_sha256': 'a' * 64}
    expected = {
        'gint_runtime_gate_receipt': str(receipt),
        'gint_runtime_source_digest': 'a' * 64,
    }
    observed = {}

    def recorded(runtime_args, source, *, execution_mode):
        observed.update(
            runtime_args=runtime_args,
            source=source,
            execution_mode=execution_mode,
        )
        return expected

    monkeypatch.setattr(
        audit_driver.BENCHMARK,
        '_recorded_gint_runtime_options',
        recorded,
    )
    assert audit_driver._gint_runtime_options(args, source_state) is expected
    assert observed['runtime_args'].gint_column_backend == 'selected'
    assert observed['runtime_args'].gint_runtime_gate_receipt == receipt
    assert observed['source'] is source_state
    assert observed['execution_mode'] == 'consumer-thc-complete-audit'

    args.gint_column_backend = 'restricted-reference'
    assert audit_driver._gint_runtime_options(args, source_state) == {}


def test_run_persists_gpu_identity_and_non_peak_hbm_evidence(
    tmp_path,
    monkeypatch,
):
    binding, _hcore, _fock_cd = _tiny_binding(seed=2105)
    fake = _FakeCuPy()
    output = tmp_path / 'audit-run.json'
    arguments = [
        '--output',
        str(output),
        '--eri-tol',
        '1e-8',
        '--rr-eig-cutoff',
        '1e-6',
        '--amplitude-fit-tolerances',
        '1e-3',
        '--amplitude-initial-rank',
        '32',
        '--amplitude-max-rank',
        '128',
        '--eri-candidate',
        '256:1e-4',
        '--eri-candidate',
        '384:1e-5',
        '--occupation-change-floor',
        '1e-12',
    ]
    args = audit_driver._parser().parse_args(arguments)
    monkeypatch.setitem(sys.modules, 'cupy', fake)
    monkeypatch.setattr(
        audit_driver,
        '_nvidia_smi_uuid_for_pci',
        lambda _pci: 'GPU-mocked-run',
    )
    source_state = {
        'repository': str(audit_driver.REPOSITORY_ROOT),
        'tree_sha256': 'a' * 64,
    }
    monkeypatch.setattr(audit_driver, '_git_state', lambda: source_state)
    monkeypatch.setattr(
        audit_driver.BENCHMARK,
        '_record_source_stability',
        lambda _record: True,
    )

    def fake_build_rr_solver(_args, *, source_state: dict):
        assert source_state is source_state_at_start
        return (
            binding.solver,
            {
                'hf_seconds': 0.0,
                'rr_post_hf_seconds': 0.0,
                'hf_energy': -1.0,
                'rr_correlation_energy': -0.1,
                'orbital': {'orbital_fingerprint': 'sha256:mock-run'},
            },
        )

    source_state_at_start = source_state
    monkeypatch.setattr(audit_driver, '_build_rr_solver', fake_build_rr_solver)

    def fake_staged(_binding, *, hbm_observer, **_kwargs):
        assert _binding.denominator is binding.denominator
        hbm_observer.sample('mock-audit-stage')
        return {'status': 'completed', **audit_driver._copy_flags()}

    monkeypatch.setattr(audit_driver, 'run_staged_audit', fake_staged)
    assert audit_driver.run(args) == 0

    result = audit_driver.json.loads(output.read_text(encoding='utf-8'))
    assert result['status'] == 'completed'
    assert result['schema'] == 'gpu4pyscf.water2-thc-complete-audit.v2'
    assert result['source'] == source_state
    assert result['source_at_end'] == source_state
    assert result['source_stability'] is True
    assert result['gpu_identity']['validated'] is True
    assert result['gpu_identity']['uuid'] == 'GPU-mocked-run'
    assert result['gpu_identity']['primary_identity_kind'] == 'uuid'
    hbm = result['hbm_observations']
    assert [sample['label'] for sample in hbm['samples']] == [
        'process-entry',
        'rr-lifecycle-converged',
        'mock-audit-stage',
        'process-exit',
    ]
    assert hbm['gate_passed'] is True
    assert hbm['observed_peak_is_true_peak'] is False
    assert hbm['formal_performance_measurement'] is False
    _assert_flags_false(result)


def test_fixed_lifecycle_builds_distinct_physical_and_effective_fhat_semantics():
    binding, hcore, fock_cd = _tiny_binding()

    physical, effective = audit_driver.build_fhat_semantics(
        binding,
        auxiliary_block_size=2,
        orbital_fingerprint='sha256:unit-orbitals',
    )

    np.testing.assert_allclose(physical.hcore_mo, hcore, atol=2e-15, rtol=0.0)
    np.testing.assert_allclose(physical.zero_t1_fock, fock_cd, atol=2e-14)
    np.testing.assert_allclose(
        effective.hcore_mo,
        hcore + (binding.fock - fock_cd),
        atol=2e-14,
    )
    assert physical.name == 'physical-cd-hamiltonian'
    assert effective.name == 'direct-cd-effective-one-body'
    assert effective.correction_max_abs > 0.0
    assert physical.tokens.digest != effective.tokens.digest
    assert effective.tokens.orbitals['t1_runtime_identity'] == id(binding.t1)
    assert effective.tokens.orbitals['doubles_runtime_identity'] == id(binding.doubles)
    assert effective.tokens.integrals['mp2_doubles_runtime_identity'] == id(binding.mp2_doubles)
    assert effective.tokens.orbitals['denominator_runtime_identity'] == id(binding.denominator)
    lifecycle_metadata = binding.metadata()
    assert lifecycle_metadata['denominator_runtime_identity'] == id(binding.denominator)
    assert lifecycle_metadata['object_and_pointer_identity_validated'] is True
    assert lifecycle_metadata['content_immutability_cryptographically_attested'] is False
    assert lifecycle_metadata['tensor_contents_hashed'] is False
    assert (
        lifecycle_metadata['array_pointer_snapshot']['denominator_matrix']
        == (binding.denominator.matrix.__array_interface__['data'][0])
    )
    _assert_flags_false(physical.metadata())
    _assert_flags_false(effective.metadata())

    replacement = RRDoubles(
        binding.projector,
        binding.doubles.core.copy(),
        nocc=binding.nocc,
        nvir=binding.nvir,
    )
    binding.solver.t2 = replacement
    with pytest.raises(ValueError, match='solver.t2'):
        binding.validate()
    binding.solver.t2 = binding.doubles
    binding.validate()

    replacement_denominator = replace(binding.denominator)
    binding.engine.denominator = replacement_denominator
    with pytest.raises(ValueError, match='engine.denominator'):
        binding.validate()
    binding.engine.denominator = binding.denominator
    binding.validate()


def test_exact_algorithms_1_to_10_anchor_uses_one_bound_rr_state():
    binding, _hcore, _fock_cd = _tiny_binding(seed=2102)
    physical, effective = audit_driver.build_fhat_semantics(
        binding,
        auxiliary_block_size=2,
        orbital_fingerprint='sha256:unit-exact-anchor',
    )
    physical_energies = np.diag(physical.zero_t1_fock).copy()
    physical_oracle = audit_driver.build_rr_equation_oracle(
        binding,
        semantics=physical.name,
        fock=physical.zero_t1_fock,
        orbital_energies=physical_energies,
        auxiliary_block_size=2,
        virtual_block_size=2,
    )
    effective_oracle = audit_driver.build_rr_equation_oracle(
        binding,
        semantics=effective.name,
        fock=binding.engine.fock,
        orbital_energies=binding.engine.orbital_energies,
        auxiliary_block_size=2,
        virtual_block_size=2,
    )
    assert effective_oracle.denominator is binding.denominator
    assert effective_oracle.denominator_matrix is binding.denominator.matrix
    amplitude = audit_driver.fit_amplitude_factors(
        binding,
        fit_tolerance=0.0,
        initial_rank=binding.pair_dimension,
        max_rank=binding.pair_dimension,
        max_iterations=5,
        als_convergence_tolerance=1e-10,
        ridge=1e-12,
        seed=0,
        orthogonality_cutoff=1e-12,
        orthogonality_tolerance=1e-10,
    )
    assert amplitude.analytic_full_pair_endpoint is True
    assert amplitude.exact_pair_endpoint is True
    assert amplitude.fit_tolerance == 0.0
    assert 0.0 < amplitude.weighted_fit_residual <= (amplitude.exact_pair_roundoff_bound)
    amplitude_metadata = amplitude.metadata()
    assert amplitude_metadata['exact_pair_endpoint'] is True
    assert amplitude_metadata['exact_pair_roundoff_gate_passed'] is True
    assert amplitude_metadata['weighted_fit_gate_passed'] is True
    corrupted_endpoint = replace(
        amplitude,
        orthogonality_error=2.0 * amplitude.exact_pair_roundoff_bound,
    )
    assert corrupted_endpoint.exact_pair_roundoff_gate_passed is False
    assert corrupted_endpoint.weighted_fit_gate_passed is False
    with pytest.raises(RuntimeError, match='roundoff gate'):
        _validate_factor_lifecycle(
            binding.doubles,
            corrupted_endpoint,
            binding.loo,
            binding.lov,
            binding.lvv,
        )
    _validate_factor_lifecycle(
        binding.doubles,
        amplitude,
        binding.loo,
        binding.lov,
        binding.lvv,
    )
    budget = audit_driver.ResourceBudget(
        hbm_limit_bytes=8 * audit_driver.GIB,
        host_rss_limit_bytes=8 * audit_driver.GIB,
        reserve_bytes=0,
    )
    preflight = audit_driver._candidate_preflight(
        binding,
        budget,
        amplitude_rank=binding.pair_dimension,
        eri_rank=binding.pair_dimension,
        auxiliary_block_size=2,
        x_block_size=1,
    )
    result = audit_driver.evaluate_candidate(
        binding,
        amplitude,
        eri_spec=audit_driver.ERIFitSpec(binding.pair_dimension, 0.0),
        exact_eri_endpoint=True,
        semantic_oracles=(
            (physical, physical_oracle),
            (effective, effective_oracle),
        ),
        tolerances=audit_driver.AuditTolerances(
            exact_identity_atol=1e-9,
            candidate_delta_norm=1e-6,
            candidate_projected_residual_norm=1e3,
            amplitude_core_symmetry_atol=1e-10,
        ),
        occupation_change_floor=1e-12,
        fit_max_iterations=5,
        als_convergence_tolerance=1e-10,
        ridge=1e-12,
        seed=0,
        auxiliary_block_size=2,
        virtual_block_size=2,
        algorithm_x_block_size=1,
        preflight=preflight,
        stage='exact-anchor',
    )

    assert result['diagnostic_gate_passed'] is True
    assert result['eri_fit_spec'] == {
        'rank': binding.pair_dimension,
        'fit_tolerance': 0.0,
    }
    assert result['eri_preprocess']['analytic_exact_pair_endpoint'] is True
    assert result['eri_preprocess']['rr_doubles_runtime_identity'] == id(binding.mp2_doubles)
    assert result['eri_preprocess']['amplitude_thc_fit_runtime_identity'] == id(amplitude)
    assert result['eri_preprocess']['cholesky_runtime_identity'] == id(binding.lov)
    weights = result['eri_preprocess']['weights']
    assert weights['thc_fit_exact_pair_endpoint'] is True
    assert weights['thc_fit_exact_pair_roundoff_bound'] == (amplitude.exact_pair_roundoff_bound)
    for row in result['semantics']:
        comparison = row['comparison']
        assert comparison['combined_delta_norm'] <= 1e-9
        assert comparison['audit_denominator_subtraction_count'] == 0
        assert comparison['oracle_denominator_subtraction_count'] == 1
        assert comparison['diagnostic_gate_passed'] is True
    _assert_flags_false(result)


def test_staged_audit_uses_the_first_passing_amplitude_without_a_cartesian_scan(
    monkeypatch,
):
    binding, _hcore, _fock_cd = _tiny_binding(seed=2104)
    monkeypatch.setattr(audit_driver, 'CASE_ID', binding.case_id)
    monkeypatch.setattr(audit_driver, 'PAIR_DIMENSION', binding.pair_dimension)
    fake = _FakeCuPy()
    gpu_identity = audit_driver.capture_a100_gpu_identity(
        fake,
        uuid_lookup=lambda _pci: 'GPU-staged-audit',
    )
    hbm_observer = audit_driver.SynchronizedHBMObserver(
        fake,
        audit_driver.ResourceBudget(reserve_bytes=0),
        gpu_identity,
        host_rss_reader=lambda: 2 * audit_driver.GIB,
    )
    result = audit_driver.run_staged_audit(
        binding,
        amplitude_tolerances=(10.0, 5.0),
        amplitude_initial_rank=binding.projector.rank,
        amplitude_max_rank=binding.pair_dimension - 1,
        eri_specs=(
            audit_driver.ERIFitSpec(binding.projector.rank, 10.0),
            audit_driver.ERIFitSpec(binding.projector.rank + 1, 5.0),
        ),
        tolerances=audit_driver.AuditTolerances(
            exact_identity_atol=1e-9,
            candidate_delta_norm=1e6,
            candidate_projected_residual_norm=1e6,
            amplitude_core_symmetry_atol=1e-10,
        ),
        budget=audit_driver.ResourceBudget(
            hbm_limit_bytes=8 * audit_driver.GIB,
            host_rss_limit_bytes=8 * audit_driver.GIB,
            reserve_bytes=0,
        ),
        occupation_change_floor=1e-12,
        fit_max_iterations=20,
        als_convergence_tolerance=1e-10,
        ridge=1e-12,
        amplitude_seed=0,
        eri_seed=0,
        orthogonality_cutoff=1e-12,
        orthogonality_tolerance=1e-10,
        auxiliary_block_size=2,
        virtual_block_size=2,
        algorithm_x_block_size=1,
        orbital_fingerprint='sha256:unit-staged-audit',
        hbm_observer=hbm_observer,
    )

    assert result['exact_anchor']['diagnostic_gate_passed'] is True
    assert result['cartesian_product_scan_performed'] is False
    assert len(result['amplitude_scan']) == 2
    assert result['selected_amplitude_scan_index'] == 0
    assert len(result['eri_scan']) == 2
    assert result['final_candidate_scan_indices'] == [0, 1]
    assert result['staged_search_found_candidate'] is True
    assert result['hbm_observations']['gate_passed'] is True
    assert [sample['label'] for sample in result['hbm_observations']['samples']] == [
        'audit-entry',
        'fhat-semantics-built',
        'rr-oracles-built',
        'exact-anchor-exit',
        'amplitude-scan-0-exit',
        'amplitude-scan-1-exit',
        'eri-scan-0-exit',
        'eri-scan-1-exit',
        'audit-exit',
    ]
    _assert_flags_false(result)


def test_zero_tolerance_remains_fail_closed_for_an_inexact_amplitude_rank():
    binding, _hcore, _fock_cd = _tiny_binding(seed=2103)

    with pytest.raises(RuntimeError, match='no THC projector rank'):
        audit_driver.fit_amplitude_factors(
            binding,
            fit_tolerance=0.0,
            initial_rank=binding.projector.rank,
            max_rank=binding.projector.rank,
            max_iterations=5,
            als_convergence_tolerance=1e-10,
            ridge=1e-12,
            seed=0,
            orthogonality_cutoff=1e-12,
            orthogonality_tolerance=1e-10,
        )

    loose = audit_driver.fit_amplitude_factors(
        binding,
        fit_tolerance=10.0,
        initial_rank=binding.projector.rank,
        max_rank=binding.projector.rank,
        max_iterations=5,
        als_convergence_tolerance=1e-10,
        ridge=1e-12,
        seed=0,
        orthogonality_cutoff=1e-12,
        orthogonality_tolerance=1e-10,
    )
    assert loose.exact_pair_endpoint is False
    invalid = replace(loose, fit_tolerance=0.0)
    assert invalid.weighted_fit_gate_passed is False
    with pytest.raises(ValueError, match='recorded gate'):
        build_factorized_mp2_natural_occupation_weights(
            binding.mp2_doubles,
            invalid,
            occupation_change_floor=1e-12,
            transfer_counter=binding.transfer_counter,
        )
    with pytest.raises(RuntimeError, match='weighted fit gate'):
        _validate_factor_lifecycle(
            binding.doubles,
            invalid,
            binding.loo,
            binding.lov,
            binding.lvv,
        )
