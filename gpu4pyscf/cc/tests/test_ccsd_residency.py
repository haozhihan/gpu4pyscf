"""CPU checks for the canonical GPU-resident residual boundary.

The CUDA driver is exercised on MTU.  These tests keep the algebra and the
host-transfer contract reviewable on machines without PySCF or CuPy.
"""

import ast
import contextlib
import importlib.util
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import pytest


_CC_DIR = Path(__file__).parents[1]
_RUNTIME_PATH = _CC_DIR / "device_runtime.py"
_SPEC = importlib.util.spec_from_file_location(
    "gpu4pyscf_cc_device_runtime_residency", _RUNTIME_PATH
)
runtime = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = runtime
_SPEC.loader.exec_module(runtime)


def _legacy_cpu_residual(
    t1,
    t2,
    *,
    fock,
    mo_e_o,
    mo_e_v,
    orbo,
    orbv,
    wpq,
    t1new,
    t2new,
    wVOov,
    wVooV,
    oooo,
    ovoo,
    oovv,
    ovvo,
):
    """Independent NumPy transcription of the pre-residency update."""

    einsum = np.einsum
    nocc = t1.shape[0]
    t2new *= 0.5

    fov = fock[:nocc, nocc:].copy()
    t1new += fock[:nocc, nocc:]
    foo = fock[:nocc, :nocc] - np.diag(mo_e_o)
    foo += 0.5 * einsum("ia,ja->ij", fock[:nocc, nocc:], t1)
    fvv = einsum("pa,qp,qb->ab", orbv, wpq, orbv)
    t1new -= einsum("ab,ib->ia", fvv, t1)
    fvv += fock[nocc:, nocc:] - np.diag(mo_e_v)
    fvv -= 0.5 * einsum("ia,ib->ab", t1, fock[:nocc, nocc:])
    foo += einsum("pi,qp,qj->ij", orbo, wpq, orbo)
    fov += einsum("pi,qp,qa->ia", orbo, wpq, orbv)

    tau = einsum("ia,jb->ijab", t1, t1) + t2
    woooo = einsum("ijab,kabl->ijkl", tau, ovvo)
    woooo += oooo.transpose(0, 2, 1, 3)
    tmp = einsum("la,jaik->lkji", t1, ovoo)
    woooo += tmp + tmp.transpose(1, 0, 3, 2)
    t2new += 0.5 * einsum("ijkl,klab->ijab", woooo, tau)

    wVOov -= einsum("jbik,ka->bjia", ovoo, t1)
    t2new += wVOov.transpose(1, 2, 0, 3)
    wVooV += einsum("kbij,ka->bija", ovoo, t1)
    wVooV -= oovv.transpose(2, 0, 1, 3)
    wVOov += 0.5 * wVooV
    wVOov += ovvo.transpose(2, 3, 0, 1)
    t2new += (0.5 * ovvo).transpose(0, 3, 1, 2)
    t1new += einsum("pi,pq,qa->ia", orbo, wpq, orbv)
    tmp = einsum("ic,kjbc->ikjb", t1, oovv)
    tmp += einsum("jbck,ic->jkib", ovvo, t1)
    t2new -= einsum("ka,jkib->jiba", t1, tmp)

    tau = 0.5 * t2 + einsum("ia,jb->ijab", t1, t1)
    wVooV += einsum("kbci,jkca->bija", ovvo, tau)
    tmp = einsum("jkca,ckib->jaib", t2, wVooV)
    t2new += tmp.transpose(2, 0, 1, 3)
    tmp *= 0.5
    t2new += tmp.transpose(0, 2, 1, 3)

    tau = einsum("ia,jb->iajb", 0.5 * t1, t1)
    tau += t2.transpose(0, 2, 1, 3)
    ovOV = 2 * ovvo.transpose(0, 1, 3, 2)
    ovOV -= ovvo.transpose(3, 1, 0, 2)
    fvv -= einsum("jcia,jcib->ab", tau, ovOV)
    foo += einsum("iakb,jakb->ij", ovOV, tau)

    theta = 2 * t2.transpose(0, 2, 1, 3)
    theta -= t2.transpose(1, 2, 0, 3)
    tau = 0.25 * theta
    tau -= einsum("ia,jb->jaib", 0.5 * t1, t1)
    wVOov += einsum("kcia,kcjb->aijb", ovOV, tau)
    t2new += einsum("kcia,ckjb->ijab", theta, wVOov)

    t1new += 2 * einsum("jb,ijab->ia", fov, t2)
    t1new -= einsum("jb,ijba->ia", fov, t2)
    ovoo_antisym = 2 * ovoo - ovoo.transpose(2, 1, 0, 3)
    t1new -= einsum("jbki,jkba->ia", ovoo_antisym, t2)

    ft_ij = foo + einsum("ja,ia->ij", 0.5 * t1, fov)
    ft_ab = fvv - einsum("ia,ib->ab", 0.5 * t1, fov)
    t2new += einsum("ijac,bc->ijab", t2, ft_ab)
    t2new -= einsum("ki,kjab->ijab", ft_ij, t2)

    eia = mo_e_o[:, None] - mo_e_v
    t1new += einsum("ib,ab->ia", t1, fvv)
    t1new -= einsum("ja,ji->ia", t1, foo)
    t1new /= eia
    t2new = t2new + t2new.transpose(1, 0, 3, 2)
    t2new /= eia[:, None, :, None] + eia[:, None, :]
    return t1new, t2new


def _small_inputs(seed=118):
    rng = np.random.default_rng(seed)
    nocc, nvir, nao = 2, 3, 5
    random = rng.normal
    return {
        "t1": random(size=(nocc, nvir)) * 0.03,
        "t2": random(size=(nocc, nocc, nvir, nvir)) * 0.02,
        "fock": random(size=(nocc + nvir, nocc + nvir)) * 0.1,
        "mo_e_o": np.array([-1.2, -0.7]),
        "mo_e_v": np.array([0.4, 0.9, 1.4]),
        "orbo": random(size=(nao, nocc)),
        "orbv": random(size=(nao, nvir)),
        "wpq": random(size=(nao, nao)) * 0.05,
        "t1new": random(size=(nocc, nvir)) * 0.1,
        "t2new": random(size=(nocc, nocc, nvir, nvir)) * 0.1,
        "wVOov": random(size=(nvir, nocc, nocc, nvir)) * 0.04,
        "wVooV": random(size=(nvir, nocc, nocc, nvir)) * 0.04,
        "oooo": random(size=(nocc, nocc, nocc, nocc)) * 0.04,
        "ovoo": random(size=(nocc, nvir, nocc, nocc)) * 0.04,
        "oovv": random(size=(nocc, nocc, nvir, nvir)) * 0.04,
        "ovvo": random(size=(nocc, nvir, nvir, nocc)) * 0.04,
    }


def _copy_inputs(inputs):
    return {name: value.copy() for name, value in inputs.items()}


def test_resident_residual_matches_pre_residency_algebra():
    inputs = _small_inputs()
    reference = _legacy_cpu_residual(**_copy_inputs(inputs))
    observed = runtime.canonical_ccsd_residual_update(
        **_copy_inputs(inputs), array_module=np
    )

    np.testing.assert_allclose(observed[0], reference[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(observed[1], reference[1], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        observed[1], observed[1].transpose(1, 0, 3, 2),
        rtol=2e-13, atol=2e-13,
    )


def test_traced_ao_metric_matches_optimized_einsum_without_outer_product():
    rng = np.random.default_rng(912)
    staging = rng.normal(size=(7, 7, 3, 3))
    coefficients = rng.normal(size=(7, 5))

    expected = np.einsum(
        "pqkk,pi,qj->ij", staging, coefficients, coefficients,
        optimize=True,
    )
    observed = runtime.contract_traced_ao_metric(
        staging, coefficients, array_module=np
    )
    observed_transposed = runtime.contract_traced_ao_metric(
        staging, coefficients, array_module=np, transpose_output=True
    )

    np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        observed_transposed, expected.T, rtol=2e-14, atol=2e-14
    )


def test_nonresident_direct_path_preserves_host_staging_policy():
    source = (_CC_DIR / "ccsd_incore.py").read_text()
    tree = ast.parse(source)
    direct = ast.get_source_segment(
        source, _function_node(tree, "_direct_ovvv_vvvv")
    )
    staging_allocator = ast.get_source_segment(
        source, _function_node(tree, "_allocate_direct_resident_staging")
    )

    assert "staging_on_device = resident_cache is not None" in direct
    assert "contract_traced_ao_metric" in direct
    assert "'pqkk,pi,qj->" not in direct
    assert "block_lower_staging = staging_on_device" in direct
    assert "contract_block_lower_ao_to_virtual" in direct
    assert "wVVoo-packed" in staging_allocator
    assert "gpu4pyscf.cc.block-lower-ao.v1" in direct
    assert "allocated_bytes" in direct
    assert "dense_equivalent_bytes" in direct
    assert "saved_live_bytes" in direct


def test_resident_residual_keeps_phase_ledger_without_transfers():
    phases = []

    @contextlib.contextmanager
    def phase(name):
        phases.append(name)
        yield

    runtime.canonical_ccsd_residual_update(
        **_copy_inputs(_small_inputs()), array_module=np, phase=phase
    )
    assert phases == [
        "residual_fock_intermediates",
        "residual_oooo_ladder",
        "residual_linear_ov",
        "residual_wVooV_ring",
        "residual_fock_dressing",
        "residual_wVOov_ring",
        "residual_singles_doubles_coupling",
        "residual_one_body_doubles",
        "residual_denominator_and_symmetry",
    ]


def _function_node(tree, name):
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _load_ccsd_incore_function(name, namespace):
    """Load one dependency-injected helper without importing CUDA/PySCF."""
    tree = ast.parse((_CC_DIR / "ccsd_incore.py").read_text())
    function = _function_node(tree, name)
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    exec(compile(module, str(_CC_DIR / "ccsd_incore.py"), "exec"), namespace)
    return namespace[name]


class _StagingArray:
    def __init__(self, name):
        self.name = name
        self.fill_values = []

    def fill(self, value):
        self.fill_values.append(value)


class _StagingMetrics:
    def __init__(self):
        self.counters = {}

    def increment(self, name, value=1):
        self.counters[name] = self.counters.get(name, 0) + value


_STAGING_OOM = object()


def _run_staging_allocation(outcomes):
    class FakeOOM(Exception):
        pass

    class Planner:
        def __init__(self):
            self.released = []

        def release(self, phase):
            self.released.append(phase)

    class Pool:
        def __init__(self):
            self.free_calls = 0

        def free_all_blocks(self):
            self.free_calls += 1

    pending = iter(outcomes)
    calls = []

    def workspace_array(mycc, phase, name, shape, **kwargs):
        calls.append((phase, name, shape, kwargs))
        outcome = next(pending)
        if outcome is _STAGING_OOM:
            raise FakeOOM("simulated device allocation failure")
        return outcome

    metrics = _StagingMetrics()
    planner = Planner()
    pool = Pool()
    solver = SimpleNamespace(
        run_metrics=metrics,
        _ccsd_workspace_planner=planner,
    )
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            memory=SimpleNamespace(OutOfMemoryError=FakeOOM)
        ),
        get_default_memory_pool=lambda: pool,
    )
    allocate = _load_ccsd_incore_function(
        "_allocate_direct_resident_staging",
        {
            "cupy": fake_cupy,
            "_workspace_array": workspace_array,
        },
    )
    result = allocate(solver, (17, 2, 2), (5, 5, 2, 2))
    return result, metrics, planner, pool, calls


def test_direct_resident_staging_allocation_has_no_retry_on_success():
    packed = _StagingArray("packed")
    dense = _StagingArray("dense")

    result, metrics, planner, pool, calls = _run_staging_allocation(
        [packed, dense]
    )

    assert result == (packed, dense, True)
    assert packed.fill_values == [0]
    assert dense.fill_values == [0]
    assert metrics.counters == {}
    assert planner.released == []
    assert pool.free_calls == 0
    assert [call[1] for call in calls] == ["wVVoo-packed", "wVvoO"]
    assert all(call[3] == {} for call in calls)


def test_direct_resident_staging_retries_once_after_first_oom():
    abandoned = _StagingArray("abandoned-packed")
    packed = _StagingArray("retry-packed")
    dense = _StagingArray("retry-dense")

    result, metrics, planner, pool, calls = _run_staging_allocation(
        [abandoned, _STAGING_OOM, packed, dense]
    )
    assert result == (packed, dense, True)
    assert abandoned.fill_values == []
    assert packed.fill_values == [0]
    assert dense.fill_values == [0]
    assert metrics.counters == {
        "direct_resident_staging_allocation_retries": 1
    }
    assert planner.released == ["direct-resident-staging"]
    assert pool.free_calls == 1
    assert [call[1] for call in calls] == [
        "wVVoo-packed", "wVvoO", "wVVoo-packed", "wVvoO"
    ]


def test_direct_resident_staging_falls_back_only_after_retry_oom():
    first = _StagingArray("first-packed")
    second = _StagingArray("second-packed")

    result, metrics, planner, pool, calls = _run_staging_allocation(
        [first, _STAGING_OOM, second, _STAGING_OOM]
    )

    assert result == (None, None, False)
    assert first.fill_values == []
    assert second.fill_values == []
    assert metrics.counters == {
        "direct_resident_staging_allocation_retries": 1,
        "direct_host_staging_fallbacks": 1,
    }
    assert planner.released == [
        "direct-resident-staging", "direct-resident-staging"
    ]
    assert pool.free_calls == 2
    assert [call[1] for call in calls] == [
        "wVVoo-packed", "wVvoO", "wVVoo-packed", "wVvoO"
    ]


def test_public_update_has_only_the_two_compatibility_downloads():
    source = (_CC_DIR / "ccsd_incore.py").read_text()
    update = _function_node(ast.parse(source), "update_amps")

    direct_gets = [
        node for node in ast.walk(update)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"get", "asnumpy"}
    ]
    assert direct_gets == []

    operations = []
    for node in ast.walk(update):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_host_array"
        ):
            continue
        operation = next(
            keyword.value for keyword in node.keywords
            if keyword.arg == "operation"
        )
        assert isinstance(operation, ast.Constant)
        operations.append(operation.value)
    assert sorted(operations) == [
        "iteration-t1-result-download",
        "iteration-t2-result-download",
    ]


def test_direct_downloads_are_confined_to_explicit_host_staging_fallback():
    source = (_CC_DIR / "ccsd_incore.py").read_text()
    direct = _function_node(ast.parse(source), "_direct_ovvv_vvvv")
    operations = []
    for node in ast.walk(direct):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_host_array"
        ):
            continue
        operation = next(
            keyword.value for keyword in node.keywords
            if keyword.arg == "operation"
        )
        assert isinstance(operation, ast.Constant)
        operations.append(operation.value)
    assert operations
    assert all(label.startswith("direct-host-staging-") for label in operations)


def _symmetric_doubles(rng, nocc, nvir):
    doubles = rng.normal(size=(nocc, nocc, nvir, nvir))
    return doubles + doubles.transpose(1, 0, 3, 2)


def test_rccsd_vector_helpers_match_pyscf_compound_index_layout():
    rng = np.random.default_rng(723)
    nocc, nvir = 3, 4
    t1 = rng.normal(size=(nocc, nvir))
    t2 = _symmetric_doubles(rng, nocc, nvir)
    pair = t2.transpose(0, 2, 1, 3).reshape(nocc * nvir, nocc * nvir)
    lower = np.tril_indices(nocc * nvir)
    expected = np.concatenate((t1.ravel(), pair[lower]))

    observed = runtime.pack_rccsd_amplitudes(t1, t2, array_module=np)
    assert observed.size == runtime.rccsd_amplitude_vector_size(nocc, nvir)
    np.testing.assert_array_equal(observed, expected)

    unpacked_t1, unpacked_t2 = runtime.unpack_rccsd_amplitudes(
        observed, nocc, nvir, array_module=np
    )
    np.testing.assert_array_equal(unpacked_t1, t1)
    np.testing.assert_array_equal(unpacked_t2, t2)


def test_rccsd_packed_delta_reproduces_inherited_convergence_norm():
    rng = np.random.default_rng(811)
    nocc, nvir = 2, 5
    old_t1 = rng.normal(size=(nocc, nvir))
    old_t2 = _symmetric_doubles(rng, nocc, nvir)
    new_t1 = rng.normal(size=(nocc, nvir))
    new_t2 = _symmetric_doubles(rng, nocc, nvir)

    delta = runtime.pack_rccsd_amplitude_difference(
        new_t1, new_t2, old_t1, old_t2, array_module=np
    )
    expected = (
        runtime.pack_rccsd_amplitudes(new_t1, new_t2, array_module=np)
        - runtime.pack_rccsd_amplitudes(old_t1, old_t2, array_module=np)
    )
    np.testing.assert_array_equal(delta, expected)
    np.testing.assert_allclose(np.linalg.norm(delta), np.linalg.norm(expected))


def test_resident_loop_crosses_only_scalar_diis_and_final_boundaries():
    tree = ast.parse((_CC_DIR / "ccsd_incore.py").read_text())
    loop = _function_node(tree, "_resident_iteration_kernel")
    host_operations = []
    initial_uploads = []
    scalar_operations = []
    for node in ast.walk(loop):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in {
            "_host_array", "_device_array", "_device_scalar_to_float"
        }:
            operation = next(
                keyword.value for keyword in node.keywords
                if keyword.arg == "operation"
            )
            assert isinstance(operation, ast.Constant)
            if node.func.id == "_host_array":
                host_operations.append(operation.value)
            elif node.func.id == "_device_array":
                initial_uploads.append(operation.value)
            else:
                scalar_operations.append(operation.value)

    assert sorted(host_operations) == ["final-t1-download", "final-t2-download"]
    assert sorted(initial_uploads) == [
        "resident-initial-t1-upload", "resident-initial-t2-upload"
    ]
    assert sorted(scalar_operations) == [
        "initial-energy-download", "iteration-energy-download"
    ]

    diis = _function_node(tree, "_resident_diis")
    calls = {
        node.func.id: next(
            keyword.value.value for keyword in node.keywords
            if keyword.arg == "operation"
        )
        for node in ast.walk(diis)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"_download_pinned_vector", "_upload_pinned_vector"}
    }
    assert calls == {
        "_download_pinned_vector": "diis-vector-download",
        "_upload_pinned_vector": "diis-vector-upload",
    }


def test_static_eri_uploads_are_confined_to_preloop_cache_builder():
    tree = ast.parse((_CC_DIR / "ccsd_incore.py").read_text())
    prepare = _function_node(tree, "_prepare_resident_eris")
    labels = []
    for node in ast.walk(prepare):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_device_array"
        ):
            continue
        operation = next(
            keyword.value for keyword in node.keywords
            if keyword.arg == "operation"
        )
        assert isinstance(operation, ast.Constant)
        labels.append(operation.value)
    assert sorted(labels) == sorted([
        "resident-static-fock-upload",
        "resident-static-mo-energy-upload",
        "resident-static-mo-coeff-upload",
        "resident-static-oooo-upload",
        "resident-static-ovoo-upload",
        "resident-static-oovv-upload",
        "resident-static-ovvo-upload",
    ])

    loop = _function_node(tree, "_resident_iteration_kernel")
    assert not any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("resident-static-")
        for node in ast.walk(loop)
    )


def test_residual_profiler_uses_deferred_cuda_events_not_enqueue_time():
    tree = ast.parse((_CC_DIR / "ccsd_incore.py").read_text())
    profile = _function_node(tree, "_profile_phase")
    event_calls = [
        node for node in ast.walk(profile)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Event"
    ]
    assert len(event_calls) == 2
    constants = {
        node.value for node in ast.walk(profile)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "residual_" in constants

    flush = _function_node(tree, "_flush_cuda_profile_events")
    flush_constants = {
        node.value for node in ast.walk(flush)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "cuda-event-device-elapsed" in flush_constants
    assert "outer-instrumented-phase" in flush_constants


def test_exception_cleanup_cannot_carry_cuda_events_into_next_lifecycle():
    tree = ast.parse((_CC_DIR / "ccsd_incore.py").read_text())
    release = _function_node(tree, "_release_resident_state")
    assert any(
        isinstance(node, ast.Constant)
        and node.value == "_ccsd_pending_cuda_profile_events"
        for node in ast.walk(release)
    )

    kernel = _function_node(tree, "_resident_iteration_kernel")
    cleanup_calls = [
        node for try_node in ast.walk(kernel)
        if isinstance(try_node, ast.Try)
        for statement in try_node.finalbody
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_release_resident_state"
    ]
    assert len(cleanup_calls) == 1


def test_post_hf_state_scope_owns_one_exception_safe_cleanup():
    released = []
    flushed = []
    scope = _load_ccsd_incore_function(
        "_resident_post_hf_state_scope",
        {
            "contextlib": contextlib,
            "_resident_lifecycle_supported": lambda _solver: (True, None),
            "_release_resident_state": lambda solver: released.append(solver),
            "_free_cupy_cache": lambda solver, phase: flushed.append(
                (solver, phase)
            ),
        },
    )
    solver = SimpleNamespace()

    with pytest.raises(RuntimeError, match="scope failure"):
        with scope(solver) as active:
            assert active is True
            assert solver._ccsd_resident_state_scope_depth == 1
            with scope(solver) as nested_active:
                assert nested_active is True
                assert solver._ccsd_resident_state_scope_depth == 2
            assert solver._ccsd_resident_state_scope_depth == 1
            raise RuntimeError("scope failure")

    assert not hasattr(solver, "_ccsd_resident_state_scope_depth")
    assert released == [solver]
    assert flushed == [(solver, "resident_post_hf_scope_pool_flush")]


def test_post_hf_state_scope_is_noop_for_legacy_or_subclass_solver():
    released = []
    scope = _load_ccsd_incore_function(
        "_resident_post_hf_state_scope",
        {
            "contextlib": contextlib,
            "_resident_lifecycle_supported": lambda _solver: (
                False, "disabled"
            ),
            "_release_resident_state": lambda solver: released.append(solver),
            "_free_cupy_cache": lambda *_args: released.append("flush"),
        },
    )
    solver = SimpleNamespace()

    with scope(solver) as active:
        assert active is False

    assert not hasattr(solver, "_ccsd_resident_state_scope_depth")
    assert released == []


def test_final_residual_operands_reuse_device_amplitudes_and_eri_cache():
    eris = object()
    current_t1 = np.ones((2, 3))
    current_t2 = np.ones((2, 2, 3, 3))
    jacobi_t1 = current_t1 + 1.0
    jacobi_t2 = current_t2 + 2.0
    updates = []
    metrics = _StagingMetrics()
    metrics.metadata = {}
    pool_flushes = []
    synchronizations = []
    planner_releases = []
    planner = SimpleNamespace(
        specs=lambda: [
            SimpleNamespace(
                phase="direct-resident-staging", nbytes=123
            ),
            SimpleNamespace(phase="other", nbytes=456),
        ],
        release=lambda phase: planner_releases.append(phase),
    )
    pool = SimpleNamespace(
        free_all_blocks=lambda: pool_flushes.append(True),
        used_bytes=lambda: 600,
        total_bytes=lambda: 700,
    )
    fake_cupy = SimpleNamespace(
        get_default_memory_pool=lambda: pool,
        cuda=SimpleNamespace(
            get_current_stream=lambda: SimpleNamespace(
                synchronize=lambda: synchronizations.append(True)
            ),
            runtime=SimpleNamespace(memGetInfo=lambda: (200, 1000)),
        ),
    )

    def update_amps(t1, t2, observed_eris):
        assert solver._ccsd_resident_lifecycle is True
        updates.append((t1, t2, observed_eris))
        return jacobi_t1, jacobi_t2

    solver = SimpleNamespace(
        _ccsd_resident_state_scope_depth=1,
        _ccsd_resident_eri_cache={
            "eris": eris,
            "final_t1": current_t1,
            "final_t2": current_t2,
            "mo_energy": np.arange(5.0),
            "direct_vhfopt": object(),
        },
        _ccsd_workspace_planner=planner,
        update_amps=update_amps,
        run_metrics=metrics,
    )
    build = _load_ccsd_incore_function(
        "_resident_final_residual_operands",
        {
            "cupy": fake_cupy,
            "FINAL_RESIDUAL_BLOCK_TARGET_BYTES": 64 * 1024**2,
            "_record_synchronized_hbm_checkpoint": (
                lambda solver, name: {
                    "name": name,
                    "driver_used_bytes": 800,
                    "driver_total_bytes": 1000,
                    "cupy_pool_used_bytes": 600,
                    "cupy_pool_total_bytes": 700,
                    "timing_semantics": "cuda-synchronized-instantaneous",
                    "measurement_scope": "exclusive-slurm-device-global",
                    "source": "cudaMemGetInfo-and-cupy-default-memory-pool",
                }
            ),
            "_final_residual_block_plan": lambda nocc, nvir, itemsize,
                *, target_bytes: (
                    int(nvir), 3 * int(nocc) * int(nocc) * int(nvir)
                    * int(nvir) * int(itemsize)
                ),
        },
    )

    operands = build(solver, eris)

    assert updates == [(current_t1, current_t2, eris)]
    assert operands["current_t1"] is current_t1
    assert operands["current_t2"] is current_t2
    assert operands["jacobi_t1"] is jacobi_t1
    assert operands["jacobi_t2"] is jacobi_t2
    np.testing.assert_array_equal(operands["occupied_energies"], [0.0, 1.0])
    np.testing.assert_array_equal(
        operands["virtual_energies"], [2.0, 3.0, 4.0]
    )
    assert metrics.counters == {
        "resident_final_residual_updates": 1,
        "final_residual_released_staging_bytes": 123,
        "final_residual_staging_release_events": 1,
        "final_residual_temporary_bound_bytes": 864,
    }
    assert metrics.metadata == {
        "final_residual_backend": "gpu-blockwise-fp64",
        "final_residual_doubles_block_size": 3,
        "final_residual_temporary_bound_bytes": 864,
        "final_residual_released_staging_bytes": 123,
    }
    assert operands["doubles_block_size"] == 3
    assert operands["temporary_bound_bytes"] == 864
    assert operands["released_staging_bytes"] == 123
    assert "direct_vhfopt" not in solver._ccsd_resident_eri_cache
    assert planner_releases == ["direct-resident-staging"]
    assert pool_flushes == [True]
    assert operands["update_hbm_checkpoint"]["driver_used_bytes"] == 800
    assert operands["update_hbm_checkpoints"] == [
        operands["update_hbm_checkpoint"]
    ]
    assert not hasattr(solver, "_ccsd_resident_lifecycle")


def test_final_residual_operands_fail_closed_without_matching_retained_cache():
    build = _load_ccsd_incore_function(
        "_resident_final_residual_operands",
        {
            "cupy": SimpleNamespace(),
            "FINAL_RESIDUAL_BLOCK_TARGET_BYTES": 64 * 1024**2,
        },
    )
    solver = SimpleNamespace(
        _ccsd_resident_state_scope_depth=1,
        _ccsd_resident_eri_cache={"eris": object()},
    )

    assert build(solver, object()) is None
    assert not hasattr(solver, "_ccsd_resident_lifecycle")


def test_final_residual_scalar_reader_adds_stable_transfer_prefix():
    calls = []
    reader = _load_ccsd_incore_function(
        "_resident_residual_scalar_to_float",
        {
            "_device_scalar_to_float": (
                lambda solver, value, *, operation: calls.append(
                    (solver, value, operation)
                ) or 3.5
            )
        },
    )
    solver = object()
    scalar = object()

    assert reader(solver, scalar, operation="full-equation-norm") == 3.5
    assert calls == [(
        solver,
        scalar,
        "final-residual-full-equation-norm-download",
    )]


def test_synchronized_hbm_checkpoint_records_driver_and_pool_peaks():
    sync_calls = []
    pool = SimpleNamespace(
        used_bytes=lambda: 600,
        total_bytes=lambda: 700,
    )
    metrics = SimpleNamespace(metadata={}, counters={})
    solver = SimpleNamespace(run_metrics=metrics)
    record = _load_ccsd_incore_function(
        "_record_synchronized_hbm_checkpoint",
        {
            "cupy": SimpleNamespace(
                get_default_memory_pool=lambda: pool,
                cuda=SimpleNamespace(
                    get_current_stream=lambda: SimpleNamespace(
                        synchronize=lambda: sync_calls.append(True)
                    ),
                    runtime=SimpleNamespace(memGetInfo=lambda: (200, 1000)),
                ),
            )
        },
    )

    checkpoint = record(solver, "final-residual-update-live")

    assert sync_calls == [True]
    assert checkpoint == {
        "name": "final-residual-update-live",
        "driver_used_bytes": 800,
        "driver_total_bytes": 1000,
        "cupy_pool_used_bytes": 600,
        "cupy_pool_total_bytes": 700,
        "timing_semantics": "cuda-synchronized-instantaneous",
        "measurement_scope": "exclusive-slurm-device-global",
        "source": "cudaMemGetInfo-and-cupy-default-memory-pool",
    }
    assert metrics.metadata["synchronized_hbm_checkpoints"] == [checkpoint]
    assert metrics.counters == {
        "synchronized_driver_hbm_peak_bytes": 800.0,
        "synchronized_cupy_pool_used_peak_bytes": 600.0,
        "synchronized_cupy_pool_total_peak_bytes": 700.0,
    }


def test_water8_final_residual_block_plan_is_below_192_mib():
    plan = _load_ccsd_incore_function(
        "_final_residual_block_plan",
        {"FINAL_RESIDUAL_BLOCK_TARGET_BYTES": 64 * 1024**2},
    )

    block_size, temporary_bound = plan(40, 424, 8)

    assert block_size == 12
    assert temporary_bound == 195_379_200
    assert temporary_bound < 192 * 1024**2


def test_iteration_kernel_retains_only_device_final_amplitudes_for_outer_scope():
    source = (_CC_DIR / "ccsd_incore.py").read_text()
    tree = ast.parse(source)
    loop = ast.get_source_segment(
        source, _function_node(tree, "_resident_iteration_kernel")
    )

    assert "_ccsd_resident_state_scope_depth" in loop
    assert "cache['final_t1'] = t1" in loop
    assert "cache['final_t2'] = t2" in loop
    assert "if not outer_state_scope" in loop


def test_blockwise_residual_source_never_materializes_full_doubles_delta():
    source = (_CC_DIR / "residual_evaluator.py").read_text()
    tree = ast.parse(source)
    helper = ast.get_source_segment(
        source, _function_node(tree, "_blockwise_doubles_squared_norms")
    )

    assert "a0:a1" in helper
    assert "jacobi_t2 - current_t2" not in helper
    assert "eia[:, None, :, None]" not in helper


def test_resident_loop_is_private_ccsd_bridge_and_has_safe_fallbacks():
    source = (_CC_DIR / "ccsd_incore.py").read_text()
    tree = ast.parse(source)
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "CCSDBase"
            and target.attr == "ccsd"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Name)
    assert assignments[0].value.id == "_resident_ccsd"
    supported = _function_node(tree, "_resident_lifecycle_supported")
    constants = {
        node.value for node in ast.walk(supported)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"disabled", "subclass", "callback", "complex-orbitals"} <= constants


def _cuda_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA device unavailable")
def test_cupy_pack_norm_unpack_and_energy_match_numpy():
    """Exercise the actual device kernels used by the resident lifecycle."""

    import cupy as cp
    from gpu4pyscf.cc import ccsd_incore

    rng = np.random.default_rng(997)
    nocc, nvir = 3, 5
    t1 = rng.normal(size=(nocc, nvir))
    t2 = _symmetric_doubles(rng, nocc, nvir)
    old_t1 = rng.normal(size=(nocc, nvir))
    old_t2 = _symmetric_doubles(rng, nocc, nvir)

    t1_gpu, t2_gpu = cp.asarray(t1), cp.asarray(t2)
    vector_gpu = runtime.pack_rccsd_amplitudes(
        t1_gpu, t2_gpu, array_module=cp
    )
    expected_vector = runtime.pack_rccsd_amplitudes(
        t1, t2, array_module=np
    )
    np.testing.assert_allclose(cp.asnumpy(vector_gpu), expected_vector)
    unpacked = runtime.unpack_rccsd_amplitudes(
        vector_gpu, nocc, nvir, array_module=cp
    )
    np.testing.assert_allclose(cp.asnumpy(unpacked[0]), t1)
    np.testing.assert_allclose(cp.asnumpy(unpacked[1]), t2)

    delta_gpu = runtime.pack_rccsd_amplitude_difference(
        t1_gpu, t2_gpu, cp.asarray(old_t1), cp.asarray(old_t2),
        array_module=cp,
    )
    expected_delta = runtime.pack_rccsd_amplitude_difference(
        t1, t2, old_t1, old_t2, array_module=np
    )
    np.testing.assert_allclose(cp.asnumpy(delta_gpu), expected_delta)

    nmo = nocc + nvir
    fock = rng.normal(size=(nmo, nmo))
    ovvo = rng.normal(size=(nocc, nvir, nvir, nocc))
    eris = SimpleNamespace()
    mycc = SimpleNamespace(
        _ccsd_resident_lifecycle=True,
        _ccsd_resident_eri_cache={
            "eris": eris,
            "fock": cp.asarray(fock),
            "ovvo": cp.asarray(ovvo),
        },
    )
    observed_energy = float(
        ccsd_incore._resident_energy_device(
            mycc, t1_gpu, t2_gpu, eris
        ).get()
    )
    expected_energy = 2 * np.einsum(
        "ia,ia", fock[:nocc, nocc:], t1
    )
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    expected_energy += 2 * np.einsum("ijab,iabj", tau, ovvo)
    expected_energy -= np.einsum("jiab,iabj", tau, ovvo)
    np.testing.assert_allclose(observed_energy, expected_energy, rtol=2e-13)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA device unavailable")
def test_cupy_blockwise_residual_norms_match_dense_cpu_oracle():
    import cupy as cp
    from gpu4pyscf.cc import ccsd_incore
    from gpu4pyscf.cc.residual_evaluator import evaluate_dense_ccsd_residual

    rng = np.random.default_rng(2401)
    nocc, nvir = 2, 5
    current_t1 = rng.normal(size=(nocc, nvir))
    current_t2 = rng.normal(size=(nocc, nocc, nvir, nvir))
    jacobi_t1 = current_t1 + rng.normal(size=current_t1.shape) * 1e-5
    jacobi_t2 = current_t2 + rng.normal(size=current_t2.shape) * 1e-5
    occupied = np.array([-1.1, -0.8])
    virtual = np.linspace(0.2, 1.4, nvir)
    expected = evaluate_dense_ccsd_residual(
        current_t1,
        current_t2,
        jacobi_t1,
        jacobi_t2,
        occupied,
        virtual,
    )
    solver = SimpleNamespace(
        run_metrics=runtime.RunMetrics("cupy-blockwise-residual")
    )
    observed = evaluate_dense_ccsd_residual(
        cp.asarray(current_t1),
        cp.asarray(current_t2),
        cp.asarray(jacobi_t1),
        cp.asarray(jacobi_t2),
        cp.asarray(occupied),
        cp.asarray(virtual),
        doubles_block_size=2,
        scalar_to_float=lambda value, *, operation: (
            ccsd_incore._resident_residual_scalar_to_float(
                solver, value, operation=operation
            )
        ),
    )

    for field in (
        "full_space_equation_norm",
        "full_space_jacobi_update_norm",
        "singles_equation_norm",
        "full_space_doubles_equation_norm",
        "singles_jacobi_update_norm",
        "full_space_doubles_jacobi_update_norm",
    ):
        np.testing.assert_allclose(
            getattr(observed, field), getattr(expected, field),
            rtol=3e-13, atol=3e-15,
        )
    scalar_downloads = solver.run_metrics.transfers.to_dict()[
        "by_operation"
    ]["d2h"]
    assert len(scalar_downloads) == 6
    assert all(
        payload == {"bytes": 8, "count": 1}
        for payload in scalar_downloads.values()
    )


@pytest.mark.skipif(not _cuda_available(), reason="CUDA device unavailable")
def test_end_to_end_resident_lifecycle_matches_compatibility_loop():
    """Compare PySCF DIIS state as well as final resident amplitudes."""

    import pyscf
    from pyscf import lib
    from gpu4pyscf.cc import ccsd_incore

    mol = pyscf.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        verbose=0,
    )
    mf = mol.RHF().run()

    reference = ccsd_incore.CCSD(mf)
    reference.resident_iterations = False
    reference.max_cycle = 4
    reference.conv_tol = 0.0
    reference.conv_tol_normt = 0.0
    reference.iterative_damping = 0.83
    reference.diis_start_cycle = 0
    reference.diis_start_energy_diff = np.inf
    reference.diis = lib.diis.DIIS(reference, incore=True)
    reference.diis.space = 4
    reference.kernel()

    candidate = ccsd_incore.CCSD(mf)
    candidate.run_metrics = runtime.RunMetrics("resident-water-sto3g")
    candidate.max_cycle = reference.max_cycle
    candidate.conv_tol = reference.conv_tol
    candidate.conv_tol_normt = reference.conv_tol_normt
    candidate.iterative_damping = reference.iterative_damping
    candidate.diis_start_cycle = reference.diis_start_cycle
    candidate.diis_start_energy_diff = reference.diis_start_energy_diff
    candidate.diis = lib.diis.DIIS(candidate, incore=True)
    candidate.diis.space = reference.diis.space
    candidate.kernel()

    assert not reference.converged and not candidate.converged
    assert reference.cycles == candidate.cycles == 4
    np.testing.assert_allclose(candidate.e_corr, reference.e_corr, atol=1e-10)
    np.testing.assert_allclose(candidate.t1, reference.t1, atol=2e-8)
    np.testing.assert_allclose(candidate.t2, reference.t2, atol=2e-8)
    assert isinstance(candidate.t1, np.ndarray)
    assert isinstance(candidate.t2, np.ndarray)

    assert reference.diis.get_num_vec() == candidate.diis.get_num_vec()
    for index in range(reference.diis.get_num_vec()):
        np.testing.assert_allclose(
            np.asarray(candidate.diis.get_vec(index)),
            np.asarray(reference.diis.get_vec(index)),
            rtol=2e-10,
            atol=2e-10,
        )
        np.testing.assert_allclose(
            np.asarray(candidate.diis.get_err_vec(index)),
            np.asarray(reference.diis.get_err_vec(index)),
            rtol=2e-9,
            atol=2e-10,
        )

    transfers = candidate.run_metrics.transfers.to_dict()["by_operation"]
    h2d = transfers.get("h2d", {})
    d2h = transfers.get("d2h", {})
    assert "iteration-t1-upload" not in h2d
    assert "iteration-t2-upload" not in h2d
    assert "iteration-t1-result-download" not in d2h
    assert "iteration-t2-result-download" not in d2h
    assert d2h["diis-vector-download"]["count"] == candidate.cycles
    assert h2d["diis-vector-upload"]["count"] == candidate.cycles
    assert d2h["final-t1-download"]["count"] == 1
    assert d2h["final-t2-download"]["count"] == 1
    assert candidate.run_metrics.counters["resident_diis_steps"] == 4

    residual_names = {
        "residual_fock_intermediates",
        "residual_oooo_ladder",
        "residual_linear_ov",
        "residual_wVooV_ring",
        "residual_fock_dressing",
        "residual_wVOov_ring",
        "residual_singles_doubles_coupling",
        "residual_one_body_doubles",
        "residual_denominator_and_symmetry",
    }
    residual_phases = [
        phase for phase in candidate.run_metrics.phases
        if phase["name"] in residual_names
    ]
    assert len(residual_phases) == candidate.cycles * len(residual_names)
    assert {phase["name"] for phase in residual_phases} == residual_names
    assert all(phase["elapsed_s"] >= 0.0 for phase in residual_phases)
    assert sum(phase["elapsed_s"] for phase in residual_phases) > 0.0
    assert all(
        phase["metadata"] == {
            "timing_semantics": "cuda-event-device-elapsed",
            "synchronization": "outer-instrumented-phase",
        }
        for phase in residual_phases
    )
    update_phases = [
        phase for phase in candidate.run_metrics.phases
        if phase["name"] == "ccsd_update"
    ]
    assert len(update_phases) == candidate.cycles
    assert all(
        phase["metadata"]["timing_semantics"] == "cuda-synchronized-wall"
        for phase in update_phases
    )
    assert all(phase["depth"] == 1 for phase in residual_phases)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA device unavailable")
def test_timed_final_residual_reuses_cache_and_downloads_only_norm_scalars():
    import pyscf
    from gpu4pyscf.cc import ccsd_incore
    from gpu4pyscf.cc.residual_evaluator import evaluate_dense_ccsd_residual

    mol = pyscf.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        verbose=0,
    )
    solver = ccsd_incore.CCSD(mol.RHF().run())
    solver.max_cycle = 1
    solver.conv_tol = 0.0
    solver.conv_tol_normt = 0.0
    solver.diis = False
    solver.run_metrics = runtime.RunMetrics("resident-final-residual")

    with solver._resident_post_hf_state_scope() as active:
        assert active is True
        eris = solver.ao2mo()
        solver.kernel(eris=eris)
        retained_cache = solver._ccsd_resident_eri_cache
        operands = solver._resident_final_residual_operands(eris)
        assert solver._ccsd_resident_eri_cache is retained_cache
        block_checkpoints = []
        evaluation = evaluate_dense_ccsd_residual(
            operands["current_t1"],
            operands["current_t2"],
            operands["jacobi_t1"],
            operands["jacobi_t2"],
            operands["occupied_energies"],
            operands["virtual_energies"],
            level_shift=solver.level_shift,
            doubles_block_size=operands["doubles_block_size"],
            scalar_to_float=solver._resident_residual_scalar_to_float,
            block_observer=lambda a0, a1: block_checkpoints.append(
                solver._record_synchronized_hbm_checkpoint(
                    f"final-residual-block-{a0}-{a1}-live"
                )
            ),
        )
        assert evaluation.full_space_equation_norm >= 0.0
        assert "direct_vhfopt" not in retained_cache
        assert block_checkpoints

    assert not hasattr(solver, "_ccsd_resident_eri_cache")
    assert not hasattr(solver, "_ccsd_workspace_planner")
    counters = solver.run_metrics.counters
    assert counters["resident_iterations"] == 1
    assert counters["resident_final_residual_updates"] == 1
    assert counters["direct_resident_staging_iterations"] == 2
    assert counters["final_residual_staging_release_events"] == 1
    assert counters["synchronized_driver_hbm_peak_bytes"] > 0
    transfers = solver.run_metrics.transfers.to_dict()["by_operation"]
    for operation in (
        "resident-static-fock-upload",
        "resident-static-mo-energy-upload",
        "resident-static-mo-coeff-upload",
        "resident-static-oooo-upload",
        "resident-static-ovoo-upload",
        "resident-static-oovv-upload",
        "resident-static-ovvo-upload",
    ):
        assert transfers["h2d"][operation]["count"] == 1
    assert not any(
        "host-staging" in operation
        for direction in transfers.values()
        for operation in direction
    )
    scalar_downloads = {
        operation: payload
        for operation, payload in transfers["d2h"].items()
        if operation.startswith("final-residual-")
    }
    assert len(scalar_downloads) == 6
    assert all(payload == {"bytes": 8, "count": 1} for payload in scalar_downloads.values())


@pytest.mark.skipif(not _cuda_available(), reason="CUDA device unavailable")
def test_resident_exception_discards_pending_cuda_profile_events():
    import cupy as cp
    import pyscf
    from gpu4pyscf.cc import ccsd_incore

    mol = pyscf.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        verbose=0,
    )
    solver = ccsd_incore.CCSD(mol.RHF().run())
    eris = solver.ao2mo()
    solver.run_metrics = runtime.RunMetrics("resident-exception")

    def failing_update(self, _t1, _t2, _eris):
        with ccsd_incore._profile_phase(self, "residual_failure_probe"):
            _ = cp.arange(16.0).sum()
            raise RuntimeError("intentional resident update failure")

    solver.update_amps = MethodType(failing_update, solver)
    with pytest.raises(RuntimeError, match="intentional resident update failure"):
        ccsd_incore._resident_iteration_kernel(
            solver, eris, max_cycle=1, tol=0.0, tolnormt=0.0
        )

    assert not hasattr(solver, "_ccsd_pending_cuda_profile_events")
    assert not hasattr(solver, "_ccsd_resident_eri_cache")
    assert not hasattr(solver, "_ccsd_workspace_planner")
