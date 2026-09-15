"""CPU-only checks for benchmark serialization helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "water8_benchmark", ROOT / "benchmark.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)
SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "water8_snapshot_manifest_for_benchmark_helpers", ROOT / "snapshot_manifest.py"
)
assert SNAPSHOT_SPEC is not None and SNAPSHOT_SPEC.loader is not None
SNAPSHOT = importlib.util.module_from_spec(SNAPSHOT_SPEC)
SNAPSHOT_SPEC.loader.exec_module(SNAPSHOT)


def test_hbm_monitor_close_before_start_is_safe() -> None:
    monitor = BENCHMARK._HBMMonitor()

    monitor.close()

    assert monitor.stop.is_set()
    assert monitor.thread.is_alive() is False


def test_gpu_telemetry_parser_and_post_hf_summary() -> None:
    monitor = BENCHMARK._HBMMonitor()
    monitor.samples = [{"time": 19.0, "MiB": 10, "gpu_uuid": "GPU-a"}]
    monitor.post_hf_started_at = 20.0
    monitor.post_hf_finished_at = 30.0
    monitor.telemetry_samples = (
        BENCHMARK._parse_gpu_telemetry(
            "GPU-a, 0, 1005, 1215, 55.5, 31, P0\n",
            sampled_at=10.0,
        )
        + BENCHMARK._parse_gpu_telemetry(
            "GPU-a, 80, 1410, 1215, 285.0, 48, P0\n",
            sampled_at=22.0,
        )
        + BENCHMARK._parse_gpu_telemetry(
            "GPU-a, 100, 1410, 1215, N/A, 51, P0\n",
            sampled_at=28.0,
        )
    )

    payload = BENCHMARK._gpu_telemetry_payload(monitor)

    overall = payload["overall"]["devices"]["GPU-a"]
    post_hf = payload["post_hf"]["devices"]["GPU-a"]
    assert overall["sample_count"] == 3
    assert overall["gpu_utilization_percent"]["mean"] == 60.0
    assert overall["power_draw_w"]["count"] == 2
    assert post_hf["sample_count"] == 2
    assert post_hf["gpu_utilization_percent"]["median"] == 90.0
    assert post_hf["active_sample_fraction"] == 1.0
    assert post_hf["sm_clock_mhz"]["min"] == 1410.0


def test_gpu_telemetry_excludes_unrelated_visible_devices() -> None:
    monitor = BENCHMARK._HBMMonitor()
    monitor.samples = [{"time": 1.0, "MiB": 20, "gpu_uuid": "GPU-owned"}]
    monitor.telemetry_samples = (
        BENCHMARK._parse_gpu_telemetry(
            "GPU-owned, 75, 1410, 1215, 250, 45, P0\n"
            "GPU-other, 100, 900, 1215, 350, 70, P0\n",
            sampled_at=1.0,
        )
    )

    payload = BENCHMARK._gpu_telemetry_payload(monitor)

    assert list(payload["overall"]["devices"]) == ["GPU-owned"]
    assert payload["attribution"] == {
        "eligible": True,
        "process_gpu_uuids": ["GPU-owned"],
        "unattributed_device_uuids": ["GPU-other"],
        "policy": "process-compute-app-uuid-match",
    }


def test_gpu_telemetry_parser_skips_malformed_rows() -> None:
    rows = BENCHMARK._parse_gpu_telemetry(
        "bad,row\nGPU-b, N/A, N/A, N/A, N/A, N/A, P8\n",
        sampled_at=1.25,
    )

    assert rows == [{
        "time": 1.25,
        "gpu_uuid": "GPU-b",
        "gpu_utilization_percent": None,
        "sm_clock_mhz": None,
        "memory_clock_mhz": None,
        "power_draw_w": None,
        "temperature_c": None,
        "pstate": "P8",
    }]


def _installed_snapshot(tmp_path: Path) -> tuple[Path, Path, Path]:
    task = tmp_path / "task"
    staging = task / "snapshots" / "staging"
    source = staging / "source"
    source.mkdir(parents=True)
    (source / "module.py").write_text("answer = 42\n", encoding="utf-8")
    binary_root = source / "gpu4pyscf" / "lib"
    binary_root.mkdir(parents=True)
    (source / "gpu4pyscf" / "runtime_targets.py").write_text(
        "from gpu4pyscf.lib.utils import load_library\n"
        + "".join(
            f"load_library({name.removesuffix('.so')!r})\n"
            for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
        ),
        encoding="utf-8",
    )
    for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES:
        (binary_root / name).write_bytes(
            f"test-runtime-binary:{name}\n".encode()
        )
    digest, _ = BENCHMARK._source_tree_digest(source)
    target = task / "snapshots" / digest
    staging.rename(target)
    source = target / "source"
    binaries = [
        source / "gpu4pyscf" / "lib" / name
        for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
    ]
    binary = source / "gpu4pyscf" / "lib" / "libgint.so"
    manifest_path = target / "manifest.json"
    allowed = {
        "prefixes": ["benchmarks/cc/a100_water8/"],
        "files": ["gpu4pyscf/cc/device_runtime.py"],
    }
    entries: list[str] = []
    status_bytes = b""
    policy_bytes = json.dumps(
        {"allowed_untracked": allowed, "status": entries},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    manifest_path.write_text(json.dumps({
        "schema": "gpu4pyscf.source-snapshot.v2",
        "tree_sha256": digest,
        "base_revision": BENCHMARK.FROZEN_BASE_COMMIT,
        "source": str(source.resolve()),
        "immutable": True,
        "local_provenance": {
            "schema": "gpu4pyscf.local-provenance.v1",
            "deployment_profile": "candidate",
            "repository_root": "/local/repository",
            "head_revision": BENCHMARK.FROZEN_BASE_COMMIT,
            "frozen_base_revision": BENCHMARK.FROZEN_BASE_COMMIT,
            "head_matches_frozen_base": True,
            "tracked_pristine": True,
            "untracked_paths_allowed": True,
            "canonical_pristine": False,
            "allowed_untracked": allowed,
            "git_status": {
                "format": "porcelain-v1-lines",
                "sha256": hashlib.sha256(status_bytes).hexdigest(),
                "allowed_paths_status_sha256": hashlib.sha256(
                    policy_bytes
                ).hexdigest(),
                "entry_count": 0,
                "entries": entries,
            },
        },
        "runtime_binaries": {
            "source_root": "/pinned/gpu4pyscf/lib",
            "complete": True,
            "required_library_names": list(SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES),
            "inventory_library_names": list(SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES),
            "closure_complete": True,
            "files": [{
                "relative_path": f"gpu4pyscf/lib/{item.name}",
                "sha256": BENCHMARK._file_sha256(item),
                "bytes": item.stat().st_size,
            } for item in binaries],
        },
    }), encoding="utf-8")
    for path in (
        source / "module.py",
        source / "gpu4pyscf" / "runtime_targets.py",
        *binaries,
        manifest_path,
    ):
        os.chmod(path, 0o444)
    for path in (binary.parent, binary.parent.parent, source, target):
        os.chmod(path, 0o555)
    return task, source, binary


def _make_snapshot_writable(source: Path) -> None:
    target = source.parent
    binary_root = source / "gpu4pyscf" / "lib"
    for path in (target, source, binary_root.parent, binary_root):
        os.chmod(path, 0o755)
    for path in (
        source / "module.py",
        source / "gpu4pyscf" / "runtime_targets.py",
        *binary_root.glob("*.so"),
        target / "manifest.json",
    ):
        os.chmod(path, 0o644)


class _CC:
    e_corr = -0.9
    e_tot = -80.9
    t1 = np.ones((1, 2))
    t2 = np.eye(2).reshape(1, 1, 2, 2)


class _FNO:
    corrected_e_corr = -1.0
    corrected_e_tot = -81.0
    delta_mp2 = -0.1


class _Projector:
    vectors = np.eye(2)
    eigenvalues = np.array([2.0, 1.0])


class _Compressed:
    projector = _Projector()
    core = np.diag([2.0, 1.0])
    nocc = 1
    nvir = 2

    def reconstruct_t2(self):
        raise AssertionError("normal benchmark path reconstructed dense t2")

    def metadata(self):
        return {"representation": "rr-doubles"}


def test_fno_energy_payload_uses_delta_mp2_corrected_values() -> None:
    payload = BENCHMARK._energy_payload("fno", _FNO(), _CC())
    assert payload["e_corr"] == -1.0
    assert payload["e_tot"] == -81.0
    assert payload["raw_fno_e_corr"] == -0.9
    assert payload["delta_mp2"] == -0.1
    assert payload["corrected_e_corr"] == -1.0
    assert "Delta" in payload["energy_definition"]


def test_fno_approximation_signature_tracks_effective_selector() -> None:
    threshold = SimpleNamespace(
        fno_thresh=1e-5, fno_pct_occ=None, fno_nvir_act=None
    )
    other_threshold = SimpleNamespace(
        fno_thresh=1e-6, fno_pct_occ=None, fno_nvir_act=None
    )
    fixed_rank = SimpleNamespace(
        fno_thresh=1e-4, fno_pct_occ=0.9, fno_nvir_act=17
    )
    same_fixed_rank = SimpleNamespace(
        fno_thresh=1e-8, fno_pct_occ=0.99, fno_nvir_act=17
    )
    payload, signature = BENCHMARK.approximation_identity("fno", threshold)
    _, other_signature = BENCHMARK.approximation_identity("fno", other_threshold)
    fixed_payload, fixed_signature = BENCHMARK.approximation_identity(
        "fno", fixed_rank
    )
    _, same_fixed_signature = BENCHMARK.approximation_identity(
        "fno", same_fixed_rank
    )
    assert payload["selection"] == {
        "mode": "occupation_threshold", "value": 1e-5
    }
    assert signature != other_signature
    assert fixed_payload["selection"] == {"mode": "nvir_act", "value": 17}
    assert fixed_signature == same_fixed_signature


def test_compressed_summary_and_checkpoint_do_not_reconstruct(tmp_path: Path) -> None:
    doubles = _Compressed()
    result = (-0.9, _CC.t1, doubles)
    summary = BENCHMARK._amplitude_payload(result, _CC(), np)
    assert summary["dense_t2_materialized_for_summary"] is False
    assert summary["t2_shape"] == [1, 1, 2, 2]

    path = tmp_path / "restart.npz"
    checkpoint = BENCHMARK._write_checkpoint_once(
        path, "rr", result, _CC(), np
    )
    cached = checkpoint.pop("_cached_amplitude_payload")
    summary = BENCHMARK._amplitude_payload(
        result, _CC(), np, cached_payload=cached
    )
    assert checkpoint["included_in_post_hf"] is True
    assert checkpoint["representation"] == "rr-doubles"
    assert checkpoint["dense_t2_reconstructed"] is False
    with np.load(path) as payload:
        assert payload["checkpoint_schema"][0] == "gpu4pyscf.cc.restart.v1"
        assert payload["representation"][0] == "rr-doubles"
        assert "rr_projector" in payload
        assert "rr_eigenvalues" in payload
        assert "compressed_core" in payload
        assert "t2" not in payload
    assert summary["summary_transfer_mode"] == (
        "reused-counted-checkpoint-host-buffers"
    )


def test_checkpoint_summary_reuses_counted_device_buffers_without_redownload(
    tmp_path: Path,
) -> None:
    class Ledger:
        def __init__(self):
            self.operations = []

        def record_d2h(self, nbytes, *, operation):
            self.operations.append((operation, nbytes))

    class DeviceArray:
        def __init__(self, host):
            self.host = np.asarray(host)
            self.shape = self.host.shape
            self.nbytes = self.host.nbytes
            self.get_calls = 0

        def get(self):
            self.get_calls += 1
            return self.host.copy()

    class Projector:
        vectors = DeviceArray(np.eye(2))
        eigenvalues = DeviceArray(np.array([2.0, 1.0]))
        source = "test-projector"
        cutoff = 1e-6

    class Doubles:
        projector = Projector()
        core = DeviceArray(np.diag([2.0, 1.0]))
        nocc = 1
        nvir = 2
        storage_nbytes = projector.vectors.nbytes + core.nbytes

        def reconstruct_t2(self):
            raise AssertionError("summary reconstructed dense t2")

        def metadata(self):
            raise AssertionError("summary redownloaded doubles metadata")

    ledger = Ledger()
    t1 = DeviceArray(np.ones((1, 2)))
    doubles = Doubles()
    cc_obj = SimpleNamespace(
        e_corr=-0.9,
        t1=t1,
        t2=doubles,
        run_metrics=SimpleNamespace(transfers=ledger),
    )
    result = (-0.9, t1, doubles)
    checkpoint = BENCHMARK._write_checkpoint_once(
        tmp_path / "device.npz", "rr_cd", result, cc_obj, np
    )
    cached = checkpoint.pop("_cached_amplitude_payload")
    calls_after_checkpoint = {
        "t1": t1.get_calls,
        "projector": doubles.projector.vectors.get_calls,
        "eigenvalues": doubles.projector.eigenvalues.get_calls,
        "core": doubles.core.get_calls,
    }

    summary = BENCHMARK._amplitude_payload(
        result, cc_obj, np, cached_payload=cached
    )

    assert calls_after_checkpoint == {
        "t1": 1,
        "projector": 1,
        "eigenvalues": 1,
        "core": 1,
    }
    assert t1.get_calls == 1
    assert doubles.projector.vectors.get_calls == 1
    assert doubles.projector.eigenvalues.get_calls == 1
    assert doubles.core.get_calls == 1
    assert ledger.operations == [
        ("checkpoint_t1", t1.nbytes),
        ("checkpoint_rr_projector", doubles.projector.vectors.nbytes),
        ("checkpoint_rr_eigenvalues", doubles.projector.eigenvalues.nbytes),
        ("checkpoint_rr_core", doubles.core.nbytes),
    ]
    assert summary["t1_norm"] == np.linalg.norm(t1.host)
    assert summary["t2_norm"] == np.linalg.norm(doubles.core.host)
    assert summary["dense_t2_materialized_for_summary"] is False


def test_checkpoint_skip_uses_device_norms_and_counts_only_scalar_downloads(
    monkeypatch,
) -> None:
    class Ledger:
        def __init__(self):
            self.operations = []

        def record_d2h(self, nbytes, *, operation):
            self.operations.append((operation, nbytes))

    class DeviceArray:
        def __init__(self, shape, norm):
            self.shape = shape
            self.norm = norm
            self.nbytes = int(np.prod(shape)) * 8

        def get(self):
            raise AssertionError("amplitude summary downloaded a bulk array")

    DeviceArray.__module__ = "cupy._core.core"

    scalar_gets = []

    class DeviceScalar:
        nbytes = 8

        def __init__(self, value):
            self.value = value

        def get(self):
            scalar_gets.append(self.value)
            return np.asarray(self.value)

    fake_cupy = ModuleType("cupy")
    fake_cupy.linalg = SimpleNamespace(
        norm=lambda value: DeviceScalar(value.norm)
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    class Projector:
        vectors = DeviceArray((2, 2), 2.0)
        eigenvalues = DeviceArray((2,), 2.0)
        source = "test-projector"
        cutoff = 1e-6

    class Doubles:
        projector = Projector()
        core = DeviceArray((2, 2), 3.0)
        nocc = 1
        nvir = 2
        storage_nbytes = projector.vectors.nbytes + core.nbytes

        def reconstruct_t2(self):
            raise AssertionError("summary reconstructed dense t2")

        def metadata(self):
            raise AssertionError("summary downloaded projector metadata")

    ledger = Ledger()
    t1 = DeviceArray((1, 2), 2.5)
    doubles = Doubles()
    cc_obj = SimpleNamespace(
        t1=t1,
        t2=doubles,
        run_metrics=SimpleNamespace(transfers=ledger),
    )

    summary = BENCHMARK._amplitude_payload(
        (-0.9, t1, doubles), cc_obj, np
    )

    assert summary["t1_norm"] == 2.5
    assert summary["t2_norm"] == 3.0
    assert summary["summary_transfer_mode"] == "device-norm-scalars-only"
    assert scalar_gets == [2.5, 3.0]
    assert ledger.operations == [
        ("amplitude_summary_t1_norm", 8),
        ("amplitude_summary_rr_core_norm", 8),
    ]


def test_run_snapshots_experiment_record_after_all_amplitude_accounting() -> None:
    source = inspect.getsource(BENCHMARK.run)

    summary_position = source.index("amplitude_payload = _amplitude_payload(")
    snapshot_position = source.index(
        "experiment_record = cc_obj.experiment_record()"
    )

    assert summary_position < snapshot_position


def test_rr_cd_kernel_lifecycle_never_calls_harness_ao2mo() -> None:
    result = (-0.9, np.ones((1, 2)), _Compressed())

    class DirectSolver:
        def ao2mo(self, *_args, **_kwargs):
            raise AssertionError("rr_cd benchmark invoked the dense ao2mo seam")

        def kernel(self):
            return result

    solver = DirectSolver()
    observed, eris = BENCHMARK._execute_solver_kernel("rr_cd", solver, solver)

    assert observed is result
    assert eris is None


def test_thc_cd_kernel_lifecycle_never_calls_harness_ao2mo() -> None:
    result = (-0.9, np.ones((1, 2)), _Compressed())

    class DirectSolver:
        def ao2mo(self, *_args, **_kwargs):
            raise AssertionError("thc_cd benchmark invoked the dense ao2mo seam")

        def kernel(self):
            return result

    solver = DirectSolver()
    observed, eris = BENCHMARK._execute_solver_kernel(
        "thc_cd", solver, solver
    )

    assert observed is result
    assert eris is None


def test_rr_cd_constructor_mapping_exposes_every_public_control() -> None:
    args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "rr_cd",
        "--eri-tol", "1e-6",
        "--direct-scf-tol", "2e-13",
        "--cd-max-rank", "301",
        "--gint-column-backend", "restricted-reference",
        "--gint-group-size", "7",
        "--gint-max-block-bytes", "999999",
        "--gint-max-batch-size", "13",
        "--cd-mo-block-size", "11",
        "--rr-eig-cutoff", "3e-5",
        "--rr-max-rank", "79",
        "--denominator-tolerance", "4e-10",
        "--denominator-max-rank", "83",
        "--rr-initial-rank", "13",
        "--rr-solver-tolerance", "5e-11",
        "--rr-solver-maxiter", "107",
        "--rr-dense-fallback-dimension", "17",
        "--rr-ritz-residual-tolerance", "6e-9",
        "--rr-auxiliary-block-size", "19",
        "--rr-virtual-block-size", "23",
    ])

    assert BENCHMARK._rr_cd_constructor_options(args) == {
        "eri_backend": "cd",
        "eri_tol": 1e-6,
        "direct_scf_tol": 2e-13,
        "cd_max_rank": 301,
        "gint_column_backend": "restricted-reference",
        "gint_group_size": 7,
        "gint_max_block_bytes": 999999,
        "gint_max_batch_size": 13,
        "cd_mo_block_size": 11,
        "rr_eig_cutoff": 3e-5,
        "rr_max_rank": 79,
        "denominator_tolerance": 4e-10,
        "denominator_max_rank": 83,
        "rr_initial_rank": 13,
        "rr_solver_tolerance": 5e-11,
        "rr_solver_maxiter": 107,
        "rr_dense_fallback_dimension": 17,
        "rr_ritz_residual_tolerance": 6e-9,
        "rr_auxiliary_block_size": 19,
        "rr_virtual_block_size": 23,
        "precision": "fp64",
    }


def test_thc_cd_constructor_mapping_exposes_all_rr_and_thc_controls() -> None:
    args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "thc_cd",
        "--eri-tol", "1e-7",
        "--direct-scf-tol", "2e-13",
        "--cd-max-rank", "301",
        "--gint-group-size", "7",
        "--gint-max-block-bytes", "999999",
        "--cd-mo-block-size", "11",
        "--rr-eig-cutoff", "3e-5",
        "--rr-max-rank", "79",
        "--denominator-tolerance", "4e-10",
        "--denominator-max-rank", "83",
        "--rr-initial-rank", "13",
        "--rr-solver-tolerance", "5e-11",
        "--rr-solver-maxiter", "107",
        "--rr-dense-fallback-dimension", "17",
        "--rr-ritz-residual-tolerance", "6e-9",
        "--rr-auxiliary-block-size", "19",
        "--rr-virtual-block-size", "23",
        "--thc-fit-tol", "7e-7",
        "--thc-rank", "1060",
        "--thc-max-rank", "1060",
        "--thc-rank-growth", "1.7",
        "--thc-orthogonality-cutoff", "8e-13",
        "--thc-orthogonality-tolerance", "9e-11",
        "--thc-max-iterations", "41",
        "--thc-als-convergence-tolerance", "2e-11",
        "--thc-ridge", "3e-13",
        "--thc-seed", "17",
        "--thc-replacement-validation-atol", "4e-12",
        "--thc-replacement-validation-rtol", "5e-12",
    ])

    options = BENCHMARK._thc_cd_constructor_options(args)

    assert {
        key: options[key]
        for key in BENCHMARK._rr_cd_constructor_options(args)
    } == BENCHMARK._rr_cd_constructor_options(args)
    assert options["eri_backend"] == "cd"
    assert options["thc_fit_tol"] == 7e-7
    assert options["thc_rank"] == 1060
    assert options["thc_initial_rank"] is None
    assert options["thc_max_rank"] == 1060
    assert options["thc_rank_growth"] == 1.7
    assert options["thc_orthogonality_cutoff"] == 8e-13
    assert options["thc_orthogonality_tolerance"] == 9e-11
    assert options["thc_max_iterations"] == 41
    assert options["thc_als_convergence_tolerance"] == 2e-11
    assert options["thc_ridge"] == 3e-13
    assert options["thc_seed"] == 17
    assert options["thc_replacement_validation_atol"] == 4e-12
    assert options["thc_replacement_validation_rtol"] == 5e-12


def test_make_solver_routes_rr_cd_to_rrccsd_and_never_constructs_thc(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeRRCCSD:
        def __init__(self, mf, **kwargs):
            captured["mf"] = mf
            captured["kwargs"] = kwargs

    class ForbiddenTHCRRCCSD:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("rr_cd fell through to THCRRCCSD")

    package = ModuleType("gpu4pyscf")
    cc_package = ModuleType("gpu4pyscf.cc")
    incore_module = ModuleType("gpu4pyscf.cc.ccsd_incore")
    rr_module = ModuleType("gpu4pyscf.cc.rrccsd")
    rr_module.RRCCSD = FakeRRCCSD
    thc_module = ModuleType("gpu4pyscf.cc.thc_rrccsd")
    thc_module.THCRRCCSD = ForbiddenTHCRRCCSD
    package.cc = cc_package
    cc_package.ccsd_incore = incore_module
    for name, module in {
        "gpu4pyscf": package,
        "gpu4pyscf.cc": cc_package,
        "gpu4pyscf.cc.ccsd_incore": incore_module,
        "gpu4pyscf.cc.rrccsd": rr_module,
        "gpu4pyscf.cc.thc_rrccsd": thc_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "rr_cd",
        "--eri-tol", "1e-6",
        "--direct-scf-tol", "2e-13",
        "--denominator-tolerance", "3e-10",
        "--rr-eig-cutoff", "4e-5",
    ])
    mf = object()
    solver, metadata = BENCHMARK._make_solver(args, mf)

    assert isinstance(solver, FakeRRCCSD)
    assert captured["mf"] is mf
    assert captured["kwargs"] == BENCHMARK._rr_cd_constructor_options(args)
    assert captured["kwargs"]["eri_backend"] == "cd"
    assert solver.frozen == 0
    assert metadata is BENCHMARK.METHOD_INFO["rr_cd"]


def test_make_solver_routes_thc_cd_with_fail_closed_endpoint_controls(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class ForbiddenRRCCSD:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("thc_cd was routed to RRCCSD")

    class FakeTHCRRCCSD:
        def __init__(self, mf, **kwargs):
            captured["mf"] = mf
            captured["kwargs"] = kwargs

    package = ModuleType("gpu4pyscf")
    cc_package = ModuleType("gpu4pyscf.cc")
    incore_module = ModuleType("gpu4pyscf.cc.ccsd_incore")
    rr_module = ModuleType("gpu4pyscf.cc.rrccsd")
    rr_module.RRCCSD = ForbiddenRRCCSD
    thc_module = ModuleType("gpu4pyscf.cc.thc_rrccsd")
    thc_module.THCRRCCSD = FakeTHCRRCCSD
    package.cc = cc_package
    cc_package.ccsd_incore = incore_module
    for name, module in {
        "gpu4pyscf": package,
        "gpu4pyscf.cc": cc_package,
        "gpu4pyscf.cc.ccsd_incore": incore_module,
        "gpu4pyscf.cc.rrccsd": rr_module,
        "gpu4pyscf.cc.thc_rrccsd": thc_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "thc_cd",
        "--thc-rank", "1060",
        "--thc-replacement-validation-atol", "1e-11",
        "--thc-replacement-validation-rtol", "2e-11",
    ])
    mf = object()

    solver, metadata = BENCHMARK._make_solver(args, mf)

    assert isinstance(solver, FakeTHCRRCCSD)
    assert captured["mf"] is mf
    assert captured["kwargs"] == BENCHMARK._thc_cd_constructor_options(args)
    assert captured["kwargs"]["thc_rank"] == 1060
    assert captured["kwargs"]["thc_replacement_validation_atol"] == 1e-11
    assert captured["kwargs"]["thc_replacement_validation_rtol"] == 2e-11
    assert solver.frozen == 0
    assert metadata is BENCHMARK.METHOD_INFO["thc_cd"]
    assert metadata["performance_eligible"] is False


def test_canonical_legacy_routes_only_the_iteration_execution_flag(
    monkeypatch,
) -> None:
    created = []

    class FakeCanonical:
        resident_iterations = True

        def __init__(self, mf):
            self.mf = mf
            created.append(self)

    package = ModuleType("gpu4pyscf")
    cc_package = ModuleType("gpu4pyscf.cc")
    incore_module = ModuleType("gpu4pyscf.cc.ccsd_incore")
    incore_module.CCSD = FakeCanonical
    package.cc = cc_package
    cc_package.ccsd_incore = incore_module
    for name, module in {
        "gpu4pyscf": package,
        "gpu4pyscf.cc": cc_package,
        "gpu4pyscf.cc.ccsd_incore": incore_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    canonical_args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz", "--method", "canonical",
    ])
    legacy_args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz", "--method", "canonical_legacy",
    ])
    mf = object()

    canonical, canonical_metadata = BENCHMARK._make_solver(canonical_args, mf)
    legacy, legacy_metadata = BENCHMARK._make_solver(legacy_args, mf)

    assert canonical.resident_iterations is True
    assert legacy.resident_iterations is False
    assert canonical.frozen == legacy.frozen == 0
    assert canonical.conv_tol == legacy.conv_tol
    assert canonical.conv_tol_normt == legacy.conv_tol_normt
    assert canonical.max_cycle == legacy.max_cycle
    assert canonical.max_memory == legacy.max_memory
    assert canonical_metadata["acceptance_table"] == "S"
    assert legacy_metadata["acceptance_table"] == "S"
    assert canonical_metadata["resident_iterations"] is True
    assert legacy_metadata["resident_iterations"] is False
    assert canonical_metadata["resident_iterations_policy"] == "solver-default"
    assert legacy_metadata["resident_iterations_policy"] == "forced-disabled"
    case = BENCHMARK.load_cases()["water2-tz"]
    canonical_plan = BENCHMARK._plan(canonical_args, case)
    legacy_plan = BENCHMARK._plan(legacy_args, case)
    assert canonical_plan["solver_lifecycle"]["resident_iterations"] is None
    assert canonical_plan["solver_lifecycle"][
        "resident_iterations_policy"
    ] == "solver-default"
    assert legacy_plan["solver_lifecycle"]["resident_iterations"] is False
    assert legacy_plan["solver_lifecycle"][
        "resident_iterations_policy"
    ] == "forced-disabled"
    canonical_approximation = BENCHMARK.approximation_identity(
        "canonical", canonical_args
    )
    legacy_approximation = BENCHMARK.approximation_identity(
        "canonical_legacy", legacy_args
    )
    assert legacy_approximation == canonical_approximation
    assert created == [canonical, legacy]


def test_rr_cd_approximation_identity_separates_integral_and_denominator_controls() -> None:
    controls = SimpleNamespace(
        eri_tol=1e-8,
        direct_scf_tol=1e-13,
        cd_max_rank=None,
        rr_eig_cutoff=1e-6,
        rr_max_rank=None,
        denominator_tolerance=1e-10,
        denominator_max_rank=None,
        rr_solver_tolerance=1e-10,
        rr_solver_maxiter=None,
        rr_dense_fallback_dimension=64,
        rr_ritz_residual_tolerance=None,
        precision="fp64",
    )
    changed_screening = SimpleNamespace(**vars(controls))
    changed_screening.direct_scf_tol = 1e-12
    changed_denominator = SimpleNamespace(**vars(controls))
    changed_denominator.denominator_tolerance = 1e-9

    payload, signature = BENCHMARK.approximation_identity("rr_cd", controls)
    _, screening_signature = BENCHMARK.approximation_identity(
        "rr_cd", changed_screening
    )
    _, denominator_signature = BENCHMARK.approximation_identity(
        "rr_cd", changed_denominator
    )

    assert payload["eri_backend"] == "cd"
    assert payload["direct_scf_tol"] == 1e-13
    assert payload["denominator_tolerance"] == 1e-10
    assert signature != screening_signature
    assert signature != denominator_signature


def test_thc_cd_identity_includes_every_cd_rr_and_thc_endpoint_control() -> None:
    args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "thc_cd",
        "--thc-rank", "1060",
        "--thc-max-rank", "1060",
        "--thc-replacement-validation-atol", "1e-11",
        "--thc-replacement-validation-rtol", "2e-11",
    ])
    changed = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "thc_cd",
        "--thc-rank", "1060",
        "--thc-max-rank", "1060",
        "--thc-replacement-validation-atol", "3e-11",
        "--thc-replacement-validation-rtol", "2e-11",
    ])

    payload, signature = BENCHMARK.approximation_identity("thc_cd", args)
    _, changed_signature = BENCHMARK.approximation_identity("thc_cd", changed)

    for name in BENCHMARK._rr_cd_constructor_options(args):
        if name not in {
            "gint_column_backend", "gint_group_size", "gint_max_block_bytes",
            "gint_max_batch_size", "cd_mo_block_size",
            "rr_initial_rank", "rr_auxiliary_block_size",
            "rr_virtual_block_size",
        }:
            assert name in payload
    for name in (
        "thc_fit_tol", "thc_rank", "thc_initial_rank", "thc_max_rank",
        "thc_rank_growth", "thc_orthogonality_cutoff",
        "thc_orthogonality_tolerance", "thc_max_iterations",
        "thc_als_convergence_tolerance", "thc_ridge", "thc_seed",
        "thc_replacement_validation_atol",
        "thc_replacement_validation_rtol",
    ):
        assert name in payload
    assert payload["eri_backend"] == "cd"
    assert payload["thc_rank"] == 1060
    assert signature != changed_signature


def test_thc_cd_configuration_requires_exact_explicit_endpoint() -> None:
    missing_rank = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "thc_cd",
        "--eri-tol", "1e-8",
        "--rr-eig-cutoff", "1e-6",
        "--thc-fit-tol", "1e-6",
        "--thc-replacement-validation-atol", "1e-11",
        "--thc-replacement-validation-rtol", "1e-11",
    ])
    inexact_rank = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "thc_cd",
        "--eri-tol", "1e-8",
        "--rr-eig-cutoff", "1e-6",
        "--thc-fit-tol", "1e-6",
        "--thc-rank", "1059",
        "--thc-replacement-validation-atol", "1e-11",
        "--thc-replacement-validation-rtol", "1e-11",
    ])
    exact_rank = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "thc_cd",
        "--eri-tol", "1e-8",
        "--rr-eig-cutoff", "1e-6",
        "--thc-fit-tol", "1e-6",
        "--thc-rank", "1060",
        "--thc-replacement-validation-atol", "1e-11",
        "--thc-replacement-validation-rtol", "1e-11",
    ])

    with np.testing.assert_raises_regex(ValueError, "explicit --thc-rank"):
        BENCHMARK._validate_run_configuration(missing_rank)
    with np.testing.assert_raises_regex(ValueError, "inexact thc_cd"):
        BENCHMARK._validate_run_configuration(inexact_rank)
    BENCHMARK._validate_run_configuration(exact_rank)
    assert BENCHMARK.METHOD_INFO["thc_cd"]["performance_eligible"] is False


def test_rr_and_thc_methods_reject_implicit_approximation_defaults() -> None:
    for method in ("rr_cd", "rr_canonical", "thc_cd", "thc_canonical"):
        args = BENCHMARK._parser().parse_args([
            "--case", "water2-tz", "--method", method,
        ])
        with np.testing.assert_raises_regex(
            ValueError, "requires explicit approximation controls"
        ):
            BENCHMARK._validate_run_configuration(args)

    for method in ("canonical", "canonical_legacy", "fno"):
        args = BENCHMARK._parser().parse_args([
            "--case", "water2-tz", "--method", method,
        ])
        BENCHMARK._validate_run_configuration(args)


def test_fno_configuration_rejects_nonfinite_or_invalid_selectors() -> None:
    invalid_argv = (
        ["--fno-thresh", "nan"],
        # argparse in Python 3.10 treats a separate negative scientific-
        # notation token as an option; the equals spelling reaches the same
        # float validator consistently on both the local 3.10 and MTU 3.11.
        ["--fno-thresh=-1e-6"],
        ["--fno-pct-occ", "1.01"],
        ["--fno-nvir-act", "0"],
    )
    for extra in invalid_argv:
        args = BENCHMARK._parser().parse_args([
            "--case", "water2-tz", "--method", "fno", *extra,
        ])
        with np.testing.assert_raises(ValueError):
            BENCHMARK._validate_run_configuration(args)


def test_cli_tracks_required_approximation_controls_in_the_plan() -> None:
    args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "thc_canonical",
        "--eri-tol", "1e-8",
        "--rr-eig-cutoff", "1e-6",
        "--thc-fit-tol", "1e-5",
    ])

    BENCHMARK._validate_run_configuration(args)
    plan = BENCHMARK._plan(
        args, BENCHMARK.load_cases()["water2-tz"]
    )

    assert args._explicit_approximation_controls == frozenset({
        "eri_tol", "rr_eig_cutoff", "thc_fit_tol",
    })
    provenance = plan["approximation_control_provenance"]
    assert provenance["required_explicit"] == [
        "eri_tol", "rr_eig_cutoff", "thc_fit_tol",
    ]
    assert provenance["provided_via_cli_or_config"] == [
        "eri_tol", "rr_eig_cutoff", "thc_fit_tol",
    ]
    assert provenance["all_required_explicit"] is True
    assert provenance["source_by_control"] == {
        "eri_tol": "cli",
        "rr_eig_cutoff": "cli",
        "thc_fit_tol": "cli",
    }


def test_config_values_count_as_explicit_approximation_controls(
    tmp_path: Path,
) -> None:
    config = tmp_path / "rr-config.json"
    config.write_text(json.dumps({
        "eri_tol": 1e-7,
        "rr_eig_cutoff": 2e-6,
    }))
    args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "rr_cd",
        "--config", str(config),
    ])

    args = BENCHMARK._apply_config(args)
    BENCHMARK._validate_run_configuration(args)
    plan = BENCHMARK._plan(
        args, BENCHMARK.load_cases()["water2-tz"]
    )

    assert args.eri_tol == 1e-7
    assert args.rr_eig_cutoff == 2e-6
    assert args._explicit_approximation_controls == frozenset({
        "eri_tol", "rr_eig_cutoff",
    })
    assert plan["approximation_control_provenance"][
        "all_required_explicit"
    ] is True
    assert plan["approximation_control_provenance"][
        "source_by_control"
    ] == {
        "eri_tol": "config",
        "rr_eig_cutoff": "config",
    }


def test_missing_controls_fail_before_output_or_compute_setup(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"
    args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "rr_canonical",
        "--output-dir", str(output),
    ])

    with np.testing.assert_raises_regex(
        ValueError, "requires explicit approximation controls"
    ):
        BENCHMARK.run(args)

    assert output.exists() is False


def test_rr_cd_dry_run_preserves_source_and_full_protocol(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    source = {
        "revision": "tree-sha256:abc",
        "base_revision": BENCHMARK.FROZEN_BASE_COMMIT,
        "tree_sha256": "abc",
        "source_kind": "content-manifest",
    }
    monkeypatch.setattr(BENCHMARK, "_git_state", lambda: source)
    args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "rr_cd",
        "--device", "gpu",
        "--output-dir", str(tmp_path),
        "--dry-run",
        "--eri-tol", "1e-6",
        "--direct-scf-tol", "2e-13",
        "--denominator-tolerance", "3e-10",
        "--denominator-max-rank", "91",
        "--rr-eig-cutoff", "1e-5",
        "--rr-ritz-residual-tolerance", "4e-9",
    ])

    assert BENCHMARK.run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    plan = payload["plan"]

    assert payload["dry_run"] is True
    assert payload["source"] == source
    assert plan["protocol_schema"] == BENCHMARK.PROTOCOL_SCHEMA
    assert plan["source_protocol"]["frozen_base_commit"] == (
        BENCHMARK.FROZEN_BASE_COMMIT
    )
    assert plan["solver_lifecycle"]["benchmark_prebuilds_eris"] is False
    assert plan["solver_lifecycle"]["dense_ao2mo_in_normal_path"] is False
    assert plan["solver_lifecycle"]["dense_t2_in_normal_path"] is False
    assert plan["settings"]["eri_tol"] == 1e-6
    assert plan["settings"]["direct_scf_tol"] == 2e-13
    assert plan["settings"]["denominator_tolerance"] == 3e-10
    assert plan["settings"]["denominator_max_rank"] == 91
    assert plan["settings"]["rr_eig_cutoff"] == 1e-5
    assert plan["settings"]["rr_ritz_residual_tolerance"] == 4e-9
    assert plan["settings"]["eri_backend"] == "cd"
    assert plan["performance_eligible"] is False
    assert "rr_cd" in plan["approximation_signature"]
    assert list(tmp_path.iterdir()) == []


def test_dry_run_does_not_create_output_directory(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    source = {
        "revision": "tree-sha256:abc",
        "base_revision": BENCHMARK.FROZEN_BASE_COMMIT,
        "tree_sha256": "abc",
        "source_kind": "content-manifest",
    }
    monkeypatch.setattr(BENCHMARK, "_git_state", lambda: source)
    output = tmp_path / "immutable-snapshot" / "results"
    args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "canonical",
        "--device", "cpu",
        "--output-dir", str(output),
        "--dry-run",
    ])

    assert BENCHMARK.run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["source"] == source
    assert output.exists() is False


def test_rr_cd_static_gates_cannot_be_overridden_by_runtime_metadata() -> None:
    solver = SimpleNamespace(
        method_metadata=lambda: {
            "performance_eligible": True,
            "projected_equation_residual": 7e-7,
        }
    )
    metadata = BENCHMARK._runtime_method_metadata(
        solver, BENCHMARK.METHOD_INFO["rr_cd"]
    )
    record = {}
    BENCHMARK._apply_method_eligibility(record, metadata)

    assert metadata["performance_eligible"] is False
    assert metadata["projected_equation_residual"] == 7e-7
    assert record["performance_eligible"] is False
    assert record["limitations"] == (
        BENCHMARK.METHOD_INFO["rr_cd"]["performance_limitations"]
    )


def test_threshold_grid_includes_direct_cd_numerical_controls() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "make_threshold_grid.py"), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert "rr_cd" in payload["methods"]
    assert payload["direct_scf_tol"] == [1e-13]
    assert payload["denominator_tolerance"] == [1e-10]
    assert payload["validation_status"]["rr_cd"] == (
        "direct-cd-compressed-rr-reference"
    )
    assert payload["performance_eligibility"]["rr_cd"]["eligible"] is False


def test_runtime_cc_controls_are_assigned_after_construction() -> None:
    solver = SimpleNamespace()
    args = SimpleNamespace(
        cc_conv_tol=1e-9,
        cc_conv_tol_normt=2e-7,
        max_cycle=37,
        max_memory_mb=12345,
    )

    returned = BENCHMARK._configure_solver(solver, args)

    assert returned is solver
    assert solver.conv_tol == 1e-9
    assert solver.conv_tol_normt == 2e-7
    assert solver.max_cycle == 37
    assert solver.max_memory == 12345


def test_final_dense_residual_applies_orbital_denominators() -> None:
    class Solver:
        t1 = np.zeros((1, 2))
        t2 = np.zeros((1, 1, 2, 2))
        level_shift = 0.0

        @staticmethod
        def update_amps(t1, t2, eris):
            return t1 + np.array([[1.0, 2.0]]), t2 + np.ones_like(t2)

    eris = SimpleNamespace(mo_energy=np.array([-1.0, 1.0, 2.0]))

    residual = BENCHMARK._dense_final_residual(Solver(), eris, np)

    eia = np.array([[-2.0, -3.0]])
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    expected_equation = np.sqrt(
        np.vdot(np.array([[1.0, 2.0]]) * eia,
                 np.array([[1.0, 2.0]]) * eia).real
        + np.vdot(eijab, eijab).real
    )
    assert np.isclose(residual["equation_norm"], expected_equation)
    assert np.isclose(residual["jacobi_update_norm"], np.sqrt(9.0))
    assert residual["included_in_post_hf"] is True
    assert residual["is_full_space"] is True

    fno_residual = BENCHMARK._dense_final_residual(
        Solver(), eris, np, orbital_space="fno-active"
    )
    assert fno_residual["is_full_space"] is False
    assert fno_residual["orbital_space"] == "fno-active"
    assert fno_residual["definition"] == "FNO active-space FP64 equation residual"

    diagnostic = BENCHMARK._full_space_residual_diagnostic("fno")
    assert diagnostic["available"] is False
    assert diagnostic["required_for_this_probe"] is False
    assert "no validated lifting" in diagnostic["reason"]
    rr_diagnostic = BENCHMARK._full_space_residual_diagnostic("rr_cd")
    assert rr_diagnostic["available"] is True
    assert rr_diagnostic["execution_status"] == "not-requested"
    assert rr_diagnostic["required_for_final_acceptance"] is True
    assert rr_diagnostic["included_in_normal_timed_path"] is False
    thc_diagnostic = BENCHMARK._full_space_residual_diagnostic("thc_cd")
    assert thc_diagnostic["available"] is True
    assert thc_diagnostic["execution_status"] == "not-requested"
    assert BENCHMARK._full_space_residual_diagnostic("canonical") is None


def test_final_dense_residual_keeps_resident_tensors_on_device_boundary(
    monkeypatch,
) -> None:
    current_t1 = np.zeros((1, 2))
    current_t2 = np.zeros((1, 1, 2, 2))
    jacobi_t1 = np.array([[1.0, 2.0]])
    jacobi_t2 = np.ones((1, 1, 2, 2))
    scalar_operations = []

    class Solver:
        level_shift = 0.0
        t1 = object()
        t2 = object()

        @staticmethod
        def _resident_final_residual_operands(eris):
            assert eris is ERIS
            return {
                "current_t1": current_t1,
                "current_t2": current_t2,
                "jacobi_t1": jacobi_t1,
                "jacobi_t2": jacobi_t2,
                "occupied_energies": np.array([-1.0]),
                "virtual_energies": np.array([1.0, 2.0]),
                "doubles_block_size": 1,
                "temporary_bound_bytes": 96,
                "released_staging_bytes": 4096,
                "update_hbm_checkpoint": {
                    "name": "final-residual-update-live",
                    "driver_used_bytes": 500,
                    "driver_total_bytes": 1000,
                    "cupy_pool_used_bytes": 400,
                    "cupy_pool_total_bytes": 450,
                    "timing_semantics": "cuda-synchronized-instantaneous",
                    "measurement_scope": "exclusive-slurm-device-global",
                    "source": "cudaMemGetInfo-and-cupy-default-memory-pool",
                },
                "update_hbm_checkpoints": [{
                    "name": "final-residual-update-live",
                    "driver_used_bytes": 500,
                    "driver_total_bytes": 1000,
                    "cupy_pool_used_bytes": 400,
                    "cupy_pool_total_bytes": 450,
                    "timing_semantics": "cuda-synchronized-instantaneous",
                    "measurement_scope": "exclusive-slurm-device-global",
                    "source": "cudaMemGetInfo-and-cupy-default-memory-pool",
                }],
            }

        @staticmethod
        def _resident_residual_scalar_to_float(value, *, operation):
            scalar_operations.append(operation)
            return value.item()

        @staticmethod
        def _record_synchronized_hbm_checkpoint(name):
            assert name in {
                "final-residual-block-0-1-live",
                "final-residual-block-1-2-live",
            }
            return {
                "name": name,
                "driver_used_bytes": 400,
                "driver_total_bytes": 1000,
                "cupy_pool_used_bytes": 300,
                "cupy_pool_total_bytes": 350,
                "timing_semantics": "cuda-synchronized-instantaneous",
                "measurement_scope": "exclusive-slurm-device-global",
                "source": "cudaMemGetInfo-and-cupy-default-memory-pool",
            }

        @staticmethod
        def update_amps(*_args):
            raise AssertionError("resident diagnostic used host update path")

    ERIS = SimpleNamespace(mo_energy=np.array([-1.0, 1.0, 2.0]))
    monkeypatch.setattr(
        BENCHMARK,
        "_array",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("resident diagnostic materialized a dense tensor")
        ),
    )

    residual = BENCHMARK._dense_final_residual(Solver(), ERIS, np)

    assert residual["available"] is True
    assert residual["evaluation_backend"] == "gpu-blockwise-fp64"
    assert residual["doubles_block_size"] == 1
    assert residual["temporary_workspace_bound_bytes"] == 96
    assert residual["released_staging_bytes"] == 4096
    assert residual["dense_t2_inputs_present"] is True
    assert residual["full_size_residual_temporaries_materialized"] is False
    assert "dense_t2_materialized" not in residual
    assert residual[
        "final_residual_synchronized_driver_hbm_peak_bytes"
    ] == 500
    assert residual[
        "whole_stage_synchronized_driver_hbm_peak_bytes"
    ] == 500
    assert [
        item["name"] for item in residual["synchronized_hbm_checkpoints"]
    ] == [
        "final-residual-update-live",
        "final-residual-block-0-1-live",
        "final-residual-block-1-2-live",
    ]
    assert scalar_operations == [
        "singles-equation-norm",
        "doubles-equation-norm",
        "singles-jacobi-update-norm",
        "doubles-jacobi-update-norm",
        "full-equation-norm",
        "full-jacobi-update-norm",
    ]


def test_timed_resident_scope_routes_only_canonical():
    calls = []

    class Solver:
        @staticmethod
        def _resident_post_hf_state_scope():
            calls.append("canonical")
            return "scope"

    solver = Solver()

    assert BENCHMARK._timed_resident_diagnostic_scope(
        "canonical", solver
    ) == "scope"
    for method in ("canonical_legacy", "fno", "rr_cd", "thc_cd"):
        with BENCHMARK._timed_resident_diagnostic_scope(
            method, solver
        ) as active:
            assert active is False
    assert calls == ["canonical"]


def test_rr_thc_residual_serialization_requires_labeled_schema() -> None:
    payload = BENCHMARK._residual_module.labeled_low_rank_residuals(
        projected_equation=7.0e-7,
        projected_jacobi_update=2.0e-5,
    )
    solver = SimpleNamespace(residual_metadata=lambda: payload)

    residual = BENCHMARK._labeled_low_rank_residual(solver)

    assert residual["schema"] == BENCHMARK.LOW_RANK_RESIDUAL_SCHEMA
    assert residual["projected_equation"] == 7.0e-7
    assert "norm" not in residual
    projected = residual["measurements"]["projected_equation"]
    jacobi = residual["measurements"]["projected_jacobi_update"]
    full = residual["measurements"]["full_space_equation"]
    assert projected == {
        "available": True,
        "norm": 7.0e-7,
        "kind": "equation-residual",
        "space": "rr-projected-active-pair",
        "is_full_space": False,
    }
    assert jacobi["kind"] == "jacobi-update"
    assert jacobi["is_full_space"] is False
    assert full["available"] is False
    assert full["is_full_space"] is True


def test_rr_thc_residual_serialization_fails_closed_on_bad_semantics() -> None:
    malformed = BENCHMARK._residual_module.labeled_low_rank_residuals(
        projected_equation=7.0e-7,
    )
    malformed["measurements"]["projected_equation"]["kind"] = (
        "jacobi-update"
    )

    with np.testing.assert_raises_regex(ValueError, "invalid semantics"):
        BENCHMARK._labeled_low_rank_residual(
            SimpleNamespace(residual_metadata=lambda: malformed)
        )
    with np.testing.assert_raises_regex(TypeError, "residual_metadata"):
        BENCHMARK._labeled_low_rank_residual(SimpleNamespace())


def test_opt_in_full_space_diagnostic_has_its_own_wall_time_and_metadata() -> None:
    class Diagnostic:
        @staticmethod
        def metadata():
            return {"residual_norm": 4e-8, "pair_dimension": 17}

    calls = []
    solver = SimpleNamespace(
        converged=True,
        run_full_space_residual_diagnostic=lambda: (
            calls.append("diagnostic") or Diagnostic()
        ),
        full_space_diagnostic_metadata={
            "resource_accounting": {"pair_matrix_nbytes": 2312},
            "transfer_accounting": {"delta": {"total_bytes": 24}},
        },
    )
    ticks = iter((100.0, 102.75))

    payload = BENCHMARK._run_post_hf_full_space_diagnostic(
        "rr_cd", solver, None, clock=lambda: next(ticks)
    )

    assert calls == ["diagnostic"]
    assert payload["wall_seconds"] == 2.75
    assert payload["included_in_post_hf"] is False
    assert payload["included_in_normal_timed_path"] is False
    assert payload["execution_status"] == "completed"
    assert payload["kind"] == "equation-residual"
    assert payload["space"] == "full-active-pair"
    assert payload["is_full_space"] is True
    assert payload["residual_norm"] == 4e-8
    assert payload["resource_accounting"]["pair_matrix_nbytes"] == 2312
    assert payload["transfer_accounting"]["delta"]["total_bytes"] == 24


def test_full_space_diagnostic_flag_fails_for_unsupported_method() -> None:
    args = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "canonical",
        "--run-full-space-diagnostic",
    ])

    with np.testing.assert_raises_regex(
        ValueError, "supported only for rr_cd and thc_cd"
    ):
        BENCHMARK._validate_run_configuration(args)


def test_requested_diagnostic_is_invoked_after_post_hf_timer_closes() -> None:
    source = inspect.getsource(BENCHMARK.run)

    timer_position = source.index(
        "post_hf_seconds = time.perf_counter() - post_start"
    )
    diagnostic_position = source.index(
        "_run_post_hf_full_space_diagnostic(", timer_position
    )
    amplitude_position = source.index(
        "amplitude_payload = _amplitude_payload(", diagnostic_position
    )

    assert timer_position < diagnostic_position < amplitude_position


def test_pcie_query_uses_mtu_supported_identity_fields(monkeypatch) -> None:
    def command(arguments):
        query = next(item for item in arguments if item.startswith("--query-gpu="))
        assert "pci.link" not in query
        return (
            "NVIDIA A100-SXM4-80GB, GPU-1, 81920, 580.159.03, "
            "00000000:ff:00.0"
        )

    monkeypatch.setattr(BENCHMARK, "_command_text", command)
    payload = BENCHMARK._pcie_info()

    assert payload["available"] is True
    gpu = payload["visible_gpus"][0]
    assert gpu["name"] == "NVIDIA A100-SXM4-80GB"
    assert gpu["sysfs_bus_id"] == "0000:ff:00.0"
    assert gpu["max_link_width"] is None
    assert gpu["pcie_gen_max"] is None


def test_pcie_generation_maps_linux_link_rates() -> None:
    assert BENCHMARK._pcie_generation("16.0 GT/s PCIe") == "4"
    assert BENCHMARK._pcie_generation("32 GT/s") == "5"
    assert BENCHMARK._pcie_generation(None) is None
    assert BENCHMARK._pcie_generation("unknown") is None


def test_git_state_falls_back_to_content_manifest(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "gpu4pyscf").mkdir()
    (tmp_path / "gpu4pyscf" / "kernel.py").write_text("answer = 42\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ignored.json").write_text("{}\n")

    def no_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(BENCHMARK, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(BENCHMARK.subprocess, "check_output", no_git)
    state = BENCHMARK._git_state()

    assert state["source_kind"] == "content-manifest"
    assert state["revision"].startswith("tree-sha256:")
    assert state["tree_sha256"] == state["revision"].split(":", 1)[1]
    assert state["files_hashed"] == 1
    assert state["base_revision"] == BENCHMARK.FROZEN_BASE_COMMIT
    assert state["snapshot"]["valid_at_start"] is False
    assert state["formal_performance_eligible"] is False


def test_source_mutation_keeps_record_but_invalidates_performance(monkeypatch) -> None:
    record = {"source": {"tree_sha256": "start"}, "status": "completed"}
    monkeypatch.setattr(
        BENCHMARK, "_source_tree_digest", lambda root: ("end", 19)
    )

    stable = BENCHMARK._record_source_stability(record)

    assert stable is False
    assert record["status"] == "completed"
    assert record["performance_eligible"] is False
    assert record["source"]["tree_sha256_at_start"] == "start"
    assert record["source"]["tree_sha256_at_end"] == "end"
    assert record["source"]["files_hashed_at_end"] == 19
    assert record["source"]["stable_during_run"] is False
    assert record["limitations"] == [
        "source snapshot or manifest changed during the timed run"
    ]


def test_valid_v2_snapshot_qualifies_without_inventing_top_level_eligibility(
    monkeypatch, tmp_path: Path
) -> None:
    task, source, _ = _installed_snapshot(tmp_path)
    monkeypatch.setattr(BENCHMARK, "REPOSITORY_ROOT", source)
    monkeypatch.setenv("CCSD_TASK_ROOT", str(task))
    monkeypatch.setenv("CCSD_EXPECTED_DEPLOYMENT_PROFILE", "candidate")
    try:
        state = BENCHMARK._git_state()
        record = {"source": state}

        assert BENCHMARK._record_source_stability(record) is True
        assert state["stable_during_run"] is True
        assert state["formal_performance_eligible"] is True
        assert "performance_eligible" not in record
    finally:
        _make_snapshot_writable(source)


def test_runtime_binary_mutation_invalidates_formal_performance(
    monkeypatch, tmp_path: Path
) -> None:
    task, source, binary = _installed_snapshot(tmp_path)
    monkeypatch.setattr(BENCHMARK, "REPOSITORY_ROOT", source)
    monkeypatch.setenv("CCSD_TASK_ROOT", str(task))
    monkeypatch.setenv("CCSD_EXPECTED_DEPLOYMENT_PROFILE", "candidate")
    try:
        state = BENCHMARK._git_state()
        os.chmod(binary, 0o644)
        binary.write_bytes(b"modified-runtime-binary\n")
        os.chmod(binary, 0o444)
        record = {"source": state, "performance_eligible": True}

        assert BENCHMARK._record_source_stability(record) is False
        assert state["snapshot"]["valid_at_end"] is False
        assert state["snapshot"]["runtime_binaries_valid_at_end"] is False
        assert state["snapshot"]["stable_during_run"] is False
        assert state["formal_performance_eligible"] is False
        assert record["performance_eligible"] is False
    finally:
        _make_snapshot_writable(source)
