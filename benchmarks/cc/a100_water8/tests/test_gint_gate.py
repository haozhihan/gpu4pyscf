"""CPU-only tests for the fail-closed water2 GINT/direct-CD gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("water2_gint_gate", ROOT / "gint_gate.py")
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)

BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "water8_benchmark_digest", ROOT / "benchmark.py"
)
assert BENCHMARK_SPEC is not None and BENCHMARK_SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_SPEC.name] = BENCHMARK
BENCHMARK_SPEC.loader.exec_module(BENCHMARK)
SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "water8_snapshot_manifest_for_gint", ROOT / "snapshot_manifest.py"
)
assert SNAPSHOT_SPEC is not None and SNAPSHOT_SPEC.loader is not None
SNAPSHOT = importlib.util.module_from_spec(SNAPSHOT_SPEC)
SNAPSHOT_SPEC.loader.exec_module(SNAPSHOT)


SOURCE_DIGEST = "a" * 64
MANIFEST_DIGEST = "b" * 64


def _passing_source_evidence() -> dict:
    source = f"/task/snapshots/{SOURCE_DIGEST}/source"
    provenance = _local_provenance()
    return {
        "root": source,
        "tree_sha256_at_start": SOURCE_DIGEST,
        "tree_sha256_at_end": SOURCE_DIGEST,
        "stable_during_run": True,
        "base_revision": GATE.FROZEN_BASE_COMMIT,
        "snapshot": {
            "schema": "gpu4pyscf.snapshot-evidence.v1",
            "valid": True,
            "valid_at_start": True,
            "valid_at_end": True,
            "source": source,
            "snapshot_root": f"/task/snapshots/{SOURCE_DIGEST}",
            "manifest_path": (
                f"/task/snapshots/{SOURCE_DIGEST}/manifest.json"
            ),
            "tree_sha256": SOURCE_DIGEST,
            "tree_sha256_at_end": SOURCE_DIGEST,
            "read_only": True,
            "read_only_at_start": True,
            "read_only_at_end": True,
            "writable_paths": [],
            "reasons": [],
            "reasons_at_end": [],
            "stable_during_run": True,
            "manifest_sha256_at_start": MANIFEST_DIGEST,
            "manifest_sha256_at_end": MANIFEST_DIGEST,
            "expected_deployment_profile": "candidate",
            "canonical_pristine_required": False,
            "manifest": {
                "schema": "gpu4pyscf.source-snapshot.v2",
                "tree_sha256": SOURCE_DIGEST,
                "base_revision": GATE.FROZEN_BASE_COMMIT,
                "source": source,
                "immutable": True,
                "local_provenance": provenance,
                "runtime_binaries": {
                    "complete": True,
                    "source_root": "/pinned/gpu4pyscf/lib",
                    "required_library_names": list(
                        SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
                    ),
                    "inventory_library_names": list(
                        SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
                    ),
                    "closure_complete": True,
                    "files": [{
                        "relative_path": f"gpu4pyscf/lib/{name}",
                        "sha256": "c" * 64,
                        "bytes": 123,
                    } for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES],
                },
            },
        },
    }


def _local_provenance() -> dict:
    allowed = {
        "prefixes": ["benchmarks/cc/a100_water8/"],
        "files": ["gpu4pyscf/cc/device_runtime.py"],
    }
    entries: list[str] = []
    policy = json.dumps(
        {"allowed_untracked": allowed, "status": entries},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "gpu4pyscf.local-provenance.v1",
        "deployment_profile": "candidate",
        "repository_root": "/local/repository",
        "head_revision": GATE.FROZEN_BASE_COMMIT,
        "frozen_base_revision": GATE.FROZEN_BASE_COMMIT,
        "head_matches_frozen_base": True,
        "tracked_pristine": True,
        "untracked_paths_allowed": True,
        "canonical_pristine": False,
        "allowed_untracked": allowed,
        "git_status": {
            "format": "porcelain-v1-lines",
            "sha256": hashlib.sha256(b"").hexdigest(),
            "allowed_paths_status_sha256": hashlib.sha256(policy).hexdigest(),
            "entry_count": 0,
            "entries": entries,
        },
    }


def _passing_record():
    dimension = 116 * 117 // 2
    h2d_operations = {
        "selected_device_schedule": {"bytes": 4096, "count": 8},
        "selected_column_request": {"bytes": 1024, "count": 4},
        "selected_gint_constant_cache": {"bytes": 224, "count": 2},
    }
    d2h_operations = {
        "selected_schedule_coeff_support": {"bytes": 1024, "count": 1},
        "selected_schedule_log_q": {"bytes": 512, "count": 3},
    }
    h2d_bytes = sum(item["bytes"] for item in h2d_operations.values())
    d2h_bytes = sum(item["bytes"] for item in d2h_operations.values())
    transfer_total = h2d_bytes + d2h_bytes
    return {
        "status": "completed",
        "source": _passing_source_evidence(),
        "topology": {
            "passed": True,
            "stable_during_run": True,
            "reasons": [],
        },
        "hardware": {"passed": True, "reasons": []},
        "case": {
            "id": "water2-tz",
            "basis": "cc-pVTZ",
            "actual_nao": 116,
            "geometry_sha256": GATE.EXPECTED_WATER2_GEOMETRY_SHA256,
        },
        "thresholds": {
            "eri_tol": 1e-8,
            "selected_column_tolerance": 2e-10,
            "diagonal_tolerance": 2e-10,
        },
        "provider": {
            "factorization": GATE.SELECTED_PROVIDER_FACTORIZATION,
            "backend": GATE.SELECTED_PROVIDER_BACKEND,
            "c_abi": {
                "columns": GATE.SELECTED_COLUMNS_SYMBOL,
                "diagonal": GATE.SELECTED_DIAGONAL_SYMBOL,
            },
            "dimension": dimension,
            "nao": 116,
            "precision": "fp64",
            "output_residency": "gpu",
            "output_layout": GATE.SELECTED_OUTPUT_LAYOUT,
            "max_batch_size": 32,
            "maximum_batch_observed": 32,
            "diagonal_calls": 1,
            "cabi_calls": 2,
            "materializes_pair_matrix": False,
            "materializes_four_index_tensor": False,
            "materializes_per_pivot_ao_matrix": False,
            "diag_block_with_triu": True,
            "device_schedule": True,
            "release_scope": GATE.SELECTED_RELEASE_SCOPE,
            "schedule": {
                "single_shell_support": True,
                "npair": dimension,
                "nao_original": 116,
            },
            "kernel_launches": 2,
            "performance_eligible": True,
            "performance_gate": "water2-selected-provider-gate-complete",
            "explicit_transfers": {
                "schema": "gpu4pyscf.gint-selected-transfer-ledger.v1",
                "directions": {
                    "h2d": {
                        "bytes": h2d_bytes,
                        "count": sum(
                            item["count"] for item in h2d_operations.values()
                        ),
                        "operations": h2d_operations,
                    },
                    "d2h": {
                        "bytes": d2h_bytes,
                        "count": sum(
                            item["count"] for item in d2h_operations.values()
                        ),
                        "operations": d2h_operations,
                    },
                },
                "total_bytes": transfer_total,
                "output_residency": "gpu",
                "known_provider_arrays_byte_counted": True,
                "complete_for_vhfopt_internal_setup": True,
                "unresolved_operations": [],
            },
        },
        "numerical_validation": {
            "materialized_nao4": False,
            "materialized_pair_matrix": False,
            "selected_column_max_abs_error": 1e-12,
            "diagonal_max_abs_error": 2e-12,
            "ao2mo_oracle": {
                "backend": "pyscf.ao2mo.general",
                "precision": "fp64",
                "batch_size": 32,
                "within_working_limit": True,
                "materialized_nao4": False,
                "materialized_pair_matrix": False,
            },
            "selected_column_batches": [{
                "batch_size": batch_size,
                "shape": [batch_size, dimension],
                "dtype": "float64",
                "gpu_resident": True,
                "output_nbytes": batch_size * dimension * 8,
                "max_abs_error": 1e-12,
            } for batch_size in GATE.SELECTED_COLUMN_BATCH_SIZES],
            "diagonal": {
                "shape": [dimension],
                "dtype": "float64",
                "gpu_resident": True,
                "source": GATE.SELECTED_DIAGONAL_SYMBOL,
            },
        },
        "allocation_validation": {
            "max_returned_batch_bytes": 256 * 1024**2,
            "returned_batches": [{
                "batch_size": batch_size,
                "shape": [batch_size, dimension],
                "nbytes": batch_size * dimension * 8,
                "live_pool_delta_bytes": batch_size * dimension * 8,
                "within_limit": True,
            } for batch_size in GATE.SELECTED_COLUMN_BATCH_SIZES],
            "packed_diagonal": {
                "shape": [dimension],
                "nbytes": dimension * 8,
                "live_pool_delta_bytes": dimension * 8,
                "within_limit": True,
            },
            "pair_matrix_allocated": False,
            "four_index_ao_tensor_allocated": False,
            "per_pivot_ao_matrix_allocated": False,
        },
        "timed_transfer_counter": {
            "total_bytes": transfer_total,
            "by_operation": {
                "h2d": h2d_operations,
                "d2h": d2h_operations,
            },
        },
        "direct_cd": {
            "rank": 321,
            "capped": False,
            "residual_diagonal": 8e-9,
            "provider_materializes_pair_matrix": False,
            "column_batch_size": 32,
            "requested_column_batch_size": 32,
            "provider_batch_calls": 11,
        },
        "timing": {
            "direct_cd_end_to_end_seconds": 12.3,
            "boundary": "provider + diagonal + all direct-CD iterations + sync",
        },
        "resources": {
            "hbm": {"peak_process_gib": 4.0},
            "direct_cd_peak_host_rss_gib": 1.0,
        },
    }


def test_gate_passes_only_complete_explicit_evidence_and_has_no_speed_target():
    decision = GATE.evaluate_gate(_passing_record())
    assert decision["passed"] is True
    assert decision["correctness_passed"] is True
    assert decision["performance_eligible"] is True
    assert decision["reasons"] == []
    assert decision["speed_threshold_seconds"] is None
    timing = next(
        item for item in decision["checks"]
        if item["id"] == "complete_timing_boundary"
    )
    assert timing["evidence"]["speed_threshold_seconds"] is None


def test_gate_keeps_correctness_separate_from_pending_performance_audit():
    record = _passing_record()
    provider = record["provider"]
    provider["performance_eligible"] = False
    transfers = provider["explicit_transfers"]
    transfers["complete_for_vhfopt_internal_setup"] = False
    transfers["unresolved_operations"] = [{
        "operation": "vhfopt-internal-basis-cache-and-coeff-build",
        "bytes": None,
    }]

    decision = GATE.evaluate_gate(record)

    assert decision["passed"] is True
    assert decision["correctness_passed"] is True
    assert decision["reasons"] == []
    assert decision["performance_eligible"] is False
    assert len(decision["performance_reasons"]) == 1
    release = next(
        item for item in decision["checks"]
        if item["id"] == "performance_transfer_release"
    )
    assert release["scope"] == "performance"
    assert release["passed"] is False


def test_gate_requires_all_three_batch_sizes_and_bounded_allocations():
    record = _passing_record()
    record["numerical_validation"]["selected_column_batches"].pop()
    allocation = record["allocation_validation"]["returned_batches"][-1]
    allocation["within_limit"] = False

    decision = GATE.evaluate_gate(record)

    failed = {item["id"] for item in decision["checks"] if not item["passed"]}
    assert "selected_batch_oracle" in failed
    assert "bounded_allocations" in failed
    assert decision["correctness_passed"] is False
    assert decision["performance_eligible"] is False


def test_bounded_ao2mo_oracle_extracts_requested_packed_columns():
    nao = 3
    eri = np.arange(nao**4, dtype=np.float64).reshape((nao,) * 4)
    pair_map = SimpleNamespace(
        dimension=6,
        pair_i=np.asarray([0, 1, 1, 2, 2, 2]),
        pair_j=np.asarray([0, 0, 1, 0, 1, 2]),
    )
    molecule = SimpleNamespace(nao_nr=lambda: nao)

    class FakeAO2MO:
        @staticmethod
        def general(_mol, coefficients, compact):
            assert compact is False
            first, second, third, fourth = coefficients
            transformed = np.einsum(
                "pqrs,pi,qj,rk,sl->ijkl",
                eri,
                first,
                second,
                third,
                fourth,
                optimize=True,
            )
            return transformed.reshape(
                first.shape[1] * second.shape[1],
                third.shape[1] * fourth.shape[1],
            )

    pivots = (0, 4)
    columns, metadata = GATE.ao2mo_selected_columns(
        molecule,
        pair_map,
        pivots,
        max_working_bytes=nao * nao * len(pivots) ** 2 * 8,
        ao2mo_module=FakeAO2MO,
    )
    expected = np.stack([
        eri[
            pair_map.pair_i,
            pair_map.pair_j,
            pair_map.pair_i[pivot],
            pair_map.pair_j[pivot],
        ]
        for pivot in pivots
    ])
    np.testing.assert_array_equal(columns, expected)
    assert metadata["transformed_shape"] == [9, 4]
    assert metadata["materialized_nao4"] is False
    assert metadata["materialized_pair_matrix"] is False

    with pytest.raises(MemoryError, match="exceeding max_working_bytes"):
        GATE.ao2mo_selected_columns(
            molecule,
            pair_map,
            pivots,
            max_working_bytes=metadata["transformed_bytes"] - 1,
            ao2mo_module=FakeAO2MO,
        )


def test_validation_pivots_are_distinct_and_span_the_pair_range():
    pivots = GATE.selected_validation_pivots(116 * 117 // 2)
    assert len(pivots) == 32
    assert len(set(pivots)) == 32
    assert pivots[0] == 0
    assert pivots[-1] == 116 * 117 // 2 - 1
    for batch_size in GATE.SELECTED_COLUMN_BATCH_SIZES:
        batch = GATE.pivots_for_batch(pivots, batch_size)
        assert len(batch) == batch_size
        assert len(set(batch)) == batch_size


def test_gate_fails_closed_for_capped_cd_and_missing_known_transfer():
    record = _passing_record()
    record["direct_cd"]["capped"] = True
    transfers = record["provider"]["explicit_transfers"]
    transfers["known_provider_arrays_byte_counted"] = False
    decision = GATE.evaluate_gate(record)
    failed = {item["id"] for item in decision["checks"] if not item["passed"]}
    assert decision["performance_eligible"] is False
    assert failed == {"direct_cd_converged", "selected_transfer_accounting"}


def test_gate_requires_v2_runtime_binary_inventory():
    record = _passing_record()
    record["source"]["snapshot"]["manifest"]["runtime_binaries"][
        "files"
    ] = []

    decision = GATE.evaluate_gate(record)

    failed = {item["id"] for item in decision["checks"] if not item["passed"]}
    assert "immutable_source" in failed
    assert decision["performance_eligible"] is False


def test_topology_evidence_binds_record_to_current_job_and_driver(tmp_path: Path):
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(json.dumps({
        "performance_eligible": True,
        "benchmark_track": "performance",
        "gpu_numa_node": 3,
        "selected_cpulist": "24-31",
        "observed_cpulist_after_bind": "24-31",
        "kernel_memory_policy": {
            "verified": True,
            "after": {"mode_name": "preferred", "nodes": [3]},
        },
        "slurm": {"SLURM_JOB_ID": "70001"},
        "command": ["python", "/snapshot/benchmarks/cc/a100_water8/gint_gate.py"],
    }))
    environment = {
        "SLURM_JOB_ID": "70001",
        "CCSD_PERFORMANCE_ELIGIBLE": "true",
    }
    evidence = GATE.topology_evidence(topology_path, environment=environment)
    assert evidence["passed"] is True
    assert evidence["sha256"] == GATE._sha256(topology_path)
    assert GATE.finish_topology_evidence(evidence) is True
    assert evidence["stable_during_run"] is True

    environment["SLURM_JOB_ID"] = "70002"
    mismatch = GATE.topology_evidence(topology_path, environment=environment)
    assert mismatch["passed"] is False
    assert "different Slurm job" in " ".join(mismatch["reasons"])


def test_source_digest_is_stable_and_changes_with_deployable_source(tmp_path: Path):
    (tmp_path / "module.py").write_text("answer = 1\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ignored.json").write_text("{}\n")
    first, count = GATE.source_tree_digest(tmp_path)
    assert count == 1
    assert (first, count) == BENCHMARK._source_tree_digest(tmp_path)
    (tmp_path / "results" / "ignored.json").write_text('{"x": 1}\n')
    assert GATE.source_tree_digest(tmp_path)[0] == first
    (tmp_path / "module.py").write_text("answer = 2\n")
    assert GATE.source_tree_digest(tmp_path)[0] != first


def test_source_digest_rejects_an_empty_deployable_tree(tmp_path: Path):
    (tmp_path / "ignored.bin").write_bytes(b"not deployable source")
    try:
        GATE.source_tree_digest(tmp_path)
    except RuntimeError as exc:
        assert "no source files" in str(exc)
    else:
        raise AssertionError("an empty source manifest must fail closed")


def test_snapshot_evidence_binds_hash_manifest_and_read_only_tree(
    monkeypatch, tmp_path: Path,
):
    task = tmp_path / "task"
    staging = task / "snapshots" / "staging" / "source"
    staging.mkdir(parents=True)
    (staging / "module.py").write_text("answer = 42\n")
    binary_root = staging / "gpu4pyscf" / "lib"
    binary_root.mkdir(parents=True)
    (staging / "gpu4pyscf" / "runtime_targets.py").write_text(
        "from gpu4pyscf.lib.utils import load_library\n"
        + "".join(
            f"load_library({name.removesuffix('.so')!r})\n"
            for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
        )
    )
    for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES:
        (binary_root / name).write_bytes(
            f"test-runtime-binary:{name}\n".encode()
        )
    digest, _ = GATE.source_tree_digest(staging)
    snapshot_root = task / "snapshots" / digest
    source = snapshot_root / "source"
    staging.parent.rename(snapshot_root)
    binaries = [
        source / "gpu4pyscf" / "lib" / name
        for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
    ]
    manifest = snapshot_root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "gpu4pyscf.source-snapshot.v2",
        "tree_sha256": digest,
        "base_revision": GATE.FROZEN_BASE_COMMIT,
        "source": str(source.resolve()),
        "immutable": True,
        "local_provenance": _local_provenance(),
        "runtime_binaries": {
            "source_root": "/pinned/gpu4pyscf/lib",
            "complete": True,
            "required_library_names": list(SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES),
            "inventory_library_names": list(SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES),
            "closure_complete": True,
            "files": [{
                "relative_path": f"gpu4pyscf/lib/{item.name}",
                "sha256": GATE._sha256(item),
                "bytes": item.stat().st_size,
            } for item in binaries],
        },
    }))
    monkeypatch.setenv("CCSD_TASK_ROOT", str(task))
    monkeypatch.setenv("CCSD_EXPECTED_DEPLOYMENT_PROFILE", "candidate")
    (source / "module.py").chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    (source / "gpu4pyscf" / "runtime_targets.py").chmod(
        stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    )
    for binary in binaries:
        binary.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    binaries[0].parent.chmod(0o555)
    binaries[0].parent.parent.chmod(0o555)
    source.chmod(
        stat.S_IRUSR | stat.S_IXUSR
        | stat.S_IRGRP | stat.S_IXGRP
        | stat.S_IROTH | stat.S_IXOTH
    )
    manifest.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    snapshot_root.chmod(
        stat.S_IRUSR | stat.S_IXUSR
        | stat.S_IRGRP | stat.S_IXGRP
        | stat.S_IROTH | stat.S_IXOTH
    )

    try:
        evidence = GATE.source_evidence(source)
        assert evidence["snapshot"]["valid"] is True
        assert evidence["snapshot"]["valid_at_start"] is True
        assert evidence["snapshot"]["manifest"]["runtime_binaries"][
            "complete"
        ] is True
        assert evidence["tree_sha256_at_start"] == digest
        assert GATE.finish_source_evidence(evidence, source) is True
        assert evidence["snapshot"]["valid_at_end"] is True
        assert evidence["snapshot"]["stable_during_run"] is True
    finally:
        snapshot_root.chmod(0o755)
        source.chmod(0o755)
        manifest.chmod(0o644)
        binaries[0].parent.parent.chmod(0o755)
        binaries[0].parent.chmod(0o755)
        for binary in binaries:
            binary.chmod(0o644)
        (source / "gpu4pyscf" / "runtime_targets.py").chmod(0o644)
        (source / "module.py").chmod(0o644)


def test_mtu_launcher_is_snapshot_aware_and_physical8_guarded():
    launcher = ROOT / "run_mtu_gint_gate.sbatch"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text(encoding="utf-8")
    assert "CCSD_SOURCE_ROOT:?" in text
    assert 'PYTHON_BIN="${PYTHON_BIN:-' in text
    assert '"${TASK_ROOT}"/snapshots/*/source' in text
    assert "GINT gate results must be outside" in text
    assert "--mode physical8" in text
    assert "--require-performance" in text
    assert '"${SCRIPT_DIR}/topology_guard.py"' in text
    assert '"${SCRIPT_DIR}/gint_gate.py"' in text


def test_dry_run_records_thresholds_and_does_not_write_output(tmp_path: Path):
    args = SimpleNamespace(
        output=tmp_path / "gate.json",
        topology_record=None,
        eri_tol=1e-8,
        direct_scf_tol=1e-14,
        selected_column_tolerance=2e-10,
        diagonal_tolerance=2e-10,
        group_size=16,
        max_rank=None,
        max_block_bytes=256 * 1024**2,
        oracle_max_bytes=256 * 1024**2,
        dry_run=True,
    )
    output, record = GATE.run(args)
    assert output == args.output.resolve()
    assert record["dry_run"] is True
    assert record["thresholds"]["eri_tol"] == 1e-8
    assert record["configuration"]["selected_column_batch_sizes"] == [1, 8, 32]
    assert record["speed_threshold_seconds"] is None
    assert not output.exists()
