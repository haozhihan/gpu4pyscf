"""CPU-only tests for the fail-closed water2 GINT/direct-CD gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import gpu4pyscf.cc.gint_transfer_audit as GINT_AUDIT
from gpu4pyscf.cc.gint_transfer_audit import (
    validate_runtime_performance_gate,
)


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
NORMALIZER_SPEC = importlib.util.spec_from_file_location(
    "water8_snapshot_normalizer_for_gint",
    ROOT / "snapshot_symlink_normalize.py",
)
assert NORMALIZER_SPEC is not None and NORMALIZER_SPEC.loader is not None
NORMALIZER = importlib.util.module_from_spec(NORMALIZER_SPEC)
NORMALIZER_SPEC.loader.exec_module(NORMALIZER)

RECEIPT_SPEC = importlib.util.spec_from_file_location(
    "water8_gint_gate_receipt", ROOT / "gint_gate_receipt.py"
)
assert RECEIPT_SPEC is not None and RECEIPT_SPEC.loader is not None
RECEIPT = importlib.util.module_from_spec(RECEIPT_SPEC)
sys.modules[RECEIPT_SPEC.name] = RECEIPT
RECEIPT_SPEC.loader.exec_module(RECEIPT)
PROMOTION_SPEC = importlib.util.spec_from_file_location(
    "water8_gint_release_snapshot", ROOT / "gint_release_snapshot.py"
)
assert PROMOTION_SPEC is not None and PROMOTION_SPEC.loader is not None
PROMOTION = importlib.util.module_from_spec(PROMOTION_SPEC)
sys.modules[PROMOTION_SPEC.name] = PROMOTION
PROMOTION_SPEC.loader.exec_module(PROMOTION)


SOURCE_DIGEST = "a" * 64
MANIFEST_DIGEST = "b" * 64
_TEST_RELEASE_RUNTIME: dict = {}
_REAL_RELEASE_RUNTIME_OBSERVER = GINT_AUDIT._observe_current_release_runtime


def _source_normalization(links: list[dict] | None = None) -> dict:
    links = [] if links is None else links
    inventory = {
        "schema": GINT_AUDIT.SOURCE_NORMALIZATION_SCHEMA,
        "policy": GINT_AUDIT.SOURCE_NORMALIZATION_POLICY,
        "links": links,
        "link_count": len(links),
        "regular_file_count_before": 3,
        "directory_count": 4,
    }
    encoded = (
        json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    value = {
        "schema": GINT_AUDIT.SOURCE_NORMALIZATION_SCHEMA,
        "policy": GINT_AUDIT.SOURCE_NORMALIZATION_POLICY,
        "source_link_inventory": inventory,
        "source_link_inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        "materialized_link_count": len(links),
        "materialized_links": links,
        "published_symlink_count": 0,
        "published_special_file_count": 0,
        "regular_file_count_after": 3 + len(links),
        "directory_count_after": 4,
    }
    uploaded = (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    value.update({
        "remote_verified_after_runtime_binding": True,
        "remote_regular_file_count_after_runtime_binding": 16 + len(links),
        "remote_directory_count_after_runtime_binding": 4,
        "uploaded_record_sha256": hashlib.sha256(uploaded).hexdigest(),
        "uploaded_record_bytes": len(uploaded),
    })
    return value


@pytest.fixture(autouse=True)
def _independent_release_runtime_observer(monkeypatch):
    """Replace only the OS/CUDA probe; public gate arguments cannot replace it."""

    monkeypatch.setattr(
        GINT_AUDIT,
        "_observe_current_release_runtime",
        lambda: json.loads(json.dumps(_TEST_RELEASE_RUNTIME)),
    )


def test_release_runtime_observer_reads_process_linux_slurm_and_cuda(
    monkeypatch, tmp_path: Path,
):
    pci_root = tmp_path / "pci"
    device_root = pci_root / "0000:01:00.0"
    device_root.mkdir(parents=True)
    (device_root / "numa_node").write_text("3\n")
    node_root = tmp_path / "nodes"
    (node_root / "node3").mkdir(parents=True)
    (node_root / "node3" / "cpulist").write_text("24-31,88-95\n")
    environment = {
        "SLURM_JOB_ID": "70002",
        "SLURM_JOB_NAME": "gint-release-" + "a" * 64,
        "SLURM_JOB_NODELIST": "compute-1-6",
        "SLURM_JOB_PARTITION": "mrigpu",
        "SLURM_CPUS_PER_TASK": "64",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for name in GINT_AUDIT._RELEASE_ENVIRONMENT_NAMES:
        if name in environment:
            monkeypatch.setenv(name, environment[name])
        else:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(GINT_AUDIT, "_PCI_SYSFS_DEVICES", pci_root)
    monkeypatch.setattr(GINT_AUDIT, "_NODE_SYSFS_DEVICES", node_root)
    monkeypatch.setattr(GINT_AUDIT.os, "getpid", lambda: 42002)
    monkeypatch.setattr(
        GINT_AUDIT.os,
        "sched_getaffinity",
        lambda _pid: set(range(24, 32)),
        raising=False,
    )
    monkeypatch.setattr(
        GINT_AUDIT.socket, "gethostname", lambda: "compute-1-6"
    )
    monkeypatch.setattr(
        GINT_AUDIT, "_proc_status_value", lambda _field: "3"
    )
    monkeypatch.setattr(
        GINT_AUDIT,
        "_current_kernel_memory_policy",
        lambda: {"mode": 1, "mode_name": "preferred", "nodes": [3]},
    )
    monkeypatch.setattr(
        GINT_AUDIT,
        "_query_slurm_controller",
        lambda _job: {
            "JobId": "70002",
            "JobName": environment["SLURM_JOB_NAME"],
            "Partition": "mrigpu",
            "NodeList": "compute-1-6",
            "ReqNodeList": "compute-1-6",
            "JobState": "RUNNING",
            "NumCPUs": "64",
            "NumTasks": "1",
            "CPUs/Task": "64",
            "Command": "/snapshot/run_mtu_gint_gate.sbatch",
        },
    )
    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                getDeviceCount=lambda: 1,
                getDevice=lambda: 0,
            ),
            Device=lambda _device: SimpleNamespace(
                pci_bus_id=b"00000000:01:00.0"
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    observed = _REAL_RELEASE_RUNTIME_OBSERVER()

    assert observed["errors"] == []
    assert observed["pid"] == 42002
    assert observed["host"] == "compute-1-6"
    assert observed["affinity"] == "24-31"
    assert observed["mems_allowed_list"] == "3"
    assert observed["gpu_node_cpulist"] == "24-31,88-95"
    assert observed["kernel_memory_policy"]["mode_name"] == "preferred"
    assert observed["slurm_controller"]["JobState"] == "RUNNING"
    assert observed["cuda"] == {
        "visible_devices_environment": "0",
        "device_count": 1,
        "current_device": 0,
        "pci_bus_id": "0000:01:00.0",
        "numa_node": 3,
    }


def test_slurm_controller_probe_ignores_path_and_parses_live_job(
    monkeypatch, tmp_path: Path,
):
    assert GINT_AUDIT._SCONTROL_PATH == Path("/usr/bin/scontrol")
    attacker = tmp_path / "scontrol"
    attacker.write_text("#!/bin/sh\nexit 0\n")
    attacker.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    # /usr/bin/env provides a root-owned executable for this CPU-only test;
    # subprocess itself is mocked, so no command is executed.
    trusted_test_binary = Path("/usr/bin/env")
    monkeypatch.setattr(GINT_AUDIT, "_SCONTROL_PATH", trusted_test_binary)
    captured: list[list[str]] = []

    def fake_check_output(arguments, **_kwargs):
        captured.append(list(arguments))
        return (
            "JobId=70002 JobName=gint-release-x Partition=mrigpu "
            "NodeList=compute-1-6 ReqNodeList=compute-1-6 "
            "JobState=RUNNING NumCPUs=64 NumTasks=1 CPUs/Task=64 "
            "Command=/snapshot/run_mtu_gint_gate.sbatch\n"
        )

    monkeypatch.setattr(GINT_AUDIT.subprocess, "check_output", fake_check_output)
    observed = GINT_AUDIT._query_slurm_controller("70002")
    assert captured[0][0] == str(trusted_test_binary)
    assert captured[0][0] != str(attacker)
    assert observed["ReqNodeList"] == "compute-1-6"
    assert observed["CPUs/Task"] == "64"


def _candidate_runtime_manifest(
    *,
    source_sha256: str,
    source_file_count: int,
    task_root: str,
    files: list[dict] | None = None,
) -> dict:
    required = list(SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES)
    if files is None:
        files = [{
            "relative_path": f"gpu4pyscf/lib/{name}",
            "sha256": "c" * 64,
            "bytes": 123,
        } for name in required]
    inventory = [{
        "name": name,
        "bytes": entry["bytes"],
        "sha256": entry["sha256"],
    } for name, entry in zip(required, files, strict=True)]
    bundle_id = SNAPSHOT._canonical_json_sha256(inventory)
    bundle_root = f"{task_root}/runtime-bundles/{bundle_id}"
    evidence = {
        "schema": SNAPSHOT.RUNTIME_VALIDATION_EVIDENCE_SCHEMA,
        "validated": True,
        "bundle_id": bundle_id,
        "bundle_root": bundle_root,
        "manifest_path": (
            f"{task_root}/runtime-bundle-manifests/{bundle_id}.json"
        ),
        "build_verification_path": (
            f"{task_root}/builds/"
            f"{source_sha256}-gint-bounded-workspace-v2-sm80/"
            "build-verification.json"
        ),
        "library_count": len(required),
        "inventory_sha256": bundle_id,
        "inventory_payload_sha256": bundle_id,
        "source_sha256": source_sha256,
        "source_file_count": source_file_count,
        "cuda_version": "12.8",
        "cuda_architecture": "80-real",
        "abi_version": 2,
        "maximum_stack_bytes": 1024,
        "build_verification_bound": True,
        "sidecar_sha256": "d" * 64,
        "sidecar_bytes": 4096,
        "payload_sha256": "e" * 64,
        "build_verification_sha256": "f" * 64,
        "build_verification_bytes": 8192,
        "build_verification_payload_sha256": "1" * 64,
    }
    return {
        "source_root": bundle_root,
        "complete": True,
        "required_library_names": required,
        "inventory_library_names": required,
        "closure_complete": True,
        "files": files,
        "contract_schema": "gpu4pyscf.runtime-bundle.v2",
        "bundle_id": bundle_id,
        "inventory_sha256": bundle_id,
        "sidecar_sha256": evidence["sidecar_sha256"],
        "build_verification_sha256": evidence[
            "build_verification_sha256"
        ],
        "validation_evidence": evidence,
        "validation_evidence_sha256": SNAPSHOT._canonical_json_sha256(
            evidence
        ),
    }


def _passing_source_evidence() -> dict:
    source = f"/task/snapshots/{SOURCE_DIGEST}/source"
    provenance = _local_provenance()
    source_file_count = 3
    return {
        "root": source,
        "tree_sha256_at_start": SOURCE_DIGEST,
        "tree_sha256_at_end": SOURCE_DIGEST,
        "files_hashed_at_start": source_file_count,
        "files_hashed_at_end": source_file_count,
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
                "schema": GINT_AUDIT.SOURCE_SNAPSHOT_SCHEMA,
                "tree_sha256": SOURCE_DIGEST,
                "base_revision": GATE.FROZEN_BASE_COMMIT,
                "source": source,
                "immutable": True,
                "source_normalization": _source_normalization(),
                "local_provenance": provenance,
                "publication_policy": SNAPSHOT.PUBLICATION_POLICY,
                "runtime_binaries": _candidate_runtime_manifest(
                    source_sha256=SOURCE_DIGEST,
                    source_file_count=source_file_count,
                    task_root="/task",
                ),
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
    timed_h2d_operations = json.loads(json.dumps(h2d_operations))
    timed_d2h_operations = json.loads(json.dumps(d2h_operations))
    for operations in (h2d_operations, d2h_operations):
        for name, item in operations.items():
            item.update({
                "logical_payload": name.replace("_", "-"),
                "provenance": "unit-test-provider",
            })
    timed_d2h_operations["direct_cd_residual_maximum"] = {
        "bytes": 80,
        "count": 10,
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
                "workspace_size": GATE.SELECTED_WORKSPACE_SIZE_SYMBOL,
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
            "maximum_rys_order": 7,
            "high_rys_workspace": {
                "strategy": "provider-owned-bounded-global-grid-stride",
                "minimum_rys_order": 7,
                "maximum_rys_order": 7,
                "threads_per_block": GATE.SELECTED_HIGH_RYS_THREADS,
                "multiprocessor_count": GATE.EXPECTED_A100_MULTIPROCESSORS,
                "slot_count": (
                    GATE.EXPECTED_A100_MULTIPROCESSORS
                    * GATE.SELECTED_HIGH_RYS_THREADS
                ),
                "nbytes": GATE.EXPECTED_RYS7_WORKSPACE_BYTES,
                "resident": True,
                "static_thread_gout_eliminated_for_orders": [7, 8],
            },
            "stream_safety": {
                "strategy": GATE.SELECTED_STREAM_SAFETY_STRATEGY,
                "policy_version": (
                    GATE.SELECTED_STREAM_SAFETY_POLICY_VERSION
                ),
                "binding_scope": "process-wide-per-cuda-device",
                "construction_process_id": 4321,
                "construction_device_id": 0,
                "bound_stream_ptr": 123456,
                "strong_stream_lifetime": True,
                "host_enqueue_lock_scope": "entire-public-provider-call",
                "shared_constant_cache_ordering": "bound-cuda-stream",
                "provider_workspace_ordering": "bound-cuda-stream",
                "cross_stream_behavior": "reject-before-device-enqueue",
                "per_call_device_synchronize": False,
                "successful_call_count": 42,
            },
            "schedule": {
                "single_shell_support": True,
                "npair": dimension,
                "nao_original": 116,
            },
            "kernel_launches": 2,
            "performance_eligible": True,
            "performance_gate": "water2-selected-provider-gate-complete",
            "runtime_gate": {
                "validated": True,
                "receipt_binding_validated": True,
                "status": "release-accepted",
            },
            "explicit_transfers": {
                "schema": "gpu4pyscf.gint-selected-transfer-ledger.v2",
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
                "complete_for_declared_payloads": True,
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
            "total_bytes": transfer_total + 80,
            "total_transfers": (
                sum(item["count"] for item in timed_h2d_operations.values())
                + sum(item["count"] for item in timed_d2h_operations.values())
            ),
            "by_kind": {
                "h2d": {
                    "bytes": sum(
                        item["bytes"] for item in timed_h2d_operations.values()
                    ),
                    "count": sum(
                        item["count"] for item in timed_h2d_operations.values()
                    ),
                },
                "d2h": {
                    "bytes": sum(
                        item["bytes"] for item in timed_d2h_operations.values()
                    ),
                    "count": sum(
                        item["count"] for item in timed_d2h_operations.values()
                    ),
                },
            },
            "by_operation": {
                "h2d": timed_h2d_operations,
                "d2h": timed_d2h_operations,
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
            "synchronized_hbm": {
                "schema": "gpu4pyscf.gint-synchronized-hbm.v1",
                "checkpoints": [{
                    "name": name,
                    "driver_used_bytes": 4 * 1024**3,
                    "driver_total_bytes": 80 * 1024**3,
                    "cupy_pool_used_bytes": 2 * 1024**3,
                    "cupy_pool_total_bytes": 3 * 1024**3,
                    "timing_semantics": (
                        "cuda-device-synchronized-instantaneous"
                    ),
                    "measurement_scope": "exclusive-slurm-device-global",
                    "source": (
                        "cudaMemGetInfo-and-cupy-default-memory-pool"
                    ),
                } for name in GATE.SYNCHRONIZED_HBM_CHECKPOINT_NAMES],
                "peak_driver_used_bytes": 4 * 1024**3,
                "peak_driver_used_gib": 4.0,
                "budget_gib": GATE.HBM_LIMIT_GIB,
            },
            "direct_cd_peak_host_rss_gib": 1.0,
        },
    }



def _seal_snapshot(source: Path, manifest_path: Path) -> None:
    for path in source.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    manifest_path.chmod(0o444)
    source.parent.chmod(0o555)


def _rewrite_readonly_json(path: Path, value: dict) -> None:
    path.chmod(0o600)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    path.chmod(0o444)


def _materialize_qualification(
    tmp_path: Path, *, with_internal_link: bool = False,
    post_publish_hook=None,
):
    """Create qualification A, a receipt, and receipt-pinned release B."""

    task = tmp_path / "task"
    snapshots = task / "snapshots"
    source = snapshots / "staging-a" / "source"
    library_dir = source / "gpu4pyscf" / "lib"
    package_dir = source / "gpu4pyscf" / "cc"
    library_dir.mkdir(parents=True)
    package_dir.mkdir(parents=True)
    (package_dir / "provider.py").write_text("QUALIFIED_SOURCE = True\n")
    if with_internal_link:
        link_dir = source / "builder"
        target_dir = source / "dockerfiles" / "manylinux"
        link_dir.mkdir()
        target_dir.mkdir(parents=True)
        (target_dir / "build_wheels.sh").write_text("#!/bin/sh\nexit 0\n")
        (link_dir / "build_wheels.sh").symlink_to(
            "../dockerfiles/manylinux/build_wheels.sh"
        )
    libgint = library_dir / "libgint.so"
    libgint.write_bytes(b"unit-test-libgint\0")
    (source / "gpu4pyscf" / "__init__.py").write_text("\n")
    (library_dir / "__init__.py").write_text("\n")
    (library_dir / "utils.py").write_text(
        "def load_library(name):\n    return name\n"
    )
    (library_dir / "runtime_targets.py").write_text(
        "from gpu4pyscf.lib.utils import load_library\n"
        + "".join(
            f"load_library({name.removesuffix('.so')!r})\n"
            for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
        )
    )
    for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES:
        binary = library_dir / name
        if not binary.exists():
            binary.write_bytes(f"unit-test-{name}\0".encode())
    lib_sha = hashlib.sha256(libgint.read_bytes()).hexdigest()
    source_inventory = NORMALIZER.inspect_source(source)
    source_normalization = NORMALIZER.materialize_source_links(
        source, source_inventory
    )
    uploaded_normalization = (
        json.dumps(source_normalization, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    normalized = NORMALIZER.assert_normalized_source(source)
    source_normalization.update({
        "remote_verified_after_runtime_binding": True,
        "remote_regular_file_count_after_runtime_binding": normalized[
            "regular_file_count_after"
        ],
        "remote_directory_count_after_runtime_binding": normalized[
            "directory_count_after"
        ],
        "uploaded_record_sha256": hashlib.sha256(
            uploaded_normalization
        ).hexdigest(),
        "uploaded_record_bytes": len(uploaded_normalization),
    })
    source_digest, source_file_count = RECEIPT.release_source_tree_digest(source)
    qualification_snapshot = snapshots / source_digest
    source.parent.rename(qualification_snapshot)
    source = qualification_snapshot / "source"
    libgint = source / "gpu4pyscf" / "lib" / "libgint.so"

    manifest_path = source.parent / "manifest.json"
    manifest = _passing_source_evidence()["snapshot"]["manifest"]
    manifest.update({"source": str(source), "tree_sha256": source_digest})
    manifest["source_normalization"] = source_normalization
    runtime_files = [
        {
            "relative_path": f"gpu4pyscf/lib/{name}",
            "sha256": hashlib.sha256(
                (source / "gpu4pyscf" / "lib" / name).read_bytes()
            ).hexdigest(),
            "bytes": (source / "gpu4pyscf" / "lib" / name).stat().st_size,
        }
        for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
    ]
    manifest["runtime_binaries"] = _candidate_runtime_manifest(
        source_sha256=source_digest,
        source_file_count=source_file_count,
        task_root=str(task),
        files=runtime_files,
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    attestation = {
        "schema": SNAPSHOT.PUBLICATION_ATTESTATION_SCHEMA,
        "status": "published",
        "source_tree_sha256": source_digest,
        "snapshot_name": source_digest,
        "published_directory_identity": {
            "device": qualification_snapshot.stat().st_dev,
            "inode": qualification_snapshot.stat().st_ino,
        },
        "publish_protocol": "atomic-exclusive-rename",
        "trust_boundary": SNAPSHOT.PUBLISH_TRUST_BOUNDARY,
        "publication_policy_sha256": SNAPSHOT._canonical_json_sha256(
            SNAPSHOT.PUBLICATION_POLICY
        ),
        "published_manifest_sha256": manifest_sha,
        "prepublication_manifest_sha256": manifest_sha,
    }
    attestation["attestation_sha256"] = SNAPSHOT._canonical_json_sha256(
        attestation
    )
    attestation_path = qualification_snapshot.parent / (
        f"{source_digest}{SNAPSHOT.PUBLICATION_ATTESTATION_SUFFIX}"
    )
    attestation_path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n"
    )
    attestation_path.chmod(0o444)

    topology_path = (
        task / "results" / "gint-gate" / "topology-physical8-70001.json"
    )
    topology_path.parent.mkdir(parents=True)
    topology = {
        "pid": 41001,
        "performance_eligible": True,
        "benchmark_track": "performance",
        "host": "compute-1-6",
        "mode": "physical8",
        "expected_gpu_numa_node": 3,
        "requested_physical_cores": 8,
        "requested_threads": 8,
        "gpu_numa_node": 3,
        "gpu_pci_bus_ids": ["0000:01:00.0"],
        "gpu_node_cpulist": "24-31,88-95",
        "allowed_cpulist_before": "0-31,64-95",
        "node_allowed_intersection": "24-31,88-95",
        "selected_cpulist": "24-31",
        "observed_cpulist_after_bind": "24-31",
        "mems_allowed_list": "3",
        "slurm": {
            "SLURM_JOB_ID": "70001",
            "SLURM_JOB_NAME": "gint-water2-gate",
            "SLURM_JOB_NODELIST": "compute-1-6",
            "SLURM_JOB_PARTITION": "mrigpu",
            "SLURM_CPUS_PER_TASK": "64",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    }
    topology_path.write_text(json.dumps(topology, sort_keys=True) + "\n")
    topology_sha = hashlib.sha256(topology_path.read_bytes()).hexdigest()
    record = _passing_record()
    record["source"] = _passing_source_evidence()
    record["source"].update({
        "root": str(source),
        "tree_sha256_at_start": source_digest,
        "tree_sha256_at_end": source_digest,
        "files_hashed_at_start": source_file_count,
        "files_hashed_at_end": source_file_count,
        "stable_during_run": True,
    })
    snapshot = record["source"]["snapshot"]
    snapshot.update({
        "source": str(source),
        "snapshot_root": str(source.parent),
        "manifest_path": str(manifest_path),
        "tree_sha256": source_digest,
        "tree_sha256_at_end": source_digest,
        "manifest_sha256_at_start": manifest_sha,
        "manifest_sha256_at_end": manifest_sha,
        "manifest": manifest,
    })
    record["topology"] = {
        "path": str(topology_path),
        "sha256": topology_sha,
        "sha256_at_start": topology_sha,
        "sha256_at_end": topology_sha,
        "passed": True,
        "stable_during_run": True,
        "reasons": [],
        "record": topology,
    }
    bpcache_abi = {
        "symbol": "GINTsizeof_basis_prod_cache",
        "python_ctypes_size_bytes": 144,
        "libgint_size_bytes": 144,
        "libgint_path": str(libgint),
        "libgint_sha256": lib_sha,
        "libgint_bytes": libgint.stat().st_size,
        "verified": True,
    }
    selected_abi = {
        "schema": "gpu4pyscf.gint-selected-pair-abi.v1",
        "size_symbol": "GINTsizeof_selected_pair_data",
        "offset_symbol": "GINToffsetof_selected_pair_data",
        "version_symbol": "GINTselected_pair_data_abi_version",
        "python_ctypes_size_bytes": 112,
        "libgint_size_bytes": 112,
        "abi_version": 2,
        "fields": [{
            "index": 0,
            "name": "npair",
            "python_offset_bytes": 0,
            "libgint_offset_bytes": 0,
        }],
        "verified": True,
    }
    record["provider"]["c_abi"].update({
        "basis_prod_cache_identity": bpcache_abi,
        "selected_pair_identity": selected_abi,
    })
    record["provider"]["performance_eligible"] = False
    record["provider"]["runtime_gate"] = {
        "validated": False,
        "receipt_binding_validated": False,
        "status": "pending",
    }
    record["gate_decision"] = GATE.evaluate_gate(record)
    assert record["gate_decision"]["qualification_passed"] is True
    assert record["gate_decision"]["performance_eligible"] is False
    record["performance_eligible"] = False

    result_path = (
        task / "results" / "gint-gate" / "water2-gint-direct-cd-70001.json"
    )
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    slurm_output = task / "logs" / "gint-gate-70001.log"
    slurm_output.parent.mkdir()
    slurm_output.write_text("qualification completed\n")
    terminal = {
        "job_id": "70001",
        "job_name": "gint-water2-gate",
        "node": "compute-1-6",
        "partition": "mrigpu",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "submit_utc": "2026-09-14T00:00:00",
        "start_utc": "2026-09-14T00:01:00",
        "end_utc": "2026-09-14T00:02:00",
        "elapsed": "00:01:00",
    }
    _seal_snapshot(source, manifest_path)
    payload = RECEIPT.build_receipt_payload(
        result_path, slurm_output, slurm_terminal=terminal
    )
    receipt_path = (
        task / "results" / "gint-runtime-gate-receipt" / "receipt.json"
    )
    RECEIPT.write_once_receipt(receipt_path, payload)
    release_pin = RECEIPT.build_release_pin(receipt_path)
    release_pin_staging = task / "results" / "gint_release_pin.json"
    RECEIPT.write_once_release_pin(release_pin_staging, release_pin)

    promotion = PROMOTION.promote_release_snapshot(
        source,
        manifest_path,
        release_pin_staging,
        task_root=task,
        post_publish_hook=post_publish_hook,
    )
    release_digest = promotion["release_source_sha256"]
    release_source = Path(promotion["release_source"])
    release_manifest_path = Path(promotion["release_manifest"])
    release_libgint = release_source / "gpu4pyscf" / "lib" / "libgint.so"

    release_topology_path = (
        task / "results" / "gint-gate" / "topology-physical8-70002.json"
    )
    release_topology = {
        **topology,
        "pid": 42002,
        "kernel_memory_policy": {
            "verified": True,
            "after": {"mode_name": "preferred", "nodes": [3]},
        },
        "command": [
            "python",
            str(release_source / "benchmarks/cc/a100_water8/gint_gate.py"),
        ],
        "slurm": {
            **topology["slurm"],
            "SLURM_JOB_ID": "70002",
            "SLURM_JOB_NAME": (
                "gint-release-" + RECEIPT.canonical_json_sha256(payload)
            ),
        },
    }
    release_topology_path.write_text(
        json.dumps(release_topology, sort_keys=True) + "\n"
    )
    release_topology_path.chmod(0o444)
    release_topology_sha = hashlib.sha256(
        release_topology_path.read_bytes()
    ).hexdigest()
    release_environment = {
        **release_topology["slurm"],
        "CCSD_PERFORMANCE_ELIGIBLE": "true",
        "CCSD_BENCHMARK_TRACK": "performance",
        "CCSD_TOPOLOGY_RECORD": str(release_topology_path),
        "CCSD_TOPOLOGY_SHA256": release_topology_sha,
        "CCSD_BOUND_CPUS": "24-31",
        "GPU4PYSCF_NUMA": "3",
    }
    _TEST_RELEASE_RUNTIME.clear()
    _TEST_RELEASE_RUNTIME.update({
        "schema": GINT_AUDIT.RELEASE_RUNTIME_OBSERVATION_SCHEMA,
        "source": "current-process-linux-cuda",
        "pid": 42002,
        "host": "compute-1-6",
        "affinity": "24-31",
        "mems_allowed_list": "3",
        "gpu_node_cpulist": "24-31,88-95",
        "kernel_memory_policy": {
            "mode": 1,
            "mode_name": "preferred",
            "nodes": [3],
        },
        "environment": release_environment,
        "slurm_controller": {
            "JobId": "70002",
            "JobName": release_environment["SLURM_JOB_NAME"],
            "Partition": "mrigpu",
            "NodeList": "compute-1-6",
            "ReqNodeList": "compute-1-6",
            "JobState": "RUNNING",
            "NumCPUs": "64",
            "NumTasks": "1",
            "CPUs/Task": "64",
            "Command": str(
                release_source
                / "benchmarks"
                / "cc"
                / "a100_water8"
                / "run_mtu_gint_gate.sbatch"
            ),
        },
        "cuda": {
            "visible_devices_environment": "0",
            "device_count": 1,
            "current_device": 0,
            "pci_bus_id": "0000:01:00.0",
            "numa_node": 3,
        },
        "errors": [],
    })
    validation_arguments = {
        "basis_prod_cache_size_bytes": 144,
        "loaded_libgint_sha256": lib_sha,
        "loaded_libgint_path": release_libgint,
        "loaded_libgint_bytes": release_libgint.stat().st_size,
        "selected_pair_abi": selected_abi,
        "selected_c_abi": {
            "columns": GATE.SELECTED_COLUMNS_SYMBOL,
            "diagonal": GATE.SELECTED_DIAGONAL_SYMBOL,
            "workspace_size": GATE.SELECTED_WORKSPACE_SIZE_SYMBOL,
        },
        "loaded_source_digest": release_digest,
        "loaded_source_root": release_source,
        "loaded_manifest_path": release_manifest_path,
        "release_topology_path": release_topology_path,
        "loaded_release_job_id": "70002",
        "loaded_release_context": {
            "pid": 42002,
            "host": "compute-1-6",
            "affinity": "24-31",
            "gpu_pci_bus_id": "0000:01:00.0",
            "gpu_numa_node": 3,
            "environment": release_environment,
        },
    }
    return {
        "receipt": receipt_path,
        "result": result_path,
        "manifest": release_manifest_path,
        "qualification_manifest": manifest_path,
        "libgint": release_libgint,
        "qualification_source": source,
        "release_source": release_source,
        "release_pin": release_source / RECEIPT.RELEASE_PIN_RELATIVE_PATH,
        "release_pin_staging": release_pin_staging,
        "arguments": validation_arguments,
    }


def test_write_once_qualification_receipt_unlocks_matching_release(tmp_path: Path):
    evidence = _materialize_qualification(tmp_path)
    accepted = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    assert accepted["payload_validated"] is True
    assert accepted["receipt_binding_validated"] is True
    assert accepted["validated"] is True
    assert accepted["status"] == "release-accepted"
    assert accepted["validation_errors"] == []
    assert accepted["binding_contract"][
        "caller_supplied_runtime_context_authoritative"
    ] is False
    assert accepted["receipt"]["current_runtime"]["source"] == (
        "current-process-linux-cuda"
    )
    assert len(accepted["receipt"]["current_runtime_sha256"]) == 64
    assert accepted["receipt"]["release_runtime_contract"][
        "cpu_affinity"
    ] == "24-31"
    assert accepted["receipt"]["job_id"] == "70001"
    assert (
        accepted["receipt"]["qualification_source_sha256"]
        != accepted["receipt"]["release_source_sha256"]
    )
    assert RECEIPT.release_source_tree_digest(
        evidence["release_source"], exclude_release_pin=True
    )[0] == RECEIPT.release_source_tree_digest(
        evidence["qualification_source"]
    )[0]
    assert stat.S_IMODE(evidence["receipt"].stat().st_mode) == 0o444
    assert stat.S_IMODE(
        Path(str(evidence["receipt"]) + ".sha256").stat().st_mode
    ) == 0o444
    assert stat.S_IMODE(evidence["release_pin"].stat().st_mode) == 0o444
    assert stat.S_IMODE(evidence["release_source"].stat().st_mode) == 0o555
    release_manifest = json.loads(evidence["manifest"].read_text())
    assert release_manifest["gint_release_lineage"]["release_pin_sha256"] == (
        accepted["receipt"]["release_pin_sha256"]
    )
    qualification_attestation = json.loads(
        (evidence["qualification_source"].parent.parent
         / f"{evidence['qualification_source'].parent.name}{SNAPSHOT.PUBLICATION_ATTESTATION_SUFFIX}").read_text()
    )
    assert release_manifest["gint_release_lineage"][
        "qualification_publication_attestation_sha256"
    ] == qualification_attestation["attestation_sha256"]
    release_attestation_path = evidence["release_source"].parent.parent / (
        f"{evidence['release_source'].parent.name}{SNAPSHOT.PUBLICATION_ATTESTATION_SUFFIX}"
    )
    assert release_attestation_path.is_file()
    assert json.loads(release_attestation_path.read_text())["source_tree_sha256"] == (
        evidence["release_source"].parent.name
    )

    with pytest.raises(FileExistsError, match="refusing to"):
        RECEIPT.write_once_receipt(
            evidence["receipt"],
            RECEIPT.build_receipt_payload(
                evidence["result"],
                tmp_path / "task" / "logs" / "gint-gate-70001.log",
                slurm_terminal={
                    "job_id": "70001",
                    "job_name": "gint-water2-gate",
                    "node": "compute-1-6",
                    "partition": "mrigpu",
                    "state": "COMPLETED",
                    "exit_code": "0:0",
                    "submit_utc": "x",
                    "start_utc": "x",
                    "end_utc": "x",
                    "elapsed": "x",
                },
            ),
        )


@pytest.mark.parametrize("mutation", ["tamper", "extra", "missing"])
def test_release_lineage_binds_exact_qualification_attestation(
    tmp_path: Path, mutation: str
):
    evidence = _materialize_qualification(tmp_path)
    manifest_path = evidence["manifest"]
    manifest = json.loads(manifest_path.read_text())
    lineage = manifest["gint_release_lineage"]
    if mutation == "tamper":
        lineage["qualification_publication_attestation"]["status"] = "forged"
    elif mutation == "extra":
        lineage["qualification_publication_attestation_extra"] = True
    else:
        lineage.pop("qualification_publication_attestation")
    _rewrite_readonly_json(manifest_path, manifest)
    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    reasons = "\n".join(rejected["validation_errors"])
    assert rejected["validated"] is False
    assert "A-to-B lineage" in reasons


@pytest.mark.parametrize(
    "protocol",
    [
        "atomic-exclusive-rename-commit-confirmed",
        "reserved-empty-directory-rename-commit-confirmed",
    ],
)
def test_a_attestation_accepts_commit_confirmed_native_and_fallback(
    tmp_path: Path, protocol: str
):
    evidence = _materialize_qualification(tmp_path)
    source = evidence["qualification_source"]
    attestation_path = source.parent.parent / (
        f"{source.parent.name}{SNAPSHOT.PUBLICATION_ATTESTATION_SUFFIX}"
    )
    attestation = json.loads(attestation_path.read_text())
    attestation["publish_protocol"] = protocol
    unsigned = dict(attestation)
    unsigned.pop("attestation_sha256", None)
    attestation["attestation_sha256"] = GINT_AUDIT.canonical_json_sha256(
        unsigned
    )
    _rewrite_readonly_json(attestation_path, attestation)
    manifest = json.loads(evidence["qualification_manifest"].read_text())
    errors, loaded = GINT_AUDIT._publication_attestation_checks(
        manifest, source_root=source
    )
    assert errors == []
    assert loaded["publish_protocol"] == protocol


def test_a_attestation_rejects_unknown_publication_protocol(tmp_path: Path):
    evidence = _materialize_qualification(tmp_path)
    source = evidence["qualification_source"]
    attestation_path = source.parent.parent / (
        f"{source.parent.name}{SNAPSHOT.PUBLICATION_ATTESTATION_SUFFIX}"
    )
    attestation = json.loads(attestation_path.read_text())
    attestation["publish_protocol"] = "fallback-link-reservation"
    unsigned = dict(attestation)
    unsigned.pop("attestation_sha256", None)
    attestation["attestation_sha256"] = GINT_AUDIT.canonical_json_sha256(
        unsigned
    )
    _rewrite_readonly_json(attestation_path, attestation)
    manifest = json.loads(evidence["qualification_manifest"].read_text())
    errors, loaded = GINT_AUDIT._publication_attestation_checks(
        manifest, source_root=source
    )
    assert loaded is not None
    assert any("protocol is invalid" in error for error in errors)


@pytest.mark.parametrize(
    "raw_pin",
    [
        '{"schema":"x","schema":"y"}\n',
        '{"schema":"x","value":NaN}\n',
        '{"schema":"x","value":Infinity}\n',
    ],
)
def test_release_pin_parse_or_schema_failure_publishes_no_new_b(
    tmp_path: Path, raw_pin: str
):
    evidence = _materialize_qualification(tmp_path)
    pin_path = evidence["release_pin_staging"]
    snapshots = evidence["qualification_source"].parent.parent
    before = {path.name for path in snapshots.iterdir()}
    pin_path.chmod(0o600)
    pin_path.write_text(raw_pin)
    pin_path.chmod(0o444)
    with pytest.raises(ValueError):
        PROMOTION.promote_release_snapshot(
            evidence["qualification_source"],
            evidence["qualification_manifest"],
            pin_path,
            task_root=snapshots.parent,
        )
    assert {path.name for path in snapshots.iterdir()} == before


def test_release_promotion_rejects_postpublish_ancestor_swap(tmp_path: Path):
    task = tmp_path / "task"
    snapshots = task / "snapshots"
    replacement = task / "replacement-snapshots"

    def swap_ancestor():
        snapshots.rename(task / "snapshots-held")
        replacement.mkdir()
        snapshots.symlink_to(replacement, target_is_directory=True)

    with pytest.raises((RuntimeError, ValueError), match="(ancestor|symbolic|canonical|identity)"):
        _materialize_qualification(
            tmp_path, post_publish_hook=swap_ancestor
        )
    assert not any(replacement.iterdir())
    assert (task / "snapshots-held").is_dir()


def test_normalized_internal_link_survives_a_to_b_as_bound_regular_file(
    tmp_path: Path,
):
    evidence = _materialize_qualification(
        tmp_path, with_internal_link=True
    )
    accepted = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    assert accepted["validated"] is True

    relative = Path("builder/build_wheels.sh")
    qualification_file = evidence["qualification_source"] / relative
    release_file = evidence["release_source"] / relative
    assert qualification_file.is_file() and not qualification_file.is_symlink()
    assert release_file.is_file() and not release_file.is_symlink()
    assert qualification_file.read_bytes() == release_file.read_bytes()

    qualification_manifest = json.loads(
        evidence["qualification_manifest"].read_text()
    )
    release_manifest = json.loads(evidence["manifest"].read_text())
    normalization = qualification_manifest["source_normalization"]
    assert normalization["materialized_link_count"] == 1
    assert normalization["materialized_links"] == [{
        "relative_path": relative.as_posix(),
        "link_type": "relative-leaf-file",
        "target": "../dockerfiles/manylinux/build_wheels.sh",
        "resolved_target_relative_path": (
            "dockerfiles/manylinux/build_wheels.sh"
        ),
        "target_bytes": len(b"#!/bin/sh\nexit 0\n"),
        "target_sha256": hashlib.sha256(
            b"#!/bin/sh\nexit 0\n"
        ).hexdigest(),
    }]
    assert release_manifest["source_normalization"] == normalization


def _configure_consumer_evidence(
    evidence: dict, *, mode: str, job_id: str = "71002"
) -> dict:
    task = evidence["release_source"].parent.parent.parent
    source = evidence["release_source"]
    old_topology_path = evidence["arguments"]["release_topology_path"]
    topology = json.loads(old_topology_path.read_text())
    if mode == "consumer-benchmark":
        job_name = "ccsd-benchmark"
        driver = source / "benchmarks/cc/a100_water8/benchmark.py"
        launcher = source / "benchmarks/cc/a100_water8/run_mtu_benchmark.sbatch"
        topology_path = (
            task / "results" / "topology"
            / f"physical8-benchmark-{job_id}.json"
        )
    elif mode == "consumer-counterpoise":
        job_name = "ccsd-counterpoise"
        driver = source / "benchmarks/cc/a100_water8/counterpoise.py"
        launcher = source / "benchmarks/cc/a100_water8/run_mtu_counterpoise.sbatch"
        topology_path = (
            task / "results" / "counterpoise" / "candidate"
            / f"topology-physical8-{job_id}.json"
        )
    else:  # pragma: no cover - helper callers are exhaustive
        raise AssertionError(mode)
    topology.update({
        "pid": 43002,
        "command": ["python", str(driver)],
        "slurm": {
            **topology["slurm"],
            "SLURM_JOB_ID": job_id,
            "SLURM_JOB_NAME": job_name,
        },
    })
    topology_path.parent.mkdir(parents=True, exist_ok=True)
    topology_path.write_text(json.dumps(topology, sort_keys=True) + "\n")
    topology_path.chmod(0o444)
    topology_sha = hashlib.sha256(topology_path.read_bytes()).hexdigest()
    environment = {
        **topology["slurm"],
        "CCSD_PERFORMANCE_ELIGIBLE": "true",
        "CCSD_BENCHMARK_TRACK": "performance",
        "CCSD_TOPOLOGY_RECORD": str(topology_path),
        "CCSD_TOPOLOGY_SHA256": topology_sha,
        "CCSD_BOUND_CPUS": "24-31",
        "GPU4PYSCF_NUMA": "3",
    }
    _TEST_RELEASE_RUNTIME.clear()
    _TEST_RELEASE_RUNTIME.update({
        "schema": GINT_AUDIT.RELEASE_RUNTIME_OBSERVATION_SCHEMA,
        "source": "current-process-linux-cuda",
        "pid": 43002,
        "host": "compute-1-6",
        "affinity": "24-31",
        "mems_allowed_list": "3",
        "gpu_node_cpulist": "24-31,88-95",
        "kernel_memory_policy": {
            "mode": 1,
            "mode_name": "preferred",
            "nodes": [3],
        },
        "environment": environment,
        "slurm_controller": {
            "JobId": job_id,
            "JobName": job_name,
            "Partition": "mrigpu",
            "NodeList": "compute-1-6",
            "ReqNodeList": "compute-1-6",
            "JobState": "RUNNING",
            "NumCPUs": "64",
            "NumTasks": "1",
            "CPUs/Task": "64",
            "Command": str(launcher),
        },
        "cuda": {
            "visible_devices_environment": "0",
            "device_count": 1,
            "current_device": 0,
            "pci_bus_id": "0000:01:00.0",
            "numa_node": 3,
        },
        "errors": [],
    })
    arguments = dict(evidence["arguments"])
    arguments.update({
        "execution_mode": mode,
        "release_topology_path": topology_path,
        "loaded_release_job_id": job_id,
    })
    return {
        "arguments": arguments,
        "topology": topology,
        "topology_path": topology_path,
    }


@pytest.mark.parametrize(
    "mode", ("consumer-benchmark", "consumer-counterpoise")
)
def test_qualified_release_unlocks_strict_current_consumer(
    tmp_path: Path, mode: str,
):
    evidence = _materialize_qualification(tmp_path)
    consumer = _configure_consumer_evidence(evidence, mode=mode)

    accepted = validate_runtime_performance_gate(
        evidence["receipt"], **consumer["arguments"]
    )

    assert accepted["validated"] is True
    assert accepted["status"] == "consumer-accepted"
    assert accepted["execution_mode"] == mode
    assert accepted["binding_contract"]["execution_mode"] == mode
    assert accepted["validation_errors"] == []


@pytest.mark.parametrize(
    ("mode", "tamper", "reason"),
    (
        (
            "consumer-benchmark",
            "controller-command",
            "did not launch snapshot B consumer",
        ),
        (
            "consumer-benchmark",
            "topology-directory",
            "fixed task/results job path",
        ),
        (
            "consumer-counterpoise",
            "topology-job-key",
            "not keyed by SLURM_JOB_ID",
        ),
        (
            "consumer-counterpoise",
            "live-affinity",
            "CPU affinity is not 24-31",
        ),
    ),
)
def test_consumer_runtime_gate_fails_closed_on_current_job_tampering(
    tmp_path: Path, mode: str, tamper: str, reason: str,
):
    evidence = _materialize_qualification(tmp_path)
    consumer = _configure_consumer_evidence(evidence, mode=mode)
    arguments = consumer["arguments"]
    if tamper == "controller-command":
        _TEST_RELEASE_RUNTIME["slurm_controller"]["Command"] = (
            "/tmp/forged-consumer.sbatch"
        )
    elif tamper in {"topology-directory", "topology-job-key"}:
        original = consumer["topology_path"]
        forged = original.parent / (
            "forged.json" if tamper == "topology-job-key" else original.name
        )
        if tamper == "topology-directory":
            forged = original.parent / "forged" / original.name
        forged.parent.mkdir(parents=True, exist_ok=True)
        forged.write_bytes(original.read_bytes())
        forged.chmod(0o444)
        arguments["release_topology_path"] = forged
        _TEST_RELEASE_RUNTIME["environment"]["CCSD_TOPOLOGY_RECORD"] = str(forged)
        _TEST_RELEASE_RUNTIME["environment"]["CCSD_TOPOLOGY_SHA256"] = (
            hashlib.sha256(forged.read_bytes()).hexdigest()
        )
    elif tamper == "live-affinity":
        _TEST_RELEASE_RUNTIME["affinity"] = "25-32"
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(tamper)

    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **arguments
    )

    assert rejected["validated"] is False
    assert reason in "\n".join(rejected["validation_errors"])


def test_receipt_rejects_tampered_bound_result_and_runtime_mismatch(tmp_path: Path):
    evidence = _materialize_qualification(tmp_path)
    evidence["result"].write_text(evidence["result"].read_text() + " \n")
    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    assert rejected["validated"] is False
    assert any(
        "qualification result SHA-256" in reason
        for reason in rejected["validation_errors"]
    )

    evidence = _materialize_qualification(tmp_path / "second")
    bad_arguments = dict(evidence["arguments"])
    bad_arguments["loaded_source_digest"] = "f" * 64
    bad_arguments["loaded_libgint_sha256"] = "e" * 64
    bad_arguments["selected_pair_abi"] = {
        **bad_arguments["selected_pair_abi"],
        "abi_version": 3,
    }
    bad_arguments["selected_c_abi"] = {
        **bad_arguments["selected_c_abi"],
        "workspace_size": "missing_workspace_query",
    }
    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **bad_arguments
    )
    reasons = "\n".join(rejected["validation_errors"])
    assert "source digest" in reasons
    assert "libgint digest" in reasons
    assert "ABI size/offset/version" in reasons
    assert "selected C ABI symbol identity" in reasons

    evidence = _materialize_qualification(tmp_path / "binary")
    release_source = evidence["release_source"]
    libgint = evidence["libgint"]
    release_source.chmod(0o755)
    libgint.parent.chmod(0o755)
    libgint.chmod(0o644)
    libgint.write_bytes(b"tampered-release-libgint\0")
    libgint.chmod(0o444)
    libgint.parent.chmod(0o555)
    release_source.chmod(0o555)
    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    reasons = "\n".join(rejected["validation_errors"])
    assert rejected["validated"] is False
    assert "loaded release libgint byte length changed" in reasons
    assert "release snapshot B libgint differs from qualification A" in reasons
    assert "release source manifest libgint identity mismatch" in reasons


def test_receipt_rejects_mismatched_current_release_topology(tmp_path: Path):
    evidence = _materialize_qualification(tmp_path)
    topology_path = evidence["arguments"]["release_topology_path"]
    topology = json.loads(topology_path.read_text())
    topology["host"] = "compute-1-7"
    _rewrite_readonly_json(topology_path, topology)
    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    assert rejected["validated"] is False
    assert "release topology differs" in "\n".join(
        rejected["validation_errors"]
    )

    topology["host"] = "compute-1-6"
    topology["slurm"]["SLURM_JOB_ID"] = "70003"
    _rewrite_readonly_json(topology_path, topology)
    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    assert rejected["validated"] is False
    assert "not bound to release topology job" in "\n".join(
        rejected["validation_errors"]
    )


def test_release_topology_must_be_read_only_fixed_current_job_path(tmp_path: Path):
    evidence = _materialize_qualification(tmp_path)
    topology_path = evidence["arguments"]["release_topology_path"]
    topology_path.chmod(0o644)
    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    reasons = "\n".join(rejected["validation_errors"])
    assert rejected["validated"] is False
    assert "current release topology must have no filesystem write bits" in reasons

    fresh = _materialize_qualification(tmp_path / "fixed-path")
    original = fresh["arguments"]["release_topology_path"]
    forged_directory = original.parent / "forged"
    forged_directory.mkdir()
    forged = forged_directory / original.name
    forged.write_bytes(original.read_bytes())
    forged.chmod(0o444)
    arguments = dict(fresh["arguments"])
    arguments["release_topology_path"] = forged
    arguments["loaded_release_context"] = json.loads(json.dumps(
        arguments["loaded_release_context"]
    ))
    arguments["loaded_release_context"]["environment"][
        "CCSD_TOPOLOGY_RECORD"
    ] = str(forged)
    arguments["loaded_release_context"]["environment"][
        "CCSD_TOPOLOGY_SHA256"
    ] = hashlib.sha256(forged.read_bytes()).hexdigest()
    rejected = validate_runtime_performance_gate(
        fresh["receipt"], **arguments
    )
    reasons = "\n".join(rejected["validation_errors"])
    assert rejected["validated"] is False
    assert "fixed task/results job path" in reasons


def test_release_slurm_allocation_must_request_qualification_node(tmp_path: Path):
    evidence = _materialize_qualification(tmp_path)
    _TEST_RELEASE_RUNTIME["slurm_controller"]["ReqNodeList"] = "(null)"
    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    reasons = "\n".join(rejected["validation_errors"])
    assert rejected["validated"] is False
    assert "not pinned to the qualification node" in reasons


@pytest.mark.parametrize(
    ("field", "expected_reason"),
    (
        ("job_name", "Slurm job name"),
        ("node", "hostname"),
        ("partition", "Slurm partition"),
        ("affinity", "CPU affinity"),
        ("pci", "GPU PCI"),
        ("numa", "GPU NUMA"),
        ("policy", "memory policy"),
        ("cuda_visible", "CUDA visible device"),
    ),
)
def test_same_job_forged_topology_and_caller_context_cannot_unlock_release(
    tmp_path: Path, field: str, expected_reason: str,
):
    """The caller mapping may match a forgery; the OS/CUDA probe still wins."""

    evidence = _materialize_qualification(tmp_path)
    topology_path = evidence["arguments"]["release_topology_path"]
    topology = json.loads(topology_path.read_text())
    context = evidence["arguments"]["loaded_release_context"]
    environment = context["environment"]
    if field == "job_name":
        value = "gint-release-" + "f" * 64
        topology["slurm"]["SLURM_JOB_NAME"] = value
        environment["SLURM_JOB_NAME"] = value
    elif field == "node":
        topology["host"] = "compute-1-7"
        topology["slurm"]["SLURM_JOB_NODELIST"] = "compute-1-7"
        context["host"] = "compute-1-7"
        environment["SLURM_JOB_NODELIST"] = "compute-1-7"
    elif field == "partition":
        topology["slurm"]["SLURM_JOB_PARTITION"] = "debug"
        environment["SLURM_JOB_PARTITION"] = "debug"
    elif field == "affinity":
        topology["selected_cpulist"] = "25-32"
        topology["observed_cpulist_after_bind"] = "25-32"
        context["affinity"] = "25-32"
        environment["CCSD_BOUND_CPUS"] = "25-32"
    elif field == "pci":
        topology["gpu_pci_bus_ids"] = ["0000:02:00.0"]
        context["gpu_pci_bus_id"] = "0000:02:00.0"
    elif field == "numa":
        topology["gpu_numa_node"] = 2
        context["gpu_numa_node"] = 2
        environment["GPU4PYSCF_NUMA"] = "2"
    elif field == "policy":
        topology["kernel_memory_policy"] = {
            "verified": True,
            "after": {"mode_name": "default", "nodes": []},
        }
    elif field == "cuda_visible":
        topology["slurm"]["CUDA_VISIBLE_DEVICES"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = "1"
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(field)
    _rewrite_readonly_json(topology_path, topology)
    environment["CCSD_TOPOLOGY_SHA256"] = hashlib.sha256(
        topology_path.read_bytes()
    ).hexdigest()

    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    reasons = "\n".join(rejected["validation_errors"])
    assert rejected["validated"] is False
    assert expected_reason in reasons
    assert rejected["receipt"]["current_runtime"]["affinity"] == "24-31"
    assert rejected["receipt"]["current_runtime"]["cuda"][
        "pci_bus_id"
    ] == "0000:01:00.0"


def test_receipt_rejects_symlink_writable_or_mapping_input(tmp_path: Path):
    evidence = _materialize_qualification(tmp_path)
    alias = evidence["receipt"].parent.parent / "receipt-alias.json"
    alias.symlink_to(evidence["receipt"])
    rejected = validate_runtime_performance_gate(alias, **evidence["arguments"])
    assert rejected["validated"] is False
    assert "symbolic link" in "\n".join(rejected["validation_errors"])

    evidence["receipt"].chmod(0o644)
    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    assert rejected["validated"] is False
    assert "write bits" in "\n".join(rejected["validation_errors"])
    with pytest.raises(TypeError, match="mappings are forbidden"):
        validate_runtime_performance_gate(
            {"path": str(evidence["receipt"])},
            **evidence["arguments"],
        )


def test_receipt_rejects_parent_component_symlink_on_issue_and_release(
    tmp_path: Path,
):
    evidence = _materialize_qualification(tmp_path)
    envelope = json.loads(evidence["receipt"].read_text())

    source_link = tmp_path / "source-link"
    source_link.symlink_to(evidence["arguments"]["loaded_source_root"])
    through_source = source_link / "forged-receipt" / "receipt.json"
    with pytest.raises(ValueError, match="must not traverse symlinks"):
        RECEIPT.write_once_receipt(through_source, envelope["payload"])

    receipt_parent_link = tmp_path / "receipt-parent-link"
    receipt_parent_link.symlink_to(evidence["receipt"].parent)
    rejected = validate_runtime_performance_gate(
        receipt_parent_link / evidence["receipt"].name,
        **evidence["arguments"],
    )
    assert rejected["validated"] is False
    assert "path traverses symbolic link" in "\n".join(
        rejected["validation_errors"]
    )

    release_source_link = tmp_path / "release-source-link"
    release_source_link.symlink_to(evidence["arguments"]["loaded_source_root"])
    linked_arguments = dict(evidence["arguments"])
    linked_arguments["loaded_source_root"] = release_source_link
    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **linked_arguments
    )
    assert rejected["validated"] is False
    assert "path traverses symbolic link" in "\n".join(
        rejected["validation_errors"]
    )


def test_receipt_content_sidecar_and_terminal_slurm_fail_closed(
    tmp_path: Path,
):
    evidence = _materialize_qualification(tmp_path)
    sidecar = Path(str(evidence["receipt"]) + ".sha256")
    sidecar.chmod(0o600)
    sidecar.write_text("0" * 64 + "  wrong.json\n")
    sidecar.chmod(0o444)
    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    assert rejected["validated"] is False
    assert "sidecar is invalid" in "\n".join(rejected["validation_errors"])

    fresh = _materialize_qualification(tmp_path / "terminal")
    with pytest.raises(ValueError, match="terminal Slurm state"):
        RECEIPT.build_receipt_payload(
            fresh["result"],
            tmp_path / "terminal" / "task" / "logs" / "gint-gate-70001.log",
            slurm_terminal={
                "job_id": "70001",
                "job_name": "gint-water2-gate",
                "node": "compute-1-6",
                "partition": "mrigpu",
                "state": "FAILED",
                "exit_code": "1:0",
                "submit_utc": "x",
                "start_utc": "x",
                "end_utc": "x",
                "elapsed": "x",
            },
        )


def test_replacing_both_receipt_files_is_rejected_by_snapshot_b_pin(
    tmp_path: Path,
):
    evidence = _materialize_qualification(tmp_path)
    receipt = evidence["receipt"]
    sidecar = Path(str(receipt) + ".sha256")
    envelope = json.loads(receipt.read_text())
    envelope["payload"]["replacement_nonce"] = 1
    envelope["payload_sha256"] = RECEIPT.canonical_json_sha256(
        envelope["payload"]
    )
    replacement = (
        json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    ).encode()
    replacement_sha = hashlib.sha256(replacement).hexdigest()

    receipt.parent.chmod(0o700)
    receipt.chmod(0o600)
    sidecar.chmod(0o600)
    receipt.write_bytes(replacement)
    sidecar.write_text(f"{replacement_sha}  {receipt.name}\n")
    receipt.chmod(0o444)
    sidecar.chmod(0o444)
    receipt.parent.chmod(0o555)

    # Also rewrite the external release topology's job-name anchor.  The
    # authoritative pin remains in content-addressed snapshot B and therefore
    # still rejects the jointly replaced receipt and sidecar.
    topology_path = evidence["arguments"]["release_topology_path"]
    topology = json.loads(topology_path.read_text())
    topology["slurm"]["SLURM_JOB_NAME"] = (
        "gint-release-" + envelope["payload_sha256"]
    )
    _rewrite_readonly_json(topology_path, topology)
    environment = evidence["arguments"]["loaded_release_context"][
        "environment"
    ]
    environment["SLURM_JOB_NAME"] = topology["slurm"]["SLURM_JOB_NAME"]
    environment["CCSD_TOPOLOGY_SHA256"] = hashlib.sha256(
        topology_path.read_bytes()
    ).hexdigest()

    rejected = validate_runtime_performance_gate(
        receipt, **evidence["arguments"]
    )
    assert rejected["validated"] is False
    assert "release pin does not bind the exact qualification receipt" in "\n".join(
        rejected["validation_errors"]
    )


def test_release_pin_or_source_change_requires_new_snapshot_b(tmp_path: Path):
    evidence = _materialize_qualification(tmp_path)
    release_source = evidence["release_source"]
    release_pin = evidence["release_pin"]

    release_source.chmod(0o755)
    release_pin.parent.chmod(0o755)
    release_pin.chmod(0o644)
    pin = json.loads(release_pin.read_text())
    pin["qualification"]["job_id"] = "79999"
    release_pin.write_text(json.dumps(pin, indent=2, sort_keys=True) + "\n")
    release_pin.chmod(0o444)
    release_pin.parent.chmod(0o555)
    release_source.chmod(0o555)

    rejected = validate_runtime_performance_gate(
        evidence["receipt"], **evidence["arguments"]
    )
    reasons = "\n".join(rejected["validation_errors"])
    assert rejected["validated"] is False
    assert "loaded release source digest differs from snapshot B" in reasons
    assert "release pin does not bind the exact qualification receipt" in reasons

    fresh = _materialize_qualification(tmp_path / "source-change")
    release_source = fresh["release_source"]
    provider_source = release_source / "gpu4pyscf" / "cc" / "provider.py"
    release_source.chmod(0o755)
    provider_source.parent.chmod(0o755)
    provider_source.chmod(0o644)
    provider_source.write_text("QUALIFIED_SOURCE = False\n")
    provider_source.chmod(0o444)
    provider_source.parent.chmod(0o555)
    release_source.chmod(0o555)
    rejected = validate_runtime_performance_gate(
        fresh["receipt"], **fresh["arguments"]
    )
    reasons = "\n".join(rejected["validation_errors"])
    assert rejected["validated"] is False
    assert "loaded release source digest differs from snapshot B" in reasons
    assert "does not reproduce qualification snapshot A" in reasons


def test_receipt_must_be_below_same_task_approved_results_root(tmp_path: Path):
    evidence = _materialize_qualification(tmp_path)
    outside_directory = tmp_path / "outside-receipt"
    outside_directory.mkdir()
    outside_receipt = outside_directory / evidence["receipt"].name
    outside_sidecar = Path(str(outside_receipt) + ".sha256")
    shutil.copy2(evidence["receipt"], outside_receipt)
    shutil.copy2(
        Path(str(evidence["receipt"]) + ".sha256"), outside_sidecar
    )
    outside_directory.chmod(0o555)

    rejected = validate_runtime_performance_gate(
        outside_receipt, **evidence["arguments"]
    )
    assert rejected["validated"] is False
    assert "outside the approved task results root" in "\n".join(
        rejected["validation_errors"]
    )


def test_query_terminal_slurm_parses_exact_allocation(monkeypatch):
    output = (
        "70001|gint-water2-gate|mrigpu|compute-1-6|COMPLETED|0:0|"
        "2026-09-14T00:00:00|2026-09-14T00:01:00|"
        "2026-09-14T00:02:00|00:01:00|\n"
    )
    monkeypatch.setattr(
        RECEIPT.subprocess,
        "check_output",
        lambda *args, **kwargs: output,
    )
    record = RECEIPT.query_terminal_slurm("70001")
    assert record == {
        "job_id": "70001",
        "job_name": "gint-water2-gate",
        "partition": "mrigpu",
        "node": "compute-1-6",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "submit_utc": "2026-09-14T00:00:00",
        "start_utc": "2026-09-14T00:01:00",
        "end_utc": "2026-09-14T00:02:00",
        "elapsed": "00:01:00",
    }

def test_gate_passes_only_complete_explicit_evidence_and_has_no_speed_target():
    decision = GATE.evaluate_gate(_passing_record())
    assert decision["passed"] is True
    assert decision["correctness_passed"] is True
    assert decision["qualification_passed"] is True
    assert decision["performance_eligible"] is True
    assert decision["reasons"] == []
    assert decision["speed_threshold_seconds"] is None
    timing = next(
        item for item in decision["checks"]
        if item["id"] == "complete_timing_boundary"
    )
    assert timing["evidence"]["speed_threshold_seconds"] is None


def test_synchronized_hbm_checkpoint_records_driver_and_pool_state():
    synchronizations = []
    pool = SimpleNamespace(used_bytes=lambda: 600, total_bytes=lambda: 700)
    cupy = SimpleNamespace(
        get_default_memory_pool=lambda: pool,
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                deviceSynchronize=lambda: synchronizations.append(True),
                memGetInfo=lambda: (200, 1000),
            ),
        ),
    )

    checkpoint = GATE.synchronized_hbm_checkpoint(cupy, "after-direct-cd")
    evidence = GATE.synchronized_hbm_evidence([checkpoint])

    assert synchronizations == [True]
    assert checkpoint["driver_used_bytes"] == 800
    assert checkpoint["cupy_pool_used_bytes"] == 600
    assert evidence["peak_driver_used_bytes"] == 800
    assert np.isclose(evidence["peak_driver_used_gib"], 800 / 1024**3)


def test_gate_rejects_missing_or_over_budget_synchronized_hbm_evidence():
    missing = _passing_record()
    del missing["resources"]["synchronized_hbm"]
    missing_decision = GATE.evaluate_gate(missing)
    assert "resource_budget" in {
        item["id"] for item in missing_decision["checks"]
        if not item["passed"]
    }

    over = _passing_record()
    transient = 72 * 1024**3 + 1
    synchronized = over["resources"]["synchronized_hbm"]
    synchronized["checkpoints"][1]["driver_used_bytes"] = transient
    synchronized["peak_driver_used_bytes"] = transient
    synchronized["peak_driver_used_gib"] = transient / float(1024**3)
    over_decision = GATE.evaluate_gate(over)
    resource = next(
        item for item in over_decision["checks"]
        if item["id"] == "resource_budget"
    )
    assert resource["passed"] is False
    assert resource["evidence"]["synchronized_hbm_complete"] is True


def test_gate_requires_exact_root7_bounded_workspace_contract():
    record = _passing_record()
    record["provider"]["high_rys_workspace"]["nbytes"] += 8

    decision = GATE.evaluate_gate(record)

    failed = {item["id"] for item in decision["checks"] if not item["passed"]}
    assert "bounded_selected_provider" in failed


def test_gate_requires_fail_closed_selected_provider_stream_contract():
    record = _passing_record()
    record["provider"]["stream_safety"]["cross_stream_behavior"] = "allow"

    decision = GATE.evaluate_gate(record)

    failed = {item["id"] for item in decision["checks"] if not item["passed"]}
    assert "selected_provider_stream_safety" in failed


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
        if item["id"] == "performance_transfer_qualification"
    )
    assert release["scope"] == "qualification"
    assert release["passed"] is False
    assert decision["qualification_passed"] is False


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
    source_inventory = NORMALIZER.inspect_source(staging)
    source_normalization = NORMALIZER.materialize_source_links(
        staging, source_inventory
    )
    uploaded_normalization = (
        json.dumps(source_normalization, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    normalized = NORMALIZER.assert_normalized_source(staging)
    source_normalization.update({
        "remote_verified_after_runtime_binding": True,
        "remote_regular_file_count_after_runtime_binding": normalized[
            "regular_file_count_after"
        ],
        "remote_directory_count_after_runtime_binding": normalized[
            "directory_count_after"
        ],
        "uploaded_record_sha256": hashlib.sha256(
            uploaded_normalization
        ).hexdigest(),
        "uploaded_record_bytes": len(uploaded_normalization),
    })
    digest, source_file_count = GATE.source_tree_digest(staging)
    snapshot_root = task / "snapshots" / digest
    source = snapshot_root / "source"
    staging.parent.rename(snapshot_root)
    binaries = [
        source / "gpu4pyscf" / "lib" / name
        for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
    ]
    manifest = snapshot_root / "manifest.json"
    runtime_files = [{
        "relative_path": f"gpu4pyscf/lib/{item.name}",
        "sha256": GATE._sha256(item),
        "bytes": item.stat().st_size,
    } for item in binaries]
    manifest.write_text(json.dumps({
        "schema": SNAPSHOT.SOURCE_SNAPSHOT_SCHEMA,
        "tree_sha256": digest,
        "base_revision": GATE.FROZEN_BASE_COMMIT,
        "source": str(source.resolve()),
        "immutable": True,
        "source_normalization": source_normalization,
        "local_provenance": _local_provenance(),
        "publication_policy": SNAPSHOT.PUBLICATION_POLICY,
        "runtime_binaries": _candidate_runtime_manifest(
            source_sha256=digest,
            source_file_count=source_file_count,
            task_root=str(task),
            files=runtime_files,
        ),
    }))
    manifest_bytes = manifest.read_bytes()
    attestation = {
        "schema": SNAPSHOT.PUBLICATION_ATTESTATION_SCHEMA,
        "status": "published",
        "source_tree_sha256": digest,
        "snapshot_name": digest,
        "published_directory_identity": {
            "device": snapshot_root.stat().st_dev,
            "inode": snapshot_root.stat().st_ino,
        },
        "publish_protocol": "atomic-exclusive-rename",
        "trust_boundary": SNAPSHOT.PUBLISH_TRUST_BOUNDARY,
        "publication_policy_sha256": SNAPSHOT._canonical_json_sha256(
            SNAPSHOT.PUBLICATION_POLICY
        ),
        "published_manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "prepublication_manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
    }
    attestation["attestation_sha256"] = SNAPSHOT._canonical_json_sha256(
        attestation
    )
    attestation_path = snapshot_root.parent / (
        f"{digest}{SNAPSHOT.PUBLICATION_ATTESTATION_SUFFIX}"
    )
    attestation_path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n"
    )
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
    attestation_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
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
        attestation_path.chmod(0o644)
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
    assert 'RESULT_ROOT="${TASK_ROOT}/results/gint-gate"' in text
    assert "GINT_GATE_RESULT_ROOT must be the fixed path" in text
    assert "#SBATCH --nodelist=compute-1-6" in text
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
