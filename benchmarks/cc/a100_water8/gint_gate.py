#!/usr/bin/env python3
"""Fail-closed WATER27 water2 gate for selected-pair GINT/direct Cholesky.

The timed boundary starts before construction of the GINT column provider and
ends after blocked matrix-free pivoted Cholesky has converged and the GPU
device has synchronized.  Batches of 1, 8, and 32 selected columns are then
checked against a bounded rectangular PySCF AO2MO oracle; the direct packed
diagonal is checked against CPU shell quartets.  No ``nao**4`` tensor or
square AO-pair matrix is built.

This gate has no speed target.  It decides whether one measured direct-CD run
has complete, stable, numerically correct evidence on the frozen MTU hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[2]
CASE_FILE = ROOT / "cases.json"

try:
    from .snapshot_manifest import (
        CANDIDATE_DEPLOYMENT_PROFILE,
        FROZEN_BASE_COMMIT,
        G0_DEPLOYMENT_PROFILE,
        SOURCE_SNAPSHOT_SCHEMA,
        source_tree_digest,
        validate_snapshot,
    )
except ImportError:  # direct execution and importlib-based unit tests
    _snapshot_spec = importlib.util.spec_from_file_location(
        "water2_gint_snapshot_manifest", ROOT / "snapshot_manifest.py"
    )
    if _snapshot_spec is None or _snapshot_spec.loader is None:
        raise RuntimeError("cannot load snapshot_manifest.py")
    _snapshot_module = importlib.util.module_from_spec(_snapshot_spec)
    _snapshot_spec.loader.exec_module(_snapshot_module)
    CANDIDATE_DEPLOYMENT_PROFILE = (
        _snapshot_module.CANDIDATE_DEPLOYMENT_PROFILE
    )
    FROZEN_BASE_COMMIT = _snapshot_module.FROZEN_BASE_COMMIT
    G0_DEPLOYMENT_PROFILE = _snapshot_module.G0_DEPLOYMENT_PROFILE
    SOURCE_SNAPSHOT_SCHEMA = _snapshot_module.SOURCE_SNAPSHOT_SCHEMA
    source_tree_digest = _snapshot_module.source_tree_digest
    validate_snapshot = _snapshot_module.validate_snapshot

CASE_ID = "water2-tz"
ERI_THRESHOLDS = (1.0e-4, 1.0e-6, 1.0e-8)
HBM_LIMIT_GIB = 72.0
RSS_LIMIT_GIB = 110.0
EXPECTED_GPU = "NVIDIA A100-SXM4-80GB"
EXPECTED_CPU_FRAGMENT = "AMD EPYC 7513"
EXPECTED_NUMA_NODE = 3
EXPECTED_PCIE_SPEED = "16.0 GT/s"
EXPECTED_PCIE_WIDTH = "16"
EXPECTED_WATER2_GEOMETRY_SHA256 = (
    "d392c0d09b349aa8044d7d652d8ca073bee8a77111db973fdcaf297b13dfedc3"
)
SELECTED_COLUMN_BATCH_SIZES = (1, 8, 32)
SELECTED_PROVIDER_FACTORIZATION = "gint-selected-shell-pair-columns"
SELECTED_PROVIDER_BACKEND = "cupy-gint-selected-cabi"
SELECTED_COLUMNS_SYMBOL = "GINTfill_selected_int2e_columns"
SELECTED_DIAGONAL_SYMBOL = "GINTfill_selected_int2e_diagonal"
SELECTED_WORKSPACE_SIZE_SYMBOL = "GINTselected_workspace_size"
SELECTED_OUTPUT_LAYOUT = "batch-by-packed-lower-triangle"
SELECTED_RELEASE_SCOPE = "spherical-original-aos-with-single-shell-support"
SELECTED_STREAM_SAFETY_STRATEGY = (
    "process-wide-per-device-single-stream-with-host-enqueue-lock"
)
SELECTED_STREAM_SAFETY_POLICY_VERSION = 1
SYNCHRONIZED_HBM_CHECKPOINT_NAMES = (
    "before-provider-setup",
    "after-provider-setup",
    "after-direct-cd",
)
EXPECTED_A100_MULTIPROCESSORS = 108
SELECTED_HIGH_RYS_THREADS = 32
RYS7_GOUT_DOUBLES = 21_720
EXPECTED_RYS7_WORKSPACE_BYTES = (
    EXPECTED_A100_MULTIPROCESSORS
    * SELECTED_HIGH_RYS_THREADS
    * RYS7_GOUT_DOUBLES
    * 8
)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_task_root() -> Path | None:
    value = os.getenv("CCSD_TASK_ROOT")
    return None if not value else Path(value).expanduser().resolve()


def _expected_deployment_profile() -> str | None:
    return os.getenv("CCSD_EXPECTED_DEPLOYMENT_PROFILE") or None


def _require_canonical_pristine() -> bool:
    return os.getenv(
        "CCSD_REQUIRE_CANONICAL_PRISTINE", ""
    ).strip().lower() in {"1", "true", "yes"}


def snapshot_evidence(
    root: Path,
    *,
    tree_sha256: str,
    base_revision: str,
) -> dict[str, Any]:
    """Capture shared v2 snapshot validation evidence at run start."""

    expected_profile = _expected_deployment_profile()
    require_canonical = _require_canonical_pristine()
    evidence = validate_snapshot(
        root,
        expected_base_revision=base_revision,
        expected_task_root=_expected_task_root(),
        expected_deployment_profile=expected_profile,
        require_canonical_pristine=require_canonical,
    )
    if expected_profile is None:
        evidence["valid"] = False
        evidence.setdefault("reasons", []).append(
            "expected deployment profile is not configured"
        )
    if evidence.get("tree_sha256") != tree_sha256:
        evidence["valid"] = False
        evidence.setdefault("reasons", []).append(
            "captured source digest differs from snapshot validation"
        )
    manifest_path = evidence.get("manifest_path")
    try:
        manifest_sha256 = _sha256(Path(str(manifest_path)))
    except Exception as exc:
        manifest_sha256 = None
        evidence["manifest_hash_error_at_start"] = repr(exc)
    evidence["valid_at_start"] = evidence.get("valid") is True
    evidence["read_only_at_start"] = evidence.get("read_only") is True
    evidence["manifest_sha256_at_start"] = manifest_sha256
    task_root = _expected_task_root()
    evidence["expected_task_root"] = (
        None if task_root is None else str(task_root)
    )
    return evidence


def source_evidence(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    digest, count = source_tree_digest(root)
    base_revision = os.getenv(
        "FROZEN_GPU4PYSCF_COMMIT", FROZEN_BASE_COMMIT
    )
    evidence: dict[str, Any] = {
        "root": str(root.resolve()),
        "tree_sha256_at_start": digest,
        "files_hashed_at_start": count,
        "base_revision": base_revision,
        "snapshot": snapshot_evidence(
            root,
            tree_sha256=digest,
            base_revision=base_revision,
        ),
    }
    try:
        evidence["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        evidence["git_dirty"] = bool(status)
        evidence["git_dirty_paths"] = [line[3:] for line in status.splitlines()]
    except Exception as exc:
        evidence["git_head"] = None
        evidence["git_dirty"] = None
        evidence["git_error"] = repr(exc)
    return evidence


def finish_source_evidence(
    evidence: dict[str, Any], root: Path = REPOSITORY_ROOT
) -> bool:
    try:
        digest, count = source_tree_digest(root)
    except Exception as exc:
        evidence.update({
            "tree_sha256_at_end": None,
            "files_hashed_at_end": None,
            "stable_during_run": None,
            "stability_error": repr(exc),
        })
        return False
    tree_stable = digest == evidence.get("tree_sha256_at_start")
    snapshot = evidence.setdefault("snapshot", {})
    try:
        end_evidence = validate_snapshot(
            root,
            expected_base_revision=evidence.get(
                "base_revision", FROZEN_BASE_COMMIT
            ),
            expected_task_root=_expected_task_root(),
            expected_deployment_profile=snapshot.get(
                "expected_deployment_profile"
            ),
            require_canonical_pristine=bool(
                snapshot.get("canonical_pristine_required")
            ),
        )
    except Exception as exc:
        end_evidence = {
            "valid": False,
            "read_only": False,
            "tree_sha256": None,
            "files_hashed": None,
            "reasons": [f"snapshot validation at end failed: {exc!r}"],
        }
    manifest_path = snapshot.get("manifest_path")
    manifest_at_start = snapshot.get("manifest_sha256_at_start")
    try:
        manifest_at_end = _sha256(Path(manifest_path))
    except Exception as exc:
        manifest_at_end = None
        snapshot["stability_error"] = repr(exc)
    manifest_stable = bool(
        manifest_at_start is not None and manifest_at_start == manifest_at_end
    )
    snapshot_stable = bool(
        tree_stable
        and manifest_stable
        and snapshot.get("valid_at_start") is True
        and end_evidence.get("valid") is True
    )
    snapshot.update({
        "valid_at_end": end_evidence.get("valid") is True,
        "read_only_at_end": end_evidence.get("read_only") is True,
        "tree_sha256_at_end": end_evidence.get("tree_sha256"),
        "files_hashed_at_end": end_evidence.get("files_hashed"),
        "manifest_sha256_at_end": manifest_at_end,
        "stable_during_run": snapshot_stable,
        "reasons_at_end": end_evidence.get("reasons", []),
    })
    evidence.update({
        "tree_sha256_at_end": digest,
        "files_hashed_at_end": count,
        "stable_during_run": snapshot_stable,
    })
    return snapshot_stable


def _command(arguments: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(
            list(arguments), text=True, stderr=subprocess.DEVNULL, timeout=15
        ).strip()
    except Exception:
        return None


def topology_evidence(
    path: str | Path | None,
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load and fail-closed validate the topology-guard sidecar."""

    env = dict(os.environ if environment is None else environment)
    reasons: list[str] = []
    if path is None:
        path = env.get("CCSD_TOPOLOGY_RECORD")
    if not path:
        return {
            "available": False,
            "passed": False,
            "reasons": ["topology record path is missing"],
        }
    record_path = Path(path).expanduser().resolve()
    if not record_path.is_file():
        return {
            "available": False,
            "path": str(record_path),
            "passed": False,
            "reasons": ["topology record does not exist"],
        }
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "available": True,
            "path": str(record_path),
            "sha256": _sha256(record_path),
            "passed": False,
            "reasons": [f"topology record cannot be decoded: {exc!r}"],
        }

    if payload.get("performance_eligible") is not True:
        reasons.append("topology guard did not mark the launch performance eligible")
    if payload.get("benchmark_track") != "performance":
        reasons.append("topology benchmark track is not performance")
    if payload.get("gpu_numa_node") != EXPECTED_NUMA_NODE:
        reasons.append("visible GPU is not attached to NUMA node 3")
    if payload.get("selected_cpulist") != payload.get(
        "observed_cpulist_after_bind"
    ):
        reasons.append("observed CPU affinity differs from the selected cpulist")
    policy = payload.get("kernel_memory_policy") or {}
    after = policy.get("after") or {}
    if not policy.get("verified"):
        reasons.append("kernel preferred-memory policy is unverified")
    if after.get("mode_name") != "preferred" or after.get("nodes") != [3]:
        reasons.append("kernel memory policy is not preferred NUMA node 3")

    current_job = env.get("SLURM_JOB_ID")
    record_job = (payload.get("slurm") or {}).get("SLURM_JOB_ID")
    if not current_job:
        reasons.append("current Slurm job id is unavailable")
    elif current_job != record_job:
        reasons.append("topology record belongs to a different Slurm job")
    if env.get("CCSD_PERFORMANCE_ELIGIBLE", "").lower() != "true":
        reasons.append("topology guard performance environment is not true")
    environment_path = env.get("CCSD_TOPOLOGY_RECORD")
    if environment_path and Path(environment_path).expanduser().resolve() != record_path:
        reasons.append("explicit topology path differs from the guard environment")
    command_names = [Path(str(item)).name for item in payload.get("command", [])]
    if "gint_gate.py" not in command_names:
        reasons.append("topology guard did not launch gint_gate.py")

    topology_sha256 = _sha256(record_path)
    return {
        "available": True,
        "path": str(record_path),
        "sha256": topology_sha256,
        "sha256_at_start": topology_sha256,
        "stable_during_run": None,
        "record": payload,
        "current_slurm_job_id": current_job,
        "record_slurm_job_id": record_job,
        "passed": not reasons,
        "reasons": reasons,
    }


def finish_topology_evidence(evidence: dict[str, Any]) -> bool:
    """Verify that the topology sidecar did not change during the gate."""

    path = evidence.get("path")
    start = evidence.get("sha256_at_start") or evidence.get("sha256")
    if not path or not start:
        evidence["sha256_at_end"] = None
        evidence["stable_during_run"] = False
        return False
    try:
        end = _sha256(Path(path))
    except Exception as exc:
        evidence["sha256_at_end"] = None
        evidence["stable_during_run"] = False
        evidence["stability_error"] = repr(exc)
        return False
    stable = end == start
    evidence["sha256_at_end"] = end
    evidence["stable_during_run"] = stable
    if not stable:
        evidence["passed"] = False
        evidence.setdefault("reasons", []).append(
            "topology record changed during the gate"
        )
    return stable


def _normalise_pci_bus_id(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("00000000:"):
        return "0000:" + value.split(":", 1)[1]
    return value


def hardware_evidence() -> dict[str, Any]:
    """Capture and validate the frozen A100/EPYC/PCIe hardware contract."""

    reasons: list[str] = []
    query = _command([
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version,pci.bus_id",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    if query:
        for line in query.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5:
                rows.append(dict(zip(
                    ("name", "uuid", "memory_total_mib", "driver", "pci_bus_id"),
                    fields,
                )))
    if len(rows) != 1:
        reasons.append(f"expected one visible GPU, observed {len(rows)}")

    pci: dict[str, Any] = {}
    if len(rows) == 1:
        gpu = rows[0]
        if gpu["name"] != EXPECTED_GPU:
            reasons.append(f"GPU model is {gpu['name']!r}, expected {EXPECTED_GPU!r}")
        try:
            if float(gpu["memory_total_mib"]) < 80000:
                reasons.append("visible A100 reports less than 80000 MiB")
        except ValueError:
            reasons.append("GPU memory size is not numeric")
        bus = _normalise_pci_bus_id(gpu["pci_bus_id"])
        device = Path("/sys/bus/pci/devices") / bus
        pci = {"sysfs_bus_id": bus}
        for name in (
            "current_link_speed", "current_link_width", "max_link_speed",
            "max_link_width", "numa_node",
        ):
            try:
                pci[name] = (device / name).read_text(encoding="utf-8").strip()
            except OSError:
                pci[name] = None
        if not str(pci.get("max_link_speed") or "").startswith(
            EXPECTED_PCIE_SPEED
        ):
            reasons.append("GPU maximum PCIe link speed is not 16.0 GT/s (Gen4)")
        if pci.get("max_link_width") != EXPECTED_PCIE_WIDTH:
            reasons.append("GPU maximum PCIe link width is not x16")
        if pci.get("numa_node") != str(EXPECTED_NUMA_NODE):
            reasons.append("GPU sysfs NUMA node is not 3")

    lscpu = _command(["lscpu", "-J"])
    cpu_model = None
    if lscpu:
        try:
            for item in json.loads(lscpu).get("lscpu", []):
                if str(item.get("field", "")).rstrip(":") == "Model name":
                    cpu_model = str(item.get("data"))
                    break
        except Exception:
            pass
    cpu_model = cpu_model or platform.processor() or None
    if not cpu_model or EXPECTED_CPU_FRAGMENT not in cpu_model:
        reasons.append("CPU model is not AMD EPYC 7513")
    return {
        "host": socket.gethostname(),
        "visible_gpus": rows,
        "pci": pci,
        "cpu_model": cpu_model,
        "affinity": sorted(os.sched_getaffinity(0)),
        "passed": not reasons,
        "reasons": reasons,
    }


class HBMMonitor:
    """Sample process HBM through nvidia-smi during the timed boundary."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = float(interval_seconds)
        self.samples_mib: list[int] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        pid = str(os.getpid())
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=4,
                )
                for line in output.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 2 and fields[0] == pid:
                        self.samples_mib.append(int(fields[1]))
            except Exception as exc:
                self.error = repr(exc)
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def metadata(self) -> dict[str, Any]:
        peak = max(self.samples_mib, default=None)
        return {
            "backend": "nvidia-smi-process-sampling",
            "interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples_mib),
            "peak_process_mib": peak,
            "peak_process_gib": None if peak is None else peak / 1024.0,
            "error": self.error,
        }


def synchronized_hbm_checkpoint(cupy: Any, name: str) -> dict[str, Any]:
    """Measure device-global and CuPy-pool HBM at a device-wide boundary."""

    cupy.cuda.runtime.deviceSynchronize()
    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    pool = cupy.get_default_memory_pool()
    return {
        "name": str(name),
        "driver_used_bytes": int(total_bytes) - int(free_bytes),
        "driver_total_bytes": int(total_bytes),
        "cupy_pool_used_bytes": int(pool.used_bytes()),
        "cupy_pool_total_bytes": int(pool.total_bytes()),
        "timing_semantics": "cuda-device-synchronized-instantaneous",
        "measurement_scope": "exclusive-slurm-device-global",
        "source": "cudaMemGetInfo-and-cupy-default-memory-pool",
    }


def synchronized_hbm_evidence(
    checkpoints: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Serialize the complete named HBM checkpoint ledger."""

    peaks = [int(item["driver_used_bytes"]) for item in checkpoints]
    peak = max(peaks, default=None)
    return {
        "schema": "gpu4pyscf.gint-synchronized-hbm.v1",
        "checkpoints": list(checkpoints),
        "peak_driver_used_bytes": peak,
        "peak_driver_used_gib": (
            None if peak is None else peak / float(1024**3)
        ),
        "budget_gib": HBM_LIMIT_GIB,
    }


def _host_peak_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0**3)
    return value / (1024.0**2)


def load_water2_case() -> dict[str, Any]:
    payload = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    matches = [case for case in payload["cases"] if case["id"] == CASE_ID]
    if len(matches) != 1:
        raise RuntimeError("cases.json does not contain exactly one water2-tz")
    case = matches[0]
    if case.get("source") != "WATER27:H2O2" or case.get("basis") != "cc-pVTZ":
        raise RuntimeError("water2 gate case contract changed")
    if case.get("expected_dimensions") != {"nao": 116, "nocc": 10, "nvir": 106}:
        raise RuntimeError("water2 expected dimensions changed")
    canonical = json.dumps(
        case["geometry_angstrom"], separators=(",", ":"), ensure_ascii=True
    )
    result = dict(case)
    result["geometry_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    if result["geometry_sha256"] != EXPECTED_WATER2_GEOMETRY_SHA256:
        raise RuntimeError("water2 frozen geometry hash changed")
    result["cases_file_sha256"] = _sha256(CASE_FILE)
    return result


def _shell_and_local(ao_locs: Any, ao: int) -> tuple[int, int]:
    import numpy as np

    shell = int(np.searchsorted(ao_locs, ao, side="right") - 1)
    return shell, int(ao - ao_locs[shell])


def shell_sliced_column(mol: Any, pair_map: Any, pivot: int):
    """Build one CPU AO-pair column from bounded shell quartets."""

    import numpy as np

    first, second = pair_map.pair(pivot)
    ao_locs = np.asarray(mol.ao_loc_nr(), dtype=np.int64)
    shell_k, local_k = _shell_and_local(ao_locs, first)
    shell_l, local_l = _shell_and_local(ao_locs, second)
    result = np.empty(pair_map.dimension, dtype=np.float64)
    for shell_i in range(mol.nbas):
        for shell_j in range(shell_i + 1):
            block = mol.intor_by_shell(
                "int2e_sph", (shell_i, shell_j, shell_k, shell_l)
            )
            for ao_i in range(ao_locs[shell_i], ao_locs[shell_i + 1]):
                for ao_j in range(ao_locs[shell_j], ao_locs[shell_j + 1]):
                    if ao_i < ao_j:
                        continue
                    result[pair_map.index(ao_i, ao_j)] = block[
                        ao_i - ao_locs[shell_i],
                        ao_j - ao_locs[shell_j],
                        local_k,
                        local_l,
                    ]
    return result


def shell_sliced_diagonal(mol: Any, pair_map: Any):
    """Build the CPU AO-pair diagonal from bounded shell quartets."""

    import numpy as np

    ao_locs = np.asarray(mol.ao_loc_nr(), dtype=np.int64)
    result = np.empty(pair_map.dimension, dtype=np.float64)
    for shell_i in range(mol.nbas):
        for shell_j in range(shell_i + 1):
            block = mol.intor_by_shell(
                "int2e_sph", (shell_i, shell_j, shell_i, shell_j)
            )
            for ao_i in range(ao_locs[shell_i], ao_locs[shell_i + 1]):
                local_i = ao_i - ao_locs[shell_i]
                for ao_j in range(ao_locs[shell_j], ao_locs[shell_j + 1]):
                    if ao_i < ao_j:
                        continue
                    local_j = ao_j - ao_locs[shell_j]
                    result[pair_map.index(ao_i, ao_j)] = block[
                        local_i, local_j, local_i, local_j
                    ]
    return result


def selected_validation_pivots(
    dimension: int,
    *,
    maximum_batch_size: int = max(SELECTED_COLUMN_BATCH_SIZES),
) -> tuple[int, ...]:
    """Choose deterministic, distinct AO-pair pivots across the full range."""

    import numpy as np

    dimension = int(dimension)
    maximum_batch_size = int(maximum_batch_size)
    if dimension < maximum_batch_size or maximum_batch_size < 1:
        raise ValueError("pair dimension must cover the validation batch")
    pivots = np.linspace(
        0, dimension - 1, maximum_batch_size, dtype=np.int64
    )
    if np.unique(pivots).size != maximum_batch_size:
        raise RuntimeError("validation pivot selection contains duplicates")
    return tuple(int(item) for item in pivots)


def pivots_for_batch(
    master_pivots: Sequence[int], batch_size: int
) -> tuple[int, ...]:
    """Select a spread-out nested subset for one required API batch size."""

    import numpy as np

    master = tuple(int(item) for item in master_pivots)
    batch_size = int(batch_size)
    if batch_size < 1 or batch_size > len(master):
        raise ValueError("batch_size must lie within the master pivot set")
    positions = np.linspace(0, len(master) - 1, batch_size, dtype=np.int64)
    result = tuple(master[int(position)] for position in positions)
    if len(set(result)) != batch_size:
        raise RuntimeError("batch pivot selection contains duplicates")
    return result


def ao2mo_selected_columns(
    mol: Any,
    pair_map: Any,
    pivots: Sequence[int],
    *,
    max_working_bytes: int,
    ao2mo_module: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Build selected packed columns with a bounded rectangular AO2MO.

    The last two coefficient matrices each contain ``B`` unit vectors.  PySCF
    therefore materializes ``nao**2 * B**2`` transformed elements.  Extracting
    the paired diagonal of those last two axes yields exactly the requested
    ``B`` ERI columns.  The allocation is checked before calling AO2MO and can
    never become a square AO-pair matrix or a four-index AO tensor.
    """

    import numpy as np

    pivots = tuple(int(item) for item in pivots)
    if not pivots:
        raise ValueError("AO2MO oracle requires at least one pivot")
    dimension = int(pair_map.dimension)
    if any(pivot < 0 or pivot >= dimension for pivot in pivots):
        raise IndexError("AO2MO oracle pivot is outside the packed pair range")
    if len(set(pivots)) != len(pivots):
        raise ValueError("AO2MO oracle pivots must be distinct")
    nao = int(mol.nao_nr())
    batch_size = len(pivots)
    dtype = np.dtype(np.float64)
    transformed_shape = (nao * nao, batch_size * batch_size)
    transformed_elements = int(np.prod(transformed_shape, dtype=object))
    transformed_bytes = transformed_elements * dtype.itemsize
    max_working_bytes = int(max_working_bytes)
    if max_working_bytes < dtype.itemsize:
        raise ValueError("AO2MO working-byte limit must hold one FP64 value")
    if transformed_bytes > max_working_bytes:
        raise MemoryError(
            "bounded AO2MO oracle requires "
            f"{transformed_bytes} bytes, exceeding max_working_bytes="
            f"{max_working_bytes}"
        )

    if ao2mo_module is None:
        from pyscf import ao2mo as ao2mo_module

    identity = np.eye(nao, dtype=dtype)
    pair_i = np.asarray(pair_map.pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_map.pair_j, dtype=np.int64)
    selected_i = identity[:, pair_i[list(pivots)]]
    selected_j = identity[:, pair_j[list(pivots)]]
    transformed = np.asarray(
        ao2mo_module.general(
            mol,
            (identity, identity, selected_i, selected_j),
            compact=False,
        )
    )
    if transformed.size != transformed_elements:
        raise RuntimeError(
            "PySCF AO2MO returned an unexpected rectangular allocation"
        )
    if transformed.dtype != dtype:
        raise TypeError("PySCF AO2MO oracle must return FP64 values")
    transformed = transformed.reshape(nao, nao, batch_size, batch_size)
    paired = transformed[
        :, :, np.arange(batch_size), np.arange(batch_size)
    ]
    columns = np.ascontiguousarray(paired[pair_i, pair_j, :].T)
    if columns.shape != (batch_size, dimension):
        raise RuntimeError("AO2MO oracle returned an invalid packed shape")
    return columns, {
        "backend": "pyscf.ao2mo.general",
        "compact": False,
        "precision": "fp64",
        "batch_size": batch_size,
        "transformed_shape": list(transformed_shape),
        "transformed_elements": transformed_elements,
        "transformed_bytes": transformed_bytes,
        "max_working_bytes": max_working_bytes,
        "within_working_limit": transformed_bytes <= max_working_bytes,
        "materialized_nao4": False,
        "materialized_pair_matrix": False,
    }


def _snapshot_evidence_valid(source: dict[str, Any]) -> bool:
    """Validate captured v3 normalized-snapshot evidence without I/O."""

    digest = source.get("tree_sha256_at_start")
    snapshot = source.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    manifest = snapshot.get("manifest")
    if not isinstance(manifest, dict):
        return False
    runtime = manifest.get("runtime_binaries")
    files = runtime.get("files") if isinstance(runtime, dict) else None
    manifest_start = snapshot.get("manifest_sha256_at_start")
    snapshot_root = snapshot.get("snapshot_root")
    source_root = source.get("root")
    return bool(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        and source.get("tree_sha256_at_end") == digest
        and source.get("stable_during_run") is True
        and source.get("base_revision") == FROZEN_BASE_COMMIT
        and snapshot.get("schema") == "gpu4pyscf.snapshot-evidence.v1"
        and snapshot.get("valid") is True
        and snapshot.get("valid_at_start") is True
        and snapshot.get("valid_at_end") is True
        and snapshot.get("read_only") is True
        and snapshot.get("read_only_at_start") is True
        and snapshot.get("read_only_at_end") is True
        and snapshot.get("writable_paths") == []
        and snapshot.get("reasons") == []
        and snapshot.get("reasons_at_end") == []
        and snapshot.get("stable_during_run") is True
        and snapshot.get("tree_sha256") == digest
        and snapshot.get("tree_sha256_at_end") == digest
        and isinstance(manifest_start, str)
        and len(manifest_start) == 64
        and manifest_start == snapshot.get("manifest_sha256_at_end")
        and manifest.get("schema") == SOURCE_SNAPSHOT_SCHEMA
        and isinstance(manifest.get("source_normalization"), dict)
        and manifest["source_normalization"].get("published_symlink_count") == 0
        and manifest["source_normalization"].get(
            "published_special_file_count"
        ) == 0
        and manifest["source_normalization"].get(
            "remote_verified_after_runtime_binding"
        ) is True
        and manifest.get("tree_sha256") == digest
        and manifest.get("base_revision") == FROZEN_BASE_COMMIT
        and manifest.get("source") == source_root
        and manifest.get("immutable") is True
        and snapshot.get("expected_deployment_profile")
        in {CANDIDATE_DEPLOYMENT_PROFILE, G0_DEPLOYMENT_PROFILE}
        and isinstance(snapshot.get("canonical_pristine_required"), bool)
        and isinstance(manifest.get("local_provenance"), dict)
        and manifest["local_provenance"].get("deployment_profile")
        == snapshot.get("expected_deployment_profile")
        and (
            snapshot.get("expected_deployment_profile")
            != G0_DEPLOYMENT_PROFILE
            or snapshot.get("canonical_pristine_required") is True
            and manifest["local_provenance"].get("canonical_pristine") is True
        )
        and isinstance(runtime, dict)
        and runtime.get("complete") is True
        and isinstance(files, list)
        and bool(files)
        and snapshot.get("source") == source_root
        and isinstance(snapshot_root, str)
        and Path(snapshot_root).parent.name == "snapshots"
        and Path(snapshot_root).name == digest
        and snapshot.get("manifest_path")
        == str(Path(snapshot_root) / "manifest.json")
        and Path(str(source_root)).name == "source"
        and Path(str(source_root)).parent == Path(snapshot_root)
    )


def evaluate_gate(record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate correctness and the stricter performance-release contract."""

    checks: list[dict[str, Any]] = []

    def check(
        identifier: str,
        passed: bool,
        reason: str,
        evidence: Any,
        *,
        scope: str = "correctness",
    ) -> None:
        if scope not in {"correctness", "qualification", "performance"}:
            raise ValueError(
                "gate check scope must be correctness, qualification, or performance"
            )
        checks.append({
            "id": identifier,
            "scope": scope,
            "passed": bool(passed),
            "reason_if_failed": None if passed else reason,
            "evidence": evidence,
        })

    source = record.get("source", {})
    snapshot = source.get("snapshot") or {}
    check(
        "immutable_source",
        _snapshot_evidence_valid(source),
        (
            "source is not a stable, read-only content-addressed snapshot "
            "of the frozen GPU4PySCF revision"
        ),
        source,
    )
    topology = record.get("topology", {})
    check(
        "mtu_topology",
        topology.get("passed") is True
        and topology.get("stable_during_run") is True,
        "MTU topology guard evidence is missing or failed",
        topology.get("reasons"),
    )
    hardware = record.get("hardware", {})
    check(
        "fixed_hardware",
        hardware.get("passed") is True,
        "A100/EPYC/PCIe hardware contract failed",
        hardware.get("reasons"),
    )
    case = record.get("case", {})
    check(
        "water2_case",
        case.get("id") == CASE_ID
        and case.get("basis") == "cc-pVTZ"
        and case.get("actual_nao") == 116
        and case.get("geometry_sha256") == EXPECTED_WATER2_GEOMETRY_SHA256,
        "case is not frozen WATER27 water2/cc-pVTZ with nao=116",
        {
            key: case.get(key)
            for key in ("id", "basis", "actual_nao", "geometry_sha256")
        },
    )
    provider = record.get("provider", {})
    schedule = provider.get("schedule") or {}
    c_abi = provider.get("c_abi") or {}
    high_rys_workspace = provider.get("high_rys_workspace") or {}
    stream_safety = provider.get("stream_safety") or {}
    expected_dimension = 116 * 117 // 2
    check(
        "selected_provider_identity",
        provider.get("factorization") == SELECTED_PROVIDER_FACTORIZATION
        and provider.get("backend") == SELECTED_PROVIDER_BACKEND
        and provider.get("dimension") == expected_dimension
        and provider.get("nao") == 116
        and provider.get("precision") == "fp64"
        and provider.get("output_residency") == "gpu"
        and provider.get("output_layout") == SELECTED_OUTPUT_LAYOUT
        and provider.get("max_batch_size")
        == max(SELECTED_COLUMN_BATCH_SIZES)
        and provider.get("maximum_batch_observed")
        == max(SELECTED_COLUMN_BATCH_SIZES)
        and isinstance(provider.get("diagonal_calls"), int)
        and provider["diagonal_calls"] >= 1
        and isinstance(provider.get("cabi_calls"), int)
        and provider["cabi_calls"] > 0
        and isinstance(provider.get("kernel_launches"), int)
        and provider["kernel_launches"] > 0
        and c_abi.get("columns") == SELECTED_COLUMNS_SYMBOL
        and c_abi.get("diagonal") == SELECTED_DIAGONAL_SYMBOL
        and c_abi.get("workspace_size") == SELECTED_WORKSPACE_SIZE_SYMBOL,
        "run did not use the FP64 selected-pair C-ABI provider",
        {
            key: provider.get(key)
            for key in (
                "factorization", "backend", "dimension", "nao", "precision",
                "output_residency", "output_layout", "max_batch_size",
                "maximum_batch_observed", "diagonal_calls", "cabi_calls",
                "kernel_launches",
            )
        } | {"c_abi": c_abi},
    )
    check(
        "bounded_selected_provider",
        provider.get("materializes_pair_matrix") is False
        and provider.get("materializes_four_index_tensor") is False
        and provider.get("materializes_per_pivot_ao_matrix") is False
        and provider.get("diag_block_with_triu") is True
        and provider.get("device_schedule") is True
        and schedule.get("single_shell_support") is True
        and schedule.get("npair") == expected_dimension
        and schedule.get("nao_original") == 116
        and provider.get("release_scope") == SELECTED_RELEASE_SCOPE
        and provider.get("maximum_rys_order") == 7
        and high_rys_workspace.get("strategy")
            == "provider-owned-bounded-global-grid-stride"
        and high_rys_workspace.get("minimum_rys_order") == 7
        and high_rys_workspace.get("maximum_rys_order") == 7
        and high_rys_workspace.get("threads_per_block")
            == SELECTED_HIGH_RYS_THREADS
        and high_rys_workspace.get("multiprocessor_count")
            == EXPECTED_A100_MULTIPROCESSORS
        and high_rys_workspace.get("slot_count")
            == EXPECTED_A100_MULTIPROCESSORS * SELECTED_HIGH_RYS_THREADS
        and high_rys_workspace.get("nbytes")
            == EXPECTED_RYS7_WORKSPACE_BYTES
        and high_rys_workspace.get("resident") is True
        and high_rys_workspace.get("static_thread_gout_eliminated_for_orders")
            == [7, 8],
        "provider representation or release-scope invariant failed",
        {
            key: provider.get(key)
            for key in (
                "materializes_pair_matrix", "materializes_four_index_tensor",
                "materializes_per_pivot_ao_matrix", "diag_block_with_triu",
                "device_schedule", "release_scope",
            )
        } | {
            "schedule": schedule,
            "maximum_rys_order": provider.get("maximum_rys_order"),
            "high_rys_workspace": high_rys_workspace,
        },
    )
    check(
        "selected_provider_stream_safety",
        stream_safety.get("strategy") == SELECTED_STREAM_SAFETY_STRATEGY
        and stream_safety.get("policy_version")
            == SELECTED_STREAM_SAFETY_POLICY_VERSION
        and stream_safety.get("binding_scope")
            == "process-wide-per-cuda-device"
        and isinstance(stream_safety.get("construction_process_id"), int)
        and not isinstance(stream_safety.get("construction_process_id"), bool)
        and stream_safety.get("construction_device_id") == 0
        and isinstance(stream_safety.get("bound_stream_ptr"), int)
        and not isinstance(stream_safety.get("bound_stream_ptr"), bool)
        and stream_safety.get("bound_stream_ptr") >= 0
        and stream_safety.get("strong_stream_lifetime") is True
        and stream_safety.get("host_enqueue_lock_scope")
            == "entire-public-provider-call"
        and stream_safety.get("shared_constant_cache_ordering")
            == "bound-cuda-stream"
        and stream_safety.get("provider_workspace_ordering")
            == "bound-cuda-stream"
        and stream_safety.get("cross_stream_behavior")
            == "reject-before-device-enqueue"
        and stream_safety.get("per_call_device_synchronize") is False
        and isinstance(stream_safety.get("successful_call_count"), int)
        and not isinstance(stream_safety.get("successful_call_count"), bool)
        and stream_safety.get("successful_call_count") > 0,
        "selected provider did not prove the process-wide single-stream "
        "constant-cache/workspace safety contract",
        stream_safety,
    )
    thresholds = record.get("thresholds", {})
    numerical = record.get("numerical_validation", {})
    selected_batches = numerical.get("selected_column_batches")
    if not isinstance(selected_batches, list):
        selected_batches = []
    batch_by_size = {
        item["batch_size"]: item
        for item in selected_batches
        if isinstance(item, dict)
        and isinstance(item.get("batch_size"), int)
        and not isinstance(item.get("batch_size"), bool)
    }
    column_error = numerical.get("selected_column_max_abs_error")
    diagonal_error = numerical.get("diagonal_max_abs_error")
    batch_checks_complete = all(
        isinstance(batch_by_size.get(batch_size), dict)
        and batch_by_size[batch_size].get("shape")
        == [batch_size, expected_dimension]
        and batch_by_size[batch_size].get("dtype") == "float64"
        and batch_by_size[batch_size].get("gpu_resident") is True
        and batch_by_size[batch_size].get("output_nbytes")
        == batch_size * expected_dimension * 8
        and isinstance(batch_by_size[batch_size].get("max_abs_error"), (int, float))
        and batch_by_size[batch_size]["max_abs_error"]
        <= thresholds.get("selected_column_tolerance", -1)
        for batch_size in SELECTED_COLUMN_BATCH_SIZES
    )
    ao2mo_oracle = numerical.get("ao2mo_oracle") or {}
    check(
        "selected_batch_oracle",
        batch_checks_complete
        and len(selected_batches) == len(SELECTED_COLUMN_BATCH_SIZES)
        and set(batch_by_size) == set(SELECTED_COLUMN_BATCH_SIZES)
        and isinstance(column_error, (int, float))
        and column_error <= thresholds.get("selected_column_tolerance", -1)
        and ao2mo_oracle.get("backend") == "pyscf.ao2mo.general"
        and ao2mo_oracle.get("precision") == "fp64"
        and ao2mo_oracle.get("batch_size") == max(SELECTED_COLUMN_BATCH_SIZES)
        and ao2mo_oracle.get("within_working_limit") is True
        and ao2mo_oracle.get("materialized_nao4") is False
        and ao2mo_oracle.get("materialized_pair_matrix") is False
        and numerical.get("materialized_nao4") is False
        and numerical.get("materialized_pair_matrix") is False,
        "selected GINT B=1/8/32 columns failed the bounded AO2MO oracle",
        {
            "batches": selected_batches,
            "max_error": column_error,
            "threshold": thresholds.get("selected_column_tolerance"),
            "ao2mo_oracle": ao2mo_oracle,
        },
    )
    check(
        "direct_packed_diagonal_oracle",
        isinstance(diagonal_error, (int, float))
        and diagonal_error <= thresholds.get("diagonal_tolerance", -1)
        and (numerical.get("diagonal") or {}).get("shape")
        == [expected_dimension]
        and (numerical.get("diagonal") or {}).get("dtype") == "float64"
        and (numerical.get("diagonal") or {}).get("gpu_resident") is True
        and (numerical.get("diagonal") or {}).get("source")
        == SELECTED_DIAGONAL_SYMBOL,
        "direct selected-GINT packed diagonal exceeded its oracle tolerance",
        {
            "error": diagonal_error,
            "threshold": thresholds.get("diagonal_tolerance"),
            "validation": numerical.get("diagonal"),
        },
    )
    direct_cd = record.get("direct_cd", {})
    eri_tol = thresholds.get("eri_tol")
    check(
        "direct_cd_converged",
        isinstance(direct_cd.get("rank"), int)
        and direct_cd["rank"] > 0
        and direct_cd.get("capped") is False
        and isinstance(direct_cd.get("residual_diagonal"), (int, float))
        and isinstance(eri_tol, (int, float))
        and direct_cd["residual_diagonal"] <= eri_tol
        and direct_cd.get("provider_materializes_pair_matrix") is False
        and direct_cd.get("column_batch_size")
        == max(SELECTED_COLUMN_BATCH_SIZES)
        and direct_cd.get("requested_column_batch_size")
        == max(SELECTED_COLUMN_BATCH_SIZES)
        and isinstance(direct_cd.get("provider_batch_calls"), int)
        and direct_cd["provider_batch_calls"] > 0,
        "blocked direct Cholesky was capped, unbatched, or above eri_tol",
        {
            "rank": direct_cd.get("rank"),
            "capped": direct_cd.get("capped"),
            "residual_diagonal": direct_cd.get("residual_diagonal"),
            "eri_tol": eri_tol,
            "column_batch_size": direct_cd.get("column_batch_size"),
            "requested_column_batch_size": direct_cd.get(
                "requested_column_batch_size"
            ),
            "provider_batch_calls": direct_cd.get("provider_batch_calls"),
        },
    )
    elapsed = (record.get("timing") or {}).get("direct_cd_end_to_end_seconds")
    check(
        "complete_timing_boundary",
        isinstance(elapsed, (int, float)) and elapsed > 0,
        "end-to-end direct-CD time is missing",
        {
            "seconds": elapsed,
            "speed_threshold_seconds": None,
            "boundary": (record.get("timing") or {}).get("boundary"),
        },
    )
    allocations = record.get("allocation_validation") or {}
    returned = allocations.get("returned_batches")
    if not isinstance(returned, list):
        returned = []
    returned_by_size = {
        item["batch_size"]: item
        for item in returned
        if isinstance(item, dict)
        and isinstance(item.get("batch_size"), int)
        and not isinstance(item.get("batch_size"), bool)
    }
    allocation_limit = allocations.get("max_returned_batch_bytes")
    allocation_checks_complete = all(
        isinstance(returned_by_size.get(batch_size), dict)
        and returned_by_size[batch_size].get("shape")
        == [batch_size, expected_dimension]
        and returned_by_size[batch_size].get("nbytes")
        == batch_size * expected_dimension * 8
        and returned_by_size[batch_size].get("within_limit") is True
        and isinstance(
            returned_by_size[batch_size].get("live_pool_delta_bytes"), int
        )
        and returned_by_size[batch_size]["live_pool_delta_bytes"] >= 0
        for batch_size in SELECTED_COLUMN_BATCH_SIZES
    )
    diagonal_allocation = allocations.get("packed_diagonal") or {}
    check(
        "bounded_allocations",
        allocation_checks_complete
        and len(returned) == len(SELECTED_COLUMN_BATCH_SIZES)
        and set(returned_by_size) == set(SELECTED_COLUMN_BATCH_SIZES)
        and isinstance(allocation_limit, int)
        and allocation_limit
        >= max(SELECTED_COLUMN_BATCH_SIZES) * expected_dimension * 8
        and diagonal_allocation.get("shape") == [expected_dimension]
        and diagonal_allocation.get("nbytes") == expected_dimension * 8
        and diagonal_allocation.get("within_limit") is True
        and allocations.get("pair_matrix_allocated") is False
        and allocations.get("four_index_ao_tensor_allocated") is False
        and allocations.get("per_pivot_ao_matrix_allocated") is False,
        "selected provider allocation evidence is missing or exceeds its bound",
        allocations,
    )
    resources = record.get("resources", {})
    hbm = (resources.get("hbm") or {}).get("peak_process_gib")
    synchronized_hbm = resources.get("synchronized_hbm") or {}
    synchronized_checkpoints = synchronized_hbm.get("checkpoints")
    synchronized_names: list[str] = []
    synchronized_driver_peaks: list[int] = []
    synchronized_complete = isinstance(synchronized_checkpoints, list)
    if synchronized_complete:
        for checkpoint in synchronized_checkpoints:
            if not isinstance(checkpoint, dict):
                synchronized_complete = False
                break
            name = checkpoint.get("name")
            driver_used = checkpoint.get("driver_used_bytes")
            driver_total = checkpoint.get("driver_total_bytes")
            pool_used = checkpoint.get("cupy_pool_used_bytes")
            pool_total = checkpoint.get("cupy_pool_total_bytes")
            numeric = all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (driver_used, driver_total, pool_used, pool_total)
            )
            if (
                not isinstance(name, str)
                or not numeric
                or not 0 <= driver_used <= driver_total
                or not 0 <= pool_used <= pool_total <= driver_used
                or checkpoint.get("timing_semantics")
                    != "cuda-device-synchronized-instantaneous"
                or checkpoint.get("measurement_scope")
                    != "exclusive-slurm-device-global"
                or checkpoint.get("source")
                    != "cudaMemGetInfo-and-cupy-default-memory-pool"
            ):
                synchronized_complete = False
                break
            synchronized_names.append(name)
            synchronized_driver_peaks.append(driver_used)
    synchronized_peak = synchronized_hbm.get("peak_driver_used_bytes")
    synchronized_peak_gib = synchronized_hbm.get("peak_driver_used_gib")
    synchronized_complete = bool(
        synchronized_complete
        and synchronized_names == list(SYNCHRONIZED_HBM_CHECKPOINT_NAMES)
        and synchronized_driver_peaks
        and isinstance(synchronized_peak, int)
        and not isinstance(synchronized_peak, bool)
        and synchronized_peak == max(synchronized_driver_peaks)
        and isinstance(synchronized_peak_gib, (int, float))
        and not isinstance(synchronized_peak_gib, bool)
        and abs(
            synchronized_peak_gib
            - synchronized_peak / float(1024**3)
        ) <= 1e-12
        and synchronized_hbm.get("schema")
            == "gpu4pyscf.gint-synchronized-hbm.v1"
        and synchronized_hbm.get("budget_gib") == HBM_LIMIT_GIB
    )
    rss = resources.get("direct_cd_peak_host_rss_gib")
    check(
        "resource_budget",
        isinstance(hbm, (int, float))
        and hbm <= HBM_LIMIT_GIB
        and synchronized_complete
        and synchronized_peak_gib <= HBM_LIMIT_GIB
        and isinstance(rss, (int, float))
        and rss <= RSS_LIMIT_GIB,
        "HBM/RSS evidence is missing or exceeds the fixed budget",
        {
            "peak_hbm_gib": hbm,
            "synchronized_hbm_complete": synchronized_complete,
            "synchronized_peak_driver_hbm_gib": synchronized_peak_gib,
            "synchronized_checkpoint_names": synchronized_names,
            "hbm_limit_gib": HBM_LIMIT_GIB,
            "peak_rss_gib": rss,
            "rss_limit_gib": RSS_LIMIT_GIB,
        },
    )
    transfers = provider.get("explicit_transfers") or {}
    complete_transfers = record.get("timed_transfer_counter") or {}
    directions = transfers.get("directions") or {}
    h2d = directions.get("h2d") or {}
    d2h = directions.get("d2h") or {}
    h2d_operations = h2d.get("operations") or {}
    d2h_operations = d2h.get("operations") or {}
    timed_operations = complete_transfers.get("by_operation") or {}
    timed_by_kind = complete_transfers.get("by_kind") or {}

    def positive_operation(operations: dict[str, Any], name: str) -> bool:
        item = operations.get(name) or {}
        return (
            isinstance(item.get("bytes"), int)
            and item["bytes"] > 0
            and isinstance(item.get("count"), int)
            and item["count"] > 0
        )

    operation_ledgers_agree = all(
        isinstance(
            (timed_operations.get(direction) or {}).get(name), dict
        )
        and (timed_operations.get(direction) or {})[name].get("bytes")
        == item.get("bytes")
        and (timed_operations.get(direction) or {})[name].get("count")
        == item.get("count")
        for direction, payload in directions.items()
        if isinstance(payload, dict)
        for name, item in (payload.get("operations") or {}).items()
    )
    timed_ledger_fully_attributed = (
        isinstance(complete_transfers.get("total_bytes"), int)
        and isinstance(complete_transfers.get("total_transfers"), int)
        and complete_transfers["total_bytes"]
        == sum(
            item.get("bytes", -1)
            for item in timed_by_kind.values()
            if isinstance(item, dict)
        )
        and complete_transfers["total_transfers"]
        == sum(
            item.get("count", -1)
            for item in timed_by_kind.values()
            if isinstance(item, dict)
        )
        and all(
            isinstance(kind_record, dict)
            and kind_record.get("bytes") == sum(
                item.get("bytes", -1)
                for item in (timed_operations.get(kind) or {}).values()
                if isinstance(item, dict)
            )
            and kind_record.get("count") == sum(
                item.get("count", -1)
                for item in (timed_operations.get(kind) or {}).values()
                if isinstance(item, dict)
            )
            for kind, kind_record in timed_by_kind.items()
        )
    )
    check(
        "selected_transfer_accounting",
        transfers.get("schema")
        == "gpu4pyscf.gint-selected-transfer-ledger.v2"
        and transfers.get("known_provider_arrays_byte_counted") is True
        and transfers.get("output_residency") == "gpu"
        and isinstance(h2d.get("bytes"), int)
        and h2d["bytes"] > 0
        and isinstance(d2h.get("bytes"), int)
        and d2h["bytes"] > 0
        and transfers.get("total_bytes") == h2d["bytes"] + d2h["bytes"]
        and complete_transfers.get("total_bytes") >= transfers.get("total_bytes")
        and timed_ledger_fully_attributed
        and operation_ledgers_agree
        and positive_operation(
            h2d_operations, "selected_device_schedule"
        )
        and positive_operation(
            h2d_operations, "selected_column_request"
        )
        and positive_operation(
            h2d_operations, "selected_gint_constant_cache"
        )
        and positive_operation(
            d2h_operations, "selected_schedule_coeff_support"
        )
        and positive_operation(
            d2h_operations, "selected_schedule_log_q"
        )
        and h2d_operations["selected_gint_constant_cache"]["count"]
        == provider.get("kernel_launches"),
        (
            "selected provider's known array transfers are missing, "
            "double-counted, or disagree with the timed counter"
        ),
        {
            "provider_explicit": transfers,
            "complete_timed_counter": complete_transfers,
        },
    )
    unresolved = transfers.get("unresolved_operations")
    check(
        "performance_transfer_qualification",
        transfers.get("complete_for_vhfopt_internal_setup") is True
        and transfers.get("complete_for_declared_payloads") is True
        and unresolved == [],
        (
            "provider has not completed its _VHFOpt/runtime transfer audit"
        ),
        {
            "complete_for_vhfopt_internal_setup": transfers.get(
                "complete_for_vhfopt_internal_setup"
            ),
            "complete_for_declared_payloads": transfers.get(
                "complete_for_declared_payloads"
            ),
            "unresolved_operations": unresolved,
        },
        scope="qualification",
    )
    runtime_gate = provider.get("runtime_gate") or {}
    check(
        "performance_receipt_release",
        provider.get("performance_eligible") is True
        and runtime_gate.get("validated") is True
        and runtime_gate.get("receipt_binding_validated") is True
        and runtime_gate.get("status") == "release-accepted",
        (
            "provider has no valid write-once MTU qualification receipt"
        ),
        {
            "provider_performance_eligible": provider.get(
                "performance_eligible"
            ),
            "performance_gate": provider.get("performance_gate"),
            "runtime_gate": runtime_gate,
        },
        scope="performance",
    )

    if record.get("status") != "completed":
        checks.append({
            "id": "completed_run",
            "scope": "correctness",
            "passed": False,
            "reason_if_failed": "gate execution did not complete",
            "evidence": record.get("status"),
        })
    correctness_passed = all(
        item["passed"] for item in checks
        if item["scope"] == "correctness"
    )
    qualification_passed = correctness_passed and all(
        item["passed"] for item in checks
        if item["scope"] == "qualification"
    )
    performance_eligible = qualification_passed and all(
        item["passed"] for item in checks
        if item["scope"] == "performance"
    )
    return {
        "schema": "gpu4pyscf.water2.gint-direct-cd-gate-decision.v2",
        "passed": correctness_passed,
        "correctness_passed": correctness_passed,
        "qualification_passed": qualification_passed,
        "performance_eligible": performance_eligible,
        "checks": checks,
        "reasons": [
            item["reason_if_failed"] for item in checks
            if item["scope"] == "correctness" and not item["passed"]
        ],
        "performance_reasons": [
            item["reason_if_failed"] for item in checks
            if item["scope"] == "performance" and not item["passed"]
        ],
        "qualification_reasons": [
            item["reason_if_failed"] for item in checks
            if item["scope"] == "qualification" and not item["passed"]
        ],
        "speed_threshold_seconds": None,
        "speed_threshold_policy": "no speed threshold belongs to this gate",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--topology-record", type=Path,
        default=os.getenv("CCSD_TOPOLOGY_RECORD"),
    )
    parser.add_argument("--eri-tol", type=float, choices=ERI_THRESHOLDS, default=1e-8)
    parser.add_argument("--direct-scf-tol", type=float, default=1e-14)
    parser.add_argument("--selected-column-tolerance", type=float, default=2e-10)
    parser.add_argument("--diagonal-tolerance", type=float, default=2e-10)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--max-block-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--oracle-max-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--runtime-gate-receipt",
        type=Path,
        default=os.getenv("GINT_RUNTIME_GATE_RECEIPT"),
        help="read-only qualification receipt; omit for the qualification run",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.direct_scf_tol <= 1:
        raise ValueError("direct_scf_tol must lie in (0, 1]")
    if args.selected_column_tolerance < 0 or args.diagonal_tolerance < 0:
        raise ValueError("oracle tolerances must be non-negative")
    if args.group_size < 1:
        raise ValueError("group_size must be positive")
    if args.max_rank is not None and args.max_rank < 1:
        raise ValueError("max_rank must be positive")
    if args.max_block_bytes < 8:
        raise ValueError("max_block_bytes must be at least 8")
    if args.oracle_max_bytes < 8:
        raise ValueError("oracle_max_bytes must be at least 8")
    expected_dimension = 116 * 117 // 2
    required_batch_bytes = (
        max(SELECTED_COLUMN_BATCH_SIZES) * expected_dimension * 8
    )
    if args.max_block_bytes < required_batch_bytes:
        raise ValueError(
            "max_block_bytes cannot hold the required B=32 FP64 output"
        )
    required_oracle_bytes = (
        116 * 116 * max(SELECTED_COLUMN_BATCH_SIZES) ** 2 * 8
    )
    if args.oracle_max_bytes < required_oracle_bytes:
        raise ValueError(
            "oracle_max_bytes cannot hold the required bounded B=32 AO2MO"
        )


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    _validate_args(args)
    runtime_gate_receipt = getattr(args, "runtime_gate_receipt", None)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    try:
        relative_output = output.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        relative_output = None
    if relative_output is not None and "results" not in relative_output.parts:
        raise ValueError(
            "an output inside the source tree must be placed under a results "
            "directory so it is excluded from the immutable source hash"
        )
    case = load_water2_case()
    if args.dry_run:
        record = {
            "schema": "gpu4pyscf.water2.gint-direct-cd-gate.v2",
            "dry_run": True,
            "case": case,
            "thresholds": {
                "eri_tol": args.eri_tol,
                "direct_scf_tol": args.direct_scf_tol,
                "selected_column_tolerance": args.selected_column_tolerance,
                "diagonal_tolerance": args.diagonal_tolerance,
            },
            "timing_boundary": (
                "selected-pair provider construction + direct packed diagonal "
                "+ all blocked matrix-free pivoted-Cholesky iterations + "
                "device synchronize"
            ),
            "configuration": {
                "selected_column_batch_sizes": list(
                    SELECTED_COLUMN_BATCH_SIZES
                ),
                "direct_cd_column_batch_size": max(
                    SELECTED_COLUMN_BATCH_SIZES
                ),
                "max_returned_batch_bytes": int(args.max_block_bytes),
                "oracle_max_bytes": int(args.oracle_max_bytes),
                "precision": "fp64",
            },
            "speed_threshold_seconds": None,
            "runtime_gate_receipt": (
                None if runtime_gate_receipt is None
                else str(runtime_gate_receipt.expanduser().resolve())
            ),
        }
        return output, record

    output.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "gpu4pyscf.water2.gint-direct-cd-gate.v2",
        "created_utc": _utc_now(),
        "status": "running",
        "command": list(sys.argv),
        "source": source_evidence(),
        "topology": topology_evidence(args.topology_record),
        "hardware": hardware_evidence(),
        "case": case,
        "thresholds": {
            "eri_tol": float(args.eri_tol),
            "direct_scf_tol": float(args.direct_scf_tol),
            "selected_column_tolerance": float(args.selected_column_tolerance),
            "diagonal_tolerance": float(args.diagonal_tolerance),
        },
        "configuration": {
            "group_size": int(args.group_size),
            "max_rank": args.max_rank,
            "selected_column_batch_sizes": list(
                SELECTED_COLUMN_BATCH_SIZES
            ),
            "selected_provider_max_batch_size": max(
                SELECTED_COLUMN_BATCH_SIZES
            ),
            "direct_cd_column_batch_size": max(
                SELECTED_COLUMN_BATCH_SIZES
            ),
            "max_returned_batch_bytes": int(args.max_block_bytes),
            "oracle_max_bytes": int(args.oracle_max_bytes),
            "precision": "fp64",
            "runtime_gate_receipt": (
                None if runtime_gate_receipt is None
                else str(runtime_gate_receipt.expanduser().resolve())
            ),
        },
        "performance_eligible": False,
    }
    monitor = HBMMonitor()
    synchronized_hbm_checkpoints: list[dict[str, Any]] = []
    provider = None
    timed_transfer_snapshot = None
    try:
        import cupy
        import numpy as np
        from pyscf import gto

        from gpu4pyscf.cc.device_runtime import TransferCounter
        from gpu4pyscf.cc.direct_cd import pivoted_cholesky_from_columns
        from gpu4pyscf.cc.gint_selected_columns import (
            GINTSelectedAOPairColumnProvider,
        )

        molecule = gto.M(
            atom=[tuple(row) for row in case["geometry_angstrom"]],
            basis=case["basis"],
            unit="Angstrom",
            cart=False,
            verbose=0,
        )
        record["case"]["actual_nao"] = int(molecule.nao_nr())
        if molecule.nao_nr() != case["expected_dimensions"]["nao"]:
            raise RuntimeError("water2 AO dimension differs from frozen contract")

        transfers = TransferCounter()
        monitor.start()
        cupy.cuda.runtime.deviceSynchronize()
        synchronized_hbm_checkpoints.append(synchronized_hbm_checkpoint(
            cupy, SYNCHRONIZED_HBM_CHECKPOINT_NAMES[0]
        ))
        timed_start = time.perf_counter()
        setup_start = timed_start
        provider = GINTSelectedAOPairColumnProvider(
            molecule,
            direct_scf_tol=args.direct_scf_tol,
            group_size=args.group_size,
            max_batch_size=max(SELECTED_COLUMN_BATCH_SIZES),
            transfer_counter=transfers,
            runtime_gate_receipt=runtime_gate_receipt,
            runtime_source_digest=record["source"]["tree_sha256_at_start"],
            runtime_source_root=record["source"]["root"],
            runtime_manifest_path=record["source"]["snapshot"][
                "manifest_path"
            ],
            runtime_release_topology_path=args.topology_record,
        )
        synchronized_hbm_checkpoints.append(synchronized_hbm_checkpoint(
            cupy, SYNCHRONIZED_HBM_CHECKPOINT_NAMES[1]
        ))
        setup_end = time.perf_counter()
        direct_cd = pivoted_cholesky_from_columns(
            provider,
            threshold=args.eri_tol,
            max_rank=args.max_rank,
            transfer_counter=transfers,
            column_batch_size=max(SELECTED_COLUMN_BATCH_SIZES),
        )
        synchronized_hbm_checkpoints.append(synchronized_hbm_checkpoint(
            cupy, SYNCHRONIZED_HBM_CHECKPOINT_NAMES[2]
        ))
        timed_end = time.perf_counter()
        monitor.close()
        timed_transfer_snapshot = transfers.to_dict()
        timed_provider_metadata = provider.metadata()
        record["timing"] = {
            "direct_cd_end_to_end_seconds": timed_end - timed_start,
            "provider_setup_seconds": setup_end - setup_start,
            "factorization_after_setup_seconds": timed_end - setup_end,
            "boundary": (
                "selected-pair provider construction + direct packed diagonal "
                "+ all blocked matrix-free pivoted-Cholesky iterations + "
                "device synchronize"
            ),
            "oracle_validation_included": False,
            "speed_threshold_seconds": None,
        }
        record["direct_cd"] = direct_cd.metadata()
        record["provider"] = timed_provider_metadata
        record["timed_transfer_counter"] = timed_transfer_snapshot
        record["resources"] = {
            "hbm": monitor.metadata(),
            "synchronized_hbm": synchronized_hbm_evidence(
                synchronized_hbm_checkpoints
            ),
            "direct_cd_peak_host_rss_gib": _host_peak_rss_gib(),
            "limits": {
                "hbm_gib": HBM_LIMIT_GIB,
                "host_rss_gib": RSS_LIMIT_GIB,
            },
        }

        master_pivots = selected_validation_pivots(provider.dimension)
        oracle_start = time.perf_counter()
        expected_master, ao2mo_metadata = ao2mo_selected_columns(
            molecule,
            provider.pair_map,
            master_pivots,
            max_working_bytes=args.oracle_max_bytes,
        )
        ao2mo_metadata["seconds"] = time.perf_counter() - oracle_start
        oracle_row = {
            pivot: index for index, pivot in enumerate(master_pivots)
        }
        pool = cupy.get_default_memory_pool()
        column_checks = []
        returned_allocations = []
        for batch_size in SELECTED_COLUMN_BATCH_SIZES:
            pivots = pivots_for_batch(master_pivots, batch_size)
            before_used = int(pool.used_bytes())
            validation_start = time.perf_counter()
            actual_device = provider.columns(pivots)
            cupy.cuda.runtime.deviceSynchronize()
            validation_seconds = time.perf_counter() - validation_start
            live_used = int(pool.used_bytes())
            actual = cupy.asnumpy(actual_device)
            provider._record_transfer(
                "d2h", "gate_validation_columns", int(actual.nbytes)
            )
            expected = expected_master[
                [oracle_row[pivot] for pivot in pivots]
            ]
            error = float(np.max(np.abs(actual - expected)))
            output_nbytes = int(actual_device.nbytes)
            shape = [int(value) for value in actual_device.shape]
            allocation = {
                "batch_size": int(batch_size),
                "shape": shape,
                "nbytes": output_nbytes,
                "expected_nbytes": (
                    int(batch_size) * int(provider.dimension) * 8
                ),
                "live_pool_delta_bytes": max(0, live_used - before_used),
                "within_limit": output_nbytes <= args.max_block_bytes,
            }
            returned_allocations.append(allocation)
            column_checks.append({
                "batch_size": int(batch_size),
                "pivots": list(pivots),
                "shape": shape,
                "dtype": str(actual_device.dtype),
                "gpu_resident": (
                    type(actual_device).__module__.split(".", 1)[0] == "cupy"
                ),
                "output_nbytes": output_nbytes,
                "max_abs_error": error,
                "seconds": validation_seconds,
            })
            del actual_device, actual, expected

        diagonal_before_used = int(pool.used_bytes())
        expected_diagonal = shell_sliced_diagonal(molecule, provider.pair_map)
        diagonal_device = provider.diagonal()
        cupy.cuda.runtime.deviceSynchronize()
        diagonal_live_used = int(pool.used_bytes())
        actual_diagonal = cupy.asnumpy(diagonal_device)
        provider._record_transfer(
            "d2h", "gate_validation_diagonal", int(actual_diagonal.nbytes)
        )
        diagonal_error = float(
            np.max(np.abs(actual_diagonal - expected_diagonal))
        )
        record["numerical_validation"] = {
            "selected_column_oracle": (
                "bounded rectangular PySCF AO2MO with unit-vector right "
                "coefficients"
            ),
            "diagonal_oracle": "CPU int2e_sph shell quartets",
            "materialized_nao4": False,
            "materialized_pair_matrix": False,
            "ao2mo_oracle": ao2mo_metadata,
            "selected_column_batches": column_checks,
            "selected_column_max_abs_error": max(
                item["max_abs_error"] for item in column_checks
            ),
            "diagonal": {
                "shape": [int(value) for value in diagonal_device.shape],
                "dtype": str(diagonal_device.dtype),
                "gpu_resident": (
                    type(diagonal_device).__module__.split(".", 1)[0]
                    == "cupy"
                ),
                "source": SELECTED_DIAGONAL_SYMBOL,
            },
            "diagonal_max_abs_error": diagonal_error,
        }
        record["allocation_validation"] = {
            "method": "returned-array-shape-plus-CuPy-live-pool-sampling",
            "max_returned_batch_bytes": int(args.max_block_bytes),
            "returned_batches": returned_allocations,
            "packed_diagonal": {
                "shape": [int(value) for value in diagonal_device.shape],
                "nbytes": int(diagonal_device.nbytes),
                "expected_nbytes": int(provider.dimension) * 8,
                "live_pool_delta_bytes": max(
                    0, diagonal_live_used - diagonal_before_used
                ),
                "within_limit": (
                    int(diagonal_device.nbytes) <= args.max_block_bytes
                ),
            },
            "pair_matrix_bytes_if_materialized": (
                int(provider.dimension) ** 2 * 8
            ),
            "four_index_ao_bytes_if_materialized": (
                int(molecule.nao_nr()) ** 4 * 8
            ),
            "pair_matrix_allocated": False,
            "four_index_ao_tensor_allocated": False,
            "per_pivot_ao_matrix_allocated": False,
        }
        record["post_validation_provider"] = provider.metadata()
        record["post_validation_transfer_counter"] = transfers.to_dict()
        record["resources"]["final_peak_host_rss_gib"] = _host_peak_rss_gib()
        record["status"] = "completed"
    except Exception as exc:
        monitor.close()
        record.update({
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        })
    finally:
        record["finished_utc"] = _utc_now()
        finish_source_evidence(record["source"])
        finish_topology_evidence(record["topology"])
        record["gate_decision"] = evaluate_gate(record)
        record["performance_eligible"] = record["gate_decision"][
            "performance_eligible"
        ]
        with output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
    return output, record


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output, record = run(args)
    if args.dry_run:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    release_mode = args.runtime_gate_receipt is not None
    print(json.dumps({
        "output": str(output),
        "mode": "release" if release_mode else "qualification",
        "status": record["status"],
        "correctness_passed": record["gate_decision"][
            "correctness_passed"
        ],
        "qualification_passed": record["gate_decision"][
            "qualification_passed"
        ],
        "performance_eligible": record["performance_eligible"],
        "gate_reasons": record["gate_decision"]["reasons"],
        "performance_reasons": record["gate_decision"][
            "performance_reasons"
        ],
        "direct_cd_seconds": (record.get("timing") or {}).get(
            "direct_cd_end_to_end_seconds"
        ),
        "rank": (record.get("direct_cd") or {}).get("rank"),
        "residual_diagonal": (record.get("direct_cd") or {}).get(
            "residual_diagonal"
        ),
    }, sort_keys=True))
    decision = record["gate_decision"]
    accepted = (
        decision["performance_eligible"] is True
        if release_mode else decision["qualification_passed"] is True
    )
    return 0 if record["status"] == "completed" and accepted else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"gint gate configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
