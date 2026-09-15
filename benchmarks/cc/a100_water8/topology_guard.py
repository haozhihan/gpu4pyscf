#!/usr/bin/env python3
"""Audit and enforce the MTU A100/NUMA-3 benchmark launch contract.

The MTU compute image does not provide ``srun`` or ``numactl`` inside a batch
job.  This helper therefore derives the allocated GPU's NUMA node from sysfs,
intersects that node's CPUs with the Slurm cgroup affinity, and narrows the
benchmark process with ``sched_setaffinity``.  A run is performance eligible
only when all checks pass.  Every decision is written to a sidecar JSON file.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


EXPECTED_NUMA_NODE = 3
MODE_SHAPES = {
    "physical8": (8, 1),
    "logical16": (8, 2),
}
MPOL_PREFERRED = 1
MPOL_NAMES = {
    0: "default",
    1: "preferred",
    2: "bind",
    3: "interleave",
    4: "local",
    5: "preferred_many",
}


def parse_cpu_list(value: str) -> tuple[int, ...]:
    """Expand a Linux cpulist such as ``24-31,88-95``."""

    cpus: set[int] = set()
    stripped = value.strip()
    if not stripped:
        return ()
    for field in stripped.split(","):
        field = field.strip()
        if not field:
            raise ValueError(f"invalid empty cpulist field in {value!r}")
        if "-" in field:
            pieces = field.split("-")
            if len(pieces) != 2:
                raise ValueError(f"invalid cpulist range {field!r}")
            start, stop = (int(piece) for piece in pieces)
            if start < 0 or stop < start:
                raise ValueError(f"invalid cpulist range {field!r}")
            cpus.update(range(start, stop + 1))
        else:
            cpu = int(field)
            if cpu < 0:
                raise ValueError(f"invalid CPU id {cpu}")
            cpus.add(cpu)
    return tuple(sorted(cpus))


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Compress CPU ids into the kernel cpulist representation."""

    ordered = sorted(set(int(cpu) for cpu in cpus))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def normalize_pci_bus_id(value: str) -> str:
    """Convert NVIDIA's eight-digit PCI domain to Linux sysfs form."""

    value = value.strip().lower()
    fields = value.split(":")
    if len(fields) != 3 or "." not in fields[2]:
        raise ValueError(f"invalid PCI bus id {value!r}")
    domain, bus, device_function = fields
    if len(domain) == 8 and domain.startswith("0000"):
        domain = domain[4:]
    if len(domain) != 4 or len(bus) != 2:
        raise ValueError(f"invalid PCI bus id {value!r}")
    device, function = device_function.split(".", 1)
    if len(device) != 2 or len(function) != 1:
        raise ValueError(f"invalid PCI bus id {value!r}")
    int(domain, 16)
    int(bus, 16)
    int(device, 16)
    int(function, 16)
    return f"{domain}:{bus}:{device}.{function}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _core_key(sysfs_root: Path, cpu: int) -> tuple[int, int]:
    topology = sysfs_root / "devices/system/cpu" / f"cpu{cpu}" / "topology"
    return (
        int(_read_text(topology / "physical_package_id")),
        int(_read_text(topology / "core_id")),
    )


def select_cpus(
    cpus: Iterable[int], mode: str, *, sysfs_root: Path = Path("/sys")
) -> tuple[int, ...]:
    """Select eight cores with one or two hardware threads per core."""

    try:
        core_count, threads_per_core = MODE_SHAPES[mode]
    except KeyError as exc:
        raise ValueError(f"unknown topology mode {mode!r}") from exc

    grouped: dict[tuple[int, int], list[int]] = {}
    for cpu in sorted(set(cpus)):
        grouped.setdefault(_core_key(sysfs_root, cpu), []).append(cpu)

    eligible_groups = [
        members
        for members in grouped.values()
        if len(members) >= threads_per_core
    ]
    if len(eligible_groups) < core_count:
        return ()
    selected = [
        cpu
        for members in eligible_groups[:core_count]
        for cpu in members[:threads_per_core]
    ]
    return tuple(sorted(selected))


def _query_visible_gpu_bus_ids() -> tuple[str, ...]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return tuple(
        normalize_pci_bus_id(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    )


def _proc_status_list(field: str) -> str:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key == field:
            return value.strip()
    raise RuntimeError(f"/proc/self/status has no {field} field")


def _read_kernel_memory_policy(libnuma: Any) -> dict[str, Any]:
    """Read this thread's default Linux memory policy via get_mempolicy."""

    max_nodes = 1024
    bits_per_word = ctypes.sizeof(ctypes.c_ulong) * 8
    word_count = (max_nodes + bits_per_word - 1) // bits_per_word
    nodemask = (ctypes.c_ulong * word_count)()
    mode = ctypes.c_int(-1)
    get_mempolicy = libnuma.get_mempolicy
    get_mempolicy.argtypes = (
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
    )
    get_mempolicy.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = get_mempolicy(
        ctypes.byref(mode), nodemask, max_nodes, None, 0
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    nodes = [
        node
        for node in range(max_nodes)
        if nodemask[node // bits_per_word] & (1 << (node % bits_per_word))
    ]
    return {
        "mode": int(mode.value),
        "mode_name": MPOL_NAMES.get(int(mode.value), "unknown"),
        "nodes": nodes,
    }


def set_and_verify_preferred_memory(node: int) -> dict[str, Any]:
    """Apply ``numactl --preferred=NODE`` semantics and verify the kernel state."""

    libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    libnuma.numa_available.argtypes = ()
    libnuma.numa_available.restype = ctypes.c_int
    if libnuma.numa_available() < 0:
        raise RuntimeError("libnuma reports that NUMA is unavailable")
    before = _read_kernel_memory_policy(libnuma)
    set_preferred = libnuma.numa_set_preferred
    set_preferred.argtypes = (ctypes.c_int,)
    set_preferred.restype = None
    ctypes.set_errno(0)
    set_preferred(node)
    after = _read_kernel_memory_policy(libnuma)
    if after["mode"] != MPOL_PREFERRED or after["nodes"] != [node]:
        raise RuntimeError(
            "kernel memory policy verification failed: "
            f"expected preferred node {node}, observed {after}"
        )
    return {
        "backend": "libnuma.so.1",
        "requested_semantics": f"numactl --preferred={node}",
        "before": before,
        "after": after,
        "verified": True,
    }


@dataclass(frozen=True)
class TopologyDecision:
    schema_version: int
    created_utc: str
    host: str
    pid: int
    mode: str
    expected_gpu_numa_node: int
    requested_physical_cores: int
    requested_threads: int
    gpu_pci_bus_ids: list[str]
    gpu_numa_node: int | None
    gpu_node_cpulist: str | None
    allowed_cpulist_before: str
    node_allowed_intersection: str
    selected_cpulist: str
    observed_cpulist_after_bind: str
    mems_allowed_list: str | None
    slurm: dict[str, str | None]
    commands: dict[str, str | None]
    memory_binding_evidence: dict[str, str | None]
    kernel_memory_policy: dict[str, Any] | None
    performance_eligible: bool
    benchmark_track: str
    reasons: list[str]
    command: list[str]


def decide(
    *,
    mode: str,
    expected_numa_node: int = EXPECTED_NUMA_NODE,
    sysfs_root: Path = Path("/sys"),
    gpu_bus_ids: Sequence[str] | None = None,
    allowed_cpus: Iterable[int] | None = None,
    apply_binding: bool = True,
    memory_policy_setter: Callable[[int], dict[str, Any]] | None = None,
    command: Sequence[str] = (),
) -> TopologyDecision:
    """Build a fail-closed topology decision and optionally bind this process."""

    core_count, threads_per_core = MODE_SHAPES[mode]
    requested_threads = core_count * threads_per_core
    reasons: list[str] = []
    actual_allowed = set(
        os.sched_getaffinity(0) if allowed_cpus is None else allowed_cpus
    )
    allowed_before = format_cpu_list(actual_allowed)

    queried_bus_ids: tuple[str, ...]
    try:
        queried_bus_ids = tuple(
            normalize_pci_bus_id(value)
            for value in (
                _query_visible_gpu_bus_ids() if gpu_bus_ids is None else gpu_bus_ids
            )
        )
    except Exception as exc:
        queried_bus_ids = ()
        reasons.append(f"visible GPU PCI query failed: {exc!r}")

    gpu_numa_node: int | None = None
    gpu_node_cpulist: str | None = None
    node_cpus: set[int] = set()
    if len(queried_bus_ids) != 1:
        reasons.append(
            f"expected exactly one visible GPU, observed {len(queried_bus_ids)}"
        )
    else:
        pci_path = sysfs_root / "bus/pci/devices" / queried_bus_ids[0]
        try:
            gpu_numa_node = int(_read_text(pci_path / "numa_node"))
        except Exception as exc:
            reasons.append(f"cannot read GPU NUMA node from {pci_path}: {exc!r}")
        if gpu_numa_node != expected_numa_node:
            reasons.append(
                f"GPU NUMA node is {gpu_numa_node}, expected {expected_numa_node}"
            )
        if gpu_numa_node is not None and gpu_numa_node >= 0:
            node_path = (
                sysfs_root
                / "devices/system/node"
                / f"node{gpu_numa_node}"
                / "cpulist"
            )
            try:
                gpu_node_cpulist = _read_text(node_path)
                node_cpus = set(parse_cpu_list(gpu_node_cpulist))
            except Exception as exc:
                reasons.append(f"cannot read GPU-node cpulist from {node_path}: {exc!r}")

    intersection = actual_allowed & node_cpus
    selected: tuple[int, ...] = ()
    if gpu_numa_node == expected_numa_node and intersection:
        try:
            selected = select_cpus(intersection, mode, sysfs_root=sysfs_root)
        except Exception as exc:
            reasons.append(f"cannot classify CPU cores: {exc!r}")
    if len(selected) != requested_threads:
        reasons.append(
            "Slurm cgroup does not contain the required eight GPU-local cores "
            f"for {mode}; selected {len(selected)} of {requested_threads} threads"
        )

    observed_after = allowed_before
    preliminarily_eligible = not reasons
    if preliminarily_eligible and apply_binding:
        try:
            os.sched_setaffinity(0, set(selected))
            observed_after = format_cpu_list(os.sched_getaffinity(0))
        except Exception as exc:
            reasons.append(f"sched_setaffinity failed: {exc!r}")
        if observed_after != format_cpu_list(selected):
            reasons.append(
                "post-bind affinity differs from selected affinity: "
                f"observed {observed_after!r}"
            )

    try:
        mems_allowed = _proc_status_list("Mems_allowed_list")
        if expected_numa_node not in parse_cpu_list(mems_allowed):
            reasons.append(
                f"memory node {expected_numa_node} is outside Mems_allowed_list"
            )
    except Exception as exc:
        mems_allowed = None
        reasons.append(f"cannot audit Mems_allowed_list: {exc!r}")

    kernel_memory_policy: dict[str, Any] | None = None
    if not reasons:
        setter = memory_policy_setter or set_and_verify_preferred_memory
        try:
            kernel_memory_policy = setter(expected_numa_node)
            if not kernel_memory_policy.get("verified", False):
                raise RuntimeError("memory-policy setter returned unverified state")
        except Exception as exc:
            reasons.append(f"cannot enforce preferred memory policy: {exc!r}")

    performance_eligible = not reasons
    return TopologyDecision(
        schema_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        host=socket.gethostname(),
        pid=os.getpid(),
        mode=mode,
        expected_gpu_numa_node=expected_numa_node,
        requested_physical_cores=core_count,
        requested_threads=requested_threads,
        gpu_pci_bus_ids=list(queried_bus_ids),
        gpu_numa_node=gpu_numa_node,
        gpu_node_cpulist=gpu_node_cpulist,
        allowed_cpulist_before=allowed_before,
        node_allowed_intersection=format_cpu_list(intersection),
        selected_cpulist=format_cpu_list(selected),
        observed_cpulist_after_bind=observed_after,
        mems_allowed_list=mems_allowed,
        slurm={
            key: os.getenv(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_NODELIST",
                "SLURM_JOB_PARTITION",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        commands={name: shutil.which(name) for name in ("srun", "numactl", "taskset")},
        memory_binding_evidence={
            key: os.getenv(key)
            for key in (
                "SLURM_MEM_BIND",
                "SLURM_MEM_BIND_LIST",
                "SLURM_MEM_BIND_PREFER",
                "SLURM_MEM_BIND_SORT",
                "SLURM_MEM_BIND_TYPE",
                "SLURM_MEM_BIND_VERBOSE",
            )
        },
        kernel_memory_policy=kernel_memory_policy,
        performance_eligible=performance_eligible,
        benchmark_track=(
            "performance" if performance_eligible else "correctness-only"
        ),
        reasons=reasons,
        command=list(command),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=tuple(MODE_SHAPES))
    parser.add_argument("--expected-gpu-numa", type=int, default=EXPECTED_NUMA_NODE)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument(
        "--require-performance",
        action="store_true",
        help="do not launch the command when topology is correctness-only",
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _write_new_record(path: Path, decision: TopologyDecision) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(asdict(decision), stream, indent=2, sort_keys=True)
        stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if args.probe_only and command:
        raise ValueError("--probe-only cannot be combined with a command")
    if not args.probe_only and not command:
        raise ValueError("a command is required unless --probe-only is used")

    decision = decide(
        mode=args.mode,
        expected_numa_node=args.expected_gpu_numa,
        command=command,
    )
    _write_new_record(args.record, decision)
    print(
        "TOPOLOGY_GUARD=" + json.dumps(asdict(decision), sort_keys=True),
        flush=True,
    )

    if args.probe_only:
        return 0 if decision.performance_eligible else 2
    if args.require_performance and not decision.performance_eligible:
        print(
            "topology guard refused a performance run; see " + str(args.record),
            file=sys.stderr,
        )
        return 3

    environment = os.environ.copy()
    environment.update(
        {
            "CCSD_PERFORMANCE_ELIGIBLE": str(decision.performance_eligible).lower(),
            "CCSD_BENCHMARK_TRACK": decision.benchmark_track,
            "CCSD_TOPOLOGY_RECORD": str(args.record),
            "CCSD_BOUND_CPUS": decision.observed_cpulist_after_bind,
        }
    )
    if decision.gpu_numa_node is not None and decision.gpu_numa_node >= 0:
        environment["GPU4PYSCF_NUMA"] = str(decision.gpu_numa_node)
    else:
        environment.pop("GPU4PYSCF_NUMA", None)
    os.execvpe(command[0], command, environment)
    raise AssertionError("os.execvpe unexpectedly returned")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"topology guard error: {exc}", file=sys.stderr)
        raise SystemExit(2)
