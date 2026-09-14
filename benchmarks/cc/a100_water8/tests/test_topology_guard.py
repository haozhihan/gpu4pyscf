from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "topology_guard.py"
SPEC = importlib.util.spec_from_file_location("a100_topology_guard", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
topology_guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = topology_guard
SPEC.loader.exec_module(topology_guard)


def _fake_topology(root: Path) -> None:
    pci = root / "bus/pci/devices/0000:01:00.0"
    pci.mkdir(parents=True)
    (pci / "numa_node").write_text("3\n")
    node = root / "devices/system/node/node3"
    node.mkdir(parents=True)
    (node / "cpulist").write_text("24-31,88-95\n")
    for low, high in zip(range(24, 32), range(88, 96)):
        for cpu in (low, high):
            cpu_topology = root / f"devices/system/cpu/cpu{cpu}/topology"
            cpu_topology.mkdir(parents=True)
            (cpu_topology / "physical_package_id").write_text("1\n")
            (cpu_topology / "core_id").write_text(f"{low}\n")


def _verified_memory_policy(node):
    return {
        "backend": "test",
        "before": {"mode": 0, "mode_name": "default", "nodes": []},
        "after": {"mode": 1, "mode_name": "preferred", "nodes": [node]},
        "verified": True,
    }


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("", ()),
        ("3", (3,)),
        ("0-2,5,7-8", (0, 1, 2, 5, 7, 8)),
    ],
)
def test_cpu_list_round_trip(source, expected):
    assert topology_guard.parse_cpu_list(source) == expected
    assert topology_guard.parse_cpu_list(
        topology_guard.format_cpu_list(expected)
    ) == expected


def test_normalize_nvidia_pci_domain():
    assert (
        topology_guard.normalize_pci_bus_id("00000000:01:00.0")
        == "0000:01:00.0"
    )
    assert topology_guard.normalize_pci_bus_id("0000:01:00.0") == "0000:01:00.0"


def test_select_exact_gpu_local_shapes(tmp_path):
    _fake_topology(tmp_path)
    allowed = topology_guard.parse_cpu_list("0-31,64-95")
    node_cpus = set(topology_guard.parse_cpu_list("24-31,88-95"))
    intersection = set(allowed) & node_cpus

    assert topology_guard.select_cpus(
        intersection, "physical8", sysfs_root=tmp_path
    ) == tuple(range(24, 32))
    assert topology_guard.select_cpus(
        intersection, "logical16", sysfs_root=tmp_path
    ) == tuple(range(24, 32)) + tuple(range(88, 96))


def test_incomplete_cgroup_fails_closed(tmp_path, monkeypatch):
    _fake_topology(tmp_path)
    monkeypatch.setattr(topology_guard, "_proc_status_list", lambda field: "0-3")
    decision = topology_guard.decide(
        mode="physical8",
        expected_numa_node=3,
        sysfs_root=tmp_path,
        gpu_bus_ids=("00000000:01:00.0",),
        allowed_cpus=range(0, 8),
        apply_binding=False,
        memory_policy_setter=_verified_memory_policy,
    )

    assert decision.performance_eligible is False
    assert decision.benchmark_track == "correctness-only"
    assert decision.selected_cpulist == ""
    assert any("Slurm cgroup" in reason for reason in decision.reasons)


def test_gpu_local_cgroup_is_performance_eligible(tmp_path, monkeypatch):
    _fake_topology(tmp_path)
    monkeypatch.setattr(topology_guard, "_proc_status_list", lambda field: "0-3")
    decision = topology_guard.decide(
        mode="logical16",
        expected_numa_node=3,
        sysfs_root=tmp_path,
        gpu_bus_ids=("00000000:01:00.0",),
        allowed_cpus=topology_guard.parse_cpu_list("0-31,64-95"),
        apply_binding=False,
        memory_policy_setter=_verified_memory_policy,
    )

    assert decision.performance_eligible is True
    assert decision.selected_cpulist == "24-31,88-95"
    assert decision.observed_cpulist_after_bind == "0-31,64-95"


def test_unverified_memory_policy_fails_closed(tmp_path, monkeypatch):
    _fake_topology(tmp_path)
    monkeypatch.setattr(topology_guard, "_proc_status_list", lambda field: "0-3")
    decision = topology_guard.decide(
        mode="physical8",
        expected_numa_node=3,
        sysfs_root=tmp_path,
        gpu_bus_ids=("00000000:01:00.0",),
        allowed_cpus=topology_guard.parse_cpu_list("0-31,64-95"),
        apply_binding=False,
        memory_policy_setter=lambda node: {"verified": False},
    )

    assert decision.performance_eligible is False
    assert any("preferred memory policy" in reason for reason in decision.reasons)
