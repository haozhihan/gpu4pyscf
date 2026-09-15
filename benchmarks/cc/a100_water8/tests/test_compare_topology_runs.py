"""Tests for fail-closed MTU topology selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from compare_topology_runs import compare  # noqa: E402


def _write_run(root: Path, mode: str, index: int, seconds: float) -> Path:
    physical = mode == "physical8"
    job = str(9000 + index + (0 if physical else 100))
    affinity = list(range(24, 32)) + ([] if physical else list(range(88, 96)))
    threads = "8" if physical else "16"
    record = {
        "status": "completed",
        "cc_converged": True,
        "post_hf_seconds": seconds,
        "e_corr": -0.5,
        "case_id": "water2-tz",
        "method": "canonical",
        "geometry_sha256": "geometry",
        "affinity": affinity,
        "thread_environment": {
            "OMP_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "GPU4PYSCF_NUMA": "3",
        },
        "gpu": {
            "device_count": 1,
            "device_0": {"name": "NVIDIA A100-SXM4-80GB"},
        },
        "pcie": {"visible_gpus": [{
            "numa_node": "3",
            "max_link_speed": "16.0 GT/s PCIe",
            "max_link_width": "16",
        }]},
        "plan": {
            "basis": "cc-pVTZ",
            "reference": "RHF",
            "spherical": True,
            "settings": {"cc_conv_tol": 1.0e-8},
        },
        "timing_definition": "complete post-HF",
        "residual": {
            "available": True,
            "definition": "full-space canonical FP64 equation residual",
            "equation_norm": 1.0e-7,
        },
        "slurm": {"SLURM_JOB_ID": job},
    }
    path = root / f"{mode}-{index}.json"
    path.write_text(json.dumps(record))
    topology = {
        "performance_eligible": True,
        "mode": mode,
        "gpu_numa_node": 3,
        "selected_cpulist": "24-31" if physical else "24-31,88-95",
        "kernel_memory_policy": {
            "verified": True,
            "mode_name": "preferred",
            "nodes": [3],
        },
    }
    topo_name = "physical8" if physical else "logical16"
    (root / f"{topo_name}-{job}.json").write_text(json.dumps(topology))
    return path


def test_three_runs_select_faster_mode(tmp_path: Path) -> None:
    physical = [_write_run(tmp_path, "physical8", i, value) for i, value in enumerate((10.8, 10.9, 11.0))]
    logical = [_write_run(tmp_path, "logical16", i, value) for i, value in enumerate((11.2, 11.3, 11.4))]

    result = compare(physical, logical, topology_dir=tmp_path)

    assert result["groups"]["physical8"]["median_seconds"] == 10.9
    assert result["decision"]["selected_mode"] == "physical8"
    assert result["decision"]["selected_affinity"] == list(range(24, 32))


def test_below_two_percent_uses_physical_tie_policy(tmp_path: Path) -> None:
    physical = [_write_run(tmp_path, "physical8", i, value) for i, value in enumerate((10.0, 10.0, 10.0))]
    logical = [_write_run(tmp_path, "logical16", i, value) for i, value in enumerate((9.9, 9.9, 9.9))]

    result = compare(physical, logical, topology_dir=tmp_path)

    assert result["decision"]["relative_median_difference"] < 0.02
    assert result["decision"]["selected_mode"] == "physical8"


def test_ineligible_topology_fails_closed(tmp_path: Path) -> None:
    physical = [_write_run(tmp_path, "physical8", i, 10.0) for i in range(3)]
    logical = [_write_run(tmp_path, "logical16", i, 10.1) for i in range(3)]
    topology = tmp_path / "physical8-9000.json"
    payload = json.loads(topology.read_text())
    payload["performance_eligible"] = False
    topology.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="ineligible"):
        compare(physical, logical, topology_dir=tmp_path)
