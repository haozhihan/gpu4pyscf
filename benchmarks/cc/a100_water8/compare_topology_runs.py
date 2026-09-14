#!/usr/bin/env python3
"""Select the frozen MTU CPU binding from paired canonical benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED = {
    "physical8": {
        "affinity": list(range(24, 32)),
        "threads": "8",
        "selected_cpulist": "24-31",
    },
    "logical16": {
        "affinity": list(range(24, 32)) + list(range(88, 96)),
        "threads": "16",
        "selected_cpulist": "24-31,88-95",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _scientific_signature(record: dict[str, Any]) -> dict[str, Any]:
    plan = record.get("plan", {})
    return {
        "case_id": record.get("case_id"),
        "method": record.get("method"),
        "geometry_sha256": record.get("geometry_sha256"),
        "basis": plan.get("basis"),
        "reference": plan.get("reference"),
        "spherical": plan.get("spherical"),
        "settings": plan.get("settings"),
        "timing_definition": record.get("timing_definition"),
    }


def _topology_path(directory: Path, mode: str, job_id: str) -> Path:
    stem = "physical8" if mode == "physical8" else "logical16"
    return directory / f"{stem}-{job_id}.json"


def validate_run(
    path: Path,
    *,
    mode: str,
    topology_dir: Path,
) -> dict[str, Any]:
    """Validate one timing and its independent fail-closed topology record."""

    if mode not in EXPECTED:
        raise ValueError(f"unknown topology mode {mode!r}")
    record = _load(path)
    reasons: list[str] = []
    expected = EXPECTED[mode]
    if record.get("status") != "completed" or record.get("cc_converged") is not True:
        reasons.append("benchmark did not complete and converge")
    seconds = record.get("post_hf_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        reasons.append("post_hf_seconds is missing or invalid")
    if record.get("affinity") != expected["affinity"]:
        reasons.append("CPU affinity does not match the requested mode")
    env = record.get("thread_environment", {})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if str(env.get(name)) != expected["threads"]:
            reasons.append(f"{name} does not match the requested mode")
    if str(env.get("GPU4PYSCF_NUMA")) != "3":
        reasons.append("GPU4PYSCF_NUMA is not 3")
    gpu = record.get("gpu", {})
    device = gpu.get("device_0", {}) if isinstance(gpu, dict) else {}
    if gpu.get("device_count") != 1 or "A100-SXM4-80GB" not in str(device.get("name", "")):
        reasons.append("run did not use one A100-SXM4-80GB")
    visible = record.get("pcie", {}).get("visible_gpus", [])
    pcie = visible[0] if len(visible) == 1 else {}
    if str(pcie.get("numa_node")) != "3":
        reasons.append("GPU PCI device is not on NUMA node 3")
    if not str(pcie.get("max_link_speed", "")).startswith("16.0") or str(pcie.get("max_link_width")) != "16":
        reasons.append("GPU link is not PCIe 4.0 x16")

    job_id = str(record.get("slurm", {}).get("SLURM_JOB_ID", ""))
    topology_path = _topology_path(topology_dir, mode, job_id)
    if not topology_path.is_file():
        reasons.append(f"missing topology record {topology_path}")
        topology: dict[str, Any] = {}
    else:
        topology = _load(topology_path)
        if topology.get("performance_eligible") is not True:
            reasons.append("topology guard marked the run ineligible")
        if topology.get("mode") != mode:
            reasons.append("topology record mode mismatch")
        if topology.get("gpu_numa_node") != 3:
            reasons.append("topology record GPU NUMA mismatch")
        if topology.get("selected_cpulist") != expected["selected_cpulist"]:
            reasons.append("topology record CPU selection mismatch")
        policy = topology.get("kernel_memory_policy", {})
        observed_policy = policy.get("after", policy)
        if policy.get("verified") is not True or observed_policy.get("mode_name") != "preferred" or observed_policy.get("nodes") != [3]:
            reasons.append("preferred NUMA-3 memory policy is unverified")
    residual = record.get("residual", {})
    if residual.get("available") is not True or residual.get("definition") != "full-space canonical FP64 equation residual":
        reasons.append("canonical FP64 final residual evidence is missing")
    if reasons:
        raise ValueError(f"{path}: " + "; ".join(reasons))
    return {
        "result": str(path),
        "result_sha256": _sha256(path),
        "topology_record": str(topology_path),
        "topology_sha256": _sha256(topology_path),
        "job_id": job_id,
        "seconds": float(seconds),
        "e_corr": float(record["e_corr"]),
        "equation_residual": float(residual["equation_norm"]),
        "scientific_signature": _scientific_signature(record),
    }


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 3:
        raise ValueError("each topology requires at least three eligible fresh-process runs")
    seconds = [run["seconds"] for run in runs]
    mean = statistics.mean(seconds)
    return {
        "count": len(runs),
        "seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "sample_cv": statistics.stdev(seconds) / mean,
        "runs": [{key: value for key, value in run.items() if key != "scientific_signature"} for run in runs],
    }


def compare(
    physical_paths: Iterable[Path],
    logical_paths: Iterable[Path],
    *,
    topology_dir: Path,
) -> dict[str, Any]:
    physical = [validate_run(path, mode="physical8", topology_dir=topology_dir) for path in physical_paths]
    logical = [validate_run(path, mode="logical16", topology_dir=topology_dir) for path in logical_paths]
    signatures = [run["scientific_signature"] for run in physical + logical]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("scientific settings or timing boundary differ across topology runs")
    energies = [run["e_corr"] for run in physical + logical]
    if max(energies) - min(energies) > 1.0e-10:
        raise ValueError("correlation energies differ by more than 1e-10 Eh")
    residuals = [run["equation_residual"] for run in physical + logical]
    if max(residuals) - min(residuals) > 1.0e-10:
        raise ValueError("final equation residuals are inconsistent")

    groups = {
        "physical8": _summarize(physical),
        "logical16": _summarize(logical),
    }
    physical_median = groups["physical8"]["median_seconds"]
    logical_median = groups["logical16"]["median_seconds"]
    difference = abs(logical_median - physical_median) / min(
        physical_median, logical_median
    )
    selected = "physical8" if difference < 0.02 else (
        "physical8" if physical_median < logical_median else "logical16"
    )
    return {
        "schema": "gpu4pyscf.water8.topology-comparison.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_signature": signatures[0],
        "groups": groups,
        "decision": {
            "relative_median_difference": difference,
            "tie_threshold": 0.02,
            "tie_policy": "select physical8 when relative difference is below 2 percent",
            "selected_mode": selected,
            "selected_affinity": EXPECTED[selected]["affinity"],
            "selected_threads": int(EXPECTED[selected]["threads"]),
            "selected_memory_policy": "preferred NUMA node 3",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical8", nargs="+", required=True, type=Path)
    parser.add_argument("--logical16", nargs="+", required=True, type=Path)
    parser.add_argument("--topology-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = compare(
        args.physical8, args.logical16, topology_dir=args.topology_dir
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {args.output}") from exc
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
