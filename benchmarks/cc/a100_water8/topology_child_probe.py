#!/usr/bin/env python3
"""Verify that CPU affinity and memory policy survive the guarded exec."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys

import topology_guard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-affinity", required=True)
    parser.add_argument("--expected-numa", type=int, required=True)
    args = parser.parse_args()

    libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    observed_affinity = topology_guard.format_cpu_list(os.sched_getaffinity(0))
    policy = topology_guard._read_kernel_memory_policy(libnuma)
    expected_affinity = topology_guard.format_cpu_list(
        topology_guard.parse_cpu_list(args.expected_affinity)
    )
    passed = (
        observed_affinity == expected_affinity
        and policy["mode"] == topology_guard.MPOL_PREFERRED
        and policy["nodes"] == [args.expected_numa]
        and os.getenv("GPU4PYSCF_NUMA") == str(args.expected_numa)
        and os.getenv("CCSD_PERFORMANCE_ELIGIBLE") == "true"
        and os.getenv("CCSD_BENCHMARK_TRACK") == "performance"
    )
    print(
        "TOPOLOGY_CHILD="
        + json.dumps(
            {
                "affinity": observed_affinity,
                "expected_affinity": expected_affinity,
                "kernel_memory_policy": policy,
                "GPU4PYSCF_NUMA": os.getenv("GPU4PYSCF_NUMA"),
                "CCSD_PERFORMANCE_ELIGIBLE": os.getenv(
                    "CCSD_PERFORMANCE_ELIGIBLE"
                ),
                "CCSD_BENCHMARK_TRACK": os.getenv("CCSD_BENCHMARK_TRACK"),
                "passed": passed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"topology child probe error: {exc}", file=sys.stderr)
        raise SystemExit(2)
