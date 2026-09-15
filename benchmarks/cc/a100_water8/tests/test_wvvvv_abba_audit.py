"""CPU-only tests for the same-state Wvvvv ABBA audit."""

from __future__ import annotations

import copy
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "water8_wvvvv_abba_audit", ROOT / "wvvvv_abba_audit.py"
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _comparison(layer: str) -> dict:
    array_names, scalar_names = AUDIT.EXPECTED_OUTPUTS[layer]
    arrays = {
        name: {
            "passed": True,
            "finite": True,
            "shape": [2, 2],
            "dtype": "float64",
        }
        for name in array_names
    }
    scalars = {
        name: {
            "passed": True,
            "finite": True,
            "reference": float(index + 1),
            "candidate": float(index + 1),
        }
        for index, name in enumerate(sorted(scalar_names))
    }
    return {"passed": True, "arrays": arrays, "scalars": scalars}


def _passing_rows(rank: int = 17, state: str = "a" * 64) -> list[dict]:
    rows = []
    sequence_index = 0
    for layer in AUDIT.LAYERS:
        comparison = _comparison(layer)
        scalars = {
            name: check["candidate"]
            for name, check in comparison["scalars"].items()
        }
        for layer_index, selector in enumerate(AUDIT.ABBA):
            rows.append({
                "sequence_index": sequence_index,
                "layer_sequence_index": layer_index,
                "layer": layer,
                "selector": selector,
                "status": "passed",
                "error": None,
                "cuda_event_seconds": 1.0 + 0.01 * sequence_index,
                "rank": rank,
                "state_identity_sha256": state,
                "state_content_sha256": "c" * 64,
                "state_pointer_sha256": "b" * 64,
                "state_manifest": {
                    "identity_sha256": state,
                    "content_sha256": "c" * 64,
                    "arrays": {
                        name: {
                            "shape": [2, 2],
                            "dtype": "float64",
                            "nbytes": 32,
                            "value_sha256": "e" * 64,
                        }
                        for name in AUDIT.EXPECTED_STATE_ARRAYS
                    },
                    "pointers": {"identity_sha256": "b" * 64},
                },
                "state_observation": {
                    "computed_outside_cuda_event": True,
                    "host_staging": "bounded-D2H-for-content-hash",
                    "included_arrays": sorted(AUDIT.EXPECTED_STATE_ARRAYS),
                },
                "kernel_metadata": AUDIT.KERNEL_METADATA[selector],
                "selector_metadata": selector,
                "scientific_scalars": dict(scalars),
                "comparison_to_first_two_ladder": copy.deepcopy(comparison),
            })
            sequence_index += 1
    return rows


def test_array_comparison_uses_elementwise_strict_tolerance() -> None:
    reference = np.array([[1.0, -2.0], [0.0, 4.0]])
    candidate = reference.copy()
    candidate[0, 0] += 5.0e-12

    passing = AUDIT._compare_arrays(
        reference, candidate, np, atol=1.0e-11, rtol=0.0
    )
    failing = AUDIT._compare_arrays(
        reference, candidate, np, atol=1.0e-13, rtol=0.0
    )

    assert passing["passed"] is True
    assert passing["violation_count"] == 0
    assert failing["passed"] is False
    assert failing["violation_count"] == 1


@pytest.mark.parametrize(
    "candidate",
    [np.array([np.nan]), np.array([np.inf]), np.array([-np.inf])],
)
def test_array_comparison_rejects_nonfinite_values(candidate) -> None:
    comparison = AUDIT._compare_arrays(
        np.zeros(1), candidate, np, atol=1.0, rtol=1.0
    )

    assert comparison["passed"] is False
    assert comparison["finite"] is False


def test_fail_closed_summary_accepts_only_complete_abba_blocks() -> None:
    rows = _passing_rows()

    summary = AUDIT._summarize_measurements(
        rows,
        expected_rank=17,
        expected_state_identity="a" * 64,
        expected_state_pointer="b" * 64,
        expected_state_content="c" * 64,
    )

    assert summary["status"] == "passed"
    assert summary["performance_eligible"] is False
    assert summary["production_qualified"] is False
    assert set(summary["layers"]) == set(AUDIT.LAYERS)
    assert all(
        value["sample_count_per_selector"] == 2
        for value in summary["layers"].values()
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.pop(),
        lambda rows: rows[1].update(status="error", error="boom"),
        lambda rows: rows[3].update(rank=18),
        lambda rows: rows[4].update(state_identity_sha256="b" * 64),
        lambda rows: rows[5].update(state_pointer_sha256="c" * 64),
        lambda rows: rows[5].update(state_content_sha256="d" * 64),
        lambda rows: rows[6].update(selector="two-ladder"),
        lambda rows: rows[8]["comparison_to_first_two_ladder"].update(
            passed=False
        ),
        lambda rows: rows[9]["comparison_to_first_two_ladder"][
            "arrays"
        ].pop("doubles_core"),
        lambda rows: rows[10]["scientific_scalars"].update(energy=999.0),
    ],
)
def test_fail_closed_summary_rejects_missing_failed_or_inconsistent_rows(
    mutate,
) -> None:
    rows = _passing_rows()
    mutate(rows)

    with pytest.raises(ValueError):
        AUDIT._summarize_measurements(
            rows,
            expected_rank=17,
            expected_state_identity="a" * 64,
            expected_state_pointer="b" * 64,
            expected_state_content="c" * 64,
        )


def test_argument_validation_requires_explicit_artifact_and_write_once_output(
    tmp_path,
) -> None:
    artifact = tmp_path / "orbitals.npz"
    artifact.write_bytes(b"placeholder")
    output = tmp_path / "audit.json"
    args = AUDIT._parser().parse_args([
        "--orbital-artifact-in", str(artifact),
        "--output", str(output),
        "--eri-tol", "1e-8",
        "--rr-eig-cutoff", "1e-6",
    ])

    AUDIT._validate_args(args)
    args.threads = 16
    with pytest.raises(ValueError, match="exactly 8 CPU threads"):
        AUDIT._validate_args(args)
    args.threads = 8
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        AUDIT._validate_args(args)


def test_json_writer_is_exclusive_and_rejects_nan(tmp_path) -> None:
    output = tmp_path / "record.json"
    AUDIT._json_write_once(output, {"status": "passed"})

    with pytest.raises(FileExistsError, match="overwrite"):
        AUDIT._json_write_once(output, {"status": "second"})
    with pytest.raises(ValueError):
        AUDIT._json_write_once(tmp_path / "nan.json", {"value": np.nan})


def test_same_pointer_with_changed_content_is_rejected() -> None:
    rows = _passing_rows()
    # The pointer witness is unchanged.  A changed content witness must still
    # fail the lifecycle gate.
    rows[2]["state_content_sha256"] = "d" * 64
    rows[2]["state_manifest"]["content_sha256"] = "d" * 64

    with pytest.raises(ValueError, match="content"):
        AUDIT._summarize_measurements(
            rows,
            expected_rank=17,
            expected_state_identity="a" * 64,
            expected_state_pointer="b" * 64,
            expected_state_content="c" * 64,
        )


def test_jacobi_selector_is_restored_when_callback_raises() -> None:
    def fail(*_args):
        raise RuntimeError("synthetic Jacobi failure")

    engine = SimpleNamespace(wvvvv_kernel="two-ladder", jacobi=fail)

    with pytest.raises(RuntimeError, match="synthetic"):
        AUDIT._jacobi_with_wvvvv_selector(
            engine, None, None, selector="fused"
        )
    assert engine.wvvvv_kernel == "two-ladder"


class _FakeCudaRuntime:
    def __init__(self, name: str, memory: int):
        self._properties = {
            "name": name.encode(),
            "major": 8,
            "minor": 0,
            "totalGlobalMem": memory,
        }

    def getDeviceCount(self):
        return 1

    def getDeviceProperties(self, _device_id):
        return self._properties

    def runtimeGetVersion(self):
        return 12000

    def driverGetVersion(self):
        return 12000


class _FakeCuda:
    def __init__(self, name: str, memory: int):
        self.runtime = _FakeCudaRuntime(name, memory)

    @staticmethod
    def Device(device_id=0):
        return SimpleNamespace(id=device_id, pci_bus_id="0000:17:00.0")


@pytest.mark.parametrize(
    "name,memory",
    [
        ("NVIDIA A100-PCIE-40GB", 40 << 30),
        ("NVIDIA A100-SXM4-80GB", 40 << 30),
    ],
)
def test_gpu_identity_rejects_pcie_or_undersized_a100(name, memory) -> None:
    cp = SimpleNamespace(cuda=_FakeCuda(name, memory))
    with pytest.raises(RuntimeError, match="A100-SXM4|72 GiB"):
        AUDIT._gpu_identity(cp)


def test_gpu_identity_accepts_only_full_sxm4_memory_contract() -> None:
    cp = SimpleNamespace(
        cuda=_FakeCuda("NVIDIA A100-SXM4-80GB", 80 << 30)
    )
    identity = AUDIT._gpu_identity(cp)
    assert identity["compute_capability"] == "8.0"
    assert identity["total_global_memory_bytes"] == 80 << 30


def test_process_affinity_is_exactly_the_fixed_cpu_set(monkeypatch) -> None:
    monkeypatch.setattr(
        AUDIT.os,
        "sched_getaffinity",
        lambda _pid: set(range(24, 32)),
        raising=False,
    )
    assert AUDIT._process_affinity() == list(range(24, 32))

    monkeypatch.setattr(
        AUDIT.os,
        "sched_getaffinity",
        lambda _pid: set(range(23, 32)),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="24-31"):
        AUDIT._process_affinity()


def test_cpu_model_identity_requires_epyc_7513() -> None:
    assert (
        AUDIT._cpu_model_identity(
            lambda: "AMD EPYC 7513 32-Core Processor"
        )
        == "AMD EPYC 7513 32-Core Processor"
    )
    with pytest.raises(RuntimeError, match="EPYC 7513"):
        AUDIT._cpu_model_identity(lambda: "AMD EPYC 7763")
