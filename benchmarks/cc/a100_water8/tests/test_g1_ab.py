"""CPU-only tests for the fail-closed G1 shared-orbital A/B orchestrator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import copy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("water8_g1_ab", ROOT / "g1_ab.py")
assert SPEC is not None and SPEC.loader is not None
G1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = G1
SPEC.loader.exec_module(G1)


def _task_and_source(tmp_path: Path) -> tuple[Path, Path]:
    task = tmp_path / "task"
    source = task / "snapshots" / ("a" * 64) / "source"
    return task, source


def _plan(tmp_path: Path, *, dependency: str | None = None) -> dict:
    task, source = _task_and_source(tmp_path)
    return G1.make_plan(
        task_root=task,
        source_root=source,
        run_id="g1-test",
        initial_dependency=dependency,
    )


def test_plan_is_fixed_six_job_serial_shared_orbital_dag(tmp_path: Path) -> None:
    plan = _plan(tmp_path, dependency="afterok:62434")

    assert plan["case_order"] == ["water2-tz", "water4-tz", "water8-tz"]
    assert plan["node"] == "compute-1-6"
    assert len(plan["jobs"]) == 6
    assert [job["id"] for job in plan["jobs"]] == [
        "water2-legacy", "water2-resident", "water4-legacy",
        "water4-resident", "water8-legacy", "water8-resident",
    ]
    for index, job in enumerate(plan["jobs"]):
        assert job["ordinal"] == index
        assert job["slurm_options"] == ["--nodelist=compute-1-6"]
        assert job["depends_on"] == ([] if index == 0 else [plan["jobs"][index - 1]["id"]])
    for legacy, resident in zip(plan["jobs"][::2], plan["jobs"][1::2]):
        assert legacy["method"] == "canonical_legacy"
        assert resident["method"] == "canonical"
        assert (
            legacy["exports"]["BENCHMARK_ORBITAL_ARTIFACT_OUT"]
            == resident["exports"]["BENCHMARK_ORBITAL_ARTIFACT_IN"]
        )


def test_preview_shows_initial_and_symbolic_dependencies(tmp_path: Path) -> None:
    plan = _plan(tmp_path, dependency="afterok:62434")

    preview = G1.submit_plan(plan, execute=False)

    assert preview["stage"] == "all"
    assert len(preview["jobs"]) == 6
    assert "--dependency=afterok:62434" in preview["jobs"][0]["argv"]
    assert (
        "--dependency=afterok:<water2-legacy>"
        in preview["jobs"][1]["argv"]
    )
    assert all(
        "--nodelist=compute-1-6" in item["argv"]
        for item in preview["jobs"]
    )


def _receipt_evidence(
    tmp_path: Path, receipt: dict, name: str
) -> tuple[dict, dict]:
    path = tmp_path / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return G1._read_json_evidence(path, name)


def _receipt(
    tmp_path: Path, plan: dict, stage: str, *, first_job_id: int
) -> tuple[dict, dict]:
    receipt = G1.submit_plan(plan, execute=False, stage=stage)
    receipt["executed"] = True
    for offset, item in enumerate(receipt["jobs"]):
        item["job_id"] = str(first_job_id + offset)
    return _receipt_evidence(tmp_path, receipt, f"{stage}-receipt")


def _passing_gate_analysis(
    plan: dict, gate_receipt: dict, gate_evidence: dict
) -> dict:
    return {
        "schema": G1.SCHEMA,
        "run_id": plan["run_id"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "plan_sha256": G1._plan_sha256(plan),
        "rows": [
            {"case_id": "water2-tz", "passed": True},
            {"case_id": "water4-tz", "passed": True},
            {"case_id": "water8-tz", "passed": False},
        ],
        "receipts": {
            "gate": G1._receipt_summary(
                gate_receipt, gate_evidence, plan, "gate"
            ),
            "water8": None,
        },
        "decision": {"ready_for_water8": True},
    }


def test_execution_is_two_stage_and_water8_requires_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = _plan(tmp_path)
    ids = iter(("70001", "70002", "70003", "70004", "70005", "70006"))
    monkeypatch.setattr(
        G1.subprocess, "check_output", lambda *args, **kwargs: next(ids) + "\n"
    )

    with pytest.raises(ValueError, match="cannot execute in one step"):
        G1.submit_plan(plan, execute=True, stage="all")
    with pytest.raises(ValueError, match="requires a passing gate analysis"):
        G1.submit_plan(plan, execute=True, stage="water8")

    gate_receipt = G1.submit_plan(plan, execute=True, stage="gate")
    assert [item["job_id"] for item in gate_receipt["jobs"]] == [
        "70001", "70002", "70003", "70004"
    ]
    assert gate_receipt["jobs"][0]["case_id"] == "water2-tz"
    assert gate_receipt["jobs"][0]["method"] == "canonical_legacy"
    assert gate_receipt["jobs"][0]["repeat"] == (
        plan["jobs"][0]["exports"]["REPEAT"]
    )
    assert gate_receipt["jobs"][0]["expected_outputs"] == (
        plan["jobs"][0]["expected_outputs"]
    )
    gate_receipt, gate_evidence = _receipt_evidence(
        tmp_path, gate_receipt, "executed-gate-receipt"
    )
    water8 = G1.submit_plan(
        plan,
        execute=True,
        stage="water8",
        gate_analysis=_passing_gate_analysis(
            plan, gate_receipt, gate_evidence
        ),
        prior_receipt=gate_receipt,
        prior_receipt_evidence=gate_evidence,
    )
    assert "--dependency=afterok:70004" in water8["jobs"][0]["argv"]
    assert [item["job_id"] for item in water8["jobs"]] == ["70005", "70006"]


def _source(digest: str) -> dict:
    return {
        "tree_sha256": digest,
        "tree_sha256_at_end": digest,
        "stable_during_run": True,
        "formal_performance_eligible": True,
        "snapshot": {
            "tree_sha256": digest,
            "tree_sha256_at_end": digest,
            "stable_during_run": True,
            "read_only_at_end": True,
            "runtime_binaries_valid_at_end": True,
            "manifest": {
                "local_provenance": {"deployment_profile": "candidate"}
            },
        },
    }


def _hardware() -> dict:
    return {
        "host": "compute-1-6",
        "cpu_model": "AMD EPYC 7513 32-Core Processor",
        "affinity": list(range(24, 32)),
        "thread_environment": {
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "GPU4PYSCF_NUMA": "3",
        },
        "gpu": {
            "device_count": 1,
            "device_0": {
                "name": "NVIDIA A100-SXM4-80GB",
                "totalGlobalMem": 80 * 1024**3,
            },
        },
        "pcie": {
            "visible_gpus": [{
                "name": "NVIDIA A100-SXM4-80GB",
                "numa_node": "3",
                "pcie_gen_current": "4",
                "pcie_width_current": "16",
                "pci_bus_id": "00000000:01:00.0",
            }],
        },
    }


def _orbitals(
    case_id: str,
    digest: str,
    *,
    mode: str,
    artifact_path: str,
    artifact_sha256: str,
) -> dict:
    return {
        "schema": "gpu4pyscf.cc.canonical-orbitals.v1",
        "mode": mode,
        "applied_to_mean_field": True,
        "included_in_post_hf": False,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "orbital_fingerprint": "c" * 64,
        "identity": {"case_id": case_id},
        "producer": {"tree_sha256": digest},
    }


def _transfers(*, h2d: int, d2h: int, count: int, host_staging: int = 0) -> dict:
    h2d_count = count // 2
    d2h_count = count - h2d_count
    operations = {"h2d": {}, "d2h": {}}
    if host_staging:
        operations["d2h"]["direct-host-staging-test"] = {
            "bytes": host_staging,
            "count": 1,
        }
    return {
        "total_bytes": h2d + d2h,
        "total_transfers": count,
        "by_kind": {
            "h2d": {"bytes": h2d, "count": h2d_count},
            "d2h": {"bytes": d2h, "count": d2h_count},
        },
        "by_operation": operations,
    }


def _record(
    sync_root: Path,
    plan: dict,
    case_id: str,
    method: str,
    receipt_jobs: dict[str, dict],
) -> dict:
    job = next(
        item for item in plan["jobs"]
        if item["case_id"] == case_id and item["method"] == method
    )
    relative_json = Path(job["expected_outputs"][0]).relative_to(
        Path(plan["result_root"])
    )
    path = sync_root / relative_json
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = sync_root / Path(job["expected_outputs"][1]).relative_to(
        Path(plan["result_root"])
    )
    checkpoint_path.write_bytes(f"checkpoint:{job['id']}".encode("ascii"))
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    legacy_job = next(
        item for item in plan["jobs"]
        if item["case_id"] == case_id
        and item["method"] == "canonical_legacy"
    )
    artifact_remote = legacy_job["expected_outputs"][2]
    artifact_path = sync_root / Path(artifact_remote).relative_to(
        Path(plan["result_root"])
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if not artifact_path.exists():
        artifact_path.write_bytes(f"orbitals:{case_id}".encode("ascii"))
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    legacy = method == "canonical_legacy"
    final_residual_peak = 60 * 1024**3
    synchronized_peak = 61 * 1024**3

    def hbm_checkpoint(name: str, driver_used: int) -> dict:
        return {
            "name": name,
            "driver_used_bytes": driver_used,
            "driver_total_bytes": 80 * 1024**3,
            "cupy_pool_used_bytes": min(driver_used, 58 * 1024**3),
            "cupy_pool_total_bytes": min(driver_used, 59 * 1024**3),
            "timing_semantics": "cuda-synchronized-instantaneous",
            "measurement_scope": "exclusive-slurm-device-global",
            "source": "cudaMemGetInfo-and-cupy-default-memory-pool",
        }

    final_update_checkpoint_names = [
        "final-residual-internal-direct-transform-live",
        "resident-update-direct-output-live",
        "final-residual-internal-direct-output-live",
        "final-residual-internal-residual_fock_intermediates",
        "final-residual-internal-residual_oooo_ladder",
        "final-residual-internal-residual_linear_ov",
        "final-residual-internal-residual_wVooV_ring",
        "final-residual-internal-residual_fock_dressing",
        "final-residual-internal-residual_wVOov_ring",
        "final-residual-internal-residual_singles_doubles_coupling",
        "final-residual-internal-residual_one_body_doubles",
        "final-residual-internal-residual_denominator_and_symmetry",
        "final-residual-update-live",
    ]
    synchronized_checkpoints = None
    all_synchronized_checkpoints = None
    if not legacy:
        synchronized_checkpoints = [
            hbm_checkpoint(name, final_residual_peak)
            for name in final_update_checkpoint_names
        ]
        synchronized_checkpoints.append(hbm_checkpoint(
            "final-residual-block-0-8-live", 57 * 1024**3
        ))
        # Five timed CC cycles precede the diagnostic.  Its own resident-update
        # checkpoint is already included in the slice above.
        all_synchronized_checkpoints = [
            hbm_checkpoint(
                "resident-update-direct-output-live", synchronized_peak
            )
            for _ in range(5)
        ] + synchronized_checkpoints
    transfers = _transfers(
        h2d=1000 if legacy else 100,
        d2h=2000 if legacy else 200,
        count=30 if legacy else 20,
        host_staging=100 if legacy else 0,
    )
    if not legacy:
        for operation in (
            "resident-static-fock-upload",
            "resident-static-mo-energy-upload",
            "resident-static-mo-coeff-upload",
            "resident-static-oooo-upload",
            "resident-static-ovoo-upload",
            "resident-static-oovv-upload",
            "resident-static-ovvo-upload",
        ):
            transfers["by_operation"]["h2d"][operation] = {
                "bytes": 1,
                "count": 1,
            }
        for operation in (
            "final-residual-singles-equation-norm-download",
            "final-residual-doubles-equation-norm-download",
            "final-residual-singles-jacobi-update-norm-download",
            "final-residual-doubles-jacobi-update-norm-download",
            "final-residual-full-equation-norm-download",
            "final-residual-full-jacobi-update-norm-download",
        ):
            transfers["by_operation"]["d2h"][operation] = {
                "bytes": 8,
                "count": 1,
            }
    receipt_job = receipt_jobs[job["id"]]
    record = {
        "_g1_ab_input_path": str(path),
        "case_id": case_id,
        "method": method,
        "repeat": job["exports"]["REPEAT"],
        "status": "completed",
        "cc_converged": True,
        "cc_iterations": 5,
        "e_corr": -1.0,
        "e_tot": -100.0,
        "post_hf_seconds": 20.0 if legacy else 10.0,
        "output": job["expected_outputs"][0],
        "slurm": {
            "SLURM_JOB_ID": receipt_job["job_id"],
            "SLURM_JOB_NAME": job["id"],
            "SLURM_JOB_NODELIST": "compute-1-6",
            "SLURM_JOB_PARTITION": "mrigpu",
            "SLURM_CPUS_PER_TASK": "64",
        },
        "residual": {
            "available": True,
            "equation_norm": 2e-7,
            "jacobi_update_norm": 1e-7,
            "evaluation_backend": (
                "host-dense" if legacy else "gpu-blockwise-fp64"
            ),
            "doubles_block_size": None if legacy else 8,
            "active_nvir": None if legacy else 8,
            "temporary_workspace_bound_bytes": (
                None if legacy else 1024
            ),
            "released_staging_bytes": None if legacy else 2048,
            "final_residual_synchronized_driver_hbm_peak_bytes": (
                None if legacy else final_residual_peak
            ),
            "whole_stage_synchronized_driver_hbm_peak_bytes": (
                None if legacy else synchronized_peak
            ),
            "synchronized_hbm_checkpoints": synchronized_checkpoints,
        },
        "source": _source(plan["source_tree_sha256"]),
        "canonical_orbitals": _orbitals(
            case_id,
            plan["source_tree_sha256"],
            mode="written-and-reloaded" if legacy else "loaded",
            artifact_path=artifact_remote,
            artifact_sha256=artifact_sha,
        ),
        "checkpoint": {
            "path": job["expected_outputs"][1],
            "sha256": checkpoint_sha,
            "included_in_post_hf": True,
        },
        "hbm": {"peak_process_MiB": 60 * 1024, "sample_count": 2},
        "peak_host_RSS_GiB": 70.0,
        "run_metrics": {
            "counters": {
                "direct_host_staging_fallbacks": 5.0 if legacy else 0.0,
                "resident_iterations": 0.0 if legacy else 5.0,
                "direct_resident_staging_iterations": (
                    0.0 if legacy else 6.0
                ),
                "resident_final_residual_updates": 0.0 if legacy else 1.0,
                "final_residual_staging_release_events": (
                    0.0 if legacy else 1.0
                ),
                "final_residual_released_staging_bytes": (
                    0.0 if legacy else 2048.0
                ),
                "final_residual_temporary_bound_bytes": (
                    0.0 if legacy else 1024.0
                ),
                "synchronized_driver_hbm_peak_bytes": (
                    0.0 if legacy else float(synchronized_peak)
                ),
                "resident_update_hbm_checkpoint_count": (
                    0.0 if legacy else 6.0
                ),
            },
            "transfers": transfers,
            "metadata": {
                "synchronized_hbm_checkpoints": all_synchronized_checkpoints
            } if not legacy else {},
        },
    }
    record.update(_hardware())
    serialized = {
        key: value for key, value in record.items()
        if key != "_g1_ab_input_path"
    }
    path.write_text(json.dumps(serialized, sort_keys=True), encoding="utf-8")
    return record


def _records(
    tmp_path: Path,
    plan: dict,
    gate_receipt: dict,
    water8_receipt: dict,
) -> list[dict]:
    receipt_jobs = {
        item["job"]: item
        for receipt in (gate_receipt, water8_receipt)
        for item in receipt["jobs"]
    }
    sync_root = tmp_path / "synchronized" / plan["run_id"]
    return [
        _record(sync_root, plan, case_id, method, receipt_jobs)
        for case_id in G1.CASES for method in G1.METHODS
    ]


def test_analysis_uses_strict_comparator_and_unlocks_water8(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = _plan(tmp_path)
    gate_receipt, gate_evidence = _receipt(
        tmp_path, plan, "gate", first_job_id=71001
    )
    water8_receipt, water8_evidence = _receipt(
        tmp_path, plan, "water8", first_job_id=71005
    )
    calls: list[dict] = []

    def compare(*args, **kwargs):
        calls.append(kwargs)
        return {"passed": True, "checks": {"amplitudes": {"passed": True}}}

    monkeypatch.setattr(G1._checkpoint_comparator, "compare_checkpoints", compare)

    result = G1.analyze_g1_ab(
        _records(tmp_path, plan, gate_receipt, water8_receipt),
        plan,
        gate_receipt=gate_receipt,
        gate_receipt_evidence=gate_evidence,
        water8_receipt=water8_receipt,
        water8_receipt_evidence=water8_evidence,
    )

    assert result["decision"] == {
        "small_case_gate_passed": True,
        "ready_for_water8": True,
        "water8_complete": True,
        "final_g1_passed": True,
        "water8_submission_policy": (
            "requires ready_for_water8=true plus an executed small-case receipt"
        ),
    }
    assert all(row["passed"] for row in result["rows"])
    assert len(calls) == 3
    assert all(call["energy_tol"] == 1e-8 for call in calls)
    assert all(call["residual_tol"] == 1e-6 for call in calls)
    assert all(call["amplitude_max_tol"] == 1e-6 for call in calls)
    assert all(call["amplitude_l2_tol"] == 1e-6 for call in calls)


def test_analysis_fails_closed_on_fallback_resource_and_energy(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = _plan(tmp_path)
    gate_receipt, gate_evidence = _receipt(
        tmp_path, plan, "gate", first_job_id=72001
    )
    water8_receipt, water8_evidence = _receipt(
        tmp_path, plan, "water8", first_job_id=72005
    )
    monkeypatch.setattr(
        G1._checkpoint_comparator,
        "compare_checkpoints",
        lambda *args, **kwargs: {"passed": True},
    )
    records = _records(tmp_path, plan, gate_receipt, water8_receipt)
    resident_water4 = next(
        item for item in records
        if item["case_id"] == "water4-tz" and item["method"] == "canonical"
    )
    resident_water4["run_metrics"]["counters"][
        "direct_host_staging_fallbacks"
    ] = 1.0
    resident_water4["hbm"]["peak_process_MiB"] = 73 * 1024
    resident_water4["e_tot"] += 2e-8

    result = G1.analyze_g1_ab(
        records,
        plan,
        gate_receipt=gate_receipt,
        gate_receipt_evidence=gate_evidence,
        water8_receipt=water8_receipt,
        water8_receipt_evidence=water8_evidence,
    )
    row = next(item for item in result["rows"] if item["case_id"] == "water4-tz")

    assert row["passed"] is False
    assert row["energy"]["passed"] is False
    assert row["resident_resources"]["passed"] is False
    assert row["residency_and_transfers"]["passed"] is False
    assert result["decision"]["ready_for_water8"] is False


def test_sparse_zero_event_counter_is_accepted_with_complete_transfer_ledger(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = _plan(tmp_path)
    gate_receipt, gate_evidence = _receipt(
        tmp_path, plan, "gate", first_job_id=73001
    )
    water8_receipt, water8_evidence = _receipt(
        tmp_path, plan, "water8", first_job_id=73005
    )
    monkeypatch.setattr(
        G1._checkpoint_comparator,
        "compare_checkpoints",
        lambda *args, **kwargs: {"passed": True},
    )
    records = _records(tmp_path, plan, gate_receipt, water8_receipt)
    for record in records:
        if record["method"] == "canonical":
            del record["run_metrics"]["counters"][
                "direct_host_staging_fallbacks"
            ]

    result = G1.analyze_g1_ab(
        records,
        plan,
        gate_receipt=gate_receipt,
        gate_receipt_evidence=gate_evidence,
        water8_receipt=water8_receipt,
        water8_receipt_evidence=water8_evidence,
    )

    assert all(row["passed"] for row in result["rows"])
    assert all(
        row["residency_and_transfers"][
            "direct_host_staging_counter_serialized"
        ] is False
        for row in result["rows"]
    )


def test_residency_gate_rejects_host_final_residual_regression(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    gate_receipt, _evidence = _receipt(
        tmp_path, plan, "gate", first_job_id=73501
    )
    receipt_jobs = {item["job"]: item for item in gate_receipt["jobs"]}
    sync_root = tmp_path / "synchronized" / plan["run_id"]
    legacy = _record(
        sync_root, plan, "water2-tz", "canonical_legacy", receipt_jobs
    )
    resident = _record(
        sync_root, plan, "water2-tz", "canonical", receipt_jobs
    )
    resident["residual"]["evaluation_backend"] = "host-dense"
    resident["run_metrics"]["counters"][
        "direct_resident_staging_iterations"
    ] = resident["cc_iterations"]
    del resident["run_metrics"]["transfers"]["by_operation"]["d2h"][
        "final-residual-full-equation-norm-download"
    ]

    gate = G1._residency_and_transfer_gate(legacy, resident)

    assert gate["passed"] is False
    assert gate["checks"]["final_residual_gpu_blockwise"] is False
    assert gate["checks"][
        "direct_staging_covers_iterations_and_final_residual"
    ] is False
    assert gate["checks"]["final_residual_norm_scalars_accounted"] is False


def test_residency_gate_rejects_transient_hbm_peak_missed_by_sampler(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    gate_receipt, _evidence = _receipt(
        tmp_path, plan, "gate", first_job_id=73601
    )
    receipt_jobs = {item["job"]: item for item in gate_receipt["jobs"]}
    sync_root = tmp_path / "synchronized" / plan["run_id"]
    legacy = _record(
        sync_root, plan, "water2-tz", "canonical_legacy", receipt_jobs
    )
    resident = _record(
        sync_root, plan, "water2-tz", "canonical", receipt_jobs
    )
    transient_peak = 72 * 1024**3 + 1
    # The sampled process maximum remains a seemingly valid 60 GiB.
    assert resident["hbm"]["peak_process_MiB"] == 60 * 1024
    resident["run_metrics"]["metadata"][
        "synchronized_hbm_checkpoints"
    ][0]["driver_used_bytes"] = transient_peak
    resident["residual"][
        "whole_stage_synchronized_driver_hbm_peak_bytes"
    ] = (
        transient_peak
    )
    resident["run_metrics"]["counters"][
        "synchronized_driver_hbm_peak_bytes"
    ] = float(transient_peak)

    gate = G1._residency_and_transfer_gate(legacy, resident)

    assert gate["passed"] is False
    assert gate["checks"]["synchronized_hbm_checkpoints_complete"] is True
    assert gate["checks"]["synchronized_hbm_within_budget"] is False


def test_old_record_with_same_repeat_but_wrong_job_and_paths_is_rejected(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = _plan(tmp_path)
    gate_receipt, gate_evidence = _receipt(
        tmp_path, plan, "gate", first_job_id=74001
    )
    water8_receipt, water8_evidence = _receipt(
        tmp_path, plan, "water8", first_job_id=74005
    )
    monkeypatch.setattr(
        G1._checkpoint_comparator,
        "compare_checkpoints",
        lambda *args, **kwargs: {"passed": True},
    )
    records = _records(tmp_path, plan, gate_receipt, water8_receipt)
    stale = next(
        item for item in records
        if item["case_id"] == "water2-tz" and item["method"] == "canonical"
    )
    # The old record deliberately retains the expected repeat, which was the
    # only selection key before receipt binding was added.
    stale["slurm"]["SLURM_JOB_ID"] = "63999"
    stale["output"] = stale["output"].replace("water2-tz", "old-water2", 1)
    stale["checkpoint"]["path"] = stale["checkpoint"]["path"].replace(
        "water2-tz", "old-water2", 1
    )

    result = G1.analyze_g1_ab(
        records,
        plan,
        gate_receipt=gate_receipt,
        gate_receipt_evidence=gate_evidence,
        water8_receipt=water8_receipt,
        water8_receipt_evidence=water8_evidence,
    )
    row = result["rows"][0]
    binding = row["receipt_and_files"]["resident"]

    assert row["passed"] is False
    assert binding["checks"]["slurm_job_id"] is False
    assert binding["checks"]["record_output"] is False
    assert binding["checks"]["checkpoint_declared_path"] is False


def test_duplicate_copied_record_with_same_repeat_is_rejected(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = _plan(tmp_path)
    gate_receipt, gate_evidence = _receipt(
        tmp_path, plan, "gate", first_job_id=75001
    )
    water8_receipt, water8_evidence = _receipt(
        tmp_path, plan, "water8", first_job_id=75005
    )
    monkeypatch.setattr(
        G1._checkpoint_comparator,
        "compare_checkpoints",
        lambda *args, **kwargs: {"passed": True},
    )
    records = _records(tmp_path, plan, gate_receipt, water8_receipt)
    duplicate = copy.deepcopy(records[0])
    copied_path = Path(duplicate["_g1_ab_input_path"]).with_name("copied.json")
    copied_path.write_text("{}", encoding="utf-8")
    duplicate["_g1_ab_input_path"] = str(copied_path)
    records.append(duplicate)

    result = G1.analyze_g1_ab(
        records,
        plan,
        gate_receipt=gate_receipt,
        gate_receipt_evidence=gate_evidence,
        water8_receipt=water8_receipt,
        water8_receipt_evidence=water8_evidence,
    )

    assert result["rows"][0]["complete"] is False
    assert any("found 2" in reason for reason in result["rows"][0]["reasons"])


def test_copied_record_at_wrong_local_result_path_is_rejected(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = _plan(tmp_path)
    gate_receipt, gate_evidence = _receipt(
        tmp_path, plan, "gate", first_job_id=75501
    )
    water8_receipt, water8_evidence = _receipt(
        tmp_path, plan, "water8", first_job_id=75505
    )
    monkeypatch.setattr(
        G1._checkpoint_comparator,
        "compare_checkpoints",
        lambda *args, **kwargs: {"passed": True},
    )
    records = _records(tmp_path, plan, gate_receipt, water8_receipt)
    copied = next(
        item for item in records
        if item["case_id"] == "water4-tz"
        and item["method"] == "canonical"
    )
    original = Path(copied["_g1_ab_input_path"])
    wrong_path = tmp_path / "flattened-copy" / original.name
    wrong_path.parent.mkdir(parents=True)
    wrong_path.write_bytes(original.read_bytes())
    copied["_g1_ab_input_path"] = str(wrong_path)

    result = G1.analyze_g1_ab(
        records,
        plan,
        gate_receipt=gate_receipt,
        gate_receipt_evidence=gate_evidence,
        water8_receipt=water8_receipt,
        water8_receipt_evidence=water8_evidence,
    )
    binding = result["rows"][1]["receipt_and_files"]["resident"]

    assert result["rows"][1]["passed"] is False
    assert binding["checks"]["local_file_binding"] is False
    assert "planned result-tree suffix" in binding["error"]


def test_gate_analysis_binds_receipt_content_but_allows_path_migration(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    gate_receipt, gate_evidence = _receipt(
        tmp_path, plan, "gate", first_job_id=76001
    )
    analysis = _passing_gate_analysis(plan, gate_receipt, gate_evidence)

    copied_receipt, copied_evidence = _receipt_evidence(
        tmp_path / "copied", gate_receipt, "gate-receipt-copy"
    )
    preview = G1.submit_plan(
        plan,
        execute=False,
        stage="water8",
        gate_analysis=analysis,
        prior_receipt=copied_receipt,
        prior_receipt_evidence=copied_evidence,
    )
    assert "--dependency=afterok:76004" in preview["jobs"][0]["argv"]

    changed = copy.deepcopy(gate_receipt)
    changed["jobs"][0]["job_id"] = "76991"
    changed_receipt, changed_evidence = _receipt_evidence(
        tmp_path / "changed", changed, "gate-receipt-changed"
    )
    with pytest.raises(ValueError, match="not bound"):
        G1.submit_plan(
            plan,
            execute=False,
            stage="water8",
            gate_analysis=analysis,
            prior_receipt=changed_receipt,
            prior_receipt_evidence=changed_evidence,
        )


def test_final_analysis_requires_executed_water8_receipt(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = _plan(tmp_path)
    gate_receipt, gate_evidence = _receipt(
        tmp_path, plan, "gate", first_job_id=77001
    )
    water8_receipt, _ = _receipt(
        tmp_path, plan, "water8", first_job_id=77005
    )
    monkeypatch.setattr(
        G1._checkpoint_comparator,
        "compare_checkpoints",
        lambda *args, **kwargs: {"passed": True},
    )

    result = G1.analyze_g1_ab(
        _records(tmp_path, plan, gate_receipt, water8_receipt),
        plan,
        gate_receipt=gate_receipt,
        gate_receipt_evidence=gate_evidence,
    )

    assert result["decision"]["ready_for_water8"] is True
    assert result["decision"]["water8_complete"] is False
    assert result["decision"]["final_g1_passed"] is False
    assert "executed receipt is missing" in result["rows"][2]["reasons"][0]


@pytest.mark.parametrize("dangling_symlink", [False, True])
def test_existing_receipt_preflight_makes_zero_sbatch_calls(
    tmp_path: Path, monkeypatch, dangling_symlink: bool,
) -> None:
    plan_path = tmp_path / "plan.json"
    G1._json_write_once(plan_path, _plan(tmp_path))
    receipt_path = tmp_path / "occupied-receipt.json"
    if dangling_symlink:
        receipt_path.symlink_to(tmp_path / "missing-target")
    else:
        receipt_path.write_text("occupied", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        G1.subprocess,
        "check_output",
        lambda argv, **kwargs: calls.append(argv) or "80001\n",
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        G1.main([
            "submit", "--plan", str(plan_path), "--stage", "gate",
            "--execute", "--receipt", str(receipt_path),
        ])

    assert calls == []


def test_write_once_refuses_existing_plan_or_receipt(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    G1._json_write_once(output, {"value": 1})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        G1._json_write_once(output, {"value": 2})


@pytest.mark.parametrize(
    "dependency", ["62434", "afterany:62434", "afterok:0", "afterok:abc"]
)
def test_initial_dependency_is_fail_closed(tmp_path: Path, dependency: str) -> None:
    task, source = _task_and_source(tmp_path)
    with pytest.raises(ValueError, match="initial dependency"):
        G1.make_plan(
            task_root=task,
            source_root=source,
            run_id="g1-test",
            initial_dependency=dependency,
        )
