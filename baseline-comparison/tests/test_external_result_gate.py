from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sys

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import external_result_gate as gate


GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> dict:
    return {
        "cases": [
            {
                "id": "water2-tz",
                "role": "sanity",
                "source": "WATER27:H2O2",
                "basis": "cc-pVTZ",
                "expected_dimensions": {"nao": 116, "nocc": 10, "nvir": 106},
                "geometry_angstrom": [
                    ["O", -1.3305065, 0.9990741, 0.0],
                    ["H", -1.0635455, 1.9210871, 0.0],
                ],
            },
            {
                "id": "water4-tz",
                "role": "development",
                "source": "WATER27:H2O4",
                "basis": "cc-pVTZ",
                "expected_dimensions": {"nao": 232, "nocc": 20, "nvir": 212},
                "geometry_angstrom": [["O", 0.0, 0.0, 0.0]],
            },
        ]
    }


def geometry_sha(document: dict) -> str:
    record = document["cases"][0]
    encoded = json.dumps(
        record["geometry_angstrom"], separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def topology(job_id: str = "70001", node: str = "compute-1-2") -> dict:
    return {
        "hostname": node,
        "cpu_model": "AMD EPYC 7513 32-Core Processor",
        "cpu_affinity": list(range(24, 32)),
        "unique_physical_cores": 8,
        "allocated_gpu_query_selector": "3",
        "visible_gpus": [
            {
                "name": "NVIDIA A100-SXM4-80GB",
                "uuid": GPU_UUID,
                "pci_bus_id": "00000000:81:00.0",
                "sysfs_bus_id": "0000:81:00.0",
                "memory_total_mib": 81920,
                "numa_node": "3",
                "current_link_width": "16",
                "current_link_speed": "16.0 GT/s PCIe",
            }
        ],
        "slurm": {
            "SLURM_JOB_ID": job_id,
            "SLURM_JOB_NODELIST": node,
        },
    }


def oracle(case_doc: dict, e_corr: float = -0.5644876834) -> dict:
    tree = "a" * 64
    return {
        "protocol_schema": "gpu4pyscf.water8-ccsd-a100.v2",
        "status": "completed",
        "method": "canonical",
        "case_id": "water2-tz",
        "case": case_doc["cases"][0],
        "geometry_sha256": geometry_sha(case_doc),
        "orbital_dimensions": {"nao": 116, "nocc": 10, "nvir": 106},
        "cc_converged": True,
        "e_corr": e_corr,
        "source": {
            "revision": f"tree-sha256:{tree}",
            "formal_performance_eligible": True,
            "snapshot": {
                "tree_sha256": tree,
                "valid": True,
                "valid_at_start": True,
                "valid_at_end": True,
                "stable_during_run": True,
                "read_only_at_start": True,
                "read_only_at_end": True,
            },
        },
        "checkpoint": {
            "included_in_post_hf": True,
            "sha256": "b" * 64,
        },
        "residual": {
            "available": True,
            "is_full_space": True,
            "included_in_post_hf": True,
            "equation_norm": 5e-8,
        },
    }


def byteqc_fixture(tmp_path: Path) -> dict[str, Path | dict]:
    case_doc = cases()
    cases_path = write_json(tmp_path / "cases.json", case_doc)
    runner = tmp_path / "byteqc_benchmark.py"
    runner.write_text("# pinned byteqc runner\n", encoding="utf-8")
    digests = {
        "byteqc_benchmark.py": sha(runner),
        "cases": sha(cases_path),
        "libgint": "2" * 64,
        "libgvhf": "3" * 64,
    }
    result = {
        "schema": "agent-qc.single-a100-byteqc-rccsd.v2",
        "status": "completed",
        "case": case_doc["cases"][0],
        "geometry_sha256": geometry_sha(case_doc),
        "dimensions": {"nao": 116, "nocc": 10, "nvir": 106},
        "comparison_contract": {
            "software": "ByteQC",
            "release_commit": gate.SOFTWARE["byteqc"]["release"],
            "acceptance_table": "S-candidate-pending-oracle-validation",
            "density_fitting": False,
            "precision": "FP64",
            "basis": "cc-pVTZ",
            "spherical": True,
            "charge": 0,
            "spin": 0,
            "frozen_occupied": 0,
            "scf_excluded_from_timer": True,
            "eri_representation": "exact four-index ERIs",
            "normal_checkpoint": False,
            "separate_final_full_space_residual": False,
        },
        "runtime_method_classifier": {
            "verified": True,
            "actual_class": "byteqc.cucc.ccsd.RCCSD",
        },
        "provenance": {
            "source_root": "/remote/byteqc",
            "head_commit": gate.SOFTWARE["byteqc"]["release"],
            "expected_commit": gate.SOFTWARE["byteqc"]["release"],
            "tracked_source_clean": True,
            "harness": {"sha256": digests["byteqc_benchmark.py"]},
            "cases": {"path": "/remote/byteqc/cases.json", "sha256": digests["cases"]},
            "native_libraries": {
                "libgint.so": {"sha256": digests["libgint"]},
                "libgvhf.so": {"sha256": digests["libgvhf"]},
            },
        },
        "bound_topology": topology(),
        "execution_gpu_identity": {"verified": True},
        "gpu_telemetry": {
            "identity_verified": True,
            "allocated_gpu_uuid": GPU_UUID,
            "sample_count": 3,
            "peak_memory_used_mib": 2048.0,
        },
        "host_max_rss_gib": 9.5,
        "timing_seconds": {"post_hf_contract": 7.5},
        "timing_definition": {"post_hf_contract": "setup + AO2MO + all iterations"},
        "scf": {"converged": True, "energy_eh": -152.1205477194},
        "convergence": {
            "converged": True,
            "conv_tol": 1e-8,
            "conv_tol_normt": 1e-6,
            "max_cycle": 50,
            "cycles_observed": 10,
            "history": [{"update_norm": 5e-7}],
        },
        "energies_eh": {"e_corr": -0.56448768339},
    }
    scheduler = {
        "terminal_evidence": {
            "job_id": "70001",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "node": "compute-1-2",
        },
        "source_sha256": digests,
    }
    capability = {
        "schema": gate.CAPABILITY_SCHEMA,
        "software": "byteqc",
        "result_schema": gate.SOFTWARE["byteqc"]["result_schema"],
        "runner_sha256": sha(runner),
        "supported_cases": ["water2-tz", "water4-tz"],
        "water4_argv_supported": True,
    }
    paths = {
        "runner": runner,
        "result": write_json(tmp_path / "result.json", result),
        "scheduler": write_json(tmp_path / "scheduler.json", scheduler),
        "cases": cases_path,
        "capabilities": write_json(tmp_path / "capabilities.json", capability),
        "oracle": write_json(tmp_path / "oracle.json", oracle(case_doc)),
    }
    return {**paths, "result_payload": result, "scheduler_payload": scheduler}


def run_byteqc_audit(bundle: dict, **overrides) -> dict:
    kwargs = {
        "software": "byteqc",
        "result_path": bundle["result"],
        "scheduler_path": bundle["scheduler"],
        "cases_path": bundle["cases"],
        "capabilities_path": bundle["capabilities"],
        "oracle_path": bundle["oracle"],
        "energy_tolerance": 1e-8,
    }
    kwargs.update(overrides)
    return gate.audit_external_result(**kwargs)


def gansu_fixture(tmp_path: Path) -> dict:
    case_doc = cases()
    cases_path = write_json(tmp_path / "gansu-cases.json", case_doc)
    runner = tmp_path / "gansu_benchmark.py"
    runner.write_text("# pinned gansu runner\n", encoding="utf-8")
    keys = (
        "harness",
        "cases",
        "basis",
        "auxiliary_basis",
        "native_library",
        "python_wrapper",
        "native_metadata",
        "libgfortran",
        "libquadmath",
        "distribution_metadata",
        "distribution_record",
        "distribution_wheel",
        "cuda_runtime",
    )
    digests = {key: f"{index:x}" * 64 for index, key in enumerate(keys, start=1)}
    digests["harness"] = sha(runner)
    digests["cases"] = sha(cases_path)
    files = {key: {"path": f"/remote/{key}", "sha256": value} for key, value in digests.items()}
    result = {
        "schema": "agent-qc.single-a100-gansu-ri-rccsd.v2",
        "status": "complete",
        "case": case_doc["cases"][0],
        "geometry_sha256": geometry_sha(case_doc),
        "dimensions": {"nao": 116, "nocc": 10, "nvir": 106, "nelectron": 20},
        "method": {
            "software": "GANSU",
            "version": "2026.8.10",
            "acceptance_table": "I",
            "canonical_four_index": False,
            "density_fitting": True,
            "thc": False,
        },
        "backend_verification": {
            "requested_eri_method": "ri",
            "requested_bnative_environment": True,
            "single_gpu_gate_passed": True,
            "ccsd_callback_shape_consistent_with_bnative": True,
        },
        "provenance": {
            "release": {"official_tag_commit": gate.SOFTWARE["gansu"]["release"]},
            "runtime_root": "/remote/venv",
            "files": files,
        },
        "bound_topology": topology(job_id="70002", node="compute-1-3"),
        "cuda_execution_identity": {
            "verified_before_gansu_init": True,
            "verified_before_public_run": True,
            "validation_errors": [],
        },
        "gpu_monitor": {
            "identity_verified": True,
            "expected_uuid": GPU_UUID,
            "sample_count": 4,
            "peak_memory_used_mib": 1024.0,
        },
        "resources": {"peak_hbm_mib": 1024.0, "host_max_rss_gib": 4.0},
        "resource_budget_violations": [],
        "timing_seconds": {
            "whole_rhf_ri_rccsd_public_run_wall": 12.0,
            "primary_reportable_seconds": 12.0,
        },
        "timing_comparability": {"primary_public_whole_run_reportable": True},
        "convergence": {
            "scf_converged_inferred": True,
            "ccsd_converged_inferred": True,
            "ccsd_callback_shape_consistent_with_bnative": True,
            "ccsd_threshold_eh": 1e-10,
            "ccsd_callback_count": 8,
        },
        "energies_eh": {"ccsd_correlation": -0.56448768339},
    }
    scheduler = {
        "job": {
            "id": 70002,
            "scontrol": (
                "JobId=70002 JobState=COMPLETED ExitCode=0:0 "
                "NodeList=compute-1-3"
            ),
        },
        "files": {"runtime_assets": digests},
    }
    capability = {
        "schema": gate.CAPABILITY_SCHEMA,
        "software": "gansu",
        "result_schema": gate.SOFTWARE["gansu"]["result_schema"],
        "runner_sha256": sha(runner),
        "supported_cases": ["water2-tz"],
        "water4_argv_supported": False,
    }
    return {
        "runner": runner,
        "result": write_json(tmp_path / "gansu-result.json", result),
        "scheduler": write_json(tmp_path / "gansu-scheduler.json", scheduler),
        "cases": cases_path,
        "capabilities": write_json(tmp_path / "gansu-capabilities.json", capability),
        "oracle": write_json(tmp_path / "gansu-oracle.json", oracle(case_doc)),
    }


def test_valid_byteqc_is_s_l1_w4_candidate_but_not_formal_v1(tmp_path: Path):
    bundle = byteqc_fixture(tmp_path)
    decision = run_byteqc_audit(bundle)

    assert decision["water2_gate_passed"] is True
    assert decision["w4_eligible"] is True
    assert decision["classification"]["acceptance_table"] == "S"
    assert decision["classification"]["evidence_level"] == "L1"
    formal = decision["classification"]["formal_v1_post_hf_comparability"]
    assert formal["eligible"] is False
    assert any("checkpoint" in reason for reason in formal["reasons"])
    assert any("full-space residual" in reason for reason in formal["reasons"])
    assert any("source-tree content SHA-256" in reason for reason in formal["reasons"])
    assert decision["gates"]["numerical_oracle"]["absolute_error_eh"] < 1e-8


@pytest.mark.parametrize(
    ("mutator", "gate_name"),
    [
        (lambda result: result.update(status="failed"), "result_schema_status"),
        (lambda result: result["timing_seconds"].update(post_hf_contract=0.0), "timing"),
        (lambda result: result["case"].update(source="invented"), "case_contract"),
        (lambda result: result["dimensions"].update(nvir=105), "case_contract"),
        (lambda result: result["bound_topology"].update(hostname="compute-1-5"), "topology"),
        (lambda result: result["gpu_telemetry"].update(peak_memory_used_mib=73729), "resource_budget"),
        (lambda result: result["convergence"].update(converged=False), "convergence"),
    ],
)
def test_result_tampering_fails_closed(tmp_path: Path, mutator, gate_name: str):
    bundle = byteqc_fixture(tmp_path)
    result = json.loads(Path(bundle["result"]).read_text())
    mutator(result)
    write_json(Path(bundle["result"]), result)
    decision = run_byteqc_audit(bundle)
    assert decision["w4_eligible"] is False
    assert decision["gates"][gate_name]["passed"] is False


def test_source_sha_job_node_and_gpu_evidence_are_all_bound(tmp_path: Path):
    bundle = byteqc_fixture(tmp_path)
    result = json.loads(Path(bundle["result"]).read_text())
    result["provenance"]["native_libraries"]["libgint.so"]["sha256"] = "f" * 64
    result["bound_topology"]["slurm"]["SLURM_JOB_ID"] = "other"
    result["bound_topology"]["visible_gpus"][0]["uuid"] = "bad"
    write_json(Path(bundle["result"]), result)
    decision = run_byteqc_audit(bundle)
    assert decision["gates"]["source_receipt_binding"]["passed"] is False
    assert decision["gates"]["scheduler_terminal"]["passed"] is False
    assert decision["gates"]["topology"]["passed"] is False


def test_pending_scheduler_is_not_terminal_evidence(tmp_path: Path):
    bundle = byteqc_fixture(tmp_path)
    scheduler = json.loads(Path(bundle["scheduler"]).read_text())
    scheduler["terminal_evidence"].update(state="PENDING", exit_code=None, node=None)
    write_json(Path(bundle["scheduler"]), scheduler)
    decision = run_byteqc_audit(bundle)
    assert decision["water2_gate_passed"] is False
    assert decision["gates"]["scheduler_terminal"]["passed"] is False


@pytest.mark.parametrize(
    "oracle_path,tolerance",
    [(None, 1e-8), ("present", None), ("present", 1e-3)],
)
def test_oracle_and_explicit_project_bounded_tolerance_are_required(
    tmp_path: Path, oracle_path, tolerance
):
    bundle = byteqc_fixture(tmp_path)
    selected = bundle["oracle"] if oracle_path == "present" else None
    decision = run_byteqc_audit(
        bundle, oracle_path=selected, energy_tolerance=tolerance
    )
    assert decision["gates"]["numerical_oracle"]["passed"] is False
    assert decision["w4_eligible"] is False


def test_oracle_must_have_immutable_source_checkpoint_and_residual(tmp_path: Path):
    bundle = byteqc_fixture(tmp_path)
    value = json.loads(Path(bundle["oracle"]).read_text())
    value["source"]["snapshot"]["stable_during_run"] = False
    value.pop("checkpoint")
    value["residual"]["equation_norm"] = 2e-6
    write_json(Path(bundle["oracle"]), value)
    decision = run_byteqc_audit(bundle)
    assert decision["gates"]["numerical_oracle"]["passed"] is False
    assert len(decision["gates"]["numerical_oracle"]["reasons"]) >= 3


def test_gansu_is_i_l2_diagnostic_and_current_runner_refuses_water4(tmp_path: Path):
    bundle = gansu_fixture(tmp_path)
    decision = gate.audit_external_result(
        software="gansu",
        result_path=bundle["result"],
        scheduler_path=bundle["scheduler"],
        cases_path=bundle["cases"],
        capabilities_path=bundle["capabilities"],
        oracle_path=bundle["oracle"],
        energy_tolerance=1e-8,
    )
    assert decision["water2_gate_passed"] is False
    assert decision["w4_eligible"] is False
    assert decision["w4_blocking_gates"] == [
        "convergence",
        "water4_runner_capability",
    ]
    assert any(
        "no final observed CCSD residual" in reason
        for reason in decision["gates"]["convergence"]["reasons"]
    )
    assert decision["classification"]["acceptance_table"] == "I"
    assert decision["classification"]["evidence_level"] == "L2"
    assert decision["classification"]["formal_v1_post_hf_comparability"]["eligible"] is False


def test_missing_or_invalid_required_fields_never_pass(tmp_path: Path):
    bundle = byteqc_fixture(tmp_path)
    result = json.loads(Path(bundle["result"]).read_text())
    del result["convergence"]["history"]
    del result["energies_eh"]["e_corr"]
    result["host_max_rss_gib"] = "not-a-number"
    write_json(Path(bundle["result"]), result)
    decision = run_byteqc_audit(bundle)
    assert decision["gates"]["convergence"]["passed"] is False
    assert decision["gates"]["resource_budget"]["passed"] is False
    assert decision["gates"]["numerical_oracle"]["passed"] is False


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize(
    "evidence_name",
    ["result", "scheduler", "cases", "capabilities", "oracle"],
)
def test_every_json_evidence_input_rejects_nonstandard_numeric_constants(
    tmp_path: Path, evidence_name: str, token: str
):
    bundle = byteqc_fixture(tmp_path)
    Path(bundle[evidence_name]).write_text(
        '{"illegal_constant": ' + token + "}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-standard JSON numeric constant"):
        run_byteqc_audit(bundle)


def test_canonical_payload_hash_refuses_nonfinite_python_values():
    with pytest.raises(ValueError, match="Out of range float values"):
        gate._payload_sha256({"coordinate": float("nan")})


@pytest.mark.parametrize(
    "bad_geometry",
    [
        [["O", 0.0, 0.0]],
        [["not-an-atom", 0.0, 0.0, 0.0]],
        [["O", 0.0, "bad", 0.0]],
        ["not-a-row"],
        [],
    ],
)
def test_malformed_geometry_rows_fail_the_case_gate(tmp_path: Path, bad_geometry):
    bundle = byteqc_fixture(tmp_path)
    result = json.loads(Path(bundle["result"]).read_text())
    result["case"]["geometry_angstrom"] = bad_geometry
    write_json(Path(bundle["result"]), result)
    decision = run_byteqc_audit(bundle)
    assert decision["gates"]["case_contract"]["passed"] is False
    assert decision["w4_eligible"] is False


def test_valid_byteqc_plan_is_manifest_only_and_contains_water4_argv(tmp_path: Path):
    bundle = byteqc_fixture(tmp_path)
    decision = run_byteqc_audit(bundle)
    decision_path = write_json(tmp_path / "decision.json", decision)
    output_dir = tmp_path / "future-results"
    plan = gate.make_water4_plan(
        decision_path=decision_path,
        runner_path=bundle["runner"],
        python_executable=Path(sys.executable),
        run_id="byteqc-w4-a",
        output_dir=output_dir,
    )
    assert plan["w4_eligible"] is True
    assert plan["dry_run"] is True
    assert plan["would_submit"] is False
    assert plan["submission_argv"] is None
    assert plan["audit_recomputation"]["exact_payload_match"] is True
    assert plan["audit_recomputation"]["semantic_fields_match"] is True
    assert not output_dir.exists()
    argv = plan["job"]["benchmark_argv"]
    assert argv[argv.index("--case") + 1] == "water4-tz"
    assert "sbatch" not in argv
    assert plan["job"]["environment"]["BYTEQC_EXPECTED_HARNESS_SHA256"] == sha(bundle["runner"])


def test_plan_emits_no_argv_when_water2_or_capability_gate_failed(tmp_path: Path):
    bundle = byteqc_fixture(tmp_path)
    decision = run_byteqc_audit(bundle, oracle_path=None)
    decision_path = write_json(tmp_path / "decision.json", decision)
    plan = gate.make_water4_plan(
        decision_path=decision_path,
        runner_path=bundle["runner"],
        python_executable=Path(sys.executable),
        run_id="refused-w4",
        output_dir=tmp_path / "future-results",
    )
    assert plan["w4_eligible"] is False
    assert "job" not in plan
    assert any("WATER2 audit" in reason for reason in plan["refusal_reasons"])


def test_plan_recomputes_real_failed_result_and_rejects_tampered_flags(tmp_path: Path):
    bundle = byteqc_fixture(tmp_path)
    result = json.loads(Path(bundle["result"]).read_text())
    result["status"] = "failed"
    write_json(Path(bundle["result"]), result)
    decision = run_byteqc_audit(bundle)
    assert decision["w4_eligible"] is False

    decision["water2_gate_passed"] = True
    decision["water2_blocking_gates"] = []
    decision["w4_eligible"] = True
    decision["w4_blocking_gates"] = []
    decision["w4_reasons"] = []
    decision_path = write_json(tmp_path / "forged-decision.json", decision)
    plan = gate.make_water4_plan(
        decision_path=decision_path,
        runner_path=bundle["runner"],
        python_executable=Path(sys.executable),
        run_id="forged-w4",
        output_dir=tmp_path / "future-results",
    )
    assert plan["w4_eligible"] is False
    assert "job" not in plan
    assert plan["audit_recomputation"]["exact_payload_match"] is False
    assert plan["audit_recomputation"]["semantic_fields_match"] is False
    assert any(
        "recomputed WATER2 audit did not pass" in reason
        for reason in plan["refusal_reasons"]
    )


def test_plan_rechecks_input_hashes_and_runner_digest(tmp_path: Path):
    bundle = byteqc_fixture(tmp_path)
    decision_path = write_json(tmp_path / "decision.json", run_byteqc_audit(bundle))
    Path(bundle["result"]).write_text("{}\n", encoding="utf-8")
    Path(bundle["runner"]).write_text("# changed\n", encoding="utf-8")
    plan = gate.make_water4_plan(
        decision_path=decision_path,
        runner_path=bundle["runner"],
        python_executable=Path(sys.executable),
        run_id="tampered-w4",
        output_dir=tmp_path / "future-results",
    )
    assert plan["w4_eligible"] is False
    assert "job" not in plan
    assert any("changed after audit" in reason for reason in plan["refusal_reasons"])
    assert any("runner SHA-256" in reason for reason in plan["refusal_reasons"])


def test_cli_writes_audit_and_plan_manifests_once(tmp_path: Path, capsys):
    bundle = byteqc_fixture(tmp_path)
    decision = tmp_path / "audit.json"
    rc = gate.main(
        [
            "audit",
            "--software",
            "byteqc",
            "--result",
            str(bundle["result"]),
            "--scheduler-evidence",
            str(bundle["scheduler"]),
            "--cases",
            str(bundle["cases"]),
            "--runner-capabilities",
            str(bundle["capabilities"]),
            "--oracle",
            str(bundle["oracle"]),
            "--energy-tolerance",
            "1e-8",
            "--output",
            str(decision),
        ]
    )
    assert rc == 0
    assert json.loads(decision.read_text())["w4_eligible"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gate.main(
            [
                "audit",
                "--software",
                "byteqc",
                "--result",
                str(bundle["result"]),
                "--scheduler-evidence",
                str(bundle["scheduler"]),
                "--cases",
                str(bundle["cases"]),
                "--runner-capabilities",
                str(bundle["capabilities"]),
                "--oracle",
                str(bundle["oracle"]),
                "--energy-tolerance",
                "1e-8",
                "--output",
                str(decision),
            ]
        )
    manifest = tmp_path / "plan.json"
    rc = gate.main(
        [
            "plan-water4",
            "--decision",
            str(decision),
            "--runner",
            str(bundle["runner"]),
            "--python",
            sys.executable,
            "--run-id",
            "cli-w4-a",
            "--output-dir",
            str(tmp_path / "future"),
            "--manifest",
            str(manifest),
        ]
    )
    assert rc == 0
    assert json.loads(manifest.read_text())["job"]["benchmark_argv"]
    capsys.readouterr()


def test_source_contains_no_submission_execution_path():
    source = (HERE / "external_result_gate.py").read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "os.system" not in source
