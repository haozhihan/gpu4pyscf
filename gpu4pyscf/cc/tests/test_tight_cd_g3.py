from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "benchmarks/cc/a100_water8/tight_cd_g3.py"
SPEC = importlib.util.spec_from_file_location("tight_cd_g3_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
G3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G3)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _task_source_receipt(
    tmp_path: Path, node: str = "compute-1-3",
) -> tuple[Path, Path, Path, dict]:
    task = tmp_path / "task"
    source_digest = "a" * 64
    source = task / "snapshots" / source_digest / "source"
    (source / "gpu4pyscf/cc").mkdir(parents=True)
    receipt = task / "results/gint/receipt.json"
    payload = {
        "schema": G3._g4.PAYLOAD_SCHEMA,
        "qualification": {"slurm": {"node": node}},
    }
    payload_sha = G3._g4._canonical_json_sha256(payload)
    envelope = {
        "schema": G3._g4.RECEIPT_SCHEMA,
        "payload_sha256": payload_sha,
        "payload": payload,
    }
    _write_json(receipt, envelope)
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    Path(str(receipt) + ".sha256").write_text(
        f"{receipt_sha}  {receipt.name}\n", encoding="ascii",
    )
    pin = {
        "schema": G3._g4.PIN_SCHEMA,
        "lineage": {
            "model": "qualification-source-plus-fixed-release-pin",
            "qualification_source_tree_sha256": "b" * 64,
            "release_source_without_pin_sha256": "b" * 64,
        },
        "receipt": {"sha256": receipt_sha, "payload_sha256": payload_sha},
        "release_runtime_contract": {
            "schema": G3._g4.RUNTIME_CONTRACT_SCHEMA,
            "node": node,
            "host": node,
        },
    }
    _write_json(source / "gpu4pyscf/cc/gint_release_pin.json", pin)
    contract = G3._gint_contract(source, receipt, task)
    return task, source, receipt, contract


def _executed_receipt(tmp_path: Path, plan: dict) -> tuple[Path, dict[str, str]]:
    plan_path = _write_json(tmp_path / f"{plan['run_id']}-plan.json", plan)
    preview = G3.submit_plan(plan, execute=False, plan_path=plan_path)
    preview["executed"] = True
    logical = {}
    for index, item in enumerate(preview["jobs"], start=1001):
        item["job_id"] = str(index)
        logical[item["job"]] = str(index)
    receipt = _write_json(tmp_path / f"{plan['run_id']}-receipt.json", preview)
    return receipt, logical


def _artifact_payload(path: Path, *, mode: str, case_id: str, source: str) -> dict:
    payload = b"exact serialized orbitals"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    dimensions = G3.CASE_DIMENSIONS[case_id]
    return {
        "schema": "gpu4pyscf.cc.canonical-orbitals.v1",
        "mode": mode,
        "artifact_path": str(path),
        "artifact_sha256": sha,
        "orbital_fingerprint": "c" * 64,
        "identity": {
            "case_id": case_id,
            "nao": dimensions["nao"],
            "nocc": dimensions["nocc"],
            "nvir": dimensions["nvir"],
        },
        "producer": {"tree_sha256": source},
        "applied_to_mean_field": True,
        "included_in_post_hf": False,
    }


def _runtime_receipt(contract: dict, source_digest: str) -> dict:
    return {
        "path": contract["path"],
        "sha256": contract["sha256"],
        "payload_sha256": contract["payload_sha256"],
        "release_source_sha256": source_digest,
        "current_runtime": {"environment": {}},
    }


def _record_set(plan: dict, logical: dict[str, str]) -> list[dict]:
    records: list[dict] = []
    source_digest = plan["source_tree_sha256"]
    contract = plan["gint_release_receipt"]
    for job in plan["jobs"]:
        tolerance = job["eri_tol"]
        method = job["method"]
        expected = Path(job["expected_outputs"][0])
        expected.parent.mkdir(parents=True, exist_ok=True)
        slurm = {
            "SLURM_JOB_NAME": job["id"],
            "SLURM_JOB_ID": logical[job["id"]],
            "SLURM_JOB_NODELIST": plan["node"],
        }
        if job["phase"].endswith("timing"):
            artifact_path = Path(
                job["exports"].get("BENCHMARK_ORBITAL_ARTIFACT_OUT")
                or job["exports"]["BENCHMARK_ORBITAL_ARTIFACT_IN"]
            )
            orbitals = _artifact_payload(
                artifact_path,
                mode="written-and-reloaded" if method == "rr_canonical" else "loaded",
                case_id=plan["case_id"], source=source_digest,
            )
            rank = 200 if method == "rr_canonical" else 201
            seconds = {
                1e-4: 100.0,
                1e-6: 102.0,
                1e-8: 110.0,
            }[tolerance]
            record = {
                "_tight_cd_g3_input_path": str(expected),
                "case_id": plan["case_id"],
                "method": method,
                "status": "completed",
                "cc_converged": True,
                "e_corr": -1.0 + (5e-5 if method == "rr_cd" else 0.0),
                "post_hf_seconds": seconds if method == "rr_cd" else 120.0,
                "plan": {"settings": {
                    "eri_tol": tolerance,
                    "rr_eig_cutoff": G3.RR_EIG_CUTOFF,
                }},
                "source": {
                    "tree_sha256": source_digest,
                    "tree_sha256_at_start": source_digest,
                    "tree_sha256_at_end": source_digest,
                    "stable_during_run": True,
                },
                "slurm": slurm,
                "canonical_orbitals": orbitals,
                "checkpoint": {
                    "included_in_post_hf": True,
                    "orbital_artifact_sha256": orbitals["artifact_sha256"],
                    "orbital_fingerprint": orbitals["orbital_fingerprint"],
                },
                "ranks": {"rr_rank": rank, "rr_full_dimension": 1060},
                "method_metadata": {"rr_projector_build": {"projector": {
                    "orthogonality_error": 1e-12,
                }}},
                "residual": {"projected_equation": 1e-7},
            }
            if method == "rr_cd":
                runtime = _runtime_receipt(contract, source_digest)
                runtime["current_runtime"]["environment"]["SLURM_JOB_ID"] = logical[job["id"]]
                record["provider_runtime_gate"] = {
                    "validated": True,
                    "execution_mode": "consumer-benchmark",
                    "receipt": runtime,
                }
                record["full_space_residual_diagnostic"] = {
                    "available": True,
                    "execution_status": "completed",
                    "residual_norm": 0.02,
                }
        else:
            bundle_path = Path(
                job["exports"].get("CP_ORBITAL_BUNDLE_OUT")
                or job["exports"]["CP_ORBITAL_BUNDLE_IN"]
            )
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            if not bundle_path.exists():
                bundle_path.write_bytes(b"complete CP orbital bundle")
            bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
            fragments = [{} for _ in range(plan["dimensions"]["fragments"])]
            record = {
                "_tight_cd_g3_input_path": str(expected),
                "schema": "gpu4pyscf.water8.counterpoise.v1",
                "case": {"id": plan["case_id"]},
                "method": method,
                "protocol": {"options": {
                    "eri_tol": tolerance,
                    "rr_eig_cutoff": G3.RR_EIG_CUTOFF,
                }},
                "source": {
                    "tree_sha256": source_digest,
                    "tree_sha256_at_start": source_digest,
                    "tree_sha256_at_end": source_digest,
                    "stable_during_run": True,
                },
                "slurm": slurm,
                "orchestration": {
                    "logical_job_id": job["id"],
                    "run_id": job["exports"]["CP_RUN_ID"],
                    "stage": f"tight-cd-g3-{plan['stage']}",
                },
                "orbital_bundle": {
                    "path": str(bundle_path), "sha256": bundle_sha,
                    "complete": True,
                },
                "fragments": fragments,
                "interaction_energy_cp_kcal_mol": (
                    -10.0 + (0.01 if method == "rr_cd" else 0.0)
                ),
            }
            if method == "rr_cd":
                runtime = _runtime_receipt(contract, source_digest)
                runtime["current_runtime"]["environment"]["SLURM_JOB_ID"] = logical[job["id"]]
                gate = {
                    "validated": True,
                    "execution_mode": "consumer-counterpoise",
                    "receipt": runtime,
                }
                record["provider_runtime_gates"] = {
                    "cluster": copy.deepcopy(gate),
                    "fragments": [copy.deepcopy(gate) for _ in fragments],
                }
        _write_json(expected, {
            key: value for key, value in record.items()
            if key != "_tight_cd_g3_input_path"
        })
        records.append(record)
    return records


def _patch_formal_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(G3._analysis, "_source_stable", lambda record: True)
    monkeypatch.setattr(G3._analysis, "_timed_snapshot_evidence_reasons", lambda record: [])
    monkeypatch.setattr(G3._analysis, "_scientific_signature", lambda record: {"same": True})
    monkeypatch.setattr(G3._analysis, "_hardware_fingerprint", lambda record: "same-hardware")
    monkeypatch.setattr(G3._analysis, "_timing_boundary", lambda record: "same-boundary")
    monkeypatch.setattr(
        G3._analysis, "_resources",
        lambda record: {"passed": True, "hbm_gib": 60.0, "rss_gib": 10.0},
    )
    monkeypatch.setattr(G3._analysis, "_cp_eligibility", lambda record, source: (True, []))
    monkeypatch.setattr(G3._analysis, "_cp_signature", lambda record: {"same": True})
    monkeypatch.setattr(
        G3._analysis, "shared_bundle_proof",
        lambda left, right: {"record_match_key": "same-bundle"},
    )


def test_plan_is_receipt_node_bound_and_names_scientific_limit(tmp_path: Path) -> None:
    task, source, receipt, contract = _task_source_receipt(tmp_path)
    plan = G3.make_water2_plan(
        task_root=task, source_root=source,
        gint_runtime_gate_receipt=receipt, run_id="g3-w2",
    )
    assert plan["node"] == contract["node"] == "compute-1-3"
    assert plan["eri_tolerances"] == [1e-4, 1e-6, 1e-8]
    assert len(plan["jobs"]) == 12
    assert plan["scientific_boundary"] == {
        "comparison": "rr_canonical versus rr_cd complete pipelines",
        "same_rr_eig_cutoff": True,
        "same_serialized_rhf_orbitals": True,
        "projector_artifact_shared": False,
        "projectors_built_independently": True,
        "measured_effect": "CD increment in the final RR pipeline",
        "full_space_canonical_cd_equivalence": False,
        "acceptance_table": "A",
        "integral_subtrack": "CD increment diagnostic inside A-track RR",
        "performance_claim_eligible": False,
    }
    assert all(
        job["slurm_options"] == ["--nodelist=compute-1-3"]
        for job in plan["jobs"]
    )
    assert all(
        job["exports"]["GINT_RUNTIME_GATE_RECEIPT"] == str(receipt.resolve())
        for job in plan["jobs"] if job["method"] == "rr_cd"
    )
    G3._validate_plan(plan)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda plan: plan["scientific_boundary"].update(projector_artifact_shared=True),
        lambda plan: plan["jobs"][1]["exports"].pop("GINT_RUNTIME_GATE_RECEIPT"),
        lambda plan: plan["jobs"][1].update(depends_on=[]),
        lambda plan: plan["jobs"][0]["exports"].update(RR_EIG_CUTOFF="1e-6"),
        lambda plan: plan["jobs"].pop(),
    ],
)
def test_plan_tampering_fails_closed(tmp_path: Path, mutation) -> None:
    task, source, receipt, _ = _task_source_receipt(tmp_path)
    plan = G3.make_water2_plan(
        task_root=task, source_root=source,
        gint_runtime_gate_receipt=receipt, run_id="tamper",
    )
    mutation(plan)
    with pytest.raises(G3.ContractError):
        G3._validate_plan(plan)


def test_gint_receipt_and_release_pin_must_be_one_lineage(tmp_path: Path) -> None:
    task, source, receipt, _ = _task_source_receipt(tmp_path)
    pin_path = source / "gpu4pyscf/cc/gint_release_pin.json"
    pin = json.loads(pin_path.read_text())
    pin["lineage"]["release_source_without_pin_sha256"] = "d" * 64
    _write_json(pin_path, pin)
    with pytest.raises(G3.ContractError, match="lineage"):
        G3.make_water2_plan(
            task_root=task, source_root=source,
            gint_runtime_gate_receipt=receipt, run_id="bad-lineage",
        )


def test_submit_preview_is_reviewable_and_does_not_call_sbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, source, receipt, _ = _task_source_receipt(tmp_path)
    plan = G3.make_water2_plan(
        task_root=task, source_root=source,
        gint_runtime_gate_receipt=receipt, run_id="preview",
    )
    monkeypatch.setattr(
        G3.subprocess, "check_output",
        lambda *args, **kwargs: pytest.fail("preview called sbatch"),
    )
    output = G3.submit_plan(plan, execute=False)
    assert output["executed"] is False
    assert len(output["jobs"]) == 12
    assert all(item["argv"][0:2] == ["sbatch", "--parsable"] for item in output["jobs"])
    assert "--dependency=afterok:" in output["jobs"][1]["shell_preview"]


def test_tie_rule_anchors_at_loosest_and_prefers_tighter_within_three_percent() -> None:
    def row(tolerance: float, seconds: float, qualifies: bool = True) -> dict:
        return {
            "eri_tol": tolerance, "qualifies": qualifies,
            "timing": {"rr_cd_post_hf_s": seconds},
        }

    selected = G3.select_eri_tolerance([
        row(1e-4, 100.0), row(1e-6, 102.999), row(1e-8, 103.001),
    ])
    assert selected["anchor_eri_tol"] == 1e-4
    assert selected["selected_eri_tol"] == 1e-6
    assert selected["tie_group"] == [1e-4, 1e-6]
    selected = G3.select_eri_tolerance([
        row(1e-4, 100.0, False), row(1e-6, 80.0), row(1e-8, 81.0),
    ])
    assert selected["anchor_eri_tol"] == 1e-6
    assert selected["selected_eri_tol"] == 1e-8


def test_complete_analysis_is_i_track_and_does_not_unlock_water8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_formal_evidence(monkeypatch)
    task, source, gint_receipt, _ = _task_source_receipt(tmp_path)
    plan = G3.make_water2_plan(
        task_root=task, source_root=source,
        gint_runtime_gate_receipt=gint_receipt, run_id="analysis",
    )
    submission, logical = _executed_receipt(tmp_path, plan)
    records = _record_set(plan, logical)
    analysis = G3.analyze_stage(records, submission)
    assert analysis["grid_complete"] is True
    assert analysis["selection"]["ready_for_water4"] is True
    assert analysis["selection"]["ready_for_water8"] is False
    assert analysis["selection"]["selected_eri_tol"] == 1e-6
    assert analysis["claim_state"] == {
        "stage_gate_passed": True,
        "g3_complete_claim_eligible": False,
        "full_space_canonical_cd_equivalence_proven": False,
        "projector_identity_proven": False,
        "water8_unlock_from_g3_alone": False,
        "speedup_claim_eligible": False,
    }
    for row in analysis["rows"]:
        assert row["acceptance_table"] == "A"
        assert row["projectors"]["same_artifact_claimed"] is False
        assert row["projectors"]["rank_delta"] == 1


def test_missing_result_and_incomplete_cp_fragments_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_formal_evidence(monkeypatch)
    task, source, gint_receipt, _ = _task_source_receipt(tmp_path)
    plan = G3.make_water2_plan(
        task_root=task, source_root=source,
        gint_runtime_gate_receipt=gint_receipt, run_id="negative",
    )
    submission, logical = _executed_receipt(tmp_path, plan)
    records = _record_set(plan, logical)
    incomplete = copy.deepcopy(records)
    target = next(
        item for item in incomplete
        if item.get("schema") == "gpu4pyscf.water8.counterpoise.v1"
        and item.get("method") == "rr_cd"
    )
    target["fragments"].pop()
    target["provider_runtime_gates"]["fragments"].pop()
    analysis = G3.analyze_stage(incomplete, submission)
    assert analysis["grid_complete"] is False
    assert analysis["selection"]["ready_for_water4"] is False
    assert any("ghost monomers" in reason for reason in analysis["rows"][0]["reasons"])

    analysis = G3.analyze_stage(records[:-1], submission)
    assert analysis["grid_complete"] is False
    assert any("without exactly one synchronized result" in reason
               for reason in analysis["selection"]["reasons"])


def test_projector_metadata_and_gint_consumer_binding_are_hard_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_formal_evidence(monkeypatch)
    task, source, gint_receipt, _ = _task_source_receipt(tmp_path)
    plan = G3.make_water2_plan(
        task_root=task, source_root=source,
        gint_runtime_gate_receipt=gint_receipt, run_id="projector",
    )
    submission, logical = _executed_receipt(tmp_path, plan)
    records = _record_set(plan, logical)
    candidate = next(
        item for item in records
        if item.get("method") == "rr_cd"
        and item.get("schema") != "gpu4pyscf.water8.counterpoise.v1"
    )
    candidate["method_metadata"]["rr_projector_build"]["projector"].pop(
        "orthogonality_error"
    )
    candidate["provider_runtime_gate"]["receipt"]["sha256"] = "0" * 64
    analysis = G3.analyze_stage(records, submission)
    assert analysis["grid_complete"] is False
    row = analysis["rows"][0]
    assert any("orthogonality" in reason for reason in row["reasons"])
    assert any("pinned GINT receipt" in reason for reason in row["reasons"])


def test_g4_authorization_must_match_selected_eri_tol_and_receipt(tmp_path: Path) -> None:
    task, source, gint_receipt, contract = _task_source_receipt(tmp_path)
    result = {
        "method": "rr_cd",
        "plan": {"settings": {"eri_tol": 1e-6}},
        "source": {"tree_sha256": "a" * 64},
        "provider_runtime_gate": {"validated": True, "receipt": {
            "sha256": contract["sha256"],
            "payload_sha256": contract["payload_sha256"],
        }},
    }
    result_path = _write_json(task / "results/g4/result.json", result)
    gate = {
        "schema": G3._g4.GATE_SCHEMA,
        "case_id": "water4-tz",
        "source_tree_sha256": "a" * 64,
        "sequence_complete": True,
        "advance_to_water8": True,
        "promotable_cutoffs": [1e-9],
        "result_evidence": {
            "path": str(result_path.resolve()),
            "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        },
    }
    gate_path = _write_json(task / "results/g4/gate.json", gate)
    evidence = G3._require_g4_authorization(
        gate_path, task_root=task, source_digest="a" * 64,
        selected_tol=1e-6, gint_contract=contract,
    )
    assert evidence["payload"]["advance_to_water8"] is True
    with pytest.raises(G3.ContractError, match="selected G3 eri_tol"):
        G3._require_g4_authorization(
            gate_path, task_root=task, source_digest="a" * 64,
            selected_tol=1e-8, gint_contract=contract,
        )
    gate["advance_to_water8"] = False
    _write_json(gate_path, gate)
    with pytest.raises(G3.ContractError, match="does not authorize"):
        G3._require_g4_authorization(
            gate_path, task_root=task, source_digest="a" * 64,
            selected_tol=1e-6, gint_contract=contract,
        )
