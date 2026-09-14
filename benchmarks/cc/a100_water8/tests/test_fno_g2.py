"""CPU-only tests for the staged water4 FNO G2 probe."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("water8_fno_g2", ROOT / "fno_g2.py")
assert SPEC is not None and SPEC.loader is not None
FNO_G2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FNO_G2
SPEC.loader.exec_module(FNO_G2)


def _task_and_source(tmp_path: Path) -> tuple[Path, Path]:
    task = tmp_path / "task"
    source = task / "snapshots" / ("a" * 64) / "source"
    return task, source


def test_plan_contains_exact_grid_serial_ab_pairs_and_shared_cp_bundle(
    tmp_path: Path,
) -> None:
    task, source = _task_and_source(tmp_path)
    plan = FNO_G2.make_plan(
        task_root=task,
        source_root=source,
        run_id="g2-test",
        node="compute-1-6",
    )

    assert plan["threshold_order"] == list(FNO_G2.FNO_THRESHOLDS)
    assert len(plan["jobs"]) == 22
    timing = [job for job in plan["jobs"] if job["phase"] == "water4-timing"]
    cp = [job for job in plan["jobs"] if job["phase"] == "water4-counterpoise"]
    assert len(timing) == 14
    assert len(cp) == 8
    for index, cutoff in enumerate(FNO_G2.FNO_THRESHOLDS):
        oracle, candidate = timing[2 * index:2 * index + 2]
        assert oracle["exports"]["METHOD"] == "canonical"
        assert candidate["exports"]["METHOD"] == "fno"
        assert candidate["fno_thresh"] == cutoff
        assert candidate["depends_on"] == [oracle["id"]]
        assert (
            oracle["exports"]["BENCHMARK_ORBITAL_ARTIFACT_OUT"]
            == candidate["exports"]["BENCHMARK_ORBITAL_ARTIFACT_IN"]
        )
        if index:
            assert oracle["depends_on"] == [timing[2 * index - 1]["id"]]
    bundle = cp[0]["exports"]["CP_ORBITAL_BUNDLE_OUT"]
    assert cp[0]["depends_on"] == [timing[-1]["id"]]
    for predecessor, successor in zip(cp, cp[1:]):
        assert successor["depends_on"] == [predecessor["id"]]
    assert all(
        job["exports"]["CP_ORBITAL_BUNDLE_IN"] == bundle
        for job in cp[1:]
    )
    assert all(job["slurm_options"] == ["--nodelist=compute-1-6"] for job in plan["jobs"])


def test_submit_preview_preserves_symbolic_dependencies(tmp_path: Path) -> None:
    task, source = _task_and_source(tmp_path)
    plan = FNO_G2.make_plan(
        task_root=task,
        source_root=source,
        run_id="g2-test",
        node="compute-1-6",
    )

    receipt = FNO_G2.submit_plan(plan, execute=False)

    assert receipt["executed"] is False
    first, second = receipt["jobs"][:2]
    assert first["job_id"] == "<timing-1e-4-canonical>"
    assert not any(value.startswith("--dependency=") for value in first["argv"])
    assert (
        "--dependency=afterok:<timing-1e-4-canonical>" in second["argv"]
    )
    assert second["argv"][-1].endswith("run_mtu_benchmark.sbatch")


def _artifact_payload(token: str, *, mode: str) -> dict:
    fingerprint = f"fingerprint-{token}"
    sha = f"artifact-{token}"
    return {
        "schema": "gpu4pyscf.cc.canonical-orbitals.v1",
        "mode": mode,
        "applied_to_mean_field": True,
        "included_in_post_hf": False,
        "artifact_path": f"/task/results/{token}.npz",
        "artifact_sha256": sha,
        "orbital_fingerprint": fingerprint,
        "identity": {"case_id": "water4-tz", "nocc": 20, "nvir": 212},
        "producer": {"tree_sha256": "a" * 64},
    }


def _checkpoint(artifact: dict) -> dict:
    return {
        "included_in_post_hf": True,
        "orbital_artifact_sha256": artifact["artifact_sha256"],
        "orbital_fingerprint": artifact["orbital_fingerprint"],
    }


def _canonical(cutoff: float) -> dict:
    token = FNO_G2._cutoff_token(cutoff)
    artifact = _artifact_payload(token, mode="written-and-reloaded")
    return {
        "_fno_g2_input_path": f"canonical-{token}.json",
        "case_id": "water4-tz",
        "method": "canonical",
        "status": "completed",
        "cc_converged": True,
        "e_corr": -1.0,
        "post_hf_seconds": 100.0,
        "canonical_orbitals": artifact,
        "checkpoint": _checkpoint(artifact),
    }


def _fno(cutoff: float, *, ecorr_error: float) -> dict:
    token = FNO_G2._cutoff_token(cutoff)
    artifact = _artifact_payload(token, mode="loaded")
    delta = -0.01
    raw_corr = -1.0 + ecorr_error - delta
    raw_total = -80.0 + ecorr_error - delta
    occupations = [float(212 - index) * 1e-8 for index in range(212)]
    return {
        "_fno_g2_input_path": f"fno-{token}.json",
        "case_id": "water4-tz",
        "method": "fno",
        "status": "completed",
        "cc_converged": True,
        "post_hf_seconds": 50.0,
        "e_corr": raw_corr + delta,
        "e_tot": raw_total + delta,
        "raw_fno_e_corr": raw_corr,
        "raw_fno_e_tot": raw_total,
        "delta_mp2": delta,
        "energy_definition": (
            "FNO-CCSD + Delta-MP2 (full-space MP2 minus FNO-space MP2)"
        ),
        "approximation_signature": f"fno:{token}",
        "approximation": {
            "method": "fno",
            "selection": {"mode": "occupation_threshold", "value": cutoff},
            "correction": "delta-mp2",
        },
        "canonical_orbitals": artifact,
        "checkpoint": _checkpoint(artifact),
        "residual": {
            "equation_norm": 1e-7,
            "orbital_space": "fno-active",
            "is_full_space": False,
        },
        "run_metrics": {
            "phase_totals": {
                "method_setup": {"total_s": 4.0},
                "integral_transform": {"total_s": 6.0},
            }
        },
        "fno": {
            "threshold": cutoff,
            "pct_occ": None,
            "requested_nvir_act": None,
            "selection_mode": "occupation_threshold",
            "selection_value": cutoff,
            "nocc": 20,
            "nvir": 212,
            "nvir_active": 200,
            "frozen_virtual": list(range(200, 212)),
            "active_virtual": list(range(200)),
            "virtual_occupations": occupations,
            "occupation_boundary": {
                "lowest_retained": occupations[199],
                "highest_discarded": occupations[200],
            },
            "semicanonical_offdiag_max": 1e-12,
            "mp2": {
                "full_correlation_energy_eh": -0.5,
                "fno_correlation_energy_eh": -0.49,
                "delta_mp2_eh": delta,
            },
            "timings_s": {
                "mp2_full": 1.0,
                "occupation_and_fno_orbitals": 0.5,
                "semicanonical_fock_validation": 0.25,
                "mp2_fno": 0.75,
                "total": 2.6,
            },
        },
    }


def _patch_formal_helpers(monkeypatch) -> None:
    monkeypatch.setattr(
        FNO_G2._analysis,
        "compatibility",
        lambda baseline, candidate: {"compatible": True, "reasons": []},
    )
    monkeypatch.setattr(
        FNO_G2._analysis,
        "_resources",
        lambda candidate: {"passed": True, "hbm_gib": 20.0, "rss_gib": 4.0},
    )
    monkeypatch.setattr(
        FNO_G2._analysis,
        "_source_digest",
        lambda candidate: "a" * 64,
    )
    monkeypatch.setattr(
        FNO_G2._analysis,
        "_counterpoise_error",
        lambda *args, **kwargs: (0.01, []),
    )


def test_analysis_selects_first_pass_and_immediate_tighter_setting(
    monkeypatch,
) -> None:
    _patch_formal_helpers(monkeypatch)
    records: list[dict] = []
    for index, cutoff in enumerate(FNO_G2.FNO_THRESHOLDS):
        records.extend((
            _canonical(cutoff),
            _fno(cutoff, ecorr_error=2e-4 if index < 3 else 5e-5),
        ))

    result = FNO_G2.analyze_fno_g2(records)

    assert result["grid_complete"] is True
    assert result["selection"]["ready_for_water8"] is True
    assert result["selection"]["first_passing_cutoff"] == 3e-6
    assert result["selection"]["next_tighter_cutoff"] == 1e-6
    assert [
        item["fno_thresh"]
        for item in result["selection"]["water8_candidates"]
    ] == [3e-6, 1e-6]
    selected = result["rows"][3]
    assert selected["timing"]["speedup"] == 2.0
    assert selected["fno"]["delta_mp2_eh"] == -0.01
    assert selected["canonical_orbital_pair"]["passed"] is True


def test_analysis_fails_closed_on_delta_mp2_or_orbital_mismatch(
    monkeypatch,
) -> None:
    _patch_formal_helpers(monkeypatch)
    records: list[dict] = []
    for cutoff in FNO_G2.FNO_THRESHOLDS:
        baseline = _canonical(cutoff)
        candidate = _fno(cutoff, ecorr_error=5e-5)
        records.extend((baseline, candidate))
    records[1]["delta_mp2"] = -0.02
    records[3]["canonical_orbitals"]["orbital_fingerprint"] = "different"

    result = FNO_G2.analyze_fno_g2(records)

    assert result["rows"][0]["qualifies_for_water8"] is False
    assert any("Delta-MP2" in reason for reason in result["rows"][0]["reasons"])
    assert result["rows"][1]["qualifies_for_water8"] is False
    assert any(
        "exactly one canonical timing record" in reason
        for reason in result["rows"][1]["reasons"]
    )


def test_analysis_requires_the_complete_fixed_grid(monkeypatch) -> None:
    _patch_formal_helpers(monkeypatch)
    records: list[dict] = []
    for cutoff in FNO_G2.FNO_THRESHOLDS[:-1]:
        records.extend((_canonical(cutoff), _fno(cutoff, ecorr_error=5e-5)))

    result = FNO_G2.analyze_fno_g2(records)

    assert result["grid_complete"] is False
    assert result["selection"]["ready_for_water8"] is False
    assert result["rows"][-1]["qualifies_for_water8"] is False


def test_water2_plan_is_a_full_seven_threshold_ab_and_cp_dag(tmp_path: Path) -> None:
    task, source = _task_and_source(tmp_path)
    plan = FNO_G2.make_water2_plan(
        task_root=task, source_root=source, run_id="water2-grid", node="compute-1-6"
    )
    assert plan["stage"] == "water2"
    assert plan["case_id"] == "water2-tz"
    assert plan["dimensions"] == {"nao": 116, "nocc": 10, "nvir": 106}
    assert len(plan["jobs"]) == 22
    assert all(job["exports"]["CASE"] == "water2-tz" for job in plan["jobs"])
    assert {job["phase"] for job in plan["jobs"]} == {
        "water2-timing", "water2-counterpoise"
    }
    with pytest.raises(ValueError, match="compute-1-6"):
        FNO_G2.make_water2_plan(
            task_root=task, source_root=source, run_id="other-node", node="compute-1-5"
        )


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _write_plan_and_receipt(tmp_path: Path, plan: dict, *, stage: str, tag: str) -> Path:
    import hashlib
    import json
    plan_path = tmp_path / f"{tag}-plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    receipt = {
        "schema": FNO_G2.SUBMISSION_SCHEMA,
        "executed": True,
        "stage": stage,
        "case_id": plan["case_id"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "plan_path": str(plan_path.resolve()),
        "plan_payload_sha256": hashlib.sha256(
            FNO_G2._canonical_json(plan).encode("ascii")
        ).hexdigest(),
        "jobs": [
            {"job": job["id"], "job_id": str(index + 1000)}
            for index, job in enumerate(plan["jobs"])
        ],
    }
    return _write_json(tmp_path / f"{tag}-receipt.json", receipt)


def _bind_analysis_to_receipt(analysis: Path, receipt: Path, plan: dict) -> None:
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_bytes = receipt.read_bytes()
    binding = {
        "receipt_path": str(receipt.resolve()),
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt_payload_sha256": FNO_G2._payload_sha256(payload),
        "plan_path": str(Path(payload["plan_path"]).resolve()),
        "plan_file_sha256": hashlib.sha256(Path(payload["plan_path"]).read_bytes()).hexdigest(),
        "plan_payload_sha256": payload["plan_payload_sha256"],
        "run_id": plan["run_id"],
        "logical_job_ids": {job["job"]: job["job_id"] for job in payload["jobs"]},
    }
    value = json.loads(analysis.read_text(encoding="utf-8"))
    value["receipt_binding"] = binding
    analysis.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_water4_plan_requires_immutable_passing_water2_evidence(tmp_path: Path) -> None:
    task, source = _task_and_source(tmp_path)
    analysis = _write_json(task / "results" / "water2-analysis.json", {
        "schema": FNO_G2.SCHEMA, "stage": "water2", "case_id": "water2-tz",
        "source_tree_sha256": "a" * 64,
        "selection": {"ready_for_next_stage": True},
    })
    water2_plan = FNO_G2.make_water2_plan(
        task_root=task, source_root=source, run_id="water2", node="compute-1-6"
    )
    receipt = _write_plan_and_receipt(tmp_path, water2_plan, stage="water2", tag="water2")
    _bind_analysis_to_receipt(analysis, receipt, water2_plan)
    plan = FNO_G2.make_water4_plan(
        task_root=task, source_root=source, run_id="water4-grid", node="compute-1-6",
        water2_analysis=analysis, water2_receipt=receipt,
    )
    assert plan["stage"] == "water4"
    assert plan["prerequisite"]["analysis"]["payload_sha256"]
    analysis.write_text(analysis.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with __import__("pytest").raises(ValueError, match="modified"):
        FNO_G2._validate_plan(plan)


def test_strict_virtual_partition_rejects_relative_or_duplicate_indexes(monkeypatch) -> None:
    _patch_formal_helpers(monkeypatch)
    records: list[dict] = []
    for cutoff in FNO_G2.FNO_THRESHOLDS:
        baseline = _canonical(cutoff)
        candidate = _fno(cutoff, ecorr_error=5e-5)
        candidate["source"] = {"tree_sha256": "a" * 64}
        baseline["source"] = {"tree_sha256": "a" * 64}
        candidate["fno"]["active_virtual"] = list(range(20, 220))
        candidate["fno"]["frozen_virtual"] = list(range(219, 231))
        records.extend((baseline, candidate))
    result = FNO_G2.analyze_water4(records)
    assert result["selection"]["ready_for_water8"] is False
    assert any("exact full partition" in reason for reason in result["rows"][0]["reasons"])


def test_water8_dry_run_contains_exactly_two_selected_candidates(tmp_path: Path) -> None:
    task, source = _task_and_source(tmp_path)
    rows = [
        {"threshold": cutoff, "qualifies_for_water8": index >= 3}
        for index, cutoff in enumerate(FNO_G2.FNO_THRESHOLDS)
    ]
    analysis_payload = {
        "schema": FNO_G2.SCHEMA, "stage": "water4", "case_id": "water4-tz",
        "source_tree_sha256": "a" * 64, "grid_complete": True, "rows": rows,
        "selection": {
            "ready_for_water8": True,
            "water8_candidates": [
                {"fno_thresh": 3e-6, "approximation_signature": "fno:3e-6"},
                {"fno_thresh": 1e-6, "approximation_signature": "fno:1e-6"},
            ],
        },
    }
    analysis = _write_json(task / "results" / "water4-analysis.json", analysis_payload)
    water2_analysis = _write_json(task / "results" / "water2-analysis.json", {
        "schema": FNO_G2.SCHEMA, "stage": "water2", "case_id": "water2-tz",
        "source_tree_sha256": "a" * 64, "selection": {"ready_for_next_stage": True},
    })
    water2_plan = FNO_G2.make_water2_plan(
        task_root=task, source_root=source, run_id="water2", node="compute-1-6"
    )
    water2_receipt = _write_plan_and_receipt(tmp_path, water2_plan, stage="water2", tag="water2")
    _bind_analysis_to_receipt(water2_analysis, water2_receipt, water2_plan)
    water4_plan = FNO_G2.make_water4_plan(
        task_root=task, source_root=source, run_id="water4", node="compute-1-6",
        water2_analysis=water2_analysis, water2_receipt=water2_receipt,
    )
    receipt = _write_plan_and_receipt(tmp_path, water4_plan, stage="water4", tag="water4")
    water4_plan_path = tmp_path / "water4-plan.json"
    water4_plan_path.write_text(__import__("json").dumps(water4_plan, sort_keys=True), encoding="utf-8")
    # Repoint the receipt to the exact plan path used by the gate.
    receipt_payload = __import__("json").loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["plan_path"] = str(water4_plan_path.resolve())
    receipt_payload["plan_payload_sha256"] = FNO_G2._payload_sha256(water4_plan)
    receipt.write_text(__import__("json").dumps(receipt_payload, sort_keys=True), encoding="utf-8")
    _bind_analysis_to_receipt(analysis, receipt, water4_plan)
    plan = FNO_G2.make_water8_plan(
        task_root=task, source_root=source, run_id="water8-grid", node="compute-1-6",
        water4_analysis=analysis, water4_receipt=receipt,
    )
    assert plan["candidate_thresholds"] == [3e-6, 1e-6]
    assert len(plan["jobs"]) == 8
    assert [job["phase"] for job in plan["jobs"]].count("water8-timing") == 4
    assert [job["phase"] for job in plan["jobs"]].count("water8-counterpoise") == 4
    preview = FNO_G2.submit_plan(plan, execute=False)
    assert len(preview["jobs"]) == 8
    assert preview["prerequisite"]["analysis"]["file_sha256"]


def test_receipt_requires_exact_schema_plan_jobs_and_numeric_ids(tmp_path: Path) -> None:
    task, source = _task_and_source(tmp_path)
    plan = FNO_G2.make_water2_plan(
        task_root=task, source_root=source, run_id="receipt", node="compute-1-6"
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    receipt = {
        "schema": FNO_G2.SUBMISSION_SCHEMA, "executed": True,
        "stage": "water2", "case_id": "water2-tz",
        "source_tree_sha256": plan["source_tree_sha256"],
        "plan_payload_sha256": FNO_G2._payload_sha256(plan),
        "plan_path": str(plan_path),
        "jobs": [{"job": job["id"], "job_id": str(i + 1)} for i, job in enumerate(plan["jobs"])],
    }
    assert len(FNO_G2._validate_receipt_jobs(receipt, plan)) == 22
    for mutation in (
        lambda value: value.update(schema="wrong"),
        lambda value: value.update(plan_payload_sha256="0" * 64),
        lambda value: value.update(jobs=value["jobs"][:-1]),
        lambda value: value.update(jobs=value["jobs"] + [{"job": "extra", "job_id": "9999"}]),
        lambda value: value["jobs"].__setitem__(0, {"job": value["jobs"][0]["job"], "job_id": "x"}),
    ):
        bad = copy.deepcopy(receipt)
        mutation(bad)
        with pytest.raises(ValueError):
            FNO_G2._validate_receipt_jobs(bad, plan)


def test_result_slurm_id_must_be_inside_receipt(tmp_path: Path) -> None:
    task, source = _task_and_source(tmp_path)
    plan = FNO_G2.make_water2_plan(
        task_root=task, source_root=source, run_id="binding", node="compute-1-6"
    )
    receipt = {
        "schema": FNO_G2.SUBMISSION_SCHEMA, "executed": True,
        "stage": "water2", "case_id": "water2-tz",
        "source_tree_sha256": plan["source_tree_sha256"],
        "plan_payload_sha256": FNO_G2._payload_sha256(plan),
        "jobs": [{"job": job["id"], "job_id": str(i + 1)} for i, job in enumerate(plan["jobs"])],
    }
    record = {"slurm": {"SLURM_JOB_NAME": plan["jobs"][0]["id"], "SLURM_JOB_ID": "99999"}}
    assert any("outside" in reason for reason in FNO_G2._bind_records_to_receipt([record], receipt, plan))


def test_complete_water2_record_set_binds_without_false_cp_errors(tmp_path: Path) -> None:
    task, source = _task_and_source(tmp_path)
    plan = FNO_G2.make_water2_plan(task_root=task, source_root=source, run_id="complete", node="compute-1-6")
    receipt = {
        "schema": FNO_G2.SUBMISSION_SCHEMA, "executed": True,
        "stage": "water2", "case_id": "water2-tz",
        "source_tree_sha256": plan["source_tree_sha256"],
        "plan_payload_sha256": FNO_G2._payload_sha256(plan),
        "jobs": [{"job": job["id"], "job_id": str(i + 3000)} for i, job in enumerate(plan["jobs"])],
    }
    records = []
    for item, job in zip(receipt["jobs"], plan["jobs"]):
        result_path = Path(job["expected_outputs"][0])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}", encoding="utf-8")
        record = {"_fno_g2_input_path": str(result_path), "slurm": {
            "SLURM_JOB_NAME": item["job"], "SLURM_JOB_ID": item["job_id"]}}
        if job["phase"] == "water2-counterpoise":
            bundle_path = Path(job["exports"].get("CP_ORBITAL_BUNDLE_OUT") or job["exports"]["CP_ORBITAL_BUNDLE_IN"])
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            bundle_path.write_bytes(b"bundle")
            record.update({
                "schema": "gpu4pyscf.water8.counterpoise.v1",
                "orchestration": {"logical_job_id": item["job"], "run_id": job["exports"]["CP_RUN_ID"], "stage": "water2"},
                "orbital_bundle": {"path": str(bundle_path), "complete": True, "sha256": hashlib.sha256(b"bundle").hexdigest()},
            })
        records.append(record)
    assert FNO_G2._bind_records_to_receipt(records, receipt, plan) == []


def test_counterpoise_wrong_run_or_stage_is_rejected(tmp_path: Path) -> None:
    task, source = _task_and_source(tmp_path)
    plan = FNO_G2.make_water2_plan(task_root=task, source_root=source, run_id="cpidentity", node="compute-1-6")
    job = next(item for item in plan["jobs"] if item["phase"] == "water2-counterpoise")
    receipt = {"schema": FNO_G2.SUBMISSION_SCHEMA, "executed": True, "stage": "water2", "case_id": "water2-tz", "source_tree_sha256": plan["source_tree_sha256"], "plan_payload_sha256": FNO_G2._payload_sha256(plan), "jobs": [{"job": item["id"], "job_id": str(i + 1)} for i, item in enumerate(plan["jobs"])]}
    path = Path(job["expected_outputs"][0]); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}", encoding="utf-8")
    bundle = Path(job["exports"].get("CP_ORBITAL_BUNDLE_OUT") or job["exports"]["CP_ORBITAL_BUNDLE_IN"]); bundle.parent.mkdir(parents=True, exist_ok=True); bundle.write_bytes(b"b")
    record = {"_fno_g2_input_path": str(path), "schema": "gpu4pyscf.water8.counterpoise.v1", "slurm": {"SLURM_JOB_NAME": job["id"], "SLURM_JOB_ID": next(i["job_id"] for i in receipt["jobs"] if i["job"] == job["id"])}, "orchestration": {"logical_job_id": job["id"], "run_id": "wrong", "stage": "wrong"}, "orbital_bundle": {"path": str(bundle), "complete": True, "sha256": hashlib.sha256(b"b").hexdigest()}}
    errors = FNO_G2._bind_records_to_receipt([record], receipt, plan)
    assert any("run ID" in error for error in errors)
    assert any("stage" in error for error in errors)


def test_plan_regeneration_rejects_field_phase_and_dag_tampering(tmp_path: Path) -> None:
    task, source = _task_and_source(tmp_path)
    plan = FNO_G2.make_water2_plan(
        task_root=task, source_root=source, run_id="tamper", node="compute-1-6"
    )
    mutations = []
    for key, value in (("CASE", "water8-tz"), ("METHOD", "canonical"), ("FNO_THRESH", "1e-7")):
        mutations.append(lambda candidate, key=key, value=value: candidate["jobs"][1]["exports"].update({key: value}))
    mutations.append(lambda candidate: candidate["jobs"][0].update(phase="water2-counterpoise"))
    mutations.append(lambda candidate: candidate["jobs"].pop())
    mutations.append(lambda candidate: candidate["jobs"][1].update(depends_on=[]))
    mutations.append(lambda candidate: candidate["jobs"][0]["expected_outputs"].append("tampered"))
    mutations.append(lambda candidate: candidate["jobs"][0]["exports"].update(BENCHMARK_ORBITAL_ARTIFACT_OUT="/tmp/other.npz"))
    for mutation in mutations:
        bad = copy.deepcopy(plan)
        mutation(bad)
        with pytest.raises(ValueError, match="canonical regeneration|exact|phase|job"):
            FNO_G2._validate_plan(bad)


def test_water8_counterpoise_to_timing_mutation_is_rejected(tmp_path: Path) -> None:
    task, source = _task_and_source(tmp_path)
    # Build a strict water8 plan through the public helper, using a minimal
    # cryptographically valid water4 prerequisite as in the dry-run test.
    analysis = _write_json(task / "a.json", {
        "schema": FNO_G2.SCHEMA, "stage": "water4", "case_id": "water4-tz",
        "source_tree_sha256": "a" * 64, "grid_complete": True,
        "rows": [{"threshold": x, "qualifies_for_water8": i >= 3} for i, x in enumerate(FNO_G2.FNO_THRESHOLDS)],
        "selection": {"ready_for_water8": True, "water8_candidates": [{"fno_thresh": 3e-6}, {"fno_thresh": 1e-6}]},
    })
    w2_analysis = _write_json(task / "w2a.json", {
        "schema": FNO_G2.SCHEMA, "stage": "water2", "case_id": "water2-tz",
        "source_tree_sha256": "a" * 64, "selection": {"ready_for_next_stage": True},
    })
    w2_plan = FNO_G2.make_water2_plan(task_root=task, source_root=source, run_id="w2", node="compute-1-6")
    w2_receipt = _write_plan_and_receipt(tmp_path, w2_plan, stage="water2", tag="w2")
    _bind_analysis_to_receipt(w2_analysis, w2_receipt, w2_plan)
    w4_plan = FNO_G2.make_water4_plan(task_root=task, source_root=source, run_id="w4", node="compute-1-6", water2_analysis=w2_analysis, water2_receipt=w2_receipt)
    w4_receipt = _write_plan_and_receipt(tmp_path, w4_plan, stage="water4", tag="w4")
    w4_plan_path = tmp_path / "w4-plan.json"
    w4_plan_path.write_text(json.dumps(w4_plan, sort_keys=True), encoding="utf-8")
    payload = json.loads(w4_receipt.read_text(encoding="utf-8")); payload["plan_path"] = str(w4_plan_path); payload["plan_payload_sha256"] = FNO_G2._payload_sha256(w4_plan); w4_receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    _bind_analysis_to_receipt(analysis, w4_receipt, w4_plan)
    plan = FNO_G2.make_water8_plan(task_root=task, source_root=source, run_id="w8", node="compute-1-6", water4_analysis=analysis, water4_receipt=w4_receipt)
    bad = copy.deepcopy(plan)
    bad["jobs"][4]["phase"], bad["jobs"][0]["phase"] = bad["jobs"][0]["phase"], bad["jobs"][4]["phase"]
    with pytest.raises(ValueError):
        FNO_G2._validate_plan(bad)


def test_same_source_distinct_plan_receipts_cannot_unlock_transitions(tmp_path: Path) -> None:
    task, source = _task_and_source(tmp_path)
    analysis = _write_json(task / "analysis.json", {
        "schema": FNO_G2.SCHEMA, "stage": "water2", "case_id": "water2-tz",
        "source_tree_sha256": "a" * 64, "selection": {"ready_for_next_stage": True},
    })
    p1 = FNO_G2.make_water2_plan(task_root=task, source_root=source, run_id="one", node="compute-1-6")
    p2 = FNO_G2.make_water2_plan(task_root=task, source_root=source, run_id="two", node="compute-1-6")
    r1 = _write_plan_and_receipt(tmp_path, p1, stage="water2", tag="one")
    r2 = _write_plan_and_receipt(tmp_path, p2, stage="water2", tag="two")
    _bind_analysis_to_receipt(analysis, r1, p1)
    with pytest.raises(ValueError, match="identity"):
        FNO_G2.make_water4_plan(task_root=task, source_root=source, run_id="next", node="compute-1-6", water2_analysis=analysis, water2_receipt=r2)
