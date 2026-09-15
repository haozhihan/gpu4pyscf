"""CPU-only tests for the staged water4 FNO G2 probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


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
