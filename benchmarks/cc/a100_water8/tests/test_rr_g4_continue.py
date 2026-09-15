from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "rr_g4_continue.py"
SPEC = importlib.util.spec_from_file_location("rr_g4_continue", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
rr_g4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rr_g4)


def test_water4_rank_stop_boundary_is_exact_contract_value() -> None:
    assert rr_g4.PAIR_DIMENSION == 4240
    assert rr_g4.RANK_STOP_BOUNDARY == 3392
    assert rr_g4.RANK_STOP_BOUNDARY == int(0.8 * rr_g4.PAIR_DIMENSION)


@pytest.mark.parametrize("rank", [1495, 1914, 3391])
def test_failed_candidate_below_boundary_requires_tighter_rank(rank: int) -> None:
    decision = rr_g4.decide_gate(
        scientific_pass=False, rank=rank, cutoff=1e-8, previous_gate=None,
    )
    assert decision["route_stop"] is False
    assert decision["requires_tighter"] is True
    assert decision["sequence_complete"] is False


@pytest.mark.parametrize("rank", [3392, 4000, 4240])
def test_failed_candidate_at_boundary_stops_route(rank: int) -> None:
    decision = rr_g4.decide_gate(
        scientific_pass=False, rank=rank, cutoff=1e-11, previous_gate=None,
    )
    assert decision["route_stop"] is True
    assert decision["requires_tighter"] is False
    assert decision["sequence_complete"] is False


def test_first_pass_and_next_tighter_complete_accuracy_sequence() -> None:
    first = rr_g4.decide_gate(
        scientific_pass=True, rank=2500, cutoff=1e-9, previous_gate=None,
    )
    assert first["requires_tighter"] is True
    assert first["next_cutoff"] == pytest.approx(1e-11)
    second = rr_g4.decide_gate(
        scientific_pass=False, rank=2800, cutoff=1e-11,
        previous_gate={"scientific_pass": True, "cutoff": 1e-9},
    )
    assert second["sequence_complete"] is True


def test_accurate_but_slower_sequence_never_advances_to_water8() -> None:
    decision = rr_g4.promotion_decision(
        sequence_complete=True,
        candidates=[{
            "cutoff": 1e-9,
            "scientific_pass": True,
            "complete_iteration_faster_than_canonical": False,
        }],
    )
    assert decision == {
        "promotable_cutoffs": [],
        "advance_to_water8": False,
        "performance_stop": True,
    }


def test_faster_accurate_candidate_promotes_only_after_sequence_complete() -> None:
    candidate = [{
        "cutoff": 1e-9, "scientific_pass": True,
        "complete_iteration_faster_than_canonical": True,
    }]
    assert rr_g4.promotion_decision(
        sequence_complete=False, candidates=candidate,
    )["advance_to_water8"] is False
    assert rr_g4.promotion_decision(
        sequence_complete=True, candidates=candidate,
    )["advance_to_water8"] is True


def test_complete_iteration_timing_supports_rr_and_canonical_records() -> None:
    rr = {
        "cc_iterations": 2,
        "experiment_record": {"phases": [{
            "name": "ccsd_iterations", "depth": 0, "elapsed_s": 12.5,
        }]},
    }
    canonical = {
        "cc_iterations": 2,
        "run_metrics": {"phases": [
            {"name": "ccsd_update", "depth": 1, "elapsed_s": 1.0},
            {"name": "diis", "depth": 1, "elapsed_s": 0.1},
            {"name": "ccsd_energy", "depth": 1, "elapsed_s": 0.2},
            {"name": "ccsd_update", "depth": 1, "elapsed_s": 1.1},
            {"name": "update_norm", "depth": 1, "elapsed_s": 0.05},
        ]},
    }
    assert rr_g4._complete_iteration_timing(rr)["seconds"] == 12.5
    assert rr_g4._complete_iteration_timing(canonical)["seconds"] == pytest.approx(2.45)


def _write_receipt_and_pin(tmp_path: Path, node: str = "compute-1-3") -> tuple[Path, Path]:
    task = tmp_path / "task"
    source = task / "snapshots" / ("a" * 64) / "source"
    receipt = task / "results/gint/receipt.json"
    receipt.parent.mkdir(parents=True)
    (source / "gpu4pyscf/cc").mkdir(parents=True)
    payload = {
        "schema": rr_g4.PAYLOAD_SCHEMA,
        "qualification": {"slurm": {"node": node}},
    }
    payload_sha = rr_g4._canonical_json_sha256(payload)
    envelope = {
        "schema": rr_g4.RECEIPT_SCHEMA,
        "payload_sha256": payload_sha, "payload": payload,
    }
    receipt.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    Path(str(receipt) + ".sha256").write_text(
        f"{receipt_sha}  {receipt.name}\n"
    )
    pin = {
        "schema": rr_g4.PIN_SCHEMA,
        "receipt": {"sha256": receipt_sha, "payload_sha256": payload_sha},
        "release_runtime_contract": {
            "schema": rr_g4.RUNTIME_CONTRACT_SCHEMA,
            "node": node, "host": node,
        },
    }
    (source / "gpu4pyscf/cc/gint_release_pin.json").write_text(
        json.dumps(pin, indent=2, sort_keys=True) + "\n"
    )
    return source, receipt


def test_receipt_node_comes_from_matching_receipt_and_release_pin(tmp_path: Path) -> None:
    source, receipt = _write_receipt_and_pin(tmp_path, "compute-1-3")
    contract = rr_g4._receipt_contract(source, receipt)
    assert contract["node"] == "compute-1-3"

    pin_path = source / "gpu4pyscf/cc/gint_release_pin.json"
    pin = json.loads(pin_path.read_text())
    pin["release_runtime_contract"]["node"] = "compute-1-6"
    pin_path.write_text(json.dumps(pin))
    with pytest.raises(rr_g4.ContractError, match="disagree"):
        rr_g4._receipt_contract(source, receipt)


def test_submitter_uses_only_standard_receipt_compatible_launcher() -> None:
    text = (ROOT / "submit_mtu_rr_g4_water4.sh").read_text(encoding="utf-8")
    for required in (
        "run_mtu_benchmark.sbatch", "receipt-node", "authorize",
        "--nodes=1", "--ntasks=1", "--gres=gpu:nvidia:1",
        '--nodelist="${NODE}"', "GINT_RUNTIME_GATE_RECEIPT",
        "--max-cycle 1", "--skip-checkpoint",
        "RUN_FULL_SPACE_DIAGNOSTIC=1",
        "refusing to reuse output directory",
    ):
        assert required in text
    assert "run_mtu_rr_g4_water4.sbatch" not in text
    assert "compute-1-6" not in text


def test_driver_never_runs_benchmark_or_names_water8_case() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'CASE_ID = "water4-tz"' in source
    assert '"water8-tz"' not in source
    assert "subprocess.run" in source  # sacct only
    assert "def _benchmark_base" not in source
    assert '"--case"' not in source
