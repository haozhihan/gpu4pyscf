from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def _diagnostic_record(orthogonality: object, residual: object) -> dict:
    return {
        "method_metadata": {
            "rr_projector_build": {
                "projector": {"orthogonality_error": orthogonality},
            },
        },
        "full_space_residual_diagnostic": {
            "available": True,
            "execution_status": "completed",
            "residual_norm": residual,
        },
    }


def test_diagnostic_gate_accepts_orthogonal_projector_and_finite_residual() -> None:
    checks, measurements = rr_g4._diagnostic_gate(
        _diagnostic_record(1e-12, 1.25e4)
    )
    assert checks == {
        "projector_orthogonality_1e_10": True,
        "full_space_diagnostic_recorded": True,
    }
    assert measurements == {
        "projector_orthogonality_error": 1e-12,
        "full_space_equation_residual_diagnostic": 1.25e4,
    }


def test_diagnostic_gate_accepts_exact_orthogonality_threshold() -> None:
    checks, measurements = rr_g4._diagnostic_gate(
        _diagnostic_record(rr_g4.PROJECTOR_ORTHOGONALITY_TOLERANCE, 0.0)
    )
    assert checks["projector_orthogonality_1e_10"] is True
    assert checks["full_space_diagnostic_recorded"] is True
    assert measurements["projector_orthogonality_error"] == pytest.approx(1e-10)


def test_projected_residual_norm_rejects_negative_value() -> None:
    record = {"residual": {"projected_equation": -1e-12}}
    assert rr_g4._projected_residual(record) is None
    record["residual"]["projected_equation"] = 1e-12
    assert rr_g4._projected_residual(record) == 1e-12


@pytest.mark.parametrize(
    "value", [True, False, float("nan"), float("inf"), -1.0, None],
)
def test_resource_scalar_rejects_bool_nonfinite_and_negative(value: object) -> None:
    assert rr_g4._finite_nonnegative(value) is None


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1e-14, 1.1e-10])
def test_diagnostic_gate_rejects_invalid_or_nonorthogonal_projector(
    value: object,
) -> None:
    checks, measurements = rr_g4._diagnostic_gate(
        _diagnostic_record(value, 0.02)
    )
    assert checks["projector_orthogonality_1e_10"] is False
    assert checks["full_space_diagnostic_recorded"] is True
    if value != 1.1e-10:
        assert measurements["projector_orthogonality_error"] is None


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1e-14])
def test_diagnostic_gate_rejects_invalid_full_space_residual(value: object) -> None:
    checks, measurements = rr_g4._diagnostic_gate(
        _diagnostic_record(1e-12, value)
    )
    assert checks["projector_orthogonality_1e_10"] is True
    assert checks["full_space_diagnostic_recorded"] is False
    assert measurements["full_space_equation_residual_diagnostic"] is None


def test_write_once_failure_preserves_owned_inode(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "gate.json"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(rr_g4.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        rr_g4._exclusive_json(output, {"gate": "partial"})
    assert output.exists()
    with pytest.raises(FileExistsError):
        rr_g4._exclusive_json(output, {"gate": "replacement"})


def _write_receipt_and_pin(
    tmp_path: Path, node: str = "compute-1-3",
) -> tuple[Path, Path, Path]:
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
    manifest = _write_read_only_json(
        source.parent / "manifest.json", {"synthetic": True},
    )
    evidence_dir = task / "results/release-evidence"
    evidence = {
        "submission": _write_read_only_json(
            evidence_dir / "submission.json", {"job_id": "70002"},
        ),
        "result": _write_read_only_json(
            evidence_dir / "result.json", {"performance_eligible": True},
        ),
        "topology": _write_read_only_json(
            evidence_dir / "topology.json", {"node": node},
        ),
    }
    slurm_output = task / "logs/gint-gate-70002.log"
    slurm_output.parent.mkdir(parents=True)
    slurm_output.write_text("release accepted\n", encoding="utf-8")
    evidence["slurm_output"] = slurm_output
    release_gate = rr_g4._release_gate
    acceptance_payload = {
        "schema": release_gate.ACCEPTANCE_SCHEMA,
        "created_utc": "2026-09-15T00:00:00+00:00",
        "status": "accepted", "performance_eligible": True,
        "source": {
            "root": str(source.resolve()), "tree_sha256": "a" * 64,
            "manifest": {
                "path": str(manifest.resolve()),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
        },
        "qualification_receipt": {
            "path": str(receipt.resolve()), "sha256": receipt_sha,
            "payload_sha256": payload_sha,
        },
        "release_pin": {
            "path": str((source / "gpu4pyscf/cc/gint_release_pin.json").resolve()),
            "sha256": hashlib.sha256(
                (source / "gpu4pyscf/cc/gint_release_pin.json").read_bytes()
            ).hexdigest(),
        },
        "node": node,
        "release_job": {"terminal_slurm": {
            "job_id": "70002", "job_name": f"gint-release-{payload_sha}",
            "partition": "mrigpu", "node": node,
            "state": "COMPLETED", "exit_code": "0:0",
        }},
        "evidence": {
            label: {"path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for label, path in evidence.items()
        },
    }
    acceptance = task / "results/release-acceptance/acceptance.json"
    release_gate.write_once_acceptance(acceptance, acceptance_payload)
    return source, receipt, acceptance


def test_receipt_node_comes_from_matching_receipt_and_release_pin(tmp_path: Path) -> None:
    source, receipt, acceptance = _write_receipt_and_pin(tmp_path, "compute-1-3")
    contract = rr_g4._receipt_contract(source, receipt)
    assert contract["node"] == "compute-1-3"

    pin_path = source / "gpu4pyscf/cc/gint_release_pin.json"
    pin = json.loads(pin_path.read_text())
    pin["release_runtime_contract"]["node"] = "compute-1-6"
    pin_path.write_text(json.dumps(pin))
    with pytest.raises(rr_g4.ContractError, match="disagree"):
        rr_g4._receipt_contract(source, receipt)


def _rewrite_acceptance(path: Path, mutation) -> None:
    directory = path.parent
    sidecar = Path(str(path) + ".sha256")
    directory.chmod(0o755)
    path.chmod(0o644)
    sidecar.chmod(0o644)
    payload = json.loads(path.read_text())
    mutation(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {path.name}\n")
    path.chmod(0o444)
    sidecar.chmod(0o444)
    directory.chmod(0o555)


@pytest.mark.parametrize(
    ("label", "mutation"),
    [
        ("pending", lambda value: value["release_job"]["terminal_slurm"].update(
            state="PENDING"
        )),
        ("failed", lambda value: value["release_job"]["terminal_slurm"].update(
            state="FAILED", exit_code="1:0"
        )),
        ("node", lambda value: value.update(node="compute-1-0")),
        ("source", lambda value: value["source"].update(tree_sha256="f" * 64)),
        ("receipt", lambda value: value["qualification_receipt"].update(
            sha256="e" * 64
        )),
    ],
)
def test_g4_refuses_forged_or_nonterminal_release_acceptance(
    tmp_path: Path, label: str, mutation,
) -> None:
    source, receipt, acceptance = _write_receipt_and_pin(tmp_path)
    if label == "missing":
        acceptance.unlink()
    else:
        _rewrite_acceptance(acceptance, mutation)
    with pytest.raises(rr_g4.ContractError, match="not accepted|differs"):
        rr_g4._accepted_receipt_contract(source, receipt, acceptance)


def test_g4_requires_release_acceptance_cli_argument(tmp_path: Path) -> None:
    source, receipt, _ = _write_receipt_and_pin(tmp_path)
    with pytest.raises(SystemExit):
        rr_g4.main([
            "receipt-node", "--source-root", str(source),
            "--gint-runtime-gate-receipt", str(receipt),
        ])


def test_submitter_uses_only_standard_receipt_compatible_launcher() -> None:
    text = (ROOT / "submit_mtu_rr_g4_water4.sh").read_text(encoding="utf-8")
    for required in (
        "run_mtu_benchmark.sbatch", "receipt-node", "authorize",
        "--nodes=1", "--ntasks=1", "--gres=gpu:nvidia:1",
        '--nodelist="${NODE}"', "GINT_RUNTIME_GATE_RECEIPT",
        "GINT_RELEASE_GATE_ACCEPTANCE",
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


def _write_read_only_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return path


class _SyntheticG4Chain:
    """Build a release-bound WATER4 evidence chain for CLI integration tests."""

    def __init__(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
        rank_1e9: int = 2500, rank_1e11: int = 2800,
    ) -> None:
        self.node = "compute-1-3"
        self.source, self.receipt, self.acceptance = _write_receipt_and_pin(
            tmp_path, self.node
        )
        self.task = self.source.parents[2]
        # Full release-evidence reconstruction is covered by test_gint_gate.py;
        # these compact fixtures isolate the G4 sequence and science gates.
        monkeypatch.setattr(
            rr_g4._release_gate, "build_acceptance",
            lambda **_kwargs: json.loads(
                self.acceptance.read_text(encoding="utf-8")
            ),
        )
        self.results = self.task / "results"
        self.chain_dir = self.results / "rr-g4-water4/synthetic"
        self.chain_dir.mkdir(parents=True)
        self.source_digest = self.source.parent.name
        self.orbital_sha = "b" * 64
        self.orbital_fingerprint = "c" * 64
        self.orbital_path = str(self.results / "orbitals/water4.npz")
        self.jobs: dict[str, tuple[str, bool]] = {}
        self.sacct_calls: list[list[str]] = []

        receipt_envelope = json.loads(self.receipt.read_text(encoding="utf-8"))
        self.receipt_sha = hashlib.sha256(self.receipt.read_bytes()).hexdigest()
        self.payload_sha = receipt_envelope["payload_sha256"]
        self.pin = self.source / "gpu4pyscf/cc/gint_release_pin.json"
        # These are immutable inputs to every continuation command in this fixture.
        self.receipt.chmod(0o444)
        Path(str(self.receipt) + ".sha256").chmod(0o444)
        self.pin.chmod(0o444)

        def fake_sacct(
            command: list[str], *, check: bool, capture_output: bool, text: bool,
        ) -> SimpleNamespace:
            assert command[0] == "/usr/bin/sacct"
            assert check and capture_output and text
            job_id = command[command.index("--jobs") + 1]
            job_name, probe = self.jobs[job_id]
            state, exit_code = ("FAILED", "1:0") if probe else ("COMPLETED", "0:0")
            row = "|".join((
                job_id, job_name, "mrigpu", self.node, state, exit_code,
                "2026-09-15T00:00:00", "2026-09-15T00:00:01",
                "2026-09-15T00:01:00", "00:00:59",
            ))
            self.sacct_calls.append(command)
            return SimpleNamespace(stdout=row + "\n")

        monkeypatch.setattr(rr_g4.subprocess, "run", fake_sacct)
        self.oracle = self.record(
            "oracle", method="canonical", job_id="71000", e_corr=-1.0,
            iteration_seconds=10.0,
        )
        self.probe_1e9 = self.record(
            "probe-1e9", method=rr_g4.METHOD, job_id="71001", probe=True,
            cutoff=1e-9, rank=rank_1e9,
        )
        self.probe_1e11 = self.record(
            "probe-1e11", method=rr_g4.METHOD, job_id="71002", probe=True,
            cutoff=1e-11, rank=rank_1e11,
        )
        self.spectrum = self.chain_dir / "spectrum.json"

    def _source(self) -> dict[str, Any]:
        return {
            "tree_sha256": self.source_digest,
            "tree_sha256_at_start": self.source_digest,
            "tree_sha256_at_end": self.source_digest,
            "stable_during_run": True,
            "repository": str(self.source),
            "snapshot": {
                "snapshot_root": str(self.source.parent),
                "valid_at_start": True,
                "valid_at_end": True,
                "read_only_at_start": True,
                "read_only_at_end": True,
                "runtime_binaries_valid_at_start": True,
                "runtime_binaries_valid_at_end": True,
                "stable_during_run": True,
            },
        }

    def _provider_gate(self, job_id: str) -> dict[str, Any]:
        return {
            "validated": True,
            "execution_mode": "consumer-benchmark",
            "receipt": {
                "path": str(self.receipt),
                "sha256": self.receipt_sha,
                "payload_sha256": self.payload_sha,
                "release_source_sha256": self.source_digest,
                "current_runtime": {
                    "environment": {"SLURM_JOB_ID": job_id},
                },
            },
        }

    def record(
        self, label: str, *, method: str, job_id: str,
        cutoff: float | None = None, rank: int | None = None,
        probe: bool = False, e_corr: float = -1.0,
        iteration_seconds: float = 8.0,
    ) -> Path:
        job_name = f"g4-{label}"
        self.jobs[job_id] = (job_name, probe)
        record: dict[str, Any] = {
            "case_id": rr_g4.CASE_ID,
            "method": method,
            "status": "not_converged" if probe else "completed",
            "cc_converged": False if probe else True,
            "cc_iterations": 1 if probe else 5,
            "e_corr": e_corr,
            "post_hf_seconds": iteration_seconds + 5.0,
            "source": self._source(),
            "canonical_orbitals": {
                "artifact_sha256": self.orbital_sha,
                "orbital_fingerprint": self.orbital_fingerprint,
                "artifact_path": self.orbital_path,
            },
            "slurm": {
                "SLURM_JOB_ID": job_id,
                "SLURM_JOB_NAME": job_name,
                "SLURM_JOB_NODELIST": self.node,
                "SLURM_JOB_PARTITION": "mrigpu",
                "SLURM_CPUS_PER_TASK": "64",
            },
            "experiment_record": {"phases": [{
                "name": "ccsd_iterations", "depth": 0,
                "elapsed_s": iteration_seconds,
            }]},
        }
        if method == rr_g4.METHOD:
            assert cutoff is not None and rank is not None
            record.update({
                "approximation": {"rr_eig_cutoff": cutoff},
                "ranks": {
                    "rr_rank": rank,
                    "rr_full_dimension": rr_g4.PAIR_DIMENSION,
                },
                "provider_runtime_gate": self._provider_gate(job_id),
                "method_metadata": {"rr_projector_build": {
                    "cutoff_bracketed": True,
                    "capped": False,
                    "solver": "synthetic-lanczos",
                    "requested_subspace": rank,
                    "max_ritz_residual": 1e-12,
                    "projector": {"orthogonality_error": 1e-12},
                }},
            })
            if not probe:
                record.update({
                    "residual": {"projected_equation": 1e-7},
                    "full_space_residual_diagnostic": {
                        "available": True,
                        "execution_status": "completed",
                        "residual_norm": 2.5e-3,
                    },
                    "hbm": {"peak_process_MiB": 70 * 1024},
                    "peak_host_RSS_GiB": 100.0,
                    "checkpoint": {"included_in_post_hf": True},
                })
        path = _write_read_only_json(
            self.results / "records" / f"{label}.json", record,
        )
        topology = {
            "performance_eligible": True,
            "mode": "physical8",
            "slurm": {
                "SLURM_JOB_ID": job_id,
                "SLURM_JOB_NODELIST": self.node,
            },
            "command": [
                str(self.source / "benchmarks/cc/a100_water8/benchmark.py"),
                "--case", rr_g4.CASE_ID,
            ],
        }
        _write_read_only_json(
            self.results / "topology" / f"physical8-benchmark-{job_id}.json",
            topology,
        )
        return path

    def make_spectrum(self) -> dict[str, Any]:
        assert rr_g4.main([
            "spectrum",
            "--task-root", str(self.task),
            "--source-root", str(self.source),
            "--gint-runtime-gate-receipt", str(self.receipt),
            "--gint-release-gate-acceptance", str(self.acceptance),
            "--output", str(self.spectrum),
            "--oracle-record", str(self.oracle),
            "--probe-record", str(self.probe_1e9),
            "--probe-record", str(self.probe_1e11),
        ]) == 0
        return json.loads(self.spectrum.read_text(encoding="utf-8"))

    def gate(
        self, *, label: str, cutoff: float, record: Path,
        previous: Path | None = None, probe_summary: Path | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        output = self.chain_dir / f"{label}-gate.json"
        argv = [
            "gate",
            "--task-root", str(self.task),
            "--source-root", str(self.source),
            "--gint-runtime-gate-receipt", str(self.receipt),
            "--gint-release-gate-acceptance", str(self.acceptance),
            "--output", str(output),
            "--rr-eig-cutoff", str(cutoff),
            "--spectrum-summary", str(self.spectrum),
            "--rr-record", str(record),
        ]
        if previous is not None:
            argv += ["--previous-gate", str(previous)]
        if probe_summary is not None:
            argv += ["--probe-summary", str(probe_summary)]
        assert rr_g4.main(argv) == 0
        return output, json.loads(output.read_text(encoding="utf-8"))

    def authorize(
        self, *, stage: str, cutoff: float, previous: Path | None = None,
        probe_summary: Path | None = None,
    ) -> int:
        argv = [
            "authorize",
            "--task-root", str(self.task),
            "--source-root", str(self.source),
            "--gint-runtime-gate-receipt", str(self.receipt),
            "--gint-release-gate-acceptance", str(self.acceptance),
            "--stage", stage,
            "--rr-eig-cutoff", str(cutoff),
            "--spectrum-summary", str(self.spectrum),
        ]
        if previous is not None:
            argv += ["--previous-gate", str(previous)]
        if probe_summary is not None:
            argv += ["--probe-summary", str(probe_summary)]
        return rr_g4.main(argv)

    def custom_probe(
        self, *, label: str, cutoff: float, rank: int, previous: Path,
    ) -> tuple[Path, dict[str, Any]]:
        record = self.record(
            f"{label}-record", method=rr_g4.METHOD, job_id="71020",
            probe=True, cutoff=cutoff, rank=rank,
        )
        output = self.chain_dir / f"{label}.json"
        assert rr_g4.main([
            "probe",
            "--task-root", str(self.task),
            "--source-root", str(self.source),
            "--gint-runtime-gate-receipt", str(self.receipt),
            "--gint-release-gate-acceptance", str(self.acceptance),
            "--output", str(output),
            "--rr-eig-cutoff", str(cutoff),
            "--spectrum-summary", str(self.spectrum),
            "--previous-gate", str(previous),
            "--rr-record", str(record),
        ]) == 0
        return output, json.loads(output.read_text(encoding="utf-8"))


def test_cli_first_pass_requires_next_tighter_and_slow_sequence_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _SyntheticG4Chain(tmp_path, monkeypatch)
    spectrum = chain.make_spectrum()
    assert [item["cutoff"] for item in spectrum["probes"]] == pytest.approx(
        [1e-9, 1e-11]
    )
    assert spectrum["oracle"]["complete_iteration_timing"]["seconds"] == 10.0
    first_record = chain.record(
        "converged-1e9", method=rr_g4.METHOD, job_id="71003",
        cutoff=1e-9, rank=2500, e_corr=-1.0, iteration_seconds=12.0,
    )
    first_path, first = chain.gate(
        label="first", cutoff=1e-9, record=first_record,
    )
    assert first["scientific_pass"] is True
    assert first["sequence_complete"] is False
    assert first["requires_tighter"] is True
    assert first["next_cutoff"] == pytest.approx(1e-11)
    assert first["advance_to_water8"] is False

    assert chain.authorize(
        stage="converge", cutoff=1e-11, previous=first_path,
    ) == 0
    second_record = chain.record(
        "converged-1e11", method=rr_g4.METHOD, job_id="71004",
        cutoff=1e-11, rank=2800, e_corr=-1.0, iteration_seconds=11.0,
    )
    _, second = chain.gate(
        label="second", cutoff=1e-11, record=second_record,
        previous=first_path,
    )
    assert second["scientific_pass"] is True
    assert second["sequence_complete"] is True
    assert second["requires_tighter"] is False
    assert second["promotable_cutoffs"] == []
    assert second["advance_to_water8"] is False
    assert second["performance_stop"] is True
    assert len(chain.sacct_calls) == 5
    for immutable in (chain.receipt, chain.pin, chain.spectrum, first_path):
        assert immutable.stat().st_mode & 0o222 == 0


def test_cli_failed_rank_below_boundary_authorizes_custom_probe_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _SyntheticG4Chain(
        tmp_path, monkeypatch, rank_1e9=3391, rank_1e11=3300,
    )
    chain.make_spectrum()
    failed_1e9 = chain.record(
        "failed-accuracy-1e9", method=rr_g4.METHOD, job_id="71005",
        cutoff=1e-9, rank=3391, e_corr=-0.9998,
    )
    gate_1e9_path, gate_1e9 = chain.gate(
        label="failed-1e9", cutoff=1e-9, record=failed_1e9,
    )
    assert gate_1e9["scientific_pass"] is False
    assert gate_1e9["rank"] == rr_g4.RANK_STOP_BOUNDARY - 1
    assert gate_1e9["route_stop"] is False
    assert gate_1e9["requires_tighter"] is True
    assert chain.authorize(
        stage="converge", cutoff=1e-11, previous=gate_1e9_path,
    ) == 0

    failed_1e11 = chain.record(
        "failed-accuracy-1e11", method=rr_g4.METHOD, job_id="71006",
        cutoff=1e-11, rank=3300, e_corr=-0.9998,
    )
    gate_1e11_path, gate_1e11 = chain.gate(
        label="failed-1e11", cutoff=1e-11, record=failed_1e11,
        previous=gate_1e9_path,
    )
    assert gate_1e11["route_stop"] is False
    assert gate_1e11["next_cutoff"] == pytest.approx(1e-13)
    assert chain.authorize(
        stage="probe", cutoff=1e-13, previous=gate_1e11_path,
    ) == 0
    custom_path, custom = chain.custom_probe(
        label="probe-1e13", cutoff=1e-13, rank=3350,
        previous=gate_1e11_path,
    )
    assert custom["rank"] == 3350
    assert custom["performance_eligible"] is False
    assert chain.authorize(
        stage="converge", cutoff=1e-13, previous=gate_1e11_path,
        probe_summary=custom_path,
    ) == 0


def test_cli_failed_rank_at_boundary_closes_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _SyntheticG4Chain(
        tmp_path, monkeypatch, rank_1e9=rr_g4.RANK_STOP_BOUNDARY,
    )
    chain.make_spectrum()
    failed_record = chain.record(
        "failed-at-boundary", method=rr_g4.METHOD, job_id="71007",
        cutoff=1e-9, rank=rr_g4.RANK_STOP_BOUNDARY, e_corr=-0.9998,
    )
    gate_path, gate = chain.gate(
        label="boundary", cutoff=1e-9, record=failed_record,
    )
    assert gate["scientific_pass"] is False
    assert gate["route_stop"] is True
    assert gate["requires_tighter"] is False
    assert gate["sequence_complete"] is False
    assert gate["advance_to_water8"] is False
    with pytest.raises(rr_g4.ContractError, match="closes the continuation chain"):
        chain.authorize(
            stage="converge", cutoff=1e-11, previous=gate_path,
        )
