#!/usr/bin/env python3
"""Validate the receipt-bound WATER4 RR continuation from standard jobs.

The selected-GINT receipt binds consumers to ``run_mtu_benchmark.sbatch`` and
to the node recorded by the qualification. This program therefore never runs
``benchmark.py`` itself. It validates standard-launcher artifacts, queries
terminal Slurm accounting, and writes summaries with exclusive-create
semantics. WATER8 is outside this driver by construction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


CASE_ID = "water4-tz"
METHOD = "rr_cd"
PAIR_DIMENSION = 4240
RANK_STOP_BOUNDARY = 3392  # 0.8 * (20 occupied * 212 virtual)
INITIAL_CUTOFFS = (1e-9, 1e-11)
TIGHTENING_FACTOR = 100.0
ENERGY_TOLERANCE_EH = 1e-4
PROJECTED_RESIDUAL_TOLERANCE = 1e-6
PROJECTOR_ORTHOGONALITY_TOLERANCE = 1e-10
HBM_LIMIT_MIB = 72 * 1024
HOST_RSS_LIMIT_GIB = 110.0
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_MTU_A100_NODES = frozenset({
    "compute-1-0", "compute-1-2", "compute-1-3", "compute-1-5",
    "compute-1-6",
})
RECEIPT_SCHEMA = "gpu4pyscf.gint-selected-runtime-gate-receipt.v2"
PAYLOAD_SCHEMA = "gpu4pyscf.gint-selected-runtime-gate-payload.v2"
PIN_SCHEMA = "gpu4pyscf.gint-selected-runtime-release-pin.v2"
RUNTIME_CONTRACT_SCHEMA = (
    "gpu4pyscf.gint-selected-release-runtime-contract.v2"
)
SPECTRUM_SCHEMA = "gpu4pyscf.rr-g4-water4-spectrum.v2"
PROBE_SCHEMA = "gpu4pyscf.rr-g4-water4-probe.v2"
GATE_SCHEMA = "gpu4pyscf.rr-g4-water4-gate.v2"


def _load_release_gate() -> Any:
    path = Path(__file__).resolve().with_name("gint_release_gate.py")
    spec = importlib.util.spec_from_file_location("rr_g4_release_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load gint_release_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_release_gate = _load_release_gate()


class ContractError(RuntimeError):
    """Raised when a continuation artifact violates the G4 contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON artifact must contain an object: {path}")
    return value


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    if not path.parent.is_dir():
        raise ContractError(f"output parent does not exist: {path.parent}")
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    # Preserve a partial file if writing or fsync fails. Removing by pathname
    # would create a TOCTOU window in which another process could replace this
    # inode and have its write-once artifact deleted by our exception handler.
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _input_file(path: Path, *, task_root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    result_root = (task_root / "results").resolve(strict=True)
    if not resolved.is_file() or not _is_below(resolved, result_root):
        raise ContractError(f"{label} must be a regular file below {result_root}")
    return resolved


def _output_file(path: Path, *, task_root: Path) -> Path:
    result_root = (task_root / "results").resolve(strict=True)
    parent = path.expanduser().parent.resolve(strict=True)
    result = parent / path.name
    if not _is_below(result, result_root):
        raise ContractError(f"output must be below {result_root}")
    if result.exists() or result.is_symlink():
        raise ContractError(f"refusing to overwrite output: {result}")
    return result


def _same_cutoff(left: Any, right: Any) -> bool:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)


def _cutoff(value: Any) -> float:
    if isinstance(value, bool):
        raise ContractError("RR cutoff must be a finite positive scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("RR cutoff must be a finite positive scalar") from exc
    if not math.isfinite(result) or result <= 0.0 or result > 1e-9:
        raise ContractError("RR continuation cutoff must lie in (0, 1e-9]")
    return result


def _receipt_contract(
    source_root: Path, receipt_path: Path, *, task_root: Path | None = None,
) -> dict[str, Any]:
    """Return the receipt-pinned node after checking the envelope and pin."""

    source = source_root.expanduser().resolve(strict=True)
    receipt = receipt_path.expanduser().resolve(strict=True)
    task = (
        source.parent.parent.parent if task_root is None
        else task_root.expanduser().resolve(strict=True)
    )
    if not receipt.is_file() or not _is_below(receipt, task / "results"):
        raise ContractError("GINT receipt must be a file below task results")
    envelope = _load_json(receipt)
    payload = envelope.get("payload")
    if envelope.get("schema") != RECEIPT_SCHEMA or not isinstance(payload, dict):
        raise ContractError("GINT receipt envelope is invalid")
    if payload.get("schema") != PAYLOAD_SCHEMA:
        raise ContractError("GINT receipt payload schema is invalid")
    payload_sha = _canonical_json_sha256(payload)
    if envelope.get("payload_sha256") != payload_sha:
        raise ContractError("GINT receipt payload digest is invalid")
    receipt_sha = _sha256(receipt)
    sidecar = Path(str(receipt) + ".sha256")
    expected_sidecar = f"{receipt_sha}  {receipt.name}\n"
    if not sidecar.is_file() or sidecar.read_text("ascii") != expected_sidecar:
        raise ContractError("GINT receipt SHA-256 sidecar is invalid")

    pin_path = source / "gpu4pyscf/cc/gint_release_pin.json"
    pin = _load_json(pin_path)
    runtime = pin.get("release_runtime_contract")
    if pin.get("schema") != PIN_SCHEMA or not isinstance(runtime, dict):
        raise ContractError("release snapshot lacks a valid GINT pin")
    if runtime.get("schema") != RUNTIME_CONTRACT_SCHEMA:
        raise ContractError("release runtime contract schema is invalid")
    pin_receipt = pin.get("receipt")
    if not isinstance(pin_receipt, dict) or (
        pin_receipt.get("sha256") != receipt_sha
        or pin_receipt.get("payload_sha256") != payload_sha
    ):
        raise ContractError("release pin identifies a different GINT receipt")
    qualification = payload.get("qualification")
    slurm = qualification.get("slurm") if isinstance(qualification, dict) else None
    node = slurm.get("node") if isinstance(slurm, dict) else None
    if (
        node not in ALLOWED_MTU_A100_NODES
        or runtime.get("node") != node
        or runtime.get("host") != node
    ):
        raise ContractError("receipt and release pin disagree on the MTU A100 node")
    return {
        "path": str(receipt), "sha256": receipt_sha,
        "payload_sha256": payload_sha, "node": node,
        "release_pin_path": str(pin_path),
        "release_pin_sha256": _sha256(pin_path),
    }


def _accepted_receipt_contract(
    source_root: Path, receipt_path: Path, acceptance_path: Path, *,
    task_root: Path | None = None,
) -> dict[str, Any]:
    """Require the post-release terminal acceptance before any G4 work."""

    source = source_root.expanduser().resolve(strict=True)
    task = (
        source.parent.parent.parent if task_root is None
        else task_root.expanduser().resolve(strict=True)
    )
    receipt = _receipt_contract(source, receipt_path, task_root=task)
    try:
        accepted = _release_gate.validate_release_acceptance(
            acceptance_path, task_root=task, source_root=source,
            qualification_receipt=receipt_path,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise ContractError(f"selected-GINT release gate is not accepted: {exc}") from exc
    if not all((
        accepted.get("node") == receipt["node"],
        accepted.get("receipt_sha256") == receipt["sha256"],
        accepted.get("receipt_payload_sha256") == receipt["payload_sha256"],
        accepted.get("release_pin_sha256") == receipt["release_pin_sha256"],
        accepted.get("source_tree_sha256") == source.parent.name,
    )):
        raise ContractError("release-gate acceptance differs from receipt/source")
    return {**receipt, "release_gate_acceptance": accepted}


def query_terminal_slurm(job_id: str) -> dict[str, str]:
    fields = (
        "JobIDRaw,JobName,Partition,NodeList,State,ExitCode,"
        "Submit,Start,End,Elapsed"
    )
    completed = subprocess.run(
        ["/usr/bin/sacct", "--noheader", "--parsable2", "--allocations",
         "--jobs", job_id, "--format", fields],
        check=True, capture_output=True, text=True,
    )
    rows = [line.split("|") for line in completed.stdout.splitlines() if line]
    rows = [row for row in rows if row[0] == job_id]
    if len(rows) != 1 or len(rows[0]) < 10:
        raise ContractError(f"sacct did not return one allocation for job {job_id}")
    return dict(zip(
        ("job_id", "job_name", "partition", "node", "state", "exit_code",
         "submit_utc", "start_utc", "end_utc", "elapsed"),
        rows[0][:10], strict=True,
    ))


def _formal_source(record: Mapping[str, Any]) -> tuple[str, str]:
    source = record.get("source")
    snapshot = source.get("snapshot") if isinstance(source, dict) else None
    if not isinstance(source, dict) or not isinstance(snapshot, dict):
        raise ContractError("benchmark record lacks source snapshot evidence")
    digest = source.get("tree_sha256")
    required = (
        isinstance(digest, str) and bool(HEX64_RE.fullmatch(digest)),
        source.get("tree_sha256_at_start") == digest,
        source.get("tree_sha256_at_end") == digest,
        source.get("stable_during_run") is True,
        snapshot.get("valid_at_start") is True,
        snapshot.get("valid_at_end") is True,
        snapshot.get("read_only_at_start") is True,
        snapshot.get("read_only_at_end") is True,
        snapshot.get("runtime_binaries_valid_at_start") is True,
        snapshot.get("runtime_binaries_valid_at_end") is True,
        snapshot.get("stable_during_run") is True,
    )
    if not all(required):
        raise ContractError("benchmark source snapshot is not stable and release-valid")
    root = snapshot.get("snapshot_root")
    if not isinstance(root, str):
        raise ContractError("benchmark source snapshot root is missing")
    return str(digest), root


def _orbital_identity(record: Mapping[str, Any]) -> tuple[str, str, str]:
    orbitals = record.get("canonical_orbitals")
    if not isinstance(orbitals, dict):
        raise ContractError("benchmark record lacks canonical-orbital evidence")
    values = (
        orbitals.get("artifact_sha256"), orbitals.get("orbital_fingerprint"),
        orbitals.get("artifact_path"),
    )
    if not (
        isinstance(values[0], str) and HEX64_RE.fullmatch(values[0])
        and isinstance(values[1], str) and HEX64_RE.fullmatch(values[1])
        and isinstance(values[2], str)
    ):
        raise ContractError("canonical-orbital identity is incomplete")
    return values  # type: ignore[return-value]


def _topology_and_terminal(
    record: Mapping[str, Any], *, task_root: Path, source_root: Path,
    node: str, probe: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    slurm = record.get("slurm")
    if not isinstance(slurm, dict):
        raise ContractError("benchmark record lacks Slurm provenance")
    job_id = slurm.get("SLURM_JOB_ID")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ContractError("benchmark record has an invalid Slurm job id")
    if (
        slurm.get("SLURM_JOB_NODELIST") != node
        or slurm.get("SLURM_JOB_PARTITION") != "mrigpu"
        or slurm.get("SLURM_CPUS_PER_TASK") != "64"
    ):
        raise ContractError("benchmark record differs from the receipt node contract")
    topology_path = (
        task_root / "results/topology" / f"physical8-benchmark-{job_id}.json"
    ).resolve(strict=True)
    topology = _load_json(topology_path)
    topology_slurm = topology.get("slurm")
    command = topology.get("command")
    expected_driver = str(
        source_root / "benchmarks/cc/a100_water8/benchmark.py"
    )
    if (
        topology.get("performance_eligible") is not True
        or topology.get("mode") != "physical8"
        or not isinstance(topology_slurm, dict)
        or topology_slurm.get("SLURM_JOB_ID") != job_id
        or topology_slurm.get("SLURM_JOB_NODELIST") != node
        or not isinstance(command, list)
        or expected_driver not in [str(item) for item in command]
    ):
        raise ContractError("benchmark topology does not bind the standard job")
    terminal = query_terminal_slurm(job_id)
    expected = ("FAILED", "1:0") if probe else ("COMPLETED", "0:0")
    if (
        terminal.get("node") != node
        or terminal.get("partition") != "mrigpu"
        or (terminal.get("state"), terminal.get("exit_code")) != expected
        or terminal.get("job_name") != slurm.get("SLURM_JOB_NAME")
    ):
        raise ContractError("terminal Slurm accounting does not match the record")
    for key in ("submit_utc", "start_utc", "end_utc", "elapsed"):
        if not terminal.get(key):
            raise ContractError("terminal Slurm accounting is incomplete")
    return {
        "path": str(topology_path), "sha256": _sha256(topology_path),
    }, terminal


def _gint_receipt_identity(
    record: Mapping[str, Any], receipt: Mapping[str, Any], *,
    source_digest: str, job_id: str,
) -> None:
    gate = record.get("provider_runtime_gate")
    summary = gate.get("receipt") if isinstance(gate, dict) else None
    current = summary.get("current_runtime") if isinstance(summary, dict) else None
    environment = current.get("environment") if isinstance(current, dict) else None
    if not (
        isinstance(gate, dict) and gate.get("validated") is True
        and gate.get("execution_mode") == "consumer-benchmark"
        and isinstance(summary, dict)
        and summary.get("path") == receipt["path"]
        and summary.get("sha256") == receipt["sha256"]
        and summary.get("payload_sha256") == receipt["payload_sha256"]
        and summary.get("release_source_sha256") == source_digest
        and isinstance(environment, dict)
        and environment.get("SLURM_JOB_ID") == job_id
    ):
        raise ContractError("RR record does not validate the pinned GINT receipt")


def _record_evidence(
    path: Path, *, task_root: Path, source_root: Path,
    receipt: Mapping[str, Any], method: str, probe: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _input_file(path, task_root=task_root, label="benchmark record")
    record = _load_json(resolved)
    if record.get("case_id") != CASE_ID or record.get("method") != method:
        raise ContractError("benchmark record identity does not match WATER4 G4")
    digest, snapshot_root = _formal_source(record)
    source_state = record.get("source")
    repository = source_state.get("repository") if isinstance(source_state, dict) else None
    if (
        snapshot_root != str(source_root.parent)
        or repository != str(source_root)
        or digest != source_root.parent.name
    ):
        raise ContractError("benchmark record belongs to another source snapshot")
    orbitals = _orbital_identity(record)
    topology, terminal = _topology_and_terminal(
        record, task_root=task_root, source_root=source_root,
        node=str(receipt["node"]), probe=probe,
    )
    job_id = str(record["slurm"]["SLURM_JOB_ID"])
    if method == METHOD:
        _gint_receipt_identity(
            record, receipt, source_digest=digest, job_id=job_id,
        )
    return record, {
        "path": str(resolved), "sha256": _sha256(resolved),
        "source_tree_sha256": digest, "snapshot_root": snapshot_root,
        "orbital_artifact_sha256": orbitals[0],
        "orbital_fingerprint": orbitals[1],
        "orbital_artifact_path": orbitals[2],
        "topology": topology, "terminal_slurm": terminal,
    }


def _complete_iteration_timing(record: Mapping[str, Any]) -> dict[str, Any]:
    experiment = record.get("experiment_record")
    phases = experiment.get("phases") if isinstance(experiment, dict) else None
    if isinstance(phases, list):
        matches = [
            phase for phase in phases if isinstance(phase, dict)
            and phase.get("name") == "ccsd_iterations"
            and phase.get("depth") == 0
        ]
        if len(matches) == 1 and isinstance(matches[0].get("elapsed_s"), (int, float)):
            seconds = float(matches[0]["elapsed_s"])
            if math.isfinite(seconds) and seconds > 0:
                return {
                    "seconds": seconds,
                    "source": "experiment_record.phases.ccsd_iterations",
                    "iteration_count": record.get("cc_iterations"),
                }
    metrics = record.get("run_metrics")
    phases = metrics.get("phases") if isinstance(metrics, dict) else None
    names = {"ccsd_energy", "ccsd_update", "update_norm", "diis"}
    selected = [
        phase for phase in phases or [] if isinstance(phase, dict)
        and phase.get("depth") == 1 and phase.get("name") in names
        and isinstance(phase.get("elapsed_s"), (int, float))
    ]
    updates = sum(phase.get("name") == "ccsd_update" for phase in selected)
    if updates != record.get("cc_iterations") or not selected:
        raise ContractError("record lacks complete CCSD-iteration timing")
    seconds = sum(float(phase["elapsed_s"]) for phase in selected)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ContractError("complete CCSD-iteration timing is invalid")
    return {
        "seconds": seconds,
        "source": "sum(run_metrics depth-1 CCSD iteration phases)",
        "phase_names": sorted(names), "phase_count": len(selected),
        "iteration_count": updates,
    }


def _probe_measurement(record: Mapping[str, Any], cutoff: float) -> dict[str, Any]:
    if record.get("status") != "not_converged" or record.get("cc_iterations") != 1:
        raise ContractError("rank probe must be exactly one non-converged RR cycle")
    approximation = record.get("approximation")
    if not isinstance(approximation, dict) or not _same_cutoff(
        approximation.get("rr_eig_cutoff"), cutoff,
    ):
        raise ContractError("rank probe cutoff does not match")
    ranks = record.get("ranks")
    rank = ranks.get("rr_rank") if isinstance(ranks, dict) else None
    dimension = ranks.get("rr_full_dimension") if isinstance(ranks, dict) else None
    projector = record.get("method_metadata", {}).get("rr_projector_build", {})
    if (
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
        or dimension != PAIR_DIMENSION
        or not isinstance(projector, dict)
        or projector.get("cutoff_bracketed") is not True
        or projector.get("capped") is not False
    ):
        raise ContractError("rank probe lacks a valid bracketed WATER4 rank")
    return {
        "cutoff": cutoff, "rank": rank,
        "rank_fraction": rank / PAIR_DIMENSION,
        "at_or_above_stop_boundary": rank >= RANK_STOP_BOUNDARY,
        "projector_solver": projector.get("solver"),
        "requested_subspace": projector.get("requested_subspace"),
        "max_ritz_residual": projector.get("max_ritz_residual"),
        "post_hf_seconds": record.get("post_hf_seconds"),
        "performance_eligible": False,
    }


def decide_gate(
    *, scientific_pass: bool, rank: int, cutoff: float,
    previous_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the accuracy/rank sequence; speed promotion is separate."""

    previous_pass = bool(
        previous_gate is not None
        and previous_gate.get("scientific_pass") is True
    )
    if previous_pass:
        return {
            "route_stop": False, "sequence_complete": True,
            "requires_tighter": False, "next_cutoff": None,
            "reason": "first passing cutoff and its next tighter point are complete",
        }
    if scientific_pass:
        return {
            "route_stop": False, "sequence_complete": False,
            "requires_tighter": True,
            "next_cutoff": cutoff / TIGHTENING_FACTOR,
            "reason": "first passing cutoff found; one next tighter point is required",
        }
    if rank >= RANK_STOP_BOUNDARY:
        return {
            "route_stop": True, "sequence_complete": False,
            "requires_tighter": False, "next_cutoff": None,
            "reason": "accuracy failed at or above the 0.8*OV rank boundary",
        }
    return {
        "route_stop": False, "sequence_complete": False,
        "requires_tighter": True,
        "next_cutoff": cutoff / TIGHTENING_FACTOR,
        "reason": "accuracy failed below the 0.8*OV rank boundary",
    }


def promotion_decision(
    *, sequence_complete: bool, candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    promotable = [
        float(item["cutoff"]) for item in candidates
        if item.get("scientific_pass") is True
        and item.get("complete_iteration_faster_than_canonical") is True
    ]
    return {
        "promotable_cutoffs": sorted(set(promotable), reverse=True),
        "advance_to_water8": bool(sequence_complete and promotable),
        "performance_stop": bool(sequence_complete and not promotable),
    }


def _validate_spectrum(
    path: Path, *, task_root: Path, source_root: Path,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    resolved = _input_file(path, task_root=task_root, label="spectrum summary")
    payload = _load_json(resolved)
    if (
        payload.get("schema") != SPECTRUM_SCHEMA
        or payload.get("case_id") != CASE_ID
        or payload.get("pair_dimension") != PAIR_DIMENSION
        or payload.get("rank_stop_boundary") != RANK_STOP_BOUNDARY
        or payload.get("source", {}).get("snapshot_root") != str(source_root.parent)
        or payload.get("gint_runtime_gate_receipt", {}).get("sha256")
        != receipt["sha256"]
        or payload.get("gint_release_gate_acceptance", {}).get("sha256")
        != receipt["release_gate_acceptance"]["sha256"]
    ):
        raise ContractError("invalid or mismatched WATER4 spectrum summary")
    probes = payload.get("probes")
    if not isinstance(probes, list) or len(probes) != 2:
        raise ContractError("initial spectrum must contain exactly two probes")
    for expected, probe in zip(INITIAL_CUTOFFS, probes, strict=True):
        if not isinstance(probe, dict) or not _same_cutoff(probe.get("cutoff"), expected):
            raise ContractError("initial spectrum cutoffs must be 1e-9 then 1e-11")
    return payload, resolved


def _validate_previous_gate(
    path: Path, *, task_root: Path, spectrum_sha256: str,
) -> tuple[dict[str, Any], Path]:
    resolved = _input_file(path, task_root=task_root, label="previous G4 gate")
    payload = _load_json(resolved)
    if (
        payload.get("schema") != GATE_SCHEMA
        or payload.get("case_id") != CASE_ID
        or payload.get("spectrum_summary_sha256") != spectrum_sha256
    ):
        raise ContractError("previous gate is outside this WATER4 chain")
    if payload.get("route_stop") is True or payload.get("sequence_complete") is True:
        raise ContractError("previous gate closes the continuation chain")
    if payload.get("requires_tighter") is not True:
        raise ContractError("previous gate does not authorize a tighter cutoff")
    return payload, resolved


def _matching_probe(
    spectrum: Mapping[str, Any], cutoff: float,
    custom: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    for probe in spectrum["probes"]:
        if _same_cutoff(probe.get("cutoff"), cutoff):
            return probe
    if custom is None or not _same_cutoff(custom.get("cutoff"), cutoff):
        raise ContractError("converged run needs a matching rank probe")
    return custom


def _spectrum(args: argparse.Namespace) -> int:
    task = args.task_root.resolve(strict=True)
    source = args.source_root.resolve(strict=True)
    receipt = _accepted_receipt_contract(
        source, args.gint_runtime_gate_receipt,
        args.gint_release_gate_acceptance, task_root=task,
    )
    output = _output_file(args.output, task_root=task)
    oracle, oracle_evidence = _record_evidence(
        args.oracle_record, task_root=task, source_root=source,
        receipt=receipt, method="canonical", probe=False,
    )
    if oracle.get("status") != "completed" or oracle.get("cc_converged") is not True:
        raise ContractError("paired canonical WATER4 oracle did not converge")
    probes: list[dict[str, Any]] = []
    for cutoff, path in zip(INITIAL_CUTOFFS, args.probe_record, strict=True):
        record, evidence = _record_evidence(
            path, task_root=task, source_root=source, receipt=receipt,
            method=METHOD, probe=True,
        )
        if (
            evidence["orbital_artifact_sha256"]
            != oracle_evidence["orbital_artifact_sha256"]
            or evidence["orbital_fingerprint"]
            != oracle_evidence["orbital_fingerprint"]
        ):
            raise ContractError("rank probe does not share canonical orbitals")
        probes.append({**_probe_measurement(record, cutoff), "evidence": evidence})
    payload = {
        "schema": SPECTRUM_SCHEMA, "created_utc": _utc_now(),
        "case_id": CASE_ID, "stage": "rank-spectrum",
        "scientific_role": "rank-and-cutoff-bracketing-only",
        "performance_eligible": False,
        "pair_dimension": PAIR_DIMENSION,
        "rank_stop_boundary": RANK_STOP_BOUNDARY,
        "rank_stop_fraction": 0.8,
        "source": {
            "tree_sha256": oracle_evidence["source_tree_sha256"],
            "snapshot_root": oracle_evidence["snapshot_root"],
        },
        "oracle": {
            "e_corr_eh": oracle.get("e_corr"),
            "post_hf_seconds": oracle.get("post_hf_seconds"),
            "complete_iteration_timing": _complete_iteration_timing(oracle),
            "evidence": oracle_evidence,
        },
        "gint_runtime_gate_receipt": receipt,
        "gint_release_gate_acceptance": receipt["release_gate_acceptance"],
        "probes": probes,
        "decision": "converge 1e-9 first; follow each gate exactly",
    }
    _exclusive_json(output, payload)
    print(json.dumps({"spectrum_summary": str(output), "probes": probes}))
    return 0


def _probe(args: argparse.Namespace) -> int:
    task = args.task_root.resolve(strict=True)
    source = args.source_root.resolve(strict=True)
    receipt = _accepted_receipt_contract(
        source, args.gint_runtime_gate_receipt,
        args.gint_release_gate_acceptance, task_root=task,
    )
    spectrum, spectrum_path = _validate_spectrum(
        args.spectrum_summary, task_root=task, source_root=source, receipt=receipt,
    )
    spectrum_sha = _sha256(spectrum_path)
    previous, previous_path = _validate_previous_gate(
        args.previous_gate, task_root=task, spectrum_sha256=spectrum_sha,
    )
    cutoff = _cutoff(args.rr_eig_cutoff)
    expected = float(previous["cutoff"]) / TIGHTENING_FACTOR
    if not _same_cutoff(cutoff, expected) or cutoff >= 1e-11:
        raise ContractError("custom probe must be the next 100-fold tighter cutoff")
    record, evidence = _record_evidence(
        args.rr_record, task_root=task, source_root=source, receipt=receipt,
        method=METHOD, probe=True,
    )
    if evidence["orbital_artifact_sha256"] != (
        spectrum["oracle"]["evidence"]["orbital_artifact_sha256"]
    ):
        raise ContractError("custom probe does not share canonical orbitals")
    output = _output_file(args.output, task_root=task)
    payload = {
        "schema": PROBE_SCHEMA, "created_utc": _utc_now(),
        "case_id": CASE_ID,
        "spectrum_summary": str(spectrum_path),
        "spectrum_summary_sha256": spectrum_sha,
        "previous_gate": str(previous_path),
        "previous_gate_sha256": _sha256(previous_path),
        **_probe_measurement(record, cutoff), "evidence": evidence,
    }
    _exclusive_json(output, payload)
    print(json.dumps({"probe_summary": str(output), "rank": payload["rank"]}))
    return 0


def _projected_residual(record: Mapping[str, Any]) -> float | None:
    residual = record.get("residual")
    value = residual.get("projected_equation") if isinstance(residual, dict) else None
    if value is None and isinstance(residual, dict):
        value = residual.get("measurements", {}).get("projected_equation", {}).get("norm")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0.0:
        return None
    return result


def _diagnostic_gate(
    record: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, float | None]]:
    method = record.get("method_metadata")
    build = method.get("rr_projector_build") if isinstance(method, dict) else None
    projector = build.get("projector") if isinstance(build, dict) else None
    orthogonality = _finite_nonnegative(
        projector.get("orthogonality_error")
        if isinstance(projector, dict) else None
    )
    full_space = record.get("full_space_residual_diagnostic")
    full_residual = _finite_nonnegative(
        full_space.get("residual_norm")
        if isinstance(full_space, dict) else None
    )
    checks = {
        "projector_orthogonality_1e_10": orthogonality is not None
        and orthogonality <= PROJECTOR_ORTHOGONALITY_TOLERANCE,
        "full_space_diagnostic_recorded": isinstance(full_space, dict)
        and full_space.get("available") is True
        and full_space.get("execution_status") == "completed"
        and full_residual is not None,
    }
    measurements = {
        "projector_orthogonality_error": orthogonality,
        "full_space_equation_residual_diagnostic": full_residual,
    }
    return checks, measurements


def _gate(args: argparse.Namespace) -> int:
    task = args.task_root.resolve(strict=True)
    source = args.source_root.resolve(strict=True)
    receipt = _accepted_receipt_contract(
        source, args.gint_runtime_gate_receipt,
        args.gint_release_gate_acceptance, task_root=task,
    )
    spectrum, spectrum_path = _validate_spectrum(
        args.spectrum_summary, task_root=task, source_root=source, receipt=receipt,
    )
    spectrum_sha = _sha256(spectrum_path)
    previous = previous_path = None
    if args.previous_gate:
        previous, previous_path = _validate_previous_gate(
            args.previous_gate, task_root=task, spectrum_sha256=spectrum_sha,
        )
    cutoff = _cutoff(args.rr_eig_cutoff)
    if _same_cutoff(cutoff, 1e-9):
        if previous is not None:
            raise ContractError("1e-9 must be the first converged point")
    elif previous is None or not _same_cutoff(
        cutoff, float(previous["cutoff"]) / TIGHTENING_FACTOR,
    ):
        raise ContractError("cutoffs must continue in 100-fold tighter steps")
    custom = custom_path = None
    if args.probe_summary:
        custom_path = _input_file(
            args.probe_summary, task_root=task, label="custom probe summary",
        )
        custom = _load_json(custom_path)
        if custom.get("schema") != PROBE_SCHEMA or (
            custom.get("spectrum_summary_sha256") != spectrum_sha
        ):
            raise ContractError("custom probe is outside this chain")
    probe = _matching_probe(spectrum, cutoff, custom)
    record, evidence = _record_evidence(
        args.rr_record, task_root=task, source_root=source, receipt=receipt,
        method=METHOD, probe=False,
    )
    rank = record.get("ranks", {}).get("rr_rank")
    dimension = record.get("ranks", {}).get("rr_full_dimension")
    if (
        isinstance(rank, bool) or not isinstance(rank, int)
        or dimension != PAIR_DIMENSION or rank != probe.get("rank")
    ):
        raise ContractError("converged result rank differs from its probe")
    if evidence["orbital_artifact_sha256"] != (
        spectrum["oracle"]["evidence"]["orbital_artifact_sha256"]
    ) or evidence["orbital_fingerprint"] != (
        spectrum["oracle"]["evidence"]["orbital_fingerprint"]
    ):
        raise ContractError("converged result does not share canonical orbitals")
    approximation = record.get("approximation", {})
    if not _same_cutoff(approximation.get("rr_eig_cutoff"), cutoff):
        raise ContractError("converged result cutoff differs from request")

    try:
        oracle_energy = float(spectrum["oracle"]["e_corr_eh"])
        candidate_energy = float(record.get("e_corr"))
    except (TypeError, ValueError):
        oracle_energy = candidate_energy = math.nan
    energy_error = (
        abs(candidate_energy - oracle_energy)
        if math.isfinite(oracle_energy) and math.isfinite(candidate_energy)
        else None
    )
    projected = _projected_residual(record)
    hbm = _finite_nonnegative(record.get("hbm", {}).get("peak_process_MiB"))
    rss = _finite_nonnegative(record.get("peak_host_RSS_GiB"))
    checkpoint = record.get("checkpoint", {})
    diagnostic_checks, diagnostic_measurements = _diagnostic_gate(record)
    checks = {
        "job_completed_0_0": evidence["terminal_slurm"]["state"] == "COMPLETED"
        and evidence["terminal_slurm"]["exit_code"] == "0:0",
        "cc_converged": record.get("status") == "completed"
        and record.get("cc_converged") is True,
        "energy_1e_4_eh": energy_error is not None
        and energy_error <= ENERGY_TOLERANCE_EH,
        "projected_equation_1e_6": projected is not None
        and projected <= PROJECTED_RESIDUAL_TOLERANCE,
        **diagnostic_checks,
        "hbm_72_gib": hbm is not None and hbm <= HBM_LIMIT_MIB,
        "host_rss_110_gib": rss is not None and rss <= HOST_RSS_LIMIT_GIB,
        "normal_checkpoint": isinstance(checkpoint, dict)
        and checkpoint.get("included_in_post_hf") is True,
    }
    scientific_pass = all(checks.values())
    candidate_iterations = _complete_iteration_timing(record)
    oracle_iterations = spectrum["oracle"]["complete_iteration_timing"]
    iteration_faster = (
        candidate_iterations["seconds"] < oracle_iterations["seconds"]
    )
    performance = {
        "paired_same_source_orbitals_node": True,
        "candidate_complete_iteration_seconds": candidate_iterations["seconds"],
        "canonical_complete_iteration_seconds": oracle_iterations["seconds"],
        "complete_iteration_faster_than_canonical": iteration_faster,
        "complete_iteration_speedup": (
            oracle_iterations["seconds"] / candidate_iterations["seconds"]
        ),
        "candidate_post_hf_seconds": record.get("post_hf_seconds"),
        "canonical_post_hf_seconds": spectrum["oracle"]["post_hf_seconds"],
        "whole_stage_speedup_diagnostic": (
            float(spectrum["oracle"]["post_hf_seconds"])
            / float(record["post_hf_seconds"])
        ),
    }
    decision = decide_gate(
        scientific_pass=scientific_pass, rank=rank, cutoff=cutoff,
        previous_gate=previous,
    )
    candidates: list[dict[str, Any]] = []
    if previous is not None:
        candidates.extend(previous.get("accuracy_candidates", []))
    candidates.append({
        "cutoff": cutoff, "scientific_pass": scientific_pass,
        "complete_iteration_faster_than_canonical": iteration_faster,
    })
    promotion = promotion_decision(
        sequence_complete=decision["sequence_complete"], candidates=candidates,
    )
    output = _output_file(args.output, task_root=task)
    payload = {
        "schema": GATE_SCHEMA, "created_utc": _utc_now(),
        "case_id": CASE_ID, "method": METHOD, "cutoff": cutoff,
        "rank": rank, "pair_dimension": PAIR_DIMENSION,
        "rank_fraction": rank / PAIR_DIMENSION,
        "rank_stop_boundary": RANK_STOP_BOUNDARY,
        "scientific_pass": scientific_pass, "checks": checks,
        "measurements": {
            "e_corr_eh": candidate_energy,
            "oracle_e_corr_eh": oracle_energy,
            "abs_delta_e_corr_eh": energy_error,
            "projected_equation_residual": projected,
            **diagnostic_measurements,
            "peak_process_hbm_mib": hbm, "peak_host_rss_gib": rss,
        },
        "performance_gate": performance,
        "accuracy_candidates": candidates,
        **decision, **promotion,
        "water8_promotion_note": (
            "accuracy sequence and a faster complete WATER4 iteration are both required"
        ),
        "target_speed_claim_eligible": False,
        "source_tree_sha256": evidence["source_tree_sha256"],
        "spectrum_summary": str(spectrum_path),
        "spectrum_summary_sha256": spectrum_sha,
        "rank_probe": probe,
        "previous_gate": None if previous_path is None else str(previous_path),
        "previous_gate_sha256": (
            None if previous_path is None else _sha256(previous_path)
        ),
        "custom_probe_summary": None if custom_path is None else str(custom_path),
        "custom_probe_summary_sha256": (
            None if custom_path is None else _sha256(custom_path)
        ),
        "result_evidence": evidence,
        "gint_release_gate_acceptance": receipt["release_gate_acceptance"],
    }
    _exclusive_json(output, payload)
    print(json.dumps({
        "g4_gate": str(output), "accuracy_decision": decision,
        "promotion_decision": promotion,
    }))
    return 0


def _receipt_node(args: argparse.Namespace) -> int:
    contract = _accepted_receipt_contract(
        args.source_root, args.gint_runtime_gate_receipt,
        args.gint_release_gate_acceptance,
    )
    print(contract["node"])
    return 0


def _authorize(args: argparse.Namespace) -> int:
    """Fail closed before a standard RR job is submitted."""

    task = args.task_root.resolve(strict=True)
    source = args.source_root.resolve(strict=True)
    receipt = _accepted_receipt_contract(
        source, args.gint_runtime_gate_receipt,
        args.gint_release_gate_acceptance, task_root=task,
    )
    spectrum, spectrum_path = _validate_spectrum(
        args.spectrum_summary, task_root=task, source_root=source, receipt=receipt,
    )
    spectrum_sha = _sha256(spectrum_path)
    previous = None
    if args.previous_gate:
        previous, _ = _validate_previous_gate(
            args.previous_gate, task_root=task, spectrum_sha256=spectrum_sha,
        )
    cutoff = _cutoff(args.rr_eig_cutoff)
    if args.stage == "probe":
        if previous is None or cutoff >= 1e-11 or not _same_cutoff(
            cutoff, float(previous["cutoff"]) / TIGHTENING_FACTOR,
        ):
            raise ContractError("custom probe is not the authorized next cutoff")
    elif _same_cutoff(cutoff, 1e-9):
        if previous is not None:
            raise ContractError("1e-9 must be the first converged point")
    elif previous is None or not _same_cutoff(
        cutoff, float(previous["cutoff"]) / TIGHTENING_FACTOR,
    ):
        raise ContractError("converged cutoff is not the authorized next point")
    if args.stage == "converge":
        custom = None
        if args.probe_summary:
            custom_path = _input_file(
                args.probe_summary, task_root=task, label="custom probe summary",
            )
            custom = _load_json(custom_path)
            if custom.get("schema") != PROBE_SCHEMA or (
                custom.get("spectrum_summary_sha256") != spectrum_sha
            ):
                raise ContractError("custom probe is outside this chain")
        _matching_probe(spectrum, cutoff, custom)
    print(json.dumps({
        "node": receipt["node"], "cutoff": cutoff,
        "orbital_artifact_path": (
            spectrum["oracle"]["evidence"]["orbital_artifact_path"]
        ),
    }))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    node = commands.add_parser("receipt-node")
    node.add_argument("--source-root", type=Path, required=True)
    node.add_argument("--gint-runtime-gate-receipt", type=Path, required=True)
    node.add_argument("--gint-release-gate-acceptance", type=Path, required=True)
    node.set_defaults(handler=_receipt_node)

    authorize = commands.add_parser("authorize")
    authorize.add_argument("--task-root", type=Path, required=True)
    authorize.add_argument("--source-root", type=Path, required=True)
    authorize.add_argument(
        "--gint-runtime-gate-receipt", type=Path, required=True,
    )
    authorize.add_argument(
        "--gint-release-gate-acceptance", type=Path, required=True,
    )
    authorize.add_argument("--stage", choices=("probe", "converge"), required=True)
    authorize.add_argument("--rr-eig-cutoff", type=float, required=True)
    authorize.add_argument("--spectrum-summary", type=Path, required=True)
    authorize.add_argument("--previous-gate", type=Path)
    authorize.add_argument("--probe-summary", type=Path)
    authorize.set_defaults(handler=_authorize)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task-root", type=Path, required=True)
    common.add_argument("--source-root", type=Path, required=True)
    common.add_argument("--gint-runtime-gate-receipt", type=Path, required=True)
    common.add_argument("--gint-release-gate-acceptance", type=Path, required=True)
    common.add_argument("--output", type=Path, required=True)
    spectrum = commands.add_parser("spectrum", parents=[common])
    spectrum.add_argument("--oracle-record", type=Path, required=True)
    spectrum.add_argument("--probe-record", type=Path, action="append", required=True)
    spectrum.set_defaults(handler=_spectrum)
    probe = commands.add_parser("probe", parents=[common])
    probe.add_argument("--rr-eig-cutoff", type=float, required=True)
    probe.add_argument("--spectrum-summary", type=Path, required=True)
    probe.add_argument("--previous-gate", type=Path, required=True)
    probe.add_argument("--rr-record", type=Path, required=True)
    probe.set_defaults(handler=_probe)
    gate = commands.add_parser("gate", parents=[common])
    gate.add_argument("--rr-eig-cutoff", type=float, required=True)
    gate.add_argument("--spectrum-summary", type=Path, required=True)
    gate.add_argument("--previous-gate", type=Path)
    gate.add_argument("--probe-summary", type=Path)
    gate.add_argument("--rr-record", type=Path, required=True)
    gate.set_defaults(handler=_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "spectrum" and len(args.probe_record) != 2:
        raise ContractError("spectrum requires exactly two --probe-record values")
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"G4 continuation contract error: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
