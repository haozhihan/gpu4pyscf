#!/usr/bin/env python3
"""Submit and accept the receipt-pinned selected-GINT release gate.

The qualification allocation cannot attest its own terminal scheduler state.
Likewise, a successful release allocation is not an authorization artifact by
itself.  This module creates a reviewable submission plan, submits that exact
plan on the node pinned by the qualification receipt, and only then issues a
write-once acceptance after validating the release result, topology, Slurm
output, and terminal ``sacct`` row.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PLAN_SCHEMA = "gpu4pyscf.gint-release-gate-plan.v1"
SUBMISSION_SCHEMA = "gpu4pyscf.gint-release-gate-submission.v1"
ACCEPTANCE_SCHEMA = "gpu4pyscf.gint-release-gate-acceptance.v1"
SIDECAR_SUFFIX = ".sha256"
ALLOWED_MTU_A100_NODES = frozenset({
    "compute-1-0", "compute-1-2", "compute-1-3", "compute-1-5",
    "compute-1-6",
})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """Raised when the release-gate evidence chain is not exact."""


def _load_sibling(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_receipt = _load_sibling(
    "water8_gint_release_gate_receipt", "gint_gate_receipt.py"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value), indent=2, sort_keys=True, ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
    ).encode("ascii")


def _json_object(path: Path, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    try:
        resolved, raw, _ = _receipt._regular_file(path, name=label)
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    try:
        value = _receipt._json_object(raw, name=label)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"cannot decode {label}: {exc}") from exc
    return resolved, raw, value


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve(strict=False)
    if target != path.expanduser().absolute():
        raise ContractError("output path must not traverse symlinks")
    if not target.parent.is_dir():
        raise ContractError(f"output parent does not exist: {target.parent}")
    encoded = _json_bytes(payload)
    descriptor = os.open(
        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _file_evidence(path: Path, label: str) -> dict[str, Any]:
    try:
        resolved, raw, info = _receipt._regular_file(path, name=label)
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    return {
        "path": str(resolved), "sha256": _sha256_bytes(raw),
        "bytes": int(info.st_size),
    }


def _release_contract(
    task_root: Path, source_root: Path, receipt_path: Path,
) -> dict[str, Any]:
    task = task_root.expanduser().resolve(strict=True)
    source = source_root.expanduser().resolve(strict=True)
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    try:
        relative = source.relative_to(task / "snapshots")
    except ValueError as exc:
        raise ContractError("release source is outside the task snapshots") from exc
    if (
        len(relative.parts) != 2 or relative.parts[1] != "source"
        or SHA256_RE.fullmatch(relative.parts[0]) is None
    ):
        raise ContractError("release source must be snapshots/<sha256>/source")
    source_digest = relative.parts[0]
    if not _is_below(receipt_path, task / "results"):
        raise ContractError("qualification receipt is outside task results")
    try:
        expected_pin = _receipt.build_release_pin(receipt_path)
    except (OSError, ValueError) as exc:
        raise ContractError(f"qualification receipt is invalid: {exc}") from exc
    receipt_file, receipt_raw, envelope = _json_object(
        receipt_path, "qualification receipt"
    )
    payload = envelope.get("payload")
    payload_sha = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or _payload_sha256(payload) != payload_sha:
        raise ContractError("qualification receipt payload digest is invalid")
    pin_path, pin_raw, pin = _json_object(
        source / _receipt.RELEASE_PIN_RELATIVE_PATH, "release pin"
    )
    if pin != expected_pin:
        raise ContractError("release pin does not bind the qualification receipt")
    pin_info = pin_path.lstat()
    if pin_info.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ContractError("release pin must be read-only")
    runtime = pin.get("release_runtime_contract")
    node = runtime.get("node") if isinstance(runtime, dict) else None
    job_name = runtime.get("job_name") if isinstance(runtime, dict) else None
    if (
        not isinstance(runtime, dict)
        or node not in ALLOWED_MTU_A100_NODES
        or runtime.get("host") != node
        or runtime.get("partition") != "mrigpu"
        or job_name != f"gint-release-{payload_sha}"
    ):
        raise ContractError("release runtime contract has an invalid node/job identity")
    manifest_path, manifest_raw, manifest = _json_object(
        source.parent / "manifest.json", "release manifest"
    )
    lineage = manifest.get("gint_release_lineage")
    if not isinstance(lineage, dict) or not all((
        manifest.get("source") == str(source),
        manifest.get("tree_sha256") == source_digest,
        manifest.get("immutable") is True,
        lineage.get("receipt_sha256") == _sha256_bytes(receipt_raw),
        lineage.get("receipt_payload_sha256") == payload_sha,
        lineage.get("release_pin_sha256") == _sha256_bytes(pin_raw),
    )):
        raise ContractError("release manifest lineage is invalid")
    try:
        actual_digest, _ = _receipt.release_source_tree_digest(source)
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot hash release source: {exc}") from exc
    if actual_digest != source_digest:
        raise ContractError("release source content differs from its digest")
    return {
        "task_root": str(task),
        "source_root": str(source),
        "source_tree_sha256": source_digest,
        "receipt": {
            "path": str(receipt_file),
            "sha256": _sha256_bytes(receipt_raw),
            "payload_sha256": payload_sha,
        },
        "pin": {"path": str(pin_path), "sha256": _sha256_bytes(pin_raw)},
        "manifest": {
            "path": str(manifest_path), "sha256": _sha256_bytes(manifest_raw),
        },
        "node": node,
        "job_name": job_name,
    }


def make_plan(
    *, task_root: Path, source_root: Path, qualification_receipt: Path,
) -> dict[str, Any]:
    contract = _release_contract(task_root, source_root, qualification_receipt)
    task = Path(contract["task_root"])
    source = Path(contract["source_root"])
    launcher = source / "benchmarks/cc/a100_water8/run_mtu_gint_gate.sbatch"
    if not launcher.is_file():
        raise ContractError("release source is missing run_mtu_gint_gate.sbatch")
    exports = {
        "CCSD_TASK_ROOT": str(task),
        "CCSD_SOURCE_ROOT": str(source),
        "CCSD_EXPECTED_DEPLOYMENT_PROFILE": "candidate",
        "CCSD_REQUIRE_CANONICAL_PRISTINE": "0",
        "GINT_RUNTIME_GATE_RECEIPT": contract["receipt"]["path"],
    }
    options = [
        "--nodes=1", "--ntasks=1", "--gres=gpu:nvidia:1",
        "--gres-flags=enforce-binding", f"--nodelist={contract['node']}",
        f"--job-name={contract['job_name']}",
    ]
    return {
        "schema": PLAN_SCHEMA,
        "created_utc": _utc_now(),
        "contract": contract,
        "launcher": str(launcher),
        "sbatch_options": options,
        "exports": exports,
        "expected_outputs": {
            "result_template": str(
                task / "results/gint-gate/water2-gint-direct-cd-${SLURM_JOB_ID}.json"
            ),
            "topology_template": str(
                task / "results/gint-gate/topology-physical8-${SLURM_JOB_ID}.json"
            ),
            "slurm_output_template": str(task / "logs/gint-gate-${SLURM_JOB_ID}.log"),
        },
    }


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ContractError("release-gate plan schema is invalid")
    contract = plan.get("contract")
    if not isinstance(contract, dict):
        raise ContractError("release-gate plan contract is missing")
    receipt = contract.get("receipt")
    if not isinstance(receipt, dict):
        raise ContractError("release-gate plan receipt binding is missing")
    fresh = make_plan(
        task_root=Path(str(contract.get("task_root", ""))),
        source_root=Path(str(contract.get("source_root", ""))),
        qualification_receipt=Path(str(receipt.get("path", ""))),
    )
    # Creation time is the only intentionally non-deterministic plan field.
    fresh["created_utc"] = plan.get("created_utc")
    if _canonical_json(fresh) != _canonical_json(plan):
        raise ContractError("release-gate plan differs from canonical regeneration")
    return dict(plan)


def _sbatch_argv(plan: Mapping[str, Any]) -> list[str]:
    exports = ["ALL"] + [
        f"{key}={value}" for key, value in sorted(plan["exports"].items())
    ]
    return [
        "sbatch", "--parsable", *plan["sbatch_options"],
        "--export=" + ",".join(exports), str(plan["launcher"]),
    ]


def submit_plan(
    plan: Mapping[str, Any], *, execute: bool, plan_path: Path | None = None,
) -> dict[str, Any]:
    checked = _validate_plan(plan)
    argv = _sbatch_argv(checked)
    job_id: str | None = None
    if execute:
        if plan_path is None:
            raise ContractError("executed submission requires its plan path")
        raw = subprocess.check_output(argv, text=True).strip()
        job_id = raw.split(";", 1)[0]
        if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
            raise ContractError(f"sbatch returned an invalid job id: {raw!r}")
    outputs = {
        key.removesuffix("_template"): value.replace("${SLURM_JOB_ID}", job_id or "<job-id>")
        for key, value in checked["expected_outputs"].items()
    }
    return {
        "schema": SUBMISSION_SCHEMA,
        "created_utc": _utc_now(),
        "executed": execute,
        "job_id": job_id,
        "job_name": checked["contract"]["job_name"],
        "node": checked["contract"]["node"],
        "source_tree_sha256": checked["contract"]["source_tree_sha256"],
        "qualification_receipt": checked["contract"]["receipt"],
        "release_pin": checked["contract"]["pin"],
        "plan_path": None if plan_path is None else str(plan_path.resolve(strict=True)),
        "plan_sha256": None if plan_path is None else _sha256(plan_path.resolve(strict=True)),
        "plan_payload_sha256": _payload_sha256(checked),
        "argv": argv,
        "shell_preview": shlex.join(argv),
        "expected_outputs": outputs,
    }


def _validate_submission(
    submission_path: Path,
) -> tuple[Path, bytes, dict[str, Any], dict[str, Any]]:
    path, raw, submission = _json_object(submission_path, "release submission receipt")
    if (
        submission.get("schema") != SUBMISSION_SCHEMA
        or submission.get("executed") is not True
        or re.fullmatch(r"[1-9][0-9]*", str(submission.get("job_id", ""))) is None
    ):
        raise ContractError("release submission receipt is not executed")
    plan_path = Path(str(submission.get("plan_path", "")))
    resolved_plan, plan_raw, plan = _json_object(plan_path, "release submission plan")
    checked = _validate_plan(plan)
    if not all((
        submission.get("plan_sha256") == _sha256_bytes(plan_raw),
        submission.get("plan_payload_sha256") == _payload_sha256(checked),
        submission.get("job_name") == checked["contract"]["job_name"],
        submission.get("node") == checked["contract"]["node"],
        submission.get("source_tree_sha256") == checked["contract"]["source_tree_sha256"],
        submission.get("qualification_receipt") == checked["contract"]["receipt"],
        submission.get("release_pin") == checked["contract"]["pin"],
        submission.get("argv") == _sbatch_argv(checked),
        resolved_plan == plan_path.resolve(strict=True),
    )):
        raise ContractError("release submission receipt differs from its plan")
    job_id = str(submission["job_id"])
    expected_outputs = {
        key.removesuffix("_template"): value.replace("${SLURM_JOB_ID}", job_id)
        for key, value in checked["expected_outputs"].items()
    }
    if submission.get("expected_outputs") != expected_outputs:
        raise ContractError("release submission output paths were changed")
    return path, raw, submission, checked


def query_terminal_slurm(job_id: str) -> dict[str, str]:
    fields = (
        "JobIDRaw,JobName,Partition,NodeList,State,ExitCode,"
        "Submit,Start,End,Elapsed"
    )
    output = subprocess.check_output([
        "/usr/bin/sacct", "--noheader", "--parsable2", "--allocations",
        "--jobs", job_id, "--format", fields,
    ], text=True, stderr=subprocess.STDOUT, timeout=30)
    names = fields.split(",")
    matches: list[dict[str, str]] = []
    for line in output.splitlines():
        values = line.split("|")
        if values and values[-1] == "":
            values.pop()
        if len(values) == len(names):
            row = dict(zip(names, values, strict=True))
            if row["JobIDRaw"] == job_id:
                matches.append(row)
    if len(matches) != 1:
        raise ContractError("sacct did not return exactly one release allocation")
    row = matches[0]
    return dict(zip(
        ("job_id", "job_name", "partition", "node", "state", "exit_code",
         "submit_utc", "start_utc", "end_utc", "elapsed"),
        (row[name] for name in names), strict=True,
    ))


def build_acceptance(
    *, submission_receipt: Path, release_result: Path,
    release_topology: Path, slurm_output: Path,
    slurm_terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    submission_path, submission_raw, submission, plan = _validate_submission(
        submission_receipt
    )
    contract = plan["contract"]
    task = Path(contract["task_root"])
    plan_path = Path(str(submission["plan_path"])).resolve(strict=True)
    if not all((
        _is_below(submission_path, task / "results"),
        _is_below(plan_path, task / "results"),
    )):
        raise ContractError("release plan and submission must be below task results")
    job_id = str(submission["job_id"])
    terminal = dict(
        query_terminal_slurm(job_id) if slurm_terminal is None else slurm_terminal
    )
    expected_terminal = {
        "job_id": job_id,
        "job_name": contract["job_name"],
        "partition": "mrigpu",
        "node": contract["node"],
        "state": "COMPLETED",
        "exit_code": "0:0",
    }
    for key, expected in expected_terminal.items():
        if terminal.get(key) != expected:
            raise ContractError(
                f"terminal Slurm {key} is {terminal.get(key)!r}; expected {expected!r}"
            )
    for key in ("submit_utc", "start_utc", "end_utc", "elapsed"):
        if not isinstance(terminal.get(key), str) or not terminal[key]:
            raise ContractError(f"terminal Slurm {key} is missing")

    expected_paths = submission["expected_outputs"]
    result_path, result_raw, result = _json_object(release_result, "release result")
    topology_path, topology_raw, topology = _json_object(
        release_topology, "release topology"
    )
    slurm_path = slurm_output.expanduser().resolve(strict=True)
    if not all((
        str(result_path) == expected_paths["result"],
        str(topology_path) == expected_paths["topology"],
        str(slurm_path) == expected_paths["slurm_output"],
    )):
        raise ContractError("release evidence paths differ from the submission")
    slurm_evidence = _file_evidence(slurm_path, "release Slurm output")
    source = result.get("source")
    source_snapshot = source.get("snapshot") if isinstance(source, dict) else None
    topology_evidence = result.get("topology")
    provider = result.get("provider")
    runtime_gate = provider.get("runtime_gate") if isinstance(provider, dict) else None
    runtime_receipt = runtime_gate.get("receipt") if isinstance(runtime_gate, dict) else None
    decision = result.get("gate_decision")
    if not all(isinstance(item, dict) for item in (
        source, source_snapshot, topology_evidence, provider, runtime_gate,
        runtime_receipt, decision,
    )):
        raise ContractError("release result has incomplete performance evidence")
    if not all((
        result.get("schema") == _receipt.GINT_GATE_RESULT_SCHEMA,
        result.get("status") == "completed",
        result.get("performance_eligible") is True,
        decision.get("correctness_passed") is True,
        decision.get("qualification_passed") is True,
        decision.get("performance_eligible") is True,
        provider.get("performance_eligible") is True,
        runtime_gate.get("validated") is True,
        runtime_gate.get("receipt_binding_validated") is True,
        runtime_gate.get("status") == "release-accepted",
        runtime_gate.get("execution_mode") == "release-gate",
        runtime_receipt.get("path") == contract["receipt"]["path"],
        runtime_receipt.get("sha256") == contract["receipt"]["sha256"],
        runtime_receipt.get("payload_sha256") == contract["receipt"]["payload_sha256"],
        runtime_receipt.get("release_source_sha256") == contract["source_tree_sha256"],
        runtime_receipt.get("release_pin_sha256") == contract["pin"]["sha256"],
        runtime_receipt.get("release_job_id") == job_id,
        runtime_receipt.get("release_topology_path") == str(topology_path),
        runtime_receipt.get("release_topology_sha256") == _sha256_bytes(topology_raw),
        source.get("root") == contract["source_root"],
        source.get("tree_sha256_at_start") == contract["source_tree_sha256"],
        source.get("tree_sha256_at_end") == contract["source_tree_sha256"],
        source.get("stable_during_run") is True,
        source_snapshot.get("manifest_path") == contract["manifest"]["path"],
        source_snapshot.get("manifest_sha256_at_start") == contract["manifest"]["sha256"],
        source_snapshot.get("manifest_sha256_at_end") == contract["manifest"]["sha256"],
        source_snapshot.get("stable_during_run") is True,
        topology_evidence.get("path") == str(topology_path),
        topology_evidence.get("sha256_at_start") == _sha256_bytes(topology_raw),
        topology_evidence.get("sha256_at_end") == _sha256_bytes(topology_raw),
        topology_evidence.get("passed") is True,
        topology_evidence.get("stable_during_run") is True,
    )):
        raise ContractError("release result does not prove performance eligibility")
    topology_slurm = topology.get("slurm")
    command = topology.get("command")
    expected_driver = str(
        Path(contract["source_root"]) / "benchmarks/cc/a100_water8/gint_gate.py"
    )
    if not all((
        topology.get("performance_eligible") is True,
        topology.get("benchmark_track") == "performance",
        topology.get("host") == contract["node"],
        topology.get("mode") == "physical8",
        isinstance(topology_slurm, dict),
        topology_slurm.get("SLURM_JOB_ID") == job_id,
        topology_slurm.get("SLURM_JOB_NAME") == contract["job_name"],
        topology_slurm.get("SLURM_JOB_NODELIST") == contract["node"],
        topology_slurm.get("SLURM_JOB_PARTITION") == "mrigpu",
        isinstance(command, list),
        expected_driver in [str(item) for item in command],
    )):
        raise ContractError("release topology does not bind the exact allocation")
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "created_utc": _utc_now(),
        "status": "accepted",
        "performance_eligible": True,
        "source": {
            "root": contract["source_root"],
            "tree_sha256": contract["source_tree_sha256"],
            "manifest": contract["manifest"],
        },
        "qualification_receipt": contract["receipt"],
        "release_pin": contract["pin"],
        "node": contract["node"],
        "release_job": {"terminal_slurm": terminal},
        "evidence": {
            "submission": {
                "path": str(submission_path),
                "sha256": _sha256_bytes(submission_raw),
            },
            "result": {"path": str(result_path), "sha256": _sha256_bytes(result_raw)},
            "topology": {
                "path": str(topology_path), "sha256": _sha256_bytes(topology_raw),
            },
            "slurm_output": slurm_evidence,
        },
    }


def write_once_acceptance(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create a sealed two-file acceptance directory."""

    if not all((
        payload.get("schema") == ACCEPTANCE_SCHEMA,
        payload.get("status") == "accepted",
        payload.get("performance_eligible") is True,
    )):
        raise ContractError("only an accepted release gate can be sealed")
    target = path.expanduser().resolve(strict=False)
    if target.name != "acceptance.json":
        raise ContractError("release-gate acceptance must be named acceptance.json")
    sidecar = Path(str(target) + SIDECAR_SUFFIX)
    if target.parent.exists() or target.parent.is_symlink():
        raise FileExistsError("refusing to reuse release-gate acceptance directory")
    encoded = _json_bytes(payload)
    digest = _sha256_bytes(encoded)
    sidecar_bytes = f"{digest}  {target.name}\n".encode("ascii")
    os.mkdir(target.parent, 0o700)
    try:
        for output, data in ((target, encoded), (sidecar, sidecar_bytes)):
            descriptor = os.open(
                output, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0), 0o444,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        os.chmod(target.parent, 0o555)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        # Preserve any created inode.  Recovery must inspect it; deleting by
        # pathname would risk removing a concurrently replaced artifact.
        raise
    return {
        "path": str(target), "sha256": digest,
        "payload_sha256": _payload_sha256(payload), "sidecar": str(sidecar),
    }


def validate_release_acceptance(
    acceptance_path: Path, *, task_root: Path, source_root: Path,
    qualification_receipt: Path,
) -> dict[str, Any]:
    """Validate a sealed acceptance and return its downstream identity."""

    task = task_root.expanduser().resolve(strict=True)
    path, raw, payload = _json_object(acceptance_path, "release-gate acceptance")
    if path.name != "acceptance.json":
        raise ContractError("release-gate acceptance must be named acceptance.json")
    if not _is_below(path, task / "results"):
        raise ContractError("release-gate acceptance is outside task results")
    sidecar_path, sidecar_raw, sidecar_info = _receipt._regular_file(
        str(path) + SIDECAR_SUFFIX, name="release-gate acceptance sidecar"
    )
    digest = _sha256_bytes(raw)
    if sidecar_raw != f"{digest}  {path.name}\n".encode("ascii"):
        raise ContractError("release-gate acceptance sidecar is invalid")
    for item, label in ((path.lstat(), "acceptance"), (sidecar_info, "sidecar"),
                        (path.parent.lstat(), "directory")):
        if item.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ContractError(f"release-gate {label} must be sealed")
    if {item.name for item in path.parent.iterdir()} != {path.name, sidecar_path.name}:
        raise ContractError("release-gate acceptance directory has unexpected files")
    source = source_root.expanduser().resolve(strict=True)
    receipt = qualification_receipt.expanduser().resolve(strict=True)
    try:
        source_relative = source.relative_to(task / "snapshots")
    except ValueError as exc:
        raise ContractError("release source is outside task snapshots") from exc
    if len(source_relative.parts) != 2 or source_relative.parts[1] != "source":
        raise ContractError("release source must be snapshots/<sha256>/source")
    if not _is_below(receipt, task / "results"):
        raise ContractError("qualification receipt is outside task results")
    _, receipt_raw, envelope = _json_object(receipt, "qualification receipt")
    receipt_payload = envelope.get("payload")
    receipt_payload_sha = envelope.get("payload_sha256")
    if not isinstance(receipt_payload, dict) or (
        _payload_sha256(receipt_payload) != receipt_payload_sha
    ):
        raise ContractError("qualification receipt payload digest is invalid")
    pin_path, pin_raw, pin = _json_object(
        source / _receipt.RELEASE_PIN_RELATIVE_PATH, "release pin"
    )
    pin_receipt = pin.get("receipt")
    runtime = pin.get("release_runtime_contract")
    node = runtime.get("node") if isinstance(runtime, dict) else None
    source_digest = source.parent.name
    if SHA256_RE.fullmatch(source_digest) is None:
        raise ContractError("release source directory is not content-addressed")
    accepted_source = payload.get("source")
    accepted_manifest = (
        accepted_source.get("manifest")
        if isinstance(accepted_source, dict) else None
    )
    if not all((
        payload.get("schema") == ACCEPTANCE_SCHEMA,
        payload.get("status") == "accepted",
        payload.get("performance_eligible") is True,
        accepted_source == {
            "root": str(source),
            "tree_sha256": source_digest,
            "manifest": accepted_manifest,
        },
        payload.get("qualification_receipt") == {
            "path": str(receipt), "sha256": _sha256_bytes(receipt_raw),
            "payload_sha256": receipt_payload_sha,
        },
        payload.get("release_pin") == {
            "path": str(pin_path), "sha256": _sha256_bytes(pin_raw),
        },
        isinstance(pin_receipt, dict),
        pin_receipt.get("sha256") == _sha256_bytes(receipt_raw)
        if isinstance(pin_receipt, dict) else False,
        pin_receipt.get("payload_sha256") == receipt_payload_sha
        if isinstance(pin_receipt, dict) else False,
        node in ALLOWED_MTU_A100_NODES,
        isinstance(runtime, dict),
        runtime.get("host") == node if isinstance(runtime, dict) else False,
        payload.get("node") == node,
    )):
        raise ContractError("release-gate acceptance identity is inconsistent")
    release_job = payload.get("release_job")
    terminal = (
        release_job.get("terminal_slurm")
        if isinstance(release_job, dict) else None
    )
    expected_name = f"gint-release-{receipt_payload_sha}"
    if not isinstance(terminal, dict) or not all((
        terminal.get("job_name") == expected_name,
        terminal.get("node") == node,
        terminal.get("partition") == "mrigpu",
        terminal.get("state") == "COMPLETED",
        terminal.get("exit_code") == "0:0",
        isinstance(terminal.get("job_id"), str),
        terminal.get("job_id", "").isdigit(),
    )):
        raise ContractError("release-gate terminal evidence is not accepted")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "submission", "result", "topology", "slurm_output"
    }:
        raise ContractError("release-gate acceptance evidence set is incomplete")
    for label, binding in evidence.items():
        if not isinstance(binding, dict):
            raise ContractError(f"release-gate {label} evidence is malformed")
        try:
            evidence_path = Path(str(binding.get("path", ""))).resolve(
                strict=True
            )
        except OSError as exc:
            raise ContractError(
                f"release-gate {label} evidence is unavailable: {exc}"
            ) from exc
        if not evidence_path.is_file():
            raise ContractError(f"release-gate {label} evidence is not a file")
        if not _is_below(evidence_path, task / "results") and label != "slurm_output":
            raise ContractError(f"release-gate {label} evidence is outside results")
        if label == "slurm_output" and not _is_below(evidence_path, task / "logs"):
            raise ContractError("release-gate Slurm output is outside task logs")
        if _sha256(evidence_path) != binding.get("sha256"):
            raise ContractError(f"release-gate {label} evidence changed")
    manifest = accepted_manifest
    if not isinstance(manifest, dict):
        raise ContractError("release-gate acceptance has no release manifest binding")
    manifest_path = Path(str(manifest.get("path", ""))).resolve(strict=True)
    if manifest_path != source.parent / "manifest.json" or (
        _sha256(manifest_path) != manifest.get("sha256")
    ):
        raise ContractError("release-gate release manifest changed")
    created_utc = payload.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc:
        raise ContractError("release-gate acceptance creation time is missing")
    # The sidecar detects mutation but is not a signing authority.  Rebuild the
    # acceptance from every bound file so a fabricated JSON+sidecar pair cannot
    # bypass result, topology, submission, or source validation.
    try:
        rebuilt = build_acceptance(
            submission_receipt=Path(evidence["submission"]["path"]),
            release_result=Path(evidence["result"]["path"]),
            release_topology=Path(evidence["topology"]["path"]),
            slurm_output=Path(evidence["slurm_output"]["path"]),
            slurm_terminal=terminal,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise ContractError(
            f"release-gate acceptance evidence is invalid: {exc}"
        ) from exc
    rebuilt["created_utc"] = created_utc
    if _canonical_json(rebuilt) != _canonical_json(payload):
        raise ContractError(
            "release-gate acceptance differs from regenerated evidence"
        )
    return {
        "path": str(path), "sha256": digest,
        "payload_sha256": _payload_sha256(payload),
        "node": node, "source_tree_sha256": source_digest,
        "receipt_sha256": _sha256_bytes(receipt_raw),
        "receipt_payload_sha256": receipt_payload_sha,
        "release_pin_sha256": _sha256_bytes(pin_raw),
        "release_job_id": terminal["job_id"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--task-root", type=Path, required=True)
    plan.add_argument("--source-root", type=Path, required=True)
    plan.add_argument("--qualification-receipt", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--execute", action="store_true")
    submit.add_argument("--receipt", type=Path)
    accept = commands.add_parser("accept")
    accept.add_argument("--submission-receipt", type=Path, required=True)
    accept.add_argument("--release-result", type=Path, required=True)
    accept.add_argument("--release-topology", type=Path, required=True)
    accept.add_argument("--slurm-output", type=Path, required=True)
    accept.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--acceptance", type=Path, required=True)
    verify.add_argument("--task-root", type=Path, required=True)
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--qualification-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        payload = make_plan(
            task_root=args.task_root, source_root=args.source_root,
            qualification_receipt=args.qualification_receipt,
        )
        _write_once_json(args.output, payload)
        print(args.output.resolve())
        return 0
    if args.command == "submit":
        _, _, plan = _json_object(args.plan, "release-gate plan")
        if args.execute and args.receipt is None:
            raise ContractError("--execute requires --receipt")
        if not args.execute and args.receipt is not None:
            raise ContractError("--receipt is valid only with --execute")
        payload = submit_plan(plan, execute=args.execute, plan_path=args.plan)
        if args.execute:
            _write_once_json(args.receipt, payload)
            print(args.receipt.resolve())
        else:
            print(payload["shell_preview"])
        return 0
    if args.command == "accept":
        payload = build_acceptance(
            submission_receipt=args.submission_receipt,
            release_result=args.release_result,
            release_topology=args.release_topology,
            slurm_output=args.slurm_output,
        )
        evidence = write_once_acceptance(args.output, payload)
        print(json.dumps(evidence, sort_keys=True))
        return 0
    evidence = validate_release_acceptance(
        args.acceptance, task_root=args.task_root,
        source_root=args.source_root,
        qualification_receipt=args.qualification_receipt,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"selected-GINT release gate failed: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
