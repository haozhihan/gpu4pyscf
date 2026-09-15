#!/usr/bin/env python3
"""Issue a sealed receipt and source-addressed release pin after qualification.

Receipt issuance is deliberately separate from ``gint_gate.py``.  A Slurm job
cannot prove its own terminal state or hash its final scheduler output while it
is still running.  Run this command only after ``sacct`` reports COMPLETED/0:0;
the subsequent release run passes the resulting path back to ``gint_gate.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence

try:
    from .gint_gate import (
        SELECTED_COLUMNS_SYMBOL,
        SELECTED_DIAGONAL_SYMBOL,
        SELECTED_WORKSPACE_SIZE_SYMBOL,
        GINT_GATE_RESULT_SCHEMA,
        SCONTROL_QUALIFICATION_EVIDENCE_SCHEMA,
        evaluate_gate,
    )
except ImportError:
    _gate_spec = importlib.util.spec_from_file_location(
        "water2_gint_gate_for_receipt", Path(__file__).resolve().with_name(
            "gint_gate.py"
        )
    )
    if _gate_spec is None or _gate_spec.loader is None:
        raise RuntimeError("cannot load gint_gate.py")
    _gate_module = importlib.util.module_from_spec(_gate_spec)
    _gate_spec.loader.exec_module(_gate_module)
    evaluate_gate = _gate_module.evaluate_gate
    SELECTED_COLUMNS_SYMBOL = _gate_module.SELECTED_COLUMNS_SYMBOL
    SELECTED_DIAGONAL_SYMBOL = _gate_module.SELECTED_DIAGONAL_SYMBOL
    SELECTED_WORKSPACE_SIZE_SYMBOL = (
        _gate_module.SELECTED_WORKSPACE_SIZE_SYMBOL
    )
    GINT_GATE_RESULT_SCHEMA = _gate_module.GINT_GATE_RESULT_SCHEMA
    SCONTROL_QUALIFICATION_EVIDENCE_SCHEMA = (
        _gate_module.SCONTROL_QUALIFICATION_EVIDENCE_SCHEMA
    )

from gpu4pyscf.cc.gint_transfer_audit import (
    RELEASE_PIN_RELATIVE_PATH,
    RELEASE_PIN_SCHEMA,
    RUNTIME_GATE_PAYLOAD_SCHEMA,
    RUNTIME_GATE_RECEIPT_SCHEMA,
    RUNTIME_GATE_SIDECAR_SUFFIX,
    SCONTROL_HELPER_SCHEMA,
    SCONTROL_LOADER_EVIDENCE_SCHEMA,
    canonical_json_sha256,
    expected_runtime_release_pin,
    observe_scontrol_helper_identity,
    release_source_tree_digest,
    scontrol_helper_portable_binding,
    scontrol_loader_portable_binding,
    source_snapshot_normalization_errors,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_file(
    value: str | os.PathLike[str], *, name: str
) -> tuple[Path, bytes, os.stat_result]:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{name} path traverses symbolic link {current}")
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{name} must not be a symbolic link")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError(f"{name} changed while it was opened")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        data = b"".join(chunks)
    finally:
        os.close(descriptor)
    final = path.lstat()
    if (
        (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    ):
        raise ValueError(f"{name} changed while it was read")
    return path, data, final


def _json_object(data: bytes, *, name: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(data.decode("utf-8"), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{name} root must be an object")
    return value


def _binding(path: Path, data: bytes, info: os.stat_result) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_bytes(data),
        "bytes": int(info.st_size),
    }


def query_terminal_slurm(job_id: str) -> dict[str, str]:
    """Read one terminal allocation row from sacct."""

    fields = (
        "JobIDRaw,JobName,Partition,NodeList,State,ExitCode,"
        "Submit,Start,End,Elapsed"
    )
    output = subprocess.check_output(
        [
            "sacct", "--noheader", "--parsable2", "--allocations",
            "--jobs", job_id, "--format", fields,
        ],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    names = fields.split(",")
    matches = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if values and values[-1] == "":
            values.pop()
        if len(values) != len(names):
            continue
        row = dict(zip(names, values))
        if row["JobIDRaw"] == job_id:
            matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(
            f"sacct did not return exactly one allocation row for job {job_id}"
        )
    row = matches[0]
    return {
        "job_id": row["JobIDRaw"],
        "job_name": row["JobName"],
        "partition": row["Partition"],
        "node": row["NodeList"],
        "state": row["State"],
        "exit_code": row["ExitCode"],
        "submit_utc": row["Submit"],
        "start_utc": row["Start"],
        "end_utc": row["End"],
        "elapsed": row["Elapsed"],
    }


def build_receipt_payload(
    result_path: str | os.PathLike[str],
    slurm_output_path: str | os.PathLike[str],
    *,
    slurm_terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate qualification evidence and return the bound receipt payload."""

    result_file, result_bytes, result_stat = _regular_file(
        result_path, name="qualification result"
    )
    result = _json_object(result_bytes, name="qualification result")
    if result.get("schema") != GINT_GATE_RESULT_SCHEMA:
        raise ValueError("qualification result schema mismatch")
    decision = evaluate_gate(result)
    if decision.get("correctness_passed") is not True:
        raise ValueError("qualification result failed the correctness gate")
    if decision.get("qualification_passed") is not True:
        raise ValueError("qualification result failed the transfer gate")
    provider = result.get("provider") or {}
    runtime_gate = provider.get("runtime_gate") or {}
    if provider.get("performance_eligible") is not False:
        raise ValueError("receipt issuance requires an unreleased qualification run")
    if runtime_gate.get("validated") is not False:
        raise ValueError("qualification run already consumed a runtime receipt")

    source = result.get("source") or {}
    snapshot = source.get("snapshot") or {}
    source_root = Path(os.path.abspath(str(source.get("root", ""))))
    if not source_root.is_absolute() or source_root.name != "source":
        raise ValueError("qualification source root is invalid")
    current = Path(source_root.anchor)
    for part in source_root.parts[1:]:
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(
                f"qualification source path traverses symbolic link {current}"
            )
    if source_root.parent.parent.name != "snapshots":
        raise ValueError("qualification source is outside a snapshots directory")
    digest = source.get("tree_sha256_at_start")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("qualification source digest is invalid")
    if source.get("tree_sha256_at_end") != digest:
        raise ValueError("qualification source digest changed during the run")
    if source_root.parent.name != digest:
        raise ValueError("qualification snapshot directory does not match digest")
    if (source_root / RELEASE_PIN_RELATIVE_PATH).exists():
        raise ValueError("qualification snapshot A must not contain a release pin")
    actual_digest, _ = release_source_tree_digest(source_root)
    if actual_digest != digest:
        raise ValueError("qualification source content differs from its digest")
    for path in (source_root, *source_root.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"qualification source contains symlink {path}")
        if info.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(f"qualification source contains writable path {path}")

    manifest_file, manifest_bytes, manifest_stat = _regular_file(
        snapshot.get("manifest_path"), name="source snapshot manifest"
    )
    if manifest_file != source_root.parent / "manifest.json":
        raise ValueError("qualification manifest is outside snapshot A")
    if manifest_stat.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("qualification source manifest must be read-only")
    manifest = _json_object(manifest_bytes, name="source snapshot manifest")
    manifest_digest = _sha256_bytes(manifest_bytes)
    normalization_errors = source_snapshot_normalization_errors(
        manifest, source_root=source_root
    )
    if normalization_errors:
        raise ValueError(
            "qualification snapshot normalization is invalid: "
            + "; ".join(normalization_errors)
        )
    if (
        manifest_digest != snapshot.get("manifest_sha256_at_start")
        or manifest_digest != snapshot.get("manifest_sha256_at_end")
        or manifest.get("tree_sha256") != digest
        or manifest.get("source") != str(source_root)
        or manifest.get("immutable") is not True
    ):
        raise ValueError("qualification snapshot manifest binding is invalid")

    topology_file, topology_bytes, topology_stat = _regular_file(
        (result.get("topology") or {}).get("path"),
        name="qualification topology",
    )
    topology = _json_object(topology_bytes, name="qualification topology")
    topology_digest = _sha256_bytes(topology_bytes)
    topology_evidence = result.get("topology") or {}
    if (
        topology_digest != topology_evidence.get("sha256_at_start")
        or topology_digest != topology_evidence.get("sha256_at_end")
        or topology_evidence.get("passed") is not True
        or topology_evidence.get("stable_during_run") is not True
        or topology.get("performance_eligible") is not True
    ):
        raise ValueError("qualification topology binding is invalid")

    topology_slurm = topology.get("slurm") or {}
    job_id = topology_slurm.get("SLURM_JOB_ID")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("qualification topology has no valid Slurm job id")
    terminal = dict(
        query_terminal_slurm(job_id)
        if slurm_terminal is None else slurm_terminal
    )
    required_terminal = {
        "job_id": job_id,
        "job_name": topology_slurm.get("SLURM_JOB_NAME"),
        "node": topology_slurm.get("SLURM_JOB_NODELIST"),
        "partition": topology_slurm.get("SLURM_JOB_PARTITION"),
        "state": "COMPLETED",
        "exit_code": "0:0",
    }
    for key, expected in required_terminal.items():
        if terminal.get(key) != expected:
            raise ValueError(
                f"terminal Slurm {key} is {terminal.get(key)!r}; expected {expected!r}"
            )
    for key in ("submit_utc", "start_utc", "end_utc", "elapsed"):
        if not isinstance(terminal.get(key), str) or not terminal[key]:
            raise ValueError(f"terminal Slurm {key} is missing")

    output_file, output_bytes, output_stat = _regular_file(
        slurm_output_path, name="qualification Slurm output"
    )
    if job_id not in output_file.name:
        raise ValueError("qualification Slurm output name does not contain job id")

    c_abi = provider.get("c_abi") or {}
    selected_c_abi = {
        "columns": c_abi.get("columns"),
        "diagonal": c_abi.get("diagonal"),
        "workspace_size": c_abi.get("workspace_size"),
    }
    if selected_c_abi != {
        "columns": SELECTED_COLUMNS_SYMBOL,
        "diagonal": SELECTED_DIAGONAL_SYMBOL,
        "workspace_size": SELECTED_WORKSPACE_SIZE_SYMBOL,
    }:
        raise ValueError("qualification selected C ABI symbols are invalid")
    bpcache_abi = c_abi.get("basis_prod_cache_identity")
    selected_abi = c_abi.get("selected_pair_identity")
    if not isinstance(bpcache_abi, dict) or bpcache_abi.get("verified") is not True:
        raise ValueError("qualification BasisProdCache ABI identity is missing")
    if not isinstance(selected_abi, dict) or selected_abi.get("verified") is not True:
        raise ValueError("qualification selected-pair ABI identity is missing")
    libgint_file, libgint_bytes, libgint_stat = _regular_file(
        bpcache_abi.get("libgint_path"), name="qualification libgint"
    )
    if libgint_stat.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("qualification libgint must be read-only")
    if (
        _sha256_bytes(libgint_bytes) != bpcache_abi.get("libgint_sha256")
        or libgint_stat.st_size != bpcache_abi.get("libgint_bytes")
    ):
        raise ValueError("qualification libgint identity changed")
    try:
        libgint_relative = libgint_file.relative_to(source_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            "qualification libgint is outside the source snapshot"
        ) from exc
    runtime_files = ((manifest.get("runtime_binaries") or {}).get("files"))
    manifest_matches = [
        item for item in runtime_files or []
        if isinstance(item, dict) and item.get("relative_path") == libgint_relative
    ]
    if len(manifest_matches) != 1 or (
        manifest_matches[0].get("sha256") != _sha256_bytes(libgint_bytes)
        or manifest_matches[0].get("bytes") != libgint_stat.st_size
    ):
        raise ValueError("snapshot manifest does not bind the loaded libgint")

    ledger = provider.get("explicit_transfers")
    if not isinstance(ledger, dict):
        raise ValueError("qualification transfer ledger is missing")
    if (
        ledger.get("complete_for_vhfopt_internal_setup") is not True
        or ledger.get("complete_for_declared_payloads") is not True
        or ledger.get("unresolved_operations") != []
    ):
        raise ValueError("qualification transfer ledger is incomplete")
    timed_ledger = result.get("timed_transfer_counter")
    if not isinstance(timed_ledger, dict):
        raise ValueError("qualification timed transfer ledger is missing")
    timed_by_kind = timed_ledger.get("by_kind") or {}
    timed_by_operation = timed_ledger.get("by_operation") or {}
    if (
        timed_ledger.get("total_bytes")
        != sum(item.get("bytes", -1) for item in timed_by_kind.values())
        or timed_ledger.get("total_transfers")
        != sum(item.get("count", -1) for item in timed_by_kind.values())
        or any(
            item.get("bytes") != sum(
                operation.get("bytes", -1)
                for operation in (timed_by_operation.get(kind) or {}).values()
            )
            or item.get("count") != sum(
                operation.get("count", -1)
                for operation in (timed_by_operation.get(kind) or {}).values()
            )
            for kind, item in timed_by_kind.items()
        )
    ):
        raise ValueError("qualification timed transfer ledger is not fully attributed")
    timing = result.get("timing") or {}
    if not isinstance(timing.get("direct_cd_end_to_end_seconds"), (int, float)):
        raise ValueError("qualification timing evidence is missing")

    helper_evidence = result.get("slurm_controller_helper") or {}
    if (
        helper_evidence.get("schema")
        != SCONTROL_QUALIFICATION_EVIDENCE_SCHEMA
        or helper_evidence.get("helper_schema") != SCONTROL_HELPER_SCHEMA
        or helper_evidence.get("loader_schema")
        != SCONTROL_LOADER_EVIDENCE_SCHEMA
        or helper_evidence.get("stable_during_run") is not True
        or helper_evidence.get("errors") != []
    ):
        raise ValueError("qualification controlled scontrol evidence is invalid")
    try:
        helper_start = scontrol_helper_portable_binding(
            helper_evidence.get("start")
        )
        helper_end = scontrol_helper_portable_binding(
            helper_evidence.get("end")
        )
        helper_recorded = helper_evidence.get("portable_binding")
        loader_start = scontrol_loader_portable_binding(
            (helper_evidence.get("start_query") or {}).get("loader_evidence")
        )
        loader_end = scontrol_loader_portable_binding(
            (helper_evidence.get("end_query") or {}).get("loader_evidence")
        )
        loader_recorded = helper_evidence.get("portable_loader_binding")
        helper_at_issuance = scontrol_helper_portable_binding(
            observe_scontrol_helper_identity(require_library_search_path=False)
        )
    except Exception as exc:
        raise ValueError(
            f"qualification controlled scontrol binding is invalid: {exc}"
        ) from exc
    if not (
        helper_start == helper_end == helper_recorded == helper_at_issuance
    ):
        raise ValueError(
            "controlled scontrol helper changed between qualification and receipt"
        )
    if not (
        loader_start == loader_end == loader_recorded
        and loader_recorded.get("query_helper") == helper_recorded
    ):
        raise ValueError(
            "actual scontrol library mapping changed during qualification"
        )

    return {
        "schema": RUNTIME_GATE_PAYLOAD_SCHEMA,
        "status": "passed",
        "evidence_state": "local_measured",
        "site": "mtu",
        "case": "water2-tz",
        "source": {
            "root": str(source_root),
            "tree_sha256": digest,
            "base_revision": source.get("base_revision"),
            "manifest": _binding(
                manifest_file, manifest_bytes, manifest_stat
            ),
        },
        "runtime": {
            "libgint": _binding(
                libgint_file, libgint_bytes, libgint_stat
            ),
            "basis_prod_cache_abi": bpcache_abi,
            "selected_pair_abi": selected_abi,
            "selected_c_abi": selected_c_abi,
        },
        "qualification": {
            "result": _binding(result_file, result_bytes, result_stat),
            "topology": _binding(
                topology_file, topology_bytes, topology_stat
            ),
            "slurm_output": _binding(
                output_file, output_bytes, output_stat
            ),
            "slurm": terminal,
            "topology_fingerprint": {
                "host": topology.get("host"),
                "mode": topology.get("mode"),
                "expected_gpu_numa_node": topology.get(
                    "expected_gpu_numa_node"
                ),
                "requested_physical_cores": topology.get(
                    "requested_physical_cores"
                ),
                "requested_threads": topology.get("requested_threads"),
                "gpu_numa_node": topology.get("gpu_numa_node"),
                "gpu_pci_bus_ids": topology.get("gpu_pci_bus_ids"),
                "gpu_node_cpulist": topology.get("gpu_node_cpulist"),
                "allowed_cpulist_before": topology.get(
                    "allowed_cpulist_before"
                ),
                "node_allowed_intersection": topology.get(
                    "node_allowed_intersection"
                ),
                "selected_cpulist": topology.get("selected_cpulist"),
                "observed_cpulist_after_bind": topology.get(
                    "observed_cpulist_after_bind"
                ),
                "mems_allowed_list": topology.get("mems_allowed_list"),
                "benchmark_track": topology.get("benchmark_track"),
            },
            "slurm_controller_helper": helper_recorded,
            "slurm_controller_loader": loader_recorded,
        },
        "transfer_audit": {
            "ledger": ledger,
            "ledger_sha256": canonical_json_sha256(ledger),
            "timed_ledger": timed_ledger,
            "timed_ledger_sha256": canonical_json_sha256(timed_ledger),
            "unresolved_count": 0,
        },
        "truths": {
            "cuda_parity_passed": True,
            "complete_transfer_audit_passed": True,
            "timing_gate_passed": True,
            "qualification_passed": True,
            "slurm_terminal_success": True,
        },
    }


def write_once_receipt(
    receipt_path: str | os.PathLike[str], payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Create an exclusive receipt directory for later anchoring in snapshot B."""

    path = Path(os.path.abspath(os.path.expanduser(os.fspath(receipt_path))))
    sidecar = Path(str(path) + RUNTIME_GATE_SIDECAR_SUFFIX)
    source_root = Path(str(payload["source"]["root"])).resolve(strict=True)
    canonical_path = path.resolve(strict=False)
    if canonical_path != path:
        raise ValueError("runtime gate receipt path must not traverse symlinks")
    if canonical_path.is_relative_to(source_root):
        raise ValueError("runtime gate receipt must be outside the source snapshot")
    task_root = source_root.parent.parent.parent
    approved_receipt_root = task_root / "results"
    if not canonical_path.is_relative_to(approved_receipt_root):
        raise ValueError(
            "runtime gate receipt must be under the qualification task results root"
        )
    if not path.parent.parent.is_dir():
        raise ValueError("runtime gate receipt directory parent does not exist")
    if path.parent.exists() or path.parent.is_symlink():
        raise FileExistsError(
            "refusing to reuse a runtime gate receipt directory"
        )
    if path.parent.parent.resolve(strict=True) != path.parent.parent:
        raise ValueError("runtime gate receipt parent path traverses a symlink")
    envelope = {
        "schema": RUNTIME_GATE_RECEIPT_SCHEMA,
        "created_utc": _utc_now(),
        "payload_sha256": canonical_json_sha256(payload),
        "payload": dict(payload),
    }
    encoded = (
        json.dumps(
            envelope,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")
    receipt_digest = _sha256_bytes(encoded)
    sidecar_bytes = f"{receipt_digest}  {path.name}\n".encode("ascii")
    created_receipt = False
    created_directory = False

    def write_all(descriptor: int, value: bytes) -> None:
        view = memoryview(value)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("write-once receipt write made no progress")
            view = view[written:]

    try:
        os.mkdir(path.parent, 0o700)
        created_directory = True
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        created_receipt = True
        try:
            write_all(descriptor, encoded)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
        descriptor = os.open(
            sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
        try:
            write_all(descriptor, sidecar_bytes)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
        os.chmod(path.parent, 0o555)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if created_directory and path.parent.exists() and not path.parent.is_symlink():
            path.parent.chmod(0o700)
        if sidecar.exists() and not sidecar.is_symlink():
            sidecar.chmod(0o600)
            sidecar.unlink()
        if created_receipt and path.exists() and not path.is_symlink():
            path.chmod(0o600)
            path.unlink()
        if created_directory and path.parent.exists() and not any(
            path.parent.iterdir()
        ):
            path.parent.rmdir()
        raise
    return {
        "receipt": str(path),
        "receipt_sha256": receipt_digest,
        "sidecar": str(sidecar),
        "payload_sha256": envelope["payload_sha256"],
    }


def build_release_pin(
    receipt_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build the deterministic A-to-B pin for one sealed receipt."""

    receipt_file, receipt_bytes, receipt_stat = _regular_file(
        receipt_path, name="runtime gate receipt"
    )
    if receipt_stat.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("runtime gate receipt must be read-only")
    sidecar_file, sidecar_bytes, sidecar_stat = _regular_file(
        str(receipt_file) + RUNTIME_GATE_SIDECAR_SUFFIX,
        name="runtime gate receipt SHA-256 sidecar",
    )
    if sidecar_stat.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("runtime gate receipt sidecar must be read-only")
    receipt_sha256 = _sha256_bytes(receipt_bytes)
    expected_sidecar = f"{receipt_sha256}  {receipt_file.name}\n".encode("ascii")
    if sidecar_bytes != expected_sidecar:
        raise ValueError("runtime gate receipt sidecar is invalid")
    if sidecar_file != Path(str(receipt_file) + RUNTIME_GATE_SIDECAR_SUFFIX):
        raise ValueError("runtime gate receipt sidecar path is inconsistent")
    directory_stat = receipt_file.parent.lstat()
    if directory_stat.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("runtime gate receipt directory must be sealed")
    if {path.name for path in receipt_file.parent.iterdir()} != {
        receipt_file.name,
        sidecar_file.name,
    }:
        raise ValueError("runtime gate receipt directory has unexpected files")

    envelope = _json_object(receipt_bytes, name="runtime gate receipt")
    if envelope.get("schema") != RUNTIME_GATE_RECEIPT_SCHEMA:
        raise ValueError("runtime gate receipt schema mismatch")
    payload = envelope.get("payload")
    payload_sha256 = envelope.get("payload_sha256")
    if not isinstance(payload, dict):
        raise ValueError("runtime gate receipt payload must be an object")
    if canonical_json_sha256(payload) != payload_sha256:
        raise ValueError("runtime gate receipt payload digest mismatch")
    return expected_runtime_release_pin(
        payload,
        receipt_sha256=receipt_sha256,
        payload_sha256=payload_sha256,
    )


def write_once_release_pin(
    output_path: str | os.PathLike[str], pin: Mapping[str, Any]
) -> dict[str, Any]:
    """Write a staging pin which must later be embedded in snapshot B."""

    if pin.get("schema") != RELEASE_PIN_SCHEMA:
        raise ValueError("release pin schema mismatch")
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(output_path))))
    if path.name != RELEASE_PIN_RELATIVE_PATH.name:
        raise ValueError(
            f"release pin must be named {RELEASE_PIN_RELATIVE_PATH.name}"
        )
    canonical_path = path.resolve(strict=False)
    if canonical_path != path:
        raise ValueError("release pin output path must not traverse symlinks")
    if not path.parent.is_dir():
        raise ValueError("release pin output directory does not exist")
    qualification_root = Path(
        str((pin.get("qualification_source") or {}).get("root", ""))
    ).resolve(strict=True)
    approved_results = qualification_root.parent.parent.parent / "results"
    if not path.is_relative_to(approved_results):
        raise ValueError("release pin must be below the task results root")
    encoded = (
        json.dumps(
            dict(pin), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        ) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("release pin write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
    except Exception:
        if path.exists() and not path.is_symlink():
            path.chmod(0o600)
            path.unlink()
        raise
    return {
        "release_pin": str(path),
        "release_pin_sha256": _sha256_bytes(encoded),
        "embed_at": RELEASE_PIN_RELATIVE_PATH.as_posix(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--slurm-output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--release-pin", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if os.path.lexists(args.release_pin):
        raise FileExistsError(f"refusing to overwrite release pin {args.release_pin}")
    payload = build_receipt_payload(args.result, args.slurm_output)
    created = write_once_receipt(args.receipt, payload)
    pin = build_release_pin(args.receipt)
    pin_created = write_once_release_pin(args.release_pin, pin)
    print(json.dumps({**created, **pin_created}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"GINT receipt issuance failed: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
