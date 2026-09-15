# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact, JSON-safe setup-transfer provenance for selected GINT columns.

The audit object is deliberately independent of CuPy.  ``basis_seg_contraction``
and ``_VHFOpt`` receive it through a backwards-compatible optional argument and
report only application-visible array payloads.  Device allocations, device to
device transformations, and CUDA launch argument marshalling are classified
separately so that resident capacity is never added to PCIe traffic.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
from typing import Any, Mapping, Optional


RUNTIME_GATE_RECEIPT_SCHEMA = (
    "gpu4pyscf.gint-selected-runtime-gate-receipt.v2"
)
RUNTIME_GATE_PAYLOAD_SCHEMA = (
    "gpu4pyscf.gint-selected-runtime-gate-payload.v2"
)
RUNTIME_GATE_SIDECAR_SUFFIX = ".sha256"
RELEASE_PIN_SCHEMA = "gpu4pyscf.gint-selected-runtime-release-pin.v2"
RELEASE_PIN_RELATIVE_PATH = Path("gpu4pyscf/cc/gint_release_pin.json")
PUBLICATION_ATTESTATION_SCHEMA = "gpu4pyscf.snapshot-publication-attestation.v1"
PUBLICATION_ATTESTATION_SUFFIX = ".publication-attestation.json"
PUBLICATION_TRUST_BOUNDARY = "private-anchor-owner-and-root-are-trusted-cooperators"
PUBLICATION_POLICY = {
    "schema": "gpu4pyscf.snapshot-publication-policy.v1",
    "trust_boundary": PUBLICATION_TRUST_BOUNDARY,
    "attestation_schema": PUBLICATION_ATTESTATION_SCHEMA,
    "attestation_location": (
        "snapshots-root sibling named "
        "<snapshot-name>.publication-attestation.json"
    ),
}
PUBLICATION_PROTOCOLS = frozenset({
    "atomic-exclusive-rename",
    "atomic-exclusive-rename-commit-confirmed",
    "reserved-empty-directory-rename",
    "reserved-empty-directory-rename-commit-confirmed",
})
PUBLICATION_ATTESTATION_KEYS = frozenset({
    "schema", "status", "source_tree_sha256", "snapshot_name",
    "published_directory_identity", "publish_protocol", "trust_boundary",
    "publication_policy_sha256", "published_manifest_sha256",
    "prepublication_manifest_sha256", "attestation_sha256",
})
DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
RELEASE_RUNTIME_OBSERVATION_SCHEMA = (
    "gpu4pyscf.gint-selected-release-runtime-observation.v2"
)
RELEASE_RUNTIME_CONTRACT_SCHEMA = (
    "gpu4pyscf.gint-selected-release-runtime-contract.v2"
)
GINT_GATE_RESULT_SCHEMA = "gpu4pyscf.water2.gint-direct-cd-gate.v3"
SCONTROL_HELPER_SCHEMA = "gpu4pyscf.controlled-scontrol-helper.v2"
SCONTROL_LOADER_EVIDENCE_SCHEMA = (
    "gpu4pyscf.controlled-scontrol-loader-evidence.v1"
)
SCONTROL_ELF_LOADER_POLICY_SCHEMA = (
    "gpu4pyscf.controlled-scontrol-elf-loader-policy.v1"
)
SCONTROL_HELPER_TRUST_BOUNDARY = (
    "task-root-owner-and-root-are-trusted-cooperators"
)
THREAD_AFFINITY_OBSERVATION_SCHEMA = (
    "gpu4pyscf.linux-thread-affinity-observation.v1"
)
SOURCE_SNAPSHOT_SCHEMA = "gpu4pyscf.source-snapshot.v3"
SOURCE_NORMALIZATION_SCHEMA = "gpu4pyscf.source-symlink-normalization.v1"
SOURCE_NORMALIZATION_POLICY = "validated-relative-leaf-links-materialized"
RUNTIME_GATE_EXECUTION_MODES = frozenset({
    "release-gate",
    "consumer-benchmark",
    "consumer-counterpoise",
})
_EXPECTED_RELEASE_PARTITION = "mrigpu"
_EXPECTED_RELEASE_NODE = "compute-1-6"
_EXPECTED_RELEASE_AFFINITY = "24-31"
_EXPECTED_RELEASE_NUMA_NODE = 3
_PCI_SYSFS_DEVICES = Path("/sys/bus/pci/devices")
_NODE_SYSFS_DEVICES = Path("/sys/devices/system/node")
_SCONTROL_DISTRIBUTION_RELATIVE_PATH = Path(
    "control-tools/slurm-23.02.4-local-rpath-v1"
)
_SCONTROL_RELATIVE_PATH = (
    _SCONTROL_DISTRIBUTION_RELATIVE_PATH / "bin" / "scontrol"
)
_SCONTROL_LIBRARY_RELATIVE_PATH = Path(
    _SCONTROL_DISTRIBUTION_RELATIVE_PATH / "lib" / "libslurmfull.so"
)
_SCONTROL_SHA256 = (
    "12259ab80333f9c1b7033747bde5b5b3bb9d73e5b6453f3e4049a4597cdddcdf"
)
_SCONTROL_BYTES = 1_128_736
_SCONTROL_LIBRARY_SHA256 = (
    "722e5978486e5231ed1e1cae497f0a860858695ee69e4caef1c9525075b1dee2"
)
_SCONTROL_LIBRARY_BYTES = 12_187_568
_SCONTROL_ELF_LOADER_POLICY = {
    "schema": SCONTROL_ELF_LOADER_POLICY_SCHEMA,
    "dynamic_tag": "DT_RPATH",
    "search_path": "$ORIGIN/../lib",
    "needed_soname": "libslurmfull.so",
    "mapping_evidence": "glibc-LD_DEBUG=libs-calling-init",
}
_SCONTROL_DIRECTORY_RELATIVE_PATHS = {
    "control_tools": Path("control-tools"),
    "distribution": _SCONTROL_DISTRIBUTION_RELATIVE_PATH,
    "bin": _SCONTROL_DISTRIBUTION_RELATIVE_PATH / "bin",
    "lib": _SCONTROL_DISTRIBUTION_RELATIVE_PATH / "lib",
}
_PROC_SELF_TASK = Path("/proc/self/task")
_RELEASE_ENVIRONMENT_NAMES = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_PARTITION",
    "SLURM_CPUS_PER_TASK",
    "CUDA_VISIBLE_DEVICES",
    "CCSD_PERFORMANCE_ELIGIBLE",
    "CCSD_BENCHMARK_TRACK",
    "CCSD_TOPOLOGY_RECORD",
    "CCSD_TOPOLOGY_SHA256",
    "CCSD_BOUND_CPUS",
    "GPU4PYSCF_NUMA",
    "CCSD_SCONTROL_PATH",
)
_SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".ini",
    ".json", ".md", ".py", ".pyx", ".pxd", ".sbatch", ".sh",
    ".toml", ".txt", ".yaml", ".yml",
}
_SOURCE_EXCLUDED_PARTS = {
    ".git", ".pytest_cache", "__pycache__", "build", "dist", "results",
}


REQUIRED_SETUP_TRANSFERS = (
    "basis_c2s_blocks",
    "basis_high_l_decontraction_blocks",
    "basis_block_diag_rows",
    "basis_block_diag_cols",
    "basis_block_diag_offsets",
    "vhfopt_ao_sort_index",
    "vhfopt_log_q",
    "gint_basis_cache_ao_loc",
    "gint_basis_cache_bas_coords",
    "gint_basis_cache_bas_atm",
    "gint_basis_cache_aexyz",
    "gint_basis_cache_bas_pair2shls",
)

REQUIRED_DEVICE_PAYLOADS = (
    "basis_block_diag_concatenated_blocks",
    "basis_block_diag_coeff",
    "vhfopt_sorted_coeff",
)


def _label(value: Any, *, name: str) -> str:
    value = str(value)
    if not value or "/" in value:
        raise ValueError(f"{name} must be a non-empty path segment")
    return value


def _text(value: Any, *, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


class GINTSetupTransferAudit:
    """Accumulate exact setup transfers and device-generated payloads.

    A required operation may have zero bytes and zero calls.  This is needed
    for the high-angular-momentum decontraction path, which is absent for many
    basis sets but must still be explicitly classified.  Presence and source
    provenance, rather than a guessed positive size, establish coverage.
    """

    schema = "gpu4pyscf.gint-setup-transfer-audit.v1"

    def __init__(self, *, transfer_counter: Any = None) -> None:
        self.transfer_counter = transfer_counter
        self._directions = {
            "h2d": OrderedDict(),
            "d2h": OrderedDict(),
        }
        self._device_payloads: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._excluded: OrderedDict[str, dict[str, str]] = OrderedDict()

    def record_transfer(
        self,
        direction: str,
        operation: str,
        nbytes: int,
        *,
        count: int = 1,
        logical_payload: str,
        provenance: str,
    ) -> None:
        direction = str(direction)
        if direction not in self._directions:
            raise ValueError("transfer direction must be h2d or d2h")
        operation = _label(operation, name="transfer operation")
        logical_payload = _text(logical_payload, name="logical payload")
        provenance = _text(provenance, name="transfer provenance")
        nbytes, count = int(nbytes), int(count)
        if nbytes < 0 or count < 0:
            raise ValueError("transfer bytes and count must be non-negative")
        operations = self._directions[direction]
        current = operations.get(operation)
        if current is None:
            current = {
                "bytes": 0,
                "count": 0,
                "logical_payload": logical_payload,
                "provenance": provenance,
            }
            operations[operation] = current
        elif (
            current["logical_payload"] != logical_payload
            or current["provenance"] != provenance
        ):
            raise ValueError(
                f"transfer operation {operation!r} changed payload provenance"
            )
        current["bytes"] += nbytes
        current["count"] += count
        if self.transfer_counter is not None:
            self.transfer_counter.record(
                direction, nbytes, count=count, operation=operation
            )

    def record_device_payload(
        self,
        operation: str,
        resident_bytes: int,
        *,
        logical_payload: str,
        provenance: str,
        origin: str = "device-generated",
        lifetime: str = "provider-lifetime",
    ) -> None:
        operation = _label(operation, name="device payload operation")
        logical_payload = _text(logical_payload, name="logical payload")
        provenance = _text(provenance, name="device payload provenance")
        origin = _text(origin, name="device payload origin")
        lifetime = _text(lifetime, name="device payload lifetime")
        if origin not in {"device-generated", "host-uploaded", "device-alias"}:
            raise ValueError("unsupported device payload origin")
        if lifetime not in {"provider-lifetime", "transient-setup"}:
            raise ValueError("unsupported device payload lifetime")
        resident_bytes = int(resident_bytes)
        if resident_bytes < 0:
            raise ValueError("resident payload bytes must be non-negative")
        if operation in self._device_payloads:
            raise ValueError(f"device payload {operation!r} is duplicated")
        self._device_payloads[operation] = {
            "resident_bytes": resident_bytes,
            "actual_transfer_bytes": 0 if origin == "device-generated" else None,
            "classification": "logical-device-resident-payload",
            "origin": origin,
            "lifetime": lifetime,
            "logical_payload": logical_payload,
            "provenance": provenance,
        }

    def record_exclusion(
        self,
        operation: str,
        *,
        reason: str,
        classification: str,
    ) -> None:
        operation = _label(operation, name="excluded operation")
        reason = _text(reason, name="exclusion reason")
        classification = _text(classification, name="exclusion classification")
        if operation in self._excluded:
            raise ValueError(f"excluded operation {operation!r} is duplicated")
        self._excluded[operation] = {
            "operation": operation,
            "classification": classification,
            "reason": reason,
        }

    def to_dict(self) -> dict[str, Any]:
        directions: dict[str, Any] = {}
        for direction, raw_operations in self._directions.items():
            operations = deepcopy(dict(raw_operations))
            directions[direction] = {
                "bytes": sum(item["bytes"] for item in operations.values()),
                "count": sum(item["count"] for item in operations.values()),
                "operations": operations,
            }
        observed = set(self._directions["h2d"])
        missing_operations = sorted(set(REQUIRED_SETUP_TRANSFERS) - observed)
        missing_payloads = sorted(
            set(REQUIRED_DEVICE_PAYLOADS) - set(self._device_payloads)
        )
        unresolved = [
            {
                "operation": operation,
                "bytes": None,
                "reason": "required setup transfer was not reported",
            }
            for operation in missing_operations
        ] + [
            {
                "operation": operation,
                "bytes": None,
                "reason": "required device-resident payload was not classified",
            }
            for operation in missing_payloads
        ]
        complete = not unresolved
        return {
            "schema": self.schema,
            "directions": directions,
            "total_bytes": sum(item["bytes"] for item in directions.values()),
            "logical_device_payloads": deepcopy(dict(self._device_payloads)),
            "coverage": {
                "required_transfer_operations": list(REQUIRED_SETUP_TRANSFERS),
                "required_device_payloads": list(REQUIRED_DEVICE_PAYLOADS),
                "missing_transfer_operations": missing_operations,
                "missing_device_payloads": missing_payloads,
            },
            "excluded_operations": list(deepcopy(self._excluded).values()),
            "unresolved_operations": unresolved,
            "complete_for_declared_payloads": complete,
        }

    @property
    def complete(self) -> bool:
        return bool(self.to_dict()["complete_for_declared_payloads"])


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value with the receipt's deterministic encoding."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _format_cpu_list(values: Any) -> str:
    ordered = sorted({int(value) for value in values})
    ranges: list[str] = []
    start = previous = None
    for value in ordered:
        if start is None:
            start = previous = value
        elif value == previous + 1:
            previous = value
        else:
            ranges.append(
                str(start) if start == previous else f"{start}-{previous}"
            )
            start = previous = value
    if start is not None:
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _parse_cpu_list(value: Any) -> tuple[int, ...]:
    if not isinstance(value, str) or not value.strip():
        return ()
    result: set[int] = set()
    for field in value.split(","):
        pieces = field.strip().split("-")
        if len(pieces) == 1:
            start = stop = int(pieces[0])
        elif len(pieces) == 2:
            start, stop = (int(piece) for piece in pieces)
        else:
            raise ValueError(f"invalid CPU list field {field!r}")
        if start < 0 or stop < start:
            raise ValueError(f"invalid CPU list range {field!r}")
        result.update(range(start, stop + 1))
    return tuple(sorted(result))


def _status_cpu_list(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key == "Cpus_allowed_list":
            cpus = _parse_cpu_list(value.strip())
            if not cpus:
                raise ValueError(f"{path} has an empty Cpus_allowed_list")
            return _format_cpu_list(cpus)
    raise ValueError(f"{path} has no Cpus_allowed_list")


def observe_process_thread_affinity(
    task_directory: Optional[os.PathLike[str] | str] = None,
) -> dict[str, Any]:
    """Read one stable, fail-closed ``/proc/self/task`` affinity snapshot.

    Linux affinity is a per-thread property.  OpenMP may pin the Python caller
    to one CPU while its workers cover the complete requested CPU set, so
    ``sched_getaffinity(0)`` is not a process-wide observation.  A changing
    task inventory or a vanished worker invalidates this sample rather than
    being silently treated as a smaller process affinity.
    """

    root = _PROC_SELF_TASK if task_directory is None else Path(task_directory)
    errors: list[str] = []
    threads: list[dict[str, Any]] = []
    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    try:
        before = tuple(sorted(
            (item.name for item in root.iterdir() if item.name.isdigit()),
            key=int,
        ))
        if not before:
            raise RuntimeError("no Linux threads were visible")
        for name in before:
            status_path = root / name / "status"
            threads.append({
                "tid": int(name),
                "affinity": _status_cpu_list(status_path),
            })
        after = tuple(sorted(
            (item.name for item in root.iterdir() if item.name.isdigit()),
            key=int,
        ))
        if after != before:
            raise RuntimeError(
                "Linux thread inventory changed while affinity was sampled"
            )
    except Exception as exc:
        errors.append(f"thread affinity snapshot failed: {exc}")

    expected = set(_parse_cpu_list(_EXPECTED_RELEASE_AFFINITY))
    observed_sets: list[set[int]] = []
    for item in threads:
        try:
            observed_sets.append(set(_parse_cpu_list(item["affinity"])))
        except Exception as exc:
            errors.append(
                f"thread {item.get('tid')} affinity is invalid: {exc}"
            )
    union = set().union(*observed_sets) if observed_sets else set()
    all_subset = bool(observed_sets) and all(
        cpus.issubset(expected) for cpus in observed_sets
    )
    covers_expected = expected.issubset(union)
    if not all_subset:
        errors.append("one or more thread affinities leave CPUs 24-31")
    if not covers_expected:
        errors.append("thread-affinity union does not cover CPUs 24-31")
    return {
        "schema": THREAD_AFFINITY_OBSERVATION_SCHEMA,
        "source": "stable-proc-self-task-status",
        "task_directory": str(root),
        "thread_count": len(threads),
        "thread_ids_before": [int(value) for value in before],
        "thread_ids_after": [int(value) for value in after],
        "threads": threads,
        "union_affinity": _format_cpu_list(union),
        "expected_affinity": _EXPECTED_RELEASE_AFFINITY,
        "all_threads_within_expected": all_subset,
        "union_covers_expected": covers_expected,
        "stable": bool(before) and before == after and not errors,
        "errors": errors,
    }


def _controlled_directory_binding(
    path: Path,
    *,
    label: str,
    expected_owner_uid: int,
    expected_mode: int = 0o555,
) -> dict[str, Any]:
    """Bind one sealed helper directory inside the trusted task root."""

    _reject_symlink_components(path, name=label)
    before = path.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{label} must be a directory")
    if path.resolve(strict=True) != path:
        raise ValueError(f"{label} path must be canonical")
    if stat.S_IMODE(before.st_mode) != expected_mode:
        raise ValueError(f"{label} mode must be {expected_mode:#o}")
    if before.st_uid != expected_owner_uid:
        raise ValueError(f"{label} owner differs from the task owner")
    descriptor = os.open(path, DIRECTORY_FLAGS | NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_nlink,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_gid,
            before.st_nlink,
        ):
            raise ValueError(f"{label} identity changed while opened")
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_nlink,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
    ):
        raise ValueError(f"{label} identity changed while observed")
    return {
        "path": str(path),
        "mode": stat.S_IMODE(before.st_mode),
        "owner_uid": int(before.st_uid),
        "owner_gid": int(before.st_gid),
        "hard_links": int(before.st_nlink),
        "directory_identity": {
            "device": int(before.st_dev),
            "inode": int(before.st_ino),
        },
    }


def _controlled_file_binding(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_bytes: int,
    expected_owner_uid: int,
    require_executable: bool,
    expected_mode: Optional[int] = None,
) -> dict[str, Any]:
    _reject_symlink_components(path, name=label)
    before = path.lstat()
    execute_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if path.resolve(strict=True) != path:
        raise ValueError(f"{label} path must be canonical")
    if before.st_mode & (write_bits | stat.S_ISUID | stat.S_ISGID):
        raise ValueError(f"{label} has unsafe permission bits")
    if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
        raise ValueError(f"{label} mode must be {expected_mode:#o}")
    if before.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")
    if before.st_uid != expected_owner_uid:
        raise ValueError(f"{label} owner differs from the task owner")
    if require_executable and (
        not before.st_mode & execute_bits or not os.access(path, os.X_OK)
    ):
        raise ValueError(f"{label} must be executable")
    descriptor = os.open(path, os.O_RDONLY | NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} identity changed while opened")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_nlink,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
    ):
        raise ValueError(f"{label} identity changed while read")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256 or size != expected_bytes:
        raise ValueError(f"{label} content identity mismatch")
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "bytes": size,
        "mode": stat.S_IMODE(before.st_mode),
        "owner_uid": int(before.st_uid),
        "owner_gid": int(before.st_gid),
        "hard_links": int(before.st_nlink),
        "directory_identity": {
            "device": int(before.st_dev),
            "inode": int(before.st_ino),
        },
    }


def observe_scontrol_helper_identity(
    *, require_library_search_path: bool = True,
) -> dict[str, Any]:
    """Validate the explicit MTU ``scontrol`` helper and its private library."""

    errors: list[str] = []
    task_root: Optional[Path] = None
    directories: Optional[dict[str, Any]] = None
    executable: Optional[dict[str, Any]] = None
    library: Optional[dict[str, Any]] = None
    library_dir: Optional[Path] = None
    configured_path = os.getenv("CCSD_SCONTROL_PATH")
    try:
        configured_task_root = os.getenv("CCSD_TASK_ROOT")
        if not configured_task_root:
            raise ValueError("CCSD_TASK_ROOT is missing")
        task_root = Path(configured_task_root)
        if not task_root.is_absolute() or task_root.resolve(strict=True) != task_root:
            raise ValueError("CCSD_TASK_ROOT must be an absolute canonical path")
        _reject_symlink_components(task_root, name="controlled helper task root")
        task_info = task_root.lstat()
        if not stat.S_ISDIR(task_info.st_mode):
            raise ValueError("CCSD_TASK_ROOT must be a directory")
        if task_info.st_uid != os.geteuid():
            raise ValueError("CCSD_TASK_ROOT must be owned by the current user")
        directories = {
            role: _controlled_directory_binding(
                task_root / relative,
                label=f"controlled scontrol {role.replace('_', '-')} directory",
                expected_owner_uid=task_info.st_uid,
            )
            for role, relative in _SCONTROL_DIRECTORY_RELATIVE_PATHS.items()
        }
        expected_path = task_root / _SCONTROL_RELATIVE_PATH
        if not configured_path:
            raise ValueError("CCSD_SCONTROL_PATH is missing")
        path = Path(configured_path)
        if not path.is_absolute() or path != expected_path:
            raise ValueError(
                "CCSD_SCONTROL_PATH must name the fixed task control helper"
            )
        library_path = task_root / _SCONTROL_LIBRARY_RELATIVE_PATH
        library_dir = library_path.parent
        executable = _controlled_file_binding(
            path,
            label="controlled scontrol executable",
            expected_sha256=_SCONTROL_SHA256,
            expected_bytes=_SCONTROL_BYTES,
            expected_owner_uid=task_info.st_uid,
            require_executable=True,
            expected_mode=0o555,
        )
        library = _controlled_file_binding(
            library_path,
            label="controlled scontrol library",
            expected_sha256=_SCONTROL_LIBRARY_SHA256,
            expected_bytes=_SCONTROL_LIBRARY_BYTES,
            expected_owner_uid=task_info.st_uid,
            require_executable=False,
            expected_mode=0o444,
        )
        if require_library_search_path:
            loader_paths = os.getenv("LD_LIBRARY_PATH", "").split(":")
            if not loader_paths or loader_paths[0] != str(library_dir):
                raise ValueError(
                    "controlled scontrol library directory must be first in "
                    "LD_LIBRARY_PATH"
                )
    except Exception as exc:
        errors.append(str(exc))
    return {
        "schema": SCONTROL_HELPER_SCHEMA,
        "status": "validated" if not errors else "invalid",
        "trust_boundary": SCONTROL_HELPER_TRUST_BOUNDARY,
        "task_root": None if task_root is None else str(task_root),
        "configured_path": configured_path,
        "library_search_path_prefix": (
            None if library_dir is None else str(library_dir)
        ),
        "elf_loader_policy": deepcopy(_SCONTROL_ELF_LOADER_POLICY),
        "directories": directories,
        "executable": executable,
        "library": library,
        "errors": errors,
    }


def _portable_filesystem_inode(item: Mapping[str, Any], *, label: str) -> int:
    identity = item.get("directory_identity")
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"device", "inode"}
        or type(identity.get("device")) is not int
        or identity["device"] < 0
        or type(identity.get("inode")) is not int
        or identity["inode"] <= 0
    ):
        raise ValueError(f"controlled {label} filesystem identity is invalid")
    return identity["inode"]


def _portable_controlled_file(item: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(item, Mapping) or set(item) != {
        "path", "sha256", "bytes", "mode", "owner_uid", "owner_gid",
        "hard_links", "directory_identity",
    }:
        raise ValueError(f"controlled {label} identity shape is invalid")
    inode = _portable_filesystem_inode(item, label=label)
    if not _is_sha256(item.get("sha256")):
        raise ValueError(f"controlled {label} SHA-256 is invalid")
    for name in ("bytes", "mode", "owner_uid", "owner_gid", "hard_links"):
        if type(item.get(name)) is not int or item[name] < 0:
            raise ValueError(f"controlled {label} {name} is invalid")
    return {
        key: deepcopy(item[key])
        for key in (
            "path", "sha256", "bytes", "mode", "owner_uid",
            "owner_gid", "hard_links",
        )
    } | {"inode": inode}


def _portable_controlled_directory(item: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(item, Mapping) or set(item) != {
        "path", "mode", "owner_uid", "owner_gid", "hard_links",
        "directory_identity",
    }:
        raise ValueError(f"controlled {label} directory shape is invalid")
    inode = _portable_filesystem_inode(item, label=f"{label} directory")
    for name in ("mode", "owner_uid", "owner_gid", "hard_links"):
        if type(item.get(name)) is not int or item[name] < 0:
            raise ValueError(f"controlled {label} directory {name} is invalid")
    return {
        key: deepcopy(item[key])
        for key in ("path", "mode", "owner_uid", "owner_gid", "hard_links")
    } | {"inode": inode}


def scontrol_helper_portable_binding(value: Any) -> dict[str, Any]:
    """Return the helper identity stable across NFS client mount namespaces."""

    if not isinstance(value, Mapping):
        raise ValueError("controlled scontrol helper evidence is unavailable")
    expected_keys = {
        "schema", "status", "trust_boundary", "task_root",
        "configured_path", "library_search_path_prefix", "elf_loader_policy",
        "directories", "executable", "library", "errors",
    }
    if set(value) != expected_keys or value.get("schema") != SCONTROL_HELPER_SCHEMA:
        raise ValueError("controlled scontrol helper evidence shape is invalid")
    if value.get("status") != "validated" or value.get("errors") != []:
        raise ValueError("controlled scontrol helper evidence is invalid")

    raw_directories = value.get("directories")
    if not isinstance(raw_directories, Mapping) or set(raw_directories) != set(
        _SCONTROL_DIRECTORY_RELATIVE_PATHS
    ):
        raise ValueError("controlled scontrol directory chain is invalid")

    portable = {
        "schema": SCONTROL_HELPER_SCHEMA,
        "trust_boundary": value.get("trust_boundary"),
        "task_root": value.get("task_root"),
        "configured_path": value.get("configured_path"),
        "library_search_path_prefix": value.get("library_search_path_prefix"),
        "elf_loader_policy": deepcopy(value.get("elf_loader_policy")),
        "directories": {
            role: _portable_controlled_directory(
                raw_directories.get(role), label=role.replace("_", "-")
            )
            for role in _SCONTROL_DIRECTORY_RELATIVE_PATHS
        },
        "executable": _portable_controlled_file(
            value.get("executable"), label="executable"
        ),
        "library": _portable_controlled_file(
            value.get("library"), label="library"
        ),
    }
    _validate_scontrol_helper_portable_binding(portable)
    return portable


def _validate_scontrol_helper_portable_binding(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "trust_boundary", "task_root", "configured_path",
        "library_search_path_prefix", "elf_loader_policy", "directories",
        "executable", "library",
    }:
        raise ValueError("portable controlled scontrol helper shape is invalid")
    if value.get("schema") != SCONTROL_HELPER_SCHEMA:
        raise ValueError("portable controlled scontrol helper schema is invalid")
    if value.get("trust_boundary") != SCONTROL_HELPER_TRUST_BOUNDARY:
        raise ValueError("portable controlled scontrol trust boundary is invalid")
    if value.get("elf_loader_policy") != _SCONTROL_ELF_LOADER_POLICY:
        raise ValueError("portable controlled scontrol ELF loader policy is invalid")
    task_root = value.get("task_root")
    if not isinstance(task_root, str) or not Path(task_root).is_absolute():
        raise ValueError("portable controlled scontrol task root is invalid")
    expected_executable = Path(task_root) / _SCONTROL_RELATIVE_PATH
    expected_library = Path(task_root) / _SCONTROL_LIBRARY_RELATIVE_PATH
    if value.get("configured_path") != str(expected_executable):
        raise ValueError("portable controlled scontrol path is invalid")
    if value.get("library_search_path_prefix") != str(expected_library.parent):
        raise ValueError("portable controlled scontrol library path is invalid")
    directories = value.get("directories")
    if not isinstance(directories, Mapping) or set(directories) != set(
        _SCONTROL_DIRECTORY_RELATIVE_PATHS
    ):
        raise ValueError("portable controlled scontrol directory chain is invalid")
    directory_owners: set[tuple[int, int]] = set()
    for role, relative in _SCONTROL_DIRECTORY_RELATIVE_PATHS.items():
        item = directories.get(role)
        if not isinstance(item, Mapping) or set(item) != {
            "path", "mode", "owner_uid", "owner_gid", "hard_links", "inode",
        }:
            raise ValueError(
                f"portable controlled scontrol {role} directory shape is invalid"
            )
        if item.get("path") != str(Path(task_root) / relative):
            raise ValueError(
                f"portable controlled scontrol {role} directory path is invalid"
            )
        if (
            item.get("mode") != 0o555
            or type(item.get("owner_uid")) is not int
            or item["owner_uid"] < 0
            or type(item.get("owner_gid")) is not int
            or item["owner_gid"] < 0
            or type(item.get("hard_links")) is not int
            or item["hard_links"] < 1
            or type(item.get("inode")) is not int
            or item["inode"] <= 0
        ):
            raise ValueError(
                f"portable controlled scontrol {role} directory metadata is invalid"
            )
        directory_owners.add((item["owner_uid"], item["owner_gid"]))
    if len(directory_owners) != 1:
        raise ValueError("portable controlled scontrol directory owners differ")
    for label, item, expected_path, expected_sha, expected_size, expected_mode in (
        (
            "executable", value.get("executable"), expected_executable,
            _SCONTROL_SHA256, _SCONTROL_BYTES, 0o555,
        ),
        (
            "library", value.get("library"), expected_library,
            _SCONTROL_LIBRARY_SHA256, _SCONTROL_LIBRARY_BYTES, 0o444,
        ),
    ):
        if not isinstance(item, Mapping) or set(item) != {
            "path", "sha256", "bytes", "mode", "owner_uid", "owner_gid",
            "hard_links", "inode",
        }:
            raise ValueError(f"portable controlled {label} shape is invalid")
        if item.get("path") != str(expected_path):
            raise ValueError(f"portable controlled {label} path is invalid")
        if item.get("sha256") != expected_sha or item.get("bytes") != expected_size:
            raise ValueError(f"portable controlled {label} content is invalid")
        if (
            item.get("mode") != expected_mode
            or type(item.get("owner_uid")) is not int
            or item["owner_uid"] < 0
            or type(item.get("owner_gid")) is not int
            or item["owner_gid"] < 0
            or item.get("hard_links") != 1
            or type(item.get("inode")) is not int
            or item["inode"] <= 0
        ):
            raise ValueError(f"portable controlled {label} metadata is invalid")
    if not value["executable"]["mode"] & (
        stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    ):
        raise ValueError("portable controlled executable is not executable")
    if value["executable"]["owner_uid"] != value["library"]["owner_uid"]:
        raise ValueError("portable controlled helper owners differ")
    owner = next(iter(directory_owners))
    if (
        value["executable"]["owner_uid"],
        value["executable"]["owner_gid"],
    ) != owner or (
        value["library"]["owner_uid"],
        value["library"]["owner_gid"],
    ) != owner:
        raise ValueError("portable controlled helper directory/file owners differ")


def validate_scontrol_helper_portable_binding(value: Any) -> None:
    """Validate a receipt/pin helper identity without client-local ``st_dev``."""

    _validate_scontrol_helper_portable_binding(value)


def _scontrol_loader_evidence(
    debug_stderr: str,
    *,
    child_pid: int,
    helper_before: Mapping[str, Any],
    helper_after: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove which ``libslurmfull.so`` glibc mapped for one query."""

    if dict(helper_before) != dict(helper_after):
        raise RuntimeError("controlled scontrol helper changed while queried")
    helper = scontrol_helper_portable_binding(helper_before)
    policy = helper["elf_loader_policy"]
    soname = policy["needed_soname"]
    candidates: list[tuple[str, str]] = []
    pattern = re.compile(
        rf"^\s*{child_pid}:\s+calling init:\s+(.*/{re.escape(soname)})"
        r"(?:\s+\[0\])?\s*$"
    )
    for line in debug_stderr.splitlines():
        match = pattern.fullmatch(line)
        if match is not None:
            candidates.append((line.strip(), match.group(1)))
    if len(candidates) != 1:
        raise RuntimeError(
            "glibc loader evidence did not identify exactly one "
            f"{soname} mapping"
        )
    debug_line, raw_path = candidates[0]
    if not Path(raw_path).is_absolute():
        raise RuntimeError("glibc loader reported a non-absolute library path")
    canonical_path = Path(raw_path).resolve(strict=True)
    expected_path = Path(str(helper["library"]["path"]))
    if canonical_path != expected_path:
        raise RuntimeError(
            "controlled scontrol mapped libslurmfull from an unexpected path: "
            f"{canonical_path}"
        )
    loaded_library = _controlled_file_binding(
        canonical_path,
        label="actually mapped controlled scontrol library",
        expected_sha256=_SCONTROL_LIBRARY_SHA256,
        expected_bytes=_SCONTROL_LIBRARY_BYTES,
        expected_owner_uid=helper["library"]["owner_uid"],
        require_executable=False,
        expected_mode=0o444,
    )
    if _portable_controlled_file(
        loaded_library, label="actually mapped library"
    ) != helper["library"]:
        raise RuntimeError(
            "actually mapped libslurmfull identity differs from the helper binding"
        )
    encoded_debug = debug_stderr.encode("utf-8")
    return {
        "schema": SCONTROL_LOADER_EVIDENCE_SCHEMA,
        "status": "validated",
        "source": "glibc-LD_DEBUG=libs-calling-init",
        "child_pid": child_pid,
        "task_root": helper["task_root"],
        "soname": soname,
        "elf_loader_policy": deepcopy(policy),
        "raw_loaded_path": raw_path,
        "canonical_loaded_path": str(canonical_path),
        "library": loaded_library,
        "debug_line": debug_line,
        "debug_stderr_sha256": hashlib.sha256(encoded_debug).hexdigest(),
        "debug_stderr_bytes": len(encoded_debug),
        "helper_before": deepcopy(dict(helper_before)),
        "helper_after": deepcopy(dict(helper_after)),
        "stable_during_query": True,
        "errors": [],
    }


def scontrol_loader_portable_binding(value: Any) -> dict[str, Any]:
    """Return actual-loader evidence stable across clients and process IDs."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema", "status", "source", "child_pid", "task_root", "soname",
        "elf_loader_policy", "raw_loaded_path", "canonical_loaded_path",
        "library", "debug_line", "debug_stderr_sha256",
        "debug_stderr_bytes", "helper_before", "helper_after",
        "stable_during_query", "errors",
    }:
        raise ValueError("controlled scontrol loader evidence shape is invalid")
    if (
        value.get("schema") != SCONTROL_LOADER_EVIDENCE_SCHEMA
        or value.get("status") != "validated"
        or value.get("errors") != []
        or value.get("source") != "glibc-LD_DEBUG=libs-calling-init"
        or value.get("elf_loader_policy") != _SCONTROL_ELF_LOADER_POLICY
        or value.get("soname") != _SCONTROL_ELF_LOADER_POLICY["needed_soname"]
        or type(value.get("child_pid")) is not int
        or value["child_pid"] <= 0
        or not _is_sha256(value.get("debug_stderr_sha256"))
        or type(value.get("debug_stderr_bytes")) is not int
        or value["debug_stderr_bytes"] <= 0
        or not isinstance(value.get("debug_line"), str)
        or "calling init:" not in value["debug_line"]
        or value.get("stable_during_query") is not True
        or not isinstance(value.get("raw_loaded_path"), str)
        or not Path(value["raw_loaded_path"]).is_absolute()
    ):
        raise ValueError("controlled scontrol loader evidence is invalid")
    if dict(value.get("helper_before") or {}) != dict(
        value.get("helper_after") or {}
    ):
        raise ValueError("controlled scontrol helper changed during loader query")
    query_helper = scontrol_helper_portable_binding(value.get("helper_before"))
    try:
        resolved_raw = Path(value["raw_loaded_path"]).resolve(strict=True)
    except OSError as exc:
        raise ValueError("controlled scontrol loader path is unavailable") from exc
    if str(resolved_raw) != value.get("canonical_loaded_path"):
        raise ValueError("controlled scontrol loader raw path is inconsistent")
    if value["raw_loaded_path"] not in value["debug_line"]:
        raise ValueError("controlled scontrol loader debug line is inconsistent")
    portable = {
        "schema": SCONTROL_LOADER_EVIDENCE_SCHEMA,
        "source": value["source"],
        "task_root": value.get("task_root"),
        "soname": value["soname"],
        "elf_loader_policy": deepcopy(value["elf_loader_policy"]),
        "canonical_loaded_path": value.get("canonical_loaded_path"),
        "query_helper": query_helper,
        "library": _portable_controlled_file(
            value.get("library"), label="actually mapped library"
        ),
    }
    _validate_scontrol_loader_portable_binding(portable)
    return portable


def _validate_scontrol_loader_portable_binding(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "source", "task_root", "soname", "elf_loader_policy",
        "canonical_loaded_path", "query_helper", "library",
    }:
        raise ValueError("portable controlled scontrol loader shape is invalid")
    task_root = value.get("task_root")
    expected_path = (
        Path(task_root) / _SCONTROL_LIBRARY_RELATIVE_PATH
        if isinstance(task_root, str) and Path(task_root).is_absolute()
        else None
    )
    library = value.get("library")
    query_helper = value.get("query_helper")
    if (
        value.get("schema") != SCONTROL_LOADER_EVIDENCE_SCHEMA
        or value.get("source") != "glibc-LD_DEBUG=libs-calling-init"
        or value.get("soname") != _SCONTROL_ELF_LOADER_POLICY["needed_soname"]
        or value.get("elf_loader_policy") != _SCONTROL_ELF_LOADER_POLICY
        or expected_path is None
        or value.get("canonical_loaded_path") != str(expected_path)
        or not isinstance(query_helper, Mapping)
        or not isinstance(library, Mapping)
        or set(library) != {
            "path", "sha256", "bytes", "mode", "owner_uid", "owner_gid",
            "hard_links", "inode",
        }
        or library.get("path") != str(expected_path)
        or library.get("sha256") != _SCONTROL_LIBRARY_SHA256
        or library.get("bytes") != _SCONTROL_LIBRARY_BYTES
        or library.get("mode") != 0o444
        or type(library.get("owner_uid")) is not int
        or library["owner_uid"] < 0
        or type(library.get("owner_gid")) is not int
        or library["owner_gid"] < 0
        or library.get("hard_links") != 1
        or type(library.get("inode")) is not int
        or library["inode"] <= 0
    ):
        raise ValueError("portable controlled scontrol loader binding is invalid")
    _validate_scontrol_helper_portable_binding(query_helper)
    if (
        query_helper.get("task_root") != task_root
        or query_helper.get("library") != library
        or query_helper.get("elf_loader_policy") != value.get("elf_loader_policy")
    ):
        raise ValueError("portable scontrol loader/helper binding is inconsistent")


def validate_scontrol_loader_portable_binding(value: Any) -> None:
    """Validate receipt/pin proof of the actually mapped helper library."""

    _validate_scontrol_loader_portable_binding(value)


def _thread_affinity_errors(observation: Any, *, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(observation, Mapping):
        return [f"current {label} thread-affinity observation is unavailable"]
    if set(observation) != {
        "schema", "source", "task_directory", "thread_count",
        "thread_ids_before", "thread_ids_after", "threads",
        "union_affinity", "expected_affinity",
        "all_threads_within_expected", "union_covers_expected", "stable",
        "errors",
    }:
        errors.append(f"current {label} thread-affinity fields are not exact")
    if observation.get("schema") != THREAD_AFFINITY_OBSERVATION_SCHEMA:
        errors.append(f"current {label} thread-affinity schema mismatch")
    if observation.get("source") != "stable-proc-self-task-status":
        errors.append(f"current {label} thread-affinity source mismatch")
    if observation.get("task_directory") != str(_PROC_SELF_TASK):
        errors.append(f"current {label} thread-affinity task path mismatch")
    threads = observation.get("threads")
    if not isinstance(threads, list) or not threads:
        return errors + [f"current {label} thread-affinity set is empty"]
    if observation.get("thread_count") != len(threads):
        errors.append(f"current {label} thread-affinity count mismatch")
    tids: list[int] = []
    observed_sets: list[set[int]] = []
    for item in threads:
        if not isinstance(item, Mapping) or set(item) != {"tid", "affinity"}:
            errors.append(f"current {label} thread-affinity row is invalid")
            continue
        tid = item.get("tid")
        if type(tid) is not int or tid <= 0 or tid in tids:
            errors.append(f"current {label} thread id is invalid")
            continue
        tids.append(tid)
        try:
            cpus = set(_parse_cpu_list(item.get("affinity")))
            if not cpus:
                raise ValueError("empty")
            observed_sets.append(cpus)
        except Exception:
            errors.append(f"current {label} thread {tid} affinity is invalid")
    expected = set(_parse_cpu_list(_EXPECTED_RELEASE_AFFINITY))
    union = set().union(*observed_sets) if observed_sets else set()
    all_subset = bool(observed_sets) and all(
        values.issubset(expected) for values in observed_sets
    )
    covers = expected.issubset(union)
    if not all_subset:
        errors.append(f"current {label} has a thread outside CPUs 24-31")
    if not covers:
        errors.append(f"current {label} thread union does not cover CPUs 24-31")
    if observation.get("union_affinity") != _format_cpu_list(union):
        errors.append(f"current {label} thread-affinity union mismatch")
    if observation.get("expected_affinity") != _EXPECTED_RELEASE_AFFINITY:
        errors.append(f"current {label} expected thread affinity mismatch")
    if observation.get("all_threads_within_expected") is not all_subset:
        errors.append(f"current {label} thread subset verdict mismatch")
    if observation.get("union_covers_expected") is not covers:
        errors.append(f"current {label} thread coverage verdict mismatch")
    if observation.get("thread_ids_before") != tids:
        errors.append(f"current {label} initial thread inventory mismatch")
    if observation.get("thread_ids_after") != tids:
        errors.append(f"current {label} final thread inventory mismatch")
    if observation.get("stable") is not True or observation.get("errors") != []:
        errors.append(f"current {label} thread-affinity snapshot is not stable")
    return errors


def _current_scontrol_helper_errors(
    observation: Any, *, expected: Any, label: str
) -> list[str]:
    try:
        current = scontrol_helper_portable_binding(observation)
    except Exception as exc:
        return [f"current {label} controlled scontrol helper is invalid: {exc}"]
    try:
        _validate_scontrol_helper_portable_binding(expected)
    except Exception as exc:
        return [f"pinned controlled scontrol helper is invalid: {exc}"]
    if current != expected:
        return [f"current {label} controlled scontrol helper differs from release pin"]
    return []


def _current_scontrol_loader_errors(
    observation: Any,
    *,
    expected: Any,
    helper_observation: Any,
    label: str,
) -> list[str]:
    try:
        current = scontrol_loader_portable_binding(observation)
    except Exception as exc:
        return [f"current {label} scontrol loader proof is invalid: {exc}"]
    try:
        _validate_scontrol_loader_portable_binding(expected)
    except Exception as exc:
        return [f"pinned scontrol loader proof is invalid: {exc}"]
    try:
        helper = scontrol_helper_portable_binding(helper_observation)
    except Exception as exc:
        return [f"current {label} controlled scontrol helper is invalid: {exc}"]
    errors: list[str] = []
    if current != expected:
        errors.append(
            f"current {label} actual scontrol library mapping differs from release pin"
        )
    if (
        current.get("library") != helper.get("library")
        or current.get("elf_loader_policy") != helper.get("elf_loader_policy")
    ):
        errors.append(
            f"current {label} actual scontrol library differs from helper identity"
        )
    return errors


def _normalise_pci_bus_id(value: Any) -> Optional[str]:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="strict")
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip().lower()
    if value.startswith("00000000:"):
        value = "0000:" + value.split(":", 1)[1]
    return value


def _proc_status_value(field: str) -> str:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key == field:
            return value.strip()
    raise RuntimeError(f"/proc/self/status has no {field} field")


def _current_kernel_memory_policy() -> dict[str, Any]:
    """Read the calling thread's Linux default memory policy."""

    max_nodes = 1024
    bits_per_word = ctypes.sizeof(ctypes.c_ulong) * 8
    word_count = (max_nodes + bits_per_word - 1) // bits_per_word
    nodemask = (ctypes.c_ulong * word_count)()
    mode = ctypes.c_int(-1)
    libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    get_mempolicy = libnuma.get_mempolicy
    get_mempolicy.argtypes = (
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
    )
    get_mempolicy.restype = ctypes.c_int
    ctypes.set_errno(0)
    if get_mempolicy(ctypes.byref(mode), nodemask, max_nodes, None, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    nodes = [
        node for node in range(max_nodes)
        if nodemask[node // bits_per_word] & (1 << (node % bits_per_word))
    ]
    return {
        "mode": int(mode.value),
        "mode_name": {
            0: "default",
            1: "preferred",
            2: "bind",
            3: "interleave",
            4: "local",
            5: "preferred_many",
        }.get(int(mode.value), "unknown"),
        "nodes": nodes,
    }


def _query_slurm_controller(
    job_id: str,
    *,
    helper_identity: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Read the live allocation from Slurm rather than trusting job variables."""

    before = observe_scontrol_helper_identity()
    portable_before = scontrol_helper_portable_binding(before)
    if helper_identity is not None and dict(helper_identity) != before:
        raise RuntimeError(
            "caller helper identity differs from the live pre-query identity"
        )
    executable = Path(str(portable_before["executable"]["path"]))
    command_environment = os.environ.copy()
    library_prefix = str(portable_before["library_search_path_prefix"])
    for name in tuple(command_environment):
        if name.startswith("LD_"):
            command_environment.pop(name, None)
    command_environment["LD_LIBRARY_PATH"] = library_prefix
    command_environment["LD_DEBUG"] = "libs"
    command_environment["LC_ALL"] = "C"
    process: Optional[subprocess.Popen[bytes]] = None
    query_error: Optional[Exception] = None
    stdout_bytes = b""
    stderr_bytes = b""
    try:
        process = subprocess.Popen(
            [str(executable), "show", "job", "--oneliner", job_id],
            text=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=command_environment,
        )
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise RuntimeError("controlled scontrol query timed out")
    except Exception as exc:
        query_error = exc
    after = observe_scontrol_helper_identity()
    if after != before:
        raise RuntimeError("controlled scontrol helper changed while queried")
    if query_error is not None:
        raise RuntimeError("controlled scontrol query failed") from query_error
    if process is None:  # pragma: no cover - guarded by query_error
        raise RuntimeError("controlled scontrol query did not start")
    try:
        output = stdout_bytes.decode("utf-8", errors="strict").strip()
        debug_stderr = stderr_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError("controlled scontrol emitted invalid UTF-8") from exc
    loader_evidence = _scontrol_loader_evidence(
        debug_stderr,
        child_pid=process.pid,
        helper_before=before,
        helper_after=after,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"controlled scontrol exited with status {process.returncode}"
        )
    records = [line for line in output.splitlines() if line.strip()]
    if len(records) != 1:
        raise RuntimeError(
            f"scontrol returned {len(records)} records for current job {job_id}"
        )
    fields: dict[str, str] = {}
    for token in shlex.split(records[0]):
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    if fields.get("JobId") != job_id:
        raise RuntimeError("scontrol result does not identify the current job")
    result = {
        key: fields.get(key)
        for key in (
            "JobId",
            "JobName",
            "Partition",
            "NodeList",
            "ReqNodeList",
            "JobState",
            "NumCPUs",
            "NumTasks",
            "CPUs/Task",
            "Command",
        )
    }
    result["loader_evidence"] = loader_evidence
    return result


def _observe_current_release_runtime() -> dict[str, Any]:
    """Observe release facts from this process, Linux, and CUDA.

    This deliberately accepts no caller-supplied state.  The topology JSON is
    evidence to compare against, never the authority for current affinity,
    memory policy, host identity, or the active CUDA device.
    """

    errors: list[str] = []
    environment = {name: os.getenv(name) for name in _RELEASE_ENVIRONMENT_NAMES}
    job_id = environment.get("SLURM_JOB_ID")
    helper_identity = observe_scontrol_helper_identity()
    slurm_controller_loader: Optional[dict[str, Any]] = None
    try:
        if not isinstance(job_id, str) or not job_id.isdigit():
            raise ValueError("SLURM_JOB_ID is missing or invalid")
        slurm_controller = _query_slurm_controller(
            job_id, helper_identity=helper_identity
        )
        slurm_controller_loader = slurm_controller.pop("loader_evidence", None)
    except Exception as exc:
        slurm_controller = None
        errors.append(f"current Slurm allocation could not be observed: {exc}")
    thread_affinity = observe_process_thread_affinity()
    affinity = thread_affinity.get("union_affinity")
    if thread_affinity.get("stable") is not True:
        errors.extend(str(item) for item in thread_affinity.get("errors", []))
    try:
        mems_allowed = _proc_status_value("Mems_allowed_list")
    except Exception as exc:
        mems_allowed = None
        errors.append(f"current Mems_allowed_list could not be observed: {exc}")
    try:
        memory_policy = _current_kernel_memory_policy()
    except Exception as exc:
        memory_policy = None
        errors.append(f"current kernel memory policy could not be observed: {exc}")

    cuda: dict[str, Any] = {
        "visible_devices_environment": environment.get("CUDA_VISIBLE_DEVICES"),
        "device_count": None,
        "current_device": None,
        "pci_bus_id": None,
        "numa_node": None,
    }
    gpu_node_cpulist: Optional[str] = None
    try:
        import cupy  # imported only on the GPU release path

        cuda["device_count"] = int(cupy.cuda.runtime.getDeviceCount())
        cuda["current_device"] = int(cupy.cuda.runtime.getDevice())
        device = cupy.cuda.Device(cuda["current_device"])
        bus_id = _normalise_pci_bus_id(device.pci_bus_id)
        cuda["pci_bus_id"] = bus_id
        if bus_id is not None:
            cuda["numa_node"] = int(
                (_PCI_SYSFS_DEVICES / bus_id / "numa_node")
                .read_text(encoding="utf-8")
                .strip()
            )
            gpu_node_cpulist = (
                _NODE_SYSFS_DEVICES
                / f"node{cuda['numa_node']}"
                / "cpulist"
            ).read_text(encoding="utf-8").strip()
    except Exception as exc:
        errors.append(f"current CUDA device could not be observed: {exc}")

    return {
        "schema": RELEASE_RUNTIME_OBSERVATION_SCHEMA,
        "source": "current-process-linux-cuda",
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "affinity": affinity,
        "thread_affinity": thread_affinity,
        "mems_allowed_list": mems_allowed,
        "gpu_node_cpulist": gpu_node_cpulist,
        "kernel_memory_policy": memory_policy,
        "environment": environment,
        "slurm_controller": slurm_controller,
        "slurm_controller_helper": helper_identity,
        "slurm_controller_loader": slurm_controller_loader,
        "cuda": cuda,
        "errors": errors,
    }


def _release_runtime_contract(
    *,
    payload_sha256: str,
    slurm: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    slurm_controller_helper: Mapping[str, Any],
    slurm_controller_loader: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the future release-job contract committed by snapshot B."""

    return {
        "schema": RELEASE_RUNTIME_CONTRACT_SCHEMA,
        "observation_source": "current-process-linux-cuda",
        "job_name": f"gint-release-{payload_sha256}",
        "node": slurm.get("node"),
        "host": fingerprint.get("host"),
        "partition": _EXPECTED_RELEASE_PARTITION,
        "cpu_affinity": _EXPECTED_RELEASE_AFFINITY,
        "gpu_pci_bus_ids": fingerprint.get("gpu_pci_bus_ids"),
        "gpu_node_cpulist": fingerprint.get("gpu_node_cpulist"),
        "gpu_numa_node": _EXPECTED_RELEASE_NUMA_NODE,
        "mems_allowed_list": fingerprint.get("mems_allowed_list"),
        "memory_policy": {"mode_name": "preferred", "nodes": [3]},
        "topology_path_template": (
            "results/gint-gate/topology-physical8-${SLURM_JOB_ID}.json"
        ),
        "qualification_topology_fingerprint_sha256": canonical_json_sha256(
            fingerprint
        ),
        "slurm_controller_helper": deepcopy(dict(slurm_controller_helper)),
        "slurm_controller_loader": deepcopy(dict(slurm_controller_loader)),
    }


def expected_runtime_release_pin(
    payload: Mapping[str, Any],
    *,
    receipt_sha256: str,
    payload_sha256: str,
) -> dict[str, Any]:
    """Return the deterministic pin embedded in release snapshot B.

    The receipt remains outside the source snapshots.  Snapshot B contains
    this pin at :data:`RELEASE_PIN_RELATIVE_PATH`; its normal source-tree hash
    therefore commits to the exact qualification receipt.  Removing only the
    pin from B must reproduce qualification snapshot A byte for byte.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("runtime gate payload must be a mapping")
    if not _is_sha256(receipt_sha256):
        raise ValueError("receipt_sha256 must be a lowercase SHA-256")
    if not _is_sha256(payload_sha256):
        raise ValueError("payload_sha256 must be a lowercase SHA-256")
    if canonical_json_sha256(payload) != payload_sha256:
        raise ValueError("payload_sha256 does not identify the payload")

    source = payload.get("source")
    runtime = payload.get("runtime")
    qualification = payload.get("qualification")
    transfer = payload.get("transfer_audit")
    if not all(
        isinstance(value, Mapping)
        for value in (source, runtime, qualification, transfer)
    ):
        raise ValueError("runtime gate payload lacks release-pin bindings")
    manifest = source.get("manifest")
    libgint = runtime.get("libgint")
    slurm = qualification.get("slurm")
    topology_fingerprint = qualification.get("topology_fingerprint")
    slurm_controller_helper = qualification.get("slurm_controller_helper")
    slurm_controller_loader = qualification.get("slurm_controller_loader")
    if not all(
        isinstance(value, Mapping)
        for value in (
            manifest, libgint, slurm, topology_fingerprint,
            slurm_controller_helper, slurm_controller_loader,
        )
    ):
        raise ValueError("runtime gate payload has incomplete release-pin bindings")
    tree_sha256 = source.get("tree_sha256")
    if not _is_sha256(tree_sha256):
        raise ValueError("qualification source digest is invalid")
    _validate_scontrol_helper_portable_binding(slurm_controller_helper)
    _validate_scontrol_loader_portable_binding(slurm_controller_loader)
    if (
        slurm_controller_loader.get("library")
        != slurm_controller_helper.get("library")
        or slurm_controller_loader.get("elf_loader_policy")
        != slurm_controller_helper.get("elf_loader_policy")
    ):
        raise ValueError("qualification scontrol loader does not match helper")
    qualification_root = Path(str(source.get("root", "")))
    if (
        not qualification_root.is_absolute()
        or qualification_root.name != "source"
        or slurm_controller_helper.get("task_root")
        != str(qualification_root.parent.parent.parent)
    ):
        raise ValueError("controlled scontrol helper task root is invalid")

    return {
        "schema": RELEASE_PIN_SCHEMA,
        "status": "release-pinned",
        "release_pin_path": RELEASE_PIN_RELATIVE_PATH.as_posix(),
        "lineage": {
            "model": "qualification-source-plus-fixed-release-pin",
            "qualification_source_tree_sha256": tree_sha256,
            "release_source_without_pin_sha256": tree_sha256,
        },
        "receipt": {
            "sha256": receipt_sha256,
            "payload_sha256": payload_sha256,
        },
        "qualification_source": {
            "root": source.get("root"),
            "tree_sha256": tree_sha256,
            "manifest_sha256": manifest.get("sha256"),
            "manifest_bytes": manifest.get("bytes"),
            "binding_sha256": canonical_json_sha256(source),
        },
        "qualification": {
            "job_id": slurm.get("job_id"),
            "topology_fingerprint": deepcopy(dict(topology_fingerprint)),
            "topology_fingerprint_sha256": canonical_json_sha256(
                topology_fingerprint
            ),
            "binding_sha256": canonical_json_sha256(qualification),
        },
        "release_runtime_contract": _release_runtime_contract(
            payload_sha256=payload_sha256,
            slurm=slurm,
            fingerprint=topology_fingerprint,
            slurm_controller_helper=slurm_controller_helper,
            slurm_controller_loader=slurm_controller_loader,
        ),
        "runtime": {
            "libgint_sha256": libgint.get("sha256"),
            "libgint_bytes": libgint.get("bytes"),
            "basis_prod_cache_abi_sha256": canonical_json_sha256(
                runtime.get("basis_prod_cache_abi")
            ),
            "selected_pair_abi_sha256": canonical_json_sha256(
                runtime.get("selected_pair_abi")
            ),
            "selected_c_abi_sha256": canonical_json_sha256(
                runtime.get("selected_c_abi")
            ),
            "binding_sha256": canonical_json_sha256(runtime),
        },
        "transfer_audit_sha256": canonical_json_sha256(transfer),
    }


def release_source_tree_digest(
    source_root: str | os.PathLike[str], *, exclude_release_pin: bool = False
) -> tuple[str, int]:
    """Reproduce the snapshot digest, optionally removing only the release pin."""

    root = _path_without_final_resolution(source_root, name="release source root")
    _reject_symlink_components(root, name="release source root")
    root_stat = root.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("release source root must be a directory")
    if root.resolve(strict=True) != root:
        raise ValueError("release source root must be canonical")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"release source path traverses symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _SOURCE_EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        if exclude_release_pin and relative == RELEASE_PIN_RELATIVE_PATH:
            continue
        if path.resolve() != path:
            raise ValueError(f"release source path traverses symlink: {path}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
        count += 1
    if count == 0:
        raise ValueError("release source tree contains no deployable files")
    return digest.hexdigest(), count


def source_snapshot_normalization_errors(
    manifest: Any, *, source_root: Optional[Path] = None
) -> list[str]:
    """Validate the v3 proof that all checkout links were materialized."""

    if not isinstance(manifest, Mapping):
        return ["source snapshot manifest root must be an object"]
    if manifest.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
        return ["source snapshot manifest schema mismatch"]
    value = manifest.get("source_normalization")
    if not isinstance(value, Mapping):
        return ["source normalization manifest is missing"]
    errors: list[str] = []
    if value.get("schema") != SOURCE_NORMALIZATION_SCHEMA:
        errors.append("source normalization schema mismatch")
    if value.get("policy") != SOURCE_NORMALIZATION_POLICY:
        errors.append("source normalization policy mismatch")
    inventory = value.get("source_link_inventory")
    if not isinstance(inventory, Mapping):
        errors.append("source normalization inventory is missing")
        inventory = {}
    if inventory.get("schema") != SOURCE_NORMALIZATION_SCHEMA:
        errors.append("source normalization inventory schema mismatch")
    if inventory.get("policy") != SOURCE_NORMALIZATION_POLICY:
        errors.append("source normalization inventory policy mismatch")
    links = inventory.get("links")
    if not isinstance(links, list):
        errors.append("source normalization links must be a list")
        links = []
    elif not all(isinstance(link, Mapping) for link in links):
        errors.append("source normalization link entry must be an object")
        links = []
    if inventory.get("link_count") != len(links):
        errors.append("source normalization link count mismatch")
    seen: set[str] = set()
    for link in links:
        relative = link.get("relative_path")
        target = link.get("target")
        resolved = link.get("resolved_target_relative_path")
        relative_path = Path(relative) if isinstance(relative, str) else None
        resolved_path = Path(resolved) if isinstance(resolved, str) else None
        if (
            not isinstance(relative, str)
            or not relative
            or relative_path is None
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or any(part in _SOURCE_EXCLUDED_PARTS for part in relative_path.parts)
            or relative_path.as_posix() != relative
            or relative in seen
        ):
            errors.append("source normalization link path is invalid")
        else:
            seen.add(relative)
        if (
            link.get("link_type") != "relative-leaf-file"
            or not isinstance(target, str)
            or not target
            or os.path.isabs(target)
            or not isinstance(resolved, str)
            or not resolved
            or resolved_path is None
            or resolved_path.is_absolute()
            or ".." in resolved_path.parts
            or any(part in _SOURCE_EXCLUDED_PARTS for part in resolved_path.parts)
            or resolved_path.as_posix() != resolved
            or not isinstance(link.get("target_bytes"), int)
            or link.get("target_bytes") < 0
            or not _is_sha256(link.get("target_sha256"))
        ):
            errors.append("source normalization link identity is invalid")
            continue
        target_parts = list(relative_path.parent.parts)
        target_components = target.split("/")
        if (
            "\\" in target
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in target
            )
            or any(component in {"", "."} for component in target_components)
        ):
            errors.append("source normalization literal target is invalid")
            continue
        escaped = False
        for component in target_components:
            if component == "..":
                if not target_parts:
                    escaped = True
                    break
                target_parts.pop()
            else:
                target_parts.append(component)
        if escaped or Path(*target_parts) != resolved_path:
            errors.append("source normalization resolved target mismatch")
            continue
        if source_root is not None:
            for label, path in (
                ("materialized link", source_root / relative_path),
                ("resolved target", source_root / resolved_path),
            ):
                try:
                    _, data, info = _read_regular_file(
                        path, name=f"source normalization {label}",
                        minimum_bytes=0,
                    )
                    if (
                        info.st_size != link.get("target_bytes")
                        or _sha256_bytes(data) != link.get("target_sha256")
                    ):
                        raise ValueError("content mismatch")
                except Exception:
                    errors.append(
                        f"source normalization {label} content mismatch: {path}"
                    )
    encoded_inventory = (
        json.dumps(
            dict(inventory), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ) + "\n"
    ).encode("ascii")
    if (
        _sha256_bytes(encoded_inventory)
        != value.get("source_link_inventory_sha256")
    ):
        errors.append("source normalization inventory digest mismatch")
    if value.get("materialized_link_count") != len(links):
        errors.append("source normalization materialized-link count mismatch")
    if value.get("materialized_links") != links:
        errors.append("source normalization materialized-link identity mismatch")
    if value.get("published_symlink_count") != 0:
        errors.append("source normalization published-link count is not zero")
    if value.get("published_special_file_count") != 0:
        errors.append("source normalization published-special-file count is not zero")
    if value.get("remote_verified_after_runtime_binding") is not True:
        errors.append("source normalization remote verification is missing")
    uploaded_view = dict(value)
    for name in (
        "remote_verified_after_runtime_binding",
        "remote_regular_file_count_after_runtime_binding",
        "remote_directory_count_after_runtime_binding",
        "uploaded_record_sha256",
        "uploaded_record_bytes",
    ):
        uploaded_view.pop(name, None)
    uploaded_data = (
        json.dumps(uploaded_view, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if (
        not _is_sha256(value.get("uploaded_record_sha256"))
        or value.get("uploaded_record_sha256")
        != _sha256_bytes(uploaded_data)
        or value.get("uploaded_record_bytes") != len(uploaded_data)
    ):
        errors.append("source normalization uploaded-record binding mismatch")
    for name in (
        "regular_file_count_after", "directory_count_after",
        "remote_regular_file_count_after_runtime_binding",
        "remote_directory_count_after_runtime_binding",
    ):
        if not isinstance(value.get(name), int) or value[name] < 0:
            errors.append(f"source normalization {name} is invalid")
    return errors


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_loads(value: bytes, *, label: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = item
        return result

    def reject_constant(name: str) -> Any:
        raise ValueError(f"{label} contains non-finite JSON number {name}")

    return json.loads(
        value.decode("utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=reject_constant,
    )


def _path_without_final_resolution(value: Any, *, name: str) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be a filesystem path")
    text = os.fspath(value)
    if isinstance(text, bytes):
        text = os.fsdecode(text)
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return Path(os.path.abspath(os.path.expanduser(text)))


def _reject_symlink_components(path: Path, *, name: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{name} path traverses symbolic link {current}")


def _read_regular_file(
    value: Any,
    *,
    name: str,
    require_read_only: bool = False,
    minimum_bytes: int = 1,
    maximum_bytes: int = 256 * 1024 * 1024,
) -> tuple[Path, bytes, os.stat_result]:
    path = _path_without_final_resolution(value, name=name)
    _reject_symlink_components(path, name=name)
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{name} must not be a symbolic link")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} must be a regular file")
    if require_read_only and before.st_mode & (
        stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    ):
        raise ValueError(f"{name} must have no filesystem write bits")
    if (
        minimum_bytes < 0
        or before.st_size < minimum_bytes
        or before.st_size > maximum_bytes
    ):
        raise ValueError(f"{name} has an invalid byte length")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{name} changed while it was opened")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > maximum_bytes:
            raise ValueError(f"{name} exceeds its maximum byte length")
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    ):
        raise ValueError(f"{name} changed while it was read")
    return path, data, after


def _file_binding_errors(
    binding: Any,
    *,
    label: str,
    require_read_only: bool = False,
) -> tuple[list[str], Optional[Path], Optional[bytes]]:
    errors: list[str] = []
    if not isinstance(binding, Mapping):
        return [f"{label} binding must be an object"], None, None
    path_value = binding.get("path")
    try:
        path, data, file_stat = _read_regular_file(
            path_value,
            name=label,
            require_read_only=require_read_only,
        )
    except Exception as exc:
        return [str(exc)], None, None
    if binding.get("path") != str(path):
        errors.append(f"{label} path must be absolute and normalized")
    if binding.get("bytes") != file_stat.st_size:
        errors.append(f"{label} byte length differs from the receipt")
    if binding.get("sha256") != _sha256_bytes(data):
        errors.append(f"{label} SHA-256 differs from the receipt")
    return errors, path, data


def _normalised_loaded_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(_path_without_final_resolution(value, name="loaded path"))
    except (TypeError, ValueError):
        return None


def _source_root_errors(
    value: Any,
    *,
    label: str,
    require_read_only: bool,
) -> tuple[list[str], Optional[Path]]:
    """Validate a canonical, symlink-free snapshot ``source`` directory."""

    errors: list[str] = []
    try:
        root = _path_without_final_resolution(value, name=label)
        _reject_symlink_components(root, name=label)
        root_stat = root.lstat()
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"{label} must be a directory")
        if root.resolve(strict=True) != root:
            raise ValueError(f"{label} must be canonical")
    except Exception as exc:
        return [str(exc)], None
    if root.name != "source" or root.parent.parent.name != "snapshots":
        errors.append(f"{label} is outside a content-addressed snapshot")
    if require_read_only:
        for path in (root, *root.rglob("*")):
            try:
                info = path.lstat()
            except Exception as exc:
                errors.append(f"{label} path could not be inspected: {exc}")
                continue
            if stat.S_ISLNK(info.st_mode):
                errors.append(f"{label} contains symbolic link {path}")
            if info.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                errors.append(f"{label} contains writable path {path}")
    return errors, root


def _snapshot_manifest_checks(
    manifest: Any,
    *,
    source_root: Path,
    tree_sha256: Any,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        return [f"{label} root must be an object"]
    normalization_errors = source_snapshot_normalization_errors(
        manifest, source_root=source_root
    )
    errors.extend(f"{label} {error}" for error in normalization_errors)
    if manifest.get("tree_sha256") != tree_sha256:
        errors.append(f"{label} tree digest mismatch")
    if manifest.get("source") != str(source_root):
        errors.append(f"{label} source root mismatch")
    if manifest.get("immutable") is not True:
        errors.append(f"{label} is not immutable")
    return errors


def _publication_attestation_checks(
    manifest: Any,
    *,
    source_root: Path,
) -> tuple[list[str], Optional[dict[str, Any]]]:
    """Validate the v3 sibling attestation and return its exact object."""

    if not isinstance(manifest, Mapping):
        return [], None
    if manifest.get("schema") != "gpu4pyscf.source-snapshot.v3":
        return [], None
    errors: list[str] = []
    policy = manifest.get("publication_policy")
    if policy != PUBLICATION_POLICY:
        return ["v3 publication policy is invalid"], None
    snapshots_root = source_root.parent.parent
    attestation_name = source_root.parent.name + PUBLICATION_ATTESTATION_SUFFIX
    try:
        _reject_symlink_components(snapshots_root, name="publication snapshots root")
        root_before = snapshots_root.lstat()
        if not stat.S_ISDIR(root_before.st_mode):
            raise ValueError("publication snapshots root must be a directory")
        root_fd = os.open(snapshots_root, DIRECTORY_FLAGS | NOFOLLOW)
        try:
            root_opened = os.fstat(root_fd)
            if (root_opened.st_dev, root_opened.st_ino) != (
                root_before.st_dev, root_before.st_ino
            ):
                raise ValueError("publication snapshots root changed while opened")
            snapshot_before = os.stat(
                source_root.parent.name, dir_fd=root_fd, follow_symlinks=False
            )
            source_identity = source_root.parent.lstat()
            if (
                stat.S_ISLNK(snapshot_before.st_mode)
                or not stat.S_ISDIR(snapshot_before.st_mode)
                or (snapshot_before.st_dev, snapshot_before.st_ino)
                != (source_identity.st_dev, source_identity.st_ino)
            ):
                raise ValueError("publication snapshot identity changed")
            attestation_before = os.stat(
                attestation_name, dir_fd=root_fd, follow_symlinks=False
            )
            if (
                stat.S_ISLNK(attestation_before.st_mode)
                or not stat.S_ISREG(attestation_before.st_mode)
                or attestation_before.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                or attestation_before.st_size > 256 * 1024
            ):
                raise ValueError("publication attestation file identity is invalid")
            attestation_fd = os.open(
                attestation_name, os.O_RDONLY | NOFOLLOW, dir_fd=root_fd
            )
            try:
                opened = os.fstat(attestation_fd)
                if (opened.st_dev, opened.st_ino) != (
                    attestation_before.st_dev, attestation_before.st_ino
                ):
                    raise ValueError("publication attestation changed while opened")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(attestation_fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if sum(map(len, chunks)) > 256 * 1024:
                        raise ValueError("publication attestation is too large")
                attestation_bytes = b"".join(chunks)
            finally:
                os.close(attestation_fd)
            attestation_after = os.stat(
                attestation_name, dir_fd=root_fd, follow_symlinks=False
            )
            snapshot_after = os.stat(
                source_root.parent.name, dir_fd=root_fd, follow_symlinks=False
            )
            root_after = os.fstat(root_fd)
            if (
                (attestation_after.st_dev, attestation_after.st_ino,
                 attestation_after.st_size, attestation_after.st_mtime_ns)
                != (attestation_before.st_dev, attestation_before.st_ino,
                    attestation_before.st_size, attestation_before.st_mtime_ns)
                or (snapshot_after.st_dev, snapshot_after.st_ino)
                != (snapshot_before.st_dev, snapshot_before.st_ino)
                or (root_after.st_dev, root_after.st_ino)
                != (root_before.st_dev, root_before.st_ino)
            ):
                raise ValueError("publication attestation ancestry changed while read")
        finally:
            os.close(root_fd)
        attestation = _strict_json_loads(
            attestation_bytes, label="qualification publication attestation"
        )
    except Exception as exc:
        return [f"qualification publication attestation is invalid: {exc}"], None
    if not isinstance(attestation, Mapping):
        return ["qualification publication attestation root must be an object"], None
    if set(attestation) != PUBLICATION_ATTESTATION_KEYS:
        errors.append("qualification publication attestation keys are invalid")
    if attestation.get("schema") != PUBLICATION_ATTESTATION_SCHEMA:
        errors.append("qualification publication attestation schema mismatch")
    if attestation.get("status") != "published":
        errors.append("qualification publication attestation status is invalid")
    digest = source_root.parent.name
    if attestation.get("source_tree_sha256") != digest:
        errors.append("qualification publication attestation source digest mismatch")
    if attestation.get("snapshot_name") != digest:
        errors.append("qualification publication attestation snapshot name mismatch")
    if attestation.get("publish_protocol") not in PUBLICATION_PROTOCOLS:
        errors.append("qualification publication attestation protocol is invalid")
    if attestation.get("trust_boundary") != PUBLICATION_TRUST_BOUNDARY:
        errors.append("qualification publication attestation trust boundary mismatch")
    if attestation.get("publication_policy_sha256") != canonical_json_sha256(policy):
        errors.append("qualification publication attestation policy mismatch")
    manifest_path = source_root.parent / "manifest.json"
    try:
        _, manifest_bytes, _ = _read_regular_file(
            manifest_path,
            name="qualification snapshot manifest",
            require_read_only=True,
            maximum_bytes=16 * 1024 * 1024,
        )
    except Exception as exc:
        errors.append(f"qualification snapshot manifest could not be read: {exc}")
        manifest_bytes = b""
    manifest_hash = _sha256_bytes(manifest_bytes)
    if attestation.get("published_manifest_sha256") != manifest_hash:
        errors.append("qualification publication attestation manifest hash mismatch")
    if attestation.get("prepublication_manifest_sha256") != manifest_hash:
        errors.append("qualification publication attestation prepublication hash mismatch")
    identity = attestation.get("published_directory_identity")
    root_stat = source_root.parent.lstat()
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"device", "inode"}
        or type(identity.get("device")) is not int
        or identity["device"] < 0
        or type(identity.get("inode")) is not int
        or identity["inode"] <= 0
    ):
        errors.append(
            "qualification publication attestation directory identity is invalid"
        )
    # st_dev is local to the observing client's mount namespace.  Preserve it
    # in the signed attestation as a diagnostic, but use the portable inode for
    # the shared NFS directory identity check.
    elif identity["inode"] != root_stat.st_ino:
        errors.append("qualification publication attestation directory identity mismatch")
    unsigned = dict(attestation)
    unsigned.pop("attestation_sha256", None)
    if attestation.get("attestation_sha256") != canonical_json_sha256(unsigned):
        errors.append("qualification publication attestation self-hash mismatch")
    return errors, dict(attestation)


def _current_release_binding_errors(
    observation: Any,
    *,
    release_topology: Any,
    qualification_topology: Any,
    qualification_fingerprint: Any,
    release_topology_file: Optional[Path],
    release_topology_sha256: Optional[str],
    release_task_root: Optional[Path],
    release_source_root: Optional[Path],
    release_runtime_contract: Any,
    payload_sha256: Any,
    loaded_release_job_id: Optional[str],
) -> list[str]:
    """Compare independently observed runtime state with pinned evidence."""

    errors: list[str] = []
    if not isinstance(observation, Mapping):
        return ["current release runtime observation is unavailable"]
    if observation.get("schema") != RELEASE_RUNTIME_OBSERVATION_SCHEMA:
        errors.append("current release runtime observation schema mismatch")
    observed_errors = observation.get("errors")
    if not isinstance(observed_errors, list):
        errors.append("current release runtime observation errors are invalid")
    elif observed_errors:
        errors.extend(str(error) for error in observed_errors)
    thread_affinity = observation.get("thread_affinity")
    errors.extend(_thread_affinity_errors(thread_affinity, label="release"))

    if not isinstance(release_topology, Mapping):
        errors.append("current release topology root must be an object")
        release_topology = {}
    if not isinstance(qualification_topology, Mapping):
        errors.append("qualification topology root must be an object")
        qualification_topology = {}
    if not isinstance(qualification_fingerprint, Mapping):
        errors.append("qualification topology fingerprint is unavailable")
        qualification_fingerprint = {}
    if not isinstance(release_runtime_contract, Mapping):
        errors.append("release pin runtime contract is unavailable")
        release_runtime_contract = {}
    errors.extend(_current_scontrol_helper_errors(
        observation.get("slurm_controller_helper"),
        expected=release_runtime_contract.get("slurm_controller_helper"),
        label="release",
    ))
    errors.extend(_current_scontrol_loader_errors(
        observation.get("slurm_controller_loader"),
        expected=release_runtime_contract.get("slurm_controller_loader"),
        helper_observation=observation.get("slurm_controller_helper"),
        label="release",
    ))

    environment = observation.get("environment")
    if not isinstance(environment, Mapping):
        errors.append("current release process environment is unavailable")
        environment = {}
    cuda = observation.get("cuda")
    if not isinstance(cuda, Mapping):
        errors.append("current release CUDA observation is unavailable")
        cuda = {}
    slurm_controller = observation.get("slurm_controller")
    if not isinstance(slurm_controller, Mapping):
        errors.append("current live Slurm-controller observation is unavailable")
        slurm_controller = {}
    release_slurm = release_topology.get("slurm")
    if not isinstance(release_slurm, Mapping):
        errors.append("current release topology Slurm binding is unavailable")
        release_slurm = {}
    qualification_slurm = qualification_topology.get("slurm")
    if not isinstance(qualification_slurm, Mapping):
        errors.append("qualification topology Slurm binding is unavailable")
        qualification_slurm = {}

    actual_job_id = environment.get("SLURM_JOB_ID")
    if (
        not isinstance(actual_job_id, str)
        or not actual_job_id.isdigit()
        or int(actual_job_id) < 1
    ):
        errors.append("current process Slurm job id is unavailable or invalid")
    if loaded_release_job_id is not None and loaded_release_job_id != actual_job_id:
        errors.append("caller release job id differs from the current environment")
    if release_slurm.get("SLURM_JOB_ID") != actual_job_id:
        errors.append("current process is not bound to release topology job")
    for controller_key, environment_key in (
        ("JobId", "SLURM_JOB_ID"),
        ("JobName", "SLURM_JOB_NAME"),
        ("Partition", "SLURM_JOB_PARTITION"),
        ("NodeList", "SLURM_JOB_NODELIST"),
        ("CPUs/Task", "SLURM_CPUS_PER_TASK"),
    ):
        if slurm_controller.get(controller_key) != environment.get(environment_key):
            errors.append(
                "current Slurm controller differs from environment for "
                f"{controller_key}"
            )
    if slurm_controller.get("JobState") not in {"RUNNING", "COMPLETING"}:
        errors.append("current release Slurm job is not running")
    if (
        slurm_controller.get("NumTasks") != "1"
        or slurm_controller.get("CPUs/Task") != "64"
    ):
        errors.append("current release Slurm CPU/task allocation mismatch")
    if slurm_controller.get("ReqNodeList") != release_runtime_contract.get("node"):
        errors.append(
            "current release Slurm job was not pinned to the qualification node"
        )

    expected_topology_file = None
    if release_task_root is not None and isinstance(actual_job_id, str):
        expected_topology_file = (
            release_task_root
            / "results"
            / "gint-gate"
            / f"topology-physical8-{actual_job_id}.json"
        )
    if release_topology_file != expected_topology_file:
        errors.append(
            "current release topology is not the fixed task/results job path"
        )

    expected_name = (
        f"gint-release-{payload_sha256}" if _is_sha256(payload_sha256) else None
    )
    expected_environment = {
        "SLURM_JOB_NAME": expected_name,
        "SLURM_JOB_NODELIST": release_runtime_contract.get("node"),
        "SLURM_JOB_PARTITION": _EXPECTED_RELEASE_PARTITION,
        "CCSD_PERFORMANCE_ELIGIBLE": "true",
        "CCSD_BENCHMARK_TRACK": "performance",
        "CCSD_TOPOLOGY_RECORD": (
            str(expected_topology_file)
            if expected_topology_file is not None else None
        ),
        "CCSD_TOPOLOGY_SHA256": release_topology_sha256,
        "CCSD_BOUND_CPUS": _EXPECTED_RELEASE_AFFINITY,
        "GPU4PYSCF_NUMA": str(_EXPECTED_RELEASE_NUMA_NODE),
        "CCSD_SCONTROL_PATH": (
            (release_runtime_contract.get("slurm_controller_helper") or {}).get(
                "configured_path"
            )
        ),
    }
    if release_runtime_contract.get("node") != _EXPECTED_RELEASE_NODE:
        errors.append("qualification node differs from fixed MTU node compute-1-6")
    if release_runtime_contract.get("host") != _EXPECTED_RELEASE_NODE:
        errors.append("qualification host differs from fixed MTU node compute-1-6")
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            errors.append(f"current release environment {name} mismatch")
    for name in (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "CUDA_VISIBLE_DEVICES",
    ):
        if environment.get(name) != release_slurm.get(name):
            errors.append(
                f"current release environment differs from topology for {name}"
            )

    if release_slurm.get("SLURM_JOB_NAME") != expected_name:
        errors.append("current release Slurm job name does not bind receipt payload")
    if release_slurm.get("SLURM_JOB_PARTITION") != _EXPECTED_RELEASE_PARTITION:
        errors.append("current release Slurm partition mismatch")
    if observation.get("host") != release_topology.get("host"):
        errors.append("current release topology host mismatch")
    if observation.get("host") != environment.get("SLURM_JOB_NODELIST"):
        errors.append("current hostname differs from the Slurm node allocation")
    if observation.get("host") != release_runtime_contract.get("host"):
        errors.append("current hostname differs from the pinned qualification host")
    if release_topology.get("pid") != observation.get("pid"):
        errors.append("current release topology PID mismatch")

    if observation.get("affinity") != _EXPECTED_RELEASE_AFFINITY:
        errors.append("current release CPU affinity is not 24-31")
    if observation.get("affinity") != (
        thread_affinity.get("union_affinity")
        if isinstance(thread_affinity, Mapping) else None
    ):
        errors.append("current release aggregate affinity is inconsistent")
    if release_topology.get("selected_cpulist") != observation.get("affinity"):
        errors.append("current release CPU affinity differs from topology")
    if release_topology.get("observed_cpulist_after_bind") != observation.get(
        "affinity"
    ):
        errors.append("current release post-bind CPU affinity mismatch")
    if release_runtime_contract.get("cpu_affinity") != observation.get("affinity"):
        errors.append("current release CPU affinity differs from the release pin")
    gpu_node_cpulist = observation.get("gpu_node_cpulist")
    try:
        if not set(_parse_cpu_list(observation.get("affinity"))).issubset(
            _parse_cpu_list(gpu_node_cpulist)
        ):
            errors.append("current CPU affinity is outside the GPU-local NUMA node")
    except Exception:
        errors.append("current GPU-node CPU list is invalid")
    if release_topology.get("gpu_node_cpulist") != gpu_node_cpulist:
        errors.append("current GPU-node CPU list differs from topology")
    if qualification_fingerprint.get("gpu_node_cpulist") != gpu_node_cpulist:
        errors.append("current GPU-node CPU list differs from qualification")
    if release_runtime_contract.get("gpu_node_cpulist") != gpu_node_cpulist:
        errors.append("current GPU-node CPU list differs from the release pin")

    if release_topology.get("mode") != "physical8":
        errors.append("current release topology mode is not physical8")
    if (
        release_topology.get("requested_physical_cores") != 8
        or release_topology.get("requested_threads") != 8
    ):
        errors.append("current release topology does not request eight physical cores")
    if release_topology.get("performance_eligible") is not True:
        errors.append("current release topology is not performance eligible")
    if release_topology.get("benchmark_track") != "performance":
        errors.append("current release topology benchmark track mismatch")

    if cuda.get("device_count") != 1:
        errors.append("current CUDA runtime does not expose exactly one GPU")
    if cuda.get("current_device") != 0:
        errors.append("current CUDA runtime device is not visible-device index 0")
    visible_devices = environment.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(visible_devices, str) or not visible_devices.strip():
        errors.append("current CUDA_VISIBLE_DEVICES is unavailable")
    if cuda.get("visible_devices_environment") != visible_devices:
        errors.append("current CUDA visibility observation is inconsistent")
    if qualification_slurm.get("CUDA_VISIBLE_DEVICES") != visible_devices:
        errors.append("current CUDA visible device differs from qualification")

    observed_bus_id = _normalise_pci_bus_id(cuda.get("pci_bus_id"))
    release_bus_ids = [
        _normalise_pci_bus_id(value)
        for value in (release_topology.get("gpu_pci_bus_ids") or [])
    ]
    qualification_bus_ids = [
        _normalise_pci_bus_id(value)
        for value in (qualification_fingerprint.get("gpu_pci_bus_ids") or [])
    ]
    pinned_bus_ids = [
        _normalise_pci_bus_id(value)
        for value in (release_runtime_contract.get("gpu_pci_bus_ids") or [])
    ]
    if not observed_bus_id or [observed_bus_id] != release_bus_ids:
        errors.append("current release GPU PCI binding mismatch")
    if [observed_bus_id] != qualification_bus_ids:
        errors.append("current GPU PCI bus differs from qualification")
    if [observed_bus_id] != pinned_bus_ids:
        errors.append("current GPU PCI bus differs from the release pin")
    if cuda.get("numa_node") != _EXPECTED_RELEASE_NUMA_NODE:
        errors.append("current CUDA device is not attached to NUMA node 3")
    if release_topology.get("gpu_numa_node") != cuda.get("numa_node"):
        errors.append("current release GPU NUMA binding mismatch")
    if release_runtime_contract.get("gpu_numa_node") != cuda.get("numa_node"):
        errors.append("current GPU NUMA node differs from the release pin")

    mems_allowed = observation.get("mems_allowed_list")
    try:
        if _EXPECTED_RELEASE_NUMA_NODE not in _parse_cpu_list(mems_allowed):
            errors.append("current Mems_allowed_list excludes NUMA node 3")
    except Exception:
        errors.append("current Mems_allowed_list is invalid")
    if release_topology.get("mems_allowed_list") != mems_allowed:
        errors.append("current Mems_allowed_list differs from topology")
    if release_runtime_contract.get("mems_allowed_list") != mems_allowed:
        errors.append("current Mems_allowed_list differs from the release pin")

    observed_policy = observation.get("kernel_memory_policy")
    if not isinstance(observed_policy, Mapping):
        errors.append("current kernel memory policy is unavailable")
        observed_policy = {}
    if (
        observed_policy.get("mode_name") != "preferred"
        or observed_policy.get("nodes") != [_EXPECTED_RELEASE_NUMA_NODE]
    ):
        errors.append("current kernel memory policy is not preferred NUMA node 3")
    topology_policy = release_topology.get("kernel_memory_policy") or {}
    topology_after = topology_policy.get("after") or {}
    if (
        topology_policy.get("verified") is not True
        or topology_after.get("mode_name") != observed_policy.get("mode_name")
        or topology_after.get("nodes") != observed_policy.get("nodes")
    ):
        errors.append("current release NUMA memory policy mismatch")
    pinned_policy = release_runtime_contract.get("memory_policy") or {}
    if (
        pinned_policy.get("mode_name") != observed_policy.get("mode_name")
        or pinned_policy.get("nodes") != observed_policy.get("nodes")
    ):
        errors.append("current memory policy differs from the release pin")

    expected_gate_script = (
        release_source_root
        / "benchmarks"
        / "cc"
        / "a100_water8"
        / "gint_gate.py"
        if release_source_root is not None else None
    )
    command_paths = [
        _normalised_loaded_path(item)
        for item in release_topology.get("command", [])
    ]
    if expected_gate_script is None or str(expected_gate_script) not in command_paths:
        errors.append("current release topology did not launch snapshot B gate")
    expected_launcher = (
        release_source_root
        / "benchmarks"
        / "cc"
        / "a100_water8"
        / "run_mtu_gint_gate.sbatch"
        if release_source_root is not None else None
    )
    if (
        expected_launcher is None
        or _normalised_loaded_path(slurm_controller.get("Command"))
        != str(expected_launcher)
    ):
        errors.append("current Slurm allocation did not launch snapshot B entry point")
    return errors


def _topology_fingerprint(topology: Mapping[str, Any]) -> dict[str, Any]:
    """Return the hardware/allocation fields committed by qualification."""

    return {
        "host": topology.get("host"),
        "mode": topology.get("mode"),
        "expected_gpu_numa_node": topology.get("expected_gpu_numa_node"),
        "requested_physical_cores": topology.get("requested_physical_cores"),
        "requested_threads": topology.get("requested_threads"),
        "gpu_numa_node": topology.get("gpu_numa_node"),
        "gpu_pci_bus_ids": topology.get("gpu_pci_bus_ids"),
        "gpu_node_cpulist": topology.get("gpu_node_cpulist"),
        "allowed_cpulist_before": topology.get("allowed_cpulist_before"),
        "node_allowed_intersection": topology.get("node_allowed_intersection"),
        "selected_cpulist": topology.get("selected_cpulist"),
        "observed_cpulist_after_bind": topology.get(
            "observed_cpulist_after_bind"
        ),
        "mems_allowed_list": topology.get("mems_allowed_list"),
        "benchmark_track": topology.get("benchmark_track"),
    }


def _current_consumer_binding_errors(
    observation: Any,
    *,
    execution_mode: str,
    consumer_topology: Any,
    qualification_topology: Any,
    qualification_fingerprint: Any,
    consumer_topology_file: Optional[Path],
    consumer_topology_sha256: Optional[str],
    release_task_root: Optional[Path],
    release_source_root: Optional[Path],
    release_runtime_contract: Any,
    loaded_job_id: Optional[str],
) -> list[str]:
    """Validate a benchmark/CP consumer from current process observations.

    Qualification evidence establishes the immutable B source, binary, ABI,
    transfer ledger, node, GPU, and NUMA contract.  Each consumer must still
    prove its own live Slurm allocation, topology record, command, affinity,
    memory policy, and CUDA device.  No caller-supplied mapping is treated as
    authority for any of those observations.
    """

    errors: list[str] = []
    if execution_mode not in {
        "consumer-benchmark", "consumer-counterpoise"
    }:
        return ["consumer runtime gate execution mode is invalid"]
    if not isinstance(observation, Mapping):
        return ["current consumer runtime observation is unavailable"]
    if observation.get("schema") != RELEASE_RUNTIME_OBSERVATION_SCHEMA:
        errors.append("current consumer runtime observation schema mismatch")
    observed_errors = observation.get("errors")
    if not isinstance(observed_errors, list):
        errors.append("current consumer runtime observation errors are invalid")
    elif observed_errors:
        errors.extend(str(error) for error in observed_errors)
    thread_affinity = observation.get("thread_affinity")
    errors.extend(_thread_affinity_errors(thread_affinity, label="consumer"))

    if not isinstance(consumer_topology, Mapping):
        errors.append("current consumer topology root must be an object")
        consumer_topology = {}
    if not isinstance(qualification_topology, Mapping):
        errors.append("qualification topology root must be an object")
        qualification_topology = {}
    if not isinstance(qualification_fingerprint, Mapping):
        errors.append("qualification topology fingerprint is unavailable")
        qualification_fingerprint = {}
    if not isinstance(release_runtime_contract, Mapping):
        errors.append("release pin runtime contract is unavailable")
        release_runtime_contract = {}
    errors.extend(_current_scontrol_helper_errors(
        observation.get("slurm_controller_helper"),
        expected=release_runtime_contract.get("slurm_controller_helper"),
        label="consumer",
    ))
    errors.extend(_current_scontrol_loader_errors(
        observation.get("slurm_controller_loader"),
        expected=release_runtime_contract.get("slurm_controller_loader"),
        helper_observation=observation.get("slurm_controller_helper"),
        label="consumer",
    ))

    environment = observation.get("environment")
    if not isinstance(environment, Mapping):
        errors.append("current consumer process environment is unavailable")
        environment = {}
    cuda = observation.get("cuda")
    if not isinstance(cuda, Mapping):
        errors.append("current consumer CUDA observation is unavailable")
        cuda = {}
    controller = observation.get("slurm_controller")
    if not isinstance(controller, Mapping):
        errors.append("current live Slurm-controller observation is unavailable")
        controller = {}
    consumer_slurm = consumer_topology.get("slurm")
    if not isinstance(consumer_slurm, Mapping):
        errors.append("current consumer topology Slurm binding is unavailable")
        consumer_slurm = {}
    qualification_slurm = qualification_topology.get("slurm")
    if not isinstance(qualification_slurm, Mapping):
        errors.append("qualification topology Slurm binding is unavailable")
        qualification_slurm = {}

    job_id = environment.get("SLURM_JOB_ID")
    if (
        not isinstance(job_id, str)
        or not job_id.isdigit()
        or int(job_id) < 1
    ):
        errors.append("current consumer Slurm job id is unavailable or invalid")
    if loaded_job_id is not None and loaded_job_id != job_id:
        errors.append("caller consumer job id differs from the current environment")
    if consumer_slurm.get("SLURM_JOB_ID") != job_id:
        errors.append("current process is not bound to consumer topology job")

    for controller_key, environment_key in (
        ("JobId", "SLURM_JOB_ID"),
        ("JobName", "SLURM_JOB_NAME"),
        ("Partition", "SLURM_JOB_PARTITION"),
        ("NodeList", "SLURM_JOB_NODELIST"),
        ("CPUs/Task", "SLURM_CPUS_PER_TASK"),
    ):
        if controller.get(controller_key) != environment.get(environment_key):
            errors.append(
                "current Slurm controller differs from environment for "
                f"{controller_key}"
            )
    if controller.get("JobState") not in {"RUNNING", "COMPLETING"}:
        errors.append("current consumer Slurm job is not running")
    if controller.get("NumTasks") != "1" or controller.get("CPUs/Task") != "64":
        errors.append("current consumer Slurm CPU/task allocation mismatch")
    if controller.get("ReqNodeList") != release_runtime_contract.get("node"):
        errors.append("current consumer Slurm job was not pinned to the qualified node")

    expected_topology_file: Optional[Path] = None
    if release_task_root is not None and isinstance(job_id, str):
        if execution_mode == "consumer-benchmark":
            expected_topology_file = (
                release_task_root / "results" / "topology"
                / f"physical8-benchmark-{job_id}.json"
            )
            if consumer_topology_file != expected_topology_file:
                errors.append(
                    "current benchmark topology is not the fixed task/results job path"
                )
        else:
            if (
                consumer_topology_file is None
                or consumer_topology_file.name
                != f"topology-physical8-{job_id}.json"
                or not consumer_topology_file.is_relative_to(
                    release_task_root / "results"
                )
            ):
                errors.append(
                    "current counterpoise topology is not keyed by SLURM_JOB_ID "
                    "below the fixed task results root"
                )
            else:
                expected_topology_file = consumer_topology_file

    expected_environment = {
        "SLURM_JOB_NODELIST": release_runtime_contract.get("node"),
        "SLURM_JOB_PARTITION": _EXPECTED_RELEASE_PARTITION,
        "CCSD_PERFORMANCE_ELIGIBLE": "true",
        "CCSD_BENCHMARK_TRACK": "performance",
        "CCSD_TOPOLOGY_RECORD": (
            str(expected_topology_file)
            if expected_topology_file is not None else None
        ),
        "CCSD_TOPOLOGY_SHA256": consumer_topology_sha256,
        "CCSD_BOUND_CPUS": _EXPECTED_RELEASE_AFFINITY,
        "GPU4PYSCF_NUMA": str(_EXPECTED_RELEASE_NUMA_NODE),
        "CCSD_SCONTROL_PATH": (
            (release_runtime_contract.get("slurm_controller_helper") or {}).get(
                "configured_path"
            )
        ),
    }
    if release_runtime_contract.get("node") != _EXPECTED_RELEASE_NODE:
        errors.append("qualification node differs from fixed MTU node compute-1-6")
    if release_runtime_contract.get("host") != _EXPECTED_RELEASE_NODE:
        errors.append("qualification host differs from fixed MTU node compute-1-6")
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            errors.append(f"current consumer environment {name} mismatch")
    for name in (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "CUDA_VISIBLE_DEVICES",
    ):
        if environment.get(name) != consumer_slurm.get(name):
            errors.append(
                f"current consumer environment differs from topology for {name}"
            )

    if observation.get("host") != consumer_topology.get("host"):
        errors.append("current consumer topology host mismatch")
    if observation.get("host") != environment.get("SLURM_JOB_NODELIST"):
        errors.append("current hostname differs from the Slurm node allocation")
    if observation.get("host") != release_runtime_contract.get("host"):
        errors.append("current hostname differs from the qualified host")
    if consumer_topology.get("pid") != observation.get("pid"):
        errors.append("current consumer topology PID mismatch")

    affinity = observation.get("affinity")
    if affinity != _EXPECTED_RELEASE_AFFINITY:
        errors.append("current consumer CPU affinity is not 24-31")
    if affinity != (
        thread_affinity.get("union_affinity")
        if isinstance(thread_affinity, Mapping) else None
    ):
        errors.append("current consumer aggregate affinity is inconsistent")
    if consumer_topology.get("selected_cpulist") != affinity:
        errors.append("current consumer CPU affinity differs from topology")
    if consumer_topology.get("observed_cpulist_after_bind") != affinity:
        errors.append("current consumer post-bind CPU affinity mismatch")
    if release_runtime_contract.get("cpu_affinity") != affinity:
        errors.append("current consumer CPU affinity differs from the release pin")
    gpu_node_cpulist = observation.get("gpu_node_cpulist")
    try:
        if not set(_parse_cpu_list(affinity)).issubset(
            _parse_cpu_list(gpu_node_cpulist)
        ):
            errors.append("current consumer CPU affinity is outside GPU-local NUMA")
    except Exception:
        errors.append("current consumer GPU-node CPU list is invalid")
    for label, value in (
        ("topology", consumer_topology.get("gpu_node_cpulist")),
        ("qualification", qualification_fingerprint.get("gpu_node_cpulist")),
        ("release pin", release_runtime_contract.get("gpu_node_cpulist")),
    ):
        if value != gpu_node_cpulist:
            errors.append(f"current consumer GPU-node CPU list differs from {label}")

    if consumer_topology.get("mode") != "physical8":
        errors.append("current consumer topology mode is not physical8")
    if (
        consumer_topology.get("requested_physical_cores") != 8
        or consumer_topology.get("requested_threads") != 8
    ):
        errors.append("current consumer topology does not request eight physical cores")
    if consumer_topology.get("performance_eligible") is not True:
        errors.append("current consumer topology is not performance eligible")
    if consumer_topology.get("benchmark_track") != "performance":
        errors.append("current consumer topology benchmark track mismatch")
    if _topology_fingerprint(consumer_topology) != dict(qualification_fingerprint):
        errors.append("current consumer topology differs from qualification")

    if cuda.get("device_count") != 1:
        errors.append("current consumer CUDA runtime does not expose exactly one GPU")
    if cuda.get("current_device") != 0:
        errors.append("current consumer CUDA device is not visible-device index 0")
    visible_devices = environment.get("CUDA_VISIBLE_DEVICES")
    if not isinstance(visible_devices, str) or not visible_devices.strip():
        errors.append("current consumer CUDA_VISIBLE_DEVICES is unavailable")
    if cuda.get("visible_devices_environment") != visible_devices:
        errors.append("current consumer CUDA visibility observation is inconsistent")
    if qualification_slurm.get("CUDA_VISIBLE_DEVICES") != visible_devices:
        errors.append("current consumer CUDA visible device differs from qualification")

    observed_bus_id = _normalise_pci_bus_id(cuda.get("pci_bus_id"))
    observed_bus_ids = [observed_bus_id] if observed_bus_id else []
    for label, values in (
        ("topology", consumer_topology.get("gpu_pci_bus_ids") or []),
        ("qualification", qualification_fingerprint.get("gpu_pci_bus_ids") or []),
        ("release pin", release_runtime_contract.get("gpu_pci_bus_ids") or []),
    ):
        normalized = [_normalise_pci_bus_id(value) for value in values]
        if observed_bus_ids != normalized:
            errors.append(f"current consumer GPU PCI bus differs from {label}")
    if cuda.get("numa_node") != _EXPECTED_RELEASE_NUMA_NODE:
        errors.append("current consumer CUDA device is not attached to NUMA node 3")
    if consumer_topology.get("gpu_numa_node") != cuda.get("numa_node"):
        errors.append("current consumer GPU NUMA binding differs from topology")
    if release_runtime_contract.get("gpu_numa_node") != cuda.get("numa_node"):
        errors.append("current consumer GPU NUMA node differs from the release pin")

    mems_allowed = observation.get("mems_allowed_list")
    try:
        if _EXPECTED_RELEASE_NUMA_NODE not in _parse_cpu_list(mems_allowed):
            errors.append("current consumer Mems_allowed_list excludes NUMA node 3")
    except Exception:
        errors.append("current consumer Mems_allowed_list is invalid")
    if consumer_topology.get("mems_allowed_list") != mems_allowed:
        errors.append("current consumer Mems_allowed_list differs from topology")
    if release_runtime_contract.get("mems_allowed_list") != mems_allowed:
        errors.append("current consumer Mems_allowed_list differs from release pin")

    policy = observation.get("kernel_memory_policy")
    if not isinstance(policy, Mapping):
        errors.append("current consumer kernel memory policy is unavailable")
        policy = {}
    if (
        policy.get("mode_name") != "preferred"
        or policy.get("nodes") != [_EXPECTED_RELEASE_NUMA_NODE]
    ):
        errors.append("current consumer kernel memory policy is not preferred NUMA 3")
    topology_policy = consumer_topology.get("kernel_memory_policy") or {}
    topology_after = topology_policy.get("after") or {}
    if (
        topology_policy.get("verified") is not True
        or topology_after.get("mode_name") != policy.get("mode_name")
        or topology_after.get("nodes") != policy.get("nodes")
    ):
        errors.append("current consumer NUMA memory policy differs from topology")
    pinned_policy = release_runtime_contract.get("memory_policy") or {}
    if (
        pinned_policy.get("mode_name") != policy.get("mode_name")
        or pinned_policy.get("nodes") != policy.get("nodes")
    ):
        errors.append("current consumer memory policy differs from release pin")

    script_name = (
        "benchmark.py"
        if execution_mode == "consumer-benchmark" else "counterpoise.py"
    )
    launcher_name = (
        "run_mtu_benchmark.sbatch"
        if execution_mode == "consumer-benchmark"
        else "run_mtu_counterpoise.sbatch"
    )
    expected_script = (
        release_source_root / "benchmarks" / "cc" / "a100_water8" / script_name
        if release_source_root is not None else None
    )
    command_paths = [
        _normalised_loaded_path(item)
        for item in consumer_topology.get("command", [])
    ]
    if expected_script is None or str(expected_script) not in command_paths:
        errors.append("current consumer topology did not launch snapshot B driver")
    expected_launcher = (
        release_source_root / "benchmarks" / "cc" / "a100_water8" / launcher_name
        if release_source_root is not None else None
    )
    if (
        expected_launcher is None
        or _normalised_loaded_path(controller.get("Command"))
        != str(expected_launcher)
    ):
        errors.append("current Slurm allocation did not launch snapshot B consumer")
    return errors


def validate_runtime_performance_gate(
    receipt_path: Optional[os.PathLike[str] | str],
    *,
    basis_prod_cache_size_bytes: int,
    loaded_libgint_sha256: Optional[str] = None,
    loaded_libgint_path: Optional[os.PathLike[str] | str] = None,
    loaded_libgint_bytes: Optional[int] = None,
    selected_pair_abi: Optional[Mapping[str, Any]] = None,
    selected_c_abi: Optional[Mapping[str, Any]] = None,
    loaded_source_digest: Optional[str] = None,
    loaded_source_root: Optional[os.PathLike[str] | str] = None,
    loaded_manifest_path: Optional[os.PathLike[str] | str] = None,
    release_topology_path: Optional[os.PathLike[str] | str] = None,
    loaded_release_job_id: Optional[str] = None,
    loaded_release_context: Optional[Mapping[str, Any]] = None,
    execution_mode: str = "release-gate",
) -> dict[str, Any]:
    """Validate an MTU qualification receipt against release snapshot B.

    The selected-pair provider accepts only a filesystem path.  The receipt,
    its content-hash sidecar, and every evidence file named by the receipt are
    re-read on each provider construction.  The fixed release pin inside the
    loaded read-only source tree commits B to that exact receipt, while B with
    only the pin removed must hash to qualification snapshot A.  A structurally
    plausible Python mapping can therefore never unlock the performance path.
    """

    execution_mode = str(execution_mode).strip().lower()
    if execution_mode not in RUNTIME_GATE_EXECUTION_MODES:
        raise ValueError(
            "runtime gate execution_mode must be release-gate, "
            "consumer-benchmark, or consumer-counterpoise"
        )
    binding_contract = {
        "schema": "gpu4pyscf.gint-selected-runtime-gate-binding.v2",
        "caller_supplied_runtime_context_authoritative": False,
        "runtime_observation": "current-process-linux-cuda",
        "execution_mode": execution_mode,
        "required": [
            "read-only regular receipt plus content-SHA-256 sidecar",
            "terminal COMPLETED Slurm job and immutable output identity",
            "qualification result and topology identities",
            "qualification A and release B source/manifest identities",
            "fixed release pin binding B to the exact external receipt",
            "B without its pin reproduces A's source-tree SHA-256",
            "loaded libgint SHA-256, byte length, ABI size/offset/version",
            "loaded selected-column, diagonal, and workspace-size symbols",
            "complete transfer ledger with zero unresolved operations",
            (
                "current release Slurm, Linux affinity/NUMA policy, and CUDA "
                "device observed independently of topology JSON"
            ),
            (
                "controlled scontrol executable and libslurmfull path, hash, "
                "size, owner, exact mode, and portable inode"
            ),
            (
                "sealed control-tools/distribution/bin/lib directory identities "
                "within the trusted task-root-owner boundary"
            ),
            (
                "glibc loader proof that the query actually mapped the pinned "
                "$ORIGIN/../lib libslurmfull object"
            ),
            (
                "stable /proc/self/task affinity snapshot with every thread "
                "inside and the union covering CPUs 24-31"
            ),
            "mode-specific task/results topology bound to current Slurm job id",
        ],
        "loader_status": "implemented",
    }
    if receipt_path is None:
        return {
            "required": True,
            "payload_validated": False,
            "receipt_binding_validated": False,
            "validated": False,
            "status": "pending",
            "validation_errors": ["no MTU runtime gate receipt path was supplied"],
            "receipt": None,
            "binding_contract": binding_contract,
            "execution_mode": execution_mode,
        }
    if isinstance(receipt_path, Mapping):
        raise TypeError(
            "runtime gate evidence mappings are forbidden; supply a receipt path"
        )

    errors: list[str] = []
    current_release_runtime: Any = None
    try:
        current_release_runtime = _observe_current_release_runtime()
    except Exception as exc:
        errors.append(f"current release runtime observation failed: {exc}")
    receipt: Any = None
    receipt_file: Optional[Path] = None
    receipt_sha256: Optional[str] = None
    try:
        receipt_file, receipt_bytes, _ = _read_regular_file(
            receipt_path,
            name="runtime gate receipt",
            require_read_only=True,
            maximum_bytes=32 * 1024 * 1024,
        )
        receipt_sha256 = _sha256_bytes(receipt_bytes)
        receipt_directory = receipt_file.parent
        directory_stat = receipt_directory.lstat()
        if not stat.S_ISDIR(directory_stat.st_mode):
            errors.append("runtime gate receipt parent must be a directory")
        if directory_stat.st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            errors.append("runtime gate receipt directory must have no write bits")
        sidecar = Path(str(receipt_file) + RUNTIME_GATE_SIDECAR_SUFFIX)
        sidecar_path, sidecar_bytes, sidecar_stat = _read_regular_file(
            sidecar,
            name="runtime gate receipt SHA-256 sidecar",
            require_read_only=True,
            maximum_bytes=1024,
        )
        expected_sidecar = (
            f"{receipt_sha256}  {receipt_file.name}\n".encode("ascii")
        )
        if sidecar_bytes != expected_sidecar:
            errors.append("runtime gate receipt SHA-256 sidecar is invalid")
        if sidecar_path != Path(str(receipt_file) + RUNTIME_GATE_SIDECAR_SUFFIX):
            errors.append("runtime gate receipt sidecar path is inconsistent")
        directory_entries = {item.name for item in receipt_directory.iterdir()}
        if directory_entries != {receipt_file.name, sidecar.name}:
            errors.append(
                "runtime gate receipt directory must contain only receipt and sidecar"
            )
        receipt_stat = receipt_file.lstat()
        if not (
            receipt_stat.st_uid == sidecar_stat.st_uid == directory_stat.st_uid
            and receipt_stat.st_nlink == 1
            and sidecar_stat.st_nlink == 1
        ):
            errors.append("runtime gate receipt storage identity is invalid")
        receipt = _strict_json_loads(receipt_bytes, label="runtime gate receipt")
    except Exception as exc:
        errors.append(f"runtime gate receipt could not be loaded: {exc}")

    payload: Any = None
    if isinstance(receipt, Mapping):
        if receipt.get("schema") != RUNTIME_GATE_RECEIPT_SCHEMA:
            errors.append("runtime gate receipt schema mismatch")
        payload = receipt.get("payload")
        payload_digest = receipt.get("payload_sha256")
        if not isinstance(payload, Mapping):
            errors.append("runtime gate receipt payload must be an object")
        elif not _is_sha256(payload_digest):
            errors.append("runtime gate receipt payload SHA-256 is invalid")
        elif canonical_json_sha256(payload) != payload_digest:
            errors.append("runtime gate receipt payload SHA-256 mismatch")
    elif receipt is not None:
        errors.append("runtime gate receipt root must be an object")

    result_data: Any = None
    topology_data: Any = None
    source_manifest: Any = None
    release_manifest: Any = None
    release_pin: Any = None
    release_pin_sha256: Optional[str] = None
    qualification_source_root: Optional[Path] = None
    release_source_root: Optional[Path] = None
    release_task_root: Optional[Path] = None
    release_topology: Any = None
    release_topology_file: Optional[Path] = None
    release_topology_sha256: Optional[str] = None
    qualification_publication_attestation: Optional[dict[str, Any]] = None
    if isinstance(payload, Mapping):
        if payload.get("schema") != RUNTIME_GATE_PAYLOAD_SCHEMA:
            errors.append("runtime gate payload schema mismatch")
        for name, expected in {
            "status": "passed",
            "evidence_state": "local_measured",
            "site": "mtu",
            "case": "water2-tz",
        }.items():
            if payload.get(name) != expected:
                errors.append(f"runtime gate payload {name} must equal {expected!r}")
        truths = payload.get("truths")
        for name in (
            "cuda_parity_passed",
            "complete_transfer_audit_passed",
            "timing_gate_passed",
            "qualification_passed",
            "slurm_terminal_success",
        ):
            if not isinstance(truths, Mapping) or truths.get(name) is not True:
                errors.append(f"runtime gate truth {name} must be true")

        source = payload.get("source")
        if not isinstance(source, Mapping):
            errors.append("runtime gate source binding must be an object")
            source = {}
        if not _is_sha256(source.get("tree_sha256")):
            errors.append("runtime gate source tree SHA-256 is invalid")
        source_root_errors, qualification_source_root = _source_root_errors(
            source.get("root"),
            label="qualification source root",
            require_read_only=True,
        )
        errors.extend(source_root_errors)
        if (
            qualification_source_root is not None
            and source.get("root") != str(qualification_source_root)
        ):
            errors.append("qualification source root must be absolute and canonical")
        if qualification_source_root is not None:
            if qualification_source_root.parent.name != source.get("tree_sha256"):
                errors.append(
                    "qualification snapshot directory does not match its source digest"
                )
            if (qualification_source_root / RELEASE_PIN_RELATIVE_PATH).exists():
                errors.append("qualification snapshot A must not contain a release pin")
            try:
                qualification_digest, _ = release_source_tree_digest(
                    qualification_source_root
                )
                if qualification_digest != source.get("tree_sha256"):
                    errors.append(
                        "qualification source tree differs from the receipt digest"
                    )
            except Exception as exc:
                errors.append(f"qualification source tree could not be hashed: {exc}")
        manifest_binding = source.get("manifest")
        manifest_errors, manifest_path, manifest_bytes = _file_binding_errors(
            manifest_binding,
            label="qualification source snapshot manifest",
            require_read_only=True,
        )
        errors.extend(manifest_errors)
        if qualification_source_root is not None and manifest_path is not None:
            expected_snapshot_manifest = str(
                qualification_source_root.parent / "manifest.json"
            )
            if str(manifest_path) != expected_snapshot_manifest:
                errors.append(
                    "qualification manifest is outside snapshot A"
                )
        if manifest_bytes is not None:
            try:
                source_manifest = _strict_json_loads(
                    manifest_bytes, label="qualification source snapshot manifest"
                )
            except Exception as exc:
                errors.append(f"qualification source manifest is invalid: {exc}")
        if qualification_source_root is not None:
            errors.extend(_snapshot_manifest_checks(
                source_manifest,
                source_root=qualification_source_root,
                tree_sha256=source.get("tree_sha256"),
                label="qualification source snapshot manifest",
            ))
            attestation_errors, qualification_publication_attestation = (
                _publication_attestation_checks(
                    source_manifest, source_root=qualification_source_root
                )
            )
            errors.extend(attestation_errors)

        release_root_errors, release_source_root = _source_root_errors(
            loaded_source_root,
            label="loaded release source root",
            require_read_only=True,
        )
        errors.extend(release_root_errors)
        if not _is_sha256(loaded_source_digest):
            errors.append("the loaded release source digest was not supplied")
        if release_source_root is not None:
            if release_source_root.parent.name != loaded_source_digest:
                errors.append(
                    "release snapshot directory does not match its source digest"
                )
            try:
                release_digest, _ = release_source_tree_digest(
                    release_source_root
                )
                release_without_pin_digest, _ = release_source_tree_digest(
                    release_source_root, exclude_release_pin=True
                )
                if release_digest != loaded_source_digest:
                    errors.append(
                        "loaded release source digest differs from snapshot B"
                    )
                if release_without_pin_digest != source.get("tree_sha256"):
                    errors.append(
                        "release snapshot B without its pin does not reproduce "
                        "qualification snapshot A"
                    )
            except Exception as exc:
                errors.append(f"loaded release source tree could not be hashed: {exc}")

            expected_release_manifest = release_source_root.parent / "manifest.json"
            supplied_release_manifest = _normalised_loaded_path(
                loaded_manifest_path
            )
            if supplied_release_manifest != str(expected_release_manifest):
                errors.append("loaded release manifest is outside snapshot B")
            try:
                _, release_manifest_bytes, _ = _read_regular_file(
                    expected_release_manifest,
                    name="release source snapshot manifest",
                    require_read_only=True,
                )
                release_manifest = _strict_json_loads(
                    release_manifest_bytes,
                    label="release source snapshot manifest",
                )
            except Exception as exc:
                errors.append(f"release source manifest could not be loaded: {exc}")
            errors.extend(_snapshot_manifest_checks(
                release_manifest,
                source_root=release_source_root,
                tree_sha256=loaded_source_digest,
                label="release source snapshot manifest",
            ))
            if (
                isinstance(source_manifest, Mapping)
                and isinstance(release_manifest, Mapping)
            ):
                qualification_manifest_core = deepcopy(dict(source_manifest))
                release_manifest_core = deepcopy(dict(release_manifest))
                for value in (qualification_manifest_core, release_manifest_core):
                    value.pop("created_utc", None)
                    value.pop("tree_sha256", None)
                    value.pop("source", None)
                    value.pop("gint_release_lineage", None)
                if release_manifest_core != qualification_manifest_core:
                    errors.append(
                        "release snapshot B manifest provenance differs from A"
                    )

            release_pin_path = release_source_root / RELEASE_PIN_RELATIVE_PATH
            pin_candidates = [
                path for path in release_source_root.rglob(
                    RELEASE_PIN_RELATIVE_PATH.name
                ) if path.is_file() or path.is_symlink()
            ]
            if pin_candidates != [release_pin_path]:
                errors.append(
                    "release snapshot B must contain exactly one fixed-path release pin"
                )
            try:
                _, release_pin_bytes, _ = _read_regular_file(
                    release_pin_path,
                    name="release source pin",
                    require_read_only=True,
                    maximum_bytes=1024 * 1024,
                )
                release_pin_sha256 = _sha256_bytes(release_pin_bytes)
                release_pin = _strict_json_loads(
                    release_pin_bytes, label="release source pin"
                )
            except Exception as exc:
                errors.append(f"release source pin could not be loaded: {exc}")
            try:
                expected_pin = expected_runtime_release_pin(
                    payload,
                    receipt_sha256=str(receipt_sha256),
                    payload_sha256=str(payload_digest),
                )
                if release_pin != expected_pin:
                    errors.append(
                        "release pin does not bind the exact qualification receipt"
                    )
            except Exception as exc:
                errors.append(f"expected release pin could not be built: {exc}")
            if isinstance(release_manifest, Mapping):
                expected_lineage = {
                    "schema": RELEASE_PIN_SCHEMA,
                    "qualification_source_tree_sha256": source.get(
                        "tree_sha256"
                    ),
                    "release_pin_relative_path": (
                        RELEASE_PIN_RELATIVE_PATH.as_posix()
                    ),
                    "release_pin_sha256": release_pin_sha256,
                    "receipt_sha256": receipt_sha256,
                    "receipt_payload_sha256": payload_digest,
                }
                source_is_v3 = (
                    isinstance(source_manifest, Mapping)
                    and source_manifest.get("schema")
                    == "gpu4pyscf.source-snapshot.v3"
                )
                if source_is_v3:
                    if qualification_publication_attestation is None:
                        errors.append(
                            "v3 lineage lacks the qualification publication attestation"
                        )
                    else:
                        expected_lineage.update({
                            "qualification_publication_attestation": (
                                qualification_publication_attestation
                            ),
                            "qualification_publication_attestation_sha256": (
                                qualification_publication_attestation.get(
                                    "attestation_sha256"
                                )
                            ),
                        })
                if release_manifest.get("gint_release_lineage") != expected_lineage:
                    errors.append(
                        "release snapshot manifest does not bind A-to-B lineage"
                    )

            release_task_root = release_source_root.parent.parent.parent
            approved_receipt_root = release_task_root / "results"
            try:
                _reject_symlink_components(
                    approved_receipt_root, name="approved receipt root"
                )
            except Exception as exc:
                errors.append(f"approved receipt root is invalid: {exc}")
            if (
                receipt_file is None
                or not receipt_file.is_relative_to(approved_receipt_root)
            ):
                errors.append(
                    "runtime gate receipt is outside the approved task results root"
                )
            if (
                qualification_source_root is not None
                and qualification_source_root.parent.parent.parent
                != release_task_root
            ):
                errors.append(
                    "qualification snapshot A and release snapshot B use "
                    "different task roots"
                )

        qualification = payload.get("qualification")
        if not isinstance(qualification, Mapping):
            errors.append("runtime gate qualification binding must be an object")
            qualification = {}
        for label, key in (
            ("qualification result", "result"),
            ("qualification topology", "topology"),
            ("qualification Slurm output", "slurm_output"),
        ):
            file_errors, _, file_bytes = _file_binding_errors(
                qualification.get(key), label=label
            )
            errors.extend(file_errors)
            if key == "result" and file_bytes is not None:
                try:
                    result_data = _strict_json_loads(
                        file_bytes, label="qualification result"
                    )
                except Exception as exc:
                    errors.append(f"qualification result is invalid: {exc}")
            elif key == "topology" and file_bytes is not None:
                try:
                    topology_data = _strict_json_loads(
                        file_bytes, label="qualification topology"
                    )
                except Exception as exc:
                    errors.append(f"qualification topology is invalid: {exc}")

        slurm = qualification.get("slurm")
        if not isinstance(slurm, Mapping):
            errors.append("runtime gate Slurm binding must be an object")
            slurm = {}
        job_id = slurm.get("job_id")
        if not isinstance(job_id, str) or not job_id.isdigit() or int(job_id) < 1:
            errors.append("runtime gate Slurm job id is invalid")
        for name in ("job_name", "node", "partition"):
            if not isinstance(slurm.get(name), str) or not slurm.get(name):
                errors.append(f"runtime gate Slurm {name} is invalid")
        if slurm.get("state") != "COMPLETED" or slurm.get("exit_code") != "0:0":
            errors.append("runtime gate Slurm job did not terminate successfully")
        if slurm.get("job_name") != "gint-water2-gate":
            errors.append("runtime gate Slurm job name mismatch")
        if slurm.get("partition") != "mrigpu":
            errors.append("runtime gate Slurm partition mismatch")

        runtime = payload.get("runtime")
        if not isinstance(runtime, Mapping):
            errors.append("runtime gate library binding must be an object")
            runtime = {}
        lib_binding = runtime.get("libgint")
        lib_errors, lib_path, lib_bytes = _file_binding_errors(
            lib_binding,
            label="qualification libgint",
            require_read_only=True,
        )
        errors.extend(lib_errors)
        if qualification_source_root is not None and lib_path is not None:
            qualification_libgint = (
                qualification_source_root / "gpu4pyscf" / "lib" / "libgint.so"
            )
            if lib_path != qualification_libgint:
                errors.append(
                    "qualification libgint is outside snapshot A runtime"
                )

        current_lib_path = _normalised_loaded_path(loaded_libgint_path)
        if current_lib_path is None:
            errors.append("the loaded libgint path is unavailable")
        elif release_source_root is not None:
            expected_libgint = (
                release_source_root / "gpu4pyscf" / "lib" / "libgint.so"
            )
            if current_lib_path != str(expected_libgint):
                errors.append("loaded libgint is outside snapshot B runtime")

        loaded_lib_bytes: Optional[bytes] = None
        if current_lib_path is not None:
            try:
                _, loaded_lib_bytes, loaded_lib_stat = _read_regular_file(
                    current_lib_path,
                    name="loaded release libgint",
                    require_read_only=True,
                )
                if loaded_libgint_bytes is not None and (
                    loaded_lib_stat.st_size != int(loaded_libgint_bytes)
                ):
                    errors.append(
                        "loaded release libgint byte length changed during validation"
                    )
            except Exception as exc:
                errors.append(f"loaded release libgint could not be read: {exc}")
        if loaded_libgint_sha256 is None:
            errors.append("the loaded libgint SHA-256 is unavailable")
        elif not isinstance(lib_binding, Mapping) or (
            lib_binding.get("sha256") != loaded_libgint_sha256
        ):
            errors.append("runtime gate libgint digest differs from the loaded library")
        if loaded_libgint_bytes is None:
            errors.append("the loaded libgint byte length is unavailable")
        elif not isinstance(lib_binding, Mapping) or (
            lib_binding.get("bytes") != int(loaded_libgint_bytes)
        ):
            errors.append(
                "runtime gate libgint byte length differs from the loaded library"
            )
        if loaded_lib_bytes is not None and loaded_libgint_sha256 is not None:
            if _sha256_bytes(loaded_lib_bytes) != loaded_libgint_sha256:
                errors.append("loaded release libgint changed after ABI binding")
        if lib_bytes is not None and loaded_lib_bytes is not None:
            if _sha256_bytes(lib_bytes) != _sha256_bytes(loaded_lib_bytes):
                errors.append("release snapshot B libgint differs from qualification A")

        bpcache_abi = runtime.get("basis_prod_cache_abi")
        if not isinstance(bpcache_abi, Mapping):
            errors.append("runtime gate BasisProdCache ABI binding is missing")
        elif (
            bpcache_abi.get("verified") is not True
            or bpcache_abi.get("libgint_size_bytes")
            != int(basis_prod_cache_size_bytes)
            or bpcache_abi.get("python_ctypes_size_bytes")
            != int(basis_prod_cache_size_bytes)
        ):
            errors.append("runtime gate BasisProdCache ABI differs from the loaded ABI")
        receipt_selected_abi = runtime.get("selected_pair_abi")
        if not isinstance(selected_pair_abi, Mapping):
            errors.append("the loaded selected-pair ABI identity is unavailable")
        elif receipt_selected_abi != dict(selected_pair_abi):
            errors.append("runtime gate selected-pair ABI size/offset/version mismatch")
        receipt_selected_c_abi = runtime.get("selected_c_abi")
        if not isinstance(selected_c_abi, Mapping):
            errors.append("the loaded selected C ABI symbol identity is unavailable")
        elif receipt_selected_c_abi != dict(selected_c_abi):
            errors.append("runtime gate selected C ABI symbol identity mismatch")

        transfer = payload.get("transfer_audit")
        if not isinstance(transfer, Mapping):
            errors.append("runtime gate transfer audit binding must be an object")
            transfer = {}
        ledger = transfer.get("ledger")
        timed_ledger = transfer.get("timed_ledger")
        if not isinstance(ledger, Mapping):
            errors.append("runtime gate transfer ledger must be an object")
        else:
            if transfer.get("ledger_sha256") != canonical_json_sha256(ledger):
                errors.append("runtime gate transfer ledger digest mismatch")
            if ledger.get("complete_for_declared_payloads") is not True:
                errors.append("runtime gate setup transfer ledger is incomplete")
            if ledger.get("complete_for_vhfopt_internal_setup") is not True:
                errors.append("runtime gate VHFOpt transfer ledger is incomplete")
            if ledger.get("unresolved_operations") != []:
                errors.append("runtime gate transfer ledger has unresolved operations")
            if transfer.get("unresolved_count") != 0:
                errors.append("runtime gate unresolved transfer count is not zero")
        if not isinstance(timed_ledger, Mapping):
            errors.append("runtime gate timed transfer ledger must be an object")
        else:
            if transfer.get("timed_ledger_sha256") != canonical_json_sha256(
                timed_ledger
            ):
                errors.append("runtime gate timed transfer ledger digest mismatch")
            timed_by_kind = timed_ledger.get("by_kind") or {}
            timed_by_operation = timed_ledger.get("by_operation") or {}
            if (
                timed_ledger.get("total_bytes")
                != sum(
                    item.get("bytes", -1)
                    for item in timed_by_kind.values()
                    if isinstance(item, Mapping)
                )
                or timed_ledger.get("total_transfers")
                != sum(
                    item.get("count", -1)
                    for item in timed_by_kind.values()
                    if isinstance(item, Mapping)
                )
                or any(
                    not isinstance(item, Mapping)
                    or item.get("bytes") != sum(
                        operation.get("bytes", -1)
                        for operation in (
                            timed_by_operation.get(kind) or {}
                        ).values()
                        if isinstance(operation, Mapping)
                    )
                    or item.get("count") != sum(
                        operation.get("count", -1)
                        for operation in (
                            timed_by_operation.get(kind) or {}
                        ).values()
                        if isinstance(operation, Mapping)
                    )
                    for kind, item in timed_by_kind.items()
                )
            ):
                errors.append(
                    "runtime gate timed transfer ledger is not fully attributed"
                )

        if isinstance(result_data, Mapping):
            if result_data.get("schema") != GINT_GATE_RESULT_SCHEMA:
                errors.append("qualification result schema mismatch")
            result_source = result_data.get("source") or {}
            result_snapshot = result_source.get("snapshot") or {}
            result_provider = result_data.get("provider") or {}
            decision = result_data.get("gate_decision") or {}
            if result_data.get("status") != "completed":
                errors.append("qualification result did not complete")
            if decision.get("correctness_passed") is not True:
                errors.append("qualification correctness gate did not pass")
            if decision.get("qualification_passed") is not True:
                errors.append("qualification transfer gate did not pass")
            if result_source.get("stable_during_run") is not True:
                errors.append("qualification source was not stable")
            if result_source.get("tree_sha256_at_start") != source.get("tree_sha256"):
                errors.append("qualification result source digest mismatch")
            if result_snapshot.get("manifest_sha256_at_end") != (
                manifest_binding.get("sha256")
                if isinstance(manifest_binding, Mapping) else None
            ):
                errors.append("qualification result manifest digest mismatch")
            if result_provider.get("explicit_transfers") != ledger:
                errors.append("qualification result transfer ledger mismatch")
            if result_data.get("timed_transfer_counter") != timed_ledger:
                errors.append("qualification result timed transfer ledger mismatch")
            result_topology = result_data.get("topology") or {}
            topology_binding = qualification.get("topology") or {}
            if (
                result_topology.get("path") != topology_binding.get("path")
                or result_topology.get("sha256_at_end")
                != topology_binding.get("sha256")
            ):
                errors.append("qualification result topology binding mismatch")
            if (result_provider.get("c_abi") or {}).get(
                "basis_prod_cache_identity"
            ) != bpcache_abi:
                errors.append("qualification result BasisProdCache ABI mismatch")
            if (result_provider.get("c_abi") or {}).get(
                "selected_pair_identity"
            ) != receipt_selected_abi:
                errors.append("qualification result selected-pair ABI mismatch")
            result_c_abi = result_provider.get("c_abi") or {}
            if {
                key: result_c_abi.get(key)
                for key in ("columns", "diagonal", "workspace_size")
            } != receipt_selected_c_abi:
                errors.append("qualification result selected C ABI mismatch")
            if (result_data.get("case") or {}).get("id") != payload.get("case"):
                errors.append("qualification result case mismatch")

        if isinstance(topology_data, Mapping):
            topology_slurm = topology_data.get("slurm") or {}
            if topology_data.get("performance_eligible") is not True:
                errors.append("qualification topology was not performance eligible")
            for receipt_key, topology_key in (
                ("job_id", "SLURM_JOB_ID"),
                ("job_name", "SLURM_JOB_NAME"),
                ("node", "SLURM_JOB_NODELIST"),
                ("partition", "SLURM_JOB_PARTITION"),
            ):
                if slurm.get(receipt_key) != topology_slurm.get(topology_key):
                    errors.append(
                        f"qualification topology Slurm {receipt_key} mismatch"
                    )
            fingerprint = qualification.get("topology_fingerprint")
            expected_fingerprint = {
                "host": topology_data.get("host"),
                "mode": topology_data.get("mode"),
                "expected_gpu_numa_node": topology_data.get(
                    "expected_gpu_numa_node"
                ),
                "requested_physical_cores": topology_data.get(
                    "requested_physical_cores"
                ),
                "requested_threads": topology_data.get("requested_threads"),
                "gpu_numa_node": topology_data.get("gpu_numa_node"),
                "gpu_pci_bus_ids": topology_data.get("gpu_pci_bus_ids"),
                "gpu_node_cpulist": topology_data.get("gpu_node_cpulist"),
                "allowed_cpulist_before": topology_data.get(
                    "allowed_cpulist_before"
                ),
                "node_allowed_intersection": topology_data.get(
                    "node_allowed_intersection"
                ),
                "selected_cpulist": topology_data.get("selected_cpulist"),
                "observed_cpulist_after_bind": topology_data.get(
                    "observed_cpulist_after_bind"
                ),
                "mems_allowed_list": topology_data.get("mems_allowed_list"),
                "benchmark_track": topology_data.get("benchmark_track"),
            }
            if fingerprint != expected_fingerprint:
                errors.append("qualification topology fingerprint mismatch")

            try:
                release_topology_file, release_topology_bytes, _ = (
                    _read_regular_file(
                    release_topology_path,
                    name="current release topology",
                    require_read_only=True,
                    maximum_bytes=4 * 1024 * 1024,
                    )
                )
                release_topology_sha256 = _sha256_bytes(
                    release_topology_bytes
                )
                release_topology = _strict_json_loads(
                    release_topology_bytes, label="current release topology"
                )
            except Exception as exc:
                errors.append(f"current release topology could not be loaded: {exc}")
            if release_topology_file is not None:
                if (
                    release_task_root is None
                    or not release_topology_file.is_relative_to(
                        release_task_root / "results"
                    )
                ):
                    errors.append(
                        "current release topology is outside the task results root"
                    )
            if execution_mode == "release-gate":
                if isinstance(release_topology, Mapping):
                    release_slurm = release_topology.get("slurm") or {}
                    current_fingerprint = {
                        "host": release_topology.get("host"),
                        "mode": release_topology.get("mode"),
                        "expected_gpu_numa_node": release_topology.get(
                            "expected_gpu_numa_node"
                        ),
                        "requested_physical_cores": release_topology.get(
                            "requested_physical_cores"
                        ),
                        "requested_threads": release_topology.get(
                            "requested_threads"
                        ),
                        "gpu_numa_node": release_topology.get("gpu_numa_node"),
                        "gpu_pci_bus_ids": release_topology.get("gpu_pci_bus_ids"),
                        "gpu_node_cpulist": release_topology.get(
                            "gpu_node_cpulist"
                        ),
                        "allowed_cpulist_before": release_topology.get(
                            "allowed_cpulist_before"
                        ),
                        "node_allowed_intersection": release_topology.get(
                            "node_allowed_intersection"
                        ),
                        "selected_cpulist": release_topology.get("selected_cpulist"),
                        "observed_cpulist_after_bind": release_topology.get(
                            "observed_cpulist_after_bind"
                        ),
                        "mems_allowed_list": release_topology.get(
                            "mems_allowed_list"
                        ),
                        "benchmark_track": release_topology.get("benchmark_track"),
                    }
                    if current_fingerprint != fingerprint:
                        errors.append(
                            "current release topology differs from qualification"
                        )
                    if release_topology.get("performance_eligible") is not True:
                        errors.append(
                            "current release topology is not performance eligible"
                        )
                    current_policy = release_topology.get("kernel_memory_policy") or {}
                    current_after = current_policy.get("after") or {}
                    if (
                        current_policy.get("verified") is not True
                        or current_after.get("mode_name") != "preferred"
                        or current_after.get("nodes") != [3]
                    ):
                        errors.append("current release NUMA memory policy mismatch")
                    expected_release_name = (
                        f"gint-release-{payload_digest}"
                        if _is_sha256(payload_digest) else None
                    )
                    if release_slurm.get("SLURM_JOB_NAME") != expected_release_name:
                        errors.append(
                            "current release Slurm job name does not bind receipt payload"
                        )
                    if release_slurm.get("SLURM_JOB_PARTITION") != "mrigpu":
                        errors.append("current release Slurm partition mismatch")
                    if (
                        not isinstance(loaded_release_job_id, str)
                        or not loaded_release_job_id.isdigit()
                        or release_slurm.get("SLURM_JOB_ID")
                        != loaded_release_job_id
                    ):
                        errors.append(
                            "current process is not bound to release topology job"
                        )
                    expected_topology_name = (
                        f"topology-physical8-{loaded_release_job_id}.json"
                        if isinstance(loaded_release_job_id, str) else None
                    )
                    if (
                        release_topology_file is None
                        or release_topology_file.name != expected_topology_name
                    ):
                        errors.append("current release topology filename mismatch")
                    expected_gate_script = (
                        release_source_root
                        / "benchmarks" / "cc" / "a100_water8" / "gint_gate.py"
                        if release_source_root is not None else None
                    )
                    command_paths = [
                        _normalised_loaded_path(item)
                        for item in release_topology.get("command", [])
                    ]
                    if (
                        expected_gate_script is None
                        or str(expected_gate_script) not in command_paths
                    ):
                        errors.append(
                            "current release topology did not launch snapshot B gate"
                        )

                    if not isinstance(loaded_release_context, Mapping):
                        errors.append("current release process context is unavailable")
                    else:
                        context_environment = loaded_release_context.get(
                            "environment"
                        )
                        if not isinstance(context_environment, Mapping):
                            errors.append(
                                "current release process environment is unavailable"
                            )
                            context_environment = {}
                        if release_topology.get("pid") != loaded_release_context.get(
                            "pid"
                        ):
                            errors.append("current release topology PID mismatch")
                        if release_topology.get("host") != loaded_release_context.get(
                            "host"
                        ):
                            errors.append("current release topology host mismatch")
                        if release_topology.get("selected_cpulist") != (
                            loaded_release_context.get("affinity")
                        ):
                            errors.append("current release CPU affinity mismatch")
                        release_bus_ids = release_topology.get("gpu_pci_bus_ids")
                        if release_bus_ids != [
                            loaded_release_context.get("gpu_pci_bus_id")
                        ]:
                            errors.append("current release GPU PCI binding mismatch")
                        if release_topology.get("gpu_numa_node") != (
                            loaded_release_context.get("gpu_numa_node")
                        ):
                            errors.append("current release GPU NUMA binding mismatch")

                        for environment_key, record_key in (
                            ("SLURM_JOB_ID", "SLURM_JOB_ID"),
                            ("SLURM_JOB_NAME", "SLURM_JOB_NAME"),
                            ("SLURM_JOB_NODELIST", "SLURM_JOB_NODELIST"),
                            ("SLURM_JOB_PARTITION", "SLURM_JOB_PARTITION"),
                            ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_PER_TASK"),
                            ("CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"),
                        ):
                            if context_environment.get(environment_key) != (
                                release_slurm.get(record_key)
                            ):
                                errors.append(
                                    "current release environment differs from "
                                    f"topology for {environment_key}"
                                )
                        expected_environment = {
                            "CCSD_PERFORMANCE_ELIGIBLE": "true",
                            "CCSD_BENCHMARK_TRACK": "performance",
                            "CCSD_TOPOLOGY_RECORD": (
                                str(release_topology_file)
                                if release_topology_file is not None else None
                            ),
                            "CCSD_TOPOLOGY_SHA256": release_topology_sha256,
                            "CCSD_BOUND_CPUS": release_topology.get(
                                "selected_cpulist"
                            ),
                            "GPU4PYSCF_NUMA": str(
                                release_topology.get("gpu_numa_node")
                            ),
                        }
                        for name, expected in expected_environment.items():
                            if context_environment.get(name) != expected:
                                errors.append(
                                    f"current release environment {name} mismatch"
                                )
                elif release_topology is not None:
                    errors.append("current release topology root must be an object")

            pinned_runtime_contract = (
                release_pin.get("release_runtime_contract")
                if isinstance(release_pin, Mapping) else None
            )
            if execution_mode == "release-gate":
                errors.extend(_current_release_binding_errors(
                    current_release_runtime,
                    release_topology=release_topology,
                    qualification_topology=topology_data,
                    qualification_fingerprint=fingerprint,
                    release_topology_file=release_topology_file,
                    release_topology_sha256=release_topology_sha256,
                    release_task_root=release_task_root,
                    release_source_root=release_source_root,
                    release_runtime_contract=pinned_runtime_contract,
                    payload_sha256=payload_digest,
                    loaded_release_job_id=loaded_release_job_id,
                ))
            else:
                errors.extend(_current_consumer_binding_errors(
                    current_release_runtime,
                    execution_mode=execution_mode,
                    consumer_topology=release_topology,
                    qualification_topology=topology_data,
                    qualification_fingerprint=fingerprint,
                    consumer_topology_file=release_topology_file,
                    consumer_topology_sha256=release_topology_sha256,
                    release_task_root=release_task_root,
                    release_source_root=release_source_root,
                    release_runtime_contract=pinned_runtime_contract,
                    loaded_job_id=loaded_release_job_id,
                ))

        if isinstance(lib_binding, Mapping):
            for manifest_value, label in (
                (source_manifest, "qualification source manifest"),
                (release_manifest, "release source manifest"),
            ):
                if not isinstance(manifest_value, Mapping):
                    continue
                files = (
                    (manifest_value.get("runtime_binaries") or {}).get("files")
                )
                matches = [
                    item for item in files or []
                    if isinstance(item, Mapping)
                    and item.get("relative_path")
                    == "gpu4pyscf/lib/libgint.so"
                ]
                if len(matches) != 1:
                    errors.append(
                        f"{label} does not bind exactly one libgint.so"
                    )
                elif (
                    matches[0].get("sha256") != lib_binding.get("sha256")
                    or matches[0].get("bytes") != lib_binding.get("bytes")
                ):
                    errors.append(f"{label} libgint identity mismatch")
                elif (
                    label == "release source manifest"
                    and loaded_lib_bytes is not None
                    and (
                        matches[0].get("sha256")
                        != _sha256_bytes(loaded_lib_bytes)
                        or matches[0].get("bytes") != len(loaded_lib_bytes)
                    )
                ):
                    errors.append(f"{label} libgint identity mismatch")

    payload_validated = isinstance(payload, Mapping) and not errors
    summary = {
        "path": None if receipt_file is None else str(receipt_file),
        "sha256": receipt_sha256,
        "payload_sha256": (
            receipt.get("payload_sha256") if isinstance(receipt, Mapping) else None
        ),
        "job_id": (
            (((payload or {}).get("qualification") or {}).get("slurm") or {}).get(
                "job_id"
            ) if isinstance(payload, Mapping) else None
        ),
        "release_job_id": loaded_release_job_id,
        "release_topology_sha256": release_topology_sha256,
        "release_pin_sha256": release_pin_sha256,
        "qualification_source_sha256": (
            ((payload or {}).get("source") or {}).get("tree_sha256")
            if isinstance(payload, Mapping) else None
        ),
        "release_source_sha256": loaded_source_digest,
        "release_topology_path": (
            None if release_topology_file is None else str(release_topology_file)
        ),
        "execution_mode": execution_mode,
        "current_runtime": deepcopy(current_release_runtime),
        "current_runtime_sha256": (
            canonical_json_sha256(current_release_runtime)
            if isinstance(current_release_runtime, Mapping) else None
        ),
        "release_runtime_contract": (
            deepcopy(release_pin.get("release_runtime_contract"))
            if isinstance(release_pin, Mapping) else None
        ),
    }
    return {
        "required": True,
        "payload_validated": payload_validated,
        "receipt_binding_validated": payload_validated,
        "validated": payload_validated,
        "status": (
            (
                "release-accepted"
                if execution_mode == "release-gate"
                else "consumer-accepted"
            )
            if payload_validated else "rejected"
        ),
        "validation_errors": errors,
        "receipt": summary,
        "binding_contract": binding_contract,
        "execution_mode": execution_mode,
    }


__all__ = [
    "GINTSetupTransferAudit",
    "REQUIRED_DEVICE_PAYLOADS",
    "REQUIRED_SETUP_TRANSFERS",
    "RUNTIME_GATE_PAYLOAD_SCHEMA",
    "RUNTIME_GATE_RECEIPT_SCHEMA",
    "RUNTIME_GATE_SIDECAR_SUFFIX",
    "RELEASE_PIN_RELATIVE_PATH",
    "RELEASE_PIN_SCHEMA",
    "RELEASE_RUNTIME_CONTRACT_SCHEMA",
    "RELEASE_RUNTIME_OBSERVATION_SCHEMA",
    "SCONTROL_HELPER_SCHEMA",
    "SCONTROL_HELPER_TRUST_BOUNDARY",
    "SCONTROL_LOADER_EVIDENCE_SCHEMA",
    "THREAD_AFFINITY_OBSERVATION_SCHEMA",
    "RUNTIME_GATE_EXECUTION_MODES",
    "SOURCE_NORMALIZATION_POLICY",
    "SOURCE_NORMALIZATION_SCHEMA",
    "SOURCE_SNAPSHOT_SCHEMA",
    "canonical_json_sha256",
    "expected_runtime_release_pin",
    "observe_process_thread_affinity",
    "observe_scontrol_helper_identity",
    "release_source_tree_digest",
    "scontrol_helper_portable_binding",
    "scontrol_loader_portable_binding",
    "source_snapshot_normalization_errors",
    "validate_scontrol_helper_portable_binding",
    "validate_scontrol_loader_portable_binding",
    "validate_runtime_performance_gate",
]
