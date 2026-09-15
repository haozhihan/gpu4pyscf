#!/usr/bin/env python3
"""Build and seal the CUDA 12.8/sm_80 GINT runtime bundle.

The command is deliberately fail-closed.  It accepts only a normalized source
tree (regular files and directories), verifies the declared source identity,
builds ``libgint.so`` in an external workspace, and seals a content-addressed
bundle containing the 13-library runtime closure.  The default mode is a
read-only plan; ``--execute`` is required before cmake, filesystem writes, or
bundle assembly occur.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


TOOL_SCHEMA = "gpu4pyscf.gint-sm80-build.v1"
BUNDLE_SCHEMA = "gpu4pyscf.runtime-bundle.v2"
CUDA_VERSION = "12.8"
CUDA_ARCHITECTURE = "80-real"
BUILD_TYPE = "RelWithDebInfo"
CUDA_TOOL_PATHS = {
    "nvcc": "/usr/local/cuda-12.8/bin/nvcc",
    "cuobjdump": "/usr/local/cuda-12.8/bin/cuobjdump",
}
NVCC_PATH = Path(CUDA_TOOL_PATHS["nvcc"])
CUOBJDUMP_PATH = Path(CUDA_TOOL_PATHS["cuobjdump"])
ABI_VERSION = 2
EXPECTED_BASIS_PROD_SIZE = 144
EXPECTED_SELECTED_PAIR_SIZE = 112
EXPECTED_SELECTED_OFFSETS = (
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
)
EXPECTED_WORKSPACE_BYTES = {6: 0, 7: 600_514_560, 8: 1_264_896_000}
MAX_STACK_BYTES = 4096
REQUIRED_SYMBOLS = (
    "GINTfill_selected_int2e_columns",
    "GINTfill_selected_int2e_diagonal",
    "GINTselected_workspace_size",
    "GINTsizeof_basis_prod_cache",
    "GINTsizeof_selected_pair_data",
    "GINToffsetof_selected_pair_data",
    "GINTselected_pair_data_abi_version",
)
REQUIRED_LIBRARIES = (
    "libcupy_helper.so", "libgdft.so", "libgecp.so", "libgint.so",
    "libgvhf.so", "libgvhf_md.so", "libgvhf_rys.so", "libmgrid.so",
    "libmgrid_v2.so", "libmgrid_v3.so", "libpbc.so", "libsem.so",
    "libsolvent.so",
)


class BuildError(RuntimeError):
    """Raised when a build or artifact cannot satisfy the release contract."""


# This source is executed by a short-lived child process.  Keeping the
# ctypes handle in that process guarantees that the library is unmapped when
# the process exits, before the parent's TemporaryDirectory cleanup runs.
ABI_PROBE_SOURCE = r'''
import ctypes
import json
import sys


def main():
    path = sys.argv[1]
    offset_count = int(sys.argv[2])
    workspace_orders = [int(value) for value in json.loads(sys.argv[3])]
    handle = ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_LOCAL", 0))
    size_t = ctypes.c_size_t
    handle.GINTsizeof_basis_prod_cache.restype = size_t
    handle.GINTsizeof_selected_pair_data.restype = size_t
    handle.GINToffsetof_selected_pair_data.argtypes = [ctypes.c_int]
    handle.GINToffsetof_selected_pair_data.restype = size_t
    handle.GINTselected_pair_data_abi_version.restype = ctypes.c_int
    handle.GINTselected_workspace_size.argtypes = [ctypes.c_int, ctypes.c_int]
    handle.GINTselected_workspace_size.restype = size_t
    values = {
        "basis_prod_cache_size": int(handle.GINTsizeof_basis_prod_cache()),
        "selected_pair_data_size": int(handle.GINTsizeof_selected_pair_data()),
        "offsets": [
            int(handle.GINToffsetof_selected_pair_data(index))
            for index in range(offset_count)
        ],
        "abi_version": int(handle.GINTselected_pair_data_abi_version()),
        "workspace_bytes": {
            order: int(handle.GINTselected_workspace_size(order, 108))
            for order in workspace_orders
        },
    }
    print(json.dumps(values, sort_keys=True))


try:
    main()
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
    raise SystemExit(1)
'''


def canonical_json_bytes(value: Any, *, trailing_newline: bool = True) -> bytes:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return encoded + (b"\n" if trailing_newline else b"")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file_identity(path: Path, *, label: str) -> dict[str, Any]:
    """Hash one regular file with pre/open/post identity checks."""

    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise BuildError(f"{label} is not a regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise BuildError(f"{label} changed while it was opened: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        raise BuildError(f"{label} changed while it was read: {path}")
    return {"name": path.name, "bytes": int(after.st_size),
            "sha256": digest.hexdigest()}


def assert_regular_tree(root: Path) -> tuple[int, int]:
    """Reject links/special files and return (regular-file-count, dir-count)."""

    root = root.absolute()
    info = root.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise BuildError(f"source root must be a canonical directory: {root}")
    if root.resolve(strict=True) != root:
        raise BuildError(f"source root is not canonical: {root}")
    files = directories = 0
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        names.sort()
        filenames.sort()
        for name in names:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise BuildError(f"source contains non-directory entry: {path}")
            directories += 1
        for name in filenames:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise BuildError(f"source contains non-regular entry: {path}")
            files += 1
    return files, directories


def source_tree_digest(root: Path) -> tuple[str, int]:
    """Use the benchmark source identity, after requiring a link-free tree."""

    files, _ = assert_regular_tree(root)
    digest = hashlib.sha256()
    count = 0
    excluded = {".git", ".pytest_cache", "__pycache__", "build", "dist", "results"}
    suffixes = {
        ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".ini",
        ".json", ".md", ".py", ".pyx", ".pxd", ".sbatch", ".sh",
        ".toml", ".txt", ".yaml", ".yml",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        if path.suffix.lower() not in suffixes:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    if count == 0 or files == 0:
        raise BuildError(f"source has no digestible files: {root}")
    return digest.hexdigest(), count


def regular_file(path: Path, *, label: str) -> Path:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise BuildError(f"{label} is not a regular file: {path}")
    return path


def validate_base_bundle(base: Path) -> dict[str, dict[str, Any]]:
    """Return the exact 13-library base inventory, rejecting substitutions."""

    info = base.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise BuildError(f"base bundle must be a directory: {base}")
    if base.resolve(strict=True) != base:
        raise BuildError(f"base bundle is not canonical: {base}")
    actual = sorted(path.name for path in base.iterdir()
                    if path.name.endswith(".so"))
    if actual != sorted(REQUIRED_LIBRARIES):
        raise BuildError(f"base bundle inventory mismatch: {actual!r}")
    inventory: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_LIBRARIES:
        inventory[name] = regular_file_identity(
            base / name, label=f"base library {name}"
        )
    return inventory


def parse_nm_symbols(output: str) -> set[str]:
    symbols: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            symbols.add(fields[-1])
    return symbols


def validate_required_symbols(output: str) -> dict[str, Any]:
    symbols = parse_nm_symbols(output)
    missing = [name for name in REQUIRED_SYMBOLS if name not in symbols]
    if missing:
        raise BuildError(f"libgint is missing required exported symbols: {missing}")
    return {"required": list(REQUIRED_SYMBOLS), "missing": [],
            "present": [name for name in REQUIRED_SYMBOLS if name in symbols]}


def validate_abi_values(values: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "basis_prod_cache_size": EXPECTED_BASIS_PROD_SIZE,
        "selected_pair_data_size": EXPECTED_SELECTED_PAIR_SIZE,
        "offsets": list(EXPECTED_SELECTED_OFFSETS),
        "abi_version": ABI_VERSION,
        "workspace_bytes": {str(key): value
                            for key, value in EXPECTED_WORKSPACE_BYTES.items()},
    }
    workspace_values = values.get("workspace_bytes", {})
    actual = {
        "basis_prod_cache_size": values.get("basis_prod_cache_size"),
        "selected_pair_data_size": values.get("selected_pair_data_size"),
        "offsets": values.get("offsets"),
        "abi_version": values.get("abi_version"),
        "workspace_bytes": {
            str(key): workspace_values.get(
                key, workspace_values.get(str(key))
            )
            for key in EXPECTED_WORKSPACE_BYTES
        },
    }
    if actual != expected:
        raise BuildError(f"GINT ABI mismatch: expected {expected!r}, got {actual!r}")
    return {"expected": expected, "actual": actual, "verified": True}


def parse_architectures(output: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:sm|compute)_[0-9]+\b", output)))


def validate_sm80_cubin(output: str) -> dict[str, Any]:
    architectures = parse_architectures(output)
    sm = [arch for arch in architectures if arch.startswith("sm_")]
    if "sm_80" not in sm or any(arch != "sm_80" for arch in sm):
        raise BuildError(f"CUDA binary is not sm_80-only: {architectures!r}")
    if any(arch.startswith("compute_") and arch != "compute_80"
           for arch in architectures):
        raise BuildError(f"CUDA PTX target is not compute_80: {architectures!r}")
    return {"architectures": architectures, "sm_cubin": sm, "sm80_only": True}


def parse_ldd_missing(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines()
            if "not found" in line.lower()]


def validate_ldd(output: str, returncode: int = 0) -> dict[str, Any]:
    missing = parse_ldd_missing(output)
    if returncode != 0 or missing:
        raise BuildError(f"ldd reported missing dependencies: {missing!r}")
    return {"missing": [], "returncode": returncode, "verified": True}


def validate_nvcc_version(output: str) -> dict[str, Any]:
    """Require an actual CUDA 12.8 compiler banner, not just a path name."""

    return _validate_cuda_12_8_banner(output, tool="nvcc")


def _validate_cuda_12_8_banner(output: str, *, tool: str) -> dict[str, Any]:
    """Require an actual CUDA 12.8 tool banner, not just a path name."""

    match = re.search(r"\brelease\s+12\.8(?:\D|$)", output, re.IGNORECASE)
    if not match:
        raise BuildError(f"{tool} is not CUDA 12.8")
    return {"release": "12.8", "output": output, "verified": True}


def validate_cuobjdump_version(output: str) -> dict[str, Any]:
    """Require an actual CUDA 12.8 cuobjdump banner."""

    return _validate_cuda_12_8_banner(output, tool="cuobjdump")


def parse_stack_usage(output: str) -> dict[str, int]:
    """Extract stack frames for selected columns/diagonal Rys orders 7/8.

    cuobjdump versions differ in formatting.  We accept either a compact line
    (``columns root 7 ... stack frame: 3712``) or a function label followed by
    a stack-frame line, but never infer a missing order.
    """

    records: dict[str, int] = {}
    pending: list[tuple[str, int]] = []
    for line in output.splitlines():
        lower = line.lower()
        kind = "columns" if "column" in lower else "diagonal" if "diagonal" in lower else None
        order_matches = re.findall(
            r"(?:root|rys|order)[^0-9]*([78])\b|ili([78])", lower
        )
        # The alternation above returns a two-tuple.  Keep the non-empty
        # capture so raw cuobjdump names such as ``...kernelILi7...`` are
        # handled along with the human-readable form.
        orders = [int(first or second) for first, second in order_matches]
        if kind and orders:
            pending = [(kind, order) for order in orders]
        match = re.search(r"(?:stack(?:_frame| frame)?|local memory)[^0-9]*(\d+)\s*(?:bytes)?", lower)
        if not match:
            match = re.search(r"(\d+)\s*bytes[^\n]*(?:stack|local)", lower)
        if not match:
            continue
        size = int(match.group(1))
        targets = pending.copy()
        if kind and orders:
            targets = [(kind, order) for order in orders]
        for target_kind, order in targets:
            records[f"{target_kind}_rys{order}"] = size
        pending = []
    required = {f"{kind}_rys{order}"
                for kind in ("columns", "diagonal") for order in (7, 8)}
    missing = sorted(required - records.keys())
    if missing:
        raise BuildError(f"cuobjdump stack report lacks required kernels: {missing}")
    oversized = {key: value for key, value in records.items()
                 if key in required and value > MAX_STACK_BYTES}
    if oversized:
        raise BuildError(f"selected GINT stack frame exceeds limit: {oversized}")
    return {"records": {key: records[key] for key in sorted(required)},
            "max_bytes": max(records[key] for key in required),
            "limit_bytes": MAX_STACK_BYTES, "verified": True}


def _run(command: Sequence[str], *, cwd: Path | None = None,
         log: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    rendered = "$ " + " ".join(subprocess.list2cmdline([part]) for part in command)
    if log is not None:
        log.append(rendered)
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if log is not None:
        log.append(result.stdout)
    if result.returncode:
        raise BuildError(f"command failed ({result.returncode}): {rendered}\n{result.stdout}")
    return result


def _preflight_tools(*, log: list[str] | None = None) -> dict[str, Any]:
    """Check pinned CUDA tools before starting a potentially long build."""

    checks: dict[str, dict[str, Any]] = {}
    for name, raw_path in CUDA_TOOL_PATHS.items():
        path = Path(raw_path)
        try:
            regular_file(path, label=f"CUDA {name}")
        except (FileNotFoundError, OSError) as exc:
            raise BuildError(f"required CUDA tool is unavailable: {path}") from exc
        if not os.access(path, os.X_OK):
            raise BuildError(f"required CUDA tool is not executable: {path}")
        checks[name] = {"path": str(path), "regular_file": True,
                        "executable": True}

    nvcc_version = validate_nvcc_version(
        _run([str(NVCC_PATH), "--version"], log=log).stdout
    )
    cuobjdump_version = validate_cuobjdump_version(
        _run([str(CUOBJDUMP_PATH), "--version"], log=log).stdout
    )
    return {"paths": dict(CUDA_TOOL_PATHS), "checks": checks,
            "nvcc_version": nvcc_version,
            "cuobjdump_version": cuobjdump_version, "verified": True}


def _probe_abi_subprocess(libgint: Path, *, log: list[str] | None = None) -> dict[str, Any]:
    """Load libgint in a child process and return its ABI values as JSON."""

    command = [
        sys.executable,
        "-c",
        ABI_PROBE_SOURCE,
        str(libgint),
        str(len(EXPECTED_SELECTED_OFFSETS)),
        json.dumps(sorted(EXPECTED_WORKSPACE_BYTES)),
    ]
    result = _run(command, log=log)
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BuildError(
            f"ABI probe did not return JSON: {result.stdout!r}"
        ) from exc
    if not isinstance(values, dict) or "error" in values:
        raise BuildError(f"ABI probe returned invalid JSON: {values!r}")
    return values


def verify_library(libgint: Path, *, log: list[str] | None = None) -> dict[str, Any]:
    regular_file(libgint, label="built libgint")
    nm = _run(["nm", "-D", "--defined-only", str(libgint)], log=log)
    symbols = validate_required_symbols(nm.stdout)
    values = _probe_abi_subprocess(libgint, log=log)
    abi = validate_abi_values(values)
    cubin = validate_sm80_cubin(
        _run([str(CUOBJDUMP_PATH), "--list-elf", str(libgint)], log=log).stdout
    )
    resources = _run(
        [str(CUOBJDUMP_PATH), "--dump-resource-usage", str(libgint)], log=log
    ).stdout
    stack = parse_stack_usage(resources)
    ldd_result = _run(["ldd", str(libgint)], log=log)
    ldd = validate_ldd(ldd_result.stdout, ldd_result.returncode)
    return {"path": str(libgint), "sha256": sha256_file(libgint),
            "bytes": libgint.stat().st_size, "symbols": symbols,
            "abi": abi,
            "abi_probe": {"mode": "subprocess", "executable": sys.executable,
                          "verified": True},
            "cuobjdump_path": str(CUOBJDUMP_PATH), "cubin": cubin,
            "stack": stack, "ldd": ldd}


def library_inventory(bundle: Path) -> list[dict[str, Any]]:
    entries = []
    for name in REQUIRED_LIBRARIES:
        entries.append(regular_file_identity(
            bundle / name, label=f"bundle library {name}"
        ))
    return entries


def make_bundle_id(inventory: Sequence[Mapping[str, Any]]) -> str:
    """Return the established ID: canonical inventory list, no newline."""

    return sha256_bytes(canonical_json_bytes(inventory, trailing_newline=False))


def _readonly_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        os.chmod(path, mode & ~0o222)
    os.chmod(root, root.stat().st_mode & ~0o222)


def _make_writable_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        os.chmod(path, mode | (0o200 if path.is_file() else 0o700))
    os.chmod(root, root.stat().st_mode | 0o700)


def _write_once(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == data:
            return
        raise BuildError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError(f"short write while creating {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_and_verify_source(source: Path, destination: Path,
                            expected_sha: str, expected_count: int) -> None:
    if destination.exists() or destination.is_symlink():
        raise BuildError(f"staging source already exists: {destination}")
    shutil.copytree(source, destination, symlinks=False)
    actual_sha, actual_count = source_tree_digest(destination)
    if (actual_sha, actual_count) != (expected_sha, expected_count):
        raise BuildError("staged source changed while it was copied")


def _copy_and_verify_base(base: Path, destination: Path,
                          expected: Mapping[str, Mapping[str, Any]]) -> None:
    if destination.exists() or destination.is_symlink():
        raise BuildError(f"staging base bundle already exists: {destination}")
    destination.mkdir(parents=True)
    for name in REQUIRED_LIBRARIES:
        shutil.copy2(base / name, destination / name)
    actual = validate_base_bundle(destination)
    if actual != dict(expected):
        raise BuildError("staged base bundle inventory differs from source base")


def _assert_source_unchanged(source: Path, expected_sha: str,
                             expected_count: int) -> None:
    actual = source_tree_digest(source)
    if actual != (expected_sha, expected_count):
        raise BuildError(
            f"source changed during build: expected {expected_sha}/{expected_count}, "
            f"got {actual[0]}/{actual[1]}"
        )


def _assert_base_unchanged(base: Path,
                           expected: Mapping[str, Mapping[str, Any]]) -> None:
    actual = validate_base_bundle(base)
    if actual != dict(expected):
        raise BuildError("base bundle changed during build")


def _assert_bundle_files(bundle: Path) -> None:
    entries = list(bundle.iterdir())
    names = sorted(path.name for path in entries)
    if names != sorted(REQUIRED_LIBRARIES):
        raise BuildError(f"published bundle must contain exactly 13 libraries: {names}")
    for path in entries:
        regular_file(path, label="bundle member")


def _acquire_publish_lock(root: Path) -> Path:
    lock = root / ".gint-sm80-publish.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock, flags, 0o600)
    except FileExistsError as exc:
        raise BuildError(f"another GINT bundle publish is in progress: {lock}") from exc
    os.close(descriptor)
    return lock


def _publish_transaction(
    bundle_staging: Path,
    manifest_bytes: bytes,
    verification_bytes: bytes,
    log_bytes: bytes,
    output_task_root: Path,
    bundle_id: str,
    source_sha: str,
) -> tuple[Path, Path, Path]:
    """Publish bundle, sidecar, and build evidence with rollback on failure."""

    bundle_target = output_task_root / "runtime-bundles" / bundle_id
    manifest_target = output_task_root / "runtime-bundle-manifests" / f"{bundle_id}.json"
    build_target = output_task_root / "builds" / f"{source_sha}-gint-bounded-workspace-v2-sm80"
    for target in (bundle_target, manifest_target, build_target):
        if target.exists() or target.is_symlink():
            raise BuildError(f"refusing to overwrite existing artifact: {target}")

    output_task_root.mkdir(parents=True, exist_ok=True)
    lock = _acquire_publish_lock(output_task_root)
    transaction = Path(tempfile.mkdtemp(prefix=".gint-sm80-publish-",
                                         dir=str(output_task_root)))
    published: list[Path] = []
    try:
        bundle_stage = transaction / "bundle"
        manifest_stage = transaction / "manifest.json"
        build_stage = transaction / "build"
        shutil.copytree(bundle_staging, bundle_stage, symlinks=False)
        _assert_bundle_files(bundle_stage)
        _write_once(manifest_stage, manifest_bytes)
        build_stage.mkdir()
        _write_once(build_stage / "build-verification.json", verification_bytes)
        _write_once(build_stage / "build.log", log_bytes)
        bundle_target.parent.mkdir(parents=True, exist_ok=True)
        manifest_target.parent.mkdir(parents=True, exist_ok=True)
        build_target.parent.mkdir(parents=True, exist_ok=True)
        # The lock makes the preflight and three no-clobber renames one local
        # transaction.  Every destination was absent before staging began.
        for target in (bundle_target, manifest_target, build_target):
            if target.exists() or target.is_symlink():
                raise BuildError(f"publish destination appeared concurrently: {target}")
        os.rename(bundle_stage, bundle_target)
        published.append(bundle_target)
        _readonly_tree(bundle_target)
        os.rename(manifest_stage, manifest_target)
        published.append(manifest_target)
        os.chmod(manifest_target, manifest_target.stat().st_mode & ~0o222)
        os.rename(build_stage, build_target)
        published.append(build_target)
        _readonly_tree(build_target)
        return bundle_target, manifest_target, build_target
    except Exception:
        for path in reversed(published):
            if path.is_dir() and not path.is_symlink():
                _make_writable_tree(path)
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
        raise
    finally:
        shutil.rmtree(transaction, ignore_errors=True)
        lock.unlink(missing_ok=True)


def plan(source_root: Path, source_sha: str, source_count: int,
         base_bundle: Path, output_task_root: Path) -> dict[str, Any]:
    actual_sha, actual_count = source_tree_digest(source_root)
    if actual_sha != source_sha or actual_count != source_count:
        raise BuildError(
            f"source identity mismatch: declared {source_sha}/{source_count}, "
            f"actual {actual_sha}/{actual_count}"
        )
    base = validate_base_bundle(base_bundle)
    return {
        "schema": TOOL_SCHEMA, "execute": False,
        "source_root": str(source_root.absolute()), "source_sha256": actual_sha,
        "source_file_count": actual_count, "base_bundle": str(base_bundle.absolute()),
        "base_library_count": len(base), "output_task_root": str(output_task_root.absolute()),
        "build_type": BUILD_TYPE, "cuda_version": CUDA_VERSION,
        "cuda_architecture": CUDA_ARCHITECTURE, "required_symbols": list(REQUIRED_SYMBOLS),
        "tool_paths": dict(CUDA_TOOL_PATHS),
        "abi_version": ABI_VERSION, "workspace_bytes": {
            str(key): value for key, value in EXPECTED_WORKSPACE_BYTES.items()
        },
    }


def execute_build(source_root: Path, source_sha: str, source_count: int,
                  base_bundle: Path, output_task_root: Path,
                  *, build_jobs: int = 8) -> dict[str, Any]:
    preview = plan(source_root, source_sha, source_count, base_bundle, output_task_root)
    output_task_root.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    tool_preflight = _preflight_tools(log=logs)
    nvcc_version = tool_preflight["nvcc_version"]
    base_inventory = validate_base_bundle(base_bundle)
    with tempfile.TemporaryDirectory(prefix="gint-sm80-build-",
                                     dir=str(output_task_root)) as temporary:
        work = Path(temporary)
        work_source = work / "source"
        staged_base = work / "base-bundle"
        _copy_and_verify_source(source_root, work_source, source_sha, source_count)
        _copy_and_verify_base(base_bundle, staged_base, base_inventory)
        # The original inputs are rechecked after both private copies exist.
        # This catches a mutation during copy before any compiler process runs.
        _assert_source_unchanged(source_root, source_sha, source_count)
        _assert_base_unchanged(base_bundle, base_inventory)
        cmake_source = work_source / "gpu4pyscf" / "lib"
        work_build = work / "cmake"
        _run(["cmake", "-S", str(cmake_source), "-B", str(work_build),
              f"-DCMAKE_BUILD_TYPE={BUILD_TYPE}",
              f"-DCMAKE_CUDA_COMPILER={NVCC_PATH}",
              "-DCMAKE_Fortran_COMPILER=/usr/bin/gfortran",
              f"-DCUDA_ARCHITECTURES={CUDA_ARCHITECTURE}"], log=logs)
        _run(["cmake", "--build", str(work_build), "--target", "gint",
              "--parallel", str(build_jobs)], log=logs)
        # Check again immediately before consuming the compiler output.
        _assert_source_unchanged(source_root, source_sha, source_count)
        _assert_base_unchanged(base_bundle, base_inventory)
        built = cmake_source / "libgint.so"
        verification = verify_library(built, log=logs)
        staging = work / "bundle"
        staging.mkdir()
        for name in REQUIRED_LIBRARIES:
            origin = built if name == "libgint.so" else staged_base / name
            regular_file(origin, label=f"bundle input {name}")
            shutil.copy2(origin, staging / name)
        _assert_bundle_files(staging)
        inventory = library_inventory(staging)
        bundle_id = make_bundle_id(inventory)
        # Final input checks bind the exact immutable source/base used by this
        # build to the bundle identity immediately before publication.
        _assert_source_unchanged(source_root, source_sha, source_count)
        _assert_base_unchanged(base_bundle, base_inventory)
        if validate_base_bundle(staged_base) != base_inventory:
            raise BuildError("staged base bundle changed before publication")
        inventory_digest = sha256_bytes(
            canonical_json_bytes(inventory, trailing_newline=False)
        )
        bundle_target = output_task_root / "runtime-bundles" / bundle_id
        manifest_target = output_task_root / "runtime-bundle-manifests" / f"{bundle_id}.json"
        build_target = output_task_root / "builds" / f"{source_sha}-gint-bounded-workspace-v2-sm80"
        manifest = {
            "schema": BUNDLE_SCHEMA, "bundle_id": bundle_id,
            "identity_algorithm": "sha256(canonical-json-inventory-list-no-newline)",
            "inventory_sha256": inventory_digest,
            "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
            "bundle_root": str(bundle_target),
            "manifest_path": str(manifest_target),
            "source_sha256": source_sha, "source_file_count": source_count,
            "source_root": str(source_root.absolute()), "base_bundle": str(base_bundle.absolute()),
            "build": {"type": BUILD_TYPE, "cuda_version": CUDA_VERSION,
                      "cuda_architecture": CUDA_ARCHITECTURE,
                      "tool_paths": dict(CUDA_TOOL_PATHS),
                      "tool_preflight": tool_preflight,
                      "nvcc_version": nvcc_version, "verification": verification},
            "inventory": inventory,
        }
        verification_payload = {
            "schema": TOOL_SCHEMA, "source": preview, "nvcc": nvcc_version,
            "tool_paths": dict(CUDA_TOOL_PATHS), "tool_preflight": tool_preflight,
            "commands_log_sha256": sha256_bytes("\n".join(logs).encode()),
            "libgint": verification, "inventory": inventory,
            "bundle_id": bundle_id, "bundle_path": str(bundle_target),
            "manifest_path": str(manifest_target), "build_root": str(build_target),
        }
        _publish_transaction(
            staging,
            canonical_json_bytes(manifest),
            canonical_json_bytes(verification_payload),
            "\n".join(logs).encode(),
            output_task_root, bundle_id, source_sha,
        )
    return verification_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-file-count", required=True, type=int)
    parser.add_argument("--base-bundle", required=True, type=Path)
    parser.add_argument("--output-task-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--build-jobs", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.build_jobs < 1:
        raise SystemExit("--build-jobs must be positive")
    if args.execute:
        result = execute_build(args.source_root, args.source_sha256,
                               args.source_file_count, args.base_bundle,
                               args.output_task_root, build_jobs=args.build_jobs)
    else:
        result = plan(args.source_root, args.source_sha256, args.source_file_count,
                      args.base_bundle, args.output_task_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"build_gint_sm80_bundle: {exc}", file=sys.stderr)
        raise SystemExit(2)
