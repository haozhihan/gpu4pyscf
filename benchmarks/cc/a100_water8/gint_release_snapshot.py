#!/usr/bin/env python3
"""Promote qualified snapshot A into receipt-pinned release snapshot B."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

from gpu4pyscf.cc.gint_transfer_audit import (
    RELEASE_PIN_RELATIVE_PATH,
    RELEASE_PIN_SCHEMA,
    release_source_tree_digest,
    source_snapshot_normalization_errors,
)


WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
PUBLICATION_ATTESTATION_SUFFIX = ".publication-attestation.json"
DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _snapshot_manifest_module():
    path = Path(__file__).with_name("snapshot_manifest.py")
    spec = importlib.util.spec_from_file_location(
        "water8_release_snapshot_manifest", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load snapshot manifest validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot_assembly_module():
    path = Path(__file__).with_name("snapshot_remote_assembly.py")
    spec = importlib.util.spec_from_file_location(
        "water8_release_snapshot_assembly", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load snapshot publication helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_path(value: str | os.PathLike[str], *, name: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{name} path traverses symbolic link {current}")
    if path.resolve(strict=True) != path:
        raise ValueError(f"{name} path is not canonical")
    return path


def _read_regular_json(
    value: str | os.PathLike[str], *, name: str, require_read_only: bool = True
) -> tuple[Path, dict[str, Any], bytes]:
    path = _canonical_path(value, name=name)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular file")
    if require_read_only and info.st_mode & WRITE_BITS:
        raise ValueError(f"{name} must be read-only")
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

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{name} contains non-finite JSON number {value}")

    loaded = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(loaded, dict):
        raise ValueError(f"{name} root must be an object")
    return path, loaded, data


def _open_anchored_directory(path: Path, *, name: str) -> tuple[Path, int, os.stat_result]:
    """Open a canonical directory while checking every ancestor by descriptor."""

    path = _canonical_path(path, name=name)
    if path == Path(path.anchor):
        raise ValueError(f"{name} cannot be the filesystem root")
    descriptor = os.open(Path(path.anchor), DIRECTORY_FLAGS | NOFOLLOW)
    private_anchor = False
    try:
        for component in path.parts[1:]:
            before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"{name} component is not a directory")
            if (
                not private_anchor
                and before.st_uid == os.geteuid()
                and not (before.st_mode & (stat.S_IRWXG | stat.S_IRWXO))
            ):
                private_anchor = True
            child = os.open(component, DIRECTORY_FLAGS | NOFOLLOW, dir_fd=descriptor)
            opened = os.fstat(child)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                os.close(child)
                raise ValueError(f"{name} component changed while opened")
            os.close(descriptor)
            descriptor = child
        if not private_anchor:
            raise ValueError(f"{name} has no deployment-user-owned private anchor")
        return path, descriptor, os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory(parent_fd: int, name: str, *, expected: tuple[int, int], label: str) -> int:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{label} is not a directory")
    descriptor = os.open(name, DIRECTORY_FLAGS | NOFOLLOW, dir_fd=parent_fd)
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino) or (opened.st_dev, opened.st_ino) != expected:
        os.close(descriptor)
        raise ValueError(f"{label} identity changed")
    return descriptor


def _read_child_regular(parent_fd: int, name: str, *, label: str, maximum_bytes: int = 8 * 1024 * 1024) -> tuple[bytes, os.stat_result]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if before.st_mode & WRITE_BITS:
        raise ValueError(f"{label} must be read-only")
    descriptor = os.open(name, os.O_RDONLY | NOFOLLOW, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} identity changed while opened")
        data = bytearray()
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > maximum_bytes:
                raise ValueError(f"{label} is too large")
    finally:
        os.close(descriptor)
    after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
        raise ValueError(f"{label} changed while read")
    return bytes(data), after


def _validate_release_pin_shape(pin: Mapping[str, Any]) -> None:
    """Reject schema extensions before any release tree is created."""

    def exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(f"release pin {label} keys are invalid")
        return value

    top = {
        "schema", "status", "release_pin_path", "lineage", "receipt",
        "qualification_source", "qualification", "release_runtime_contract",
        "runtime", "transfer_audit_sha256",
    }
    if set(pin) != top:
        raise ValueError("release pin keys are invalid")
    if pin["schema"] != RELEASE_PIN_SCHEMA or pin["status"] != "release-pinned":
        raise ValueError("release pin schema or status is invalid")
    if pin["release_pin_path"] != RELEASE_PIN_RELATIVE_PATH.as_posix():
        raise ValueError("release pin path is invalid")
    lineage = exact(pin["lineage"], {
        "model", "qualification_source_tree_sha256",
        "release_source_without_pin_sha256",
    }, "lineage")
    if lineage["model"] != "qualification-source-plus-fixed-release-pin":
        raise ValueError("release pin lineage model is invalid")
    exact(pin["receipt"], {"sha256", "payload_sha256"}, "receipt")
    exact(pin["qualification_source"], {
        "root", "tree_sha256", "manifest_sha256", "manifest_bytes",
        "binding_sha256",
    }, "qualification_source")
    exact(pin["qualification"], {
        "job_id", "topology_fingerprint", "topology_fingerprint_sha256",
        "binding_sha256",
    }, "qualification")
    exact(pin["release_runtime_contract"], {
        "schema", "observation_source", "job_name", "node", "host",
        "partition", "cpu_affinity", "gpu_pci_bus_ids", "gpu_node_cpulist",
        "gpu_numa_node", "mems_allowed_list", "memory_policy",
        "topology_path_template", "qualification_topology_fingerprint_sha256",
    }, "release_runtime_contract")
    exact(pin["runtime"], {
        "libgint_sha256", "libgint_bytes", "basis_prod_cache_abi_sha256",
        "selected_pair_abi_sha256", "selected_c_abi_sha256", "binding_sha256",
    }, "runtime")


def _mkdtemp_at(parent_fd: int, parent_path: Path) -> Path:
    """Create staging below a held directory fd, never through a raced path."""

    for candidate in tempfile._get_candidate_names():
        name = ".gint-release-" + candidate
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        return parent_path / name
    raise FileExistsError("could not allocate release staging directory")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seal_snapshot(snapshot: Path) -> None:
    source = snapshot / "source"
    manifest = snapshot / "manifest.json"
    for directory, directories, files in os.walk(source):
        for name in files:
            os.chmod(Path(directory) / name, 0o444)
        for name in directories:
            os.chmod(Path(directory) / name, 0o555)
    os.chmod(source, 0o555)
    os.chmod(manifest, 0o444)
    os.chmod(snapshot, 0o555)


def _remove_staging(path: Path) -> None:
    if not path.exists() or path.is_symlink():
        return
    for directory, directories, files in os.walk(path):
        for name in files:
            candidate = Path(directory) / name
            if not candidate.is_symlink():
                candidate.chmod(0o600)
        for name in directories:
            candidate = Path(directory) / name
            if not candidate.is_symlink():
                candidate.chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)


def promote_release_snapshot(
    qualification_source: str | os.PathLike[str],
    qualification_manifest: str | os.PathLike[str],
    release_pin: str | os.PathLike[str],
    *,
    task_root: str | os.PathLike[str],
    post_publish_hook: Any = None,
) -> dict[str, Any]:
    """Copy A plus its fixed pin into a new content-addressed, read-only B."""

    task = _canonical_path(task_root, name="task root")
    if not task.is_dir():
        raise ValueError("task root must be a directory")
    snapshots = _canonical_path(task / "snapshots", name="snapshots root")
    source = _canonical_path(
        qualification_source, name="qualification source"
    )
    if (
        source.name != "source"
        or source.parent.parent != snapshots
        or source.parent.name == ""
    ):
        raise ValueError("qualification source is outside task snapshots")
    if (source / RELEASE_PIN_RELATIVE_PATH).exists():
        raise ValueError("qualification snapshot A already contains a release pin")
    for path in (source, *source.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"qualification snapshot A contains symlink {path}")
        if info.st_mode & WRITE_BITS:
            raise ValueError(f"qualification snapshot A contains writable path {path}")

    manifest_path, manifest, manifest_bytes = _read_regular_json(
        qualification_manifest, name="qualification manifest"
    )
    if manifest_path != source.parent / "manifest.json":
        raise ValueError("qualification manifest is outside snapshot A")
    qualification_digest, _ = release_source_tree_digest(source)
    snapshot_manifest = _snapshot_manifest_module()
    if manifest.get("publication_policy") != snapshot_manifest.PUBLICATION_POLICY:
        raise ValueError("qualification publication policy is invalid")
    qualification_attestation, attestation_errors = (
        snapshot_manifest._read_publication_attestation(
            source.parent,
            expected_digest=qualification_digest,
            manifest_bytes=manifest_bytes,
            policy=manifest.get("publication_policy"),
        )
    )
    if attestation_errors or qualification_attestation is None:
        raise ValueError(
            "qualification publication attestation is invalid: "
            + "; ".join(attestation_errors)
        )
    normalization_errors = source_snapshot_normalization_errors(
        manifest, source_root=source
    )
    if (
        source.parent.name != qualification_digest
        or normalization_errors
        or manifest.get("tree_sha256") != qualification_digest
        or manifest.get("source") != str(source)
        or manifest.get("immutable") is not True
    ):
        details = (
            "" if not normalization_errors
            else ": " + "; ".join(normalization_errors)
        )
        raise ValueError("qualification snapshot A identity is invalid" + details)

    pin_path, pin, pin_bytes = _read_regular_json(
        release_pin, name="release pin"
    )
    _validate_release_pin_shape(pin)
    if pin.get("schema") != RELEASE_PIN_SCHEMA:
        raise ValueError("release pin schema mismatch")
    lineage = pin.get("lineage") or {}
    if (
        pin.get("status") != "release-pinned"
        or pin.get("release_pin_path") != RELEASE_PIN_RELATIVE_PATH.as_posix()
        or lineage.get("model")
        != "qualification-source-plus-fixed-release-pin"
        or lineage.get("qualification_source_tree_sha256")
        != qualification_digest
        or lineage.get("release_source_without_pin_sha256")
        != qualification_digest
        or (pin.get("qualification_source") or {}).get("tree_sha256")
        != qualification_digest
        or (pin.get("qualification_source") or {}).get("root") != str(source)
    ):
        raise ValueError("release pin does not identify qualification snapshot A")
    qualification_source_binding = pin.get("qualification_source") or {}
    if (
        qualification_source_binding.get("manifest_sha256")
        != _sha256(manifest_bytes)
        or qualification_source_binding.get("manifest_bytes")
        != len(manifest_bytes)
    ):
        raise ValueError("release pin does not bind qualification manifest")
    runtime_binding = pin.get("runtime") or {}
    libgint = source / "gpu4pyscf" / "lib" / "libgint.so"
    libgint_info = libgint.lstat()
    if (
        not stat.S_ISREG(libgint_info.st_mode)
        or _sha256(libgint.read_bytes()) != runtime_binding.get("libgint_sha256")
        or libgint_info.st_size != runtime_binding.get("libgint_bytes")
    ):
        raise ValueError("release pin does not bind qualification libgint")
    receipt_binding = pin.get("receipt") or {}
    if not all(
        isinstance(receipt_binding.get(name), str)
        and len(receipt_binding[name]) == 64
        and all(
            character in "0123456789abcdef"
            for character in receipt_binding[name]
        )
        for name in ("sha256", "payload_sha256")
    ):
        raise ValueError("release pin receipt identity is invalid")

    _, snapshots_fd, snapshots_info = _open_anchored_directory(
        snapshots, name="snapshots root"
    )
    snapshots_identity = (snapshots_info.st_dev, snapshots_info.st_ino)
    staging: Path | None = _mkdtemp_at(snapshots_fd, snapshots)
    publication_protocol: str | None = None
    staging_fd = target_fd = source_fd = -1
    attestation_path: Path | None = None
    attestation_bytes = b""
    release_attestation: dict[str, Any] = {}
    try:
        assert staging is not None
        release_source = staging / "source"
        shutil.copytree(source, release_source)
        embedded_pin = release_source / RELEASE_PIN_RELATIVE_PATH
        embedded_pin.parent.chmod(0o755)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(embedded_pin, flags, 0o444)
        try:
            view = memoryview(pin_bytes)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("release pin write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)

        release_digest, _ = release_source_tree_digest(release_source)
        without_pin_digest, _ = release_source_tree_digest(
            release_source, exclude_release_pin=True
        )
        if without_pin_digest != qualification_digest:
            raise RuntimeError("release snapshot B without pin differs from A")
        target = snapshots / release_digest

        release_manifest = deepcopy(manifest)
        release_manifest.update({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tree_sha256": release_digest,
            "source": str(target / "source"),
            "gint_release_lineage": {
                "schema": RELEASE_PIN_SCHEMA,
                "qualification_source_tree_sha256": qualification_digest,
                "release_pin_relative_path": RELEASE_PIN_RELATIVE_PATH.as_posix(),
                "release_pin_sha256": _sha256(pin_bytes),
                "receipt_sha256": (pin.get("receipt") or {}).get("sha256"),
                "receipt_payload_sha256": (
                    pin.get("receipt") or {}
                ).get("payload_sha256"),
                "qualification_publication_attestation": deepcopy(
                    qualification_attestation
                ),
                "qualification_publication_attestation_sha256": (
                    qualification_attestation["attestation_sha256"]
                ),
            },
        })
        (staging / "manifest.json").write_text(
            json.dumps(
                release_manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        _seal_snapshot(staging)
        sealed_digest, _ = release_source_tree_digest(release_source)
        if sealed_digest != release_digest:
            raise RuntimeError("sealed release snapshot B digest changed")
        assembly = _snapshot_assembly_module()
        staging_fd = os.open(
            staging.name, DIRECTORY_FLAGS | NOFOLLOW, dir_fd=snapshots_fd
        )
        staging_info = os.fstat(staging_fd)
        publication_protocol = assembly._publish_directory_noreplace(
            snapshots_fd,
            staging_fd,
            staging.name,
            target.name,
            (staging_info.st_dev, staging_info.st_ino),
        )
        staging = None
        if post_publish_hook is not None:
            post_publish_hook()
        current_snapshots = os.fstat(snapshots_fd)
        if (current_snapshots.st_dev, current_snapshots.st_ino) != snapshots_identity:
            raise RuntimeError("held snapshots-root identity changed after publication")
        target_stat = os.stat(
            target.name, dir_fd=snapshots_fd, follow_symlinks=False
        )
        target_fd = _open_child_directory(
            snapshots_fd, target.name,
            expected=(target_stat.st_dev, target_stat.st_ino),
            label="published release snapshot",
        )
        target_info = os.fstat(target_fd)
        release_manifest_bytes, _ = _read_child_regular(
            target_fd, "manifest.json", label="published release manifest"
        )
        source_stat = os.stat("source", dir_fd=target_fd, follow_symlinks=False)
        source_fd = _open_child_directory(
            target_fd, "source",
            expected=(source_stat.st_dev, source_stat.st_ino),
            label="published release source",
        )
        release_attestation = {
            "schema": snapshot_manifest.PUBLICATION_ATTESTATION_SCHEMA,
            "status": "published",
            "source_tree_sha256": release_digest,
            "snapshot_name": release_digest,
            "published_directory_identity": {
                "device": target_info.st_dev,
                "inode": target_info.st_ino,
            },
            "publish_protocol": publication_protocol,
            "trust_boundary": snapshot_manifest.PUBLISH_TRUST_BOUNDARY,
            "publication_policy_sha256": snapshot_manifest._canonical_json_sha256(
                snapshot_manifest.PUBLICATION_POLICY
            ),
            "published_manifest_sha256": _sha256(release_manifest_bytes),
            "prepublication_manifest_sha256": _sha256(release_manifest_bytes),
        }
        release_attestation["attestation_sha256"] = (
            snapshot_manifest._canonical_json_sha256(release_attestation)
        )
        attestation_name = f"{release_digest}{PUBLICATION_ATTESTATION_SUFFIX}"
        attestation_path = snapshots / attestation_name
        attestation_bytes = (
            json.dumps(
                release_attestation,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
        attestation_fd = os.open(
            attestation_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW,
            0o600,
            dir_fd=snapshots_fd,
        )
        try:
            view = memoryview(attestation_bytes)
            while view:
                written = os.write(attestation_fd, view)
                if written < 1:
                    raise OSError("publication attestation write made no progress")
                view = view[written:]
            os.fsync(attestation_fd)
            os.fchmod(attestation_fd, 0o444)
        finally:
            os.close(attestation_fd)
        release_evidence = snapshot_manifest.validate_snapshot(
            target / "source", expected_task_root=task
        )
        if not release_evidence["valid"]:
            raise RuntimeError(
                "published release snapshot B failed strict-v3 validation: "
                + "; ".join(release_evidence["reasons"])
            )
        # The strict validator uses the public spelling for its full evidence
        # walk; the held descriptors prove that walk still names the exact
        # tree which was published, even if an ancestor was raced meanwhile.
        if (os.fstat(target_fd).st_dev, os.fstat(target_fd).st_ino) != (
            target_info.st_dev, target_info.st_ino
        ) or (os.fstat(source_fd).st_dev, os.fstat(source_fd).st_ino) != (
            source_stat.st_dev, source_stat.st_ino
        ):
            raise RuntimeError("published release target changed during validation")
        # Re-open the public path after all reads and writes.  This catches an
        # ancestor swap while the held descriptors keep the published tree
        # recoverable and prevent a replacement tree from being written.
        _, check_fd, check_info = _open_anchored_directory(
            snapshots, name="snapshots root after publication"
        )
        try:
            if (check_info.st_dev, check_info.st_ino) != snapshots_identity:
                raise RuntimeError("snapshots-root ancestor changed after publication")
        finally:
            os.close(check_fd)
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if target_fd >= 0:
            os.close(target_fd)
        if staging_fd >= 0:
            os.close(staging_fd)
        if snapshots_fd >= 0:
            os.close(snapshots_fd)
        if staging is not None:
            _remove_staging(staging)

    return {
        "schema": "gpu4pyscf.gint-release-snapshot-promotion.v1",
        "status": "created",
        "qualification_source": str(source),
        "qualification_source_sha256": qualification_digest,
        "release_source": str(target / "source"),
        "release_source_sha256": release_digest,
        "release_manifest": str(target / "manifest.json"),
        "publication_protocol": publication_protocol,
        "publication_attestation": str(attestation_path),
        "publication_attestation_sha256": release_attestation["attestation_sha256"],
        "publication_attestation_file_sha256": _sha256(attestation_bytes),
        "release_pin_source": str(pin_path),
        "release_pin_sha256": _sha256(pin_bytes),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-source", required=True, type=Path)
    parser.add_argument("--qualification-manifest", required=True, type=Path)
    parser.add_argument("--release-pin", required=True, type=Path)
    parser.add_argument("--task-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = promote_release_snapshot(
        args.qualification_source,
        args.qualification_manifest,
        args.release_pin,
        task_root=args.task_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"GINT release snapshot promotion failed: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
