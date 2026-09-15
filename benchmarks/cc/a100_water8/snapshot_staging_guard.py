#!/usr/bin/env python3
"""Create, verify, and remove remote snapshot staging directories safely.

This file is streamed to the remote Python interpreter by
``deploy_mtu_snapshot.sh``.  All operations are anchored to open directory
descriptors and reject symbolic links in the absolute snapshots-root path.
The cleanup operation only removes the direct child whose device/inode pair
was returned by ``create``; links found *inside* that child are unlinked rather
than followed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import NoReturn


GUARD_SCHEMA = "gpu4pyscf.remote-snapshot-staging-guard.v1"
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OTHER_PRINCIPAL_WRITE_BITS = stat.S_IWGRP | stat.S_IWOTH
_OTHER_PRINCIPAL_ACCESS_BITS = stat.S_IRWXG | stat.S_IRWXO


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _directory_identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _parse_identity(device: str | int, inode: str | int) -> tuple[int, int]:
    try:
        result = int(device), int(inode)
    except (TypeError, ValueError) as exc:
        raise ValueError("directory identity must contain integers") from exc
    if result[0] < 0 or result[1] <= 0:
        raise ValueError("directory identity is outside the accepted range")
    return result


def _verify_isolated_directory(
    info: os.stat_result, *, label: str, require_owner: bool
) -> None:
    if info.st_mode & _OTHER_PRINCIPAL_WRITE_BITS:
        _fail(f"{label} is writable by another principal")
    if require_owner and info.st_uid != os.geteuid():
        _fail(f"{label} is not owned by the deployment user")


def _absolute_lexical_path(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if not raw or not os.path.isabs(raw):
        raise ValueError("snapshots root must be an absolute path")
    normalized = os.path.normpath(raw)
    if normalized != raw or "\x00" in raw:
        raise ValueError("snapshots root must be a normalized lexical path")
    return Path(raw)


def _open_snapshots_root(
    value: str | os.PathLike[str], *, create_final: bool = False
) -> tuple[Path, int, os.stat_result]:
    """Open every path component with no-follow semantics.

    Only the final ``snapshots`` directory may be created.  Missing ancestors
    are rejected so a typo cannot create a second deployment hierarchy.
    """

    path = _absolute_lexical_path(value)
    components = path.parts[1:]
    if not components:
        raise ValueError("filesystem root cannot be the snapshots root")
    current_fd = os.open("/", _DIRECTORY_FLAGS | _NOFOLLOW)
    private_anchor_found = False
    try:
        for index, component in enumerate(components):
            final = index == len(components) - 1
            try:
                before = os.stat(
                    component, dir_fd=current_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                if not (create_final and final):
                    raise RuntimeError(
                        f"snapshots-root component is missing: {component}"
                    ) from None
                os.mkdir(component, 0o700, dir_fd=current_fd)
                before = os.stat(
                    component, dir_fd=current_fd, follow_symlinks=False
                )
            if stat.S_ISLNK(before.st_mode):
                _fail(f"snapshots-root component is a symbolic link: {component}")
            if not stat.S_ISDIR(before.st_mode):
                _fail(f"snapshots-root component is not a directory: {component}")
            if (
                not private_anchor_found
                and before.st_uid == os.geteuid()
                and not (before.st_mode & _OTHER_PRINCIPAL_ACCESS_BITS)
            ):
                private_anchor_found = True
            if private_anchor_found:
                _verify_isolated_directory(
                    before,
                    label=f"isolated deployment component {component}",
                    require_owner=True,
                )
            child_fd = os.open(
                component, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=current_fd
            )
            opened = os.fstat(child_fd)
            if _directory_identity(opened) != _directory_identity(before):
                os.close(child_fd)
                _fail(f"snapshots-root component changed while opened: {component}")
            if private_anchor_found:
                _verify_isolated_directory(
                    opened,
                    label=f"opened isolated deployment component {component}",
                    require_owner=True,
                )
            os.close(current_fd)
            current_fd = child_fd
        if not private_anchor_found:
            _fail(
                "snapshots root has no deployment-user-owned private anchor"
            )
        return path, current_fd, os.fstat(current_fd)
    except BaseException:
        os.close(current_fd)
        raise


def _validate_child_name(name: str) -> str:
    if (
        not name
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
        or "\x00" in name
        or Path(name).name != name
    ):
        raise ValueError("snapshot child must be one safe path component")
    return name


def _open_child_directory(
    parent_fd: int,
    name: str,
    *,
    expected: tuple[int, int] | None = None,
    label: str,
) -> tuple[int, os.stat_result]:
    name = _validate_child_name(name)
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} is missing: {name}") from exc
    if stat.S_ISLNK(before.st_mode):
        _fail(f"{label} is a symbolic link: {name}")
    if not stat.S_ISDIR(before.st_mode):
        _fail(f"{label} is not a directory: {name}")
    _verify_isolated_directory(before, label=label, require_owner=True)
    descriptor = os.open(
        name, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=parent_fd
    )
    opened = os.fstat(descriptor)
    actual = _directory_identity(opened)
    if actual != _directory_identity(before):
        os.close(descriptor)
        _fail(f"{label} changed while opened: {name}")
    _verify_isolated_directory(opened, label=label, require_owner=True)
    if expected is not None and actual != expected:
        os.close(descriptor)
        _fail(f"{label} identity mismatch: {name}")
    return descriptor, opened


def inspect_child(root_value: str, name: str) -> dict[str, object]:
    """Inspect a prospective published snapshot without following a link."""

    name = _validate_child_name(name)
    root, root_fd, root_info = _open_snapshots_root(
        root_value, create_final=True
    )
    try:
        result: dict[str, object] = {
            "schema": GUARD_SCHEMA,
            "snapshots_root": str(root),
            "snapshots_dev": root_info.st_dev,
            "snapshots_ino": root_info.st_ino,
            "name": name,
        }
        try:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            result["state"] = "missing"
            return result
        if stat.S_ISLNK(info.st_mode):
            _fail(f"snapshot child is a symbolic link: {name}")
        if not stat.S_ISDIR(info.st_mode):
            _fail(f"snapshot child is not a directory: {name}")
        child_fd, opened = _open_child_directory(
            root_fd, name, label="snapshot child"
        )
        os.close(child_fd)
        result.update({
            "state": "directory",
            "child_dev": opened.st_dev,
            "child_ino": opened.st_ino,
        })
        return result
    finally:
        os.close(root_fd)


def create_staging(root_value: str, tree_sha256: str) -> dict[str, object]:
    """Atomically create an unpredictable private staging child and source."""

    if _SHA256.fullmatch(tree_sha256) is None:
        raise ValueError("tree SHA-256 is invalid")
    root, root_fd, root_info = _open_snapshots_root(
        root_value, create_final=True
    )
    staging_name = ""
    staging_fd = -1
    try:
        # mkdirat with an unpredictable name has the same atomic-exclusion
        # property as mktemp -d while keeping path resolution anchored to the
        # already validated snapshots-root descriptor.
        for _ in range(128):
            candidate = (
                f".staging-{tree_sha256[:12]}-{secrets.token_hex(12)}"
            )
            try:
                os.mkdir(candidate, 0o700, dir_fd=root_fd)
            except FileExistsError:
                continue
            staging_name = candidate
            break
        if not staging_name:
            _fail("could not allocate a unique snapshot staging directory")
        staging_fd, staging_info = _open_child_directory(
            root_fd, staging_name, label="new snapshot staging"
        )
        os.mkdir("source", 0o700, dir_fd=staging_fd)
        source_fd, source_info = _open_child_directory(
            staging_fd, "source", label="new staging source"
        )
        os.close(source_fd)
        return {
            "schema": GUARD_SCHEMA,
            "snapshots_root": str(root),
            "snapshots_dev": root_info.st_dev,
            "snapshots_ino": root_info.st_ino,
            "staging_name": staging_name,
            "staging_path": str(root / staging_name),
            "staging_dev": staging_info.st_dev,
            "staging_ino": staging_info.st_ino,
            "source_dev": source_info.st_dev,
            "source_ino": source_info.st_ino,
        }
    except BaseException:
        # Creation failures are rare.  Delete only the direct child just made;
        # no pre-existing name is ever selected by this branch.
        if staging_fd >= 0:
            try:
                _purge_open_directory(staging_fd)
                current = os.stat(
                    staging_name, dir_fd=root_fd, follow_symlinks=False
                )
                if (
                    stat.S_ISDIR(current.st_mode)
                    and _directory_identity(current)
                    == _directory_identity(os.fstat(staging_fd))
                ):
                    os.rmdir(staging_name, dir_fd=root_fd)
            except Exception:
                pass
            os.close(staging_fd)
            staging_fd = -1
        elif staging_name:
            try:
                os.rmdir(staging_name, dir_fd=root_fd)
            except OSError:
                pass
        raise
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)
        os.close(root_fd)


def verify_child(
    root_value: str,
    name: str,
    root_identity: tuple[int, int],
    child_identity: tuple[int, int],
) -> dict[str, object]:
    root, root_fd, root_info = _open_snapshots_root(root_value)
    child_fd = -1
    try:
        if _directory_identity(root_info) != root_identity:
            _fail("snapshots-root identity mismatch")
        child_fd, child_info = _open_child_directory(
            root_fd, name, expected=child_identity, label="snapshot child"
        )
        return {
            "schema": GUARD_SCHEMA,
            "snapshots_root": str(root),
            "name": name,
            "child_dev": child_info.st_dev,
            "child_ino": child_info.st_ino,
            "verified": True,
        }
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        os.close(root_fd)


def verify_staging(
    root_value: str,
    name: str,
    root_identity: tuple[int, int],
    staging_identity: tuple[int, int],
    source_identity: tuple[int, int],
) -> dict[str, object]:
    root, root_fd, root_info = _open_snapshots_root(root_value)
    staging_fd = source_fd = -1
    try:
        if _directory_identity(root_info) != root_identity:
            _fail("snapshots-root identity mismatch")
        staging_fd, staging_info = _open_child_directory(
            root_fd,
            name,
            expected=staging_identity,
            label="snapshot staging",
        )
        source_fd, source_info = _open_child_directory(
            staging_fd,
            "source",
            expected=source_identity,
            label="staging source",
        )
        return {
            "schema": GUARD_SCHEMA,
            "snapshots_root": str(root),
            "staging_name": name,
            "staging_dev": staging_info.st_dev,
            "staging_ino": staging_info.st_ino,
            "source_dev": source_info.st_dev,
            "source_ino": source_info.st_ino,
            "verified": True,
        }
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if staging_fd >= 0:
            os.close(staging_fd)
        os.close(root_fd)


def _purge_open_directory(directory_fd: int) -> None:
    """Remove contents relative to an open directory, never following links."""

    os.fchmod(directory_fd, 0o700)
    for name in os.listdir(directory_fd):
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(before.st_mode):
            child_fd = os.open(
                name, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=directory_fd
            )
            try:
                if _directory_identity(os.fstat(child_fd)) != _directory_identity(
                    before
                ):
                    _fail(f"cleanup child changed while opened: {name}")
                _purge_open_directory(child_fd)
                current = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                if _directory_identity(current) != _directory_identity(before):
                    _fail(f"cleanup child identity changed: {name}")
                os.rmdir(name, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
        else:
            # unlinkat removes a symlink or special-file directory entry; it
            # cannot traverse its target.
            os.unlink(name, dir_fd=directory_fd)


def remove_child(
    root_value: str,
    name: str,
    root_identity: tuple[int, int],
    child_identity: tuple[int, int],
) -> dict[str, object]:
    """Remove only the identity-bound direct child created by this run."""

    root, root_fd, root_info = _open_snapshots_root(root_value)
    child_fd = -1
    try:
        if _directory_identity(root_info) != root_identity:
            _fail("snapshots-root identity mismatch during cleanup")
        child_fd, before = _open_child_directory(
            root_fd,
            name,
            expected=child_identity,
            label="cleanup snapshot child",
        )
        _purge_open_directory(child_fd)
        current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or _directory_identity(current) != child_identity
            or _directory_identity(os.fstat(child_fd)) != child_identity
        ):
            _fail("cleanup snapshot child identity changed before removal")
        os.rmdir(name, dir_fd=root_fd)
        return {
            "schema": GUARD_SCHEMA,
            "snapshots_root": str(root),
            "name": name,
            "removed": True,
        }
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        os.close(root_fd)


def _usage() -> NoReturn:
    raise SystemExit(
        "usage: snapshot_staging_guard.py ACTION SNAPSHOTS_ROOT ..."
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2:
        _usage()
    action, root, *values = arguments
    if action == "inspect" and len(values) == 1:
        result = inspect_child(root, values[0])
    elif action == "create" and len(values) == 1:
        result = create_staging(root, values[0])
    elif action == "verify-child" and len(values) == 5:
        result = verify_child(
            root,
            values[0],
            _parse_identity(values[1], values[2]),
            _parse_identity(values[3], values[4]),
        )
    elif action == "verify-staging" and len(values) == 7:
        result = verify_staging(
            root,
            values[0],
            _parse_identity(values[1], values[2]),
            _parse_identity(values[3], values[4]),
            _parse_identity(values[5], values[6]),
        )
    elif action == "remove" and len(values) == 5:
        result = remove_child(
            root,
            values[0],
            _parse_identity(values[1], values[2]),
            _parse_identity(values[3], values[4]),
        )
    else:
        _usage()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
