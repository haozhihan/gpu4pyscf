#!/usr/bin/env python3
"""Normalize approved source links before publishing an MTU snapshot.

The source checkout may contain repository-managed convenience links.  A
published runtime snapshot may not: receipt and release validation require a
tree made only of directories and regular files.  This module accepts the
small, deterministic subset needed by the repository (a relative leaf link
which points directly to an in-tree regular file), binds its target bytes in
an inventory, and materializes it without ever following an unchecked link.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Mapping


NORMALIZATION_SCHEMA = "gpu4pyscf.source-symlink-normalization.v1"
NORMALIZATION_POLICY = "validated-relative-leaf-links-materialized"
EXCLUDED_PARTS = {
    ".git", ".pytest_cache", "__pycache__", "build", "dist", "results",
}


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    """Return the fields used for pre/open/post TOCTOU checks."""

    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _directory_identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _canonical_root(value: str | os.PathLike[str]) -> Path:
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    info = root.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("source normalization root must be a directory")
    if root.resolve(strict=True) != root:
        raise ValueError("source normalization root must be canonical")
    return root


def _open_relative_directory_anchored(
    root: Path, relative: Path, *, label: str
) -> tuple[list[int], list[tuple[int, int, str, os.stat_result]], os.stat_result]:
    """Open ``relative`` from root and retain every parent for postchecks."""

    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} is not a safe relative directory")
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    root_before = root.lstat()
    root_fd = os.open(root, os.O_RDONLY | directory_flag | nofollow_flag)
    descriptors = [root_fd]
    opened: list[tuple[int, int, str, os.stat_result]] = []
    try:
        if _directory_identity(os.fstat(root_fd)) != _directory_identity(root_before):
            raise ValueError(f"{label} root changed while opened")
        parent_fd = root_fd
        for component in relative.parts:
            if component in {"", "."}:
                continue
            before = os.stat(
                component, dir_fd=parent_fd, follow_symlinks=False
            )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"{label} traverses a linked/non-directory parent")
            descriptor = os.open(
                component,
                os.O_RDONLY | directory_flag | nofollow_flag,
                dir_fd=parent_fd,
            )
            if _directory_identity(os.fstat(descriptor)) != _directory_identity(
                before
            ):
                os.close(descriptor)
                raise ValueError(f"{label} parent changed while opened")
            descriptors.append(descriptor)
            opened.append((parent_fd, descriptor, component, before))
            parent_fd = descriptor
        return descriptors, opened, root_before
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _verify_anchored_directory_chain(
    root: Path,
    descriptors: list[int],
    opened: list[tuple[int, int, str, os.stat_result]],
    root_before: os.stat_result,
    *,
    label: str,
) -> None:
    for parent_fd, descriptor, component, before in opened:
        current = os.stat(
            component, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or _directory_identity(current) != _directory_identity(before)
            or _directory_identity(os.fstat(descriptor))
            != _directory_identity(before)
        ):
            raise ValueError(f"{label} parent path changed during operation")
    if (
        _directory_identity(root.lstat()) != _directory_identity(root_before)
        or _directory_identity(os.fstat(descriptors[0]))
        != _directory_identity(root_before)
    ):
        raise ValueError(f"{label} root changed during operation")


def _read_link_child_nofollow(
    parent_fd: int, name: str, *, label: str
) -> tuple[str, os.stat_result]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} is no longer a symbolic link")
    target = os.readlink(name, dir_fd=parent_fd)
    after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _identity(after) != _identity(before) or os.readlink(
        name, dir_fd=parent_fd
    ) != target:
        raise ValueError(f"{label} changed while it was read")
    return target, before


def _read_regular_child_nofollow(
    parent_fd: int, name: str, *, label: str
) -> tuple[bytes, os.stat_result]:
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file: {name}")
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd
    )
    try:
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise ValueError(f"{label} changed while it was opened: {name}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _identity(after) != _identity(before):
        raise ValueError(f"{label} changed while it was read: {name}")
    return b"".join(chunks), after


def _read_relative_regular_anchored(
    root: Path, relative: Path, *, link_relative: Path
) -> tuple[bytes, os.stat_result]:
    """Read an in-tree target through an fd-anchored, no-follow path walk."""

    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    root_before = root.lstat()
    root_fd = os.open(root, os.O_RDONLY | directory_flag | nofollow_flag)
    opened: list[tuple[int, int, str, os.stat_result]] = []
    descriptors = [root_fd]
    try:
        if _identity(os.fstat(root_fd)) != _identity(root_before):
            raise ValueError("source root changed during anchored target walk")
        parent_fd = root_fd
        final_info: os.stat_result | None = None
        for index, component in enumerate(relative.parts):
            final = index == len(relative.parts) - 1
            try:
                before = os.stat(
                    component, dir_fd=parent_fd, follow_symlinks=False
                )
            except FileNotFoundError as exc:
                raise ValueError(f"source link is dangling: {link_relative}") from exc
            if stat.S_ISLNK(before.st_mode):
                kind = "link chain or cycle" if final else "linked directory"
                raise ValueError(f"source link uses a {kind}: {link_relative}")
            if final and stat.S_ISDIR(before.st_mode):
                raise ValueError(
                    f"source link points to a directory: {link_relative}"
                )
            if not final and not stat.S_ISDIR(before.st_mode):
                raise ValueError(
                    "source link has a non-directory path component: "
                    f"{link_relative}"
                )
            if final and not stat.S_ISREG(before.st_mode):
                raise ValueError(
                    f"source link target is not a regular file: {link_relative}"
                )
            flags = os.O_RDONLY | nofollow_flag
            if not final:
                flags |= directory_flag
            descriptor = os.open(component, flags, dir_fd=parent_fd)
            descriptors.append(descriptor)
            opened_info = os.fstat(descriptor)
            if _identity(opened_info) != _identity(before):
                raise ValueError(
                    f"source link target changed while opened: {link_relative}"
                )
            opened.append((parent_fd, descriptor, component, before))
            parent_fd = descriptor
            if final:
                final_info = opened_info

        if final_info is None:
            raise ValueError(f"source link points to a directory: {link_relative}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptors[-1], 1024 * 1024):
            chunks.append(chunk)
        data = b"".join(chunks)

        # Recheck every directory entry through its still-open parent fd.  If
        # any intermediate directory was renamed and replaced, this detects
        # that the anchored object is no longer the path named by the record.
        for parent_fd, descriptor, component, before in opened:
            current = os.stat(
                component, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                _identity(current) != _identity(before)
                or _identity(os.fstat(descriptor)) != _identity(before)
            ):
                raise ValueError(
                    "source link target path changed during anchored read: "
                    f"{link_relative}"
                )
        if _identity(root.lstat()) != _identity(root_before):
            raise ValueError("source root changed during anchored target read")
        return data, final_info
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _iter_entries(root: Path):
    """Yield every entry that the deployment rsync can copy, without follow."""

    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_names[:] = sorted(
            name for name in directory_names if name not in EXCLUDED_PARTS
        )
        for name in sorted((*directory_names, *file_names)):
            if name in EXCLUDED_PARTS or name.endswith(".pyc"):
                continue
            path = Path(directory) / name
            relative = path.relative_to(root)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            yield relative, path


def _inspect_link(
    root: Path, relative: Path, path: Path
) -> tuple[dict[str, Any], bytes, os.stat_result, os.stat_result]:
    del path  # All link operations below are descriptor-relative.
    descriptors, opened, root_before = _open_relative_directory_anchored(
        root, relative.parent, label=f"source link parent {relative}"
    )
    parent_fd = descriptors[-1]
    try:
        target_text, link_before = _read_link_child_nofollow(
            parent_fd, relative.name, label=f"source link {relative}"
        )
        if not target_text:
            raise ValueError(f"source link has an empty target: {relative}")
        if os.path.isabs(target_text):
            raise ValueError(f"source link has an absolute target: {relative}")
        if "\\" in target_text or any(
            ord(character) < 32 or ord(character) == 127
            for character in target_text
        ):
            raise ValueError(f"source link target is suspicious: {relative}")
        components = target_text.split("/")
        if any(component in {"", "."} for component in components):
            raise ValueError(f"source link target is suspicious: {relative}")

        target_parts = list(relative.parent.parts)
        for component in components:
            if component == "..":
                if not target_parts:
                    raise ValueError(f"source link escapes source root: {relative}")
                target_parts.pop()
            else:
                if component in EXCLUDED_PARTS or component.endswith(".pyc"):
                    raise ValueError(
                        f"source link enters an excluded path: {relative}"
                    )
                target_parts.append(component)
        target_relative = Path(*target_parts)

        target_data, target_info = _read_relative_regular_anchored(
            root, target_relative, link_relative=relative
        )
        final_target, final_link = _read_link_child_nofollow(
            parent_fd, relative.name, label=f"source link {relative}"
        )
        if (
            _identity(final_link) != _identity(link_before)
            or final_target != target_text
        ):
            raise ValueError(f"source link changed during inspection: {relative}")
        _verify_anchored_directory_chain(
            root,
            descriptors,
            opened,
            root_before,
            label=f"source link parent {relative}",
        )
        record = {
            "relative_path": relative.as_posix(),
            "link_type": "relative-leaf-file",
            "target": target_text,
            "resolved_target_relative_path": target_relative.as_posix(),
            "target_bytes": len(target_data),
            "target_sha256": hashlib.sha256(target_data).hexdigest(),
        }
        return record, target_data, target_info, link_before
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def inspect_source(root_value: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate all copied entries and return the logical link inventory."""

    root = _canonical_root(root_value)
    links: list[dict[str, Any]] = []
    regular_files = 0
    directories = 0
    for relative, path in _iter_entries(root):
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise ValueError(
                f"source entry disappeared during inspection: {relative}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            record, _, _, _ = _inspect_link(root, relative, path)
            links.append(record)
        elif stat.S_ISREG(info.st_mode):
            regular_files += 1
        elif stat.S_ISDIR(info.st_mode):
            directories += 1
        else:
            raise ValueError(f"source contains a special file: {relative}")
    return {
        "schema": NORMALIZATION_SCHEMA,
        "policy": NORMALIZATION_POLICY,
        "links": links,
        "link_count": len(links),
        "regular_file_count_before": regular_files,
        "directory_count": directories,
    }


def canonical_inventory_bytes(inventory: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(inventory), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ) + "\n"
    ).encode("ascii")


def inventory_sha256(inventory: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_inventory_bytes(inventory)).hexdigest()


def write_inventory(
    root: str | os.PathLike[str], output: str | os.PathLike[str]
) -> dict[str, Any]:
    inventory = inspect_source(root)
    path = Path(output)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        data = canonical_inventory_bytes(inventory)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("inventory write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return inventory


def load_inventory(value: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(value)

    def reject_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"source link inventory duplicates key {key!r}")
            result[key] = item
        return result

    loaded = json.loads(
        path.read_text(encoding="ascii"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(loaded, dict):
        raise ValueError("source link inventory root must be an object")
    if loaded.get("schema") != NORMALIZATION_SCHEMA:
        raise ValueError("source link inventory schema mismatch")
    if loaded.get("policy") != NORMALIZATION_POLICY:
        raise ValueError("source link inventory policy mismatch")
    if loaded.get("link_count") != len(loaded.get("links", [])):
        raise ValueError("source link inventory count mismatch")
    return loaded


def assert_normalized_source(root_value: str | os.PathLike[str]) -> dict[str, int]:
    """Prove a publishable tree contains no links or special files."""

    root = _canonical_root(root_value)
    regular_files = 0
    directories = 0
    for relative, path in _iter_entries(root):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"normalized source still contains a link: {relative}")
        if stat.S_ISREG(info.st_mode):
            regular_files += 1
        elif stat.S_ISDIR(info.st_mode):
            directories += 1
        else:
            raise ValueError(f"normalized source contains a special file: {relative}")
    return {
        "published_symlink_count": 0,
        "published_special_file_count": 0,
        "regular_file_count_after": regular_files,
        "directory_count_after": directories,
    }


def _scan_normalized_directory_fd(directory_fd: int) -> tuple[int, int]:
    """Count a regular tree beneath an already trusted directory descriptor."""

    regular_files = 0
    directories = 0
    for name in sorted(os.listdir(directory_fd)):
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(before.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                if _directory_identity(os.fstat(child_fd)) != _directory_identity(
                    before
                ):
                    raise ValueError(
                        f"normalized source directory changed while opened: {name}"
                    )
                child_files, child_directories = _scan_normalized_directory_fd(
                    child_fd
                )
                after = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    _directory_identity(after) != _directory_identity(before)
                    or _directory_identity(os.fstat(child_fd))
                    != _directory_identity(before)
                ):
                    raise ValueError(
                        f"normalized source directory changed while scanned: {name}"
                    )
            finally:
                os.close(child_fd)
            regular_files += child_files
            directories += 1 + child_directories
        elif stat.S_ISREG(before.st_mode):
            regular_files += 1
        elif stat.S_ISLNK(before.st_mode):
            raise ValueError(f"normalized source still contains a link: {name}")
        else:
            raise ValueError(f"normalized source contains a special file: {name}")
    return regular_files, directories


def assert_normalized_source_fd(directory_fd: int) -> dict[str, int]:
    """Prove a publishable tree through a caller-held directory descriptor."""

    root_before = os.fstat(directory_fd)
    if not stat.S_ISDIR(root_before.st_mode):
        raise ValueError("normalized source descriptor is not a directory")
    regular_files, directories = _scan_normalized_directory_fd(directory_fd)
    if _directory_identity(os.fstat(directory_fd)) != _directory_identity(root_before):
        raise ValueError("normalized source descriptor changed while scanned")
    return {
        "published_symlink_count": 0,
        "published_special_file_count": 0,
        "regular_file_count_after": regular_files,
        "directory_count_after": directories,
    }


def materialize_source_links(
    root_value: str | os.PathLike[str], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace every validated link with a stable copy of its target bytes."""

    root = _canonical_root(root_value)
    observed = inspect_source(root)
    if observed != dict(expected):
        raise ValueError("source link inventory changed before materialization")

    for expected_record in observed["links"]:
        relative = Path(expected_record["relative_path"])
        path = root / relative
        record, target_data, target_info, link_info = _inspect_link(
            root, relative, path
        )
        if record != expected_record:
            raise ValueError(f"source link changed before replacement: {relative}")
        descriptors, opened, root_before = _open_relative_directory_anchored(
            root, relative.parent,
            label=f"source materialization parent {relative}",
        )
        parent_fd = descriptors[-1]
        descriptor = -1
        temporary_name = ""
        try:
            current_target, current_link = _read_link_child_nofollow(
                parent_fd, relative.name, label=f"source link {relative}"
            )
            if (
                _identity(current_link) != _identity(link_info)
                or current_target != expected_record["target"]
            ):
                raise ValueError(
                    f"source link changed before atomic replacement: {relative}"
                )
            for _ in range(128):
                candidate = (
                    ".source-link-materialize-" + secrets.token_hex(12)
                )
                try:
                    descriptor = os.open(
                        candidate,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=parent_fd,
                    )
                except FileExistsError:
                    continue
                temporary_name = candidate
                break
            if descriptor < 0:
                raise RuntimeError(
                    "could not allocate a source-link materialization file"
                )
            view = memoryview(target_data)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("source link materialization made no progress")
                view = view[written:]
            os.fchmod(descriptor, stat.S_IMODE(target_info.st_mode))
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            final_target, final_link = _read_link_child_nofollow(
                parent_fd, relative.name, label=f"source link {relative}"
            )
            if (
                _identity(final_link) != _identity(link_info)
                or final_target != expected_record["target"]
            ):
                raise ValueError(
                    f"source link changed before atomic replacement: {relative}"
                )
            os.replace(
                temporary_name,
                relative.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            temporary_name = ""
            replaced_data, _ = _read_regular_child_nofollow(
                parent_fd,
                relative.name,
                label="materialized source link",
            )
            if (
                len(replaced_data) != expected_record["target_bytes"]
                or hashlib.sha256(replaced_data).hexdigest()
                != expected_record["target_sha256"]
            ):
                raise ValueError(
                    f"materialized source link content mismatch: {relative}"
                )
            _verify_anchored_directory_chain(
                root,
                descriptors,
                opened,
                root_before,
                label=f"source materialization parent {relative}",
            )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_name:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            for directory_descriptor in reversed(descriptors):
                os.close(directory_descriptor)

    normalized = assert_normalized_source(root)
    return {
        "schema": NORMALIZATION_SCHEMA,
        "policy": NORMALIZATION_POLICY,
        "source_link_inventory": observed,
        "source_link_inventory_sha256": inventory_sha256(observed),
        "materialized_link_count": observed["link_count"],
        "materialized_links": observed["links"],
        **normalized,
    }


__all__ = [
    "NORMALIZATION_POLICY",
    "NORMALIZATION_SCHEMA",
    "assert_normalized_source",
    "assert_normalized_source_fd",
    "canonical_inventory_bytes",
    "inspect_source",
    "inventory_sha256",
    "load_inventory",
    "materialize_source_links",
    "write_inventory",
]
