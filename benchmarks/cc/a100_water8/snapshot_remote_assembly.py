#!/usr/bin/env python3
"""Assemble and atomically publish a received regular-tree snapshot.

This module is embedded into ``snapshot_bundle_receiver.py`` output.  Its
``post_receive`` callback runs before the receiver releases the validated
snapshots, staging, and source directory descriptors, so every mutation is
rooted at those descriptors even if a mount ancestor is renamed concurrently.
"""

from __future__ import annotations

import base64
import ctypes
import datetime
import errno
import hashlib
import importlib.util
import json
import os
import pathlib
import stat
import sys
from typing import Any, Sequence

sys.dont_write_bytecode = True


_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_UNSUPPORTED_EXCLUSIVE_RENAME = {
    errno.EINVAL,
    errno.ENOSYS,
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    getattr(errno, "ENOTSUP", errno.EINVAL),
}
PUBLISH_TRUST_BOUNDARY = "private-anchor-owner-and-root-are-trusted-cooperators"
PUBLICATION_POLICY_SCHEMA = "gpu4pyscf.snapshot-publication-policy.v1"
PUBLICATION_ATTESTATION_SCHEMA = "gpu4pyscf.snapshot-publication-attestation.v1"
PUBLICATION_ATTESTATION_SUFFIX = ".publication-attestation.json"
PUBLICATION_POLICY = {
    "schema": PUBLICATION_POLICY_SCHEMA,
    "trust_boundary": PUBLISH_TRUST_BOUNDARY,
    "attestation_schema": PUBLICATION_ATTESTATION_SCHEMA,
    "attestation_location": (
        "snapshots-root sibling named "
        "<snapshot-name>.publication-attestation.json"
    ),
}


class PublicationIndeterminateError(RuntimeError):
    """A rename failed and its namespace outcome could not be observed safely."""

    state = "indeterminate"


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_all(descriptor: int, payload: bytes, *, replace: bool = False) -> None:
    if replace:
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written < 1:
            raise OSError("snapshot attestation write made no progress")
        view = view[written:]
    os.fsync(descriptor)


def _directory_identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _safe_child_name(value: str, *, label: str) -> str:
    if (
        not value or value in {".", ".."} or os.sep in value
        or (os.altsep is not None and os.altsep in value)
        or pathlib.Path(value).name != value or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RuntimeError(f"{label} is not one safe path component")
    return value


def _exclusive_rename_noreplace(
    parent_fd: int, source_name: str, destination_name: str
) -> bool:
    """Return false only when the filesystem lacks an exclusive rename."""

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source_name)
    encoded_destination = os.fsencode(destination_name)
    if sys.platform == "darwin":
        primitive = getattr(libc, "renameatx_np", None)
        if primitive is None:
            return False
        primitive.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        primitive.restype = ctypes.c_int
        result = primitive(
            parent_fd, encoded_source,
            parent_fd, encoded_destination,
            0x00000004,  # RENAME_EXCL
        )
    else:
        primitive = getattr(libc, "renameat2", None)
        if primitive is None:
            return False
        primitive.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        primitive.restype = ctypes.c_int
        result = primitive(
            parent_fd, encoded_source,
            parent_fd, encoded_destination,
            1,  # RENAME_NOREPLACE
        )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise RuntimeError("snapshot target appeared concurrently")
    if error in _UNSUPPORTED_EXCLUSIVE_RENAME:
        return False
    raise OSError(error, os.strerror(error))


def _reconcile_rename_error(
    parent_fd: int,
    source_fd: int,
    source_name: str,
    destination_name: str,
    expected_source_identity: tuple[int, int],
    *,
    uncommitted_target_identity: tuple[int, int] | None,
    stat_entry=os.stat,
) -> str:
    """Return committed/not-committed or raise for an unobservable outcome."""

    def inspect(name: str) -> os.stat_result | None:
        try:
            return stat_entry(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None

    try:
        source = inspect(source_name)
        target = inspect(destination_name)
        held_source_identity = _directory_identity(os.fstat(source_fd))
    except OSError as exc:
        raise PublicationIndeterminateError(
            "publication outcome is indeterminate: namespace reconciliation "
            f"failed for staging={source_name!r}, target={destination_name!r}, "
            f"expected_inode={expected_source_identity!r}"
        ) from exc
    if held_source_identity != expected_source_identity:
        raise PublicationIndeterminateError(
            "publication outcome is indeterminate: held staging identity changed"
        )
    source_identity = None if source is None else _directory_identity(source)
    target_identity = None if target is None else _directory_identity(target)
    if (
        source is None
        and target is not None
        and stat.S_ISDIR(target.st_mode)
        and target_identity == expected_source_identity
    ):
        return "committed"
    if (
        source is not None
        and stat.S_ISDIR(source.st_mode)
        and source_identity == expected_source_identity
        and (
            target is None
            or (
                uncommitted_target_identity is not None
                and target is not None
                and stat.S_ISDIR(target.st_mode)
                and target_identity == uncommitted_target_identity
            )
        )
    ):
        return "not-committed"
    raise PublicationIndeterminateError(
        "publication outcome is indeterminate: observed namespace does not "
        f"prove commit or rollback for staging={source_name!r}, "
        f"target={destination_name!r}, expected_inode={expected_source_identity!r}"
    )


def _publish_directory_noreplace(
    parent_fd: int,
    source_fd: int,
    source_name: str,
    destination_name: str,
    expected_source_identity: tuple[int, int],
    *,
    exclusive_rename=_exclusive_rename_noreplace,
    fallback_rename=os.rename,
    reconciliation_stat=os.stat,
    reservation_hook=None,
) -> str:
    """Atomically publish ``source_name`` without overwriting a prior name.

    Linux local filesystems use ``renameat2(RENAME_NOREPLACE)``.  Some NFS
    servers return ``EINVAL`` for that flag.  The fallback first creates the
    destination as an exclusive empty directory, binds its inode through an
    fd, then atomically renames the source over only that still-empty, still-
    bound reservation.  All namespace operations remain relative to the held
    parent descriptor.  This contract excludes hostile processes running as
    the deployment UID (and root): POSIX/NFS has no inode-conditional rename,
    so those principals must be trusted cooperators under the private anchor.
    """

    source_name = _safe_child_name(source_name, label="snapshot staging name")
    destination_name = _safe_child_name(
        destination_name, label="snapshot target name"
    )

    def verify_source_entry() -> None:
        current = os.stat(
            source_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or _directory_identity(current) != expected_source_identity
            or _directory_identity(os.fstat(source_fd)) != expected_source_identity
        ):
            raise RuntimeError("snapshot staging identity changed before publish")

    verify_source_entry()
    try:
        os.stat(destination_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("snapshot target appeared concurrently")

    protocol: str | None = None
    try:
        exclusive_supported = exclusive_rename(
            parent_fd, source_name, destination_name
        )
    except OSError:
        native_outcome = _reconcile_rename_error(
            parent_fd,
            source_fd,
            source_name,
            destination_name,
            expected_source_identity,
            uncommitted_target_identity=None,
            stat_entry=reconciliation_stat,
        )
        if native_outcome != "committed":
            raise
        protocol = "atomic-exclusive-rename-commit-confirmed"
        exclusive_supported = True
    if exclusive_supported:
        if protocol is None:
            protocol = "atomic-exclusive-rename"
    else:
        reservation_fd = -1
        reservation_identity: tuple[int, int] | None = None
        try:
            try:
                os.mkdir(destination_name, 0o700, dir_fd=parent_fd)
            except FileExistsError as exc:
                raise RuntimeError(
                    "snapshot target appeared during reservation"
                ) from exc
            before = os.stat(
                destination_name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISDIR(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                raise RuntimeError("snapshot target reservation is unsafe")
            reservation_identity = _directory_identity(before)
            reservation_fd = os.open(
                destination_name,
                _DIRECTORY_FLAGS | _NOFOLLOW,
                dir_fd=parent_fd,
            )
            if _directory_identity(os.fstat(reservation_fd)) != reservation_identity:
                raise RuntimeError(
                    "snapshot target reservation changed while opened"
                )
            if reservation_hook is not None:
                reservation_hook(
                    parent_fd, destination_name, reservation_fd,
                    reservation_identity,
                )
            current = os.stat(
                destination_name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISDIR(current.st_mode)
                or _directory_identity(current) != reservation_identity
                or _directory_identity(os.fstat(reservation_fd))
                != reservation_identity
            ):
                raise RuntimeError("snapshot target reservation was replaced")
            if os.listdir(reservation_fd):
                raise RuntimeError("snapshot target reservation is not empty")
            verify_source_entry()
            try:
                fallback_rename(
                    source_name,
                    destination_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
            except OSError:
                # An NFS server may commit RENAME but lose the reply.  Treat
                # the error as success only when the held source inode is now
                # installed at the target and the staging name is absent.
                fallback_outcome = _reconcile_rename_error(
                    parent_fd,
                    source_fd,
                    source_name,
                    destination_name,
                    expected_source_identity,
                    uncommitted_target_identity=reservation_identity,
                    stat_entry=reconciliation_stat,
                )
                if fallback_outcome != "committed":
                    raise
                protocol = (
                    "reserved-empty-directory-rename-commit-confirmed"
                )
            else:
                protocol = "reserved-empty-directory-rename"
        except BaseException:
            if reservation_fd >= 0 and reservation_identity is not None:
                try:
                    current = os.stat(
                        destination_name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if (
                        stat.S_ISDIR(current.st_mode)
                        and _directory_identity(current) == reservation_identity
                        and _directory_identity(os.fstat(reservation_fd))
                        == reservation_identity
                        and not os.listdir(reservation_fd)
                    ):
                        os.rmdir(destination_name, dir_fd=parent_fd)
                except (FileNotFoundError, OSError, RuntimeError):
                    pass
            raise
        finally:
            if reservation_fd >= 0:
                os.close(reservation_fd)

    published = os.stat(
        destination_name, dir_fd=parent_fd, follow_symlinks=False
    )
    if (
        not stat.S_ISDIR(published.st_mode)
        or _directory_identity(published) != expected_source_identity
        or _directory_identity(os.fstat(source_fd)) != expected_source_identity
    ):
        raise RuntimeError("published snapshot identity differs from staging")
    try:
        os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("staging name remained after atomic publication")
    if protocol is None:
        raise PublicationIndeterminateError(
            "publication outcome is indeterminate: no protocol was recorded"
        )
    return protocol


def _assemble(context: dict[str, Any]) -> dict[str, Any]:
    snapshots_root_text = sys.argv[1]
    staging_name = sys.argv[2]
    expected_snapshots_identity = (int(sys.argv[3]), int(sys.argv[4]))
    expected_staging_identity = (int(sys.argv[5]), int(sys.argv[6]))
    expected_source_identity = (int(sys.argv[7]), int(sys.argv[8]))
    target_name = sys.argv[9]
    tree_sha256 = sys.argv[10]
    tree_file_count = int(sys.argv[11])
    normalization_sha256 = sys.argv[12]
    normalization_bytes = int(sys.argv[13])
    base_revision = sys.argv[14]
    runtime_binary_source_text = sys.argv[15]
    required_library_names = json.loads(base64.b64decode(sys.argv[16]).decode("utf-8"))
    deployment_profile = sys.argv[17]
    if deployment_profile not in {"candidate", "g0-canonical-pristine"}:
        raise RuntimeError("snapshot assembly deployment profile is invalid")
    suffixes = {
        ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".ini",
        ".json", ".md", ".py", ".pyx", ".pxd", ".sbatch", ".sh", ".toml",
        ".txt", ".yaml", ".yml",
    }
    excluded = {".git", ".pytest_cache", "__pycache__", "build", "dist", "results"}
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    other_principal_write_bits = stat.S_IWGRP | stat.S_IWOTH
    other_principal_access_bits = stat.S_IRWXG | stat.S_IRWXO

    def directory_identity(info):
        return info.st_dev, info.st_ino

    def lexical_absolute(value, *, label):
        if not value or not os.path.isabs(value) or os.path.normpath(value) != value:
            raise RuntimeError(f"{label} must be a normalized absolute path")
        if "\x00" in value:
            raise RuntimeError(f"{label} contains a NUL byte")
        if sys.platform == "darwin":
            for alias, canonical in (
                ("/var", "/private/var"),
                ("/tmp", "/private/tmp"),
                ("/etc", "/private/etc"),
            ):
                if value == alias or value.startswith(alias + os.sep):
                    value = canonical + value[len(alias):]
                    break
        return pathlib.Path(value)

    def verify_isolated_directory(info, *, label, require_owner):
        if info.st_mode & other_principal_write_bits:
            raise RuntimeError(f"{label} is writable by another principal")
        if require_owner and info.st_uid != os.geteuid():
            raise RuntimeError(f"{label} is not owned by the deployment user")

    def open_absolute_directory(value, *, label, isolated=False):
        path = lexical_absolute(value, label=label)
        components = path.parts[1:]
        if not components:
            raise RuntimeError(f"{label} cannot be the filesystem root")
        descriptor = os.open("/", directory_flags | nofollow)
        private_anchor_found = False
        try:
            for component in components:
                before = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False
                )
                if stat.S_ISLNK(before.st_mode):
                    raise RuntimeError(
                        f"{label} component is a symbolic link: {component}"
                    )
                if not stat.S_ISDIR(before.st_mode):
                    raise RuntimeError(
                        f"{label} component is not a directory: {component}"
                    )
                if (
                    isolated and not private_anchor_found
                    and before.st_uid == os.geteuid()
                    and not (before.st_mode & other_principal_access_bits)
                ):
                    private_anchor_found = True
                if isolated and private_anchor_found:
                    verify_isolated_directory(
                        before, label=f"{label} component {component}",
                        require_owner=True,
                    )
                child = os.open(
                    component, directory_flags | nofollow, dir_fd=descriptor
                )
                try:
                    opened = os.fstat(child)
                    if directory_identity(opened) != directory_identity(before):
                        raise RuntimeError(
                            f"{label} component changed while opened: {component}"
                        )
                    if isolated and private_anchor_found:
                        verify_isolated_directory(
                            opened,
                            label=f"opened {label} component {component}",
                            require_owner=True,
                        )
                    after = os.stat(
                        component, dir_fd=descriptor, follow_symlinks=False
                    )
                    if (
                        not stat.S_ISDIR(after.st_mode)
                        or directory_identity(after) != directory_identity(before)
                    ):
                        raise RuntimeError(
                            f"{label} component changed after open: {component}"
                        )
                except BaseException:
                    os.close(child)
                    raise
                os.close(descriptor)
                descriptor = child
            if isolated and not private_anchor_found:
                raise RuntimeError(
                    f"{label} has no deployment-user-owned private anchor"
                )
            return path, descriptor, os.fstat(descriptor)
        except BaseException:
            os.close(descriptor)
            raise

    def validate_child_name(value, *, label):
        if (
            not value or value in {".", ".."} or os.sep in value
            or (os.altsep is not None and os.altsep in value)
            or pathlib.Path(value).name != value or "\x00" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise RuntimeError(f"{label} is not one safe path component")
        return value

    def open_child_directory(parent_fd, name, expected=None, *, label):
        validate_child_name(name, label=label)
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise RuntimeError(f"{label} is a symbolic link")
        if not stat.S_ISDIR(before.st_mode):
            raise RuntimeError(f"{label} is not a directory")
        verify_isolated_directory(before, label=label, require_owner=True)
        descriptor = os.open(
            name, directory_flags | nofollow, dir_fd=parent_fd
        )
        try:
            opened = os.fstat(descriptor)
            actual = directory_identity(opened)
            if (
                actual != directory_identity(before)
                or (expected is not None and actual != expected)
            ):
                raise RuntimeError(f"{label} identity mismatch")
            verify_isolated_directory(opened, label=label, require_owner=True)
            after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(after.st_mode)
                or directory_identity(after) != actual
            ):
                raise RuntimeError(f"{label} changed after it was opened")
            verify_isolated_directory(after, label=label, require_owner=True)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    snapshots_root = pathlib.Path(context["destination_root"])
    snapshots_fd = context["root_fd"]
    staging_fd = context["container_fd"]
    source_fd = context["tree_fd"]
    snapshots_info = os.fstat(snapshots_fd)
    if directory_identity(snapshots_info) != expected_snapshots_identity:
        raise RuntimeError("snapshots-root identity changed before assembly")
    if directory_identity(os.fstat(staging_fd)) != expected_staging_identity:
        raise RuntimeError("snapshot-staging identity changed before assembly")
    if directory_identity(os.fstat(source_fd)) != expected_source_identity:
        raise RuntimeError("staging-source identity changed before assembly")
    canonical_snapshots_root = lexical_absolute(
        snapshots_root_text, label="snapshots root"
    )
    if snapshots_root != canonical_snapshots_root:
        raise RuntimeError("receiver and assembly snapshots roots differ")
    if snapshots_root.name != "snapshots":
        raise RuntimeError("snapshot root basename must be 'snapshots'")
    task_root_public = snapshots_root.parent
    task_fd = os.open("..", directory_flags | nofollow, dir_fd=snapshots_fd)
    task_identity = directory_identity(os.fstat(task_fd))
    verify_isolated_directory(
        os.fstat(task_fd), label="held task root", require_owner=True
    )
    _, public_task_fd, public_task_info = open_absolute_directory(
        str(task_root_public), label="task root", isolated=True
    )
    try:
        if directory_identity(public_task_info) != task_identity:
            raise RuntimeError("public task-root path differs from held task root")
    finally:
        os.close(public_task_fd)
    task_snapshots_fd = open_child_directory(
        task_fd,
        "snapshots",
        expected_snapshots_identity,
        label="task snapshots root",
    )
    os.close(task_snapshots_fd)
    # fchdir pins every relative Path operation below to the already opened
    # staging directory.  Renaming any lexical ancestor cannot redirect a read,
    # write, chmod, unlink, or module load into a replacement hierarchy.
    os.fchdir(staging_fd)
    if directory_identity(os.stat(".")) != expected_staging_identity:
        raise RuntimeError("working directory differs from snapshot staging")
    validate_child_name(target_name, label="snapshot target")
    if target_name != tree_sha256:
        raise RuntimeError("snapshot target name differs from source digest")
    target_public = snapshots_root / target_name
    # All mutation below is relative to the descriptor-anchored working
    # directory.  The separately held source fd supplies repeated identity
    # checks before sealing and publication.
    staging = pathlib.Path(".")
    source = pathlib.Path("source")

    def verify_bound_staging():
        current_staging = os.stat(
            staging_name, dir_fd=snapshots_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(current_staging.st_mode)
            or directory_identity(current_staging) != expected_staging_identity
            or directory_identity(os.fstat(staging_fd)) != expected_staging_identity
        ):
            raise RuntimeError("snapshot staging identity changed during assembly")
        current_source = os.stat(
            "source", dir_fd=staging_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(current_source.st_mode)
            or directory_identity(current_source) != expected_source_identity
            or directory_identity(os.fstat(source_fd)) != expected_source_identity
        ):
            raise RuntimeError("staging source identity changed during assembly")

    def require_target_missing():
        try:
            info = os.stat(
                target_name, dir_fd=snapshots_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            return
        kind = "symbolic link" if stat.S_ISLNK(info.st_mode) else "filesystem entry"
        raise RuntimeError(f"snapshot target appeared concurrently as a {kind}")

    verify_bound_staging()
    require_target_missing()
    runtime_binary_fd = -1
    runtime_binary_identity = None
    runtime_binary_source_public = None
    runtime_validation_evidence = None
    runtime_validation_evidence_sha256 = None

    def file_identity(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    def file_sha256(path: pathlib.Path) -> str:
        digest = hashlib.sha256()
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"hash input is not a regular file: {path}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if file_identity(opened) != file_identity(before):
                raise RuntimeError(f"hash input changed while opened: {path}")
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(descriptor)
        if file_identity(path.lstat()) != file_identity(before):
            raise RuntimeError(f"hash input changed while read: {path}")
        return digest.hexdigest()

    def read_regular_child_nofollow(
        parent_fd, name, *, label, maximum_bytes=None, require_readonly=False
    ):
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {name}")
        if require_readonly and before.st_mode & write_bits:
            raise RuntimeError(f"{label} is writable: {name}")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise RuntimeError(f"{label} is too large: {name}")
        descriptor = os.open(name, os.O_RDONLY | nofollow, dir_fd=parent_fd)
        try:
            opened = os.fstat(descriptor)
            if file_identity(opened) != file_identity(before):
                raise RuntimeError(f"{label} changed while opened: {name}")
            if require_readonly and opened.st_mode & write_bits:
                raise RuntimeError(f"opened {label} is writable: {name}")
            chunks = []
            total_bytes = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
                total_bytes += len(chunk)
                if maximum_bytes is not None and total_bytes > maximum_bytes:
                    raise RuntimeError(f"{label} grew beyond its size limit: {name}")
            after_open = os.fstat(descriptor)
            if file_identity(after_open) != file_identity(before):
                raise RuntimeError(f"{label} changed on its held descriptor: {name}")
        finally:
            os.close(descriptor)
        after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if file_identity(after) != file_identity(before):
            raise RuntimeError(f"{label} changed while read: {name}")
        if require_readonly and after.st_mode & write_bits:
            raise RuntimeError(f"{label} became writable: {name}")
        return b"".join(chunks)

    def exact_regular_inventory(parent_fd, names, *, label, require_readonly):
        actual_names = sorted(os.listdir(parent_fd))
        if actual_names != sorted(names):
            raise RuntimeError(
                f"{label} does not contain the exact required file set"
            )
        result = []
        for name in names:
            validate_child_name(name, label=f"{label} member")
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"{label} member is not regular: {name}")
            if require_readonly and before.st_mode & write_bits:
                raise RuntimeError(f"{label} member is writable: {name}")
            descriptor = os.open(
                name, os.O_RDONLY | nofollow, dir_fd=parent_fd
            )
            digest = hashlib.sha256()
            byte_count = 0
            try:
                opened = os.fstat(descriptor)
                if file_identity(opened) != file_identity(before):
                    raise RuntimeError(
                        f"{label} member changed while opened: {name}"
                    )
                if require_readonly and opened.st_mode & write_bits:
                    raise RuntimeError(f"opened {label} member is writable: {name}")
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
                    byte_count += len(chunk)
                after_open = os.fstat(descriptor)
                if file_identity(after_open) != file_identity(before):
                    raise RuntimeError(
                        f"{label} member changed while read: {name}"
                    )
            finally:
                os.close(descriptor)
            after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if file_identity(after) != file_identity(before):
                raise RuntimeError(
                    f"{label} member changed after read: {name}"
                )
            if byte_count != before.st_size:
                raise RuntimeError(f"{label} member byte count changed: {name}")
            result.append({
                "name": name,
                "bytes": byte_count,
                "sha256": digest.hexdigest(),
            })
        if sorted(os.listdir(parent_fd)) != actual_names:
            raise RuntimeError(f"{label} directory changed while inventoried")
        return result

    def copy_regular_nofollow(source_dir_fd, source_name, target_path) -> None:
        before = os.stat(
            source_name, dir_fd=source_dir_fd, follow_symlinks=False
        )
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"runtime binary is not regular: {source_name}")
        source_fd = os.open(
            source_name, os.O_RDONLY | nofollow, dir_fd=source_dir_fd
        )
        target_fd = -1
        try:
            opened = os.fstat(source_fd)
            if file_identity(opened) != file_identity(before):
                raise RuntimeError(
                    f"runtime binary changed while opened: {source_name}"
                )
            target_fd = os.open(
                target_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                stat.S_IMODE(before.st_mode),
            )
            while chunk := os.read(source_fd, 1024 * 1024):
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    if written < 1:
                        raise OSError("runtime binary copy made no progress")
                    view = view[written:]
            os.fsync(target_fd)
            after = os.stat(
                source_name, dir_fd=source_dir_fd, follow_symlinks=False
            )
            after_open = os.fstat(source_fd)
            if (
                file_identity(after) != file_identity(before)
                or file_identity(after_open) != file_identity(before)
            ):
                raise RuntimeError(
                    f"runtime binary changed while copied: {source_name}"
                )
        finally:
            os.close(source_fd)
            if target_fd >= 0:
                os.close(target_fd)

    def source_digest(root: pathlib.Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        count = 0
        for path in sorted(root.rglob("*")):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise RuntimeError(f"source link appeared while hashing: {path}")
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"source special file appeared while hashing: {path}")
            relative = path.relative_to(root)
            if any(part in excluded for part in relative.parts):
                continue
            if path.suffix.lower() not in suffixes:
                continue
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(b"\0")
            descriptor = os.open(path, os.O_RDONLY | nofollow)
            try:
                if file_identity(os.fstat(descriptor)) != file_identity(info):
                    raise RuntimeError(f"source file changed while opened: {path}")
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
            finally:
                os.close(descriptor)
            if file_identity(path.lstat()) != file_identity(info):
                raise RuntimeError(f"source file changed while hashed: {path}")
            digest.update(b"\0")
            count += 1
        return digest.hexdigest(), count

    def assert_regular_tree_nofollow(root):
        for directory, subdirectories, files in os.walk(
            root, topdown=True, followlinks=False
        ):
            for name in subdirectories:
                path = pathlib.Path(directory) / name
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise RuntimeError(f"staged source directory is unsafe: {path}")
            for name in files:
                path = pathlib.Path(directory) / name
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    raise RuntimeError(f"staged source file is unsafe: {path}")

    verify_bound_staging()
    require_target_missing()
    assert_regular_tree_nofollow(source)
    verify_bound_staging()
    provenance_data = read_regular_child_nofollow(
        staging_fd, "local-provenance.json", label="local provenance record"
    )
    provenance = json.loads(provenance_data.decode("utf-8"))
    if provenance.get("frozen_base_revision") != base_revision:
        raise RuntimeError("local provenance base revision mismatch")
    if provenance.get("deployment_profile") != deployment_profile:
        raise RuntimeError("local provenance deployment profile mismatch")
    normalization_data = read_regular_child_nofollow(
        staging_fd, "source-normalization.json", label="source normalization record"
    )
    if (
        len(normalization_data) != normalization_bytes
        or hashlib.sha256(normalization_data).hexdigest() != normalization_sha256
    ):
        raise RuntimeError("remote source normalization record differs from local staging")
    normalization = json.loads(normalization_data.decode("utf-8"))
    normalizer_path = source / "benchmarks/cc/a100_water8/snapshot_symlink_normalize.py"
    normalizer_spec = importlib.util.spec_from_file_location(
        "water8_snapshot_symlink_normalize_remote", normalizer_path
    )
    normalizer = importlib.util.module_from_spec(normalizer_spec)
    if normalizer_spec.loader is None:
        raise RuntimeError("cannot load staged source normalization validator")
    normalizer_spec.loader.exec_module(normalizer)
    if normalization.get("schema") != normalizer.NORMALIZATION_SCHEMA:
        raise RuntimeError("source normalization schema mismatch")
    if normalization.get("policy") != normalizer.NORMALIZATION_POLICY:
        raise RuntimeError("source normalization policy mismatch")
    inventory = normalization.get("source_link_inventory")
    if not isinstance(inventory, dict):
        raise RuntimeError("source normalization inventory is missing")
    if (
        normalizer.inventory_sha256(inventory)
        != normalization.get("source_link_inventory_sha256")
    ):
        raise RuntimeError("source normalization inventory digest mismatch")
    if normalization.get("materialized_link_count") != inventory.get("link_count"):
        raise RuntimeError("source normalization materialized-link count mismatch")
    if normalization.get("materialized_links") != inventory.get("links"):
        raise RuntimeError("source normalization link identities mismatch")
    if (
        normalization.get("published_symlink_count") != 0
        or normalization.get("published_special_file_count") != 0
    ):
        raise RuntimeError("source normalization does not declare a regular tree")
    normalizer.assert_normalized_source_fd(source_fd)

    pre_runtime_digest, pre_runtime_count = source_digest(source)
    if (
        pre_runtime_digest != tree_sha256
        or pre_runtime_count != tree_file_count
    ):
        raise RuntimeError(
            "received source identity differs before runtime binding"
        )

    runtime_target = source / "gpu4pyscf/lib"
    if (
        not isinstance(required_library_names, list)
        or not required_library_names
        or not all(isinstance(name, str) for name in required_library_names)
        or len(set(required_library_names)) != len(required_library_names)
    ):
        raise RuntimeError("required runtime library list is invalid")

    observed_runtime_inventory = None
    if deployment_profile == "candidate":
        runtime_contract_path = (
            source / "benchmarks/cc/a100_water8/runtime_bundle_contract.py"
        )
        runtime_contract_spec = importlib.util.spec_from_file_location(
            "water8_runtime_bundle_contract_remote", runtime_contract_path
        )
        if runtime_contract_spec is None or runtime_contract_spec.loader is None:
            raise RuntimeError("cannot load staged runtime-bundle contract")
        runtime_contract = importlib.util.module_from_spec(runtime_contract_spec)
        runtime_contract_spec.loader.exec_module(runtime_contract)
        if tuple(required_library_names) != tuple(
            runtime_contract.REQUIRED_LIBRARIES
        ):
            raise RuntimeError(
                "runtime-bundle contract and source library closure differ"
            )

        runtime_binary_source_public = lexical_absolute(
            runtime_binary_source_text, label="candidate runtime bundle root"
        )
        bundle_id = runtime_binary_source_public.name
        if (
            len(bundle_id) != 64
            or any(character not in "0123456789abcdef" for character in bundle_id)
            or runtime_binary_source_public
            != task_root_public / "runtime-bundles" / bundle_id
        ):
            raise RuntimeError(
                "candidate runtime bundle must be a content-addressed child "
                "of the snapshot task root"
            )

        runtime_bundles_fd = open_child_directory(
            task_fd, "runtime-bundles", label="runtime-bundles root"
        )
        try:
            runtime_binary_fd = open_child_directory(
                runtime_bundles_fd,
                bundle_id,
                label="candidate runtime bundle",
            )
        finally:
            os.close(runtime_bundles_fd)
        runtime_binary_info = os.fstat(runtime_binary_fd)
        if runtime_binary_info.st_mode & write_bits:
            raise RuntimeError("candidate runtime bundle directory is writable")
        runtime_binary_identity = directory_identity(runtime_binary_info)

        sidecars_fd = open_child_directory(
            task_fd,
            "runtime-bundle-manifests",
            label="runtime-bundle-manifests root",
        )
        sidecar_name = f"{bundle_id}.json"
        sidecar_public = task_root_public / "runtime-bundle-manifests" / sidecar_name
        try:
            sidecar_payload = read_regular_child_nofollow(
                sidecars_fd,
                sidecar_name,
                label="candidate runtime-bundle sidecar",
                maximum_bytes=4 * 1024 * 1024,
                require_readonly=True,
            )
        finally:
            os.close(sidecars_fd)

        builds_fd = open_child_directory(
            task_fd, "builds", label="runtime build-evidence root"
        )
        build_name = f"{tree_sha256}-gint-bounded-workspace-v2-sm80"
        build_fd = -1
        try:
            build_fd = open_child_directory(
                builds_fd, build_name, label="candidate runtime build root"
            )
            if os.fstat(build_fd).st_mode & write_bits:
                raise RuntimeError("candidate runtime build root is writable")
            build_verification_payload = read_regular_child_nofollow(
                build_fd,
                "build-verification.json",
                label="candidate detached build verification",
                maximum_bytes=8 * 1024 * 1024,
                require_readonly=True,
            )
        finally:
            if build_fd >= 0:
                os.close(build_fd)
            os.close(builds_fd)
        build_verification_public = (
            task_root_public / "builds" / build_name / "build-verification.json"
        )

        observed_runtime_inventory = exact_regular_inventory(
            runtime_binary_fd,
            list(runtime_contract.REQUIRED_LIBRARIES),
            label="candidate runtime bundle",
            require_readonly=True,
        )
        sidecar = runtime_contract.parse_runtime_bundle_sidecar(
            sidecar_payload
        )
        build_verification = runtime_contract.parse_build_verification(
            build_verification_payload
        )
        runtime_validation_evidence = (
            runtime_contract.validate_runtime_bundle_sidecar(
                sidecar,
                observed_runtime_inventory,
                sidecar_payload=sidecar_payload,
                public_bundle_root=str(runtime_binary_source_public),
                public_sidecar_path=str(sidecar_public),
                expected_source_sha256=tree_sha256,
                expected_source_file_count=tree_file_count,
                build_verification=build_verification,
                build_verification_payload=build_verification_payload,
                public_build_verification_path=str(
                    build_verification_public
                ),
            )
        )
        if runtime_validation_evidence.get("build_verification_bound") is not True:
            raise RuntimeError(
                "candidate runtime bundle lacks detached build verification"
            )
        runtime_validation_evidence_sha256 = _canonical_json_sha256(
            runtime_validation_evidence
        )
        binary_names = list(runtime_contract.REQUIRED_LIBRARIES)
    else:
        runtime_binary_source_public, runtime_binary_fd, runtime_binary_info = (
            open_absolute_directory(
                runtime_binary_source_text,
                label="legacy G0 runtime binary source root",
                isolated=True,
            )
        )
        runtime_binary_identity = directory_identity(runtime_binary_info)
        binary_names = sorted(
            name for name in os.listdir(runtime_binary_fd) if name.endswith(".so")
        )
        if not binary_names:
            raise RuntimeError(
                "no GPU4PySCF runtime binaries found under the legacy G0 root"
            )
        for name in binary_names:
            validate_child_name(name, label="runtime binary")
            info = os.stat(name, dir_fd=runtime_binary_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(f"runtime binary is not regular: {name}")
        missing_libraries = sorted(
            set(required_library_names) - set(binary_names)
        )
        if missing_libraries:
            raise RuntimeError(
                "legacy G0 runtime root is missing required libraries: "
                + ", ".join(missing_libraries)
            )
    os.close(task_fd)
    task_fd = -1

    # No staged runtime path is mutated until the candidate sidecar, detached
    # build evidence, source identity, and exact held-fd inventory all pass.
    for existing in source.rglob("*.so"):
        info = existing.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"staged runtime path is not regular: {existing}")
        existing.unlink()
    runtime_target.mkdir(parents=True, exist_ok=True)
    for name in binary_names:
        copy_regular_nofollow(runtime_binary_fd, name, runtime_target / name)

    remote_normalized = normalizer.assert_normalized_source_fd(source_fd)
    assert_regular_tree_nofollow(source)
    verify_bound_staging()
    normalization["remote_verified_after_runtime_binding"] = True
    normalization["uploaded_record_sha256"] = normalization_sha256
    normalization["uploaded_record_bytes"] = normalization_bytes
    normalization["remote_regular_file_count_after_runtime_binding"] = (
        remote_normalized["regular_file_count_after"]
    )
    normalization["remote_directory_count_after_runtime_binding"] = (
        remote_normalized["directory_count_after"]
    )
    snapshot_manifest_path = source / "benchmarks/cc/a100_water8/snapshot_manifest.py"
    snapshot_manifest_spec = importlib.util.spec_from_file_location(
        "water8_snapshot_manifest_remote", snapshot_manifest_path
    )
    snapshot_manifest = importlib.util.module_from_spec(snapshot_manifest_spec)
    if snapshot_manifest_spec.loader is None:
        raise RuntimeError("cannot load staged snapshot-manifest validator")
    snapshot_manifest_spec.loader.exec_module(snapshot_manifest)
    normalization_errors = snapshot_manifest._normalization_errors(
        normalization, source=source
    )
    if normalization_errors:
        raise RuntimeError(
            "remote source normalization proof is invalid: "
            + " | ".join(normalization_errors)
        )

    actual_digest, source_count = source_digest(source)
    if source_count != tree_file_count or actual_digest != tree_sha256:
        raise RuntimeError(f"remote source digest mismatch: {actual_digest} != {tree_sha256}")
    runtime_files = [
        {
            "relative_path": path.relative_to(source).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(source.rglob("*.so"))
        if stat.S_ISREG(path.lstat().st_mode)
    ]
    if not runtime_files:
        raise RuntimeError("staged snapshot has no GPU4PySCF runtime binaries")
    expected_runtime_paths = {
        (runtime_target / name).relative_to(source).as_posix()
        for name in binary_names
    }
    if {entry["relative_path"] for entry in runtime_files} != expected_runtime_paths:
        raise RuntimeError("staged runtime inventory differs from the pinned runtime root")
    inventory_library_names = sorted(entry["relative_path"].rsplit("/", 1)[-1] for entry in runtime_files)
    copied_runtime_inventory = [
        {
            "name": name,
            "bytes": next(
                entry["bytes"] for entry in runtime_files
                if pathlib.PurePosixPath(entry["relative_path"]).name == name
            ),
            "sha256": next(
                entry["sha256"] for entry in runtime_files
                if pathlib.PurePosixPath(entry["relative_path"]).name == name
            ),
        }
        for name in binary_names
    ]
    if (
        deployment_profile == "candidate"
        and copied_runtime_inventory != observed_runtime_inventory
    ):
        raise RuntimeError(
            "copied candidate runtime inventory differs from held bundle"
        )

    # The policy is sealed into the snapshot before publication.  The
    # outcome-specific protocol is retained in a sibling attestation because
    # the protocol is only known after the no-replace rename returns.
    publication_policy = dict(PUBLICATION_POLICY)
    runtime_manifest = {
        "source_root": str(runtime_binary_source_public), "complete": True,
        "required_library_names": required_library_names,
        "inventory_library_names": inventory_library_names,
        "closure_complete": True,
        "files": runtime_files,
    }
    if deployment_profile == "candidate":
        runtime_manifest.update({
            "contract_schema": runtime_contract.BUNDLE_SCHEMA,
            "bundle_id": runtime_validation_evidence["bundle_id"],
            "inventory_sha256": runtime_validation_evidence[
                "inventory_sha256"
            ],
            "sidecar_sha256": runtime_validation_evidence[
                "sidecar_sha256"
            ],
            "build_verification_sha256": runtime_validation_evidence[
                "build_verification_sha256"
            ],
            "validation_evidence": runtime_validation_evidence,
            "validation_evidence_sha256": (
                runtime_validation_evidence_sha256
            ),
        })
    manifest = {
        "schema": snapshot_manifest.SOURCE_SNAPSHOT_SCHEMA,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tree_sha256": tree_sha256,
        "base_revision": base_revision,
        "source": str(target_public / "source"),
        "immutable": True,
        "source_normalization": normalization,
        "local_provenance": provenance,
        "publication_policy": publication_policy,
        "runtime_binaries": runtime_manifest,
    }
    manifest_data = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_fd = os.open(
        "manifest.json",
        os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow,
        0o600,
        dir_fd=staging_fd,
    )
    context["_deferred_manifest_fd"] = manifest_fd
    _write_all(manifest_fd, manifest_data)
    prepublication_manifest_sha256 = hashlib.sha256(manifest_data).hexdigest()
    os.unlink("local-provenance.json", dir_fd=staging_fd)
    os.unlink("source-normalization.json", dir_fd=staging_fd)

    # Seal and re-check staging completely before the only operation that publishes it.
    for directory, subdirectories, files in os.walk(source):
        for name in files:
            path = pathlib.Path(directory) / name
            if path.is_symlink():
                raise RuntimeError(f"source link appeared before sealing: {path}")
            os.chmod(path, 0o444)
        for name in subdirectories:
            path = pathlib.Path(directory) / name
            if path.is_symlink():
                raise RuntimeError(f"source directory link appeared before sealing: {path}")
            os.chmod(path, 0o555)
    os.fchmod(source_fd, 0o555)
    os.chmod(staging / "manifest.json", 0o444)
    os.fchmod(staging_fd, 0o555)
    sealed_digest, sealed_count = source_digest(source)
    if sealed_count != source_count or sealed_digest != tree_sha256:
        raise RuntimeError("sealed staging source differs from verified source")
    normalizer.assert_normalized_source_fd(source_fd)
    for entry in runtime_files:
        path = source / entry["relative_path"]
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != entry["bytes"] \
                or file_sha256(path) != entry["sha256"]:
            raise RuntimeError(f"sealed runtime binary differs: {entry['relative_path']}")
    if deployment_profile == "candidate":
        final_held_inventory = exact_regular_inventory(
            runtime_binary_fd,
            binary_names,
            label="candidate runtime bundle final check",
            require_readonly=True,
        )
        if final_held_inventory != observed_runtime_inventory:
            raise RuntimeError(
                "candidate runtime bundle changed before snapshot publication"
            )
    for root_fd, label in ((staging_fd, "staging"), (source_fd, "source")):
        info = os.fstat(root_fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & write_bits:
            raise RuntimeError(f"writable or invalid {label} root remains after sealing")
    for path in (staging / "manifest.json", *source.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"symbolic link remains in sealed staging: {path}")
        if info.st_mode & write_bits:
            raise RuntimeError(f"writable path remains in sealed staging: {path}")
    if json.loads((staging / "manifest.json").read_text()) != manifest:
        raise RuntimeError("sealed staging manifest differs from assembled manifest")

    # The manifest is complete and read-only before publication.  Keep the
    # descriptor only until this point so a successful rename can never be
    # followed by a mutation of the published snapshot.
    os.close(manifest_fd)
    manifest_fd = -1
    context.pop("_deferred_manifest_fd", None)
    published_manifest_sha256 = prepublication_manifest_sha256

    verify_bound_staging()
    require_target_missing()
    _, final_snapshots_fd, final_snapshots_info = open_absolute_directory(
        snapshots_root_text, label="snapshots root", isolated=True
    )
    _, final_runtime_fd, final_runtime_info = open_absolute_directory(
        runtime_binary_source_text,
        label="runtime binary source root",
        isolated=True,
    )
    try:
        if directory_identity(final_snapshots_info) != expected_snapshots_identity:
            raise RuntimeError("snapshots-root path identity changed before publish")
        if directory_identity(final_runtime_info) != runtime_binary_identity:
            raise RuntimeError("runtime-binary root changed before publish")
        if (
            deployment_profile == "candidate"
            and final_runtime_info.st_mode & write_bits
        ):
            raise RuntimeError("candidate runtime bundle became writable")
    finally:
        os.close(final_snapshots_fd)
        os.close(final_runtime_fd)

    publish_protocol = _publish_directory_noreplace(
        snapshots_fd,
        staging_fd,
        staging_name,
        target_name,
        expected_staging_identity,
    )
    publication_attestation = {
        "schema": PUBLICATION_ATTESTATION_SCHEMA,
        "status": "published",
        "source_tree_sha256": tree_sha256,
        "snapshot_name": target_name,
        "published_directory_identity": {
            "device": expected_staging_identity[0],
            "inode": expected_staging_identity[1],
        },
        "publish_protocol": publish_protocol,
        "trust_boundary": PUBLISH_TRUST_BOUNDARY,
        "publication_policy_sha256": _canonical_json_sha256(
            publication_policy
        ),
        "published_manifest_sha256": prepublication_manifest_sha256,
        "prepublication_manifest_sha256": prepublication_manifest_sha256,
    }
    publication_attestation["attestation_sha256"] = _canonical_json_sha256(
        publication_attestation
    )
    attestation_name = f"{target_name}{PUBLICATION_ATTESTATION_SUFFIX}"
    attestation_fd = os.open(
        attestation_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
        0o600,
        dir_fd=snapshots_fd,
    )
    try:
        attestation_data = (
            json.dumps(
                publication_attestation,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
        _write_all(attestation_fd, attestation_data)
        os.fchmod(attestation_fd, 0o444)
    finally:
        os.close(attestation_fd)
    if read_regular_child_nofollow(
        snapshots_fd, attestation_name, label="publication attestation"
    ) != attestation_data:
        raise RuntimeError("publication attestation changed after creation")
    published_fd = open_child_directory(
        snapshots_fd, target_name, expected_staging_identity,
        label="published snapshot",
    )
    try:
        published_source_fd = open_child_directory(
            published_fd, "source", expected_source_identity,
            label="published snapshot source",
        )
        os.close(published_source_fd)
    finally:
        os.close(published_fd)

    # Re-open the public root once more after publication.  The actual rename
    # used the held descriptor, so an ancestor swap can never redirect it; this
    # final check makes such a swap fail closed instead of reporting success for
    # an unreachable publication.
    _, published_snapshots_fd, published_snapshots_info = open_absolute_directory(
        snapshots_root_text, label="published snapshots root", isolated=True
    )
    try:
        if directory_identity(published_snapshots_info) != expected_snapshots_identity:
            raise RuntimeError("snapshots-root path identity changed after publish")
    finally:
        os.close(published_snapshots_fd)

    os.close(runtime_binary_fd)
    result = {
        "published_snapshot": target_name,
        "tree_sha256": tree_sha256,
        "source_file_count": source_count,
        "runtime_library_count": len(runtime_files),
        "runtime_contract_validated": deployment_profile == "candidate",
        "runtime_bundle_id": (
            runtime_validation_evidence["bundle_id"]
            if runtime_validation_evidence is not None else None
        ),
        "runtime_inventory_sha256": (
            runtime_validation_evidence["inventory_sha256"]
            if runtime_validation_evidence is not None else None
        ),
        "runtime_sidecar_sha256": (
            runtime_validation_evidence["sidecar_sha256"]
            if runtime_validation_evidence is not None else None
        ),
        "runtime_build_verification_sha256": (
            runtime_validation_evidence["build_verification_sha256"]
            if runtime_validation_evidence is not None else None
        ),
        "runtime_validation_evidence_sha256": (
            runtime_validation_evidence_sha256
        ),
        "publish_protocol": publish_protocol,
        "publish_trust_boundary": PUBLISH_TRUST_BOUNDARY,
        "publication_attestation": publication_attestation,
        "publication_attestation_name": attestation_name,
        "publication_attestation_sha256": publication_attestation[
            "attestation_sha256"
        ],
        "publication_attestation_file_sha256": hashlib.sha256(
            attestation_data
        ).hexdigest(),
        "published_manifest_sha256": published_manifest_sha256,
    }
    return result



def post_receive(
    context: dict[str, Any], argv: Sequence[str]
) -> dict[str, Any]:
    """Run runtime binding, validation, sealing, and publish on held fds."""

    if len(argv) != 9:
        raise ValueError("snapshot assembly requires exactly nine arguments")
    root_dev, root_ino = context["root_identity"]
    container_dev, container_ino = context["container_identity"]
    tree_dev, tree_ino = context["tree_identity"]
    legacy_argv = [
        "snapshot_remote_assembly.py",
        str(context["destination_root"]),
        str(context["container_name"]),
        str(root_dev),
        str(root_ino),
        str(container_dev),
        str(container_ino),
        str(tree_dev),
        str(tree_ino),
        *map(str, argv),
    ]
    previous_argv = sys.argv
    previous_cwd_fd = os.open(
        ".", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    sys.argv = legacy_argv
    try:
        return _assemble(context)
    finally:
        deferred_manifest_fd = context.pop("_deferred_manifest_fd", None)
        if deferred_manifest_fd is not None:
            os.close(deferred_manifest_fd)
        sys.argv = previous_argv
        os.fchdir(previous_cwd_fd)
        os.close(previous_cwd_fd)
