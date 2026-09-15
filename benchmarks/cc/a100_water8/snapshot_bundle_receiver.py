#!/usr/bin/env python3
"""Build and receive a no-link source bundle through anchored directory fds.

The deploy script embeds this module plus a compressed tar payload in one
Python program and streams that program to the remote interpreter.  The remote
process keeps the deployment-user private anchor, destination root, container,
and tree descriptors open for the complete receive.  It creates only validated
directories and regular files with mkdirat/openat and never resolves a payload
path through the filesystem.

The receiver is generic: callers may use any identity-bound destination root,
container, and tree name.  The snapshot deploy uses ``snapshots/<random>/source``;
the same protocol can populate an identity-bound ``build-inputs`` container.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
from typing import Any, Callable, Iterable, NamedTuple, NoReturn, Sequence


BUNDLE_SCHEMA = "gpu4pyscf.regular-tree-bundle.v1"
RECEIPT_SCHEMA = "gpu4pyscf.regular-tree-bundle-receipt.v1"
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_OTHER_WRITE = stat.S_IWGRP | stat.S_IWOTH
_OTHER_ACCESS = stat.S_IRWXG | stat.S_IRWXO
_MAX_MEMBERS = 200_000
_MAX_BUNDLE_BYTES = 2 * 1024**3
_MAX_UNPACKED_BYTES = 8 * 1024**3
_CHUNK = 1024 * 1024


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _dir_identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _parse_identity(device: str | int, inode: str | int) -> tuple[int, int]:
    try:
        result = int(device), int(inode)
    except (TypeError, ValueError) as exc:
        raise ValueError("directory identity must contain integers") from exc
    if result[0] < 0 or result[1] <= 0:
        raise ValueError("directory identity is outside the accepted range")
    return result


def _safe_child_name(value: str, *, label: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or os.sep in value
        or (os.altsep is not None and os.altsep in value)
        or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or Path(value).name != value
    ):
        raise ValueError(f"{label} must be one safe path component")
    return value


def _lexical_absolute(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if (
        not raw
        or not os.path.isabs(raw)
        or os.path.normpath(raw) != raw
        or "\x00" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise ValueError("destination root must be a normalized absolute path")
    if sys.platform == "darwin":
        for alias, canonical in (
            ("/var", "/private/var"),
            ("/tmp", "/private/tmp"),
            ("/etc", "/private/etc"),
        ):
            if raw == alias or raw.startswith(alias + os.sep):
                raw = canonical + raw[len(alias):]
                break
    return Path(raw)


def _verify_isolated(info: os.stat_result, *, label: str) -> None:
    if info.st_uid != os.geteuid():
        _fail(f"{label} is not owned by the deployment user")
    if info.st_mode & _OTHER_WRITE:
        _fail(f"{label} is writable by another principal")


def _open_isolated_root(
    value: str | os.PathLike[str], expected: tuple[int, int]
) -> tuple[Path, int]:
    """Walk from /, then retain the first euid-owned mode-700-like anchor."""

    root = _lexical_absolute(value)
    descriptor = os.open("/", _DIRECTORY_FLAGS | _NOFOLLOW)
    private_anchor_found = False
    try:
        for component in root.parts[1:]:
            before = os.stat(
                component, dir_fd=descriptor, follow_symlinks=False
            )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                _fail(f"destination-root component is unsafe: {component}")
            if (
                not private_anchor_found
                and before.st_uid == os.geteuid()
                and not (before.st_mode & _OTHER_ACCESS)
            ):
                private_anchor_found = True
            if private_anchor_found:
                _verify_isolated(
                    before, label=f"isolated destination component {component}"
                )
            child = os.open(
                component, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=descriptor
            )
            opened = os.fstat(child)
            if _dir_identity(opened) != _dir_identity(before):
                os.close(child)
                _fail(f"destination-root component changed: {component}")
            if private_anchor_found:
                _verify_isolated(
                    opened,
                    label=f"opened isolated destination component {component}",
                )
            os.close(descriptor)
            descriptor = child
        if not private_anchor_found:
            _fail("destination root has no deployment-user-owned private anchor")
        if _dir_identity(os.fstat(descriptor)) != expected:
            _fail("destination-root identity mismatch")
        return root, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_bound_child(
    parent_fd: int,
    name: str,
    expected: tuple[int, int],
    *,
    label: str,
) -> int:
    name = _safe_child_name(name, label=label)
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        _fail(f"{label} is not a no-link directory")
    _verify_isolated(before, label=label)
    descriptor = os.open(
        name, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=parent_fd
    )
    opened = os.fstat(descriptor)
    if _dir_identity(opened) != _dir_identity(before):
        os.close(descriptor)
        _fail(f"{label} changed while opened")
    if _dir_identity(opened) != expected:
        os.close(descriptor)
        _fail(f"{label} identity mismatch")
    _verify_isolated(opened, label=label)
    return descriptor


def _read_regular_stable(path: Path, *, label: str) -> tuple[bytes, int]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW)
    try:
        if _file_identity(os.fstat(descriptor)) != _file_identity(before):
            raise ValueError(f"{label} changed while opened")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _CHUNK):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    if _file_identity(path.lstat()) != _file_identity(before):
        raise ValueError(f"{label} changed while read")
    return b"".join(chunks), stat.S_IMODE(before.st_mode)


def _source_entries(source: Path) -> tuple[list[tuple[str, int]], list[tuple[str, bytes, int]]]:
    directories: list[tuple[str, int]] = []
    files: list[tuple[str, bytes, int]] = []
    for directory, directory_names, file_names in os.walk(
        source, topdown=True, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = Path(directory) / name
            relative = path.relative_to(source).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"bundle source directory is unsafe: {relative}")
            directories.append((f"tree/{relative}", stat.S_IMODE(info.st_mode)))
        for name in file_names:
            path = Path(directory) / name
            relative = path.relative_to(source).as_posix()
            data, mode = _read_regular_stable(
                path, label=f"bundle source file {relative}"
            )
            files.append((f"tree/{relative}", data, mode))
    directories.sort(key=lambda item: (item[0].count("/"), item[0]))
    files.sort(key=lambda item: item[0])
    return directories, files


def build_bundle(
    source_root: str | os.PathLike[str],
    provenance_file: str | os.PathLike[str],
    normalization_file: str | os.PathLike[str],
) -> bytes:
    """Return a deterministic gzip-compressed tar of regular source inputs."""

    source = Path(os.path.abspath(os.fspath(source_root)))
    if source.resolve(strict=True) != source or not source.is_dir():
        raise ValueError("bundle source root must be a canonical directory")
    directories, files = _source_entries(source)
    for archive_name, value, label in (
        ("local-provenance.json", Path(provenance_file), "local provenance"),
        (
            "source-normalization.json",
            Path(normalization_file),
            "source normalization",
        ),
    ):
        data, mode = _read_regular_stable(value, label=label)
        files.append((archive_name, data, mode))
    files.sort(key=lambda item: item[0])

    output = BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(
            fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
        ) as archive:
            metadata = tarfile.TarInfo(".bundle-metadata.json")
            metadata_data = (
                json.dumps(
                    {"schema": BUNDLE_SCHEMA},
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n"
            ).encode("ascii")
            metadata.type = tarfile.REGTYPE
            metadata.mode = 0o600
            metadata.size = len(metadata_data)
            metadata.mtime = metadata.uid = metadata.gid = 0
            metadata.uname = metadata.gname = ""
            archive.addfile(metadata, BytesIO(metadata_data))
            for name, mode in directories:
                entry = tarfile.TarInfo(name)
                entry.type = tarfile.DIRTYPE
                entry.mode = mode & 0o777
                entry.mtime = entry.uid = entry.gid = 0
                entry.uname = entry.gname = ""
                archive.addfile(entry)
            for name, data, mode in files:
                entry = tarfile.TarInfo(name)
                entry.type = tarfile.REGTYPE
                entry.mode = mode & 0o777
                entry.size = len(data)
                entry.mtime = entry.uid = entry.gid = 0
                entry.uname = entry.gname = ""
                archive.addfile(entry, BytesIO(data))
    return output.getvalue()


class _ValidatedMember(NamedTuple):
    member: tarfile.TarInfo
    area: str
    parts: tuple[str, ...]


def _member_path(member: tarfile.TarInfo) -> _ValidatedMember:
    name = member.name
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise ValueError("bundle member path is invalid")
    pure = PurePosixPath(name)
    if pure.is_absolute() or pure.as_posix() != name or ".." in pure.parts:
        raise ValueError(f"bundle member path is unsafe: {name}")
    if name in {
        ".bundle-metadata.json",
        "local-provenance.json",
        "source-normalization.json",
    }:
        return _ValidatedMember(member, "container", (name,))
    if len(pure.parts) < 2 or pure.parts[0] != "tree":
        raise ValueError(f"bundle member is outside the tree: {name}")
    return _ValidatedMember(member, "tree", tuple(pure.parts[1:]))


def _validate_archive(archive: tarfile.TarFile) -> list[_ValidatedMember]:
    members = archive.getmembers()
    if not members or len(members) > _MAX_MEMBERS:
        raise ValueError("bundle member count is outside the accepted range")
    validated: list[_ValidatedMember] = []
    seen: set[str] = set()
    directories: set[str] = set()
    unpacked = 0
    for member in members:
        item = _member_path(member)
        if member.name in seen:
            raise ValueError(f"duplicate bundle member: {member.name}")
        seen.add(member.name)
        if member.isdir():
            if item.area != "tree":
                raise ValueError("container-level bundle member may not be a directory")
            directories.add(member.name)
        elif member.isreg():
            if member.size < 0:
                raise ValueError(f"bundle member has negative size: {member.name}")
            unpacked += member.size
            if unpacked > _MAX_UNPACKED_BYTES:
                raise ValueError("bundle expands beyond the accepted byte limit")
        else:
            raise ValueError(f"bundle member is not regular/dir: {member.name}")
        validated.append(item)
    required = {
        ".bundle-metadata.json",
        "local-provenance.json",
        "source-normalization.json",
    }
    if not required.issubset(seen):
        raise ValueError("bundle metadata/provenance/normalization is incomplete")
    for item in validated:
        if item.area != "tree" or len(item.parts) < 2:
            continue
        for length in range(1, len(item.parts)):
            parent = "tree/" + "/".join(item.parts[:length])
            if parent not in directories:
                raise ValueError(f"bundle member lacks directory parent: {item.member.name}")
    return validated


def _open_relative_directory(base_fd: int, parts: Iterable[str]) -> int:
    descriptor = os.dup(base_fd)
    try:
        for component in parts:
            child = os.open(
                component, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=descriptor
            )
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                _fail(f"bundle parent is not a directory: {component}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_directory(base_fd: int, parts: tuple[str, ...]) -> None:
    parent_fd = _open_relative_directory(base_fd, parts[:-1])
    try:
        os.mkdir(parts[-1], 0o700, dir_fd=parent_fd)
        child_fd = os.open(
            parts[-1], _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=parent_fd
        )
        try:
            info = os.fstat(child_fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
                _fail(f"created bundle directory identity is invalid: {parts[-1]}")
        finally:
            os.close(child_fd)
    finally:
        os.close(parent_fd)


def _create_regular(
    base_fd: int,
    parts: tuple[str, ...],
    member: tarfile.TarInfo,
    archive: tarfile.TarFile,
) -> tuple[int, str]:
    parent_fd = _open_relative_directory(base_fd, parts[:-1])
    descriptor = -1
    digest = hashlib.sha256()
    try:
        descriptor = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"bundle regular member has no data: {member.name}")
        copied = 0
        with source:
            while chunk := source.read(_CHUNK):
                copied += len(chunk)
                if copied > member.size:
                    raise ValueError(
                        f"bundle member exceeds declared size: {member.name}"
                    )
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written < 1:
                        raise OSError("bundle receive write made no progress")
                    view = view[written:]
        if copied != member.size:
            raise ValueError(f"bundle member size mismatch: {member.name}")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        current = os.stat(
            parts[-1], dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or _file_identity(current) != _file_identity(opened)
        ):
            _fail(f"received bundle file identity changed: {member.name}")
        return copied, digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _scan_regular_tree(directory_fd: int) -> tuple[int, int, int]:
    directories = files = byte_count = 0
    for name in os.listdir(directory_fd):
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child = os.open(
                name, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=directory_fd
            )
            try:
                if _dir_identity(os.fstat(child)) != _dir_identity(info):
                    _fail(f"received directory changed while opened: {name}")
                child_dirs, child_files, child_bytes = _scan_regular_tree(child)
            finally:
                os.close(child)
            directories += 1 + child_dirs
            files += child_files
            byte_count += child_bytes
        elif stat.S_ISREG(info.st_mode):
            files += 1
            byte_count += info.st_size
        else:
            _fail(f"received tree contains a link/special file: {name}")
    return directories, files, byte_count


def receive_bundle(
    payload: bytes,
    *,
    destination_root: str,
    container_name: str,
    root_identity: tuple[int, int],
    container_identity: tuple[int, int],
    tree_name: str,
    tree_identity: tuple[int, int],
    expected_sha256: str,
    expected_bytes: int,
    post_receive: Callable[[dict[str, Any], Sequence[str]], dict[str, Any]]
    | None = None,
    post_receive_args: Sequence[str] = (),
) -> dict[str, Any]:
    if expected_bytes < 1 or expected_bytes > _MAX_BUNDLE_BYTES:
        raise ValueError("embedded bundle byte count is outside the accepted range")
    if len(payload) != expected_bytes:
        raise ValueError("embedded bundle byte count mismatch")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("embedded bundle SHA-256 mismatch")
    container_name = _safe_child_name(container_name, label="container name")
    tree_name = _safe_child_name(tree_name, label="tree name")
    root, root_fd = _open_isolated_root(destination_root, root_identity)
    container_fd = tree_fd = -1
    try:
        container_fd = _open_bound_child(
            root_fd, container_name, container_identity, label="bundle container"
        )
        tree_fd = _open_bound_child(
            container_fd, tree_name, tree_identity, label="bundle tree"
        )
        if os.listdir(tree_fd):
            raise ValueError("bundle tree is not empty before receive")
        if set(os.listdir(container_fd)) != {tree_name}:
            raise ValueError("bundle container has unexpected pre-existing entries")
        with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
            members = _validate_archive(archive)
            directories = sorted(
                (item for item in members if item.member.isdir()),
                key=lambda item: (len(item.parts), item.member.name),
            )
            regulars = sorted(
                (item for item in members if item.member.isreg()),
                key=lambda item: item.member.name,
            )
            for item in directories:
                _create_directory(tree_fd, item.parts)
            received_files: list[dict[str, Any]] = []
            for item in regulars:
                base_fd = tree_fd if item.area == "tree" else container_fd
                copied, digest = _create_regular(
                    base_fd, item.parts, item.member, archive
                )
                received_files.append({
                    "relative_path": item.member.name,
                    "bytes": copied,
                    "sha256": digest,
                })
        metadata_fd = os.open(
            ".bundle-metadata.json", os.O_RDONLY | _NOFOLLOW,
            dir_fd=container_fd,
        )
        try:
            metadata = json.loads(os.read(metadata_fd, 4096).decode("ascii"))
        finally:
            os.close(metadata_fd)
        if metadata != {"schema": BUNDLE_SCHEMA}:
            raise ValueError("bundle protocol metadata mismatch")
        os.unlink(".bundle-metadata.json", dir_fd=container_fd)
        if set(os.listdir(container_fd)) != {
            tree_name,
            "local-provenance.json",
            "source-normalization.json",
        }:
            raise ValueError("received bundle container inventory mismatch")
        directory_count, file_count, unpacked_bytes = _scan_regular_tree(tree_fd)
        current_container = os.stat(
            container_name, dir_fd=root_fd, follow_symlinks=False
        )
        current_tree = os.stat(
            tree_name, dir_fd=container_fd, follow_symlinks=False
        )
        if (
            _dir_identity(current_container) != container_identity
            or _dir_identity(os.fstat(container_fd)) != container_identity
            or _dir_identity(current_tree) != tree_identity
            or _dir_identity(os.fstat(tree_fd)) != tree_identity
        ):
            _fail("bundle destination identity changed during receive")
        _, final_root_fd = _open_isolated_root(
            destination_root, root_identity
        )
        os.close(final_root_fd)
        result: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "destination_root": str(root),
            "container_name": container_name,
            "tree_name": tree_name,
            "bundle_sha256": actual_sha256,
            "bundle_bytes": len(payload),
            "tree_directory_count": directory_count,
            "tree_regular_file_count": file_count,
            "tree_regular_bytes": unpacked_bytes,
            "received_files": received_files,
            "symlink_count": 0,
            "special_file_count": 0,
        }
        if post_receive is not None:
            post_result = post_receive(
                {
                    "destination_root": root,
                    "container_name": container_name,
                    "tree_name": tree_name,
                    "root_fd": root_fd,
                    "container_fd": container_fd,
                    "tree_fd": tree_fd,
                    "root_identity": root_identity,
                    "container_identity": container_identity,
                    "tree_identity": tree_identity,
                },
                tuple(post_receive_args),
            )
            if not isinstance(post_result, dict):
                raise TypeError("post-receive callback must return a dictionary")
            result["post_receive"] = post_result
        return result
    finally:
        if tree_fd >= 0:
            os.close(tree_fd)
        if container_fd >= 0:
            os.close(container_fd)
        os.close(root_fd)


def build_embedded_program(
    receiver_source: bytes,
    payload: bytes,
    post_receive_source: bytes | None = None,
) -> bytes:
    source64 = base64.b64encode(receiver_source).decode("ascii")
    payload64 = base64.b64encode(payload).decode("ascii")
    post_receive_setup = "post_receive=None\n"
    if post_receive_source is not None:
        post64 = base64.b64encode(post_receive_source).decode("ascii")
        post_receive_setup = (
            "post_namespace={'__name__':'gpu4pyscf_embedded_snapshot_assembly'}\n"
            f"post_source=base64.b64decode('{post64}')\n"
            "exec(compile(post_source,'<embedded-snapshot-assembly>',"
            "'exec'),post_namespace)\n"
            "post_receive=post_namespace['post_receive']\n"
        )
    program = (
        "import base64,sys\n"
        "namespace={'__name__':'gpu4pyscf_embedded_bundle_receiver'}\n"
        f"source=base64.b64decode('{source64}')\n"
        "exec(compile(source,'<embedded-bundle-receiver>','exec'),namespace)\n"
        f"payload=base64.b64decode('{payload64}')\n"
        f"{post_receive_setup}"
        "raise SystemExit(namespace['_embedded_main'](sys.argv[1:],payload,"
        "post_receive=post_receive))\n"
    )
    return program.encode("ascii")


def _receive_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("destination_root")
    parser.add_argument("container_name")
    parser.add_argument("root_dev")
    parser.add_argument("root_ino")
    parser.add_argument("container_dev")
    parser.add_argument("container_ino")
    parser.add_argument("tree_name")
    parser.add_argument("tree_dev")
    parser.add_argument("tree_ino")
    parser.add_argument("bundle_sha256")
    parser.add_argument("bundle_bytes", type=int)
    return parser


def _embedded_main(
    argv: Sequence[str],
    payload: bytes,
    *,
    post_receive: Callable[[dict[str, Any], Sequence[str]], dict[str, Any]]
    | None = None,
) -> int:
    parser = _receive_parser()
    args, remaining = parser.parse_known_args(argv)
    if post_receive is None and remaining:
        parser.error("unexpected post-receive arguments")
    result = receive_bundle(
        payload,
        destination_root=args.destination_root,
        container_name=args.container_name,
        root_identity=_parse_identity(args.root_dev, args.root_ino),
        container_identity=_parse_identity(args.container_dev, args.container_ino),
        tree_name=args.tree_name,
        tree_identity=_parse_identity(args.tree_dev, args.tree_ino),
        expected_sha256=args.bundle_sha256,
        expected_bytes=args.bundle_bytes,
        post_receive=post_receive,
        post_receive_args=remaining,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build-program")
    build.add_argument("--source-root", required=True)
    build.add_argument("--provenance", required=True)
    build.add_argument("--normalization", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--post-receive-script")
    receive = subparsers.add_parser("receive")
    for action in (receive,):
        action.add_argument("destination_root")
        action.add_argument("container_name")
        action.add_argument("root_dev")
        action.add_argument("root_ino")
        action.add_argument("container_dev")
        action.add_argument("container_ino")
        action.add_argument("tree_name")
        action.add_argument("tree_dev")
        action.add_argument("tree_ino")
        action.add_argument("bundle_sha256")
        action.add_argument("bundle_bytes", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "build-program":
        payload = build_bundle(
            args.source_root, args.provenance, args.normalization
        )
        receiver_source, _ = _read_regular_stable(
            Path(__file__).resolve(), label="bundle receiver source"
        )
        post_receive_source = None
        if args.post_receive_script:
            post_receive_source, _ = _read_regular_stable(
                Path(args.post_receive_script).resolve(),
                label="post-receive script",
            )
        program = build_embedded_program(
            receiver_source, payload, post_receive_source
        )
        output = Path(args.output)
        before = output.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("embedded receiver output must be a regular file")
        descriptor = os.open(output, os.O_WRONLY | os.O_TRUNC | _NOFOLLOW)
        try:
            view = memoryview(program)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("embedded receiver program write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        print(json.dumps({
            "schema": BUNDLE_SCHEMA,
            "bundle_sha256": hashlib.sha256(payload).hexdigest(),
            "bundle_bytes": len(payload),
            "program_bytes": len(program),
            "post_receive_embedded": post_receive_source is not None,
        }, sort_keys=True, separators=(",", ":")))
        return 0
    payload = sys.stdin.buffer.read()
    return _embedded_main([
        args.destination_root,
        args.container_name,
        args.root_dev,
        args.root_ino,
        args.container_dev,
        args.container_ino,
        args.tree_name,
        args.tree_dev,
        args.tree_ino,
        args.bundle_sha256,
        str(args.bundle_bytes),
    ], payload)


if __name__ == "__main__":
    raise SystemExit(main())
