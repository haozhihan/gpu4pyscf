#!/usr/bin/env python3
"""Validate content-addressed, read-only benchmark source snapshots."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Sequence


FROZEN_BASE_COMMIT = "a89b3ae018d4e82968323ef95645d91adde294aa"
G0_DEPLOYMENT_PROFILE = "g0-canonical-pristine"
CANDIDATE_DEPLOYMENT_PROFILE = "candidate"
SOURCE_SNAPSHOT_SCHEMA = "gpu4pyscf.source-snapshot.v3"
LEGACY_SOURCE_SNAPSHOT_SCHEMA = "gpu4pyscf.source-snapshot.v2"
PUBLICATION_POLICY_SCHEMA = "gpu4pyscf.snapshot-publication-policy.v1"
PUBLICATION_ATTESTATION_SCHEMA = "gpu4pyscf.snapshot-publication-attestation.v1"
PUBLICATION_ATTESTATION_SUFFIX = ".publication-attestation.json"
PUBLISH_TRUST_BOUNDARY = "private-anchor-owner-and-root-are-trusted-cooperators"
PUBLICATION_PROTOCOLS = {
    "atomic-exclusive-rename",
    "atomic-exclusive-rename-commit-confirmed",
    "reserved-empty-directory-rename",
    "reserved-empty-directory-rename-commit-confirmed",
}
MAX_PUBLICATION_ATTESTATION_BYTES = 64 * 1024
PUBLICATION_ATTESTATION_KEYS = {
    "schema",
    "status",
    "source_tree_sha256",
    "snapshot_name",
    "published_directory_identity",
    "publish_protocol",
    "trust_boundary",
    "publication_policy_sha256",
    "published_manifest_sha256",
    "prepublication_manifest_sha256",
    "attestation_sha256",
}
PUBLICATION_POLICY = {
    "schema": PUBLICATION_POLICY_SCHEMA,
    "trust_boundary": PUBLISH_TRUST_BOUNDARY,
    "attestation_schema": PUBLICATION_ATTESTATION_SCHEMA,
    "attestation_location": (
        "snapshots-root sibling named "
        "<snapshot-name>.publication-attestation.json"
    ),
}
SOURCE_NORMALIZATION_SCHEMA = "gpu4pyscf.source-symlink-normalization.v1"
SOURCE_NORMALIZATION_POLICY = "validated-relative-leaf-links-materialized"
DEPLOYMENT_PROFILES = {
    G0_DEPLOYMENT_PROFILE,
    CANDIDATE_DEPLOYMENT_PROFILE,
}
REQUIRED_RUNTIME_LIBRARIES = (
    "libcupy_helper.so",
    "libgdft.so",
    "libgecp.so",
    "libgint.so",
    "libgvhf.so",
    "libgvhf_md.so",
    "libgvhf_rys.so",
    "libmgrid.so",
    "libmgrid_v2.so",
    "libmgrid_v3.so",
    "libpbc.so",
    "libsem.so",
    "libsolvent.so",
)
RUNTIME_VALIDATION_EVIDENCE_SCHEMA = (
    "gpu4pyscf.runtime-bundle-validation.v1"
)
RUNTIME_BUNDLE_SCHEMA = "gpu4pyscf.runtime-bundle.v2"
RUNTIME_MANIFEST_BASE_KEYS = {
    "source_root",
    "complete",
    "required_library_names",
    "inventory_library_names",
    "closure_complete",
    "files",
}
RUNTIME_MANIFEST_CANDIDATE_KEYS = RUNTIME_MANIFEST_BASE_KEYS | {
    "contract_schema",
    "bundle_id",
    "inventory_sha256",
    "sidecar_sha256",
    "build_verification_sha256",
    "validation_evidence",
    "validation_evidence_sha256",
}
RUNTIME_VALIDATION_EVIDENCE_KEYS = {
    "schema",
    "validated",
    "bundle_id",
    "bundle_root",
    "manifest_path",
    "build_verification_path",
    "library_count",
    "inventory_sha256",
    "inventory_payload_sha256",
    "source_sha256",
    "source_file_count",
    "cuda_version",
    "cuda_architecture",
    "abi_version",
    "maximum_stack_bytes",
    "build_verification_bound",
    "sidecar_sha256",
    "sidecar_bytes",
    "payload_sha256",
    "build_verification_sha256",
    "build_verification_bytes",
    "build_verification_payload_sha256",
}
RELEASE_PIN_SCHEMA = "gpu4pyscf.gint-selected-runtime-release-pin.v1"
RELEASE_PIN_RELATIVE_PATH = Path("gpu4pyscf/cc/gint_release_pin.json")
GINT_RELEASE_LINEAGE_KEYS = {
    "schema",
    "qualification_source_tree_sha256",
    "release_pin_relative_path",
    "release_pin_sha256",
    "receipt_sha256",
    "receipt_payload_sha256",
    "qualification_publication_attestation",
    "qualification_publication_attestation_sha256",
}
SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".ini",
    ".json", ".md", ".py", ".pyx", ".pxd", ".sbatch", ".sh",
    ".toml", ".txt", ".yaml", ".yml",
}
SOURCE_EXCLUDED_PARTS = {
    ".git", ".pytest_cache", "__pycache__", "build", "dist", "results",
}
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _json_object(data: bytes, *, name: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{name} contains non-finite number {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{name} root must be an object")
    return value


def _read_publication_attestation(
    snapshot: Path,
    *,
    expected_digest: str,
    manifest_bytes: bytes,
    policy: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Read and validate the immutable attestation beside a v3 snapshot."""

    errors: list[str] = []
    name = f"{expected_digest}{PUBLICATION_ATTESTATION_SUFFIX}"
    snapshots_root = snapshot.parent
    root_fd = -1
    attestation_fd = -1
    data = b""
    try:
        snapshot_info = snapshot.lstat()
        snapshots_root_info = snapshots_root.lstat()
        root_fd = os.open(snapshots_root, DIRECTORY_FLAGS | NOFOLLOW)
        opened_root = os.fstat(root_fd)
        if (opened_root.st_dev, opened_root.st_ino) != (
            snapshots_root_info.st_dev, snapshots_root_info.st_ino
        ):
            errors.append("publication attestation snapshots-root identity changed")
            return None, errors
        snapshot_entry = os.stat(
            expected_digest, dir_fd=root_fd, follow_symlinks=False
        )
        if (snapshot_entry.st_dev, snapshot_entry.st_ino) != (
            snapshot_info.st_dev, snapshot_info.st_ino
        ) or not stat.S_ISDIR(snapshot_entry.st_mode):
            errors.append("publication attestation snapshot identity changed")
            return None, errors
        try:
            before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            errors.append("publication attestation is missing")
            return None, errors
        if not stat.S_ISREG(before.st_mode):
            errors.append("publication attestation is not a regular file")
            return None, errors
        if before.st_mode & WRITE_BITS:
            errors.append("publication attestation is writable")
            return None, errors
        if before.st_size > MAX_PUBLICATION_ATTESTATION_BYTES:
            errors.append("publication attestation is too large")
            return None, errors
        attestation_fd = os.open(name, os.O_RDONLY | NOFOLLOW, dir_fd=root_fd)
        opened = os.fstat(attestation_fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            errors.append("publication attestation identity changed while opened")
            return None, errors
        if opened.st_mode & WRITE_BITS:
            errors.append("publication attestation is writable")
            return None, errors
        chunks: list[bytes] = []
        while chunk := os.read(attestation_fd, 1024 * 1024):
            chunks.append(chunk)
        data = b"".join(chunks)
        if len(data) > MAX_PUBLICATION_ATTESTATION_BYTES:
            errors.append("publication attestation is too large")
            return None, errors
        after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ):
            errors.append("publication attestation changed while read")
            return None, errors
        if (os.fstat(root_fd).st_dev, os.fstat(root_fd).st_ino) != (
            snapshots_root_info.st_dev, snapshots_root_info.st_ino
        ):
            errors.append("publication attestation snapshots-root changed while read")
            return None, errors
        snapshot_after = os.stat(
            expected_digest, dir_fd=root_fd, follow_symlinks=False
        )
        if (snapshot_after.st_dev, snapshot_after.st_ino) != (
            snapshot_info.st_dev, snapshot_info.st_ino
        ):
            errors.append("publication attestation snapshot changed while read")
            return None, errors
    except (OSError, ValueError) as exc:
        errors.append(f"publication attestation is unreadable: {exc!r}")
        return None, errors
    finally:
        if attestation_fd >= 0:
            os.close(attestation_fd)
        if root_fd >= 0:
            os.close(root_fd)

    try:
        value = _json_object(data, name="publication attestation")
    except Exception as exc:
        errors.append(f"publication attestation JSON is invalid: {exc!r}")
        return None, errors
    if not isinstance(value, dict):
        errors.append("publication attestation root is not an object")
        return None, errors
    if set(value) != PUBLICATION_ATTESTATION_KEYS:
        errors.append("publication attestation fields are not exact")
    if value.get("schema") != PUBLICATION_ATTESTATION_SCHEMA:
        errors.append("publication attestation schema mismatch")
    if value.get("status") != "published":
        errors.append("publication attestation status is not published")
    if value.get("source_tree_sha256") != expected_digest:
        errors.append("publication attestation source digest mismatch")
    if value.get("snapshot_name") != expected_digest:
        errors.append("publication attestation snapshot name mismatch")
    if value.get("publish_protocol") not in PUBLICATION_PROTOCOLS:
        errors.append("publication attestation protocol is invalid")
    if value.get("trust_boundary") != policy.get("trust_boundary"):
        errors.append("publication attestation trust boundary mismatch")
    if value.get("publication_policy_sha256") != _canonical_json_sha256(policy):
        errors.append("publication attestation policy digest mismatch")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if value.get("published_manifest_sha256") != manifest_digest:
        errors.append("publication attestation manifest digest mismatch")
    if value.get("prepublication_manifest_sha256") != manifest_digest:
        errors.append("publication attestation prepublication manifest digest mismatch")
    try:
        snapshot_info = snapshot.lstat()
        identity = value.get("published_directory_identity")
        if identity != {
            "device": snapshot_info.st_dev,
            "inode": snapshot_info.st_ino,
        }:
            errors.append("publication attestation directory identity mismatch")
    except OSError as exc:
        errors.append(f"publication attestation directory identity failed: {exc!r}")
    self_hash = value.get("attestation_sha256")
    unsigned = dict(value)
    unsigned.pop("attestation_sha256", None)
    if self_hash != _canonical_json_sha256(unsigned):
        errors.append("publication attestation self-hash mismatch")
    return value, errors


def _lexical_absolute_path(value: str | os.PathLike[str]) -> Path:
    text = os.path.abspath(os.path.expanduser(os.fspath(value)))
    # macOS exposes these fixed OS-managed aliases even when tempfile returns
    # their short spelling.  Canonicalize only this closed platform list; user
    # or snapshot-tree links remain visible to the no-follow component walk.
    if sys.platform == "darwin":
        for alias, canonical in (
            ("/var", "/private/var"),
            ("/tmp", "/private/tmp"),
            ("/etc", "/private/etc"),
        ):
            if text == alias or text.startswith(alias + os.sep):
                text = canonical + text[len(alias):]
                break
    return Path(text)


def _verify_directory_path_nofollow(path: Path, *, label: str) -> None:
    """Reject links in every component and bind each lstat to an open fd."""

    if not path.is_absolute() or Path(os.path.normpath(path)) != path:
        raise RuntimeError(f"{label} is not a normalized absolute path")
    components = path.parts[1:]
    if not components:
        raise RuntimeError(f"{label} cannot be the filesystem root")
    descriptor = os.open("/", DIRECTORY_FLAGS | NOFOLLOW)
    try:
        for component in components:
            try:
                before = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"{label} component is missing: {component}"
                ) from exc
            if stat.S_ISLNK(before.st_mode):
                raise RuntimeError(
                    f"{label} component is a symbolic link: {component}"
                )
            if not stat.S_ISDIR(before.st_mode):
                raise RuntimeError(
                    f"{label} component is not a directory: {component}"
                )
            child = os.open(
                component, DIRECTORY_FLAGS | NOFOLLOW, dir_fd=descriptor
            )
            opened = os.fstat(child)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                os.close(child)
                raise RuntimeError(
                    f"{label} component changed while opened: {component}"
                )
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"hash input is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = lambda info: (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
        )
        if identity(opened) != identity(before):
            raise RuntimeError(f"hash input changed while opened: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    if identity(path.lstat()) != identity(before):
        raise RuntimeError(f"hash input changed while read: {path}")
    return digest.hexdigest()


def _normalization_errors(value: Any, *, source: Path | None = None) -> list[str]:
    """Validate the v3 proof that repository links were materialized locally."""

    if not isinstance(value, dict):
        return ["source normalization manifest is missing"]
    errors: list[str] = []
    if value.get("schema") != SOURCE_NORMALIZATION_SCHEMA:
        errors.append("source normalization schema mismatch")
    if value.get("policy") != SOURCE_NORMALIZATION_POLICY:
        errors.append("source normalization policy mismatch")
    inventory = value.get("source_link_inventory")
    if not isinstance(inventory, dict):
        errors.append("source normalization inventory is missing")
        inventory = {}
    if inventory.get("schema") != SOURCE_NORMALIZATION_SCHEMA:
        errors.append("source normalization inventory schema mismatch")
    if inventory.get("policy") != SOURCE_NORMALIZATION_POLICY:
        errors.append("source normalization inventory policy mismatch")
    links = inventory.get("links")
    if not isinstance(links, list) or not all(
        isinstance(link, dict) for link in links
    ):
        errors.append("source normalization link inventory is invalid")
        links = []
    if inventory.get("link_count") != len(links):
        errors.append("source normalization link count mismatch")
    seen: set[str] = set()
    for link in links:
        relative_text = link.get("relative_path")
        target_text = link.get("target")
        resolved_text = link.get("resolved_target_relative_path")
        try:
            relative = Path(relative_text)
            resolved = Path(resolved_text)
        except TypeError:
            errors.append("source normalization link path is invalid")
            continue
        if (
            not isinstance(relative_text, str)
            or not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or any(part in SOURCE_EXCLUDED_PARTS for part in relative.parts)
            or relative.as_posix() != relative_text
            or relative_text in seen
        ):
            errors.append("source normalization link path is invalid")
            continue
        seen.add(relative_text)
        if (
            link.get("link_type") != "relative-leaf-file"
            or not isinstance(target_text, str)
            or not target_text
            or os.path.isabs(target_text)
            or not isinstance(resolved_text, str)
            or not resolved_text
            or resolved.is_absolute()
            or ".." in resolved.parts
            or any(part in SOURCE_EXCLUDED_PARTS for part in resolved.parts)
            or resolved.as_posix() != resolved_text
            or not isinstance(link.get("target_bytes"), int)
            or link["target_bytes"] < 0
            or not _is_sha256(link.get("target_sha256"))
        ):
            errors.append("source normalization link identity is invalid")
            continue
        target_parts = list(relative.parent.parts)
        target_components = target_text.split("/")
        if (
            "\\" in target_text
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in target_text
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
        if escaped or Path(*target_parts) != resolved:
            errors.append("source normalization resolved target mismatch")
            continue
        if source is not None:
            for label, path in (
                ("materialized link", source / relative),
                ("resolved target", source / resolved),
            ):
                try:
                    info = path.lstat()
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError("not regular")
                    if (
                        info.st_size != link["target_bytes"]
                        or _file_sha256(path) != link["target_sha256"]
                    ):
                        raise ValueError("content mismatch")
                except Exception:
                    errors.append(
                        f"source normalization {label} content mismatch: {path}"
                    )
    encoded_inventory = (
        json.dumps(
            inventory, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ) + "\n"
    ).encode("ascii")
    if (
        hashlib.sha256(encoded_inventory).hexdigest()
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
        errors.append("source normalization lacks remote post-binding verification")
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
        != hashlib.sha256(uploaded_data).hexdigest()
        or value.get("uploaded_record_bytes") != len(uploaded_data)
    ):
        errors.append("source normalization uploaded-record binding mismatch")
    for name in (
        "regular_file_count_after",
        "directory_count_after",
        "remote_regular_file_count_after_runtime_binding",
        "remote_directory_count_after_runtime_binding",
    ):
        if not isinstance(value.get(name), int) or value[name] < 0:
            errors.append(f"source normalization {name} is invalid")
    return errors


def discover_required_runtime_libraries(root: Path) -> tuple[str, ...]:
    """Find literal GPU4PySCF ``load_library`` targets in this source tree."""

    root = _lexical_absolute_path(root)
    _verify_directory_path_nofollow(root, label="runtime discovery root")
    package = root / "gpu4pyscf"
    targets: set[str] = set()
    provider_modules = {
        "gpu4pyscf.lib.cupy_helper",
        "gpu4pyscf.lib.utils",
    }
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        loader_aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in provider_modules:
                for name in node.names:
                    if name.name == "load_library":
                        loader_aliases.add(name.asname or name.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Name) or node.func.id not in loader_aliases:
                continue
            argument = node.args[0]
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                raise RuntimeError(f"dynamic GPU runtime library target in {path}")
            name = argument.value
            targets.add(name if name.endswith(".so") else f"{name}.so")
    if not targets:
        raise RuntimeError(f"no GPU runtime library targets found under {package}")
    return tuple(sorted(targets))


def source_tree_digest(
    root: Path, *, exclude_release_pin: bool = False
) -> tuple[str, int]:
    """Return the deployment digest, optionally omitting the one release pin.

    The boolean is deliberately the only exclusion interface.  Release
    snapshot B may differ from qualified snapshot A solely by the fixed
    ``gpu4pyscf/cc/gint_release_pin.json`` file.
    """

    root = _lexical_absolute_path(root)
    _verify_directory_path_nofollow(root, label="source digest root")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in SOURCE_EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if exclude_release_pin and relative == RELEASE_PIN_RELATIVE_PATH:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"source symlink leaves snapshot root: {path}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
        count += 1
    if count == 0:
        raise RuntimeError(f"no source files found under {root}")
    return digest.hexdigest(), count


def _positive_plain_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _nonnegative_plain_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _gint_release_lineage_shape_errors(
    lineage: Any, *, release_tree_sha256: str
) -> tuple[list[str], str | None]:
    """Validate the sealed A-to-B lineage fields without external reads."""

    errors: list[str] = []
    if not isinstance(lineage, dict) or set(lineage) != GINT_RELEASE_LINEAGE_KEYS:
        return ["GINT release lineage fields are not exact"], None
    if lineage.get("schema") != RELEASE_PIN_SCHEMA:
        errors.append("GINT release lineage schema mismatch")
    qualification_sha = lineage.get("qualification_source_tree_sha256")
    if not _is_sha256(qualification_sha):
        errors.append("GINT release qualification source digest is invalid")
        qualification_sha = None
    elif qualification_sha == release_tree_sha256:
        errors.append("GINT release A and B source digests are identical")
    if lineage.get("release_pin_relative_path") != RELEASE_PIN_RELATIVE_PATH.as_posix():
        errors.append("GINT release pin path is not the fixed allowed path")
    for name in (
        "release_pin_sha256",
        "receipt_sha256",
        "receipt_payload_sha256",
        "qualification_publication_attestation_sha256",
    ):
        if not _is_sha256(lineage.get(name)):
            errors.append(f"GINT release lineage {name} is invalid")

    attestation = lineage.get("qualification_publication_attestation")
    if not isinstance(attestation, dict) or set(attestation) != PUBLICATION_ATTESTATION_KEYS:
        errors.append("GINT release qualification attestation fields are not exact")
        return errors, qualification_sha
    if attestation.get("schema") != PUBLICATION_ATTESTATION_SCHEMA:
        errors.append("GINT release qualification attestation schema mismatch")
    if attestation.get("status") != "published":
        errors.append("GINT release qualification attestation status mismatch")
    if qualification_sha is not None:
        if attestation.get("source_tree_sha256") != qualification_sha:
            errors.append("GINT release qualification attestation source mismatch")
        if attestation.get("snapshot_name") != qualification_sha:
            errors.append("GINT release qualification attestation snapshot mismatch")
    if attestation.get("publish_protocol") not in PUBLICATION_PROTOCOLS:
        errors.append("GINT release qualification attestation protocol is invalid")
    if attestation.get("trust_boundary") != PUBLISH_TRUST_BOUNDARY:
        errors.append("GINT release qualification attestation trust boundary mismatch")
    if attestation.get("publication_policy_sha256") != _canonical_json_sha256(
        PUBLICATION_POLICY
    ):
        errors.append("GINT release qualification attestation policy mismatch")
    identity = attestation.get("published_directory_identity")
    if (
        not isinstance(identity, dict)
        or set(identity) != {"device", "inode"}
        or not _nonnegative_plain_int(identity.get("device"))
        or not _positive_plain_int(identity.get("inode"))
    ):
        errors.append("GINT release qualification directory identity is invalid")
    for name in (
        "published_manifest_sha256",
        "prepublication_manifest_sha256",
        "attestation_sha256",
    ):
        if not _is_sha256(attestation.get(name)):
            errors.append(f"GINT release qualification attestation {name} is invalid")
    if attestation.get("published_manifest_sha256") != attestation.get(
        "prepublication_manifest_sha256"
    ):
        errors.append("GINT release qualification manifest digests disagree")
    unsigned = dict(attestation)
    self_hash = unsigned.pop("attestation_sha256", None)
    try:
        calculated = _canonical_json_sha256(unsigned)
    except (TypeError, ValueError):
        calculated = None
    if self_hash != calculated:
        errors.append("GINT release qualification attestation self-hash mismatch")
    if lineage.get("qualification_publication_attestation_sha256") != self_hash:
        errors.append("GINT release qualification attestation digest mismatch")
    return errors, qualification_sha


def _gint_release_tree_errors(
    source: Path,
    lineage: Any,
    *,
    release_tree_sha256: str,
    release_file_count: int,
) -> list[str]:
    """Prove that B is A plus exactly the fixed, lineage-bound release pin."""

    errors: list[str] = []
    shape_errors, qualification_sha = _gint_release_lineage_shape_errors(
        lineage, release_tree_sha256=release_tree_sha256
    )
    errors.extend(shape_errors)
    if shape_errors or qualification_sha is None:
        return errors

    pin_path = source / RELEASE_PIN_RELATIVE_PATH
    candidates = sorted(
        path.relative_to(source).as_posix()
        for path in source.rglob(RELEASE_PIN_RELATIVE_PATH.name)
        if path.is_file() or path.is_symlink()
    )
    if candidates != [RELEASE_PIN_RELATIVE_PATH.as_posix()]:
        errors.append("release snapshot B does not contain exactly one fixed release pin")
        return errors
    try:
        pin_info = pin_path.lstat()
        if not stat.S_ISREG(pin_info.st_mode):
            raise ValueError("not a regular file")
        pin_bytes = pin_path.read_bytes()
        if hashlib.sha256(pin_bytes).hexdigest() != lineage["release_pin_sha256"]:
            errors.append("release pin digest differs from GINT release lineage")
        pin = _json_object(pin_bytes, name="GINT release pin")
    except Exception as exc:
        errors.append(f"GINT release pin is invalid: {exc!r}")
        return errors

    expected_pin_keys = {
        "schema", "status", "release_pin_path", "lineage", "receipt",
        "qualification_source", "qualification", "release_runtime_contract",
        "runtime", "transfer_audit_sha256",
    }
    if set(pin) != expected_pin_keys:
        errors.append("GINT release pin fields are not exact")
    if pin.get("schema") != RELEASE_PIN_SCHEMA or pin.get("status") != "release-pinned":
        errors.append("GINT release pin schema or status mismatch")
    if pin.get("release_pin_path") != RELEASE_PIN_RELATIVE_PATH.as_posix():
        errors.append("GINT release pin embeds a non-fixed path")
    if pin.get("lineage") != {
        "model": "qualification-source-plus-fixed-release-pin",
        "qualification_source_tree_sha256": qualification_sha,
        "release_source_without_pin_sha256": qualification_sha,
    }:
        errors.append("GINT release pin does not bind qualification snapshot A")
    if pin.get("receipt") != {
        "sha256": lineage.get("receipt_sha256"),
        "payload_sha256": lineage.get("receipt_payload_sha256"),
    }:
        errors.append("GINT release pin receipt binding differs from lineage")
    qualification_source = pin.get("qualification_source")
    if (
        not isinstance(qualification_source, dict)
        or qualification_source.get("tree_sha256") != qualification_sha
    ):
        errors.append("GINT release pin qualification source mismatch")

    try:
        without_pin_sha, without_pin_count = source_tree_digest(
            source, exclude_release_pin=True
        )
    except Exception as exc:
        errors.append(f"release snapshot B without its pin could not be hashed: {exc!r}")
    else:
        if without_pin_sha != qualification_sha:
            errors.append(
                "release snapshot B excluding only the fixed pin does not reproduce A"
            )
        if without_pin_count != release_file_count - 1:
            errors.append("release snapshot B source count does not differ from A by one pin")
    return errors


def _runtime_inventory_from_manifest(
    runtime: Any,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """Return the exact sidecar-form inventory sealed in runtime ``files``."""

    errors: list[str] = []
    if not isinstance(runtime, dict):
        return None, ["runtime-binary manifest is missing"]
    if runtime.get("complete") is not True:
        errors.append("runtime-binary manifest is incomplete")
    required = list(REQUIRED_RUNTIME_LIBRARIES)
    if runtime.get("required_library_names") != required:
        errors.append("runtime-binary required-library closure mismatch")
    if runtime.get("inventory_library_names") != required:
        errors.append("runtime-binary inventory names are not the exact required order")
    if runtime.get("closure_complete") is not True:
        errors.append("runtime-binary closure is not marked complete")
    entries = runtime.get("files")
    if not isinstance(entries, list) or len(entries) != len(required):
        errors.append("runtime-binary inventory must contain exactly 13 files")
        return None, errors
    inventory: list[dict[str, Any]] = []
    for index, (entry, name) in enumerate(zip(entries, required, strict=True)):
        if not isinstance(entry, dict) or set(entry) != {
            "relative_path", "bytes", "sha256",
        }:
            errors.append(f"runtime-binary entry {index} fields are not exact")
            continue
        expected_path = f"gpu4pyscf/lib/{name}"
        if entry.get("relative_path") != expected_path:
            errors.append(f"runtime-binary entry {index} path/order mismatch")
        if not _positive_plain_int(entry.get("bytes")):
            errors.append(f"runtime-binary entry {index} byte count is invalid")
        if not _is_sha256(entry.get("sha256")):
            errors.append(f"runtime-binary entry {index} digest is invalid")
        if (
            entry.get("relative_path") == expected_path
            and _positive_plain_int(entry.get("bytes"))
            and _is_sha256(entry.get("sha256"))
        ):
            inventory.append({
                "name": name,
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
            })
    if errors or len(inventory) != len(required):
        return None, errors
    return inventory, errors


def captured_runtime_bundle_evidence_errors(
    runtime: Any,
    *,
    deployment_profile: str | None,
    source_tree_sha256: str,
    source_file_count: int,
    task_root: str | Path,
    gint_release_lineage: Any = None,
) -> list[str]:
    """Validate sealed runtime evidence using no external sidecar/build files.

    ``runtime.files`` is the observed copied inventory for captured manifests.
    Live snapshot validation additionally hashes those files before calling
    this helper.
    """

    evidence = runtime.get("validation_evidence") if isinstance(runtime, dict) else None
    evidence_sha = (
        runtime.get("validation_evidence_sha256")
        if isinstance(runtime, dict) else None
    )
    evidence_present = evidence is not None or evidence_sha is not None
    if deployment_profile == G0_DEPLOYMENT_PROFILE and not evidence_present:
        return []
    inventory, errors = _runtime_inventory_from_manifest(runtime)
    if deployment_profile == CANDIDATE_DEPLOYMENT_PROFILE:
        if not isinstance(runtime, dict):
            errors.append("candidate runtime-binary manifest is missing")
        elif set(runtime) != RUNTIME_MANIFEST_CANDIDATE_KEYS:
            errors.append("candidate runtime-binary manifest fields are not exact")
    if deployment_profile == CANDIDATE_DEPLOYMENT_PROFILE and not evidence_present:
        errors.append("candidate runtime-bundle validation evidence is missing")
        return errors
    if not evidence_present:
        return errors
    if not isinstance(evidence, dict) or set(evidence) != RUNTIME_VALIDATION_EVIDENCE_KEYS:
        errors.append("runtime-bundle validation evidence fields are not exact")
        return errors
    if not _is_sha256(evidence_sha):
        errors.append("runtime-bundle validation evidence digest is invalid")
    else:
        try:
            actual_evidence_sha = _canonical_json_sha256(evidence)
        except (TypeError, ValueError):
            actual_evidence_sha = None
        if evidence_sha != actual_evidence_sha:
            errors.append("runtime-bundle validation evidence digest mismatch")
    if evidence.get("schema") != RUNTIME_VALIDATION_EVIDENCE_SCHEMA:
        errors.append("runtime-bundle validation evidence schema mismatch")
    if evidence.get("validated") is not True:
        errors.append("runtime bundle is not recorded as validated")
    if evidence.get("build_verification_bound") is not True:
        errors.append("runtime bundle lacks detached build verification")
    if evidence.get("library_count") != len(REQUIRED_RUNTIME_LIBRARIES):
        errors.append("runtime-bundle validation library count mismatch")
    if deployment_profile == CANDIDATE_DEPLOYMENT_PROFILE and isinstance(
        runtime, dict
    ):
        if runtime.get("contract_schema") != RUNTIME_BUNDLE_SCHEMA:
            errors.append("candidate runtime-bundle contract schema mismatch")
        for alias in (
            "bundle_id",
            "inventory_sha256",
            "sidecar_sha256",
            "build_verification_sha256",
        ):
            if runtime.get(alias) != evidence.get(alias):
                errors.append(
                    f"candidate runtime-bundle {alias} alias mismatch"
                )

    if gint_release_lineage is None:
        expected_source_sha = source_tree_sha256
        expected_source_count = source_file_count
    else:
        lineage_errors, qualification_sha = _gint_release_lineage_shape_errors(
            gint_release_lineage, release_tree_sha256=source_tree_sha256
        )
        errors.extend(lineage_errors)
        expected_source_sha = qualification_sha
        expected_source_count = source_file_count - 1
    if evidence.get("source_sha256") != expected_source_sha:
        errors.append("runtime-bundle evidence source digest mismatch")
    if (
        not _positive_plain_int(evidence.get("source_file_count"))
        or evidence.get("source_file_count") != expected_source_count
    ):
        errors.append("runtime-bundle evidence source file count mismatch")

    inventory_sha: str | None = None
    if inventory is not None:
        inventory_sha = hashlib.sha256(
            json.dumps(
                inventory,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        for name in ("bundle_id", "inventory_sha256", "inventory_payload_sha256"):
            if evidence.get(name) != inventory_sha:
                errors.append(f"runtime-bundle evidence {name} mismatch")

    for name in (
        "sidecar_sha256",
        "payload_sha256",
        "build_verification_sha256",
        "build_verification_payload_sha256",
    ):
        if not _is_sha256(evidence.get(name)):
            errors.append(f"runtime-bundle evidence {name} is invalid")
    for name in ("sidecar_bytes", "build_verification_bytes"):
        if not _positive_plain_int(evidence.get(name)):
            errors.append(f"runtime-bundle evidence {name} is invalid")
    if evidence.get("cuda_version") != "12.8":
        errors.append("runtime-bundle evidence CUDA version mismatch")
    if evidence.get("cuda_architecture") != "80-real":
        errors.append("runtime-bundle evidence CUDA architecture mismatch")
    if evidence.get("abi_version") != 2:
        errors.append("runtime-bundle evidence ABI version mismatch")
    maximum_stack = evidence.get("maximum_stack_bytes")
    if not _nonnegative_plain_int(maximum_stack) or maximum_stack > 4096:
        errors.append("runtime-bundle evidence maximum stack size is invalid")

    try:
        task_text = os.fspath(task_root)
        if (
            not isinstance(task_text, str)
            or not os.path.isabs(task_text)
            or os.path.normpath(task_text) != task_text
        ):
            raise ValueError("not a normalized absolute path")
        task = _lexical_absolute_path(task_text)
        if str(task) != task_text:
            raise ValueError("not a canonical absolute path")
    except (TypeError, ValueError, OSError):
        task = None
        errors.append("runtime-bundle expected task root is invalid")
    bundle_id = evidence.get("bundle_id")
    source_sha = evidence.get("source_sha256")
    if task is not None and _is_sha256(bundle_id) and _is_sha256(source_sha):
        expected_bundle_root = str(task / "runtime-bundles" / bundle_id)
        expected_sidecar_path = str(
            task / "runtime-bundle-manifests" / f"{bundle_id}.json"
        )
        expected_build_path = str(
            task / "builds"
            / f"{source_sha}-gint-bounded-workspace-v2-sm80"
            / "build-verification.json"
        )
        if evidence.get("bundle_root") != expected_bundle_root:
            errors.append("runtime-bundle evidence bundle path is outside task root")
        if evidence.get("manifest_path") != expected_sidecar_path:
            errors.append("runtime-bundle evidence sidecar path is outside task root")
        if evidence.get("build_verification_path") != expected_build_path:
            errors.append(
                "runtime-bundle evidence build verification path is outside task root"
            )
        if isinstance(runtime, dict) and runtime.get("source_root") != expected_bundle_root:
            errors.append("runtime-binary source root differs from validated bundle root")
    return errors


def validate_snapshot(
    source_root: str | Path,
    *,
    expected_base_revision: str = FROZEN_BASE_COMMIT,
    expected_task_root: str | Path | None = None,
    expected_deployment_profile: str | None = None,
    require_canonical_pristine: bool = False,
) -> dict[str, Any]:
    """Return fail-closed evidence for one installed source snapshot."""

    source = _lexical_absolute_path(source_root)
    snapshot = source.parent
    manifest_path = snapshot / "manifest.json"
    reasons: list[str] = []
    manifest: dict[str, Any] | None = None
    manifest_bytes = b""
    publication_attestation: dict[str, Any] | None = None
    publication_attestation_file_sha256: str | None = None
    deployment_profile: str | None = None
    validation_task_root = (
        _lexical_absolute_path(expected_task_root)
        if expected_task_root is not None
        else snapshot.parent.parent
    )

    source_path_safe = True
    try:
        _verify_directory_path_nofollow(source, label="snapshot source path")
    except Exception as exc:
        source_path_safe = False
        reasons.append(f"snapshot source path is unsafe: {exc}")

    if source.name != "source":
        reasons.append("source root basename is not 'source'")
    if snapshot.parent.name != "snapshots":
        reasons.append("source root is not below a snapshots directory")
    if expected_task_root is not None:
        if snapshot.parent.parent != validation_task_root:
            reasons.append("snapshot is outside the expected task root")
    if not source_path_safe:
        reasons.append("snapshot source directory is missing or unsafe")
    elif not stat.S_ISDIR(source.lstat().st_mode):
        reasons.append("snapshot source directory is missing")
    try:
        manifest_info = manifest_path.lstat()
    except FileNotFoundError:
        manifest_info = None
    if manifest_info is None or not stat.S_ISREG(manifest_info.st_mode):
        reasons.append("snapshot manifest is missing")
    else:
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = _json_object(manifest_bytes, name="snapshot manifest")
        except Exception as exc:
            reasons.append(f"snapshot manifest is unreadable: {exc!r}")

    actual_digest: str | None = None
    files_hashed: int | None = None
    published_symlinks: list[str] = []
    published_special_files: list[str] = []
    if source_path_safe:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source).as_posix()
            try:
                info = path.lstat()
            except FileNotFoundError:
                reasons.append(f"source entry disappeared: {relative}")
                continue
            if stat.S_ISLNK(info.st_mode):
                published_symlinks.append(relative)
            elif not (
                stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
            ):
                published_special_files.append(relative)
        if published_symlinks:
            reasons.append("published snapshot contains symbolic links")
        if published_special_files:
            reasons.append("published snapshot contains special files")
        if not published_symlinks and not published_special_files:
            try:
                actual_digest, files_hashed = source_tree_digest(source)
            except Exception as exc:
                reasons.append(f"source digest failed: {exc!r}")

    expected_digest = snapshot.name
    if not _is_sha256(expected_digest):
        reasons.append("snapshot directory name is not a lowercase SHA-256")
    if actual_digest is not None and actual_digest != expected_digest:
        reasons.append("source digest differs from snapshot directory name")

    if manifest is not None:
        manifest_schema = manifest.get("schema")
        if manifest_schema not in {
            LEGACY_SOURCE_SNAPSHOT_SCHEMA, SOURCE_SNAPSHOT_SCHEMA,
        }:
            reasons.append("snapshot manifest schema mismatch")
        if manifest_schema == SOURCE_SNAPSHOT_SCHEMA:
            reasons.extend(_normalization_errors(
                manifest.get("source_normalization"), source=source,
            ))
            publication_policy = manifest.get("publication_policy")
            if publication_policy != PUBLICATION_POLICY:
                reasons.append("publication policy is missing or mismatched")
                publication_policy = (
                    publication_policy
                    if isinstance(publication_policy, dict)
                    else PUBLICATION_POLICY
                )
            publication_attestation, attestation_errors = _read_publication_attestation(
                snapshot,
                expected_digest=expected_digest,
                manifest_bytes=manifest_bytes,
                policy=publication_policy,
            )
            reasons.extend(attestation_errors)
            if publication_attestation is not None and not attestation_errors:
                publication_attestation_file_sha256 = _file_sha256(
                    snapshot.parent
                    / f"{expected_digest}{PUBLICATION_ATTESTATION_SUFFIX}"
                )
        if manifest.get("tree_sha256") != expected_digest:
            reasons.append("snapshot manifest digest mismatch")
        if manifest.get("base_revision") != expected_base_revision:
            reasons.append("snapshot manifest base revision mismatch")
        manifest_source_text = manifest.get("source")
        manifest_source = None
        if (
            isinstance(manifest_source_text, str)
            and os.path.isabs(manifest_source_text)
            and os.path.normpath(manifest_source_text) == manifest_source_text
        ):
            manifest_source = Path(manifest_source_text)
        if manifest_source != source:
            reasons.append("snapshot manifest source path mismatch")
        if manifest.get("immutable") is not True:
            reasons.append("snapshot manifest does not declare immutable=true")

        provenance = manifest.get("local_provenance")
        if not isinstance(provenance, dict):
            reasons.append("local provenance is missing")
        else:
            if provenance.get("schema") != "gpu4pyscf.local-provenance.v1":
                reasons.append("local provenance schema mismatch")
            if provenance.get("frozen_base_revision") != expected_base_revision:
                reasons.append("local provenance base revision mismatch")
            repository_root = provenance.get("repository_root")
            if not isinstance(repository_root, str) or not repository_root.startswith("/"):
                reasons.append("local provenance repository root is invalid")
            profile = provenance.get("deployment_profile")
            deployment_profile = profile if isinstance(profile, str) else None
            if profile not in DEPLOYMENT_PROFILES:
                reasons.append("local provenance deployment profile is invalid")
            if (
                expected_deployment_profile is not None
                and profile != expected_deployment_profile
            ):
                reasons.append("snapshot deployment profile mismatch")

            head = provenance.get("head_revision")
            if not (
                isinstance(head, str)
                and len(head) == 40
                and all(character in "0123456789abcdef" for character in head)
            ):
                reasons.append("local provenance HEAD is invalid")
            head_matches_frozen = head == expected_base_revision
            if provenance.get("head_matches_frozen_base") is not head_matches_frozen:
                reasons.append("local provenance frozen-HEAD claim is inconsistent")

            tracked_pristine = provenance.get("tracked_pristine")
            if not isinstance(tracked_pristine, bool):
                reasons.append("local provenance tracked-pristine claim is invalid")
            canonical_pristine = provenance.get("canonical_pristine")
            if not isinstance(canonical_pristine, bool):
                reasons.append("local provenance canonical-pristine claim is invalid")
            expected_canonical = bool(
                profile == G0_DEPLOYMENT_PROFILE
                and head_matches_frozen
                and tracked_pristine is True
                and provenance.get("untracked_paths_allowed") is True
            )
            if canonical_pristine is not expected_canonical:
                reasons.append(
                    "local provenance canonical-pristine claim is inconsistent"
                )
            if profile == CANDIDATE_DEPLOYMENT_PROFILE and canonical_pristine is True:
                reasons.append("candidate deployment is mislabeled canonical-pristine")

            status = provenance.get("git_status")
            if not isinstance(status, dict):
                reasons.append("local provenance git status is missing")
            else:
                status_sha = status.get("sha256")
                policy_status_sha = status.get("allowed_paths_status_sha256")
                entries = status.get("entries")
                if (
                    not _is_sha256(status_sha)
                    or not _is_sha256(policy_status_sha)
                    or not isinstance(entries, list)
                    or not all(isinstance(entry, str) for entry in entries)
                    or status.get("entry_count") != len(entries)
                    or status.get("format") != "porcelain-v1-lines"
                ):
                    reasons.append("local provenance git status evidence is invalid")
                elif isinstance(allowed := provenance.get("allowed_untracked"), dict):
                    status_bytes = (
                        "\n".join(entries) + ("\n" if entries else "")
                    ).encode("utf-8")
                    if hashlib.sha256(status_bytes).hexdigest() != status_sha:
                        reasons.append(
                            "local provenance git status digest is inconsistent"
                        )
                    policy_bytes = json.dumps(
                        {"allowed_untracked": allowed, "status": entries},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    if (
                        hashlib.sha256(policy_bytes).hexdigest()
                        != policy_status_sha
                    ):
                        reasons.append(
                            "local provenance allowed-path/status digest is inconsistent"
                        )
                    prefixes = allowed.get("prefixes", [])
                    files = allowed.get("files", [])
                    tracked_entries_pristine = all(
                        len(entry) >= 4 and entry[:2] == "??" for entry in entries
                    )
                    untracked_entries_allowed = all(
                        any(entry[3:].startswith(prefix) for prefix in prefixes)
                        or entry[3:] in files
                        for entry in entries
                        if len(entry) >= 4 and entry[:2] == "??"
                    )
                    if tracked_pristine is not tracked_entries_pristine:
                        reasons.append(
                            "local provenance tracked-pristine claim is inconsistent"
                        )
                    if (
                        provenance.get("untracked_paths_allowed")
                        is not untracked_entries_allowed
                    ):
                        reasons.append(
                            "local provenance allowed-path claim is inconsistent"
                        )

            allowed = provenance.get("allowed_untracked")
            if not isinstance(allowed, dict):
                reasons.append("local provenance allowed-path policy is missing")
            elif (
                allowed.get("prefixes") != ["benchmarks/cc/a100_water8/"]
                or allowed.get("files") != ["gpu4pyscf/cc/device_runtime.py"]
            ):
                reasons.append("local provenance allowed-path policy mismatch")

            if require_canonical_pristine and canonical_pristine is not True:
                reasons.append("canonical-pristine deployment is required")

        runtime = manifest.get("runtime_binaries")
        if not isinstance(runtime, dict) or runtime.get("complete") is not True:
            reasons.append("runtime-binary manifest is missing or incomplete")
        else:
            runtime_source_root = runtime.get("source_root")
            if not (
                isinstance(runtime_source_root, str)
                and runtime_source_root.startswith("/")
            ):
                reasons.append("runtime-binary source root is invalid")
            entries = runtime.get("files")
            if published_symlinks or published_special_files:
                reasons.append(
                    "runtime-binary inspection requires a normalized source tree"
                )
            elif not isinstance(entries, list) or not entries:
                reasons.append("runtime-binary manifest has no files")
            else:
                listed: set[str] = set()
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        reasons.append(
                            f"runtime-binary entry {index} is not an object"
                        )
                        continue
                    relative = Path(str(entry.get("relative_path", "")))
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or relative.suffix != ".so"
                    ):
                        reasons.append(
                            f"runtime-binary entry {index} has an invalid path"
                        )
                        continue
                    path = source / relative
                    try:
                        path_info = path.lstat()
                    except FileNotFoundError:
                        path_info = None
                    if path_info is None or not stat.S_ISREG(path_info.st_mode):
                        reasons.append(f"runtime binary is missing: {relative}")
                        continue
                    relative_text = relative.as_posix()
                    if relative_text in listed:
                        reasons.append(
                            f"duplicate runtime-binary entry: {relative}"
                        )
                    listed.add(relative_text)
                    if path.stat().st_size != entry.get("bytes"):
                        reasons.append(
                            f"runtime binary size mismatch: {relative}"
                        )
                    if _file_sha256(path) != entry.get("sha256"):
                        reasons.append(
                            f"runtime binary digest mismatch: {relative}"
                        )
                actual = {
                    path.relative_to(source).as_posix()
                    for path in source.rglob("*.so")
                    if stat.S_ISREG(path.lstat().st_mode)
                }
                if actual != listed:
                    reasons.append(
                        "runtime-binary file set differs from the manifest"
                    )
                inventory_names = sorted(Path(path).name for path in listed)
                if runtime.get("inventory_library_names") != inventory_names:
                    reasons.append("runtime-binary inventory names mismatch")
                required_names = list(REQUIRED_RUNTIME_LIBRARIES)
                if runtime.get("required_library_names") != required_names:
                    reasons.append("runtime-binary required-library closure mismatch")
                missing = sorted(set(required_names) - set(inventory_names))
                if missing:
                    reasons.append(
                        "runtime-binary closure is missing: " + ", ".join(missing)
                    )
                if runtime.get("closure_complete") is not True:
                    reasons.append("runtime-binary closure is not marked complete")

        lineage = manifest.get("gint_release_lineage")
        if lineage is not None and manifest_schema != SOURCE_SNAPSHOT_SCHEMA:
            reasons.append("GINT release lineage requires a v3 snapshot manifest")
        if lineage is None and source_path_safe:
            try:
                if (source / RELEASE_PIN_RELATIVE_PATH).exists():
                    reasons.append("GINT release pin exists without sealed A-to-B lineage")
            except OSError as exc:
                reasons.append(f"GINT release pin presence check failed: {exc!r}")
        if (
            lineage is not None
            and source_path_safe
            and actual_digest is not None
            and files_hashed is not None
        ):
            reasons.extend(_gint_release_tree_errors(
                source,
                lineage,
                release_tree_sha256=actual_digest,
                release_file_count=files_hashed,
            ))
        if actual_digest is not None and files_hashed is not None:
            reasons.extend(captured_runtime_bundle_evidence_errors(
                runtime,
                deployment_profile=deployment_profile,
                source_tree_sha256=actual_digest,
                source_file_count=files_hashed,
                task_root=validation_task_root,
                gint_release_lineage=lineage,
            ))

        if (
            source_path_safe
            and not published_symlinks
            and not published_special_files
        ):
            try:
                discovered = discover_required_runtime_libraries(source)
                if discovered != REQUIRED_RUNTIME_LIBRARIES:
                    reasons.append(
                        "source load_library targets differ from the required-runtime closure"
                    )
            except Exception as exc:
                reasons.append(f"runtime-library discovery failed: {exc!r}")

    writable_paths: list[str] = []
    if source_path_safe:
        paths = [snapshot, manifest_path, source]
        if manifest is not None and manifest.get("schema") == SOURCE_SNAPSHOT_SCHEMA:
            paths.append(
                snapshot.parent
                / f"{expected_digest}{PUBLICATION_ATTESTATION_SUFFIX}"
            )
        paths.extend(source.rglob("*"))
        for path in paths:
            try:
                if path.lstat().st_mode & WRITE_BITS:
                    writable_paths.append(str(path))
            except FileNotFoundError:
                continue
    if writable_paths:
        reasons.append("snapshot contains writable paths")

    return {
        "schema": "gpu4pyscf.snapshot-evidence.v1",
        "valid": not reasons,
        "source": str(source),
        "snapshot_root": str(snapshot),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "manifest_sha256": (
            hashlib.sha256(manifest_bytes).hexdigest()
            if manifest_bytes else None
        ),
        "publication_attestation": publication_attestation,
        "publication_attestation_file_sha256": publication_attestation_file_sha256,
        "expected_base_revision": expected_base_revision,
        "expected_deployment_profile": expected_deployment_profile,
        "canonical_pristine_required": require_canonical_pristine,
        "required_runtime_libraries": list(REQUIRED_RUNTIME_LIBRARIES),
        "tree_sha256": actual_digest,
        "files_hashed": files_hashed,
        "read_only": not writable_paths,
        "writable_paths": writable_paths,
        "published_symlinks": published_symlinks,
        "published_special_files": published_special_files,
        "reasons": reasons,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-task-root", type=Path)
    parser.add_argument(
        "--expected-base-revision", default=FROZEN_BASE_COMMIT
    )
    parser.add_argument(
        "--expected-deployment-profile", choices=sorted(DEPLOYMENT_PROFILES)
    )
    parser.add_argument("--require-canonical-pristine", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = validate_snapshot(
        args.source_root,
        expected_base_revision=args.expected_base_revision,
        expected_task_root=args.expected_task_root,
        expected_deployment_profile=args.expected_deployment_profile,
        require_canonical_pristine=args.require_canonical_pristine,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if evidence["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
