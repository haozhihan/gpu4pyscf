#!/usr/bin/env python3
"""Validate content-addressed, read-only benchmark source snapshots."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import stat
from pathlib import Path
from typing import Any, Sequence


FROZEN_BASE_COMMIT = "a89b3ae018d4e82968323ef95645d91adde294aa"
G0_DEPLOYMENT_PROFILE = "g0-canonical-pristine"
CANDIDATE_DEPLOYMENT_PROFILE = "candidate"
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
SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".ini",
    ".json", ".md", ".py", ".pyx", ".pxd", ".sbatch", ".sh",
    ".toml", ".txt", ".yaml", ".yml",
}
SOURCE_EXCLUDED_PARTS = {
    ".git", ".pytest_cache", "__pycache__", "build", "dist", "results",
}
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def discover_required_runtime_libraries(root: Path) -> tuple[str, ...]:
    """Find literal GPU4PySCF ``load_library`` targets in this source tree."""

    package = root.resolve() / "gpu4pyscf"
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


def source_tree_digest(root: Path) -> tuple[str, int]:
    """Return the exact digest used by benchmark.py and snapshot deployment."""

    root = root.resolve()
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


def validate_snapshot(
    source_root: str | Path,
    *,
    expected_base_revision: str = FROZEN_BASE_COMMIT,
    expected_task_root: str | Path | None = None,
    expected_deployment_profile: str | None = None,
    require_canonical_pristine: bool = False,
) -> dict[str, Any]:
    """Return fail-closed evidence for one installed source snapshot."""

    source = Path(source_root).expanduser().resolve()
    snapshot = source.parent
    manifest_path = snapshot / "manifest.json"
    reasons: list[str] = []
    manifest: dict[str, Any] | None = None

    if source.name != "source":
        reasons.append("source root basename is not 'source'")
    if snapshot.parent.name != "snapshots":
        reasons.append("source root is not below a snapshots directory")
    if expected_task_root is not None:
        task_root = Path(expected_task_root).expanduser().resolve()
        if snapshot.parent.parent != task_root:
            reasons.append("snapshot is outside the expected task root")
    if not source.is_dir():
        reasons.append("snapshot source directory is missing")
    if not manifest_path.is_file():
        reasons.append("snapshot manifest is missing")
    else:
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise TypeError("manifest root is not an object")
            manifest = loaded
        except Exception as exc:
            reasons.append(f"snapshot manifest is unreadable: {exc!r}")

    actual_digest: str | None = None
    files_hashed: int | None = None
    if source.is_dir():
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
        if manifest.get("schema") != "gpu4pyscf.source-snapshot.v2":
            reasons.append("snapshot manifest schema mismatch")
        if manifest.get("tree_sha256") != expected_digest:
            reasons.append("snapshot manifest digest mismatch")
        if manifest.get("base_revision") != expected_base_revision:
            reasons.append("snapshot manifest base revision mismatch")
        try:
            manifest_source = Path(str(manifest.get("source", ""))).resolve()
        except Exception:
            manifest_source = None
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
            if not isinstance(entries, list) or not entries:
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
                        inside = path.resolve().is_relative_to(source)
                    except Exception:
                        inside = False
                    if not path.is_file() or not inside:
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
                    for path in source.rglob("*.so") if path.is_file()
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

        if source.is_dir():
            try:
                discovered = discover_required_runtime_libraries(source)
                if discovered != REQUIRED_RUNTIME_LIBRARIES:
                    reasons.append(
                        "source load_library targets differ from the required-runtime closure"
                    )
            except Exception as exc:
                reasons.append(f"runtime-library discovery failed: {exc!r}")

    writable_paths: list[str] = []
    if snapshot.exists():
        paths = [snapshot, manifest_path, source]
        if source.is_dir():
            paths.extend(source.rglob("*"))
        for path in paths:
            try:
                if path.stat().st_mode & WRITE_BITS:
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
        "expected_base_revision": expected_base_revision,
        "expected_deployment_profile": expected_deployment_profile,
        "canonical_pristine_required": require_canonical_pristine,
        "required_runtime_libraries": list(REQUIRED_RUNTIME_LIBRARIES),
        "tree_sha256": actual_digest,
        "files_hashed": files_hashed,
        "read_only": not writable_paths,
        "writable_paths": writable_paths,
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
