"""Security tests for the remote snapshot staging lifecycle guard."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import stat

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "water8_snapshot_staging_guard_test", ROOT / "snapshot_staging_guard.py"
)
GUARD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GUARD)


def _snapshots_root(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "task" / "snapshots"
    root.parent.mkdir(parents=True)
    return root


def _identities(created: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (created["snapshots_dev"], created["snapshots_ino"]),
        (created["staging_dev"], created["staging_ino"]),
    )


def test_staging_is_unpredictable_direct_child_and_identity_bound(tmp_path: Path):
    root = _snapshots_root(tmp_path)
    first = GUARD.create_staging(str(root), "a" * 64)
    second = GUARD.create_staging(str(root), "a" * 64)
    assert first["staging_name"] != second["staging_name"]
    assert Path(first["staging_path"]).parent == root
    root_identity, staging_identity = _identities(first)
    result = GUARD.verify_staging(
        str(root), first["staging_name"], root_identity, staging_identity,
        (first["source_dev"], first["source_ino"]),
    )
    assert result["verified"] is True


def test_prepositioned_staging_symlink_is_rejected_without_escape(tmp_path: Path):
    root = _snapshots_root(tmp_path)
    root.mkdir()
    outside = tmp_path.resolve() / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("must survive\n", encoding="utf-8")
    forged = root / ".staging-forged"
    forged.symlink_to(outside, target_is_directory=True)
    root_info = root.stat()
    link_info = forged.lstat()

    with pytest.raises(RuntimeError, match="symbolic link"):
        GUARD.inspect_child(str(root), forged.name)
    with pytest.raises(RuntimeError, match="symbolic link"):
        GUARD.remove_child(
            str(root), forged.name,
            (root_info.st_dev, root_info.st_ino),
            (link_info.st_dev, link_info.st_ino),
        )
    assert sentinel.read_text(encoding="utf-8") == "must survive\n"
    assert forged.is_symlink()


def test_cleanup_unlinks_nested_links_and_never_follows_them(tmp_path: Path):
    root = _snapshots_root(tmp_path)
    created = GUARD.create_staging(str(root), "b" * 64)
    staging = Path(created["staging_path"])
    outside = tmp_path.resolve() / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("must survive\n", encoding="utf-8")
    (staging / "source" / "nested").mkdir()
    (staging / "source" / "nested" / "file.txt").write_text("inside\n")
    (staging / "source" / "outside-link").symlink_to(
        outside, target_is_directory=True
    )

    root_identity, staging_identity = _identities(created)
    result = GUARD.remove_child(
        str(root), created["staging_name"], root_identity, staging_identity
    )
    assert result["removed"] is True
    assert not os.path.lexists(staging)
    assert sentinel.read_text(encoding="utf-8") == "must survive\n"


def test_cleanup_refuses_replaced_direct_child(tmp_path: Path):
    root = _snapshots_root(tmp_path)
    created = GUARD.create_staging(str(root), "c" * 64)
    staging = Path(created["staging_path"])
    moved = root / ".original-staging"
    staging.rename(moved)
    staging.mkdir()
    replacement = staging / "replacement.txt"
    replacement.write_text("do not remove\n", encoding="utf-8")

    root_identity, staging_identity = _identities(created)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        GUARD.remove_child(
            str(root), created["staging_name"], root_identity, staging_identity
        )
    assert replacement.read_text(encoding="utf-8") == "do not remove\n"
    assert moved.is_dir()


def test_cleanup_removes_same_inode_after_atomic_publish_rename(tmp_path: Path):
    root = _snapshots_root(tmp_path)
    digest = "d" * 64
    created = GUARD.create_staging(str(root), digest)
    staging = Path(created["staging_path"])
    published = root / digest
    (staging / "source" / "payload.bin").write_bytes(b"payload\n")
    staging.rename(published)

    root_identity, staging_identity = _identities(created)
    result = GUARD.remove_child(
        str(root), digest, root_identity, staging_identity
    )

    assert result["removed"] is True
    assert not os.path.lexists(published)


def test_cleanup_ancestor_swap_refuses_outside_mutation(tmp_path: Path):
    private = tmp_path.resolve() / "private"
    root = private / "task" / "snapshots"
    root.parent.mkdir(parents=True)
    private.chmod(0o700)
    created = GUARD.create_staging(str(root), "e" * 64)
    root_identity, staging_identity = _identities(created)
    original = tmp_path.resolve() / "private-original"
    outside = tmp_path.resolve() / "outside"
    outside_staging = outside / "task" / "snapshots" / created["staging_name"]
    outside_staging.mkdir(parents=True)
    sentinel = outside_staging / "sentinel.txt"
    sentinel.write_text("must survive\n", encoding="utf-8")
    private.rename(original)
    private.symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises(RuntimeError, match="symbolic link"):
            GUARD.remove_child(
                str(root), created["staging_name"],
                root_identity, staging_identity,
            )
        assert sentinel.read_text(encoding="utf-8") == "must survive\n"
    finally:
        private.unlink()
        original.rename(private)
        GUARD.remove_child(
            str(root), created["staging_name"],
            root_identity, staging_identity,
        )


def test_snapshots_root_path_component_may_not_be_a_link(tmp_path: Path):
    canonical = tmp_path.resolve()
    real_parent = canonical / "real-task"
    real_parent.mkdir()
    linked_parent = canonical / "linked-task"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="component is a symbolic link"):
        GUARD.inspect_child(str(linked_parent / "snapshots"), "snapshot")


def test_task_and_snapshots_root_may_not_be_writable_by_other_principals(
    tmp_path: Path,
):
    root = _snapshots_root(tmp_path)
    root.mkdir()
    task = root.parent
    original_task_mode = stat.S_IMODE(task.stat().st_mode)
    original_root_mode = stat.S_IMODE(root.stat().st_mode)
    try:
        task.chmod(original_task_mode | stat.S_IWGRP)
        with pytest.raises(RuntimeError, match="writable by another principal"):
            GUARD.inspect_child(str(root), "snapshot")
        task.chmod(original_task_mode)
        root.chmod(original_root_mode | stat.S_IWOTH)
        with pytest.raises(RuntimeError, match="writable by another principal"):
            GUARD.inspect_child(str(root), "snapshot")
    finally:
        task.chmod(original_task_mode)
        root.chmod(original_root_mode)
