"""Security and identity tests for runtime snapshot link normalization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


NORMALIZER = _load(
    "water8_snapshot_symlink_normalize_test",
    ROOT / "snapshot_symlink_normalize.py",
)
SNAPSHOT = _load(
    "water8_snapshot_manifest_for_normalize_test",
    ROOT / "snapshot_manifest.py",
)


def _source_with_target(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    target = source / "dockerfiles" / "manylinux" / "build_wheels.sh"
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (source / "builder").mkdir()
    return source, target


def test_valid_internal_leaf_link_is_materialized_without_digest_change(
    tmp_path: Path,
):
    source, target = _source_with_target(tmp_path)
    link = source / "builder" / "build_wheels.sh"
    link.symlink_to("../dockerfiles/manylinux/build_wheels.sh")

    digest_before = SNAPSHOT.source_tree_digest(source)
    inventory = NORMALIZER.inspect_source(source)
    result = NORMALIZER.materialize_source_links(source, inventory)
    digest_after = SNAPSHOT.source_tree_digest(source)

    assert digest_after == digest_before
    assert result["materialized_link_count"] == 1
    assert result["published_symlink_count"] == 0
    assert link.is_file() and not link.is_symlink()
    assert link.read_bytes() == target.read_bytes()
    assert result["materialized_links"][0]["target"] == (
        "../dockerfiles/manylinux/build_wheels.sh"
    )

    normalized = NORMALIZER.assert_normalized_source(source)
    source_fd = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        assert NORMALIZER.assert_normalized_source_fd(source_fd) == normalized
    finally:
        os.close(source_fd)
    uploaded = (
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    result.update({
        "remote_verified_after_runtime_binding": True,
        "remote_regular_file_count_after_runtime_binding": normalized[
            "regular_file_count_after"
        ],
        "remote_directory_count_after_runtime_binding": normalized[
            "directory_count_after"
        ],
        "uploaded_record_sha256": hashlib.sha256(uploaded).hexdigest(),
        "uploaded_record_bytes": len(uploaded),
    })
    assert SNAPSHOT._normalization_errors(result, source=source) == []

    link.write_text("tampered\n")
    assert "content mismatch" in " ".join(
        SNAPSHOT._normalization_errors(result, source=source)
    )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("/etc/passwd", "absolute target"),
        ("../../../outside.sh", "escapes source root"),
        ("missing.sh", "dangling"),
        ("../dockerfiles", "points to a directory"),
        ("./target.sh", "suspicious"),
        ("target//file.sh", "suspicious"),
    ],
)
def test_invalid_link_targets_fail_closed(
    tmp_path: Path, target: str, message: str,
):
    source, _ = _source_with_target(tmp_path)
    (tmp_path / "outside.sh").write_text("outside\n")
    link = source / "builder" / "bad.sh"
    link.symlink_to(target)
    with pytest.raises(ValueError, match=message):
        NORMALIZER.inspect_source(source)


def test_link_chain_and_cycle_fail_closed(tmp_path: Path):
    chain_source, _ = _source_with_target(tmp_path / "chain")
    first = chain_source / "builder" / "first.sh"
    second = chain_source / "builder" / "second.sh"
    second.symlink_to("../dockerfiles/manylinux/build_wheels.sh")
    first.symlink_to("second.sh")
    with pytest.raises(ValueError, match="chain or cycle"):
        NORMALIZER.inspect_source(chain_source)

    cycle_source, _ = _source_with_target(tmp_path / "cycle")
    left = cycle_source / "builder" / "left.sh"
    right = cycle_source / "builder" / "right.sh"
    left.symlink_to("right.sh")
    right.symlink_to("left.sh")
    with pytest.raises(ValueError, match="chain or cycle"):
        NORMALIZER.inspect_source(cycle_source)


def test_linked_directory_and_special_file_fail_closed(tmp_path: Path):
    source, _ = _source_with_target(tmp_path)
    (source / "directory-link").symlink_to("dockerfiles")
    with pytest.raises(ValueError, match="points to a directory"):
        NORMALIZER.inspect_source(source)

    (source / "directory-link").unlink()
    fifo = source / "unexpected.pipe"
    try:
        fifo.parent.mkdir(parents=True, exist_ok=True)
        import os
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="special file"):
            NORMALIZER.inspect_source(source)
    finally:
        if fifo.exists():
            fifo.unlink()


def test_inventory_binds_target_path_and_content(tmp_path: Path):
    source, first_target = _source_with_target(tmp_path)
    second_target = first_target.with_name("build_wheels_copy.sh")
    second_target.write_bytes(first_target.read_bytes())
    link = source / "builder" / "build_wheels.sh"
    link.symlink_to("../dockerfiles/manylinux/build_wheels.sh")
    first = NORMALIZER.inspect_source(source)

    link.unlink()
    link.symlink_to("../dockerfiles/manylinux/build_wheels_copy.sh")
    second = NORMALIZER.inspect_source(source)
    assert NORMALIZER.inventory_sha256(second) != NORMALIZER.inventory_sha256(first)

    second_target.write_text("changed\n")
    third = NORMALIZER.inspect_source(source)
    assert NORMALIZER.inventory_sha256(third) != NORMALIZER.inventory_sha256(second)


def test_normalizer_rejects_inventory_change_before_replacement(tmp_path: Path):
    source, target = _source_with_target(tmp_path)
    link = source / "builder" / "build_wheels.sh"
    link.symlink_to("../dockerfiles/manylinux/build_wheels.sh")
    inventory = NORMALIZER.inspect_source(source)
    target.write_text("changed after inventory\n")
    with pytest.raises(ValueError, match="inventory changed"):
        NORMALIZER.materialize_source_links(source, inventory)


def test_target_directory_swap_is_detected_during_anchored_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source, _ = _source_with_target(tmp_path)
    link = source / "builder" / "build_wheels.sh"
    link.symlink_to("../dockerfiles/manylinux/build_wheels.sh")
    original_open = NORMALIZER.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "dockerfiles" and dir_fd is not None and not swapped:
            swapped = True
            source.joinpath("dockerfiles").rename(source / "dockerfiles-old")
            replacement = source / "dockerfiles" / "manylinux"
            replacement.mkdir(parents=True)
            (replacement / "build_wheels.sh").write_text(
                "malicious replacement\n", encoding="utf-8"
            )
        return descriptor

    monkeypatch.setattr(NORMALIZER.os, "open", swapping_open)
    with pytest.raises(ValueError, match="path changed during anchored read"):
        NORMALIZER.inspect_source(source)
    assert swapped is True


def test_materialization_parent_swap_cannot_write_through_outside_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source, _ = _source_with_target(tmp_path)
    link = source / "builder" / "build_wheels.sh"
    link.symlink_to("../dockerfiles/manylinux/build_wheels.sh")
    inventory = NORMALIZER.inspect_source(source)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("must survive\n", encoding="utf-8")
    original_open = NORMALIZER.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if (
            isinstance(path, str)
            and path.startswith(".source-link-materialize-")
            and dir_fd is not None
            and not swapped
        ):
            swapped = True
            (source / "builder").rename(source / "builder-original")
            (source / "builder").symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(NORMALIZER.os, "open", swapping_open)
    with pytest.raises(ValueError, match="parent path changed during operation"):
        NORMALIZER.materialize_source_links(source, inventory)
    assert swapped is True
    assert sentinel.read_text(encoding="utf-8") == "must survive\n"
    assert sorted(path.name for path in outside.iterdir()) == ["sentinel.txt"]
