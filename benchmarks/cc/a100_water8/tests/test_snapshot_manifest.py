from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "water8_snapshot_manifest", ROOT / "snapshot_manifest.py"
)
SNAPSHOT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SNAPSHOT)


def _provenance(profile: str = "g0-canonical-pristine") -> dict:
    entries = [
        "?? benchmarks/cc/a100_water8/benchmark.py",
        "?? gpu4pyscf/cc/device_runtime.py",
    ]
    allowed = {
        "prefixes": ["benchmarks/cc/a100_water8/"],
        "files": ["gpu4pyscf/cc/device_runtime.py"],
    }
    status_bytes = ("\n".join(entries) + "\n").encode("utf-8")
    policy_bytes = json.dumps(
        {"allowed_untracked": allowed, "status": entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    canonical = profile == "g0-canonical-pristine"
    return {
        "schema": "gpu4pyscf.local-provenance.v1",
        "deployment_profile": profile,
        "repository_root": "/local/gpu4pyscf",
        "head_revision": SNAPSHOT.FROZEN_BASE_COMMIT,
        "frozen_base_revision": SNAPSHOT.FROZEN_BASE_COMMIT,
        "head_matches_frozen_base": True,
        "tracked_pristine": True,
        "untracked_paths_allowed": True,
        "canonical_pristine": canonical,
        "allowed_untracked": allowed,
        "git_status": {
            "format": "porcelain-v1-lines",
            "sha256": hashlib.sha256(status_bytes).hexdigest(),
            "allowed_paths_status_sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "entry_count": len(entries),
            "entries": entries,
        },
    }


def _installed_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    task = tmp_path / "task"
    staging = task / "snapshots" / "staging"
    source = staging / "source"
    source.mkdir(parents=True)
    (source / "module.py").write_text("answer = 42\n", encoding="utf-8")
    binary_root = source / "gpu4pyscf" / "lib"
    binary_root.mkdir(parents=True)
    (source / "gpu4pyscf" / "runtime_targets.py").write_text(
        "from gpu4pyscf.lib.utils import load_library\n"
        + "".join(
            f"load_library({name.removesuffix('.so')!r})\n"
            for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
        ),
        encoding="utf-8",
    )
    for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES:
        (binary_root / name).write_bytes(f"test-runtime-binary:{name}\n".encode())
    digest, _ = SNAPSHOT.source_tree_digest(source)
    target = task / "snapshots" / digest
    staging.rename(target)
    source = target / "source"
    binaries = [source / "gpu4pyscf" / "lib" / name
                for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES]
    manifest = {
        "schema": "gpu4pyscf.source-snapshot.v2",
        "tree_sha256": digest,
        "base_revision": SNAPSHOT.FROZEN_BASE_COMMIT,
        "source": str(source.resolve()),
        "immutable": True,
        "local_provenance": _provenance(),
        "runtime_binaries": {
            "source_root": "/pinned/gpu4pyscf/lib",
            "complete": True,
            "required_library_names": list(SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES),
            "inventory_library_names": list(SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES),
            "closure_complete": True,
            "files": [{
                "relative_path": f"gpu4pyscf/lib/{binary.name}",
                "sha256": SNAPSHOT._file_sha256(binary),
                "bytes": binary.stat().st_size,
            } for binary in binaries],
        },
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    os.chmod(source / "module.py", 0o444)
    os.chmod(source / "gpu4pyscf" / "runtime_targets.py", 0o444)
    for binary in binaries:
        os.chmod(binary, 0o444)
    os.chmod(binaries[0].parent, 0o555)
    os.chmod(binaries[0].parent.parent, 0o555)
    os.chmod(source, 0o555)
    os.chmod(target / "manifest.json", 0o444)
    os.chmod(target, 0o555)
    return task, source


def _make_writable(source: Path) -> None:
    os.chmod(source.parent, 0o755)
    os.chmod(source, 0o755)
    os.chmod(source / "module.py", 0o644)
    os.chmod(source / "gpu4pyscf" / "runtime_targets.py", 0o644)
    binary_root = source / "gpu4pyscf" / "lib"
    os.chmod(binary_root.parent, 0o755)
    os.chmod(binary_root, 0o755)
    for binary in binary_root.glob("*.so"):
        os.chmod(binary, 0o644)
    os.chmod(source.parent / "manifest.json", 0o644)


def test_valid_content_addressed_read_only_snapshot(tmp_path: Path):
    task, source = _installed_snapshot(tmp_path)
    try:
        evidence = SNAPSHOT.validate_snapshot(
            source,
            expected_task_root=task,
            expected_deployment_profile="g0-canonical-pristine",
            require_canonical_pristine=True,
        )
        assert evidence["valid"] is True
        assert evidence["tree_sha256"] == source.parent.name
        assert evidence["read_only"] is True
        assert evidence["required_runtime_libraries"] == list(
            SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
        )
        assert evidence["manifest"]["runtime_binaries"][
            "closure_complete"
        ] is True
    finally:
        _make_writable(source)


def test_stable_mutable_snapshot_is_rejected(tmp_path: Path):
    task, source = _installed_snapshot(tmp_path)
    _make_writable(source)
    evidence = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
    assert evidence["valid"] is False
    assert "writable" in " ".join(evidence["reasons"])


def test_manifest_and_task_root_mismatches_are_rejected(tmp_path: Path):
    task, source = _installed_snapshot(tmp_path)
    try:
        os.chmod(source.parent, 0o755)
        manifest_path = source.parent / "manifest.json"
        os.chmod(manifest_path, 0o644)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["base_revision"] = "wrong"
        manifest["source"] = str(tmp_path / "wrong-source")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        os.chmod(manifest_path, 0o444)
        os.chmod(source.parent, 0o555)
        evidence = SNAPSHOT.validate_snapshot(
            source, expected_task_root=tmp_path / "other-task"
        )
        assert evidence["valid"] is False
        reasons = " ".join(evidence["reasons"])
        assert "base revision" in reasons
        assert "source path" in reasons
        assert "expected task root" in reasons
    finally:
        _make_writable(source)


def test_runtime_binary_digest_and_inventory_are_required(tmp_path: Path):
    task, source = _installed_snapshot(tmp_path)
    try:
        os.chmod(source.parent, 0o755)
        os.chmod(source, 0o755)
        binary = source / "gpu4pyscf" / "lib" / "libgint.so"
        os.chmod(binary.parent.parent, 0o755)
        os.chmod(binary.parent, 0o755)
        os.chmod(binary, 0o644)
        binary.write_bytes(b"changed\n")
        os.chmod(binary, 0o444)
        os.chmod(binary.parent, 0o555)
        os.chmod(binary.parent.parent, 0o555)
        os.chmod(source, 0o555)
        os.chmod(source.parent, 0o555)
        evidence = SNAPSHOT.validate_snapshot(
            source, expected_task_root=task
        )
        assert evidence["valid"] is False
        assert "runtime binary" in " ".join(evidence["reasons"])
    finally:
        _make_writable(source)


def test_digest_protocol_matches_benchmark(tmp_path: Path):
    benchmark_spec = importlib.util.spec_from_file_location(
        "water8_benchmark_for_snapshot_test", ROOT / "benchmark.py"
    )
    benchmark = importlib.util.module_from_spec(benchmark_spec)
    assert benchmark_spec.loader is not None
    benchmark_spec.loader.exec_module(benchmark)
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.json").write_text("{}\n", encoding="utf-8")
    assert SNAPSHOT.source_tree_digest(tmp_path) == benchmark._source_tree_digest(
        tmp_path
    )


def test_required_runtime_closure_is_discovered_from_source():
    discovered = SNAPSHOT.discover_required_runtime_libraries(
        ROOT.parents[2]
    )
    assert discovered == SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
    assert "libmgrid_v3.so" in discovered


def test_missing_mgrid_v3_is_rejected_before_runtime_use(tmp_path: Path):
    task, source = _installed_snapshot(tmp_path)
    try:
        os.chmod(source.parent, 0o755)
        os.chmod(source, 0o755)
        binary_root = source / "gpu4pyscf" / "lib"
        os.chmod(binary_root.parent, 0o755)
        os.chmod(binary_root, 0o755)
        missing = binary_root / "libmgrid_v3.so"
        os.chmod(missing, 0o644)
        missing.unlink()
        os.chmod(binary_root, 0o555)
        os.chmod(binary_root.parent, 0o555)
        os.chmod(source, 0o555)
        os.chmod(source.parent, 0o555)
        evidence = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert evidence["valid"] is False
        reasons = " ".join(evidence["reasons"])
        assert "libmgrid_v3.so" in reasons
        assert "closure is missing" in reasons
    finally:
        _make_writable(source)


def test_candidate_profile_cannot_satisfy_canonical_pristine_gate(tmp_path: Path):
    task, source = _installed_snapshot(tmp_path)
    try:
        os.chmod(source.parent, 0o755)
        manifest_path = source.parent / "manifest.json"
        os.chmod(manifest_path, 0o644)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["local_provenance"] = _provenance("candidate")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        os.chmod(manifest_path, 0o444)
        os.chmod(source.parent, 0o555)

        candidate = SNAPSHOT.validate_snapshot(
            source,
            expected_task_root=task,
            expected_deployment_profile="candidate",
        )
        assert candidate["valid"] is True
        g0 = SNAPSHOT.validate_snapshot(
            source,
            expected_task_root=task,
            expected_deployment_profile="g0-canonical-pristine",
            require_canonical_pristine=True,
        )
        assert g0["valid"] is False
        reasons = " ".join(g0["reasons"])
        assert "deployment profile" in reasons
        assert "canonical-pristine" in reasons
    finally:
        _make_writable(source)


def test_tampered_provenance_status_digest_is_rejected(tmp_path: Path):
    task, source = _installed_snapshot(tmp_path)
    try:
        os.chmod(source.parent, 0o755)
        manifest_path = source.parent / "manifest.json"
        os.chmod(manifest_path, 0o644)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["local_provenance"]["git_status"]["entries"].append(
            "?? outside-allowlist.txt"
        )
        manifest["local_provenance"]["git_status"]["entry_count"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        os.chmod(manifest_path, 0o444)
        os.chmod(source.parent, 0o555)
        evidence = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert evidence["valid"] is False
        reasons = " ".join(evidence["reasons"])
        assert "status digest" in reasons
        assert "allowed-path/status digest" in reasons
    finally:
        _make_writable(source)


def test_deploy_seals_and_checks_staging_before_atomic_rename():
    script = (ROOT / "deploy_mtu_snapshot.sh").read_text(encoding="utf-8")
    seal = script.index("# Seal and re-check staging completely")
    sealed_digest = script.index("sealed_digest, sealed_count")
    atomic_rename = script.index("os.rename(staging, target)")
    assert seal < sealed_digest < atomic_rename
    assert "missing required libraries" in script
    assert '"required_library_names": required_library_names' in script
    assert '"inventory_library_names": inventory_library_names' in script
    # The developer host uses Bash 3.2; an empty array expansion under
    # ``set -u`` aborts candidate-profile validation before upload.
    assert "local pristine_argument=\"\"" in script
    assert "pristine_argument[@]" not in script
    assert "requested runtime inventory differs from sealed snapshot" in script
    assert "requested runtime binary differs from sealed snapshot" in script
    assert 'requested_root.glob("*.so")' in script
    existing = script.index('if remote_ssh test -e "${SNAPSHOT_ROOT}"')
    runtime_identity = script.index(
        "requested runtime inventory differs from sealed snapshot"
    )
    publish = script.index("os.rename(staging, target)")
    assert runtime_identity < existing < publish


def test_mgrid_build_helper_uses_external_source_copy_and_curated_inventory():
    script = (ROOT / "build_mgrid_v3_bundle.sh").read_text(encoding="utf-8")
    assert '"${SOURCE_ROOT}/gpu4pyscf/lib/" "${WORK_ROOT}/source/"' in script
    assert '--target mgrid_v3' in script
    assert '"${WORK_ROOT}/source/libmgrid_v3.so"' in script
    assert 'inventory != list(required)' in script
