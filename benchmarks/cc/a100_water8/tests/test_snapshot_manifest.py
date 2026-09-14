from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "water8_snapshot_manifest", ROOT / "snapshot_manifest.py"
)
SNAPSHOT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SNAPSHOT)
NORMALIZER_SPEC = importlib.util.spec_from_file_location(
    "water8_snapshot_normalizer_for_manifest_test",
    ROOT / "snapshot_symlink_normalize.py",
)
NORMALIZER = importlib.util.module_from_spec(NORMALIZER_SPEC)
assert NORMALIZER_SPEC.loader is not None
NORMALIZER_SPEC.loader.exec_module(NORMALIZER)


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


def _write_publication_attestation(
    target: Path, *, protocol: str = "atomic-exclusive-rename"
) -> None:
    manifest_bytes = (target / "manifest.json").read_bytes()
    identity = target.stat()
    attestation = {
        "schema": SNAPSHOT.PUBLICATION_ATTESTATION_SCHEMA,
        "status": "published",
        "source_tree_sha256": target.name,
        "snapshot_name": target.name,
        "published_directory_identity": {
            "device": identity.st_dev,
            "inode": identity.st_ino,
        },
        "publish_protocol": protocol,
        "trust_boundary": SNAPSHOT.PUBLISH_TRUST_BOUNDARY,
        "publication_policy_sha256": SNAPSHOT._canonical_json_sha256(
            SNAPSHOT.PUBLICATION_POLICY
        ),
        "published_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "prepublication_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    attestation["attestation_sha256"] = SNAPSHOT._canonical_json_sha256(
        attestation
    )
    path = target.parent / f"{target.name}{SNAPSHOT.PUBLICATION_ATTESTATION_SUFFIX}"
    path.write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def _upgrade_to_v3(source: Path) -> Path:
    manifest_path = source.parent / "manifest.json"
    inventory = NORMALIZER.inspect_source(source)
    normalization = NORMALIZER.materialize_source_links(source, inventory)
    normalized = NORMALIZER.assert_normalized_source(source)
    uploaded = (
        json.dumps(normalization, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    normalization.update({
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
    os.chmod(source.parent, 0o755)
    os.chmod(manifest_path, 0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "schema": SNAPSHOT.SOURCE_SNAPSHOT_SCHEMA,
        "source_normalization": normalization,
        "publication_policy": SNAPSHOT.PUBLICATION_POLICY,
    })
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    os.chmod(manifest_path, 0o444)
    _write_publication_attestation(source.parent)
    os.chmod(source.parent, 0o555)
    return manifest_path


def _rewrite_v3_manifest(source: Path, manifest: dict) -> None:
    target = source.parent
    manifest_path = target / "manifest.json"
    target.chmod(0o755)
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o444)
    attestation_path = target.parent / (
        f"{target.name}{SNAPSHOT.PUBLICATION_ATTESTATION_SUFFIX}"
    )
    if attestation_path.exists():
        attestation_path.unlink()
    _write_publication_attestation(target)
    target.chmod(0o555)


def _candidate_runtime_evidence(task: Path, source: Path) -> dict:
    runtime_files = json.loads(
        (source.parent / "manifest.json").read_text(encoding="utf-8")
    )["runtime_binaries"]["files"]
    inventory = [{
        "name": name,
        "bytes": entry["bytes"],
        "sha256": entry["sha256"],
    } for name, entry in zip(
        SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES, runtime_files, strict=True
    )]
    inventory_sha = SNAPSHOT._canonical_json_sha256(inventory)
    source_sha, source_count = SNAPSHOT.source_tree_digest(source)
    bundle_root = task / "runtime-bundles" / inventory_sha
    evidence = {
        "schema": SNAPSHOT.RUNTIME_VALIDATION_EVIDENCE_SCHEMA,
        "validated": True,
        "bundle_id": inventory_sha,
        "bundle_root": str(bundle_root),
        "manifest_path": str(
            task / "runtime-bundle-manifests" / f"{inventory_sha}.json"
        ),
        "build_verification_path": str(
            task / "builds"
            / f"{source_sha}-gint-bounded-workspace-v2-sm80"
            / "build-verification.json"
        ),
        "library_count": len(inventory),
        "inventory_sha256": inventory_sha,
        "inventory_payload_sha256": inventory_sha,
        "source_sha256": source_sha,
        "source_file_count": source_count,
        "cuda_version": "12.8",
        "cuda_architecture": "80-real",
        "abi_version": 2,
        "maximum_stack_bytes": 1024,
        "build_verification_bound": True,
        "sidecar_sha256": "a" * 64,
        "sidecar_bytes": 4096,
        "payload_sha256": "b" * 64,
        "build_verification_sha256": "c" * 64,
        "build_verification_bytes": 8192,
        "build_verification_payload_sha256": "d" * 64,
    }
    return evidence


def _installed_candidate_v3(tmp_path: Path) -> tuple[Path, Path]:
    task, source = _installed_snapshot(tmp_path)
    manifest_path = source.parent / "manifest.json"
    source.parent.chmod(0o755)
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["local_provenance"] = _provenance("candidate")
    evidence = _candidate_runtime_evidence(task, source)
    runtime = manifest["runtime_binaries"]
    runtime["source_root"] = evidence["bundle_root"]
    runtime.update({
        "contract_schema": SNAPSHOT.RUNTIME_BUNDLE_SCHEMA,
        "bundle_id": evidence["bundle_id"],
        "inventory_sha256": evidence["inventory_sha256"],
        "sidecar_sha256": evidence["sidecar_sha256"],
        "build_verification_sha256": evidence[
            "build_verification_sha256"
        ],
    })
    runtime["validation_evidence"] = evidence
    runtime["validation_evidence_sha256"] = SNAPSHOT._canonical_json_sha256(
        evidence
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o444)
    _upgrade_to_v3(source)
    return task, source


def _release_pin(
    source: Path, qualification_sha: str, *, receipt_sha: str, payload_sha: str
) -> dict:
    manifest_bytes = (source.parent / "manifest.json").read_bytes()
    return {
        "schema": SNAPSHOT.RELEASE_PIN_SCHEMA,
        "status": "release-pinned",
        "release_pin_path": SNAPSHOT.RELEASE_PIN_RELATIVE_PATH.as_posix(),
        "lineage": {
            "model": "qualification-source-plus-fixed-release-pin",
            "qualification_source_tree_sha256": qualification_sha,
            "release_source_without_pin_sha256": qualification_sha,
        },
        "receipt": {
            "sha256": receipt_sha,
            "payload_sha256": payload_sha,
        },
        "qualification_source": {
            "root": str(source),
            "tree_sha256": qualification_sha,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "manifest_bytes": len(manifest_bytes),
            "binding_sha256": "e" * 64,
        },
        "qualification": {
            "job_id": "70001",
            "topology_fingerprint": {},
            "topology_fingerprint_sha256": "f" * 64,
            "binding_sha256": "1" * 64,
        },
        "release_runtime_contract": {
            "schema": "test",
            "observation_source": "test",
            "job_name": "test",
            "node": "test",
            "host": "test",
            "partition": "test",
            "cpu_affinity": "24-31",
            "gpu_pci_bus_ids": ["0000:01:00.0"],
            "gpu_node_cpulist": "24-31",
            "gpu_numa_node": 3,
            "mems_allowed_list": "3",
            "memory_policy": "preferred: 3",
            "topology_path_template": "test",
            "qualification_topology_fingerprint_sha256": "2" * 64,
        },
        "runtime": {
            "libgint_sha256": "3" * 64,
            "libgint_bytes": 1,
            "basis_prod_cache_abi_sha256": "4" * 64,
            "selected_pair_abi_sha256": "5" * 64,
            "selected_c_abi_sha256": "6" * 64,
            "binding_sha256": "7" * 64,
        },
        "transfer_audit_sha256": "8" * 64,
    }


def _promote_candidate_to_release_b(
    task: Path, source: Path
) -> tuple[Path, str]:
    qualification_sha = source.parent.name
    qualification_attestation_path = source.parent.parent / (
        f"{qualification_sha}{SNAPSHOT.PUBLICATION_ATTESTATION_SUFFIX}"
    )
    qualification_attestation = json.loads(
        qualification_attestation_path.read_text(encoding="utf-8")
    )
    receipt_sha = "9" * 64
    payload_sha = "0" * 64
    pin = _release_pin(
        source, qualification_sha,
        receipt_sha=receipt_sha,
        payload_sha=payload_sha,
    )
    package = source / "gpu4pyscf"
    package.chmod(0o755)
    cc = package / "cc"
    cc.mkdir()
    pin_path = cc / SNAPSHOT.RELEASE_PIN_RELATIVE_PATH.name
    pin_bytes = (json.dumps(pin, indent=2, sort_keys=True) + "\n").encode("utf-8")
    pin_path.write_bytes(pin_bytes)
    pin_path.chmod(0o444)
    cc.chmod(0o555)
    package.chmod(0o555)

    release_sha, _ = SNAPSHOT.source_tree_digest(source)
    old_target = source.parent
    release_target = old_target.parent / release_sha
    old_target.rename(release_target)
    release_source = release_target / "source"
    manifest = json.loads(
        (release_target / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["tree_sha256"] = release_sha
    manifest["source"] = str(release_source)
    manifest["gint_release_lineage"] = {
        "schema": SNAPSHOT.RELEASE_PIN_SCHEMA,
        "qualification_source_tree_sha256": qualification_sha,
        "release_pin_relative_path": SNAPSHOT.RELEASE_PIN_RELATIVE_PATH.as_posix(),
        "release_pin_sha256": hashlib.sha256(pin_bytes).hexdigest(),
        "receipt_sha256": receipt_sha,
        "receipt_payload_sha256": payload_sha,
        "qualification_publication_attestation": qualification_attestation,
        "qualification_publication_attestation_sha256": (
            qualification_attestation["attestation_sha256"]
        ),
    }
    _rewrite_v3_manifest(release_source, manifest)
    return release_source, qualification_sha


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


def test_v3_normalization_manifest_is_verified(tmp_path: Path):
    task, source = _installed_snapshot(tmp_path)
    manifest_path = source.parent / "manifest.json"
    try:
        inventory = NORMALIZER.inspect_source(source)
        normalization = NORMALIZER.materialize_source_links(source, inventory)
        uploaded = (
            json.dumps(normalization, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        normalized = NORMALIZER.assert_normalized_source(source)
        normalization.update({
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
        os.chmod(source.parent, 0o755)
        os.chmod(manifest_path, 0o644)
        manifest = json.loads(manifest_path.read_text())
        manifest["schema"] = SNAPSHOT.SOURCE_SNAPSHOT_SCHEMA
        manifest["source_normalization"] = normalization
        manifest["publication_policy"] = SNAPSHOT.PUBLICATION_POLICY
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        os.chmod(manifest_path, 0o444)
        _write_publication_attestation(source.parent)
        os.chmod(source.parent, 0o555)

        evidence = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert evidence["valid"] is True

        os.chmod(source.parent, 0o755)
        os.chmod(manifest_path, 0o644)
        manifest["source_normalization"]["policy"] = "forged"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        os.chmod(manifest_path, 0o444)
        os.chmod(source.parent, 0o555)
        rejected = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert rejected["valid"] is False
        assert "normalization policy" in " ".join(rejected["reasons"])
    finally:
        _make_writable(source)


def test_candidate_v3_binds_exact_runtime_bundle_evidence(tmp_path: Path):
    task, source = _installed_candidate_v3(tmp_path)
    try:
        accepted = SNAPSHOT.validate_snapshot(
            source,
            expected_task_root=task,
            expected_deployment_profile="candidate",
        )
        assert accepted["valid"] is True, accepted["reasons"]
        manifest = accepted["manifest"]
        assert SNAPSHOT.captured_runtime_bundle_evidence_errors(
            manifest["runtime_binaries"],
            deployment_profile="candidate",
            source_tree_sha256=accepted["tree_sha256"],
            source_file_count=accepted["files_hashed"],
            task_root=task,
        ) == []
    finally:
        _make_writable(source)


@pytest.mark.parametrize("missing", ["validation_evidence", "validation_evidence_sha256"])
def test_candidate_v3_requires_complete_runtime_evidence(
    tmp_path: Path, missing: str
):
    task, source = _installed_candidate_v3(tmp_path)
    try:
        manifest = json.loads(
            (source.parent / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["runtime_binaries"].pop(missing)
        _rewrite_v3_manifest(source, manifest)
        rejected = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert rejected["valid"] is False
        assert "runtime-bundle validation evidence" in " ".join(
            rejected["reasons"]
        )
    finally:
        _make_writable(source)


def test_candidate_v3_rejects_tampered_runtime_evidence_hash(tmp_path: Path):
    task, source = _installed_candidate_v3(tmp_path)
    try:
        manifest = json.loads(
            (source.parent / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["runtime_binaries"]["validation_evidence_sha256"] = "0" * 64
        _rewrite_v3_manifest(source, manifest)
        rejected = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert rejected["valid"] is False
        assert "evidence digest mismatch" in " ".join(rejected["reasons"])
    finally:
        _make_writable(source)


@pytest.mark.parametrize(
    "field",
    [
        "contract_schema",
        "bundle_id",
        "inventory_sha256",
        "sidecar_sha256",
        "build_verification_sha256",
        "unsealed_extension",
    ],
)
def test_candidate_v3_runtime_aliases_are_exact(
    tmp_path: Path, field: str
):
    task, source = _installed_candidate_v3(tmp_path)
    try:
        manifest = json.loads(
            (source.parent / "manifest.json").read_text(encoding="utf-8")
        )
        runtime = manifest["runtime_binaries"]
        runtime[field] = (
            "gpu4pyscf.runtime-bundle.v1"
            if field == "contract_schema"
            else (True if field == "unsealed_extension" else "0" * 64)
        )
        _rewrite_v3_manifest(source, manifest)
        rejected = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert rejected["valid"] is False
        reasons = " ".join(rejected["reasons"])
        assert "runtime-bundle" in reasons or "runtime-binary" in reasons
    finally:
        _make_writable(source)


@pytest.mark.parametrize("mutation", ["extra", "missing", "float"])
def test_candidate_v3_runtime_evidence_shape_is_exact(
    tmp_path: Path, mutation: str
):
    task, source = _installed_candidate_v3(tmp_path)
    try:
        manifest = json.loads(
            (source.parent / "manifest.json").read_text(encoding="utf-8")
        )
        runtime = manifest["runtime_binaries"]
        evidence = runtime["validation_evidence"]
        if mutation == "extra":
            evidence["unsealed_extension"] = True
        elif mutation == "missing":
            evidence.pop("build_verification_path")
        else:
            evidence["sidecar_bytes"] = 1.5
        runtime["validation_evidence_sha256"] = SNAPSHOT._canonical_json_sha256(
            evidence
        )
        _rewrite_v3_manifest(source, manifest)
        rejected = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert rejected["valid"] is False
        reasons = " ".join(rejected["reasons"])
        assert "runtime-bundle" in reasons and "evidence" in reasons
    finally:
        _make_writable(source)


@pytest.mark.parametrize("field", ["source_sha256", "source_file_count"])
def test_candidate_v3_runtime_evidence_binds_source_identity(
    tmp_path: Path, field: str
):
    task, source = _installed_candidate_v3(tmp_path)
    try:
        manifest = json.loads(
            (source.parent / "manifest.json").read_text(encoding="utf-8")
        )
        runtime = manifest["runtime_binaries"]
        evidence = runtime["validation_evidence"]
        if field == "source_sha256":
            evidence[field] = "0" * 64
        else:
            evidence[field] += 1
        runtime["validation_evidence_sha256"] = SNAPSHOT._canonical_json_sha256(
            evidence
        )
        _rewrite_v3_manifest(source, manifest)
        rejected = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert rejected["valid"] is False
        identity_name = "digest" if field.endswith("sha256") else "file count"
        assert f"source {identity_name} mismatch" in " ".join(rejected["reasons"])
    finally:
        _make_writable(source)


@pytest.mark.parametrize(
    ("field", "component"),
    [
        ("bundle_root", "runtime-bundles"),
        ("manifest_path", "runtime-bundle-manifests"),
        ("build_verification_path", "builds"),
    ],
)
def test_candidate_v3_runtime_evidence_paths_stay_under_task_root(
    tmp_path: Path, field: str, component: str
):
    task, source = _installed_candidate_v3(tmp_path)
    try:
        manifest = json.loads(
            (source.parent / "manifest.json").read_text(encoding="utf-8")
        )
        runtime = manifest["runtime_binaries"]
        evidence = runtime["validation_evidence"]
        evidence[field] = str(tmp_path / "other-task" / component / "forged")
        runtime["validation_evidence_sha256"] = SNAPSHOT._canonical_json_sha256(
            evidence
        )
        _rewrite_v3_manifest(source, manifest)
        rejected = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert rejected["valid"] is False
        assert "outside task root" in " ".join(rejected["reasons"])
    finally:
        _make_writable(source)


def test_captured_runtime_evidence_rejects_relative_task_root(tmp_path: Path):
    task, source = _installed_candidate_v3(tmp_path)
    try:
        manifest = json.loads(
            (source.parent / "manifest.json").read_text(encoding="utf-8")
        )
        digest, count = SNAPSHOT.source_tree_digest(source)
        reasons = SNAPSHOT.captured_runtime_bundle_evidence_errors(
            manifest["runtime_binaries"],
            deployment_profile="candidate",
            source_tree_sha256=digest,
            source_file_count=count,
            task_root=Path("relative-task"),
        )
        assert "runtime-bundle expected task root is invalid" in reasons
    finally:
        _make_writable(source)


@pytest.mark.parametrize("tamper", ["digest", "path", "extra"])
def test_candidate_v3_binds_exact_copied_runtime_inventory(
    tmp_path: Path, tamper: str
):
    task, source = _installed_candidate_v3(tmp_path)
    try:
        manifest = json.loads(
            (source.parent / "manifest.json").read_text(encoding="utf-8")
        )
        entries = manifest["runtime_binaries"]["files"]
        if tamper == "digest":
            entries[3]["sha256"] = "0" * 64
        elif tamper == "path":
            entries[3]["relative_path"] = "gpu4pyscf/lib/renamed.so"
        else:
            entries.append(dict(entries[-1]))
        _rewrite_v3_manifest(source, manifest)
        rejected = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert rejected["valid"] is False
        reasons = " ".join(rejected["reasons"])
        assert "runtime binary" in reasons or "runtime-binary" in reasons
        assert any(token in reasons for token in ("digest", "path", "exactly 13"))
    finally:
        _make_writable(source)


def test_candidate_release_b_runtime_evidence_remains_bound_to_a(tmp_path: Path):
    task, source = _installed_candidate_v3(tmp_path)
    release_source, qualification_sha = _promote_candidate_to_release_b(
        task, source
    )
    try:
        accepted = SNAPSHOT.validate_snapshot(
            release_source,
            expected_task_root=task,
            expected_deployment_profile="candidate",
        )
        assert accepted["valid"] is True, accepted["reasons"]
        runtime_evidence = accepted["manifest"]["runtime_binaries"][
            "validation_evidence"
        ]
        assert runtime_evidence["source_sha256"] == qualification_sha
        assert runtime_evidence["source_file_count"] == accepted["files_hashed"] - 1
        assert SNAPSHOT.source_tree_digest(
            release_source, exclude_release_pin=True
        )[0] == qualification_sha
    finally:
        _make_writable(release_source)


@pytest.mark.parametrize("tamper", ["pin", "qualification_digest", "extra_source"])
def test_candidate_release_b_rejects_invalid_a_to_b_lineage(
    tmp_path: Path, tamper: str
):
    task, source = _installed_candidate_v3(tmp_path)
    release_source, _ = _promote_candidate_to_release_b(task, source)
    if tamper == "pin":
        pin = release_source / SNAPSHOT.RELEASE_PIN_RELATIVE_PATH
        pin.chmod(0o644)
        pin.write_bytes(pin.read_bytes() + b" ")
        pin.chmod(0o444)
    elif tamper == "qualification_digest":
        manifest = json.loads(
            (release_source.parent / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["gint_release_lineage"][
            "qualification_source_tree_sha256"
        ] = "0" * 64
        _rewrite_v3_manifest(release_source, manifest)
    else:
        release_source.chmod(0o755)
        extra = release_source / "unapproved-release-delta.json"
        extra.write_text("{}\n", encoding="utf-8")
        extra.chmod(0o444)
        release_source.chmod(0o555)
    try:
        rejected = SNAPSHOT.validate_snapshot(
            release_source, expected_task_root=task
        )
        assert rejected["valid"] is False
        reasons = " ".join(rejected["reasons"])
        assert any(token in reasons for token in (
            "release pin digest",
            "qualification attestation source",
            "excluding only the fixed pin",
            "source digest differs",
        ))
    finally:
        _make_writable(release_source)


def test_g0_release_b_cannot_hide_malformed_lineage_without_runtime_evidence(
    tmp_path: Path,
):
    task, source = _installed_candidate_v3(tmp_path)
    release_source, _ = _promote_candidate_to_release_b(task, source)
    try:
        manifest = json.loads(
            (release_source.parent / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        manifest["local_provenance"] = _provenance(
            "g0-canonical-pristine"
        )
        runtime = manifest["runtime_binaries"]
        runtime["source_root"] = "/legacy/gpu4pyscf/lib"
        for field in (
            "contract_schema",
            "bundle_id",
            "inventory_sha256",
            "sidecar_sha256",
            "build_verification_sha256",
            "validation_evidence",
            "validation_evidence_sha256",
        ):
            runtime.pop(field)
        manifest["gint_release_lineage"].pop("receipt_sha256")
        _rewrite_v3_manifest(release_source, manifest)
        rejected = SNAPSHOT.validate_snapshot(
            release_source,
            expected_task_root=task,
            expected_deployment_profile="g0-canonical-pristine",
            require_canonical_pristine=True,
        )
        assert rejected["valid"] is False
        assert "GINT release lineage fields are not exact" in " ".join(
            rejected["reasons"]
        )
    finally:
        _make_writable(release_source)


@pytest.mark.parametrize(
    "tamper", ["missing", "malformed", "duplicate", "replaced", "writable", "mismatched"]
)
def test_v3_publication_attestation_is_required_and_bound(tmp_path: Path, tamper: str):
    task, source = _installed_snapshot(tmp_path)
    _upgrade_to_v3(source)
    attestation_path = source.parent.parent / (
        f"{source.parent.name}{SNAPSHOT.PUBLICATION_ATTESTATION_SUFFIX}"
    )
    source.parent.chmod(0o755)
    if tamper == "missing":
        attestation_path.unlink()
    elif tamper == "malformed":
        attestation_path.chmod(0o644)
        attestation_path.write_text("not-json\n", encoding="utf-8")
        attestation_path.chmod(0o444)
    elif tamper == "duplicate":
        attestation_path.chmod(0o644)
        attestation_path.write_text(
            '{"schema":"x","schema":"y"}\n', encoding="utf-8"
        )
        attestation_path.chmod(0o444)
    elif tamper == "replaced":
        attestation_path.chmod(0o644)
        value = json.loads(attestation_path.read_text())
        value["published_directory_identity"]["inode"] = 1
        attestation_path.write_text(json.dumps(value), encoding="utf-8")
        attestation_path.chmod(0o444)
    elif tamper == "writable":
        attestation_path.chmod(0o644)
    else:
        attestation_path.chmod(0o644)
        value = json.loads(attestation_path.read_text())
        value["source_tree_sha256"] = "0" * 64
        attestation_path.write_text(json.dumps(value), encoding="utf-8")
        attestation_path.chmod(0o444)
    evidence = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
    assert evidence["valid"] is False
    reasons = " ".join(evidence["reasons"])
    assert "publication attestation" in reasons
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


def test_candidate_without_v2_runtime_evidence_fails_closed(tmp_path: Path):
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
        assert candidate["valid"] is False
        assert "runtime-bundle validation evidence is missing" in " ".join(
            candidate["reasons"]
        )
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


def test_legacy_manifest_cannot_publish_a_symbolic_link(tmp_path: Path):
    task, source = _installed_snapshot(tmp_path)
    alias = source / "module-alias.py"
    try:
        os.chmod(source.parent, 0o755)
        os.chmod(source, 0o755)
        alias.symlink_to("module.py")
        os.chmod(source, 0o555)
        os.chmod(source.parent, 0o555)
        evidence = SNAPSHOT.validate_snapshot(source, expected_task_root=task)
        assert evidence["valid"] is False
        assert evidence["published_symlinks"] == ["module-alias.py"]
        assert "symbolic links" in " ".join(evidence["reasons"])
    finally:
        os.chmod(source.parent, 0o755)
        os.chmod(source, 0o755)
        if alias.is_symlink():
            alias.unlink()
        _make_writable(source)


def test_source_root_symlink_is_rejected_before_resolution(tmp_path: Path):
    task, source = _installed_snapshot(tmp_path)
    alias = tmp_path / "source-alias"
    alias.symlink_to(source, target_is_directory=True)
    try:
        evidence = SNAPSHOT.validate_snapshot(alias)
        assert evidence["valid"] is False
        reasons = " ".join(evidence["reasons"])
        assert "source path is unsafe" in reasons
        assert "symbolic link" in reasons
        with pytest.raises(RuntimeError, match="symbolic link"):
            SNAPSHOT.source_tree_digest(alias)
    finally:
        alias.unlink()
        _make_writable(source)


def test_deploy_seals_and_checks_staging_before_atomic_rename():
    script = (ROOT / "deploy_mtu_snapshot.sh").read_text(encoding="utf-8")
    assembly = (ROOT / "snapshot_remote_assembly.py").read_text(encoding="utf-8")
    seal = assembly.index("# Seal and re-check staging completely")
    sealed_digest = assembly.index("sealed_digest, sealed_count")
    atomic_rename = assembly.index(
        "publish_protocol = _publish_directory_noreplace"
    )
    manifest_close = assembly.index("os.close(manifest_fd)")
    attestation_create = assembly.index("attestation_fd = os.open")
    assert seal < sealed_digest < atomic_rename
    assert manifest_close < atomic_rename < attestation_create
    assert assembly.index("_write_all(manifest_fd, manifest_data)") < manifest_close
    assert assembly.index("_write_all(attestation_fd, attestation_data)") > atomic_rename
    assert "missing required libraries" in assembly
    assert '"required_library_names": required_library_names' in assembly
    assert '"inventory_library_names": inventory_library_names' in assembly
    assert 'LOCAL_STAGING_ROOT="$(mktemp -d' in script
    assert script.count('"frozen_base_revision": frozen,') == 1
    assert 'materialize_local_source "${LOCAL_STAGING_ROOT}"' in script
    assert 'materialize_local_source "${LOCAL_CHECK_ROOT}"' in script
    assert 'snapshot_bundle_receiver.py' in script
    assert 'snapshot_remote_assembly.py' in script
    assert 'REMOTE_BUNDLE_RECEIVER="${LOCAL_SOURCE_ROOT}/' in script
    assert 'REMOTE_SNAPSHOT_ASSEMBLY="${LOCAL_SOURCE_ROOT}/' in script
    assert '--post-receive-script "${REMOTE_SNAPSHOT_ASSEMBLY}"' in script
    assert 'BUNDLE_RECEIPT="$(remote_ssh' in script
    assert '<"${REMOTE_UPLOAD_PROGRAM}"' in script
    assert 'before releasing the fds' in script
    assert 'rsync -az' not in script
    assert 'os.fchdir(staging_fd)' in assembly
    assert 'staging = pathlib.Path(".")' in assembly
    assert 'source = pathlib.Path("source")' in assembly
    assert 'os.fchdir(previous_cwd_fd)' in assembly
    assert 'normalizer.assert_normalized_source_fd(source_fd)' in assembly
    assert 'os.fchmod(source_fd, 0o555)' in assembly
    assert 'os.fchmod(staging_fd, 0o555)' in assembly
    assert '"schema": snapshot_manifest.SOURCE_SNAPSHOT_SCHEMA' in assembly
    assert 'source-link normalization inventory changed during bundle creation' in script
    assert 'local normalized source digest or file count changed' in script
    assert (
        'copy_regular_nofollow(runtime_binary_fd, name, runtime_target / name)'
        in assembly
    )
    assert (
        'label="runtime binary source root"' in assembly
        and 'runtime_binary_source_text' in assembly
        and 'runtime-binary root changed before publish' in assembly
    )
    assert 'normalization_errors = snapshot_manifest._normalization_errors' in assembly
    assert 'verify_existing_snapshot()' not in script
    assert 'automatic reuse is disabled: ${SNAPSHOT_ROOT}' in script
    assert 'if [[ "${SNAPSHOT_STATE}" == "directory" ]]' in script
    assert 'remote_staging_guard create' in script
    assert script.count('remote_staging_guard verify-staging') == 1
    assert script.count('remote_staging_guard verify-child') == 1
    assert 'remote_staging_guard remove' in script
    # Before a callback receipt exists, cleanup is restricted to the old
    # staging name; probing the digest target would destroy an indeterminate
    # publication.  The digest name is eligible only after a complete receipt.
    assert '"${STAGING_NAME}" \\\n        "${SNAPSHOTS_DEV}"' in script
    assert 'cleanup_name="${SNAPSHOT_NAME:-}"' in script
    assert 'digest target ${SNAPSHOT_NAME:-unset} was deliberately not probed' in script
    assert 'REMOTE_VALIDATION="$(remote_ssh' in script
    assert script.index('PUBLICATION_RECEIPT_OBTAINED=1') > script.index(
        'REMOTE_VALIDATION="$(remote_ssh'
    )
    assert '"${STAGING_DEV}" "${STAGING_INO}"' in script
    assert script.index('remote_staging_guard remove') < script.index(
        'rm -rf "${LOCAL_STAGING_ROOT}"'
    )
    assert 'snapshot_staging_guard.py' in script
    assert '.staging-${TREE_SHA256}-$$' not in script
    assert 'remote_ssh mkdir -p "${STAGING_ROOT}' not in script
    assert 'remote_ssh test -e "${SNAPSHOT_ROOT}"' not in script
    assert 'remote_ssh chmod -R' not in script
    assert 'remote_ssh rm -rf' not in script
    assert 'renameat2(RENAME_NOREPLACE)' in assembly
    assert 'reserved-empty-directory-rename' in assembly
    assert 'snapshot target reservation was replaced' in assembly
    assert 'snapshot target reservation is not empty' in assembly
    assert 'atomic-exclusive-rename-commit-confirmed' in script
    assert 'reserved-empty-directory-rename-commit-confirmed' in script
    assert 'private-anchor-owner-and-root-are-trusted-cooperators' in script
    assert '"publish_trust_boundary": PUBLISH_TRUST_BOUNDARY' in assembly
    assert 'PUBLICATION_POLICY = {' in assembly
    assert 'publication_attestation_name' in assembly
    assert 'PUBLICATION_ATTESTATION_SUFFIX' in assembly
    assert 'manifest["publication_attestation"]' not in assembly
    assert 'pwd -P)' in script
    assert 'Path(sys.argv[1]).resolve(strict=True)' in script


def test_deploy_refuses_existing_digest_without_reuse_validation():
    script = (ROOT / "deploy_mtu_snapshot.sh").read_text(encoding="utf-8")
    start = script.index('if [[ "${SNAPSHOT_STATE}" == "directory" ]]')
    end = script.index("\nfi", start)
    branch = script[start:end]

    assert 'automatic reuse is disabled: ${SNAPSHOT_ROOT}' in branch
    assert "exit 5" in branch
    assert "remote_ssh" not in branch
    assert "snapshot_manifest.py" not in branch
    assert "manifest.json" not in branch
    assert "MTU_GPU4PYSCF_BINARY_ROOT" not in branch
    assert "verify_existing_snapshot" not in script


def test_candidate_deploy_requires_explicit_runtime_bundle_before_ssh(
    tmp_path: Path,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "ssh-was-called"
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        f"#!/bin/sh\ntouch {marker!s}\nexit 97\n", encoding="utf-8"
    )
    fake_ssh.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = str(fake_bin) + os.pathsep + environment["PATH"]
    environment["DEPLOYMENT_PROFILE"] = "candidate"
    environment["MTU_RUNTIME_BUNDLE_ROOT"] = ""
    completed = subprocess.run(
        ["bash", str(ROOT / "deploy_mtu_snapshot.sh")],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "requires explicit MTU_RUNTIME_BUNDLE_ROOT" in completed.stderr
    assert not marker.exists()


def test_mgrid_build_helper_uses_external_source_copy_and_curated_inventory():
    script = (ROOT / "build_mgrid_v3_bundle.sh").read_text(encoding="utf-8")
    assert '"${SOURCE_ROOT}/gpu4pyscf/lib/" "${WORK_ROOT}/source/"' in script
    assert '--target mgrid_v3' in script
    assert '"${WORK_ROOT}/source/libmgrid_v3.so"' in script
    assert 'inventory != list(required)' in script
