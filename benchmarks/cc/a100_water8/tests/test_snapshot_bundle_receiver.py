"""Security tests for the descriptor-anchored snapshot bundle receiver."""

from __future__ import annotations

import base64
import errno
from io import BytesIO
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RECEIVER = _load(
    "water8_snapshot_bundle_receiver_test",
    ROOT / "snapshot_bundle_receiver.py",
)
GUARD = _load(
    "water8_snapshot_staging_guard_for_bundle_test",
    ROOT / "snapshot_staging_guard.py",
)
NORMALIZER = _load(
    "water8_snapshot_normalizer_for_bundle_test",
    ROOT / "snapshot_symlink_normalize.py",
)
SNAPSHOT = _load(
    "water8_snapshot_manifest_for_bundle_test",
    ROOT / "snapshot_manifest.py",
)
ASSEMBLY = _load(
    "water8_snapshot_remote_assembly_test",
    ROOT / "snapshot_remote_assembly.py",
)
RUNTIME_CONTRACT = _load(
    "water8_runtime_bundle_contract_for_bundle_test",
    ROOT / "runtime_bundle_contract.py",
)


def _publish_fixture(tmp_path: Path) -> tuple[Path, int, int, tuple[int, int]]:
    root = tmp_path.resolve() / "publish-root"
    source = root / "staging"
    source.mkdir(parents=True)
    (source / "payload.txt").write_text("staged payload\n", encoding="utf-8")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    root_fd = os.open(root, directory_flags)
    source_fd = os.open("staging", directory_flags, dir_fd=root_fd)
    source_info = os.fstat(source_fd)
    return root, root_fd, source_fd, (source_info.st_dev, source_info.st_ino)


def _source_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path.resolve() / "local-source"
    (source / "nested").mkdir(parents=True)
    (source / "root.txt").write_bytes(b"root payload\n")
    (source / "nested" / "module.py").write_bytes(b"answer = 42\n")
    provenance = tmp_path.resolve() / "local-provenance.json"
    normalization = tmp_path.resolve() / "source-normalization.json"
    provenance.write_text('{"schema":"test-provenance"}\n', encoding="utf-8")
    normalization.write_text(
        '{"schema":"test-normalization"}\n', encoding="utf-8"
    )
    return source, provenance, normalization


def _remote_staging(tmp_path: Path, marker: str = "a") -> tuple[Path, dict]:
    root = tmp_path.resolve() / "private" / "task" / "snapshots"
    root.parent.mkdir(parents=True, exist_ok=True)
    # Model the deployment user's private /mnt/mridata/vxu anchor.
    (tmp_path.resolve() / "private").chmod(0o700)
    created = GUARD.create_staging(str(root), marker * 64)
    return root, created


def _receive_arguments(root: Path, created: dict, payload: bytes) -> list[str]:
    return [
        str(root),
        created["staging_name"],
        str(created["snapshots_dev"]),
        str(created["snapshots_ino"]),
        str(created["staging_dev"]),
        str(created["staging_ino"]),
        "source",
        str(created["source_dev"]),
        str(created["source_ino"]),
        hashlib.sha256(payload).hexdigest(),
        str(len(payload)),
    ]


def _receive_direct(root: Path, created: dict, payload: bytes):
    arguments = _receive_arguments(root, created, payload)
    return RECEIVER.receive_bundle(
        payload,
        destination_root=arguments[0],
        container_name=arguments[1],
        root_identity=(int(arguments[2]), int(arguments[3])),
        container_identity=(int(arguments[4]), int(arguments[5])),
        tree_name=arguments[6],
        tree_identity=(int(arguments[7]), int(arguments[8])),
        expected_sha256=arguments[9],
        expected_bytes=int(arguments[10]),
    )


def _tar_payload(
    extra: list[tuple[tarfile.TarInfo, bytes | None]],
) -> bytes:
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []
    required = {
        ".bundle-metadata.json": (
            json.dumps(
                {"schema": RECEIVER.BUNDLE_SCHEMA},
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
        ).encode("ascii"),
        "local-provenance.json": b"{}\n",
        "source-normalization.json": b"{}\n",
    }
    for name, data in required.items():
        member = tarfile.TarInfo(name)
        member.type = tarfile.REGTYPE
        member.size = len(data)
        members.append((member, data))
    members.extend(extra)
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for member, data in members:
            archive.addfile(member, None if data is None else BytesIO(data))
    return output.getvalue()


def test_embedded_program_roundtrip_uses_only_regular_files(tmp_path: Path):
    source, provenance, normalization = _source_inputs(tmp_path)
    payload = RECEIVER.build_bundle(source, provenance, normalization)
    receiver_source = (ROOT / "snapshot_bundle_receiver.py").read_bytes()
    program = RECEIVER.build_embedded_program(receiver_source, payload)
    root, created = _remote_staging(tmp_path)

    completed = subprocess.run(
        [sys.executable, "-", *_receive_arguments(root, created, payload)],
        input=program,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    receipt = json.loads(completed.stdout)
    staging = Path(created["staging_path"])
    received = staging / "source"

    assert receipt["schema"] == RECEIVER.RECEIPT_SCHEMA
    assert receipt["bundle_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["symlink_count"] == receipt["special_file_count"] == 0
    assert (received / "root.txt").read_bytes() == b"root payload\n"
    assert (received / "nested" / "module.py").read_bytes() == b"answer = 42\n"
    assert (staging / "local-provenance.json").read_bytes() == provenance.read_bytes()
    assert (
        staging / "source-normalization.json"
    ).read_bytes() == normalization.read_bytes()
    assert not (staging / ".bundle-metadata.json").exists()
    for directory, names, files in os.walk(staging, followlinks=False):
        for name in [*names, *files]:
            info = (Path(directory) / name).lstat()
            assert stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)


def test_receiver_contract_is_reusable_for_build_inputs(tmp_path: Path):
    source, provenance, normalization = _source_inputs(tmp_path)
    payload = RECEIVER.build_bundle(source, provenance, normalization)
    private = tmp_path.resolve() / "private"
    root = private / "task" / "build-inputs"
    container = root / "exact-source-sm80"
    tree = container / "normalized-tree"
    tree.mkdir(parents=True)
    private.chmod(0o700)
    for path in (root, container, tree):
        path.chmod(0o700)
    identity = lambda path: (path.stat().st_dev, path.stat().st_ino)

    receipt = RECEIVER.receive_bundle(
        payload,
        destination_root=str(root),
        container_name=container.name,
        root_identity=identity(root),
        container_identity=identity(container),
        tree_name=tree.name,
        tree_identity=identity(tree),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_bytes=len(payload),
    )

    assert receipt["container_name"] == "exact-source-sm80"
    assert receipt["tree_name"] == "normalized-tree"
    assert (tree / "nested" / "module.py").read_bytes() == b"answer = 42\n"


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for directory, names, files in os.walk(root):
        Path(directory).chmod(0o700)
        for name in names:
            (Path(directory) / name).chmod(0o700)
        for name in files:
            (Path(directory) / name).chmod(0o600)


def _candidate_runtime_bundle(
    task_root: Path, source_sha256: str, source_file_count: int
) -> Path:
    payloads = {
        name: f"runtime:{name}\n".encode("ascii")
        for name in RUNTIME_CONTRACT.REQUIRED_LIBRARIES
    }
    inventory = [
        {
            "name": name,
            "bytes": len(payloads[name]),
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }
        for name in RUNTIME_CONTRACT.REQUIRED_LIBRARIES
    ]
    bundle_id = hashlib.sha256(
        RUNTIME_CONTRACT.canonical_json_bytes(inventory)
    ).hexdigest()
    bundle = task_root / "runtime-bundles" / bundle_id
    sidecar_path = (
        task_root / "runtime-bundle-manifests" / f"{bundle_id}.json"
    )
    build_root = (
        task_root
        / "builds"
        / f"{source_sha256}-gint-bounded-workspace-v2-sm80"
    )
    bundle.mkdir(parents=True)
    for name, payload in payloads.items():
        path = bundle / name
        path.write_bytes(payload)
        path.chmod(0o444)
    bundle.chmod(0o555)

    tool_paths = dict(RUNTIME_CONTRACT.CUDA_TOOL_PATHS)
    nvcc = {
        "release": "12.8",
        "output": "Cuda compilation tools, release 12.8, V12.8.61\n",
        "verified": True,
    }
    cuobjdump = {
        "release": "12.8",
        "output": "cuobjdump release 12.8, V12.8.61\n",
        "verified": True,
    }
    preflight = {
        "paths": dict(tool_paths),
        "checks": {
            tool: {"path": path, "regular_file": True, "executable": True}
            for tool, path in tool_paths.items()
        },
        "nvcc_version": dict(nvcc),
        "cuobjdump_version": cuobjdump,
        "verified": True,
    }
    libgint = next(item for item in inventory if item["name"] == "libgint.so")
    abi = {
        "basis_prod_cache_size": RUNTIME_CONTRACT.EXPECTED_BASIS_PROD_SIZE,
        "selected_pair_data_size": RUNTIME_CONTRACT.EXPECTED_SELECTED_PAIR_SIZE,
        "offsets": list(RUNTIME_CONTRACT.EXPECTED_SELECTED_OFFSETS),
        "abi_version": RUNTIME_CONTRACT.ABI_VERSION,
        "workspace_bytes": dict(RUNTIME_CONTRACT.EXPECTED_WORKSPACE_BYTES),
    }
    library_verification = {
        "path": "/private/build/gpu4pyscf/lib/libgint.so",
        "sha256": libgint["sha256"],
        "bytes": libgint["bytes"],
        "symbols": {
            "required": list(RUNTIME_CONTRACT.REQUIRED_SYMBOLS),
            "missing": [],
            "present": list(RUNTIME_CONTRACT.REQUIRED_SYMBOLS),
        },
        "abi": {"expected": abi, "actual": dict(abi), "verified": True},
        "abi_probe": {
            "mode": "subprocess",
            "executable": "/usr/bin/python3",
            "verified": True,
        },
        "cuobjdump_path": tool_paths["cuobjdump"],
        "cubin": {
            "architectures": ["compute_80", "sm_80"],
            "sm_cubin": ["sm_80"],
            "sm80_only": True,
        },
        "stack": {
            "records": {
                "columns_rys7": 3712,
                "columns_rys8": 3728,
                "diagonal_rys7": 1920,
                "diagonal_rys8": 1936,
            },
            "max_bytes": 3728,
            "limit_bytes": 4096,
            "verified": True,
        },
        "ldd": {"missing": [], "returncode": 0, "verified": True},
    }
    source_root = task_root / "build-inputs" / source_sha256 / "source"
    base_bundle = task_root / "runtime-bundles" / "base-bundle"
    build = {
        "type": "RelWithDebInfo",
        "cuda_version": "12.8",
        "cuda_architecture": "80-real",
        "tool_paths": tool_paths,
        "tool_preflight": preflight,
        "nvcc_version": nvcc,
        "verification": library_verification,
    }
    sidecar = {
        "schema": RUNTIME_CONTRACT.BUNDLE_SCHEMA,
        "bundle_id": bundle_id,
        "identity_algorithm": RUNTIME_CONTRACT.IDENTITY_ALGORITHM,
        "inventory_sha256": bundle_id,
        "complete": True,
        "created_utc": "2026-09-14T12:34:56+00:00",
        "bundle_root": str(bundle),
        "manifest_path": str(sidecar_path),
        "source_sha256": source_sha256,
        "source_file_count": source_file_count,
        "source_root": str(source_root),
        "base_bundle": str(base_bundle),
        "build": build,
        "inventory": inventory,
    }
    source_plan = {
        "schema": RUNTIME_CONTRACT.BUILD_SCHEMA,
        "execute": False,
        "source_root": str(source_root),
        "source_sha256": source_sha256,
        "source_file_count": source_file_count,
        "base_bundle": str(base_bundle),
        "base_library_count": 13,
        "output_task_root": str(task_root),
        "build_type": "RelWithDebInfo",
        "cuda_version": "12.8",
        "cuda_architecture": "80-real",
        "required_symbols": list(RUNTIME_CONTRACT.REQUIRED_SYMBOLS),
        "tool_paths": tool_paths,
        "abi_version": RUNTIME_CONTRACT.ABI_VERSION,
        "workspace_bytes": dict(RUNTIME_CONTRACT.EXPECTED_WORKSPACE_BYTES),
    }
    detached = {
        "schema": RUNTIME_CONTRACT.BUILD_SCHEMA,
        "source": source_plan,
        "nvcc": nvcc,
        "tool_paths": tool_paths,
        "tool_preflight": preflight,
        "commands_log_sha256": "b" * 64,
        "libgint": library_verification,
        "inventory": inventory,
        "bundle_id": bundle_id,
        "bundle_path": str(bundle),
        "manifest_path": str(sidecar_path),
        "build_root": str(build_root),
    }
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_bytes(
        RUNTIME_CONTRACT.canonical_json_bytes(sidecar, trailing_newline=True)
    )
    sidecar_path.chmod(0o444)
    build_root.mkdir(parents=True)
    verification_path = build_root / "build-verification.json"
    verification_path.write_bytes(
        RUNTIME_CONTRACT.canonical_json_bytes(detached, trailing_newline=True)
    )
    verification_path.chmod(0o444)
    build_root.chmod(0o555)
    return bundle


def _publish_inputs(
    tmp_path: Path, profile: str = "g0-canonical-pristine"
) -> dict:
    source = tmp_path.resolve() / "publish-source"
    module_root = source / "benchmarks" / "cc" / "a100_water8"
    module_root.mkdir(parents=True)
    for name in (
        "snapshot_symlink_normalize.py",
        "snapshot_manifest.py",
        "runtime_bundle_contract.py",
    ):
        (module_root / name).write_bytes((ROOT / name).read_bytes())
    (source / "case.py").write_text("case = 'water8'\n", encoding="utf-8")
    runtime_package = source / "gpu4pyscf" / "lib"
    runtime_package.mkdir(parents=True)
    (source / "gpu4pyscf" / "__init__.py").write_text("\n", encoding="utf-8")
    (source / "gpu4pyscf" / "lib" / "__init__.py").write_text(
        "\n", encoding="utf-8"
    )
    (runtime_package / "utils.py").write_text(
        "def load_library(name):\n    return name\n", encoding="utf-8"
    )
    (runtime_package / "runtime_targets.py").write_text(
        "from gpu4pyscf.lib.utils import load_library\n"
        + "".join(
            f"load_library({name.removesuffix('.so')!r})\n"
            for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
        ),
        encoding="utf-8",
    )
    (runtime_package / "prebinding-sentinel.so").write_bytes(
        b"must survive candidate validation failures\n"
    )

    inventory = NORMALIZER.inspect_source(source)
    normalization_value = NORMALIZER.materialize_source_links(source, inventory)
    normalization = tmp_path.resolve() / "publish-normalization.json"
    normalization.write_text(
        json.dumps(normalization_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = tmp_path.resolve() / "publish-provenance.json"
    allowed = {
        "prefixes": ["benchmarks/cc/a100_water8/"],
        "files": ["gpu4pyscf/cc/device_runtime.py"],
    }
    policy_bytes = json.dumps(
        {"allowed_untracked": allowed, "status": []},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    canonical_pristine = profile == "g0-canonical-pristine"
    provenance.write_text(
        json.dumps({
            "schema": "gpu4pyscf.local-provenance.v1",
            "deployment_profile": profile,
            "repository_root": "/local/gpu4pyscf",
            "head_revision": SNAPSHOT.FROZEN_BASE_COMMIT,
            "frozen_base_revision": SNAPSHOT.FROZEN_BASE_COMMIT,
            "head_matches_frozen_base": True,
            "tracked_pristine": True,
            "untracked_paths_allowed": True,
            "canonical_pristine": canonical_pristine,
            "allowed_untracked": allowed,
            "git_status": {
                "format": "porcelain-v1-lines",
                "sha256": hashlib.sha256(b"").hexdigest(),
                "allowed_paths_status_sha256": hashlib.sha256(policy_bytes).hexdigest(),
                "entry_count": 0,
                "entries": [],
            },
        }) + "\n",
        encoding="utf-8",
    )
    digest, file_count = SNAPSHOT.source_tree_digest(source)
    if profile == "candidate":
        task_root = tmp_path.resolve() / "private" / "task"
        task_root.mkdir(parents=True, exist_ok=True)
        (tmp_path.resolve() / "private").chmod(0o700)
        runtime = _candidate_runtime_bundle(task_root, digest, file_count)
    else:
        runtime = tmp_path.resolve() / "runtime"
        runtime.mkdir()
        for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES:
            (runtime / name).write_bytes(f"runtime:{name}\n".encode("ascii"))

    payload = RECEIVER.build_bundle(source, provenance, normalization)
    normalization_data = normalization.read_bytes()
    required_b64 = base64.b64encode(
        json.dumps(list(SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES)).encode("utf-8")
    ).decode("ascii")
    runtime_argument = str(runtime)
    if sys.platform == "darwin" and runtime_argument.startswith("/private/var/"):
        # Exercise the exact short spelling commonly returned through TMPDIR.
        runtime_argument = runtime_argument.removeprefix("/private")
    assembly_arguments = [
        digest,
        digest,
        str(file_count),
        hashlib.sha256(normalization_data).hexdigest(),
        str(len(normalization_data)),
        SNAPSHOT.FROZEN_BASE_COMMIT,
        runtime_argument,
        required_b64,
        profile,
    ]
    return {
        "source": source,
        "normalization": normalization,
        "payload": payload,
        "digest": digest,
        "assembly_arguments": assembly_arguments,
        "runtime": runtime,
        "profile": profile,
    }


def _receive_with_assembly(tmp_path: Path, inputs: dict):
    payload = inputs["payload"]
    root, created = _remote_staging(tmp_path, "9")
    arguments = _receive_arguments(root, created, payload)
    try:
        result = RECEIVER.receive_bundle(
            payload,
            destination_root=arguments[0],
            container_name=arguments[1],
            root_identity=(int(arguments[2]), int(arguments[3])),
            container_identity=(int(arguments[4]), int(arguments[5])),
            tree_name=arguments[6],
            tree_identity=(int(arguments[7]), int(arguments[8])),
            expected_sha256=arguments[9],
            expected_bytes=int(arguments[10]),
            post_receive=ASSEMBLY.post_receive,
            post_receive_args=inputs["assembly_arguments"],
        )
        return result, root, created
    except BaseException as exc:
        setattr(exc, "staging_root_for_test", root)
        setattr(exc, "staging_identity_for_test", created)
        raise


@pytest.mark.parametrize("profile", ["g0-canonical-pristine", "candidate"])
def test_embedded_receive_assembly_and_publish_share_fd_lifetime(
    tmp_path: Path, profile: str
):
    inputs = _publish_inputs(tmp_path, profile)
    payload = inputs["payload"]
    digest = inputs["digest"]
    normalization_data = inputs["normalization"].read_bytes()
    program_path = tmp_path.resolve() / "embedded-receiver.py"
    program_path.touch()
    built = subprocess.run(
        [
            sys.executable,
            str(ROOT / "snapshot_bundle_receiver.py"),
            "build-program",
            "--source-root",
            str(inputs["source"]),
            "--provenance",
            str(tmp_path.resolve() / "publish-provenance.json"),
            "--normalization",
            str(inputs["normalization"]),
            "--post-receive-script",
            str(ROOT / "snapshot_remote_assembly.py"),
            "--output",
            str(program_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    build_receipt = json.loads(built.stdout)
    assert build_receipt["post_receive_embedded"] is True
    assert build_receipt["bundle_sha256"] == hashlib.sha256(payload).hexdigest()
    program = program_path.read_bytes()
    root, created = _remote_staging(tmp_path, "e")
    target = root / digest
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-",
                *_receive_arguments(root, created, payload),
                *inputs["assembly_arguments"],
            ],
            input=program,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        receipt = json.loads(completed.stdout)
        assert receipt["post_receive"]["published_snapshot"] == digest
        assert receipt["post_receive"]["publish_protocol"] in {
            "atomic-exclusive-rename",
            "reserved-empty-directory-rename",
            "reserved-empty-directory-rename-commit-confirmed",
        }
        assert receipt["post_receive"]["publish_trust_boundary"] == (
            ASSEMBLY.PUBLISH_TRUST_BOUNDARY
        )
        attestation_name = receipt["post_receive"]["publication_attestation_name"]
        assert attestation_name == f"{digest}.publication-attestation.json"
        attestation_bytes = (root / attestation_name).read_bytes()
        attestation = json.loads(attestation_bytes)
        assert attestation["source_tree_sha256"] == digest
        assert attestation["publish_protocol"] == receipt["post_receive"][
            "publish_protocol"
        ]
        attestation_without_hash = dict(attestation)
        attestation_without_hash.pop("attestation_sha256")
        assert attestation["attestation_sha256"] == (
            ASSEMBLY._canonical_json_sha256(attestation_without_hash)
        )
        assert attestation["attestation_sha256"] == (
            receipt["post_receive"]["publication_attestation_sha256"]
        )
        assert stat.S_IMODE((root / attestation_name).stat().st_mode) == 0o444
        assert not Path(created["staging_path"]).exists()
        assert target.is_dir()
        assert (target / "manifest.json").is_file()
        manifest = json.loads((target / "manifest.json").read_text())
        assert manifest["schema"] == SNAPSHOT.SOURCE_SNAPSHOT_SCHEMA
        assert manifest["tree_sha256"] == digest
        assert manifest["publication_policy"] == {
            "schema": ASSEMBLY.PUBLICATION_POLICY_SCHEMA,
            "trust_boundary": ASSEMBLY.PUBLISH_TRUST_BOUNDARY,
            "attestation_schema": ASSEMBLY.PUBLICATION_ATTESTATION_SCHEMA,
            "attestation_location": (
                "snapshots-root sibling named "
                "<snapshot-name>.publication-attestation.json"
            ),
        }
        assert "publication_attestation" not in manifest
        validated = SNAPSHOT.validate_snapshot(
            target / "source", expected_task_root=root.parent
        )
        assert validated["valid"] is True, validated["reasons"]
        assert manifest["source_normalization"]["uploaded_record_sha256"] == (
            hashlib.sha256(normalization_data).hexdigest()
        )
        assert {
            entry["relative_path"].rsplit("/", 1)[-1]
            for entry in manifest["runtime_binaries"]["files"]
        } == set(SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES)
        if profile == "candidate":
            runtime = manifest["runtime_binaries"]
            runtime_receipt = receipt["post_receive"]
            assert runtime_receipt["runtime_contract_validated"] is True
            assert runtime_receipt["runtime_library_count"] == 13
            assert runtime_receipt["runtime_bundle_id"] == runtime["bundle_id"]
            assert runtime_receipt["runtime_sidecar_sha256"] == runtime[
                "sidecar_sha256"
            ]
            assert runtime_receipt[
                "runtime_build_verification_sha256"
            ] == runtime["build_verification_sha256"]
            assert runtime_receipt[
                "runtime_validation_evidence_sha256"
            ] == runtime["validation_evidence_sha256"]
            assert runtime["validation_evidence"][
                "build_verification_bound"
            ] is True
        else:
            assert receipt["post_receive"]["runtime_contract_validated"] is False
        for directory, names, files in os.walk(target, followlinks=False):
            for name in [*names, *files]:
                info = (Path(directory) / name).lstat()
                assert not stat.S_ISLNK(info.st_mode)
                assert not info.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    finally:
        _make_writable(target)


def _candidate_sidecar_path(inputs: dict) -> Path:
    runtime = inputs["runtime"]
    return (
        runtime.parent.parent
        / "runtime-bundle-manifests"
        / f"{runtime.name}.json"
    )


def _candidate_build_verification_path(inputs: dict) -> Path:
    runtime = inputs["runtime"]
    return (
        runtime.parent.parent
        / "builds"
        / f'{inputs["digest"]}-gint-bounded-workspace-v2-sm80'
        / "build-verification.json"
    )


def test_candidate_missing_sidecar_fails_before_runtime_copy(tmp_path: Path):
    inputs = _publish_inputs(tmp_path, "candidate")
    _candidate_sidecar_path(inputs).unlink()
    with pytest.raises(FileNotFoundError) as caught:
        _receive_with_assembly(tmp_path, inputs)
    created = caught.value.staging_identity_for_test
    sentinel = (
        Path(created["staging_path"])
        / "source/gpu4pyscf/lib/prebinding-sentinel.so"
    )
    assert sentinel.read_bytes() == b"must survive candidate validation failures\n"


def test_candidate_requires_detached_build_verification(tmp_path: Path):
    inputs = _publish_inputs(tmp_path, "candidate")
    verification = _candidate_build_verification_path(inputs)
    verification.parent.chmod(0o755)
    verification.unlink()
    verification.parent.chmod(0o555)
    with pytest.raises(FileNotFoundError):
        _receive_with_assembly(tmp_path, inputs)


def test_candidate_detached_build_verification_symlink_is_rejected(
    tmp_path: Path,
):
    inputs = _publish_inputs(tmp_path, "candidate")
    verification = _candidate_build_verification_path(inputs)
    payload = verification.read_bytes()
    outside = tmp_path / "outside-build-verification.json"
    outside.write_bytes(payload)
    verification.parent.chmod(0o755)
    verification.unlink()
    verification.symlink_to(outside)
    verification.parent.chmod(0o555)
    with pytest.raises(RuntimeError, match="not a regular file"):
        _receive_with_assembly(tmp_path, inputs)


def test_candidate_sidecar_source_identity_tamper_is_rejected(
    tmp_path: Path,
):
    inputs = _publish_inputs(tmp_path, "candidate")
    sidecar = _candidate_sidecar_path(inputs)
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    value["source_sha256"] = "0" * 64
    sidecar.chmod(0o644)
    sidecar.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar.chmod(0o444)
    with pytest.raises(ValueError, match="source SHA-256"):
        _receive_with_assembly(tmp_path, inputs)


def test_candidate_sidecar_symlink_is_rejected(tmp_path: Path):
    inputs = _publish_inputs(tmp_path, "candidate")
    sidecar = _candidate_sidecar_path(inputs)
    payload = sidecar.read_bytes()
    outside = tmp_path / "outside-sidecar.json"
    outside.write_bytes(payload)
    sidecar.unlink()
    sidecar.symlink_to(outside)
    with pytest.raises(RuntimeError, match="not a regular file"):
        _receive_with_assembly(tmp_path, inputs)


@pytest.mark.parametrize("mutation", ["missing", "extra", "writable"])
def test_candidate_bundle_inventory_is_exact_and_immutable(
    tmp_path: Path, mutation: str
):
    inputs = _publish_inputs(tmp_path, "candidate")
    bundle = inputs["runtime"]
    bundle.chmod(0o755)
    if mutation == "missing":
        (bundle / RUNTIME_CONTRACT.REQUIRED_LIBRARIES[-1]).unlink()
    elif mutation == "extra":
        extra = bundle / "extra.so"
        extra.write_bytes(b"extra\n")
        extra.chmod(0o444)
    else:
        target = bundle / RUNTIME_CONTRACT.REQUIRED_LIBRARIES[0]
        target.chmod(0o644)
    bundle.chmod(0o555)
    with pytest.raises(RuntimeError, match="exact required|writable"):
        _receive_with_assembly(tmp_path, inputs)


def test_candidate_bundle_from_another_task_root_is_rejected(tmp_path: Path):
    inputs = _publish_inputs(tmp_path, "candidate")
    outside = tmp_path.resolve() / "other-private" / "task" / "runtime-bundles"
    outside.parent.parent.mkdir(parents=True)
    outside.parent.parent.chmod(0o700)
    outside.mkdir(parents=True)
    copied = outside / inputs["runtime"].name
    shutil.copytree(inputs["runtime"], copied)
    copied.chmod(0o555)
    inputs["assembly_arguments"][-3] = str(copied)
    with pytest.raises(RuntimeError, match="snapshot task root"):
        _receive_with_assembly(tmp_path, inputs)


def test_assembly_ancestor_swap_cannot_mutate_replacement_tree(
    tmp_path: Path,
):
    inputs = _publish_inputs(tmp_path)
    payload = inputs["payload"]
    root, created = _remote_staging(tmp_path, "f")
    private = tmp_path.resolve() / "private"
    original_private = tmp_path.resolve() / "private-original"
    outside = tmp_path.resolve() / "outside"
    mirror_source = (
        outside / "task" / "snapshots" / created["staging_name"] / "source"
    )
    mirror_source.parent.mkdir(parents=True)
    shutil.copytree(inputs["source"], mirror_source)
    victim = mirror_source / "gpu4pyscf" / "lib" / "victim.so"
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_bytes(b"outside must not change\n")
    victim_mode = stat.S_IMODE(victim.stat().st_mode)

    def swap_then_assemble(context, arguments):
        private.rename(original_private)
        private.symlink_to(outside, target_is_directory=True)
        return ASSEMBLY.post_receive(context, arguments)

    receive_arguments = _receive_arguments(root, created, payload)
    try:
        with pytest.raises(RuntimeError, match="component is a symbolic link"):
            RECEIVER.receive_bundle(
                payload,
                destination_root=receive_arguments[0],
                container_name=receive_arguments[1],
                root_identity=(int(receive_arguments[2]), int(receive_arguments[3])),
                container_identity=(
                    int(receive_arguments[4]), int(receive_arguments[5])
                ),
                tree_name=receive_arguments[6],
                tree_identity=(int(receive_arguments[7]), int(receive_arguments[8])),
                expected_sha256=receive_arguments[9],
                expected_bytes=int(receive_arguments[10]),
                post_receive=swap_then_assemble,
                post_receive_args=inputs["assembly_arguments"],
            )
        assert victim.read_bytes() == b"outside must not change\n"
        assert stat.S_IMODE(victim.stat().st_mode) == victim_mode
        assert not (mirror_source.parent / "manifest.json").exists()
        assert sorted(path.name for path in victim.parent.glob("*.so")) == [
            "prebinding-sentinel.so", "victim.so"
        ]
    finally:
        if private.is_symlink():
            private.unlink()
            original_private.rename(private)
        staging = root / created["staging_name"]
        _make_writable(staging)


def test_exclusive_rename_einval_selects_nfs_fallback(monkeypatch, tmp_path: Path):
    root, root_fd, source_fd, _ = _publish_fixture(tmp_path)

    class Primitive:
        argtypes = None
        restype = None

        def __call__(self, *arguments):
            del arguments
            ASSEMBLY.ctypes.set_errno(errno.EINVAL)
            return -1

    class LibC:
        renameat2 = Primitive()

    monkeypatch.setattr(ASSEMBLY.sys, "platform", "linux")
    monkeypatch.setattr(ASSEMBLY.ctypes, "CDLL", lambda *args, **kwargs: LibC())
    try:
        supported = ASSEMBLY._exclusive_rename_noreplace(
            root_fd, "staging", "published"
        )
        assert supported is False
        assert (root / "staging").is_dir()
        assert not (root / "published").exists()
    finally:
        os.close(source_fd)
        os.close(root_fd)


def test_nfs_reservation_fallback_publishes_original_staging_inode(tmp_path: Path):
    root, root_fd, source_fd, source_identity = _publish_fixture(tmp_path)
    try:
        protocol = ASSEMBLY._publish_directory_noreplace(
            root_fd,
            source_fd,
            "staging",
            "published",
            source_identity,
            exclusive_rename=lambda *arguments: False,
        )

        assert protocol == "reserved-empty-directory-rename"
        assert not (root / "staging").exists()
        assert (root / "published" / "payload.txt").read_text() == (
            "staged payload\n"
        )
        published = (root / "published").stat()
        assert (published.st_dev, published.st_ino) == source_identity
        assert sorted(path.name for path in root.iterdir()) == ["published"]
    finally:
        os.close(source_fd)
        os.close(root_fd)


def test_nfs_reservation_never_overwrites_preexisting_target(tmp_path: Path):
    root, root_fd, source_fd, source_identity = _publish_fixture(tmp_path)
    target = root / "published"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("must survive\n", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="target appeared"):
            ASSEMBLY._publish_directory_noreplace(
                root_fd,
                source_fd,
                "staging",
                "published",
                source_identity,
                exclusive_rename=lambda *arguments: False,
            )
        assert sentinel.read_text(encoding="utf-8") == "must survive\n"
        assert (root / "staging" / "payload.txt").is_file()
    finally:
        os.close(source_fd)
        os.close(root_fd)


def test_nfs_reservation_replacement_is_not_overwritten(tmp_path: Path):
    root, root_fd, source_fd, source_identity = _publish_fixture(tmp_path)

    def replace_reservation(parent_fd, name, reservation_fd, identity):
        del reservation_fd, identity
        os.rename(name, "moved-reservation", src_dir_fd=parent_fd,
                  dst_dir_fd=parent_fd)
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        replacement_root_fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_fd,
        )
        try:
            replacement_fd = os.open(
                "replacement.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=replacement_root_fd,
            )
            try:
                os.write(replacement_fd, b"must survive\n")
            finally:
                os.close(replacement_fd)
        finally:
            os.close(replacement_root_fd)

    try:
        with pytest.raises(RuntimeError, match="reservation was replaced"):
            ASSEMBLY._publish_directory_noreplace(
                root_fd,
                source_fd,
                "staging",
                "published",
                source_identity,
                exclusive_rename=lambda *arguments: False,
                reservation_hook=replace_reservation,
            )
        assert (root / "published" / "replacement.txt").read_bytes() == (
            b"must survive\n"
        )
        assert (root / "staging" / "payload.txt").is_file()
        assert (root / "moved-reservation").is_dir()
    finally:
        os.close(source_fd)
        os.close(root_fd)


def test_nfs_nonempty_reservation_is_preserved_without_publishing(tmp_path: Path):
    root, root_fd, source_fd, source_identity = _publish_fixture(tmp_path)

    def fill_reservation(parent_fd, name, reservation_fd, identity):
        del parent_fd, name, identity
        descriptor = os.open(
            "unexpected.txt",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=reservation_fd,
        )
        try:
            os.write(descriptor, b"unexpected\n")
        finally:
            os.close(descriptor)

    try:
        with pytest.raises(RuntimeError, match="reservation is not empty"):
            ASSEMBLY._publish_directory_noreplace(
                root_fd,
                source_fd,
                "staging",
                "published",
                source_identity,
                exclusive_rename=lambda *arguments: False,
                reservation_hook=fill_reservation,
            )
        assert (root / "published" / "unexpected.txt").read_bytes() == (
            b"unexpected\n"
        )
        assert (root / "staging" / "payload.txt").is_file()
    finally:
        os.close(source_fd)
        os.close(root_fd)


def test_nfs_ambiguous_rename_error_recognizes_committed_inode(tmp_path: Path):
    root, root_fd, source_fd, source_identity = _publish_fixture(tmp_path)

    def commit_then_lose_reply(
        source_name, destination_name, *, src_dir_fd, dst_dir_fd,
    ):
        os.rename(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        raise OSError(errno.EIO, "simulated lost NFS rename reply")

    try:
        protocol = ASSEMBLY._publish_directory_noreplace(
            root_fd,
            source_fd,
            "staging",
            "published",
            source_identity,
            exclusive_rename=lambda *arguments: False,
            fallback_rename=commit_then_lose_reply,
        )

        assert protocol == (
            "reserved-empty-directory-rename-commit-confirmed"
        )
        assert not (root / "staging").exists()
        published = (root / "published").stat()
        assert (published.st_dev, published.st_ino) == source_identity
        assert (root / "published" / "payload.txt").read_text() == (
            "staged payload\n"
        )
    finally:
        os.close(source_fd)
        os.close(root_fd)


@pytest.mark.parametrize("native", [True, False], ids=["native", "fallback"])
def test_rename_syscall_error_reconciles_committed_held_inode(
    tmp_path: Path, native: bool,
):
    root, root_fd, source_fd, source_identity = _publish_fixture(tmp_path)

    def native_commit_then_lose_reply(parent_fd, source_name, destination_name):
        os.rename(
            source_name, destination_name,
            src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
        )
        raise OSError(errno.EIO, "simulated lost rename reply")

    def fallback_commit_then_lose_reply(
        source_name, destination_name, *, src_dir_fd, dst_dir_fd,
    ):
        os.rename(
            source_name, destination_name,
            src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
        )
        raise OSError(errno.EIO, "simulated lost rename reply")

    try:
        protocol = ASSEMBLY._publish_directory_noreplace(
            root_fd,
            source_fd,
            "staging",
            "published",
            source_identity,
            exclusive_rename=(
                native_commit_then_lose_reply if native else lambda *args: False
            ),
            fallback_rename=fallback_commit_then_lose_reply,
        )
        assert protocol == (
            "atomic-exclusive-rename-commit-confirmed"
            if native else "reserved-empty-directory-rename-commit-confirmed"
        )
        assert not (root / "staging").exists()
        assert (root / "published").stat().st_ino == source_identity[1]
    finally:
        os.close(source_fd)
        os.close(root_fd)


@pytest.mark.parametrize("native", [True, False], ids=["native", "fallback"])
def test_rename_error_reconciliation_stat_failure_is_indeterminate(
    tmp_path: Path, native: bool,
):
    root, root_fd, source_fd, source_identity = _publish_fixture(tmp_path)

    def reconciliation_stat(*arguments, **keywords):
        del arguments, keywords
        raise OSError(errno.EIO, "simulated reconciliation stat failure")

    def native_commit_then_lose_reply(parent_fd, source_name, destination_name):
        os.rename(
            source_name, destination_name,
            src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
        )
        raise OSError(errno.EIO, "simulated lost rename reply")

    def fallback_commit_then_lose_reply(
        source_name, destination_name, *, src_dir_fd, dst_dir_fd,
    ):
        os.rename(
            source_name, destination_name,
            src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
        )
        raise OSError(errno.EIO, "simulated lost rename reply")

    try:
        with pytest.raises(
            ASSEMBLY.PublicationIndeterminateError,
            match="publication outcome is indeterminate",
        ) as raised:
            ASSEMBLY._publish_directory_noreplace(
                root_fd,
                source_fd,
                "staging",
                "published",
                source_identity,
                exclusive_rename=(
                    native_commit_then_lose_reply
                    if native else lambda *args: False
                ),
                fallback_rename=fallback_commit_then_lose_reply,
                reconciliation_stat=reconciliation_stat,
            )
        assert raised.value.state == "indeterminate"
        assert not (root / "staging").exists()
        published = root / "published"
        assert published.is_dir()
        assert (published.stat().st_dev, published.stat().st_ino) == source_identity
    finally:
        os.close(source_fd)
        os.close(root_fd)


def test_nfs_uncommitted_rename_error_removes_only_empty_reservation(
    tmp_path: Path,
):
    root, root_fd, source_fd, source_identity = _publish_fixture(tmp_path)

    def fail_without_commit(*arguments, **keywords):
        del arguments, keywords
        raise OSError(errno.EIO, "simulated uncommitted NFS rename")

    try:
        with pytest.raises(OSError, match="uncommitted NFS rename"):
            ASSEMBLY._publish_directory_noreplace(
                root_fd,
                source_fd,
                "staging",
                "published",
                source_identity,
                exclusive_rename=lambda *arguments: False,
                fallback_rename=fail_without_commit,
            )
        assert not (root / "published").exists()
        assert (root / "staging" / "payload.txt").is_file()
    finally:
        os.close(source_fd)
        os.close(root_fd)


def test_nfs_fallback_ancestor_swap_never_writes_outside(tmp_path: Path):
    private = tmp_path.resolve() / "private"
    root = private / "task" / "snapshots"
    source = root / "staging"
    source.mkdir(parents=True)
    private.chmod(0o700)
    (source / "payload.txt").write_text("inside\n", encoding="utf-8")
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    source_fd = os.open(
        "staging",
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        dir_fd=root_fd,
    )
    source_info = os.fstat(source_fd)
    source_identity = (source_info.st_dev, source_info.st_ino)
    original = tmp_path.resolve() / "private-original"
    outside = tmp_path.resolve() / "outside"
    outside_target = outside / "task" / "snapshots" / "published"
    outside_target.mkdir(parents=True)
    sentinel = outside_target / "sentinel.txt"
    sentinel.write_text("outside must survive\n", encoding="utf-8")

    def swap_ancestor(parent_fd, name, reservation_fd, identity):
        del parent_fd, name, reservation_fd, identity
        private.rename(original)
        private.symlink_to(outside, target_is_directory=True)

    try:
        protocol = ASSEMBLY._publish_directory_noreplace(
            root_fd,
            source_fd,
            "staging",
            "published",
            source_identity,
            exclusive_rename=lambda *arguments: False,
            reservation_hook=swap_ancestor,
        )
        assert protocol == "reserved-empty-directory-rename"
        assert sentinel.read_text(encoding="utf-8") == "outside must survive\n"
        published = original / "task" / "snapshots" / "published"
        assert (published / "payload.txt").read_text() == "inside\n"
        assert (published.stat().st_dev, published.stat().st_ino) == source_identity
    finally:
        os.close(source_fd)
        os.close(root_fd)
        if private.is_symlink():
            private.unlink()
            original.rename(private)


@pytest.mark.parametrize(
    ("name", "kind", "message"),
    [
        ("/absolute.txt", "regular", "unsafe"),
        ("tree/../escape.txt", "regular", "unsafe"),
        ("tree/link", "symlink", "not regular/dir"),
        ("tree/fifo", "fifo", "not regular/dir"),
    ],
)
def test_unsafe_archive_members_fail_before_writes(
    tmp_path: Path, name: str, kind: str, message: str,
):
    member = tarfile.TarInfo(name)
    data: bytes | None = b"malicious\n"
    if kind == "regular":
        member.type = tarfile.REGTYPE
        member.size = len(data)
    elif kind == "symlink":
        member.type = tarfile.SYMTYPE
        member.linkname = "/tmp/outside"
        member.size = 0
        data = None
    else:
        member.type = tarfile.FIFOTYPE
        member.size = 0
        data = None
    payload = _tar_payload([(member, data)])
    root, created = _remote_staging(tmp_path, "b")

    with pytest.raises(ValueError, match=message):
        _receive_direct(root, created, payload)

    staging = Path(created["staging_path"])
    assert list((staging / "source").iterdir()) == []
    assert sorted(path.name for path in staging.iterdir()) == ["source"]


def test_duplicate_archive_member_is_rejected(tmp_path: Path):
    first = tarfile.TarInfo("tree/value.txt")
    first.type = tarfile.REGTYPE
    first.size = 3
    second = tarfile.TarInfo("tree/value.txt")
    second.type = tarfile.REGTYPE
    second.size = 3
    payload = _tar_payload([(first, b"one"), (second, b"two")])
    root, created = _remote_staging(tmp_path, "c")
    with pytest.raises(ValueError, match="duplicate bundle member"):
        _receive_direct(root, created, payload)
    assert list((Path(created["staging_path"]) / "source").iterdir()) == []


def test_anchor_swap_cannot_redirect_receiver_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source, provenance, normalization = _source_inputs(tmp_path)
    payload = RECEIVER.build_bundle(source, provenance, normalization)
    root, created = _remote_staging(tmp_path, "d")
    private = tmp_path.resolve() / "private"
    original_private = tmp_path.resolve() / "private-original"
    outside = tmp_path.resolve() / "outside"
    mirror_source = (
        outside / "task" / "snapshots" / created["staging_name"] / "source"
    )
    mirror_source.mkdir(parents=True)
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("must survive\n", encoding="utf-8")
    original_create = RECEIVER._create_regular
    swapped = False

    def swapping_create(base_fd, parts, member, archive):
        nonlocal swapped
        if member.name.startswith("tree/") and not swapped:
            swapped = True
            private.rename(original_private)
            private.symlink_to(outside, target_is_directory=True)
        return original_create(base_fd, parts, member, archive)

    monkeypatch.setattr(RECEIVER, "_create_regular", swapping_create)
    with pytest.raises(RuntimeError, match="unsafe|symbolic link"):
        _receive_direct(root, created, payload)

    assert swapped is True
    assert sentinel.read_text(encoding="utf-8") == "must survive\n"
    assert list(mirror_source.iterdir()) == []
    anchored_source = (
        original_private
        / "task"
        / "snapshots"
        / created["staging_name"]
        / "source"
    )
    assert (anchored_source / "root.txt").read_bytes() == b"root payload\n"


def test_build_bundle_requires_canonical_source_root(tmp_path: Path):
    source, provenance, normalization = _source_inputs(tmp_path)
    alias = tmp_path.resolve() / "source-alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical directory"):
        RECEIVER.build_bundle(alias, provenance, normalization)


def test_darwin_system_alias_mapping_is_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(RECEIVER.sys, "platform", "darwin")
    assert RECEIVER._lexical_absolute("/var/folders/example") == Path(
        "/private/var/folders/example"
    )
    assert RECEIVER._lexical_absolute("/tmp/example") == Path(
        "/private/tmp/example"
    )
    assert RECEIVER._lexical_absolute("/users/custom-link/example") == Path(
        "/users/custom-link/example"
    )
