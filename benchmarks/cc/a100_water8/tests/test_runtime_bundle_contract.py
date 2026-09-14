from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONTRACT = _load(
    "water8_runtime_bundle_contract",
    ROOT / "runtime_bundle_contract.py",
)
BUILD = _load(
    "water8_runtime_bundle_builder_for_contract_test",
    ROOT / "build_gint_sm80_bundle.py",
)

SOURCE_SHA256 = "a" * 64
SOURCE_FILE_COUNT = 971
TASK_ROOT = "/mnt/mridata/private/task"


def _inventory() -> list[dict]:
    return [
        {
            "name": name,
            "bytes": index + 10,
            "sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
        }
        for index, name in enumerate(CONTRACT.REQUIRED_LIBRARIES)
    ]


def _version(tool: str) -> dict:
    return {
        "release": "12.8",
        "output": (
            f"{tool}: NVIDIA CUDA tool\n"
            "Cuda compilation tools, release 12.8, V12.8.61\n"
        ),
        "verified": True,
    }


def _verification(inventory: list[dict]) -> dict:
    libgint = next(entry for entry in inventory if entry["name"] == "libgint.so")
    abi = {
        "basis_prod_cache_size": 144,
        "selected_pair_data_size": 112,
        "offsets": list(CONTRACT.EXPECTED_SELECTED_OFFSETS),
        "abi_version": 2,
        "workspace_bytes": dict(CONTRACT.EXPECTED_WORKSPACE_BYTES),
    }
    return {
        "path": "/mnt/mridata/private/build/source/gpu4pyscf/lib/libgint.so",
        "sha256": libgint["sha256"],
        "bytes": libgint["bytes"],
        "symbols": {
            "required": list(CONTRACT.REQUIRED_SYMBOLS),
            "missing": [],
            "present": list(CONTRACT.REQUIRED_SYMBOLS),
        },
        "abi": {"expected": abi, "actual": deepcopy(abi), "verified": True},
        "abi_probe": {
            "mode": "subprocess",
            "executable": "/usr/bin/python3",
            "verified": True,
        },
        "cuobjdump_path": "/usr/local/cuda-12.8/bin/cuobjdump",
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


def _sidecar() -> dict:
    inventory = _inventory()
    bundle_id = hashlib.sha256(
        CONTRACT.canonical_json_bytes(inventory, trailing_newline=False)
    ).hexdigest()
    bundle_root = f"{TASK_ROOT}/runtime-bundles/{bundle_id}"
    manifest_path = f"{TASK_ROOT}/runtime-bundle-manifests/{bundle_id}.json"
    tool_paths = dict(CONTRACT.CUDA_TOOL_PATHS)
    nvcc = _version("nvcc")
    cuobjdump = _version("cuobjdump")
    preflight = {
        "paths": deepcopy(tool_paths),
        "checks": {
            tool: {
                "path": path,
                "regular_file": True,
                "executable": True,
            }
            for tool, path in tool_paths.items()
        },
        "nvcc_version": deepcopy(nvcc),
        "cuobjdump_version": cuobjdump,
        "verified": True,
    }
    return {
        "schema": "gpu4pyscf.runtime-bundle.v2",
        "bundle_id": bundle_id,
        "identity_algorithm": (
            "sha256(canonical-json-inventory-list-no-newline)"
        ),
        "inventory_sha256": bundle_id,
        "complete": True,
        "created_utc": "2026-09-14T12:34:56.123456+00:00",
        "bundle_root": bundle_root,
        "manifest_path": manifest_path,
        "source_sha256": SOURCE_SHA256,
        "source_file_count": SOURCE_FILE_COUNT,
        "source_root": f"{TASK_ROOT}/build-inputs/{SOURCE_SHA256}/source",
        "base_bundle": f"{TASK_ROOT}/runtime-bundles/base-bundle",
        "build": {
            "type": "RelWithDebInfo",
            "cuda_version": "12.8",
            "cuda_architecture": "80-real",
            "tool_paths": tool_paths,
            "tool_preflight": preflight,
            "nvcc_version": nvcc,
            "verification": _verification(inventory),
        },
        "inventory": inventory,
    }


def _build_verification(sidecar: dict) -> dict:
    build_root = (
        f"{TASK_ROOT}/builds/{SOURCE_SHA256}-gint-bounded-workspace-v2-sm80"
    )
    return {
        "schema": "gpu4pyscf.gint-sm80-build.v1",
        "source": {
            "schema": "gpu4pyscf.gint-sm80-build.v1",
            "execute": False,
            "source_root": sidecar["source_root"],
            "source_sha256": SOURCE_SHA256,
            "source_file_count": SOURCE_FILE_COUNT,
            "base_bundle": sidecar["base_bundle"],
            "base_library_count": 13,
            "output_task_root": TASK_ROOT,
            "build_type": "RelWithDebInfo",
            "cuda_version": "12.8",
            "cuda_architecture": "80-real",
            "required_symbols": list(CONTRACT.REQUIRED_SYMBOLS),
            "tool_paths": dict(CONTRACT.CUDA_TOOL_PATHS),
            "abi_version": 2,
            "workspace_bytes": dict(CONTRACT.EXPECTED_WORKSPACE_BYTES),
        },
        "nvcc": deepcopy(sidecar["build"]["nvcc_version"]),
        "tool_paths": deepcopy(sidecar["build"]["tool_paths"]),
        "tool_preflight": deepcopy(sidecar["build"]["tool_preflight"]),
        "commands_log_sha256": "b" * 64,
        "libgint": deepcopy(sidecar["build"]["verification"]),
        "inventory": deepcopy(sidecar["inventory"]),
        "bundle_id": sidecar["bundle_id"],
        "bundle_path": sidecar["bundle_root"],
        "manifest_path": sidecar["manifest_path"],
        "build_root": build_root,
    }


def _payload(sidecar: dict) -> bytes:
    return CONTRACT.canonical_json_bytes(sidecar, trailing_newline=True)


def _validate(
    sidecar: dict,
    *,
    observed_inventory=None,
    payload: bytes | None = None,
    build_verification: dict | None = None,
    build_verification_payload: bytes | None = None,
    public_build_verification_path: str | None = None,
):
    return CONTRACT.validate_runtime_bundle_sidecar(
        sidecar,
        deepcopy(sidecar["inventory"])
        if observed_inventory is None
        else observed_inventory,
        sidecar_payload=_payload(sidecar) if payload is None else payload,
        public_bundle_root=sidecar["bundle_root"],
        public_sidecar_path=sidecar["manifest_path"],
        expected_source_sha256=SOURCE_SHA256,
        expected_source_file_count=SOURCE_FILE_COUNT,
        build_verification=build_verification,
        build_verification_payload=build_verification_payload,
        public_build_verification_path=(
            public_build_verification_path
            if public_build_verification_path is not None
            else (
                f'{build_verification["build_root"]}/build-verification.json'
                if build_verification is not None
                and build_verification_payload is not None
                else None
            )
        ),
    )


def test_builder_compatible_sidecar_validates_and_returns_compact_hashes():
    sidecar = _sidecar()
    payload = _payload(sidecar)
    evidence = _validate(sidecar, payload=payload)

    assert tuple(BUILD.REQUIRED_LIBRARIES) == CONTRACT.REQUIRED_LIBRARIES
    assert tuple(BUILD.REQUIRED_SYMBOLS) == CONTRACT.REQUIRED_SYMBOLS
    assert BUILD.make_bundle_id(sidecar["inventory"]) == sidecar["bundle_id"]
    assert evidence == {
        "schema": "gpu4pyscf.runtime-bundle-validation.v1",
        "validated": True,
        "bundle_id": sidecar["bundle_id"],
        "bundle_root": sidecar["bundle_root"],
        "manifest_path": sidecar["manifest_path"],
        "library_count": 13,
        "inventory_sha256": sidecar["bundle_id"],
        "inventory_payload_sha256": sidecar["bundle_id"],
        "source_sha256": SOURCE_SHA256,
        "source_file_count": SOURCE_FILE_COUNT,
        "cuda_version": "12.8",
        "cuda_architecture": "80-real",
        "abi_version": 2,
        "maximum_stack_bytes": 3728,
        "build_verification_bound": False,
        "sidecar_sha256": hashlib.sha256(payload).hexdigest(),
        "sidecar_bytes": len(payload),
        "payload_sha256": hashlib.sha256(
            CONTRACT.canonical_json_bytes(sidecar)
        ).hexdigest(),
    }


def test_optional_builder_verification_is_source_and_bundle_bound():
    sidecar = _sidecar()
    verification = _build_verification(sidecar)
    verification_payload = CONTRACT.canonical_json_bytes(
        verification, trailing_newline=True
    )
    evidence = _validate(
        sidecar,
        build_verification=verification,
        build_verification_payload=verification_payload,
    )
    assert evidence["build_verification_bound"] is True
    assert evidence["build_verification_sha256"] == hashlib.sha256(
        verification_payload
    ).hexdigest()
    assert evidence["build_verification_bytes"] == len(verification_payload)
    assert evidence["build_verification_path"] == (
        f'{verification["build_root"]}/build-verification.json'
    )


def test_execute_build_generated_sidecar_and_detached_evidence_validate(
    tmp_path: Path, monkeypatch,
):
    """Exercise the builder's real v2 manifest/evidence assembly code."""

    source = tmp_path / "source"
    source_library = source / "gpu4pyscf" / "lib"
    source_library.mkdir(parents=True)
    (source_library / "CMakeLists.txt").write_text(
        "add_library(gint SHARED dummy.cu)\n", encoding="utf-8"
    )
    source_sha256, source_count = BUILD.source_tree_digest(source)
    base = tmp_path / "base"
    base.mkdir()
    for name in BUILD.REQUIRED_LIBRARIES:
        (base / name).write_bytes(f"base:{name}\n".encode("ascii"))
    task = tmp_path / "task"
    configured_source: dict[str, Path] = {}
    built_bytes = b"builder-produced-libgint\0"

    def fake_run(command, *, cwd=None, log=None):
        del cwd
        if log is not None:
            log.extend(("$ " + " ".join(command), "simulated\n"))
        if "-S" in command:
            configured_source["path"] = Path(command[command.index("-S") + 1])
        if "--target" in command:
            (configured_source["path"] / "libgint.so").write_bytes(built_bytes)
        return subprocess.CompletedProcess(command, 0, "simulated\n")

    template = _sidecar()

    def fake_preflight(*, log=None):
        if log is not None:
            log.append("simulated CUDA tool preflight")
        return deepcopy(template["build"]["tool_preflight"])

    def fake_verify(path, *, log=None):
        del log
        inventory = _inventory()
        libgint = next(
            entry for entry in inventory if entry["name"] == "libgint.so"
        )
        libgint["bytes"] = len(built_bytes)
        libgint["sha256"] = hashlib.sha256(built_bytes).hexdigest()
        result = _verification(inventory)
        result["path"] = str(path)
        return result

    captured: dict[str, bytes] = {}

    def fake_publish(
        bundle_staging,
        manifest_bytes,
        verification_bytes,
        log_bytes,
        output_task_root,
        bundle_id,
        declared_source_sha256,
    ):
        del bundle_staging, log_bytes
        captured["sidecar"] = manifest_bytes
        captured["verification"] = verification_bytes
        return (
            output_task_root / "runtime-bundles" / bundle_id,
            output_task_root / "runtime-bundle-manifests" / f"{bundle_id}.json",
            output_task_root
            / "builds"
            / f"{declared_source_sha256}-gint-bounded-workspace-v2-sm80",
        )

    monkeypatch.setattr(BUILD, "_run", fake_run)
    monkeypatch.setattr(BUILD, "_preflight_tools", fake_preflight)
    monkeypatch.setattr(BUILD, "verify_library", fake_verify)
    monkeypatch.setattr(BUILD, "_publish_transaction", fake_publish)

    returned_verification = BUILD.execute_build(
        source,
        source_sha256,
        source_count,
        base,
        task,
        build_jobs=1,
    )
    sidecar = CONTRACT.parse_runtime_bundle_sidecar(captured["sidecar"])
    detached = json.loads(captured["verification"])
    assert detached == returned_verification

    evidence = CONTRACT.validate_runtime_bundle_sidecar(
        sidecar,
        sidecar["inventory"],
        sidecar_payload=captured["sidecar"],
        public_bundle_root=sidecar["bundle_root"],
        public_sidecar_path=sidecar["manifest_path"],
        expected_source_sha256=source_sha256,
        expected_source_file_count=source_count,
        build_verification=detached,
        build_verification_payload=captured["verification"],
        public_build_verification_path=(
            f'{detached["build_root"]}/build-verification.json'
        ),
    )
    assert evidence["validated"] is True
    assert evidence["build_verification_bound"] is True


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(schema="gpu4pyscf.runtime-bundle.v1"), "schema"),
        (lambda value: value.update(complete=False), "completeness"),
        (lambda value: value.update(extra=True), "keys mismatch"),
        (lambda value: value.pop("created_utc"), "keys mismatch"),
        (lambda value: value.update(identity_algorithm="sha256(files)"), "algorithm"),
        (lambda value: value.update(created_utc="yesterday"), "created_utc"),
    ],
)
def test_top_level_contract_fails_closed(mutation, match):
    sidecar = _sidecar()
    mutation(sidecar)
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match=match):
        _validate(sidecar)


@pytest.mark.parametrize(
    "payload,match",
    [
        (b'{"schema":"a","schema":"b"}', "duplicate"),
        (b'{"nested":{"x":1,"x":2}}', "duplicate"),
        (b'{"x":NaN}', "non-integer"),
        (b'{"x":Infinity}', "non-integer"),
        (b'{"x":1.5}', "non-integer"),
        (b"[]", "root"),
        (b"\xff", "UTF-8"),
    ],
)
def test_parser_rejects_ambiguous_or_noncanonical_json(payload, match):
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match=match):
        CONTRACT.parse_runtime_bundle_sidecar(payload)


def test_validator_rejects_mapping_subclasses_and_unbound_payload():
    class Sidecar(dict):
        pass

    sidecar = _sidecar()
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="plain object"):
        CONTRACT.validate_runtime_bundle_sidecar(
            Sidecar(sidecar),
            sidecar["inventory"],
            sidecar_payload=_payload(sidecar),
            public_bundle_root=sidecar["bundle_root"],
            public_sidecar_path=sidecar["manifest_path"],
            expected_source_sha256=SOURCE_SHA256,
            expected_source_file_count=SOURCE_FILE_COUNT,
        )

    different = deepcopy(sidecar)
    different["created_utc"] = "2026-09-14T12:34:57+00:00"
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="does not match"):
        _validate(sidecar, payload=_payload(different))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda inv: inv.pop(), "exactly 13"),
        (lambda inv: inv.reverse(), "name mismatch"),
        (lambda inv: inv[0].update(extra="forbidden"), "keys mismatch"),
        (lambda inv: inv[0].update(name="libother.so"), "name mismatch"),
        (lambda inv: inv[0].update(bytes=0), "positive integer"),
        (lambda inv: inv[0].update(bytes=True), "positive integer"),
        (lambda inv: inv[0].update(sha256="A" * 64), "lowercase"),
    ],
)
def test_inventory_order_shape_names_sizes_and_hashes_are_exact(mutation, match):
    sidecar = _sidecar()
    mutation(sidecar["inventory"])
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match=match):
        _validate(sidecar)


def test_observed_inventory_must_be_plain_ordered_and_byte_exact():
    sidecar = _sidecar()
    observed = deepcopy(sidecar["inventory"])
    observed[4]["bytes"] += 1
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="sidecar/observed"):
        _validate(sidecar, observed_inventory=observed)
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="must be a list"):
        _validate(sidecar, observed_inventory=tuple(sidecar["inventory"]))


@pytest.mark.parametrize(
    "field,match",
    [
        ("bundle_id", "do not agree"),
        ("inventory_sha256", "inventory digest"),
    ],
)
def test_content_address_is_canonical_inventory_without_newline(field, match):
    sidecar = _sidecar()
    sidecar[field] = "0" * 64
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match=match):
        _validate(sidecar)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("bundle_root", "/mnt/task/runtime-bundles/wrong", "do not agree"),
        ("manifest_path", "/mnt/task/manifests/wrong.json", "do not agree"),
        ("source_sha256", "b" * 64, "source SHA-256"),
        ("source_file_count", SOURCE_FILE_COUNT + 1, "source file count"),
        ("source_root", "/mnt/task/../source", "canonical POSIX"),
    ],
)
def test_public_layout_and_source_identity_are_bound(field, value, match):
    sidecar = _sidecar()
    sidecar[field] = value
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match=match):
        _validate(sidecar)


def test_caller_observed_public_paths_must_match_sidecar_paths():
    sidecar = _sidecar()
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="root binding"):
        CONTRACT.validate_runtime_bundle_sidecar(
            sidecar,
            sidecar["inventory"],
            sidecar_payload=_payload(sidecar),
            public_bundle_root=f"{TASK_ROOT}/runtime-bundles/{'0' * 64}",
            public_sidecar_path=sidecar["manifest_path"],
            expected_source_sha256=SOURCE_SHA256,
            expected_source_file_count=SOURCE_FILE_COUNT,
        )


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("cuda_version",), "12.7", "CUDA version"),
        (("cuda_architecture",), "90-real", "CUDA architecture"),
        (("tool_paths", "nvcc"), "/usr/local/cuda/bin/nvcc", "nvcc"),
        (("tool_preflight", "verified"), False, "preflight verification"),
        (("nvcc_version", "release"), "12.7", "nvcc release"),
        (("nvcc_version", "output"), "release 12.7", "does not prove"),
    ],
)
def test_cuda_12_8_and_pinned_tool_evidence_are_mandatory(path, value, match):
    sidecar = _sidecar()
    cursor = sidecar["build"]
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = value
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match=match):
        _validate(sidecar)


@pytest.mark.parametrize(
    "field,value",
    [
        ("basis_prod_cache_size", 143),
        ("selected_pair_data_size", 111),
        ("abi_version", 1),
        ("offsets", [0]),
        ("workspace_bytes", {"6": 0, "7": 1, "8": 2}),
    ],
)
def test_abi_sizes_offsets_version_and_workspaces_are_exact(field, value):
    sidecar = _sidecar()
    sidecar["build"]["verification"]["abi"]["actual"][field] = value
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="ABI actual"):
        _validate(sidecar)


def test_required_symbol_set_is_exact():
    sidecar = _sidecar()
    symbols = sidecar["build"]["verification"]["symbols"]
    symbols["present"] = symbols["present"][:-1]
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="present"):
        _validate(sidecar)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda verification: verification["cubin"].update(
                architectures=["sm_80", "sm_90"]
            ),
            "architectures",
        ),
        (
            lambda verification: verification["ldd"].update(
                missing=["libcudart.so"]
            ),
            "missing dependencies",
        ),
        (
            lambda verification: verification["stack"]["records"].update(
                columns_rys8=4097
            ),
            "exceeds 4096",
        ),
        (
            lambda verification: verification["stack"].update(max_bytes=1),
            "stack maximum",
        ),
    ],
)
def test_cubin_ldd_and_stack_evidence_fail_closed(mutation, match):
    sidecar = _sidecar()
    mutation(sidecar["build"]["verification"])
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match=match):
        _validate(sidecar)


@pytest.mark.parametrize("field", ["bytes", "sha256"])
def test_built_libgint_identity_must_match_observed_bundle(field):
    sidecar = _sidecar()
    verification = sidecar["build"]["verification"]
    verification[field] = (
        verification[field] + 1 if field == "bytes" else "f" * 64
    )
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="libgint"):
        _validate(sidecar)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["source"].update(source_sha256="c" * 64),
            "build source SHA-256",
        ),
        (
            lambda value: value.update(bundle_path="/mnt/elsewhere/bundle"),
            "verified bundle path",
        ),
        (
            lambda value: value["source"].update(workspace_bytes={"6": 1}),
            "workspaces",
        ),
        (
            lambda value: value["inventory"][0].update(bytes=999),
            "verified inventory",
        ),
        (
            lambda value: value.update(build_root="/mnt/other/builds/unknown"),
            "build root does not agree",
        ),
    ],
)
def test_optional_build_verification_cannot_break_source_binding(mutation, match):
    sidecar = _sidecar()
    verification = _build_verification(sidecar)
    mutation(verification)
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match=match):
        _validate(
            sidecar,
            build_verification=verification,
            build_verification_payload=CONTRACT.canonical_json_bytes(
                verification, trailing_newline=True
            ),
        )


def test_detached_build_verification_requires_parsed_and_raw_evidence_together():
    sidecar = _sidecar()
    verification = _build_verification(sidecar)
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="together"):
        _validate(sidecar, build_verification=verification)
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="together"):
        _validate(
            sidecar,
            build_verification_payload=CONTRACT.canonical_json_bytes(
                verification, trailing_newline=True
            ),
        )


def test_detached_build_verification_public_path_is_exactly_bound():
    sidecar = _sidecar()
    verification = _build_verification(sidecar)
    payload = CONTRACT.canonical_json_bytes(
        verification, trailing_newline=True
    )
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="path binding"):
        _validate(
            sidecar,
            build_verification=verification,
            build_verification_payload=payload,
            public_build_verification_path=(
                f"{TASK_ROOT}/builds/other/build-verification.json"
            ),
        )


def test_detached_build_verification_payload_is_duplicate_safe_and_bound():
    sidecar = _sidecar()
    verification = _build_verification(sidecar)
    different = deepcopy(verification)
    different["commands_log_sha256"] = "d" * 64
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="does not match"):
        _validate(
            sidecar,
            build_verification=verification,
            build_verification_payload=CONTRACT.canonical_json_bytes(different),
        )
    malformed_payloads = (
        (b'{"schema":"one","schema":"two"}', "duplicate"),
        (b'{"value":NaN}', "non-integer"),
        (b'{"value":1.5}', "non-integer"),
    )
    for malformed, match in malformed_payloads:
        with pytest.raises(CONTRACT.RuntimeBundleContractError, match=match):
            _validate(
                sidecar,
                build_verification=verification,
                build_verification_payload=malformed,
            )


def test_non_json_values_in_preparsed_input_are_rejected():
    sidecar = _sidecar()
    sidecar["unexpected_float"] = float("nan")
    with pytest.raises(CONTRACT.RuntimeBundleContractError, match="forbidden"):
        CONTRACT.validate_runtime_bundle_sidecar(
            sidecar,
            sidecar["inventory"],
            sidecar_payload=json.dumps(sidecar).encode("utf-8"),
            public_bundle_root=sidecar["bundle_root"],
            public_sidecar_path=sidecar["manifest_path"],
            expected_source_sha256=SOURCE_SHA256,
            expected_source_file_count=SOURCE_FILE_COUNT,
        )
