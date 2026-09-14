from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_gint_sm80_bundle", ROOT / "build_gint_sm80_bundle.py"
)
BUILD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILD)


def test_required_symbols_are_exactly_checked():
    text = "\n".join(f"000000 T {symbol}" for symbol in BUILD.REQUIRED_SYMBOLS)
    result = BUILD.validate_required_symbols(text)
    assert result["missing"] == []
    assert result["present"] == list(BUILD.REQUIRED_SYMBOLS)


def test_missing_symbol_fails_closed():
    text = "000000 T GINTselected_workspace_size\n"
    with pytest.raises(BUILD.BuildError, match="missing"):
        BUILD.validate_required_symbols(text)


def test_abi_contract_and_workspace_values():
    values = {
        "basis_prod_cache_size": 144,
        "selected_pair_data_size": 112,
        "offsets": list(BUILD.EXPECTED_SELECTED_OFFSETS),
        "abi_version": 2,
        "workspace_bytes": dict(BUILD.EXPECTED_WORKSPACE_BYTES),
    }
    assert BUILD.validate_abi_values(values)["verified"] is True


def test_abi_contract_rejects_workspace_drift():
    values = {
        "basis_prod_cache_size": 144,
        "selected_pair_data_size": 112,
        "offsets": list(BUILD.EXPECTED_SELECTED_OFFSETS),
        "abi_version": 2,
        "workspace_bytes": {**BUILD.EXPECTED_WORKSPACE_BYTES, 7: 1},
    }
    with pytest.raises(BUILD.BuildError, match="ABI mismatch"):
        BUILD.validate_abi_values(values)


def test_sm80_cubin_rejects_other_sm_targets():
    assert BUILD.validate_sm80_cubin("arch = sm_80\n")['sm80_only'] is True
    assert BUILD.validate_sm80_cubin(
        "ELF file    1: libgint.1.sm_80.cubin\n"
    )["sm80_only"] is True
    with pytest.raises(BUILD.BuildError, match="sm_80-only"):
        BUILD.validate_sm80_cubin("arch = sm_80\narch = sm_90\n")


def test_ldd_rejects_missing_dependency():
    with pytest.raises(BUILD.BuildError, match="missing"):
        BUILD.validate_ldd("libfoo.so => not found\n", 0)
    assert BUILD.validate_ldd("libc.so.6 => /lib/libc.so.6\n", 0)["verified"]


def test_stack_parser_requires_all_high_rys_kernels_and_limit():
    text = "\n".join([
        "columns root 7 stack frame: 3712 bytes",
        "columns root 8 stack frame: 3728 bytes",
        "diagonal root 7 stack frame: 1920 bytes",
        "diagonal root 8 stack frame: 1936 bytes",
    ])
    result = BUILD.parse_stack_usage(text)
    assert result["max_bytes"] == 3728
    assert result["records"]["columns_rys8"] == 3728


def test_stack_parser_accepts_raw_cuobjdump_function_names():
    text = "\n".join([
        "Function _Z24selected_diagonal_kernelILi8ELi45750EEv",
        " REG:130 STACK:1936 SHARED:0",
        "Function _Z24selected_diagonal_kernelILi7ELi21720EEv",
        " REG:130 STACK:1920 SHARED:0",
        "Function _Z30selected_columns_kernel_cutoffILi8ELi45750EEv",
        " REG:130 STACK:3728 SHARED:0",
        "Function _Z30selected_columns_kernel_cutoffILi7ELi21720EEv",
        " REG:130 STACK:3712 SHARED:0",
    ])
    assert BUILD.parse_stack_usage(text)["max_bytes"] == 3728


def test_stack_parser_rejects_missing_or_oversized_kernel():
    with pytest.raises(BUILD.BuildError, match="lacks required"):
        BUILD.parse_stack_usage("columns root 7 stack frame: 1 bytes\n")
    text = "\n".join([
        "columns root 7 stack frame: 4097 bytes",
        "columns root 8 stack frame: 1 bytes",
        "diagonal root 7 stack frame: 1 bytes",
        "diagonal root 8 stack frame: 1 bytes",
    ])
    with pytest.raises(BUILD.BuildError, match="exceeds limit"):
        BUILD.parse_stack_usage(text)


def test_source_tree_digest_rejects_symlink(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    target = source / "target.py"
    target.write_text("x = 1\n", encoding="utf-8")
    (source / "link.py").symlink_to(target.name)
    with pytest.raises(BUILD.BuildError, match="non-regular"):
        BUILD.source_tree_digest(source)


def test_content_addressed_inventory_is_stable(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in BUILD.REQUIRED_LIBRARIES:
        (bundle / name).write_bytes(name.encode("ascii"))
    inventory = BUILD.library_inventory(bundle)
    assert len(inventory) == 13
    assert BUILD.make_bundle_id(inventory) == BUILD.make_bundle_id(inventory)


def test_established_bundle_id_fixture_and_layout(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in BUILD.REQUIRED_LIBRARIES:
        (bundle / name).write_bytes(name.encode("ascii"))
    inventory = BUILD.library_inventory(bundle)
    # Contract is canonical JSON of the inventory list without a newline.
    expected = BUILD.sha256_bytes(
        BUILD.canonical_json_bytes(inventory, trailing_newline=False)
    )
    assert BUILD.make_bundle_id(inventory) == expected
    assert len(expected) == 64
    assert BUILD.make_bundle_id(inventory) != BUILD.sha256_bytes(
        BUILD.canonical_json_bytes(inventory)
    )

    # Fixed regression fixture: any accidental newline, dict wrapper, or
    # inventory ordering change must alter this established algorithm.
    fixture = [
        {"bytes": 1, "name": "a.so", "sha256": "0" * 64},
        {"bytes": 2, "name": "b.so", "sha256": "1" * 64},
    ]
    assert BUILD.make_bundle_id(fixture) == (
        "1af790e41c6c37a54d9d5b036490d835fc498f2725162292a44d08d57a1abee5"
    )


def test_nvcc_version_is_strict():
    assert BUILD.validate_nvcc_version(
        "Cuda compilation tools, release 12.8, V12.8.89\n"
    )["verified"] is True
    with pytest.raises(BUILD.BuildError, match="CUDA 12.8"):
        BUILD.validate_nvcc_version("Cuda compilation tools, release 12.7, V12.7.1\n")


def test_cuobjdump_version_is_strict():
    banner = "Cuda compilation tools, release 12.8, V12.8.55\n"
    assert BUILD.validate_cuobjdump_version(banner)["verified"] is True
    for release in ("12.7", "13.1"):
        with pytest.raises(BUILD.BuildError, match="cuobjdump is not CUDA 12.8"):
            BUILD.validate_cuobjdump_version(
                f"Cuda compilation tools, release {release}, V{release}.1\n"
            )


def test_plan_records_pinned_cuda_tool_paths(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("x = 1\n", encoding="utf-8")
    source_sha, source_count = BUILD.source_tree_digest(source)
    base = tmp_path / "base"
    base.mkdir()
    for name in BUILD.REQUIRED_LIBRARIES:
        (base / name).write_bytes(name.encode("ascii"))

    plan = BUILD.plan(source, source_sha, source_count, base, tmp_path / "task")

    assert plan["tool_paths"] == {
        "nvcc": "/usr/local/cuda-12.8/bin/nvcc",
        "cuobjdump": "/usr/local/cuda-12.8/bin/cuobjdump",
    }


def test_tool_preflight_checks_pinned_paths_before_build(monkeypatch):
    calls = []

    def fake_run(command, *, cwd=None, log=None):
        calls.append(command)
        output = "Cuda compilation tools, release 12.8, V12.8.55\n"
        return subprocess.CompletedProcess(command, 0,
                                           output)

    monkeypatch.setattr(BUILD, "regular_file", lambda path, label: path)
    monkeypatch.setattr(BUILD.os, "access", lambda path, mode: True)
    monkeypatch.setattr(BUILD, "_run", fake_run)

    evidence = BUILD._preflight_tools()

    assert evidence["paths"] == BUILD.CUDA_TOOL_PATHS
    assert evidence["verified"] is True
    assert evidence["cuobjdump_version"]["release"] == "12.8"
    assert calls == [
        [str(BUILD.NVCC_PATH), "--version"],
        [str(BUILD.CUOBJDUMP_PATH), "--version"],
    ]


def test_verify_source_has_no_bare_cuobjdump_command():
    source = (ROOT / "build_gint_sm80_bundle.py").read_text(encoding="utf-8")
    assert re.search(r"_run\(\[\s*[\"']cuobjdump", source) is None
    assert '"--list",' not in source
    assert '"--list-elf"' in source


def test_abi_probe_isolated_in_child_process_and_verify_uses_pinned_cuobjdump(
    tmp_path: Path, monkeypatch
):
    libgint = tmp_path / "libgint.so"
    libgint.write_bytes(b"test library")
    abi_values = {
        "basis_prod_cache_size": BUILD.EXPECTED_BASIS_PROD_SIZE,
        "selected_pair_data_size": BUILD.EXPECTED_SELECTED_PAIR_SIZE,
        "offsets": list(BUILD.EXPECTED_SELECTED_OFFSETS),
        "abi_version": BUILD.ABI_VERSION,
        "workspace_bytes": dict(BUILD.EXPECTED_WORKSPACE_BYTES),
    }
    # The subprocess emits JSON, so integer workspace keys become strings on
    # the parent-side round trip.  Keep this contract explicit in the test.
    abi_json = json.loads(json.dumps(abi_values))
    assert set(abi_json["workspace_bytes"]) == {"6", "7", "8"}
    nm_output = "\n".join(f"000000 T {symbol}" for symbol in BUILD.REQUIRED_SYMBOLS)
    stack_output = "\n".join([
        "columns root 7 stack frame: 3712 bytes",
        "columns root 8 stack frame: 3728 bytes",
        "diagonal root 7 stack frame: 1920 bytes",
        "diagonal root 8 stack frame: 1936 bytes",
    ])
    calls = []

    def fake_run(command, *, cwd=None, log=None):
        calls.append(command)
        if command[0] == sys.executable:
            stdout = json.dumps(abi_json)
        elif command[0] == "nm":
            stdout = nm_output
        elif command[0] == str(BUILD.CUOBJDUMP_PATH) and "--list-elf" in command:
            stdout = "arch = sm_80\n"
        elif command[0] == str(BUILD.CUOBJDUMP_PATH):
            stdout = stack_output
        elif command[0] == "ldd":
            stdout = "libc.so.6 => /lib/libc.so.6\n"
        else:
            raise AssertionError(f"unexpected command: {command!r}")
        return subprocess.CompletedProcess(command, 0, stdout)

    monkeypatch.setattr(BUILD, "_run", fake_run)
    result = BUILD.verify_library(libgint)

    assert result["abi_probe"] == {
        "mode": "subprocess", "executable": sys.executable, "verified": True,
    }
    assert result["abi"]["actual"]["workspace_bytes"] == abi_json["workspace_bytes"]
    assert result["cuobjdump_path"] == str(BUILD.CUOBJDUMP_PATH)
    assert calls[1][0] == sys.executable
    assert calls[1][1] == "-c"
    assert [command[0] for command in calls if "cuobjdump" in command[0]] == [
        str(BUILD.CUOBJDUMP_PATH), str(BUILD.CUOBJDUMP_PATH),
    ]


def test_publish_layout_is_content_addressed_and_sidecared(tmp_path: Path):
    staged = tmp_path / "staged"
    staged.mkdir()
    for name in BUILD.REQUIRED_LIBRARIES:
        (staged / name).write_bytes(name.encode("ascii"))
    inventory = BUILD.library_inventory(staged)
    bundle_id = BUILD.make_bundle_id(inventory)
    result = BUILD._publish_transaction(
        staged, b'{"manifest":true}\n', b'{"verification":true}\n', b"build\n",
        tmp_path / "task", bundle_id, "a" * 64,
    )
    bundle, sidecar, build = result
    assert bundle == tmp_path / "task" / "runtime-bundles" / bundle_id
    assert sidecar == tmp_path / "task" / "runtime-bundle-manifests" / f"{bundle_id}.json"
    assert build == tmp_path / "task" / "builds" / f"{'a' * 64}-gint-bounded-workspace-v2-sm80"
    assert sorted(path.name for path in bundle.iterdir()) == sorted(BUILD.REQUIRED_LIBRARIES)
    assert sidecar.is_file()
    assert (build / "build-verification.json").is_file()
    assert not (bundle / "bundle-manifest.json").exists()


def test_publish_failure_rolls_back_all_outputs(tmp_path: Path, monkeypatch):
    staged = tmp_path / "staged"
    staged.mkdir()
    for name in BUILD.REQUIRED_LIBRARIES:
        (staged / name).write_bytes(name.encode("ascii"))
    bundle_id = BUILD.make_bundle_id(BUILD.library_inventory(staged))
    real_rename = BUILD.os.rename
    calls = {"count": 0}

    def fail_second(source, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected publish failure")
        return real_rename(source, target)

    monkeypatch.setattr(BUILD.os, "rename", fail_second)
    with pytest.raises(OSError, match="injected"):
        BUILD._publish_transaction(
            staged, b"{}\n", b"{}\n", b"log\n", tmp_path / "task",
            bundle_id, "b" * 64,
        )
    task = tmp_path / "task"
    assert not (task / "runtime-bundles" / bundle_id).exists()
    assert not (task / "runtime-bundle-manifests" / f"{bundle_id}.json").exists()
    assert not (task / "builds" / f"{'b' * 64}-gint-bounded-workspace-v2-sm80").exists()
    assert not (task / ".gint-sm80-publish.lock").exists()
    assert not list(task.glob(".gint-sm80-publish-*"))


def test_source_and_base_mutation_is_rejected(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("x = 1\n", encoding="utf-8")
    expected_source = BUILD.source_tree_digest(source)
    (source / "module.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(BUILD.BuildError, match="source changed"):
        BUILD._assert_source_unchanged(source, *expected_source)

    base = tmp_path / "base"
    base.mkdir()
    for name in BUILD.REQUIRED_LIBRARIES:
        (base / name).write_bytes(name.encode("ascii"))
    expected_base = BUILD.validate_base_bundle(base)
    (base / "libgint.so").write_bytes(b"mutated")
    with pytest.raises(BUILD.BuildError, match="base bundle changed"):
        BUILD._assert_base_unchanged(base, expected_base)
