"""Contract tests for the freeze-only MTU build-input staging step."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "stage_mtu_build_input.sh"


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _shell_function(name: str, *, next_name: str) -> str:
    script = _script()
    start = script.index(f"{name}() {{")
    end = script.index(f"\n\n{next_name}() {{", start)
    return script[start:end]


def _function_python(name: str) -> str:
    script = _script()
    start = script.index(f"{name}() {{")
    code_start = script.index("<<'PY'\n", start) + len("<<'PY'\n")
    code_end = script.index("\nPY\n}", code_start)
    return script[code_start:code_end]


def test_stage_script_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_safe_ssh_host_regex_is_bash_32_compatible() -> None:
    regex = "^[A-Za-z0-9_.:@%+-]+$"
    assert regex in _script()
    deploy = (ROOT / "deploy_mtu_snapshot.sh").read_text(encoding="utf-8")
    assert regex in deploy
    for token in ("mtu", "user@mtu", "host-name.example:22"):
        completed = subprocess.run(
            [
                "bash",
                "-c",
                '[[ "$1" =~ ^[A-Za-z0-9_.:@%+-]+$ ]]',
                "host-regex-test",
                token,
            ],
        )
        assert completed.returncode == 0
    for token in ("-oProxyCommand=bad", "bad host", "bad;host", "[::1]"):
        completed = subprocess.run(
            [
                "bash",
                "-c",
                '[[ "$1" =~ ^[A-Za-z0-9_.:@%+-]+$ ]]',
                "host-regex-test",
                token,
            ],
        )
        assert completed.returncode != 0


def test_guard_digest_supports_a_local_python_path_with_spaces(
    tmp_path: Path,
) -> None:
    wrapper_dir = tmp_path / "python wrapper"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "python executable"
    wrapper.write_text(
        f"#!/bin/sh\nexec {sys.executable!r} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    guard = tmp_path / "guard.py"
    guard.write_bytes(b"guard payload\n")
    expected = hashlib.sha256(guard.read_bytes()).hexdigest()
    assignment = next(
        line for line in _script().splitlines()
        if line.startswith("GUARD_SHA256=")
    )
    completed = subprocess.run(
        [
            "bash", "-c",
            "\n".join((
                'set -euo pipefail',
                'PYTHON_BIN="$1"',
                'REMOTE_STAGING_GUARD="$2"',
                assignment,
                'printf "%s\\n" "$GUARD_SHA256"',
            )),
            "guard-digest-test", str(wrapper), str(guard),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == expected
    assert '$("${PYTHON_BIN}" -c' in assignment


def test_two_independent_normalized_copies_are_required() -> None:
    script = _script()
    assert script.count("materialize_local_source") >= 3
    assert "LOCAL_STAGING_ROOT" in script
    assert "LOCAL_CHECK_ROOT" in script
    assert 'cmp -s "${PROVENANCE_FILE}" "${PROVENANCE_CHECK_FILE}"' in script
    assert 'cmp -s "${NORMALIZATION_FILE}" "${NORMALIZATION_CHECK_FILE}"' in script
    assert '[[ "${TREE_IDENTITY_CHECK_JSON}" == "${TREE_IDENTITY_JSON}" ]]' in script
    assert "snapshot_symlink_normalize.py" in script
    assert "module.inspect_source(source)" in script
    assert "module.materialize_source_links(source, inventory)" in script
    assert "module.assert_normalized_source(source)" in script
    assert script.index("PROGRAM_INFO=") < script.index(
        'write_local_provenance "${STATUS_CHECK_FILE}"'
    )


def test_copy_exclusions_match_snapshot_deploy_contract() -> None:
    script = _script()
    for exclusion in (
        "--exclude='.git'",
        "--exclude='.pytest_cache/'",
        "--exclude='__pycache__/'",
        "--exclude='build/'",
        "--exclude='dist/'",
        "--exclude='results/'",
        "--exclude='*.pyc'",
    ):
        assert exclusion in script


def test_receiver_has_no_publish_callback_or_runtime_mutation() -> None:
    script = _script()
    assert "snapshot_bundle_receiver.py" in script
    assert "build-program" in script
    assert "--post-receive-script" not in script
    assert "snapshot_remote_assembly.py" not in script
    assert "REMOTE_GPU4PYSCF" not in script
    assert "REMOTE_SNAPSHOTS_ROOT" not in script
    assert '"preserved": True' in script
    assert "validate_bundle_program_info" in script
    assert 'value["post_receive_embedded"] is not False' in script


def test_identity_bound_guard_covers_allocate_verify_and_failure_cleanup() -> None:
    script = _script()
    assert 'remote_guard create "${MTU_BUILD_INPUTS_ROOT}"' in script
    assert script.count('remote_guard verify-staging "${MTU_BUILD_INPUTS_ROOT}"') >= 2
    assert 'remote_guard remove "${MTU_BUILD_INPUTS_ROOT}" "${CONTAINER_NAME}"' in script
    assert '"${SUCCESS}" != "1"' in script
    assert '"${ROOT_DEV}" "${ROOT_INO}" "${CONTAINER_DEV}" "${CONTAINER_INO}"' in script
    assert '"guarded_cleanup_arguments"' in script
    assert '"remove", *sys.argv[1:]' in script
    assert '"guarded_cleanup"' in script
    assert '"guard_sha256"' in script
    assert '"argv": [sys.argv[18], "-", *json.loads(sys.argv[15])]' in script
    assert '>${ALLOCATED_RECEIPT_FILE}' not in script
    assert '>"${ALLOCATED_RECEIPT_FILE}"' in script
    assert "validate_guard_create_receipt" in script
    assert "transport-failed" in script


def test_remote_argv_round_trip_does_not_execute_shell_payload(
    tmp_path: Path,
) -> None:
    function = _shell_function("remote_argv_command", next_name="remote_ssh")
    marker = tmp_path / "remote-argv-was-executed"
    payload = f"single'quote; $(touch {marker})\nsecond line"
    wrapper = "\n".join((
        "set -euo pipefail",
        'PYTHON_BIN="$1"',
        "shift",
        function,
        'remote_argv_command "$@"',
    ))
    encoded = subprocess.run(
        [
            "bash", "-c", wrapper, "stage-test", sys.executable,
            "/usr/bin/printf", "%s", payload,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip("\n")
    completed = subprocess.run(
        ["/bin/sh", "-c", encoded], check=True, capture_output=True, text=True
    )
    assert completed.stdout == payload
    assert not marker.exists()
    assert "shlex.join(sys.argv[1:])" in function
    assert 'ssh "${SSH_OPTIONS[@]}" -- "${MTU_HOST_NAME}" "${remote_command}"' in _script()


def test_unsafe_ssh_host_fails_before_any_ssh_side_effect(tmp_path: Path) -> None:
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
    environment["MTU_HOST_NAME"] = "-oProxyCommand=touch-injected"
    completed = subprocess.run(
        ["bash", str(SCRIPT)], env=environment, capture_output=True, text=True
    )
    assert completed.returncode == 2
    assert "safe SSH host token" in completed.stderr
    assert not marker.exists()
    assert _script().index('validate_ssh_host "${MTU_HOST_NAME}"') < _script().index(
        "start_remote_session"
    )


def test_program_receipt_requires_literal_false_post_receive() -> None:
    validator = _function_python("validate_bundle_program_info")
    valid = {
        "schema": "gpu4pyscf.regular-tree-bundle.v1",
        "bundle_sha256": "a" * 64,
        "bundle_bytes": 101,
        "program_bytes": 303,
        "post_receive_embedded": False,
    }
    completed = subprocess.run(
        [sys.executable, "-", json.dumps(valid, separators=(",", ":"))],
        input=validator,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == f'{"a" * 64}\t101'

    valid["post_receive_embedded"] = True
    rejected = subprocess.run(
        [sys.executable, "-", json.dumps(valid, separators=(",", ":"))],
        input=validator,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "must not embed post-receive code" in rejected.stderr


def _guard_receipt(root: str, tree_sha256: str) -> dict[str, object]:
    name = f'.staging-{tree_sha256[:12]}-{"b" * 24}'
    return {
        "schema": "gpu4pyscf.remote-snapshot-staging-guard.v1",
        "snapshots_root": root,
        "snapshots_dev": 11,
        "snapshots_ino": 12,
        "staging_name": name,
        "staging_path": f"{root}/{name}",
        "staging_dev": 13,
        "staging_ino": 14,
        "source_dev": 15,
        "source_ino": 16,
    }


def _run_guard_receipt_validator(
    tmp_path: Path, raw: bytes, *, mode: str = "validate"
) -> subprocess.CompletedProcess[str]:
    receipt_path = tmp_path / "guard-create.json"
    receipt_path.write_bytes(raw)
    root = "/task/build-inputs"
    tree_sha256 = "a" * 64
    return subprocess.run(
        [
            sys.executable, "-", str(receipt_path), root, tree_sha256, mode,
        ],
        input=_function_python("validate_guard_create_receipt"),
        capture_output=True,
        text=True,
    )


def test_guard_create_receipt_is_fully_validated_before_state_assignment(
    tmp_path: Path,
) -> None:
    root = "/task/build-inputs"
    tree_sha256 = "a" * 64
    receipt = _guard_receipt(root, tree_sha256)
    completed = _run_guard_receipt_validator(
        tmp_path, json.dumps(receipt, separators=(",", ":")).encode()
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().split("\t") == [
        "11", "12", receipt["staging_name"], "13", "14", "15", "16",
    ]

    script = _script()
    validation = script.index('if ! ALLOCATION_FIELDS="$(validate_guard_create_receipt')
    activation = script.index("read -r ROOT_DEV ROOT_INO CONTAINER_NAME")
    verify = script.index('remote_guard verify-staging "${MTU_BUILD_INPUTS_ROOT}"')
    assert validation < activation < verify
    assert 'json_field "${ALLOCATED}"' not in script


def test_bad_guard_receipt_emits_bounded_raw_and_cleanup_evidence(
    tmp_path: Path,
) -> None:
    root = "/task/build-inputs"
    tree_sha256 = "a" * 64
    receipt = _guard_receipt(root, tree_sha256)
    raw = (
        json.dumps(receipt, separators=(",", ":"))[:-1]
        + ',"unexpected":true}'
    ).encode()
    completed = _run_guard_receipt_validator(tmp_path, raw)
    assert completed.returncode != 0
    match = re.search(r"recovery evidence=(\{.*\})", completed.stderr)
    assert match is not None, completed.stderr
    evidence = json.loads(match.group(1))
    assert evidence["schema"] == "gpu4pyscf.guard-create-recovery.v1"
    assert evidence["receipt_bytes_observed"] == len(raw)
    assert evidence["receipt_sha256_observed"] == hashlib.sha256(raw).hexdigest()
    assert evidence["raw_receipt_truncated"] is False
    assert evidence["cleanup_identity"]["argv"] == [
        "remove", root, receipt["staging_name"], "11", "12", "13", "14",
    ]


def test_oversized_guard_receipt_hashes_full_input_but_bounds_raw_evidence(
    tmp_path: Path,
) -> None:
    raw = b"x" * 20000
    completed = _run_guard_receipt_validator(tmp_path, raw)
    assert completed.returncode != 0
    match = re.search(r"recovery evidence=(\{.*\})", completed.stderr)
    assert match is not None, completed.stderr
    evidence = json.loads(match.group(1))
    assert evidence["receipt_bytes_observed"] == len(raw)
    assert evidence["receipt_sha256_observed"] == hashlib.sha256(raw).hexdigest()
    assert evidence["receipt_oversized"] is True
    assert evidence["raw_receipt_truncated"] is True
    assert len(evidence["raw_receipt_base64"]) < len(raw)


def test_transport_failure_never_activates_cleanup_state_but_reports_identity(
    tmp_path: Path,
) -> None:
    root = "/task/build-inputs"
    tree_sha256 = "a" * 64
    receipt = _guard_receipt(root, tree_sha256)
    raw = json.dumps(receipt, separators=(",", ":")).encode()
    completed = _run_guard_receipt_validator(
        tmp_path, raw, mode="transport-failed"
    )
    assert completed.returncode != 0
    assert "possible allocation" in completed.stderr
    assert receipt["staging_name"] in completed.stderr
    script = _script()
    failed_branch = script[script.index("if ! remote_guard create"):]
    assert "transport-failed" in failed_branch
    assert failed_branch.index("transport-failed") < failed_branch.index(
        'read -r ROOT_DEV ROOT_INO CONTAINER_NAME'
    )


def test_machine_readable_output_declares_required_fields() -> None:
    script = _script()
    fields = (
        '"source_sha256"',
        '"source_file_count"',
        '"remote_source_root"',
        '"normalization_identity"',
        '"provenance_identity"',
        '"bundle_receipt"',
        '"root_identity"',
        '"container_identity"',
        '"source_identity"',
        '"guarded_cleanup_arguments"',
    )
    for field in fields:
        assert field in script


def test_json_field_shape_documented_by_script_is_serializable() -> None:
    value = {
        "schema": "gpu4pyscf.mtu-build-input.v1",
        "source_sha256": "a" * 64,
        "source_file_count": 1,
        "remote_source_root": "/task/build-inputs/.staging/source",
        "normalization_identity": {"sha256": "b" * 64, "bytes": 10},
        "provenance_identity": {"sha256": "c" * 64, "bytes": 10},
        "bundle_receipt": {},
        "root_identity": {"dev": 1, "ino": 2},
        "container_identity": {"dev": 1, "ino": 3},
        "source_identity": {"dev": 1, "ino": 4},
        "guarded_cleanup_arguments": ["remove"],
    }
    assert json.loads(json.dumps(value))["schema"] == (
        "gpu4pyscf.mtu-build-input.v1"
    )


def test_embedded_remote_digest_matches_snapshot_manifest(tmp_path: Path) -> None:
    script = _script()
    function_start = script.index("verify_remote_source() {")
    code_start = script.index("<<'PY'\n", function_start) + len("<<'PY'\n")
    code_end = script.index("\nPY\n}", code_start)
    verifier = script[code_start:code_end]

    source = tmp_path / "build-inputs" / ".staging-test" / "source"
    (source / "nested").mkdir(parents=True)
    (source / "alpha.py").write_text("alpha = 1\n", encoding="utf-8")
    (source / "nested" / "beta.json").write_text("{}\n", encoding="utf-8")
    (source / "ignored.so").write_bytes(b"runtime\n")
    (source / "results").mkdir()
    (source / "results" / "ignored.py").write_text("ignored\n", encoding="utf-8")
    for directory in (
        source.parent.parent,
        source.parent,
        source,
        source / "nested",
        source / "results",
    ):
        os.chmod(directory, 0o700)

    manifest_spec = importlib.util.spec_from_file_location(
        "stage_snapshot_manifest_for_test", ROOT / "snapshot_manifest.py"
    )
    assert manifest_spec is not None and manifest_spec.loader is not None
    manifest = importlib.util.module_from_spec(manifest_spec)
    manifest_spec.loader.exec_module(manifest)
    expected, expected_count = manifest.source_tree_digest(source)
    root = source.parent.parent
    container = source.parent.name
    root_info = root.stat()
    container_info = source.parent.stat()
    source_info = source.stat()
    args = [
        str(root), container,
        str(root_info.st_dev), str(root_info.st_ino),
        str(container_info.st_dev), str(container_info.st_ino),
        str(source_info.st_dev), str(source_info.st_ino), expected,
    ]
    completed = subprocess.run(
        [sys.executable, "-"] + args,
        input=verifier.encode("utf-8"),
        check=True,
        capture_output=True,
    )
    observed = json.loads(completed.stdout)
    assert observed["source_sha256"] == expected
    assert observed["source_file_count"] == expected_count
    assert observed["regular_file_count"] == 4
    assert "digest.update(b\"\\0\")" in verifier
