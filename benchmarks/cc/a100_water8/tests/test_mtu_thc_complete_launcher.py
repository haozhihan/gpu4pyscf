"""Static fail-closed contract for the MTU complete-THC WATER2 launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_mtu_thc_complete_water2_audit.sbatch"


def _text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_has_valid_bash_and_explicit_single_gpu_allocation() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = _text()

    for directive in (
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gres=gpu:nvidia:1",
        "#SBATCH --gres-flags=enforce-binding",
        "#SBATCH --cpus-per-task=64",
        "#SBATCH --hint=nomultithread",
        "#SBATCH --mem-bind=verbose,map_mem:3",
    ):
        assert directive in text
    assert "#SBATCH --nodelist=" not in text


def test_launcher_requires_release_b_candidate_before_gpu_payload() -> None:
    text = _text()
    payload = text.rindex('"${SCRIPT_DIR}/thc_complete_water2_audit.py"')

    required_before_payload = (
        '${CCSD_SOURCE_ROOT:?set CCSD_SOURCE_ROOT to immutable release snapshot B}',
        '"${TASK_ROOT}"/snapshots/*/source',
        "SOURCE_TREE_SHA256",
        "gint_release_pin.json",
        '"${DEPLOYMENT_PROFILE}" != "candidate"',
        "snapshot_manifest.py",
        "--expected-deployment-profile candidate",
        'manifest.get("gint_release_lineage")',
        'manifest.get("immutable") is not True',
        'deployment_profile") != "candidate"',
    )
    for marker in required_before_payload:
        assert marker in text
        assert text.index(marker) < payload


def test_launcher_binds_exact_external_receipt_pin_tree_and_bundle() -> None:
    text = _text()

    for marker in (
        '${GINT_RUNTIME_GATE_RECEIPT:?set GINT_RUNTIME_GATE_RECEIPT',
        'GINT_RUNTIME_GATE_RECEIPT must be below ${TASK_ROOT}/results',
        "runtime gate receipt must be read-only",
        "build_release_pin(receipt_path)",
        "release snapshot B pin does not bind the supplied receipt",
        'lineage.get("receipt_sha256")',
        'lineage.get("receipt_payload_sha256")',
        'lineage.get("release_pin_sha256")',
        "manifest_module.source_tree_digest(source)",
        'runtime.get("bundle_id")',
        '"receipt_source_tree_bundle"',
        '"runtime_bundle_id": bundle_id',
        '"qualification_node": pinned_node',
        'os.getenv("SLURM_JOB_NODELIST") != pinned_node',
        'submit with sbatch --nodelist={pinned_node}',
    ):
        assert marker in text

    preflight = text.index("build_release_pin(receipt_path)")
    pin_match = text.index(
        "release snapshot B pin does not bind the supplied receipt"
    )
    node_match = text.index(
        'os.getenv("SLURM_JOB_NODELIST") != pinned_node'
    )
    topology = text.rindex('"${SCRIPT_DIR}/topology_guard.py"')
    assert preflight < pin_match < node_match < topology


def test_launcher_uses_dedicated_physical8_numa3_topology_contract() -> None:
    text = _text()

    for marker in (
        'RESULT_ROOT="${TASK_ROOT}/results/thc-complete-water2-audit"',
        'TOPOLOGY_RECORD="${RESULT_ROOT}/topology-physical8-${SLURM_JOB_ID}.json"',
        "--mode physical8",
        "--expected-gpu-numa 3",
        "--require-performance --",
        '"${SCRIPT_DIR}/thc_complete_water2_audit.py"',
    ):
        assert marker in text
    assert text.index("--mode physical8") < text.index(
        '"${SCRIPT_DIR}/thc_complete_water2_audit.py"'
    )


def test_launcher_passes_selected_reference_receipt_explicitly() -> None:
    text = _text()
    driver = text.rindex('"${SCRIPT_DIR}/thc_complete_water2_audit.py"')

    for marker in (
        "--gint-column-backend selected",
        '--gint-runtime-gate-receipt "${GINT_RUNTIME_GATE_RECEIPT}"',
        "--gint-column-kernel reference",
        "--rr-ring-kernel reference",
        '"gint_column_backend": "selected"',
        '"gint_column_kernel": "reference"',
    ):
        assert marker in text
        assert text.index(marker) < driver or marker.startswith('"gint_')


def test_launcher_freezes_the_staged_full_pair_and_inexact_scan() -> None:
    text = _text()

    for marker in (
        "--eri-tol 1e-8",
        "--rr-eig-cutoff 1e-6",
        "--amplitude-fit-tolerances 1e-4 1e-5 1e-6 1e-7",
        "--amplitude-initial-rank 512",
        "--amplitude-max-rank 1024",
        "--eri-candidate 512:1e-4",
        "--eri-candidate 768:1e-5",
        "--eri-candidate 1024:1e-6",
        "--occupation-change-floor 1e-12",
        "--exact-identity-atol 1e-9",
        "--candidate-delta-norm 1e-6",
        "--candidate-projected-residual-norm 1e-6",
        '"full-pair anchor"',
        '"staged inexact amplitude scan"',
    ):
        assert marker in text


def test_launcher_is_write_once_and_records_child_return_code() -> None:
    text = _text()

    for marker in (
        "refusing to overwrite complete THC audit artifact",
        "os.O_EXCL",
        "AUDIT_RETURN_CODE=$?",
        'RETURN_CODE_RECORD="${RESULT_ROOT}/launcher-return-${SLURM_JOB_ID}.json"',
        '"audit_return_code": audit_return_code',
        '"launcher_return_code": launcher_return_code',
        'exit "${AUDIT_RETURN_CODE}"',
        'exit "${POSTCHECK_RETURN_CODE}"',
    ):
        assert marker in text
    assert "--overwrite" not in text
    assert '"${AUDIT_ARGUMENTS[@]}" "$@"' not in text
    assert "accepts no positional overrides" in text


def test_launcher_cannot_publish_energy_or_performance_claims() -> None:
    text = _text()

    for marker in (
        '"audit_only": True',
        '"iterative_thc_energy_evaluated": False',
        '"performance_claim_eligible": False',
        '"accepted"',
        '"production_enabled"',
        '"performance_eligible"',
        '"formal_validation_eligible"',
        '"complete_validated"',
    ):
        assert marker in text
    assert "not an iterative THC-CCSD energy or performance claim" in text


def test_launcher_uses_the_receipt_pinned_controlled_scontrol_runtime() -> None:
    text = _text()

    for marker in (
        'export CCSD_SCONTROL_PATH="${EXPECTED_SCONTROL_PATH}"',
        "control-tools/slurm-23.02.4-local-rpath-v1/bin/scontrol",
        "control-tools/slurm-23.02.4-local-rpath-v1/lib",
        'RUNTIME_LIB_DIRS="${SCONTROL_LIBRARY_DIR}',
    ):
        assert marker in text
