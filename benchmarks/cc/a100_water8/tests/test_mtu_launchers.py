"""CPU-only contract checks for the MTU batch entry points.

The scientific drivers validate the snapshot again when they start, but every
Slurm entry point must reject a profile mismatch before allocating hours to a
run.  These tests keep the shell-level and Python-level provenance contracts in
lockstep without requiring Slurm, CUDA, or access to MTU.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

MANDATORY_SNAPSHOT_LAUNCHERS = {
    "run_mtu_benchmark.sbatch": "benchmark.py",
    "run_mtu_counterpoise.sbatch": "counterpoise.py",
    "run_mtu_gint_gate.sbatch": "gint_gate.py",
    "run_mtu_gpu_tests.sbatch": "gpu4pyscf/cc/tests/test_ccsd.py",
    "run_mtu_ncu.sbatch": "benchmark.py",
    "run_mtu_nsys.sbatch": "nsys_profile.sh",
    "run_mtu_rr_smoke.sbatch": "validate_rr_engine.py",
}

SNAPSHOT_OPTIONAL_LAUNCHERS = {
    "run_mtu_topology_probe.sbatch": "topology_child_probe.py",
    "run_numa3_8core.sbatch": "benchmark.py",
    "run_numa3_16thread.sbatch": "benchmark.py",
}

CANDIDATE_ONLY_LAUNCHERS = {
    "run_mtu_gint_gate.sbatch",
    "run_mtu_gpu_tests.sbatch",
    "run_mtu_rr_smoke.sbatch",
}

STRICT_NUMA_LAUNCHERS = (
    set(MANDATORY_SNAPSHOT_LAUNCHERS)
    | set(SNAPSHOT_OPTIONAL_LAUNCHERS)
) - {"run_mtu_gpu_tests.sbatch"}

GENERIC_SNAPSHOT_LAUNCHERS = (
    set(MANDATORY_SNAPSHOT_LAUNCHERS)
    | set(SNAPSHOT_OPTIONAL_LAUNCHERS)
) - CANDIDATE_ONLY_LAUNCHERS


@pytest.mark.parametrize(
    "name",
    sorted(set(MANDATORY_SNAPSHOT_LAUNCHERS) | set(SNAPSHOT_OPTIONAL_LAUNCHERS)),
)
def test_gpu_launchers_request_slurm_cpu_gpu_binding(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")

    assert "#SBATCH --gres-flags=enforce-binding" in text


@pytest.mark.parametrize("name", sorted(STRICT_NUMA_LAUNCHERS))
def test_strict_numa_launchers_reserve_all_physical_cores(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")

    assert "#SBATCH --cpus-per-task=64" in text
    assert "#SBATCH --hint=nomultithread" in text


@pytest.mark.parametrize(
    "name", ("run_mtu_gint_gate.sbatch", "run_mtu_gpu_tests.sbatch")
)
def test_candidate_validation_launchers_pin_fixed_mtu_node(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")

    # Snapshot A/B gates and the GPU regression suite must execute on the
    # fixed A100/EPYC target.  A command-line override is not sufficient for a
    # future direct ``sbatch`` invocation, so keep the pin in each artifact.
    assert "#SBATCH --nodelist=compute-1-6" in text


@pytest.mark.parametrize(
    "name",
    sorted(
        set(MANDATORY_SNAPSHOT_LAUNCHERS)
        | set(SNAPSHOT_OPTIONAL_LAUNCHERS)
        | {"nsys_profile.sh"}
    ),
)
def test_launcher_has_valid_bash_syntax(name: str) -> None:
    subprocess.run(["bash", "-n", str(ROOT / name)], check=True)


@pytest.mark.parametrize(
    ("name", "payload_marker"),
    sorted(
        {
            **MANDATORY_SNAPSHOT_LAUNCHERS,
            **SNAPSHOT_OPTIONAL_LAUNCHERS,
        }.items()
    ),
)
def test_snapshot_launchers_bind_profile_before_python_payload(
    name: str,
    payload_marker: str,
) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")

    for required in (
        "${CCSD_EXPECTED_DEPLOYMENT_PROFILE:-candidate}",
        "${CCSD_REQUIRE_CANONICAL_PRISTINE:-0}",
        "snapshot_manifest.py",
        "--expected-deployment-profile",
        "--require-canonical-pristine",
        "CCSD_REQUIRE_CANONICAL_PRISTINE must be 0 or 1",
        "export CCSD_TASK_ROOT=",
        "export CCSD_EXPECTED_DEPLOYMENT_PROFILE=",
        "export CCSD_REQUIRE_CANONICAL_PRISTINE=",
    ):
        assert required in text, f"{name} is missing {required!r}"

    validation = text.index("--expected-deployment-profile")
    payload = text.rindex(payload_marker)
    assert validation < payload
    for variable in (
        "export CCSD_TASK_ROOT=",
        "export CCSD_EXPECTED_DEPLOYMENT_PROFILE=",
        "export CCSD_REQUIRE_CANONICAL_PRISTINE=",
    ):
        assert validation < text.rindex(variable) < payload


@pytest.mark.parametrize("name", sorted(CANDIDATE_ONLY_LAUNCHERS))
def test_candidate_only_launchers_reject_non_candidate_snapshots(
    name: str,
) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")

    assert "requires the candidate deployment profile" in text
    assert '"${DEPLOYMENT_PROFILE}" != "candidate"' in text or (
        '"$DEPLOYMENT_PROFILE" != "candidate"' in text
    )


@pytest.mark.parametrize("name", sorted(GENERIC_SNAPSHOT_LAUNCHERS))
def test_generic_launchers_accept_explicit_g0_profile(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")

    assert "candidate|g0-canonical-pristine" in text
    assert "requires the candidate deployment profile" not in text


def test_gpu_launcher_runs_low_rank_residual_and_public_driver_tests() -> None:
    text = (ROOT / "run_mtu_gpu_tests.sbatch").read_text(encoding="utf-8")

    assert "gpu4pyscf/cc/tests/test_full_space_residual.py" in text
    assert "gpu4pyscf/cc/tests/test_thc_engine.py" in text
    assert "gpu4pyscf/cc/tests/test_thc_eri.py" in text
    assert "gpu4pyscf/cc/tests/test_thc_eri_factorization.py" in text
    assert "gpu4pyscf/cc/tests/test_thc_eri_preprocess.py" in text
    assert "gpu4pyscf/cc/tests/test_thc_complete_audit.py" in text
    assert "gpu4pyscf/cc/tests/test_thc_fhat.py" in text
    assert "gpu4pyscf/cc/tests/test_thc_mp2_weights.py" in text
    assert "gpu4pyscf/cc/tests/test_thc_omega_cd_audit.py" in text
    assert "gpu4pyscf/cc/tests/test_thc_omega_d.py" in text
    assert "gpu4pyscf/cc/tests/test_thc_omega_ghi.py" in text
    assert "gpu4pyscf/cc/tests/test_thc_rrccsd_cd.py" in text


def test_counterpoise_launcher_passes_one_external_orbital_bundle() -> None:
    text = (ROOT / "run_mtu_counterpoise.sbatch").read_text(
        encoding="utf-8"
    )

    for required in (
        "${CP_ORBITAL_BUNDLE_IN:-}",
        "${CP_ORBITAL_BUNDLE_OUT:-}",
        "CP_ORBITAL_BUNDLE_IN and CP_ORBITAL_BUNDLE_OUT are mutually exclusive",
        "CP orbital bundles must be below ${TASK_ROOT}/results",
        "CP orbital bundles must be outside the read-only source snapshot",
        "--orbital-bundle-in \"${ORBITAL_BUNDLE_IN}\"",
        "--orbital-bundle-out \"${ORBITAL_BUNDLE_OUT}\"",
    ):
        assert required in text
    assert '[[ ! -f "${ORBITAL_BUNDLE_IN}" ]]' in text
    assert '[[ -e "${ORBITAL_BUNDLE_OUT}" ]]' in text
    payload = text.rindex("counterpoise.py")
    assert text.index("CP_ORBITAL_BUNDLE_IN and") < payload
    assert text.index("--orbital-bundle-in") < payload
    assert text.index("--orbital-bundle-out") < payload


def test_benchmark_launcher_passes_fno_threshold_and_shared_orbital_artifact() -> None:
    text = (ROOT / "run_mtu_benchmark.sbatch").read_text(encoding="utf-8")

    for required in (
        "${BENCHMARK_ORBITAL_ARTIFACT_IN:-}",
        "${BENCHMARK_ORBITAL_ARTIFACT_OUT:-}",
        "BENCHMARK_ORBITAL_ARTIFACT_IN and BENCHMARK_ORBITAL_ARTIFACT_OUT are mutually exclusive",
        "benchmark orbital artifacts must be below ${TASKDIR}/results",
        '--orbital-artifact-in "${ORBITAL_ARTIFACT_IN}"',
        '--orbital-artifact-out "${ORBITAL_ARTIFACT_OUT}"',
        '"fno-thresh:${FNO_THRESH:-}"',
    ):
        assert required in text
    assert '[[ ! -f "${ORBITAL_ARTIFACT_IN}" ]]' in text
    assert '[[ -e "${ORBITAL_ARTIFACT_OUT}" ]]' in text
    payload = text.rindex("benchmark.py")
    assert text.index("BENCHMARK_ORBITAL_ARTIFACT_IN and") < payload
    assert text.index("--orbital-artifact-in") < payload
    assert text.index("--orbital-artifact-out") < payload


def test_benchmark_launcher_controls_separately_timed_full_space_diagnostic() -> None:
    text = (ROOT / "run_mtu_benchmark.sbatch").read_text(encoding="utf-8")

    for required in (
        '${RUN_FULL_SPACE_DIAGNOSTIC:-0}',
        'RUN_FULL_SPACE_DIAGNOSTIC must be 0 or 1',
        'if [[ "${RUN_FULL_SPACE_DIAGNOSTIC}" == "1" ]]',
        'BENCHMARK_ARGUMENTS+=(--run-full-space-diagnostic)',
    ):
        assert required in text
    validation = text.index("RUN_FULL_SPACE_DIAGNOSTIC must be 0 or 1")
    argument = text.index("BENCHMARK_ARGUMENTS+=(--run-full-space-diagnostic)")
    payload = text.rindex("benchmark.py")
    assert validation < argument < payload


def test_benchmark_launcher_passes_explicit_rr_and_thc_controls() -> None:
    text = (ROOT / "run_mtu_benchmark.sbatch").read_text(encoding="utf-8")

    for required in (
        'rr_cd|rr_canonical)',
        'thc_cd|thc_canonical)',
        '--eri-tol "${ERI_TOL:-1e-8}"',
        '--rr-eig-cutoff "${RR_EIG_CUTOFF:-1e-6}"',
        '--thc-fit-tol "${THC_FIT_TOL:-1e-6}"',
        '--rr-ring-kernel "${RR_RING_KERNEL:-reference}"',
    ):
        assert required in text
    controls = text.index("--eri-tol")
    payload = text.rindex("benchmark.py")
    assert controls < payload


def test_counterpoise_launcher_passes_rr_ring_kernel() -> None:
    text = (ROOT / "run_mtu_counterpoise.sbatch").read_text(encoding="utf-8")
    assert '--rr-ring-kernel "${RR_RING_KERNEL:-reference}"' in text
    assert text.index("--rr-ring-kernel") < text.rindex("counterpoise.py")
