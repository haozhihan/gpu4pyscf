#!/bin/bash
# Submit one stage through the receipt-compatible standard MTU launcher.
set -euo pipefail

TASK_ROOT=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913
BASELINE_ROOT=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913
PYTHON="${BASELINE_ROOT}/.venv/bin/python"
SOURCE_ROOT="${CCSD_SOURCE_ROOT:?set the release-B snapshot source}"
RECEIPT="${GINT_RUNTIME_GATE_RECEIPT:?set the release-B GINT receipt}"
ACCEPTANCE="${GINT_RELEASE_GATE_ACCEPTANCE:?set the completed release-gate acceptance}"
STAGE="${G4_STAGE:?set G4_STAGE to spectrum, probe, or converge}"
RUN_ID="${G4_RUN_ID:?set a unique G4_RUN_ID}"
case "${STAGE}" in spectrum|probe|converge) ;; *) exit 2 ;; esac
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]]; then
  echo "invalid G4_RUN_ID" >&2
  exit 2
fi
case "${SOURCE_ROOT}" in "${TASK_ROOT}"/snapshots/*/source) ;; *) exit 2 ;; esac

DRIVER="${SOURCE_ROOT}/benchmarks/cc/a100_water8/rr_g4_continue.py"
STANDARD_LAUNCHER="${SOURCE_ROOT}/benchmarks/cc/a100_water8/run_mtu_benchmark.sbatch"
COMMON_SBATCH=(
  --parsable --nodes=1 --ntasks=1 --gres=gpu:nvidia:1
  --gres-flags=enforce-binding
)
COMMON_EXPORT="CCSD_SOURCE_ROOT=${SOURCE_ROOT},CCSD_EXPECTED_DEPLOYMENT_PROFILE=candidate,CCSD_REQUIRE_CANONICAL_PRISTINE=0"
COMMON_RR_ARGS=(
  --rr-auxiliary-block-size 32 --rr-virtual-block-size 32
  --retain-cupy-cache
)

if [[ "${STAGE}" == "spectrum" ]]; then
  if [[ -n "${G4_RR_EIG_CUTOFF:-}${G4_SPECTRUM_SUMMARY:-}${G4_PREVIOUS_GATE:-}${G4_PROBE_SUMMARY:-}" ]]; then
    echo "spectrum does not accept continuation inputs" >&2
    exit 2
  fi
  NODE=$("${PYTHON}" "${DRIVER}" receipt-node \
    --source-root "${SOURCE_ROOT}" \
    --gint-runtime-gate-receipt "${RECEIPT}" \
    --gint-release-gate-acceptance "${ACCEPTANCE}")
else
  CUTOFF="${G4_RR_EIG_CUTOFF:?set G4_RR_EIG_CUTOFF}"
  SPECTRUM="${G4_SPECTRUM_SUMMARY:?set G4_SPECTRUM_SUMMARY}"
  AUTHORIZE=(
    "${PYTHON}" "${DRIVER}" authorize
    --task-root "${TASK_ROOT}" --source-root "${SOURCE_ROOT}"
    --gint-runtime-gate-receipt "${RECEIPT}"
    --gint-release-gate-acceptance "${ACCEPTANCE}"
    --stage "${STAGE}" --rr-eig-cutoff "${CUTOFF}"
    --spectrum-summary "${SPECTRUM}"
  )
  [[ -z "${G4_PREVIOUS_GATE:-}" ]] || AUTHORIZE+=(--previous-gate "${G4_PREVIOUS_GATE}")
  [[ -z "${G4_PROBE_SUMMARY:-}" ]] || AUTHORIZE+=(--probe-summary "${G4_PROBE_SUMMARY}")
  AUTHORIZATION=$("${AUTHORIZE[@]}")
  read -r NODE ORBITAL_ARTIFACT < <("${PYTHON}" -c \
    'import json,sys; x=json.load(sys.stdin); print(x["node"], x["orbital_artifact_path"])' \
    <<<"${AUTHORIZATION}")
fi
COMMON_SBATCH+=(--nodelist="${NODE}")

OUTPUT_DIR="${TASK_ROOT}/results/rr-g4-water4/${RUN_ID}-${STAGE}"
if ! mkdir "${OUTPUT_DIR}"; then
  echo "refusing to reuse output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
"${PYTHON}" - "${OUTPUT_DIR}/request.json" "${STAGE}" "${RUN_ID}" \
  "${SOURCE_ROOT}" "${RECEIPT}" "${ACCEPTANCE}" "${NODE}" <<'PY'
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = dict(zip(
    ("stage", "run_id", "source_root", "receipt", "release_gate_acceptance", "node"),
    sys.argv[2:], strict=True,
))
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

submit() {
  local export_values=$1 dependency=$2
  shift 2
  local command=(sbatch "${COMMON_SBATCH[@]}")
  [[ -z "${dependency}" ]] || command+=(--dependency="${dependency}")
  command+=(--export="ALL,${COMMON_EXPORT},${export_values}" "${STANDARD_LAUNCHER}")
  command+=("$@")
  local result
  result=$("${command[@]}")
  result=${result%%;*}
  [[ "${result}" =~ ^[0-9]+$ ]] || return 2
  printf '%s\n' "${result}"
}

if [[ "${STAGE}" == "spectrum" ]]; then
  ORBITAL_ARTIFACT="${OUTPUT_DIR}/water4-canonical-orbitals.npz"
  ORACLE=$(submit \
    "CASE=water4-tz,METHOD=canonical,REPEAT=${RUN_ID}-oracle,BENCHMARK_OUTPUT_DIR=${OUTPUT_DIR},BENCHMARK_ORBITAL_ARTIFACT_OUT=${ORBITAL_ARTIFACT}" "")
  PROBE_1E9=$(submit \
    "CASE=water4-tz,METHOD=rr_cd,REPEAT=${RUN_ID}-rank-1e9,BENCHMARK_OUTPUT_DIR=${OUTPUT_DIR},BENCHMARK_ORBITAL_ARTIFACT_IN=${ORBITAL_ARTIFACT},GINT_RUNTIME_GATE_RECEIPT=${RECEIPT},RR_EIG_CUTOFF=1e-9,RR_RING_KERNEL=gemm" \
    "afterok:${ORACLE}" --max-cycle 1 --skip-checkpoint "${COMMON_RR_ARGS[@]}")
  PROBE_1E11=$(submit \
    "CASE=water4-tz,METHOD=rr_cd,REPEAT=${RUN_ID}-rank-1e11,BENCHMARK_OUTPUT_DIR=${OUTPUT_DIR},BENCHMARK_ORBITAL_ARTIFACT_IN=${ORBITAL_ARTIFACT},GINT_RUNTIME_GATE_RECEIPT=${RECEIPT},RR_EIG_CUTOFF=1e-11,RR_RING_KERNEL=gemm" \
    "afterok:${ORACLE}" --max-cycle 1 --skip-checkpoint "${COMMON_RR_ARGS[@]}")
  printf '{"oracle":"%s","probe_1e9":"%s","probe_1e11":"%s","output":"%s"}\n' \
    "${ORACLE}" "${PROBE_1E9}" "${PROBE_1E11}" "${OUTPUT_DIR}"
elif [[ "${STAGE}" == "probe" ]]; then
  JOB=$(submit \
    "CASE=water4-tz,METHOD=rr_cd,REPEAT=${RUN_ID}-rank,BENCHMARK_OUTPUT_DIR=${OUTPUT_DIR},BENCHMARK_ORBITAL_ARTIFACT_IN=${ORBITAL_ARTIFACT},GINT_RUNTIME_GATE_RECEIPT=${RECEIPT},RR_EIG_CUTOFF=${CUTOFF},RR_RING_KERNEL=gemm" \
    "" --max-cycle 1 --skip-checkpoint "${COMMON_RR_ARGS[@]}")
  printf '{"probe":"%s","output":"%s"}\n' "${JOB}" "${OUTPUT_DIR}"
else
  JOB=$(submit \
    "CASE=water4-tz,METHOD=rr_cd,REPEAT=${RUN_ID}-rr,BENCHMARK_OUTPUT_DIR=${OUTPUT_DIR},BENCHMARK_ORBITAL_ARTIFACT_IN=${ORBITAL_ARTIFACT},GINT_RUNTIME_GATE_RECEIPT=${RECEIPT},RR_EIG_CUTOFF=${CUTOFF},RR_RING_KERNEL=gemm,RUN_FULL_SPACE_DIAGNOSTIC=1" \
    "" "${COMMON_RR_ARGS[@]}")
  printf '{"converge":"%s","output":"%s"}\n' "${JOB}" "${OUTPUT_DIR}"
fi
