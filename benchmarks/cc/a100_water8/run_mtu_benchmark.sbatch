#!/bin/bash
#SBATCH --job-name=ccsd-benchmark
#SBATCH --account=mri
#SBATCH --partition=mrigpu
#SBATCH --nodelist=compute-1-6
#SBATCH --gres=gpu:nvidia:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem-bind=verbose,map_mem:3
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913/logs/benchmark-%j.log
set -euo pipefail

TASKDIR=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913
BASELINE=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913
VENV="${BASELINE}/.venv"
EXPECTED_SCONTROL_PATH="${TASKDIR}/control-tools/slurm-23.02.4-local-rpath-v1/bin/scontrol"
if [[ -n "${CCSD_SCONTROL_PATH:-}" && \
      "${CCSD_SCONTROL_PATH}" != "${EXPECTED_SCONTROL_PATH}" ]]; then
  echo "CCSD_SCONTROL_PATH must be ${EXPECTED_SCONTROL_PATH}" >&2
  exit 2
fi
export CCSD_SCONTROL_PATH="${EXPECTED_SCONTROL_PATH}"
SCONTROL_LIBRARY_DIR="${TASKDIR}/control-tools/slurm-23.02.4-local-rpath-v1/lib"
SOURCE_ROOT="${CCSD_SOURCE_ROOT:?set CCSD_SOURCE_ROOT to an immutable MTU snapshot source directory}"
case "${SOURCE_ROOT}" in
  "${TASKDIR}"/snapshots/*/source) ;;
  *)
    echo "CCSD_SOURCE_ROOT must point below ${TASKDIR}/snapshots/<sha256>/source" >&2
    exit 2
    ;;
esac
EXPECTED_SHA=$(basename "$(dirname "${SOURCE_ROOT}")")
if [[ ! "${EXPECTED_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "snapshot directory is not content-addressed: ${SOURCE_ROOT}" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
ACTUAL_SHA=$("${VENV}/bin/python" - "${SOURCE_ROOT}" <<'PY'
import importlib.util
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
module_path = root / "benchmarks/cc/a100_water8/benchmark.py"
spec = importlib.util.spec_from_file_location("water8_benchmark_manifest", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(module._source_tree_digest(root)[0])
PY
)
ACTUAL_SHA=$(printf '%s\n' "${ACTUAL_SHA}" | tail -n 1)
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  echo "snapshot digest mismatch: ${ACTUAL_SHA} != ${EXPECTED_SHA}" >&2
  exit 2
fi
DEPLOYMENT_PROFILE="${CCSD_EXPECTED_DEPLOYMENT_PROFILE:-candidate}"
REQUIRE_CANONICAL_PRISTINE="${CCSD_REQUIRE_CANONICAL_PRISTINE:-0}"
case "${DEPLOYMENT_PROFILE}" in
  candidate|g0-canonical-pristine) ;;
  *) echo "invalid CCSD_EXPECTED_DEPLOYMENT_PROFILE: ${DEPLOYMENT_PROFILE}" >&2; exit 2 ;;
esac
case "${REQUIRE_CANONICAL_PRISTINE}" in
  0|1) ;;
  *) echo "CCSD_REQUIRE_CANONICAL_PRISTINE must be 0 or 1" >&2; exit 2 ;;
esac
SNAPSHOT_VALIDATION_ARGS=(
  --source-root "${SOURCE_ROOT}"
  --expected-task-root "${TASKDIR}"
  --expected-deployment-profile "${DEPLOYMENT_PROFILE}"
)
if [[ "${REQUIRE_CANONICAL_PRISTINE}" == "1" ]]; then
  SNAPSHOT_VALIDATION_ARGS+=(--require-canonical-pristine)
fi
"${VENV}/bin/python" \
  "${SOURCE_ROOT}/benchmarks/cc/a100_water8/snapshot_manifest.py" \
  "${SNAPSHOT_VALIDATION_ARGS[@]}" >/dev/null

CASE_ID="${CASE:?set CASE to a benchmark case id}"
METHOD_ID="${METHOD:?set METHOD to a benchmark method id}"
REPEAT_ID="${REPEAT:?set REPEAT to a unique fresh-process repeat id}"
GINT_RUNTIME_GATE_RECEIPT="${GINT_RUNTIME_GATE_RECEIPT:-}"
GINT_COLUMN_BACKEND_VALUE="${GINT_COLUMN_BACKEND:-selected}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${TASKDIR}/results/formal}"
RUN_FULL_SPACE_DIAGNOSTIC="${RUN_FULL_SPACE_DIAGNOSTIC:-0}"
case "${RUN_FULL_SPACE_DIAGNOSTIC}" in
  0|1) ;;
  *) echo "RUN_FULL_SPACE_DIAGNOSTIC must be 0 or 1" >&2; exit 2 ;;
esac
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd -P)
case "${OUTPUT_DIR}" in
  "${TASKDIR}"/results/*) ;;
  *)
    echo "BENCHMARK_OUTPUT_DIR must be below ${TASKDIR}/results" >&2
    exit 2
    ;;
esac

# FNO G2 A/B runs pass the canonical-orbital artifact through the environment
# so an sbatch dependency can name the producer and consumer without fragile
# positional shell quoting.  Keep these paths outside the immutable snapshot
# and fail before starting HF when the dependency artifact is missing.
ORBITAL_ARTIFACT_IN="${BENCHMARK_ORBITAL_ARTIFACT_IN:-}"
ORBITAL_ARTIFACT_OUT="${BENCHMARK_ORBITAL_ARTIFACT_OUT:-}"
if [[ -n "${ORBITAL_ARTIFACT_IN}" && -n "${ORBITAL_ARTIFACT_OUT}" ]]; then
  echo "BENCHMARK_ORBITAL_ARTIFACT_IN and BENCHMARK_ORBITAL_ARTIFACT_OUT are mutually exclusive" >&2
  exit 2
fi
resolve_artifact_path() {
  "${VENV}/bin/python" - "$1" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
}
if [[ -n "${ORBITAL_ARTIFACT_IN}" ]]; then
  ORBITAL_ARTIFACT_IN=$(resolve_artifact_path "${ORBITAL_ARTIFACT_IN}")
  if [[ ! -f "${ORBITAL_ARTIFACT_IN}" ]]; then
    echo "BENCHMARK_ORBITAL_ARTIFACT_IN is not a regular file: ${ORBITAL_ARTIFACT_IN}" >&2
    exit 2
  fi
fi
if [[ -n "${ORBITAL_ARTIFACT_OUT}" ]]; then
  ORBITAL_ARTIFACT_OUT=$(resolve_artifact_path "${ORBITAL_ARTIFACT_OUT}")
  if [[ -e "${ORBITAL_ARTIFACT_OUT}" ]]; then
    echo "refusing to overwrite BENCHMARK_ORBITAL_ARTIFACT_OUT: ${ORBITAL_ARTIFACT_OUT}" >&2
    exit 2
  fi
fi
for artifact_path in "${ORBITAL_ARTIFACT_IN}" "${ORBITAL_ARTIFACT_OUT}"; do
  [[ -z "${artifact_path}" ]] && continue
  case "${artifact_path}" in
    "${TASKDIR}"/results/*) ;;
    *)
      echo "benchmark orbital artifacts must be below ${TASKDIR}/results" >&2
      exit 2
      ;;
  esac
  case "${artifact_path}" in
    "${SOURCE_ROOT}"/*)
      echo "benchmark orbital artifacts must be outside the read-only source snapshot" >&2
      exit 2
      ;;
  esac
done

CUDA_SITE="${VENV}/lib/python3.11/site-packages"
CUDA_LIB_DIRS=$(find "${CUDA_SITE}/nvidia" "${CUDA_SITE}/cutensor" \
  -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="${SCONTROL_LIBRARY_DIR}:${BASELINE}/cuda-loader-lib:${CUDA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${SOURCE_ROOT}"
export CUPY_CACHE_DIR="${TASKDIR}/.cache/cupy"
export PYSCF_TMPDIR="${TASKDIR}/tmp"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PYTHONUNBUFFERED=1
export FROZEN_GPU4PYSCF_COMMIT=a89b3ae018d4e82968323ef95645d91adde294aa
export CCSD_TASK_ROOT="${TASKDIR}"
export CCSD_EXPECTED_DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE}"
export CCSD_REQUIRE_CANONICAL_PRISTINE="${REQUIRE_CANONICAL_PRISTINE}"

TOPOLOGY_DIR="${TASKDIR}/results/topology"
mkdir -p "${CUPY_CACHE_DIR}" "${PYSCF_TMPDIR}" "${TOPOLOGY_DIR}"
TOPOLOGY_RECORD="${TOPOLOGY_DIR}/physical8-benchmark-${SLURM_JOB_ID}.json"
cd "${TASKDIR}"

BENCHMARK_ARGUMENTS=()
if [[ "${RUN_FULL_SPACE_DIAGNOSTIC}" == "1" ]]; then
  BENCHMARK_ARGUMENTS+=(--run-full-space-diagnostic)
fi
# The benchmark deliberately rejects RR/THC runs whose approximation controls
# were inherited from argparse defaults.  Map the batch environment into real
# command-line arguments so every Slurm artifact records an explicit scientific
# choice and can be reproduced from its launch metadata.
case "${METHOD_ID}" in
  rr_cd|rr_canonical)
    BENCHMARK_ARGUMENTS+=(
      --eri-tol "${ERI_TOL:-1e-8}"
      --rr-eig-cutoff "${RR_EIG_CUTOFF:-1e-6}"
      --rr-ring-kernel "${RR_RING_KERNEL:-reference}"
    )
    ;;
  thc_cd|thc_canonical)
    BENCHMARK_ARGUMENTS+=(
      --eri-tol "${ERI_TOL:-1e-8}"
      --rr-eig-cutoff "${RR_EIG_CUTOFF:-1e-6}"
      --thc-fit-tol "${THC_FIT_TOL:-1e-6}"
      --rr-ring-kernel "${RR_RING_KERNEL:-reference}"
    )
    ;;
esac
if [[ -n "${GINT_RUNTIME_GATE_RECEIPT}" ]]; then
  GINT_RUNTIME_GATE_RECEIPT=$(resolve_artifact_path "${GINT_RUNTIME_GATE_RECEIPT}")
  if [[ ! -f "${GINT_RUNTIME_GATE_RECEIPT}" ]]; then
    echo "GINT_RUNTIME_GATE_RECEIPT is not a regular file: ${GINT_RUNTIME_GATE_RECEIPT}" >&2
    exit 2
  fi
  case "${GINT_RUNTIME_GATE_RECEIPT}/" in
    "${TASKDIR}/results/"*) ;;
    *)
      echo "GINT_RUNTIME_GATE_RECEIPT must be below ${TASKDIR}/results" >&2
      exit 2
      ;;
  esac
  if [[ "${GINT_COLUMN_BACKEND_VALUE}" != "selected" ]]; then
    echo "GINT_RUNTIME_GATE_RECEIPT requires GINT_COLUMN_BACKEND=selected" >&2
    exit 2
  fi
  case "${METHOD_ID}" in
    rr_cd|thc_cd) ;;
    *)
      echo "GINT_RUNTIME_GATE_RECEIPT applies only to rr_cd or thc_cd" >&2
      exit 2
      ;;
  esac
  BENCHMARK_ARGUMENTS+=(
    --gint-runtime-gate-receipt "${GINT_RUNTIME_GATE_RECEIPT}"
  )
fi
if [[ -n "${ORBITAL_ARTIFACT_IN}" ]]; then
  BENCHMARK_ARGUMENTS+=(--orbital-artifact-in "${ORBITAL_ARTIFACT_IN}")
fi
if [[ -n "${ORBITAL_ARTIFACT_OUT}" ]]; then
  BENCHMARK_ARGUMENTS+=(--orbital-artifact-out "${ORBITAL_ARTIFACT_OUT}")
fi
for specification in \
  "fno-thresh:${FNO_THRESH:-}" \
  "fno-nvir-act:${FNO_NVIR_ACT:-}" \
  "fno-pct-occ:${FNO_PCT_OCC:-}"; do
  name="${specification%%:*}"
  value="${specification#*:}"
  if [[ -n "${value}" ]]; then
    BENCHMARK_ARGUMENTS+=("--${name}" "${value}")
  fi
done

exec "${VENV}/bin/python" "${SOURCE_ROOT}/benchmarks/cc/a100_water8/topology_guard.py" \
  --mode physical8 --record "${TOPOLOGY_RECORD}" --require-performance -- \
  "${VENV}/bin/python" "${SOURCE_ROOT}/benchmarks/cc/a100_water8/benchmark.py" \
  --case "${CASE_ID}" --method "${METHOD_ID}" --device gpu --threads 8 \
  --repeat "${REPEAT_ID}" --output-dir "${OUTPUT_DIR}" \
  "${BENCHMARK_ARGUMENTS[@]}" "$@"
