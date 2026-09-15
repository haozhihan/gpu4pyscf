#!/bin/bash
#SBATCH --account=mri
#SBATCH --partition=mrigpu
#SBATCH --gres=gpu:nvidia:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem-bind=verbose,map_mem:3
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

RUN_THREADS=16
export OMP_NUM_THREADS="${RUN_THREADS}"
export OPENBLAS_NUM_THREADS="${RUN_THREADS}"
export MKL_NUM_THREADS="${RUN_THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=threads

if [[ -n "${CCSD_SOURCE_ROOT:-}" ]]; then
  REPOSITORY_ROOT="${CCSD_SOURCE_ROOT}"
  case "${REPOSITORY_ROOT}" in
    */snapshots/*/source)
      DETECTED_TASK_ROOT="$(cd -- "${REPOSITORY_ROOT}/../../.." && pwd)"
      ;;
    *)
      echo "CCSD_SOURCE_ROOT must point to snapshots/<sha256>/source" >&2
      exit 2
      ;;
  esac
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/benchmarks/cc/a100_water8/benchmark.py" ]]; then
  REPOSITORY_ROOT="${SLURM_SUBMIT_DIR}"
  DETECTED_TASK_ROOT=""
else
  SCRIPT_LOCATION="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPOSITORY_ROOT="$(cd -- "${SCRIPT_LOCATION}/../../.." && pwd)"
  DETECTED_TASK_ROOT=""
fi
SCRIPT_DIR="${REPOSITORY_ROOT}/benchmarks/cc/a100_water8"
cd "${REPOSITORY_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${DETECTED_TASK_ROOT}" ]]; then
  export PYTHONDONTWRITEBYTECODE=1
  DEPLOYMENT_PROFILE="${CCSD_EXPECTED_DEPLOYMENT_PROFILE:-candidate}"
  REQUIRE_CANONICAL_PRISTINE="${CCSD_REQUIRE_CANONICAL_PRISTINE:-0}"
  case "${DEPLOYMENT_PROFILE}" in
    candidate|g0-canonical-pristine) ;;
    *)
      echo "invalid CCSD_EXPECTED_DEPLOYMENT_PROFILE: ${DEPLOYMENT_PROFILE}" >&2
      exit 2
      ;;
  esac
  case "${REQUIRE_CANONICAL_PRISTINE}" in
    0|1) ;;
    *)
      echo "CCSD_REQUIRE_CANONICAL_PRISTINE must be 0 or 1" >&2
      exit 2
      ;;
  esac
  SNAPSHOT_VALIDATION_ARGS=(
    --source-root "${REPOSITORY_ROOT}"
    --expected-task-root "${DETECTED_TASK_ROOT}"
    --expected-deployment-profile "${DEPLOYMENT_PROFILE}"
  )
  if [[ "${REQUIRE_CANONICAL_PRISTINE}" == "1" ]]; then
    SNAPSHOT_VALIDATION_ARGS+=(--require-canonical-pristine)
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/snapshot_manifest.py" \
    "${SNAPSHOT_VALIDATION_ARGS[@]}" >/dev/null
  export CCSD_TASK_ROOT="${DETECTED_TASK_ROOT}"
  export CCSD_EXPECTED_DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE}"
  export CCSD_REQUIRE_CANONICAL_PRISTINE="${REQUIRE_CANONICAL_PRISTINE}"
fi
if [[ -n "${DETECTED_TASK_ROOT}" ]]; then
  DEFAULT_RESULT_DIR="${DETECTED_TASK_ROOT}/results/formal"
  DEFAULT_TOPOLOGY_DIR="${DETECTED_TASK_ROOT}/results/topology"
else
  DEFAULT_RESULT_DIR="${SCRIPT_DIR}/results"
  DEFAULT_TOPOLOGY_DIR="${SCRIPT_DIR}/results/topology"
fi
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${DEFAULT_RESULT_DIR}}"
TOPOLOGY_RECORD_DIR="${TOPOLOGY_RECORD_DIR:-${DEFAULT_TOPOLOGY_DIR}}"
mkdir -p "${BENCHMARK_OUTPUT_DIR}" "${TOPOLOGY_RECORD_DIR}"
TOPOLOGY_RECORD="${TOPOLOGY_RECORD_DIR}/logical16-${SLURM_JOB_ID:-manual}.json"
GUARD_POLICY=(--require-performance)
if [[ "${ALLOW_CORRECTNESS_ONLY:-0}" == "1" ]]; then
  GUARD_POLICY=()
fi

# 32 allocated physical cores expose their SMT siblings in this cluster's
# batch cgroup.  topology_guard.py selects the GPU-local eight cores only.
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/topology_guard.py" \
  --mode logical16 --record "${TOPOLOGY_RECORD}" "${GUARD_POLICY[@]}" -- \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark.py" \
  --case "${CASE:-water8-tz}" --method "${METHOD:-canonical}" \
  --device gpu --threads "${RUN_THREADS}" \
  --output-dir "${BENCHMARK_OUTPUT_DIR}" "$@"
