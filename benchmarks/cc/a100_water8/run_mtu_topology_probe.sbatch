#!/bin/bash
#SBATCH --job-name=ccsd-topology-probe
#SBATCH --account=mri
#SBATCH --partition=mrigpu
#SBATCH --gres=gpu:nvidia:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem-bind=verbose,map_mem:3
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

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
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/benchmarks/cc/a100_water8/topology_guard.py" ]]; then
  REPOSITORY_ROOT="${SLURM_SUBMIT_DIR}"
  DETECTED_TASK_ROOT=""
else
  SCRIPT_LOCATION="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPOSITORY_ROOT="$(cd -- "${SCRIPT_LOCATION}/../../.." && pwd)"
  DETECTED_TASK_ROOT=""
fi
SCRIPT_DIR="${REPOSITORY_ROOT}/benchmarks/cc/a100_water8"
PYTHON_BIN="${PYTHON_BIN:-python3}"
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
  DEFAULT_RECORD_DIR="${DETECTED_TASK_ROOT}/results/topology"
else
  DEFAULT_RECORD_DIR="${SCRIPT_DIR}/results/topology"
fi
RECORD_DIR="${TOPOLOGY_RECORD_DIR:-${DEFAULT_RECORD_DIR}}"
mkdir -p "${RECORD_DIR}"

hostname
PROBE_STATUS=0
if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/topology_guard.py" \
    --mode physical8 \
    --record "${RECORD_DIR}/probe-physical8-${SLURM_JOB_ID}.json" \
    --require-performance -- \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/topology_child_probe.py" \
    --expected-affinity 24-31 --expected-numa 3; then
  PROBE_STATUS=2
fi

if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/topology_guard.py" \
    --mode logical16 \
    --record "${RECORD_DIR}/probe-logical16-${SLURM_JOB_ID}.json" \
    --require-performance -- \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/topology_child_probe.py" \
    --expected-affinity 24-31,88-95 --expected-numa 3; then
  PROBE_STATUS=2
fi
exit "${PROBE_STATUS}"
