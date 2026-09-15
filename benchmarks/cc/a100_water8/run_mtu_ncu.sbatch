#!/bin/bash
#SBATCH --job-name=ccsd-ncu
#SBATCH --account=mri
#SBATCH --partition=mrigpu
#SBATCH --gres=gpu:nvidia:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem-bind=verbose,map_mem:3
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913/logs/ncu-%j.log
set -euo pipefail

TASKDIR=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913
BASELINE=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913
VENV="${BASELINE}/.venv"
SOURCE_ROOT="${CCSD_SOURCE_ROOT:?set CCSD_SOURCE_ROOT to an immutable MTU snapshot source directory}"
case "${SOURCE_ROOT}" in
  "${TASKDIR}"/snapshots/*/source) ;;
  *)
    echo "CCSD_SOURCE_ROOT must point below ${TASKDIR}/snapshots/<sha256>/source" >&2
    exit 2
    ;;
esac
if [[ $# -lt 2 ]]; then
  echo "usage: sbatch $0 PROFILE_NAME KERNEL_REGEX [benchmark.py arguments...]" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
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
PROFILE_NAME=$1
KERNEL_REGEX=$2
shift 2
if [[ "${PROFILE_NAME}" == */* || -z "${PROFILE_NAME}" ]]; then
  echo "PROFILE_NAME must be a nonempty filename component" >&2
  exit 2
fi

NCU=/usr/local/cuda/bin/ncu
if [[ ! -x "${NCU}" ]]; then
  echo "missing ncu at ${NCU}" >&2
  exit 2
fi
CUDA_SITE="${VENV}/lib/python3.11/site-packages"
CUDA_LIB_DIRS=$(find "${CUDA_SITE}/nvidia" "${CUDA_SITE}/cutensor" \
  -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="${BASELINE}/cuda-loader-lib:${CUDA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${SOURCE_ROOT}"
export CUPY_CACHE_DIR="${TASKDIR}/.cache/cupy"
export PYSCF_TMPDIR="${TASKDIR}/tmp"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PYTHONUNBUFFERED=1
export CCSD_TASK_ROOT="${TASKDIR}"
export CCSD_EXPECTED_DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE}"
export CCSD_REQUIRE_CANONICAL_PRISTINE="${REQUIRE_CANONICAL_PRISTINE}"

PROFILE_DIR="${TASKDIR}/results/nsight-compute"
BENCHMARK_OUTPUT_DIR="${NCU_BENCHMARK_OUTPUT_DIR:-${TASKDIR}/results/nsight-compute-benchmarks}"
TOPOLOGY_DIR="${TASKDIR}/results/topology"
mkdir -p "${CUPY_CACHE_DIR}" "${PYSCF_TMPDIR}" "${PROFILE_DIR}" \
  "${BENCHMARK_OUTPUT_DIR}" "${TOPOLOGY_DIR}"
PREFIX="${PROFILE_DIR}/${PROFILE_NAME}"
if [[ -e "${PREFIX}.ncu-rep" ]]; then
  echo "refusing to overwrite Nsight Compute output: ${PREFIX}.ncu-rep" >&2
  exit 2
fi
TOPOLOGY_RECORD="${TOPOLOGY_DIR}/physical8-ncu-${SLURM_JOB_ID}.json"
cd "${TASKDIR}"

exec "${VENV}/bin/python" "${SOURCE_ROOT}/benchmarks/cc/a100_water8/topology_guard.py" \
  --mode physical8 --record "${TOPOLOGY_RECORD}" --require-performance -- \
  "${NCU}" --target-processes all --kernel-name-base demangled \
  --kernel-name "regex:${KERNEL_REGEX}" --launch-skip-before-match "${NCU_LAUNCH_SKIP_BEFORE_MATCH:-0}" \
  --launch-count "${NCU_LAUNCH_COUNT:-1}" --replay-mode kernel \
  --set "${NCU_SECTION_SET:-full}" --export "${PREFIX}" \
  "${VENV}/bin/python" "${SOURCE_ROOT}/benchmarks/cc/a100_water8/benchmark.py" \
  "$@" --output-dir "${BENCHMARK_OUTPUT_DIR}"
