#!/bin/bash
#SBATCH --job-name=ccsd-counterpoise
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
#SBATCH --output=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913/logs/counterpoise-%j.log
set -euo pipefail

TASK_ROOT="${CCSD_TASK_ROOT:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913}"
SOURCE_ROOT="${CCSD_SOURCE_ROOT:?set CCSD_SOURCE_ROOT to an immutable MTU snapshot source directory}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run_mtu_counterpoise.sbatch requires a Slurm job" >&2
  exit 2
fi

TASK_ROOT="$("${PYTHON_BIN}" - "${TASK_ROOT}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
SOURCE_ROOT="$("${PYTHON_BIN}" - "${SOURCE_ROOT}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
EXPECTED_SCONTROL_PATH="${TASK_ROOT}/control-tools/slurm-23.02.4-local-rpath-v1/bin/scontrol"
if [[ -n "${CCSD_SCONTROL_PATH:-}" && \
      "${CCSD_SCONTROL_PATH}" != "${EXPECTED_SCONTROL_PATH}" ]]; then
  echo "CCSD_SCONTROL_PATH must be ${EXPECTED_SCONTROL_PATH}" >&2
  exit 2
fi
export CCSD_SCONTROL_PATH="${EXPECTED_SCONTROL_PATH}"
SCONTROL_LIBRARY_DIR="${TASK_ROOT}/control-tools/slurm-23.02.4-local-rpath-v1/lib"
case "${SOURCE_ROOT}" in
  "${TASK_ROOT}"/snapshots/*/source) ;;
  *)
    echo "CCSD_SOURCE_ROOT must be ${TASK_ROOT}/snapshots/<sha256>/source" >&2
    exit 2
    ;;
esac
SCRIPT_DIR="${SOURCE_ROOT}/benchmarks/cc/a100_water8"
if [[ ! -f "${SCRIPT_DIR}/counterpoise.py" || ! -f "${SCRIPT_DIR}/snapshot_manifest.py" || ! -f "${SCRIPT_DIR}/topology_guard.py" ]]; then
  echo "source snapshot is missing a required counterpoise launcher component" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export FROZEN_GPU4PYSCF_COMMIT="a89b3ae018d4e82968323ef95645d91adde294aa"
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
  --source-root "${SOURCE_ROOT}"
  --expected-task-root "${TASK_ROOT}"
  --expected-base-revision "${FROZEN_GPU4PYSCF_COMMIT}"
  --expected-deployment-profile "${DEPLOYMENT_PROFILE}"
)
if [[ "${REQUIRE_CANONICAL_PRISTINE}" == "1" ]]; then
  SNAPSHOT_VALIDATION_ARGS+=(--require-canonical-pristine)
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/snapshot_manifest.py" \
  "${SNAPSHOT_VALIDATION_ARGS[@]}" >/dev/null
export CCSD_TASK_ROOT="${TASK_ROOT}"
export CCSD_EXPECTED_DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE}"
export CCSD_REQUIRE_CANONICAL_PRISTINE="${REQUIRE_CANONICAL_PRISTINE}"

RESULT_ROOT="${CP_OUTPUT_ROOT:-${TASK_ROOT}/results/counterpoise}"
RESULT_ROOT="$("${PYTHON_BIN}" - "${RESULT_ROOT}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
case "${RESULT_ROOT}/" in
  "${TASK_ROOT}/results/"*) ;;
  *)
    echo "CP_OUTPUT_ROOT must be below ${TASK_ROOT}/results and outside the source snapshot" >&2
    exit 2
    ;;
esac
case "${RESULT_ROOT}/" in
  "${SOURCE_ROOT}/"*)
    echo "counterpoise output must be outside the read-only source snapshot" >&2
    exit 2
    ;;
esac

ORBITAL_BUNDLE_IN="${CP_ORBITAL_BUNDLE_IN:-}"
ORBITAL_BUNDLE_OUT="${CP_ORBITAL_BUNDLE_OUT:-}"
if [[ -n "${ORBITAL_BUNDLE_IN}" && -n "${ORBITAL_BUNDLE_OUT}" ]]; then
  echo "CP_ORBITAL_BUNDLE_IN and CP_ORBITAL_BUNDLE_OUT are mutually exclusive" >&2
  exit 2
fi
if [[ -n "${ORBITAL_BUNDLE_IN}" ]]; then
  ORBITAL_BUNDLE_IN="$("${PYTHON_BIN}" - "${ORBITAL_BUNDLE_IN}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
  if [[ ! -f "${ORBITAL_BUNDLE_IN}" ]]; then
    echo "CP_ORBITAL_BUNDLE_IN is not a regular file: ${ORBITAL_BUNDLE_IN}" >&2
    exit 2
  fi
fi
if [[ -n "${ORBITAL_BUNDLE_OUT}" ]]; then
  ORBITAL_BUNDLE_OUT="$("${PYTHON_BIN}" - "${ORBITAL_BUNDLE_OUT}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
  if [[ -e "${ORBITAL_BUNDLE_OUT}" ]]; then
    echo "refusing to overwrite CP_ORBITAL_BUNDLE_OUT: ${ORBITAL_BUNDLE_OUT}" >&2
    exit 2
  fi
fi
for orbital_bundle_path in "${ORBITAL_BUNDLE_IN}" "${ORBITAL_BUNDLE_OUT}"; do
  if [[ -z "${orbital_bundle_path}" ]]; then
    continue
  fi
  case "${orbital_bundle_path}/" in
    "${TASK_ROOT}/results/"*) ;;
    *)
      echo "CP orbital bundles must be below ${TASK_ROOT}/results" >&2
      exit 2
      ;;
  esac
  case "${orbital_bundle_path}/" in
    "${SOURCE_ROOT}/"*)
      echo "CP orbital bundles must be outside the read-only source snapshot" >&2
      exit 2
      ;;
  esac
done

CASE_ID="${CASE:?set CASE to a WATER27 benchmark case id}"
METHOD_ID="${METHOD:?set METHOD to a counterpoise method id}"
RUN_ID="${CP_RUN_ID:-${SLURM_JOB_ID}}"
GINT_RUNTIME_GATE_RECEIPT="${GINT_RUNTIME_GATE_RECEIPT:-}"
GINT_COLUMN_BACKEND_VALUE="${GINT_COLUMN_BACKEND:-selected}"
GINT_COLUMN_KERNEL_VALUE="${GINT_COLUMN_KERNEL:-reference}"
case "${GINT_COLUMN_KERNEL_VALUE}" in
  reference|grouped) ;;
  *)
    echo "GINT_COLUMN_KERNEL must be reference or grouped" >&2
    exit 2
    ;;
esac
if [[ ! "${CASE_ID}" =~ ^[A-Za-z0-9._-]+$ || ! "${METHOD_ID}" =~ ^[A-Za-z0-9._-]+$ || ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "CASE, METHOD, and CP_RUN_ID may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

export PYTHONPATH="${SOURCE_ROOT}"
export CUPY_CACHE_DIR="$("${PYTHON_BIN}" - "${CCSD_CUPY_CACHE_DIR:-${TASK_ROOT}/.cache/cupy}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
export PYSCF_TMPDIR="$("${PYTHON_BIN}" - "${CCSD_PYSCF_TMPDIR:-${TASK_ROOT}/tmp}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
for external_path in "${CUPY_CACHE_DIR}" "${PYSCF_TMPDIR}"; do
  case "${external_path}/" in
    "${SOURCE_ROOT}/"*)
      echo "cache and scratch paths must be outside the read-only source snapshot" >&2
      exit 2
      ;;
  esac
done
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export GPU4PYSCF_NUMA=3

PYTHON_SITE="$("${PYTHON_BIN}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
CUDA_LIBRARY_ROOTS=()
for root in "${PYTHON_SITE}/nvidia" "${PYTHON_SITE}/cutensor"; do
  if [[ -d "${root}" ]]; then
    CUDA_LIBRARY_ROOTS+=("${root}")
  fi
done
CUDA_LIB_DIRS=""
if (( ${#CUDA_LIBRARY_ROOTS[@]} )); then
  CUDA_LIB_DIRS="$(find "${CUDA_LIBRARY_ROOTS[@]}" -maxdepth 2 -type d -name lib -print | paste -sd: -)"
fi
VIRTUAL_ENV_ROOT="$(cd -- "$(dirname -- "${PYTHON_BIN}")/.." && pwd)"
CUDA_LOADER_LIB="${CCSD_CUDA_LOADER_LIB:-$(dirname -- "${VIRTUAL_ENV_ROOT}")/cuda-loader-lib}"
RUNTIME_LIB_DIRS="${SCONTROL_LIBRARY_DIR}${CUDA_LIB_DIRS:+:${CUDA_LIB_DIRS}}"
if [[ -d "${CUDA_LOADER_LIB}" ]]; then
  RUNTIME_LIB_DIRS="${SCONTROL_LIBRARY_DIR}:${CUDA_LOADER_LIB}${CUDA_LIB_DIRS:+:${CUDA_LIB_DIRS}}"
fi
if [[ -n "${RUNTIME_LIB_DIRS}" ]]; then
  export LD_LIBRARY_PATH="${RUNTIME_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

mkdir -p "${RESULT_ROOT}" "${CUPY_CACHE_DIR}" "${PYSCF_TMPDIR}" "${TASK_ROOT}/logs"
OUTPUT_FILE="${RESULT_ROOT}/${CASE_ID}__${METHOD_ID}__${RUN_ID}.json"
TOPOLOGY_RECORD="${RESULT_ROOT}/topology-physical8-${SLURM_JOB_ID}.json"
if [[ -e "${OUTPUT_FILE}" || -e "${TOPOLOGY_RECORD}" ]]; then
  echo "refusing to overwrite an existing counterpoise or topology artifact" >&2
  exit 2
fi

CP_ARGUMENTS=(
  --case "${CASE_ID}"
  --method "${METHOD_ID}"
  --device gpu
  --output "${OUTPUT_FILE}"
  --scf-conv-tol "${SCF_CONV_TOL:-1e-10}"
  --cc-conv-tol "${CC_CONV_TOL:-1e-8}"
  --cc-conv-tol-normt "${CC_CONV_TOL_NORMT:-1e-6}"
  --cc-max-cycle "${CC_MAX_CYCLE:-50}"
  --max-memory-mb "${MAX_MEMORY_MB:-110000}"
  --eri-tol "${ERI_TOL:-1e-8}"
  --direct-scf-tol "${DIRECT_SCF_TOL:-1e-13}"
  --gint-column-backend "${GINT_COLUMN_BACKEND:-selected}"
  --gint-column-kernel "${GINT_COLUMN_KERNEL_VALUE}"
  --gint-group-size "${GINT_GROUP_SIZE:-16}"
  --gint-max-batch-size "${GINT_MAX_BATCH_SIZE:-32}"
  --cd-mo-block-size "${CD_MO_BLOCK_SIZE:-32}"
  --rr-eig-cutoff "${RR_EIG_CUTOFF:-1e-6}"
  --denominator-tolerance "${DENOMINATOR_TOLERANCE:-1e-10}"
  --rr-initial-rank "${RR_INITIAL_RANK:-32}"
  --rr-solver-tolerance "${RR_SOLVER_TOLERANCE:-1e-10}"
  --rr-dense-fallback-dimension "${RR_DENSE_FALLBACK_DIMENSION:-64}"
  --rr-auxiliary-block-size "${RR_AUXILIARY_BLOCK_SIZE:-1}"
  --rr-virtual-block-size "${RR_VIRTUAL_BLOCK_SIZE:-8}"
  --rr-ring-kernel "${RR_RING_KERNEL:-reference}"
  --precision fp64
  --thc-fit-tol "${THC_FIT_TOL:-1e-6}"
  --thc-rank-growth "${THC_RANK_GROWTH:-1.5}"
  --thc-orthogonality-cutoff "${THC_ORTHOGONALITY_CUTOFF:-1e-12}"
  --thc-orthogonality-tolerance "${THC_ORTHOGONALITY_TOLERANCE:-1e-10}"
  --thc-max-iterations "${THC_MAX_ITERATIONS:-500}"
  --thc-als-convergence-tolerance "${THC_ALS_CONVERGENCE_TOLERANCE:-1e-10}"
  --thc-ridge "${THC_RIDGE:-1e-12}"
  --thc-seed "${THC_SEED:-0}"
  --fno-thresh "${FNO_THRESH:-1e-6}"
)
if [[ -n "${GINT_RUNTIME_GATE_RECEIPT}" ]]; then
  GINT_RUNTIME_GATE_RECEIPT="$("${PYTHON_BIN}" - "${GINT_RUNTIME_GATE_RECEIPT}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
)"
  if [[ ! -f "${GINT_RUNTIME_GATE_RECEIPT}" ]]; then
    echo "GINT_RUNTIME_GATE_RECEIPT is not a regular file: ${GINT_RUNTIME_GATE_RECEIPT}" >&2
    exit 2
  fi
  case "${GINT_RUNTIME_GATE_RECEIPT}/" in
    "${TASK_ROOT}/results/"*) ;;
    *)
      echo "GINT_RUNTIME_GATE_RECEIPT must be below ${TASK_ROOT}/results" >&2
      exit 2
      ;;
  esac
  if [[ "${GINT_COLUMN_BACKEND_VALUE}" != "selected" ]]; then
    echo "GINT_RUNTIME_GATE_RECEIPT requires GINT_COLUMN_BACKEND=selected" >&2
    exit 2
  fi
  if [[ "${GINT_COLUMN_KERNEL_VALUE}" != "reference" ]]; then
    echo "grouped selected-column prototype cannot consume a release receipt" >&2
    exit 2
  fi
  case "${METHOD_ID}" in
    rr_cd|thc_cd) ;;
    *)
      echo "GINT_RUNTIME_GATE_RECEIPT applies only to rr_cd or thc_cd" >&2
      exit 2
      ;;
  esac
  CP_ARGUMENTS+=(
    --gint-runtime-gate-receipt "${GINT_RUNTIME_GATE_RECEIPT}"
  )
fi
if [[ -n "${ORBITAL_BUNDLE_IN}" ]]; then
  CP_ARGUMENTS+=(--orbital-bundle-in "${ORBITAL_BUNDLE_IN}")
fi
if [[ -n "${ORBITAL_BUNDLE_OUT}" ]]; then
  CP_ARGUMENTS+=(--orbital-bundle-out "${ORBITAL_BUNDLE_OUT}")
fi
for specification in \
  "cd-max-rank:${CD_MAX_RANK:-}" \
  "gint-max-block-bytes:${GINT_MAX_BLOCK_BYTES:-}" \
  "rr-max-rank:${RR_MAX_RANK:-}" \
  "denominator-max-rank:${DENOMINATOR_MAX_RANK:-}" \
  "rr-solver-maxiter:${RR_SOLVER_MAXITER:-}" \
  "rr-ritz-residual-tolerance:${RR_RITZ_RESIDUAL_TOLERANCE:-}" \
  "thc-rank:${THC_RANK:-}" \
  "thc-initial-rank:${THC_INITIAL_RANK:-}" \
  "thc-max-rank:${THC_MAX_RANK:-}" \
  "thc-replacement-validation-atol:${THC_REPLACEMENT_VALIDATION_ATOL:-}" \
  "thc-replacement-validation-rtol:${THC_REPLACEMENT_VALIDATION_RTOL:-}" \
  "fno-nvir-act:${FNO_NVIR_ACT:-}" \
  "fno-pct-occ:${FNO_PCT_OCC:-}"; do
  name="${specification%%:*}"
  value="${specification#*:}"
  if [[ -n "${value}" ]]; then
    CP_ARGUMENTS+=("--${name}" "${value}")
  fi
done

cd "${SOURCE_ROOT}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/topology_guard.py" \
  --mode physical8 \
  --record "${TOPOLOGY_RECORD}" \
  --require-performance -- \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/counterpoise.py" "${CP_ARGUMENTS[@]}"
