#!/bin/bash
#SBATCH --job-name=gint-water2-gate
#SBATCH --account=mri
#SBATCH --partition=mrigpu
#SBATCH --gres=gpu:nvidia:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem-bind=verbose,map_mem:3
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913/logs/gint-gate-%j.log
set -euo pipefail

TASK_ROOT="${CCSD_TASK_ROOT:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913}"
SOURCE_ROOT="${CCSD_SOURCE_ROOT:?set CCSD_SOURCE_ROOT to an immutable MTU snapshot source directory}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run_mtu_gint_gate.sbatch requires a Slurm job" >&2
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
if [[ ! -f "${SCRIPT_DIR}/gint_gate.py" || \
      ! -f "${SCRIPT_DIR}/gint_gate_receipt.py" || \
      ! -f "${SCRIPT_DIR}/snapshot_manifest.py" || \
      ! -f "${SCRIPT_DIR}/topology_guard.py" ]]; then
  echo "snapshot is missing the GINT gate or topology guard" >&2
  exit 2
fi

DEPLOYMENT_PROFILE="${CCSD_EXPECTED_DEPLOYMENT_PROFILE:-candidate}"
REQUIRE_CANONICAL_PRISTINE="${CCSD_REQUIRE_CANONICAL_PRISTINE:-0}"
COLUMN_KERNEL="${GINT_COLUMN_KERNEL:-reference}"
if [[ "${DEPLOYMENT_PROFILE}" != "candidate" ]]; then
  echo "run_mtu_gint_gate.sbatch requires the candidate deployment profile" >&2
  exit 2
fi
case "${REQUIRE_CANONICAL_PRISTINE}" in
  0|1) ;;
  *)
    echo "CCSD_REQUIRE_CANONICAL_PRISTINE must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${COLUMN_KERNEL}" in
  reference|grouped) ;;
  *)
    echo "GINT_COLUMN_KERNEL must be reference or grouped" >&2
    exit 2
    ;;
esac
SNAPSHOT_VALIDATION_ARGS=(
  --source-root "${SOURCE_ROOT}"
  --expected-task-root "${TASK_ROOT}"
  --expected-base-revision "a89b3ae018d4e82968323ef95645d91adde294aa"
  --expected-deployment-profile "${DEPLOYMENT_PROFILE}"
)
if [[ "${REQUIRE_CANONICAL_PRISTINE}" == "1" ]]; then
  SNAPSHOT_VALIDATION_ARGS+=(--require-canonical-pristine)
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/snapshot_manifest.py" \
  "${SNAPSHOT_VALIDATION_ARGS[@]}" >/dev/null

RESULT_ROOT="${TASK_ROOT}/results/gint-gate"
if [[ -n "${GINT_GATE_RESULT_ROOT:-}" ]]; then
  REQUESTED_RESULT_ROOT="$("${PYTHON_BIN}" - "${GINT_GATE_RESULT_ROOT}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
  )"
  if [[ "${REQUESTED_RESULT_ROOT}" != "${RESULT_ROOT}" ]]; then
    echo "GINT_GATE_RESULT_ROOT must be the fixed path ${RESULT_ROOT}" >&2
    exit 2
  fi
fi

export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CCSD_TASK_ROOT="${TASK_ROOT}"
export CCSD_EXPECTED_DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE}"
export CCSD_REQUIRE_CANONICAL_PRISTINE="${REQUIRE_CANONICAL_PRISTINE}"
export CUPY_CACHE_DIR="${CCSD_CUPY_CACHE_DIR:-${TASK_ROOT}/.cache/cupy}"
export PYSCF_TMPDIR="${CCSD_PYSCF_TMPDIR:-${TASK_ROOT}/tmp}"
export FROZEN_GPU4PYSCF_COMMIT="a89b3ae018d4e82968323ef95645d91adde294aa"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores

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
  CUDA_LIB_DIRS="$(find "${CUDA_LIBRARY_ROOTS[@]}" -maxdepth 2 \
    -type d -name lib -print | paste -sd: -)"
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
mkdir -p "${RESULT_ROOT}" "${CUPY_CACHE_DIR}" "${PYSCF_TMPDIR}" \
  "${TASK_ROOT}/logs"

TOPOLOGY_RECORD="${RESULT_ROOT}/topology-physical8-${SLURM_JOB_ID}.json"
GATE_OUTPUT="${RESULT_ROOT}/water2-gint-direct-cd-${SLURM_JOB_ID}.json"
if [[ -e "${TOPOLOGY_RECORD}" || -e "${GATE_OUTPUT}" ]]; then
  echo "refusing to overwrite an existing GINT gate artifact" >&2
  exit 2
fi

GATE_ARGUMENTS=(
  --output "${GATE_OUTPUT}"
  --topology-record "${TOPOLOGY_RECORD}"
  --eri-tol "${ERI_TOL:-1e-8}"
  --direct-scf-tol "${DIRECT_SCF_TOL:-1e-14}"
  --selected-column-tolerance "${COLUMN_TOLERANCE:-2e-10}"
  --diagonal-tolerance "${DIAGONAL_TOLERANCE:-2e-10}"
  --group-size "${GINT_GROUP_SIZE:-16}"
  --column-kernel "${COLUMN_KERNEL}"
  --max-block-bytes "${GINT_MAX_BLOCK_BYTES:-268435456}"
)
if [[ -n "${GINT_MAX_RANK:-}" ]]; then
  GATE_ARGUMENTS+=(--max-rank "${GINT_MAX_RANK}")
fi
if [[ -n "${GINT_RUNTIME_GATE_RECEIPT:-}" ]]; then
  if [[ "${COLUMN_KERNEL}" != "reference" ]]; then
    echo "grouped selected-column prototype cannot consume a release receipt" >&2
    exit 2
  fi
  RELEASE_PIN_PATH="${SOURCE_ROOT}/gpu4pyscf/cc/gint_release_pin.json"
  if [[ ! -f "${RELEASE_PIN_PATH}" || -L "${RELEASE_PIN_PATH}" ]]; then
    echo "release snapshot B is missing its fixed regular-file GINT pin" >&2
    exit 2
  fi
  RECEIPT_PAYLOAD_SHA="$("${PYTHON_BIN}" - "${GINT_RUNTIME_GATE_RECEIPT}" <<'PY'
import json
import pathlib
import sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
digest = value.get("payload_sha256")
if not isinstance(digest, str) or len(digest) != 64:
    raise SystemExit("runtime receipt has no valid payload_sha256")
print(digest)
PY
)"
  EXPECTED_RELEASE_JOB_NAME="gint-release-${RECEIPT_PAYLOAD_SHA}"
  if [[ "${SLURM_JOB_NAME:-}" != "${EXPECTED_RELEASE_JOB_NAME}" ]]; then
    echo "release job name must bind receipt payload: ${EXPECTED_RELEASE_JOB_NAME}" >&2
    exit 2
  fi
  GATE_ARGUMENTS+=(--runtime-gate-receipt "${GINT_RUNTIME_GATE_RECEIPT}")
else
  if [[ -e "${SOURCE_ROOT}/gpu4pyscf/cc/gint_release_pin.json" || \
        -L "${SOURCE_ROOT}/gpu4pyscf/cc/gint_release_pin.json" ]]; then
    echo "qualification snapshot A must not contain a GINT release pin" >&2
    exit 2
  fi
  if [[ "${SLURM_JOB_NAME:-}" != "gint-water2-gate" ]]; then
    echo "qualification job name must be gint-water2-gate" >&2
    exit 2
  fi
fi

cd "${SOURCE_ROOT}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/topology_guard.py" \
  --mode physical8 \
  --record "${TOPOLOGY_RECORD}" \
  --require-performance -- \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/gint_gate.py" "${GATE_ARGUMENTS[@]}"
