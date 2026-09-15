#!/bin/bash
#SBATCH --job-name=rr-engine-smoke
#SBATCH --account=mri
#SBATCH --partition=mrigpu
#SBATCH --nodelist=compute-1-6
#SBATCH --gres=gpu:nvidia:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913/logs/rr-smoke-%j.log
set -euo pipefail

TASKDIR=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913
BASELINE=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913
VENV="$BASELINE/.venv"
SOURCE_ROOT="${CCSD_SOURCE_ROOT:?set CCSD_SOURCE_ROOT to an immutable MTU snapshot source directory}"
case "$SOURCE_ROOT" in
  "$TASKDIR"/snapshots/*/source) ;;
  *)
    echo "CCSD_SOURCE_ROOT must point below $TASKDIR/snapshots/<sha256>/source" >&2
    exit 2
    ;;
esac
export PYTHONDONTWRITEBYTECODE=1
EXPECTED_SHA=$(basename "$(dirname "$SOURCE_ROOT")")
ACTUAL_SHA=$("$VENV/bin/python" - "$SOURCE_ROOT" <<'PY'
import importlib.util
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
module_path = root / "benchmarks/cc/a100_water8/benchmark.py"
spec = importlib.util.spec_from_file_location("rr_smoke_manifest", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(module._source_tree_digest(root)[0])
PY
)
ACTUAL_SHA=$(printf '%s\n' "$ACTUAL_SHA" | tail -n 1)
if [[ ! "$EXPECTED_SHA" =~ ^[0-9a-f]{64}$ || "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "immutable source digest mismatch: $ACTUAL_SHA != $EXPECTED_SHA" >&2
  exit 2
fi
DEPLOYMENT_PROFILE="${CCSD_EXPECTED_DEPLOYMENT_PROFILE:-candidate}"
REQUIRE_CANONICAL_PRISTINE="${CCSD_REQUIRE_CANONICAL_PRISTINE:-0}"
if [[ "$DEPLOYMENT_PROFILE" != "candidate" ]]; then
  echo "run_mtu_rr_smoke.sbatch requires the candidate deployment profile" >&2
  exit 2
fi
case "$REQUIRE_CANONICAL_PRISTINE" in
  0|1) ;;
  *)
    echo "CCSD_REQUIRE_CANONICAL_PRISTINE must be 0 or 1" >&2
    exit 2
    ;;
esac
SNAPSHOT_VALIDATION_ARGS=(
  --source-root "$SOURCE_ROOT"
  --expected-task-root "$TASKDIR"
  --expected-deployment-profile "$DEPLOYMENT_PROFILE"
)
if [[ "$REQUIRE_CANONICAL_PRISTINE" == "1" ]]; then
  SNAPSHOT_VALIDATION_ARGS+=(--require-canonical-pristine)
fi
"$VENV/bin/python" \
  "$SOURCE_ROOT/benchmarks/cc/a100_water8/snapshot_manifest.py" \
  "${SNAPSHOT_VALIDATION_ARGS[@]}" >/dev/null
CUDA_SITE="$VENV/lib/python3.11/site-packages"
CUDA_LIB_DIRS=$(find "$CUDA_SITE/nvidia" "$CUDA_SITE/cutensor" \
  -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="$BASELINE/cuda-loader-lib:$CUDA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$SOURCE_ROOT"
export CUPY_CACHE_DIR="$TASKDIR/.cache/cupy"
export PYSCF_TMPDIR="$TASKDIR/tmp"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export GPU4PYSCF_NUMA=3
export FROZEN_GPU4PYSCF_COMMIT=a89b3ae018d4e82968323ef95645d91adde294aa
export CCSD_TASK_ROOT="$TASKDIR"
export CCSD_EXPECTED_DEPLOYMENT_PROFILE="$DEPLOYMENT_PROFILE"
export CCSD_REQUIRE_CANONICAL_PRISTINE="$REQUIRE_CANONICAL_PRISTINE"
mkdir -p "$CUPY_CACHE_DIR" "$PYSCF_TMPDIR" "$TASKDIR/results" "$TASKDIR/logs"
cd "$SOURCE_ROOT"

hostname
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,pci.bus_id \
  --format=csv
echo "correctness smoke: recording Slurm's actual affinity in result JSON"
RR_RING_KERNEL_VALUE="${RR_RING_KERNEL:-reference}"
case "${RR_RING_KERNEL_VALUE}" in
  reference|gemm) ;;
  *) echo "RR_RING_KERNEL must be reference or gemm" >&2; exit 2 ;;
esac

# The minimal full-rank gate must pass before the capped water2 integration
# diagnostic is attempted.  Each invocation writes a separate auditable JSON.
"$VENV/bin/python" benchmarks/cc/a100_water8/validate_rr_engine.py \
  --case water1-sto3g --water1-convergence \
  --rr-ring-kernel "${RR_RING_KERNEL_VALUE}" \
  --output "$TASKDIR/results/rr-smoke-water1-sto3g-${SLURM_JOB_ID}.json"

"$VENV/bin/python" benchmarks/cc/a100_water8/validate_rr_engine.py \
  --case water2-tz --rr-eig-cutoff 1e-6 --rr-max-rank 32 \
  --rr-ring-kernel "${RR_RING_KERNEL_VALUE}" \
  --output "$TASKDIR/results/rr-smoke-water2-tz-${SLURM_JOB_ID}.json"
