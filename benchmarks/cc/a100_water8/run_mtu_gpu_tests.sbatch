#!/bin/bash
#SBATCH --job-name=rrccsd-gpu-tests
#SBATCH --account=mri
#SBATCH --partition=mrigpu
#SBATCH --nodelist=compute-1-6
#SBATCH --gres=gpu:nvidia:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913/logs/gpu-tests-%j.log
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
if [[ ! -f "$SOURCE_ROOT/benchmarks/cc/a100_water8/benchmark.py" ]]; then
  echo "invalid CCSD_SOURCE_ROOT: $SOURCE_ROOT" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
DEPLOYMENT_PROFILE="${CCSD_EXPECTED_DEPLOYMENT_PROFILE:-candidate}"
REQUIRE_CANONICAL_PRISTINE="${CCSD_REQUIRE_CANONICAL_PRISTINE:-0}"
if [[ "$DEPLOYMENT_PROFILE" != "candidate" ]]; then
  echo "run_mtu_gpu_tests.sbatch requires the candidate deployment profile" >&2
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
CUDA_LIB_DIRS=$(find "$CUDA_SITE/nvidia" "$CUDA_SITE/cutensor" -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="$BASELINE/cuda-loader-lib:$CUDA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$SOURCE_ROOT"
export CUPY_CACHE_DIR="$TASKDIR/.cache/cupy"
export PYSCF_TMPDIR="$TASKDIR/tmp"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export CCSD_TASK_ROOT="$TASKDIR"
export CCSD_EXPECTED_DEPLOYMENT_PROFILE="$DEPLOYMENT_PROFILE"
export CCSD_REQUIRE_CANONICAL_PRISTINE="$REQUIRE_CANONICAL_PRISTINE"
mkdir -p "$CUPY_CACHE_DIR" "$PYSCF_TMPDIR" "$TASKDIR/results"
cd "$SOURCE_ROOT"

hostname
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,pci.bus_id --format=csv
"$VENV/bin/python" -c 'import cupy as cp; import gpu4pyscf; print(gpu4pyscf.__file__); print(cp.arange(8).sum())'
shopt -s nullglob
rrccsd_tests=(gpu4pyscf/cc/tests/test_rrccsd*.py)
"$VENV/bin/python" -m pytest -q \
  gpu4pyscf/cc/tests/test_ccsd.py \
  gpu4pyscf/cc/tests/test_device_runtime.py \
  gpu4pyscf/cc/tests/test_ccsd_residency.py \
  gpu4pyscf/cc/tests/test_compressed_diis.py \
  gpu4pyscf/cc/tests/test_direct_cd.py \
  gpu4pyscf/cc/tests/test_gint_pair_columns.py \
  gpu4pyscf/cc/tests/test_gint_selected_columns.py \
  gpu4pyscf/cc/tests/test_integrals.py \
  gpu4pyscf/cc/tests/test_lowrank.py \
  gpu4pyscf/cc/tests/test_rr_projector.py \
  gpu4pyscf/cc/tests/test_rr_residual.py \
  gpu4pyscf/cc/tests/test_residual_evaluator.py \
  gpu4pyscf/cc/tests/test_rr_engine.py \
  gpu4pyscf/cc/tests/test_full_space_residual.py \
  gpu4pyscf/cc/tests/test_thc_engine.py \
  gpu4pyscf/cc/tests/test_thc_eri.py \
  gpu4pyscf/cc/tests/test_thc_eri_factorization.py \
  gpu4pyscf/cc/tests/test_thc_eri_preprocess.py \
  gpu4pyscf/cc/tests/test_thc_factorization.py \
  gpu4pyscf/cc/tests/test_thc_mp2_weights.py \
  gpu4pyscf/cc/tests/test_thc_omega_d.py \
  gpu4pyscf/cc/tests/test_thc_omega_ghi.py \
  gpu4pyscf/cc/tests/test_thc_residual.py \
  gpu4pyscf/cc/tests/test_thc_rrccsd_cd.py \
  gpu4pyscf/cc/tests/test_addons.py \
  gpu4pyscf/cc/tests/test_lowrank_ccsd.py \
  "${rrccsd_tests[@]}"
"$VENV/bin/python" -m pytest -q benchmarks/cc/a100_water8/tests
"$VENV/bin/python" benchmarks/cc/a100_water8/self_check.py
