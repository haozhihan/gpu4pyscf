#!/bin/bash
set -euo pipefail

NSYS=/usr/local/cuda/bin/nsys
if [[ ! -x "${NSYS}" ]]; then
  echo "missing nsys at ${NSYS}" >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 OUTPUT_PREFIX [benchmark.py arguments...]" >&2
  exit 2
fi
PREFIX=$1
shift
if [[ -e "${PREFIX}.qdrep" || -e "${PREFIX}.nsys-rep" || \
      -e "${PREFIX}.sqlite" ]]; then
  echo "refusing to overwrite existing Nsight output: ${PREFIX}" >&2
  exit 2
fi
SOURCE_ROOT="${CCSD_SOURCE_ROOT:-}"
if [[ -z "${SOURCE_ROOT}" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  SOURCE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
fi
BENCHMARK="${SOURCE_ROOT}/benchmarks/cc/a100_water8/benchmark.py"
if [[ ! -f "${BENCHMARK}" ]]; then
  echo "invalid CCSD source root: ${SOURCE_ROOT}" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${SOURCE_ROOT}"
exec "${NSYS}" profile --trace=cuda,nvtx,osrt --sample=none --stats=true \
  --force-overwrite=false --output="${PREFIX}" \
  "${PYTHON_BIN}" "${BENCHMARK}" "$@"
