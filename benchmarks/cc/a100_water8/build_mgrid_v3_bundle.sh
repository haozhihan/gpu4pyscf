#!/usr/bin/env bash
# Build libmgrid_v3.so outside the frozen source and assemble a curated bundle.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-${REPOSITORY_ROOT}}"
BASE_BINARY_ROOT="${BASE_BINARY_ROOT:?set BASE_BINARY_ROOT to the pinned 12-library bundle}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new curated bundle path}"
FROZEN_BASE_REVISION="${FROZEN_BASE_REVISION:-a89b3ae018d4e82968323ef95645d91adde294aa}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-80-real}"
BUILD_JOBS="${BUILD_JOBS:-8}"
BUILD_PARENT="${BUILD_PARENT:-${TMPDIR:-/tmp}}"

SOURCE_ROOT="$(cd -- "${SOURCE_ROOT}" && pwd)"
BASE_BINARY_ROOT="$(cd -- "${BASE_BINARY_ROOT}" && pwd)"
OUTPUT_PARENT="$(cd -- "$(dirname -- "${OUTPUT_ROOT}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_PARENT}/$(basename -- "${OUTPUT_ROOT}")"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "OUTPUT_ROOT already exists: ${OUTPUT_ROOT}" >&2
  exit 2
fi

HEAD_REVISION="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
if [[ "${HEAD_REVISION}" != "${FROZEN_BASE_REVISION}" ]]; then
  echo "source HEAD is not the frozen revision: ${HEAD_REVISION}" >&2
  exit 2
fi
if ! git -C "${SOURCE_ROOT}" diff --quiet \
    || ! git -C "${SOURCE_ROOT}" diff --cached --quiet; then
  echo "frozen source has tracked or staged changes" >&2
  exit 2
fi

WORK_ROOT="$(mktemp -d "${BUILD_PARENT%/}/gpu4pyscf-mgrid-v3.XXXXXX")"
BUNDLE_STAGING="${OUTPUT_PARENT}/.$(basename -- "${OUTPUT_ROOT}").staging-$$"
cleanup() {
  rm -rf "${WORK_ROOT}" "${BUNDLE_STAGING}"
}
trap cleanup EXIT
mkdir -p "${WORK_ROOT}/source" "${WORK_ROOT}/build" "${BUNDLE_STAGING}"

# CMake targets write into PROJECT_SOURCE_DIR, so compile an external source copy.
rsync -a --delete "${SOURCE_ROOT}/gpu4pyscf/lib/" "${WORK_ROOT}/source/"
cmake -S "${WORK_ROOT}/source" -B "${WORK_ROOT}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF -DBUILD_SOLVENT=OFF \
  -DENABLE_MULTIGRID_V2=OFF
cmake --build "${WORK_ROOT}/build" --target mgrid_v3 --parallel "${BUILD_JOBS}"

python - "${SOURCE_ROOT}" "${BASE_BINARY_ROOT}" \
  "${WORK_ROOT}/source/libmgrid_v3.so" "${BUNDLE_STAGING}" \
  "${HEAD_REVISION}" "${CUDA_ARCHITECTURES}" <<'PY'
import datetime
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys

source = pathlib.Path(sys.argv[1]).resolve()
base = pathlib.Path(sys.argv[2]).resolve()
built_mgrid = pathlib.Path(sys.argv[3]).resolve()
output = pathlib.Path(sys.argv[4]).resolve()
head = sys.argv[5]
cuda_architectures = sys.argv[6]
module_path = source / "benchmarks/cc/a100_water8/snapshot_manifest.py"
spec = importlib.util.spec_from_file_location("water8_snapshot_manifest", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
required = module.REQUIRED_RUNTIME_LIBRARIES
if module.discover_required_runtime_libraries(source) != required:
    raise RuntimeError("source runtime requirements differ from the declared closure")
if not built_mgrid.is_file():
    raise RuntimeError(f"mgrid_v3 build did not produce {built_mgrid}")

for name in required:
    origin = built_mgrid if name == "libmgrid_v3.so" else base / name
    if not origin.is_file() or origin.is_symlink():
        raise RuntimeError(f"required runtime library is missing or not regular: {origin}")
    shutil.copy2(origin, output / name)

def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

files = [
    {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
    for path in sorted(output.glob("*.so"))
]
inventory = [entry["name"] for entry in files]
if inventory != list(required):
    raise RuntimeError(f"curated bundle inventory mismatch: {inventory!r}")
manifest = {
    "schema": "gpu4pyscf.runtime-bundle.v1",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_root": str(source),
    "source_head_revision": head,
    "base_binary_root": str(base),
    "cuda_architectures": cuda_architectures,
    "cmake_version": subprocess.run(
        ["cmake", "--version"], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()[0],
    "nvcc_version": subprocess.run(
        ["nvcc", "--version"], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()[-1],
    "required_library_names": list(required),
    "inventory_library_names": inventory,
    "closure_complete": True,
    "files": files,
}
(output / "bundle-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

mv "${BUNDLE_STAGING}" "${OUTPUT_ROOT}"
trap - EXIT
rm -rf "${WORK_ROOT}"
echo "${OUTPUT_ROOT}"
