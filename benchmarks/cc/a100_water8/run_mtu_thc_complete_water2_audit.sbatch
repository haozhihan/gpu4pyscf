#!/bin/bash
#SBATCH --job-name=thc-water2-complete-audit
#SBATCH --account=mri
#SBATCH --partition=mrigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --mem-bind=verbose,map_mem:3
#SBATCH --mem=110G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913/logs/thc-complete-water2-audit-%j.log
set -euo pipefail

# This launcher runs the complete Algorithms 1--10 development audit.  A
# performance-qualified topology is required so the receipt-bound selected
# GINT provider can be used. This is not an iterative THC-CCSD energy or performance claim.
# Submit with ``sbatch --nodelist=<qualification-node>``.  The node comes from
# the sealed receipt/release pin; preflight below rejects any different node.
if (( $# != 0 )); then
  echo "run_mtu_thc_complete_water2_audit.sbatch accepts no positional overrides" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run_mtu_thc_complete_water2_audit.sbatch requires a Slurm job" >&2
  exit 2
fi

TASK_ROOT="${CCSD_TASK_ROOT:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913}"
SOURCE_ROOT="${CCSD_SOURCE_ROOT:?set CCSD_SOURCE_ROOT to immutable release snapshot B}"
GINT_RUNTIME_GATE_RECEIPT="${GINT_RUNTIME_GATE_RECEIPT:?set GINT_RUNTIME_GATE_RECEIPT to the sealed qualification receipt}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

canonical_existing_path() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import os
import pathlib
import stat
import sys

value, kind = sys.argv[1:]
path = pathlib.Path(os.path.abspath(os.path.expanduser(value)))
current = pathlib.Path(path.anchor)
for component in path.parts[1:]:
    current /= component
    info = current.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise SystemExit(f"path traverses symbolic link: {current}")
if path.resolve(strict=True) != path:
    raise SystemExit(f"path is not canonical: {path}")
if kind == "directory" and not path.is_dir():
    raise SystemExit(f"path is not a directory: {path}")
if kind == "file" and not path.is_file():
    raise SystemExit(f"path is not a regular file: {path}")
print(path)
PY
}

TASK_ROOT="$(canonical_existing_path "${TASK_ROOT}" directory)"
SOURCE_ROOT="$(canonical_existing_path "${SOURCE_ROOT}" directory)"
GINT_RUNTIME_GATE_RECEIPT="$(
  canonical_existing_path "${GINT_RUNTIME_GATE_RECEIPT}" file
)"
case "${SOURCE_ROOT}" in
  "${TASK_ROOT}"/snapshots/*/source) ;;
  *)
    echo "CCSD_SOURCE_ROOT must be ${TASK_ROOT}/snapshots/<sha256>/source" >&2
    exit 2
    ;;
esac
SOURCE_TREE_SHA256="$(basename "$(dirname "${SOURCE_ROOT}")")"
if [[ ! "${SOURCE_TREE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "release snapshot B is not content-addressed: ${SOURCE_ROOT}" >&2
  exit 2
fi
case "${GINT_RUNTIME_GATE_RECEIPT}/" in
  "${TASK_ROOT}/results/"*) ;;
  *)
    echo "GINT_RUNTIME_GATE_RECEIPT must be below ${TASK_ROOT}/results" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="${SOURCE_ROOT}/benchmarks/cc/a100_water8"
for required in \
  snapshot_manifest.py \
  gint_gate_receipt.py \
  topology_guard.py \
  thc_complete_water2_audit.py; do
  if [[ ! -f "${SCRIPT_DIR}/${required}" || -L "${SCRIPT_DIR}/${required}" ]]; then
    echo "release snapshot B is missing regular ${required}" >&2
    exit 2
  fi
done
RELEASE_PIN_PATH="${SOURCE_ROOT}/gpu4pyscf/cc/gint_release_pin.json"
if [[ ! -f "${RELEASE_PIN_PATH}" || -L "${RELEASE_PIN_PATH}" ]]; then
  echo "release snapshot B is missing its fixed regular-file GINT pin" >&2
  exit 2
fi

DEPLOYMENT_PROFILE="${CCSD_EXPECTED_DEPLOYMENT_PROFILE:-candidate}"
REQUIRE_CANONICAL_PRISTINE="${CCSD_REQUIRE_CANONICAL_PRISTINE:-0}"
if [[ "${DEPLOYMENT_PROFILE}" != "candidate" ]]; then
  echo "complete THC audit requires the candidate deployment profile" >&2
  exit 2
fi
if [[ "${REQUIRE_CANONICAL_PRISTINE}" != "0" ]]; then
  echo "complete THC audit requires CCSD_REQUIRE_CANONICAL_PRISTINE=0" >&2
  exit 2
fi

EXPECTED_SCONTROL_PATH="${TASK_ROOT}/control-tools/slurm-23.02.4-local-rpath-v1/bin/scontrol"
if [[ -n "${CCSD_SCONTROL_PATH:-}" && \
      "${CCSD_SCONTROL_PATH}" != "${EXPECTED_SCONTROL_PATH}" ]]; then
  echo "CCSD_SCONTROL_PATH must be ${EXPECTED_SCONTROL_PATH}" >&2
  exit 2
fi
export CCSD_SCONTROL_PATH="${EXPECTED_SCONTROL_PATH}"
SCONTROL_LIBRARY_DIR="${TASK_ROOT}/control-tools/slurm-23.02.4-local-rpath-v1/lib"

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
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${SOURCE_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CCSD_TASK_ROOT="${TASK_ROOT}"
export CCSD_EXPECTED_DEPLOYMENT_PROFILE="candidate"
export CCSD_REQUIRE_CANONICAL_PRISTINE=0
export FROZEN_GPU4PYSCF_COMMIT=a89b3ae018d4e82968323ef95645d91adde294aa
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export CUPY_CACHE_DIR="${CCSD_CUPY_CACHE_DIR:-${TASK_ROOT}/.cache/cupy}"
export PYSCF_TMPDIR="${CCSD_PYSCF_TMPDIR:-${TASK_ROOT}/tmp}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/snapshot_manifest.py" \
  --source-root "${SOURCE_ROOT}" \
  --expected-task-root "${TASK_ROOT}" \
  --expected-base-revision a89b3ae018d4e82968323ef95645d91adde294aa \
  --expected-deployment-profile candidate >/dev/null

RESULT_ROOT="${TASK_ROOT}/results/thc-complete-water2-audit"
mkdir -p "${RESULT_ROOT}" "${CUPY_CACHE_DIR}" "${PYSCF_TMPDIR}" \
  "${TASK_ROOT}/logs"
RESULT_ROOT="$(canonical_existing_path "${RESULT_ROOT}" directory)"
AUDIT_OUTPUT="${RESULT_ROOT}/water2-thc-complete-audit-${SLURM_JOB_ID}.json"
AUDIT_LOG="${RESULT_ROOT}/water2-thc-complete-audit-${SLURM_JOB_ID}.log"
TOPOLOGY_RECORD="${RESULT_ROOT}/topology-physical8-${SLURM_JOB_ID}.json"
BINDING_RECORD="${RESULT_ROOT}/release-binding-${SLURM_JOB_ID}.json"
RETURN_CODE_RECORD="${RESULT_ROOT}/launcher-return-${SLURM_JOB_ID}.json"
for output_path in \
  "${AUDIT_OUTPUT}" \
  "${AUDIT_LOG}" \
  "${TOPOLOGY_RECORD}" \
  "${BINDING_RECORD}" \
  "${RETURN_CODE_RECORD}"; do
  if [[ -e "${output_path}" || -L "${output_path}" ]]; then
    echo "refusing to overwrite complete THC audit artifact: ${output_path}" >&2
    exit 2
  fi
done

# Rebuild the expected release pin from the external sealed receipt, then
# require byte-bound equality with the fixed pin inside B.  This also validates
# the receipt sidecar, regular/read-only modes, sealed directory, source paths,
# release tree digest, and captured runtime bundle before any GPU work starts.
"${PYTHON_BIN}" - \
  "${SOURCE_ROOT}" \
  "${TASK_ROOT}" \
  "${GINT_RUNTIME_GATE_RECEIPT}" \
  "${BINDING_RECORD}" <<'PY'
import hashlib
import importlib.util
import json
import os
import pathlib
import stat
import sys

source, task, receipt, output = map(pathlib.Path, sys.argv[1:])
write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


receipt_module = load(
    "water2_thc_launcher_receipt",
    source / "benchmarks/cc/a100_water8/gint_gate_receipt.py",
)
manifest_module = load(
    "water2_thc_launcher_manifest",
    source / "benchmarks/cc/a100_water8/snapshot_manifest.py",
)

manifest_path = source.parent / "manifest.json"
manifest_path, manifest_bytes, manifest_info = receipt_module._regular_file(
    manifest_path, name="release snapshot B manifest"
)
if manifest_info.st_mode & write_bits:
    raise SystemExit("release snapshot B manifest must be read-only")
manifest = receipt_module._json_object(
    manifest_bytes, name="release snapshot B manifest"
)
if manifest.get("immutable") is not True:
    raise SystemExit("release snapshot B manifest is not immutable")
if manifest.get("source") != str(source):
    raise SystemExit("release snapshot B manifest source path mismatch")
if (manifest.get("local_provenance") or {}).get("deployment_profile") != "candidate":
    raise SystemExit("release snapshot B is not a candidate deployment")
lineage = manifest.get("gint_release_lineage")
if not isinstance(lineage, dict):
    raise SystemExit("source is not release snapshot B: GINT lineage is absent")

tree_sha256, files_hashed = manifest_module.source_tree_digest(source)
if tree_sha256 != source.parent.name or tree_sha256 != manifest.get("tree_sha256"):
    raise SystemExit("release snapshot B source/tree binding mismatch")

runtime = manifest.get("runtime_binaries")
bundle_id = runtime.get("bundle_id") if isinstance(runtime, dict) else None
if (
    not isinstance(runtime, dict)
    or runtime.get("complete") is not True
    or not isinstance(bundle_id, str)
    or len(bundle_id) != 64
    or any(character not in "0123456789abcdef" for character in bundle_id)
):
    raise SystemExit("release snapshot B runtime bundle binding is invalid")

receipt_path, receipt_bytes, receipt_info = receipt_module._regular_file(
    receipt, name="runtime gate receipt"
)
if receipt_info.st_mode & write_bits:
    raise SystemExit("runtime gate receipt must be read-only")
approved_results = task / "results"
if not receipt_path.is_relative_to(approved_results):
    raise SystemExit("runtime gate receipt is outside the same task results root")
if receipt_path.is_relative_to(source):
    raise SystemExit("runtime gate receipt must be outside release snapshot B")
expected_pin = receipt_module.build_release_pin(receipt_path)
pin_path = source / "gpu4pyscf/cc/gint_release_pin.json"
pin_path, pin_bytes, pin_info = receipt_module._regular_file(
    pin_path, name="release snapshot B GINT pin"
)
if pin_info.st_mode & write_bits:
    raise SystemExit("release snapshot B GINT pin must be read-only")
pin = receipt_module._json_object(pin_bytes, name="release snapshot B GINT pin")
if pin != expected_pin:
    raise SystemExit("release snapshot B pin does not bind the supplied receipt")
pinned_runtime = pin.get("release_runtime_contract") or {}
pinned_node = pinned_runtime.get("node")
pinned_host = pinned_runtime.get("host")
if not isinstance(pinned_node, str) or not pinned_node:
    raise SystemExit("release pin qualification node is unavailable")
if pinned_host != pinned_node:
    raise SystemExit("release pin qualification host/node mismatch")
if os.getenv("SLURM_JOB_NODELIST") != pinned_node:
    raise SystemExit(
        "Slurm node does not match the receipt-pinned qualification node; "
        f"submit with sbatch --nodelist={pinned_node}"
    )
if os.getenv("SLURM_JOB_PARTITION") != "mrigpu":
    raise SystemExit("complete THC audit requires Slurm partition mrigpu")

receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
envelope = receipt_module._json_object(receipt_bytes, name="runtime gate receipt")
payload_sha256 = envelope.get("payload_sha256")
if lineage.get("receipt_sha256") != receipt_sha256:
    raise SystemExit("release lineage receipt file digest mismatch")
if lineage.get("receipt_payload_sha256") != payload_sha256:
    raise SystemExit("release lineage receipt payload digest mismatch")
if lineage.get("release_pin_sha256") != hashlib.sha256(pin_bytes).hexdigest():
    raise SystemExit("release lineage pin digest mismatch")

record = {
    "schema": "gpu4pyscf.thc-complete-water2-release-binding.v1",
    "validated": True,
    "audit_only": True,
    "iterative_thc_energy_evaluated": False,
    "performance_claim_eligible": False,
    "gint_column_backend": "selected",
    "gint_column_kernel": "reference",
    "receipt_source_tree_bundle": {
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "receipt_payload_sha256": payload_sha256,
        "source_root": str(source),
        "release_tree_sha256": tree_sha256,
        "source_files_hashed": files_hashed,
        "runtime_bundle_id": bundle_id,
        "qualification_node": pinned_node,
    },
    "release_manifest_path": str(manifest_path),
    "release_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "release_pin_path": str(pin_path),
    "release_pin_sha256": hashlib.sha256(pin_bytes).hexdigest(),
}
encoded = (
    json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
).encode("utf-8")
descriptor = os.open(
    output,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o444,
)
try:
    view = memoryview(encoded)
    while view:
        written = os.write(descriptor, view)
        if written < 1:
            raise OSError("release binding write made no progress")
        view = view[written:]
    os.fsync(descriptor)
    os.fchmod(descriptor, 0o444)
finally:
    os.close(descriptor)
PY

# These are development-audit controls, not selected production parameters.
AUDIT_ARGUMENTS=(
  --output "${AUDIT_OUTPUT}"
  --log "${AUDIT_LOG}"
  --threads 8
  --max-memory-mb 110000
  --scf-conv-tol 1e-10
  --cc-conv-tol 1e-8
  --cc-conv-tol-normt 1e-6
  --cc-max-cycle 50
  --eri-tol 1e-8
  --rr-eig-cutoff 1e-6
  --direct-scf-tol 1e-14
  --gint-column-backend selected
  --gint-runtime-gate-receipt "${GINT_RUNTIME_GATE_RECEIPT}"
  --gint-column-kernel reference
  --gint-group-size 16
  --gint-max-batch-size 32
  --cd-mo-block-size 32
  --denominator-tolerance 1e-10
  --rr-initial-rank 32
  --rr-solver-tolerance 1e-10
  --rr-dense-fallback-dimension 64
  --rr-ring-kernel reference
  --auxiliary-block-size 32
  --virtual-block-size 32
  --amplitude-fit-tolerances 1e-4 1e-5 1e-6 1e-7
  --amplitude-initial-rank 512
  --amplitude-max-rank 1024
  --eri-candidate 512:1e-4
  --eri-candidate 768:1e-5
  --eri-candidate 1024:1e-6
  --occupation-change-floor 1e-12
  --fit-max-iterations 500
  --als-convergence-tolerance 1e-10
  --fit-ridge 1e-12
  --amplitude-seed 0
  --eri-seed 0
  --orthogonality-cutoff 1e-12
  --orthogonality-tolerance 1e-10
  --algorithm-x-block-size 1
  --exact-identity-atol 1e-9
  --candidate-delta-norm 1e-6
  --candidate-projected-residual-norm 1e-6
  --amplitude-core-symmetry-atol 1e-10
  --hbm-limit-gib 72
  --host-rss-limit-gib 110
  --preflight-reserve-gib 0.5
)

cd "${SOURCE_ROOT}"
set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/topology_guard.py" \
  --mode physical8 \
  --expected-gpu-numa 3 \
  --record "${TOPOLOGY_RECORD}" \
  --require-performance -- \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/thc_complete_water2_audit.py" \
  "${AUDIT_ARGUMENTS[@]}"
AUDIT_RETURN_CODE=$?
set -e

POSTCHECK_RETURN_CODE=0
set +e
"${PYTHON_BIN}" - \
  "${RETURN_CODE_RECORD}" \
  "${BINDING_RECORD}" \
  "${AUDIT_OUTPUT}" \
  "${TOPOLOGY_RECORD}" \
  "${AUDIT_RETURN_CODE}" \
  "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

output, binding_path, audit_path, topology_path = map(pathlib.Path, sys.argv[1:5])
audit_return_code = int(sys.argv[5])
job_id = sys.argv[6]
errors = []
binding = json.loads(binding_path.read_text(encoding="utf-8"))

if audit_return_code == 0:
    if not audit_path.is_file() or audit_path.is_symlink():
        errors.append("successful audit command did not create a regular result")
        audit = {}
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_binding = binding["receipt_source_tree_bundle"]
    source = audit.get("source") or {}
    snapshot = source.get("snapshot") or {}
    manifest = snapshot.get("manifest") or {}
    controls = audit.get("controls") or {}
    provider_gate = (audit.get("rr_run") or {}).get("provider_runtime_gate") or {}
    checks = {
        "completed schema": (
            audit.get("schema") == "gpu4pyscf.water2-thc-complete-audit.v2"
            and audit.get("status") == "completed"
        ),
        "fixed WATER2 case": audit.get("case_id") == "water2-tz",
        "release source": source.get("repository") == expected_binding["source_root"],
        "release tree": source.get("tree_sha256") == expected_binding["release_tree_sha256"],
        "runtime bundle": (
            (manifest.get("runtime_binaries") or {}).get("bundle_id")
            == expected_binding["runtime_bundle_id"]
        ),
        "receipt": (
            controls.get("gint_runtime_gate_receipt")
            == expected_binding["receipt_path"]
        ),
        "selected reference GINT": (
            controls.get("gint_column_backend") == "selected"
            and controls.get("gint_column_kernel") == "reference"
        ),
        "provider receipt gate": (
            provider_gate.get("validated") is True
            and provider_gate.get("execution_mode")
            == "consumer-thc-complete-audit"
        ),
        "full-pair anchor": (
            ((audit.get("audit") or {}).get("exact_anchor") or {}).get("stage")
            == "exact-anchor"
        ),
        "staged inexact amplitude scan": len(
            (audit.get("audit") or {}).get("amplitude_scan") or []
        ) == 4,
    }
    errors.extend(name for name, passed in checks.items() if not passed)

    fail_closed_keys = {
        "accepted",
        "production_enabled",
        "performance_eligible",
        "formal_validation_eligible",
        "complete_validated",
        "complete_ccsd_residual",
        "formal_full_residual_eligible",
    }

    def walk(value, path="result"):
        if isinstance(value, dict):
            is_audit_result = (
                "accepted" in value or "production_enabled" in value
            )
            for key, item in value.items():
                current = f"{path}.{key}"
                if is_audit_result and key in fail_closed_keys and item is not False:
                    errors.append(f"{current} is not false")
                walk(item, current)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(audit)

if audit_return_code != 0:
    errors.append(f"audit command returned {audit_return_code}")
launcher_return_code = (
    audit_return_code if audit_return_code != 0 else (3 if errors else 0)
)
record = {
    "schema": "gpu4pyscf.thc-complete-water2-launcher-return.v1",
    "slurm_job_id": job_id,
    "audit_return_code": audit_return_code,
    "launcher_return_code": launcher_return_code,
    "postconditions_passed": not errors,
    "postcondition_errors": errors,
    "audit_only": True,
    "iterative_thc_energy_evaluated": False,
    "performance_claim_eligible": False,
    "gint_column_backend": "selected",
    "gint_column_kernel": "reference",
    "release_binding": binding,
    "audit_output": str(audit_path),
    "audit_output_sha256": (
        hashlib.sha256(audit_path.read_bytes()).hexdigest()
        if audit_path.is_file() else None
    ),
    "topology_record": str(topology_path),
    "topology_record_sha256": (
        hashlib.sha256(topology_path.read_bytes()).hexdigest()
        if topology_path.is_file() else None
    ),
}
encoded = (
    json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
).encode("utf-8")
descriptor = os.open(
    output,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o444,
)
try:
    view = memoryview(encoded)
    while view:
        written = os.write(descriptor, view)
        if written < 1:
            raise OSError("launcher return write made no progress")
        view = view[written:]
    os.fsync(descriptor)
    os.fchmod(descriptor, 0o444)
finally:
    os.close(descriptor)
raise SystemExit(0 if not errors else 3)
PY
POSTCHECK_RETURN_CODE=$?
set -e

if (( AUDIT_RETURN_CODE != 0 )); then
  exit "${AUDIT_RETURN_CODE}"
fi
exit "${POSTCHECK_RETURN_CODE}"
