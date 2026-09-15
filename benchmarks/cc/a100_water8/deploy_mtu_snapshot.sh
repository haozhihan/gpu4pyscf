#!/usr/bin/env bash
# Deploy the current worktree as a content-addressed, read-only MTU snapshot.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MTU_HOST_NAME="${MTU_HOST_NAME:-mtu}"
MTU_TASK_ROOT="${MTU_TASK_ROOT:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913}"
MTU_PYTHON_BIN="${MTU_PYTHON_BIN:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913/.venv/bin/python}"
MTU_GPU4PYSCF_BINARY_ROOT="${MTU_GPU4PYSCF_BINARY_ROOT:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913/.venv/lib/python3.11/site-packages/gpu4pyscf/lib}"
FROZEN_BASE_REVISION="${FROZEN_BASE_REVISION:-a89b3ae018d4e82968323ef95645d91adde294aa}"
DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE:-candidate}"
SSH_CONTROL_PATH="${TMPDIR:-/tmp}/agentqc-mtu-ssh-$$"
SSH_OPTIONS=(
  -o ConnectTimeout=20
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=5
  -o ControlMaster=auto
  -o ControlPersist=300
  -o ControlPath="${SSH_CONTROL_PATH}"
)
RSYNC_RSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=5 -o ServerAliveCountMax=5 -o ControlMaster=auto -o ControlPersist=300 -o ControlPath=${SSH_CONTROL_PATH}"
SSH_MASTER_STARTED=0

case "${DEPLOYMENT_PROFILE}" in
  candidate|g0-canonical-pristine) ;;
  *) echo "unsupported DEPLOYMENT_PROFILE: ${DEPLOYMENT_PROFILE}" >&2; exit 2 ;;
esac

PROVENANCE_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-provenance.XXXXXX.json")"
PROVENANCE_CHECK_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-provenance-check.XXXXXX.json")"
STATUS_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-status.XXXXXX")"
STATUS_CHECK_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-status-check.XXXXXX")"
STAGING_ROOT=""

cleanup() {
  rm -f "${PROVENANCE_FILE}" "${PROVENANCE_CHECK_FILE}" \
    "${STATUS_FILE}" "${STATUS_CHECK_FILE}"
  if [[ -n "${STAGING_ROOT}" && "${SSH_MASTER_STARTED}" == "1" ]]; then
    ssh "${SSH_OPTIONS[@]}" "${MTU_HOST_NAME}" \
      chmod -R u+w "${STAGING_ROOT}" >/dev/null 2>&1 || true
    ssh "${SSH_OPTIONS[@]}" "${MTU_HOST_NAME}" \
      rm -rf "${STAGING_ROOT}" >/dev/null 2>&1 || true
  fi
  if [[ "${SSH_MASTER_STARTED}" == "1" ]]; then
    ssh "${SSH_OPTIONS[@]}" -O exit "${MTU_HOST_NAME}" >/dev/null 2>&1 || true
  fi
  rm -f "${SSH_CONTROL_PATH}"
}
trap cleanup EXIT

start_remote_session() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if ssh "${SSH_OPTIONS[@]}" -Nf "${MTU_HOST_NAME}"; then
      SSH_MASTER_STARTED=1
      return 0
    fi
    sleep "$((attempt * 5))"
  done
  echo "unable to establish a persistent SSH session to ${MTU_HOST_NAME}" >&2
  return 1
}

remote_ssh() {
  ssh "${SSH_OPTIONS[@]}" "${MTU_HOST_NAME}" "$@"
}

write_local_provenance() {
  local status_path="$1"
  local output_path="$2"
  LC_ALL=C git -C "${REPOSITORY_ROOT}" -c core.quotePath=true \
    status --porcelain=v1 --untracked-files=all >"${status_path}"
  python - "${REPOSITORY_ROOT}" "${FROZEN_BASE_REVISION}" \
    "${DEPLOYMENT_PROFILE}" "${status_path}" "${output_path}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
frozen = sys.argv[2]
profile = sys.argv[3]
status_path = Path(sys.argv[4])
output_path = Path(sys.argv[5])

def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=check,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

head = git("rev-parse", "HEAD").stdout.decode("ascii").strip()
tracked_diff_pristine = (
    git("diff", "--quiet", check=False).returncode == 0
    and git("diff", "--cached", "--quiet", check=False).returncode == 0
)
status_bytes = status_path.read_bytes()
try:
    status_text = status_bytes.decode("utf-8")
except UnicodeDecodeError as exc:
    raise SystemExit(f"git status is not valid UTF-8: {exc}")
entries = status_text.splitlines()
allowed = {
    "prefixes": ["benchmarks/cc/a100_water8/"],
    "files": ["gpu4pyscf/cc/device_runtime.py"],
}
bad_tracked_entries = []
bad_untracked_entries = []
for entry in entries:
    if len(entry) < 4:
        bad_tracked_entries.append(entry)
        continue
    status = entry[:2]
    path = entry[3:]
    if status != "??":
        bad_tracked_entries.append(entry)
    elif not (
        path.startswith(allowed["prefixes"][0]) or path in allowed["files"]
    ):
        bad_untracked_entries.append(entry)

tracked_pristine = tracked_diff_pristine and not bad_tracked_entries
untracked_paths_allowed = not bad_untracked_entries
head_matches_frozen = head == frozen
canonical_pristine = bool(
    profile == "g0-canonical-pristine"
    and head_matches_frozen and tracked_pristine and untracked_paths_allowed
)
if profile == "g0-canonical-pristine":
    failures = []
    if not head_matches_frozen:
        failures.append(f"HEAD {head} != frozen revision {frozen}")
    if not tracked_pristine:
        failures.append("tracked worktree or index is not pristine")
    if bad_tracked_entries or bad_untracked_entries:
        failures.append(
            "status violates the G0 pristine/allowlist contract: "
            + "; ".join(bad_tracked_entries + bad_untracked_entries)
        )
    if failures:
        raise SystemExit("G0 provenance check failed: " + " | ".join(failures))

policy_bytes = json.dumps(
    {"allowed_untracked": allowed, "status": entries},
    sort_keys=True, separators=(",", ":"),
).encode("utf-8")
provenance = {
    "schema": "gpu4pyscf.local-provenance.v1",
    "deployment_profile": profile,
    "repository_root": str(root),
    "head_revision": head,
    "frozen_base_revision": frozen,
    "head_matches_frozen_base": head_matches_frozen,
    "tracked_pristine": tracked_pristine,
    "untracked_paths_allowed": untracked_paths_allowed,
    "canonical_pristine": canonical_pristine,
    "allowed_untracked": allowed,
    "git_status": {
        "format": "porcelain-v1-lines",
        "sha256": hashlib.sha256(status_bytes).hexdigest(),
        "allowed_paths_status_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "entry_count": len(entries),
        "entries": entries,
    },
}
output_path.write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

write_local_provenance "${STATUS_FILE}" "${PROVENANCE_FILE}"

REQUIRED_RUNTIME_JSON="$({ python - "${REPOSITORY_ROOT}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
module_path = root / "benchmarks/cc/a100_water8/snapshot_manifest.py"
spec = importlib.util.spec_from_file_location("water8_snapshot_manifest", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
discovered = module.discover_required_runtime_libraries(root)
if discovered != module.REQUIRED_RUNTIME_LIBRARIES:
    raise SystemExit(
        "source load_library targets differ from REQUIRED_RUNTIME_LIBRARIES: "
        f"{discovered!r} != {module.REQUIRED_RUNTIME_LIBRARIES!r}"
    )
print(json.dumps(list(discovered), separators=(",", ":")))
PY
} | tail -n 1)"
REQUIRED_RUNTIME_B64="$(printf '%s' "${REQUIRED_RUNTIME_JSON}" | base64 | tr -d '\n')"

TREE_SHA256="$({ python - "${REPOSITORY_ROOT}" <<'PY'
import importlib.util
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
module_path = root / "benchmarks/cc/a100_water8/snapshot_manifest.py"
spec = importlib.util.spec_from_file_location("water8_snapshot_manifest", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(module.source_tree_digest(root)[0])
PY
} | tail -n 1)"
if [[ ! "${TREE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "failed to compute a source tree SHA-256" >&2
  exit 2
fi

SNAPSHOT_ROOT="${MTU_TASK_ROOT}/snapshots/${TREE_SHA256}"
STAGING_ROOT="${MTU_TASK_ROOT}/snapshots/.staging-${TREE_SHA256}-$$"

verify_existing_snapshot() {
  # macOS ships Bash 3.2, where expanding a declared empty array under
  # ``set -u`` raises "unbound variable".  Use an optional scalar so both the
  # candidate and pristine profiles can validate from the developer host.
  local pristine_argument=""
  if [[ "${DEPLOYMENT_PROFILE}" == "g0-canonical-pristine" ]]; then
    pristine_argument="--require-canonical-pristine"
  fi
  remote_ssh "${MTU_PYTHON_BIN}" \
    "${SNAPSHOT_ROOT}/source/benchmarks/cc/a100_water8/snapshot_manifest.py" \
    --source-root "${SNAPSHOT_ROOT}/source" \
    --expected-task-root "${MTU_TASK_ROOT}" \
    --expected-base-revision "${FROZEN_BASE_REVISION}" \
    --expected-deployment-profile "${DEPLOYMENT_PROFILE}" \
    ${pristine_argument:+"${pristine_argument}"} >/dev/null

  # The source-tree digest deliberately excludes shared libraries.  Refuse to
  # reuse an existing source-addressed snapshot unless the runtime bundle
  # requested by this invocation is byte-identical to the one sealed into the
  # snapshot.  Otherwise a rebuilt libgint.so could silently inherit evidence
  # produced by an older binary under the same source digest.
  remote_ssh "${MTU_PYTHON_BIN}" - \
    "${SNAPSHOT_ROOT}/manifest.json" \
    "${SNAPSHOT_ROOT}/source" \
    "${MTU_GPU4PYSCF_BINARY_ROOT}" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1]).resolve()
source_root = pathlib.Path(sys.argv[2]).resolve()
requested_root = pathlib.Path(sys.argv[3]).resolve()

def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
runtime = manifest.get("runtime_binaries")
if not isinstance(runtime, dict) or runtime.get("complete") is not True:
    raise SystemExit("snapshot runtime-binary manifest is incomplete")
entries = runtime.get("files")
if not isinstance(entries, list) or not entries:
    raise SystemExit("snapshot runtime-binary manifest has no files")
if not requested_root.is_dir():
    raise SystemExit(f"requested runtime root is missing: {requested_root}")

sealed = {}
for entry in entries:
    if not isinstance(entry, dict):
        raise SystemExit("snapshot runtime-binary entry is invalid")
    relative = pathlib.Path(str(entry.get("relative_path", "")))
    name = relative.name
    if not name.endswith(".so") or name in sealed:
        raise SystemExit(f"snapshot runtime-binary inventory is invalid: {name}")
    sealed_path = source_root / relative
    if not sealed_path.is_file():
        raise SystemExit(f"sealed runtime binary is missing: {relative}")
    sealed[name] = {
        "bytes": int(entry.get("bytes", -1)),
        "sha256": entry.get("sha256"),
    }

requested_paths = {
    path.name: path for path in requested_root.glob("*.so") if path.is_file()
}
if set(requested_paths) != set(sealed):
    missing = sorted(set(sealed) - set(requested_paths))
    extra = sorted(set(requested_paths) - set(sealed))
    raise SystemExit(
        "requested runtime inventory differs from sealed snapshot: "
        f"missing={missing}, extra={extra}"
    )
for name, expected in sealed.items():
    path = requested_paths[name]
    actual_size = path.stat().st_size
    actual_sha256 = file_sha256(path)
    if actual_size != expected["bytes"] or actual_sha256 != expected["sha256"]:
        raise SystemExit(
            f"requested runtime binary differs from sealed snapshot: {name}"
        )
PY
}

start_remote_session

if remote_ssh test -e "${SNAPSHOT_ROOT}"; then
  verify_existing_snapshot
  echo "${SNAPSHOT_ROOT}/source"
  exit 0
fi

remote_ssh mkdir -p "${STAGING_ROOT}/source"
rsync -az --delete -e "${RSYNC_RSH}" \
  --exclude='.git' --exclude='.pytest_cache/' --exclude='__pycache__/' \
  --exclude='build/' --exclude='dist/' --exclude='results/' --exclude='*.pyc' \
  "${REPOSITORY_ROOT}/" "${MTU_HOST_NAME}:${STAGING_ROOT}/source/"
rsync -az -e "${RSYNC_RSH}" "${PROVENANCE_FILE}" \
  "${MTU_HOST_NAME}:${STAGING_ROOT}/local-provenance.json"

# Prove that neither the Git state nor its G0 allowlist changed during upload.
write_local_provenance "${STATUS_CHECK_FILE}" "${PROVENANCE_CHECK_FILE}"
if ! cmp -s "${PROVENANCE_FILE}" "${PROVENANCE_CHECK_FILE}"; then
  echo "local Git provenance changed during snapshot upload" >&2
  exit 3
fi

remote_ssh "${MTU_PYTHON_BIN}" - \
  "${STAGING_ROOT}" "${SNAPSHOT_ROOT}" "${TREE_SHA256}" \
  "${FROZEN_BASE_REVISION}" "${MTU_GPU4PYSCF_BINARY_ROOT}" \
  "${REQUIRED_RUNTIME_B64}" <<'PY'
import base64
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import stat
import sys

staging = pathlib.Path(sys.argv[1]).resolve()
target = pathlib.Path(sys.argv[2]).resolve()
tree_sha256 = sys.argv[3]
base_revision = sys.argv[4]
runtime_binary_source = pathlib.Path(sys.argv[5]).resolve()
required_library_names = json.loads(base64.b64decode(sys.argv[6]).decode("utf-8"))
source = staging / "source"
provenance_path = staging / "local-provenance.json"
suffixes = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".ini",
    ".json", ".md", ".py", ".pyx", ".pxd", ".sbatch", ".sh", ".toml",
    ".txt", ".yaml", ".yml",
}
excluded = {".git", ".pytest_cache", "__pycache__", "build", "dist", "results"}
write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH

def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def source_digest(root: pathlib.Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        if path.suffix.lower() not in suffixes:
            continue
        if not path.resolve().is_relative_to(root):
            raise RuntimeError(f"source symlink leaves staging root: {path}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count

if target.exists():
    raise RuntimeError(f"snapshot appeared concurrently: {target}")
if not source.is_dir() or not provenance_path.is_file():
    raise RuntimeError("snapshot staging tree is incomplete")
provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
if provenance.get("frozen_base_revision") != base_revision:
    raise RuntimeError("local provenance base revision mismatch")

runtime_target = source / "gpu4pyscf/lib"
binaries = sorted(runtime_binary_source.glob("*.so"))
if not binaries:
    raise RuntimeError(f"no GPU4PySCF runtime binaries found under {runtime_binary_source}")
available_library_names = {path.name for path in binaries if path.is_file()}
missing_libraries = sorted(set(required_library_names) - available_library_names)
if missing_libraries:
    raise RuntimeError(
        "GPU4PySCF runtime bundle is missing required libraries: "
        + ", ".join(missing_libraries)
    )
for existing in source.rglob("*.so"):
    if existing.is_file() or existing.is_symlink():
        existing.unlink()
runtime_target.mkdir(parents=True, exist_ok=True)
for path in binaries:
    if not path.is_file():
        raise RuntimeError(f"runtime binary is not a regular file: {path}")
    shutil.copy2(path, runtime_target / path.name)

actual_digest, source_count = source_digest(source)
if source_count == 0 or actual_digest != tree_sha256:
    raise RuntimeError(f"remote source digest mismatch: {actual_digest} != {tree_sha256}")
runtime_files = [
    {
        "relative_path": path.relative_to(source).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    for path in sorted(source.rglob("*.so")) if path.is_file()
]
if not runtime_files:
    raise RuntimeError("staged snapshot has no GPU4PySCF runtime binaries")
expected_runtime_paths = {
    (runtime_target / path.name).relative_to(source).as_posix() for path in binaries
}
if {entry["relative_path"] for entry in runtime_files} != expected_runtime_paths:
    raise RuntimeError("staged runtime inventory differs from the pinned runtime root")
inventory_library_names = sorted(entry["relative_path"].rsplit("/", 1)[-1] for entry in runtime_files)

manifest = {
    "schema": "gpu4pyscf.source-snapshot.v2",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "tree_sha256": tree_sha256,
    "base_revision": base_revision,
    "source": str(target / "source"),
    "immutable": True,
    "local_provenance": provenance,
    "runtime_binaries": {
        "source_root": str(runtime_binary_source), "complete": True,
        "required_library_names": required_library_names,
        "inventory_library_names": inventory_library_names,
        "closure_complete": True,
        "files": runtime_files,
    },
}
(staging / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
provenance_path.unlink()

# Seal and re-check staging completely before the only operation that publishes it.
for directory, subdirectories, files in os.walk(source):
    for name in files:
        os.chmod(pathlib.Path(directory) / name, 0o444)
    for name in subdirectories:
        os.chmod(pathlib.Path(directory) / name, 0o555)
os.chmod(source, 0o555)
os.chmod(staging / "manifest.json", 0o444)
os.chmod(staging, 0o555)
sealed_digest, sealed_count = source_digest(source)
if sealed_count != source_count or sealed_digest != tree_sha256:
    raise RuntimeError("sealed staging source differs from verified source")
for entry in runtime_files:
    path = source / entry["relative_path"]
    if not path.is_file() or path.stat().st_size != entry["bytes"] \
            or file_sha256(path) != entry["sha256"]:
        raise RuntimeError(f"sealed runtime binary differs: {entry['relative_path']}")
for path in (staging, staging / "manifest.json", source, *source.rglob("*")):
    if path.stat().st_mode & write_bits:
        raise RuntimeError(f"writable path remains in sealed staging: {path}")
if json.loads((staging / "manifest.json").read_text()) != manifest:
    raise RuntimeError("sealed staging manifest differs from assembled manifest")

target.parent.mkdir(parents=True, exist_ok=True)
os.rename(staging, target)
PY

# A post-rename failure removes only the target created by this invocation.
STAGING_ROOT=""
if ! verify_existing_snapshot; then
  remote_ssh chmod -R u+w "${SNAPSHOT_ROOT}" >/dev/null 2>&1 || true
  remote_ssh rm -rf "${SNAPSHOT_ROOT}" >/dev/null 2>&1 || true
  echo "published snapshot failed final validation and was removed" >&2
  exit 4
fi

echo "${SNAPSHOT_ROOT}/source"
