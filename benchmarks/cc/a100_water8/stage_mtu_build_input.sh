#!/usr/bin/env bash
# Freeze the current worktree as an identity-bound MTU build input.
#
# This is deliberately separate from snapshot publication: it never writes a
# manifest, runtime library, or digest-addressed snapshot, and it has no
# post-receive callback.  A successful run leaves the allocated build-input
# container in place for a later build job.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MTU_HOST_NAME="${MTU_HOST_NAME:-mtu}"
MTU_TASK_ROOT="${MTU_TASK_ROOT:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913}"
MTU_PYTHON_BIN="${MTU_PYTHON_BIN:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913/.venv/bin/python}"
FROZEN_BASE_REVISION="${FROZEN_BASE_REVISION:-a89b3ae018d4e82968323ef95645d91adde294aa}"
DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE:-candidate}"
MTU_BUILD_INPUTS_ROOT="${MTU_BUILD_INPUTS_ROOT:-${MTU_TASK_ROOT}/build-inputs}"
SSH_CONTROL_PATH="${TMPDIR:-/tmp}/agentqc-mtu-build-input-ssh-$$"
SSH_OPTIONS=(
  -o ConnectTimeout=20
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=5
  -o ControlMaster=auto
  -o ControlPersist=300
  -o ControlPath="${SSH_CONTROL_PATH}"
)
SSH_MASTER_STARTED=0
SUCCESS=0
CONTAINER_NAME=""
ROOT_DEV=""
ROOT_INO=""
CONTAINER_DEV=""
CONTAINER_INO=""
SOURCE_DEV=""
SOURCE_INO=""

validate_ssh_host() {
  local value="$1"
  if [[ -z "${value}" || "${value}" == -* \
      || ! "${value}" =~ ^[A-Za-z0-9_.:@%+-]+$ ]]; then
    echo "MTU_HOST_NAME is not a safe SSH host token" >&2
    return 2
  fi
}

validate_ssh_host "${MTU_HOST_NAME}"

case "${DEPLOYMENT_PROFILE}" in
  candidate|g0-canonical-pristine) ;;
  *) echo "unsupported DEPLOYMENT_PROFILE: ${DEPLOYMENT_PROFILE}" >&2; exit 2 ;;
esac

PROVENANCE_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-build-provenance.XXXXXX.json")"
PROVENANCE_CHECK_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-build-provenance-check.XXXXXX.json")"
STATUS_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-build-status.XXXXXX")"
STATUS_CHECK_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-build-status-check.XXXXXX")"
ALLOCATED_RECEIPT_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-build-allocation.XXXXXX.json")"
LOCAL_STAGING_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/gpu4pyscf-build-stage.XXXXXX")"
LOCAL_CHECK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/gpu4pyscf-build-check.XXXXXX")"
REMOTE_UPLOAD_PROGRAM="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-build-receiver.XXXXXX.py")"
LOCAL_STAGING_ROOT="$(cd -- "${LOCAL_STAGING_ROOT}" && pwd -P)"
LOCAL_CHECK_ROOT="$(cd -- "${LOCAL_CHECK_ROOT}" && pwd -P)"
PROVENANCE_FILE="$(cd -- "$(dirname -- "${PROVENANCE_FILE}")" && pwd -P)/$(basename -- "${PROVENANCE_FILE}")"
PROVENANCE_CHECK_FILE="$(cd -- "$(dirname -- "${PROVENANCE_CHECK_FILE}")" && pwd -P)/$(basename -- "${PROVENANCE_CHECK_FILE}")"
STATUS_FILE="$(cd -- "$(dirname -- "${STATUS_FILE}")" && pwd -P)/$(basename -- "${STATUS_FILE}")"
STATUS_CHECK_FILE="$(cd -- "$(dirname -- "${STATUS_CHECK_FILE}")" && pwd -P)/$(basename -- "${STATUS_CHECK_FILE}")"
ALLOCATED_RECEIPT_FILE="$(cd -- "$(dirname -- "${ALLOCATED_RECEIPT_FILE}")" && pwd -P)/$(basename -- "${ALLOCATED_RECEIPT_FILE}")"
REMOTE_UPLOAD_PROGRAM="$(cd -- "$(dirname -- "${REMOTE_UPLOAD_PROGRAM}")" && pwd -P)/$(basename -- "${REMOTE_UPLOAD_PROGRAM}")"

NORMALIZATION_FILE="${LOCAL_STAGING_ROOT}/source-normalization.json"
NORMALIZATION_CHECK_FILE="${LOCAL_CHECK_ROOT}/source-normalization.json"
LOCAL_SOURCE_ROOT="${LOCAL_STAGING_ROOT}/source"
REMOTE_STAGING_GUARD="${SCRIPT_DIR}/snapshot_staging_guard.py"
REMOTE_BUNDLE_RECEIVER="${SCRIPT_DIR}/snapshot_bundle_receiver.py"

cleanup() {
  local exit_status=$?
  if [[ "${SUCCESS}" != "1" && "${SSH_MASTER_STARTED}" == "1" \
      && -n "${CONTAINER_NAME}" && -n "${ROOT_DEV}" && -n "${ROOT_INO}" \
      && -n "${CONTAINER_DEV}" && -n "${CONTAINER_INO}" ]]; then
    # The only remote deletion permitted here is the identity-bound container
    # allocated by this invocation.  No name-only fallback is attempted.
    remote_guard remove "${MTU_BUILD_INPUTS_ROOT}" "${CONTAINER_NAME}" \
      "${ROOT_DEV}" "${ROOT_INO}" "${CONTAINER_DEV}" "${CONTAINER_INO}" \
      >/dev/null 2>&1 || echo "identity-bound build-input cleanup failed; inspect ${MTU_BUILD_INPUTS_ROOT}/${CONTAINER_NAME} (dev=${CONTAINER_DEV}, ino=${CONTAINER_INO})" >&2
  fi
  if [[ "${SSH_MASTER_STARTED}" == "1" ]]; then
    ssh "${SSH_OPTIONS[@]}" -O exit -- "${MTU_HOST_NAME}" >/dev/null 2>&1 || true
  fi
  rm -f "${PROVENANCE_FILE}" "${PROVENANCE_CHECK_FILE}" \
    "${STATUS_FILE}" "${STATUS_CHECK_FILE}" "${ALLOCATED_RECEIPT_FILE}" \
    "${REMOTE_UPLOAD_PROGRAM}" || true
  rm -rf "${LOCAL_STAGING_ROOT}" "${LOCAL_CHECK_ROOT}" || true
  rm -f "${SSH_CONTROL_PATH}" || true
  return "${exit_status}"
}
trap cleanup EXIT

remote_argv_command() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import shlex
import sys

if len(sys.argv) < 2:
    raise SystemExit("remote command argv must not be empty")
print(shlex.join(sys.argv[1:]))
PY
}

remote_ssh() {
  local remote_command
  remote_command="$(remote_argv_command "$@")"
  # OpenSSH passes the remote command through the login shell.  Supplying one
  # shlex-quoted command string preserves the exact argv even when a configured
  # remote path contains whitespace or shell metacharacters.
  ssh "${SSH_OPTIONS[@]}" -- "${MTU_HOST_NAME}" "${remote_command}"
}

remote_guard() {
  remote_ssh "${MTU_PYTHON_BIN}" - "$@" <"${REMOTE_STAGING_GUARD}"
}

json_field() {
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$1" "$2"
}

validate_bundle_program_info() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import re
import sys

def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def reject_float(value):
    raise ValueError(f"non-integer JSON number is forbidden: {value}")

value = json.loads(
    sys.argv[1], object_pairs_hook=reject_duplicate_pairs,
    parse_float=reject_float, parse_constant=reject_float,
)
expected = {
    "schema", "bundle_sha256", "bundle_bytes", "program_bytes",
    "post_receive_embedded",
}
if not isinstance(value, dict) or set(value) != expected:
    raise ValueError("bundle program receipt has an unexpected shape")
if value["schema"] != "gpu4pyscf.regular-tree-bundle.v1":
    raise ValueError("bundle program receipt schema mismatch")
if re.fullmatch(r"[0-9a-f]{64}", value["bundle_sha256"]) is None:
    raise ValueError("bundle program SHA-256 is invalid")
for key in ("bundle_bytes", "program_bytes"):
    if type(value[key]) is not int or value[key] <= 0:
        raise ValueError(f"bundle program {key} must be a positive integer")
if value["post_receive_embedded"] is not False:
    raise ValueError("freeze-only build input must not embed post-receive code")
print(f'{value["bundle_sha256"]}\t{value["bundle_bytes"]}')
PY
}

validate_guard_create_receipt() {
  "${PYTHON_BIN}" - "$1" "$2" "$3" "${4:-validate}" <<'PY'
import base64
import hashlib
import json
import os
import posixpath
import re
import stat
import sys

receipt_path, expected_root, tree_sha256, mode = sys.argv[1:]
maximum_receipt_bytes = 16384
preview_bytes = 4096
receipt_digest = hashlib.sha256()
captured = bytearray()
receipt_bytes = 0
descriptor = -1
read_error = None
try:
    descriptor = os.open(
        receipt_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("guard create receipt is not a regular file")
    while chunk := os.read(descriptor, 1024 * 1024):
        receipt_digest.update(chunk)
        receipt_bytes += len(chunk)
        if len(captured) <= maximum_receipt_bytes:
            remaining = maximum_receipt_bytes + 1 - len(captured)
            captured.extend(chunk[:remaining])
    after = os.fstat(descriptor)
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        or receipt_bytes != after.st_size
    ):
        raise RuntimeError("guard create receipt changed while read")
except Exception as exc:
    read_error = str(exc)
finally:
    if descriptor >= 0:
        os.close(descriptor)
raw = bytes(captured[:maximum_receipt_bytes])
oversized = receipt_bytes > maximum_receipt_bytes

def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def reject_float(value):
    raise ValueError(f"non-integer JSON number is forbidden: {value}")

def valid_identity(device, inode):
    return (
        type(device) is int and device >= 0
        and type(inode) is int and inode > 0
    )

def recovery_identity(value):
    if not isinstance(value, dict):
        return None
    name = value.get("staging_name")
    fields = (
        value.get("snapshots_dev"), value.get("snapshots_ino"),
        value.get("staging_dev"), value.get("staging_ino"),
    )
    if (
        value.get("snapshots_root") != expected_root
        or not isinstance(name, str)
        or re.fullmatch(
            rf"\.staging-{re.escape(tree_sha256[:12])}-[0-9a-f]{{24}}", name
        ) is None
        or not valid_identity(*fields[:2])
        or not valid_identity(*fields[2:])
    ):
        return None
    return {
        "argv": [
            "remove", expected_root, name,
            *(str(item) for item in fields),
        ],
        "container_path": expected_root + "/" + name,
    }

parsed = None
try:
    if read_error is not None:
        raise RuntimeError(read_error)
    if re.fullmatch(r"[0-9a-f]{64}", tree_sha256) is None:
        raise ValueError("expected source tree SHA-256 is invalid")
    if (
        not posixpath.isabs(expected_root)
        or posixpath.normpath(expected_root) != expected_root
    ):
        raise ValueError("expected build-input root is not a normalized path")
    if oversized:
        raise ValueError(
            f"guard create receipt exceeds {maximum_receipt_bytes} bytes"
        )
    parsed = json.loads(
        raw.decode("utf-8"), object_pairs_hook=reject_duplicate_pairs,
        parse_float=reject_float, parse_constant=reject_float,
    )
    expected_keys = {
        "schema", "snapshots_root", "snapshots_dev", "snapshots_ino",
        "staging_name", "staging_path", "staging_dev", "staging_ino",
        "source_dev", "source_ino",
    }
    if not isinstance(parsed, dict) or set(parsed) != expected_keys:
        raise ValueError("guard create receipt has an unexpected shape")
    if parsed["schema"] != "gpu4pyscf.remote-snapshot-staging-guard.v1":
        raise ValueError("guard create receipt schema mismatch")
    if parsed["snapshots_root"] != expected_root:
        raise ValueError("guard create snapshots root mismatch")
    name = parsed["staging_name"]
    if not isinstance(name, str) or re.fullmatch(
        rf"\.staging-{re.escape(tree_sha256[:12])}-[0-9a-f]{{24}}", name
    ) is None:
        raise ValueError("guard create staging name is invalid")
    if parsed["staging_path"] != expected_root + "/" + name:
        raise ValueError("guard create staging path mismatch")
    identities = [
        (parsed["snapshots_dev"], parsed["snapshots_ino"]),
        (parsed["staging_dev"], parsed["staging_ino"]),
        (parsed["source_dev"], parsed["source_ino"]),
    ]
    if not all(valid_identity(*identity) for identity in identities):
        raise ValueError("guard create identities are invalid")
    if len(set(identities)) != len(identities):
        raise ValueError("guard create directory identities must be distinct")
    if mode == "transport-failed":
        raise RuntimeError(
            "remote guard create transport failed after possible allocation"
        )
    if mode != "validate":
        raise ValueError("unknown guard receipt validation mode")
except Exception as exc:
    evidence = {
        "schema": "gpu4pyscf.guard-create-recovery.v1",
        "error": str(exc),
        "receipt_bytes_observed": receipt_bytes,
        "receipt_oversized": oversized,
        "receipt_sha256_observed": receipt_digest.hexdigest(),
        "raw_receipt_base64": base64.b64encode(
            raw[:preview_bytes]
        ).decode("ascii"),
        "raw_receipt_truncated": oversized or len(raw) > preview_bytes,
        "expected_snapshots_root": expected_root,
        "expected_tree_sha256": tree_sha256,
        "cleanup_identity": recovery_identity(parsed),
    }
    print(
        "guard create receipt rejected; recovery evidence="
        + json.dumps(evidence, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
    )
    raise SystemExit(1)

print("\t".join(str(item) for item in (
    parsed["snapshots_dev"], parsed["snapshots_ino"], name,
    parsed["staging_dev"], parsed["staging_ino"],
    parsed["source_dev"], parsed["source_ino"],
)))
PY
}

write_local_provenance() {
  local status_path="$1"
  local output_path="$2"
  LC_ALL=C git -C "${REPOSITORY_ROOT}" -c core.quotePath=true \
    status --porcelain=v1 --untracked-files=all >"${status_path}"
  "${PYTHON_BIN}" - "${REPOSITORY_ROOT}" "${FROZEN_BASE_REVISION}" \
    "${DEPLOYMENT_PROFILE}" "${status_path}" "${output_path}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
frozen, profile = sys.argv[2:4]
status_path, output_path = map(Path, sys.argv[4:6])

def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=check,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

head = git("rev-parse", "HEAD").stdout.decode("ascii").strip()
tracked_pristine = (
    git("diff", "--quiet", check=False).returncode == 0
    and git("diff", "--cached", "--quiet", check=False).returncode == 0
)
status_bytes = status_path.read_bytes()
entries = status_bytes.decode("utf-8").splitlines()
allowed = {
    "prefixes": ["benchmarks/cc/a100_water8/"],
    "files": ["gpu4pyscf/cc/device_runtime.py"],
}
bad = []
for entry in entries:
    if len(entry) < 4 or entry[:2] != "??":
        bad.append(entry)
    elif not (entry[3:].startswith(allowed["prefixes"][0])
              or entry[3:] in allowed["files"]):
        bad.append(entry)
untracked_allowed = not bad
head_matches = head == frozen
canonical = bool(profile == "g0-canonical-pristine" and head_matches
                 and tracked_pristine and untracked_allowed)
if profile == "g0-canonical-pristine" and not (head_matches and tracked_pristine and untracked_allowed):
    raise SystemExit("G0 provenance check failed")
policy = json.dumps(
    {"allowed_untracked": allowed, "status": entries},
    sort_keys=True, separators=(",", ":"),
).encode("utf-8")
payload = {
    "schema": "gpu4pyscf.local-provenance.v1",
    "deployment_profile": profile,
    "repository_root": str(root),
    "head_revision": head,
    "frozen_base_revision": frozen,
    "head_matches_frozen_base": head_matches,
    "tracked_pristine": tracked_pristine,
    "untracked_paths_allowed": untracked_allowed,
    "canonical_pristine": canonical,
    "allowed_untracked": allowed,
    "git_status": {
        "format": "porcelain-v1-lines",
        "sha256": hashlib.sha256(status_bytes).hexdigest(),
        "allowed_paths_status_sha256": hashlib.sha256(policy).hexdigest(),
        "entry_count": len(entries), "entries": entries,
    },
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

materialize_local_source() {
  local destination_root="$1"
  local normalization_file="$2"
  local destination_source="${destination_root}/source"
  mkdir -p "${destination_source}"
  rsync -a --delete \
    --exclude='.git' --exclude='.pytest_cache/' --exclude='__pycache__/' \
    --exclude='build/' --exclude='dist/' --exclude='results/' \
    --exclude='*.pyc' \
    "${REPOSITORY_ROOT}/" "${destination_source}/"
  "${PYTHON_BIN}" - "${REPOSITORY_ROOT}" "${destination_source}" \
    "${normalization_file}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3])
module_path = repository / "benchmarks/cc/a100_water8/snapshot_symlink_normalize.py"
spec = importlib.util.spec_from_file_location("stage_snapshot_normalizer", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
inventory = module.inspect_source(source)
normalization = module.materialize_source_links(source, inventory)
module.assert_normalized_source(source)
output.write_text(json.dumps(normalization, indent=2, sort_keys=True) + "\n")
PY
}

source_tree_identity() {
  "${PYTHON_BIN}" - "${1}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
module_path = root / "benchmarks/cc/a100_water8/snapshot_manifest.py"
spec = importlib.util.spec_from_file_location("stage_snapshot_manifest", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
digest, count = module.source_tree_digest(root)
print(json.dumps({"tree_sha256": digest, "files_hashed": count}, sort_keys=True))
PY
}

start_remote_session() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if ssh "${SSH_OPTIONS[@]}" -Nf -- "${MTU_HOST_NAME}"; then
      SSH_MASTER_STARTED=1
      return 0
    fi
    sleep "$((attempt * 5))"
  done
  return 1
}

# This verifier reopens the guarded source and computes the same source-tree
# digest/count while rejecting links and special files.  It is read-only and
# does not use a callback or modify the remote build-input container.
verify_remote_source() {
  remote_ssh "${MTU_PYTHON_BIN}" - "$@" <<'PY'
import hashlib
import json
import os
import stat
import sys

root, name, root_dev, root_ino, container_dev, container_ino, source_dev, source_ino, expected = sys.argv[1:]
root_dev, root_ino = int(root_dev), int(root_ino)
container_dev, container_ino = int(container_dev), int(container_ino)
source_dev, source_ino = int(source_dev), int(source_ino)
excluded = {".git", ".pytest_cache", "__pycache__", "build", "dist", "results"}
suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".ini", ".json", ".md", ".py", ".pyx", ".pxd", ".sbatch", ".sh", ".toml", ".txt", ".yaml", ".yml"}
flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)

def child(parent, item):
    before = os.stat(item, dir_fd=parent, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise RuntimeError("remote build input contains unsafe directory")
    fd = os.open(item, flags, dir_fd=parent)
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
        os.close(fd); raise RuntimeError("remote directory identity changed")
    return fd

def open_root(value):
    fd = os.open("/", flags)
    try:
        for item in os.path.normpath(value).split(os.sep)[1:]:
            nxt = child(fd, item); os.close(fd); fd = nxt
        return fd
    except BaseException:
        os.close(fd); raise

def walk(fd, rel=()):
    for item in sorted(os.listdir(fd)):
        info = os.stat(item, dir_fd=fd, follow_symlinks=False)
        current = rel + (item,)
        if stat.S_ISDIR(info.st_mode):
            nxt = child(fd, item)
            try:
                yield from walk(nxt, current)
                opened = os.fstat(nxt)
                current_info = os.stat(item, dir_fd=fd, follow_symlinks=False)
                if ((opened.st_dev, opened.st_ino, opened.st_mtime_ns)
                        != (info.st_dev, info.st_ino, info.st_mtime_ns)
                        or (current_info.st_dev, current_info.st_ino,
                            current_info.st_mtime_ns)
                        != (info.st_dev, info.st_ino, info.st_mtime_ns)):
                    raise RuntimeError("remote directory entry changed after recursion")
            finally:
                os.close(nxt)
        elif stat.S_ISREG(info.st_mode):
            yield fd, item, current, info
        else:
            raise RuntimeError("remote build input contains a link/special file")

root_fd = open_root(root)
try:
    root_info = os.fstat(root_fd)
    if (root_info.st_dev, root_info.st_ino) != (root_dev, root_ino): raise RuntimeError("build-input root identity mismatch")
    container_fd = child(root_fd, name)
    try:
        cinfo = os.fstat(container_fd)
        if (cinfo.st_dev, cinfo.st_ino) != (container_dev, container_ino): raise RuntimeError("build-input container identity mismatch")
        source_fd = child(container_fd, "source")
        try:
            sinfo = os.fstat(source_fd)
            if (sinfo.st_dev, sinfo.st_ino) != (source_dev, source_ino): raise RuntimeError("build-input source identity mismatch")
            digest = hashlib.sha256(); count = 0; regular = 0
            for parent, item, parts, info in walk(source_fd):
                regular += 1
                if any(part in excluded for part in parts) or os.path.splitext(item)[1].lower() not in suffixes: continue
                handle = os.open(item, file_flags, dir_fd=parent)
                try:
                    opened = os.fstat(handle)
                    before_identity = (
                        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
                    )
                    if (
                        opened.st_dev, opened.st_ino, opened.st_size,
                        opened.st_mtime_ns
                    ) != before_identity:
                        raise RuntimeError("remote source file changed before read")
                    digest.update("/".join(parts).encode()); digest.update(b"\0")
                    while chunk := os.read(handle, 1024 * 1024): digest.update(chunk)
                    after = os.fstat(handle)
                    current_info = os.stat(
                        item, dir_fd=parent, follow_symlinks=False
                    )
                    if (
                        (after.st_dev, after.st_ino, after.st_size,
                         after.st_mtime_ns) != before_identity
                        or (current_info.st_dev, current_info.st_ino,
                            current_info.st_size, current_info.st_mtime_ns)
                        != before_identity
                    ):
                        raise RuntimeError("remote source file changed after read")
                    digest.update(b"\0")
                    count += 1
                finally: os.close(handle)
            result = {"source_sha256": digest.hexdigest(), "source_file_count": count, "regular_file_count": regular}
            if result["source_sha256"] != expected: raise RuntimeError("remote source digest mismatch")
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        finally: os.close(source_fd)
    finally: os.close(container_fd)
finally: os.close(root_fd)
PY
}

write_local_provenance "${STATUS_FILE}" "${PROVENANCE_FILE}"
materialize_local_source "${LOCAL_STAGING_ROOT}" "${NORMALIZATION_FILE}"
TREE_IDENTITY_JSON="$(source_tree_identity "${LOCAL_SOURCE_ROOT}" | tail -n 1)"
TREE_SHA256="$(json_field "${TREE_IDENTITY_JSON}" tree_sha256)"
TREE_FILE_COUNT="$(json_field "${TREE_IDENTITY_JSON}" files_hashed)"

# Build the receiver between the first freeze and the independent second copy.
PROGRAM_INFO="$("${PYTHON_BIN}" "${REMOTE_BUNDLE_RECEIVER}" build-program \
  --source-root "${LOCAL_SOURCE_ROOT}" --provenance "${PROVENANCE_FILE}" \
  --normalization "${NORMALIZATION_FILE}" --output "${REMOTE_UPLOAD_PROGRAM}")"
PROGRAM_FIELDS="$(validate_bundle_program_info "${PROGRAM_INFO}")"
IFS=$'\t' read -r BUNDLE_SHA256 BUNDLE_BYTES <<<"${PROGRAM_FIELDS}"

write_local_provenance "${STATUS_CHECK_FILE}" "${PROVENANCE_CHECK_FILE}"
cmp -s "${PROVENANCE_FILE}" "${PROVENANCE_CHECK_FILE}"
materialize_local_source "${LOCAL_CHECK_ROOT}" "${NORMALIZATION_CHECK_FILE}"
cmp -s "${NORMALIZATION_FILE}" "${NORMALIZATION_CHECK_FILE}"
TREE_IDENTITY_CHECK_JSON="$(source_tree_identity "${LOCAL_CHECK_ROOT}/source" | tail -n 1)"
[[ "${TREE_IDENTITY_CHECK_JSON}" == "${TREE_IDENTITY_JSON}" ]]

NORMALIZATION_SHA256="$("${PYTHON_BIN}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${NORMALIZATION_FILE}")"
NORMALIZATION_BYTES="$(wc -c <"${NORMALIZATION_FILE}" | tr -d ' ')"
PROVENANCE_SHA256="$("${PYTHON_BIN}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${PROVENANCE_FILE}")"
PROVENANCE_BYTES="$(wc -c <"${PROVENANCE_FILE}" | tr -d ' ')"

GUARD_SHA256="$("${PYTHON_BIN}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${REMOTE_STAGING_GUARD}")"

start_remote_session
if ! remote_guard create "${MTU_BUILD_INPUTS_ROOT}" "${TREE_SHA256}" \
    >"${ALLOCATED_RECEIPT_FILE}"; then
  validate_guard_create_receipt "${ALLOCATED_RECEIPT_FILE}" \
    "${MTU_BUILD_INPUTS_ROOT}" "${TREE_SHA256}" transport-failed \
    >/dev/null || true
  exit 1
fi
if ! ALLOCATION_FIELDS="$(validate_guard_create_receipt \
    "${ALLOCATED_RECEIPT_FILE}" "${MTU_BUILD_INPUTS_ROOT}" \
    "${TREE_SHA256}")"; then
  exit 1
fi
# No cleanup identity becomes active until the complete receipt, including all
# paths, identities, and schema fields, has passed the single validation above.
IFS=$'\t' read -r ROOT_DEV ROOT_INO CONTAINER_NAME CONTAINER_DEV \
  CONTAINER_INO SOURCE_DEV SOURCE_INO <<<"${ALLOCATION_FIELDS}"
remote_guard verify-staging "${MTU_BUILD_INPUTS_ROOT}" "${CONTAINER_NAME}" \
  "${ROOT_DEV}" "${ROOT_INO}" "${CONTAINER_DEV}" "${CONTAINER_INO}" \
  "${SOURCE_DEV}" "${SOURCE_INO}" >/dev/null

# No post-receive script is embedded: the receiver only creates regular source
# inputs under the identity-bound container and returns its receipt.
RECEIPT="$(remote_ssh "${MTU_PYTHON_BIN}" - \
  "${MTU_BUILD_INPUTS_ROOT}" "${CONTAINER_NAME}" \
  "${ROOT_DEV}" "${ROOT_INO}" "${CONTAINER_DEV}" "${CONTAINER_INO}" \
  source "${SOURCE_DEV}" "${SOURCE_INO}" "${BUNDLE_SHA256}" \
  "${BUNDLE_BYTES}" <"${REMOTE_UPLOAD_PROGRAM}")"
[[ "$(json_field "${RECEIPT}" bundle_sha256)" == "${BUNDLE_SHA256}" ]]
[[ "$(json_field "${RECEIPT}" bundle_bytes)" == "${BUNDLE_BYTES}" ]]
[[ "$(json_field "${RECEIPT}" symlink_count)" == "0" ]]
[[ "$(json_field "${RECEIPT}" special_file_count)" == "0" ]]

REMOTE_SOURCE_JSON="$(verify_remote_source "${MTU_BUILD_INPUTS_ROOT}" \
  "${CONTAINER_NAME}" "${ROOT_DEV}" "${ROOT_INO}" \
  "${CONTAINER_DEV}" "${CONTAINER_INO}" "${SOURCE_DEV}" \
  "${SOURCE_INO}" "${TREE_SHA256}")"
[[ "$(json_field "${REMOTE_SOURCE_JSON}" source_file_count)" == "${TREE_FILE_COUNT}" ]]
remote_guard verify-staging "${MTU_BUILD_INPUTS_ROOT}" "${CONTAINER_NAME}" \
  "${ROOT_DEV}" "${ROOT_INO}" "${CONTAINER_DEV}" "${CONTAINER_INO}" \
  "${SOURCE_DEV}" "${SOURCE_INO}" >/dev/null

REMOTE_SOURCE_ROOT="${MTU_BUILD_INPUTS_ROOT}/${CONTAINER_NAME}/source"
CLEANUP_ARGS_JSON="$("${PYTHON_BIN}" - \
  "${MTU_BUILD_INPUTS_ROOT}" "${CONTAINER_NAME}" "${ROOT_DEV}" \
  "${ROOT_INO}" "${CONTAINER_DEV}" "${CONTAINER_INO}" <<'PY'
import json, sys
print(json.dumps(["remove", *sys.argv[1:]], separators=(",", ":")))
PY
)"
"${PYTHON_BIN}" - "${RECEIPT}" "${TREE_SHA256}" "${TREE_FILE_COUNT}" \
  "${REMOTE_SOURCE_ROOT}" "${NORMALIZATION_SHA256}" "${NORMALIZATION_BYTES}" \
  "${PROVENANCE_SHA256}" "${PROVENANCE_BYTES}" "${ROOT_DEV}" "${ROOT_INO}" \
  "${CONTAINER_DEV}" "${CONTAINER_INO}" "${SOURCE_DEV}" "${SOURCE_INO}" \
  "${CLEANUP_ARGS_JSON}" "${REMOTE_SOURCE_JSON}" "${MTU_HOST_NAME}" \
  "${MTU_PYTHON_BIN}" "${REMOTE_STAGING_GUARD}" "${GUARD_SHA256}" <<'PY'
import json, sys

receipt = json.loads(sys.argv[1])
print(json.dumps({
    "schema": "gpu4pyscf.mtu-build-input.v1",
    "source_sha256": sys.argv[2],
    "source_file_count": int(sys.argv[3]),
    "remote_source_root": sys.argv[4],
    "normalization_identity": {"sha256": sys.argv[5], "bytes": int(sys.argv[6])},
    "provenance_identity": {"sha256": sys.argv[7], "bytes": int(sys.argv[8])},
    "bundle_receipt": receipt,
    "root_identity": {"dev": int(sys.argv[9]), "ino": int(sys.argv[10])},
    "container_identity": {"dev": int(sys.argv[11]), "ino": int(sys.argv[12])},
    "source_identity": {"dev": int(sys.argv[13]), "ino": int(sys.argv[14])},
    "guarded_cleanup_arguments": json.loads(sys.argv[15]),
    "remote_source_observation": json.loads(sys.argv[16]),
    "guarded_cleanup": {
        "host": sys.argv[17],
        "remote_python": sys.argv[18],
        "guard_path": sys.argv[19],
        "guard_sha256": sys.argv[20],
        "argv": [sys.argv[18], "-", *json.loads(sys.argv[15])],
    },
    "preserved": True,
}, sort_keys=True, separators=(",", ":")))
PY
SUCCESS=1
