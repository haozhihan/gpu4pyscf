#!/usr/bin/env bash
# Deploy the current worktree as a content-addressed, read-only MTU snapshot.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MTU_HOST_NAME="${MTU_HOST_NAME:-mtu}"
MTU_TASK_ROOT="${MTU_TASK_ROOT:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-a100-thc-rr-20260913}"
MTU_PYTHON_BIN="${MTU_PYTHON_BIN:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913/.venv/bin/python}"
MTU_GPU4PYSCF_BINARY_ROOT="${MTU_GPU4PYSCF_BINARY_ROOT:-/mnt/mridata/vxu/thu-likun/hanhaozhi/agent-ccsd-baseline-20260913/.venv/lib/python3.11/site-packages/gpu4pyscf/lib}"
MTU_RUNTIME_BUNDLE_ROOT="${MTU_RUNTIME_BUNDLE_ROOT:-}"
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
SSH_MASTER_STARTED=0

case "${DEPLOYMENT_PROFILE}" in
  candidate)
    if [[ -z "${MTU_RUNTIME_BUNDLE_ROOT}" ]]; then
      echo "candidate deployment requires explicit MTU_RUNTIME_BUNDLE_ROOT" >&2
      exit 2
    fi
    RUNTIME_BINARY_SOURCE_ROOT="${MTU_RUNTIME_BUNDLE_ROOT}"
    ;;
  g0-canonical-pristine)
    RUNTIME_BINARY_SOURCE_ROOT="${MTU_GPU4PYSCF_BINARY_ROOT}"
    ;;
  *) echo "unsupported DEPLOYMENT_PROFILE: ${DEPLOYMENT_PROFILE}" >&2; exit 2 ;;
esac

if [[ -z "${MTU_HOST_NAME}" || "${MTU_HOST_NAME}" == -* \
    || ! "${MTU_HOST_NAME}" =~ ^[A-Za-z0-9_.:@%+-]+$ ]]; then
  echo "MTU_HOST_NAME is not a safe SSH host token" >&2
  exit 2
fi

PROVENANCE_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-provenance.XXXXXX.json")"
PROVENANCE_CHECK_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-provenance-check.XXXXXX.json")"
STATUS_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-status.XXXXXX")"
STATUS_CHECK_FILE="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-status-check.XXXXXX")"
LOCAL_STAGING_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/gpu4pyscf-source-stage.XXXXXX")"
LOCAL_CHECK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/gpu4pyscf-source-check.XXXXXX")"
LOCAL_STAGING_ROOT="$(cd -- "${LOCAL_STAGING_ROOT}" && pwd -P)"
LOCAL_CHECK_ROOT="$(cd -- "${LOCAL_CHECK_ROOT}" && pwd -P)"
REMOTE_UPLOAD_PROGRAM="$(mktemp "${TMPDIR:-/tmp}/gpu4pyscf-source-receiver.XXXXXX.py")"
PROVENANCE_FILE="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "${PROVENANCE_FILE}")"
PROVENANCE_CHECK_FILE="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "${PROVENANCE_CHECK_FILE}")"
STATUS_FILE="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "${STATUS_FILE}")"
STATUS_CHECK_FILE="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "${STATUS_CHECK_FILE}")"
REMOTE_UPLOAD_PROGRAM="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "${REMOTE_UPLOAD_PROGRAM}")"
NORMALIZATION_FILE="${LOCAL_STAGING_ROOT}/source-normalization.json"
NORMALIZATION_CHECK_FILE="${LOCAL_CHECK_ROOT}/source-normalization.json"
REMOTE_STAGING_GUARD="${SCRIPT_DIR}/snapshot_staging_guard.py"
REMOTE_SNAPSHOTS_ROOT="${MTU_TASK_ROOT}/snapshots"
STAGING_ROOT=""
STAGING_NAME=""
SNAPSHOTS_DEV=""
SNAPSHOTS_INO=""
STAGING_DEV=""
STAGING_INO=""
STAGING_SOURCE_DEV=""
STAGING_SOURCE_INO=""
PUBLICATION_RECEIPT_OBTAINED=0

cleanup() {
  local exit_status=$?
  local cleanup_succeeded=0
  local cleanup_name
  if [[ -n "${STAGING_NAME}" && "${SSH_MASTER_STARTED}" == "1" \
      && -n "${SNAPSHOTS_DEV}" && -n "${SNAPSHOTS_INO}" \
      && -n "${STAGING_DEV}" && -n "${STAGING_INO}" ]]; then
    # The guard reopens the private root without following links, checks the
    # root and child device/inode pairs, and purges only through held dir_fds.
    if remote_staging_guard remove \
        "${REMOTE_SNAPSHOTS_ROOT}" "${STAGING_NAME}" \
        "${SNAPSHOTS_DEV}" "${SNAPSHOTS_INO}" \
        "${STAGING_DEV}" "${STAGING_INO}" >/dev/null 2>&1; then
      cleanup_succeeded=1
    elif [[ "${PUBLICATION_RECEIPT_OBTAINED}" == "1" ]]; then
      # A successful callback receipt conclusively means the inode was
      # published.  Only then may cleanup inspect the digest entry.
      cleanup_name="${SNAPSHOT_NAME:-}"
      if [[ -n "${cleanup_name}" ]] && remote_staging_guard remove \
          "${REMOTE_SNAPSHOTS_ROOT}" "${cleanup_name}" \
          "${SNAPSHOTS_DEV}" "${SNAPSHOTS_INO}" \
          "${STAGING_DEV}" "${STAGING_INO}" >/dev/null 2>&1; then
        cleanup_succeeded=1
      fi
    fi
    if [[ "${cleanup_succeeded}" != "1" ]]; then
      if [[ "${PUBLICATION_RECEIPT_OBTAINED}" == "1" ]]; then
        echo "identity-bound remote cleanup could not find the deployment inode; inspect names ${STAGING_NAME} and ${SNAPSHOT_NAME:-unset} under ${REMOTE_SNAPSHOTS_ROOT} for identity dev=${STAGING_DEV}, ino=${STAGING_INO}" >&2
      else
        echo "publication outcome may be indeterminate; digest target ${SNAPSHOT_NAME:-unset} was deliberately not probed or deleted; inspect staging=${STAGING_NAME}, inode dev=${STAGING_DEV}, ino=${STAGING_INO} under ${REMOTE_SNAPSHOTS_ROOT}" >&2
      fi
    fi
  fi
  if [[ "${SSH_MASTER_STARTED}" == "1" ]]; then
    ssh "${SSH_OPTIONS[@]}" -O exit -- "${MTU_HOST_NAME}" >/dev/null 2>&1 || true
  fi
  rm -f "${PROVENANCE_FILE}" "${PROVENANCE_CHECK_FILE}" \
    "${STATUS_FILE}" "${STATUS_CHECK_FILE}" "${REMOTE_UPLOAD_PROGRAM}" \
    || true
  rm -rf "${LOCAL_STAGING_ROOT}" "${LOCAL_CHECK_ROOT}" || true
  rm -f "${SSH_CONTROL_PATH}"
  return "${exit_status}"
}

materialize_local_source() {
  local destination_root="$1"
  local normalization_file="$2"
  local destination_source="${destination_root}/source"
  mkdir -p "${destination_source}"
  # Preserve links during this private local copy.  The no-follow validator
  # below is the only code allowed to read a target, and it replaces every
  # accepted link before any network transfer begins.
  rsync -a --delete \
    --exclude='.git' --exclude='.pytest_cache/' --exclude='__pycache__/' \
    --exclude='.ruff_cache/' \
    --exclude='build/' --exclude='dist/' --exclude='results/' \
    --exclude='*.pyc' \
    "${REPOSITORY_ROOT}/" "${destination_source}/"
  python - "${REPOSITORY_ROOT}" "${destination_source}" \
    "${normalization_file}" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3])
module_path = repository / "benchmarks/cc/a100_water8/snapshot_symlink_normalize.py"
spec = importlib.util.spec_from_file_location(
    "water8_snapshot_symlink_normalize", module_path
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
inventory = module.inspect_source(source)
normalization = module.materialize_source_links(source, inventory)
module.assert_normalized_source(source)
output.write_text(
    json.dumps(normalization, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

source_tree_identity() {
  local source_root="$1"
  python - "${source_root}" <<'PY'
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
digest, count = module.source_tree_digest(root)
print(json.dumps({"tree_sha256": digest, "files_hashed": count}, sort_keys=True))
PY
}
trap cleanup EXIT

start_remote_session() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if ssh "${SSH_OPTIONS[@]}" -Nf -- "${MTU_HOST_NAME}"; then
      SSH_MASTER_STARTED=1
      return 0
    fi
    sleep "$((attempt * 5))"
  done
  echo "unable to establish a persistent SSH session to ${MTU_HOST_NAME}" >&2
  return 1
}

remote_ssh() {
  local remote_command
  remote_command="$(python - env PYTHONDONTWRITEBYTECODE=1 "$@" <<'PY'
import shlex
import sys
if len(sys.argv) < 2:
    raise SystemExit("remote command argv must not be empty")
print(shlex.join(sys.argv[1:]))
PY
)"
  ssh "${SSH_OPTIONS[@]}" -- "${MTU_HOST_NAME}" "${remote_command}"
}

remote_staging_guard() {
  # Stream the reviewed guard on every invocation.  Nothing is installed or
  # read through an attacker-selected remote staging path.
  remote_ssh "${MTU_PYTHON_BIN}" - "$@" <"${REMOTE_STAGING_GUARD}"
}

json_field() {
  python -c \
    'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$1" "$2"
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
materialize_local_source "${LOCAL_STAGING_ROOT}" "${NORMALIZATION_FILE}"
LOCAL_SOURCE_ROOT="${LOCAL_STAGING_ROOT}/source"
REMOTE_STAGING_GUARD="${LOCAL_SOURCE_ROOT}/benchmarks/cc/a100_water8/snapshot_staging_guard.py"
REMOTE_BUNDLE_RECEIVER="${LOCAL_SOURCE_ROOT}/benchmarks/cc/a100_water8/snapshot_bundle_receiver.py"
REMOTE_SNAPSHOT_ASSEMBLY="${LOCAL_SOURCE_ROOT}/benchmarks/cc/a100_water8/snapshot_remote_assembly.py"

REQUIRED_RUNTIME_JSON="$({ python - "${LOCAL_SOURCE_ROOT}" <<'PY'
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

TREE_IDENTITY_JSON="$(source_tree_identity "${LOCAL_SOURCE_ROOT}" | tail -n 1)"
TREE_SHA256="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["tree_sha256"])' "${TREE_IDENTITY_JSON}")"
TREE_FILE_COUNT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["files_hashed"])' "${TREE_IDENTITY_JSON}")"
NORMALIZATION_SHA256="$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "${NORMALIZATION_FILE}")"
NORMALIZATION_BYTES="$(wc -c <"${NORMALIZATION_FILE}" | tr -d ' ')"
if [[ ! "${TREE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "failed to compute a source tree SHA-256" >&2
  exit 2
fi
if [[ ! "${TREE_FILE_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "failed to compute a non-empty source tree file count" >&2
  exit 2
fi
if [[ ! "${NORMALIZATION_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
    || [[ ! "${NORMALIZATION_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "failed to bind source-link normalization record" >&2
  exit 2
fi
BUNDLE_PROGRAM_INFO="$(python "${REMOTE_BUNDLE_RECEIVER}" build-program \
  --source-root "${LOCAL_SOURCE_ROOT}" \
  --provenance "${PROVENANCE_FILE}" \
  --normalization "${NORMALIZATION_FILE}" \
  --post-receive-script "${REMOTE_SNAPSHOT_ASSEMBLY}" \
  --output "${REMOTE_UPLOAD_PROGRAM}")"
BUNDLE_SHA256="$(json_field "${BUNDLE_PROGRAM_INFO}" bundle_sha256)"
BUNDLE_BYTES="$(json_field "${BUNDLE_PROGRAM_INFO}" bundle_bytes)"
if [[ ! "${BUNDLE_SHA256}" =~ ^[0-9a-f]{64}$ \
    || ! "${BUNDLE_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "failed to build the normalized source transfer bundle" >&2
  exit 2
fi

# Rebuild an independent normalized staging tree after bundle creation.  Matching Git
# provenance alone cannot detect an in-place edit to an untracked file, so the
# normalized digest, file count, and link inventory must all match as well.
write_local_provenance "${STATUS_CHECK_FILE}" "${PROVENANCE_CHECK_FILE}"
if ! cmp -s "${PROVENANCE_FILE}" "${PROVENANCE_CHECK_FILE}"; then
  echo "local Git provenance changed during bundle creation" >&2
  exit 3
fi
materialize_local_source "${LOCAL_CHECK_ROOT}" "${NORMALIZATION_CHECK_FILE}"
if ! cmp -s "${NORMALIZATION_FILE}" "${NORMALIZATION_CHECK_FILE}"; then
  echo "local source-link normalization inventory changed during bundle creation" >&2
  exit 3
fi
TREE_IDENTITY_CHECK_JSON="$(source_tree_identity "${LOCAL_CHECK_ROOT}/source" | tail -n 1)"
if [[ "${TREE_IDENTITY_CHECK_JSON}" != "${TREE_IDENTITY_JSON}" ]]; then
  echo "local normalized source digest or file count changed during bundle creation" >&2
  exit 3
fi


SNAPSHOT_NAME="${TREE_SHA256}"
SNAPSHOT_ROOT="${REMOTE_SNAPSHOTS_ROOT}/${SNAPSHOT_NAME}"

# Published digest names are single-use.  Existing snapshots are never reopened
# or reused by deployment; runtime jobs validate them independently before use.

start_remote_session

SNAPSHOT_INSPECTION="$(remote_staging_guard inspect \
  "${REMOTE_SNAPSHOTS_ROOT}" "${SNAPSHOT_NAME}")"
SNAPSHOTS_DEV="$(json_field "${SNAPSHOT_INSPECTION}" snapshots_dev)"
SNAPSHOTS_INO="$(json_field "${SNAPSHOT_INSPECTION}" snapshots_ino)"
SNAPSHOT_STATE="$(json_field "${SNAPSHOT_INSPECTION}" state)"
if [[ "${SNAPSHOT_STATE}" == "directory" ]]; then
  echo "source-addressed snapshot already exists; automatic reuse is disabled: ${SNAPSHOT_ROOT}" >&2
  exit 5
fi
if [[ "${SNAPSHOT_STATE}" != "missing" ]]; then
  echo "remote snapshot guard returned an invalid state: ${SNAPSHOT_STATE}" >&2
  exit 3
fi

STAGING_IDENTITY="$(remote_staging_guard create \
  "${REMOTE_SNAPSHOTS_ROOT}" "${TREE_SHA256}")"
if [[ "$(json_field "${STAGING_IDENTITY}" snapshots_dev)" != "${SNAPSHOTS_DEV}" \
    || "$(json_field "${STAGING_IDENTITY}" snapshots_ino)" != "${SNAPSHOTS_INO}" ]]; then
  echo "remote snapshots root changed before staging allocation" >&2
  exit 3
fi
STAGING_ROOT="$(json_field "${STAGING_IDENTITY}" staging_path)"
STAGING_NAME="$(json_field "${STAGING_IDENTITY}" staging_name)"
STAGING_DEV="$(json_field "${STAGING_IDENTITY}" staging_dev)"
STAGING_INO="$(json_field "${STAGING_IDENTITY}" staging_ino)"
STAGING_SOURCE_DEV="$(json_field "${STAGING_IDENTITY}" source_dev)"
STAGING_SOURCE_INO="$(json_field "${STAGING_IDENTITY}" source_ino)"
remote_staging_guard verify-staging "${REMOTE_SNAPSHOTS_ROOT}" \
  "${STAGING_NAME}" "${SNAPSHOTS_DEV}" "${SNAPSHOTS_INO}" \
  "${STAGING_DEV}" "${STAGING_INO}" \
  "${STAGING_SOURCE_DEV}" "${STAGING_SOURCE_INO}" >/dev/null
# One remote process holds the deployment-user private anchor plus the
# snapshots/staging/source descriptors for the whole receive.  Payload paths
# are validated tar names and are created only with mkdirat/openat relative to
# those descriptors.  Its embedded callback binds runtime libraries, validates
# and seals the tree, and publishes atomically before releasing the fds.
BUNDLE_RECEIPT="$(remote_ssh "${MTU_PYTHON_BIN}" - \
  "${REMOTE_SNAPSHOTS_ROOT}" "${STAGING_NAME}" \
  "${SNAPSHOTS_DEV}" "${SNAPSHOTS_INO}" \
  "${STAGING_DEV}" "${STAGING_INO}" source \
  "${STAGING_SOURCE_DEV}" "${STAGING_SOURCE_INO}" \
  "${BUNDLE_SHA256}" "${BUNDLE_BYTES}" \
  "${SNAPSHOT_NAME}" "${TREE_SHA256}" "${TREE_FILE_COUNT}" \
  "${NORMALIZATION_SHA256}" "${NORMALIZATION_BYTES}" \
  "${FROZEN_BASE_REVISION}" "${RUNTIME_BINARY_SOURCE_ROOT}" \
  "${REQUIRED_RUNTIME_B64}" "${DEPLOYMENT_PROFILE}" \
  <"${REMOTE_UPLOAD_PROGRAM}")"
# The callback may already have atomically renamed the staging inode.  Until a
# complete callback receipt exists, failure cleanup may inspect only the old
# staging name; this preserves a digest target whose publication outcome is
# indeterminate.
if [[ "$(json_field "${BUNDLE_RECEIPT}" bundle_sha256)" != "${BUNDLE_SHA256}" \
    || "$(json_field "${BUNDLE_RECEIPT}" bundle_bytes)" != "${BUNDLE_BYTES}" \
    || "$(json_field "${BUNDLE_RECEIPT}" symlink_count)" != "0" \
    || "$(json_field "${BUNDLE_RECEIPT}" special_file_count)" != "0" ]]; then
  echo "remote normalized source receiver returned an invalid receipt" >&2
  exit 3
fi
PUBLISHED_SNAPSHOT="$(python -c \
  'import json,sys; print(json.loads(sys.argv[1])["post_receive"]["published_snapshot"])' \
  "${BUNDLE_RECEIPT}")"
PUBLISH_PROTOCOL="$(python -c \
  'import json,sys; print(json.loads(sys.argv[1])["post_receive"]["publish_protocol"])' \
  "${BUNDLE_RECEIPT}")"
PUBLISH_TRUST_BOUNDARY="$(python -c \
  'import json,sys; print(json.loads(sys.argv[1])["post_receive"]["publish_trust_boundary"])' \
  "${BUNDLE_RECEIPT}")"
PUBLISHED_TREE_SHA256="$(python -c \
  'import json,sys; print(json.loads(sys.argv[1])["post_receive"]["tree_sha256"])' \
  "${BUNDLE_RECEIPT}")"
PUBLICATION_ATTESTATION_SHA256="$(python -c \
  'import json,sys; print(json.loads(sys.argv[1])["post_receive"]["publication_attestation_sha256"])' \
  "${BUNDLE_RECEIPT}")"
PUBLICATION_ATTESTATION_NAME="$(python -c \
  'import json,sys; print(json.loads(sys.argv[1])["post_receive"]["publication_attestation_name"])' \
  "${BUNDLE_RECEIPT}")"
PUBLISHED_MANIFEST_SHA256="$(python -c \
  'import json,sys; print(json.loads(sys.argv[1])["post_receive"]["published_manifest_sha256"])' \
  "${BUNDLE_RECEIPT}")"
PUBLICATION_ATTESTATION_FILE_SHA256="$(python -c \
  'import json,sys; print(json.loads(sys.argv[1])["post_receive"]["publication_attestation_file_sha256"])' \
  "${BUNDLE_RECEIPT}")"
RUNTIME_RECEIPT_JSON="$(python - "${BUNDLE_RECEIPT}" "${SNAPSHOT_NAME}" "${TREE_SHA256}" \
  "${STAGING_DEV}" "${STAGING_INO}" "${PUBLISHED_MANIFEST_SHA256}" \
  "${PUBLICATION_ATTESTATION_FILE_SHA256}" "${DEPLOYMENT_PROFILE}" \
  "${RUNTIME_BINARY_SOURCE_ROOT}" "${TREE_FILE_COUNT}" <<'PY'
import hashlib
import json
from pathlib import PurePosixPath
import sys

receipt, snapshot_name, tree_sha256, staging_dev, staging_ino, manifest_sha256, attestation_file_sha256, profile, runtime_root, source_file_count = sys.argv[1:]
payload = json.loads(receipt)
post = payload.get("post_receive")
if not isinstance(post, dict):
    raise SystemExit("remote snapshot assembly returned no publication receipt")
if post.get("published_snapshot") != snapshot_name:
    raise SystemExit("remote snapshot assembly returned an invalid publication receipt")
if post.get("tree_sha256") != tree_sha256:
    raise SystemExit("remote snapshot assembly returned an invalid source digest")
if post.get("source_file_count") != int(source_file_count):
    raise SystemExit("remote snapshot assembly returned an invalid source file count")
if post.get("publish_protocol") not in {
    "atomic-exclusive-rename",
    "atomic-exclusive-rename-commit-confirmed",
    "reserved-empty-directory-rename",
    "reserved-empty-directory-rename-commit-confirmed",
}:
    raise SystemExit("remote snapshot assembly returned an invalid publication protocol")
if post.get("publish_trust_boundary") != "private-anchor-owner-and-root-are-trusted-cooperators":
    raise SystemExit("remote snapshot assembly returned an invalid publication trust boundary")
if post.get("published_manifest_sha256") != manifest_sha256:
    raise SystemExit("remote snapshot assembly returned an invalid manifest digest")
if post.get("publication_attestation_file_sha256") != attestation_file_sha256:
    raise SystemExit("remote snapshot assembly returned an invalid attestation file digest")
attestation = post.get("publication_attestation")
if not isinstance(attestation, dict):
    raise SystemExit("remote snapshot assembly returned no attestation payload")
if attestation.get("source_tree_sha256") != tree_sha256 or attestation.get("snapshot_name") != snapshot_name:
    raise SystemExit("remote snapshot assembly returned an invalid attestation identity")
if attestation.get("published_manifest_sha256") != manifest_sha256:
    raise SystemExit("remote snapshot assembly returned an invalid attested manifest digest")
identity = attestation.get("published_directory_identity")
if identity != {"device": int(staging_dev), "inode": int(staging_ino)}:
    raise SystemExit("remote snapshot assembly returned an invalid published inode")
self_hash = attestation.get("attestation_sha256")
unsigned = dict(attestation)
unsigned.pop("attestation_sha256", None)
canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
if self_hash != hashlib.sha256(canonical).hexdigest() or not isinstance(self_hash, str):
    raise SystemExit("remote snapshot assembly returned an invalid attestation self-hash")
if post.get("publication_attestation_sha256") != self_hash:
    raise SystemExit("remote snapshot assembly returned an inconsistent attestation hash")
if post.get("publication_attestation_name") != snapshot_name + ".publication-attestation.json":
    raise SystemExit("remote snapshot assembly returned an invalid attestation name")
runtime = {
    "runtime_contract_validated": post.get("runtime_contract_validated"),
    "runtime_bundle_id": post.get("runtime_bundle_id"),
    "runtime_inventory_sha256": post.get("runtime_inventory_sha256"),
    "runtime_sidecar_sha256": post.get("runtime_sidecar_sha256"),
    "runtime_build_verification_sha256": post.get(
        "runtime_build_verification_sha256"
    ),
    "runtime_validation_evidence_sha256": post.get(
        "runtime_validation_evidence_sha256"
    ),
}
hash_fields = (
    "runtime_bundle_id", "runtime_inventory_sha256", "runtime_sidecar_sha256",
    "runtime_build_verification_sha256",
    "runtime_validation_evidence_sha256",
)
if profile == "candidate":
    if runtime["runtime_contract_validated"] is not True:
        raise SystemExit("candidate runtime contract was not validated")
    if post.get("runtime_library_count") != 13:
        raise SystemExit("candidate runtime receipt does not contain 13 libraries")
    if any(
        not isinstance(runtime[name], str)
        or len(runtime[name]) != 64
        or any(character not in "0123456789abcdef" for character in runtime[name])
        for name in hash_fields
    ):
        raise SystemExit("candidate runtime receipt contains an invalid digest")
    if PurePosixPath(runtime_root).name != runtime["runtime_bundle_id"]:
        raise SystemExit("candidate runtime receipt bundle ID differs from its root")
elif profile == "g0-canonical-pristine":
    if runtime["runtime_contract_validated"] is not False:
        raise SystemExit("legacy G0 runtime receipt has an invalid contract claim")
    if any(runtime[name] is not None for name in hash_fields):
        raise SystemExit("legacy G0 runtime receipt unexpectedly claims bundle evidence")
else:
    raise SystemExit("remote snapshot assembly returned an invalid profile")
print(json.dumps(runtime, sort_keys=True, separators=(",", ":")))
PY
)"
PUBLISHED_CHILD_JSON="$(remote_staging_guard verify-child "${REMOTE_SNAPSHOTS_ROOT}" \
  "${SNAPSHOT_NAME}" "${SNAPSHOTS_DEV}" "${SNAPSHOTS_INO}" \
  "${STAGING_DEV}" "${STAGING_INO}")"
python - "${PUBLISHED_CHILD_JSON}" "${STAGING_DEV}" "${STAGING_INO}" <<'PY'
import json
import sys
value = json.loads(sys.argv[1])
if (value.get("child_dev"), value.get("child_ino")) != (int(sys.argv[2]), int(sys.argv[3])):
    raise SystemExit("published digest child identity differs from the attestation")
PY
REMOTE_VALIDATION="$(remote_ssh "${MTU_PYTHON_BIN}" - \
  "${SNAPSHOT_ROOT}/source" "${MTU_TASK_ROOT}" \
  "${FROZEN_BASE_REVISION}" "${DEPLOYMENT_PROFILE}" <<'PY'
import importlib.util
import json
import sys

source = sys.argv[1]
task_root = sys.argv[2]
base_revision = sys.argv[3]
profile = sys.argv[4]
module_path = source + "/benchmarks/cc/a100_water8/snapshot_manifest.py"
spec = importlib.util.spec_from_file_location("remote_snapshot_manifest", module_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load published snapshot manifest validator")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
evidence = module.validate_snapshot(
    source,
    expected_base_revision=base_revision,
    expected_task_root=task_root,
    expected_deployment_profile=profile,
)
print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
if not evidence["valid"]:
    raise SystemExit("published snapshot strict-v3 validation failed")
PY
)"
python - "${REMOTE_VALIDATION}" "${PUBLISHED_MANIFEST_SHA256}" \
  "${PUBLICATION_ATTESTATION_FILE_SHA256}" "${PUBLICATION_ATTESTATION_SHA256}" \
  "${RUNTIME_RECEIPT_JSON}" "${DEPLOYMENT_PROFILE}" <<'PY'
import json
import sys
evidence = json.loads(sys.argv[1])
if evidence.get("manifest_sha256") != sys.argv[2]:
    raise SystemExit("published manifest hash differs from callback receipt")
if evidence.get("publication_attestation_file_sha256") != sys.argv[3]:
    raise SystemExit("published attestation file hash differs from callback receipt")
attestation = evidence.get("publication_attestation") or {}
if attestation.get("attestation_sha256") != sys.argv[4]:
    raise SystemExit("published attestation hash differs from callback receipt")
runtime_receipt = json.loads(sys.argv[5])
profile = sys.argv[6]
runtime = (evidence.get("manifest") or {}).get("runtime_binaries") or {}
if profile == "candidate":
    expected = {
        "runtime_contract_validated": True,
        "runtime_bundle_id": runtime.get("bundle_id"),
        "runtime_inventory_sha256": runtime.get("inventory_sha256"),
        "runtime_sidecar_sha256": runtime.get("sidecar_sha256"),
        "runtime_build_verification_sha256": runtime.get(
            "build_verification_sha256"
        ),
        "runtime_validation_evidence_sha256": runtime.get(
            "validation_evidence_sha256"
        ),
    }
    if runtime_receipt != expected:
        raise SystemExit(
            "published runtime evidence differs from the callback receipt"
        )
elif runtime_receipt.get("runtime_contract_validated") is not False:
    raise SystemExit("legacy G0 runtime receipt validation claim changed")
PY
PUBLICATION_RECEIPT_OBTAINED=1

# The callback already validated, sealed, and published this same held inode.
# A benchmark process performs the independent strict-v3 runtime validation.
STAGING_ROOT=""
STAGING_NAME=""

python - "${SNAPSHOT_ROOT}/source" "${SNAPSHOT_ROOT}" "${TREE_SHA256}" \
  "${TREE_FILE_COUNT}" "${DEPLOYMENT_PROFILE}" "${RUNTIME_RECEIPT_JSON}" <<'PY'
import json
import sys

runtime = json.loads(sys.argv[6])
print(json.dumps({
    "schema": "gpu4pyscf.snapshot-deployment.v1",
    "source": sys.argv[1],
    "snapshot_root": sys.argv[2],
    "source_sha256": sys.argv[3],
    "source_file_count": int(sys.argv[4]),
    "deployment_profile": sys.argv[5],
    **runtime,
}, sort_keys=True, separators=(",", ":")))
PY
