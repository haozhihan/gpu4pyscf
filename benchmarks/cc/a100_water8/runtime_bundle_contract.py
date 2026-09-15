#!/usr/bin/env python3
"""Pure validation for CUDA 12.8/sm_80 runtime-bundle sidecars.

The caller is responsible for opening the sidecar and libraries without
following links and for producing ``observed_inventory`` from those held file
descriptors.  This module deliberately performs no filesystem access.  It
only validates the parsed values and binds them to the exact bytes and public
paths supplied by the caller.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence


BUNDLE_SCHEMA = "gpu4pyscf.runtime-bundle.v2"
BUILD_SCHEMA = "gpu4pyscf.gint-sm80-build.v1"
VALIDATED_EVIDENCE_SCHEMA = "gpu4pyscf.runtime-bundle-validation.v1"
IDENTITY_ALGORITHM = "sha256(canonical-json-inventory-list-no-newline)"
CUDA_VERSION = "12.8"
CUDA_ARCHITECTURE = "80-real"
BUILD_TYPE = "RelWithDebInfo"
ABI_VERSION = 2
EXPECTED_BASIS_PROD_SIZE = 144
EXPECTED_SELECTED_PAIR_SIZE = 112
EXPECTED_SELECTED_OFFSETS = (
    0, 4, 8, 12, 16, 20, 24, 28, 32,
    40, 48, 56, 64, 72, 80, 88, 96, 104,
)
EXPECTED_WORKSPACE_BYTES = {
    "6": 0,
    "7": 600_514_560,
    "8": 1_264_896_000,
}
MAX_STACK_BYTES = 4096
CUDA_TOOL_PATHS = {
    "nvcc": "/usr/local/cuda-12.8/bin/nvcc",
    "cuobjdump": "/usr/local/cuda-12.8/bin/cuobjdump",
}
REQUIRED_SYMBOLS = (
    "GINTfill_selected_int2e_columns",
    "GINTfill_selected_int2e_diagonal",
    "GINTselected_workspace_size",
    "GINTsizeof_basis_prod_cache",
    "GINTsizeof_selected_pair_data",
    "GINToffsetof_selected_pair_data",
    "GINTselected_pair_data_abi_version",
)
REQUIRED_LIBRARIES = (
    "libcupy_helper.so",
    "libgdft.so",
    "libgecp.so",
    "libgint.so",
    "libgvhf.so",
    "libgvhf_md.so",
    "libgvhf_rys.so",
    "libmgrid.so",
    "libmgrid_v2.so",
    "libmgrid_v3.so",
    "libpbc.so",
    "libsem.so",
    "libsolvent.so",
)

_TOP_LEVEL_KEYS = {
    "schema",
    "bundle_id",
    "identity_algorithm",
    "inventory_sha256",
    "complete",
    "created_utc",
    "bundle_root",
    "manifest_path",
    "source_sha256",
    "source_file_count",
    "source_root",
    "base_bundle",
    "build",
    "inventory",
}
_BUILD_KEYS = {
    "type",
    "cuda_version",
    "cuda_architecture",
    "tool_paths",
    "tool_preflight",
    "nvcc_version",
    "verification",
}
_VERIFICATION_KEYS = {
    "path",
    "sha256",
    "bytes",
    "symbols",
    "abi",
    "abi_probe",
    "cuobjdump_path",
    "cubin",
    "stack",
    "ldd",
}
_BUILD_VERIFICATION_KEYS = {
    "schema",
    "source",
    "nvcc",
    "tool_paths",
    "tool_preflight",
    "commands_log_sha256",
    "libgint",
    "inventory",
    "bundle_id",
    "bundle_path",
    "manifest_path",
    "build_root",
}
_BUILD_SOURCE_KEYS = {
    "schema",
    "execute",
    "source_root",
    "source_sha256",
    "source_file_count",
    "base_bundle",
    "base_library_count",
    "output_task_root",
    "build_type",
    "cuda_version",
    "cuda_architecture",
    "required_symbols",
    "tool_paths",
    "abi_version",
    "workspace_bytes",
}
_STACK_KEYS = tuple(
    f"{kind}_rys{order}"
    for kind in ("columns", "diagonal")
    for order in (7, 8)
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00$"
)


class RuntimeBundleContractError(ValueError):
    """Raised when candidate runtime evidence violates the fixed contract."""


def canonical_json_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    """Encode a validated JSON value using the bundle identity algorithm."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeBundleContractError(
            "value cannot be encoded as canonical JSON"
        ) from exc
    return encoded + (b"\n" if trailing_newline else b"")


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeBundleContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_number(token: str) -> Any:
    raise RuntimeBundleContractError(
        f"non-integer JSON number is forbidden: {token!r}"
    )


def _parse_json_object_payload(payload: bytes, *, label: str) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise RuntimeBundleContractError(f"{label} payload must be bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeBundleContractError(f"{label} payload is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_float=_reject_json_number,
            parse_constant=_reject_json_number,
        )
    except RuntimeBundleContractError:
        raise
    except json.JSONDecodeError as exc:
        raise RuntimeBundleContractError(f"{label} payload is not valid JSON") from exc
    if type(value) is not dict:
        raise RuntimeBundleContractError(f"{label} JSON root must be an object")
    _reject_non_json_or_float(value, label=label)
    return value


def parse_runtime_bundle_sidecar(payload: bytes) -> dict[str, Any]:
    """Parse exact sidecar bytes while rejecting duplicates and non-integers."""

    return _parse_json_object_payload(payload, label="sidecar")


def parse_build_verification(payload: bytes) -> dict[str, Any]:
    """Parse detached build evidence with the strict sidecar JSON rules."""

    return _parse_json_object_payload(payload, label="build_verification")


def _reject_non_json_or_float(value: Any, *, label: str) -> None:
    value_type = type(value)
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise RuntimeBundleContractError(
                    f"{label} contains a non-string object key"
                )
            _reject_non_json_or_float(item, label=f"{label}.{key}")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _reject_non_json_or_float(item, label=f"{label}[{index}]")
        return
    if value_type in {str, int, bool}:
        return
    raise RuntimeBundleContractError(
        f"{label} contains forbidden JSON value type {value_type.__name__}"
    )


def _dict(
    value: Any,
    keys: set[str],
    *,
    label: str,
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise RuntimeBundleContractError(f"{label} must be an object")
    optional = set() if optional is None else optional
    actual = set(value)
    missing = keys - actual
    extras = actual - keys - optional
    if missing or extras:
        raise RuntimeBundleContractError(
            f"{label} keys mismatch: missing={sorted(missing)!r}, "
            f"extra={sorted(extras)!r}"
        )
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise RuntimeBundleContractError(f"{label} must be a list")
    return value


def _string(
    value: Any,
    *,
    label: str,
    nonempty: bool = True,
    allow_text_whitespace: bool = False,
) -> str:
    if type(value) is not str or (nonempty and not value):
        raise RuntimeBundleContractError(f"{label} must be a non-empty string")
    allowed_controls = {"\t", "\n", "\r"} if allow_text_whitespace else set()
    if any(
        (ord(character) < 32 and character not in allowed_controls)
        or ord(character) == 127
        for character in value
    ):
        raise RuntimeBundleContractError(f"{label} contains a control character")
    return value


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise RuntimeBundleContractError(f"{label} must be a {relation} integer")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeBundleContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _exact(value: Any, expected: Any, *, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise RuntimeBundleContractError(
            f"{label} mismatch: expected {expected!r}, got {value!r}"
        )


def _absolute_posix_path(value: Any, *, label: str) -> str:
    text = _string(value, label=label)
    if "\\" in text or "\x00" in text:
        raise RuntimeBundleContractError(f"{label} is not a canonical POSIX path")
    path = PurePosixPath(text)
    if (
        not path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or text.startswith("//")
    ):
        raise RuntimeBundleContractError(f"{label} is not a canonical POSIX path")
    return text


def _public_path_argument(value: str | os.PathLike[str], *, label: str) -> str:
    try:
        text = os.fspath(value)
    except TypeError as exc:
        raise RuntimeBundleContractError(f"{label} is not path-like") from exc
    if type(text) is not str:
        raise RuntimeBundleContractError(f"{label} must resolve to text")
    return _absolute_posix_path(text, label=label)


def _validate_timestamp(value: Any) -> str:
    text = _string(value, label="created_utc")
    if _UTC_TIMESTAMP_RE.fullmatch(text) is None:
        raise RuntimeBundleContractError("created_utc is not canonical UTC ISO-8601")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RuntimeBundleContractError("created_utc is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeBundleContractError("created_utc must use UTC")
    return text


def _validate_inventory(value: Any, *, label: str) -> list[dict[str, Any]]:
    entries = _list(value, label=label)
    if len(entries) != len(REQUIRED_LIBRARIES):
        raise RuntimeBundleContractError(
            f"{label} must contain exactly {len(REQUIRED_LIBRARIES)} libraries"
        )
    normalized: list[dict[str, Any]] = []
    for index, (entry, required_name) in enumerate(
        zip(entries, REQUIRED_LIBRARIES, strict=True)
    ):
        item = _dict(
            entry,
            {"name", "bytes", "sha256"},
            label=f"{label}[{index}]",
        )
        name = _string(item["name"], label=f"{label}[{index}].name")
        _exact(name, required_name, label=f"{label}[{index}].name")
        byte_count = _positive_int(
            item["bytes"], label=f"{label}[{index}].bytes"
        )
        digest = _sha256(
            item["sha256"], label=f"{label}[{index}].sha256"
        )
        normalized.append({"name": name, "bytes": byte_count, "sha256": digest})
    return normalized


def _validate_version_record(value: Any, *, tool: str) -> dict[str, Any]:
    record = _dict(
        value,
        {"release", "output", "verified"},
        label=f"{tool} version evidence",
    )
    _exact(record["release"], CUDA_VERSION, label=f"{tool} release")
    _exact(record["verified"], True, label=f"{tool} version verification")
    output = _string(
        record["output"],
        label=f"{tool} version output",
        allow_text_whitespace=True,
    )
    if re.search(r"\brelease\s+12\.8(?:\D|$)", output, re.IGNORECASE) is None:
        raise RuntimeBundleContractError(
            f"{tool} version output does not prove CUDA 12.8"
        )
    return record


def _validate_tool_paths(value: Any, *, label: str) -> dict[str, Any]:
    paths = _dict(value, set(CUDA_TOOL_PATHS), label=label)
    for tool, expected in CUDA_TOOL_PATHS.items():
        _exact(paths[tool], expected, label=f"{label}.{tool}")
    return paths


def _validate_tool_preflight(value: Any) -> dict[str, Any]:
    record = _dict(
        value,
        {"paths", "checks", "nvcc_version", "cuobjdump_version", "verified"},
        label="build.tool_preflight",
    )
    paths = _validate_tool_paths(record["paths"], label="tool_preflight.paths")
    checks = _dict(
        record["checks"], set(CUDA_TOOL_PATHS), label="tool_preflight.checks"
    )
    for tool, expected_path in CUDA_TOOL_PATHS.items():
        check = _dict(
            checks[tool],
            {"path", "regular_file", "executable"},
            label=f"tool_preflight.checks.{tool}",
        )
        _exact(check["path"], expected_path, label=f"{tool} checked path")
        _exact(check["regular_file"], True, label=f"{tool} regular-file check")
        _exact(check["executable"], True, label=f"{tool} executable check")
    _validate_version_record(record["nvcc_version"], tool="nvcc")
    _validate_version_record(record["cuobjdump_version"], tool="cuobjdump")
    _exact(record["verified"], True, label="tool preflight verification")
    return record


def _validate_symbols(value: Any) -> None:
    record = _dict(
        value, {"required", "missing", "present"}, label="build symbols"
    )
    expected = list(REQUIRED_SYMBOLS)
    _exact(record["required"], expected, label="required exported symbols")
    _exact(record["missing"], [], label="missing exported symbols")
    _exact(record["present"], expected, label="present exported symbols")


def _abi_values() -> dict[str, Any]:
    return {
        "basis_prod_cache_size": EXPECTED_BASIS_PROD_SIZE,
        "selected_pair_data_size": EXPECTED_SELECTED_PAIR_SIZE,
        "offsets": list(EXPECTED_SELECTED_OFFSETS),
        "abi_version": ABI_VERSION,
        "workspace_bytes": dict(EXPECTED_WORKSPACE_BYTES),
    }


def _validate_abi(value: Any) -> None:
    record = _dict(value, {"expected", "actual", "verified"}, label="build ABI")
    expected = _abi_values()
    for field in ("expected", "actual"):
        values = _dict(
            record[field], set(expected), label=f"build ABI {field}"
        )
        _exact(values, expected, label=f"build ABI {field}")
    _exact(record["verified"], True, label="build ABI verification")


def _validate_cubin(value: Any) -> None:
    record = _dict(
        value, {"architectures", "sm_cubin", "sm80_only"}, label="cubin"
    )
    architectures = _list(record["architectures"], label="cubin.architectures")
    if (
        not all(type(item) is str for item in architectures)
        or architectures != sorted(set(architectures))
        or "sm_80" not in architectures
        or any(item not in {"sm_80", "compute_80"} for item in architectures)
    ):
        raise RuntimeBundleContractError(
            "cubin architectures must be canonical sm_80/compute_80 evidence"
        )
    _exact(record["sm_cubin"], ["sm_80"], label="sm_80 cubin inventory")
    _exact(record["sm80_only"], True, label="sm_80-only verification")


def _validate_stack(value: Any) -> int:
    record = _dict(
        value, {"records", "max_bytes", "limit_bytes", "verified"}, label="stack"
    )
    records = _dict(record["records"], set(_STACK_KEYS), label="stack.records")
    sizes = [
        _positive_int(records[key], label=f"stack.records.{key}", allow_zero=True)
        for key in _STACK_KEYS
    ]
    maximum = max(sizes)
    if maximum > MAX_STACK_BYTES:
        raise RuntimeBundleContractError("GINT stack frame exceeds 4096 bytes")
    _exact(record["max_bytes"], maximum, label="stack maximum")
    _exact(record["limit_bytes"], MAX_STACK_BYTES, label="stack limit")
    _exact(record["verified"], True, label="stack verification")
    return maximum


def _validate_ldd(value: Any) -> None:
    record = _dict(value, {"missing", "returncode", "verified"}, label="ldd")
    _exact(record["missing"], [], label="ldd missing dependencies")
    _exact(record["returncode"], 0, label="ldd return code")
    _exact(record["verified"], True, label="ldd verification")


def _validate_build(
    value: Any, *, libgint: Mapping[str, Any]
) -> tuple[dict[str, Any], int]:
    build = _dict(value, _BUILD_KEYS, label="build")
    _exact(build["type"], BUILD_TYPE, label="build type")
    _exact(build["cuda_version"], CUDA_VERSION, label="CUDA version")
    _exact(
        build["cuda_architecture"],
        CUDA_ARCHITECTURE,
        label="CUDA architecture",
    )
    tool_paths = _validate_tool_paths(build["tool_paths"], label="build.tool_paths")
    preflight = _validate_tool_preflight(build["tool_preflight"])
    nvcc_version = _validate_version_record(build["nvcc_version"], tool="nvcc")
    _exact(preflight["paths"], tool_paths, label="preflight/build tool paths")
    _exact(
        preflight["nvcc_version"],
        nvcc_version,
        label="preflight/build nvcc evidence",
    )

    verification = _dict(
        build["verification"], _VERIFICATION_KEYS, label="build.verification"
    )
    path = _absolute_posix_path(
        verification["path"], label="built libgint path"
    )
    if PurePosixPath(path).name != "libgint.so":
        raise RuntimeBundleContractError("built library path is not libgint.so")
    _exact(verification["sha256"], libgint["sha256"], label="libgint SHA-256")
    _exact(verification["bytes"], libgint["bytes"], label="libgint byte count")
    _validate_symbols(verification["symbols"])
    _validate_abi(verification["abi"])
    abi_probe = _dict(
        verification["abi_probe"],
        {"mode", "executable", "verified"},
        label="ABI probe",
    )
    _exact(abi_probe["mode"], "subprocess", label="ABI probe mode")
    _absolute_posix_path(abi_probe["executable"], label="ABI probe executable")
    _exact(abi_probe["verified"], True, label="ABI probe verification")
    _exact(
        verification["cuobjdump_path"],
        CUDA_TOOL_PATHS["cuobjdump"],
        label="verification cuobjdump path",
    )
    _validate_cubin(verification["cubin"])
    stack_maximum = _validate_stack(verification["stack"])
    _validate_ldd(verification["ldd"])
    return build, stack_maximum


def _validate_build_verification(
    value: Any,
    *,
    sidecar: Mapping[str, Any],
    inventory: list[dict[str, Any]],
    build: Mapping[str, Any],
    public_bundle_root: str,
    public_sidecar_path: str,
    public_build_verification_path: str,
) -> None:
    record = _dict(
        value, _BUILD_VERIFICATION_KEYS, label="build_verification"
    )
    _exact(record["schema"], BUILD_SCHEMA, label="build verification schema")
    _exact(record["bundle_id"], sidecar["bundle_id"], label="verified bundle ID")
    _exact(record["bundle_path"], public_bundle_root, label="verified bundle path")
    _exact(
        record["manifest_path"],
        public_sidecar_path,
        label="verified manifest path",
    )
    build_root = PurePosixPath(
        _absolute_posix_path(record["build_root"], label="verified build root")
    )
    expected_build_name = (
        f"{sidecar['source_sha256']}-gint-bounded-workspace-v2-sm80"
    )
    if (
        build_root.name != expected_build_name
        or build_root.parent.name != "builds"
        or build_root.parent.parent
        != PurePosixPath(public_bundle_root).parent.parent
    ):
        raise RuntimeBundleContractError(
            "verified build root does not agree with source and task identity"
        )
    _exact(
        public_build_verification_path,
        str(build_root / "build-verification.json"),
        label="public build-verification path binding",
    )
    _sha256(record["commands_log_sha256"], label="commands log SHA-256")
    _exact(record["inventory"], inventory, label="verified inventory")
    _exact(record["libgint"], build["verification"], label="verified libgint")
    _exact(record["nvcc"], build["nvcc_version"], label="verified nvcc")
    _exact(record["tool_paths"], build["tool_paths"], label="verified tool paths")
    _exact(
        record["tool_preflight"],
        build["tool_preflight"],
        label="verified tool preflight",
    )

    source = _dict(record["source"], _BUILD_SOURCE_KEYS, label="build source")
    _exact(source["schema"], BUILD_SCHEMA, label="build source schema")
    _exact(source["execute"], False, label="build source execute flag")
    for key in ("source_root", "base_bundle"):
        _exact(source[key], sidecar[key], label=f"build source {key}")
    _exact(
        source["source_sha256"],
        sidecar["source_sha256"],
        label="build source SHA-256",
    )
    _exact(
        source["source_file_count"],
        sidecar["source_file_count"],
        label="build source file count",
    )
    _exact(source["base_library_count"], 13, label="base library count")
    task_root = str(PurePosixPath(public_bundle_root).parent.parent)
    _exact(source["output_task_root"], task_root, label="build output task root")
    _exact(source["build_type"], BUILD_TYPE, label="build source build type")
    _exact(source["cuda_version"], CUDA_VERSION, label="build source CUDA")
    _exact(
        source["cuda_architecture"],
        CUDA_ARCHITECTURE,
        label="build source CUDA architecture",
    )
    _exact(
        source["required_symbols"],
        list(REQUIRED_SYMBOLS),
        label="build source required symbols",
    )
    _exact(source["tool_paths"], CUDA_TOOL_PATHS, label="build source tools")
    _exact(source["abi_version"], ABI_VERSION, label="build source ABI")
    _exact(
        source["workspace_bytes"],
        EXPECTED_WORKSPACE_BYTES,
        label="build source workspaces",
    )


def validate_runtime_bundle_sidecar(
    sidecar: Mapping[str, Any],
    observed_inventory: Sequence[Mapping[str, Any]],
    *,
    sidecar_payload: bytes,
    public_bundle_root: str | os.PathLike[str],
    public_sidecar_path: str | os.PathLike[str],
    expected_source_sha256: str,
    expected_source_file_count: int,
    build_verification: Mapping[str, Any] | None = None,
    build_verification_payload: bytes | None = None,
    public_build_verification_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate and bind one securely observed candidate runtime bundle.

    ``sidecar_payload`` is mandatory so the returned evidence identifies the
    exact bytes read by the caller.  The payload is parsed again here; this
    both rejects duplicate keys and proves that ``sidecar`` represents those
    same bytes.  When the builder's detached ``build-verification.json`` is
    available, its parsed object and exact payload must be supplied together;
    they are bound to the same source, inventory, tools, and public paths.
    """

    if type(sidecar) is not dict:
        raise RuntimeBundleContractError("parsed sidecar must be a plain object")
    _reject_non_json_or_float(sidecar, label="sidecar")
    parsed = parse_runtime_bundle_sidecar(sidecar_payload)
    if parsed != sidecar:
        raise RuntimeBundleContractError(
            "parsed sidecar does not match the supplied sidecar payload"
        )
    supplied_build_evidence = (
        build_verification,
        build_verification_payload,
        public_build_verification_path,
    )
    if any(item is not None for item in supplied_build_evidence) and not all(
        item is not None for item in supplied_build_evidence
    ):
        raise RuntimeBundleContractError(
            "detached build verification, payload, and public path must be supplied together"
        )
    parsed_build_verification = None
    public_build_verification = None
    if build_verification is not None:
        if type(build_verification) is not dict:
            raise RuntimeBundleContractError(
                "parsed build verification must be a plain object"
            )
        _reject_non_json_or_float(
            build_verification, label="build_verification"
        )
        assert build_verification_payload is not None
        parsed_build_verification = parse_build_verification(
            build_verification_payload
        )
        if parsed_build_verification != build_verification:
            raise RuntimeBundleContractError(
                "parsed build verification does not match its supplied payload"
            )
        assert public_build_verification_path is not None
        public_build_verification = _public_path_argument(
            public_build_verification_path,
            label="public build-verification path",
        )
    public_root = _public_path_argument(
        public_bundle_root, label="public bundle root"
    )
    public_manifest = _public_path_argument(
        public_sidecar_path, label="public sidecar path"
    )
    source_sha256 = _sha256(
        expected_source_sha256, label="expected source SHA-256"
    )
    source_count = _positive_int(
        expected_source_file_count, label="expected source file count"
    )

    value = _dict(sidecar, _TOP_LEVEL_KEYS, label="sidecar")
    _exact(value["schema"], BUNDLE_SCHEMA, label="runtime bundle schema")
    _exact(value["complete"], True, label="runtime bundle completeness")
    _exact(
        value["identity_algorithm"],
        IDENTITY_ALGORITHM,
        label="runtime bundle identity algorithm",
    )
    _validate_timestamp(value["created_utc"])
    bundle_id = _sha256(value["bundle_id"], label="bundle ID")
    inventory_sha256 = _sha256(
        value["inventory_sha256"], label="inventory SHA-256"
    )
    _exact(value["bundle_root"], public_root, label="public bundle root binding")
    _exact(
        value["manifest_path"],
        public_manifest,
        label="public sidecar path binding",
    )
    bundle_path = PurePosixPath(public_root)
    manifest_path = PurePosixPath(public_manifest)
    if (
        bundle_path.name != bundle_id
        or bundle_path.parent.name != "runtime-bundles"
        or manifest_path.name != f"{bundle_id}.json"
        or manifest_path.parent.name != "runtime-bundle-manifests"
        or bundle_path.parent.parent != manifest_path.parent.parent
    ):
        raise RuntimeBundleContractError(
            "bundle root, bundle ID, and sidecar path do not agree"
        )
    _exact(value["source_sha256"], source_sha256, label="source SHA-256")
    _exact(value["source_file_count"], source_count, label="source file count")
    _absolute_posix_path(value["source_root"], label="source root")
    _absolute_posix_path(value["base_bundle"], label="base bundle")

    inventory = _validate_inventory(value["inventory"], label="sidecar inventory")
    observed = _validate_inventory(
        observed_inventory, label="observed library inventory"
    )
    _exact(inventory, observed, label="sidecar/observed library inventory")
    inventory_payload = canonical_json_bytes(inventory, trailing_newline=False)
    calculated_bundle_id = hashlib.sha256(inventory_payload).hexdigest()
    _exact(bundle_id, calculated_bundle_id, label="bundle ID/inventory digest")
    _exact(
        inventory_sha256,
        calculated_bundle_id,
        label="inventory SHA-256/inventory digest",
    )

    libgint = inventory[REQUIRED_LIBRARIES.index("libgint.so")]
    build, maximum_stack = _validate_build(value["build"], libgint=libgint)
    has_build_verification = build_verification is not None
    if build_verification is not None:
        _validate_build_verification(
            build_verification,
            sidecar=value,
            inventory=inventory,
            build=build,
            public_bundle_root=public_root,
            public_sidecar_path=public_manifest,
            public_build_verification_path=public_build_verification,
        )

    canonical_sidecar = canonical_json_bytes(value, trailing_newline=False)
    evidence = {
        "schema": VALIDATED_EVIDENCE_SCHEMA,
        "validated": True,
        "bundle_id": bundle_id,
        "bundle_root": public_root,
        "manifest_path": public_manifest,
        "library_count": len(inventory),
        "inventory_sha256": calculated_bundle_id,
        "inventory_payload_sha256": hashlib.sha256(inventory_payload).hexdigest(),
        "source_sha256": source_sha256,
        "source_file_count": source_count,
        "cuda_version": CUDA_VERSION,
        "cuda_architecture": CUDA_ARCHITECTURE,
        "abi_version": ABI_VERSION,
        "maximum_stack_bytes": maximum_stack,
        "build_verification_bound": has_build_verification,
        "sidecar_sha256": hashlib.sha256(sidecar_payload).hexdigest(),
        "sidecar_bytes": len(sidecar_payload),
        "payload_sha256": hashlib.sha256(canonical_sidecar).hexdigest(),
    }
    if build_verification is not None:
        assert build_verification_payload is not None
        assert parsed_build_verification is not None
        evidence.update({
            "build_verification_sha256": hashlib.sha256(
                build_verification_payload
            ).hexdigest(),
            "build_verification_bytes": len(build_verification_payload),
            "build_verification_payload_sha256": hashlib.sha256(
                canonical_json_bytes(parsed_build_verification)
            ).hexdigest(),
            "build_verification_path": public_build_verification,
        })
    return evidence


__all__ = [
    "ABI_VERSION",
    "BUNDLE_SCHEMA",
    "BUILD_SCHEMA",
    "CUDA_ARCHITECTURE",
    "CUDA_TOOL_PATHS",
    "CUDA_VERSION",
    "EXPECTED_BASIS_PROD_SIZE",
    "EXPECTED_SELECTED_OFFSETS",
    "EXPECTED_SELECTED_PAIR_SIZE",
    "EXPECTED_WORKSPACE_BYTES",
    "IDENTITY_ALGORITHM",
    "MAX_STACK_BYTES",
    "REQUIRED_LIBRARIES",
    "REQUIRED_SYMBOLS",
    "RuntimeBundleContractError",
    "VALIDATED_EVIDENCE_SCHEMA",
    "canonical_json_bytes",
    "parse_build_verification",
    "parse_runtime_bundle_sidecar",
    "validate_runtime_bundle_sidecar",
]
