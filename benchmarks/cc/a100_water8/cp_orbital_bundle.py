"""Content-addressed canonical-orbital bundles for counterpoise jobs.

One bundle contains the cluster orbitals and every complete-basis ghost
fragment.  A producing CP run serializes and reapplies each FP64 MO set before
constructing its post-SCF solver; a consuming run does the same from the saved
bundle.  Scientific identity and source checks fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


BUNDLE_SCHEMA = "gpu4pyscf.water8.cp-canonical-orbitals.v1"
ENTRY_IDENTITY_SCHEMA = "gpu4pyscf.water8.cp-orbital-identity.v1"
ORTHONORMALITY_TOLERANCE = 1.0e-8
HF_ENERGY_TOLERANCE_MIN = 1.0e-7


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def source_contract(source: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "base_revision": source.get("base_revision"),
        "revision": source.get("revision"),
        "tree_sha256": source.get("tree_sha256"),
    }
    if (
        not isinstance(contract["base_revision"], str)
        or not contract["base_revision"]
        or not isinstance(contract["revision"], str)
        or not contract["revision"]
        or not _is_sha256(contract["tree_sha256"])
    ):
        raise ValueError("CP orbital bundle source provenance is incomplete")
    return contract


def case_contract(case: dict[str, Any], atoms: list[list[Any]]) -> dict[str, Any]:
    basis = str(case.get("basis", ""))
    if not basis:
        raise ValueError("CP orbital bundle case basis is missing")
    return {
        "case_id": str(case["id"]),
        "case_source": case.get("source"),
        "basis": basis,
        "unit": "Angstrom",
        "reference": "RHF",
        "spherical": True,
        "charge": 0,
        "spin": 0,
        "cluster_atom_payload": atoms,
        "cluster_atom_payload_sha256": sha256_json(atoms),
    }


def entry_identity(
    *,
    case: dict[str, Any],
    cluster_atoms: list[list[Any]],
    atoms: list[list[Any]],
    role: str,
    entry_key: str,
    fragment_id: int | None,
    active_atom_indices: Iterable[int],
    mol: Any,
) -> dict[str, Any]:
    active = tuple(int(item) for item in active_atom_indices)
    ghost = tuple(
        index
        for index, atom in enumerate(atoms)
        if str(atom[0]).lower().startswith("ghost-")
    )
    nao = int(mol.nao_nr())
    nelectron = int(mol.nelectron)
    nocc = nelectron // 2
    return {
        "schema": ENTRY_IDENTITY_SCHEMA,
        "entry_key": str(entry_key),
        "role": str(role),
        "fragment_id": fragment_id,
        "active_atom_indices": list(active),
        "ghost_atom_indices": list(ghost),
        "atom_payload": atoms,
        "atom_payload_sha256": sha256_json(atoms),
        "cluster_atom_payload_sha256": sha256_json(cluster_atoms),
        "case_id": str(case["id"]),
        "case_source": case.get("source"),
        "basis": str(case["basis"]),
        "unit": "Angstrom",
        "reference": "RHF",
        "spherical": True,
        "basis_on_all_centers": True,
        "charge": int(getattr(mol, "charge", 0)),
        "spin": int(getattr(mol, "spin", 0)),
        "nao": nao,
        "nmo": nao,
        "nocc": nocc,
        "nvir": nao - nocc,
        "nelectron": nelectron,
    }


def _host_float64(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("CP canonical orbital arrays must be real numeric data")
    return np.asarray(array, dtype=np.float64, order="C")


def capture_orbitals(mf: Any) -> dict[str, np.ndarray]:
    return {
        name: _host_float64(getattr(mf, name))
        for name in ("mo_coeff", "mo_occ", "mo_energy")
    }


def orbital_fingerprint(
    identity: dict[str, Any], arrays: dict[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(identity).encode("ascii"))
    for name in ("mo_coeff", "mo_occ", "mo_energy"):
        array = arrays[name]
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_orbitals(
    identity: dict[str, Any],
    arrays: dict[str, np.ndarray],
    mol: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if identity.get("schema") != ENTRY_IDENTITY_SCHEMA:
        raise ValueError(f"{label}: unsupported CP orbital identity schema")
    nao = int(identity["nao"])
    nmo = int(identity["nmo"])
    nocc = int(identity["nocc"])
    nelectron = int(identity["nelectron"])
    arrays.update({name: _host_float64(value) for name, value in arrays.items()})
    coeff = arrays["mo_coeff"]
    occupation = arrays["mo_occ"]
    energy = arrays["mo_energy"]
    if coeff.shape != (nao, nmo):
        raise ValueError(f"{label}: invalid mo_coeff shape {coeff.shape}")
    if occupation.shape != (nmo,) or energy.shape != (nmo,):
        raise ValueError(f"{label}: invalid mo_occ/mo_energy shape")
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError(f"{label}: orbital array contains NaN or infinity")
    expected_occ = np.zeros(nmo, dtype=np.float64)
    expected_occ[:nocc] = 2.0
    if not np.array_equal(occupation, expected_occ):
        raise ValueError(f"{label}: invalid closed-shell RHF occupation")
    if float(occupation.sum()) != float(nelectron):
        raise ValueError(f"{label}: electron count does not match occupations")
    if np.any(np.diff(energy) < -1.0e-8):
        raise ValueError(f"{label}: MO energies are not nondecreasing")
    overlap = _host_float64(mol.intor_symmetric("int1e_ovlp"))
    if overlap.shape != (nao, nao):
        raise ValueError(f"{label}: AO overlap shape is inconsistent")
    orthogonality_error = float(
        np.max(np.abs(coeff.T @ overlap @ coeff - np.eye(nmo)))
    )
    if orthogonality_error > ORTHONORMALITY_TOLERANCE:
        raise ValueError(
            f"{label}: MOs are not AO-metric orthonormal; max error "
            f"{orthogonality_error:.3e}"
        )
    return {
        "orthonormality_max_abs": orthogonality_error,
        "orthonormality_tolerance": ORTHONORMALITY_TOLERANCE,
        "occupation": "closed-shell-rhf-contiguous",
    }


def apply_orbitals(
    mf: Any, arrays: dict[str, np.ndarray], hf_energy: float
) -> None:
    template = getattr(mf, "mo_coeff", None)
    if hasattr(template, "get"):  # pragma: no cover - exercised on A100
        import cupy as backend
    else:
        backend = np
    for name in ("mo_coeff", "mo_occ", "mo_energy"):
        setattr(mf, name, backend.asarray(arrays[name]))
    mf.e_tot = float(hf_energy)


class CPOrbitalBundle:
    """Create or consume one exact-orbital bundle for a complete CP job."""

    def __init__(
        self,
        *,
        mode: str,
        path: Path,
        case: dict[str, Any],
        cluster_atoms: list[list[Any]],
        source: dict[str, Any],
        scf_conv_tol: float,
    ) -> None:
        if mode not in {"output", "input"}:
            raise ValueError("CP orbital bundle mode must be input or output")
        self.mode = mode
        self.path = Path(path)
        self.case = case
        self.cluster_atoms = cluster_atoms
        self.case_contract = case_contract(case, cluster_atoms)
        self.source_contract = source_contract(source)
        self.scf_conv_tol = float(scf_conv_tol)
        if not math.isfinite(self.scf_conv_tol) or self.scf_conv_tol <= 0.0:
            raise ValueError("CP orbital bundle SCF tolerance must be positive")
        self.entries: dict[str, dict[str, Any]] = {}
        self.arrays: dict[str, dict[str, np.ndarray]] = {}
        self.evidence_refs: list[dict[str, Any]] = []
        self.used_keys: set[str] = set()
        self.bundle_sha256: str | None = None
        self.bundle_fingerprint: str | None = None
        self.bundle_bytes: int | None = None
        if mode == "output":
            if self.path.exists():
                raise FileExistsError(
                    f"refusing to overwrite CP orbital bundle: {self.path}"
                )
        else:
            self._load()

    def _load(self) -> None:
        try:
            payload = np.load(self.path, allow_pickle=False)
        except Exception as exc:
            raise ValueError(
                f"cannot load CP orbital bundle {self.path}: {exc}"
            ) from exc
        try:
            if "bundle_schema" not in payload.files or "manifest_json" not in payload.files:
                raise ValueError("CP orbital bundle metadata keys are missing")

            def scalar(name: str) -> str:
                value = np.asarray(payload[name])
                if value.shape != (1,):
                    raise ValueError(f"CP orbital bundle key {name!r} is malformed")
                return str(value[0])

            if scalar("bundle_schema") != BUNDLE_SCHEMA:
                raise ValueError("unsupported CP orbital bundle schema")
            try:
                manifest = json.loads(scalar("manifest_json"))
            except json.JSONDecodeError as exc:
                raise ValueError("CP orbital bundle manifest is malformed") from exc
            if not isinstance(manifest, dict):
                raise ValueError("CP orbital bundle manifest must be an object")
            fingerprint = manifest.get("bundle_fingerprint")
            core = dict(manifest)
            core.pop("bundle_fingerprint", None)
            if not _is_sha256(fingerprint) or sha256_json(core) != fingerprint:
                raise ValueError("CP orbital bundle fingerprint mismatch")
            if core.get("schema") != BUNDLE_SCHEMA:
                raise ValueError("CP orbital bundle core schema is invalid")
            if core.get("case") != self.case_contract:
                raise ValueError("CP orbital bundle case/geometry/basis mismatch")
            if core.get("source") != self.source_contract:
                raise ValueError("CP orbital bundle source tree mismatch")
            if float(core.get("scf_conv_tol", math.nan)) != self.scf_conv_tol:
                raise ValueError("CP orbital bundle SCF tolerance mismatch")
            entries = core.get("entries")
            if not isinstance(entries, dict) or not entries:
                raise ValueError("CP orbital bundle entries are missing")
            expected_keys = {"bundle_schema", "manifest_json"}
            for key, metadata in entries.items():
                if not isinstance(key, str) or not isinstance(metadata, dict):
                    raise ValueError("CP orbital bundle entry metadata is malformed")
                array_keys = {
                    name: f"{key}__{name}"
                    for name in ("mo_coeff", "mo_occ", "mo_energy")
                }
                expected_keys.update(array_keys.values())
                arrays = {
                    name: np.array(payload[array_key], copy=True)
                    for name, array_key in array_keys.items()
                    if array_key in payload.files
                }
                if set(arrays) != set(array_keys):
                    raise ValueError(f"CP orbital bundle entry {key!r} is incomplete")
                identity = metadata.get("identity")
                if not isinstance(identity, dict) or identity.get("entry_key") != key:
                    raise ValueError(f"CP orbital bundle entry {key!r} identity is invalid")
                stored_fingerprint = metadata.get("orbital_fingerprint")
                if (
                    not _is_sha256(stored_fingerprint)
                    or orbital_fingerprint(identity, arrays) != stored_fingerprint
                ):
                    raise ValueError(
                        f"CP orbital bundle entry {key!r} fingerprint mismatch"
                    )
                producer_hf = metadata.get("producer_hf_energy")
                if (
                    isinstance(producer_hf, bool)
                    or not isinstance(producer_hf, (int, float))
                    or not math.isfinite(float(producer_hf))
                ):
                    raise ValueError(
                        f"CP orbital bundle entry {key!r} HF provenance is invalid"
                    )
                self.entries[key] = metadata
                self.arrays[key] = arrays
            if set(payload.files) != expected_keys:
                raise ValueError("CP orbital bundle contains unexpected array keys")
        finally:
            payload.close()
        self.bundle_sha256 = file_sha256(self.path)
        self.bundle_fingerprint = fingerprint
        self.bundle_bytes = int(self.path.stat().st_size)

    def prepare(
        self,
        *,
        entry_key: str,
        role: str,
        fragment_id: int | None,
        active_atom_indices: Iterable[int],
        atoms: list[list[Any]],
        mol: Any,
        mf: Any,
    ) -> dict[str, Any]:
        if entry_key in self.used_keys:
            raise ValueError(f"CP orbital bundle entry {entry_key!r} was reused")
        identity = entry_identity(
            case=self.case,
            cluster_atoms=self.cluster_atoms,
            atoms=atoms,
            role=role,
            entry_key=entry_key,
            fragment_id=fragment_id,
            active_atom_indices=active_atom_indices,
            mol=mol,
        )
        fresh_hf_energy = float(mf.e_tot)
        if self.mode == "output":
            arrays = capture_orbitals(mf)
            validation = validate_orbitals(
                identity, arrays, mol, label=f"CP bundle output {entry_key}"
            )
            fingerprint = orbital_fingerprint(identity, arrays)
            producer_hf_energy = fresh_hf_energy
            metadata = {
                "identity": identity,
                "orbital_fingerprint": fingerprint,
                "producer_hf_energy": producer_hf_energy,
                "scf_conv_tol": self.scf_conv_tol,
                "validation": validation,
            }
            self.entries[entry_key] = metadata
            self.arrays[entry_key] = arrays
        else:
            if entry_key not in self.entries:
                raise ValueError(f"CP orbital bundle entry {entry_key!r} is missing")
            metadata = self.entries[entry_key]
            if metadata.get("identity") != identity:
                raise ValueError(
                    f"CP orbital bundle entry {entry_key!r} atom/ghost identity mismatch"
                )
            if float(metadata.get("scf_conv_tol", math.nan)) != self.scf_conv_tol:
                raise ValueError(
                    f"CP orbital bundle entry {entry_key!r} SCF tolerance mismatch"
                )
            arrays = self.arrays[entry_key]
            validation = validate_orbitals(
                identity, arrays, mol, label=f"CP bundle input {entry_key}"
            )
            fingerprint = orbital_fingerprint(identity, arrays)
            if fingerprint != metadata.get("orbital_fingerprint"):
                raise ValueError(
                    f"CP orbital bundle entry {entry_key!r} fingerprint mismatch"
                )
            producer_hf_energy = float(metadata["producer_hf_energy"])
        hf_delta = abs(fresh_hf_energy - producer_hf_energy)
        hf_tolerance = max(
            HF_ENERGY_TOLERANCE_MIN, 100.0 * self.scf_conv_tol
        )
        if hf_delta > hf_tolerance:
            raise ValueError(
                f"CP orbital bundle entry {entry_key!r} fresh HF energy "
                f"differs by {hf_delta:.3e} Eh"
            )
        apply_orbitals(mf, arrays, producer_hf_energy)
        evidence = {
            "schema": ENTRY_IDENTITY_SCHEMA,
            "entry_key": entry_key,
            "mode": (
                "written-and-reapplied" if self.mode == "output" else "loaded"
            ),
            "identity": identity,
            "orbital_fingerprint": fingerprint,
            "bundle_sha256": self.bundle_sha256,
            "bundle_fingerprint": self.bundle_fingerprint,
            "bundle_complete": self.mode == "input",
            "fresh_hf_energy": fresh_hf_energy,
            "producer_hf_energy": producer_hf_energy,
            "applied_hf_energy": float(mf.e_tot),
            "hf_energy_delta": hf_delta,
            "hf_energy_tolerance": hf_tolerance,
            "validation": validation,
            "applied_before_solver_construction": True,
        }
        self.evidence_refs.append(evidence)
        self.used_keys.add(entry_key)
        return evidence

    def finalize(self, expected_keys: Iterable[str]) -> dict[str, Any]:
        expected = set(expected_keys)
        if self.used_keys != expected:
            raise ValueError(
                "CP orbital bundle use is incomplete: expected "
                f"{sorted(expected)}, used {sorted(self.used_keys)}"
            )
        if set(self.entries) != expected or set(self.arrays) != expected:
            raise ValueError("CP orbital bundle entry set is inconsistent")
        if self.mode == "output":
            core = {
                "schema": BUNDLE_SCHEMA,
                "case": self.case_contract,
                "source": self.source_contract,
                "scf_conv_tol": self.scf_conv_tol,
                "entries": {key: self.entries[key] for key in sorted(expected)},
            }
            self.bundle_fingerprint = sha256_json(core)
            manifest = dict(core)
            manifest["bundle_fingerprint"] = self.bundle_fingerprint
            payload: dict[str, Any] = {
                "bundle_schema": np.asarray([BUNDLE_SCHEMA]),
                "manifest_json": np.asarray([_canonical_json(manifest)]),
            }
            for key in sorted(expected):
                for name, array in self.arrays[key].items():
                    payload[f"{key}__{name}"] = array
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.path.open("xb") as stream:
                    np.savez(stream, **payload)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite CP orbital bundle: {self.path}"
                ) from exc
            self.bundle_sha256 = file_sha256(self.path)
            self.bundle_bytes = int(self.path.stat().st_size)
        for evidence in self.evidence_refs:
            evidence.update({
                "bundle_sha256": self.bundle_sha256,
                "bundle_fingerprint": self.bundle_fingerprint,
                "bundle_complete": True,
            })
        entry_fingerprints = {
            key: self.entries[key]["orbital_fingerprint"]
            for key in sorted(expected)
        }
        match_payload = {
            "bundle_sha256": self.bundle_sha256,
            "bundle_fingerprint": self.bundle_fingerprint,
            "entry_fingerprints": entry_fingerprints,
        }
        return {
            "schema": BUNDLE_SCHEMA,
            "mode": self.mode,
            "path": str(self.path.resolve()),
            "sha256": self.bundle_sha256,
            "bytes": self.bundle_bytes,
            "bundle_fingerprint": self.bundle_fingerprint,
            "entry_count": len(expected),
            "entry_fingerprints": entry_fingerprints,
            "complete": True,
            "case_contract": self.case_contract,
            "source_contract": self.source_contract,
            "record_match_key": sha256_json(match_payload),
        }


def shared_bundle_proof(
    left_record: dict[str, Any], right_record: dict[str, Any]
) -> dict[str, Any]:
    """Prove that two completed CP records used the exact same MO bundle."""

    bundles = []
    for label, record in (("left", left_record), ("right", right_record)):
        bundle = record.get("orbital_bundle")
        if not isinstance(bundle, dict) or bundle.get("complete") is not True:
            raise ValueError(f"{label} CP record has no complete orbital bundle")
        if bundle.get("schema") != BUNDLE_SCHEMA:
            raise ValueError(f"{label} CP record bundle schema is invalid")
        if not _is_sha256(bundle.get("sha256")):
            raise ValueError(f"{label} CP record bundle SHA-256 is invalid")
        if not _is_sha256(bundle.get("bundle_fingerprint")):
            raise ValueError(f"{label} CP record bundle fingerprint is invalid")
        entry_fingerprints = bundle.get("entry_fingerprints")
        if (
            not isinstance(entry_fingerprints, dict)
            or not entry_fingerprints
            or any(not _is_sha256(value) for value in entry_fingerprints.values())
        ):
            raise ValueError(f"{label} CP record entry fingerprints are invalid")
        if (
            isinstance(bundle.get("entry_count"), bool)
            or bundle.get("entry_count") != len(entry_fingerprints)
        ):
            raise ValueError(f"{label} CP record bundle entry count is invalid")
        match_payload = {
            "bundle_sha256": bundle["sha256"],
            "bundle_fingerprint": bundle["bundle_fingerprint"],
            "entry_fingerprints": entry_fingerprints,
        }
        if bundle.get("record_match_key") != sha256_json(match_payload):
            raise ValueError(f"{label} CP record match key is invalid")
        case = record.get("case")
        contract = bundle.get("case_contract")
        source = record.get("source")
        source_identity = bundle.get("source_contract")
        if not all(isinstance(item, dict) for item in (
            case, contract, source, source_identity
        )):
            raise ValueError(f"{label} CP record bundle contracts are missing")
        cluster_payload = contract.get("cluster_atom_payload")
        cluster_payload_hash = contract.get("cluster_atom_payload_sha256")
        if (
            not isinstance(cluster_payload, list)
            or not _is_sha256(cluster_payload_hash)
            or sha256_json(cluster_payload) != cluster_payload_hash
        ):
            raise ValueError(f"{label} CP bundle cluster atom contract is invalid")
        top_case_id = record.get("case_id", case.get("id"))
        if (
            contract.get("case_id") != case.get("id")
            or contract.get("case_id") != top_case_id
            or contract.get("case_source") != case.get("source")
            or contract.get("basis") != case.get("basis")
            or contract.get("unit") != case.get("unit")
            or contract.get("reference") != "RHF"
            or contract.get("spherical") is not True
            or case.get("spherical") is not True
            or contract.get("charge") != 0
            or contract.get("spin") != 0
            or cluster_payload != case.get("geometry_angstrom")
            or cluster_payload_hash != case.get("geometry_sha256")
        ):
            raise ValueError(
                f"{label} CP bundle case/geometry/basis contract does not "
                "match the record"
            )
        if (
            source_identity.get("base_revision")
            != source.get("base_revision")
            or source_identity.get("revision") != source.get("revision")
            or source_identity.get("tree_sha256")
            != source.get("tree_sha256")
            or not _is_sha256(source_identity.get("tree_sha256"))
        ):
            raise ValueError(
                f"{label} CP bundle source contract does not match the record"
            )
        realized: dict[str, dict[str, Any]] = {}
        cluster = record.get("cluster")
        fragments = record.get("fragments")
        if not isinstance(cluster, dict) or not isinstance(fragments, list):
            raise ValueError(f"{label} CP record energy entries are missing")
        if cluster.get("atom_payload_sha256") != cluster_payload_hash:
            raise ValueError(
                f"{label} CP cluster atom evidence does not match the bundle"
            )
        energy_entries = [("cluster", cluster.get("energy"))]
        fragment_atom_hashes: dict[str, str] = {}
        for fragment in fragments:
            if not isinstance(fragment, dict):
                raise ValueError(f"{label} CP fragment record is malformed")
            fragment_id = fragment.get("fragment_id")
            if isinstance(fragment_id, bool) or not isinstance(fragment_id, int):
                raise ValueError(f"{label} CP fragment id is malformed")
            fragment_key = f"fragment-{fragment_id:03d}"
            fragment_hash = fragment.get("atom_payload_sha256")
            if fragment_key in fragment_atom_hashes or not _is_sha256(fragment_hash):
                raise ValueError(
                    f"{label} CP fragment atom evidence is malformed"
                )
            fragment_atom_hashes[fragment_key] = fragment_hash
            energy_entries.append((fragment_key, fragment.get("energy")))
        for expected_key, energy in energy_entries:
            metadata = energy.get("metadata") if isinstance(energy, dict) else None
            evidence = (
                metadata.get("canonical_orbitals")
                if isinstance(metadata, dict) else None
            )
            if not isinstance(evidence, dict):
                raise ValueError(
                    f"{label} CP entry {expected_key!r} lacks orbital evidence"
                )
            if evidence.get("entry_key") != expected_key:
                raise ValueError(
                    f"{label} CP entry {expected_key!r} orbital key mismatch"
                )
            if expected_key in realized:
                raise ValueError(f"{label} CP orbital entry is duplicated")
            if (
                evidence.get("bundle_complete") is not True
                or evidence.get("applied_before_solver_construction") is not True
                or evidence.get("bundle_sha256") != bundle["sha256"]
                or evidence.get("bundle_fingerprint")
                != bundle["bundle_fingerprint"]
                or evidence.get("orbital_fingerprint")
                != entry_fingerprints.get(expected_key)
            ):
                raise ValueError(
                    f"{label} CP entry {expected_key!r} orbital evidence "
                    "does not match the bundle"
                )
            identity = evidence.get("identity")
            if (
                not isinstance(identity, dict)
                or identity.get("entry_key") != expected_key
                or identity.get("atom_payload_sha256")
                != (
                    cluster_payload_hash
                    if expected_key == "cluster"
                    else fragment_atom_hashes.get(expected_key)
                )
            ):
                raise ValueError(
                    f"{label} CP entry {expected_key!r} atom identity mismatch"
                )
            realized[expected_key] = evidence
        if set(realized) != set(entry_fingerprints):
            raise ValueError(
                f"{label} CP record does not realize every bundle entry"
            )
        bundles.append(bundle)
    fields = (
        "sha256", "bundle_fingerprint", "entry_fingerprints",
        "record_match_key", "case_contract", "source_contract",
    )
    mismatched = [
        name for name in fields if bundles[0].get(name) != bundles[1].get(name)
    ]
    if mismatched:
        raise ValueError(
            "CP records did not use the same orbital bundle: "
            + ", ".join(mismatched)
        )
    return {
        "schema": "gpu4pyscf.water8.cp-shared-orbital-proof.v1",
        "matched": True,
        "bundle_sha256": bundles[0]["sha256"],
        "bundle_fingerprint": bundles[0]["bundle_fingerprint"],
        "entry_count": len(bundles[0]["entry_fingerprints"]),
        "record_match_key": bundles[0]["record_match_key"],
    }
