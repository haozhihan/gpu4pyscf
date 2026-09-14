"""CPU-only tests for the auditable counterpoise construction and formula."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("water8_counterpoise", ROOT / "counterpoise.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "water8_benchmark_for_cp", ROOT / "benchmark.py"
)
assert BENCHMARK_SPEC is not None and BENCHMARK_SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(BENCHMARK_SPEC)
sys.modules[BENCHMARK_SPEC.name] = BENCHMARK
BENCHMARK_SPEC.loader.exec_module(BENCHMARK)
SNAPSHOT_SPEC = importlib.util.spec_from_file_location(
    "water8_snapshot_manifest_for_cp", ROOT / "snapshot_manifest.py"
)
assert SNAPSHOT_SPEC is not None and SNAPSHOT_SPEC.loader is not None
SNAPSHOT = importlib.util.module_from_spec(SNAPSHOT_SPEC)
SNAPSHOT_SPEC.loader.exec_module(SNAPSHOT)


def _bind_candidate_runtime_evidence(
    manifest: dict, *, task_root: Path, source_file_count: int
) -> None:
    task_root = task_root.resolve()
    runtime = manifest["runtime_binaries"]
    inventory = [
        {
            "name": name,
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
        for name, entry in zip(
            SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES,
            runtime["files"],
            strict=True,
        )
    ]
    bundle_id = hashlib.sha256(json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()
    source_sha256 = manifest["tree_sha256"]
    bundle_root = str(task_root / "runtime-bundles" / bundle_id)
    evidence = {
        "schema": SNAPSHOT.RUNTIME_VALIDATION_EVIDENCE_SCHEMA,
        "validated": True,
        "bundle_id": bundle_id,
        "bundle_root": bundle_root,
        "manifest_path": str(
            task_root / "runtime-bundle-manifests" / f"{bundle_id}.json"
        ),
        "build_verification_path": str(
            task_root / "builds"
            / f"{source_sha256}-gint-bounded-workspace-v2-sm80"
            / "build-verification.json"
        ),
        "library_count": len(inventory),
        "inventory_sha256": bundle_id,
        "inventory_payload_sha256": bundle_id,
        "source_sha256": source_sha256,
        "source_file_count": source_file_count,
        "cuda_version": "12.8",
        "cuda_architecture": "80-real",
        "abi_version": 2,
        "maximum_stack_bytes": 1024,
        "build_verification_bound": True,
        "sidecar_sha256": "1" * 64,
        "sidecar_bytes": 1024,
        "payload_sha256": "2" * 64,
        "build_verification_sha256": "3" * 64,
        "build_verification_bytes": 2048,
        "build_verification_payload_sha256": "4" * 64,
    }
    runtime["source_root"] = bundle_root
    runtime["contract_schema"] = SNAPSHOT.RUNTIME_BUNDLE_SCHEMA
    runtime["bundle_id"] = evidence["bundle_id"]
    runtime["inventory_sha256"] = evidence["inventory_sha256"]
    runtime["sidecar_sha256"] = evidence["sidecar_sha256"]
    runtime["build_verification_sha256"] = evidence[
        "build_verification_sha256"
    ]
    runtime["validation_evidence"] = evidence
    runtime["validation_evidence_sha256"] = SNAPSHOT._canonical_json_sha256(
        evidence
    )


def _qualification_lineage(*, qualification_sha256: str) -> dict:
    attestation = {
        "schema": SNAPSHOT.PUBLICATION_ATTESTATION_SCHEMA,
        "status": "published",
        "source_tree_sha256": qualification_sha256,
        "snapshot_name": qualification_sha256,
        "published_directory_identity": {"device": 1, "inode": 2},
        "publish_protocol": "atomic-exclusive-rename",
        "trust_boundary": SNAPSHOT.PUBLISH_TRUST_BOUNDARY,
        "publication_policy_sha256": SNAPSHOT._canonical_json_sha256(
            SNAPSHOT.PUBLICATION_POLICY
        ),
        "published_manifest_sha256": "5" * 64,
        "prepublication_manifest_sha256": "5" * 64,
    }
    attestation["attestation_sha256"] = SNAPSHOT._canonical_json_sha256(
        attestation
    )
    return {
        "schema": SNAPSHOT.RELEASE_PIN_SCHEMA,
        "qualification_source_tree_sha256": qualification_sha256,
        "release_pin_relative_path": SNAPSHOT.RELEASE_PIN_RELATIVE_PATH.as_posix(),
        "release_pin_sha256": "6" * 64,
        "receipt_sha256": "7" * 64,
        "receipt_payload_sha256": "8" * 64,
        "qualification_publication_attestation": attestation,
        "qualification_publication_attestation_sha256": attestation[
            "attestation_sha256"
        ],
    }


def _local_provenance() -> dict:
    allowed = {
        "prefixes": ["benchmarks/cc/a100_water8/"],
        "files": ["gpu4pyscf/cc/device_runtime.py"],
    }
    entries: list[str] = []
    policy = json.dumps(
        {"allowed_untracked": allowed, "status": entries},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "gpu4pyscf.local-provenance.v1",
        "deployment_profile": "candidate",
        "repository_root": "/local/repository",
        "head_revision": MODULE.FROZEN_BASE_COMMIT,
        "frozen_base_revision": MODULE.FROZEN_BASE_COMMIT,
        "head_matches_frozen_base": True,
        "tracked_pristine": True,
        "untracked_paths_allowed": True,
        "canonical_pristine": False,
        "allowed_untracked": allowed,
        "git_status": {
            "format": "porcelain-v1-lines",
            "sha256": hashlib.sha256(b"").hexdigest(),
            "allowed_paths_status_sha256": hashlib.sha256(policy).hexdigest(),
            "entry_count": 0,
            "entries": entries,
        },
    }


def _installed_snapshot(
    directory: str,
    *,
    manifest_mode: str = "valid",
    manifest_schema: str = "v3",
    read_only: bool = True,
) -> tuple[Path, Path]:
    task = Path(directory) / "task"
    staging = task / "snapshots" / "staging"
    source = staging / "source"
    source.mkdir(parents=True)
    (source / "module.py").write_text("answer = 42\n", encoding="utf-8")
    binary_root = source / "gpu4pyscf" / "lib"
    binary_root.mkdir(parents=True)
    (source / "gpu4pyscf" / "runtime_targets.py").write_text(
        "from gpu4pyscf.lib.utils import load_library\n"
        + "".join(
            f"load_library({name.removesuffix('.so')!r})\n"
            for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
        ),
        encoding="utf-8",
    )
    for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES:
        (binary_root / name).write_bytes(
            f"counterpoise-runtime-binary:{name}\n".encode()
        )
    digest, _ = MODULE._source_tree_digest(source)
    target = task / "snapshots" / digest
    staging.rename(target)
    source = target / "source"
    binaries = [
        source / "gpu4pyscf" / "lib" / name
        for name in SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
    ]
    if manifest_mode != "missing":
        manifest = {
            "schema": (
                SNAPSHOT.SOURCE_SNAPSHOT_SCHEMA
                if manifest_schema == "v3"
                else SNAPSHOT.LEGACY_SOURCE_SNAPSHOT_SCHEMA
            ),
            "tree_sha256": digest,
            "base_revision": MODULE.FROZEN_BASE_COMMIT,
            "source": str(source.resolve()),
            "immutable": True,
            "local_provenance": _local_provenance(),
            "runtime_binaries": {
                "source_root": "/pinned/gpu4pyscf/lib",
                "complete": True,
                "required_library_names": list(
                    SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
                ),
                "inventory_library_names": list(
                    SNAPSHOT.REQUIRED_RUNTIME_LIBRARIES
                ),
                "closure_complete": True,
                "files": [{
                    "relative_path": f"gpu4pyscf/lib/{item.name}",
                    "sha256": MODULE._file_sha256(item),
                    "bytes": item.stat().st_size,
                } for item in binaries],
            },
        }
        if manifest_schema == "v3":
            inventory = {
                "schema": SNAPSHOT.SOURCE_NORMALIZATION_SCHEMA,
                "policy": SNAPSHOT.SOURCE_NORMALIZATION_POLICY,
                "links": [],
                "link_count": 0,
                "regular_file_count_before": 3,
                "directory_count_before": 4,
            }
            encoded = (
                json.dumps(
                    inventory, sort_keys=True, separators=(",", ":")
                ) + "\n"
            ).encode("ascii")
            normalization = {
                "schema": SNAPSHOT.SOURCE_NORMALIZATION_SCHEMA,
                "policy": SNAPSHOT.SOURCE_NORMALIZATION_POLICY,
                "source_link_inventory": inventory,
                "source_link_inventory_sha256": hashlib.sha256(
                    encoded
                ).hexdigest(),
                "materialized_link_count": 0,
                "materialized_links": [],
                "published_symlink_count": 0,
                "published_special_file_count": 0,
                "regular_file_count_after": 3,
                "directory_count_after": 4,
            }
            uploaded = (
                json.dumps(normalization, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            normalization.update({
                "remote_verified_after_runtime_binding": True,
                "remote_regular_file_count_after_runtime_binding": 16,
                "remote_directory_count_after_runtime_binding": 4,
                "uploaded_record_sha256": hashlib.sha256(
                    uploaded
                ).hexdigest(),
                "uploaded_record_bytes": len(uploaded),
            })
            manifest["source_normalization"] = normalization
            manifest["publication_policy"] = SNAPSHOT.PUBLICATION_POLICY
            _, source_file_count = MODULE._source_tree_digest(source)
            _bind_candidate_runtime_evidence(
                manifest,
                task_root=task,
                source_file_count=source_file_count,
            )
        if manifest_mode == "invalid":
            manifest["immutable"] = False
        if manifest_mode == "runtime_invalid":
            manifest["runtime_binaries"]["files"][0]["sha256"] = "0" * 64
        (target / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        if manifest_schema == "v3":
            manifest_bytes = (target / "manifest.json").read_bytes()
            target_info = target.stat()
            attestation = {
                "schema": SNAPSHOT.PUBLICATION_ATTESTATION_SCHEMA,
                "status": "published",
                "source_tree_sha256": digest,
                "snapshot_name": digest,
                "published_directory_identity": {
                    "device": target_info.st_dev,
                    "inode": target_info.st_ino,
                },
                "publish_protocol": "atomic-exclusive-rename",
                "trust_boundary": SNAPSHOT.PUBLISH_TRUST_BOUNDARY,
                "publication_policy_sha256": SNAPSHOT._canonical_json_sha256(
                    SNAPSHOT.PUBLICATION_POLICY
                ),
                "published_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "prepublication_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            }
            attestation["attestation_sha256"] = SNAPSHOT._canonical_json_sha256(
                attestation
            )
            sidecar = target.parent / (
                f"{digest}{SNAPSHOT.PUBLICATION_ATTESTATION_SUFFIX}"
            )
            sidecar.write_text(json.dumps(attestation, sort_keys=True) + "\n")
            sidecar.chmod(0o444)
    if read_only:
        os.chmod(source / "module.py", 0o444)
        os.chmod(source / "gpu4pyscf" / "runtime_targets.py", 0o444)
        for binary in binaries:
            os.chmod(binary, 0o444)
        os.chmod(binaries[0].parent, 0o555)
        os.chmod(binaries[0].parent.parent, 0o555)
        os.chmod(source, 0o555)
        if (target / "manifest.json").exists():
            os.chmod(target / "manifest.json", 0o444)
        os.chmod(target, 0o555)
    return task, source


def _make_snapshot_writable(source: Path) -> None:
    target = source.parent
    os.chmod(target, 0o755)
    os.chmod(source, 0o755)
    os.chmod(source / "module.py", 0o644)
    os.chmod(source / "gpu4pyscf" / "runtime_targets.py", 0o644)
    binary_root = source / "gpu4pyscf" / "lib"
    os.chmod(binary_root.parent, 0o755)
    os.chmod(binary_root, 0o755)
    for binary in binary_root.glob("*.so"):
        os.chmod(binary, 0o644)
    manifest = target / "manifest.json"
    if manifest.exists():
        os.chmod(manifest, 0o644)


def _bundle_source(tree: str = "a" * 64) -> dict:
    return {
        "base_revision": MODULE.FROZEN_BASE_COMMIT,
        "revision": f"tree-sha256:{tree}",
        "tree_sha256": tree,
        "stable_during_run": False,
        "formal_counterpoise_eligible": False,
    }


class _BundleMolecule:
    charge = 0
    spin = 0
    nelectron = 2

    @staticmethod
    def nao_nr() -> int:
        return 3

    @staticmethod
    def intor_symmetric(name: str) -> np.ndarray:
        assert name == "int1e_ovlp"
        return np.eye(3)


class _BundleMeanField:
    converged = True

    def __init__(self, *, rotated: bool = False):
        if rotated:
            angle = 0.3
            self.mo_coeff = np.asarray([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ])
            self.e_tot = -1.0 + 1.0e-9
        else:
            self.mo_coeff = np.eye(3)
            self.e_tot = -1.0
        self.mo_occ = np.asarray([2.0, 0.0, 0.0])
        self.mo_energy = np.asarray([-1.0, 0.5, 1.0])


class _BundleSolver:
    converged = True

    def __init__(self, mf, constructed_coefficients):
        constructed_coefficients.append(np.array(mf.mo_coeff, copy=True))

    @staticmethod
    def kernel():
        return -0.1, None, None


def _complete_fake_source(source: dict) -> bool:
    source["tree_sha256_at_end"] = source["tree_sha256"]
    source["stable_during_run"] = True
    source["formal_counterpoise_eligible"] = True
    return True


class CounterpoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = MODULE.load_cases(ROOT / "cases.json")

    def test_water8_fragments_are_disjoint_complete_monomers(self) -> None:
        for case_id in ("water8-tz", "water8s4-tz"):
            case = self.cases[case_id]
            fragments = MODULE.infer_water_fragments(case["geometry_angstrom"])
            self.assertEqual(len(fragments), 8)
            flattened = [
                index for fragment in fragments for index in fragment.atom_indices
            ]
            self.assertEqual(sorted(flattened), list(range(24)))
            for fragment in fragments:
                self.assertEqual(len(fragment.atom_indices), 3)
                symbols = [
                    case["geometry_angstrom"][index][0]
                    for index in fragment.atom_indices
                ]
                self.assertEqual(symbols.count("O"), 1)
                self.assertEqual(symbols.count("H"), 2)

    def test_ghost_fragment_keeps_full_cluster_basis(self) -> None:
        case = self.cases["water2-tz"]
        fragment = MODULE.infer_water_fragments(case["geometry_angstrom"])[0]
        atoms = MODULE.fragment_atoms(case["geometry_angstrom"], fragment)
        self.assertEqual(len(atoms), len(case["geometry_angstrom"]))
        active = set(fragment.atom_indices)
        for index, atom in enumerate(atoms):
            if index in active:
                self.assertFalse(atom[0].lower().startswith("ghost-"))
            else:
                self.assertTrue(atom[0].lower().startswith("ghost-"))
        self.assertEqual(atoms[0][1:], [float(value) for value in case["geometry_angstrom"][0][1:]])

    def test_counterpoise_formula_and_units(self) -> None:
        interaction = MODULE.counterpoise_energy(-100.0, [-12.0, -13.0])
        self.assertAlmostEqual(interaction, -75.0)
        self.assertAlmostEqual(interaction * MODULE.KCAL_PER_EH, -75.0 * 627.5094740631)

    def test_hash_is_stable_and_write_is_restart_safe(self) -> None:
        payload = [["O", 0.0, 0.0, 0.0]]
        self.assertEqual(MODULE.sha256_json(payload), MODULE.sha256_json(json.loads(json.dumps(payload))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            MODULE._write_json_once(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                MODULE._write_json_once(path, {"ok": False})

    def test_source_manifest_has_deterministic_start_end_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.py").write_text("b = 2\n")
            (root / "a.py").write_text("a = 1\n")
            (root / "results").mkdir()
            (root / "results" / "ignored.json").write_text("{}\n")
            first = MODULE._source_tree_digest(root)
            second = MODULE._source_tree_digest(root)
            self.assertEqual(first, second)
            with mock.patch.object(MODULE, "REPOSITORY_ROOT", root), mock.patch.object(
                MODULE, "_git_revision", return_value="revision"
            ):
                state = MODULE._source_state()
                self.assertFalse(MODULE._complete_source_stability(state))
                self.assertFalse(state["stable_during_run"])
                self.assertFalse(state["formal_counterpoise_eligible"])
                self.assertEqual(
                    state["tree_sha256_at_start"], state["tree_sha256_at_end"]
                )
                changed = MODULE._source_state()
                (root / "a.py").write_text("a = 3\n")
                self.assertFalse(MODULE._complete_source_stability(changed))
                self.assertNotEqual(
                    changed["tree_sha256_at_start"],
                    changed["tree_sha256_at_end"],
                )

    def test_formal_snapshot_evidence_valid_mutable_missing_and_invalid(self) -> None:
        for label, manifest_mode, read_only, manifest_schema, expected in (
            ("valid-v3", "valid", True, "v3", True),
            ("legacy-v2", "valid", True, "v2", False),
            ("mutable", "valid", False, "v3", False),
            ("missing", "missing", True, "v3", False),
            ("invalid", "invalid", True, "v3", False),
            ("runtime-invalid", "runtime_invalid", True, "v3", False),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                task, source_root = _installed_snapshot(
                    directory,
                    manifest_mode=manifest_mode,
                    manifest_schema=manifest_schema,
                    read_only=read_only,
                )
                try:
                    with mock.patch.object(MODULE, "REPOSITORY_ROOT", source_root), \
                            mock.patch.object(MODULE, "_git_revision", return_value=None), \
                            mock.patch.dict(
                                os.environ,
                                {
                                    "CCSD_TASK_ROOT": str(task),
                                    "CCSD_EXPECTED_DEPLOYMENT_PROFILE": "candidate",
                                    "CCSD_REQUIRE_CANONICAL_PRISTINE": "0",
                                },
                                clear=False,
                            ):
                        state = MODULE._source_state()
                        completed = MODULE._complete_source_stability(state)
                    self.assertEqual(completed, expected)
                    self.assertEqual(
                        state["formal_counterpoise_eligible"], expected
                    )
                    if expected:
                        snapshot = state["snapshot"]
                        self.assertTrue(snapshot["valid_at_start"])
                        self.assertTrue(snapshot["valid_at_end"])
                        self.assertTrue(snapshot["read_only_at_start"])
                        self.assertTrue(snapshot["read_only_at_end"])
                        self.assertEqual(
                            snapshot["manifest_sha256_at_start"],
                            snapshot["manifest_sha256_at_end"],
                        )
                        self.assertEqual(
                            state["tree_sha256_at_start"],
                            state["tree_sha256_at_end"],
                        )
                finally:
                    _make_snapshot_writable(source_root)

    def test_runtime_manifest_enforces_candidate_evidence_and_g0_legacy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task, source = _installed_snapshot(directory)
            try:
                manifest = json.loads(
                    (source.parent / "manifest.json").read_text()
                )
                digest, count = MODULE._source_tree_digest(source)
                arguments = {
                    "source_tree_sha256": digest,
                    "source_file_count": count,
                    "task_root": task.resolve(),
                }
                self.assertTrue(MODULE._runtime_manifest_valid(
                    manifest, **arguments
                ))

                for mutation in ("missing", "hash", "source", "inventory", "path"):
                    with self.subTest(mutation=mutation):
                        changed = copy.deepcopy(manifest)
                        runtime = changed["runtime_binaries"]
                        if mutation == "missing":
                            runtime.pop("validation_evidence")
                            runtime.pop("validation_evidence_sha256")
                        elif mutation == "hash":
                            runtime["validation_evidence"]["sidecar_bytes"] += 1
                        elif mutation == "source":
                            runtime["validation_evidence"]["source_sha256"] = (
                                "9" * 64
                            )
                            runtime["validation_evidence_sha256"] = (
                                SNAPSHOT._canonical_json_sha256(
                                    runtime["validation_evidence"]
                                )
                            )
                        elif mutation == "inventory":
                            runtime["files"][0]["sha256"] = "0" * 64
                        else:
                            runtime["validation_evidence"]["bundle_root"] = (
                                f"/other/runtime-bundles/"
                                f"{runtime['validation_evidence']['bundle_id']}"
                            )
                            runtime["validation_evidence_sha256"] = (
                                SNAPSHOT._canonical_json_sha256(
                                    runtime["validation_evidence"]
                                )
                            )
                        self.assertFalse(MODULE._runtime_manifest_valid(
                            changed, **arguments
                        ))

                for field, replacement in (
                    ("contract_schema", "gpu4pyscf.runtime-bundle.v1"),
                    ("bundle_id", "e" * 64),
                    ("inventory_sha256", "e" * 64),
                    ("sidecar_sha256", "e" * 64),
                    ("build_verification_sha256", "e" * 64),
                ):
                    with self.subTest(alias=field):
                        changed = copy.deepcopy(manifest)
                        changed["runtime_binaries"][field] = replacement
                        self.assertFalse(MODULE._runtime_manifest_valid(
                            changed, **arguments
                        ))

                extra = copy.deepcopy(manifest)
                extra["runtime_binaries"]["unsealed_note"] = "unexpected"
                self.assertFalse(MODULE._runtime_manifest_valid(
                    extra, **arguments
                ))

                g0 = copy.deepcopy(manifest)
                g0["local_provenance"]["deployment_profile"] = (
                    MODULE.G0_DEPLOYMENT_PROFILE
                )
                g0["runtime_binaries"].pop("validation_evidence")
                g0["runtime_binaries"].pop("validation_evidence_sha256")
                self.assertTrue(MODULE._runtime_manifest_valid(g0))

                release_sha = "d" * 64
                release = copy.deepcopy(manifest)
                release["tree_sha256"] = release_sha
                release["gint_release_lineage"] = _qualification_lineage(
                    qualification_sha256=digest
                )
                self.assertTrue(MODULE._runtime_manifest_valid(
                    release,
                    source_tree_sha256=release_sha,
                    source_file_count=count + 1,
                    task_root=task.resolve(),
                ))
                self.assertFalse(MODULE._runtime_manifest_valid(
                    release,
                    source_tree_sha256=release_sha,
                    source_file_count=count + 2,
                    task_root=task.resolve(),
                ))
            finally:
                _make_snapshot_writable(source)

    def test_fno_energy_uses_delta_mp2_correction(self) -> None:
        class MeanField:
            e_tot = -80.0
            converged = True

        class Solver:
            converged = True
            corrected_e_corr = -1.0

            def kernel(self):
                return -0.9, None, None

        def factory(mol, method, device, options):
            return MeanField(), Solver()

        record = MODULE.evaluate_energy(
            object(), method="fno", device="cpu", solver_factory=factory
        )
        self.assertEqual(record.correlation_energy_eh, -1.0)
        self.assertEqual(record.total_energy_eh, -81.0)

    def test_energy_record_keeps_realized_rr_thc_solver_metadata(self) -> None:
        class Scalar:
            def item(self):
                return 17

        class MeanField:
            e_tot = -80.0
            converged = True

        class Solver:
            converged = True

            def __init__(self):
                self.kernel_completed = False

            def kernel(self):
                self.kernel_completed = True
                return -1.0, None, None

            def method_metadata(self):
                if not self.kernel_completed:
                    raise AssertionError(
                        "realized solver metadata was sampled before kernel()"
                    )
                return {
                    "rr_projector": {"rank": Scalar()},
                    "thc_projector_factors": {"rank": 17},
                    "replacement_applied": False,
                    "inexact_thc_enabled": False,
                    "inexact_thc_blocker": (
                        "inexact amplitude THC is fail-closed"
                    ),
                    "precision": "fp64",
                }

        record = MODULE.evaluate_energy(
            object(),
            method="thc_cd",
            device="gpu",
            solver_factory=lambda *args: (MeanField(), Solver()),
        )
        realized = record.metadata["solver_method_metadata"]
        self.assertEqual(realized["rr_projector"]["rank"], 17)
        self.assertEqual(realized["thc_projector_factors"]["rank"], 17)
        self.assertIs(realized["replacement_applied"], False)
        self.assertIs(realized["inexact_thc_enabled"], False)
        self.assertIn("fail-closed", realized["inexact_thc_blocker"])
        self.assertEqual(realized["precision"], "fp64")
        json.dumps(record.metadata, allow_nan=False)

    def test_cp_persists_distinct_realized_metadata_for_every_energy(self) -> None:
        class MeanField:
            e_tot = -80.0
            converged = True

        created = []

        class Solver:
            converged = True

            def __init__(self, ordinal):
                self.ordinal = ordinal
                self.kernel_completed = False

            def kernel(self):
                self.kernel_completed = True
                return -1.0, None, None

            def method_metadata(self):
                if not self.kernel_completed:
                    raise AssertionError("metadata must describe the completed solve")
                return {
                    "rr_projector": {"rank": 10 + self.ordinal},
                    "thc_projector_factors": {"rank": 20 + self.ordinal},
                    "replacement_applied": False,
                    "inexact_thc_enabled": False,
                    "inexact_thc_blocker": "fail-closed validation endpoint",
                    "precision": "fp64",
                }

        def factory(*_args):
            solver = Solver(len(created))
            created.append(solver)
            return MeanField(), solver

        source = {
            "stable_during_run": False,
            "formal_counterpoise_eligible": False,
        }
        with mock.patch.object(MODULE, "_source_state", return_value=source), \
                mock.patch.object(
                    MODULE, "_complete_source_stability", return_value=False
                ), \
                mock.patch.object(MODULE, "build_molecule", return_value=object()):
            record = MODULE.run_counterpoise(
                self.cases["water2-tz"],
                method="thc_cd",
                device="gpu",
                solver_factory=factory,
            )

        # One cluster plus two water monomers, each with its own realized
        # post-kernel solver state rather than a copy of requested options.
        self.assertEqual(len(created), 3)
        energies = [record["cluster"]["energy"]] + [
            fragment["energy"] for fragment in record["fragments"]
        ]
        self.assertEqual(
            [
                energy["metadata"]["solver_method_metadata"]
                ["rr_projector"]["rank"]
                for energy in energies
            ],
            [10, 11, 12],
        )
        for energy in energies:
            realized = energy["metadata"]["solver_method_metadata"]
            self.assertIs(realized["replacement_applied"], False)
            self.assertIs(realized["inexact_thc_enabled"], False)
            self.assertEqual(realized["precision"], "fp64")

    def test_low_rank_cp_rejects_solver_without_method_metadata(self) -> None:
        class MeanField:
            e_tot = -80.0
            converged = True

        class Solver:
            converged = True

            def kernel(self):
                return -1.0, None, None

        with self.assertRaisesRegex(TypeError, "must expose method_metadata"):
            MODULE.evaluate_energy(
                object(),
                method="rr_cd",
                device="gpu",
                solver_factory=lambda *args: (MeanField(), Solver()),
            )

    def test_energy_evaluation_rejects_unconverged_scf_or_cc(self) -> None:
        class MeanField:
            e_tot = -80.0

            def __init__(self, converged):
                self.converged = converged

        class Solver:
            def __init__(self, converged):
                self.converged = converged

            def kernel(self):
                return -1.0, None, None

        with self.assertRaisesRegex(RuntimeError, "SCF is unconverged"):
            MODULE.evaluate_energy(
                object(),
                method="canonical",
                device="cpu",
                solver_factory=lambda *args: (MeanField(False), Solver(True)),
            )
        with self.assertRaisesRegex(RuntimeError, "CCSD is unconverged"):
            MODULE.evaluate_energy(
                object(),
                method="canonical",
                device="cpu",
                solver_factory=lambda *args: (MeanField(True), Solver(False)),
            )

    def test_cp_record_exposes_protocol_memory_and_source_stability(self) -> None:
        energy = MODULE.EnergyRecord(
            total_energy_eh=-1.0,
            hf_energy_eh=-0.9,
            correlation_energy_eh=-0.1,
            converged=True,
            wall_time_s=0.1,
            metadata={"scf_converged": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            task, source_root = _installed_snapshot(directory)
            try:
                with mock.patch.object(MODULE, "REPOSITORY_ROOT", source_root), \
                        mock.patch.object(MODULE, "build_molecule", return_value=object()), \
                        mock.patch.object(MODULE, "evaluate_energy", return_value=energy), \
                        mock.patch.dict(
                            os.environ,
                            {
                                "CCSD_TASK_ROOT": str(task),
                                "CCSD_EXPECTED_DEPLOYMENT_PROFILE": "candidate",
                                "CCSD_REQUIRE_CANONICAL_PRISTINE": "0",
                            },
                            clear=False,
                        ):
                    record = MODULE.run_counterpoise(
                        self.cases["water2-tz"],
                        method="canonical",
                        device="cpu",
                        options={"cc_max_cycle": 37, "max_memory_mb": 4567},
                    )
            finally:
                _make_snapshot_writable(source_root)
        self.assertFalse(record["accuracy_eligible"])
        self.assertIsNone(record["orbital_bundle"])
        self.assertTrue(any(
            "canonical orbital bundle" in limitation
            for limitation in record["limitations"]
        ))
        self.assertFalse(record["performance_eligible"])
        self.assertTrue(record["source"]["stable_during_run"])
        self.assertTrue(record["source"]["formal_counterpoise_eligible"])
        self.assertEqual(record["protocol"]["cc_max_cycle"], 37)
        self.assertEqual(record["protocol"]["max_memory_mb"], 4567)
        self.assertEqual(record["cc_max_cycle"], 37)
        self.assertEqual(record["max_memory_mb"], 4567)

    def test_cp_bundle_roundtrip_applies_orbitals_before_solver_and_proves_pair(
        self,
    ) -> None:
        case = self.cases["water2-tz"]
        constructed_output: list[np.ndarray] = []
        constructed_input: list[np.ndarray] = []

        def factory(constructed, *, rotated):
            def build(mol, method, device, options):
                del method, device
                mf = _BundleMeanField(rotated=rotated)
                request = options["_cp_orbital_request"]
                evidence = request["manager"].prepare(
                    entry_key=request["entry_key"],
                    role=request["role"],
                    fragment_id=request["fragment_id"],
                    active_atom_indices=request["active_atom_indices"],
                    atoms=request["atoms"],
                    mol=mol,
                    mf=mf,
                )
                mf._cp_canonical_orbitals = evidence
                return mf, _BundleSolver(mf, constructed)

            return build

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "water2-cp-orbitals.npz"
            common_patches = (
                mock.patch.object(
                    MODULE, "build_molecule", return_value=_BundleMolecule()
                ),
                mock.patch.object(
                    MODULE, "_complete_source_stability",
                    side_effect=_complete_fake_source,
                ),
                mock.patch.object(
                    MODULE, "_formal_source_evidence_valid", return_value=True
                ),
            )
            with mock.patch.object(
                MODULE, "_source_state", side_effect=lambda: _bundle_source()
            ), common_patches[0], common_patches[1], common_patches[2]:
                oracle = MODULE.run_counterpoise(
                    case,
                    method="canonical",
                    device="cpu",
                    options={"orbital_bundle_out": bundle},
                    solver_factory=factory(constructed_output, rotated=False),
                )
            self.assertTrue(bundle.is_file())
            with mock.patch.object(
                MODULE, "_source_state", side_effect=lambda: _bundle_source()
            ), mock.patch.object(
                MODULE, "build_molecule", return_value=_BundleMolecule()
            ), mock.patch.object(
                MODULE, "_complete_source_stability",
                side_effect=_complete_fake_source,
            ), mock.patch.object(
                MODULE, "_formal_source_evidence_valid", return_value=True
            ):
                candidate = MODULE.run_counterpoise(
                    case,
                    method="fno",
                    device="cpu",
                    options={"orbital_bundle_in": bundle},
                    solver_factory=factory(constructed_input, rotated=True),
                )

            self.assertTrue(oracle["accuracy_eligible"])
            self.assertTrue(candidate["accuracy_eligible"])
            self.assertEqual(oracle["orbital_bundle"]["entry_count"], 3)
            self.assertEqual(
                oracle["orbital_bundle"]["sha256"],
                candidate["orbital_bundle"]["sha256"],
            )
            proof = MODULE.shared_bundle_proof(oracle, candidate)
            self.assertTrue(proof["matched"])
            self.assertEqual(proof["entry_count"], 3)
            self.assertEqual(len(constructed_output), 3)
            self.assertEqual(len(constructed_input), 3)
            for coefficients in constructed_output + constructed_input:
                self.assertTrue(np.array_equal(coefficients, np.eye(3)))
            energies = [candidate["cluster"]["energy"]] + [
                fragment["energy"] for fragment in candidate["fragments"]
            ]
            for energy in energies:
                evidence = energy["metadata"]["canonical_orbitals"]
                self.assertEqual(evidence["bundle_sha256"], proof["bundle_sha256"])
                self.assertEqual(evidence["mode"], "loaded")
                self.assertTrue(evidence["applied_before_solver_construction"])
                self.assertEqual(evidence["applied_hf_energy"], -1.0)

            # Complete atom/ghost payloads are part of every entry identity.
            with np.load(bundle, allow_pickle=False) as payload:
                manifest = json.loads(payload["manifest_json"][0])
            cluster_identity = manifest["entries"]["cluster"]["identity"]
            fragment_identity = manifest["entries"]["fragment-000"]["identity"]
            self.assertEqual(
                cluster_identity["atom_payload"],
                MODULE.molecule_atom_payload(case["geometry_angstrom"]),
            )
            self.assertEqual(fragment_identity["role"], "ghost-monomer")
            self.assertTrue(fragment_identity["ghost_atom_indices"])
            self.assertTrue(fragment_identity["basis_on_all_centers"])

    def test_cp_bundle_rejects_case_source_and_array_tampering(self) -> None:
        case = self.cases["water2-tz"]
        geometry = MODULE.molecule_atom_payload(case["geometry_angstrom"])
        source = _bundle_source()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.npz"
            manager = MODULE.CPOrbitalBundle(
                mode="output",
                path=path,
                case=case,
                cluster_atoms=geometry,
                source=source,
                scf_conv_tol=1e-10,
            )
            expected = {"cluster"}
            manager.prepare(
                entry_key="cluster",
                role="cluster",
                fragment_id=None,
                active_atom_indices=range(len(geometry)),
                atoms=geometry,
                mol=_BundleMolecule(),
                mf=_BundleMeanField(),
            )
            manager.finalize(expected)

            wrong_case = dict(case)
            wrong_case["id"] = "different-case"
            with self.assertRaisesRegex(ValueError, "case/geometry/basis mismatch"):
                MODULE.CPOrbitalBundle(
                    mode="input",
                    path=path,
                    case=wrong_case,
                    cluster_atoms=geometry,
                    source=source,
                    scf_conv_tol=1e-10,
                )
            with self.assertRaisesRegex(ValueError, "source tree mismatch"):
                MODULE.CPOrbitalBundle(
                    mode="input",
                    path=path,
                    case=case,
                    cluster_atoms=geometry,
                    source=_bundle_source("b" * 64),
                    scf_conv_tol=1e-10,
                )

            with np.load(path, allow_pickle=False) as original:
                payload = {
                    name: np.array(original[name], copy=True)
                    for name in original.files
                }
            payload["cluster__mo_energy"][1] += 0.1
            tampered = Path(directory) / "tampered.npz"
            np.savez(tampered, **payload)
            with self.assertRaisesRegex(ValueError, "entry 'cluster' fingerprint"):
                MODULE.CPOrbitalBundle(
                    mode="input",
                    path=tampered,
                    case=case,
                    cluster_atoms=geometry,
                    source=source,
                    scf_conv_tol=1e-10,
                )

    def test_shared_bundle_proof_rejects_different_or_tampered_records(self) -> None:
        atoms = [["O", 0.0, 0.0, 0.0]]
        atom_hash = MODULE.sha256_json(atoms)
        base = {
            "case_id": "water2-tz",
            "case": {
                "id": "water2-tz",
                "source": "test",
                "basis": "cc-pVTZ",
                "unit": "Angstrom",
                "spherical": True,
                "geometry_angstrom": atoms,
                "geometry_sha256": atom_hash,
            },
            "source": {
                "base_revision": MODULE.FROZEN_BASE_COMMIT,
                "revision": "tree-sha256:" + "d" * 64,
                "tree_sha256": "d" * 64,
            },
            "orbital_bundle": {
                "schema": "gpu4pyscf.water8.cp-canonical-orbitals.v1",
                "complete": True,
                "sha256": "a" * 64,
                "bundle_fingerprint": "b" * 64,
                "entry_count": 1,
                "entry_fingerprints": {"cluster": "c" * 64},
                "case_contract": {
                    "case_id": "water2-tz",
                    "case_source": "test",
                    "basis": "cc-pVTZ",
                    "unit": "Angstrom",
                    "reference": "RHF",
                    "spherical": True,
                    "charge": 0,
                    "spin": 0,
                    "cluster_atom_payload": atoms,
                    "cluster_atom_payload_sha256": atom_hash,
                },
                "source_contract": {
                    "base_revision": MODULE.FROZEN_BASE_COMMIT,
                    "revision": "tree-sha256:" + "d" * 64,
                    "tree_sha256": "d" * 64,
                },
            },
            "cluster": {
                "atom_payload_sha256": atom_hash,
                "energy": {"metadata": {"canonical_orbitals": {
                    "entry_key": "cluster",
                    "bundle_complete": True,
                    "applied_before_solver_construction": True,
                    "bundle_sha256": "a" * 64,
                    "bundle_fingerprint": "b" * 64,
                    "orbital_fingerprint": "c" * 64,
                    "identity": {
                        "entry_key": "cluster",
                        "atom_payload_sha256": atom_hash,
                    },
                }}},
            },
            "fragments": [],
        }
        payload = {
            "bundle_sha256": "a" * 64,
            "bundle_fingerprint": "b" * 64,
            "entry_fingerprints": {"cluster": "c" * 64},
        }
        base["orbital_bundle"]["record_match_key"] = MODULE.sha256_json(payload)
        other = json.loads(json.dumps(base))
        self.assertTrue(MODULE.shared_bundle_proof(base, other)["matched"])
        other["orbital_bundle"]["sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "match key is invalid"):
            MODULE.shared_bundle_proof(base, other)

        malformed = json.loads(json.dumps(base))
        malformed["fragments"] = [{
            "fragment_id": "zero",
            "atom_payload_sha256": "e" * 64,
            "energy": {"metadata": {}},
        }]
        with self.assertRaisesRegex(ValueError, "fragment id is malformed"):
            MODULE.shared_bundle_proof(base, malformed)

    def test_real_solver_factory_applies_input_bundle_before_constructor(self) -> None:
        case = self.cases["water2-tz"]
        geometry = MODULE.molecule_atom_payload(case["geometry_angstrom"])
        source = _bundle_source()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster-only.npz"
            producer = MODULE.CPOrbitalBundle(
                mode="output",
                path=path,
                case=case,
                cluster_atoms=geometry,
                source=source,
                scf_conv_tol=1e-10,
            )
            producer.prepare(
                entry_key="cluster",
                role="cluster",
                fragment_id=None,
                active_atom_indices=range(len(geometry)),
                atoms=geometry,
                mol=_BundleMolecule(),
                mf=_BundleMeanField(),
            )
            producer.finalize({"cluster"})
            consumer = MODULE.CPOrbitalBundle(
                mode="input",
                path=path,
                case=case,
                cluster_atoms=geometry,
                source=source,
                scf_conv_tol=1e-10,
            )

            class MeanField(_BundleMeanField):
                def __init__(self):
                    super().__init__(rotated=True)

                def kernel(self):
                    return self.e_tot

            class Solver:
                def __init__(self, mf):
                    self.constructor_mo_coeff = np.array(mf.mo_coeff, copy=True)

            pyscf_module = types.ModuleType("pyscf")
            scf_module = types.ModuleType("pyscf.scf")
            cc_module = types.ModuleType("pyscf.cc")
            scf_module.RHF = lambda mol: MeanField()
            cc_module.CCSD = Solver
            pyscf_module.scf = scf_module
            request = {
                "manager": consumer,
                "entry_key": "cluster",
                "role": "cluster",
                "fragment_id": None,
                "active_atom_indices": list(range(len(geometry))),
                "atoms": geometry,
            }
            with mock.patch.dict(sys.modules, {
                "pyscf": pyscf_module,
                "pyscf.scf": scf_module,
                "pyscf.cc": cc_module,
            }):
                mf, solver = MODULE._make_solver(
                    _BundleMolecule(),
                    "canonical",
                    "cpu",
                    {"_cp_orbital_request": request},
                )
            self.assertTrue(np.array_equal(solver.constructor_mo_coeff, np.eye(3)))
            self.assertEqual(mf.e_tot, -1.0)
            self.assertTrue(
                mf._cp_canonical_orbitals[
                    "applied_before_solver_construction"
                ]
            )

    def test_formal_snapshot_requires_an_explicit_deployment_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task, source_root = _installed_snapshot(directory)
            try:
                with mock.patch.object(MODULE, "REPOSITORY_ROOT", source_root), \
                        mock.patch.object(MODULE, "_git_revision", return_value=None), \
                        mock.patch.dict(
                            os.environ,
                            {
                                "CCSD_TASK_ROOT": str(task),
                                "CCSD_EXPECTED_DEPLOYMENT_PROFILE": "",
                                "CCSD_REQUIRE_CANONICAL_PRISTINE": "0",
                            },
                            clear=False,
                        ):
                    state = MODULE._source_state()
                    self.assertFalse(MODULE._complete_source_stability(state))
            finally:
                _make_snapshot_writable(source_root)
        self.assertIn(
            "expected deployment profile is not configured",
            state["snapshot"]["reasons"],
        )

    def test_fno_signature_separates_cutoffs_and_obeys_precedence(self) -> None:
        first, first_signature = MODULE.approximation_identity(
            "fno", {"fno_thresh": 1e-5}
        )
        _, second_signature = MODULE.approximation_identity(
            "fno", {"fno_thresh": 1e-6}
        )
        rank, rank_signature = MODULE.approximation_identity(
            "fno", {"fno_thresh": 1e-4, "fno_nvir_act": 12}
        )
        _, same_rank_signature = MODULE.approximation_identity(
            "fno", {"fno_thresh": 1e-8, "fno_nvir_act": 12}
        )
        self.assertEqual(
            first["selection"],
            {"mode": "occupation_threshold", "value": 1e-5},
        )
        self.assertNotEqual(first_signature, second_signature)
        self.assertEqual(rank["selection"], {"mode": "nvir_act", "value": 12})
        self.assertEqual(rank_signature, same_rank_signature)
        benchmark_payload, benchmark_signature = BENCHMARK.approximation_identity(
            "fno",
            types.SimpleNamespace(
                fno_thresh=1e-5, fno_pct_occ=None, fno_nvir_act=None
            ),
        )
        self.assertEqual(first, benchmark_payload)
        self.assertEqual(first_signature, benchmark_signature)

    def test_rr_cd_signature_matches_benchmark_exactly(self) -> None:
        options = {
            "eri_tol": 1e-8,
            "direct_scf_tol": 1e-13,
            "cd_max_rank": 321,
            "gint_group_size": 16,
            "gint_max_block_bytes": 268435456,
            "cd_mo_block_size": 32,
            "rr_eig_cutoff": 1e-6,
            "rr_max_rank": 123,
            "denominator_tolerance": 1e-10,
            "denominator_max_rank": 222,
            "rr_initial_rank": 32,
            "rr_solver_tolerance": 1e-10,
            "rr_solver_maxiter": 77,
            "rr_dense_fallback_dimension": 64,
            "rr_ritz_residual_tolerance": 1e-8,
            "rr_auxiliary_block_size": 2,
            "rr_virtual_block_size": 8,
            "rr_ring_kernel": "gemm",
            "precision": "fp64",
        }
        cp_payload, cp_signature = MODULE.approximation_identity(
            "rr_cd", options
        )
        benchmark_payload, benchmark_signature = BENCHMARK.approximation_identity(
            "rr_cd", types.SimpleNamespace(**options)
        )
        self.assertEqual(cp_payload, benchmark_payload)
        self.assertEqual(cp_signature, benchmark_signature)

    def test_thc_cd_signature_matches_benchmark_exactly(self) -> None:
        options = {
            "eri_tol": 1e-8,
            "direct_scf_tol": 1e-13,
            "cd_max_rank": 321,
            "rr_eig_cutoff": 1e-6,
            "rr_max_rank": 123,
            "denominator_tolerance": 1e-10,
            "denominator_max_rank": 222,
            "rr_solver_tolerance": 1e-10,
            "rr_solver_maxiter": 77,
            "rr_dense_fallback_dimension": 64,
            "rr_ritz_residual_tolerance": 1e-8,
            "precision": "fp64",
            "thc_fit_tol": 1e-6,
            "thc_rank": 456,
            "thc_initial_rank": None,
            "thc_max_rank": 456,
            "thc_rank_growth": 1.5,
            "thc_orthogonality_cutoff": 1e-12,
            "thc_orthogonality_tolerance": 1e-10,
            "thc_max_iterations": 500,
            "thc_als_convergence_tolerance": 1e-10,
            "thc_ridge": 1e-12,
            "thc_seed": 7,
            "thc_replacement_validation_atol": 1e-10,
            "thc_replacement_validation_rtol": 1e-9,
        }
        cp_payload, cp_signature = MODULE.approximation_identity(
            "thc_cd", options
        )
        benchmark_payload, benchmark_signature = BENCHMARK.approximation_identity(
            "thc_cd", types.SimpleNamespace(**options)
        )
        self.assertEqual(cp_payload, benchmark_payload)
        self.assertEqual(cp_signature, benchmark_signature)

    def test_thc_cd_contract_names_shared_r123_endpoint(self) -> None:
        self.assertEqual(
            MODULE.METHOD_STATUS["thc_cd"],
            (
                "controlled-approximation",
                "direct-cd-r123-complement-full-pair-thc-endpoint",
            ),
        )

    def test_dry_run_is_explicitly_accuracy_ineligible(self) -> None:
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            status = MODULE.main([
                "--case", "water2-tz",
                "--method", "rr_cd",
                "--output", "/tmp/unused-counterpoise.json",
                "--dry-run",
            ])
        self.assertEqual(status, 0)
        plan = json.loads(output.getvalue())
        self.assertIs(plan["accuracy_eligible"], False)
        self.assertTrue(any("dry-run" in item for item in plan["limitations"]))

    def test_cp_orbital_bundle_cli_is_explicit_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.npz"
            output_bundle = root / "output.npz"
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                MODULE.main([
                    "--case", "water2-tz",
                    "--method", "canonical",
                    "--output", str(root / "result.json"),
                    "--orbital-bundle-in", str(missing),
                    "--orbital-bundle-out", str(output_bundle),
                    "--dry-run",
                ])
            with self.assertRaisesRegex(ValueError, "input does not exist"):
                MODULE.main([
                    "--case", "water2-tz",
                    "--method", "canonical",
                    "--output", str(root / "result.json"),
                    "--orbital-bundle-in", str(missing),
                    "--dry-run",
                ])
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                status = MODULE.main([
                    "--case", "water2-tz",
                    "--method", "canonical",
                    "--output", str(root / "result.json"),
                    "--orbital-bundle-out", str(output_bundle),
                    "--dry-run",
                ])
            self.assertEqual(status, 0)
            plan = json.loads(stdout.getvalue())
            self.assertEqual(
                plan["orbital_bundle"]["output"], str(output_bundle)
            )
            self.assertTrue(plan["orbital_bundle"]["required_for_accuracy"])

    def test_mtu_launcher_pins_physical8_and_external_artifacts(self) -> None:
        launcher = (ROOT / "run_mtu_counterpoise.sbatch").read_text(
            encoding="utf-8"
        )
        for required in (
            "CCSD_SOURCE_ROOT:?",
            "snapshot_manifest.py",
            "--mode physical8",
            "--require-performance",
            "CP_OUTPUT_ROOT",
            '"${TASK_ROOT}/results/"*',
            "OMP_NUM_THREADS=8",
            "GPU4PYSCF_NUMA=3",
            "PYTHONNOUSERSITE=1",
            "counterpoise output must be outside",
            "cache and scratch paths must be outside",
        ):
            self.assertIn(required, launcher)
        self.assertNotIn('"$@"', launcher)

    def test_cc_runtime_controls_are_not_passed_to_constructor(self) -> None:
        class MeanField:
            converged = True
            e_tot = -1.0

            def kernel(self):
                return self.e_tot

        class Solver:
            def __init__(self, mf):
                self.mf = mf

        pyscf_module = types.ModuleType("pyscf")
        scf_module = types.ModuleType("pyscf.scf")
        cc_module = types.ModuleType("pyscf.cc")
        scf_module.RHF = lambda mol: MeanField()
        cc_module.CCSD = Solver
        pyscf_module.scf = scf_module
        with mock.patch.dict(
            sys.modules,
            {"pyscf": pyscf_module, "pyscf.scf": scf_module, "pyscf.cc": cc_module},
        ):
            _, solver = MODULE._make_solver(
                object(),
                "canonical",
                "cpu",
                {
                    "cc_conv_tol": 2e-9,
                    "cc_conv_tol_normt": 3e-7,
                    "cc_max_cycle": 41,
                    "max_memory_mb": 1234,
                },
            )
        self.assertEqual(solver.conv_tol, 2e-9)
        self.assertEqual(solver.conv_tol_normt, 3e-7)
        self.assertEqual(solver.max_cycle, 41)
        self.assertEqual(solver.max_memory, 1234)

    def test_rr_cd_solver_uses_public_direct_cd_controls(self) -> None:
        class MeanField:
            converged = True
            e_tot = -1.0

            def kernel(self):
                return self.e_tot

        class Solver:
            def __init__(self, mf, **kwargs):
                self.mf = mf
                self.constructor_options = kwargs

        gpu_module = types.ModuleType("gpu4pyscf")
        gpu_module.__path__ = []
        scf_module = types.ModuleType("gpu4pyscf.scf")
        scf_module.RHF = lambda mol: MeanField()
        cc_module = types.ModuleType("gpu4pyscf.cc")
        cc_module.__path__ = []
        rr_module = types.ModuleType("gpu4pyscf.cc.rrccsd")
        rr_module.RRCCSD = Solver
        gpu_module.scf = scf_module
        gpu_module.cc = cc_module
        cc_module.rrccsd = rr_module
        options = {
            "eri_tol": 2e-8,
            "rr_eig_cutoff": 3e-7,
            "direct_scf_tol": 4e-13,
            "cd_max_rank": 101,
            "gint_column_backend": "restricted-reference",
            "gint_group_size": 12,
            "gint_max_block_bytes": 123456,
            "gint_max_batch_size": 11,
            "cd_mo_block_size": 24,
            "denominator_tolerance": 5e-11,
            "denominator_max_rank": 99,
            "rr_initial_rank": 20,
            "rr_solver_tolerance": 6e-10,
            "rr_solver_maxiter": 55,
            "rr_dense_fallback_dimension": 0,
            "rr_ritz_residual_tolerance": 7e-9,
            "rr_auxiliary_block_size": 3,
            "rr_virtual_block_size": 6,
            "rr_ring_kernel": "gemm",
            "precision": "fp64",
            "cc_max_cycle": 41,
            "max_memory_mb": 1234,
        }
        with mock.patch.dict(sys.modules, {
            "gpu4pyscf": gpu_module,
            "gpu4pyscf.scf": scf_module,
            "gpu4pyscf.cc": cc_module,
            "gpu4pyscf.cc.rrccsd": rr_module,
        }):
            _, solver = MODULE._make_solver(
                object(), "rr_cd", "gpu", options
            )
        self.assertEqual(solver.constructor_options["eri_backend"], "cd")
        for name in (
            "eri_tol", "rr_eig_cutoff", "direct_scf_tol", "cd_max_rank",
            "gint_column_backend", "gint_group_size", "gint_max_block_bytes",
            "gint_max_batch_size", "cd_mo_block_size",
            "denominator_tolerance", "denominator_max_rank",
            "rr_initial_rank", "rr_solver_tolerance", "rr_solver_maxiter",
            "rr_dense_fallback_dimension", "rr_ritz_residual_tolerance",
            "rr_auxiliary_block_size", "rr_virtual_block_size",
            "rr_ring_kernel", "precision",
        ):
            self.assertEqual(solver.constructor_options[name], options[name])
        self.assertEqual(solver.max_cycle, 41)
        self.assertEqual(solver.max_memory, 1234)
        self.assertEqual(solver.frozen, 0)

    def test_thc_cd_solver_routes_direct_and_factor_controls(self) -> None:
        class MeanField:
            converged = True
            e_tot = -1.0

            def kernel(self):
                return self.e_tot

        class Solver:
            def __init__(self, mf, **kwargs):
                self.mf = mf
                self.constructor_options = kwargs

        gpu_module = types.ModuleType("gpu4pyscf")
        gpu_module.__path__ = []
        scf_module = types.ModuleType("gpu4pyscf.scf")
        scf_module.RHF = lambda mol: MeanField()
        cc_module = types.ModuleType("gpu4pyscf.cc")
        cc_module.__path__ = []
        rr_module = types.ModuleType("gpu4pyscf.cc.rrccsd")
        rr_module.RRCCSD = Solver
        thc_module = types.ModuleType("gpu4pyscf.cc.thc_rrccsd")
        thc_module.THCRRCCSD = Solver
        gpu_module.scf = scf_module
        gpu_module.cc = cc_module
        cc_module.rrccsd = rr_module
        cc_module.thc_rrccsd = thc_module
        options = {
            "eri_tol": 2e-8,
            "rr_eig_cutoff": 3e-7,
            "direct_scf_tol": 4e-13,
            "thc_fit_tol": 5e-7,
            "thc_rank": 12,
            "thc_initial_rank": None,
            "thc_max_rank": 12,
            "thc_rank_growth": 1.6,
            "thc_orthogonality_cutoff": 6e-12,
            "thc_orthogonality_tolerance": 7e-10,
            "thc_max_iterations": 44,
            "thc_als_convergence_tolerance": 8e-10,
            "thc_ridge": 9e-12,
            "thc_seed": 11,
            "thc_replacement_validation_atol": 1e-9,
            "thc_replacement_validation_rtol": 2e-8,
        }
        with mock.patch.dict(sys.modules, {
            "gpu4pyscf": gpu_module,
            "gpu4pyscf.scf": scf_module,
            "gpu4pyscf.cc": cc_module,
            "gpu4pyscf.cc.rrccsd": rr_module,
            "gpu4pyscf.cc.thc_rrccsd": thc_module,
        }):
            _, solver = MODULE._make_solver(
                object(), "thc_cd", "gpu", options
            )
        self.assertEqual(solver.constructor_options["eri_backend"], "cd")
        for name, expected in options.items():
            if name in {"eri_tol", "rr_eig_cutoff"} or name.startswith("thc_"):
                self.assertEqual(solver.constructor_options[name], expected)
        self.assertEqual(solver.frozen, 0)

    def test_rr_cd_energy_does_not_reconstruct_dense_doubles(self) -> None:
        class MeanField:
            e_tot = -80.0
            converged = True

        class Doubles:
            def reconstruct_t2(self):
                raise AssertionError("CP energy path reconstructed dense t2")

        class Solver:
            converged = True

            def kernel(self):
                return -1.0, object(), Doubles()

            def method_metadata(self):
                return {
                    "rr_projector": {"rank": 12},
                    "precision": "fp64",
                }

        record = MODULE.evaluate_energy(
            object(),
            method="rr_cd",
            device="gpu",
            solver_factory=lambda *args: (MeanField(), Solver()),
        )
        self.assertEqual(record.correlation_energy_eh, -1.0)
        self.assertEqual(record.total_energy_eh, -81.0)

    def test_fno_dry_run_rejects_invalid_threshold_before_compute(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "fno_thresh must be finite and non-negative"
        ):
            MODULE.main([
                "--case", "water4-tz",
                "--method", "fno",
                "--output", "/unused.json",
                "--fno-thresh", "nan",
                "--dry-run",
            ])

    def test_rr_gemm_dry_run_records_pending_gate(self) -> None:
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            status = MODULE.main([
                "--case", "water2-tz",
                "--method", "rr_cd",
                "--rr-ring-kernel", "gemm",
                "--output", "/tmp/unused-counterpoise.json",
                "--dry-run",
            ])

        self.assertEqual(status, 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["rr_ring_kernel"], "gemm")
        self.assertEqual(
            plan["rr_ring_kernel_status"],
            "gemm-awaiting-a100-numerical-allocation-gate",
        )
        self.assertEqual(
            plan["validation_status"],
            "direct-cd-compressed-rr-gemm-gate-pending",
        )

    def test_thc_dry_run_rejects_rr_gemm_selector(self) -> None:
        with self.assertRaisesRegex(ValueError, "applies only to RRCCSD"):
            MODULE.main([
                "--case", "water2-tz",
                "--method", "thc_cd",
                "--rr-ring-kernel", "gemm",
                "--output", "/tmp/unused-counterpoise.json",
                "--dry-run",
            ])


if __name__ == "__main__":
    unittest.main()
