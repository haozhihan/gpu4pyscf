"""CPU-only tests for reusable, fail-closed canonical-MO artifacts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "water8_benchmark_orbital_tests", ROOT / "benchmark.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


class _Molecule:
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


def _case(x: float = 0.0) -> dict:
    return {
        "id": "toy",
        "role": "test",
        "source": "unit-test",
        "basis": "cc-pVTZ",
        "expected_dimensions": {"nao": 3, "nocc": 1, "nvir": 2},
        "geometry_angstrom": [["H", x, 0.0, 0.0]],
    }


def _mf(coeff: np.ndarray | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        mo_coeff=np.eye(3) if coeff is None else np.asarray(coeff),
        mo_occ=np.asarray([2.0, 0.0, 0.0]),
        mo_energy=np.asarray([-1.0, 0.5, 1.0]),
        e_tot=-1.1,
    )


def _args(*, source: Path | None = None, target: Path | None = None):
    return SimpleNamespace(
        case="toy",
        scf_conv_tol=1.0e-10,
        orbital_artifact_in=source,
        orbital_artifact_out=target,
    )


def _source() -> dict:
    return {
        "base_revision": BENCHMARK.FROZEN_BASE_COMMIT,
        "revision": "tree-sha256:" + "a" * 64,
        "tree_sha256": "a" * 64,
        "source_kind": "content-manifest",
    }


def test_artifact_roundtrip_applies_exact_serialized_orbitals(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "shared-orbitals.npz"
    mol = _Molecule()
    producer_mf = _mf()
    produced = BENCHMARK._prepare_canonical_orbitals(
        _args(target=artifact),
        producer_mf,
        mol,
        _case(),
        _source(),
        np,
        None,
    )
    assert artifact.is_file()
    assert produced["mode"] == "written-and-reloaded"
    assert produced["artifact_sha256"] == BENCHMARK._file_sha256(artifact)
    assert produced["validation"]["orthonormality_max_abs"] == 0.0

    # A fresh SCF may select a different orthogonal basis. Loading the shared
    # artifact must replace it with the exact serialized canonical MOs.
    rotated = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 2**-0.5, -2**-0.5],
        [0.0, 2**-0.5, 2**-0.5],
    ])
    consumer_mf = _mf(rotated)
    consumer_mf.e_tot += 1.0e-8
    consumed = BENCHMARK._prepare_canonical_orbitals(
        _args(source=artifact),
        consumer_mf,
        mol,
        _case(),
        _source(),
        np,
        None,
    )
    assert consumed["mode"] == "loaded"
    assert consumed["orbital_fingerprint"] == produced["orbital_fingerprint"]
    assert consumed["artifact_sha256"] == produced["artifact_sha256"]
    assert np.array_equal(consumer_mf.mo_coeff, np.eye(3))
    assert consumer_mf.e_tot == produced["producer_hf_energy"]
    assert consumed["applied_hf_energy"] == produced["producer_hf_energy"]
    assert consumed["included_in_post_hf"] is False
    assert BENCHMARK._orbital_record(consumed).get("_arrays") is None


def test_artifact_rejects_case_mismatch_tampering_and_overwrite(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "shared-orbitals.npz"
    mol = _Molecule()
    BENCHMARK._prepare_canonical_orbitals(
        _args(target=artifact), _mf(), mol, _case(), _source(), np, None
    )
    wrong_identity = BENCHMARK._canonical_orbital_identity(
        "toy", _case(0.1), mol
    )
    with pytest.raises(ValueError, match="scientific identity"):
        BENCHMARK._load_orbital_artifact(
            artifact, wrong_identity, mol, np
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        identity = BENCHMARK._canonical_orbital_identity("toy", _case(), mol)
        BENCHMARK._write_orbital_artifact_once(
            artifact,
            identity,
            BENCHMARK._capture_canonical_orbitals(_mf(), np),
            BENCHMARK._orbital_producer(
                _source(), hf_energy=-1.1, scf_conv_tol=1.0e-10
            ),
            mol,
            np,
        )

    with np.load(artifact, allow_pickle=False) as original:
        payload = {name: np.array(original[name], copy=True) for name in original.files}
    payload["mo_energy"][1] += 0.1
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **payload)
    identity = BENCHMARK._canonical_orbital_identity("toy", _case(), mol)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        BENCHMARK._load_orbital_artifact(tampered, identity, mol, np)


def test_artifact_rejects_nonorthonormal_orbitals_and_hf_energy_mismatch(
    tmp_path: Path,
) -> None:
    mol = _Molecule()
    bad_coeff = np.eye(3)
    bad_coeff[0, 0] = 2.0
    identity = BENCHMARK._canonical_orbital_identity("toy", _case(), mol)
    with pytest.raises(ValueError, match="not AO-metric orthonormal"):
        BENCHMARK._write_orbital_artifact_once(
            tmp_path / "bad.npz",
            identity,
            BENCHMARK._capture_canonical_orbitals(_mf(bad_coeff), np),
            BENCHMARK._orbital_producer(
                _source(), hf_energy=-1.1, scf_conv_tol=1.0e-10
            ),
            mol,
            np,
        )

    artifact = tmp_path / "good.npz"
    BENCHMARK._prepare_canonical_orbitals(
        _args(target=artifact), _mf(), mol, _case(), _source(), np, None
    )
    changed_hf = _mf()
    changed_hf.e_tot = -1.0
    with pytest.raises(ValueError, match="fresh SCF energy disagrees"):
        BENCHMARK._prepare_canonical_orbitals(
            _args(source=artifact),
            changed_hf,
            mol,
            _case(),
            _source(),
            np,
            None,
        )


def test_artifact_rejects_different_source_snapshot(tmp_path: Path) -> None:
    artifact = tmp_path / "shared-orbitals.npz"
    mol = _Molecule()
    BENCHMARK._prepare_canonical_orbitals(
        _args(target=artifact), _mf(), mol, _case(), _source(), np, None
    )
    other_source = dict(_source())
    other_source["tree_sha256"] = "b" * 64
    other_source["revision"] = "tree-sha256:" + "b" * 64

    with pytest.raises(ValueError, match="source provenance"):
        BENCHMARK._prepare_canonical_orbitals(
            _args(source=artifact),
            _mf(),
            mol,
            _case(),
            other_source,
            np,
            None,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("base_revision", "different-base"),
        ("revision", "tree-sha256:" + "c" * 64),
        ("source_kind", "git-working-tree"),
        ("tree_sha256", None),
    ),
)
def test_source_mismatch_fails_before_mean_field_mutation(
    tmp_path: Path, field: str, replacement: str | None
) -> None:
    artifact = tmp_path / f"source-{field}.npz"
    mol = _Molecule()
    BENCHMARK._prepare_canonical_orbitals(
        _args(target=artifact), _mf(), mol, _case(), _source(), np, None
    )
    current_source = dict(_source())
    if replacement is None:
        current_source.pop(field)
    else:
        current_source[field] = replacement
    rotated = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 2**-0.5, -2**-0.5],
        [0.0, 2**-0.5, 2**-0.5],
    ])
    consumer = _mf(rotated)
    original_energy = consumer.e_tot

    with pytest.raises(ValueError, match=f"current benchmark source: {field}"):
        BENCHMARK._prepare_canonical_orbitals(
            _args(source=artifact),
            consumer,
            mol,
            _case(),
            current_source,
            np,
            None,
        )
    assert np.array_equal(consumer.mo_coeff, rotated)
    assert consumer.e_tot == original_energy


def test_orbital_cli_and_config_are_fail_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npz"
    output = tmp_path / "output.npz"
    both = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "canonical",
        "--orbital-artifact-in", str(missing),
        "--orbital-artifact-out", str(output),
    ])
    with pytest.raises(ValueError, match="mutually exclusive"):
        BENCHMARK._validate_run_configuration(both)

    missing_only = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "canonical",
        "--orbital-artifact-in", str(missing),
    ])
    with pytest.raises(ValueError, match="does not exist"):
        BENCHMARK._validate_run_configuration(missing_only)

    output.write_bytes(b"occupied")
    existing_output = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "canonical",
        "--orbital-artifact-out", str(output),
    ])
    with pytest.raises(ValueError, match="refusing to overwrite"):
        BENCHMARK._validate_run_configuration(existing_output)

    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "orbital_artifact_in": str(output),
    }))
    configured = BENCHMARK._parser().parse_args([
        "--case", "water2-tz",
        "--method", "canonical",
        "--config", str(config),
    ])
    configured = BENCHMARK._apply_config(configured)
    assert configured.orbital_artifact_in == output


def test_restart_v2_embeds_exact_orbital_identity(tmp_path: Path) -> None:
    mol = _Molecule()
    context = BENCHMARK._prepare_canonical_orbitals(
        _args(), _mf(), mol, _case(), _source(), np, None
    )
    t1 = np.asarray([[0.1, 0.2]])
    t2 = np.arange(4.0).reshape(1, 1, 2, 2)
    cc_obj = SimpleNamespace(e_corr=-0.5, t1=t1, t2=t2)
    checkpoint = BENCHMARK._write_checkpoint_once(
        tmp_path / "restart-v2.npz",
        "canonical",
        (-0.5, t1, t2),
        cc_obj,
        np,
        orbital_context=context,
    )
    checkpoint.pop("_cached_amplitude_payload")
    assert checkpoint["checkpoint_schema"] == BENCHMARK.RESTART_SCHEMA_V2
    assert checkpoint["orbital_fingerprint"] == context["orbital_fingerprint"]
    with np.load(checkpoint["path"], allow_pickle=False) as payload:
        assert payload["checkpoint_schema"][0] == BENCHMARK.RESTART_SCHEMA_V2
        assert np.array_equal(payload["mo_coeff"], np.eye(3))
        assert json.loads(payload["orbital_identity_json"][0]) == context["identity"]
        assert payload["orbital_fingerprint"][0] == context["orbital_fingerprint"]
