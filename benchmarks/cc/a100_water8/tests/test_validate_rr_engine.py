"""CPU-only checks for the real-molecule RR integration smoke driver."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_rr_engine", ROOT / "validate_rr_engine.py"
)
assert SPEC is not None and SPEC.loader is not None
VALIDATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATE
SPEC.loader.exec_module(VALIDATE)


def test_water1_is_exact_first_monomer_from_frozen_water2() -> None:
    water1 = VALIDATE._load_case("water1-sto3g")
    water2 = VALIDATE._load_case("water2-tz")

    assert water1["geometry_angstrom"] == water2["geometry_angstrom"][:3]
    assert water1["basis"] == "sto-3g"
    assert water1["expected_dimensions"] == {
        "nao": 7,
        "nocc": 5,
        "nvir": 2,
    }


def test_initial_record_cannot_be_mistaken_for_a_performance_claim(
    monkeypatch,
) -> None:
    monkeypatch.setattr(VALIDATE, "_gpu_hardware", lambda: {"available": False})
    monkeypatch.setattr(VALIDATE, "_command", lambda _arguments: None)
    monkeypatch.setattr(VALIDATE, "_version", lambda _name: None)
    args = VALIDATE._parser().parse_args(["--case", "water2-tz"])

    record = VALIDATE._initial_record(args, VALIDATE._load_case(args.case))

    assert record["contract"]["integral_factorization"] == (
        "auxiliary-basis density fitting"
    )
    assert record["contract"]["performance_eligible"] is False
    assert record["contract"]["production_path_dense_t2_materialized"] is False
    assert (
        record["contract"]["production_path_four_index_eri_materialized"]
        is False
    )
    assert record["parameters"]["rr_eig_cutoff"] == 1e-6
    assert record["parameters"]["rr_max_rank"] == 32
    assert record["parameters"]["rr_ring_kernel"] == "reference"
    assert record["parameters"]["oracle_tensor_atol"] == 1e-10
    assert record["parameters"]["oracle_energy_atol"] == 1e-10
    assert record["parameters"]["water1_convergence"] is False
    assert record["parameters"]["convergence_energy_tol"] == 1e-10
    assert record["parameters"]["convergence_residual_tol"] == 1e-8


def test_mtu_smoke_enables_the_water1_convergence_gate() -> None:
    script = (ROOT / "run_mtu_rr_smoke.sbatch").read_text()

    assert "--case water1-sto3g --water1-convergence" in script
    assert "--case water2-tz" in script
    assert '--rr-ring-kernel "${RR_RING_KERNEL_VALUE}"' in script
    assert "#SBATCH --nodelist=compute-1-6" in script


def test_rr_smoke_parser_accepts_only_declared_ring_kernels() -> None:
    args = VALIDATE._parser().parse_args([
        "--case", "water2-tz", "--rr-ring-kernel", "gemm"
    ])
    assert args.rr_ring_kernel == "gemm"


def test_atomic_json_writer_leaves_no_temporary_file(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"

    VALIDATE._json_write(destination, {"status": "passed"})

    assert destination.read_text().endswith("\n")
    assert not destination.with_suffix(".json.tmp").exists()


def test_gpu_identity_query_uses_fields_supported_on_mtu(monkeypatch) -> None:
    def command(arguments):
        joined = " ".join(arguments)
        if "--query-gpu=" in joined:
            assert "pci.link" not in joined
            return "NVIDIA A100-SXM4-80GB, GPU-1, 81920, 580.0, 0000:01:00.0"
        if arguments[1:] == ["topo", "-m"]:
            return None
        raise AssertionError(arguments)

    monkeypatch.setattr(VALIDATE, "_command", command)

    hardware = VALIDATE._gpu_hardware()

    assert hardware["available"] is True
    assert hardware["visible_gpus"][0]["name"] == "NVIDIA A100-SXM4-80GB"
    assert hardware["topology"] == {
        "available": False,
        "nvidia_smi_topo_m": None,
    }
