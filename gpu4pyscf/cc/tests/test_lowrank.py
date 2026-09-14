"""Backend-neutral numerical tests for RR/THC doubles representations."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).parents[1] / "lowrank.py"
_SPEC = importlib.util.spec_from_file_location("gpu4pyscf_cc_lowrank", _MODULE_PATH)
lowrank = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = lowrank
_SPEC.loader.exec_module(lowrank)


def _symmetric_t2(nocc=2, nvir=3, seed=12):
    rng = np.random.default_rng(seed)
    pair_dimension = nocc * nvir
    matrix = rng.normal(size=(pair_dimension, pair_dimension))
    matrix = matrix + matrix.T
    return lowrank.t2_from_pair_matrix(matrix, nocc, nvir)


def test_pair_matrix_round_trip_preserves_index_order():
    t2 = np.arange(2 * 2 * 3 * 3, dtype=float).reshape(2, 2, 3, 3)
    matrix = lowrank.pair_matrix_from_t2(t2)
    restored = lowrank.t2_from_pair_matrix(matrix, 2, 3)
    np.testing.assert_array_equal(restored, t2)
    assert matrix[1 * 3 + 2, 0 * 3 + 1] == t2[1, 0, 2, 1]


def test_full_rank_rr_reconstructs_symmetric_doubles():
    t2 = _symmetric_t2()
    projector = lowrank.build_rr_projector(t2, 0.0)
    doubles = lowrank.RRDoubles.from_t2(t2, projector)
    assert projector.rank == 6
    assert projector.orthogonality_error() < 1e-12
    np.testing.assert_allclose(doubles.reconstruct_t2(), t2, atol=1e-12)
    assert doubles.reconstruction_error(t2) < 1e-12
    json.dumps(doubles.metadata())


def test_rr_cutoff_selects_eigenvalues_by_absolute_magnitude():
    matrix = np.diag([5.0, -2.0, 0.2, 1e-7])
    projector = lowrank.build_rr_projector(matrix, 0.5)
    assert projector.rank == 2
    np.testing.assert_allclose(np.abs(projector.eigenvalues), [5.0, 2.0])
    assert projector.compression_fraction == 0.5


def test_operator_projector_matches_dense_dominant_subspace():
    matrix = np.diag([8.0, -3.0, 0.2, 0.1, 0.01, 0.001])
    projector = lowrank.build_rr_projector_from_operator(
        matrix.__matmul__,
        matrix.shape[0],
        1.0,
        initial_rank=2,
        max_rank=5,
    )
    assert projector.rank == 2
    np.testing.assert_allclose(np.abs(projector.eigenvalues), [8.0, 3.0])
    assert projector.orthogonality_error() < 1e-10


def test_thc_validation_factorization_has_monotone_fit_and_full_endpoint():
    t2 = _symmetric_t2(seed=21)
    projector = lowrank.build_rr_projector(t2, 0.0)
    rr = lowrank.RRDoubles.from_t2(t2, projector)
    loose = lowrank.THCDoubles.from_rr(rr, 5.0)
    tight = lowrank.THCDoubles.from_rr(rr, 0.0)

    assert loose.thc_rank <= tight.thc_rank
    assert loose.fit_residual >= tight.fit_residual
    assert tight.fit_residual < 1e-12
    np.testing.assert_allclose(tight.reconstruct_t2(), t2, atol=1e-12)
    assert tight.paper_exact is False
    assert tight.metadata()["factorization_method"] == "symmetric-core-eigh"
    json.dumps(tight.metadata())


def test_invalid_shapes_and_thresholds_fail_explicitly():
    with np.testing.assert_raises(ValueError):
        lowrank.build_rr_projector(np.ones((2, 3)), 1e-5)
    with np.testing.assert_raises(ValueError):
        lowrank.build_rr_projector(np.eye(3), -1.0)
    with np.testing.assert_raises(ValueError):
        lowrank.t2_from_pair_matrix(np.eye(3), 1, 2)


def test_gpu_projector_metadata_requires_and_counts_explicit_transfer():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    projector = lowrank.RRProjector(
        cp.eye(3), cp.asarray([-3.0, -2.0, -1.0]), 0.0, 3
    )
    with pytest.raises(ValueError, match="transfer_counter"):
        projector.metadata()
    transfers = TransferCounter()
    metadata = projector.metadata(transfer_counter=transfers)
    assert metadata["rank"] == 3
    accounting = transfers.to_dict()["by_kind"]["d2h"]
    assert accounting["count"] == 2
    assert accounting["bytes"] == projector.eigenvalues.nbytes + 8
