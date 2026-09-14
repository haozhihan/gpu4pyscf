"""Equation-level tests for the shared CCSD residual semantics."""

from __future__ import annotations

import numpy as np
import pytest

from gpu4pyscf.cc.residual_evaluator import (
    RESIDUAL_SCHEMA,
    evaluate_dense_ccsd_residual,
    labeled_low_rank_residuals,
)


def _problem():
    current_t1 = np.zeros((1, 2))
    current_t2 = np.zeros((1, 1, 2, 2))
    jacobi_t1 = np.array([[1.0, 2.0]])
    jacobi_t2 = np.array([[[[1.0, 2.0], [2.0, 4.0]]]])
    occupied = np.array([-1.0])
    virtual = np.array([1.0, 2.0])
    return (
        current_t1,
        current_t2,
        jacobi_t1,
        jacobi_t2,
        occupied,
        virtual,
    )


def test_dense_equation_residual_multiplies_jacobi_update_by_denominators():
    values = _problem()
    result = evaluate_dense_ccsd_residual(*values)

    eia = np.array([[-2.0, -3.0]])
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    expected_t1_residual = values[2] * eia
    expected_t2_residual = values[3] * eijab
    expected_equation = np.sqrt(
        np.vdot(expected_t1_residual, expected_t1_residual).real
        + np.vdot(expected_t2_residual, expected_t2_residual).real
    )
    expected_update = np.sqrt(
        np.vdot(values[2], values[2]).real
        + np.vdot(values[3], values[3]).real
    )

    assert np.isclose(result.full_space_equation_norm, expected_equation)
    assert np.isclose(result.full_space_jacobi_update_norm, expected_update)
    assert not np.isclose(
        result.full_space_equation_norm,
        result.full_space_jacobi_update_norm,
    )
    assert result.projected_equation_norm is None
    assert result.projected_jacobi_update_norm is None


def test_projected_equation_projects_full_residual_after_denominator_scaling():
    values = list(_problem())
    values[2] = np.zeros_like(values[2])
    projector = np.array([[1.0], [1.0]]) / np.sqrt(2.0)

    result = evaluate_dense_ccsd_residual(
        *values, projector_vectors=projector
    )

    eia = np.array([[-2.0, -3.0]])
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    update_pair = values[3].transpose(0, 2, 1, 3).reshape(2, 2)
    equation_pair = (
        values[3] * eijab
    ).transpose(0, 2, 1, 3).reshape(2, 2)
    expected_projected_update = projector.T @ update_pair @ projector
    expected_projected_equation = projector.T @ equation_pair @ projector

    assert np.isclose(
        result.projected_jacobi_update_norm,
        np.linalg.norm(expected_projected_update),
    )
    assert np.isclose(
        result.projected_equation_norm,
        np.linalg.norm(expected_projected_equation),
    )
    # Projection and denominator scaling do not commute for this mixed pair.
    projected_eia = (
        projector.T @ np.diag(eia.ravel()) @ projector
    ).item()
    incorrect = abs(2.0 * projected_eia * expected_projected_update.item())
    assert not np.isclose(result.projected_equation_norm, incorrect)


def test_identity_projector_reproduces_full_space_norms():
    values = _problem()
    result = evaluate_dense_ccsd_residual(
        *values, projector_vectors=np.eye(2)
    )

    assert np.isclose(
        result.projected_equation_norm, result.full_space_equation_norm
    )
    assert np.isclose(
        result.projected_jacobi_update_norm,
        result.full_space_jacobi_update_norm,
    )


def test_dense_residual_can_account_each_norm_scalar_boundary():
    operations = []

    def scalar_to_float(value, *, operation):
        operations.append(operation)
        return value.item()

    expected = evaluate_dense_ccsd_residual(*_problem())
    observed = evaluate_dense_ccsd_residual(
        *_problem(), scalar_to_float=scalar_to_float
    )

    assert observed == expected
    assert operations == [
        "singles-equation-norm",
        "doubles-equation-norm",
        "singles-jacobi-update-norm",
        "doubles-jacobi-update-norm",
        "full-equation-norm",
        "full-jacobi-update-norm",
    ]


@pytest.mark.parametrize("block_size", [1, 2])
def test_blockwise_doubles_reduction_matches_dense_semantics(block_size):
    values = _problem()
    expected = evaluate_dense_ccsd_residual(*values)
    operations = []

    observed = evaluate_dense_ccsd_residual(
        *values,
        doubles_block_size=block_size,
        scalar_to_float=lambda value, *, operation: (
            operations.append(operation) or value.item()
        ),
    )

    assert observed == expected
    assert operations == [
        "singles-equation-norm",
        "doubles-equation-norm",
        "singles-jacobi-update-norm",
        "doubles-jacobi-update-norm",
        "full-equation-norm",
        "full-jacobi-update-norm",
    ]


def test_blockwise_reduction_rejects_projector_instead_of_silent_drift():
    with pytest.raises(ValueError, match="does not support a projector"):
        evaluate_dense_ccsd_residual(
            *_problem(),
            projector_vectors=np.eye(2),
            doubles_block_size=1,
        )


def test_block_observer_runs_while_each_virtual_slice_is_evaluated():
    blocks = []

    evaluate_dense_ccsd_residual(
        *_problem(),
        doubles_block_size=1,
        block_observer=lambda a0, a1: blocks.append((a0, a1)),
    )

    assert blocks == [(0, 1), (1, 2)]


def test_labeled_low_rank_record_is_fail_closed_and_has_no_generic_norm():
    missing = labeled_low_rank_residuals()

    assert missing["schema"] == RESIDUAL_SCHEMA
    assert missing["available"] is False
    assert missing["projected_equation"] is None
    assert "norm" not in missing
    for measurement in missing["measurements"].values():
        assert measurement["available"] is False
        assert measurement["norm"] is None
        assert isinstance(measurement["kind"], str)
        assert isinstance(measurement["space"], str)
        assert isinstance(measurement["is_full_space"], bool)

    populated = labeled_low_rank_residuals(
        projected_equation=2.0e-7,
        projected_jacobi_update=4.0e-6,
    )
    assert populated["available"] is True
    assert populated["projected_equation"] == 2.0e-7
    projected = populated["measurements"]["projected_equation"]
    update = populated["measurements"]["projected_jacobi_update"]
    assert projected == {
        "available": True,
        "norm": 2.0e-7,
        "kind": "equation-residual",
        "space": "rr-projected-active-pair",
        "is_full_space": False,
    }
    assert update["kind"] == "jacobi-update"
    assert update["norm"] == 4.0e-6

    with pytest.raises(ValueError, match="finite and non-negative"):
        labeled_low_rank_residuals(projected_equation=float("nan"))
