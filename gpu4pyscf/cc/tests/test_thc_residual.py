"""Dense-reference tests for Hohenstein THC Algorithms 1--3."""

import numpy as np
import pytest

from gpu4pyscf.cc.thc_residual import (
    pair_factor_residual_algorithms_1_3,
    project_thc_residual_to_rr,
    thc_algorithm_1,
    thc_algorithm_2,
    thc_algorithm_3,
    thc_residual_algorithms_1_3,
    transform_cholesky_t1,
)


def _problem(seed=201):
    rng = np.random.default_rng(seed)
    nocc, nvir, naux, rank = 3, 4, 5, 4
    t1 = rng.normal(size=(nocc, nvir))
    raw = rng.normal(size=(naux, nocc + nvir, nocc + nvir))
    factors = (raw + raw.transpose(0, 2, 1)) * 0.5
    loo = factors[:, :nocc, :nocc]
    lov = factors[:, :nocc, nocc:]
    lvv = factors[:, nocc:, nocc:]
    y_occ = rng.normal(size=(nocc, rank))
    y_vir = rng.normal(size=(nvir, rank))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    return t1, factors, loo, lov, lvv, y_occ, y_vir, core


def test_t1_transformed_cholesky_blocks_match_coefficient_transform():
    t1, factors, loo, lov, lvv, *_ = _problem()
    nocc, nvir = t1.shape
    particle = np.block([
        [np.eye(nocc), -t1],
        [np.zeros((nvir, nocc)), np.eye(nvir)],
    ])
    hole = np.block([
        [np.eye(nocc), np.zeros((nocc, nvir))],
        [t1.T, np.eye(nvir)],
    ])
    expected = np.einsum("mp,Amn,nq->Apq", particle, factors, hole)
    observed = transform_cholesky_t1(t1, loo, lov, lvv)
    np.testing.assert_allclose(observed.hoo, expected[:, :nocc, :nocc])
    np.testing.assert_allclose(observed.hov, expected[:, :nocc, nocc:])
    np.testing.assert_allclose(observed.hvv, expected[:, nocc:, nocc:])
    np.testing.assert_allclose(observed.hvo, expected[:, nocc:, :nocc])
    assert np.max(np.abs(observed.hoo - observed.hoo.transpose(0, 2, 1))) > 1e-4
    assert np.max(np.abs(observed.hvv - observed.hvv.transpose(0, 2, 1))) > 1e-4


def test_thc_algorithms_1_2_3_match_dense_equations():
    t1, _factors, loo, lov, lvv, y_occ, y_vir, core = _problem(202)
    transformed = transform_cholesky_t1(t1, loo, lov, lvv)
    pair_factor = np.einsum("iX,aX->iaX", y_occ, y_vir)
    t2 = np.einsum(
        "iaX,XY,jbY->ijab", pair_factor, core, pair_factor
    )

    dense1 = np.einsum(
        "ijcd,Aac,Abd->ijab", t2, transformed.hvv, transformed.hvv
    )
    dense1 += np.einsum(
        "klab,Aki,Alj->ijab", t2, transformed.hoo, transformed.hoo
    )
    dense1 -= np.einsum(
        "ilcb,Aac,Alj->ijab", t2, transformed.hvv, transformed.hoo
    )
    dense1 -= np.einsum(
        "kjad,Aki,Abd->ijab", t2, transformed.hoo, transformed.hvv
    )
    expected1 = np.einsum(
        "iaX,ijab,jbY->XY", pair_factor, dense1, pair_factor
    )

    spin_adapted = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    dense2 = np.einsum(
        "Aai,Abj->ijab", transformed.hvo, transformed.hvo
    )
    dense2 += np.einsum(
        "ikac,Abj,Akc->ijab",
        spin_adapted,
        transformed.hvo,
        transformed.hov,
    )
    dense2 += np.einsum(
        "jkbc,Aai,Akc->ijab",
        spin_adapted,
        transformed.hvo,
        transformed.hov,
    )
    expected2 = np.einsum(
        "iaX,ijab,jbY->XY", pair_factor, dense2, pair_factor
    )

    dense3 = -np.einsum(
        "ikac,Akj,Abc->ijab", t2, transformed.hoo, transformed.hvv
    )
    dense3 -= np.einsum(
        "jkbc,Aki,Aac->ijab", t2, transformed.hoo, transformed.hvv
    )
    expected3 = np.einsum(
        "iaX,ijab,jbY->XY", pair_factor, dense3, pair_factor
    )

    observed1 = thc_algorithm_1(y_occ, y_vir, core, transformed)
    observed2 = thc_algorithm_2(y_occ, y_vir, core, transformed)
    observed3 = thc_algorithm_3(y_occ, y_vir, core, transformed)
    np.testing.assert_allclose(observed1, expected1, atol=2e-10)
    np.testing.assert_allclose(observed2, expected2, atol=2e-10)
    np.testing.assert_allclose(observed3, expected3, atol=2e-10)

    # Regression for the index-direction defect in a literal reading of
    # Appendix Algorithm 1 line 13.  Eq. 28 requires boo.T here.
    soo = y_occ.T @ y_occ
    svv = y_vir.T @ y_vir
    boo = np.einsum("kX,Aki,iY->AXY", y_occ, transformed.hoo, y_occ)
    bvv = np.einsum("aX,Aac,cY->AXY", y_vir, transformed.hvv, y_vir)
    literal_combined = svv[None] * boo - soo[None] * bvv
    literal = np.einsum(
        "AXY,YZ,AWZ->XW", literal_combined, core, literal_combined
    )
    assert np.max(np.abs(literal - expected1)) > 1e-4


def test_general_pair_factor_reduces_to_each_thc_algorithm():
    t1, _factors, loo, lov, lvv, y_occ, y_vir, core = _problem(207)
    pair_factor = np.einsum("iX,aX->iaX", y_occ, y_vir).reshape(
        t1.size, core.shape[0]
    )
    generic = pair_factor_residual_algorithms_1_3(
        pair_factor,
        core,
        t1,
        loo,
        lov,
        lvv,
        nocc=t1.shape[0],
        nvir=t1.shape[1],
        auxiliary_block_size=2,
    )
    thc = thc_residual_algorithms_1_3(
        y_occ,
        y_vir,
        core,
        t1,
        loo,
        lov,
        lvv,
        auxiliary_block_size=2,
    )
    np.testing.assert_allclose(generic.algorithm_1, thc.algorithm_1, atol=2e-9)
    np.testing.assert_allclose(generic.algorithm_2, thc.algorithm_2, atol=2e-9)
    np.testing.assert_allclose(generic.algorithm_3, thc.algorithm_3, atol=2e-9)
    np.testing.assert_allclose(generic.sigma_pair, thc.sigma_thc, atol=3e-9)
    assert generic.metadata()["materialized_dense_t2"] is False


def test_arbitrary_rr_pair_factor_matches_dense_equations_term_by_term():
    t1, _factors, loo, lov, lvv, *_ = _problem(208)
    nocc, nvir = t1.shape
    rng = np.random.default_rng(209)
    rank = 5
    pair_factor, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    u = pair_factor.reshape(nocc, nvir, rank)
    t2 = np.einsum("iaX,XY,jbY->ijab", u, core, u)
    transformed = transform_cholesky_t1(t1, loo, lov, lvv)

    dense1 = np.einsum(
        "ijcd,Aac,Abd->ijab", t2, transformed.hvv, transformed.hvv
    )
    dense1 += np.einsum(
        "klab,Aki,Alj->ijab", t2, transformed.hoo, transformed.hoo
    )
    dense1 -= np.einsum(
        "ilcb,Aac,Alj->ijab", t2, transformed.hvv, transformed.hoo
    )
    dense1 -= np.einsum(
        "kjad,Aki,Abd->ijab", t2, transformed.hoo, transformed.hvv
    )
    spin_adapted = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    dense2 = np.einsum("Aai,Abj->ijab", transformed.hvo, transformed.hvo)
    dense2 += np.einsum(
        "ikac,Abj,Akc->ijab",
        spin_adapted,
        transformed.hvo,
        transformed.hov,
    )
    dense2 += np.einsum(
        "jkbc,Aai,Akc->ijab",
        spin_adapted,
        transformed.hvo,
        transformed.hov,
    )
    dense3 = -np.einsum(
        "ikac,Akj,Abc->ijab", t2, transformed.hoo, transformed.hvv
    )
    dense3 -= np.einsum(
        "jkbc,Aki,Aac->ijab", t2, transformed.hoo, transformed.hvv
    )
    expected = tuple(
        np.einsum("iaX,ijab,jbY->XY", u, dense, u)
        for dense in (dense1, dense2, dense3)
    )
    observed = pair_factor_residual_algorithms_1_3(
        pair_factor,
        core,
        t1,
        loo,
        lov,
        lvv,
        nocc=nocc,
        nvir=nvir,
        auxiliary_block_size=2,
    )
    np.testing.assert_allclose(observed.algorithm_1, expected[0], atol=2e-9)
    np.testing.assert_allclose(observed.algorithm_2, expected[1], atol=2e-9)
    np.testing.assert_allclose(observed.algorithm_3, expected[2], atol=2e-9)


def test_streamed_algorithms_and_rr_back_projection():
    t1, _factors, loo, lov, lvv, y_occ, y_vir, core = _problem(203)
    result = thc_residual_algorithms_1_3(
        y_occ,
        y_vir,
        core,
        t1,
        loo,
        lov,
        lvv,
        auxiliary_block_size=2,
    )
    full = transform_cholesky_t1(t1, loo, lov, lvv)
    expected = (
        thc_algorithm_1(y_occ, y_vir, core, full)
        + thc_algorithm_2(y_occ, y_vir, core, full)
        + thc_algorithm_3(y_occ, y_vir, core, full)
    )
    np.testing.assert_allclose(result.sigma_thc, expected, atol=2e-10)
    rng = np.random.default_rng(204)
    tau = rng.normal(size=(3, core.shape[0]))
    expected_rr = tau @ expected @ tau.T
    np.testing.assert_allclose(result.to_rr(tau), expected_rr)
    np.testing.assert_allclose(
        project_thc_residual_to_rr(expected, tau), expected_rr
    )
    assert result.metadata()["complete_ccsd_residual"] is False


def test_gpu_thc_algorithms_stay_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    values = _problem(205)
    t1, _factors, loo, lov, lvv, y_occ, y_vir, core = values
    cpu = thc_residual_algorithms_1_3(
        y_occ, y_vir, core, t1, loo, lov, lvv, auxiliary_block_size=2
    )
    gpu = thc_residual_algorithms_1_3(
        cp.asarray(y_occ),
        cp.asarray(y_vir),
        cp.asarray(core),
        cp.asarray(t1),
        cp.asarray(loo),
        cp.asarray(lov),
        cp.asarray(lvv),
        auxiliary_block_size=2,
    )
    assert isinstance(gpu.sigma_thc, cp.ndarray)
    cp.testing.assert_allclose(
        gpu.sigma_thc, cp.asarray(cpu.sigma_thc), atol=3e-9, rtol=3e-11
    )


def test_gpu_general_pair_reference_stays_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    t1, _factors, loo, lov, lvv, *_ = _problem(210)
    nocc, nvir = t1.shape
    rng = np.random.default_rng(211)
    rank = 4
    pair_factor, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    cpu = pair_factor_residual_algorithms_1_3(
        pair_factor,
        core,
        t1,
        loo,
        lov,
        lvv,
        nocc=nocc,
        nvir=nvir,
        auxiliary_block_size=2,
    )
    gpu = pair_factor_residual_algorithms_1_3(
        cp.asarray(pair_factor),
        cp.asarray(core),
        cp.asarray(t1),
        cp.asarray(loo),
        cp.asarray(lov),
        cp.asarray(lvv),
        nocc=nocc,
        nvir=nvir,
        auxiliary_block_size=2,
    )
    assert isinstance(gpu.sigma_pair, cp.ndarray)
    cp.testing.assert_allclose(
        gpu.sigma_pair, cp.asarray(cpu.sigma_pair), atol=5e-8, rtol=3e-11
    )
