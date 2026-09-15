"""Dense audits for THC Algorithms 8--10 and published Eqs. 39--43."""

from dataclasses import replace

import numpy as np
import pytest

from gpu4pyscf.cc.thc_omega_ghi import (
    T1TransformedCholeskyBlocks,
    thc_omega_e_algorithm9,
    thc_omega_gh_algorithm8,
    thc_omega_ij_algorithm10,
)


def _problem(seed=801, dtype=np.float64, *, symmetric_core=False):
    rng = np.random.default_rng(seed)
    nocc, nvir, amplitude_rank, nchol = 2, 3, 3, 4
    y_occ = rng.normal(size=(nocc, amplitude_rank)).astype(dtype)
    y_vir = rng.normal(size=(nvir, amplitude_rank)).astype(dtype)
    amplitude_core = rng.normal(
        size=(amplitude_rank, amplitude_rank)
    ).astype(dtype)
    if symmetric_core:
        amplitude_core = (amplitude_core + amplitude_core.T) / dtype(2)
    l_oo = rng.normal(size=(nchol, nocc, nocc)).astype(dtype)
    l_vv = rng.normal(size=(nchol, nvir, nvir)).astype(dtype)
    l_ov = rng.normal(size=(nchol, nocc, nvir)).astype(dtype)
    fhat_oo = rng.normal(size=(nocc, nocc)).astype(dtype)
    fhat_vv = rng.normal(size=(nvir, nvir)).astype(dtype)
    fhat_ov = rng.normal(size=(nocc, nvir)).astype(dtype)
    fhat_vo = rng.normal(size=(nvir, nocc)).astype(dtype)
    cholesky = T1TransformedCholeskyBlocks(l_oo, l_vv, l_ov)
    return (
        y_occ,
        y_vir,
        amplitude_core,
        cholesky,
        fhat_oo,
        fhat_vv,
        fhat_ov,
        fhat_vo,
    )


def _dense_amplitudes(y_occ, y_vir, core):
    return np.einsum(
        "iX,aX,XY,jY,bY->ijab", y_occ, y_vir, core, y_occ, y_vir
    )


def _project_dense(dense, y_occ, y_vir):
    return np.einsum(
        "iX,aX,ijab,jY,bY->XY", y_occ, y_vir, dense, y_occ, y_vir
    )


def _dense_published_eq39_parts(t2, cholesky):
    # The hatted-index orientation follows the T1-transformed l_pq arrays
    # printed in Algorithm 8, lines 3--5.  No l_oo/l_vv symmetry is used.
    ov_vv = np.einsum("Akd,Aac->kdac", cholesky.l_ov, cholesky.l_vv)
    omega_g = 2 * np.einsum("ikcd,kdac->ia", t2, ov_vv)
    omega_g -= np.einsum("ikcd,kcad->ia", t2, ov_vv)

    ov_oo = np.einsum("Alc,Aki->lcki", cholesky.l_ov, cholesky.l_oo)
    oo_ov = np.einsum("Ali,Akc->likc", cholesky.l_oo, cholesky.l_ov)
    omega_h = -2 * np.einsum("klac,lcki->ia", t2, ov_oo)
    omega_h += np.einsum("klac,likc->ia", t2, oo_ov)
    return omega_g, omega_h


def _dense_published_eq40(t2, cholesky):
    ovov = np.einsum("Ame,Akd->mekd", cholesky.l_ov, cholesky.l_ov)
    return 2 * np.einsum("jmde,mekd->kj", t2, ovov) - np.einsum(
        "jmde,mdke->kj", t2, ovov
    )


def _dense_published_eq41(t2, cholesky):
    ovov = np.einsum("Ald,Amc->ldmc", cholesky.l_ov, cholesky.l_ov)
    return -2 * np.einsum("lmdb,ldmc->bc", t2, ovov) + np.einsum(
        "lmdb,lcmd->bc", t2, ovov
    )


def _literal_algorithm8(y_occ, y_vir, core, cholesky):
    rank = core.shape[0]
    omega_g = np.zeros((y_occ.shape[0], y_vir.shape[0]), dtype=core.dtype)
    omega_h = np.zeros_like(omega_g)
    xi_oo = np.zeros((y_occ.shape[0],) * 2, dtype=core.dtype)
    xi_vv = np.zeros((y_vir.shape[0],) * 2, dtype=core.dtype)
    for l_oo, l_vv, l_ov in zip(
        cholesky.l_oo, cholesky.l_vv, cholesky.l_ov
    ):
        a_x = np.einsum("iX,aX,ia->X", y_occ, y_vir, l_ov)
        b_xy = np.einsum("iX,aY,ia->XY", y_occ, y_vir, l_ov)
        c_xy = 2 * np.diag(core @ a_x) - b_xy * core
        assert c_xy.shape == (rank, rank)
        d_ia = np.einsum("iY,aX,XY->ia", y_occ, y_vir, c_xy)
        xi_oo += np.einsum("ia,ja->ij", l_ov, d_ia)
        xi_vv -= np.einsum("ia,ib->ab", d_ia, l_ov)
        omega_g += np.einsum("ib,ab->ia", d_ia, l_vv)
        omega_h -= np.einsum("ij,ja->ia", l_oo, d_ia)
    return omega_g, omega_h, xi_oo, xi_vv


def _literal_algorithm9(y_occ, y_vir, core, fhat_oo, fhat_vv, xi_oo, xi_vv):
    overlap_occ = y_occ.T @ y_occ
    overlap_vir = y_vir.T @ y_vir
    amplitude_metric = (overlap_occ * overlap_vir) @ core
    c_ix = (fhat_oo + xi_oo).T @ y_occ
    d_ax = (fhat_vv + xi_vv) @ y_vir
    dressed_occ = y_occ.T @ c_ix
    dressed_vir = y_vir.T @ d_ax
    one_body_metric = overlap_occ * dressed_vir - dressed_occ * overlap_vir
    omega_e = (
        one_body_metric @ amplitude_metric.T
        + amplitude_metric @ one_body_metric.T
    )
    return omega_e, amplitude_metric, one_body_metric


def _literal_algorithm10(y_occ, y_vir, core, fhat_ov, fhat_vo):
    a_ix = fhat_ov @ y_vir
    b_xy = y_occ.T @ a_ix
    c_x = core @ np.diag(b_xy)
    coulomb = 2 * ((y_occ * c_x[None, :]) @ y_vir.T)
    e_xy = core * b_xy
    exchange = -((y_occ @ e_xy.T) @ y_vir.T)
    return fhat_vo.T, coulomb, exchange


def _amplitude_element_index_loop(y_occ, y_vir, core, i, j, a, b):
    value = core.dtype.type(0)
    for x in range(core.shape[0]):
        for y in range(core.shape[1]):
            value += (
                y_occ[i, x]
                * y_vir[a, x]
                * core[x, y]
                * y_occ[j, y]
                * y_vir[b, y]
            )
    return value


def _published_eq39_parts_index_loop(y_occ, y_vir, core, cholesky):
    """Independent full-index Eq. 39 oracle without einsum/matmul."""

    nocc, nvir = y_occ.shape[0], y_vir.shape[0]
    omega_g = np.zeros((nocc, nvir), dtype=core.dtype)
    omega_h = np.zeros_like(omega_g)
    for i in range(nocc):
        for a in range(nvir):
            for k in range(nocc):
                for c in range(nvir):
                    for d in range(nvir):
                        t_ikcd = _amplitude_element_index_loop(
                            y_occ, y_vir, core, i, k, c, d
                        )
                        kd_ac = core.dtype.type(0)
                        kc_ad = core.dtype.type(0)
                        for auxiliary in range(cholesky.nchol):
                            kd_ac += (
                                cholesky.l_ov[auxiliary, k, d]
                                * cholesky.l_vv[auxiliary, a, c]
                            )
                            kc_ad += (
                                cholesky.l_ov[auxiliary, k, c]
                                * cholesky.l_vv[auxiliary, a, d]
                            )
                        omega_g[i, a] += t_ikcd * (2 * kd_ac - kc_ad)
            for k in range(nocc):
                for l in range(nocc):
                    for c in range(nvir):
                        t_klac = _amplitude_element_index_loop(
                            y_occ, y_vir, core, k, l, a, c
                        )
                        lc_ki = core.dtype.type(0)
                        li_kc = core.dtype.type(0)
                        for auxiliary in range(cholesky.nchol):
                            lc_ki += (
                                cholesky.l_ov[auxiliary, l, c]
                                * cholesky.l_oo[auxiliary, k, i]
                            )
                            li_kc += (
                                cholesky.l_oo[auxiliary, l, i]
                                * cholesky.l_ov[auxiliary, k, c]
                            )
                        omega_h[i, a] -= t_klac * (2 * lc_ki - li_kc)
    return omega_g, omega_h


def _algorithm8_line8_metric_index_loop(
    y_occ, y_vir, core, cholesky, line8_metric
):
    """Scalar Appendix schedule with a caller-selected line-8 object."""

    nocc, nvir = y_occ.shape[0], y_vir.shape[0]
    rank = core.shape[0]
    omega_g = np.zeros((nocc, nvir), dtype=core.dtype)
    omega_h = np.zeros_like(omega_g)
    for auxiliary in range(cholesky.nchol):
        a_x = np.zeros(rank, dtype=core.dtype)
        b_xy = np.zeros((rank, rank), dtype=core.dtype)
        for x in range(rank):
            for i in range(nocc):
                for a in range(nvir):
                    a_x[x] += (
                        y_occ[i, x]
                        * y_vir[a, x]
                        * cholesky.l_ov[auxiliary, i, a]
                    )
                    for y in range(rank):
                        b_xy[x, y] += (
                            y_occ[i, x]
                            * y_vir[a, y]
                            * cholesky.l_ov[auxiliary, i, a]
                        )
        c_xy = np.zeros((rank, rank), dtype=core.dtype)
        for x in range(rank):
            diagonal_x = core.dtype.type(0)
            for z in range(rank):
                diagonal_x += core[x, z] * a_x[z]
            for y in range(rank):
                c_xy[x, y] = (
                    2 * line8_metric[x, y] * diagonal_x
                    - b_xy[x, y] * core[x, y]
                )
        d_ia = np.zeros((nocc, nvir), dtype=core.dtype)
        for i in range(nocc):
            for a in range(nvir):
                for x in range(rank):
                    for y in range(rank):
                        d_ia[i, a] += (
                            y_occ[i, y] * y_vir[a, x] * c_xy[x, y]
                        )
        for i in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    omega_g[i, a] += (
                        d_ia[i, b] * cholesky.l_vv[auxiliary, a, b]
                    )
                for j in range(nocc):
                    omega_h[i, a] -= (
                        cholesky.l_oo[auxiliary, j, i] * d_ia[j, a]
                    )
    return omega_g, omega_h


def _published_eq43_parts_index_loop(y_occ, y_vir, core, fhat_ov, fhat_vo):
    """Independent full-index published Eq. 43 oracle."""

    nocc, nvir = y_occ.shape[0], y_vir.shape[0]
    direct = fhat_vo.T.copy()
    coulomb = np.zeros((nocc, nvir), dtype=core.dtype)
    exchange = np.zeros_like(coulomb)
    for i in range(nocc):
        for a in range(nvir):
            for k in range(nocc):
                for c in range(nvir):
                    coulomb[i, a] += (
                        2
                        * _amplitude_element_index_loop(
                            y_occ, y_vir, core, i, k, a, c
                        )
                        * fhat_ov[k, c]
                    )
                    exchange[i, a] -= (
                        _amplitude_element_index_loop(
                            y_occ, y_vir, core, i, k, c, a
                        )
                        * fhat_ov[k, c]
                    )
    return direct, coulomb, exchange


def test_algorithm8_kronecker_delta_matches_nonorthogonal_index_loop_eq39():
    rng = np.random.default_rng(820)
    nocc, nvir, amplitude_rank, rr_rank, nchol = 3, 4, 3, 2, 2
    y_occ = rng.normal(size=(nocc, amplitude_rank))
    y_vir = rng.normal(size=(nvir, amplitude_rank))
    pair_gram = (y_occ.T @ y_occ) * (y_vir.T @ y_vir)
    values, vectors = np.linalg.eigh(pair_gram)
    inverse_sqrt = (
        vectors * (1.0 / np.sqrt(values))[None, :]
    ) @ vectors.T
    rr_rows, _ = np.linalg.qr(rng.normal(size=(amplitude_rank, rr_rank)))
    tau = rr_rows.T @ inverse_sqrt
    rr_core = rng.normal(size=(rr_rank, rr_rank))
    rr_core = (rr_core + rr_core.T) * 0.5
    core = tau.T @ rr_core @ tau
    cholesky = T1TransformedCholeskyBlocks(
        rng.normal(size=(nchol, nocc, nocc)),
        rng.normal(size=(nchol, nvir, nvir)),
        rng.normal(size=(nchol, nocc, nvir)),
    )

    assert amplitude_rank >= 3
    assert not np.allclose(pair_gram, np.eye(amplitude_rank))
    np.testing.assert_allclose(
        tau @ pair_gram @ tau.T,
        np.eye(rr_rank),
        atol=2e-14,
        rtol=2e-14,
    )
    assert (tau @ pair_gram @ tau.T).shape != np.eye(amplitude_rank).shape

    expected_g, expected_h = _published_eq39_parts_index_loop(
        y_occ, y_vir, core, cholesky
    )
    identity_g, identity_h = _algorithm8_line8_metric_index_loop(
        y_occ, y_vir, core, cholesky, np.eye(amplitude_rank)
    )
    gram_g, gram_h = _algorithm8_line8_metric_index_loop(
        y_occ, y_vir, core, cholesky, pair_gram
    )
    observed = thc_omega_gh_algorithm8(y_occ, y_vir, core, cholesky)

    np.testing.assert_allclose(observed.omega_g, expected_g, atol=2e-11)
    np.testing.assert_allclose(observed.omega_h, expected_h, atol=2e-11)
    np.testing.assert_allclose(identity_g, expected_g, atol=2e-11)
    np.testing.assert_allclose(identity_h, expected_h, atol=2e-11)
    assert np.max(np.abs(gram_g + gram_h - expected_g - expected_h)) > 1e-4


def test_algorithm8_matches_published_eq39_not_literal_line13():
    y_occ, y_vir, core, cholesky, *_ = _problem(
        802, symmetric_core=True
    )
    assert not np.allclose(cholesky.l_oo, cholesky.l_oo.swapaxes(1, 2))
    assert not np.allclose(cholesky.l_vv, cholesky.l_vv.swapaxes(1, 2))
    t2 = _dense_amplitudes(y_occ, y_vir, core)
    expected_g, expected_h = _dense_published_eq39_parts(t2, cholesky)
    observed = thc_omega_gh_algorithm8(y_occ, y_vir, core, cholesky)

    np.testing.assert_allclose(observed.omega_g, expected_g, atol=2e-11)
    np.testing.assert_allclose(observed.omega_h, expected_h, atol=2e-11)
    np.testing.assert_allclose(
        observed.singles_gh, expected_g + expected_h, atol=2e-11
    )
    assert not np.allclose(
        observed.appendix_line13_omega_h,
        expected_h,
        atol=1e-12,
        rtol=1e-12,
    )


def test_algorithm8_xi_oo_matches_published_eq40():
    y_occ, y_vir, core, cholesky, *_ = _problem(
        803, symmetric_core=True
    )
    expected = _dense_published_eq40(
        _dense_amplitudes(y_occ, y_vir, core), cholesky
    )
    observed = thc_omega_gh_algorithm8(y_occ, y_vir, core, cholesky)
    np.testing.assert_allclose(observed.xi_oo, expected, atol=2e-11)


def test_algorithm8_xi_vv_matches_published_eq41():
    y_occ, y_vir, core, cholesky, *_ = _problem(
        804, symmetric_core=True
    )
    expected = _dense_published_eq41(
        _dense_amplitudes(y_occ, y_vir, core), cholesky
    )
    observed = thc_omega_gh_algorithm8(y_occ, y_vir, core, cholesky)
    np.testing.assert_allclose(observed.xi_vv, expected, atol=2e-11)


def test_algorithm8_nonsymmetric_core_matches_literal_appendix_schedule():
    y_occ, y_vir, core, cholesky, *_ = _problem(805)
    assert not np.allclose(core, core.T)
    expected = _literal_algorithm8(y_occ, y_vir, core, cholesky)
    observed = thc_omega_gh_algorithm8(
        y_occ, y_vir, core, cholesky, cholesky_block_size=2
    )
    for actual, reference in zip(
        (
            observed.omega_g,
            observed.appendix_line13_omega_h,
            observed.xi_oo,
            observed.xi_vv,
        ),
        expected,
    ):
        np.testing.assert_allclose(actual, reference, atol=2e-12)


def test_algorithm8_line13_and_published_eq39_differ_for_nonsymmetric_loo():
    y_occ, y_vir, core, cholesky, *_ = _problem(
        819, symmetric_core=True
    )
    observed = thc_omega_gh_algorithm8(y_occ, y_vir, core, cholesky)
    assert not np.allclose(
        observed.omega_h,
        observed.appendix_line13_omega_h,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        observed.appendix_line13_singles_gh,
        observed.omega_g + observed.appendix_line13_omega_h,
        atol=0,
        rtol=0,
    )


def test_algorithm8_cholesky_blocking_does_not_change_outputs():
    y_occ, y_vir, core, cholesky, *_ = _problem(806)
    single = thc_omega_gh_algorithm8(
        y_occ, y_vir, core, cholesky, cholesky_block_size=1
    )
    paired = thc_omega_gh_algorithm8(
        y_occ, y_vir, core, cholesky, cholesky_block_size=2
    )
    full = thc_omega_gh_algorithm8(
        y_occ, y_vir, core, cholesky, cholesky_block_size=9
    )
    for name in (
        "omega_g",
        "omega_h",
        "singles_gh",
        "appendix_line13_omega_h",
        "appendix_line13_singles_gh",
        "xi_oo",
        "xi_vv",
    ):
        np.testing.assert_allclose(
            getattr(paired, name), getattr(single, name), atol=2e-12
        )
        np.testing.assert_allclose(
            getattr(full, name), getattr(single, name), atol=2e-12
        )


def test_algorithm9_matches_dense_projected_published_eq42():
    (
        y_occ,
        y_vir,
        core,
        cholesky,
        fhat_oo,
        fhat_vv,
        *_,
    ) = _problem(807, symmetric_core=True)
    algorithm8 = thc_omega_gh_algorithm8(
        y_occ, y_vir, core, cholesky
    )
    observed = thc_omega_e_algorithm9(
        y_occ, y_vir, core, fhat_oo, fhat_vv, algorithm8
    )

    t2 = _dense_amplitudes(y_occ, y_vir, core)
    effective_oo = fhat_oo + algorithm8.xi_oo
    effective_vv = fhat_vv + algorithm8.xi_vv
    dense_eq42 = np.einsum("ijac,bc->ijab", t2, effective_vv)
    dense_eq42 += np.einsum("jica,bc->ijab", t2, effective_vv)
    dense_eq42 -= np.einsum("ikab,kj->ijab", t2, effective_oo)
    dense_eq42 -= np.einsum("kiba,kj->ijab", t2, effective_oo)
    projected = _project_dense(dense_eq42, y_occ, y_vir)
    expected = (projected + projected.T) / 2
    np.testing.assert_allclose(observed.omega_e, expected, atol=3e-10)


def test_algorithm9_nonsymmetric_core_matches_literal_appendix_schedule():
    (
        y_occ,
        y_vir,
        core,
        cholesky,
        fhat_oo,
        fhat_vv,
        *_,
    ) = _problem(808)
    assert not np.allclose(core, core.T)
    algorithm8 = thc_omega_gh_algorithm8(
        y_occ, y_vir, core, cholesky
    )
    expected = _literal_algorithm9(
        y_occ,
        y_vir,
        core,
        fhat_oo,
        fhat_vv,
        algorithm8.xi_oo,
        algorithm8.xi_vv,
    )
    observed = thc_omega_e_algorithm9(
        y_occ, y_vir, core, fhat_oo, fhat_vv, algorithm8
    )
    np.testing.assert_allclose(observed.omega_e, expected[0], atol=2e-12)
    np.testing.assert_allclose(
        observed.amplitude_metric, expected[1], atol=2e-12
    )
    np.testing.assert_allclose(
        observed.one_body_metric, expected[2], atol=2e-12
    )


def test_algorithm10_matches_dense_published_eq43():
    (
        y_occ,
        y_vir,
        core,
        _cholesky,
        _fhat_oo,
        _fhat_vv,
        fhat_ov,
        fhat_vo,
    ) = _problem(809, symmetric_core=True)
    t2 = _dense_amplitudes(y_occ, y_vir, core)
    expected = fhat_vo.T.copy()
    expected += 2 * np.einsum("ikac,kc->ia", t2, fhat_ov)
    expected -= np.einsum("ikca,kc->ia", t2, fhat_ov)
    observed = thc_omega_ij_algorithm10(
        y_occ, y_vir, core, fhat_ov, fhat_vo
    )
    np.testing.assert_allclose(observed.singles_ij, expected, atol=2e-11)


def test_algorithm10_nonsymmetric_core_matches_literal_appendix_schedule():
    (
        y_occ,
        y_vir,
        core,
        _cholesky,
        _fhat_oo,
        _fhat_vv,
        fhat_ov,
        fhat_vo,
    ) = _problem(810)
    assert not np.allclose(core, core.T)
    expected = _literal_algorithm10(y_occ, y_vir, core, fhat_ov, fhat_vo)
    observed = thc_omega_ij_algorithm10(
        y_occ, y_vir, core, fhat_ov, fhat_vo
    )
    np.testing.assert_allclose(observed.direct_fhat_vo, expected[0])
    np.testing.assert_allclose(observed.coulomb_like, expected[1], atol=2e-12)
    np.testing.assert_allclose(observed.exchange_like, expected[2], atol=2e-12)
    np.testing.assert_allclose(
        observed.singles_ij, sum(expected), atol=2e-12
    )
    metadata = observed.metadata()
    assert metadata[
        "published_equation43_equivalence_requires_symmetric_amplitude_core"
    ] is True
    assert metadata["nonsymmetric_amplitude_core_status"] == (
        "appendix-literal-schedule-audit-only-not-production-eligible"
    )


def test_algorithm10_nonsymmetric_core_is_mixed_transpose_not_eq43():
    (
        y_occ,
        y_vir,
        core,
        _cholesky,
        _fhat_oo,
        _fhat_vv,
        fhat_ov,
        fhat_vo,
    ) = _problem(821)
    assert not np.allclose(core, core.T)

    published = _published_eq43_parts_index_loop(
        y_occ, y_vir, core, fhat_ov, fhat_vo
    )
    transposed = _published_eq43_parts_index_loop(
        y_occ, y_vir, core.T, fhat_ov, fhat_vo
    )
    observed = thc_omega_ij_algorithm10(
        y_occ, y_vir, core, fhat_ov, fhat_vo
    )

    np.testing.assert_allclose(observed.direct_fhat_vo, published[0], atol=0)
    np.testing.assert_allclose(observed.coulomb_like, published[1], atol=2e-12)
    np.testing.assert_allclose(observed.exchange_like, transposed[2], atol=2e-12)
    np.testing.assert_allclose(
        observed.singles_ij,
        published[0] + published[1] + transposed[2],
        atol=2e-12,
    )
    assert np.max(np.abs(observed.singles_ij - sum(published))) > 1e-6

    metadata = observed.metadata()
    assert metadata["published_equation43_nonsymmetric_core_equivalence"] is False
    assert metadata["appendix_algorithm10_nonsymmetric_core_mapping"] == (
        "published-eq43-coulomb-uses-T-exchange-uses-T-transpose"
    )
    assert metadata["nonsymmetric_amplitude_core_production_eligible"] is False


def test_algorithms8_to10_metadata_states_semantic_boundaries():
    (
        y_occ,
        y_vir,
        core,
        cholesky,
        fhat_oo,
        fhat_vv,
        fhat_ov,
        fhat_vo,
    ) = _problem(811)
    algorithm8 = thc_omega_gh_algorithm8(
        y_occ, y_vir, core, cholesky, cholesky_block_size=2
    )
    algorithm9 = thc_omega_e_algorithm9(
        y_occ, y_vir, core, fhat_oo, fhat_vv, algorithm8
    )
    algorithm10 = thc_omega_ij_algorithm10(
        y_occ, y_vir, core, fhat_ov, fhat_vo
    )
    for number, equations, arxiv_v1_equations, result in (
        (8, [39, 40, 41], [38, 39, 40], algorithm8),
        (9, [40, 41, 42], [39, 40, 41], algorithm9),
        (10, [43], [42], algorithm10),
    ):
        metadata = result.metadata()
        assert metadata["paper_algorithm"] == number
        assert metadata["paper_equations"] == equations
        assert metadata["published_equations"] == equations
        assert metadata["arxiv_v1_equations"] == arxiv_v1_equations
        assert metadata["paper_version"] == "version-of-record"
        assert metadata["equation_numbering_authority"] == (
            "published-version-of-record"
        )
        assert metadata["paper_source"] == (
            "J. Chem. Phys. 156, 054102 (2022), "
            "DOI:10.1063/5.0077770"
        )
        assert metadata["arxiv_v1_source"] == "arXiv:2111.11473v1"
        assert metadata["audit_only"] is True
        assert metadata["production_enabled"] is False
        assert metadata["performance_eligible"] is False
        assert metadata["full_equation_ledger_uniquely_validated"] is False
        assert metadata["delta_xy_convention"] == (
            "published-algorithm8-line8-kronecker-identity"
        )
        assert metadata["delta_xy_convention_uniquely_validated"] is True
        assert metadata["delta_xy_pair_gram_alternative_rejected"] is True
        assert metadata["delta_xy_primary_source_basis"] == (
            "version-of-record-Algorithm8-line8-and-Eqs24-39"
        )
        assert metadata["transformed_f_convention_uniquely_validated"] is False
        assert metadata[
            "literal_equation_oracle_requires_symmetric_amplitude_core"
        ] is True
        assert metadata["core_symmetry_enforced"] is False
        assert metadata["implicit_host_tensor_transfer"] is False
        assert metadata["materializes_dense_t2"] is False
        assert metadata["materializes_four_index_eri"] is False
    assert algorithm9.metadata()["consumes_algorithm8_xi_only"] is True
    assert algorithm9.metadata()["consumes_algorithm8_singles"] is False
    metadata10 = algorithm10.metadata()
    assert metadata10[
        "published_equation43_equivalence_requires_symmetric_amplitude_core"
    ] is True
    assert metadata10["nonsymmetric_amplitude_core_status"] == (
        "appendix-literal-schedule-audit-only-not-production-eligible"
    )
    assert metadata10["published_equation43_nonsymmetric_core_equivalence"] is False
    assert metadata10["nonsymmetric_amplitude_core_production_eligible"] is False
    metadata8 = algorithm8.metadata()
    assert metadata8["published_equation39_occupied_orientation"] == (
        "l_ji-times-D_ja"
    )
    assert metadata8[
        "legacy_arxiv_v1_equation38_occupied_orientation"
    ] == "l_ji-times-D_ja"
    assert metadata8["appendix_algorithm8_line13_orientation"] == (
        "l_ij-times-D_ja"
    )
    assert metadata8[
        "algorithm8_line13_published_equation39_equivalence"
    ] is False
    assert metadata8["algorithm8_line13_status"] == (
        "retained-separately-fail-closed"
    )


@pytest.mark.parametrize(
    "arrays,match",
    [
        ((np.ones((2, 2)), np.ones((2, 3, 3)), np.ones((2, 2, 3))), "l_oo"),
        ((np.ones((2, 2, 2)), np.ones((2, 3)), np.ones((2, 2, 3))), "l_vv"),
        ((np.ones((2, 2, 2)), np.ones((2, 3, 3)), np.ones((2, 6))), "l_ov"),
        ((np.ones((2, 2, 3)), np.ones((2, 3, 3)), np.ones((2, 2, 3))), "l_oo"),
        ((np.ones((2, 2, 2)), np.ones((3, 3, 3)), np.ones((2, 2, 3))), "nchol"),
        ((np.ones((2, 2, 2)), np.ones((2, 3, 3)), np.ones((2, 3, 3))), "l_ov"),
    ],
)
def test_cholesky_shape_errors_fail_closed(arrays, match):
    with pytest.raises(ValueError, match=match):
        T1TransformedCholeskyBlocks(*arrays)


def test_cholesky_dtype_backend_and_nonfinite_errors_fail_closed():
    with pytest.raises(TypeError, match="NumPy or CuPy"):
        T1TransformedCholeskyBlocks(
            [[[1.0]]], np.ones((1, 1, 1)), np.ones((1, 1, 1))
        )
    with pytest.raises(TypeError, match="same dtype"):
        T1TransformedCholeskyBlocks(
            np.ones((1, 1, 1), dtype=np.float32),
            np.ones((1, 1, 1), dtype=np.float64),
            np.ones((1, 1, 1), dtype=np.float64),
        )
    with pytest.raises(TypeError, match="floating-point"):
        T1TransformedCholeskyBlocks(
            np.ones((1, 1, 1), dtype=np.int64),
            np.ones((1, 1, 1), dtype=np.int64),
            np.ones((1, 1, 1), dtype=np.int64),
        )
    with pytest.raises(NotImplementedError, match="real RHF"):
        T1TransformedCholeskyBlocks(
            np.ones((1, 1, 1), dtype=np.complex128),
            np.ones((1, 1, 1), dtype=np.complex128),
            np.ones((1, 1, 1), dtype=np.complex128),
        )
    bad = np.ones((1, 1, 1))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        T1TransformedCholeskyBlocks(
            np.ones((1, 1, 1)), np.ones((1, 1, 1)), bad
        )


def test_algorithm8_contract_errors_fail_closed():
    y_occ, y_vir, core, cholesky, *_ = _problem(812)
    with pytest.raises(TypeError, match="T1TransformedCholeskyBlocks"):
        thc_omega_gh_algorithm8(y_occ, y_vir, core, object())
    with pytest.raises(TypeError, match="cholesky_block_size"):
        thc_omega_gh_algorithm8(
            y_occ, y_vir, core, cholesky, cholesky_block_size=True
        )
    with pytest.raises(ValueError, match="positive"):
        thc_omega_gh_algorithm8(
            y_occ, y_vir, core, cholesky, cholesky_block_size=0
        )
    with pytest.raises(ValueError, match="amplitude_core must be square"):
        thc_omega_gh_algorithm8(y_occ, y_vir, core[:, :2], cholesky)
    with pytest.raises(TypeError, match="same dtype"):
        thc_omega_gh_algorithm8(
            y_occ.astype(np.float32), y_vir, core, cholesky
        )
    bad = y_vir.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        thc_omega_gh_algorithm8(y_occ, bad, core, cholesky)


def test_algorithm8_orbital_dimension_error_fails_closed():
    y_occ, y_vir, core, cholesky, *_ = _problem(813)
    other = T1TransformedCholeskyBlocks(
        np.ones((2, 3, 3)), np.ones((2, 3, 3)), np.ones((2, 3, 3))
    )
    with pytest.raises(ValueError, match="orbital dimensions"):
        thc_omega_gh_algorithm8(y_occ, y_vir, core, other)


def test_algorithm9_contract_errors_fail_closed():
    y_occ, y_vir, core, cholesky, fhat_oo, fhat_vv, *_ = _problem(814)
    algorithm8 = thc_omega_gh_algorithm8(y_occ, y_vir, core, cholesky)
    with pytest.raises(TypeError, match="THCOmegaGHAlgorithm8Result"):
        thc_omega_e_algorithm9(
            y_occ, y_vir, core, fhat_oo, fhat_vv, object()
        )
    with pytest.raises(ValueError, match="fhat_oo"):
        thc_omega_e_algorithm9(
            y_occ, y_vir, core, fhat_oo[:, :1], fhat_vv, algorithm8
        )
    with pytest.raises(ValueError, match="fhat_vv"):
        thc_omega_e_algorithm9(
            y_occ, y_vir, core, fhat_oo, fhat_vv[:, :2], algorithm8
        )
    wrong_rank = replace(algorithm8, amplitude_rank=9)
    with pytest.raises(ValueError, match="amplitude rank"):
        thc_omega_e_algorithm9(
            y_occ, y_vir, core, fhat_oo, fhat_vv, wrong_rank
        )


def test_algorithm10_contract_and_nonfinite_errors_fail_closed():
    y_occ, y_vir, core, _cholesky, *_, fhat_ov, fhat_vo = _problem(815)
    with pytest.raises(ValueError, match="fhat_ov"):
        thc_omega_ij_algorithm10(
            y_occ, y_vir, core, fhat_ov[:, :2], fhat_vo
        )
    with pytest.raises(ValueError, match="fhat_vo"):
        thc_omega_ij_algorithm10(
            y_occ, y_vir, core, fhat_ov, fhat_vo[:, :1]
        )
    bad = fhat_ov.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        thc_omega_ij_algorithm10(y_occ, y_vir, core, bad, fhat_vo)


def test_float32_paths_preserve_dtype():
    (
        y_occ,
        y_vir,
        core,
        cholesky,
        fhat_oo,
        fhat_vv,
        fhat_ov,
        fhat_vo,
    ) = _problem(816, np.float32)
    algorithm8 = thc_omega_gh_algorithm8(y_occ, y_vir, core, cholesky)
    algorithm9 = thc_omega_e_algorithm9(
        y_occ, y_vir, core, fhat_oo, fhat_vv, algorithm8
    )
    algorithm10 = thc_omega_ij_algorithm10(
        y_occ, y_vir, core, fhat_ov, fhat_vo
    )
    assert algorithm8.singles_gh.dtype == np.dtype(np.float32)
    assert algorithm8.xi_oo.dtype == np.dtype(np.float32)
    assert algorithm8.xi_vv.dtype == np.dtype(np.float32)
    assert algorithm9.omega_e.dtype == np.dtype(np.float32)
    assert algorithm10.singles_ij.dtype == np.dtype(np.float32)


def test_backend_mismatch_fails_closed_with_cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    y_occ, y_vir, core, cholesky, *_ = _problem(817)
    with pytest.raises(TypeError, match="one array backend"):
        thc_omega_gh_algorithm8(cp.asarray(y_occ), y_vir, core, cholesky)


def test_cupy_paths_match_numpy_and_record_only_control_scalar_reads():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    args = _problem(818)
    y_occ, y_vir, core, chol, f_oo, f_vv, f_ov, f_vo = args
    expected8 = thc_omega_gh_algorithm8(y_occ, y_vir, core, chol)
    expected9 = thc_omega_e_algorithm9(
        y_occ, y_vir, core, f_oo, f_vv, expected8
    )
    expected10 = thc_omega_ij_algorithm10(
        y_occ, y_vir, core, f_ov, f_vo
    )

    transfers = TransferCounter()
    gpu_chol = T1TransformedCholeskyBlocks(
        cp.asarray(chol.l_oo),
        cp.asarray(chol.l_vv),
        cp.asarray(chol.l_ov),
        transfer_counter=transfers,
    )
    gpu_args = tuple(cp.asarray(value) for value in (y_occ, y_vir, core))
    observed8 = thc_omega_gh_algorithm8(*gpu_args, gpu_chol)
    observed9 = thc_omega_e_algorithm9(
        *gpu_args,
        cp.asarray(f_oo),
        cp.asarray(f_vv),
        observed8,
    )
    observed10 = thc_omega_ij_algorithm10(
        *gpu_args, cp.asarray(f_ov), cp.asarray(f_vo), transfer_counter=transfers
    )
    for actual, expected in (
        (observed8.singles_gh, expected8.singles_gh),
        (
            observed8.appendix_line13_singles_gh,
            expected8.appendix_line13_singles_gh,
        ),
        (observed8.xi_oo, expected8.xi_oo),
        (observed8.xi_vv, expected8.xi_vv),
        (observed9.omega_e, expected9.omega_e),
        (observed10.singles_ij, expected10.singles_ij),
    ):
        assert isinstance(actual, cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=3e-10)
    report = transfers.to_dict()
    assert report["total_transfers"] == 7
    assert report["by_kind"]["d2h"] == {"bytes": 7, "count": 7}
    assert "h2d" not in report["by_kind"]


def test_device_mismatch_fails_closed_with_two_gpus():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("requires two CUDA devices")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    with cp.cuda.Device(0):
        transfers = TransferCounter()
        y_occ = cp.ones((2, 2))
        y_vir = cp.ones((3, 2))
        core = cp.eye(2)
        fhat_vo = cp.ones((3, 2))
    with cp.cuda.Device(1):
        fhat_ov = cp.ones((2, 3))
    with pytest.raises(TypeError, match="one device"):
        thc_omega_ij_algorithm10(
            y_occ,
            y_vir,
            core,
            fhat_ov,
            fhat_vo,
            transfer_counter=transfers,
        )
