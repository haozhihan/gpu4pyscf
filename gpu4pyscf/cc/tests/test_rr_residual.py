"""Dense-reference tests for factorized RR residual terms."""

import numpy as np
import pytest

from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector, pair_matrix_from_t2
from gpu4pyscf.cc.rr_residual import (
    CCFockIntermediates,
    ProjectedPairDenominator,
    build_cc_lagrangian_one_body,
    build_projected_ccsd_doubles_numerator,
    build_ccsd_singles_numerator,
    build_cc_fock_intermediates,
    projected_bare_ovov,
    projected_cc_woooo,
    projected_cc_ring,
    projected_cc_wvvvv,
    projected_coulomb_ppl,
    projected_linear_t1_doubles,
    projected_one_body_dressing,
    projected_oooo_ladder,
    projected_vvvv_ladder,
    rr_ccsd_energy,
    singles_from_cd_ovoo,
    singles_from_cd_ovvo_oovv,
    singles_from_cd_ovvv,
    singles_from_fov_doubles,
    singles_from_ovoo_doubles,
)


def _dense_rccsd_doubles_numerator(
    t1,
    t2,
    fock_oo,
    fock_ov,
    fock_vv,
    occupied_energies,
    virtual_energies,
    loo,
    lov,
    lvv,
    level_shift,
):
    nocc, nvir = t1.shape
    ovov = np.einsum("Akc,Ald->kcld", lov, lov)
    ovvo = np.einsum("Akc,Aia->kcai", lov, lov)
    oovv = np.einsum("Aki,Aac->kiac", loo, lvv)
    ovvv = np.einsum("Akd,Aac->kdac", lov, lvv)
    ovoo = np.einsum("Alc,Aki->lcki", lov, loo)
    oooo = np.einsum("Aki,Alj->kilj", loo, loo)
    vvvv = np.einsum("Aab,Acd->abcd", lvv, lvv)

    foo = fock_oo.copy()
    foo += 2.0 * np.einsum("kcld,ilcd->ki", ovov, t2)
    foo -= np.einsum("kdlc,ilcd->ki", ovov, t2)
    foo += 2.0 * np.einsum("kcld,ic,ld->ki", ovov, t1, t1)
    foo -= np.einsum("kdlc,ic,ld->ki", ovov, t1, t1)
    fvv = fock_vv.copy()
    fvv -= 2.0 * np.einsum("kcld,klad->ac", ovov, t2)
    fvv += np.einsum("kdlc,klad->ac", ovov, t2)
    fvv -= 2.0 * np.einsum("kcld,ka,ld->ac", ovov, t1, t1)
    fvv += np.einsum("kdlc,ka,ld->ac", ovov, t1, t1)
    lagrangian_oo = foo + np.einsum("kc,ic->ki", fock_ov, t1)
    lagrangian_oo += 2.0 * np.einsum("lcki,lc->ki", ovoo, t1)
    lagrangian_oo -= np.einsum("kcli,lc->ki", ovoo, t1)
    lagrangian_vv = fvv - np.einsum("kc,ka->ac", fock_ov, t1)
    lagrangian_vv += 2.0 * np.einsum("kdac,kd->ac", ovvv, t1)
    lagrangian_vv -= np.einsum("kcad,kd->ac", ovvv, t1)
    lagrangian_oo[np.diag_indices(nocc)] -= occupied_energies
    lagrangian_vv[np.diag_indices(nvir)] -= virtual_energies + level_shift

    tmp2 = np.einsum("kibc,ka->abic", oovv, -t1)
    tmp2 += ovvv.transpose(1, 3, 0, 2)
    tmp = np.einsum("abic,jc->ijab", tmp2, t1)
    numerator = tmp + tmp.transpose(1, 0, 3, 2)
    tmp2 = np.einsum("kcai,jc->akij", ovvo, t1)
    tmp2 += ovoo.transpose(1, 3, 0, 2)
    tmp = np.einsum("akij,kb->ijab", tmp2, t1)
    numerator -= tmp + tmp.transpose(1, 0, 3, 2)
    numerator += ovov.transpose(0, 2, 1, 3)

    woooo = np.einsum("lcki,jc->klij", ovoo, t1)
    woooo += np.einsum("kclj,ic->klij", ovoo, t1)
    woooo += np.einsum("kcld,ijcd->klij", ovov, t2)
    woooo += np.einsum("kcld,ic,jd->klij", ovov, t1, t1)
    woooo += oooo.transpose(0, 2, 1, 3)
    wvvvv = -np.einsum("kdac,kb->abcd", ovvv, t1)
    wvvvv -= np.einsum("kcbd,ka->abcd", ovvv, t1)
    wvvvv += vvvv.transpose(0, 2, 1, 3)
    wvoov = np.einsum("kcad,id->akic", ovvv, t1)
    wvoov -= np.einsum("kcli,la->akic", ovoo, t1)
    wvoov += ovvo.transpose(2, 0, 3, 1)
    wvoov -= 0.5 * np.einsum("ldkc,ilda->akic", ovov, t2)
    wvoov -= 0.5 * np.einsum("lckd,ilad->akic", ovov, t2)
    wvoov -= np.einsum("ldkc,id,la->akic", ovov, t1, t1)
    wvoov += np.einsum("ldkc,ilad->akic", ovov, t2)
    wvovo = np.einsum("kdac,id->akci", ovvv, t1)
    wvovo -= np.einsum("lcki,la->akci", ovoo, t1)
    wvovo += oovv.transpose(2, 0, 3, 1)
    wvovo -= 0.5 * np.einsum("lckd,ilda->akci", ovov, t2)
    wvovo -= np.einsum("lckd,id,la->akci", ovov, t1, t1)
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    numerator += np.einsum("klij,klab->ijab", woooo, tau)
    numerator += np.einsum("abcd,ijcd->ijab", wvvvv, tau)
    tmp = np.einsum("ac,ijcb->ijab", lagrangian_vv, t2)
    numerator += tmp + tmp.transpose(1, 0, 3, 2)
    tmp = np.einsum("ki,kjab->ijab", lagrangian_oo, t2)
    numerator -= tmp + tmp.transpose(1, 0, 3, 2)
    tmp = 2.0 * np.einsum("akic,kjcb->ijab", wvoov, t2)
    tmp -= np.einsum("akci,kjcb->ijab", wvovo, t2)
    numerator += tmp + tmp.transpose(1, 0, 3, 2)
    tmp = np.einsum("akic,kjbc->ijab", wvoov, t2)
    numerator -= tmp + tmp.transpose(1, 0, 3, 2)
    tmp = np.einsum("bkci,kjac->ijab", wvovo, t2)
    numerator -= tmp + tmp.transpose(1, 0, 3, 2)
    return numerator


def test_projected_bare_ovov_matches_dense_reference():
    rng = np.random.default_rng(101)
    nocc, nvir, naux, rank = 3, 4, 6, 5
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    projector = RRProjector(
        vectors, np.linspace(-1.0, -0.2, rank), 0.0, nocc * nvir
    )
    lov = rng.normal(size=(naux, nocc, nvir))
    dense = np.einsum("Aia,Ajb->ijab", lov, lov)
    expected = vectors.T @ pair_matrix_from_t2(dense) @ vectors
    observed = projected_bare_ovov(projector, lov, nocc, nvir)
    np.testing.assert_allclose(observed.core, expected, atol=2e-12, rtol=2e-12)
    assert observed.materialized_dense_t2 is False


def test_projected_explicit_linear_t1_doubles_matches_dense_reference():
    rng = np.random.default_rng(121)
    nocc, nvir, naux, rank = 3, 5, 6, 4
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    projector = RRProjector(
        vectors, -np.ones(rank), 0.0, nocc * nvir
    )
    t1 = rng.normal(size=(nocc, nvir))
    raw_loo = rng.normal(size=(naux, nocc, nocc))
    loo = (raw_loo + raw_loo.transpose(0, 2, 1)) * 0.5
    lov = rng.normal(size=(naux, nocc, nvir))
    raw_lvv = rng.normal(size=(naux, nvir, nvir))
    lvv = (raw_lvv + raw_lvv.transpose(0, 2, 1)) * 0.5
    oovv = np.einsum("Aki,Abc->kibc", loo, lvv)
    ovvv = np.einsum("Akd,Aac->kdac", lov, lvv)
    ovvo = np.einsum("Akc,Aia->kcai", lov, lov)
    ovoo = np.einsum("Alc,Aki->lcki", lov, loo)

    tmp2 = np.einsum("kibc,ka->abic", oovv, -t1)
    tmp2 += ovvv.transpose(1, 3, 0, 2)
    tmp = np.einsum("abic,jc->ijab", tmp2, t1)
    dense = tmp + tmp.transpose(1, 0, 3, 2)
    tmp2 = np.einsum("kcai,jc->akij", ovvo, t1)
    tmp2 += ovoo.transpose(1, 3, 0, 2)
    tmp = np.einsum("akij,kb->ijab", tmp2, t1)
    dense -= tmp + tmp.transpose(1, 0, 3, 2)
    expected = vectors.T @ pair_matrix_from_t2(dense) @ vectors
    observed = projected_linear_t1_doubles(
        projector,
        t1,
        loo,
        lov,
        lvv,
        auxiliary_block_size=2,
    )
    np.testing.assert_allclose(observed.core, expected, atol=5e-11)
    assert observed.materialized_dense_t2 is False


def test_cc_fock_intermediates_match_dense_ovov_reference():
    rng = np.random.default_rng(102)
    nocc, nvir, naux, rank = 3, 4, 7, 5
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    lov = rng.normal(size=(naux, nocc, nvir))
    fock_oo = rng.normal(size=(nocc, nocc))
    fock_ov = rng.normal(size=(nocc, nvir))
    fock_vv = rng.normal(size=(nvir, nvir))
    t2 = doubles.reconstruct_t2()
    ovov = np.einsum("Akc,Ald->kcld", lov, lov)

    expected_foo = fock_oo.copy()
    expected_foo += 2.0 * np.einsum("kcld,ilcd->ki", ovov, t2)
    expected_foo -= np.einsum("kdlc,ilcd->ki", ovov, t2)
    expected_foo += 2.0 * np.einsum("kcld,ic,ld->ki", ovov, t1, t1)
    expected_foo -= np.einsum("kdlc,ic,ld->ki", ovov, t1, t1)

    expected_fvv = fock_vv.copy()
    expected_fvv -= 2.0 * np.einsum("kcld,klad->ac", ovov, t2)
    expected_fvv += np.einsum("kdlc,klad->ac", ovov, t2)
    expected_fvv -= 2.0 * np.einsum("kcld,ka,ld->ac", ovov, t1, t1)
    expected_fvv += np.einsum("kdlc,ka,ld->ac", ovov, t1, t1)

    expected_fov = fock_ov.copy()
    expected_fov += 2.0 * np.einsum("kcld,ld->kc", ovov, t1)
    expected_fov -= np.einsum("kdlc,ld->kc", ovov, t1)

    observed = build_cc_fock_intermediates(
        doubles,
        t1,
        lov,
        fock_oo,
        fock_ov,
        fock_vv,
        auxiliary_block_size=3,
    )
    np.testing.assert_allclose(observed.foo, expected_foo, atol=3e-11)
    np.testing.assert_allclose(observed.fov, expected_fov, atol=3e-11)
    np.testing.assert_allclose(observed.fvv, expected_fvv, atol=3e-11)
    assert observed.materialized_dense_t2 is False
    assert observed.materialized_dense_ovov is False


def test_rr_ccsd_energy_matches_dense_reference():
    rng = np.random.default_rng(104)
    nocc, nvir, naux, rank = 3, 5, 8, 6
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    fock_ov = rng.normal(size=(nocc, nvir))
    lov = rng.normal(size=(naux, nocc, nvir))
    t2 = doubles.reconstruct_t2()
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    ovov = np.einsum("Aia,Ajb->iajb", lov, lov)
    expected = 2.0 * np.einsum("ia,ia", fock_ov, t1)
    expected += 2.0 * np.einsum("ijab,iajb", tau, ovov)
    expected -= np.einsum("ijab,ibja", tau, ovov)
    observed = rr_ccsd_energy(
        doubles, t1, fock_ov, lov, auxiliary_block_size=3
    )
    np.testing.assert_allclose(observed.value, expected, atol=3e-11)
    assert observed.materialized_dense_t2 is False
    assert observed.materialized_dense_ovov is False


def test_cc_lagrangian_one_body_matches_dense_reference():
    rng = np.random.default_rng(116)
    nocc, nvir, naux = 3, 5, 7
    foo = rng.normal(size=(nocc, nocc))
    fvv = rng.normal(size=(nvir, nvir))
    fov = rng.normal(size=(nocc, nvir))
    t1 = rng.normal(size=(nocc, nvir))
    raw_loo = rng.normal(size=(naux, nocc, nocc))
    loo = (raw_loo + raw_loo.transpose(0, 2, 1)) * 0.5
    lov = rng.normal(size=(naux, nocc, nvir))
    raw_lvv = rng.normal(size=(naux, nvir, nvir))
    lvv = (raw_lvv + raw_lvv.transpose(0, 2, 1)) * 0.5
    ovoo = np.einsum("Alc,Aki->lcki", lov, loo)
    ovvv = np.einsum("Akd,Aac->kdac", lov, lvv)

    expected_loo = foo.copy()
    expected_loo += np.einsum("kc,ic->ki", fov, t1)
    expected_loo += 2.0 * np.einsum("lcki,lc->ki", ovoo, t1)
    expected_loo -= np.einsum("kcli,lc->ki", ovoo, t1)
    expected_lvv = fvv.copy()
    expected_lvv -= np.einsum("kc,ka->ac", fov, t1)
    expected_lvv += 2.0 * np.einsum("kdac,kd->ac", ovvv, t1)
    expected_lvv -= np.einsum("kcad,kd->ac", ovvv, t1)

    intermediates = CCFockIntermediates(
        foo=foo,
        fov=fov,
        fvv=fvv,
        auxiliary_block_size=naux,
        largest_intermediate_nbytes=0,
    )
    observed = build_cc_lagrangian_one_body(
        intermediates, t1, fov, loo, lov, lvv
    )
    np.testing.assert_allclose(observed.loo, expected_loo, atol=3e-12)
    np.testing.assert_allclose(observed.lvv, expected_lvv, atol=3e-12)
    assert observed.materialized_four_index_eri is False


def test_complete_cd_rr_singles_numerator_matches_dense_rccsd_equation():
    rng = np.random.default_rng(112)
    nocc, nvir, naux, rank = 3, 4, 7, 5
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    t2 = doubles.reconstruct_t2()
    t1 = rng.normal(size=(nocc, nvir))
    fock_oo = rng.normal(size=(nocc, nocc))
    fock_ov = rng.normal(size=(nocc, nvir))
    fock_vv = rng.normal(size=(nvir, nvir))
    occupied_energies = -np.linspace(1.4, 0.5, nocc)
    virtual_energies = np.linspace(0.2, 1.3, nvir)
    level_shift = 0.07
    raw_loo = rng.normal(size=(naux, nocc, nocc))
    loo = (raw_loo + raw_loo.transpose(0, 2, 1)) * 0.5
    lov = rng.normal(size=(naux, nocc, nvir))
    raw_lvv = rng.normal(size=(naux, nvir, nvir))
    lvv = (raw_lvv + raw_lvv.transpose(0, 2, 1)) * 0.5

    ovov = np.einsum("Akc,Ald->kcld", lov, lov)
    ovvo = np.einsum("Akc,Aia->kcai", lov, lov)
    oovv = np.einsum("Aki,Aac->kiac", loo, lvv)
    ovvv = np.einsum("Akd,Aac->kdac", lov, lvv)
    ovoo = np.einsum("Alc,Aki->lcki", lov, loo)
    foo = fock_oo.copy()
    foo += 2.0 * np.einsum("kcld,ilcd->ki", ovov, t2)
    foo -= np.einsum("kdlc,ilcd->ki", ovov, t2)
    foo += 2.0 * np.einsum("kcld,ic,ld->ki", ovov, t1, t1)
    foo -= np.einsum("kdlc,ic,ld->ki", ovov, t1, t1)
    fvv = fock_vv.copy()
    fvv -= 2.0 * np.einsum("kcld,klad->ac", ovov, t2)
    fvv += np.einsum("kdlc,klad->ac", ovov, t2)
    fvv -= 2.0 * np.einsum("kcld,ka,ld->ac", ovov, t1, t1)
    fvv += np.einsum("kdlc,ka,ld->ac", ovov, t1, t1)
    fov = fock_ov.copy()
    fov += 2.0 * np.einsum("kcld,ld->kc", ovov, t1)
    fov -= np.einsum("kdlc,ld->kc", ovov, t1)
    foo[np.diag_indices(nocc)] -= occupied_energies
    fvv[np.diag_indices(nvir)] -= virtual_energies + level_shift

    expected = -2.0 * np.einsum("kc,ka,ic->ia", fock_ov, t1, t1)
    expected += np.einsum("ac,ic->ia", fvv, t1)
    expected -= np.einsum("ki,ka->ia", foo, t1)
    expected += 2.0 * np.einsum("kc,kica->ia", fov, t2)
    expected -= np.einsum("kc,ikca->ia", fov, t2)
    expected += np.einsum("kc,ic,ka->ia", fov, t1, t1)
    expected += fock_ov
    expected += 2.0 * np.einsum("kcai,kc->ia", ovvo, t1)
    expected -= np.einsum("kiac,kc->ia", oovv, t1)
    expected += 2.0 * np.einsum("kdac,ikcd->ia", ovvv, t2)
    expected -= np.einsum("kcad,ikcd->ia", ovvv, t2)
    expected += 2.0 * np.einsum("kdac,kd,ic->ia", ovvv, t1, t1)
    expected -= np.einsum("kcad,kd,ic->ia", ovvv, t1, t1)
    expected -= 2.0 * np.einsum("lcki,klac->ia", ovoo, t2)
    expected += np.einsum("kcli,klac->ia", ovoo, t2)
    expected -= 2.0 * np.einsum("lcki,lc,ka->ia", ovoo, t1, t1)
    expected += np.einsum("kcli,lc,ka->ia", ovoo, t1, t1)

    observed = build_ccsd_singles_numerator(
        doubles,
        t1,
        fock_oo,
        fock_ov,
        fock_vv,
        occupied_energies,
        virtual_energies,
        loo,
        lov,
        lvv,
        level_shift=level_shift,
        auxiliary_block_size=3,
    )
    np.testing.assert_allclose(observed.numerator, expected, atol=7e-11)
    eia = occupied_energies[:, None] - virtual_energies - level_shift
    np.testing.assert_allclose(observed.amplitudes(eia), expected / eia)
    assert observed.materialized_dense_t2 is False
    assert observed.materialized_four_index_eri is False


@pytest.mark.parametrize("full_rank", [False, True])
def test_complete_projected_doubles_numerator_matches_dense_equation(full_rank):
    rng = np.random.default_rng(124 if full_rank else 123)
    nocc, nvir, naux = 2, 4, 5
    dimension = nocc * nvir
    rank = dimension if full_rank else 3
    vectors, _ = np.linalg.qr(rng.normal(size=(dimension, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, dimension),
        core,
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    occupied_energies = -np.linspace(1.3, 0.6, nocc)
    virtual_energies = np.linspace(0.2, 1.1, nvir)
    level_shift = 0.03
    fock_oo = rng.normal(size=(nocc, nocc))
    fock_ov = rng.normal(size=(nocc, nvir))
    fock_vv = rng.normal(size=(nvir, nvir))
    raw_loo = rng.normal(size=(naux, nocc, nocc))
    loo = (raw_loo + raw_loo.transpose(0, 2, 1)) * 0.5
    lov = rng.normal(size=(naux, nocc, nvir))
    raw_lvv = rng.normal(size=(naux, nvir, nvir))
    lvv = (raw_lvv + raw_lvv.transpose(0, 2, 1)) * 0.5
    t2 = doubles.reconstruct_t2()
    dense = _dense_rccsd_doubles_numerator(
        t1,
        t2,
        fock_oo,
        fock_ov,
        fock_vv,
        occupied_energies,
        virtual_energies,
        loo,
        lov,
        lvv,
        level_shift,
    )
    expected = vectors.T @ pair_matrix_from_t2(dense) @ vectors
    observed = build_projected_ccsd_doubles_numerator(
        doubles,
        t1,
        fock_oo,
        fock_ov,
        fock_vv,
        occupied_energies,
        virtual_energies,
        loo,
        lov,
        lvv,
        level_shift=level_shift,
        auxiliary_block_size=2,
        virtual_block_size=3,
    )
    np.testing.assert_allclose(observed.core, expected, atol=5e-10)
    assert observed.materialized_dense_t2 is False
    assert observed.materialized_four_index_eri is False

    if full_rank:
        denominator = ProjectedPairDenominator.build(
            doubles.projector,
            occupied_energies[:, None]
            - virtual_energies[None, :]
            - level_shift,
            nocc,
            nvir,
        )
        updated_core = observed.solve(denominator)
        updated = vectors @ updated_core @ vectors.T
        pair_differences = (
            occupied_energies[:, None]
            - virtual_energies[None, :]
            - level_shift
        ).reshape(-1)
        expected_update = pair_matrix_from_t2(dense) / (
            pair_differences[:, None] + pair_differences[None, :]
        )
        np.testing.assert_allclose(updated, expected_update, atol=7e-10)


def test_projected_one_body_dressing_matches_dense_reference():
    rng = np.random.default_rng(103)
    nocc, nvir, rank = 3, 5, 6
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    projector = RRProjector(
        vectors, np.linspace(-1.0, -0.2, rank), 0.0, nocc * nvir
    )
    doubles = RRDoubles(projector, core, nocc, nvir)
    ft_ij = rng.normal(size=(nocc, nocc))
    ft_ab = rng.normal(size=(nvir, nvir))
    dense_t2 = doubles.reconstruct_t2()
    half = np.einsum("ijac,bc->ijab", dense_t2, ft_ab)
    half -= np.einsum("ki,kjab->ijab", ft_ij, dense_t2)
    dense = half + half.transpose(1, 0, 3, 2)
    expected = vectors.T @ pair_matrix_from_t2(dense) @ vectors
    observed = projected_one_body_dressing(doubles, ft_ij, ft_ab)
    np.testing.assert_allclose(observed.core, expected, atol=2e-12, rtol=2e-12)
    assert observed.materialized_dense_t2 is False


def test_fov_doubles_singles_matches_dense_reference():
    rng = np.random.default_rng(105)
    nocc, nvir, rank = 3, 4, 5
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    fov = rng.normal(size=(nocc, nvir))
    dense = doubles.reconstruct_t2()
    expected = 2.0 * np.einsum("jb,ijab->ia", fov, dense)
    expected -= np.einsum("jb,ijba->ia", fov, dense)
    observed = singles_from_fov_doubles(doubles, fov)
    np.testing.assert_allclose(observed.amplitudes, expected, atol=2e-12)
    assert observed.materialized_dense_t2 is False


def test_ovoo_doubles_singles_matches_dense_reference():
    rng = np.random.default_rng(106)
    nocc, nvir, rank = 3, 4, 5
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    ovoo = rng.normal(size=(nocc, nvir, nocc, nocc))
    dense = doubles.reconstruct_t2()
    effective = 2.0 * ovoo - ovoo.transpose(2, 1, 0, 3)
    expected = -np.einsum("jbki,jkba->ia", effective, dense)
    observed = singles_from_ovoo_doubles(doubles, ovoo)
    np.testing.assert_allclose(observed.amplitudes, expected, atol=2e-12)
    assert observed.materialized_dense_t2 is False


def test_cd_ovvv_singles_matches_dense_reference():
    rng = np.random.default_rng(108)
    nocc, nvir, naux, rank = 3, 4, 6, 5
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    lov = rng.normal(size=(naux, nocc, nvir))
    raw_lvv = rng.normal(size=(naux, nvir, nvir))
    lvv = (raw_lvv + raw_lvv.transpose(0, 2, 1)) * 0.5
    ovvv = np.einsum("Akd,Aac->kdac", lov, lvv)
    t2 = doubles.reconstruct_t2()
    expected = 2.0 * np.einsum("kdac,ikcd->ia", ovvv, t2)
    expected -= np.einsum("kcad,ikcd->ia", ovvv, t2)
    expected += 2.0 * np.einsum("kdac,kd,ic->ia", ovvv, t1, t1)
    expected -= np.einsum("kcad,kd,ic->ia", ovvv, t1, t1)
    observed = singles_from_cd_ovvv(
        doubles, t1, lov, lvv, auxiliary_block_size=2
    )
    np.testing.assert_allclose(observed.amplitudes, expected, atol=3e-11)
    assert observed.materialized_dense_t2 is False


def test_cd_ovoo_singles_matches_dense_reference():
    rng = np.random.default_rng(109)
    nocc, nvir, naux, rank = 3, 4, 6, 5
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    lov = rng.normal(size=(naux, nocc, nvir))
    raw_loo = rng.normal(size=(naux, nocc, nocc))
    loo = (raw_loo + raw_loo.transpose(0, 2, 1)) * 0.5
    ovoo = np.einsum("Alc,Aki->lcki", lov, loo)
    t2 = doubles.reconstruct_t2()
    expected = -2.0 * np.einsum("lcki,klac->ia", ovoo, t2)
    expected += np.einsum("kcli,klac->ia", ovoo, t2)
    expected -= 2.0 * np.einsum("lcki,lc,ka->ia", ovoo, t1, t1)
    expected += np.einsum("kcli,lc,ka->ia", ovoo, t1, t1)
    observed = singles_from_cd_ovoo(
        doubles, t1, lov, loo, auxiliary_block_size=2
    )
    np.testing.assert_allclose(observed.amplitudes, expected, atol=3e-11)
    assert observed.materialized_dense_t2 is False


def test_cd_ovvo_oovv_singles_matches_dense_reference():
    rng = np.random.default_rng(111)
    nocc, nvir, naux = 3, 4, 6
    t1 = rng.normal(size=(nocc, nvir))
    lov = rng.normal(size=(naux, nocc, nvir))
    raw_loo = rng.normal(size=(naux, nocc, nocc))
    loo = (raw_loo + raw_loo.transpose(0, 2, 1)) * 0.5
    raw_lvv = rng.normal(size=(naux, nvir, nvir))
    lvv = (raw_lvv + raw_lvv.transpose(0, 2, 1)) * 0.5
    ovvo = np.einsum("Akc,Aia->kcai", lov, lov)
    oovv = np.einsum("Aki,Aac->kiac", loo, lvv)
    expected = 2.0 * np.einsum("kcai,kc->ia", ovvo, t1)
    expected -= np.einsum("kiac,kc->ia", oovv, t1)
    observed = singles_from_cd_ovvo_oovv(t1, loo, lov, lvv)
    np.testing.assert_allclose(observed.amplitudes, expected, atol=3e-11)


def test_projected_lyapunov_denominator_reproduces_full_rank_jacobi():
    rng = np.random.default_rng(107)
    nocc, nvir = 2, 3
    dimension = nocc * nvir
    vectors, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    projector = RRProjector(
        vectors, -np.arange(1, dimension + 1, dtype=float), 0.0, dimension
    )
    differences = -np.linspace(0.4, 2.1, dimension).reshape(nocc, nvir)
    numerator = rng.normal(size=(dimension, dimension))
    numerator = (numerator + numerator.T) * 0.5
    projected = vectors.T @ numerator @ vectors
    denominator = ProjectedPairDenominator.build(
        projector,
        differences,
        nocc,
        nvir,
        singular_tolerance=1e-12,
    )
    core = denominator.solve(projected)
    observed = vectors @ core @ vectors.T
    flat = differences.reshape(-1)
    expected = numerator / (flat[:, None] + flat[None, :])
    np.testing.assert_allclose(observed, expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        denominator.equation_residual(core, projected), 0.0, atol=2e-12
    )
    assert denominator.metadata()["materializes_pair_denominator"] is False


def test_projected_lyapunov_denominator_rejects_nonsymmetric_numerator():
    projector = RRProjector(np.eye(4), -np.ones(4), 0.0, 4)
    denominator = ProjectedPairDenominator.build(
        projector, -np.ones((2, 2)), 2, 2
    )
    numerator = np.eye(4)
    numerator[0, 1] = 1e-3
    with pytest.raises(ValueError, match="not symmetric"):
        denominator.solve(numerator)


def test_projected_coulomb_ppl_matches_dense_reference():
    rng = np.random.default_rng(110)
    nocc, nvir, naux, rank = 3, 5, 7, 6
    dimension = nocc * nvir
    vectors, _ = np.linalg.qr(rng.normal(size=(dimension, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    projector = RRProjector(
        vectors=vectors,
        eigenvalues=np.linspace(2.0, 0.5, rank),
        cutoff=0.0,
        full_dimension=dimension,
    )
    doubles = RRDoubles(projector, core, nocc, nvir)
    t1 = rng.normal(size=(nocc, nvir))
    raw_lvv = rng.normal(size=(naux, nvir, nvir))
    lvv = (raw_lvv + raw_lvv.transpose(0, 2, 1)) * 0.5

    observed = projected_coulomb_ppl(
        doubles, t1, lvv, auxiliary_block_size=3
    )
    dense_t2 = doubles.reconstruct_t2()
    tau = dense_t2 + np.einsum("ia,jb->ijab", t1, t1)
    dense_residual = 0.5 * np.einsum(
        "Aab,Acd,ijcd->ijab", lvv, lvv, tau
    )
    expected = vectors.T @ pair_matrix_from_t2(dense_residual) @ vectors
    expected = (expected + expected.T) * 0.5
    np.testing.assert_allclose(observed.core, expected, atol=2e-11, rtol=2e-11)
    assert observed.materialized_dense_t2 is False
    assert observed.largest_intermediate_nbytes < dense_t2.nbytes


def test_projected_vvvv_ladder_matches_dense_reference():
    rng = np.random.default_rng(114)
    nocc, nvir, naux, rank = 3, 5, 6, 4
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    raw = rng.normal(size=(naux, nvir, nvir))
    lvv = (raw + raw.transpose(0, 2, 1)) * 0.5
    tau = doubles.reconstruct_t2() + np.einsum("ia,jb->ijab", t1, t1)
    dense = np.einsum("Aac,Abd,ijcd->ijab", lvv, lvv, tau)
    expected = vectors.T @ pair_matrix_from_t2(dense) @ vectors
    observed = projected_vvvv_ladder(
        doubles, t1, lvv, auxiliary_block_size=2
    )
    np.testing.assert_allclose(observed.core, expected, atol=4e-11)
    assert observed.materialized_dense_t2 is False


def test_projected_oooo_ladder_matches_dense_reference():
    rng = np.random.default_rng(115)
    nocc, nvir, naux, rank = 3, 4, 6, 5
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    raw = rng.normal(size=(naux, nocc, nocc))
    loo = (raw + raw.transpose(0, 2, 1)) * 0.5
    tau = doubles.reconstruct_t2() + np.einsum("ia,jb->ijab", t1, t1)
    dense = np.einsum("Aki,Alj,klab->ijab", loo, loo, tau)
    expected = vectors.T @ pair_matrix_from_t2(dense) @ vectors
    observed = projected_oooo_ladder(
        doubles, t1, loo, auxiliary_block_size=2
    )
    np.testing.assert_allclose(observed.core, expected, atol=4e-11)
    assert observed.materialized_dense_t2 is False


@pytest.mark.parametrize("full_rank", [False, True])
def test_projected_complete_cc_woooo_matches_dense_reference(full_rank):
    rng = np.random.default_rng(119 if full_rank else 117)
    nocc, nvir, naux = 2, 5, 4
    dimension = nocc * nvir
    rank = dimension if full_rank else 4
    vectors, _ = np.linalg.qr(rng.normal(size=(dimension, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, dimension),
        core,
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    raw_loo = rng.normal(size=(naux, nocc, nocc))
    loo = (raw_loo + raw_loo.transpose(0, 2, 1)) * 0.5
    lov = rng.normal(size=(naux, nocc, nvir))
    t2 = doubles.reconstruct_t2()
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    ovoo = np.einsum("Alc,Aki->lcki", lov, loo)
    ovov = np.einsum("Akc,Ald->kcld", lov, lov)
    oooo = np.einsum("Aki,Alj->kilj", loo, loo)
    woooo = np.einsum("lcki,jc->klij", ovoo, t1)
    woooo += np.einsum("kclj,ic->klij", ovoo, t1)
    woooo += np.einsum("kcld,ijcd->klij", ovov, t2)
    woooo += np.einsum("kcld,ic,jd->klij", ovov, t1, t1)
    woooo += oooo.transpose(0, 2, 1, 3)
    dense = np.einsum("klij,klab->ijab", woooo, tau)
    expected = vectors.T @ pair_matrix_from_t2(dense) @ vectors
    observed = projected_cc_woooo(
        doubles,
        t1,
        loo,
        lov,
        auxiliary_block_size=2,
        virtual_block_size=2,
    )
    np.testing.assert_allclose(observed.core, expected, atol=8e-11)
    assert observed.term == "complete-cc-Woooo-ladder"
    assert observed.materialized_dense_t2 is False


def test_projected_complete_cc_wvvvv_matches_dense_reference():
    rng = np.random.default_rng(120)
    nocc, nvir, naux, rank = 3, 5, 6, 4
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    lov = rng.normal(size=(naux, nocc, nvir))
    raw_lvv = rng.normal(size=(naux, nvir, nvir))
    lvv = (raw_lvv + raw_lvv.transpose(0, 2, 1)) * 0.5
    t2 = doubles.reconstruct_t2()
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    ovvv = np.einsum("Akd,Aac->kdac", lov, lvv)
    vvvv = np.einsum("Aab,Acd->abcd", lvv, lvv)
    wvvvv = -np.einsum("kdac,kb->abcd", ovvv, t1)
    wvvvv -= np.einsum("kcbd,ka->abcd", ovvv, t1)
    wvvvv += vvvv.transpose(0, 2, 1, 3)
    dense = np.einsum("abcd,ijcd->ijab", wvvvv, tau)
    expected = vectors.T @ pair_matrix_from_t2(dense) @ vectors
    observed = projected_cc_wvvvv(
        doubles, t1, lov, lvv, auxiliary_block_size=2
    )
    np.testing.assert_allclose(observed.core, expected, atol=8e-11)
    assert observed.term == "complete-cc-Wvvvv-ladder"
    assert observed.materialized_dense_t2 is False


def test_projected_complete_cc_ring_matches_dense_reference():
    rng = np.random.default_rng(122)
    nocc, nvir, naux, rank = 2, 5, 4, 4
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    raw_loo = rng.normal(size=(naux, nocc, nocc))
    loo = (raw_loo + raw_loo.transpose(0, 2, 1)) * 0.5
    lov = rng.normal(size=(naux, nocc, nvir))
    raw_lvv = rng.normal(size=(naux, nvir, nvir))
    lvv = (raw_lvv + raw_lvv.transpose(0, 2, 1)) * 0.5
    t2 = doubles.reconstruct_t2()
    ovvv = np.einsum("Akd,Aac->kdac", lov, lvv)
    ovoo = np.einsum("Alc,Aki->lcki", lov, loo)
    ovvo = np.einsum("Akc,Aia->kcai", lov, lov)
    oovv = np.einsum("Aki,Aac->kiac", loo, lvv)
    ovov = np.einsum("Akc,Ald->kcld", lov, lov)
    wvoov = np.einsum("kcad,id->akic", ovvv, t1)
    wvoov -= np.einsum("kcli,la->akic", ovoo, t1)
    wvoov += ovvo.transpose(2, 0, 3, 1)
    wvoov -= 0.5 * np.einsum("ldkc,ilda->akic", ovov, t2)
    wvoov -= 0.5 * np.einsum("lckd,ilad->akic", ovov, t2)
    wvoov -= np.einsum("ldkc,id,la->akic", ovov, t1, t1)
    wvoov += np.einsum("ldkc,ilad->akic", ovov, t2)
    wvovo = np.einsum("kdac,id->akci", ovvv, t1)
    wvovo -= np.einsum("lcki,la->akci", ovoo, t1)
    wvovo += oovv.transpose(2, 0, 3, 1)
    wvovo -= 0.5 * np.einsum("lckd,ilda->akci", ovov, t2)
    wvovo -= np.einsum("lckd,id,la->akci", ovov, t1, t1)

    half = 2.0 * np.einsum("akic,kjcb->ijab", wvoov, t2)
    half -= np.einsum("akci,kjcb->ijab", wvovo, t2)
    half -= np.einsum("akic,kjbc->ijab", wvoov, t2)
    half -= np.einsum("bkci,kjac->ijab", wvovo, t2)
    dense = half + half.transpose(1, 0, 3, 2)
    expected = vectors.T @ pair_matrix_from_t2(dense) @ vectors
    observed = projected_cc_ring(
        doubles,
        t1,
        loo,
        lov,
        lvv,
        auxiliary_block_size=2,
        virtual_block_size=2,
    )
    np.testing.assert_allclose(observed.core, expected, atol=2e-10)
    assert observed.materialized_dense_t2 is False


def test_projected_coulomb_ppl_rejects_shape_mismatch():
    projector = RRProjector(np.eye(4), np.ones(4), 0.0, 4)
    doubles = RRDoubles(projector, np.eye(4), 2, 2)
    with np.testing.assert_raises(ValueError):
        projected_coulomb_ppl(doubles, np.zeros((2, 2)), np.zeros((3, 3, 3)))


def test_projected_coulomb_ppl_rejects_nonsymmetric_factors():
    projector = RRProjector(np.eye(4), np.ones(4), 0.0, 4)
    doubles = RRDoubles(projector, np.eye(4), 2, 2)
    lvv = np.zeros((1, 2, 2))
    lvv[0, 0, 1] = 1e-4
    with pytest.raises(ValueError, match="not symmetric"):
        projected_coulomb_ppl(doubles, np.zeros((2, 2)), lvv)


def test_projected_coulomb_ppl_rejects_integer_tensors():
    projector = RRProjector(np.eye(4), np.ones(4), 0.0, 4)
    doubles = RRDoubles(projector, np.eye(4), 2, 2)
    with pytest.raises(TypeError, match="floating-point"):
        projected_coulomb_ppl(
            doubles,
            np.zeros((2, 2), dtype=int),
            np.zeros((1, 2, 2)),
        )


def test_gpu_projected_coulomb_ppl_matches_dense_reference_and_counts_sync():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    rng = np.random.default_rng(118)
    nocc, nvir, naux, rank = 2, 3, 4, 3
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    t1 = rng.normal(size=(nocc, nvir))
    raw_lvv = rng.normal(size=(naux, nvir, nvir))
    lvv = (raw_lvv + raw_lvv.transpose(0, 2, 1)) * 0.5
    projector = RRProjector(
        cp.asarray(vectors), cp.asarray(np.ones(rank)), 0.0, nocc * nvir
    )
    doubles = RRDoubles(projector, cp.asarray(core), nocc, nvir)
    transfers = TransferCounter()
    result = projected_coulomb_ppl(
        doubles,
        cp.asarray(t1),
        cp.asarray(lvv),
        auxiliary_block_size=2,
        transfer_counter=transfers,
    )
    cpu_projector = RRProjector(vectors, np.ones(rank), 0.0, nocc * nvir)
    expected = projected_coulomb_ppl(
        RRDoubles(cpu_projector, core, nocc, nvir),
        t1,
        lvv,
        auxiliary_block_size=2,
    )
    cp.testing.assert_allclose(result.core, cp.asarray(expected.core))
    assert transfers.to_dict()["by_kind"]["d2h"]["bytes"] > 0
