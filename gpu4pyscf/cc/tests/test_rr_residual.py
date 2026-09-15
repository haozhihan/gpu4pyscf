"""Dense-reference tests for factorized RR residual terms."""

import contextlib

import numpy as np
import pytest

import gpu4pyscf.cc.rr_residual as rr_residual_module
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector, pair_matrix_from_t2
from gpu4pyscf.cc.rr_residual import (
    CCFockIntermediates,
    ProjectedPairDenominator,
    build_cc_lagrangian_one_body,
    build_projected_ccsd_doubles_components,
    build_projected_ccsd_doubles_numerator,
    build_ccsd_singles_numerator,
    build_cc_fock_intermediates,
    estimate_projected_cc_wvvvv_workspace_nbytes,
    projected_bare_ovov,
    projected_cc_woooo,
    projected_cc_ring,
    projected_cc_ring_gemm,
    estimate_rr_ring_workspace_nbytes,
    projected_cc_wvvvv,
    projected_cc_wvvvv_two_ladder,
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
    phase_records = []

    @contextlib.contextmanager
    def profile_phase(name, metadata):
        phase_records.append(("enter", name, dict(metadata)))
        try:
            yield
        finally:
            phase_records.append(("exit", name, dict(metadata)))

    components = build_projected_ccsd_doubles_components(
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
        profile_phase=profile_phase,
    )
    np.testing.assert_allclose(components.assemble_core(), observed.core)
    offset = np.eye(rank) * 0.125
    np.testing.assert_allclose(
        components.assemble_core(initial_core=-offset) + offset,
        observed.core,
    )
    assert components.metadata()["assembled_complete_core"] is False
    assert components.metadata()["wvvvv_kernel"] == "fused"
    expected_phases = [
        "rr_doubles_fock_intermediates",
        "rr_doubles_lagrangian",
        "rr_doubles_term_linear_t1",
        "rr_doubles_term_bare_ovov",
        "rr_doubles_term_woooo",
        "rr_doubles_term_wvvvv",
        "rr_doubles_term_one_body",
        "rr_doubles_term_ring",
    ]
    assert [
        name for action, name, _metadata in phase_records
        if action == "enter"
    ] == expected_phases
    assert [
        name for action, name, _metadata in phase_records
        if action == "exit"
    ] == expected_phases
    entered_metadata = [
        metadata for action, _name, metadata in phase_records
        if action == "enter"
    ]
    assert [item["coarse_term"] for item in entered_metadata] == [
        "fock_intermediates",
        "lagrangian",
        "linear_t1",
        "bare_ovov",
        "woooo",
        "wvvvv",
        "one_body",
        "ring",
    ]
    assert all(item["ring_kernel"] == "reference" for item in entered_metadata)
    assert all(item["wvvvv_kernel"] == "fused" for item in entered_metadata)
    assert all(item["auxiliary_block_size"] == 2 for item in entered_metadata)
    assert all(item["virtual_block_size"] == 3 for item in entered_metadata)
    two_ladder = build_projected_ccsd_doubles_numerator(
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
        wvvvv_kernel="two-ladder",
    )
    np.testing.assert_allclose(two_ladder.core, observed.core, atol=6e-10)
    assert observed.metadata()["wvvvv_kernel"] == "fused"
    assert two_ladder.metadata()["wvvvv_kernel"] == "two-ladder"
    observed_wvvvv = next(
        term for term in observed.metadata()["terms"]
        if term["term"] == "complete-cc-Wvvvv-ladder"
    )
    reference_wvvvv = next(
        term for term in two_ladder.metadata()["terms"]
        if term["term"] == "complete-cc-Wvvvv-ladder"
    )
    assert observed_wvvvv["kernel_name"] == "rr-wvvvv-fused"
    assert reference_wvvvv["kernel_name"] == "rr-wvvvv-two-ladder"
    gemm = build_projected_ccsd_doubles_numerator(
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
        ring_kernel="gemm",
    )
    np.testing.assert_allclose(gemm.core, observed.core, atol=6e-10)
    assert any(
        term["kernel_name"] == "rr-ring-gemm"
        for term in gemm.metadata()["terms"]
    )
    with pytest.raises(ValueError, match="ring_kernel"):
        build_projected_ccsd_doubles_numerator(
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
            ring_kernel="invalid",
        )
    with pytest.raises(ValueError, match="wvvvv_kernel"):
        build_projected_ccsd_doubles_numerator(
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
            wvvvv_kernel="invalid",
        )

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


@pytest.mark.parametrize(
    ("nocc", "nvir", "naux", "rank", "block_size"),
    [(2, 3, 5, 4, 2), (3, 4, 4, 5, 3), (2, 3, 3, 4, 1)],
)
def test_fused_cc_wvvvv_matches_two_ladder_fp64_reference(
    nocc, nvir, naux, rank, block_size
):
    rng = np.random.default_rng(8100 + nocc + 10 * nvir)
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    # Deliberately retain an asymmetric component: each old symmetric ladder
    # observes only sym(core), and the fused identity must preserve that.
    core = rng.normal(size=(rank, rank))
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    lov = rng.normal(size=(naux, nocc, nvir))
    lvv = rng.normal(size=(naux, nvir, nvir))

    reference = projected_cc_wvvvv_two_ladder(
        doubles,
        t1,
        lov,
        lvv,
        auxiliary_block_size=block_size,
    )
    observed = projected_cc_wvvvv(
        doubles,
        t1,
        lov,
        lvv,
        auxiliary_block_size=block_size,
    )

    np.testing.assert_allclose(
        observed.core, reference.core, atol=3e-11, rtol=3e-12
    )
    assert reference.kernel_name == "rr-wvvvv-two-ladder"
    assert observed.term == "complete-cc-Wvvvv-ladder"
    assert observed.auxiliary_block_size == block_size
    assert observed.kernel_name == "rr-wvvvv-fused"
    assert observed.materialized_dense_t2 is False
    assert observed.largest_intermediate_nbytes >= core.nbytes


def test_fused_cc_wvvvv_uses_bounded_subblocks_and_singleton_fallback(
    monkeypatch,
):
    rng = np.random.default_rng(8142)
    nocc, nvir, naux, rank, block_size = 2, 3, 5, 4, 2
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        rng.normal(size=(rank, rank)),
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    lov = rng.normal(size=(naux, nocc, nvir))
    lvv = rng.normal(size=(naux, nvir, nvir))
    calls = []
    original_einsum = np.einsum

    def counted_einsum(expression, *operands, **kwargs):
        calls.append(expression)
        return original_einsum(expression, *operands, **kwargs)

    monkeypatch.setattr(rr_residual_module.np, "einsum", counted_einsum)
    projected_cc_wvvvv(
        doubles,
        t1,
        lov,
        lvv,
        auxiliary_block_size=block_size,
    )

    workspace = estimate_projected_cc_wvvvv_workspace_nbytes(
        nocc,
        nvir,
        rank,
        naux,
        auxiliary_block_size=block_size,
    )
    fused_blocks = workspace["fused_auxiliary_subblock_count"]
    singleton_blocks = workspace["singleton_sequential_block_count"]
    assert calls.count("APR,RS,AQS->PQ") == 2 * singleton_blocks
    assert calls.count("APR,RS->APS") == fused_blocks
    assert calls.count("APS,AQS->PQ") == fused_blocks
    assert calls.count("AP,AQ->PQ") == fused_blocks + 2 * singleton_blocks


@pytest.mark.parametrize(
    ("naux", "block_size", "maximum_outer", "maximum_inner", "singletons"),
    [
        (8, 4, 4, 2, 0),
        (7, 4, 4, 2, 0),
        (5, 4, 4, 2, 1),
        (5, 1, 1, 0, 5),
        (8, 3, 3, 1, 0),
        (3, 8, 3, 1, 0),
    ],
)
def test_cc_wvvvv_workspace_caps_two_live_transforms_for_full_and_tail_blocks(
    naux, block_size, maximum_outer, maximum_inner, singletons
):
    nocc, nvir, rank, itemsize = 3, 7, 11, 8
    workspace = estimate_projected_cc_wvvvv_workspace_nbytes(
        nocc,
        nvir,
        rank,
        naux,
        auxiliary_block_size=block_size,
        itemsize=itemsize,
    )
    rank2 = rank * rank * itemsize

    assert workspace["maximum_actual_outer_block_size"] == maximum_outer
    assert (
        workspace["maximum_fused_auxiliary_subblock_size"]
        == maximum_inner
    )
    assert workspace["singleton_sequential_block_count"] == singletons
    assert workspace["rank_square_transform_contract_satisfied"] is True
    assert workspace["per_outer_block_contract_satisfied"] is True
    assert (
        workspace[
            "maximum_simultaneously_live_rank_square_transform_nbytes"
        ]
        <= workspace["prior_one_transform_outer_block_nbytes"]
    )
    assert workspace["prior_one_transform_outer_block_nbytes"] == (
        maximum_outer * rank2
    )
    expected_live = max(
        2 * maximum_inner * rank2,
        rank2 if singletons else 0,
    )
    assert workspace[
        "maximum_simultaneously_live_rank_square_transform_nbytes"
    ] == expected_live
    assert workspace["simultaneously_live_shape_upper_bound_nbytes"] == (
        workspace["persistent_sum_nbytes"]
        + workspace["temporary_sum_nbytes"]
    )


def test_cc_wvvvv_result_reports_tail_workspace_policy():
    rng = np.random.default_rng(8171)
    nocc, nvir, naux, rank, block_size = 2, 4, 5, 5, 4
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        rng.normal(size=(rank, rank)),
        nocc,
        nvir,
    )
    result = projected_cc_wvvvv(
        doubles,
        rng.normal(size=(nocc, nvir)),
        rng.normal(size=(naux, nocc, nvir)),
        rng.normal(size=(naux, nvir, nvir)),
        auxiliary_block_size=block_size,
    )
    workspace = result.metadata()["workspace_shape_upper_bound"]

    assert result.auxiliary_block_size == block_size
    assert workspace["maximum_actual_outer_block_size"] == 4
    assert workspace["maximum_fused_auxiliary_subblock_size"] == 2
    assert workspace["singleton_sequential_block_count"] == 1
    assert workspace["singleton_policy"] == "sequential-two-ladder"
    assert workspace["rank_square_transform_contract_satisfied"] is True
    assert result.largest_intermediate_nbytes <= workspace[
        "largest_logical_workspace_array_nbytes"
    ]


def test_gpu_fused_cc_wvvvv_matches_two_ladder_fp64_reference():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    rng = np.random.default_rng(8188)
    nocc, nvir, naux, rank, block_size = 2, 4, 5, 5, 2
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    core = rng.normal(size=(rank, rank))
    t1 = rng.normal(size=(nocc, nvir))
    lov = rng.normal(size=(naux, nocc, nvir))
    lvv = rng.normal(size=(naux, nvir, nvir))
    cpu_doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        core,
        nocc,
        nvir,
    )
    reference = projected_cc_wvvvv_two_ladder(
        cpu_doubles,
        t1,
        lov,
        lvv,
        auxiliary_block_size=block_size,
    )
    gpu_doubles = RRDoubles(
        RRProjector(
            cp.asarray(vectors),
            cp.asarray(-np.ones(rank)),
            0.0,
            nocc * nvir,
        ),
        cp.asarray(core),
        nocc,
        nvir,
    )
    observed = projected_cc_wvvvv(
        gpu_doubles,
        cp.asarray(t1),
        cp.asarray(lov),
        cp.asarray(lvv),
        auxiliary_block_size=block_size,
    )

    cp.testing.assert_allclose(
        observed.core, cp.asarray(reference.core), atol=3e-11, rtol=3e-12
    )


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


@pytest.mark.parametrize(
    ("wa", "wb", "expected_order"),
    [(1, 1, "residual-first"), (2, 3, "residual-first"),
     (5, 5, "right-project-first")],
)
def test_projected_ring_gemm_ordinary_tile_selector_matches_dense(
    wa, wb, expected_order
):
    """Random CPU oracle for both chains and unequal virtual tail tiles."""

    rng = np.random.default_rng(901 + 10 * wa + wb)
    nocc, nvir, rank = 2, 5, 4
    wvoov = rng.normal(size=(wa, nocc, nocc, nvir))
    t2_bc = rng.normal(size=(nocc, nocc, wb, nvir))
    left_u = rng.normal(size=(nocc, wa, rank))
    right_u = rng.normal(size=(nocc, wb, rank))

    residual = np.einsum("akic,kjbc->ijab", wvoov, t2_bc)
    expected = np.einsum(
        "iaP,ijab,jbQ->PQ", left_u, residual, right_u
    )
    observed, residual_nbytes, largest, selection = (
        rr_residual_module._projected_ring_gemm_tile(
            wvoov, t2_bc, left_u, right_u
        )
    )

    np.testing.assert_allclose(observed, expected, atol=2e-12, rtol=2e-12)
    assert selection["selected_order"] == expected_order
    selected = selection["candidates"][expected_order.replace("-", "_")]
    assert selected["estimated_gemm_flops"] == (
        2 * selected["scalar_multiply_count"]
    )
    assert residual_nbytes == (
        0 if expected_order == "right-project-first" else residual.nbytes
    )
    assert largest >= observed.nbytes


def test_projected_ring_gemm_wvovo_tile_keeps_bounded_residual_formula():
    rng = np.random.default_rng(949)
    nocc, nvir, rank, wa, wb = 2, 5, 4, 3, 2
    wvovo = rng.normal(size=(wb, nocc, nvir, nocc))
    t2_ac = rng.normal(size=(nocc, nocc, wa, nvir))
    left_u = rng.normal(size=(nocc, wa, rank))
    right_u = rng.normal(size=(nocc, wb, rank))

    residual = np.einsum("bkci,kjac->ijab", wvovo, t2_ac)
    expected = np.einsum(
        "iaP,ijab,jbQ->PQ", left_u, residual, right_u
    )
    observed, residual_nbytes, largest, selection = (
        rr_residual_module._projected_ring_gemm_tile(
            wvovo,
            t2_ac,
            left_u,
            right_u,
            right_is_wvovo=True,
        )
    )

    np.testing.assert_allclose(observed, expected, atol=2e-12, rtol=2e-12)
    assert residual_nbytes == residual.nbytes
    assert largest >= residual.nbytes
    assert selection["selected_order"] == "residual-first-right-project"


@pytest.mark.parametrize(
    ("case_name", "nocc", "nvir", "wa", "wb", "rank"),
    [
        ("water2", 10, 106, 32, 32, 374),
        ("water8", 40, 424, 32, 32, 9667),
        ("water8-tail", 40, 424, 32, 8, 9667),
    ],
)
def test_ring_matrix_chain_selector_rejects_costly_water_candidates(
    case_name, nocc, nvir, wa, wb, rank
):
    selection = rr_residual_module._select_projected_ring_matrix_chain(
        nocc, nvir, wa, wb, rank, 8
    )
    residual = selection["candidates"]["residual_first"]
    right = selection["candidates"]["right_project_first"]
    m = nocc * wa
    k = nocc * nvir
    n = nocc * wb

    assert selection["selected_order"] == "residual-first", case_name
    assert selection["dimensions"] == {"m": m, "k": k, "n": n, "r": rank}
    assert residual["scalar_multiply_count"] == (
        m * k * n + rank * m * n + rank * n * rank
    )
    assert right["scalar_multiply_count"] == (
        k * n * rank + m * k * rank + rank * m * rank
    )
    assert residual["logical_peak_temporary_nbytes"] == (
        2 * m * n + rank * n
    ) * 8
    assert right["logical_peak_temporary_nbytes"] == (
        k * rank + 2 * m * rank
    ) * 8
    assert residual["scalar_multiply_count"] < right[
        "scalar_multiply_count"
    ]
    assert residual["logical_peak_temporary_nbytes"] < right[
        "logical_peak_temporary_nbytes"
    ]
    assert selection["resource_contract"][
        "right_project_first_within_budget"
    ] is False
    if case_name == "water8":
        assert right["scalar_multiply_count"] > (
            3 * residual["scalar_multiply_count"]
        )


def test_ring_matrix_chain_selector_admits_beneficial_low_rank_candidate():
    selection = rr_residual_module._select_projected_ring_matrix_chain(
        2, 100, 32, 32, 1, 8
    )
    residual = selection["candidates"]["residual_first"]
    right = selection["candidates"]["right_project_first"]

    assert selection["selected_order"] == "right-project-first"
    assert right["scalar_multiply_count"] < residual[
        "scalar_multiply_count"
    ]
    assert right["logical_peak_temporary_nbytes"] < residual[
        "logical_peak_temporary_nbytes"
    ]
    assert selection["resource_contract"][
        "right_project_first_within_budget"
    ] is True


def test_ring_workspace_enumerates_mixed_tail_peak_instead_of_square_only(
    monkeypatch,
):
    """Protect the estimator if a backend policy makes a tail residual-first.

    For O=R=1, V=5 and Vblock=3, the square right-first chain needs eleven
    FP64 elements while a 3x2 residual-first chain needs fourteen.  The
    present arithmetic selector chooses right-first for both, so this test
    injects the possible mixed backend decision and audits only the workspace
    enumeration contract.
    """

    original = rr_residual_module._select_projected_ring_matrix_chain

    def mixed_tail_selector(nocc, nvir, wa, wb, rank, itemsize):
        selection = original(nocc, nvir, wa, wb, rank, itemsize)
        if (nocc, nvir, wa, wb, rank) == (1, 5, 3, 2, 1):
            selection["selected_order"] = "residual-first"
            selection["selection_reason"] = "test-mixed-tail-policy"
        return selection

    monkeypatch.setattr(
        rr_residual_module,
        "_select_projected_ring_matrix_chain",
        mixed_tail_selector,
    )
    model = estimate_rr_ring_workspace_nbytes(
        1,
        5,
        auxiliary_block_size=1,
        virtual_block_size=3,
        itemsize=8,
        rank=1,
    )

    square = model["ordinary_crossed_chain_selection"]
    peak = model["ordinary_crossed_peak_tile"]
    assert square["selected_order"] == "right-project-first"
    assert square["candidates"]["right_project_first"][
        "logical_peak_temporary_nbytes"
    ] == 11 * 8
    assert (peak["left_virtual_width"], peak["right_virtual_width"]) == (3, 2)
    assert peak["selected_order"] == "residual-first"
    assert peak["selected_candidate_temporary_sum_nbytes"] == 14 * 8
    assert model["ordinary_crossed_peak_tile_scope"] == (
        "maximum-over-all-full-and-tail-runtime-tile-shapes"
    )
    assert model["ordinary_crossed_peak_chain_temporary_nbytes"] == 14 * 8
    assert sum(model["temporaries"]["crossed_gemm"].values()) == 14 * 8
    ordinary_with_common = (
        model["ordinary_crossed_common_temporary_nbytes"] + 14 * 8
    )
    assert model["ordinary_crossed_peak_sum_nbytes"] == ordinary_with_common
    wvovo_sum = sum(
        model["temporaries"]["crossed_wvovo_gemm"].values()
    )
    assert model["crossed_gemm_peak_sum_nbytes"] == max(
        ordinary_with_common, wvovo_sum
    )
    assert model["simultaneously_live_shape_upper_bound_nbytes"] == (
        model["persistent_sum_nbytes"]
        + model["temporary_sum_nbytes"]
    )


@pytest.mark.parametrize("rank", [4, 10])
@pytest.mark.parametrize("auxiliary_block_size", [1, 2, 4])
@pytest.mark.parametrize("virtual_block_size", [1, 2, 3, 5])
def test_projected_complete_cc_ring_gemm_matches_reference_for_rr_ranks_and_blocks(
    rank, auxiliary_block_size, virtual_block_size
):
    """The GEMM path must match the bounded equation-level ring oracle."""

    rng = np.random.default_rng(812 + rank * 10 + auxiliary_block_size)
    nocc, nvir, naux = 2, 5, 4
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

    reference = projected_cc_ring(
        doubles,
        t1,
        loo,
        lov,
        lvv,
        auxiliary_block_size=auxiliary_block_size,
        virtual_block_size=virtual_block_size,
    )
    observed = projected_cc_ring_gemm(
        doubles,
        t1,
        loo,
        lov,
        lvv,
        auxiliary_block_size=auxiliary_block_size,
        virtual_block_size=virtual_block_size,
    )
    np.testing.assert_allclose(observed.core, reference.core, atol=3e-10, rtol=3e-10)
    assert observed.term == "complete-cc-Wvoov-Wvovo-ring-gemm"
    assert observed.kernel_name == "rr-ring-gemm"
    assert observed.metadata()["kernel_name"] == "rr-ring-gemm"
    workspace = observed.metadata()["workspace_shape_upper_bound"]
    assert observed.tile_residual_nbytes == workspace[
        "wvovo_crossed_ijab_tile_nbytes"
    ]
    selection = workspace["ordinary_crossed_chain_selection"]
    selected_key = selection["selected_order"].replace("-", "_")
    selected = selection["candidates"][selected_key]
    expected_full_tile_ijab = (
        0
        if selection["selected_order"] == "right-project-first"
        else workspace["ijab_tile_nbytes"]
    )
    assert workspace["ordinary_crossed_full_tile_ijab_tile_nbytes"] == (
        expected_full_tile_ijab
    )
    assert workspace["ordinary_crossed_full_tile_materialized_ijab"] is (
        selected["materialized_ijab"]
    )
    assert workspace["ordinary_crossed_contraction_order_scope"] == (
        "maximum-square-virtual-tile"
    )
    assert workspace["wvovo_crossed_materialized_ijab"] is True
    assert workspace["ordinary_crossed_contraction_order"] == selected[
        "order"
    ]
    runtime_counts = workspace[
        "ordinary_crossed_runtime_selected_order_counts"
    ]
    ntile = (nvir + virtual_block_size - 1) // virtual_block_size
    assert sum(runtime_counts.values()) == ntile**2
    runtime_tiles = workspace[
        "ordinary_crossed_runtime_tile_selections"
    ]
    assert runtime_tiles
    assert all("candidates" in item for item in runtime_tiles)
    materialized_ordinary = runtime_counts.get("residual-first", 0) > 0
    assert workspace["ordinary_crossed_materialized_ijab"] is (
        materialized_ordinary
    )
    expected_runtime_ijab = max(
        (
            nocc
            * nocc
            * item["left_virtual_width"]
            * item["right_virtual_width"]
            * np.dtype(vectors.dtype).itemsize
            for item in runtime_tiles
            if item["selected_order"] == "residual-first"
        ),
        default=0,
    )
    assert workspace["ordinary_crossed_ijab_tile_nbytes"] == (
        expected_runtime_ijab
    )
    assert workspace["ordinary_crossed_selected_orders"] == sorted(
        runtime_counts
    )
    if rank == 4 and virtual_block_size == 3:
        assert selection["selected_order"] == "right-project-first"
        assert sorted(runtime_counts) == [
            "residual-first",
            "right-project-first",
        ]
        assert workspace["ordinary_crossed_full_tile_materialized_ijab"] is (
            False
        )
        assert workspace["ordinary_crossed_materialized_ijab"] is True
    assert (
        observed.metadata()["tile_residual_nbytes"]
        == observed.tile_residual_nbytes
    )
    assert observed.materialized_dense_t2 is False


def test_projected_complete_cc_ring_gemm_default_auxiliary_batch_and_validation():
    rng = np.random.default_rng(813)
    nocc, nvir, naux, rank = 2, 3, 3, 4
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    doubles = RRDoubles(
        RRProjector(vectors, -np.ones(rank), 0.0, nocc * nvir),
        np.eye(rank),
        nocc,
        nvir,
    )
    t1 = rng.normal(size=(nocc, nvir))
    loo = rng.normal(size=(naux, nocc, nocc))
    lov = rng.normal(size=(naux, nocc, nvir))
    lvv = rng.normal(size=(naux, nvir, nvir))
    result = projected_cc_ring_gemm(
        doubles, t1, loo, lov, lvv, virtual_block_size=2
    )
    assert result.auxiliary_block_size == 1
    assert result.metadata()["largest_intermediate_nbytes"] == result.largest_intermediate_nbytes
    assert result.metadata()["largest_intermediate_scope"] == "single-logical-array"
    assert "simultaneously_live_shape_upper_bound_nbytes" in result.metadata()[
        "workspace_shape_upper_bound"
    ]
    with pytest.raises(ValueError, match="block sizes"):
        projected_cc_ring_gemm(
            doubles,
            t1,
            loo,
            lov,
            lvv,
            auxiliary_block_size=0,
        )
    with pytest.raises(ValueError, match="block sizes"):
        projected_cc_ring_gemm(
            doubles,
            t1,
            loo,
            lov,
            lvv,
            virtual_block_size=0,
        )


def test_water8_ring_workspace_estimate_is_auxiliary_block_bounded():
    estimate_one = estimate_rr_ring_workspace_nbytes(
        40, 424, auxiliary_block_size=1, virtual_block_size=32
    )
    estimate_four = estimate_rr_ring_workspace_nbytes(
        40, 424, auxiliary_block_size=4, virtual_block_size=32
    )
    # The Wvoov crossed temporary is 40^3*32 FP64 values, about 15.6 MiB
    # (the precise value is shape-dependent).  It scales with A, not naux.
    assert 15.0 * 2**20 < estimate_one["crossed_intermediate_nbytes"] < 16.0 * 2**20
    assert (
        estimate_four["crossed_intermediate_nbytes"]
        == 4 * estimate_one["crossed_intermediate_nbytes"]
    )
    assert estimate_one["largest_logical_array_nbytes"] >= estimate_one[
        "crossed_intermediate_nbytes"
    ]
    full_virtual = estimate_rr_ring_workspace_nbytes(
        40, 424, auxiliary_block_size=1, virtual_block_size=424
    )
    assert full_virtual["ijab_tile_nbytes"] == 40 * 40 * 424 * 424 * 8


def test_water8_full_rank_ring_shape_peak_model_accounts_for_rank2_temporaries():
    nocc, nvir, vblock, itemsize = 40, 424, 32, 8
    rank = nocc * nvir
    model = estimate_rr_ring_workspace_nbytes(
        nocc,
        nvir,
        auxiliary_block_size=1,
        virtual_block_size=vblock,
        itemsize=itemsize,
        rank=rank,
    )
    rank2 = rank * rank * itemsize
    ijab = nocc * nocc * vblock * vblock * itemsize
    chain = 3 * rank2
    persistent = model["persistent"]
    direct = model["temporaries"]["direct_gemm"]
    crossed_common = model["temporaries"]["crossed_gemm_common"]
    crossed = model["temporaries"]["crossed_gemm"]
    crossed_wvovo = model["temporaries"]["crossed_wvovo_gemm"]
    selection = model["ordinary_crossed_chain_selection"]

    assert persistent["half_nbytes"] == rank2
    assert persistent["pair_metric_nbytes"] == rank2
    assert direct["left_wvoov_nbytes"] == rank2
    assert direct["left_wvovo_nbytes"] == rank2
    assert direct["direct_chain_matmul_temporary_nbytes"] == chain
    assert selection["selected_order"] == "residual-first"
    assert crossed["ijab_residual_nbytes"] == ijab
    assert crossed["ijab_transpose_reshape_copy_nbytes"] == ijab
    assert crossed["projected_left_nbytes"] == (
        rank * nocc * vblock * itemsize
    )
    assert crossed_common["projected_return_nbytes"] == rank2
    assert crossed_wvovo["ijab_residual_nbytes"] == ijab
    assert crossed_wvovo["ijab_transpose_reshape_copy_nbytes"] == ijab
    assert crossed_wvovo["projected_return_nbytes"] == rank2
    assert model["ordinary_crossed_materialized_ijab"] is True
    assert model["ordinary_crossed_full_tile_materialized_ijab"] is True
    assert model["wvovo_crossed_materialized_ijab"] is True
    assert model["simultaneously_live_shape_upper_bound_nbytes"] == (
        model["persistent_sum_nbytes"]
        + model["temporary_sum_nbytes"]
    )
    assert model["simultaneously_live_shape_upper_bound_nbytes"] > rank2


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
