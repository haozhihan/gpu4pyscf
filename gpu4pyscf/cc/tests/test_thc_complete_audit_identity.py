"""Numerical identity gates for the complete Algorithms 1--10 audit.

One fixture uses the complete occupied--virtual pair basis for both the
amplitude and ERI THC representations.  A second fits a generic RR projector
at rank below ``OV``, applies the Eq. 11--14 symmetric orthogonalization, and
uses the reconstructed non-one-hot projector as an independent RR/CD equation
oracle.  Both routes compare complete singles and doubles numerators in FP64.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gpu4pyscf.cc import thc_complete_audit as complete
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.rr_residual import (
    build_ccsd_singles_numerator,
    build_projected_ccsd_doubles_components,
)
from gpu4pyscf.cc.thc_complete_audit import (
    IdentityAttestedERITHCFactors,
    assemble_complete_thc_ccsd_audit,
)
from gpu4pyscf.cc.thc_eri import ERITHCFactors
from gpu4pyscf.cc.thc_factorization import fit_weighted_thc_projector
from gpu4pyscf.cc.thc_fhat import build_t1_transformed_fhat
from gpu4pyscf.cc.thc_residual import rr_residual_algorithms_1_3


def _canonical_pair_factors(
    nocc: int, nvir: int
) -> tuple[np.ndarray, np.ndarray]:
    pair_index = np.arange(nocc * nvir)
    occupied = (
        np.arange(nocc)[:, None] == pair_index[None, :] // nvir
    ).astype(np.float64)
    virtual = (
        np.arange(nvir)[:, None] == pair_index[None, :] % nvir
    ).astype(np.float64)
    return occupied, virtual


def _full_pair_problem(
    seed: int = 1701,
    *,
    nocc: int = 2,
    nvir: int = 2,
    t1_scale: float = 0.0,
):
    rng = np.random.default_rng(seed)
    pair_dimension = nocc * nvir
    nmo = nocc + nvir

    y_occ, y_vir = _canonical_pair_factors(nocc, nvir)
    permutation = rng.permutation(pair_dimension)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=pair_dimension)
    y_occ = y_occ[:, permutation] * signs[None, :]
    y_vir = y_vir[:, permutation]
    pair_y = (y_occ[:, None, :] * y_vir[None, :, :]).reshape(
        pair_dimension, pair_dimension
    )
    projector_vectors, _ = np.linalg.qr(pair_y)
    tau = projector_vectors.T @ pair_y

    rr_core = rng.normal(size=(pair_dimension, pair_dimension))
    rr_core = (rr_core + rr_core.T) * 0.035
    amplitude_core = tau.T @ rr_core @ tau
    projector = RRProjector(
        projector_vectors,
        -np.linspace(1.0, 0.4, pair_dimension),
        cutoff=0.0,
        full_dimension=pair_dimension,
    )
    doubles = RRDoubles(
        projector, rr_core, nocc=nocc, nvir=nvir
    )

    x_occ, x_vir = _canonical_pair_factors(nocc, nvir)
    cd_ov = rng.normal(size=(pair_dimension, pair_dimension)) * 0.17
    eri_core = cd_ov.T @ cd_ov
    raw_eri = ERITHCFactors(x_occ, x_vir, eri_core)

    raw_l = np.zeros((pair_dimension, nmo, nmo), dtype=np.float64)
    raw_l[:, :nocc, nocc:] = cd_ov.reshape(pair_dimension, nocc, nvir)
    raw_l[:, nocc:, :nocc] = raw_l[:, :nocc, nocc:].swapaxes(1, 2)
    loo = rng.normal(size=(pair_dimension, nocc, nocc)) * 0.11
    lvv = rng.normal(size=(pair_dimension, nvir, nvir)) * 0.13
    raw_l[:, :nocc, :nocc] = (loo + loo.swapaxes(1, 2)) * 0.5
    raw_l[:, nocc:, nocc:] = (lvv + lvv.swapaxes(1, 2)) * 0.5

    h_mo = rng.normal(size=(nmo, nmo)) * 0.07
    h_mo = (h_mo + h_mo.T) * 0.5
    t1 = rng.normal(size=(nocc, nvir)) * t1_scale
    token = {"case": "complete-full-pair-identity", "seed": seed}
    integral_token = f"complete-full-pair-eri-{seed}"
    provenance = {
        "orbital_identity": token,
        "integral_identity": integral_token,
        "hcore_identity": token,
    }
    bare_fock = build_t1_transformed_fhat(
        h_mo,
        raw_l,
        np.zeros_like(t1),
        auxiliary_block_size=2,
        provenance_context=provenance,
    )
    fhat = build_t1_transformed_fhat(
        h_mo,
        raw_l,
        t1,
        auxiliary_block_size=2,
        provenance_context=provenance,
    )
    attested_eri = IdentityAttestedERITHCFactors(
        raw_eri,
        orbital_identity_token=token,
        integral_identity_token=integral_token,
    )

    oracle_arguments = (
        doubles,
        t1,
        bare_fock.fhat_oo,
        bare_fock.fhat_ov,
        bare_fock.fhat_vv,
        np.zeros(nocc, dtype=np.float64),
        np.zeros(nvir, dtype=np.float64),
        raw_l[:, :nocc, :nocc],
        raw_l[:, :nocc, nocc:],
        raw_l[:, nocc:, nocc:],
    )
    audit_arguments = {
        "y_occ": y_occ,
        "y_vir": y_vir,
        "amplitude_core": amplitude_core,
        "tau": tau,
        "eri_factors": attested_eri,
        "fhat": fhat,
        "orbital_identity_token": token,
        "integral_identity_token": integral_token,
        "hcore_identity_token": token,
        "algorithm4_5_outer_block_size": 2,
        "algorithm6_x_block_size": 2,
        "algorithm7_x_block_size": 2,
        "algorithm8_cholesky_block_size": 2,
    }
    return {
        "pair_y": pair_y,
        "projector_vectors": projector_vectors,
        "tau": tau,
        "doubles": doubles,
        "amplitude_core": amplitude_core,
        "raw_eri": raw_eri,
        "raw_l": raw_l,
        "nocc": nocc,
        "nvir": nvir,
        "oracle_arguments": oracle_arguments,
        "audit_arguments": audit_arguments,
    }


def _inexact_orthogonalized_cp_problem(
    seed: int,
    *,
    nocc: int = 3,
    nvir: int = 3,
    rr_rank: int = 3,
    thc_rank: int = 4,
):
    """Return a compressed, dense CP gauge after the Eq. 11--14 flow.

    The ALS target is a generic semi-unitary RR projector.  Its deliberately
    truncated CP fit is then symmetrically orthogonalized by
    :func:`fit_weighted_thc_projector`.  The independent equation oracle uses
    the *reconstructed* semi-unitary projector, rather than the pre-fit
    target.  This isolates the complete-equation identity from CP fitting
    error while exercising the same non-full-pair factors that an inexact
    amplitude-THC calculation would consume.  The ERI side deliberately stays
    at its analytic full-pair/CD endpoint so this gate diagnoses amplitude
    compression alone.
    """

    pair_dimension = nocc * nvir
    rng = np.random.default_rng(seed + 100_000)

    target_vectors, _ = np.linalg.qr(
        rng.normal(size=(pair_dimension, rr_rank))
    )
    eigenvalues = -np.linspace(1.3, 0.35, rr_rank)
    target_projector = RRProjector(
        target_vectors,
        eigenvalues,
        cutoff=1e-6,
        full_dimension=pair_dimension,
        source="generic-inexact-cp-target",
    )
    factors = fit_weighted_thc_projector(
        target_projector,
        nocc,
        nvir,
        thc_rank=thc_rank,
        fit_tolerance=0.0,
        orthogonality_cutoff=1e-12,
        orthogonality_tolerance=1e-12,
        max_iterations=40,
        als_convergence_tolerance=0.0,
        ridge=1e-12,
        seed=seed,
        allow_unconverged=True,
    )

    pair_y = np.einsum(
        "iW,aW->iaW", factors.y_occ, factors.y_vir
    ).reshape(pair_dimension, thc_rank)
    reconstructed_vectors = pair_y @ factors.tau.T
    effective_projector = RRProjector(
        reconstructed_vectors,
        eigenvalues,
        cutoff=target_projector.cutoff,
        full_dimension=pair_dimension,
        source="eq11-14-orthogonalized-inexact-cp",
    )
    rr_core = rng.normal(size=(rr_rank, rr_rank))
    rr_core = (rr_core + rr_core.T) * 0.035
    doubles = RRDoubles(
        effective_projector, rr_core, nocc=nocc, nvir=nvir
    )
    amplitude_core = factors.tau.T @ rr_core @ factors.tau
    amplitude_core = (amplitude_core + amplitude_core.T) * 0.5

    # Reuse only the independently generated physical Hamiltonian/CD data and
    # provenance fixture.  Replace every amplitude-side object by the dense,
    # compressed CP construction above.
    base = _full_pair_problem(
        seed, nocc=nocc, nvir=nvir, t1_scale=0.04
    )
    oracle_arguments = (
        doubles,
        *base["oracle_arguments"][1:],
    )
    audit_arguments = dict(base["audit_arguments"])
    audit_arguments.update(
        y_occ=factors.y_occ,
        y_vir=factors.y_vir,
        amplitude_core=amplitude_core,
        tau=factors.tau,
    )
    return {
        "target_vectors": target_vectors,
        "target_projector": target_projector,
        "factors": factors,
        "pair_y": pair_y,
        "reconstructed_vectors": reconstructed_vectors,
        "doubles": doubles,
        "amplitude_core": amplitude_core,
        "oracle_arguments": oracle_arguments,
        "audit_arguments": audit_arguments,
        "nocc": nocc,
        "nvir": nvir,
        "rr_rank": rr_rank,
        "thc_rank": thc_rank,
        "pair_dimension": pair_dimension,
    }


@pytest.mark.parametrize(
    "seed,nocc,nvir,t1_scale,omega_ac_path",
    [
        (1701, 2, 2, 0.0, "joint-6"),
        (1701, 2, 2, 0.0, "separate-4-plus-5"),
        (1702, 2, 3, 0.04, "joint-6"),
        (1702, 2, 3, 0.04, "separate-4-plus-5"),
        (1703, 3, 2, 0.03, "separate-4-plus-5"),
        (1703, 3, 2, 0.03, "joint-6"),
    ],
)
def test_complete_full_pair_identity_diagnostic(
    seed, nocc, nvir, t1_scale, omega_ac_path
):
    problem = _full_pair_problem(
        seed, nocc=nocc, nvir=nvir, t1_scale=t1_scale
    )
    np.testing.assert_allclose(
        problem["projector_vectors"],
        problem["pair_y"] @ problem["tau"].T,
        atol=2e-15,
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        problem["tau"],
        problem["projector_vectors"].T @ problem["pair_y"],
        atol=0.0,
        rtol=0.0,
    )
    dense_thc_t2 = (
        problem["pair_y"]
        @ problem["amplitude_core"]
        @ problem["pair_y"].T
    )
    np.testing.assert_allclose(
        dense_thc_t2,
        problem["doubles"].reconstruct_pair_matrix(),
        atol=3e-15,
        rtol=3e-15,
    )
    np.testing.assert_allclose(
        problem["raw_eri"].reconstruct_ovov(),
        np.einsum(
            "Aia,Ajb->iajb",
            problem["raw_l"][:, :nocc, nocc:],
            problem["raw_l"][:, :nocc, nocc:],
        ),
        atol=2e-16,
        rtol=2e-15,
    )
    assert np.min(np.linalg.eigvalsh(problem["raw_eri"].core)) >= -2e-16

    components = build_projected_ccsd_doubles_components(
        *problem["oracle_arguments"],
        auxiliary_block_size=2,
        virtual_block_size=2,
    )
    oracle_doubles = components.assemble_core()
    oracle_singles = build_ccsd_singles_numerator(
        *problem["oracle_arguments"]
    ).numerator
    audit = assemble_complete_thc_ccsd_audit(
        **problem["audit_arguments"], omega_ac_path=omega_ac_path
    )
    omega_ac = (
        audit.algorithm_6
        if audit.algorithm_6 is not None
        else audit.algorithm_4 + audit.algorithm_5
    )

    # Algorithms 1--3 have an existing independent RR/CD endpoint.  Compare
    # each one before considering the rest of the complete equation graph.
    rr_123 = rr_residual_algorithms_1_3(
        problem["doubles"],
        problem["oracle_arguments"][1],
        *problem["oracle_arguments"][-3:],
        auxiliary_block_size=2,
    )
    for rr_value, thc_value in (
        (rr_123.algorithm_1, audit.algorithm_1),
        (rr_123.algorithm_2, audit.algorithm_2),
        (rr_123.algorithm_3, audit.algorithm_3),
    ):
        np.testing.assert_allclose(
            rr_value,
            problem["tau"] @ thc_value @ problem["tau"].T,
            atol=2e-14,
            rtol=2e-13,
        )

    one_body = next(
        term.core
        for term in components.terms
        if term.term == "one-body-fock-dressing"
    )
    np.testing.assert_allclose(
        one_body,
        problem["tau"] @ audit.algorithm_9.omega_e @ problem["tau"].T,
        atol=3e-14,
        rtol=3e-13,
    )

    # The independent RR/CD total fixes the contribution required after the
    # audited A1--A3, A/C, and A9 endpoints are assembled.  It is exactly the
    # raw Algorithm 7 result plus its pair transpose.  This total-identity
    # statement does not claim a unique term mapping to printed Eq. 35.
    selected_without_7_rr = problem["tau"] @ (
        audit.algorithm_1
        + audit.algorithm_2
        + audit.algorithm_3
        + omega_ac
        + audit.algorithm_9.omega_e
    ) @ problem["tau"].T
    required_algorithm_7_rr = oracle_doubles - selected_without_7_rr
    observed_algorithm_7_rr = (
        problem["tau"]
        @ audit.algorithm_7.combined
        @ problem["tau"].T
    )
    transposed_algorithm_7_rr = (
        problem["tau"]
        @ audit.algorithm_7.combined.T
        @ problem["tau"].T
    )
    np.testing.assert_allclose(
        required_algorithm_7_rr,
        observed_algorithm_7_rr + transposed_algorithm_7_rr,
        atol=5e-14,
        rtol=5e-13,
    )
    np.testing.assert_allclose(
        audit.algorithm_7_pair_symmetrized,
        audit.algorithm_7.combined + audit.algorithm_7.combined.T,
        atol=0.0,
        rtol=0.0,
    )
    raw_once_doubles_rr = audit.doubles_rr - transposed_algorithm_7_rr
    np.testing.assert_allclose(
        raw_once_doubles_rr - oracle_doubles,
        -transposed_algorithm_7_rr,
        atol=5e-14,
        rtol=5e-13,
    )
    assert np.max(np.abs(raw_once_doubles_rr - oracle_doubles)) > 1e-10
    np.testing.assert_allclose(
        audit.doubles_rr,
        oracle_doubles,
        atol=5e-14,
        rtol=5e-13,
    )
    np.testing.assert_allclose(
        audit.singles,
        oracle_singles,
        atol=5e-14,
        rtol=5e-13,
    )

    metadata = audit.metadata()
    assert audit.accepted is False
    assert audit.production_enabled is False
    assert audit.complete_validated is False
    assert metadata["algorithm7_eq35_mapping"] == (
        "deprecated-name-mapped-to-version-of-record-eq36"
    )
    assert metadata["algorithm7_equation_version_contract_asserted"] is True
    assert metadata["algorithm7_instance_numerical_equivalence_audited"] is False
    assert metadata["algorithm7_instance_numerical_equivalence"] is None


def test_complete_assembly_pair_symmetrizes_a_nonsymmetric_raw_algorithm7(
    monkeypatch,
):
    problem = _full_pair_problem(1711, nocc=2, nvir=2, t1_scale=0.02)
    original = complete._omega_d.thc_omega_d_algorithm7
    captured = {}

    def asymmetric_algorithm7(*args, **kwargs):
        result = original(*args, **kwargs)
        antisymmetric = np.zeros_like(result.combined)
        antisymmetric[0, 1] = 0.125
        antisymmetric[1, 0] = -0.125
        result = replace(result, combined=result.combined + antisymmetric)
        captured["raw"] = result.combined
        return result

    monkeypatch.setattr(
        complete._omega_d,
        "thc_omega_d_algorithm7",
        asymmetric_algorithm7,
    )
    result = assemble_complete_thc_ccsd_audit(
        **problem["audit_arguments"], omega_ac_path="joint-6"
    )
    oracle = build_projected_ccsd_doubles_components(
        *problem["oracle_arguments"],
        auxiliary_block_size=2,
        virtual_block_size=2,
    ).assemble_core()

    assert np.max(np.abs(captured["raw"] - captured["raw"].T)) > 0.2
    np.testing.assert_allclose(
        result.algorithm_7_pair_symmetrized,
        captured["raw"] + captured["raw"].T,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.doubles_rr,
        oracle,
        atol=5e-14,
        rtol=5e-13,
    )
    algorithm7_entry = next(
        entry for entry in result.ledger.entries if entry.algorithm == 7
    )
    assert algorithm7_entry.equations == (36, 37, 38)
    assert algorithm7_entry.pair_symmetrization == (
        "raw-plus-pair-transpose"
    )
    assert algorithm7_entry.pair_symmetrization_application_count == 1


@pytest.mark.parametrize(
    "seed,nocc,nvir,rr_rank,thc_rank",
    [
        (1721, 3, 3, 3, 4),
        (1722, 2, 4, 3, 3),
    ],
)
@pytest.mark.parametrize(
    "omega_ac_path", ["joint-6", "separate-4-plus-5"]
)
def test_complete_inexact_cp_identity_against_reconstructed_rr_cd_oracle(
    seed, nocc, nvir, rr_rank, thc_rank, omega_ac_path
):
    problem = _inexact_orthogonalized_cp_problem(
        seed,
        nocc=nocc,
        nvir=nvir,
        rr_rank=rr_rank,
        thc_rank=thc_rank,
    )
    factors = problem["factors"]

    # This is a genuinely compressed and non-pair-basis amplitude factor.
    # Its fit to the original generic RR projector is intentionally inexact;
    # the complete-equation identity below is therefore evaluated against the
    # orthogonalized projector reconstructed from Eqs. 11--14.
    assert problem["rr_rank"] < problem["pair_dimension"]
    assert problem["thc_rank"] < problem["pair_dimension"]
    assert factors.exact_pair_endpoint is False
    assert factors.weighted_fit_residual > 1e-5
    assert factors.weighted_fit_gate_passed is False
    assert np.linalg.norm(problem["oracle_arguments"][1]) > 1e-3
    assert np.linalg.norm(
        problem["target_vectors"] - problem["reconstructed_vectors"]
    ) > 1e-3
    assert np.all(
        np.count_nonzero(np.abs(factors.y_occ) > 1e-10, axis=0) > 1
    )
    assert np.all(
        np.count_nonzero(np.abs(factors.y_vir) > 1e-10, axis=0) > 1
    )

    # Recompute the Eq. 11--14 symmetric orthogonalization independently from
    # raw_tau and the Khatri--Rao metric.  This guards against accidentally
    # treating the post-fit CP tensor as an arbitrary already-orthogonal U.
    y_gram = (factors.y_occ.T @ factors.y_occ) * (
        factors.y_vir.T @ factors.y_vir
    )
    raw_overlap = factors.raw_tau @ y_gram @ factors.raw_tau.T
    raw_overlap = (raw_overlap + raw_overlap.T) * 0.5
    overlap_values, overlap_vectors = np.linalg.eigh(raw_overlap)
    inverse_sqrt = (
        overlap_vectors * (1.0 / np.sqrt(overlap_values))[None, :]
    ) @ overlap_vectors.T
    expected_tau = inverse_sqrt @ factors.raw_tau
    np.testing.assert_allclose(
        factors.tau, expected_tau, atol=3e-14, rtol=3e-13
    )
    np.testing.assert_allclose(
        problem["reconstructed_vectors"].T
        @ problem["reconstructed_vectors"],
        np.eye(problem["rr_rank"]),
        atol=3e-13,
        rtol=3e-13,
    )
    np.testing.assert_allclose(
        problem["pair_y"]
        @ problem["amplitude_core"]
        @ problem["pair_y"].T,
        problem["doubles"].reconstruct_pair_matrix(),
        atol=3e-13,
        rtol=3e-13,
    )

    oracle_doubles = build_projected_ccsd_doubles_components(
        *problem["oracle_arguments"],
        auxiliary_block_size=2,
        virtual_block_size=2,
    ).assemble_core()
    oracle_singles = build_ccsd_singles_numerator(
        *problem["oracle_arguments"]
    ).numerator
    audit = assemble_complete_thc_ccsd_audit(
        **problem["audit_arguments"], omega_ac_path=omega_ac_path
    )

    np.testing.assert_allclose(
        audit.doubles_rr,
        oracle_doubles,
        atol=5e-14,
        rtol=5e-13,
    )
    np.testing.assert_allclose(
        audit.singles,
        oracle_singles,
        atol=5e-14,
        rtol=5e-13,
    )
    assert audit.accepted is False
    assert audit.complete_validated is False
    assert audit.production_enabled is False
