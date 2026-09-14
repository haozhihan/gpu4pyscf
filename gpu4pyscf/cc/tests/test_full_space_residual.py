"""Representation-neutral full pair-space residual diagnostics."""

from pathlib import Path

import numpy as np

from gpu4pyscf.cc.full_space_residual import (
    FullSpaceResidualDiagnostic,
    diagnose_reconstructed_full_space_residual,
)
from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.lowrank import RRDoubles, RRProjector
from gpu4pyscf.cc.rr_engine import RRCCSDIterationEngine


def _problem(seed=408, nocc=2, nvir=3, naux=3, rank=4):
    rng = np.random.default_rng(seed)
    nmo = nocc + nvir
    raw = rng.normal(size=(naux, nmo, nmo))
    factors = (raw + raw.transpose(0, 2, 1)) * 0.5
    vectors, _ = np.linalg.qr(rng.normal(size=(nocc * nvir, rank)))
    projector = RRProjector(
        vectors,
        -np.linspace(1.0, 0.2, rank),
        cutoff=0.0,
        full_dimension=nocc * nvir,
    )
    core = rng.normal(size=(rank, rank))
    core = (core + core.T) * 0.5
    doubles = RRDoubles(projector, core, nocc, nvir)
    t1 = rng.normal(size=(nocc, nvir))
    orbital_energies = np.concatenate((
        -np.linspace(1.4, 0.7, nocc),
        np.linspace(0.2, 1.0, nvir),
    ))
    fock = rng.normal(size=(nmo, nmo))
    provider = MOThreeIndexIntegralProvider(
        factors,
        nocc,
        factorization="cd",
        threshold=1e-8,
        source="full-space-residual-unit-test",
    )
    return provider, projector, fock, orbital_energies, t1, doubles


def test_diagnostic_uses_identity_pair_basis_explicitly():
    provider, _projector, fock, energies, t1, doubles = _problem()
    diagnostic = diagnose_reconstructed_full_space_residual(
        provider,
        t1,
        doubles,
        fock,
        energies,
        level_shift=0.02,
        auxiliary_block_size=2,
        virtual_block_size=2,
    )
    dimension = provider.nocc * provider.nvir
    identity_projector = RRProjector(
        np.eye(dimension),
        np.ones(dimension),
        cutoff=0.0,
        full_dimension=dimension,
    )
    full_doubles = RRDoubles(
        identity_projector,
        doubles.reconstruct_pair_matrix(),
        provider.nocc,
        provider.nvir,
    )
    reference_engine = RRCCSDIterationEngine(
        provider,
        identity_projector,
        fock,
        energies,
        level_shift=0.02,
        auxiliary_block_size=2,
        virtual_block_size=2,
    )
    reference = reference_engine.jacobi(t1, full_doubles)
    np.testing.assert_allclose(
        diagnostic.residual_norm, float(reference.residual_norm), atol=2e-10
    )
    assert diagnostic.pair_dimension == dimension
    assert diagnostic.materialized_dense_t2_equivalent is True
    assert diagnostic.included_in_normal_iteration_timing is False
    assert diagnostic.metadata()["norm"] == (
        "full-pair-equation-residual-frobenius"
    )


def test_thc_engine_keeps_compatibility_reexports():
    from gpu4pyscf.cc.thc_engine import (
        FullSpaceResidualDiagnostic as LegacyDiagnostic,
        diagnose_reconstructed_full_space_residual as legacy_diagnose,
    )

    assert LegacyDiagnostic is FullSpaceResidualDiagnostic
    assert legacy_diagnose is diagnose_reconstructed_full_space_residual


def test_rr_driver_has_no_thc_engine_dependency():
    source = (Path(__file__).parents[1] / "rrccsd.py").read_text(
        encoding="utf-8"
    )

    assert "gpu4pyscf.cc.thc_engine" not in source
    assert "gpu4pyscf.cc.full_space_residual" in source
