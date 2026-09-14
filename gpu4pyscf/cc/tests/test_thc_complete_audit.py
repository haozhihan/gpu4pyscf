"""Composition and fail-closed tests for the Algorithms 1--10 audit."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gpu4pyscf.cc import thc_complete_audit as complete
from gpu4pyscf.cc.device_runtime import TransferCounter
from gpu4pyscf.cc.thc_eri import ERITHCFactors
from gpu4pyscf.cc.thc_fhat import build_t1_transformed_fhat
from gpu4pyscf.cc.thc_omega_ghi import T1TransformedCholeskyBlocks


def _problem(seed=1601, *, transfer_counter=None, xp=np):
    rng = np.random.default_rng(seed)
    nocc, nvir, amplitude_rank, eri_rank, naux = 2, 3, 2, 3, 4
    nmo = nocc + nvir

    y_occ = rng.normal(size=(nocc, amplitude_rank))
    y_vir = rng.normal(size=(nvir, amplitude_rank))
    amplitude_core = rng.normal(size=(amplitude_rank, amplitude_rank))
    amplitude_core = (amplitude_core + amplitude_core.T) * 0.5
    tau = np.asarray([[1.2, 0.3], [-0.4, 0.8], [0.25, 1.1]])

    x_occ = rng.normal(size=(nocc, eri_rank))
    x_vir = rng.normal(size=(nvir, eri_rank))
    eri_core = rng.normal(size=(eri_rank, eri_rank))
    eri_core = (eri_core + eri_core.T) * 0.5

    h_mo = rng.normal(size=(nmo, nmo))
    h_mo = (h_mo + h_mo.T) * 0.5
    raw_l = rng.normal(size=(naux, nmo, nmo))
    raw_l = (raw_l + raw_l.swapaxes(1, 2)) * 0.5
    t1 = rng.normal(size=(nocc, nvir)) * 0.11

    orbital_token = {"case": "complete-audit", "seed": seed}
    integral_token = f"sha256:complete-audit-integrals-{seed}"
    hcore_token = {"case": "complete-audit", "seed": seed}
    provenance = {
        "orbital_identity": orbital_token,
        "integral_fingerprint": integral_token,
        "hcore_identity": hcore_token,
    }

    arrays = (
        y_occ,
        y_vir,
        amplitude_core,
        tau,
        x_occ,
        x_vir,
        eri_core,
        h_mo,
        raw_l,
        t1,
    )
    if xp is not np:
        arrays = tuple(xp.asarray(value) for value in arrays)
    (
        y_occ,
        y_vir,
        amplitude_core,
        tau,
        x_occ,
        x_vir,
        eri_core,
        h_mo,
        raw_l,
        t1,
    ) = arrays
    raw_eri_factors = ERITHCFactors(
        x_occ,
        x_vir,
        eri_core,
        transfer_counter=transfer_counter,
    )
    eri_factors = complete.IdentityAttestedERITHCFactors(
        raw_eri_factors,
        orbital_identity_token=orbital_token,
        integral_identity_token=integral_token,
    )
    fhat = build_t1_transformed_fhat(
        h_mo,
        raw_l,
        t1,
        auxiliary_block_size=2,
        transfer_counter=transfer_counter,
        provenance_context=provenance,
    )
    return {
        "y_occ": y_occ,
        "y_vir": y_vir,
        "amplitude_core": amplitude_core,
        "tau": tau,
        "eri_factors": eri_factors,
        "fhat": fhat,
        "orbital_identity_token": orbital_token,
        "integral_identity_token": integral_token,
        "hcore_identity_token": hcore_token,
        "transfer_counter": transfer_counter,
    }


def _assemble(problem, **overrides):
    arguments = dict(problem)
    arguments.update(overrides)
    return complete.assemble_complete_thc_ccsd_audit(**arguments)


def test_complete_schedule_sums_from_zero_and_backprojects_exactly_once(
    monkeypatch,
):
    problem = _problem()
    original_project = complete._residual.project_thc_residual_to_rr
    calls = []

    def counted_project(sigma_thc, tau):
        calls.append((sigma_thc, tau))
        return original_project(sigma_thc, tau)

    monkeypatch.setattr(
        complete._residual, "project_thc_residual_to_rr", counted_project
    )
    result = _assemble(problem)

    expected_thc = np.zeros_like(problem["amplitude_core"])
    for contribution in (
        result.algorithm_1,
        result.algorithm_2,
        result.algorithm_3,
        result.algorithm_6,
        result.algorithm_7.combined,
        result.algorithm_9.omega_e,
    ):
        expected_thc += contribution
    np.testing.assert_allclose(result.doubles_thc, expected_thc, atol=2e-12)
    np.testing.assert_allclose(
        result.doubles_rr,
        problem["tau"] @ expected_thc @ problem["tau"].T,
        atol=3e-12,
    )
    np.testing.assert_allclose(
        result.singles,
        result.algorithm_8.singles_gh + result.algorithm_10.singles_ij,
        atol=2e-12,
    )
    assert len(calls) == 1
    assert calls[0][0] is result.doubles_thc
    assert calls[0][1] is problem["tau"]
    assert result.ledger.rr_backprojection_count == 1
    assert result.ledger.singles_backprojection_count == 0
    assert not np.allclose(problem["tau"].T @ problem["tau"], np.eye(2))


def test_algorithms_4_plus_5_and_algorithm_6_are_strict_xor(monkeypatch):
    problem = _problem(seed=1602)
    calls = {4: 0, 5: 0, 6: 0}
    originals = {
        4: complete._eri.thc_omega_a_algorithm4,
        5: complete._eri.thc_omega_c_algorithm5,
        6: complete._eri.thc_omega_ac_algorithm6,
    }

    def wrap(number):
        def counted(*args, **kwargs):
            calls[number] += 1
            return originals[number](*args, **kwargs)

        return counted

    monkeypatch.setattr(complete._eri, "thc_omega_a_algorithm4", wrap(4))
    monkeypatch.setattr(complete._eri, "thc_omega_c_algorithm5", wrap(5))
    monkeypatch.setattr(complete._eri, "thc_omega_ac_algorithm6", wrap(6))

    joint = _assemble(problem, omega_ac_path="joint-6")
    assert calls == {4: 0, 5: 0, 6: 1}
    assert joint.algorithm_4 is None
    assert joint.algorithm_5 is None
    assert joint.algorithm_6 is not None
    joint_selection = {
        entry.algorithm: entry.selected
        for entry in joint.ledger.entries
        if entry.algorithm in (4, 5, 6)
    }
    assert joint_selection == {4: False, 5: False, 6: True}

    calls.update({4: 0, 5: 0, 6: 0})
    separate = _assemble(problem, omega_ac_path="separate-4-plus-5")
    assert calls == {4: 1, 5: 1, 6: 0}
    assert separate.algorithm_4 is not None
    assert separate.algorithm_5 is not None
    assert separate.algorithm_6 is None
    separate_selection = {
        entry.algorithm: entry.selected
        for entry in separate.ledger.entries
        if entry.algorithm in (4, 5, 6)
    }
    assert separate_selection == {4: True, 5: True, 6: False}
    np.testing.assert_allclose(joint.doubles_thc, separate.doubles_thc, atol=3e-10)

    for result in (joint, separate):
        for entry in result.ledger.entries:
            assert entry.coefficient == 1.0
            assert entry.individual_backprojection_count == 0


def test_algorithm7_value_is_retained_but_acceptance_stays_fail_closed():
    problem = _problem(seed=1603)
    result = _assemble(problem)
    metadata = result.metadata()

    assert np.linalg.norm(result.algorithm_7.combined) > 1e-10
    assert result.accepted is False
    assert result.production_enabled is False
    assert result.complete_validated is False
    assert result.complete_ccsd_residual is False
    assert metadata["algorithm7_computed_value_retained"] is True
    assert metadata["algorithm7_eq35_mapping"] == (
        "unresolved-by-paper-and-dense-audit"
    )
    assert metadata["algorithm7_eq35_equivalence"] is False
    assert metadata["algorithm7_sign_or_permutation_tuned"] is False
    assert metadata["accepted"] is False
    assert metadata["production_enabled"] is False
    assert metadata["complete_validated"] is False

    with pytest.raises(NotImplementedError, match="Eq. 35 equivalence is unresolved"):
        _assemble(problem, require_eq35_equivalence=True)


def test_fhat_blocks_flow_in_the_printed_directions_and_a9_reuses_a8(
    monkeypatch,
):
    problem = _problem(seed=1604)
    fhat = problem["fhat"]
    assert np.linalg.norm(fhat.l_oo - fhat.l_oo.swapaxes(1, 2)) > 1e-8
    captured = {}
    original8 = complete._omega_ghi.thc_omega_gh_algorithm8
    original9 = complete._omega_ghi.thc_omega_e_algorithm9
    original10 = complete._omega_ghi.thc_omega_ij_algorithm10

    def capture8(y_occ, y_vir, core, cholesky, **kwargs):
        captured["cholesky"] = cholesky
        result = original8(y_occ, y_vir, core, cholesky, **kwargs)
        captured["algorithm8"] = result
        return result

    def capture9(y_occ, y_vir, core, fhat_oo, fhat_vv, algorithm8, **kwargs):
        captured["fhat_oo"] = fhat_oo
        captured["fhat_vv"] = fhat_vv
        captured["algorithm9_input8"] = algorithm8
        return original9(
            y_occ,
            y_vir,
            core,
            fhat_oo,
            fhat_vv,
            algorithm8,
            **kwargs,
        )

    def capture10(y_occ, y_vir, core, fhat_ov, fhat_vo, **kwargs):
        captured["fhat_ov"] = fhat_ov
        captured["fhat_vo"] = fhat_vo
        return original10(
            y_occ, y_vir, core, fhat_ov, fhat_vo, **kwargs
        )

    monkeypatch.setattr(complete._omega_ghi, "thc_omega_gh_algorithm8", capture8)
    monkeypatch.setattr(complete._omega_ghi, "thc_omega_e_algorithm9", capture9)
    monkeypatch.setattr(complete._omega_ghi, "thc_omega_ij_algorithm10", capture10)
    result = _assemble(problem)

    cholesky = captured["cholesky"]
    assert cholesky.l_oo is fhat.l_oo
    assert cholesky.l_vv is fhat.l_vv
    assert cholesky.l_ov is fhat.l_ov
    assert captured["fhat_oo"] is fhat.fhat_oo
    assert captured["fhat_vv"] is fhat.fhat_vv
    assert captured["fhat_ov"] is fhat.fhat_ov
    assert captured["fhat_vo"] is fhat.fhat_vo
    assert captured["algorithm9_input8"] is captured["algorithm8"]
    assert result.algorithm_8 is captured["algorithm8"]
    metadata = result.metadata()
    assert metadata["fhat_identity_context_digest"].startswith("sha256:")
    assert metadata["eri_factors_identity_token_digest"].startswith("sha256:")
    assert metadata["assembly_identity_token_digest"].startswith("sha256:")
    assert metadata["identity_binding_scope"] == "caller-attested-tokens-only"
    assert metadata["numerical_content_provenance_validated"] is False
    assert metadata["factor_arrays_immutable"] is False

    wrong_direction = original8(
        problem["y_occ"],
        problem["y_vir"],
        problem["amplitude_core"],
        T1TransformedCholeskyBlocks(
            l_oo=fhat.l_oo.swapaxes(1, 2),
            l_vv=fhat.l_vv,
            l_ov=fhat.l_ov,
        ),
    )
    assert np.linalg.norm(
        result.algorithm_8.singles_gh - wrong_direction.singles_gh
    ) > 1e-8


@pytest.mark.parametrize(
    "override,match",
    [
        (
            {"orbital_identity_token": {"case": "wrong", "seed": 0}},
            "orbitals identity token does not match",
        ),
        (
            {"integral_identity_token": "sha256:wrong"},
            "integrals identity token does not match",
        ),
        (
            {"hcore_identity_token": {"case": "wrong", "seed": 0}},
            "hcore identity token does not match",
        ),
    ],
)
def test_mixed_orbital_or_integral_provenance_fails_closed(override, match):
    problem = _problem(seed=1605)
    with pytest.raises(ValueError, match=match):
        _assemble(problem, **override)


def test_unbound_or_mutated_partial_fhat_provenance_fails_closed():
    problem = _problem(seed=1606)
    unbound = replace(
        problem["fhat"], provenance_context={}, provenance_bound=False
    )
    with pytest.raises(ValueError, match="provenance must be complete"):
        _assemble(problem, fhat=unbound)

    partial_context = dict(problem["fhat"].provenance_context)
    partial_context.pop("hcore_identity")
    partial = replace(
        problem["fhat"],
        provenance_context=partial_context,
        provenance_bound=True,
    )
    with pytest.raises(ValueError, match="missing hcore"):
        _assemble(problem, fhat=partial)


def test_two_valid_but_mixed_fhat_and_eri_artifacts_fail_before_contractions(
    monkeypatch,
):
    first = _problem(seed=1613)
    second = _problem(seed=1614)

    def contraction_must_not_run(*args, **kwargs):
        raise AssertionError("provenance mismatch reached a contraction")

    monkeypatch.setattr(
        complete._residual, "thc_algorithm_1", contraction_must_not_run
    )
    with pytest.raises(
        ValueError, match="ERI factors orbital identity token does not match"
    ):
        _assemble(first, eri_factors=second["eri_factors"])


def test_valid_eri_artifact_with_only_integral_mismatch_fails_before_contraction(
    monkeypatch,
):
    first = _problem(seed=1615)
    second = _problem(seed=1616)
    mixed = complete.IdentityAttestedERITHCFactors(
        second["eri_factors"].factors,
        orbital_identity_token=first["orbital_identity_token"],
        integral_identity_token=second["integral_identity_token"],
    )

    def contraction_must_not_run(*args, **kwargs):
        raise AssertionError("integral provenance mismatch reached contraction")

    monkeypatch.setattr(
        complete._residual, "thc_algorithm_1", contraction_must_not_run
    )
    with pytest.raises(
        ValueError, match="ERI factors integral identity token does not match"
    ):
        _assemble(first, eri_factors=mixed)


def test_valid_fhat_context_with_hcore_mismatch_fails_before_contraction(
    monkeypatch,
):
    problem = _problem(seed=1617)
    foreign_context = dict(problem["fhat"].provenance_context)
    foreign_context["hcore_identity"] = {"case": "foreign-hcore"}
    foreign_fhat = replace(
        problem["fhat"], provenance_context=foreign_context
    )

    def contraction_must_not_run(*args, **kwargs):
        raise AssertionError("hcore provenance mismatch reached contraction")

    monkeypatch.setattr(
        complete._residual, "thc_algorithm_1", contraction_must_not_run
    )
    with pytest.raises(ValueError, match="hcore identity token does not match"):
        _assemble(problem, fhat=foreign_fhat)


def test_eri_provenance_wrapper_freezes_identity_snapshot():
    orbital = {"case": "before"}
    integral = ["sha256:before"]
    raw = ERITHCFactors(
        np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
    )
    bound = complete.IdentityAttestedERITHCFactors(
        raw,
        orbital_identity_token=orbital,
        integral_identity_token=integral,
    )
    expected_digest = bound.identity_token_digest
    orbital["case"] = "after"
    integral[0] = "sha256:after"
    assert bound.orbital_identity_token == {"case": "before"}
    assert bound.integral_identity_token == ["sha256:before"]
    assert bound.identity_token_digest == expected_digest


def test_algorithm10_requires_symmetric_amplitude_core():
    problem = _problem(seed=1607)
    nonsymmetric = problem["amplitude_core"].copy()
    nonsymmetric[0, 1] += 1e-8
    with pytest.raises(ValueError, match="amplitude_core must be symmetric"):
        _assemble(problem, amplitude_core=nonsymmetric)


@pytest.mark.parametrize("failure", ["nan", "tau-shape", "dtype", "fhat-shape"])
def test_shape_dtype_and_finite_contracts_fail_closed(failure):
    problem = _problem(seed=1608)
    if failure == "nan":
        bad = problem["y_occ"].copy()
        bad[0, 0] = np.nan
        override = {"y_occ": bad}
        error, match = ValueError, "non-finite"
    elif failure == "tau-shape":
        override = {"tau": np.ones((3, 3), dtype=np.float64)}
        error, match = ValueError, "tau must have shape"
    elif failure == "dtype":
        override = {"tau": problem["tau"].astype(np.float32)}
        error, match = TypeError, "one dtype"
    else:
        bad_fhat = replace(
            problem["fhat"],
            fhat_ov=np.zeros((1, 1), dtype=np.float64),
        )
        override = {"fhat": bad_fhat}
        error, match = ValueError, "fhat.fhat_ov must have shape"
    with pytest.raises(error, match=match):
        _assemble(problem, **override)


@pytest.mark.parametrize("failure", ["shape", "dtype", "nonfinite"])
def test_child_endpoint_output_contracts_are_revalidated(monkeypatch, failure):
    problem = _problem(seed=1618)
    if failure == "shape":
        replacement = np.zeros((2,), dtype=np.float64)
        error, match = ValueError, "algorithm_1 must have shape"
    elif failure == "dtype":
        replacement = np.zeros((2, 2), dtype=np.float32)
        error, match = TypeError, "one dtype"
    else:
        replacement = np.full((2, 2), np.nan, dtype=np.float64)
        error, match = FloatingPointError, "non-finite"
    monkeypatch.setattr(
        complete._residual,
        "thc_algorithm_1",
        lambda *args, **kwargs: replacement,
    )
    with pytest.raises(error, match=match):
        _assemble(problem)


@pytest.mark.parametrize("failure", ["shape", "dtype", "nonfinite"])
def test_final_rr_projection_contract_is_revalidated(monkeypatch, failure):
    problem = _problem(seed=1619)
    if failure == "shape":
        replacement = np.zeros((2, 2), dtype=np.float64)
        error, match = ValueError, "doubles_rr must have shape"
    elif failure == "dtype":
        replacement = np.zeros((3, 3), dtype=np.float32)
        error, match = TypeError, "one dtype"
    else:
        replacement = np.full((3, 3), np.nan, dtype=np.float64)
        error, match = FloatingPointError, "non-finite"
    monkeypatch.setattr(
        complete._residual,
        "project_thc_residual_to_rr",
        lambda *args, **kwargs: replacement,
    )
    with pytest.raises(error, match=match):
        _assemble(problem)


def test_mutated_eri_nonfinite_value_is_revalidated():
    problem = _problem(seed=1609)
    problem["eri_factors"].factors.core[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        _assemble(problem)


def test_nested_transfer_counters_cannot_be_mixed():
    first = TransferCounter()
    second = TransferCounter()
    problem = _problem(seed=1610, transfer_counter=first)
    with pytest.raises(ValueError, match="same transfer_counter"):
        _assemble(problem, transfer_counter=second)


def test_production_engines_do_not_import_or_call_complete_audit():
    cc_directory = Path(complete.__file__).resolve().parent
    for filename in ("thc_engine.py", "thc_rrccsd.py"):
        source = (cc_directory / filename).read_text(encoding="utf-8")
        assert "thc_complete_audit" not in source
        assert "assemble_complete_thc_ccsd_audit" not in source


def _cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    if count < 1:
        pytest.skip("no CUDA device")
    return cp, count


def test_cupy_path_requires_one_explicit_counter_and_stays_on_device():
    cp, _ = _cupy_or_skip()
    counter = TransferCounter()
    problem = _problem(seed=1611, transfer_counter=counter, xp=cp)

    with pytest.raises(ValueError, match="explicit transfer_counter"):
        _assemble(problem, transfer_counter=None)
    before = counter.total_transfers
    result = _assemble(problem, transfer_counter=counter)
    assert isinstance(result.singles, cp.ndarray)
    assert isinstance(result.doubles_thc, cp.ndarray)
    assert isinstance(result.doubles_rr, cp.ndarray)
    assert counter.total_transfers > before

    mixed_tau = cp.asnumpy(problem["tau"])
    with pytest.raises(TypeError, match="one backend"):
        _assemble(problem, tau=mixed_tau, transfer_counter=counter)


def test_cupy_device_mismatch_fails_before_contractions():
    cp, count = _cupy_or_skip()
    if count < 2:
        pytest.skip("requires two CUDA devices")
    counter = TransferCounter()
    with cp.cuda.Device(0):
        problem = _problem(seed=1612, transfer_counter=counter, xp=cp)
    with cp.cuda.Device(1):
        wrong_tau = cp.asarray(np.ones((3, 2), dtype=np.float64))
    with cp.cuda.Device(0):
        with pytest.raises(TypeError, match="one device"):
            _assemble(
                problem, tau=wrong_tau, transfer_counter=counter
            )
