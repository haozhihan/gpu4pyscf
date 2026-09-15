"""Dense joint audit tests for THC Omega-C and Omega-D."""

import json

import numpy as np
import pytest

from gpu4pyscf.cc.thc_eri import ERITHCFactors
from gpu4pyscf.cc.thc_omega_cd_audit import thc_omega_cd_joint_audit


def _problem(
    seed=801,
    *,
    nocc=2,
    nvir=3,
    amplitude_rank=3,
    eri_rank=4,
    pair_symmetric=False,
    dtype=np.float64,
):
    rng = np.random.default_rng(seed)
    y_occ = rng.normal(size=(nocc, amplitude_rank)).astype(dtype)
    y_vir = rng.normal(size=(nvir, amplitude_rank)).astype(dtype)
    amplitude_core = rng.normal(
        size=(amplitude_rank, amplitude_rank)
    ).astype(dtype)
    x_occ = rng.normal(size=(nocc, eri_rank)).astype(dtype)
    x_vir = rng.normal(size=(nvir, eri_rank)).astype(dtype)
    eri_core = rng.normal(size=(eri_rank, eri_rank)).astype(dtype)
    if pair_symmetric:
        amplitude_core = (amplitude_core + amplitude_core.T) / dtype(2)
        eri_core = (eri_core + eri_core.T) / dtype(2)
    return y_occ, y_vir, amplitude_core, ERITHCFactors(
        x_occ, x_vir, eri_core
    )


def _dense_references(y_occ, y_vir, amplitude_core, eri):
    projector = np.einsum("iX,aX->Xia", y_occ, y_vir, optimize=True)
    t2 = np.einsum(
        "iX,aX,XY,jY,bY->ijab",
        y_occ,
        y_vir,
        amplitude_core,
        y_occ,
        y_vir,
        optimize=True,
    )
    ovov = np.einsum(
        "iI,aI,IJ,jJ,bJ->iajb",
        eri.x_occ,
        eri.x_vir,
        eri.core,
        eri.x_occ,
        eri.x_vir,
        optimize=True,
    )

    eq33_residual = np.einsum(
        "ikcb,ljad,lckd->ijab", t2, t2, ovov, optimize=True
    )
    eq33 = np.einsum(
        "Xia,Yjb,ijab->XY",
        projector,
        projector,
        eq33_residual,
        optimize=True,
    )

    two_t_minus_virtual_swap = 2 * t2 - t2.swapaxes(2, 3)
    two_eri_minus_exchange = 2 * ovov - ovov.transpose(0, 3, 2, 1)
    eq36_residual = np.einsum(
        "ikac,kcld,jlbd->ijab",
        two_t_minus_virtual_swap,
        two_eri_minus_exchange,
        two_t_minus_virtual_swap,
        optimize=True,
    )
    eq36_residual += np.einsum(
        "ikca,kdlc,jldb->ijab", t2, ovov, t2, optimize=True
    )
    eq36 = 0.25 * np.einsum(
        "Xia,Yjb,ijab->XY",
        projector,
        projector,
        eq36_residual,
        optimize=True,
    )

    legacy_eq35_residual = np.einsum(
        "ikac,kcld,jlbd->ijab",
        two_t_minus_virtual_swap,
        two_eri_minus_exchange,
        t2 - t2.swapaxes(2, 3),
        optimize=True,
    )
    legacy_eq35_residual += np.einsum(
        "ikca,kdlc,jldb->ijab", t2, ovov, t2, optimize=True
    )
    legacy_eq35 = 0.25 * np.einsum(
        "Xia,Yjb,ijab->XY",
        projector,
        projector,
        legacy_eq35_residual,
        optimize=True,
    )

    r_intermediate = np.einsum(
        "Xjb,ijab->Xia",
        projector,
        two_t_minus_virtual_swap,
        optimize=True,
    )
    s_exchange = -np.einsum(
        "Xjb,ijab->Xia",
        projector,
        t2.swapaxes(2, 3),
        optimize=True,
    )
    eq38 = 0.25 * np.einsum(
        "Xia,Yjb,iajb->XY",
        r_intermediate,
        r_intermediate,
        two_eri_minus_exchange,
        optimize=True,
    )
    line19 = 0.25 * np.einsum(
        "Xia,Yjb,iajb->XY",
        s_exchange,
        s_exchange,
        ovov.transpose(0, 3, 2, 1),
        optimize=True,
    )
    return {
        "projector": projector,
        "t2": t2,
        "ovov": ovov,
        "eq33": eq33,
        "eq36": eq36,
        "legacy_eq35": legacy_eq35,
        "eq38": eq38,
        "line19": line19,
    }


def _assert_result_matches_dense(result, dense):
    np.testing.assert_allclose(
        result.omega_c_eq33, dense["eq33"], atol=2e-10, rtol=2e-12
    )
    np.testing.assert_allclose(
        result.omega_d_eq36_literal,
        dense["eq36"],
        atol=2e-10,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        result.legacy_preprint_eq35_literal,
        dense["legacy_eq35"],
        atol=2e-10,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        result.omega_d_eq38, dense["eq38"], atol=2e-10, rtol=2e-12
    )
    np.testing.assert_allclose(
        result.algorithm7_line19_correction,
        dense["line19"],
        atol=2e-10,
        rtol=2e-12,
    )
    expected_published_algorithm = dense["eq38"] + dense["line19"]
    expected_published_literal = dense["eq36"]
    expected_algorithm = dense["eq33"] + expected_published_algorithm
    expected_literal = dense["eq33"] + expected_published_literal
    np.testing.assert_allclose(
        result.published_algorithm,
        expected_published_algorithm,
        atol=4e-10,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        result.published_literal,
        expected_published_literal,
        atol=4e-10,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        result.published_difference,
        expected_published_algorithm - expected_published_literal,
        atol=4e-10,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        result.joint_algorithm,
        expected_algorithm,
        atol=4e-10,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        result.joint_literal, expected_literal, atol=4e-10, rtol=3e-12
    )
    np.testing.assert_allclose(
        result.joint_difference,
        expected_algorithm - expected_literal,
        atol=4e-10,
        rtol=3e-12,
    )
    expected_legacy_literal = dense["eq33"] + dense["legacy_eq35"]
    np.testing.assert_allclose(
        result.legacy_joint_literal,
        expected_legacy_literal,
        atol=4e-10,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        result.legacy_joint_difference,
        expected_algorithm - expected_legacy_literal,
        atol=4e-10,
        rtol=3e-12,
    )


def _brute_force_references(y_occ, y_vir, amplitude_core, eri):
    """Independent scalar-loop oracle for a deliberately tiny problem."""

    nocc, amplitude_rank = y_occ.shape
    nvir = y_vir.shape[0]
    eri_rank = eri.x_occ.shape[1]
    projector = np.zeros((amplitude_rank, nocc, nvir))
    t2 = np.zeros((nocc, nocc, nvir, nvir))
    ovov = np.zeros((nocc, nvir, nocc, nvir))
    for x in range(amplitude_rank):
        for i in range(nocc):
            for a in range(nvir):
                projector[x, i, a] = y_occ[i, x] * y_vir[a, x]
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    for x in range(amplitude_rank):
                        for y in range(amplitude_rank):
                            t2[i, j, a, b] += (
                                y_occ[i, x]
                                * y_vir[a, x]
                                * amplitude_core[x, y]
                                * y_occ[j, y]
                                * y_vir[b, y]
                            )
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    for eri_i in range(eri_rank):
                        for eri_j in range(eri_rank):
                            ovov[i, a, j, b] += (
                                eri.x_occ[i, eri_i]
                                * eri.x_vir[a, eri_i]
                                * eri.core[eri_i, eri_j]
                                * eri.x_occ[j, eri_j]
                                * eri.x_vir[b, eri_j]
                            )

    eq33 = np.zeros((amplitude_rank, amplitude_rank))
    eq36 = np.zeros_like(eq33)
    legacy_eq35 = np.zeros_like(eq33)
    r_intermediate = np.zeros((amplitude_rank, nocc, nvir))
    s_exchange = np.zeros_like(r_intermediate)
    for x in range(amplitude_rank):
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        r_intermediate[x, i, a] += projector[x, j, b] * (
                            2 * t2[i, j, a, b] - t2[i, j, b, a]
                        )
                        s_exchange[x, i, a] -= (
                            projector[x, j, b] * t2[i, j, b, a]
                        )
    for x in range(amplitude_rank):
        for y in range(amplitude_rank):
            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvir):
                        for b in range(nvir):
                            projection = (
                                projector[x, i, a] * projector[y, j, b]
                            )
                            for k in range(nocc):
                                for l in range(nocc):
                                    for c in range(nvir):
                                        for d in range(nvir):
                                            eq33[x, y] += projection * (
                                                t2[i, k, c, b]
                                                * t2[l, j, a, d]
                                                * ovov[l, c, k, d]
                                            )
                                            first = (
                                                (
                                                    2 * t2[i, k, a, c]
                                                    - t2[i, k, c, a]
                                                )
                                                * (
                                                    2 * ovov[k, c, l, d]
                                                    - ovov[k, d, l, c]
                                                )
                                                * (
                                                    2 * t2[j, l, b, d]
                                                    - t2[j, l, d, b]
                                                )
                                                + t2[i, k, c, a]
                                                * ovov[k, d, l, c]
                                                * t2[j, l, d, b]
                                            )
                                            eq36[x, y] += 0.25 * projection * first
                                            legacy_first = (
                                                (
                                                    2 * t2[i, k, a, c]
                                                    - t2[i, k, c, a]
                                                )
                                                * (
                                                    2 * ovov[k, c, l, d]
                                                    - ovov[k, d, l, c]
                                                )
                                                * (
                                                    t2[j, l, b, d]
                                                    - t2[j, l, d, b]
                                                )
                                                + t2[i, k, c, a]
                                                * ovov[k, d, l, c]
                                                * t2[j, l, d, b]
                                            )
                                            legacy_eq35[x, y] += (
                                                0.25 * projection * legacy_first
                                            )

    eq38 = np.zeros_like(eq33)
    line19 = np.zeros_like(eq33)
    for x in range(amplitude_rank):
        for y in range(amplitude_rank):
            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvir):
                        for b in range(nvir):
                            eq38[x, y] += 0.25 * (
                                r_intermediate[x, i, a]
                                * r_intermediate[y, j, b]
                                * (
                                    2 * ovov[i, a, j, b]
                                    - ovov[i, b, j, a]
                                )
                            )
                            line19[x, y] += 0.25 * (
                                s_exchange[x, i, a]
                                * s_exchange[y, j, b]
                                * ovov[i, b, j, a]
                            )
    return {
        "t2": t2,
        "ovov": ovov,
        "eq33": eq33,
        "eq36": eq36,
        "legacy_eq35": legacy_eq35,
        "eq38": eq38,
        "line19": line19,
    }


def test_all_equations_match_independent_scalar_loop_oracle():
    args = _problem(
        815,
        nocc=2,
        nvir=2,
        amplitude_rank=2,
        eri_rank=2,
        pair_symmetric=True,
    )
    brute = _brute_force_references(*args)
    result = thc_omega_cd_joint_audit(*args)

    _assert_result_matches_dense(result, brute)
    assert result.full_pair_gauge is True
    assert result.published_equivalent is True


def test_nonsymmetric_factors_match_each_literal_dense_equation():
    args = _problem(802)
    dense = _dense_references(*args)
    assert not np.allclose(args[2], args[2].T)
    assert not np.allclose(args[3].core, args[3].core.T)

    result = thc_omega_cd_joint_audit(
        *args, algorithm5_outer_block_size=2, algorithm7_x_block_size=2
    )

    _assert_result_matches_dense(result, dense)
    assert result.endpoint_consistent is True
    assert result.full_pair_gauge is False
    assert result.equivalence_attempted is False
    assert result.published_equivalent is None
    assert result.legacy_preprint_equivalent is None
    assert np.max(np.abs(result.published_difference)) > 1e-8
    assert result.symmetry_diagnostics.t2_pair_symmetric is False
    assert result.symmetry_diagnostics.eri_pair_symmetric is False
    assert result.metadata()["joint_equivalence_status"] == (
        "not-attempted-outside-full-pair-gauge"
    )


def test_one_dimensional_published_identity_and_legacy_counterexample():
    one = np.ones((1, 1))
    result = thc_omega_cd_joint_audit(
        one, one.copy(), one.copy(), ERITHCFactors(one, one.copy(), one.copy())
    )

    assert result.full_pair_gauge is True
    assert result.equivalence_attempted is True
    assert result.published_equivalent is True
    assert result.legacy_preprint_equivalence_attempted is True
    assert result.legacy_preprint_equivalent is False
    np.testing.assert_allclose(result.omega_c_eq33, [[1.0]])
    np.testing.assert_allclose(result.omega_d_eq36_literal, [[0.5]])
    np.testing.assert_allclose(result.legacy_preprint_eq35_literal, [[0.25]])
    np.testing.assert_allclose(result.omega_d_eq38, [[0.25]])
    np.testing.assert_allclose(result.algorithm7_line19_correction, [[0.25]])
    np.testing.assert_allclose(result.published_difference, [[0.0]])
    np.testing.assert_allclose(result.joint_difference, [[0.0]])
    np.testing.assert_allclose(result.legacy_joint_difference, [[0.25]])
    assert result.published_max_abs_difference <= (
        result.published_equivalence_threshold
    )
    assert result.legacy_preprint_max_abs_difference > (
        result.legacy_preprint_equivalence_threshold
    )
    assert result.metadata()["joint_equivalence_status"] == (
        "equal-within-explicit-audit-tolerance"
    )
    assert result.metadata()["legacy_preprint_equivalence_status"] == (
        "unequal-known-version-difference"
    )


def test_nontrivial_full_pair_gauge_proves_published_identity_without_tuning():
    args = _problem(804, pair_symmetric=True)
    dense = _dense_references(*args)
    np.testing.assert_allclose(
        dense["t2"], dense["t2"].transpose(1, 0, 3, 2), atol=2e-13
    )
    np.testing.assert_allclose(
        dense["ovov"], dense["ovov"].transpose(2, 3, 0, 1), atol=2e-13
    )

    result = thc_omega_cd_joint_audit(*args)

    _assert_result_matches_dense(result, dense)
    assert result.symmetry_diagnostics.amplitude_core_symmetric is True
    assert result.symmetry_diagnostics.eri_core_symmetric is True
    assert result.full_pair_gauge is True
    assert result.equivalence_attempted is True
    assert result.published_equivalent is True
    assert result.published_max_abs_difference <= (
        result.published_equivalence_threshold
    )
    assert result.legacy_preprint_equivalent is False
    assert result.legacy_preprint_max_abs_difference > (
        result.legacy_preprint_equivalence_threshold
    )


def test_reconstructed_pair_gauge_does_not_require_symmetric_factor_cores():
    rng = np.random.default_rng(805)
    y_occ = np.column_stack((rng.normal(size=2), np.zeros(2)))
    y_vir = np.column_stack((rng.normal(size=3), np.zeros(3)))
    amplitude_core = np.array([[0.7, 4.0], [-2.0, 1.3]])
    x_occ = np.column_stack((rng.normal(size=2), np.zeros(2)))
    x_vir = np.column_stack((rng.normal(size=3), np.zeros(3)))
    eri_core = np.array([[-0.4, 3.0], [8.0, 2.1]])
    eri = ERITHCFactors(x_occ, x_vir, eri_core)

    result = thc_omega_cd_joint_audit(
        y_occ, y_vir, amplitude_core, eri
    )

    assert result.symmetry_diagnostics.amplitude_core_symmetric is False
    assert result.symmetry_diagnostics.eri_core_symmetric is False
    assert result.symmetry_diagnostics.t2_pair_symmetric is True
    assert result.symmetry_diagnostics.eri_pair_symmetric is True
    assert result.full_pair_gauge is True
    assert result.equivalence_attempted is True
    assert result.published_equivalent is True


def test_zero_problem_is_a_trivial_equal_endpoint():
    y_occ, y_vir, _, eri = _problem(806, pair_symmetric=True)
    zero_core = np.zeros((y_occ.shape[1], y_occ.shape[1]))

    result = thc_omega_cd_joint_audit(y_occ, y_vir, zero_core, eri)

    for name in (
        "omega_c_eq33",
        "omega_d_eq36_literal",
        "legacy_preprint_eq35_literal",
        "omega_d_eq38",
        "algorithm7_line19_correction",
        "published_algorithm",
        "published_literal",
        "published_difference",
        "joint_algorithm",
        "joint_literal",
        "joint_difference",
        "legacy_joint_literal",
        "legacy_joint_difference",
    ):
        np.testing.assert_array_equal(getattr(result, name), 0)
    assert result.full_pair_gauge is True
    assert result.equivalence_attempted is True
    assert result.published_equivalent is True
    assert result.published_max_abs_difference == 0.0
    assert result.legacy_preprint_equivalent is True


def test_rank_one_factors_match_dense_and_remain_a_diagnostic():
    args = _problem(
        807,
        nocc=2,
        nvir=2,
        amplitude_rank=1,
        eri_rank=1,
        pair_symmetric=True,
    )
    dense = _dense_references(*args)
    result = thc_omega_cd_joint_audit(*args)

    _assert_result_matches_dense(result, dense)
    assert result.full_pair_gauge is True
    assert result.equivalence_attempted is True
    assert result.published_equivalent is True
    assert result.legacy_preprint_equivalent is False
    assert result.omega_c_eq33.shape == (1, 1)


def test_float32_defaults_are_dtype_scaled_and_recorded():
    args = _problem(816, pair_symmetric=True, dtype=np.float32)
    result = thc_omega_cd_joint_audit(*args)

    assert result.omega_c_eq33.dtype == np.dtype(np.float32)
    assert result.endpoint_consistent is True
    assert result.equivalence_rtol == pytest.approx(
        2048 * np.finfo(np.float32).eps
    )
    assert result.symmetry_diagnostics.symmetry_rtol == pytest.approx(
        256 * np.finfo(np.float32).eps
    )
    assert result.metadata()["dtype"] == "float32"


def test_algorithm_block_sizes_do_not_change_joint_audit():
    args = _problem(808, pair_symmetric=True)
    expected = thc_omega_cd_joint_audit(
        *args, algorithm5_outer_block_size=1, algorithm7_x_block_size=1
    )
    for algorithm5_block, algorithm7_block in ((2, 2), (9, 3), (3, 9)):
        observed = thc_omega_cd_joint_audit(
            *args,
            algorithm5_outer_block_size=algorithm5_block,
            algorithm7_x_block_size=algorithm7_block,
        )
        for name in (
            "omega_c_eq33",
            "omega_d_eq36_literal",
            "legacy_preprint_eq35_literal",
            "omega_d_eq38",
            "algorithm7_line19_correction",
            "published_algorithm",
            "published_literal",
            "published_difference",
            "joint_algorithm",
            "joint_literal",
            "joint_difference",
            "legacy_joint_literal",
            "legacy_joint_difference",
        ):
            np.testing.assert_allclose(
                getattr(observed, name),
                getattr(expected, name),
                atol=5e-12,
                rtol=5e-13,
            )


def test_endpoint_mismatch_disables_joint_equivalence(monkeypatch):
    import gpu4pyscf.cc.thc_omega_cd_audit as audit_module

    args = _problem(809, pair_symmetric=True)
    real_algorithm5 = audit_module.thc_omega_c_algorithm5

    def perturbed_algorithm5(*call_args, **call_kwargs):
        observed = real_algorithm5(*call_args, **call_kwargs)
        return observed + np.ones_like(observed)

    monkeypatch.setattr(
        audit_module, "thc_omega_c_algorithm5", perturbed_algorithm5
    )
    result = audit_module.thc_omega_cd_joint_audit(*args)

    assert result.full_pair_gauge is True
    assert result.endpoint_consistent is False
    assert result.equivalence_attempted is False
    assert result.published_equivalent is None
    assert result.endpoint_max_abs_difference == pytest.approx(1.0)
    assert result.metadata()["joint_equivalence_status"] == (
        "not-attempted-endpoint-mismatch"
    )


def test_metadata_records_published_gate_and_keeps_later_gates_closed():
    result = thc_omega_cd_joint_audit(*_problem(810, pair_symmetric=True))
    metadata = result.metadata()

    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["paper_source_version"] == "version-of-record"
    assert metadata["paper_equations"] == [33, 36, 37, 38]
    assert metadata["paper_algorithms"] == [5, 7]
    assert metadata["published_algorithm_definition"] == (
        "eq38+algorithm7-line19"
    )
    assert metadata["published_literal_definition"] == "literal-eq36"
    assert metadata["joint_algorithm_definition"] == (
        "eq33+eq38+algorithm7-line19"
    )
    assert metadata["joint_literal_definition"] == "eq33+literal-eq36"
    assert metadata["joint_difference_definition"] == (
        "joint_algorithm-joint_literal"
    )
    assert metadata["line19_correction_sign"] == "as-printed-positive-addend"
    assert metadata["line19_eq36_term_mapping"] == (
        "verified-in-physical-full-pair-gauge"
    )
    assert metadata["published_eq36_literal_is_algebra_oracle"] is True
    assert metadata["published_eq36_algebra_gate_passed"] is True
    assert metadata["legacy_preprint_source"] == "arXiv:2111.11473v1"
    assert metadata["legacy_preprint_equation"] == 35
    assert metadata["legacy_preprint_eq35_is_correctness_oracle"] is False
    assert metadata["legacy_preprint_equivalent"] is False
    assert metadata["materializes_dense_t2"] is True
    assert metadata["materializes_four_index_eri"] is True
    assert metadata["small_problem_only"] is True
    assert metadata["implicit_gpu_host_transfer"] is False
    assert metadata["gpu_inputs_accepted"] is False
    assert metadata["audit_only"] is True
    assert metadata["production_enabled"] is False
    assert metadata["performance_eligible"] is False
    assert metadata["complete_ccsd_residual"] is False
    assert metadata["inexact_ccsd_validated"] is False
    assert metadata["water2_validated"] is False
    assert metadata["water4_validated"] is False
    assert metadata["performance_validated"] is False


class _ImplicitHostTransferTrap:
    def __array__(self, *_args, **_kwargs):
        raise AssertionError("implicit host conversion was attempted")


def test_convertible_non_numpy_input_fails_before_implicit_host_transfer():
    _, y_vir, amplitude_core, eri = _problem(811)
    with pytest.raises(TypeError, match="NumPy ndarray"):
        thc_omega_cd_joint_audit(
            _ImplicitHostTransferTrap(), y_vir, amplitude_core, eri
        )


def test_dense_memory_limit_is_enforced_before_algorithm_endpoints(monkeypatch):
    import gpu4pyscf.cc.thc_omega_cd_audit as audit_module

    args = _problem(812)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("algorithm endpoint ran before the memory gate")

    monkeypatch.setattr(audit_module, "thc_omega_c_algorithm5", forbidden)
    monkeypatch.setattr(audit_module, "thc_omega_d_algorithm7", forbidden)
    with pytest.raises(MemoryError, match="estimated working set"):
        audit_module.thc_omega_cd_joint_audit(
            *args, max_dense_working_bytes=1
        )


@pytest.mark.parametrize(
    "keyword,value,exception,match",
    [
        ("algorithm5_outer_block_size", True, TypeError, "integer"),
        ("algorithm5_outer_block_size", 0, ValueError, "positive"),
        ("algorithm7_x_block_size", 1.5, TypeError, "integer"),
        ("algorithm7_x_block_size", -1, ValueError, "positive"),
        ("symmetry_atol", True, TypeError, "real scalar"),
        ("symmetry_atol", -1.0, ValueError, "nonnegative"),
        ("symmetry_rtol", np.inf, ValueError, "finite"),
        ("equivalence_atol", "0", TypeError, "real scalar"),
        ("equivalence_rtol", -1.0, ValueError, "nonnegative"),
        ("max_dense_working_bytes", False, TypeError, "integer"),
    ],
)
def test_scalar_options_fail_closed(keyword, value, exception, match):
    with pytest.raises(exception, match=match):
        thc_omega_cd_joint_audit(*_problem(813), **{keyword: value})


def test_shape_dtype_and_nonfinite_inputs_fail_closed():
    y_occ, y_vir, amplitude_core, eri = _problem(814)
    with pytest.raises(ValueError, match="y_occ"):
        thc_omega_cd_joint_audit(y_occ.ravel(), y_vir, amplitude_core, eri)
    with pytest.raises(TypeError, match="same dtype"):
        thc_omega_cd_joint_audit(
            y_occ.astype(np.float32), y_vir, amplitude_core, eri
        )
    with pytest.raises(NotImplementedError, match="real RHF"):
        complex_eri = ERITHCFactors(
            eri.x_occ.astype(np.complex128),
            eri.x_vir.astype(np.complex128),
            eri.core.astype(np.complex128),
        )
        thc_omega_cd_joint_audit(
            y_occ.astype(np.complex128),
            y_vir.astype(np.complex128),
            amplitude_core.astype(np.complex128),
            complex_eri,
        )
    bad = amplitude_core.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        thc_omega_cd_joint_audit(y_occ, y_vir, bad, eri)


def test_cupy_input_is_rejected_without_new_transfer_counter_entries():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    transfers = TransferCounter()
    with cp.cuda.Device(0):
        y_occ = cp.ones((1, 1), dtype=cp.float64)
        y_vir = cp.ones((1, 1), dtype=cp.float64)
        amplitude_core = cp.ones((1, 1), dtype=cp.float64)
        eri = ERITHCFactors(
            cp.ones((1, 1), dtype=cp.float64),
            cp.ones((1, 1), dtype=cp.float64),
            cp.ones((1, 1), dtype=cp.float64),
            transfer_counter=transfers,
        )
    before = transfers.to_dict()

    with pytest.raises(TypeError, match="NumPy ndarray"):
        thc_omega_cd_joint_audit(y_occ, y_vir, amplitude_core, eri)

    assert transfers.to_dict() == before


def test_cupy_eri_with_numpy_amplitudes_is_rejected_without_new_transfers():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    y_occ = np.ones((1, 1))
    y_vir = np.ones((1, 1))
    amplitude_core = np.ones((1, 1))
    transfers = TransferCounter()
    with cp.cuda.Device(0):
        eri = ERITHCFactors(
            cp.ones((1, 1), dtype=cp.float64),
            cp.ones((1, 1), dtype=cp.float64),
            cp.ones((1, 1), dtype=cp.float64),
            transfer_counter=transfers,
        )
    before = transfers.to_dict()

    with pytest.raises(TypeError, match="eri_factors.x_occ.*NumPy ndarray"):
        thc_omega_cd_joint_audit(y_occ, y_vir, amplitude_core, eri)

    assert transfers.to_dict() == before
