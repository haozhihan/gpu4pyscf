"""Dense audits for the T1-transformed inactive F-hat."""

import numpy as np
import pytest

from gpu4pyscf.cc.integrals import MOThreeIndexIntegralProvider
from gpu4pyscf.cc.thc_fhat import build_t1_transformed_fhat


def _problem(seed=1101):
    rng = np.random.default_rng(seed)
    nocc, nvir, naux = 2, 3, 5
    nmo = nocc + nvir
    h_mo = rng.normal(size=(nmo, nmo))
    h_mo = (h_mo + h_mo.T) * 0.5
    l_mo = rng.normal(size=(naux, nmo, nmo))
    l_mo = (l_mo + l_mo.swapaxes(1, 2)) * 0.5
    t1 = rng.normal(size=(nocc, nvir)) * 0.17
    return h_mo, l_mo, t1


def _provenance(seed=1101):
    return {
        "orbital_identity": {
            "case": "synthetic-provider-isomorphism",
            "seed": seed,
        },
        "integral_fingerprint": f"sha256:integrals-{seed}",
        "hcore_identity": {"source": "synthetic-symmetric-hcore"},
        "test_context": {"purpose": "audit-only"},
        "audit_chain": ["fhat", 1, 0.5, None],
    }


def _dense_oracle(h_mo, l_mo, t1):
    nocc, nvir = t1.shape
    identity_occ = np.eye(nocc)
    identity_vir = np.eye(nvir)
    zeros_vo = np.zeros((nvir, nocc))
    zeros_ov = np.zeros((nocc, nvir))
    p_matrix = np.block(
        [[identity_occ, -t1], [zeros_vo, identity_vir]]
    )
    h_matrix = np.block(
        [[identity_occ, zeros_ov], [t1.T, identity_vir]]
    )
    hhat = p_matrix.T @ h_mo @ h_matrix
    lhat = np.stack(
        [p_matrix.T @ factor @ h_matrix for factor in l_mo], axis=0
    )
    eri = np.einsum("Apq,Ars->pqrs", lhat, lhat)
    antisymmetrized = 2.0 * eri - eri.transpose(0, 3, 2, 1)
    fhat = hhat.copy()
    for occupied_index in range(nocc):
        fhat += antisymmetrized[
            occupied_index, occupied_index, :, :
        ]
    return lhat, fhat


def _assert_matches_oracle(result, lhat, fhat, nocc):
    virtual = slice(nocc, fhat.shape[0])
    occupied = slice(0, nocc)
    np.testing.assert_allclose(result.l_oo, lhat[:, occupied, occupied], atol=2e-14)
    np.testing.assert_allclose(result.l_ov, lhat[:, occupied, virtual], atol=2e-14)
    np.testing.assert_allclose(result.l_vo, lhat[:, virtual, occupied], atol=2e-14)
    np.testing.assert_allclose(result.l_vv, lhat[:, virtual, virtual], atol=2e-14)
    np.testing.assert_allclose(result.fhat_oo, fhat[occupied, occupied], atol=3e-13)
    np.testing.assert_allclose(result.fhat_ov, fhat[occupied, virtual], atol=3e-13)
    np.testing.assert_allclose(result.fhat_vo, fhat[virtual, occupied], atol=3e-13)
    np.testing.assert_allclose(result.fhat_vv, fhat[virtual, virtual], atol=3e-13)


def test_transformed_blocks_and_inactive_fhat_match_explicit_dense_oracle():
    h_mo, l_mo, t1 = _problem()
    lhat, fhat = _dense_oracle(h_mo, l_mo, t1)
    result = build_t1_transformed_fhat(
        h_mo, l_mo, t1, auxiliary_block_size=2
    )

    _assert_matches_oracle(result, lhat, fhat, t1.shape[0])
    assert np.linalg.norm(result.l_vo - result.l_ov.swapaxes(1, 2)) > 1e-3
    assert result.host_scalar_reads == 0


@pytest.mark.parametrize("block_size", [1, 2, 4, 5, 17, None])
def test_auxiliary_blocking_is_invariant(block_size):
    h_mo, l_mo, t1 = _problem(seed=1102)
    expected = build_t1_transformed_fhat(
        h_mo, l_mo, t1, auxiliary_block_size=None
    )
    observed = build_t1_transformed_fhat(
        h_mo, l_mo, t1, auxiliary_block_size=block_size
    )
    for name in (
        "l_oo",
        "l_vv",
        "l_ov",
        "l_vo",
        "fhat_oo",
        "fhat_vv",
        "fhat_ov",
        "fhat_vo",
    ):
        np.testing.assert_allclose(
            getattr(observed, name), getattr(expected, name), atol=3e-13
        )
    expected_effective = l_mo.shape[0] if block_size is None else min(
        block_size, l_mo.shape[0]
    )
    assert observed.auxiliary_block_size == expected_effective


def test_metadata_is_fail_closed_and_declares_algorithm_consumers():
    h_mo, l_mo, t1 = _problem(seed=1103)
    result = build_t1_transformed_fhat(
        h_mo, l_mo, t1, auxiliary_block_size=2
    )
    metadata = result.metadata()

    assert metadata["audit_only"] is True
    assert metadata["production_enabled"] is False
    assert metadata["performance_eligible"] is False
    assert metadata["gpu_performance_claim"] is False
    assert metadata["backend"] == "numpy"
    assert metadata["dtype"] == "float64"
    assert metadata["algorithm_consumers"] == [8, 9, 10]
    assert metadata["algorithm8_cholesky_blocks"] == ["oo", "vv", "ov"]
    assert metadata["algorithm9_fhat_blocks"] == ["oo", "vv"]
    assert metadata["algorithm10_fhat_blocks"] == ["ov", "vo"]
    assert metadata["includes_t2"] is False
    assert metadata["materializes_four_index_eri"] is False
    assert metadata["materializes_dense_t2"] is False
    assert metadata["implicit_host_tensor_transfer"] is False
    assert metadata["raw_cholesky_symmetry_required"] is True
    assert metadata["raw_cholesky_symmetry_policy"] == (
        "exact-provider-output-by-default-or-user-supplied-"
        "empirical-absolute-tolerance"
    )
    assert metadata["raw_cholesky_max_asymmetry"] == 0.0
    assert metadata["raw_cholesky_empirical_absolute_tolerance"] == 0.0
    assert metadata["raw_cholesky_tolerance_source"] == "default-exact"
    assert metadata["raw_cholesky_tolerance_is_proven_error_bound"] is False
    assert metadata["raw_cholesky_resymmetrized"] is True
    assert metadata["transformed_cholesky_symmetry_enforced"] is False
    assert metadata["host_scalar_reads"] == 0
    assert metadata["provenance_bound"] is False
    assert metadata["provenance_context"] == {}
    assert metadata["formal_full_residual_validated"] is False
    assert metadata["formal_full_residual_eligible"] is False


def test_real_mo_provider_output_is_exactly_symmetric_and_provenance_bound():
    """Exercise the actual block-Hermitized ``C.T @ L_AO @ C`` seam."""

    rng = np.random.default_rng(1)
    nocc, nvir, nao, naux = 2, 3, 31, 7
    nmo = nocc + nvir
    mo_coeff, _ = np.linalg.qr(rng.normal(size=(nao, nmo)))
    l_ao = rng.normal(size=(naux, nao, nao))
    l_ao = (l_ao + l_ao.swapaxes(1, 2)) * 0.5
    provider = MOThreeIndexIntegralProvider.from_ao_factors(
        l_ao,
        mo_coeff,
        nocc,
        factorization="custom",
        threshold=None,
        auxiliary_block_size=3,
    )
    assert np.array_equal(
        provider.factors, provider.factors.swapaxes(1, 2)
    )

    h_mo = rng.normal(size=(nmo, nmo))
    h_mo = (h_mo + h_mo.T) * 0.5
    t1 = rng.normal(size=(nocc, nvir)) * 0.05
    provenance = _provenance(seed=1)
    result = build_t1_transformed_fhat(
        h_mo,
        provider.factors,
        t1,
        auxiliary_block_size=3,
        provenance_context=provenance,
    )

    metadata = result.metadata()
    assert metadata["raw_cholesky_max_asymmetry"] == 0.0
    assert metadata["raw_cholesky_empirical_absolute_tolerance"] == 0.0
    assert metadata["raw_cholesky_tolerance_source"] == "default-exact"
    assert metadata["raw_cholesky_resymmetrized"] is True
    assert metadata["provenance_bound"] is True
    assert metadata["provenance_context"] == provenance

    lhat, fhat = _dense_oracle(h_mo, provider.factors, t1)
    _assert_matches_oracle(result, lhat, fhat, nocc)


def test_unhermitized_transform_requires_explicit_empirical_tolerance():
    rng = np.random.default_rng(1)
    nocc, nvir, nao, naux = 2, 3, 31, 7
    nmo = nocc + nvir
    mo_coeff, _ = np.linalg.qr(rng.normal(size=(nao, nmo)))
    l_ao = rng.normal(size=(naux, nao, nao))
    l_ao = (l_ao + l_ao.swapaxes(1, 2)) * 0.5
    unhermitized = np.stack(
        [mo_coeff.T @ factor @ mo_coeff for factor in l_ao], axis=0
    )
    max_asymmetry = float(
        np.max(np.abs(unhermitized - unhermitized.swapaxes(1, 2)))
    )
    assert max_asymmetry > 0.0
    h_mo = np.eye(nmo, dtype=np.float64)
    t1 = np.zeros((nocc, nvir), dtype=np.float64)

    with pytest.raises(ValueError, match="user-supplied empirical"):
        build_t1_transformed_fhat(h_mo, unhermitized, t1)
    result = build_t1_transformed_fhat(
        h_mo,
        unhermitized,
        t1,
        empirical_raw_cholesky_symmetry_tolerance=max_asymmetry,
    )
    metadata = result.metadata()
    assert metadata["raw_cholesky_max_asymmetry"] == max_asymmetry
    assert metadata["raw_cholesky_empirical_absolute_tolerance"] == max_asymmetry
    assert metadata["raw_cholesky_tolerance_source"] == "user-supplied-empirical"
    assert metadata["raw_cholesky_tolerance_is_proven_error_bound"] is False


@pytest.mark.parametrize("scale", [0.0, np.nextafter(0.0, 1.0)])
def test_zero_and_subnormal_symmetric_factors_pass_exact_default(scale):
    h_mo = np.eye(3, dtype=np.float64)
    l_mo = np.zeros((2, 3, 3), dtype=np.float64)
    l_mo[:, 0, 1] = scale
    l_mo[:, 1, 0] = scale
    t1 = np.zeros((1, 2), dtype=np.float64)
    result = build_t1_transformed_fhat(h_mo, l_mo, t1)
    metadata = result.metadata()
    assert metadata["raw_cholesky_scale"] == scale
    assert metadata["raw_cholesky_max_asymmetry"] == 0.0
    assert metadata["raw_cholesky_empirical_absolute_tolerance"] == 0.0
    assert metadata["raw_cholesky_resymmetrized"] is True


def test_provenance_context_is_complete_copied_and_never_claims_validation():
    h_mo, l_mo, t1 = _problem(seed=1110)
    provenance = _provenance(seed=1110)
    expected = _provenance(seed=1110)
    result = build_t1_transformed_fhat(
        h_mo, l_mo, t1, provenance_context=provenance
    )

    provenance["orbital_identity"]["case"] = "mutated-after-build"
    first = result.metadata()
    assert first["provenance_bound"] is True
    assert first["provenance_context"] == expected
    assert first["formal_full_residual_validated"] is False
    assert first["formal_full_residual_eligible"] is False

    first["provenance_context"]["hcore_identity"]["source"] = "mutated-copy"
    assert result.metadata()["provenance_context"] == expected


@pytest.mark.parametrize(
    "bad_context, error, match",
    [
        ("not-a-mapping", TypeError, "dictionary"),
        ({}, ValueError, "empty dictionary"),
        (
            {"orbital_fingerprint": "sha256:orbitals"},
            ValueError,
            "missing integrals, hcore",
        ),
        (
            {
                "orbital_fingerprint": "sha256:orbitals",
                "integral_identity": {},
                "hcore_fingerprint": "sha256:hcore",
            },
            ValueError,
            "empty dictionary",
        ),
        (
            {
                "orbital_identity": True,
                "integral_fingerprint": "sha256:integrals",
                "hcore_fingerprint": "sha256:hcore",
            },
            TypeError,
            "boolean identity",
        ),
        (
            {
                "orbital_identity": b"bytes",
                "integral_fingerprint": "sha256:integrals",
                "hcore_fingerprint": "sha256:hcore",
            },
            TypeError,
            "JSON-safe host tree",
        ),
        (
            {
                "orbital_identity": np.asarray([1.0]),
                "integral_fingerprint": "sha256:integrals",
                "hcore_fingerprint": "sha256:hcore",
            },
            TypeError,
            "JSON-safe host tree",
        ),
        (
            {
                "orbital_identity": {"coefficients": []},
                "integral_fingerprint": "sha256:integrals",
                "hcore_fingerprint": "sha256:hcore",
            },
            ValueError,
            "empty list",
        ),
        (
            {
                "orbital_identity": {"name": "   "},
                "integral_fingerprint": "sha256:integrals",
                "hcore_fingerprint": "sha256:hcore",
            },
            ValueError,
            "empty string",
        ),
        (
            {
                "orbital_identity": {"energy": float("nan")},
                "integral_fingerprint": "sha256:integrals",
                "hcore_fingerprint": "sha256:hcore",
            },
            ValueError,
            "finite floats",
        ),
        (
            {
                "orbital_identity": {"energy": float("inf")},
                "integral_fingerprint": "sha256:integrals",
                "hcore_fingerprint": "sha256:hcore",
            },
            ValueError,
            "finite floats",
        ),
        (
            {
                "orbital_identity": {"device": object()},
                "integral_fingerprint": "sha256:integrals",
                "hcore_fingerprint": "sha256:hcore",
            },
            TypeError,
            "JSON-safe host tree",
        ),
    ],
)
def test_partial_or_invalid_provenance_context_fails_closed(
    bad_context, error, match
):
    h_mo, l_mo, t1 = _problem(seed=1111)
    with pytest.raises(error, match=match):
        build_t1_transformed_fhat(
            h_mo, l_mo, t1, provenance_context=bad_context
        )


def test_cupy_array_and_device_object_provenance_fail_closed():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    h_mo, l_mo, t1 = _problem(seed=1113)
    for device_value in (cp.asarray([1.0]), cp.cuda.Device()):
        context = _provenance(seed=1113)
        context["orbital_identity"] = {"device_value": device_value}
        with pytest.raises(TypeError, match="JSON-safe host tree"):
            build_t1_transformed_fhat(
                h_mo, l_mo, t1, provenance_context=context
            )


@pytest.mark.parametrize("bad_block", [0, -1, True, 1.5, "2"])
def test_bad_auxiliary_block_size_fails_closed(bad_block):
    h_mo, l_mo, t1 = _problem(seed=1104)
    expected_error = (
        TypeError if isinstance(bad_block, (bool, float, str)) else ValueError
    )
    with pytest.raises(expected_error, match="auxiliary_block_size"):
        build_t1_transformed_fhat(
            h_mo, l_mo, t1, auxiliary_block_size=bad_block
        )


def test_bad_shapes_dtypes_symmetry_and_finite_values_fail_closed():
    h_mo, l_mo, t1 = _problem(seed=1105)
    with pytest.raises(ValueError, match="t1"):
        build_t1_transformed_fhat(h_mo, l_mo, t1.ravel())
    with pytest.raises(ValueError, match="h_mo"):
        build_t1_transformed_fhat(h_mo[:-1, :-1], l_mo, t1)
    with pytest.raises(ValueError, match="l_mo"):
        build_t1_transformed_fhat(h_mo, l_mo[:, :-1, :-1], t1)
    with pytest.raises(TypeError, match="same dtype"):
        build_t1_transformed_fhat(h_mo, l_mo, t1.astype(np.float32))
    with pytest.raises(TypeError, match="real FP64"):
        build_t1_transformed_fhat(
            h_mo.astype(np.float32),
            l_mo.astype(np.float32),
            t1.astype(np.float32),
        )
    with pytest.raises(NotImplementedError, match="real RHF"):
        build_t1_transformed_fhat(
            h_mo.astype(np.complex128),
            l_mo.astype(np.complex128),
            t1.astype(np.complex128),
        )

    nonsymmetric = l_mo.copy()
    nonsymmetric[0, 0, 1] += 1e-12
    with pytest.raises(ValueError, match="user-supplied empirical"):
        build_t1_transformed_fhat(h_mo, nonsymmetric, t1)
    for location in ("h_mo", "l_mo", "t1"):
        values = {
            "h_mo": h_mo.copy(),
            "l_mo": l_mo.copy(),
            "t1": t1.copy(),
        }
        values[location].flat[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            build_t1_transformed_fhat(
                values["h_mo"], values["l_mo"], values["t1"]
            )


@pytest.mark.parametrize(
    "bad_tolerance, error",
    [
        (True, TypeError),
        ("1e-12", TypeError),
        (-1e-12, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_bad_empirical_symmetry_tolerance_fails_closed(bad_tolerance, error):
    h_mo, l_mo, t1 = _problem(seed=1112)
    with pytest.raises(error, match="empirical_raw_cholesky"):
        build_t1_transformed_fhat(
            h_mo,
            l_mo,
            t1,
            empirical_raw_cholesky_symmetry_tolerance=bad_tolerance,
        )


def test_numpy_and_cupy_backends_cannot_be_mixed():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    h_mo, l_mo, t1 = _problem(seed=1106)
    with pytest.raises(TypeError, match="one array backend"):
        build_t1_transformed_fhat(cp.asarray(h_mo), l_mo, cp.asarray(t1))


def test_cupy_outputs_remain_resident_and_scalar_reads_are_accounted():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    h_mo, l_mo, t1 = _problem(seed=1107)
    device_id = int(cp.cuda.Device().id)
    gpu_h = cp.asarray(h_mo)
    gpu_l = cp.asarray(l_mo)
    gpu_t1 = cp.asarray(t1)
    with pytest.raises(ValueError, match="explicit transfer_counter"):
        build_t1_transformed_fhat(gpu_h, gpu_l, gpu_t1)

    transfers = TransferCounter()
    result = build_t1_transformed_fhat(
        gpu_h,
        gpu_l,
        gpu_t1,
        auxiliary_block_size=2,
        transfer_counter=transfers,
    )
    for name in (
        "l_oo",
        "l_vv",
        "l_ov",
        "l_vo",
        "fhat_oo",
        "fhat_vv",
        "fhat_ov",
        "fhat_vo",
    ):
        value = getattr(result, name)
        assert isinstance(value, cp.ndarray)
        assert int(value.device.id) == device_id

    assert result.host_scalar_reads == 4
    report = transfers.to_dict()
    assert report["total_transfers"] == result.host_scalar_reads
    assert report["by_kind"] == {"d2h": {"bytes": 18, "count": 4}}
    assert report["by_operation"]["d2h"] == {
        "t1_fhat_input_finite": {"bytes": 1, "count": 1},
        "t1_fhat_raw_cholesky_scale": {"bytes": 8, "count": 1},
        "t1_fhat_raw_cholesky_asymmetry": {"bytes": 8, "count": 1},
        "t1_fhat_output_finite": {"bytes": 1, "count": 1},
    }
    before = transfers.total_transfers
    metadata = result.metadata()
    assert transfers.total_transfers == before
    assert metadata["backend"] == "cupy"

    lhat, fhat = _dense_oracle(h_mo, l_mo, t1)
    cpu_result = type("CPUResult", (), {})()
    for name in (
        "l_oo",
        "l_vv",
        "l_ov",
        "l_vo",
        "fhat_oo",
        "fhat_vv",
        "fhat_ov",
        "fhat_vo",
    ):
        setattr(cpu_result, name, cp.asnumpy(getattr(result, name)))
    _assert_matches_oracle(cpu_result, lhat, fhat, t1.shape[0])


def test_cupy_mixed_devices_fail_closed_when_two_devices_are_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("two CUDA devices are required")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    h_mo, l_mo, t1 = _problem(seed=1108)
    with cp.cuda.Device(0):
        gpu_h = cp.asarray(h_mo)
        gpu_t1 = cp.asarray(t1)
    with cp.cuda.Device(1):
        gpu_l = cp.asarray(l_mo)
    with pytest.raises(TypeError, match="one device"):
        build_t1_transformed_fhat(
            gpu_h,
            gpu_l,
            gpu_t1,
            transfer_counter=TransferCounter(),
        )


def test_cupy_current_device_mismatch_fails_closed_when_two_devices_exist():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("two CUDA devices are required")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    from gpu4pyscf.cc.device_runtime import TransferCounter

    h_mo, l_mo, t1 = _problem(seed=1109)
    with cp.cuda.Device(0):
        gpu_h = cp.asarray(h_mo)
        gpu_l = cp.asarray(l_mo)
        gpu_t1 = cp.asarray(t1)
    with cp.cuda.Device(1):
        with pytest.raises(RuntimeError, match="current CuPy device"):
            build_t1_transformed_fhat(
                gpu_h,
                gpu_l,
                gpu_t1,
                transfer_counter=TransferCounter(),
            )
