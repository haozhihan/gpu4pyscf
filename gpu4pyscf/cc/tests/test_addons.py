import json
import sys
import traceback
import types

import numpy as np
import pytest

from gpu4pyscf.cc.addons import (
    FNOCCSD,
    FNOSemicanonicalizationError,
    GPUUnavailableError,
    _selection_mode_and_value,
    _verified_semicanonical_energies,
    make_fno_metadata,
)
from gpu4pyscf.cc import addons as addons_module


def _unbuilt_gpu_addon(**cc_kwargs):
    """Build the minimum wrapper state needed to inject a CC constructor."""

    addon = object.__new__(FNOCCSD)
    addon.use_gpu = True
    addon._cc = None
    addon.fno_mf = object()
    addon.metadata = types.SimpleNamespace(
        frozen_virtual=np.array([], dtype=int),
        delta_mp2=0.0,
    )
    addon.cc_kwargs = dict(cc_kwargs)
    return addon


def _small_h2():
    pyscf_gto = pytest.importorskip("pyscf.gto")
    pyscf_scf = pytest.importorskip("pyscf.scf")
    mol = pyscf_gto.M(atom="H 0 0 0; H 0 0 1.4", basis="6-31g", verbose=0)
    return mol, pyscf_scf.RHF(mol).run()


def test_fno_metadata_freezes_virtuals_only():
    _, mf = _small_h2()
    metadata = make_fno_metadata(mf, nvir_act=1)
    occupied = np.arange(metadata.nocc)
    assert np.intersect1d(metadata.frozen_virtual, occupied).size == 0
    assert metadata.nvir_active == 1
    assert metadata.frozen_virtual.size == metadata.nvir - 1
    assert metadata.no_coeff.shape == mf.mo_coeff.shape
    assert metadata.no_energy.shape == mf.mo_energy.shape
    assert np.isfinite(metadata.delta_mp2)
    assert metadata.selection_mode == "nvir_act"
    assert metadata.selection_value == 1
    assert np.all(metadata.virtual_occupations[:-1] >= metadata.virtual_occupations[1:])
    assert metadata.lowest_retained_occupation == metadata.virtual_occupations[0]
    assert metadata.highest_discarded_occupation == metadata.virtual_occupations[1]
    assert metadata.semicanonical_offdiag_max <= 1e-8
    assert set(metadata.timings_s) == {
        "mp2_full",
        "occupation_and_fno_orbitals",
        "semicanonical_fock_validation",
        "mp2_fno",
        "total",
    }
    assert all(value >= 0.0 for value in metadata.timings_s.values())
    json.dumps(metadata.audit_dict())
    assert np.isclose(
        metadata.mp2_fno_corr + metadata.delta_mp2,
        metadata.mp2_full_corr,
    )


def test_cpu_fno_ccsd_is_explicit_and_audited():
    _, mf = _small_h2()
    addon = FNOCCSD(mf, nvir_act=1, use_gpu=False)
    cc = addon.build()
    assert cc.__class__.__module__.startswith("pyscf")
    assert {"fno_metadata", "fno_delta_mp2", "fno_backend"} <= cc._keys
    assert np.isclose(cc.fno_delta_mp2, addon.metadata.delta_mp2)
    result = addon.kernel()
    assert np.isclose(result[0], cc.e_corr)
    assert np.isclose(addon.corrected_e_corr, result[0] + addon.delta_mp2)
    energies = addon.energy_metadata
    assert np.isclose(
        energies["raw_fno_ccsd_correlation_energy_eh"], result[0]
    )
    assert np.isclose(
        energies["corrected_correlation_energy_eh"], addon.corrected_e_corr
    )
    assert addon.run() is addon


def test_effective_selection_precedence_is_explicit():
    assert _selection_mode_and_value(1e-6, 0.95, 7) == ("nvir_act", 7)
    assert _selection_mode_and_value(1e-6, 0.95, None) == ("pct_occ", 0.95)
    assert _selection_mode_and_value(3e-5, None, None) == (
        "occupation_threshold",
        3e-5,
    )


def test_semicanonical_energy_builder_fails_closed_without_fock():
    class MissingFock:
        pass

    with pytest.raises(FNOSemicanonicalizationError, match="get_fock"):
        _verified_semicanonical_energies(
            MissingFock(), np.eye(2), nocc=1, nvir_active=1
        )


def test_fno_metadata_simulated_mp2_records_complete_audit(monkeypatch):
    class MeanField:
        mo_coeff = np.eye(3)
        mo_energy = np.array([-1.0, 0.5, 1.5])
        mo_occ = np.array([2.0, 0.0, 0.0])

        @staticmethod
        def get_fock():
            return np.diag([-1.0, 0.5, 1.5])

    class FakeRMP2:
        def __init__(self, mf, frozen=None):
            self._scf = mf
            self.frozen = frozen
            self.nocc = 1
            self.nmo = 3
            self.t2 = np.zeros((1, 1, 2, 2))
            if frozen:
                self.e_corr, self.e_tot = -0.15, -1.15
            else:
                self.e_corr, self.e_tot = -0.20, -1.20

        def run(self):
            return self

        @staticmethod
        def make_rdm1(t2=None, with_frozen=False):
            assert t2 is not None
            assert with_frozen is False
            return np.diag([1.99, 1e-2, 1e-7])

    pyscf_module = types.ModuleType("pyscf")
    mp_package = types.ModuleType("pyscf.mp")
    mp2_module = types.SimpleNamespace(RMP2=FakeRMP2)
    mp_package.mp2 = mp2_module
    pyscf_module.mp = mp_package
    monkeypatch.setitem(sys.modules, "pyscf", pyscf_module)
    monkeypatch.setitem(sys.modules, "pyscf.mp", mp_package)

    metadata = make_fno_metadata(MeanField(), thresh=1e-6)

    assert metadata.selection_mode == "occupation_threshold"
    assert metadata.nvir_active == 1
    assert metadata.frozen_virtual.tolist() == [2]
    assert metadata.virtual_occupations.tolist() == [1e-2, 1e-7]
    assert np.isclose(metadata.delta_mp2, -0.05)
    audit = metadata.audit_dict()
    assert audit["occupation_boundary"] == {
        "lowest_retained": 1e-2,
        "highest_discarded": 1e-7,
    }
    assert audit["mp2"]["full_correlation_energy_eh"] == -0.2
    assert audit["mp2"]["fno_correlation_energy_eh"] == -0.15


def test_fno_active_virtual_subspace_matches_pyscf_reference():
    pyscf_mp = pytest.importorskip("pyscf.mp")
    _, mf = _small_h2()
    reference_mp2 = pyscf_mp.MP2(mf).run()
    reference_frozen, reference_coeff = reference_mp2.make_fno(nvir_act=1)
    metadata = make_fno_metadata(mf, nvir_act=1)
    assert np.array_equal(metadata.frozen_virtual, np.asarray(reference_frozen))
    overlap = mf.get_ovlp()
    ours = metadata.no_coeff[:, metadata.nocc:metadata.nocc + 1]
    reference = reference_coeff[:, metadata.nocc:metadata.nocc + 1]
    # The vector sign is arbitrary; compare the S-metric subspace overlap.
    assert np.isclose(abs((ours.conj().T @ overlap @ reference)[0, 0]), 1.0)


def test_direct_ccsd_transform_uses_only_active_mo_columns():
    pytest.importorskip("cupy")
    pytest.importorskip("pyscf")
    from gpu4pyscf.cc.ccsd_incore import _active_mo_coeff

    class Solver:
        mo_coeff = np.arange(30.0).reshape(5, 6)

        @staticmethod
        def get_frozen_mask():
            return np.array([True, True, False, True, False, True])

    active = _active_mo_coeff(Solver(), nocc=2, nvir=2)
    assert np.array_equal(active, Solver.mo_coeff[:, [0, 1, 3, 5]])
    with pytest.raises(ValueError, match="active MO count"):
        _active_mo_coeff(Solver(), nocc=2, nvir=3)


def test_gpu_backend_import_failure_is_classified_as_unavailable(monkeypatch):
    addon = _unbuilt_gpu_addon()
    monkeypatch.setattr(addons_module, "gpu_available", lambda: True)

    def unavailable_import(name):
        assert name == "gpu4pyscf.cc.ccsd_incore"
        raise ModuleNotFoundError("missing GPU runtime dependency")

    monkeypatch.setattr(addons_module.importlib, "import_module", unavailable_import)
    with pytest.raises(GPUUnavailableError) as caught:
        addon.build()
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)


def test_gpu_backend_loader_failure_is_classified_as_unavailable(monkeypatch):
    addon = _unbuilt_gpu_addon()
    monkeypatch.setattr(addons_module, "gpu_available", lambda: True)

    def fail_to_load_backend(*_args, **_kwargs):
        raise OSError("failed to load CUDA-linked CCSD library")

    backend = types.SimpleNamespace(CCSD=fail_to_load_backend)
    monkeypatch.setattr(
        addons_module.importlib, "import_module", lambda _name: backend
    )
    with pytest.raises(GPUUnavailableError) as caught:
        addon.build()
    assert isinstance(caught.value.__cause__, OSError)


def test_gpu_ccsd_configuration_error_preserves_type_and_traceback(monkeypatch):
    addon = _unbuilt_gpu_addon(unsupported_option=True)
    monkeypatch.setattr(addons_module, "gpu_available", lambda: True)

    class ConfigurationError(ValueError):
        pass

    def reject_configuration(*_args, **_kwargs):
        raise ConfigurationError("invalid injected CCSD configuration")

    backend = types.SimpleNamespace(CCSD=reject_configuration)
    monkeypatch.setattr(
        addons_module.importlib, "import_module", lambda _name: backend
    )
    with pytest.raises(ConfigurationError) as caught:
        addon.build()

    frames = traceback.extract_tb(caught.tb)
    assert frames[-1].name == "reject_configuration"
    assert caught.value.__cause__ is None
