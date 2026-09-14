# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convenience helpers for MP2 frozen-natural-orbital CCSD.

The FNO construction is intentionally driven by :mod:`pyscf.mp.mp2`.  This
keeps the natural-orbital density and MP2 reference energy independent of the
GPU backend.  The resulting CCSD object is constructed from
``gpu4pyscf.cc.ccsd_incore`` only when a CUDA device is available (or when the
caller explicitly requests the CPU backend).
"""

from __future__ import annotations

import copy
import importlib
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


class GPUUnavailableError(RuntimeError):
    """Raised when GPU4PySCF CCSD was requested without a usable CUDA device."""


class FNOSemicanonicalizationError(RuntimeError):
    """Raised when production FNO orbital energies cannot be verified."""


def gpu_available() -> bool:
    """Return whether CuPy can see at least one usable CUDA device."""

    try:
        import cupy

        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _to_numpy(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def _cpu_mean_field(mf):
    """Obtain a CPU-view SCF object without mutating the caller's object."""

    if getattr(mf.__class__, "__module__", "").startswith("gpu4pyscf") and hasattr(mf, "to_cpu"):
        mf = mf.to_cpu()
    else:
        mf = copy.copy(mf)
    for name in ("mo_coeff", "mo_energy", "mo_occ"):
        if hasattr(mf, name):
            setattr(mf, name, _to_numpy(getattr(mf, name)))
    return mf


@dataclass(frozen=True)
class FNOMetadata:
    """Auditable details of an FNO truncation and its MP2 energy change."""

    threshold: float
    pct_occ: Optional[float]
    requested_nvir_act: Optional[int]
    selection_mode: str
    selection_value: float | int
    nocc: int
    nvir: int
    nvir_active: int
    frozen_virtual: np.ndarray
    active_virtual: np.ndarray
    virtual_occupations: np.ndarray
    lowest_retained_occupation: Optional[float]
    highest_discarded_occupation: Optional[float]
    no_coeff: np.ndarray
    no_energy: np.ndarray
    semicanonical_offdiag_max: float
    mp2_full_corr: float
    mp2_fno_corr: float
    delta_mp2: float
    mp2_full_total: float
    mp2_fno_total: float
    timings_s: Mapping[str, float]

    @property
    def frozen(self) -> np.ndarray:
        """PySCF-compatible frozen-orbital index list."""

        return self.frozen_virtual

    def audit_dict(self) -> dict[str, Any]:
        """Return JSON-ready FNO selection, energy, and timing provenance."""

        return {
            "threshold": self.threshold,
            "pct_occ": self.pct_occ,
            "requested_nvir_act": self.requested_nvir_act,
            "selection_mode": self.selection_mode,
            "selection_value": self.selection_value,
            "nocc": self.nocc,
            "nvir": self.nvir,
            "nvir_active": self.nvir_active,
            "frozen_virtual": self.frozen_virtual.tolist(),
            "active_virtual": self.active_virtual.tolist(),
            "virtual_occupations": self.virtual_occupations.tolist(),
            "occupation_boundary": {
                "lowest_retained": self.lowest_retained_occupation,
                "highest_discarded": self.highest_discarded_occupation,
            },
            "semicanonical_offdiag_max": self.semicanonical_offdiag_max,
            "mp2": {
                "full_correlation_energy_eh": self.mp2_full_corr,
                "fno_correlation_energy_eh": self.mp2_fno_corr,
                "delta_mp2_eh": self.delta_mp2,
                "full_total_energy_eh": self.mp2_full_total,
                "fno_total_energy_eh": self.mp2_fno_total,
            },
            "timings_s": dict(self.timings_s),
        }


def _validate_fno_options(thresh, pct_occ, nvir_act):
    if thresh < 0:
        raise ValueError("thresh must be non-negative")
    if pct_occ is not None and not 0 <= pct_occ <= 1:
        raise ValueError("pct_occ must be between zero and one")
    if nvir_act is not None and (int(nvir_act) != nvir_act or nvir_act < 1):
        raise ValueError("nvir_act must be a positive integer")


def _selection_mode_and_value(
    thresh: float, pct_occ: Optional[float], nvir_act: Optional[int]
) -> tuple[str, float | int]:
    """Return PySCF's effective FNO selector and its stable value."""

    if nvir_act is not None:
        return "nvir_act", int(nvir_act)
    if pct_occ is not None:
        return "pct_occ", float(pct_occ)
    return "occupation_threshold", float(thresh)


def _build_fno_orbitals(
    pt: Any,
    mf: Any,
    *,
    thresh: float,
    pct_occ: Optional[float],
    nvir_act: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct PySCF-compatible FNOs while retaining the occupation spectrum.

    ``pyscf.mp.MP2.make_fno`` does not return its natural occupations.  The G2
    benchmark needs those values to audit a threshold scan, so this routine
    performs the same all-electron RHF construction from the public MP2 RDM.
    The helper deliberately rejects a pre-frozen MP2 object: the frozen-space
    bookkeeping has a different orbital ordering and is outside this project's
    all-electron contract.
    """

    frozen = getattr(pt, "frozen", None)
    has_frozen = frozen is not None
    if isinstance(frozen, (int, np.integer)):
        has_frozen = int(frozen) != 0
    elif frozen is not None:
        try:
            has_frozen = len(frozen) != 0
        except TypeError:
            has_frozen = True
    if has_frozen:
        raise ValueError("FNOCCSD supports the all-electron MP2 reference only")
    nocc = int(pt.nocc)
    nmo = int(pt.nmo)
    nvir = nmo - nocc
    if nocc <= 0 or nvir <= 0:
        raise ValueError("FNO construction requires occupied and virtual orbitals")
    density = np.asarray(_to_numpy(pt.make_rdm1(t2=pt.t2, with_frozen=False)))
    if density.shape != (nmo, nmo) or not np.all(np.isfinite(density)):
        raise ValueError("MP2 one-particle density has an invalid shape or values")
    occupations, natural_virtuals = np.linalg.eigh(density[nocc:, nocc:])
    order = np.argsort(occupations)[::-1]
    occupations = np.real(occupations[order])
    natural_virtuals = natural_virtuals[:, order]

    if nvir_act is not None:
        nvir_keep = min(nvir, int(nvir_act))
    elif pct_occ is not None:
        occupation_sum = float(np.sum(occupations))
        if not np.isfinite(occupation_sum) or occupation_sum <= 0.0:
            raise ValueError("MP2 virtual occupation sum is not positive")
        cumulative = np.cumsum(occupations / occupation_sum)
        nvir_keep = int(np.count_nonzero(
            np.logical_or(cumulative <= pct_occ, np.isclose(cumulative, pct_occ))
        ))
    else:
        nvir_keep = int(np.count_nonzero(occupations > thresh))
    if nvir_keep < 1:
        raise ValueError("FNO selector freezes every virtual orbital")

    mo_coeff = np.asarray(_to_numpy(mf.mo_coeff))
    mo_energy = np.asarray(_to_numpy(mf.mo_energy))
    if mo_coeff.ndim != 2 or mo_coeff.shape[1] != nmo or mo_energy.shape != (nmo,):
        raise ValueError("SCF orbitals do not match the MP2 orbital dimensions")
    # PySCF semicanonicalizes the retained NO block with the canonical virtual
    # Fock eigenvalues.  Discarded NOs need no further rotation because they are
    # frozen in every subsequent MP2/CCSD operation.
    fvv_no = natural_virtuals.conj().T @ np.diag(mo_energy[nocc:]) @ natural_virtuals
    _, retained_rotation = np.linalg.eigh(fvv_no[:nvir_keep, :nvir_keep])
    occupied_coeff = mo_coeff[:, :nocc]
    virtual_coeff = mo_coeff[:, nocc:]
    retained_coeff = virtual_coeff @ natural_virtuals[:, :nvir_keep] @ retained_rotation
    discarded_coeff = virtual_coeff @ natural_virtuals[:, nvir_keep:]
    no_coeff = np.hstack((occupied_coeff, retained_coeff, discarded_coeff))
    frozen_virtual = np.arange(nocc + nvir_keep, nmo, dtype=int)
    return frozen_virtual, no_coeff, occupations


def _verified_semicanonical_energies(
    mf: Any,
    no_coeff: np.ndarray,
    *,
    nocc: int,
    nvir_active: int,
) -> tuple[np.ndarray, float]:
    """Build and verify the diagonal Fock energies used by FNO MP2/CCSD.

    Production calculations must never fall back to the canonical energies of
    the pre-FNO orbitals.  Such a fallback combines rotated coefficients with
    unrelated denominators and can produce a plausible but invalid result.
    """

    get_fock = getattr(mf, "get_fock", None)
    if not callable(get_fock):
        raise FNOSemicanonicalizationError(
            "the SCF object does not provide get_fock(); FNO denominators are unavailable"
        )
    try:
        fock_ao = np.asarray(_to_numpy(get_fock()))
    except Exception as exc:
        raise FNOSemicanonicalizationError(
            "failed to build the converged AO Fock matrix for FNO orbitals"
        ) from exc
    nao = no_coeff.shape[0]
    if fock_ao.shape != (nao, nao) or not np.all(np.isfinite(fock_ao)):
        raise FNOSemicanonicalizationError(
            "the AO Fock matrix has an invalid shape or non-finite values"
        )
    scale = max(1.0, float(np.max(np.abs(fock_ao))))
    hermitian_error = float(np.max(np.abs(fock_ao - fock_ao.conj().T)))
    hermitian_tol = 1.0e-10 * scale
    if hermitian_error > hermitian_tol:
        raise FNOSemicanonicalizationError(
            f"AO Fock matrix is not Hermitian: max error {hermitian_error:.3e}"
        )
    fock_mo = no_coeff.conj().T @ fock_ao @ no_coeff
    diagonal = np.diag(fock_mo)
    imaginary_error = float(np.max(np.abs(np.imag(diagonal))))
    if imaginary_error > 1.0e-10 * max(1.0, float(np.max(np.abs(diagonal)))):
        raise FNOSemicanonicalizationError(
            f"FNO orbital energies have a non-negligible imaginary part {imaginary_error:.3e}"
        )
    active_blocks = (
        fock_mo[:nocc, :nocc],
        fock_mo[nocc:nocc + nvir_active, nocc:nocc + nvir_active],
    )
    offdiag_max = 0.0
    for block in active_blocks:
        if block.size:
            offdiag = block - np.diag(np.diag(block))
            offdiag_max = max(offdiag_max, float(np.max(np.abs(offdiag))))
    # The converged canonical occupied block and the retained semicanonical
    # virtual block should both be diagonal to numerical precision.  A looser
    # 1e-8 Eh ceiling accommodates SCF roundoff while still failing decisively
    # on an incorrectly ordered or non-semicanonical coefficient matrix.
    offdiag_tol = 1.0e-8 * max(1.0, float(np.max(np.abs(diagonal))))
    if offdiag_max > offdiag_tol:
        raise FNOSemicanonicalizationError(
            "FNO active-space Fock blocks are not semicanonical: "
            f"max off-diagonal {offdiag_max:.3e} > {offdiag_tol:.3e}"
        )
    energies = np.asarray(np.real(diagonal), dtype=float)
    if not np.all(np.isfinite(energies)):
        raise FNOSemicanonicalizationError("FNO orbital energies are non-finite")
    return energies, offdiag_max


def make_fno_metadata(
    mf,
    *,
    thresh: float = 1e-6,
    pct_occ: Optional[float] = None,
    nvir_act: Optional[int] = None,
) -> FNOMetadata:
    """Run PySCF MP2 and construct a virtual-only FNO truncation.

    Every occupied orbital is retained.  The returned ``no_coeff`` follows
    PySCF's semicanonical ordering, and ``frozen_virtual`` contains only
    virtual indices in that ordering.  ``delta_mp2`` is the additive
    full-space MP2 minus FNO-space MP2 correction applied to FNO-CCSD.
    """

    total_start = time.perf_counter()
    _validate_fno_options(thresh, pct_occ, nvir_act)
    selection_mode, selection_value = _selection_mode_and_value(
        thresh, pct_occ, nvir_act
    )
    mf_cpu = _cpu_mean_field(mf)
    try:
        from pyscf import mp as pyscf_mp
    except Exception as exc:  # pragma: no cover - exercised only without PySCF
        raise RuntimeError("PySCF is required to build MP2 natural orbitals") from exc

    # Use PySCF's reference MP2 implementation explicitly.  This avoids
    # accidentally selecting gpu4pyscf.mp.MP2 through a patched SCF method.
    full_mp2_start = time.perf_counter()
    pt = pyscf_mp.mp2.RMP2(mf_cpu)
    pt.run()
    full_mp2_seconds = time.perf_counter() - full_mp2_start
    full_corr = float(np.real(pt.e_corr))
    full_total = float(np.real(pt.e_tot))
    if not np.isfinite(full_corr) or not np.isfinite(full_total):
        raise ValueError("full-space MP2 returned a non-finite energy")
    selection_start = time.perf_counter()
    frozen_virtual, no_coeff, occupations = _build_fno_orbitals(
        pt,
        mf_cpu,
        thresh=thresh,
        pct_occ=pct_occ,
        nvir_act=nvir_act,
    )
    selection_seconds = time.perf_counter() - selection_start
    mo_occ = np.asarray(_to_numpy(mf_cpu.mo_occ))
    nmo = mo_occ.size
    nocc = int(np.count_nonzero(mo_occ > 1e-8))
    if int(pt.nocc) != nocc or int(pt.nmo) != nmo:
        raise ValueError("MP2 and SCF occupied/orbital dimensions do not agree")
    if no_coeff.shape[1] != nmo:
        raise ValueError("PySCF FNO coefficients do not match the SCF orbital count")

    occupied = np.flatnonzero(mo_occ > 1e-8)
    if np.intersect1d(frozen_virtual, occupied).size:
        raise ValueError("FNO construction attempted to freeze an occupied orbital")
    virtual = np.arange(nocc, nmo, dtype=int)
    active_virtual = np.setdiff1d(virtual, frozen_virtual, assume_unique=False)
    if active_virtual.size == 0:
        raise ValueError("FNO threshold freezes every virtual orbital")

    # Re-run MP2 on the semicanonical FNO coefficients to quantify the energy
    # change.  The copy means the caller's SCF object and orbital arrays are
    # untouched.
    mf_fno = copy.copy(mf_cpu)
    mf_fno.mo_coeff = no_coeff
    mf_fno.mo_occ = mo_occ.copy()
    semicanonical_start = time.perf_counter()
    no_energy, semicanonical_offdiag_max = _verified_semicanonical_energies(
        mf_cpu,
        no_coeff,
        nocc=nocc,
        nvir_active=int(active_virtual.size),
    )
    semicanonical_seconds = time.perf_counter() - semicanonical_start
    mf_fno.mo_energy = no_energy
    fno_mp2_start = time.perf_counter()
    pt_fno = pyscf_mp.mp2.RMP2(mf_fno, frozen=frozen_virtual.tolist())
    pt_fno.run()
    fno_mp2_seconds = time.perf_counter() - fno_mp2_start
    fno_corr = float(np.real(pt_fno.e_corr))
    fno_total = float(np.real(pt_fno.e_tot))
    if not np.isfinite(fno_corr) or not np.isfinite(fno_total):
        raise ValueError("FNO-space MP2 returned a non-finite energy")
    nvir_active = int(active_virtual.size)
    timings = {
        "mp2_full": full_mp2_seconds,
        "occupation_and_fno_orbitals": selection_seconds,
        "semicanonical_fock_validation": semicanonical_seconds,
        "mp2_fno": fno_mp2_seconds,
    }
    timings["total"] = time.perf_counter() - total_start

    return FNOMetadata(
        threshold=float(thresh),
        pct_occ=None if pct_occ is None else float(pct_occ),
        requested_nvir_act=None if nvir_act is None else int(nvir_act),
        selection_mode=selection_mode,
        selection_value=selection_value,
        nocc=nocc,
        nvir=int(virtual.size),
        nvir_active=nvir_active,
        frozen_virtual=frozen_virtual,
        active_virtual=active_virtual,
        virtual_occupations=occupations,
        lowest_retained_occupation=(
            float(occupations[nvir_active - 1]) if nvir_active else None
        ),
        highest_discarded_occupation=(
            float(occupations[nvir_active]) if nvir_active < occupations.size else None
        ),
        no_coeff=no_coeff,
        no_energy=no_energy,
        semicanonical_offdiag_max=semicanonical_offdiag_max,
        mp2_full_corr=full_corr,
        mp2_fno_corr=fno_corr,
        delta_mp2=full_corr - fno_corr,
        mp2_full_total=full_total,
        mp2_fno_total=fno_total,
        timings_s=timings,
    )


class FNOCCSD:
    """Lazily constructed GPU4PySCF CCSD using MP2 frozen natural orbitals.

    Parameters are the same FNO selection controls as PySCF's
    ``MP2.make_fno``.  ``use_gpu=True`` is the default and raises
    :class:`GPUUnavailableError` when no CUDA device is available; it never
    silently substitutes a CPU CCSD object.  Set ``use_gpu=False`` only when
    an explicit CPU reference calculation is desired.
    """

    def __init__(
        self,
        mf,
        *,
        thresh: float = 1e-6,
        pct_occ: Optional[float] = None,
        nvir_act: Optional[int] = None,
        use_gpu: bool = True,
        allow_cpu_fallback: bool = False,
        **cc_kwargs: Any,
    ):
        if not isinstance(use_gpu, (bool, np.bool_)):
            raise TypeError("use_gpu must be a boolean")
        if allow_cpu_fallback:
            # The fallback is explicit in metadata and in the returned class;
            # it is provided for portable validation, never as a hidden path.
            use_gpu = bool(use_gpu and gpu_available())
        self.mf = mf
        self.use_gpu = bool(use_gpu)
        self.cc_kwargs = dict(cc_kwargs)
        if "frozen" in self.cc_kwargs:
            raise ValueError("pass FNO selection controls; frozen is managed by FNOCCSD")
        self.metadata = make_fno_metadata(
            mf, thresh=thresh, pct_occ=pct_occ, nvir_act=nvir_act
        )
        self.fno_mf = _cpu_mean_field(mf)
        self.fno_mf.mo_coeff = self.metadata.no_coeff
        self.fno_mf.mo_occ = np.asarray(_to_numpy(self.fno_mf.mo_occ))
        self.fno_mf.mo_energy = self.metadata.no_energy.copy()
        self._cc = None

    @property
    def delta_mp2(self) -> float:
        return self.metadata.delta_mp2

    @property
    def no_coeff(self) -> np.ndarray:
        return self.metadata.no_coeff

    @property
    def active_virtual(self) -> np.ndarray:
        return self.metadata.active_virtual

    @property
    def frozen(self) -> np.ndarray:
        return self.metadata.frozen_virtual

    @property
    def backend(self) -> str:
        return "gpu" if self.use_gpu else "cpu"

    @property
    def ccsd(self):
        """The configured backend CCSD object, constructing it on demand."""

        return self.build()

    def build(self):
        if self._cc is not None:
            return self._cc
        if self.use_gpu:
            if not gpu_available():
                raise GPUUnavailableError(
                    "GPU4PySCF CCSD was requested, but CuPy reports no CUDA device; "
                    "pass use_gpu=False for an explicit CPU reference"
                )
            try:
                ccsd_incore = importlib.import_module(
                    "gpu4pyscf.cc.ccsd_incore"
                )
                self._cc = ccsd_incore.CCSD(
                    self.fno_mf,
                    frozen=self.metadata.frozen_virtual.tolist(),
                    **self.cc_kwargs,
                )
            except (ImportError, OSError) as exc:
                raise GPUUnavailableError(
                    "GPU4PySCF CCSD could not be initialized on the available CUDA backend"
                ) from exc
        else:
            from pyscf import cc as pyscf_cc

            self._cc = pyscf_cc.CCSD(
                self.fno_mf, frozen=self.metadata.frozen_virtual.tolist(), **self.cc_kwargs
            )
        # Keep audit metadata attached to the solver users receive.
        # PySCF StreamObject warns when a public attribute is not declared in
        # ``_keys``.  Give this solver its own copied key set so attaching FNO
        # provenance is both quiet and does not mutate the backend class.
        audit_keys = {"fno_metadata", "fno_delta_mp2", "fno_backend"}
        self._cc._keys = set(getattr(self._cc, "_keys", ())) | audit_keys
        self._cc.fno_metadata = self.metadata
        self._cc.fno_delta_mp2 = self.metadata.delta_mp2
        self._cc.fno_backend = self.backend
        return self._cc

    def run(self, *args, **kwargs):
        """Run the selected backend and return this audited wrapper."""

        self.kernel(*args, **kwargs)
        return self

    def kernel(self, *args, **kwargs):
        """Delegate to the backend's PySCF-compatible kernel method."""

        return self.build().kernel(*args, **kwargs)

    @property
    def corrected_e_corr(self) -> float:
        """FNO-CCSD correlation energy with the additive Delta-MP2 correction."""

        solver = self.build()
        if getattr(solver, "e_corr", None) is None:
            raise RuntimeError("run FNOCCSD before requesting corrected_e_corr")
        return float(np.real(solver.e_corr)) + self.delta_mp2

    @property
    def corrected_e_tot(self) -> float:
        """FNO-CCSD total energy with the additive Delta-MP2 correction."""

        solver = self.build()
        if getattr(solver, "e_tot", None) is None:
            raise RuntimeError("run FNOCCSD before requesting corrected_e_tot")
        return float(np.real(solver.e_tot)) + self.delta_mp2

    @property
    def energy_metadata(self) -> dict[str, float | str]:
        """Return explicit raw and Delta-MP2-corrected energy semantics."""

        solver = self.build()
        if getattr(solver, "e_corr", None) is None or getattr(solver, "e_tot", None) is None:
            raise RuntimeError("run FNOCCSD before requesting energy metadata")
        raw_corr = float(np.real(solver.e_corr))
        raw_total = float(np.real(solver.e_tot))
        return {
            "definition": "FNO-CCSD + Delta-MP2",
            "raw_fno_ccsd_correlation_energy_eh": raw_corr,
            "raw_fno_ccsd_total_energy_eh": raw_total,
            "delta_mp2_eh": self.delta_mp2,
            "corrected_correlation_energy_eh": raw_corr + self.delta_mp2,
            "corrected_total_energy_eh": raw_total + self.delta_mp2,
        }

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.build(), name)


def fno_ccsd(mf, **kwargs) -> FNOCCSD:
    """Factory spelling for :class:`FNOCCSD`."""

    return FNOCCSD(mf, **kwargs)


def build_fno_ccsd(mf, **kwargs):
    """Construct and return the selected backend CCSD object immediately.

    The returned object carries ``fno_metadata``, ``fno_delta_mp2`` and
    ``fno_backend`` attributes.  For lazy access to the metadata use
    :class:`FNOCCSD` directly.
    """

    return FNOCCSD(mf, **kwargs).build()


__all__ = [
    "GPUUnavailableError",
    "FNOSemicanonicalizationError",
    "gpu_available",
    "FNOMetadata",
    "make_fno_metadata",
    "FNOCCSD",
    "fno_ccsd",
    "build_fno_ccsd",
]
