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

"""Two-level THC/RR RCCSD drivers for validation and direct-CD equations."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from gpu4pyscf.cc.device_runtime import nvtx_range
from gpu4pyscf.cc.full_space_residual import (
    FullSpaceResidualDiagnostic,
    diagnose_reconstructed_full_space_residual,
)
from gpu4pyscf.cc.lowrank import RRDoubles, THCDoubles, t2_from_pair_matrix
from gpu4pyscf.cc.rrccsd import RRCCSD, _resident_array, _synchronize
from gpu4pyscf.cc.thc_engine import THCRRCCSDIterationEngine
from gpu4pyscf.cc.thc_factorization import (
    THCProjectorFactors,
    fit_weighted_thc_projector,
)


def _array_module(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def _optional_positive_integer(name: str, value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer or None")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


class THCRRCCSD(RRCCSD):
    """RCCSD with a fixed weighted amplitude-THC projector fit.

    ``eri_backend="canonical"`` preserves the earlier deterministic
    symmetric-core validation surrogate.

    ``eri_backend="cd"`` currently admits only the analytic full-pair THC
    endpoint. It constructs the complement from the six exact RR component
    kernels with ``-R123_RR`` as the initial offset, while exact-RR and THC
    Algorithms 1--3 share each T1-transformed Cholesky block. It then applies
    the endpoint as ``complement + R123_THC``. Lower THC ranks remain fail
    closed until the water2/water4 residual and accuracy gates pass. The normal
    path never constructs dense doubles and remains performance ineligible.
    """

    implementation_stage = "dense-two-level-validation"
    _keys = set(RRCCSD._keys) | {
        "thc_fit_tol",
        "thc_rank",
        "thc_initial_rank",
        "thc_max_rank",
        "thc_rank_growth",
        "thc_orthogonality_cutoff",
        "thc_orthogonality_tolerance",
        "thc_max_iterations",
        "thc_als_convergence_tolerance",
        "thc_ridge",
        "thc_seed",
        "thc_replacement_validation_atol",
        "thc_replacement_validation_rtol",
        "thc_doubles",
        "thc_projector_factors",
        "thc_replacement_application_count",
        "full_space_diagnostic_metadata",
    }

    def __init__(
        self,
        mf: Any,
        *args: Any,
        thc_fit_tol: Optional[float] = None,
        thc_rank: Optional[int] = None,
        thc_initial_rank: Optional[int] = None,
        thc_max_rank: Optional[int] = None,
        thc_rank_growth: float = 1.5,
        thc_orthogonality_cutoff: float = 1e-12,
        thc_orthogonality_tolerance: float = 1e-10,
        thc_max_iterations: int = 500,
        thc_als_convergence_tolerance: float = 1e-10,
        thc_ridge: float = 1e-12,
        thc_seed: int = 0,
        thc_replacement_validation_atol: Optional[float] = None,
        thc_replacement_validation_rtol: Optional[float] = None,
        eri_backend: str = "cd",
        **kwargs: Any,
    ) -> None:
        if thc_fit_tol is None:
            raise TypeError("thc_fit_tol must be supplied explicitly")
        rr_ring_kernel = str(kwargs.get("rr_ring_kernel", "reference")).lower()
        if rr_ring_kernel not in {"reference", "gemm"}:
            raise ValueError("rr_ring_kernel must be 'reference' or 'gemm'")
        if rr_ring_kernel == "gemm":
            raise NotImplementedError(
                "THCRRCCSD does not consume the RR ring GEMM selector; "
                "use RRCCSD with rr_ring_kernel='gemm'"
            )
        self.thc_fit_tol = _finite_float("thc_fit_tol", thc_fit_tol)
        self.thc_rank = _optional_positive_integer("thc_rank", thc_rank)
        self.thc_initial_rank = _optional_positive_integer(
            "thc_initial_rank", thc_initial_rank
        )
        self.thc_max_rank = _optional_positive_integer(
            "thc_max_rank", thc_max_rank
        )
        if self.thc_rank is not None and self.thc_initial_rank is not None:
            raise ValueError(
                "thc_initial_rank applies only to adaptive rank selection"
            )
        if (
            self.thc_rank is not None
            and self.thc_max_rank is not None
            and self.thc_rank > self.thc_max_rank
        ):
            raise ValueError("thc_rank cannot exceed thc_max_rank")
        self.thc_rank_growth = _finite_float(
            "thc_rank_growth", thc_rank_growth, positive=True
        )
        if self.thc_rank_growth <= 1.0:
            raise ValueError("thc_rank_growth must be greater than one")
        self.thc_orthogonality_cutoff = _finite_float(
            "thc_orthogonality_cutoff",
            thc_orthogonality_cutoff,
            positive=True,
        )
        self.thc_orthogonality_tolerance = _finite_float(
            "thc_orthogonality_tolerance", thc_orthogonality_tolerance
        )
        self.thc_max_iterations = _optional_positive_integer(
            "thc_max_iterations", thc_max_iterations
        )
        self.thc_als_convergence_tolerance = _finite_float(
            "thc_als_convergence_tolerance", thc_als_convergence_tolerance
        )
        self.thc_ridge = _finite_float("thc_ridge", thc_ridge)
        if isinstance(thc_seed, (bool, np.bool_)) or not isinstance(
            thc_seed, (int, np.integer)
        ):
            raise TypeError("thc_seed must be an integer")
        self.thc_seed = int(thc_seed)
        backend = str(eri_backend).lower()
        if backend == "cd" and self.thc_rank is None:
            raise NotImplementedError(
                "direct-CD THCRRCCSD currently requires an explicit fixed "
                "thc_rank for the analytic full-pair validation endpoint; "
                "adaptive or inexact ranks are disabled until a shared "
                "RR/paper residual decomposition exists"
            )
        if backend == "cd" and (
            thc_replacement_validation_atol is None
            or thc_replacement_validation_rtol is None
        ):
            raise TypeError(
                "thc_replacement_validation_atol and "
                "thc_replacement_validation_rtol must both be supplied "
                "explicitly for the direct-CD endpoint audit"
            )
        self.thc_replacement_validation_atol = (
            None
            if thc_replacement_validation_atol is None
            else _finite_float(
                "thc_replacement_validation_atol",
                thc_replacement_validation_atol,
            )
        )
        self.thc_replacement_validation_rtol = (
            None
            if thc_replacement_validation_rtol is None
            else _finite_float(
                "thc_replacement_validation_rtol",
                thc_replacement_validation_rtol,
            )
        )
        self.thc_doubles: Optional[THCDoubles] = None
        self.thc_projector_factors: Optional[THCProjectorFactors] = None
        self.thc_replacement_application_count = 0
        self.full_space_diagnostic_metadata: Optional[dict[str, Any]] = None
        super().__init__(mf, *args, eri_backend=eri_backend, **kwargs)

    def _clear_cd_runtime(self) -> None:
        super()._clear_cd_runtime()
        self.thc_projector_factors = None
        self.thc_doubles = None
        self.thc_replacement_application_count = 0
        self.full_space_diagnostic_metadata = None

    def method_metadata(self) -> Dict[str, Any]:
        metadata = super().method_metadata()
        metadata.update({
            "rr_ring_kernel_not_applicable_to_thc_engine": True,
            "thc_fit_tol": self.thc_fit_tol,
            "thc_rank": self.thc_rank,
            "thc_initial_rank": self.thc_initial_rank,
            "thc_max_rank": self.thc_max_rank,
            "thc_rank_growth": self.thc_rank_growth,
            "thc_orthogonality_cutoff": self.thc_orthogonality_cutoff,
            "thc_orthogonality_tolerance": self.thc_orthogonality_tolerance,
            "thc_max_iterations": self.thc_max_iterations,
            "thc_als_convergence_tolerance": self.thc_als_convergence_tolerance,
            "thc_ridge": self.thc_ridge,
            "thc_seed": self.thc_seed,
            "thc_replacement_validation_atol": (
                self.thc_replacement_validation_atol
            ),
            "thc_replacement_validation_rtol": (
                self.thc_replacement_validation_rtol
            ),
            "performance_eligible": False,
        })
        if self.eri_backend == "cd":
            metadata.update({
                "implementation_stage": (
                    "direct-cd-r123-complement-full-pair-thc-endpoint"
                ),
                "thc_factorization": "weighted-projector-cp-als",
                "paper_factor_construction": True,
                "paper_exact_thc": False,
                "complete_equation_graph": True,
                "canonical_equation": False,
                "equation": "rr-r123-complement-plus-thc-r123",
                "validated_doubles_groups": {
                    "paper": "Hohenstein-2022",
                    "equations": [28, 29, 31],
                    "algorithms": [1, 2, 3],
                },
                "replacement_formula": (
                    "rr_complement + backproject(thc_algorithms_1_3)"
                ),
                "replacement_applied": bool(
                    self.thc_replacement_application_count > 0
                ),
                "replacement_application_count": int(
                    self.thc_replacement_application_count
                ),
                "inexact_thc_enabled": False,
                "inexact_thc_blocker": (
                    "water2/water4 residual and accuracy gates have not passed"
                ),
                "dense_t2_in_normal_cd_path": False,
                "quartic_residual": (
                    "rr-r123-complement-plus-thc-r123-full-pair"
                ),
                "iteration_state": "RRDoubles",
                "returned_doubles": "RRDoubles",
                "reported_energy": "rr-state-rccsd-energy-functional",
                "projected_residual": "complete-rr-equation",
                "performance_limitations": [
                    "inexact amplitude THC is fail-closed",
                    (
                        "all six exact RR component kernels and the exact "
                        "R123 subtraction oracle are still evaluated"
                    ),
                    (
                        "THC engine does not consume rr_ring_kernel; the "
                        "reference RR ring contraction stays in the complement"
                    ),
                ],
            })
            if self.thc_projector_factors is not None:
                metadata["thc_projector_factors"] = (
                    self.thc_projector_factors.metadata()
                )
            if self.full_space_diagnostic_metadata is not None:
                metadata["full_space_residual_diagnostic"] = dict(
                    self.full_space_diagnostic_metadata
                )
        else:
            metadata.update({
                "thc_factorization": "symmetric-core-eigh-validation-surrogate",
                "paper_exact_thc": False,
                "quartic_residual": False,
            })
            if self.thc_doubles is not None:
                metadata["thc_doubles"] = self.thc_doubles.metadata(
                    transfer_counter=self.run_metrics.transfers
                )
        return metadata

    def _compress_doubles(self, t2: Any):
        if self.rr_projector is None:
            raise RuntimeError("RR projector has not been initialized")
        rr = RRDoubles.from_t2(t2, self.rr_projector)
        if self.eri_backend == "cd":
            return rr
        thc = THCDoubles.from_rr(
            rr, self.thc_fit_tol, max_rank=self.thc_max_rank
        )
        if thc.thc_rank == 0:
            raise ValueError(
                "thc_fit_tol removed the complete RR doubles core; "
                "use a tighter tolerance"
            )
        return thc

    def _store_doubles(self, doubles: Any) -> None:
        if self.eri_backend == "cd":
            if not isinstance(doubles, RRDoubles):
                raise TypeError("direct-CD THC lifecycle stores RRDoubles state")
            self.doubles = doubles
            return
        if not isinstance(doubles, THCDoubles):
            raise TypeError("canonical THC validation stores THCDoubles")
        self.doubles = doubles
        self.thc_doubles = doubles

    def _fit_thc_projector(self) -> THCProjectorFactors:
        if self.rr_projector is None or self._cd_integrals is None:
            raise RuntimeError("RR projector and direct-CD factors are required")
        if self.thc_projector_factors is not None:
            return self.thc_projector_factors
        projector = self.rr_projector
        nocc = self._cd_integrals.nocc
        nvir = self._cd_integrals.nvir
        dimension = nocc * nvir
        if self.thc_rank != dimension:
            raise NotImplementedError(
                "inexact direct-CD amplitude THC is fail-closed: thc_rank "
                f"must equal the full pair dimension {dimension} until "
                "water2/water4 residual and accuracy gates pass"
            )
        for name, rank in (
            ("thc_rank", self.thc_rank),
            ("thc_initial_rank", self.thc_initial_rank),
            ("thc_max_rank", self.thc_max_rank),
        ):
            if rank is not None and rank > dimension:
                raise ValueError(f"{name} cannot exceed pair dimension {dimension}")
        common = {
            "fit_tolerance": self.thc_fit_tol,
            "orthogonality_cutoff": self.thc_orthogonality_cutoff,
            "max_iterations": self.thc_max_iterations,
            "als_convergence_tolerance": self.thc_als_convergence_tolerance,
            "ridge": self.thc_ridge,
            "seed": self.thc_seed,
            "orthogonality_tolerance": self.thc_orthogonality_tolerance,
            "transfer_counter": self.run_metrics.transfers,
        }
        with nvtx_range("thc-rr/weighted-projector-fit"):
            with self.run_metrics.phase("thc_projector_fit"):
                factors = fit_weighted_thc_projector(
                    projector,
                    nocc,
                    nvir,
                    thc_rank=self.thc_rank,
                    **common,
                )
                _synchronize(factors.tau)
        self.thc_projector_factors = factors
        return factors

    def _validate_cd_lifecycle(self) -> None:
        """Apply RR lifecycle restrictions while allowing our owned hooks."""

        if bool(getattr(self, "cc2", False)):
            raise NotImplementedError("CC2 is not implemented by the THC/RR engine")
        if getattr(self, "callback", None) is not None:
            raise NotImplementedError(
                "per-iteration callbacks are not implemented by the THC/RR engine"
            )
        if float(getattr(self, "iterative_damping", 1.0)) != 1.0:
            raise NotImplementedError(
                "iterative damping is not implemented by the THC/RR engine"
            )
        if getattr(self, "diis_file", None) is not None:
            raise NotImplementedError(
                "file-backed DIIS is unsupported; compressed DIIS stays resident"
            )
        if not isinstance(getattr(self, "diis", True), (bool, np.bool_)):
            raise NotImplementedError(
                "an external DIIS object cannot consume compressed amplitudes"
            )

    def _ccsd_cd(self, t1: Any = None, t2: Any = None, eris: Any = None):
        self.thc_replacement_application_count = 0
        self._validate_cd_lifecycle()
        if hasattr(self, "check_sanity"):
            self.check_sanity()
        self.dump_flags()
        self.e_hf = self.get_e_hf()
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        if eris is not self._cd_integrals:
            raise ValueError(
                "the direct-CD kernel requires factors built by this solver"
            )

        if self.rr_projector is None or self._mp2_operator is None:
            _emp2, default_t1, default_t2 = self.init_amps(eris)
        else:
            default_t1 = default_t2 = None
        if t1 is None:
            if default_t1 is None:
                _emp2, default_t1, rebuilt_t2 = self._cd_initial_amplitudes(
                    eris
                )
                if t2 is None:
                    default_t2 = rebuilt_t2
            t1 = default_t1
        else:
            t1 = _resident_array(
                t1,
                eris.xp,
                self.run_metrics.transfers,
                operation="thc_rr_user_initial_t1",
            )
        if t2 is None:
            if default_t2 is None:
                _emp2, rebuilt_t1, default_t2 = self._cd_initial_amplitudes(
                    eris
                )
                if t1 is None:
                    t1 = rebuilt_t1
            t2 = default_t2
        if not isinstance(t2, RRDoubles):
            raise TypeError(
                "the direct-CD THC kernel accepts RRDoubles state; use "
                "init_amps() to obtain the lifecycle-compatible guess"
            )
        if t2.projector is not self.rr_projector:
            raise ValueError("initial RRDoubles use a different RR projector")

        factors = self._fit_thc_projector()
        with nvtx_range("thc-rr/engine-setup"):
            with self.run_metrics.phase("thc_rr_engine_setup"):
                engine = THCRRCCSDIterationEngine(
                    eris,
                    self.rr_projector,
                    factors,
                    self._rr_fock,
                    self._rr_orbital_energies,
                    level_shift=float(self.level_shift),
                    auxiliary_block_size=min(
                        self.rr_auxiliary_block_size, eris.naux
                    ),
                    virtual_block_size=min(
                        self.rr_virtual_block_size, eris.nvir
                    ),
                    replacement_validation_atol=(
                        self.thc_replacement_validation_atol
                    ),
                    replacement_validation_rtol=(
                        self.thc_replacement_validation_rtol
                    ),
                    metrics=self.run_metrics,
                )
                _synchronize(eris.factors)
        self._rr_engine = engine
        diis_enabled = bool(self.diis)
        diis_start_cycle = (
            int(self.diis_start_cycle)
            if diis_enabled else int(self.max_cycle) + 1
        )
        with nvtx_range("thc-rr/ccsd-iterations"):
            with self.run_metrics.phase("ccsd_iterations"):
                _synchronize(eris.factors)
                result = engine.kernel(
                    t1,
                    t2,
                    max_cycle=int(self.max_cycle),
                    conv_tol=float(self.conv_tol),
                    conv_tol_normt=float(self.conv_tol_normt),
                    diis_space=int(self.diis_space),
                    diis_start_cycle=diis_start_cycle,
                )
                _synchronize(result.doubles.core)
        self.thc_replacement_application_count = int(
            engine.replacement_application_count
        )

        self.converged = bool(result.converged)
        self.cycles = int(result.cycles)
        self.e_corr = float(result.energy)
        self.t1 = result.t1
        self.t2 = result.doubles
        self.projected_equation_residual = float(result.residual_norm)
        self.projected_jacobi_update_norm = float(result.update_norm)
        self.full_space_residual = None
        self.full_space_jacobi_update_norm = None
        self.full_space_diagnostic_metadata = None
        self.representation_residual = None
        self.rr_history = result.history
        self._store_doubles(result.doubles)
        self.run_metrics.metadata = self.method_metadata()
        self._finalize()
        return self.e_corr, self.t1, result.doubles

    def run_full_space_residual_diagnostic(
        self,
    ) -> FullSpaceResidualDiagnostic:
        """Explicitly reconstruct and check the complete full-space equation.

        This post-convergence diagnostic is never invoked by ``kernel()`` and
        is recorded in its own phase. It materializes an ``(OV,OV)`` pair
        matrix once, so callers should run it outside normal performance
        timing.
        """

        if self.eri_backend != "cd":
            raise NotImplementedError(
                "the explicit THC full-space diagnostic is for direct CD"
            )
        if getattr(self, "converged", None) is not True:
            raise RuntimeError(
                "run a converged THCRRCCSD.kernel() before the full-space "
                "diagnostic"
            )
        if (
            self._cd_integrals is None
            or self._rr_fock is None
            or self._rr_orbital_energies is None
            or self.t1 is None
            or not isinstance(self.doubles, RRDoubles)
        ):
            raise RuntimeError(
                "run THCRRCCSD.kernel() before the full-space diagnostic"
            )
        with nvtx_range("thc-rr/post-convergence-full-space-residual"):
            with self.run_metrics.phase(
                "post_convergence_full_space_residual_diagnostic"
            ):
                diagnostic = diagnose_reconstructed_full_space_residual(
                    self._cd_integrals,
                    self.t1,
                    self.doubles,
                    self._rr_fock,
                    self._rr_orbital_energies,
                    level_shift=float(self.level_shift),
                    auxiliary_block_size=min(
                        self.rr_auxiliary_block_size,
                        self._cd_integrals.naux,
                    ),
                    virtual_block_size=min(
                        self.rr_virtual_block_size,
                        self._cd_integrals.nvir,
                    ),
                    transfer_counter=self.run_metrics.transfers,
                )
                factors = self.thc_projector_factors
                if factors is None:
                    raise RuntimeError("THC projector factors are unavailable")
                rr_pair = self.doubles.reconstruct_pair_matrix()
                thc_pair_factor = factors.khatri_rao()
                thc_core = factors.amplitude_core(self.doubles.core)
                thc_pair = thc_pair_factor @ thc_core @ thc_pair_factor.T
                xp = _array_module(rr_pair)
                difference_norm = xp.linalg.norm(thc_pair - rr_pair)
                reference_norm = xp.linalg.norm(rr_pair)
                if xp is not np:
                    self.run_metrics.transfers.record_d2h(
                        int(difference_norm.nbytes + reference_norm.nbytes),
                        2,
                        operation="thc_full_space_representation_residual",
                    )
                denominator = float(reference_norm.item())
                numerator = float(difference_norm.item())
                representation_residual = (
                    numerator
                    if denominator == 0.0
                    else numerator / denominator
                )
                _synchronize(rr_pair)
        self.full_space_residual = float(diagnostic.residual_norm)
        self.representation_residual = float(representation_residual)
        self._dense_t2_reconstructed = True
        metadata = diagnostic.metadata()
        metadata.update({
            "kind": "equation-residual",
            "space": "full-active-pair",
            "is_full_space": True,
        })
        metadata["thc_representation_residual"] = (
            self.representation_residual
        )
        metadata["normal_kernel_invokes_diagnostic"] = False
        self.full_space_diagnostic_metadata = metadata
        self.run_metrics.metadata = self.method_metadata()
        return diagnostic

    def reconstruct_t2(self):
        if self.eri_backend == "cd":
            return super().reconstruct_t2()
        if self.thc_doubles is None:
            raise RuntimeError("THCRRCCSD has not produced doubles amplitudes")
        self._dense_t2_reconstructed = True
        return self.thc_doubles.reconstruct_t2()

    def reconstruct_thc_t2(self):
        """Explicit diagnostic image used by the selected THC diagrams."""

        if self.eri_backend != "cd":
            return self.reconstruct_t2()
        if self.doubles is None or self.thc_projector_factors is None:
            raise RuntimeError("direct-CD THCRRCCSD has not produced amplitudes")
        if not isinstance(self.doubles, RRDoubles):
            raise TypeError("direct-CD THC lifecycle lost its RR state")
        factors = self.thc_projector_factors
        pair_factor = factors.khatri_rao()
        thc_core = factors.amplitude_core(self.doubles.core)
        pair_matrix = pair_factor @ thc_core @ pair_factor.T
        self._dense_t2_reconstructed = True
        return t2_from_pair_matrix(
            pair_matrix, self.doubles.nocc, self.doubles.nvir
        )


__all__ = ["THCRRCCSD", "THCDoubles"]
