"""Permit the NumPy reference tests to run on a CPU-only developer host."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def _load(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


if importlib.util.find_spec("cupy") is None:
    cc_root = Path(__file__).parents[1]
    package_roots = {
        "gpu4pyscf": cc_root.parent,
        "gpu4pyscf.cc": cc_root,
    }
    for package_name, package_root in package_roots.items():
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_root)]
            sys.modules[package_name] = package
    for module_name in (
        "lowrank",
        "device_runtime",
        "compressed_diis",
        "integrals",
        "direct_cd",
        "rr_projector",
        "rr_residual",
        "thc_factorization",
        "thc_residual",
        "rr_engine",
        "thc_engine",
    ):
        _load(f"gpu4pyscf.cc.{module_name}", cc_root / f"{module_name}.py")
