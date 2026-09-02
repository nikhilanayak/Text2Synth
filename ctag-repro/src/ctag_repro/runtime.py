"""Runtime compatibility checks for the current Google Colab image."""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata
from typing import Any, Dict, Mapping

from packaging.specifiers import SpecifierSet
from packaging.version import Version


# GoogleColab/backend-info main GPU manifest, captured 2026-09-02. Flax is the
# one intentional override: Colab's 0.11.2 uses an API removed by JAX 0.11.
COLAB_RUNTIME_SPECS = {
    "python": "==3.13.*",
    "numpy": "==2.1.3",
    "jax": "==0.11.1",
    "jaxlib": "==0.11.1",
    "flax": "==0.12.9",
    "evosax": "==0.3.1",
    "synthax": "==0.2.2",
    "laion-clap": "==1.1.7",
    "torch": "==2.11.0",
    "torchvision": "==0.26.0",
    "transformers": "==5.16.1",
    "numba": "==0.61.2",
    "scipy": "==1.16.3",
    "pandas": "==2.2.3",
}


def installed_versions() -> Dict[str, str]:
    """Return all packages that affect CTAG's numerical execution."""

    versions: Dict[str, str] = {"python": platform.python_version()}
    for package in COLAB_RUNTIME_SPECS:
        if package == "python":
            continue
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def version_issues(versions: Mapping[str, str]) -> list[str]:
    """Describe missing or out-of-matrix package versions."""

    issues: list[str] = []
    for package, requirement in COLAB_RUNTIME_SPECS.items():
        actual = versions.get(package, "missing")
        if actual == "missing":
            issues.append(f"{package} is missing (expected {requirement})")
            continue
        try:
            matches = Version(actual) in SpecifierSet(requirement)
        except ValueError:
            matches = False
        if not matches:
            issues.append(f"{package} {actual} does not satisfy {requirement}")
    return issues


def runtime_report(require_gpu: bool = False) -> Dict[str, Any]:
    """Inspect package and accelerator compatibility without changing state."""

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    versions = installed_versions()
    issues = version_issues(versions)
    jax_platforms: list[str] = []
    torch_cuda_available = False
    torch_cuda_version = None
    torch_device = None

    try:
        import jax

        jax_platforms = sorted({device.platform for device in jax.devices()})
    except Exception as exc:  # pragma: no cover - depends on host drivers
        issues.append(f"JAX device discovery failed: {type(exc).__name__}: {exc}")

    try:
        import torch

        torch_cuda_available = bool(torch.cuda.is_available())
        torch_cuda_version = torch.version.cuda
        if torch_cuda_available:
            torch_device = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - depends on host drivers
        issues.append(f"PyTorch device discovery failed: {type(exc).__name__}: {exc}")

    jax_gpu_available = "gpu" in jax_platforms or "cuda" in jax_platforms
    if require_gpu and not jax_gpu_available:
        issues.append("JAX cannot see a GPU")
    if require_gpu and not torch_cuda_available:
        issues.append("PyTorch cannot see a CUDA GPU")

    return {
        "compatible": not issues,
        "versions": versions,
        "expected": dict(COLAB_RUNTIME_SPECS),
        "accelerators": {
            "jax_platforms": jax_platforms,
            "torch_cuda_available": torch_cuda_available,
            "torch_cuda_version": torch_cuda_version,
            "torch_device": torch_device,
            "shared_gpu_memory_safe": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            )
            == "false",
        },
        "issues": issues,
        "require_gpu": require_gpu,
        "sys_executable": sys.executable,
    }
