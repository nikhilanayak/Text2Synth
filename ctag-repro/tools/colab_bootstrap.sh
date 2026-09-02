#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 - <<'PY'
import sys
from importlib import metadata

from packaging.specifiers import SpecifierSet
from packaging.version import Version

expected = {
    "numpy": "==2.1.3",
    "jax": "==0.11.1",
    "jaxlib": "==0.11.1",
    "torch": "==2.11.0",
    "torchvision": "==0.26.0",
    "transformers": "==5.16.1",
    "numba": "==0.61.2",
}
issues = []
if sys.version_info[:2] != (3, 13):
    issues.append(f"Python {sys.version.split()[0]} (expected the latest Colab 3.13 runtime)")
for package, requirement in expected.items():
    try:
        actual = metadata.version(package)
    except metadata.PackageNotFoundError:
        issues.append(f"{package} is missing")
        continue
    if Version(actual) not in SpecifierSet(requirement):
        issues.append(f"{package} {actual} does not satisfy {requirement}")
if issues:
    print("This is not the supported latest Colab runtime:", file=sys.stderr)
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)
    print("Use Runtime > Disconnect and delete runtime, reconnect, and rerun.", file=sys.stderr)
    raise SystemExit(1)
print("Latest Colab numerical and accelerator packages detected.")
PY

# Keep Colab's CUDA-enabled JAX and PyTorch wheels. Only upgrade compatible
# pure-Python layers and install CTAG's missing audio/search dependencies.
python3 -m pip install --quiet --upgrade \
  --no-build-isolation \
  --constraint constraints-colab.txt \
  --editable '.[colab]'

# LAION-CLAP 1.1.7 works with NumPy 2, but its stale package metadata still
# declares numpy<2. Installing without dependency resolution prevents pip from
# replacing Colab's NumPy/JAX stack. All of its runtime dependencies are in the
# colab extra above or already supplied by Colab.
python3 -m pip install --quiet --no-deps 'laion-clap==1.1.7'

ctag doctor --require-gpu
ctag setup --download --strict

echo
echo "CTAG is ready. Try:"
echo "ctag generate --profile smoke --prompt 'train horn' --device cuda"
