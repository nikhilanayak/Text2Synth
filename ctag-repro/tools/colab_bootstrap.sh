#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python3 -m pip install --quiet uv
uv python install 3.9
uv venv --python 3.9 --seed .venv
uv pip install --python .venv/bin/python -e '.[paper]'

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected; installing the paper-era CUDA 12 JAX wheel."
  .venv/bin/python -m pip install --quiet --force-reinstall --no-deps \
    'jax==0.4.14' \
    'jaxlib @ https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.14+cuda12.cudnn89-cp39-cp39-manylinux2014_x86_64.whl'
else
  echo "No NVIDIA GPU detected; retaining the CPU JAX wheel."
fi

.venv/bin/ctag setup --download --strict

echo
echo "CTAG is ready. Try:"
echo ".venv/bin/ctag generate --profile smoke --prompt 'train horn' --device cuda"
