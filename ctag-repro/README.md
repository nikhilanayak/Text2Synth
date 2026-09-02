# CTAG reproduction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikhilanayak/Synthesizer/blob/main/ctag-repro/CTAG_Colab.ipynb)

This repository recreates **Creative Text-to-Audio Generation via Synthesizer
Programming** (Cherep, Singh, and Shand, ICML 2024). CTAG searches the 78
interpretable parameters of the SynthAX `Voice` synthesizer and uses LAION-CLAP
similarity to choose audio that expresses a text prompt.

The current milestone is deliberately ML-first. It provides the reference
floating-point pipeline, clean component boundaries, deterministic artifacts,
tests, and performance measurements. Quantization and SystemVerilog will follow
only after this baseline is verified.

## Method reproduced

For every prompt, CTAG:

1. Computes its 512-dimensional CLAP text embedding once.
2. Asks Learned Evolution Strategies (LES) for a population of normalized
   78-parameter Voice patches.
3. Renders every patch as two seconds of 48 kHz audio with SynthAX.
4. Computes CLAP audio embeddings and minimizes negative text/audio cosine
   similarity.
5. Repeats for 300 iterations and writes the best audio and inspectable patch.

The `paper` profile uses population 50, 300 iterations, 48 kHz audio, 480 Hz
control signals, and seed 42. The `smoke` profile changes only the search budget
to population 2 and 2 iterations.

## Installation

For a GPU run, use the badge above or follow [COLAB.md](COLAB.md). The modern
path runs directly on Colab's Python 3.13, JAX 0.11 CUDA plugin, and PyTorch
2.11 CUDA runtime. It preserves those accelerator wheels while installing a
compatible Flax 0.12.9 and current Evosax/SynthAX adapters.

### Legacy paper environment

The authors released against Python 3.9 and older JAX/PyTorch packages. Start
with a clean environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade 'pip<26' setuptools wheel
python -m pip install -e '.[paper,dev]'
ctag setup --download --strict
```

`ctag setup` downloads `630k-audioset-best.pt` from the URL published in the
authors' repository. It is a large external artifact and is ignored by git.
Apple Silicon uses JAX/JAXlib 0.4.18 because the released 0.4.14 version has no
macOS arm64 wheel; all other paper dependencies remain pinned.

## Run

Fast integration run:

```bash
ctag generate --profile smoke --prompt "train horn"
```

Full released search configuration:

```bash
ctag reproduce --prompt "train horn" --device cpu
```

The latest-runtime Google Colab setup and its audited package matrix are
documented in [COLAB.md](COLAB.md).

The result of a single evolutionary run can vary substantially. The paper's
listening study drew from 10 independently generated sounds per prompt. Generate
the same number, then audition the `run-000` through `run-009` outputs:

```bash
ctag reproduce --prompt "train horn" --runs 10 --device cpu
```

Prompt files and overrides are also supported:

```bash
ctag profile --profile smoke --prompts-file data/regression-prompts.txt
ctag generate --config configs/smoke.yaml --prompt "spray" --iterations 5
```

Every run directory contains `best.wav`, `patch.yaml`, `history.csv`, the fully
resolved `config.yaml`, and `metadata.json` with checkpoint hash, upstream
revision, dependency versions, platform, timings, and best fitness.

Compare two completed runs:

```bash
ctag compare --reference path/to/reference/run-000 \
             --candidate path/to/candidate/run-000
```

## Tests

Core tests do not load or download CLAP:

```bash
pytest
```

The end-to-end smoke run is the integration test for the real JAX/PyTorch
backend. A full paper run is computationally expensive: it synthesizes and
embeds 15,000 two-second candidates for each prompt. On the validated M-series
CPU environment, one prompt took about 15.2 minutes; 96.6% of measured component
time was CLAP audio encoding.

## Provenance

- Paper: <https://arxiv.org/abs/2406.00294>
- Authors' code: <https://github.com/PapayaResearch/ctag>
- Pinned upstream commit: `fc207b271a9761a6b001e3d028e777d608c4e91f`
- SynthAX: <https://github.com/PapayaResearch/SynthAX>, version `0.2.1`
- LAION-CLAP: <https://github.com/LAION-AI/CLAP>, checkpoint
  `630k-audioset-best.pt`

The implementation is MIT licensed. Third-party packages and model files retain
their respective licenses.
