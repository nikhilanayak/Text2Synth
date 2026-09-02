# Run CTAG on the latest Google Colab runtime

[Open the runnable notebook in Colab](https://colab.research.google.com/github/nikhilanayak/Synthesizer/blob/main/ctag-repro/CTAG_Colab.ipynb), then select
**Runtime → Change runtime type → T4 GPU**.

To train the one-pass prompt-to-parameter model instead, open
[DirectPatch_Train.ipynb](https://colab.research.google.com/github/nikhilanayak/Synthesizer/blob/main/ctag-repro/DirectPatch_Train.ipynb).
That notebook mounts Drive and resumes all data, optimizer, evaluation, and
export stages automatically.

This path uses Colab's native Python 3.13 and CUDA wheels. It does not create a
second Python environment or replace JAX, JAXlib, PyTorch, torchvision, or the
CUDA plugins. The supported runtime matrix is recorded in
`constraints-colab.txt` and checked before installation.

```python
!git clone https://github.com/nikhilanayak/Synthesizer.git
%cd Synthesizer/ctag-repro
!bash tools/colab_bootstrap.sh
```

The bootstrap upgrades Colab's incompatible Flax 0.11.2 to 0.12.9, installs
the current Evosax/SynthAX adapters, validates both GPU frameworks, and
downloads the SHA-256-verified 1.86 GB CLAP checkpoint. LAION-CLAP's stale
`numpy<2` package declaration is bypassed without downgrading NumPy; its actual
inference path is validated against NumPy 2.1.3.

Run a short end-to-end benchmark first:

```python
!time ctag generate --profile smoke --prompt "train horn" \
    --population 10 --iterations 10 --device cuda --no-artifacts
```

Run one full paper-budget search after the smoke benchmark succeeds:

```python
!time ctag reproduce --prompt "train horn" --device cuda
```

The full search evaluates 15,000 candidates and writes its WAV, patch, history,
configuration, and metadata under `runs/`. Generate multiple independent
choices with `--runs 10`; this takes ten times as long.

Inspect the complete version and accelerator report at any time:

```python
!ctag doctor --require-gpu
```

If the preflight says the runtime is unsupported, choose **Runtime → Disconnect
and delete runtime**, reconnect to the default latest runtime with a GPU, and
rerun the bootstrap. Do not manually install a different JAX or PyTorch wheel.
