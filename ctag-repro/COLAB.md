# Run CTAG on Google Colab

Select **Runtime → Change runtime type → T4 GPU** before running these cells.
The paper dependency set requires Python 3.9, so the bootstrap creates an
isolated interpreter instead of modifying Colab's Python 3.13 runtime.

```python
!git clone https://github.com/nikhilanayak/Synthesizer.git
%cd Synthesizer/ctag-repro
!bash tools/colab_bootstrap.sh
```

Run a short end-to-end benchmark first:

```python
!time .venv/bin/ctag generate --profile smoke --prompt "train horn" \
    --population 10 --iterations 10 --device cuda --no-artifacts
```

Run one full paper-budget search after the smoke benchmark succeeds:

```python
!time .venv/bin/ctag reproduce --prompt "train horn" --device cuda
```

The full search evaluates 15,000 candidates and writes its WAV, patch, history,
configuration, and metadata under `runs/`. Generate multiple independent
choices with `--runs 10`; this takes ten times as long.

If Colab reports that no GPU is available, select a GPU runtime and rerun the
bootstrap. Verify the environment with:

```python
!.venv/bin/python -c "import jax, torch; print(jax.devices()); print(torch.cuda.is_available())"
```
