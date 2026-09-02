# AGENTS.md — project state and progress tracker

This file is the single source of truth for project state. Read it together
with `Project.md` before changing the roadmap. Mark work complete only after it
has been verified, keep at most one item in progress, and update the progress
log and artifact table when a milestone changes.

Box legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` skipped

## Current state

- **Project:** prompt-to-SynthAX FPGA instrument
- **Active milestone:** M1 — full Colab teacher/direct-model training
- **Last completed:** portable SynthAX parameter-to-audio hardware milestone
- **In progress:** full GPU teacher generation and direct-model training
- **Next action:** run the `balanced` or `quality` `train-live` workflow on
  Colab, review held-out/listening results, and freeze the accepted checkpoint.
- **Board blocker:** exact Cyclone IV part number and vendor pin map. Portable
  RTL, generic Quartus sources, and both transport options are complete.
- **Build/verify:** `make synthax`, then
  `ctag-repro/.venv/bin/python -m pytest -q ctag-repro/tests tests`.

## Progress tracker

### M1 — Paper and live-model software

- [x] Reproduce CTAG's released CLAP/SynthAX/LES inference path.
- [x] Provide latest-Colab bootstrap, notebooks, resumable artifacts, and tests.
- [x] Implement direct model, distillation data generation, evaluation, ONNX,
  and INT8 export paths.
- [~] Run the full GPU teacher/training profile and accept against held-out
  audio and prompt metrics. This is user-driven on Colab.

### M2 — Parameter-to-audio hardware

- [x] Freeze/hash the effective 78-value SynthAX render-time contract.
- [x] Implement and verify the fixed-point monophonic SynthAX voice.
- [x] Implement and verify compiled presets, volatile editing, note/preset
  controls, seven-segment output, and atomic external patch commit.
- [x] Implement and verify FIFO, PCM/CRC framing, JTAG UART master, physical
  UART fallback, host parser, and computer playback client.
- [x] Add deterministic ROMs, fixed-point/SynthAX comparisons, Icarus tests,
  CI, documentation, and a generic Cyclone IV Quartus project.
- [ ] Apply actual board device/pins and obtain Quartus timing/resource reports.

### M3 — Neural inference hardware

- [ ] Freeze the trained checkpoint and quantization scales.
- [ ] Inventory ONNX operations and generate bit-exact layer vectors.
- [ ] Implement INT8 layer controller, memories, requantization, and sigmoid.
- [ ] Connect its output to the existing 78-register atomic commit interface.

### M4 — Deployment and evaluation

- [ ] Measure JTAG/USB-UART throughput, FIFO occupancy, and end-to-end latency.
- [ ] Capture board audio and compare with fixed and SynthAX references.
- [ ] Demonstrate compiled prompt patches, live notes, and volatile editing.

## Key decisions

- Synthesizer: effective CTAG SynthAX `Voice`, 78 normalized parameters,
  monophonic for the first hardware closure.
- Contract: `synthax-voice-render-flat-78-v2`, unsigned Q0.16; v1 direct-model
  workspaces are invalid because they described Flax insertion order rather
  than JAX render-time order.
- Audio: 48 kHz mono, signed Q1.23 internally, framed PCM16 to the PC.
- Transport: Intel JTAG UART primary, 3 Mbaud USB-UART fallback.
- UI: four notes (60/64/67/72), eight compiled presets, edit switch, and four
  edit actions. Edits remain volatile.
- No audio input, codec, jack, ADC, or I2S in this milestone.
- Live DSP replaces SynthAX's non-causal peak normalization with fixed one-third
  headroom; documented fixed-point approximations are perceptually validated.

## Artifacts

| Area | Path | Status |
|---|---|---|
| Project/architecture | `Project.md`, `Architecture.md` | current |
| CTAG reproduction | `ctag-repro/` | tested; full training pending |
| Contract | `hardware/synthax-rtl-contract.json` | v2 frozen |
| Hardware guide | `hardware/README.md` | current |
| Synth RTL | `rtl/synthax/` | simulated |
| Transport RTL | `rtl/transport/` | simulated |
| Cyclone IV shell | `rtl/intel/`, `hardware/intel/` | generic; pins pending |
| Host playback | `host/` | parser tested; board pending |
| Fixed model/tools | `software/rtl_synth/` | tested against SynthAX |
| RTL tests | `tb/`, `tests/test_rtl_synth.py` | passing |

## Progress log

- 2026-05-31 — Created the earlier differentiable FM prototype and INT8 MAC PE.
- 2026-08-30 — Added the CTAG reference reproduction and Colab bootstrap.
- 2026-09-02 — Modernized CTAG for Colab Python 3.13/JAX/PyTorch and added the
  resumable direct prompt-to-patch training/export workflow.
- 2026-09-02 — Froze the corrected v2 render-time parameter order and completed
  the portable fixed-point SynthAX voice, interactive patch bank/UI, framed host
  audio transport, generic Cyclone IV shell, golden models, documentation, and
  regression/CI coverage. Tonal diagnostic correlation is 0.9754–0.9990.
- 2026-09-02 — Added direct local rendering of native CTAG `patch.yaml`
  artifacts through the FPGA fixed-point model and generated a verified
  48 kHz train-horn audition from the existing paper-budget search result.
