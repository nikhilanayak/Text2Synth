# AGENTS.md — Project State & Progress Tracker

This file is the **single source of truth for project state**. Any agent working
in this repo MUST read this file first, act according to it, and keep it updated.
It mirrors the checklist in `Project.md` (Section 8) and records exactly where the
project currently stands.

---

## How agents must use this file

1. **On start:** Read this entire file, then read `Project.md`. The "Current
   State" section below tells you what is done and what to do next.
2. **Before working:** Confirm the task matches "Next Action". If the user asks
   for something else, do it, then reconcile this file afterward.
3. **After completing any checklist item, you MUST:**
   - Check the box `[ ]` -> `[x]` in **both** this file's Progress Tracker and
     in `Project.md` Section 8.
   - Update **Current State** (Active Milestone, Last Completed, Next Action).
   - Append a dated line to the **Progress Log**.
   - Update **Artifacts** if you created/changed key files.
4. **Never mark an item complete unless the work is actually done and verified**
   (e.g. tests pass, a script runs). Intent is not completion.
5. **One item in progress at a time.** Mark it `[~]` (in progress) while working.

**Box legend:** `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` skipped

---

## Current State

- **Project:** Embedded Neural FM Synthesizer (CLAP-to-FM)
- **Active Milestone:** M1 — Semantic Model (Python)
- **Last Completed:** 1.2 Training Pipeline (ESC-50 + SLERP + W&B)
- **In Progress:** 1.3 Phase 1 Training (User-driven)
- **Next Action:** Run `python3 software/train_fm.py` to train the mapper; begin RTL implementation of the Hybrid Synth.
- **Overall Progress:** 4 / 20 core items complete
  (M1: 2/5 · M2: 0/4 · M3: 1/4 · M4: 0/3 · M5: 0/4)
- **Key Decisions Pending:** exact FM parameter list (aiming for 52 params: 48 FM + 4 Noise).
- **How to build/run RTL:** `make` or `make mac_pe`. Sim = Icarus Verilog.
- **Reference implementation:** `ctag-repro/` contains the verified ICML 2024 CTAG reproduction and Colab bootstrap.

---

## Progress Tracker

### M1 — Semantic Model (Python)  (2/5)
- [x] 1.1 Build a **Differentiable 4-Op FM Synthesizer** in PyTorch
- [x] 1.2 Training Pipeline: Implement ESC-50 Audio Loading + SLERP Interpolation
- [ ] 1.3 **Phase 1 Training:** Grounding Mapper via ESC-50 Audio CLAP Embeddings
- [ ] 1.4 **Phase 2 Training:** Continuity via Intra-Class Audio Interpolation
- [ ] 1.5 Quantize model to INT8 & Export weights for FPGA

### M2 — FM Synth Engine (RTL)  (0/4)
- [ ] 2.1 Sine-wave Phase Accumulator + LUT
- [ ] 2.2 Operator modulation logic (FM feedback/chaining)
- [ ] 2.3 Per-operator ADSR Envelope Generator
- [ ] 2.4 Multi-voice MIDI Decoder & Mixer

### M3 — Neural Inference Engine (RTL)  (1/4)
- [x] 3.1 Basic INT8 MAC PE (`rtl/mac_pe.v`)
- [ ] 3.2 Matrix-Vector Multiplication FSM (Fully Connected Layer Controller)
- [ ] 3.3 BRAM-based weight/activation memory system
- [ ] 3.4 Bit-exact verification (FPGA Output vs Python Golden)

### M4 — System Integration  (0/3)
- [ ] 4.1 UART Parameter Loader (Interface between Neural and Synth Engines)
- [ ] 4.2 UART Embedding Receiver (Host to Neural Engine)
- [ ] 4.3 Top-level routing: MIDI -> FM Synth; UART -> Neural -> FM Synth

### M5 — Deployment & Analysis  (0/4)
- [ ] 5.1 Final Synth/PnR on ECP5-85F
- [ ] 5.2 Latency measurement (USB-Embedding-to-Audio)
- [ ] 5.3 Project Write-up & Demo
- [ ] 5.4 Compare original vs generated audio (System Validation)

---

## Artifacts

| Area | Path | Status |
|------|------|--------|
| Project spec | `Project.md` | exists (Neural FM) |
| State tracker | `AGENTS.md` | exists |
| RTL Cell | `rtl/mac_pe.v` | verified |
| Testbenches | `tb/mac_pe_tb.v` | verified |
| Diff. Synth | `software/diff_fm.py` | verified |
| Loss Functions | `software/loss.py` | verified |
| Clean Utility | `software/clean_cache.py` | exists |
| CTAG reproduction | `ctag-repro/` | Search reproduction plus resumable direct-model training and ONNX export |

---

## Decision Log

- 2026-05-31 — **Project Pivoted to Embedded Neural FM Synth**. Reason: Superior autonomy (Inference + Synthesis both on FPGA), high technical density, and clear validation path.
- 2026-05-31 — **4-Operator FM Architecture**. Reason: Sufficiently complex to recreate almost any sound, yet efficient enough for FPGA fabric.
- 2026-05-31 — **Differentiable Teacher Model**. Reason: Allows for gradient-based training of the parameter mapper.
- 2026-05-31 — **ESC-50 Audio Training**. Reason: Focuses on actual audio textures instead of text labels; Phase 2 SLERP between samples of the same class ensures local continuity in the latent space.

---

## Progress Log

- 2026-05-31 — Completed MAC PE: wrote `rtl/mac_pe.v` and verified with `tb/mac_pe_tb.v`.
- 2026-05-31 — **Pivoted project to Embedded Neural FM Synth.** Rewrote `Project.md` and `AGENTS.md`.
- 2026-05-31 — Setup Python environment (torch, torchaudio, laion-clap).
- 2026-05-31 — Completed **M1.1**: Implemented `software/diff_fm.py` (Differentiable FM Synth).
- 2026-05-31 — Completed `software/loss.py` (DDSP + CLAP Loss functions).
- 2026-05-31 — Focused training on **ESC-50 dataset**; overhauled `software/train_fm.py` to use audio embeddings and intra-class interpolation.
- 2026-08-30 — Added the reference-compatible CTAG paper reproduction under `ctag-repro/`, including a Colab bootstrap; no roadmap checklist state changed.
- 2026-09-02 — Modernized the CTAG backend for the latest Colab Python 3.13/JAX 0.11/PyTorch 2.11 GPU runtime and added a runtime doctor plus runnable notebook; no roadmap checklist state changed.
- 2026-09-02 — Added resumable CTAG-to-MLP distillation, strict no-search prompt inference, held-out listening evaluation, and hardware-facing FP32/INT8 ONNX export; no roadmap checklist state changed.
