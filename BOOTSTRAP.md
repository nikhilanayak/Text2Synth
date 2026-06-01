# 🤖 CLAP-Synth: LLM Bootstrap Guide

This document is a hand-off guide for another AI agent to continue development on the **CLAP-Synth** project. It contains the current architecture, training status, and technical constraints.

---

## 🏗 Project Vision
**CLAP-Synth** is an autonomous generative instrument on the **ULX3S (Lattice ECP5-85F)** FPGA.
- **Input:** 512-D CLAP embedding (sent via UART from a PC prompt).
- **Brain:** On-chip INT8 MLP Mapper translates embedding -> 75 synth parameters.
- **Body:** On-chip 64-Harmonic Additive Synthesizer + Filtered Noise + Inharmonic Stretch.

---

## 🛠 Current Technical Stack

### 1. Machine Learning (Python/PyTorch)
- **Model:** 4-layer MLP (1024 hidden units, 128-D PCA Input, 75-D Parameter Output).
- **Synth Engine:** `software/diff_fm.py` (Vectorized 64-Harmonic Additive + Noise).
- **Optimizations:** 128-D PCA bottleneck (98.3% variance) to reduce FPGA BRAM/UART overhead.
- **Training Strategy:**
    - **Stage 1 (Physics):** Grounding via `software/bake_physics_data.py`. Reconstructs 20,000 random patches. MSE-based.
    - **Stage 2 (Semantic):** Dual-Modal alignment. Matches CLAP embeddings of ESC-50 recordings and labels.
- **Observability:** Full W&B integration with live audio logging.

### 2. Hardware (Verilog/RTL)
- **Completed:** 
    - `rtl/mac_pe.v`: INT8 Matrix-Vector cell.
    - `rtl/fm_phase_accumulator.v`: 32-bit DDS foundation.
    - `rtl/fm_sine_rom.v`: 1024-entry 16-bit signed Sine LUT (matching PyTorch).
- **Toolchain:** OSS CAD Suite (Yosys/nextpnr/Project Trellis).

---

## 📍 Where we left off

### ML Progress:
- The **Dataset Baker** is complete. Stage 1 training is now instantaneous if the data is baked.
- The **Mapper** architecture is finalized and optimized for your Mac (MPS) and the remote GPU (RTX 3070).
- **Next Task:** Monitor Stage 1 convergence on W&B. Once `Param_MSE` < 0.05, let the script transition to Stage 2.

### RTL Progress:
- We have the "Vocal Cords" (Sine ROM) and the "Lungs" (Phase Accumulator).
- **Next Task:** **Milestone 2.3 — The Wavetable Cooker.**
    - This module must iterate 256 times, summing 64 sine waves (using the ROM) with the 64 predicted harmonic amplitudes to "bake" a single-cycle wavetable into a BRAM block.

---

## 📝 Technical Specs for Hand-off

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16,000 Hz |
| Synth Params | 75 (64 Harmonics, 1 Amp, 4 ADSR, 1 Stretch, 4 Noise, 1 Detune) |
| PCA Dim | 128 |
| BRAM Target | 3.7 Mbit (ULX3S 85F) |
| ROM Size | 1024 entries x 16-bit |
| Polyphony | Target 16 voices via time-multiplexed wavetables |

---

## 🚀 Commands for the Next Agent

```bash
# Setup environment
make setup

# Ground the model (Stage 1)
python3 software/bake_physics_data.py
python3 software/train_fm.py

# Test hardware logic
make mac_pe
make fm_phase_accumulator
```

---
**Current Architect:** Gemini CLI
**Project ID:** Synthesizer-2026-Alpha
