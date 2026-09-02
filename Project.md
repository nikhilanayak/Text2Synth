# Project Spec: Embedded Neural FM Synthesizer (CLAP-to-FM)

**Architect:** Nikhil Nayak
**Target Applications:** Edge AI · Generative Hardware · Digital Signal Processing · FPGA/RTL
**Project Type:** High-End Portfolio Piece — **A synthesizer that uses on-chip Neural Inference to design its own internal FM patches.**

---

## 0. Scope & Intent (read first)

The **Embedded Neural FM Synthesizer** is a generative instrument where the FPGA handles both the "Decision" (Neural Inference) and the "Synthesis" (DSP).

- **Host PC Role:** Minimal. It converts a text prompt or audio clip into a fixed **512-D CLAP embedding vector** and sends it to the FPGA.
- **FPGA Brain (Neural Engine):** An on-chip INT8 MLP Decoder that takes the 512-D vector and predicts **~40-64 FM parameters** (frequencies, envelopes, feedback, etc.).
- **FPGA Instrument (FM Engine):** A **4-Operator FM Synthesizer** that takes these parameters and generates real-time audio with MIDI polyphony.

The story: *The FPGA is an autonomous sound designer. It receives a 'concept' (embedding) and internally reconfigures its entire FM synthesis engine to manifest that sound.*

---

## 1. System Overview

```
 [ PC / Python ]                              [ ULX3S ECP5-85F FPGA ]
  "Metallic Bell"                                 +-----------------------------+
       |                                          |  1. NEURAL ENGINE (INT8)    |
 [ CLAP Encoder ]  ==USB (512-D Vector)==>        |     (Decoder: MLP Inference)|
       |                                          |           |                 |
       +------------------------------------------+-----> [ FM PARAM REGISTERS ] |
                                                  |           |                 |
 [ MIDI Keyboard ] ==MIDI Link==> [MIDI Decoder]--|--> 2. FM SYNTH ENGINE (x16) |
                                                  |     (4-Op FM + ADSR + Mix)  |
                                                  +-----------|-----------------+
                                                              |
                                                       [ Audio Out (ΔΣ) ]
```

---

## 2. Target Platform & Constraints

| Item | Decision |
|------|----------|
| Board | **ULX3S (Lattice ECP5-85F)** |
| Neural Model | **MLP Decoder** (~200k-300k INT8 parameters) |
| Synth Engine | **4-Operator FM** (High-fidelity, hardware-parallel) |
| On-chip BRAM | 3.74 Mbit (Used for Neural Weights + Sine/Envelope LUTs) |
| Logic (LUTs) | FM Synthesis is LUT-efficient, leaving room for the Neural Engine |

---

## 3. The Neural "Brain" (The Parameter Mapper)

- **Input:** 512-D CLAP Vector (INT8 Quantized).
- **Architecture:** 3-4 Layer MLP (e.g., 512 -> 256 -> 128 -> 48).
- **Outputs:** Specific "knob" values for the FM engine (Normalized 0.0 to 1.0).
- **Quantization:** **INT8 Symmetric** for both weights and activations.
- **Verification:** Compare FPGA parameter output vs Python "Golden" model.

---

## 4. The FM Synth Engine

A **4-Operator FM** architecture (similar to a Yamaha DX-series) allows for massive timbral variety.
- **Operators:** 4 Sine-wave oscillators with Phase Accumulators.
- **Modulation:** Flexible routing (Op4 -> Op3 -> Op2 -> Op1).
- **Envelopes:** Dedicated ADSR for each operator (controls volume and modulation depth).
- **Polyphony:** Target 16 voices using time-multiplexing or parallel hardware.

---

## 5. Training Strategy (ESC-50 Audio Loop)

We utilize a two-phase self-supervised approach using the **ESC-50** dataset (2,000 environmental recordings):

1.  **Phase 1: Texture Grounding (Audio-to-Audio):** 
    - The system takes a target audio clip from ESC-50.
    - Target Embedding = `CLAP_Audio_Encoder(Target_Audio)`.
    - The Mapper predicts FM parameters; Synth generates `New_Audio`.
    - Loss: Minimize Cosine Distance between `Target_Embedding` and `CLAP_Audio_Encoder(New_Audio)`.
2.  **Phase 2: Class Interpolation (Continuous Texture):**
    - Group ESC-50 clips by their class (e.g., "Rain," "Cello," "Dog").
    - Pick two clips from the same class, get their CLAP embeddings, and **SLERP** between them.
    - The Mapper learns to find FM parameters that match these hybrid textures within a semantic leaf.

---

## 6. Execution Roadmap

- **M1 — Semantic Mapping (Python):** Build the training pipeline; train the CLAP-to-FM parameters model; verify "Synth Inversion" works in software.
- **M2 — FM Engine (RTL):** Build the 4-Op FM synthesis core in Verilog; verify MIDI-to-Audio functionality.
- **M3 — Neural Engine (RTL):** Implement the Matrix-Vector Multiplication engine to run the Decoder on-chip.
- **M4 — Integration:** Connect Neural Engine outputs to FM Engine control registers.
- **M5 — Bring-up:** Deploy to ULX3S; real-time Text-to-FM-Patch demo via MIDI.

---

## 7. CTAG Reference Reproduction

The repository includes a separate, reference-compatible reproduction of the
ICML 2024 CTAG search pipeline under `ctag-repro/`. It preserves the authors'
78-parameter SynthAX Voice, frozen LAION-CLAP objective, and LES optimizer. This
serves as the semantic-search baseline and as an implementation reference for
the later hardware handoff. Its GPU path tracks the latest Colab Python 3.13,
JAX, and PyTorch runtime while retaining a separate frozen Python 3.9 reference
environment; it does not change the milestone counts below.

---

## 8. Execution Checklist

### M1 — Semantic Model (Python)
- [x] 1.1 Build a **Differentiable 4-Op FM Synthesizer** in PyTorch
- [x] 1.2 Training Pipeline: Implement ESC-50 Audio Loading + SLERP Interpolation
- [ ] 1.3 **Phase 1 Training:** Grounding Mapper via ESC-50 Audio CLAP Embeddings
- [ ] 1.4 **Phase 2 Training:** Continuity via Intra-Class Audio Interpolation
- [ ] 1.5 Quantize model to INT8 & Export weights for FPGA

### M2 — FM Synth Engine (RTL)
- [ ] 2.1 Sine-wave Phase Accumulator + LUT
- [ ] 2.2 Operator modulation logic (FM feedback/chaining)
- [ ] 2.3 Per-operator ADSR Envelope Generator
- [ ] 2.4 Multi-voice MIDI Decoder & Mixer

### M3 — Neural Inference Engine (RTL)
- [x] 3.1 Basic INT8 MAC PE (`rtl/mac_pe.v`)
- [ ] 3.2 Matrix-Vector Multiplication FSM (Fully Connected Layer Controller)
- [ ] 3.3 BRAM-based weight/activation memory system
- [ ] 3.4 Bit-exact verification (FPGA Output vs Python Golden)

### M4 — System Integration
- [ ] 4.1 UART Parameter Loader (Interface between Neural and Synth Engines)
- [ ] 4.2 UART Embedding Receiver (Host to Neural Engine)
- [ ] 4.3 Top-level routing: MIDI -> FM Synth; UART -> Neural -> FM Synth

### M5 — Deployment & Analysis
- [ ] 5.1 Final Synth/PnR on ECP5-85F
- [ ] 5.2 Latency measurement (USB-Embedding-to-Audio)
- [ ] 5.3 Project Write-up & Demo
