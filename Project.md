# Project spec: prompt-to-SynthAX FPGA instrument

**Architect:** Nikhil Nayak
**Target:** a Cyclone IV teaching board, exact model and pin map pending

## Goal

Recreate *Creative Text-to-Audio Generation via Synthesizer Programming*, then
turn its synthesizer into a playable hardware instrument. A trained direct model
eventually replaces search for live prompt-to-patch inference. The hardware
boundary is deliberately stable: either path emits exactly 78 normalized
SynthAX parameters.

The computer retains CLAP text encoding and plays returned audio. The FPGA owns
patch storage/editing and real-time synthesis. The first bitstream contains
patches selected before compilation; runtime host patch loading and on-FPGA
neural inference are later extensions.

## Fixed hardware behavior

- Monophonic SynthAX-compatible voice at 48 kHz with a 480 Hz control rate.
- Four play buttons: MIDI 60, 64, 67, and 72; highest held note wins.
- One button cycles eight compiled presets.
- One switch enters edit mode; the four buttons become previous/next parameter
  and decrease/increase.
- Eight hexadecimal seven-segment digits show preset or alternating parameter
  address/value pages.
- Edits are volatile. Reset restores the ROM image.
- No audio jack, ADC, codec, or I2S dependency. Framed PCM16 returns to the PC
  over Intel JTAG UART, with a 3 Mbaud USB-UART fallback.
- A FIFO and CRC/sequence framing expose transport problems rather than
  disguising them as synthesis errors.

## Parameter and model boundary

Contract `synthax-voice-render-flat-78-v2` uses unsigned Q0.16 registers in the
actual sorted JAX PyTree order. Its hash is embedded in software and generated
preset metadata. A future INT8 neural engine writes register address/data/enable
and pulses commit; the instrument swaps the complete patch at a control tick.

The direct network currently targets a normalized 512-value CLAP embedding and
eight 78-value candidate heads. Strict live inference uses head zero. This
network remains a Colab training deliverable until measured quality is adequate.

## Milestones

### M1 — Paper and live-model software

- [x] Reproduce CTAG's released CLAP/SynthAX/LES inference path.
- [x] Provide latest-Colab bootstrap, notebooks, resumable artifacts, and tests.
- [x] Implement direct model, distillation data generation, evaluation, ONNX,
  and INT8 export paths.
- [~] Run the full GPU teacher/training profile and accept against held-out audio
  and prompt metrics.

### M2 — Parameter-to-audio hardware

- [x] Freeze and hash the effective 78-parameter render contract.
- [x] Implement DDS oscillators, noise, ADSRs, LFOs, modulation matrix, mixing,
  and fixed-point reference model.
- [x] Implement preset ROM/RAM, atomic external patch commit, live controls, and
  seven-segment output.
- [x] Implement PCM FIFO, CRC framing, JTAG UART Avalon master, physical UART
  fallback, and host playback.
- [x] Add deterministic assets, unit simulations, waveform correlation checks,
  and a board-neutral Cyclone IV Quartus project.
- [ ] Add the exact school-board device/pin assignments and complete Quartus
  timing/resource reports once the board identity is available.

### M3 — Neural inference hardware

- [ ] Select the trained model checkpoint and freeze its quantization scales.
- [ ] Inventory ONNX operations and generate bit-exact layer vectors.
- [ ] Implement the INT8 matrix-vector controller, activation/weight memories,
  requantization, and sigmoid approximation.
- [ ] Connect its 78 outputs to the existing atomic patch interface.

### M4 — Deployment and evaluation

- [ ] Measure JTAG throughput, FIFO occupancy, audio gaps, and note-to-host
  latency on the actual board; use USB-UART if JTAG misses the target.
- [ ] Compare FPGA captures with the fixed-point oracle and SynthAX perceptual
  reference across trained presets.
- [ ] Demonstrate compiled prompt patches, live playing, and no-reburn editing.

## Acceptance criteria

The portable RTL and all testbenches compile with Icarus Verilog. Tonal
diagnostic patches must correlate at least 0.97 with the SynthAX software
reference after gain/time alignment; noise is compared by RMS, mean, and
spectral distribution. RTL samples must track the fixed-point oracle at 0.95 or
better correlation. Host frames must round-trip through incremental parsing,
reject bad CRCs, and resynchronize. Board closure additionally requires zero
FIFO overflows during sustained playback and timing closure at the board clock.
