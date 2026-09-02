# Deferred hardware handoff

No hardware-specific arithmetic is introduced during the reference milestone.
The following outputs are collected now because they will be authoritative when
hardware work begins:

- Ordered 78-value patch representation and full parameter tree.
- Synth input/output shapes and 48 kHz/480 Hz timing contract.
- CLAP embedding shapes and preprocessing checkpoint identity.
- Per-stage runtime and peak process characteristics.
- Fitness histories and prompt-level regression artifacts.

The next milestone will add intermediate tensor capture, ONNX/operator
inventory, quantization calibration, and bit-exact models. SystemVerilog design
starts only after those results define precision, memory, and throughput needs.

## Direct-model contract

The amortized inference path fixes the hardware boundary as a normalized
512-value CLAP embedding followed by three ReLU hidden layers (512, 512, 256)
and eight 78-value sigmoid patch heads. Head zero is the no-search result. The
FP32 and weight-INT8 ONNX exports contain no audio or search operators; the
selected patch is written directly into the synthesizer's ordered control
registers. Every artifact carries the SHA-256 identifier for
`synthax-voice-flat-78-v1` so software and RTL cannot silently disagree about
parameter ordering.
