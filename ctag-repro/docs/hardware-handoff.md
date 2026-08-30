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
