# Hardware handoff

The parameter-to-audio hardware milestone is implemented under `../rtl/synthax`
with its host transport under `../rtl/transport`. The authoritative boundary is:

- 78 unsigned Q0.16 values in SynthAX render-time PyTree order.
- Contract `synthax-voice-render-flat-78-v2`, SHA-256
  `4a07c1ca91590e8a6f0b781057928c75c383b3fdad8346fe2d67dab7d4e2cac7`.
- Address/data/write-enable plus atomic commit at the neural-to-synth boundary.
- Signed Q1.23 at 48 kHz internally and framed little-endian PCM16 to the host.

The v2 name is intentional. Earlier metadata described Flax initialization
order, while `jax.flatten_util.ravel_pytree` sorts mapping keys before SynthAX
rendering. Search itself optimized the correct coordinates, but any direct-model
workspace made with the v1 hash must be regenerated rather than reused.

## Direct-model contract

The amortized inference path fixes the hardware boundary as a normalized
512-value CLAP embedding followed by three ReLU hidden layers (512, 512, 256)
and eight 78-value sigmoid patch heads. Head zero is the no-search result. The
FP32 and weight-INT8 ONNX exports contain no audio or search operators; the
selected patch is written directly into the synthesizer's ordered control
registers. Every artifact carries the SHA-256 identifier for
`synthax-voice-render-flat-78-v2` so software and RTL cannot silently disagree
about parameter ordering. See `../hardware/README.md` for the implemented DSP,
transport, presets, and Cyclone IV integration.
