# Reproduction contract

This document distinguishes intentional cleanup from behavior that must remain
compatible with the authors' released CTAG program.

## Locked paper behavior

- Voice architecture and SynthAX 0.2.1 parameter order: 78 normalized values.
- One shared JAX PRNG stream seeded with 42.
- Initial population initialization before each independent run.
- LES `ask`/`tell` minimization with `sigma_init=0.2693048095331496`, parameter
  initialization and clipping in `[0, 1]`, and `mean_decay=0`.
- Non-fusion `HTSAT-tiny` plus RoBERTa from `630k-audioset-best.pt`.
- Fitness `-(audio_embedding @ text_embedding.T)`.
- Fitness outside `[-1, 0]` is replaced with zero before `tell`.
- 2 seconds, 48 kHz audio, and 480 Hz control rate.

## Intentional fixes

- A single prompt is supported. The released PyPI combination documented a
  two-prompt workaround, but the orchestration itself does not require it.
- Prompt files retain their last line even without a trailing newline.
- Artifact paths use safe slugs rather than raw prompt text.
- WAV output is deterministic PCM16 and metadata records all provenance.
- Framework conversions occur inside adapters; orchestration operates on NumPy
  values and explicit protocols.
- Modern PyTorch loads the SHA-256-pinned CLAP checkpoint in restricted
  weights-only mode with only its NumPy scalar metadata types allowlisted.
- Transformers 5 omits RoBERTa's deterministic `position_ids` buffer; the
  adapter conditionally discards only that non-learned legacy key while keeping
  strict loading for all learned weights.
- Modern Evosax and JAX replace removed public APIs while retaining the released
  population-shaped initialization, parameter bounds, PRNG stream, and LES
  hyperparameters.

## Reproduction evidence

A valid real-backend reproduction must demonstrate:

1. `ctag setup --strict` reports all packages and the official checkpoint.
2. Unit tests pass without requiring a checkpoint.
3. A smoke search produces all five run artifacts and a finite fitness history.
4. The Voice backend reports exactly 78 parameters and renders `(B, 96000)`.
5. CLAP returns finite `(B, 512)` audio and `(P, 512)` text embeddings.
6. A paper-profile run completes with 300 history rows and improves over its
   initial best recorded fitness.

Exact cross-device waveform identity is not assumed. Comparisons made on one
locked environment should be deterministic for the same seed.

On macOS arm64, JAX and JAXlib 0.4.18 replace 0.4.14 because PyPI does not
provide an arm64 wheel for the released version. This is recorded in run
metadata and must pass the same component-level checks.

On the latest Colab runtime, the compatibility matrix is pinned in
`constraints-colab.txt`. Colab's CUDA-enabled JAX/JAXlib and PyTorch wheels are
never replaced. Flax is upgraded to 0.12.9 because the image's 0.11.2 release
calls `jax.core.get_opaque_trace_state`, which JAX 0.11 removed.
