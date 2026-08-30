# Validation record

Environment: macOS 15.5 arm64, Python 3.9.6, CPU execution.

`pip check` reports the installed Torch 2.0.0 and gRPC wheels as unsupported
because their internal `WHEEL` files incorrectly declare x86-64 tags. The actual
Torch extension is arm64, the gRPC extension is universal, both import and
execute natively, and all real-backend tests pass.

## External model

- File: `checkpoints/630k-audioset-best.pt`
- Size: 1,863,587,645 bytes
- SHA-256: `8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037`
- Source: the checkpoint URL listed by PapayaResearch/ctag.

## Verified behavior

- Core suite: 17 tests pass and the checkpoint-backed integration test passes
  separately. Legacy-library deprecation warnings are expected and do not
  affect numerical results.
- SynthAX Voice: exactly 78 flattened parameters.
- Candidate render: `(2, 78)` patches produced finite `(2, 96000)` audio.
- CLAP: finite, unit-length `(2, 512)` text and audio embeddings.
- Independent upstream comparison: initial population, first LES candidate batch,
  and all first-batch audio samples matched exactly; both maximum absolute errors
  were `0.0`.
- End-to-end two-iteration smoke run produced WAV, patch, configuration, history,
  and provenance metadata.
- Ten-iteration, population-ten run improved best fitness from `0.0` to
  `-0.1024380326`. Timings were 5.453 s for CLAP audio encoding, 1.211 s for
  synthesis including compilation, 0.240 s for LES, and less than 1 ms for
  scoring.
- Full paper-budget run for `train horn`: 50 candidates x 300 iterations =
  15,000 evaluations. Similarity improved from `0.1454230100` after the first
  iteration to `0.5109993219`. It produced a 96,000-frame, 48 kHz mono WAV and
  a patch containing exactly 78 scalar parameters.
- Full-run component totals: 882.866 s CLAP audio encoding, 30.551 s synthesis,
  0.797 s LES, 0.204 s scoring, and 0.062 s text embedding. The process used
  approximately 3.3 GB resident memory during the run.

The complete local artifact is under
`reference-results/20260822T222021Z-paper-seed42/train-horn/run-000/`.

Run the repeatable equivalence check with:

```bash
.venv/bin/python tools/verify_upstream.py --upstream /tmp/ctag-upstream
```
