# Direct prompt-to-patch training

The direct model amortizes CTAG's search cost: expensive SynthAX/CLAP scoring
is performed while labels are built, then a small MLP predicts patches at live
inference time. It does not replace or modify the reference reproduction.

## Fixed contracts

- Input: one normalized 512-value embedding from the released CLAP text branch.
- Output: eight ordered patches of 78 values in `[0, 1]`.
- Strict inference: head zero, with no audio encoding or candidate search.
- Optional reranking: render exactly eight heads once and select by CLAP score.
- Network: `512 → 512 → 512 → 256 → 8×78`, ReLU hidden activations and sigmoid outputs.
- Artifact identity: CLAP checkpoint SHA-256 plus the
  `synthax-voice-flat-78-v1` parameter-contract hash.

## Profiles

`smoke` uses 64 random patches and two optimizer steps for a pipeline check.
`balanced` uses 250,000 random patches, all 527 AudioSet labels, four 32×96
CTAG teacher runs for the bundled paper prompts, 50,000 surrogate steps, 75,000
generator steps, and three real-CLAP improvement rounds. `quality` raises the
random corpus to one million patches and uses the full 50×300 paper search
budget, with 120,000 and 150,000 training steps.

Random patch/audio embeddings are stored as atomic compressed shards. Prompt
teacher shards contain text embeddings, eight patches, true CLAP scores, and a
stable prompt ID. Audio is regenerated rather than retained. Training state
includes both networks, optimizers, RNG state, progress, and contract hashes.

## Commands

```bash
ctag build-distillation-data --profile balanced --workspace WORKSPACE
ctag train-direct --profile balanced --workspace WORKSPACE
ctag evaluate-direct --bundle WORKSPACE/training/model --output WORKSPACE/evaluation
ctag infer-direct --bundle WORKSPACE/training/model --prompt "train horn"
ctag infer-direct --bundle WORKSPACE/training/model --prompt "train horn" --selection clap
ctag export-direct --bundle WORKSPACE/training/model --output WORKSPACE/export --quantize
```

`ctag train-live` composes data generation, training, improvement, and safe
resume behavior. The Colab notebook is the recommended interface. A completed
model is accepted only after the held-out report is reviewed; code validation
cannot guarantee a semantic quality threshold before the actual GPU training
run has produced weights.
