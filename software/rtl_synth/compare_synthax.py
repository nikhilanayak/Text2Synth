#!/usr/bin/env python3
"""Compare diagnostic fixed-point patches with the installed SynthAX Voice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from flax import traverse_util
from jax import flatten_util
from synthax.config import SynthConfig
from synthax.synth import Voice

from fixed_model import load_preset, render


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("rtl/synthax/assets/diagnostic_presets.json")
    )
    parser.add_argument("--minimum-tonal-correlation", type=float, default=0.97)
    args = parser.parse_args()
    config = SynthConfig(
        batch_size=1, sample_rate=48_000, buffer_size_seconds=2.0,
        control_rate=480, eps=1e-6
    )
    synth = Voice(config=config)
    template = synth.init(jax.random.PRNGKey(1))
    flattened = traverse_util.flatten_dict(template)
    unbatched = traverse_util.unflatten_dict(
        {key: np.asarray(value).squeeze() for key, value in flattened.items()}
    )
    _, unravel = flatten_util.ravel_pytree(unbatched)
    descriptions = json.loads(args.manifest.read_text())["presets"]
    report = []
    failures = []
    for index, description in enumerate(descriptions):
        parameters = load_preset(args.manifest, index)
        shaped = jax.tree.map(
            lambda value: np.expand_dims(value, 0),
            unravel(parameters.astype(np.float32) / 65535.0),
        )
        reference = np.asarray(synth.apply(shaped), dtype=float).squeeze()
        fixed = render(parameters).astype(float) / 32767.0
        name = description["name"]
        item = {
            "name": name,
            "correlation": correlation(reference, fixed),
            "reference_rms": float(np.sqrt(np.mean(reference**2))),
            "fixed_rms": float(np.sqrt(np.mean(fixed**2))),
        }
        report.append(item)
        if name != "noise" and item["correlation"] < args.minimum_tonal_correlation:
            failures.append(name)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("tonal correlation below threshold: " + ", ".join(failures))


if __name__ == "__main__":
    main()
