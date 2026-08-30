"""Compare the refactored backend to the authors' exact first-step sequence.

Usage:
    git clone --depth 1 https://github.com/PapayaResearch/ctag /tmp/ctag-upstream
    .venv/bin/python tools/verify_upstream.py --upstream /tmp/ctag-upstream
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import evosax
import flax
import jax
import jax.numpy as jnp
import numpy as np
from synthax.config import SynthConfig
from synthax.synth import Voice

from ctag_repro.config import PAPER_UPSTREAM_COMMIT, RunConfig
from ctag_repro.paper_backend import EvosaxLES, JAXKeyStream, SynthAXVoice


def _load_upstream_key(upstream: Path):
    source = upstream / "ctag" / "utils" / "random.py"
    if not source.is_file():
        raise FileNotFoundError(f"not an upstream CTAG checkout: {upstream}")
    spec = importlib.util.spec_from_file_location("ctag_upstream_random", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PRNGKey


def verify(upstream: Path) -> dict:
    key_type = _load_upstream_key(upstream)
    config = RunConfig.smoke()

    ours_keys = JAXKeyStream(config.seed)
    ours_synth = SynthAXVoice(config, ours_keys)
    ours_initial = ours_synth.initialize(ours_keys)
    ours_search = EvosaxLES(config)
    ours_search.initialize(ours_keys, ours_initial)
    ours_candidates = ours_search.ask(ours_keys)
    ours_audio = ours_synth.render(ours_candidates)

    keys = key_type(config.seed)
    synth_config = SynthConfig(
        batch_size=config.population_size,
        sample_rate=config.sample_rate,
        buffer_size_seconds=config.duration_seconds,
        control_rate=config.control_rate,
        eps=1e-6,
    )
    single_config = SynthConfig(
        batch_size=1,
        sample_rate=config.sample_rate,
        buffer_size_seconds=config.duration_seconds,
        control_rate=config.control_rate,
        eps=1e-6,
    )
    synth = Voice(config=synth_config)
    single = Voice(config=single_config)
    synth.init(keys.split())
    single_parameters = single.init(keys.split())
    flat_single = flax.traverse_util.flatten_dict(single_parameters)
    unbatched = flax.traverse_util.unflatten_dict(
        {key: value.squeeze() for key, value in flat_single.items()}
    )
    reshaper = evosax.ParameterReshaper(unbatched)

    parameters = synth.init(keys.split())
    flat = flax.traverse_util.flatten_dict(parameters)
    initial = jnp.concatenate(
        [value.reshape(config.population_size, -1) for value in flat.values()], axis=1
    )
    strategy = evosax.strategies.LES(
        popsize=config.population_size,
        num_dims=reshaper.total_params,
        mean_decay=config.mean_decay,
    )
    strategy_params = strategy.default_params.replace(
        sigma_init=config.sigma_init,
        init_min=0.0,
        init_max=1.0,
        clip_min=0.0,
        clip_max=1.0,
    )
    state = strategy.initialize(keys.split(), strategy_params, initial)
    candidates, state = strategy.ask(keys.split(), state, strategy_params)
    audio = jax.jit(synth.apply)(jax.jit(reshaper.reshape)(candidates))

    result = {
        "expected_upstream_commit": PAPER_UPSTREAM_COMMIT,
        "initial_exact": bool(np.array_equal(np.asarray(ours_initial), np.asarray(initial))),
        "candidate_exact": bool(
            np.array_equal(np.asarray(ours_candidates), np.asarray(candidates))
        ),
        "audio_exact": bool(np.array_equal(ours_audio, np.asarray(audio))),
        "candidate_max_abs_error": float(
            np.max(np.abs(np.asarray(ours_candidates) - np.asarray(candidates)))
        ),
        "audio_max_abs_error": float(np.max(np.abs(ours_audio - np.asarray(audio)))),
    }
    result["passed"] = all(
        result[key] for key in ("initial_exact", "candidate_exact", "audio_exact")
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.upstream)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
