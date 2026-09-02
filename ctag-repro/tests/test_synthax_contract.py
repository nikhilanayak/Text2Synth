"""Optional contract check against the installed upstream SynthAX package."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
flax = pytest.importorskip("flax")
pytest.importorskip("synthax")

from synthax.config import SynthConfig
from synthax.synth import Voice

from ctag_repro.direct import PARAMETER_LAYOUT


def test_effective_layout_matches_jax_ravel_order():
    synth = Voice(
        config=SynthConfig(
            batch_size=1,
            sample_rate=48_000,
            buffer_size_seconds=0.05,
            control_rate=480,
            eps=1e-6,
        )
    )
    template = synth.init(jax.random.PRNGKey(0))
    flattened = flax.traverse_util.flatten_dict(template)
    actual = tuple(
        (tuple(path), int(np.asarray(value).size))
        for path, value in sorted(flattened.items())
    )
    assert actual == PARAMETER_LAYOUT
