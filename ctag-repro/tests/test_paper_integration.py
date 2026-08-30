"""Opt-in checks for the checkpoint-backed JAX/PyTorch implementation."""

import os
from pathlib import Path

import numpy as np
import pytest

from ctag_repro.config import RunConfig
from ctag_repro.pipeline import CTAGPipeline


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("CTAG_RUN_INTEGRATION") != "1",
    reason="set CTAG_RUN_INTEGRATION=1 to load the official CLAP checkpoint",
)
def test_real_backend_end_to_end():
    checkpoint = Path("checkpoints/630k-audioset-best.pt")
    if not checkpoint.is_file():
        pytest.skip("official checkpoint is not present")
    config = RunConfig.smoke(iterations=1)
    pipeline = CTAGPipeline.paper(config)
    assert pipeline.synthesizer.parameter_count == 78
    result = pipeline.run(["train horn"], write_artifacts=False)[0]
    assert result.audio.shape == (96_000,)
    assert result.best_parameters.shape == (78,)
    assert np.isfinite(result.audio).all()
    assert np.isfinite(result.best_fitness)
