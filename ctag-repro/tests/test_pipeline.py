from pathlib import Path

import numpy as np

from ctag_repro.config import RunConfig
from ctag_repro.pipeline import CTAGPipeline


class FakeEncoder:
    def embed_text(self, texts):
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(list(texts)), 1))

    def embed_audio(self, audio, sample_rate):
        value = np.clip(audio.mean(axis=1), 0.0, 1.0)
        result = np.stack([value, 1.0 - value], axis=1)
        return result / np.linalg.norm(result, axis=1, keepdims=True)


class FakeSynth:
    sample_rate = 8
    num_samples = 8
    parameter_count = 2

    def __init__(self, population_size):
        self.population_size = population_size

    def initialize(self, key_stream):
        return np.zeros((self.population_size, 2), dtype=np.float32)

    def render(self, flat_parameters):
        return np.repeat(np.asarray(flat_parameters)[:, :1], self.num_samples, axis=1)

    def render_one(self, flat_parameters):
        values = np.asarray(flat_parameters).reshape(-1)
        return np.repeat(values[0], self.num_samples), {"params": values.tolist()}


class FakeSearch:
    def __init__(self):
        self.step = 0
        self._best_fitness = float("inf")
        self._best_member = None

    def initialize(self, key_stream, initial_population):
        pass

    def ask(self, key_stream):
        first = 0.8 + 0.1 * self.step
        self.step += 1
        return np.array([[first, 0.0], [0.2, 0.0], [0.5, 0.0]], dtype=np.float32)

    def tell(self, candidates, fitness):
        index = int(np.argmin(fitness))
        if float(fitness[index]) < self._best_fitness:
            self._best_fitness = float(fitness[index])
            self._best_member = candidates[index]

    @property
    def best_member(self):
        return self._best_member

    @property
    def best_fitness(self):
        return self._best_fitness


def test_pipeline_optimizes_and_writes_artifacts(tmp_path: Path):
    config = RunConfig(
        sample_rate=8,
        control_rate=2,
        duration_seconds=1,
        population_size=3,
        iterations=2,
        output_root=str(tmp_path),
        profile="test",
    )
    encoder = FakeEncoder()
    pipeline = CTAGPipeline(
        config,
        encoder,
        encoder,
        FakeSynth(config.population_size),
        FakeSearch,
        object(),
    )
    result = pipeline.run(["train horn"])[0]
    assert result.best_fitness < -0.99
    assert result.best_similarity > 0.99
    assert result.history[1]["best_fitness"] <= result.history[0]["best_fitness"]
    assert result.artifact_dir is not None
    for filename in (
        "best.wav",
        "patch.yaml",
        "history.csv",
        "config.yaml",
        "metadata.json",
    ):
        assert (result.artifact_dir / filename).is_file()
