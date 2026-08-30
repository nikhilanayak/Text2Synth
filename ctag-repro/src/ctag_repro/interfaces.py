"""Framework-neutral component contracts used by the CTAG orchestrator."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, Tuple

import numpy as np


class TextEncoder(Protocol):
    def embed_text(self, texts: Iterable[str]) -> np.ndarray:
        """Return one normalized embedding per input string."""


class AudioEncoder(Protocol):
    def embed_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return one normalized embedding per input waveform."""


class Synthesizer(Protocol):
    sample_rate: int
    num_samples: int
    parameter_count: int

    def initialize(self, key_stream: Any) -> Any:
        """Return a population-shaped initial parameter array."""

    def render(self, flat_parameters: Any) -> np.ndarray:
        """Render a population of flattened patches."""

    def render_one(self, flat_parameters: Any) -> Tuple[np.ndarray, Mapping[str, Any]]:
        """Render one patch and return audio plus a serializable patch tree."""


class SearchStrategy(Protocol):
    def initialize(self, key_stream: Any, initial_population: Any) -> None:
        """Initialize search state around the supplied population."""

    def ask(self, key_stream: Any) -> Any:
        """Return the next candidate population."""

    def tell(self, candidates: Any, fitness: np.ndarray) -> None:
        """Update search state. Lower fitness is better."""

    @property
    def best_member(self) -> Any:
        """Return the best flattened patch seen so far."""

    @property
    def best_fitness(self) -> float:
        """Return the best minimization objective value."""
