"""Framework-neutral orchestration of the CTAG optimization loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from .artifacts import ArtifactWriter
from .config import RunConfig
from .interfaces import AudioEncoder, SearchStrategy, Synthesizer, TextEncoder
from .prompts import load_prompts
from .scoring import negative_cosine_fitness


@dataclass(frozen=True)
class RunResult:
    prompt: str
    run_index: int
    best_fitness: float
    audio: np.ndarray
    best_parameters: np.ndarray
    history: List[Dict[str, Any]]
    timings: Dict[str, float]
    artifact_dir: Optional[Path]

    @property
    def best_similarity(self) -> float:
        return -self.best_fitness


class CTAGPipeline:
    """Optimize interpretable synth patches against CLAP text embeddings."""

    def __init__(
        self,
        config: RunConfig,
        text_encoder: TextEncoder,
        audio_encoder: AudioEncoder,
        synthesizer: Synthesizer,
        search_factory: Callable[[], SearchStrategy],
        key_stream: Any,
        progress_callback: Optional[Callable[[str, int, int, float], None]] = None,
    ) -> None:
        self.config = config
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.synthesizer = synthesizer
        self.search_factory = search_factory
        self.key_stream = key_stream
        self.progress_callback = progress_callback
        if synthesizer.sample_rate != config.sample_rate:
            raise ValueError("synth and pipeline sample rates differ")
        if synthesizer.num_samples != config.num_samples:
            raise ValueError("synth and pipeline buffer sizes differ")

    @classmethod
    def paper(cls, config: RunConfig) -> "CTAGPipeline":
        from .paper_backend import build_paper_components

        encoder, synth, search_factory, keys = build_paper_components(config)
        return cls(config, encoder, encoder, synth, search_factory, keys)

    def run(
        self, prompts: Iterable[str], write_artifacts: bool = True
    ) -> List[RunResult]:
        prompt_list = load_prompts(prompts)
        started = perf_counter()
        text_embeddings = np.asarray(
            self.text_encoder.embed_text(prompt_list), dtype=np.float32
        )
        text_seconds = perf_counter() - started
        if text_embeddings.shape[0] != len(prompt_list):
            raise RuntimeError("text encoder returned the wrong batch dimension")

        artifact_writer = ArtifactWriter(self.config) if write_artifacts else None
        results: List[RunResult] = []
        for prompt_index, prompt in enumerate(prompt_list):
            for run_index in range(self.config.runs_per_prompt):
                results.append(
                    self._run_one(
                        prompt,
                        text_embeddings[prompt_index],
                        run_index,
                        text_seconds,
                        artifact_writer,
                    )
                )
        return results

    def _run_one(
        self,
        prompt: str,
        text_embedding: np.ndarray,
        run_index: int,
        text_seconds: float,
        artifact_writer: Optional[ArtifactWriter],
    ) -> RunResult:
        timings = {
            "text_embedding": text_seconds,
            "synthesis": 0.0,
            "audio_embedding": 0.0,
            "scoring": 0.0,
            "search": 0.0,
        }
        initial_population = self.synthesizer.initialize(self.key_stream)
        search = self.search_factory()
        tick = perf_counter()
        search.initialize(self.key_stream, initial_population)
        timings["search"] += perf_counter() - tick
        history: List[Dict[str, Any]] = []

        for iteration in range(self.config.iterations):
            tick = perf_counter()
            candidates = search.ask(self.key_stream)
            timings["search"] += perf_counter() - tick

            tick = perf_counter()
            audio = self.synthesizer.render(candidates)
            timings["synthesis"] += perf_counter() - tick
            audio = np.asarray(audio, dtype=np.float32)
            expected = (self.config.population_size, self.config.num_samples)
            if audio.shape != expected:
                raise RuntimeError(f"synth returned {audio.shape}; expected {expected}")

            tick = perf_counter()
            audio_embeddings = self.audio_encoder.embed_audio(
                audio, self.config.sample_rate
            )
            timings["audio_embedding"] += perf_counter() - tick

            tick = perf_counter()
            fitness = negative_cosine_fitness(audio_embeddings, text_embedding)
            timings["scoring"] += perf_counter() - tick

            tick = perf_counter()
            search.tell(candidates, fitness)
            timings["search"] += perf_counter() - tick
            history.append(
                {"iteration": iteration, "best_fitness": search.best_fitness}
            )
            if self.progress_callback is not None and (
                iteration % self.config.log_every == 0
                or iteration == self.config.iterations - 1
            ):
                self.progress_callback(
                    prompt, iteration + 1, self.config.iterations, search.best_fitness
                )

        best_parameters = np.asarray(search.best_member, dtype=np.float32)
        best_audio, patch = self.synthesizer.render_one(search.best_member)
        artifact_dir = None
        if artifact_writer is not None:
            artifact_dir = artifact_writer.write_run(
                prompt=prompt,
                run_index=run_index,
                audio=best_audio,
                patch=patch,
                history=history,
                timings=timings,
                best_fitness=search.best_fitness,
            )
        return RunResult(
            prompt=prompt,
            run_index=run_index,
            best_fitness=search.best_fitness,
            audio=np.asarray(best_audio, dtype=np.float32),
            best_parameters=best_parameters,
            history=history,
            timings=timings,
            artifact_dir=artifact_dir,
        )
