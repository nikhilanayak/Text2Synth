"""Typed, serializable configuration for the CTAG reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from typing import Any, Dict

import yaml


PAPER_UPSTREAM_COMMIT = "fc207b271a9761a6b001e3d028e777d608c4e91f"
PAPER_CHECKPOINT = "630k-audioset-best.pt"
PAPER_CHECKPOINT_URL = (
    "https://huggingface.co/lukewys/laion_clap/resolve/main/"
    "630k-audioset-best.pt"
)
PAPER_CHECKPOINT_SHA256 = (
    "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
)


@dataclass(frozen=True)
class RunConfig:
    """All behavior-affecting settings for one CTAG run.

    Defaults reproduce the configuration released with the ICML 2024 paper.
    Use :meth:`smoke` for fast development runs.
    """

    sample_rate: int = 48_000
    control_rate: int = 480
    duration_seconds: float = 2.0
    population_size: int = 50
    iterations: int = 300
    runs_per_prompt: int = 1
    seed: int = 42
    sigma_init: float = 0.2693048095331496
    mean_decay: float = 0.0
    device: str = "cpu"
    checkpoint: str = "checkpoints/630k-audioset-best.pt"
    output_root: str = "runs"
    log_every: int = 10
    profile: str = "paper"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.control_rate <= 0:
            raise ValueError("sample_rate and control_rate must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.iterations < 1 or self.runs_per_prompt < 1:
            raise ValueError("iterations and runs_per_prompt must be positive")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")

    @property
    def num_samples(self) -> int:
        return int(ceil(self.sample_rate * self.duration_seconds))

    @classmethod
    def paper(cls, **overrides: Any) -> "RunConfig":
        return replace(cls(), **overrides)

    @classmethod
    def smoke(cls, **overrides: Any) -> "RunConfig":
        values: Dict[str, Any] = {
            "population_size": 2,
            "iterations": 2,
            "profile": "smoke",
        }
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=True))

    @classmethod
    def from_yaml(cls, path: Path, **overrides: Any) -> "RunConfig":
        data = yaml.safe_load(path.read_text()) or {}
        data.update(overrides)
        return cls(**data)
