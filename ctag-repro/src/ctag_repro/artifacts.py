"""Reproducible artifact and provenance output."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
import wave
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import yaml

try:
    import resource
except ImportError:  # pragma: no cover - Windows compatibility
    resource = None

from .config import PAPER_UPSTREAM_COMMIT, RunConfig


def slugify(value: str, limit: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug or "prompt")[:limit].rstrip("-")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for name in (
        "ctag-repro",
        "jax",
        "jaxlib",
        "flax",
        "evosax",
        "synthax",
        "laion-clap",
        "torch",
        "numpy",
    ):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def peak_rss_bytes() -> int:
    if resource is None:
        return 0
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and most BSD-derived CI images report KiB.
    return value if sys.platform == "darwin" else value * 1024


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono float audio as a portable signed 16-bit PCM WAV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = np.round(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


class ArtifactWriter:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.root = Path(config.output_root) / f"{stamp}-{config.profile}-seed{config.seed}"

    def write_run(
        self,
        prompt: str,
        run_index: int,
        audio: np.ndarray,
        patch: Mapping[str, Any],
        history: Iterable[Mapping[str, Any]],
        timings: Mapping[str, float],
        best_fitness: float,
    ) -> Path:
        output = self.root / slugify(prompt) / f"run-{run_index:03d}"
        output.mkdir(parents=True, exist_ok=True)
        write_wav(output / "best.wav", audio, self.config.sample_rate)
        (output / "patch.yaml").write_text(yaml.safe_dump(dict(patch), sort_keys=True))
        rows = list(history)
        with (output / "history.csv").open("w", newline="") as handle:
            fields = list(rows[0]) if rows else ["iteration", "best_fitness"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.config.write_yaml(output / "config.yaml")
        checkpoint = Path(self.config.checkpoint)
        metadata_payload = {
            "prompt": prompt,
            "run_index": run_index,
            "best_fitness": best_fitness,
            "best_similarity": -best_fitness,
            "timings_seconds": dict(timings),
            "upstream_commit": PAPER_UPSTREAM_COMMIT,
            "implementation_revision": git_revision(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "peak_rss_bytes": peak_rss_bytes(),
            "candidate_evaluations": self.config.population_size
            * self.config.iterations,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n"
        )
        return output
