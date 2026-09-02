"""Resumable CTAG teacher-data generation for direct patch prediction."""

from __future__ import annotations

import hashlib
import csv
import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np

from .config import RunConfig
from .direct import PARAMETER_CONTRACT, PARAMETER_CONTRACT_HASH
from .pipeline import CTAGPipeline
from .prompts import load_prompts


@dataclass(frozen=True)
class DistillationProfile:
    name: str
    random_patches: int
    population_size: int
    iterations: int
    teacher_runs: int
    surrogate_steps: int
    generator_steps: int
    improvement_rounds: int


PROFILES = {
    "smoke": DistillationProfile("smoke", 64, 2, 2, 2, 2, 2, 0),
    "balanced": DistillationProfile(
        "balanced", 250_000, 32, 96, 4, 50_000, 75_000, 3
    ),
    "quality": DistillationProfile(
        "quality", 1_000_000, 50, 300, 4, 120_000, 150_000, 3
    ),
}

AUDIOSET_LABELS_URL = (
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
    "class_labels_indices.csv"
)


def get_profile(name: str) -> DistillationProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile {name!r}; choose {', '.join(PROFILES)}") from exc


class AtomicShardStore:
    """Append-only NPZ store with atomic unit-level commits and resume metadata."""

    def __init__(self, root: Path, kind: str) -> None:
        self.root = Path(root)
        self.kind = kind
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest.get("kind") != kind or self.manifest.get("format_version") != 1:
                raise ValueError(f"incompatible dataset manifest at {self.manifest_path}")
            if self.manifest.get("parameter_contract_hash") != PARAMETER_CONTRACT_HASH:
                raise ValueError("dataset parameter ordering does not match this runtime")
        else:
            self.manifest = {
                "format_version": 1,
                "kind": kind,
                "parameter_contract": PARAMETER_CONTRACT,
                "parameter_contract_hash": PARAMETER_CONTRACT_HASH,
                "units": [],
            }

    @property
    def completed_units(self) -> set[str]:
        return {entry["unit"] for entry in self.manifest["units"]}

    @property
    def count(self) -> int:
        return sum(int(entry["count"]) for entry in self.manifest["units"])

    def append(self, unit: str, arrays: Mapping[str, np.ndarray]) -> bool:
        if unit in self.completed_units:
            return False
        lengths = {np.asarray(value).shape[0] for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("all shard arrays must have the same first dimension")
        index = len(self.manifest["units"])
        filename = f"shard-{index:06d}.npz"
        final = self.root / filename
        temporary = self.root / f".{filename}.part"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, final)
        self.manifest["units"].append(
            {"unit": unit, "file": filename, "count": next(iter(lengths))}
        )
        manifest_tmp = self.manifest_path.with_suffix(".json.part")
        manifest_tmp.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")
        os.replace(manifest_tmp, self.manifest_path)
        return True

    def load(self) -> Dict[str, np.ndarray]:
        chunks: Dict[str, list[np.ndarray]] = {}
        for entry in self.manifest["units"]:
            with np.load(self.root / entry["file"], allow_pickle=False) as shard:
                for key in shard.files:
                    chunks.setdefault(key, []).append(shard[key])
        if not chunks:
            return {}
        return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def prompt_unit(prompt: str) -> str:
    digest = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()[:16]
    return f"prompt-{digest}"


def prompt_split(prompt: str) -> str:
    bucket = int(hashlib.sha256(prompt.strip().casefold().encode()).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else ("validation" if bucket == 1 else "train")


def default_training_prompts(project_root: Path) -> list[str]:
    """Load and deduplicate bundled training prompts, excluding regression tests."""

    data = Path(project_root) / "data"
    regression = set(load_prompts(data / "regression-prompts.txt"))
    direct_eval = data / "direct-eval-prompts.txt"
    if direct_eval.exists():
        regression.update(load_prompts(direct_eval))
    values: list[str] = []
    seen: set[str] = set()
    for filename in ("audioset-sounds.txt", "esc50-sounds.txt", "esc10-sounds.txt"):
        for prompt in load_prompts(data / filename):
            key = prompt.casefold()
            if prompt not in regression and key not in seen:
                seen.add(key)
                values.append(prompt)
    return values


def full_training_prompts(project_root: Path, cache_directory: Path) -> list[str]:
    """Return AudioSet's 527-label vocabulary plus the paper prompt sets."""

    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    labels_file = cache_directory / "audioset-class-labels.csv"
    if not labels_file.exists():
        partial = labels_file.with_suffix(".csv.part")
        request = urllib.request.Request(AUDIOSET_LABELS_URL, headers={"User-Agent": "ctag-repro/0.3"})
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        os.replace(partial, labels_file)
    with labels_file.open(newline="") as handle:
        labels = [row["display_name"].strip() for row in csv.DictReader(handle)]
    data_root = Path(project_root) / "data"
    regression = set(load_prompts(data_root / "regression-prompts.txt"))
    direct_eval = data_root / "direct-eval-prompts.txt"
    if direct_eval.exists():
        regression.update(load_prompts(direct_eval))
    combined = labels + default_training_prompts(project_root)
    result: list[str] = []
    seen: set[str] = set()
    for prompt in combined:
        key = prompt.casefold()
        if prompt not in regression and key not in seen:
            seen.add(key)
            result.append(prompt)
    return result


def mine_retrieval_teachers(
    audio_embeddings: np.ndarray,
    parameters: np.ndarray,
    text_embedding: np.ndarray,
    count: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    audio_values = np.asarray(audio_embeddings, dtype=np.float32)
    text_values = np.asarray(text_embedding, dtype=np.float32).reshape(-1)
    scores = np.sum(audio_values * text_values[None, :], axis=1, dtype=np.float32)
    count = min(count, len(scores))
    indices = np.argpartition(scores, -count)[-count:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return np.asarray(parameters[indices], dtype=np.float32), scores[indices].astype(np.float32)


class _RetrievalIndex:
    """Keep the corpus in one representation and accelerate repeated top-k queries."""

    def __init__(self, embeddings: np.ndarray, parameters: np.ndarray, device: str) -> None:
        self.parameters = np.asarray(parameters, dtype=np.float32)
        self.torch = None
        self.embeddings = None
        self.device = device
        if device != "cpu":
            try:
                import torch

                self.torch = torch
                self.embeddings = torch.as_tensor(
                    embeddings, dtype=torch.float32, device=device
                )
            except (ImportError, RuntimeError):
                self.torch = None
        if self.torch is None:
            # Convert only once instead of once per prompt.
            self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def query(self, text_embedding: np.ndarray, count: int = 8) -> tuple[np.ndarray, np.ndarray]:
        if self.torch is None:
            return mine_retrieval_teachers(
                self.embeddings, self.parameters, text_embedding, count
            )
        torch = self.torch
        with torch.inference_mode():
            text = torch.as_tensor(
                text_embedding, dtype=torch.float32, device=self.device
            )
            scores = torch.mv(self.embeddings, text)
            values, indices = torch.topk(scores, min(count, len(scores)))
        selected = indices.detach().cpu().numpy()
        return (
            self.parameters[selected],
            values.detach().cpu().numpy().astype(np.float32),
        )


def build_distillation_data(
    workspace: Path,
    checkpoint: Path,
    profile: DistillationProfile,
    prompts: Iterable[str],
    device: str = "cuda",
    seed: int = 0,
    progress: Optional[Callable[[str], None]] = None,
    search_prompts: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Generate random patch embeddings and prompt-specific teacher sets."""

    from .paper_backend import build_paper_components

    emit = progress or (lambda message: None)
    workspace = Path(workspace)
    random_store = AtomicShardStore(workspace / "random", "random-patches")
    teacher_store = AtomicShardStore(workspace / "teachers", "prompt-teachers")
    prompt_values = load_prompts(prompts)
    existing_manifest = workspace / "distillation.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text())
        existing_name = existing.get("profile", {}).get("name")
        if existing_name != profile.name:
            raise ValueError(
                f"workspace was created with profile {existing_name!r}, not {profile.name!r}; "
                "use a different workspace"
            )
    if (
        random_store.count >= profile.random_patches
        and all(prompt_unit(prompt) in teacher_store.completed_units for prompt in prompt_values)
        and existing_manifest.exists()
    ):
        emit("distillation data already complete")
        return json.loads(existing_manifest.read_text())
    config = RunConfig(
        population_size=profile.population_size,
        iterations=profile.iterations,
        runs_per_prompt=profile.teacher_runs,
        checkpoint=str(checkpoint),
        device=device,
        seed=seed,
        output_root=str(workspace / "teacher-runs"),
        profile=f"distill-{profile.name}",
    )
    encoder, synth, search_factory, keys = build_paper_components(config)

    shard_size = 64 if profile.name == "smoke" else (4096 if profile.name == "quality" else 2048)
    batch_index = random_store.count // shard_size
    while random_store.count < profile.random_patches:
        shard_target = min(shard_size, profile.random_patches - random_store.count)
        parameter_chunks = []
        embedding_chunks = []
        while sum(len(chunk) for chunk in parameter_chunks) < shard_target:
            wanted = min(
                profile.population_size,
                shard_target - sum(len(chunk) for chunk in parameter_chunks),
            )
            population = np.asarray(synth.initialize(keys), dtype=np.float32)
            audio = synth.render(population)
            embeddings = encoder.embed_audio(audio[:wanted], config.sample_rate)
            parameter_chunks.append(population[:wanted])
            embedding_chunks.append(embeddings)
        random_store.append(
            f"random-{batch_index:08d}",
            {
                "parameters": np.concatenate(parameter_chunks, axis=0),
                "audio_embeddings": np.asarray(
                    np.concatenate(embedding_chunks, axis=0), dtype=np.float16
                ),
            },
        )
        batch_index += 1
        emit(f"random patches: {random_store.count}/{profile.random_patches}")

    random_data = random_store.load()
    retrieval_index = _RetrievalIndex(
        random_data["audio_embeddings"], random_data["parameters"], device
    )
    pipeline = CTAGPipeline(config, encoder, encoder, synth, search_factory, keys)
    for index, prompt in enumerate(prompt_values):
        unit = prompt_unit(prompt)
        if unit in teacher_store.completed_units:
            continue
        text_embedding = encoder.embed_text([prompt])[0]
        retrieved, retrieval_scores = retrieval_index.query(text_embedding)
        should_search = search_prompts is None or prompt.casefold() in search_prompts
        if should_search:
            searched = pipeline.run([prompt], write_artifacts=False)
            search_params = np.stack([result.best_parameters for result in searched])
            search_scores = np.asarray([result.best_similarity for result in searched], dtype=np.float32)
            all_params = np.concatenate((retrieved, search_params), axis=0)
            all_scores = np.concatenate((retrieval_scores, search_scores), axis=0)
        else:
            all_params, all_scores = retrieved, retrieval_scores
        order = np.argsort(all_scores)[::-1][:8]
        selected = all_params[order]
        selected_scores = all_scores[order]
        if len(selected) < 8:
            selected = np.resize(selected, (8, 78))
            selected_scores = np.resize(selected_scores, 8)
        teacher_store.append(
            unit,
            {
                "prompts": np.asarray([prompt]),
                "text_embeddings": np.asarray(text_embedding[None], dtype=np.float16),
                "parameters": np.asarray(selected[None], dtype=np.float32),
                "scores": np.asarray(selected_scores[None], dtype=np.float32),
                "splits": np.asarray([prompt_split(prompt)]),
            },
        )
        emit(f"teacher prompts: {index + 1}/{len(prompt_values)}")

    manifest = {
        "format_version": 1,
        "profile": asdict(profile),
        "random_count": random_store.count,
        "teacher_count": teacher_store.count,
        "parameter_contract_hash": PARAMETER_CONTRACT_HASH,
    }
    output = workspace / "distillation.json"
    temporary = output.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return manifest
