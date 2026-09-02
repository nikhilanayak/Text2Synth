"""One-pass CLAP-embedding to SynthAX patch prediction."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Optional

import numpy as np


PARAMETER_CONTRACT = "synthax-voice-flat-78-v1"
_ADSR = ("attack", "decay", "sustain", "release", "alpha")
_LFO = ("frequency", "mod_depth", "initial_phase", "sin", "tri", "saw", "rsaw", "sqr")
PARAMETER_LAYOUT = (
    (("params", "modules_keyboard", "midi_f0"), 1),
    (("params", "modules_keyboard", "duration"), 1),
    *tuple((("params", "modules_lfo_1_rate_adsr", name), 1) for name in _ADSR),
    *tuple((("params", "modules_lfo_2_rate_adsr", name), 1) for name in _ADSR),
    *tuple((("params", "modules_lfo_1_amp_adsr", name), 1) for name in _ADSR),
    *tuple((("params", "modules_lfo_2_amp_adsr", name), 1) for name in _ADSR),
    *tuple((("params", "modules_lfo_1", name), 1) for name in _LFO),
    *tuple((("params", "modules_lfo_2", name), 1) for name in _LFO),
    *tuple((("params", "modules_adsr_1", name), 1) for name in _ADSR),
    *tuple((("params", "modules_adsr_2", name), 1) for name in _ADSR),
    (("params", "modules_mod_matrix", "mod"), 20),
    (("params", "modules_vco_1", "tuning"), 1),
    (("params", "modules_vco_1", "mod_depth"), 1),
    (("params", "modules_vco_1", "initial_phase"), 1),
    (("params", "modules_vco_2", "tuning"), 1),
    (("params", "modules_vco_2", "mod_depth"), 1),
    (("params", "modules_vco_2", "initial_phase"), 1),
    (("params", "modules_vco_2", "shape"), 1),
    (("params", "modules_mixer", "level"), 3),
)
PARAMETER_CONTRACT_HASH = hashlib.sha256(
    json.dumps(PARAMETER_LAYOUT, separators=(",", ":")).encode()
).hexdigest()


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the direct-model dependencies with `.[direct]`") from exc
    return torch


@dataclass(frozen=True)
class DirectModelConfig:
    embedding_dim: int = 512
    parameter_dim: int = 78
    hidden_dims: tuple[int, ...] = (512, 512, 256)
    heads: int = 8
    parameter_contract: str = PARAMETER_CONTRACT

    def validate(self) -> None:
        if self.embedding_dim != 512 or self.parameter_dim != 78:
            raise ValueError("the CTAG direct model requires 512 embeddings and 78 parameters")
        if self.heads < 1 or any(width < 1 for width in self.hidden_dims):
            raise ValueError("heads and hidden widths must be positive")
        if self.parameter_contract != PARAMETER_CONTRACT:
            raise ValueError("unsupported SynthAX parameter contract")


def build_direct_model(config: DirectModelConfig) -> Any:
    """Build the quantization-friendly MLP without importing Torch at package load."""

    torch = _torch()
    nn = torch.nn
    config.validate()

    class DirectPatchNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            widths = (config.embedding_dim,) + config.hidden_dims
            layers = []
            for input_width, output_width in zip(widths, widths[1:]):
                layers.extend((nn.Linear(input_width, output_width), nn.ReLU()))
            self.trunk = nn.Sequential(*layers)
            self.output = nn.Linear(
                config.hidden_dims[-1], config.heads * config.parameter_dim
            )

        def forward(self, embeddings: Any) -> Any:
            normalized = torch.nn.functional.normalize(embeddings, dim=-1)
            values = torch.sigmoid(self.output(self.trunk(normalized)))
            return values.reshape(-1, config.heads, config.parameter_dim)

    return DirectPatchNet()


def build_patch_surrogate(parameter_dim: int = 78, embedding_dim: int = 512) -> Any:
    """Build the training-only differentiable patch-to-CLAP approximation."""

    torch = _torch()
    nn = torch.nn

    class PatchEmbeddingSurrogate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(parameter_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, embedding_dim),
            )

        def forward(self, parameters: Any) -> Any:
            return torch.nn.functional.normalize(self.net(parameters), dim=-1)

    return PatchEmbeddingSurrogate()


def save_model_bundle(
    directory: Path,
    model: Any,
    config: DirectModelConfig,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically save safe tensor weights and a self-describing manifest."""

    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "model.safetensors"
    temporary = directory / "model.safetensors.part"
    save_file({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
    os.replace(temporary, weights)
    payload = {
        "format_version": 1,
        "model": asdict(config),
        "parameter_contract_hash": PARAMETER_CONTRACT_HASH,
        "parameter_layout": [[list(path), size] for path, size in PARAMETER_LAYOUT],
        "metadata": metadata or {},
    }
    manifest = directory / "model.json"
    temporary_manifest = directory / "model.json.part"
    temporary_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest)
    return directory


def load_model_bundle(directory: Path, device: str = "cpu") -> tuple[Any, DirectModelConfig, Dict[str, Any]]:
    """Load a model only when its dimensions and parameter ordering match."""

    from safetensors.torch import load_file

    directory = Path(directory)
    payload = json.loads((directory / "model.json").read_text())
    if payload.get("format_version") != 1:
        raise ValueError("unsupported direct-model bundle format")
    if payload.get("parameter_contract_hash") != PARAMETER_CONTRACT_HASH:
        raise ValueError("checkpoint SynthAX parameter ordering does not match this runtime")
    model_values = dict(payload["model"])
    model_values["hidden_dims"] = tuple(model_values["hidden_dims"])
    config = DirectModelConfig(**model_values)
    config.validate()
    model = build_direct_model(config).to(device)
    model.load_state_dict(load_file(directory / "model.safetensors", device=device), strict=True)
    model.eval()
    return model, config, dict(payload.get("metadata", {}))


@dataclass(frozen=True)
class DirectPrediction:
    prompt: str
    parameters: np.ndarray
    variants: np.ndarray
    timings_seconds: Dict[str, float]
    metadata: Dict[str, Any]


class DirectPatchPredictor:
    """Frozen CLAP text encoder plus a one-pass patch generator."""

    def __init__(self, model: Any, config: DirectModelConfig, text_encoder: Any, device: str) -> None:
        self.model = model
        self.config = config
        self.text_encoder = text_encoder
        self.device = device

    @classmethod
    def from_bundle(
        cls, bundle: Path, checkpoint: Path, device: str = "cpu"
    ) -> "DirectPatchPredictor":
        from .paper_backend import CLAPEncoder

        model, config, metadata = load_model_bundle(bundle, device)
        expected_checkpoint = metadata.get("clap_checkpoint_sha256")
        if expected_checkpoint:
            from .artifacts import sha256_file

            actual = sha256_file(Path(checkpoint))
            if actual != expected_checkpoint:
                raise ValueError("CLAP checkpoint hash does not match the direct model")
        return cls(model, config, CLAPEncoder(Path(checkpoint), device), device)

    def predict(self, prompts: Iterable[str]) -> list[DirectPrediction]:
        torch = _torch()
        values = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if not values:
            raise ValueError("at least one non-empty prompt is required")
        started = perf_counter()
        embeddings = self.text_encoder.embed_text(values)
        text_seconds = perf_counter() - started
        inputs = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
        started = perf_counter()
        with torch.inference_mode():
            variants = self.model(inputs).detach().cpu().numpy().astype(np.float32)
        if self.device == "cuda":
            torch.cuda.synchronize()
        model_seconds = perf_counter() - started
        return [
            DirectPrediction(
                prompt=prompt,
                parameters=variants[index, 0],
                variants=variants[index],
                timings_seconds={
                    "text_embedding": text_seconds / len(values),
                    "direct_model": model_seconds / len(values),
                },
                metadata={
                    "selection": "head0",
                    "parameter_contract": PARAMETER_CONTRACT,
                },
            )
            for index, prompt in enumerate(values)
        ]
