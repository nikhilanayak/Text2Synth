"""Lazy-loaded JAX/PyTorch backend matching the released CTAG implementation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .config import RunConfig


class MissingPaperDependencies(RuntimeError):
    """Raised when the optional paper runtime has not been installed."""


def _paper_imports() -> Dict[str, Any]:
    # CTAG runs JAX synthesis and PyTorch CLAP inference in the same process.
    # JAX's default GPU preallocation can otherwise starve PyTorch on Colab.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        import evosax
        import flax
        import jax
        import jax.numpy as jnp
        import laion_clap
        import torch
        from jax import flatten_util
        try:
            from evosax.algorithms.distribution_based import LearnedES
        except ImportError:  # evosax 0.1.x paper environment
            LearnedES = None
        from synthax.config import SynthConfig
        from synthax.synth import Voice
    except (ImportError, AttributeError) as exc:
        raise MissingPaperDependencies(
            "The ML stack is unavailable. Use `.[paper]` in the legacy "
            "Python 3.9 environment or run tools/colab_bootstrap.sh on Colab."
        ) from exc
    return {
        "evosax": evosax,
        "flax": flax,
        "jax": jax,
        "flatten_util": flatten_util,
        "jnp": jnp,
        "laion_clap": laion_clap,
        "torch": torch,
        "LearnedES": LearnedES,
        "SynthConfig": SynthConfig,
        "Voice": Voice,
    }


def _load_checkpoint(torch_module: Any, checkpoint: Path, device: str) -> Any:
    """Load the official checkpoint safely across PyTorch generations.

    PyTorch 2.6 made ``weights_only=True`` the default. The released CLAP file
    contains NumPy scalar metadata, so modern PyTorch needs three narrowly
    scoped safe globals. Older PyTorch releases do not provide this API and
    retain their historical loader behavior.
    """

    safe_globals = getattr(torch_module.serialization, "safe_globals", None)
    if safe_globals is None:
        return torch_module.load(
            str(checkpoint), map_location=device, weights_only=False
        )

    numpy_core = getattr(np, "_core", np.core)
    numpy_scalar = numpy_core.multiarray.scalar
    allowed = [
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    with safe_globals(allowed):
        return torch_module.load(
            str(checkpoint), map_location=device, weights_only=True
        )


class JAXKeyStream:
    """Stateful key splitting identical to the authors' PRNGKey helper."""

    def __init__(self, seed: int) -> None:
        deps = _paper_imports()
        self._jax = deps["jax"]
        self._key = self._jax.random.PRNGKey(seed)

    def split(self) -> Any:
        self._key, subkey = self._jax.random.split(self._key)
        return subkey


class CLAPEncoder:
    """Combined LAION-CLAP text/audio encoder from the paper checkpoint."""

    embedding_size = 512

    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        deps = _paper_imports()
        self._torch = deps["torch"]
        self.device = device
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"CLAP checkpoint not found: {checkpoint}. Run `ctag setup --download`."
            )
        self.model = deps["laion_clap"].CLAP_Module(
            enable_fusion=False,
            amodel="HTSAT-tiny",
            tmodel="roberta",
            device=device,
        )
        state = _load_checkpoint(self._torch, checkpoint, device)
        state_dict = (
            state["state_dict"]
            if isinstance(state, dict) and "state_dict" in state
            else state
        )
        first_key = next(iter(state_dict))
        if first_key.startswith("module"):
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        # Transformers 5 no longer persists RoBERTa's deterministic position
        # IDs buffer. It has no learned values, so discard only this known
        # legacy checkpoint entry and retain strict loading for everything else.
        position_ids = "text_branch.embeddings.position_ids"
        if position_ids not in self.model.model.state_dict():
            state_dict.pop(position_ids, None)
        self.model.model.load_state_dict(state_dict)
        self.model.model.eval()

    def embed_text(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            raise ValueError("at least one text prompt is required")
        # laion-clap 1.1.4 squeezes a one-item token batch to one dimension,
        # which RoBERTa rejects. The authors therefore required >=2 prompts.
        # Duplicating only at this adapter boundary yields the same first-row
        # embedding while preserving a natural one-prompt public interface.
        single = len(values) == 1
        model_values = values * 2 if single else values
        with self._torch.inference_mode():
            embedded = self.model.get_text_embedding(model_values, use_tensor=True)
        if hasattr(embedded, "detach"):
            embedded = embedded.detach().cpu().numpy()
        result = np.asarray(embedded, dtype=np.float32)
        return result[:1] if single else result

    def embed_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != 48_000:
            raise ValueError("the paper CLAP checkpoint requires 48 kHz audio")
        # JAX's NumPy view can be read-only; an owned array avoids PyTorch's
        # undefined-behavior warning without altering values.
        owned_audio = np.array(audio, dtype=np.float32, copy=True)
        tensor = self._torch.as_tensor(
            owned_audio, dtype=self._torch.float32, device=self.device
        )
        tensor = self._torch.atleast_2d(tensor)
        with self._torch.inference_mode():
            embedded = self.model.get_audio_embedding_from_data(
                tensor, use_tensor=True
            )
        return np.asarray(embedded.detach().cpu(), dtype=np.float32)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_to_builtin(item) for item in value]
    return value


class SynthAXVoice:
    """Reference Voice synth with the paper's exact Flax parameter layout."""

    parameter_count = 78

    def __init__(self, config: RunConfig, key_stream: JAXKeyStream) -> None:
        deps = _paper_imports()
        self._jax = deps["jax"]
        self._jnp = deps["jnp"]
        self._flax = deps["flax"]
        self.sample_rate = config.sample_rate
        self.num_samples = config.num_samples
        synth_config = deps["SynthConfig"](
            batch_size=config.population_size,
            sample_rate=config.sample_rate,
            buffer_size_seconds=config.duration_seconds,
            control_rate=config.control_rate,
            eps=1e-6,
        )
        single_config = deps["SynthConfig"](
            batch_size=1,
            sample_rate=config.sample_rate,
            buffer_size_seconds=config.duration_seconds,
            control_rate=config.control_rate,
            eps=1e-6,
        )
        self._synth = deps["Voice"](config=synth_config)
        self._single_synth = deps["Voice"](config=single_config)
        self._apply = self._jax.jit(self._synth.apply)
        self._single_apply = self._jax.jit(self._single_synth.apply)

        # These two setup calls deliberately consume keys in the same order as
        # the reference program before any per-prompt run begins.
        self._setup_population = self._synth.init(key_stream.split())
        single_template = self._single_synth.init(key_stream.split())
        flat_single = self._flax.traverse_util.flatten_dict(single_template)
        from .direct import PARAMETER_CONTRACT_HASH, PARAMETER_LAYOUT

        # JAX PyTrees sort dict keys during ravel/unravel; validate the
        # render-time coordinate order rather than Flax's insertion order.
        actual_layout = tuple(
            (tuple(key), int(np.asarray(value).size))
            for key, value in sorted(flat_single.items())
        )
        if actual_layout != PARAMETER_LAYOUT:
            import hashlib
            import json

            actual_hash = hashlib.sha256(
                json.dumps(actual_layout, separators=(",", ":")).encode()
            ).hexdigest()
            raise RuntimeError(
                "SynthAX parameter order changed: "
                f"expected {PARAMETER_CONTRACT_HASH}, found {actual_hash}"
            )
        unbatched = self._flax.traverse_util.unflatten_dict(
            {
                key: np.asarray(value).squeeze()
                for key, value in flat_single.items()
            }
        )
        flat_template, unravel = deps["flatten_util"].ravel_pytree(unbatched)
        self._reshape = self._jax.jit(self._jax.vmap(unravel))
        if int(flat_template.size) != self.parameter_count:
            raise RuntimeError(
                f"expected 78 Voice parameters, found {flat_template.size}"
            )

    def initialize(self, key_stream: JAXKeyStream) -> Any:
        params = self._synth.init(key_stream.split())
        flat = self._flax.traverse_util.flatten_dict(params)
        return self._jnp.concatenate(
            [value.reshape(self._synth.batch_size, -1) for value in flat.values()],
            axis=1,
        )

    def render(self, flat_parameters: Any) -> np.ndarray:
        shaped = self._reshape(flat_parameters)
        return np.asarray(self._apply(shaped), dtype=np.float32)

    def render_one(self, flat_parameters: Any) -> Tuple[np.ndarray, Mapping[str, Any]]:
        flat_parameters = self._jnp.asarray(flat_parameters)
        if flat_parameters.ndim == 1:
            flat_parameters = self._jnp.expand_dims(flat_parameters, axis=0)
        shaped = self._reshape(flat_parameters)
        audio = np.asarray(self._single_apply(shaped), dtype=np.float32).squeeze(0)
        return audio, _to_builtin(shaped)


class EvosaxLES:
    """Learned Evolution Strategy configured with the released hyperparameters."""

    def __init__(self, config: RunConfig) -> None:
        deps = _paper_imports()
        self._jax = deps["jax"]
        self._jnp = deps["jnp"]
        learned_es = deps["LearnedES"]
        self._modern = learned_es is not None
        if self._modern:
            self._strategy = learned_es(
                population_size=config.population_size,
                solution=self._jnp.zeros(
                    (SynthAXVoice.parameter_count,), dtype=self._jnp.float32
                ),
            )
            self._params = self._strategy.default_params.replace(
                std_init=config.sigma_init
            )
        else:
            self._strategy = deps["evosax"].strategies.LES(
                popsize=config.population_size,
                num_dims=SynthAXVoice.parameter_count,
                mean_decay=config.mean_decay,
            )
            self._params = self._strategy.default_params.replace(
                sigma_init=config.sigma_init,
                init_min=0.0,
                init_max=1.0,
                clip_min=0.0,
                clip_max=1.0,
            )
        self._state = None

    def initialize(self, key_stream: JAXKeyStream, initial_population: Any) -> None:
        if self._modern:
            initial_population = self._jnp.asarray(initial_population)
            self._state = self._strategy.init(
                key_stream.split(), initial_population[0], self._params
            )
            # The released CTAG program passed the full initialized SynthAX
            # population as LES's mean. Preserve that behavior even though the
            # modern Evosax API now types the initial mean as one solution.
            self._state = self._state.replace(mean=initial_population)
        else:
            self._state = self._strategy.initialize(
                key_stream.split(), self._params, initial_population
            )

    def ask(self, key_stream: JAXKeyStream) -> Any:
        candidates, self._state = self._strategy.ask(
            key_stream.split(), self._state, self._params
        )
        return self._jnp.clip(candidates, 0.0, 1.0)

    def tell(self, candidates: Any, fitness: np.ndarray) -> None:
        if self._modern:
            self._state, _ = self._strategy.tell(
                self._jax.random.PRNGKey(0),
                candidates,
                self._jnp.asarray(fitness),
                self._state,
                self._params,
            )
            self._state = self._state.replace(
                mean=self._jnp.clip(self._state.mean, 0.0, 1.0),
                std=self._jnp.clip(self._state.std, 0.0, 1.0),
            )
        else:
            self._state = self._strategy.tell(
                candidates, self._jnp.asarray(fitness), self._state, self._params
            )

    @property
    def best_member(self) -> Any:
        if self._modern:
            return self._state.best_solution
        return self._state.best_member

    @property
    def best_fitness(self) -> float:
        return float(self._state.best_fitness)


def build_paper_components(config: RunConfig) -> Tuple[Any, Any, Any, Any]:
    """Build encoder, synth, strategy factory, and shared key stream."""

    keys = JAXKeyStream(config.seed)
    encoder = CLAPEncoder(Path(config.checkpoint), config.device)
    synth = SynthAXVoice(config, keys)
    return encoder, synth, lambda: EvosaxLES(config), keys
