"""Training and evaluation for amortized CTAG patch prediction."""

from __future__ import annotations

import json
import math
import os
import random
from html import escape
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .artifacts import write_wav
from .config import PAPER_CHECKPOINT_SHA256, RunConfig
from .direct import (
    DirectModelConfig,
    PARAMETER_CONTRACT_HASH,
    build_direct_model,
    build_patch_surrogate,
    load_model_bundle,
    save_model_bundle,
)
from .distill import AtomicShardStore, DistillationProfile


def _torch() -> Any:
    import torch

    return torch


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def direct_losses(
    predicted: Any,
    teachers: Any,
    text_embeddings: Any,
    surrogate: Any,
) -> tuple[Any, Dict[str, float]]:
    """Semantic, best-teacher, set-coverage, and diversity objectives."""

    torch = _torch()
    functional = torch.nn.functional
    head0 = functional.smooth_l1_loss(predicted[:, 0], teachers[:, 0])
    distances = (predicted[:, :, None] - teachers[:, None]).abs().mean(dim=-1)
    coverage = distances.min(dim=1).values.mean()
    predicted_embeddings = surrogate(predicted.reshape(-1, predicted.shape[-1])).reshape(
        predicted.shape[0], predicted.shape[1], -1
    )
    normalized_text = functional.normalize(text_embeddings, dim=-1)
    semantic = (1.0 - (predicted_embeddings * normalized_text[:, None]).sum(dim=-1)).mean()
    if predicted.shape[1] > 1:
        pairwise = (predicted[:, :, None] - predicted[:, None]).abs().mean(dim=-1)
        mask = ~torch.eye(predicted.shape[1], dtype=torch.bool, device=predicted.device)
        diversity = functional.relu(0.05 - pairwise[:, mask]).mean()
    else:
        diversity = predicted.new_zeros(())
    total = semantic + head0 + 0.5 * coverage + 0.05 * diversity
    return total, {
        "semantic": float(semantic.detach()),
        "head0": float(head0.detach()),
        "coverage": float(coverage.detach()),
        "diversity": float(diversity.detach()),
        "total": float(total.detach()),
    }


def _load_training_arrays(workspace: Path) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    random_data = AtomicShardStore(workspace / "random", "random-patches").load()
    teacher_data = AtomicShardStore(workspace / "teachers", "prompt-teachers").load()
    if not random_data or not teacher_data:
        raise ValueError("distillation data is incomplete; run build-distillation-data first")
    online = workspace / "online-teachers.npz"
    if online.exists():
        with np.load(online, allow_pickle=False) as values:
            teacher_data = {key: values[key] for key in values.files}
    return random_data, teacher_data


def _dataset_manifest_hash(workspace: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for relative in ("random/manifest.json", "teachers/manifest.json", "distillation.json"):
        path = Path(workspace) / relative
        if path.exists():
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _save_online_teachers(workspace: Path, values: Dict[str, np.ndarray]) -> None:
    final = workspace / "online-teachers.npz"
    temporary = workspace / ".online-teachers.npz.part"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, final)


def _sample_indices(torch: Any, length: int, batch: int, device: str) -> Any:
    return torch.randint(0, length, (min(batch, length),), device=device)


def _cosine_schedule(optimizer: Any, step: int, total: int, warmup: int = 2000) -> None:
    base = optimizer.defaults["lr"]
    if step < min(warmup, total // 10):
        scale = (step + 1) / max(1, min(warmup, total // 10))
    else:
        progress = (step - min(warmup, total // 10)) / max(1, total - min(warmup, total // 10))
        scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    for group in optimizer.param_groups:
        group["lr"] = base * scale


def train_direct(
    workspace: Path,
    output: Path,
    profile: DistillationProfile,
    device: str = "cuda",
    resume: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Train the surrogate and direct model with interruption-safe checkpoints."""

    torch = _torch()
    emit = progress or (lambda message: None)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    workspace, output = Path(workspace), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset_manifest_hash = _dataset_manifest_hash(workspace)
    random_data, teacher_data = _load_training_arrays(workspace)
    random_params = torch.as_tensor(random_data["parameters"], dtype=torch.float32, device=device)
    random_embeddings = torch.as_tensor(
        random_data["audio_embeddings"], dtype=torch.float32, device=device
    )
    teacher_params = torch.as_tensor(teacher_data["parameters"], dtype=torch.float32, device=device)
    text_embeddings = torch.as_tensor(
        teacher_data["text_embeddings"], dtype=torch.float32, device=device
    )
    if "splits" in teacher_data:
        train_mask = np.asarray(teacher_data["splits"]) == "train"
        if train_mask.any():
            teacher_params = teacher_params[torch.as_tensor(train_mask, device=device)]
            text_embeddings = text_embeddings[torch.as_tensor(train_mask, device=device)]
    model_config = DirectModelConfig()
    surrogate = build_patch_surrogate().to(device)
    model = build_direct_model(model_config).to(device)
    surrogate_optimizer = torch.optim.AdamW(surrogate.parameters(), lr=3e-4, weight_decay=1e-4)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    state_path = output / "training-state.pt"
    state = {"surrogate_step": 0, "generator_step": 0}
    if resume and state_path.exists():
        loaded = torch.load(state_path, map_location=device, weights_only=False)
        if loaded.get("parameter_contract_hash") != PARAMETER_CONTRACT_HASH:
            raise ValueError("training checkpoint parameter contract mismatch")
        if loaded.get("dataset_manifest_hash") != dataset_manifest_hash:
            raise ValueError("training checkpoint dataset manifest mismatch")
        surrogate.load_state_dict(loaded["surrogate"])
        model.load_state_dict(loaded["model"])
        surrogate_optimizer.load_state_dict(loaded["surrogate_optimizer"])
        model_optimizer.load_state_dict(loaded["model_optimizer"])
        state.update({key: int(loaded[key]) for key in state})
        random.setstate(loaded["python_rng"])
        np.random.set_state(loaded["numpy_rng"])
        torch.set_rng_state(loaded["torch_rng"])
        if torch.cuda.is_available() and loaded.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(loaded["cuda_rng"])

    def save_state() -> None:
        _atomic_torch_save(
            {
                **state,
                "parameter_contract_hash": PARAMETER_CONTRACT_HASH,
                "dataset_manifest_hash": dataset_manifest_hash,
                "surrogate": surrogate.state_dict(),
                "model": model.state_dict(),
                "surrogate_optimizer": surrogate_optimizer.state_dict(),
                "model_optimizer": model_optimizer.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            state_path,
        )

    surrogate.train()
    for step in range(state["surrogate_step"], profile.surrogate_steps):
        indices = _sample_indices(torch, len(random_params), 1024, device)
        prediction = surrogate(random_params[indices])
        target = torch.nn.functional.normalize(random_embeddings[indices], dim=-1)
        loss = (1.0 - (prediction * target).sum(dim=-1)).mean()
        surrogate_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        surrogate_optimizer.step()
        state["surrogate_step"] = step + 1
        if step % 100 == 0:
            _append_jsonl(output / "metrics.jsonl", {"stage": "surrogate", "step": step, "loss": float(loss.detach())})
        if (step + 1) % 500 == 0:
            save_state()
            emit(f"surrogate: {step + 1}/{profile.surrogate_steps}")

    surrogate.eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    model.train()
    last_metrics: Dict[str, float] = {}
    for step in range(state["generator_step"], profile.generator_steps):
        _cosine_schedule(model_optimizer, step, profile.generator_steps)
        indices = _sample_indices(torch, len(teacher_params), 256, device)
        predicted = model(text_embeddings[indices])
        loss, last_metrics = direct_losses(
            predicted, teacher_params[indices], text_embeddings[indices], surrogate
        )
        model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model_optimizer.step()
        state["generator_step"] = step + 1
        if step % 100 == 0:
            _append_jsonl(output / "metrics.jsonl", {"stage": "generator", "step": step, **last_metrics})
        if (step + 1) % 500 == 0:
            save_state()
            emit(f"generator: {step + 1}/{profile.generator_steps}")
    save_state()
    save_model_bundle(
        output / "model",
        model,
        model_config,
        {
            "profile": asdict(profile),
            "clap_checkpoint_sha256": PAPER_CHECKPOINT_SHA256,
            "dataset_manifest_hash": dataset_manifest_hash,
            "training_metrics": last_metrics,
        },
    )
    return {"model": str(output / "model"), "training_metrics": last_metrics, **state}


def improve_teachers(
    workspace: Path,
    bundle: Path,
    checkpoint: Path,
    profile: DistillationProfile,
    round_index: int,
    device: str = "cuda",
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """One real SynthAX/CLAP on-policy scoring round, persisted for retraining."""

    torch = _torch()
    emit = progress or (lambda message: None)
    _, teachers = _load_training_arrays(Path(workspace))
    model, _, _ = load_model_bundle(bundle, device)
    config = RunConfig(
        population_size=profile.population_size,
        checkpoint=str(checkpoint),
        device=device,
        profile="direct-improvement",
    )
    from .paper_backend import build_paper_components

    encoder, synth, _, _ = build_paper_components(config)
    text = np.asarray(teachers["text_embeddings"], dtype=np.float32)
    with torch.inference_mode():
        predicted = model(torch.as_tensor(text, dtype=torch.float32, device=device)).cpu().numpy()
    improved = 0
    online_parameters = []
    online_embeddings = []
    for index in range(len(predicted)):
        candidates = predicted[index]
        padded = np.resize(candidates, (profile.population_size, 78))
        audio = synth.render(padded)[: len(candidates)]
        audio_embeddings = encoder.embed_audio(audio, config.sample_rate)
        online_parameters.append(candidates)
        online_embeddings.append(audio_embeddings)
        scores = audio_embeddings @ text[index]
        combined_params = np.concatenate((teachers["parameters"][index], candidates), axis=0)
        combined_scores = np.concatenate((teachers["scores"][index], scores), axis=0)
        order = np.argsort(combined_scores)[::-1][:8]
        if combined_scores[order[0]] > teachers["scores"][index, 0]:
            improved += 1
        teachers["parameters"][index] = combined_params[order]
        teachers["scores"][index] = combined_scores[order]
        emit(f"improvement scoring: {index + 1}/{len(predicted)}")
    _save_online_teachers(Path(workspace), teachers)
    AtomicShardStore(Path(workspace) / "random", "random-patches").append(
        f"on-policy-{round_index:03d}",
        {
            "parameters": np.concatenate(online_parameters, axis=0).astype(np.float32),
            "audio_embeddings": np.concatenate(online_embeddings, axis=0).astype(np.float16),
        },
    )
    return {"prompts": len(predicted), "improved": improved}


def reset_training_progress(output: Path, workspace: Optional[Path] = None) -> None:
    """Retain learned weights while starting the next scheduled training segment."""

    torch = _torch()
    path = Path(output) / "training-state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["surrogate_step"] = 0
    state["generator_step"] = 0
    if workspace is not None:
        state["dataset_manifest_hash"] = _dataset_manifest_hash(Path(workspace))
    _atomic_torch_save(state, path)


def run_live_training(
    workspace: Path,
    checkpoint: Path,
    profile: DistillationProfile,
    prompts: list[str],
    device: str = "cuda",
    progress: Optional[Callable[[str], None]] = None,
    search_prompts: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Run all resumable data, training, and real-reward improvement stages."""

    from dataclasses import replace

    from .distill import build_distillation_data

    emit = progress or print
    workspace = Path(workspace)
    output = workspace / "training"
    build_distillation_data(
        workspace,
        checkpoint,
        profile,
        prompts,
        device,
        progress=emit,
        search_prompts=search_prompts,
    )
    segment = replace(
        profile,
        surrogate_steps=max(1, math.ceil(profile.surrogate_steps / (profile.improvement_rounds + 1))),
        generator_steps=max(1, math.ceil(profile.generator_steps / (profile.improvement_rounds + 1))),
        improvement_rounds=0,
    )
    live_state_path = workspace / "live-state.json"
    live_state = json.loads(live_state_path.read_text()) if live_state_path.exists() else {
        "trained_segments": 0,
        "improvement_complete": False,
    }
    live_state.setdefault("improvement_complete", False)

    def save_live_state() -> None:
        temporary = live_state_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(live_state, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, live_state_path)

    if live_state["trained_segments"] == 0:
        train_direct(workspace, output, segment, device, resume=True, progress=emit)
        live_state["trained_segments"] = 1
        save_live_state()
    while live_state["trained_segments"] <= profile.improvement_rounds:
        round_index = live_state["trained_segments"] - 1
        if not live_state["improvement_complete"]:
            improve_teachers(
                workspace,
                output / "model",
                checkpoint,
                profile,
                round_index=round_index,
                device=device,
                progress=emit,
            )
            reset_training_progress(output, workspace)
            live_state["improvement_complete"] = True
            save_live_state()
        train_direct(workspace, output, segment, device, resume=True, progress=emit)
        live_state["trained_segments"] += 1
        live_state["improvement_complete"] = False
        save_live_state()
    return {
        "model": str(output / "model"),
        "trained_segments": live_state["trained_segments"],
        "profile": profile.name,
    }


class LiveDirectSynth:
    """Persistent prompt session that pays CLAP and JAX setup costs only once."""

    def __init__(self, bundle: Path, checkpoint: Path, device: str = "cuda") -> None:
        from .direct import DirectPatchPredictor
        from .paper_backend import JAXKeyStream, SynthAXVoice

        self.device = device
        self.checkpoint = Path(checkpoint)
        self.predictor = DirectPatchPredictor.from_bundle(bundle, checkpoint, device)
        self.config = RunConfig(
            population_size=8,
            checkpoint=str(checkpoint),
            device=device,
            profile="direct",
        )
        self.synth = SynthAXVoice(self.config, JAXKeyStream(2468))

    def generate(
        self,
        prompt: str,
        output: Path,
        selection: str = "direct",
        include_variants: bool = False,
    ) -> Dict[str, Any]:
        prediction = self.predictor.predict([prompt])[0]
        selected = 0
        similarity = None
        if selection == "clap":
            audio_batch = self.synth.render(prediction.variants)
            audio_embeddings = self.predictor.text_encoder.embed_audio(
                audio_batch, self.config.sample_rate
            )
            text_embedding = self.predictor.text_encoder.embed_text([prompt])[0]
            scores = audio_embeddings @ text_embedding
            selected = int(np.argmax(scores))
            similarity = float(scores[selected])
            audio, patch = self.synth.render_one(prediction.variants[selected])
        elif selection == "direct":
            audio, patch = self.synth.render_one(prediction.parameters)
        else:
            raise ValueError("selection must be 'direct' or 'clap'")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        wav = output / f"{hashlib_name(prompt)}.wav"
        write_wav(wav, audio, self.config.sample_rate)
        payload = {
            "prompt": prompt,
            "selection": selection,
            "selected_head": selected,
            "similarity": similarity,
            "parameters": prediction.variants[selected].tolist(),
            "patch": patch,
            "wav": str(wav),
            "timings_seconds": prediction.timings_seconds,
            "parameter_contract_hash": PARAMETER_CONTRACT_HASH,
        }
        if include_variants:
            payload["variants"] = prediction.variants.tolist()
        (output / f"{hashlib_name(prompt)}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        return payload


def infer_direct(
    bundle: Path,
    checkpoint: Path,
    prompt: str,
    output: Path,
    device: str = "cuda",
    selection: str = "direct",
    include_variants: bool = False,
) -> Dict[str, Any]:
    """Run a single prompt through a newly created live session."""

    return LiveDirectSynth(bundle, checkpoint, device).generate(
        prompt, output, selection, include_variants
    )


def export_direct(bundle: Path, output: Path, quantize: bool = False) -> Dict[str, Any]:
    """Export the hardware-facing direct network and verify ONNX parity."""

    torch = _torch()
    import onnx
    import onnxruntime as ort

    model, config, metadata = load_model_bundle(bundle, "cpu")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fp32 = output / "direct_patch_net.onnx"
    sample = torch.linspace(-1.0, 1.0, config.embedding_dim).reshape(1, -1)
    torch.onnx.export(
        model,
        sample,
        fp32,
        input_names=["clap_embedding"],
        output_names=["patches"],
        dynamic_axes={"clap_embedding": {0: "batch"}, "patches": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    expected = model(sample).detach().numpy()
    session = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"clap_embedding": sample.numpy()})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-5:
        raise RuntimeError(f"FP32 ONNX parity failed: max error {max_error}")
    graph = onnx.load(fp32)
    payload: Dict[str, Any] = {
        "fp32": str(fp32),
        "fp32_bytes": fp32.stat().st_size,
        "fp32_max_abs_error": max_error,
        "operator_types": sorted({node.op_type for node in graph.graph.node}),
        "parameter_count": sum(value.numel() for value in model.parameters()),
    }
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8 = output / "direct_patch_net.int8.onnx"
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
        quantized = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
        int8_actual = quantized.run(None, {"clap_embedding": sample.numpy()})[0]
        int8_error = float(np.max(np.abs(expected - int8_actual)))
        if int8_error > 0.02:
            raise RuntimeError(f"INT8 parameter parity failed: max error {int8_error}")
        payload.update({
            "int8": str(int8),
            "int8_bytes": int8.stat().st_size,
            "int8_max_abs_parameter_error": int8_error,
        })
    payload["metadata"] = metadata
    (output / "export.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def evaluate_direct(
    bundle: Path,
    checkpoint: Path,
    prompts: list[str],
    output: Path,
    device: str = "cuda",
    rerank: bool = True,
) -> Dict[str, Any]:
    """Render held-out direct predictions and produce machine/listening artifacts."""

    from .direct import DirectPatchPredictor
    from .paper_backend import JAXKeyStream, SynthAXVoice

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictor = DirectPatchPredictor.from_bundle(bundle, checkpoint, device)
    predictions = predictor.predict(prompts)
    config = RunConfig(population_size=8, checkpoint=str(checkpoint), device=device, profile="direct-eval")
    synth = SynthAXVoice(config, JAXKeyStream(12345))
    rows = []
    for index, prediction in enumerate(predictions):
        audio = synth.render(prediction.variants)
        audio_embeddings = predictor.text_encoder.embed_audio(audio, config.sample_rate)
        text_embedding = predictor.text_encoder.embed_text([prediction.prompt])[0]
        scores = audio_embeddings @ text_embedding
        random_parameters = np.asarray(synth.initialize(JAXKeyStream(index + 9000)), dtype=np.float32)[0]
        random_audio, _ = synth.render_one(random_parameters)
        random_embedding = predictor.text_encoder.embed_audio(random_audio[None], config.sample_rate)[0]
        random_similarity = float(random_embedding @ text_embedding)
        selected = int(np.argmax(scores)) if rerank else 0
        wav = output / f"{index:03d}-{hashlib_name(prediction.prompt)}.wav"
        write_wav(wav, audio[selected], config.sample_rate)
        rows.append({
            "prompt": prediction.prompt,
            "head0_similarity": float(scores[0]),
            "best_of_8_similarity": float(scores.max()),
            "random_similarity": random_similarity,
            "selected_head": selected,
            "wav": str(wav),
            "timings_seconds": prediction.timings_seconds,
            "parameters": prediction.variants[selected].tolist(),
        })
    report = {
        "count": len(rows),
        "median_head0_similarity": float(np.median([row["head0_similarity"] for row in rows])),
        "median_best_of_8_similarity": float(np.median([row["best_of_8_similarity"] for row in rows])),
        "median_random_similarity": float(np.median([row["random_similarity"] for row in rows])),
        "p95_prompt_to_parameters_seconds": float(np.percentile([
            sum(row["timings_seconds"].values()) for row in rows
        ], 95)),
        "results": rows,
    }
    report["checks"] = {
        "head0_beats_random_by_0.05": report["median_head0_similarity"] >= report["median_random_similarity"] + 0.05,
        "prompt_to_parameters_under_100ms": report["p95_prompt_to_parameters_seconds"] < 0.1,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    cards = "\n".join(
        f"<article><h2>{escape(row['prompt'])}</h2><audio controls "
        f"src='{escape(Path(row['wav']).name)}'></audio>"
        f"<p>head0={row['head0_similarity']:.4f}; best8={row['best_of_8_similarity']:.4f}; "
        f"random={row['random_similarity']:.4f}</p></article>" for row in rows
    )
    (output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>CTAG direct listening grid</title>"
        "<h1>CTAG direct listening grid</h1>" + cards + "\n"
    )
    return report


def hashlib_name(value: str) -> str:
    import hashlib

    slug = "-".join(value.lower().split())[:40].strip("-") or "prompt"
    return f"{slug}-{hashlib.sha256(value.encode()).hexdigest()[:8]}"
