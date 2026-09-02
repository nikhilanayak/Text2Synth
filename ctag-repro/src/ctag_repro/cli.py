"""Command-line entry points for setup, generation, comparison, and profiling."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import wave
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .artifacts import sha256_file
from .config import PAPER_CHECKPOINT_SHA256, PAPER_CHECKPOINT_URL, RunConfig
from .pipeline import CTAGPipeline
from .prompts import load_prompts


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dependency_status() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for package in (
        "numpy",
        "jax",
        "jaxlib",
        "flax",
        "evosax",
        "synthax",
        "laion-clap",
        "torch",
        "torchvision",
        "transformers",
        "numba",
    ):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "missing"
    return result


def download_checkpoint(destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        PAPER_CHECKPOINT_URL, headers={"User-Agent": "ctag-repro/0.1"}
    )
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    actual_hash = sha256_file(partial)
    if actual_hash != PAPER_CHECKPOINT_SHA256:
        raise RuntimeError(
            "downloaded checkpoint failed SHA-256 validation: "
            f"expected {PAPER_CHECKPOINT_SHA256}, got {actual_hash}; "
            f"partial file retained at {partial}"
        )
    partial.replace(destination)
    return destination


def _base_config(args: argparse.Namespace) -> RunConfig:
    if getattr(args, "config", None):
        config = RunConfig.from_yaml(Path(args.config))
    elif args.profile == "smoke":
        config = RunConfig.smoke()
    else:
        config = RunConfig.paper()
    overrides: Dict[str, Any] = {}
    for argument, field in (
        ("device", "device"),
        ("checkpoint", "checkpoint"),
        ("output", "output_root"),
        ("population", "population_size"),
        ("iterations", "iterations"),
        ("runs", "runs_per_prompt"),
        ("seed", "seed"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            overrides[field] = value
    values = config.to_dict()
    values.update(overrides)
    return RunConfig(**values)


def _prompt_source(args: argparse.Namespace) -> Iterable[str]:
    if args.prompts_file:
        return load_prompts(Path(args.prompts_file))
    if args.prompt:
        return args.prompt
    raise SystemExit("provide --prompt or --prompts-file")


def command_setup(args: argparse.Namespace) -> int:
    checkpoint = Path(args.checkpoint)
    if args.download:
        print(f"Downloading {PAPER_CHECKPOINT_URL}")
        download_checkpoint(checkpoint, force=args.force)
    actual_hash = sha256_file(checkpoint) if checkpoint.is_file() else None
    try:
        from .paper_backend import _paper_imports

        _paper_imports()
        import_error = None
    except Exception as exc:  # setup must report broken optional stacks cleanly
        import_error = f"{type(exc).__name__}: {exc}"
    payload = {
        "python": sys.version.split()[0],
        "dependencies": _dependency_status(),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        "checkpoint_sha256": actual_hash,
        "checkpoint_valid": actual_hash == PAPER_CHECKPOINT_SHA256,
        "paper_backend_import_error": import_error,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    ready = payload["checkpoint_valid"] and import_error is None and all(
        value != "missing" for value in payload["dependencies"].values()
    )
    return 0 if ready or not args.strict else 1


def command_doctor(args: argparse.Namespace) -> int:
    """Report whether package and accelerator versions match modern Colab."""

    from .runtime import runtime_report

    payload = runtime_report(require_gpu=args.require_gpu)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["compatible"] else 1


def _run_pipeline(args: argparse.Namespace, force_paper: bool = False) -> int:
    if force_paper:
        args.profile = "paper"
    config = _base_config(args)
    prompts = _prompt_source(args)
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    pipeline = CTAGPipeline.paper(config)
    pipeline.progress_callback = lambda prompt, current, total, fitness: print(
        f"[{prompt}] {current}/{total} best_fitness={fitness:.6f}",
        file=sys.stderr,
        flush=True,
    )
    results = pipeline.run(prompts, write_artifacts=not args.no_artifacts)
    summary = [
        {
            "prompt": result.prompt,
            "run_index": result.run_index,
            "best_fitness": result.best_fitness,
            "best_similarity": result.best_similarity,
            "artifact_dir": str(result.artifact_dir) if result.artifact_dir else None,
            "timings_seconds": result.timings,
        }
        for result in results
    ]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"expected mono PCM16 WAV: {path}")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(
            np.float32
        ) / 32767.0


def command_compare(args: argparse.Namespace) -> int:
    reference = Path(args.reference)
    candidate = Path(args.candidate)
    ref_audio = _read_wav(reference / "best.wav")
    new_audio = _read_wav(candidate / "best.wav")
    if ref_audio.shape != new_audio.shape:
        raise SystemExit(f"audio shapes differ: {ref_audio.shape} vs {new_audio.shape}")
    error = ref_audio - new_audio
    ref_metadata = json.loads((reference / "metadata.json").read_text())
    new_metadata = json.loads((candidate / "metadata.json").read_text())
    cosine = float(
        np.clip(
            np.dot(ref_audio, new_audio)
            / max(np.linalg.norm(ref_audio) * np.linalg.norm(new_audio), 1e-12),
            -1.0,
            1.0,
        )
    )
    report = {
        "audio_cosine_similarity": cosine,
        "audio_max_abs_error": float(np.max(np.abs(error))),
        "audio_rmse": float(np.sqrt(np.mean(error**2))),
        "reference_best_fitness": ref_metadata["best_fitness"],
        "candidate_best_fitness": new_metadata["best_fitness"],
        "fitness_delta": new_metadata["best_fitness"] - ref_metadata["best_fitness"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _direct_prompts(args: argparse.Namespace, regression: bool = False) -> list[str]:
    if getattr(args, "prompts_file", None):
        return load_prompts(Path(args.prompts_file))
    if getattr(args, "prompt", None):
        return load_prompts(args.prompt)
    if regression:
        evaluation = _project_root() / "data" / "direct-eval-prompts.txt"
        return load_prompts(evaluation if evaluation.exists() else _project_root() / "data" / "regression-prompts.txt")
    from .distill import full_training_prompts

    workspace = Path(getattr(args, "workspace", "direct-workspace"))
    return full_training_prompts(_project_root(), workspace / "vocabulary")


def command_build_distillation_data(args: argparse.Namespace) -> int:
    from .distill import build_distillation_data, default_training_prompts, get_profile

    result = build_distillation_data(
        Path(args.workspace),
        Path(args.checkpoint),
        get_profile(args.profile),
        _direct_prompts(args),
        args.device,
        args.seed,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
        search_prompts={
            prompt.casefold() for prompt in default_training_prompts(_project_root())
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_train_direct(args: argparse.Namespace) -> int:
    from .direct_training import train_direct
    from .distill import get_profile

    result = train_direct(
        Path(args.workspace),
        Path(args.output or Path(args.workspace) / "training"),
        get_profile(args.profile),
        args.device,
        resume=not args.no_resume,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_train_live(args: argparse.Namespace) -> int:
    from .direct_training import run_live_training
    from .distill import default_training_prompts, get_profile

    result = run_live_training(
        Path(args.workspace),
        Path(args.checkpoint),
        get_profile(args.profile),
        _direct_prompts(args),
        args.device,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
        search_prompts={
            prompt.casefold() for prompt in default_training_prompts(_project_root())
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_infer_direct(args: argparse.Namespace) -> int:
    from .direct_training import infer_direct

    payload = infer_direct(
        Path(args.bundle),
        Path(args.checkpoint),
        args.prompt,
        Path(args.output),
        args.device,
        args.selection,
        include_variants=args.variants == 8,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_evaluate_direct(args: argparse.Namespace) -> int:
    from .direct_training import evaluate_direct

    payload = evaluate_direct(
        Path(args.bundle),
        Path(args.checkpoint),
        _direct_prompts(args, regression=True),
        Path(args.output),
        args.device,
        rerank=not args.no_rerank,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_export_direct(args: argparse.Namespace) -> int:
    from .direct_training import export_direct

    payload = export_direct(Path(args.bundle), Path(args.output), args.quantize)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", action="append", help="prompt text; repeat as needed")
    prompts.add_argument("--prompts-file", help="one prompt per line")
    parser.add_argument("--profile", choices=("paper", "smoke"), default="smoke")
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--population", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--runs",
        type=int,
        help="independent optimizer runs per prompt (the paper auditioned 10)",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-artifacts", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctag", description="Reproduce CTAG text-to-synth optimization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="check dependencies and checkpoint")
    setup.add_argument("--checkpoint", default="checkpoints/630k-audioset-best.pt")
    setup.add_argument("--download", action="store_true")
    setup.add_argument("--force", action="store_true")
    setup.add_argument("--strict", action="store_true")
    setup.set_defaults(handler=command_setup)

    doctor = subparsers.add_parser(
        "doctor", help="validate the current Colab numerical and GPU runtime"
    )
    doctor.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail unless both JAX and PyTorch can use the GPU",
    )
    doctor.set_defaults(handler=command_doctor)

    generate = subparsers.add_parser("generate", help="run CTAG optimization")
    _add_run_arguments(generate)
    generate.set_defaults(handler=_run_pipeline)

    reproduce = subparsers.add_parser(
        "reproduce", help="run the full configuration released with the paper"
    )
    _add_run_arguments(reproduce)
    reproduce.set_defaults(handler=lambda args: _run_pipeline(args, force_paper=True))

    profile = subparsers.add_parser("profile", help="run with component timings")
    _add_run_arguments(profile)
    profile.set_defaults(handler=_run_pipeline)

    compare = subparsers.add_parser("compare", help="compare two artifact directories")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.set_defaults(handler=command_compare)

    def add_direct_runtime(target: argparse.ArgumentParser) -> None:
        target.add_argument("--checkpoint", default="checkpoints/630k-audioset-best.pt")
        target.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")

    def add_training_prompts(target: argparse.ArgumentParser) -> None:
        prompts = target.add_mutually_exclusive_group()
        prompts.add_argument("--prompt", action="append")
        prompts.add_argument("--prompts-file")

    build_data = subparsers.add_parser(
        "build-distillation-data", help="generate resumable random and CTAG teacher data"
    )
    build_data.add_argument("--workspace", required=True)
    build_data.add_argument("--profile", choices=("smoke", "balanced", "quality"), default="balanced")
    build_data.add_argument("--seed", type=int, default=0)
    build_data.add_argument("--resume", action="store_true", help="accepted for explicit restart-safe scripts; resume is always safe")
    add_training_prompts(build_data)
    add_direct_runtime(build_data)
    build_data.set_defaults(handler=command_build_distillation_data)

    train = subparsers.add_parser("train-direct", help="train the direct patch model")
    train.add_argument("--workspace", required=True)
    train.add_argument("--output")
    train.add_argument("--profile", choices=("smoke", "balanced", "quality"), default="balanced")
    train.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    train.add_argument("--no-resume", action="store_true")
    train.add_argument("--resume", dest="no_resume", action="store_false", default=False)
    train.set_defaults(handler=command_train_direct)

    live = subparsers.add_parser("train-live", help="run the complete resumable distillation workflow")
    live.add_argument("--workspace", required=True)
    live.add_argument("--profile", choices=("smoke", "balanced", "quality"), default="balanced")
    live.add_argument("--resume", action="store_true", help="resume completed shards and checkpoints")
    add_training_prompts(live)
    add_direct_runtime(live)
    live.set_defaults(handler=command_train_live)

    infer = subparsers.add_parser("infer-direct", help="predict and render a patch without search")
    infer.add_argument("--bundle", required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--output", default="direct-output")
    infer.add_argument("--selection", choices=("direct", "clap"), default="direct")
    infer.add_argument("--variants", type=int, choices=(1, 8), default=1)
    add_direct_runtime(infer)
    infer.set_defaults(handler=command_infer_direct)

    evaluate = subparsers.add_parser("evaluate-direct", help="score direct inference on held-out prompts")
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument("--prompts-file")
    evaluate.add_argument("--output", default="direct-evaluation")
    evaluate.add_argument("--no-rerank", action="store_true")
    add_direct_runtime(evaluate)
    evaluate.set_defaults(handler=command_evaluate_direct)

    export = subparsers.add_parser("export-direct", help="export the direct model to ONNX")
    export.add_argument("--bundle", required=True)
    export.add_argument("--output", default="direct-export")
    export.add_argument("--quantize", action="store_true")
    export.set_defaults(handler=command_export_direct)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
