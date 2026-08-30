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


def _dependency_status() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for package in ("jax", "flax", "evosax", "synthax", "laion-clap", "torch"):
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
