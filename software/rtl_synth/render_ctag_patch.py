#!/usr/bin/env python3
"""Render an existing CTAG patch through the FPGA fixed-point model."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from export_presets import CONTRACT_HASH
from fixed_model import render

CONTRACT = "synthax-voice-render-flat-78-v2"


def flatten_sorted_pytree(value: Any) -> list[float]:
    """Match JAX's sorted mapping-key traversal and row-major leaf flattening."""

    if isinstance(value, dict):
        result: list[float] = []
        for key in sorted(value):
            result.extend(flatten_sorted_pytree(value[key]))
        return result
    array = np.asarray(value, dtype=np.float64)
    return array.reshape(-1).tolist()


def load_patch(path: Path) -> tuple[str, np.ndarray]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "parameters" in payload:
            artifact_hash = payload.get("parameter_contract_hash")
            if artifact_hash is not None and artifact_hash != CONTRACT_HASH:
                raise ValueError("input JSON uses an incompatible parameter contract")
            values = payload["parameters"]
            name = str(payload.get("prompt", path.stem))
        else:
            values = payload
            name = path.stem
    else:
        payload = yaml.safe_load(path.read_text())
        values = flatten_sorted_pytree(payload)
        name = path.parent.name
    parameters = np.asarray(values, dtype=np.float64)
    if parameters.shape != (78,):
        raise ValueError(f"expected 78 parameters, found shape {parameters.shape}")
    if not np.isfinite(parameters).all() or np.any((parameters < 0) | (parameters > 1)):
        raise ValueError("parameters must be finite values in [0, 1]")
    return name, parameters


def write_pcm16(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patch", type=Path, help="CTAG patch.yaml or parameter JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--midi-note", type=int, default=60)
    args = parser.parse_args()
    name, parameters = load_patch(args.patch)
    quantized = np.clip(np.rint(parameters * 65535), 0, 65535).astype(np.int64)
    samples = render(
        quantized, midi_note=args.midi_note, seconds=args.seconds,
        gate_seconds=min(1.0, args.seconds),
    )
    write_pcm16(args.output, samples)
    manifest = {
        "prompt": name,
        "source": str(args.patch),
        "wav": str(args.output),
        "sample_rate_hz": 48_000,
        "sample_count": int(samples.size),
        "parameter_contract": CONTRACT,
        "parameter_contract_hash": CONTRACT_HASH,
        "parameters": parameters.tolist(),
        "parameters_q0_16": quantized.tolist(),
    }
    metadata = args.output.with_suffix(".json")
    metadata.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"wav": str(args.output), "metadata": str(metadata)}, indent=2))


if __name__ == "__main__":
    main()
