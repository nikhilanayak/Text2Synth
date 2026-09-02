#!/usr/bin/env python3
"""Convert direct-inference JSON parameter vectors into FPGA preset ROMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONTRACT_HASH = "4a07c1ca91590e8a6f0b781057928c75c383b3fdad8346fe2d67dab7d4e2cac7"


def load_parameters(path: Path) -> tuple[str, list[float]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        artifact_hash = payload.get("parameter_contract_hash")
        if artifact_hash is not None and artifact_hash != CONTRACT_HASH:
            raise ValueError(f"{path} uses an incompatible parameter contract")
        values = payload.get("parameters")
        name = str(payload.get("prompt", path.stem))
    else:
        values = payload
        name = path.stem
    if not isinstance(values, list) or len(values) != 78:
        raise ValueError(f"{path} must contain one 78-value parameters array")
    if any(not 0.0 <= float(value) <= 1.0 for value in values):
        raise ValueError(f"{path} contains a parameter outside [0, 1]")
    return name, [float(x) for x in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) > 8:
        raise SystemExit("at most eight presets fit the first UI contract")
    presets = [load_parameters(path) for path in args.inputs]
    while len(presets) < 8:
        presets.append(presets[-1])
    words = [min(65535, max(0, round(value*65535))) for _, values in presets for value in values]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{word:04x}\n" for word in words))
    manifest = {
        "format_version": 1,
        "contract": "synthax-voice-render-flat-78-v2",
        "contract_sha256": CONTRACT_HASH,
        "parameter_format": "Q0.16",
        "presets": [{"index": index, "name": name} for index, (name, _) in enumerate(presets)],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
