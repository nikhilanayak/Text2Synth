#!/usr/bin/env python3
"""Generate deterministic ROMs and diagnostic SynthAX patch presets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PARAMETER_COUNT = 78
PRESET_NAMES = ("sine", "square", "saw", "noise", "pluck", "pad", "vibrato", "mod-sweep")


def q016(value: float) -> int:
    return min(65535, max(0, round(value * 65535.0)))


def adsr(attack: float, decay: float, sustain: float, release: float, alpha: float = 1.0) -> list[float]:
    return [
        (alpha - 0.1) / 5.9,
        math.sqrt(max(attack, 0.0) / 2.0),
        math.sqrt(max(decay, 0.0) / 2.0),
        math.sqrt(max(release, 0.0) / 5.0),
        sustain,
    ]


def lfo(frequency: float = 0.0, depth: float = 0.5, shape: str = "sin") -> list[float]:
    # Effective JAX PyTree order: rsaw, saw, sin, sqr, tri.
    weights = {name: float(name == shape) for name in ("rsaw", "saw", "sin", "sqr", "tri")}
    return [
        (frequency / 20.0) ** 0.25 if frequency else 0.0,
        0.5,
        depth,
        *(weights[name] for name in ("rsaw", "saw", "sin", "sqr", "tri")),
    ]


def base_patch() -> np.ndarray:
    p = np.zeros(PARAMETER_COUNT, dtype=np.float64)
    p[0:5] = adsr(0.01, 0.2, 0.75, 0.25)
    p[5:10] = adsr(0.01, 0.2, 0.75, 0.25)
    p[10] = math.sqrt((1.0 - 0.01) / (4.0 - 0.01))
    p[11] = 60.0 / 127.0
    p[12:20] = lfo()
    p[20:25] = adsr(0.02, 0.15, 0.8, 0.2)
    p[25:30] = adsr(0.05, 0.1, 0.8, 0.2)
    p[30:38] = lfo()
    p[38:43] = adsr(0.02, 0.15, 0.8, 0.2)
    p[43:48] = adsr(0.05, 0.1, 0.8, 0.2)
    p[48:51] = [1.0, 0.0, 0.0]
    # Modulation matrix is five row-major outputs by four inputs.
    p[51 + 1 * 4 + 0] = 1.0  # ADSR 1 -> VCO 1 amplitude
    p[51 + 3 * 4 + 1] = 1.0  # ADSR 2 -> VCO 2 amplitude
    p[71:78] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5]
    return p


def diagnostic_presets() -> list[np.ndarray]:
    presets: list[np.ndarray] = []
    sine = base_patch()
    presets.append(sine)

    square = base_patch()
    square[76] = 0.0
    square[48:51] = [0.0, 0.8, 0.0]
    presets.append(square)

    saw = square.copy()
    saw[76] = 1.0
    presets.append(saw)

    noise = base_patch()
    noise[51:71] = 0.0
    noise[51 + 4 * 4 + 0] = 1.0
    noise[48:51] = [0.0, 0.0, 1.0]
    presets.append(noise)

    pluck = base_patch()
    pluck[0:5] = adsr(0.002, 0.18, 0.0, 0.12, 1.6)
    pluck[5:10] = adsr(0.002, 0.1, 0.0, 0.08, 1.4)
    pluck[76] = 0.75
    pluck[48:51] = [0.7, 0.35, 0.0]
    presets.append(pluck)

    pad = base_patch()
    pad[0:5] = adsr(0.7, 0.5, 0.8, 1.2, 0.8)
    pad[5:10] = adsr(0.9, 0.4, 0.65, 1.4, 0.8)
    pad[76] = 0.65
    pad[48:51] = [0.55, 0.35, 0.0]
    presets.append(pad)

    vibrato = base_patch()
    vibrato[12:20] = lfo(5.5, 0.54, "sin")
    vibrato[20:25] = adsr(0.1, 0.1, 0.8, 0.2)
    vibrato[51 + 0 * 4 + 2] = 0.16
    presets.append(vibrato)

    sweep = base_patch()
    sweep[12:20] = lfo(0.35, 0.64, "saw")
    sweep[51 + 0 * 4 + 2] = 0.45
    sweep[51 + 2 * 4 + 2] = 0.35
    sweep[76] = 0.8
    sweep[48:51] = [0.55, 0.45, 0.0]
    presets.append(sweep)
    return presets


def write_hex(path: Path, values: list[int], width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digits = (width + 3) // 4
    path.write_text("".join(f"{value & ((1 << width) - 1):0{digits}x}\n" for value in values))


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sine = [round(math.sin(2 * math.pi * index / 1024) * 131071) for index in range(1024)]
    write_hex(output / "sine_q1_17.hex", sine, 18)

    # Address is MIDI pitch in unsigned Q7.5, clamped to 0..127.96875.
    phase = []
    for address in range(4096):
        midi = address / 32.0
        hz = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
        phase.append(round(hz * (2**32) / 48_000.0))
    write_hex(output / "midi_phase_inc.hex", phase, 32)

    presets = diagnostic_presets()
    write_hex(output / "diagnostic_presets_q0_16.hex", [q016(x) for p in presets for x in p], 16)
    manifest = {
        "format_version": 1,
        "contract": "synthax-voice-render-flat-78-v2",
        "contract_sha256": "4a07c1ca91590e8a6f0b781057928c75c383b3fdad8346fe2d67dab7d4e2cac7",
        "parameter_format": "Q0.16",
        "presets": [
            {"index": index, "name": name, "parameters_q0_16": [q016(x) for x in patch]}
            for index, (name, patch) in enumerate(zip(PRESET_NAMES, presets))
        ],
    }
    (output / "diagnostic_presets.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("rtl/synthax/assets"))
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
