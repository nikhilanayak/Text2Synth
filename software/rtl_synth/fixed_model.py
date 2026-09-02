"""Fixed-point behavioral oracle for the portable SynthAX RTL.

This models the deliberate hardware approximations: linear ADSR ramps,
Q0.16 modulation weights, LUT oscillators, LFSR noise, and fixed 1/3 mixer
headroom. It is the numerical reference for RTL; SynthAX/JAX is the perceptual
reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
CONTROL_RATE = 480


def load_preset(path: Path, index: int) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    return np.asarray(payload["presets"][index]["parameters_q0_16"], dtype=np.int64)


def _time_ticks(value: int, maximum: int) -> int:
    return max(1, value * value * maximum >> 32)


def envelope(params: np.ndarray, gate_ticks: int, total_ticks: int) -> np.ndarray:
    attack = _time_ticks(int(params[1]), 960)
    decay = _time_ticks(int(params[2]), 960)
    sustain = int(params[4])
    release = _time_ticks(int(params[3]), 2400)
    values = np.zeros(total_ticks, dtype=np.int64)
    level = 0
    release_start = min(gate_ticks, total_ticks)
    for tick in range(total_ticks):
        if tick < attack:
            level = min(65535, level + 65535 // attack)
        elif tick < attack + decay:
            level = max(sustain, level - (65535 - sustain) // decay)
        elif tick < release_start:
            level = sustain
        elif tick < release_start + release:
            level = max(0, level - max(1, sustain // release))
        else:
            level = 0
        values[tick] = level
    return values


def lfo(params: np.ndarray, rate_env: np.ndarray, amp_env: np.ndarray) -> np.ndarray:
    del rate_env  # The v1 RTL reserves rate modulation but does not apply it.
    count = len(amp_env)
    normalized = int(params[0]) / 65535.0
    frequency = 20.0 * normalized**4
    phase = np.arange(count) * frequency / CONTROL_RATE + int(params[1]) / 65535.0 - 0.5
    frac = np.mod(phase, 1.0)
    waves = np.stack(
        (
            (1 - np.cos(2 * np.pi * phase)) / 2,
            1 - np.abs(2 * frac - 1),
            frac,
            1 - frac,
            (frac >= 0.5).astype(float),
        )
    )
    weights = params[[5, 7, 4, 3, 6]].astype(float)
    mixed = np.zeros(count) if weights.sum() == 0 else np.sum(weights[:, None] * waves, axis=0) / weights.sum()
    return np.clip(np.rint(mixed * amp_env), 0, 65535).astype(np.int64)


def _upsample(control: np.ndarray, samples: int) -> np.ndarray:
    return np.interp(
        np.arange(samples), np.arange(len(control)) * (SAMPLE_RATE / CONTROL_RATE), control
    )


def render(parameters: np.ndarray, midi_note: int = 60, seconds: float = 2.0, gate_seconds: float = 1.0) -> np.ndarray:
    p = np.asarray(parameters, dtype=np.int64)
    if p.shape != (78,):
        raise ValueError("expected one 78-value Q0.16 patch")
    samples = round(seconds * SAMPLE_RATE)
    controls = max(2, round(seconds * CONTROL_RATE))
    gate_ticks = round(gate_seconds * CONTROL_RATE)
    rate1 = envelope(p[25:30], gate_ticks, controls)
    rate2 = envelope(p[43:48], gate_ticks, controls)
    amp_lfo1 = envelope(p[20:25], gate_ticks, controls)
    amp_lfo2 = envelope(p[38:43], gate_ticks, controls)
    env1 = envelope(p[0:5], gate_ticks, controls)
    env2 = envelope(p[5:10], gate_ticks, controls)
    controls_in = np.stack((env1, env2, lfo(p[12:20], rate1, amp_lfo1), lfo(p[30:38], rate2, amp_lfo2)))
    weights = (p[51:71].reshape(5, 4) ** 2) >> 16
    modulation = (weights @ controls_in) >> 16
    mod_audio = np.stack([_upsample(row, samples) for row in modulation])

    tuning1 = (p[73] - 32768) * 48 / 32768
    tuning2 = (p[77] - 32768) * 48 / 32768
    depth1 = (p[72] - 32768) * 96 / 32768
    depth2 = (p[75] - 32768) * 96 / 32768
    midi1 = np.clip(midi_note + tuning1 + depth1 * mod_audio[0] / 65535, 0, 127.96875)
    midi2 = np.clip(midi_note + tuning2 + depth2 * mod_audio[2] / 65535, 0, 127.96875)
    inc1 = np.rint(440 * 2 ** ((midi1 - 69) / 12) * 2**32 / SAMPLE_RATE).astype(np.uint64)
    inc2 = np.rint(440 * 2 ** ((midi2 - 69) / 12) * 2**32 / SAMPLE_RATE).astype(np.uint64)
    offset1 = np.uint64((int(p[71]) << 16) - 0x80000000 & 0xFFFFFFFF)
    offset2 = np.uint64((int(p[74]) << 16) - 0x80000000 & 0xFFFFFFFF)
    phase1 = (np.cumsum(inc1, dtype=np.uint64) + offset1) & np.uint64(0xFFFFFFFF)
    phase2 = (np.cumsum(inc2, dtype=np.uint64) + offset2) & np.uint64(0xFFFFFFFF)
    sine1 = np.cos(phase1.astype(float) * (2*np.pi/2**32))
    sine2 = np.sin(phase2.astype(float) * (2*np.pi/2**32))
    square = np.where(sine2 < 0, -1.0, 1.0)
    shape = p[76] / 65535
    osc2 = (1-shape/2) * square * (1 + shape*np.cos(phase2.astype(float)*(2*np.pi/2**32)))
    noise = np.empty(samples)
    state = 0x1ACEBEEF
    for index in range(samples):
        bit = ((state >> 31) ^ (state >> 21) ^ (state >> 1) ^ state) & 1
        state = ((state << 1) & 0xFFFFFFFF) | bit
        noise[index] = ((state & 0x3FFFF) - 0x20000) / 0x20000
    mixed = (
        sine1 * mod_audio[1] / 65535 * p[48] / 65535
        + osc2 * mod_audio[3] / 65535 * p[49] / 65535
        + noise * mod_audio[4] / 65535 * p[50] / 65535
    ) / 3.0
    return np.clip(np.rint(mixed * 32767), -32768, 32767).astype("<i2")
