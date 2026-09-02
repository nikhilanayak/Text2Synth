import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software" / "rtl_synth"))

from fixed_model import load_preset, render
from protocol import FRAME_SAMPLES, FrameParser, encode_audio_frame
from export_presets import CONTRACT_HASH, load_parameters
from render_ctag_patch import flatten_sorted_pytree, load_patch


def test_contract_and_generated_presets_match():
    contract = json.loads((ROOT / "hardware" / "synthax-rtl-contract.json").read_text())
    presets = json.loads((ROOT / "rtl" / "synthax" / "assets" / "diagnostic_presets.json").read_text())
    assert contract["contract_sha256"] == presets["contract_sha256"]
    assert len(presets["presets"]) == 8
    assert all(len(item["parameters_q0_16"]) == 78 for item in presets["presets"])
    assert (ROOT / "rtl" / "synthax" / "assets" / "diagnostic_presets_q0_16.hex").read_text().count("\n") == 8*78


def test_protocol_round_trip_incremental_and_resynchronizes():
    samples = tuple(range(-128, 128))
    encoded = encode_audio_frame(65535, samples)
    parser = FrameParser()
    assert parser.feed(b"garbage" + encoded[:17]) == []
    frames = parser.feed(encoded[17:])
    assert frames[0].sequence == 65535
    assert frames[0].samples == samples


def test_protocol_rejects_bad_crc_then_finds_next_frame():
    values = [0] * FRAME_SAMPLES
    damaged = bytearray(encode_audio_frame(1, values))
    damaged[20] ^= 1
    parser = FrameParser()
    frames = parser.feed(bytes(damaged) + encode_audio_frame(2, values))
    assert [frame.sequence for frame in frames] == [2]
    assert parser.crc_errors >= 1


def test_fixed_model_renders_all_diagnostic_presets():
    manifest = ROOT / "rtl" / "synthax" / "assets" / "diagnostic_presets.json"
    for index in range(8):
        audio = render(load_preset(manifest, index), seconds=0.05, gate_seconds=0.03)
        assert audio.shape == (2400,)
        assert audio.dtype.str == "<i2"
        assert abs(audio).max() <= 32767


def test_asset_generator_is_reproducible(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "software" / "rtl_synth" / "generate_assets.py"), "--output", str(tmp_path)],
        check=True,
    )
    for filename in ("sine_q1_17.hex", "midi_phase_inc.hex", "diagnostic_presets_q0_16.hex", "diagnostic_presets.json"):
        assert (tmp_path / filename).read_bytes() == (ROOT / "rtl" / "synthax" / "assets" / filename).read_bytes()


def test_preset_export_accepts_lists_and_rejects_stale_contract(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps([0.5] * 78))
    name, values = load_parameters(raw)
    assert name == "raw" and values == [0.5] * 78
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({
        "prompt": "old", "parameters": [0.5] * 78,
        "parameter_contract_hash": "not-" + CONTRACT_HASH,
    }))
    with pytest.raises(ValueError, match="incompatible"):
        load_parameters(stale)


def test_native_ctag_patch_flattens_in_render_order(tmp_path):
    patch = tmp_path / "patch.yaml"
    patch.write_text("params:\n  b: [0.75]\n  a: [0.25, 0.5]\n")
    assert flatten_sorted_pytree({"params": {"b": [0.75], "a": [0.25, 0.5]}}) == [0.25, 0.5, 0.75]
    with pytest.raises(ValueError, match="78 parameters"):
        load_patch(patch)


def test_rtl_voice_tracks_fixed_point_oracle():
    if not shutil.which("iverilog") or not shutil.which("vvp"):
        import pytest
        pytest.skip("Icarus Verilog is not installed")
    sources = sorted((ROOT / "rtl" / "synthax").glob("*.sv"))
    subprocess.run(
        ["iverilog", "-g2012", "-I", "rtl/synthax", "-s", "synthax_voice_capture_tb",
         "-o", "build/voice_capture.vvp", *map(str, sources), "tb/synthax_voice_capture_tb.sv"],
        cwd=ROOT, check=True,
    )
    subprocess.run(["vvp", "build/voice_capture.vvp"], cwd=ROOT, check=True)
    words = [int(line, 16) for line in (ROOT / "build" / "voice_capture.hex").read_text().splitlines()]
    rtl = np.asarray([word-65536 if word & 0x8000 else word for word in words], dtype=float)
    manifest = ROOT / "rtl" / "synthax" / "assets" / "diagnostic_presets.json"
    golden = render(load_preset(manifest, 0), seconds=0.1, gate_seconds=1.0).astype(float)
    # Control-tick startup creates a bounded leading delay. Align within 5 ms.
    correlations = []
    for lag in range(240):
        a, b = rtl[lag:], golden[:len(rtl)-lag]
        correlations.append(float(np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12)))
    assert max(correlations) >= 0.95
