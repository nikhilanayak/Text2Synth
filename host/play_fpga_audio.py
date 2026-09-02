#!/usr/bin/env python3
"""Play framed PCM from an Intel JTAG bridge or a physical serial port."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software" / "rtl_synth"))
from protocol import FRAME_SAMPLES, FrameParser  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--jtag-bridge", type=Path)
    source.add_argument("--serial")
    parser.add_argument("--baud", type=int, default=3_000_000)
    args = parser.parse_args()
    process = None
    if args.jtag_bridge:
        process = subprocess.Popen([str(args.jtag_bridge)], stdout=subprocess.PIPE)
        stream = process.stdout
    else:
        import serial
        stream = serial.Serial(args.serial, args.baud, timeout=0.1)
    parser_state = FrameParser()
    expected = None
    gaps = 0
    started = time.monotonic()
    try:
        with sd.RawOutputStream(
            samplerate=48_000, channels=1, dtype="int16",
            blocksize=FRAME_SAMPLES, latency="low"
        ) as output:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    if process is not None and process.poll() is not None:
                        break
                    continue
                for frame in parser_state.feed(chunk):
                    if expected is not None and frame.sequence != expected:
                        missing = (frame.sequence - expected) & 0xFFFF
                        gaps += missing
                        for _ in range(min(missing, 16)):
                            output.write(bytes(FRAME_SAMPLES * 2))
                    expected = (frame.sequence + 1) & 0xFFFF
                    output.write(np.asarray(frame.samples, dtype="<i2").tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        if process is not None:
            process.terminate()
        elapsed = time.monotonic() - started
        print(f"frames_gaps={gaps} crc_errors={parser_state.crc_errors} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
