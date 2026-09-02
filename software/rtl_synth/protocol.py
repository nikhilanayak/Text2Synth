"""Binary CTAG PCM transport shared by tests and the host player."""

from __future__ import annotations

import dataclasses
import struct
from collections.abc import Iterable

MAGIC = b"CTAG"
VERSION = 1
AUDIO_FRAME = 1
FRAME_SAMPLES = 256


def crc16_ccitt(data: bytes, prior: int = 0xFFFF) -> int:
    value = prior
    for byte in data:
        value ^= byte << 8
        for _ in range(8):
            value = ((value << 1) ^ 0x1021) & 0xFFFF if value & 0x8000 else (value << 1) & 0xFFFF
    return value


def encode_audio_frame(sequence: int, samples: Iterable[int]) -> bytes:
    values = list(samples)
    if len(values) != FRAME_SAMPLES:
        raise ValueError(f"audio frames require exactly {FRAME_SAMPLES} samples")
    header = struct.pack("<BBHH", VERSION, AUDIO_FRAME, sequence & 0xFFFF, len(values))
    payload = struct.pack(f"<{len(values)}h", *values)
    body = header + payload
    return MAGIC + body + struct.pack("<H", crc16_ccitt(body))


@dataclasses.dataclass(frozen=True)
class AudioFrame:
    sequence: int
    samples: tuple[int, ...]


class FrameParser:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.crc_errors = 0

    def feed(self, data: bytes) -> list[AudioFrame]:
        self.buffer.extend(data)
        result: list[AudioFrame] = []
        while True:
            start = self.buffer.find(MAGIC)
            if start < 0:
                del self.buffer[:-3]
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer) < 12:
                break
            version, kind, sequence, count = struct.unpack_from("<BBHH", self.buffer, 4)
            length = 4 + 6 + count * 2 + 2
            if len(self.buffer) < length:
                break
            body = bytes(self.buffer[4:-2] if len(self.buffer) == length else self.buffer[4:length-2])
            expected = struct.unpack_from("<H", self.buffer, length - 2)[0]
            if version != VERSION or kind != AUDIO_FRAME or count > 4096 or crc16_ccitt(body) != expected:
                self.crc_errors += 1
                del self.buffer[0]
                continue
            samples = struct.unpack_from(f"<{count}h", self.buffer, 10)
            result.append(AudioFrame(sequence, tuple(samples)))
            del self.buffer[:length]
        return result
