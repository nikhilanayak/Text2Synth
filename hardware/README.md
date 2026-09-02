# SynthAX FPGA instrument

This milestone implements the parameter-to-audio half of the project as a
portable, monophonic SystemVerilog instrument. It consumes the same effective
78-value patch vector used at render time by the CTAG SynthAX `Voice`, produces
48 kHz mono PCM, and streams framed samples to the host computer. It does not
put CLAP or the direct neural mapper in RTL yet.

## User controls

The board interface needs six active-high inputs: an edit-mode switch, a preset
button, and four action buttons. In play mode, the four action buttons gate MIDI
notes 60, 64, 67, and 72. The highest held note wins. The preset button cycles
through eight compiled patches.

In edit mode the action buttons are previous parameter, next parameter,
decrease, and increase. Changes are volatile and use steps of 256 in unsigned
Q0.16. The eight hexadecimal digits alternate once a second between parameter
address and value. Leaving edit mode immediately plays the edited patch. A
preset change reloads that preset's current working copy; reset restores all
compiled values.

Button debouncing is in `button_ui.sv`. Boards with active-low buttons should
invert them only in their pin-level wrapper.

## Audio path

The DSP core uses a 32-bit phase accumulator, a 1024-entry signed Q1.17 sine
ROM, linear fixed-point ADSRs at 480 Hz, interpolated modulation, and signed
Q1.23 internal audio. The output is converted to little-endian PCM16 and put in
256-sample frames:

| Field | Bytes | Meaning |
|---|---:|---|
| Magic | 4 | ASCII `CTAG` |
| Version | 1 | `1` |
| Type | 1 | `1` = audio |
| Sequence | 2 | little-endian, wraps at 65535 |
| Count | 2 | 256 |
| PCM payload | 512 | 256 signed little-endian samples |
| CRC | 2 | CRC16-CCITT over version through payload |

At 48 kHz this requires about 98 kB/s including framing. The 1024-sample FIFO
provides roughly 21 ms of buffering and exposes a cumulative overflow counter.

Intel JTAG UART is the intended transport. `jtag_uart_tx_master.sv` is an
Avalon-MM master for the JTAG UART data/control agent. In Platform Designer,
add it as a custom component, connect its master to a JTAG UART slave, connect
the 50 MHz clock and active-low reset, and export its streaming sink toward the
synth transport. Set the JTAG UART write FIFO to at least 2048 bytes. The
board-neutral top leaves this Avalon interface visible so a generated system
can wrap it.

The independently synthesizable default instead selects the 3 Mbaud physical
UART transmitter. Route `uart_tx` through a 3.3 V USB-UART adapter if the JTAG
path cannot sustain streaming. Both paths carry the identical framed protocol.

On the host:

```bash
python3 -m pip install -r host/requirements.txt
python3 host/play_fpga_audio.py --serial /dev/ttyUSB0 --baud 3000000
```

To use JTAG, compile `host/jtag_audio_bridge.c` with the `jtag_atlantic` header
and library supplied by the installed Quartus version, then run:

```bash
python3 host/play_fpga_audio.py --jtag-bridge ./jtag_audio_bridge
```

The player verifies CRCs, tracks sequence gaps, inserts silence for short gaps,
and sends the PCM to the computer's default audio device.

## Presets and parameter interface

Eight diagnostic patches ship in `rtl/synthax/assets`. Replace them at compile
time with direct-inference JSON files:

```bash
python3 software/rtl_synth/export_presets.py \
  runs/a.json runs/b.json --output rtl/synthax/assets/presets_q0_16.hex
```

Point the patch bank's `PRESET_FILE` parameter at that file. Fewer than eight
inputs repeat the last patch. Inputs must use contract
`synthax-voice-render-flat-78-v2`, hash
`4a07c1ca91590e8a6f0b781057928c75c383b3fdad8346fe2d67dab7d4e2cac7`.

The future neural engine does not need to know the patch RAM implementation.
It writes `external_param_address`, `external_param_data`, and
`external_param_we` for indices 0 through 77, then pulses
`external_param_commit`. The patch becomes active atomically on the next 480 Hz
control boundary.

## Simulate and target Cyclone IV

```bash
make synthax
python3 -m pytest -q tests/test_rtl_synth.py
ctag-repro/.venv/bin/python software/rtl_synth/compare_synthax.py
```

`hardware/intel/cyclone_iv_generic.qsf` targets an EP4CE115F29C7 only as a
compile-time default. Change `DEVICE`, copy the vendor board's clock/button/
switch/seven-segment/UART pin assignments, and confirm button and display
polarity before programming. The top assumes a 50 MHz clock; change `CLOCK_HZ`
and the SDC together for another oscillator. No audio codec, jack, I2S, ADC, or
audio input is required.

Open or compile the QSF from `hardware/intel`; its relative source and ROM-file
assignments are part of the project. The generic project selects physical UART
so it can synthesize without generated Intel IP. A JTAG-UART build should use a
Platform Designer wrapper and set `USE_JTAG_UART=1` on the core instance.

## Deliberate differences from SynthAX/JAX

SynthAX normalizes each rendered buffer by its future peak. A live instrument
cannot do that causally, so RTL uses fixed one-third mixer headroom. ADSR curves
are linear rather than alpha-shaped, LFO rate-envelope modulation is reserved,
LFO shape weights use a linear mix rather than SynthAX's exponentiated selector,
and pseudorandom noise is validated statistically rather than sample-for-sample.
The supplied tonal presets correlate at least 0.97 with the software SynthAX
reference after gain/time alignment; the RTL is bit-checked against
`software/rtl_synth/fixed_model.py`.
