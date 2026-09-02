# CLAP-to-SynthAX system architecture

The project has two cleanly separated halves. The ML system turns a prompt into
one ordered 78-value SynthAX patch. The FPGA instrument turns that patch and
button events into a continuous 48 kHz audio stream. This boundary lets us keep
improving prompt-to-parameter quality without changing the completed sound
engine.

```text
Training / preset preparation                 Live instrument

prompt -> CLAP -> direct mapper/search         Cyclone IV buttons
                    |                           | notes / preset / edit
                    v                           v
              78 floats [0,1] -> Q0.16 ROM -> patch bank / atomic registers
                                                     |
                   ADSR + LFO + modulation matrix + two VCOs + noise
                                                     |
                                           signed Q1.23 at 48 kHz
                                                     |
                                      FIFO -> framed PCM16 -> JTAG UART
                                                     |
                                          host CRC/gap handling -> speakers
```

The portable RTL under `rtl/synthax` contains the instrument. `rtl/transport`
contains an explicit ready/valid byte boundary and both Intel Avalon JTAG UART
and physical UART transmitters. `rtl/intel` is the thin Cyclone IV shell. The
only board-specific work left is the exact FPGA part, pin map, signal polarity,
and optional Platform Designer wrapper.

The core is monophonic by design for the first hardware closure. Four play
buttons select C4, E4, G4, and C5. Eight patches are compiled into ROM, can be
selected live, and can be edited in volatile RAM using the seven-segment UI.

The neural handoff is deliberately narrow: 7-bit address, 16-bit Q0.16 data,
write enable, and commit. A future INT8 mapper can fill the shadow patch and
commit all 78 controls at a control-tick boundary, without coupling neural
latency or memory layout to the DSP datapath.

See `hardware/README.md` for protocol, controls, validation, and bring-up.
