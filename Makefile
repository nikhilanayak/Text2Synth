# Makefile - Icarus Verilog build/sim helpers.
#
# Usage:
#   make            # build + run all testbenches
#   make mac_pe     # build + run the MAC PE testbench
#   make wave       # open the last VCD in gtkwave (if installed)
#   make clean      # remove build artifacts

IVERILOG := iverilog
VVP      := vvp
IFLAGS   := -Wall -g2012
BUILD    := build

.PHONY: all setup legacy synthax synthax_patch synthax_framer synthax_instrument synthax_voice synthax_top python_tests wave clean

all: legacy synthax python_tests

setup:
	@chmod +x setup.sh
	./setup.sh

# --- MAC processing element ---
legacy: mac_pe fm_phase_accumulator

mac_pe: $(BUILD)/mac_pe_tb.vvp
	$(VVP) $<

$(BUILD)/mac_pe_tb.vvp: rtl/mac_pe.v tb/mac_pe_tb.v | $(BUILD)
	$(IVERILOG) $(IFLAGS) -o $@ $^

# --- FM Phase Accumulator (Project.md item 2.1) ---
fm_phase_accumulator: $(BUILD)/fm_phase_accumulator_tb.vvp
	$(VVP) $<

$(BUILD)/fm_phase_accumulator_tb.vvp: rtl/fm_phase_accumulator.v tb/fm_phase_accumulator_tb.v | $(BUILD)
	$(IVERILOG) $(IFLAGS) -o $@ $^

# --- SynthAX-compatible monophonic instrument and host transport ---
SYNTHAX_RTL := $(sort $(wildcard rtl/synthax/*.sv))
TRANSPORT_RTL := $(sort $(wildcard rtl/transport/*.sv))

synthax: synthax_patch synthax_framer synthax_instrument synthax_voice synthax_top

synthax_patch: $(BUILD)/synthax_patch_bank_tb.vvp
	$(VVP) $<

$(BUILD)/synthax_patch_bank_tb.vvp: $(SYNTHAX_RTL) tb/synthax_patch_bank_tb.sv | $(BUILD)
	$(IVERILOG) $(IFLAGS) -I rtl/synthax -s synthax_patch_bank_tb -o $@ $^

synthax_framer: $(BUILD)/pcm_framer_tb.vvp
	$(VVP) $<

$(BUILD)/pcm_framer_tb.vvp: $(TRANSPORT_RTL) tb/pcm_framer_tb.sv | $(BUILD)
	$(IVERILOG) $(IFLAGS) -s pcm_framer_tb -o $@ $^

synthax_instrument: $(BUILD)/synthax_instrument_tb.vvp
	$(VVP) $<

$(BUILD)/synthax_instrument_tb.vvp: $(SYNTHAX_RTL) tb/synthax_instrument_tb.sv | $(BUILD)
	$(IVERILOG) $(IFLAGS) -I rtl/synthax -s synthax_instrument_tb -o $@ $^

synthax_voice: $(BUILD)/synthax_voice_capture.vvp
	$(VVP) $<

$(BUILD)/synthax_voice_capture.vvp: $(SYNTHAX_RTL) tb/synthax_voice_capture_tb.sv | $(BUILD)
	$(IVERILOG) $(IFLAGS) -I rtl/synthax -s synthax_voice_capture_tb -o $@ $^

synthax_top: $(BUILD)/cyclone_iv_synthax_top.vvp

$(BUILD)/cyclone_iv_synthax_top.vvp: $(SYNTHAX_RTL) $(TRANSPORT_RTL) rtl/intel/seven_segment_hex.sv rtl/intel/cyclone_iv_generic_top.sv | $(BUILD)
	$(IVERILOG) $(IFLAGS) -I rtl/synthax -s cyclone_iv_synthax_top -o $@ $^

python_tests:
	python3 -m pytest -q tests/test_rtl_synth.py

$(BUILD):
	mkdir -p $(BUILD)

wave:
	@gtkwave $(BUILD)/fm_phase_accumulator_tb.vcd 2>/dev/null || echo "gtkwave not installed (brew install gtkwave)"

clean:
	rm -rf $(BUILD)
