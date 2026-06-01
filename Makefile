# Makefile - Icarus Verilog build/sim helper for the GEMM accelerator RTL.
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

.PHONY: all setup mac_pe fm_phase_accumulator wave clean

all: setup mac_pe fm_phase_accumulator

setup:
	@chmod +x setup.sh
	./setup.sh

# --- MAC processing element ---
mac_pe: $(BUILD)/mac_pe_tb.vvp
	$(VVP) $<

$(BUILD)/mac_pe_tb.vvp: rtl/mac_pe.v tb/mac_pe_tb.v | $(BUILD)
	$(IVERILOG) $(IFLAGS) -o $@ $^

# --- FM Phase Accumulator (Project.md item 2.1) ---
fm_phase_accumulator: $(BUILD)/fm_phase_accumulator_tb.vvp
	$(VVP) $<

$(BUILD)/fm_phase_accumulator_tb.vvp: rtl/fm_phase_accumulator.v tb/fm_phase_accumulator_tb.v | $(BUILD)
	$(IVERILOG) $(IFLAGS) -o $@ $^

$(BUILD):
	mkdir -p $(BUILD)

wave:
	@gtkwave $(BUILD)/fm_phase_accumulator_tb.vcd 2>/dev/null || echo "gtkwave not installed (brew install gtkwave)"

clean:
	rm -rf $(BUILD)
