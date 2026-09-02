`timescale 1ns/1ps

module synthax_midi_phase_rom #(
    parameter MEM_FILE = "rtl/synthax/assets/midi_phase_inc.hex"
) (
    input  wire [11:0] midi_q7_5,
    output wire [31:0] phase_increment
);
    reg [31:0] rom [0:4095];
    initial $readmemh(MEM_FILE, rom);
    assign phase_increment = rom[midi_q7_5];
endmodule
