`timescale 1ns/1ps

module synthax_sine_rom #(
    parameter MEM_FILE = "rtl/synthax/assets/sine_q1_17.hex"
) (
    input  wire [9:0] address,
    output wire signed [17:0] sample
);
    reg [17:0] rom [0:1023];
    initial $readmemh(MEM_FILE, rom);
    assign sample = rom[address];
endmodule
