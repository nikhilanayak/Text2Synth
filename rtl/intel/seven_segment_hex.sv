`timescale 1ns/1ps

// Eight independent hexadecimal digits. Segment order is {g,f,e,d,c,b,a};
// ACTIVE_LOW=1 matches the common-anode displays on many Cyclone IV kits.
module seven_segment_hex #(
    parameter integer ACTIVE_LOW = 1
) (
    input  wire [31:0] value,
    output wire [55:0] segments
);
    function automatic [6:0] decode;
        input [3:0] nibble;
        begin
            case (nibble)
                4'h0: decode=7'b0111111; 4'h1: decode=7'b0000110;
                4'h2: decode=7'b1011011; 4'h3: decode=7'b1001111;
                4'h4: decode=7'b1100110; 4'h5: decode=7'b1101101;
                4'h6: decode=7'b1111101; 4'h7: decode=7'b0000111;
                4'h8: decode=7'b1111111; 4'h9: decode=7'b1101111;
                4'ha: decode=7'b1110111; 4'hb: decode=7'b1111100;
                4'hc: decode=7'b0111001; 4'hd: decode=7'b1011110;
                4'he: decode=7'b1111001; default: decode=7'b1110001;
            endcase
        end
    endfunction
    genvar digit;
    generate
        for (digit=0; digit<8; digit=digit+1) begin : digits
            wire [6:0] lit = decode(value[digit*4 +: 4]);
            assign segments[digit*7 +: 7] = ACTIVE_LOW ? ~lit : lit;
        end
    endgenerate
endmodule
