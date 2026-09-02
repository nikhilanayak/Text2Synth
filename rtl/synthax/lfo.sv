`timescale 1ns/1ps

module synthax_lfo (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        control_tick,
    input  wire        gate,
    input  wire [15:0] frequency,
    input  wire [15:0] mod_depth,
    input  wire [15:0] initial_phase,
    input  wire [15:0] weight_sine,
    input  wire [15:0] weight_triangle,
    input  wire [15:0] weight_saw,
    input  wire [15:0] weight_reverse_saw,
    input  wire [15:0] weight_square,
    input  wire [15:0] rate_envelope,
    input  wire [15:0] amplitude_envelope,
    output reg  [15:0] value
);
    reg [31:0] phase;
    reg prior_gate;
    reg [63:0] f2, f4;
    reg [63:0] weighted;
    reg [18:0] weight_sum;
    reg [15:0] sine_u, triangle_u, saw_u, reverse_saw_u, square_u;
    wire signed [17:0] sine_q;
    synthax_sine_rom sine_table(.address(phase[31:22] + 10'd256), .sample(sine_q));
    wire unused_modulation = ^{mod_depth, rate_envelope};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= 0;
            prior_gate <= 0;
            value <= 0;
        end else if (control_tick) begin
            // SynthAX maps normalized LFO frequency with n^4 over 0..20 Hz.
            f2 = frequency * frequency;
            f4 = (f2 >> 16) * (f2 >> 16);
            prior_gate <= gate;
            if (gate && !prior_gate)
                phase <= ({initial_phase,16'b0} - 32'h80000000)
                    + ((f4 >> 16) * 32'd178956971) / 32'd65535;
            else
                phase <= phase + ((f4 >> 16) * 32'd178956971) / 32'd65535;
            // SynthAX's field named "sin" is (cos(argument+pi)+1)/2.
            sine_u = (18'sd131071 - sine_q) >> 2;
            saw_u = phase[31:16];
            reverse_saw_u = 16'hffff - saw_u;
            triangle_u = phase[31] ? (16'hffff - phase[30:15]) : phase[30:15];
            square_u = phase[31] ? 16'hffff : 16'd0;
            weight_sum = weight_sine + weight_triangle + weight_saw
                       + weight_reverse_saw + weight_square;
            weighted = weight_sine * sine_u + weight_triangle * triangle_u
                     + weight_saw * saw_u + weight_reverse_saw * reverse_saw_u
                     + weight_square * square_u;
            if (weight_sum == 0)
                value <= 0;
            else
                value <= ((weighted / weight_sum) * amplitude_envelope) >> 16;
        end
    end
endmodule
