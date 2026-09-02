`timescale 1ns/1ps

module synthax_control_interp (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               sample_tick,
    input  wire               control_tick,
    input  wire signed [18:0] target,
    output reg  signed [18:0] value
);
    reg signed [25:0] accumulator;
    reg signed [25:0] step;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
            step <= 0;
            value <= 0;
        end else begin
            if (control_tick)
                step <= (($signed(target) <<< 7) - accumulator) / 100;
            if (sample_tick) begin
                accumulator <= accumulator + step;
                value <= (accumulator + step) >>> 7;
            end
        end
    end
endmodule
