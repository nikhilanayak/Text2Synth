`timescale 1ns/1ps

module synthax_tick_generator #(
    parameter integer CLOCK_HZ = 50_000_000,
    parameter integer SAMPLE_HZ = 48_000
) (
    input  wire clk,
    input  wire rst_n,
    output reg  sample_tick,
    output reg  control_tick
);
    reg [31:0] accumulator;
    reg [6:0] sample_in_control;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
            sample_in_control <= 0;
            sample_tick <= 0;
            control_tick <= 0;
        end else begin
            sample_tick <= 0;
            control_tick <= 0;
            if (accumulator + SAMPLE_HZ >= CLOCK_HZ) begin
                accumulator <= accumulator + SAMPLE_HZ - CLOCK_HZ;
                sample_tick <= 1;
                if (sample_in_control == 99) begin
                    sample_in_control <= 0;
                    control_tick <= 1;
                end else sample_in_control <= sample_in_control + 1'b1;
            end else accumulator <= accumulator + SAMPLE_HZ;
        end
    end
endmodule
