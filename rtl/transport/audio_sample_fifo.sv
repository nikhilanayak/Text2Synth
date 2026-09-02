`timescale 1ns/1ps

module audio_sample_fifo #(
    parameter integer DEPTH = 1024,
    parameter integer ADDRESS_WIDTH = 10
) (
    input  wire clk,
    input  wire rst_n,
    input  wire sample_valid,
    input  wire signed [23:0] sample_q1_23,
    input  wire pop,
    output wire signed [15:0] front,
    output wire empty,
    output wire full,
    output reg  [ADDRESS_WIDTH:0] level,
    output reg  [31:0] overflow_count
);
    reg signed [15:0] memory [0:DEPTH-1];
    reg [ADDRESS_WIDTH-1:0] write_pointer, read_pointer;
    assign empty = level == 0;
    assign full = level == DEPTH;
    assign front = memory[read_pointer];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_pointer <= 0;
            read_pointer <= 0;
            level <= 0;
            overflow_count <= 0;
        end else begin
            case ({sample_valid && !full, pop && !empty})
                2'b10: begin
                    memory[write_pointer] <= sample_q1_23[23:8];
                    write_pointer <= write_pointer + 1'b1;
                    level <= level + 1'b1;
                end
                2'b01: begin
                    read_pointer <= read_pointer + 1'b1;
                    level <= level - 1'b1;
                end
                2'b11: begin
                    memory[write_pointer] <= sample_q1_23[23:8];
                    write_pointer <= write_pointer + 1'b1;
                    read_pointer <= read_pointer + 1'b1;
                end
            endcase
            if (sample_valid && full && !(pop && !empty))
                overflow_count <= overflow_count + 1'b1;
        end
    end
endmodule
