`timescale 1ns/1ps

module synthax_uart_tx #(
    parameter integer CLOCK_HZ = 50_000_000,
    parameter integer BAUD = 3_000_000
) (
    input  wire clk,
    input  wire rst_n,
    input  wire [7:0] stream_data,
    input  wire stream_valid,
    output wire stream_ready,
    output reg  tx
);
    reg [31:0] baud_accumulator;
    reg [9:0] shift;
    reg [3:0] bits_remaining;
    assign stream_ready = bits_remaining == 0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            baud_accumulator <= 0;
            shift <= 10'h3ff;
            bits_remaining <= 0;
            tx <= 1;
        end else begin
            if (stream_valid && stream_ready) begin
                shift <= {1'b1,stream_data,1'b0};
                bits_remaining <= 10;
                tx <= 0;
                baud_accumulator <= 0;
            end else if (bits_remaining != 0) begin
                if (baud_accumulator + BAUD >= CLOCK_HZ) begin
                    baud_accumulator <= baud_accumulator + BAUD - CLOCK_HZ;
                    shift <= {1'b1,shift[9:1]};
                    bits_remaining <= bits_remaining - 1'b1;
                    tx <= shift[1];
                end else baud_accumulator <= baud_accumulator + BAUD;
            end else tx <= 1;
        end
    end
endmodule
