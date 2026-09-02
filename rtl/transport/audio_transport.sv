`timescale 1ns/1ps

module synthax_audio_transport (
    input  wire clk,
    input  wire rst_n,
    input  wire signed [23:0] sample,
    input  wire sample_valid,
    output wire [7:0] stream_data,
    output wire stream_valid,
    input  wire stream_ready,
    output wire [31:0] overflow_count
);
    wire signed [15:0] fifo_front;
    wire fifo_empty, fifo_full, fifo_pop;
    wire [10:0] fifo_level;
    audio_sample_fifo fifo(
        clk,rst_n,sample_valid,sample,fifo_pop,fifo_front,fifo_empty,
        fifo_full,fifo_level,overflow_count
    );
    pcm_framer framer(
        clk,rst_n,fifo_front,fifo_level,fifo_pop,stream_data,stream_valid,stream_ready
    );
endmodule
