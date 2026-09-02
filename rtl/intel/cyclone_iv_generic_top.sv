`timescale 1ns/1ps

// Board-neutral Cyclone IV shell. A board .qsf maps the UI/JTAG/clock ports.
// Platform Designer connects avm_* to an Intel JTAG UART agent.
module cyclone_iv_synthax_top #(
    parameter integer CLOCK_HZ = 50_000_000,
    parameter integer USE_JTAG_UART = 0
) (
    input  wire clk_50,
    input  wire rst_n,
    input  wire edit_mode,
    input  wire preset_button,
    input  wire [3:0] action_buttons,
    output wire [31:0] display_word,
    output wire [55:0] seven_segment,
    output wire [2:0] selected_preset,
    output wire patch_busy,
    output wire [31:0] audio_overflows,
    output wire uart_tx,
    output wire avm_address,
    output wire avm_read,
    output wire avm_write,
    output wire [31:0] avm_writedata,
    input  wire [31:0] avm_readdata,
    input  wire avm_waitrequest,
    input  wire avm_readdatavalid
);
    seven_segment_hex display(display_word,seven_segment);
    wire signed [23:0] sample;
    wire sample_valid;
    synthax_instrument #(.CLOCK_HZ(CLOCK_HZ)) instrument(
        clk_50,rst_n,edit_mode,preset_button,action_buttons,1'b0,
        1'b0,7'b0,16'b0,1'b0,sample,sample_valid,display_word,
        selected_preset,patch_busy
    );
    wire [7:0] stream_data;
    wire stream_valid, stream_ready;
    synthax_audio_transport transport(
        clk_50,rst_n,sample,sample_valid,stream_data,stream_valid,
        stream_ready,audio_overflows
    );
    generate
        if (USE_JTAG_UART) begin : jtag_transport
            jtag_uart_tx_master jtag_tx(
                clk_50,rst_n,stream_data,stream_valid,stream_ready,
                avm_address,avm_read,avm_write,avm_writedata,
                avm_readdata,avm_waitrequest,avm_readdatavalid
            );
            assign uart_tx = 1'b1;
        end else begin : physical_uart_transport
            synthax_uart_tx #(.CLOCK_HZ(CLOCK_HZ)) serial_tx(
                clk_50,rst_n,stream_data,stream_valid,stream_ready,uart_tx
            );
            assign avm_address = 0;
            assign avm_read = 0;
            assign avm_write = 0;
            assign avm_writedata = 0;
        end
    endgenerate
endmodule
