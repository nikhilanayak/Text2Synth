`timescale 1ns/1ps

// Avalon-MM master for Intel's JTAG UART agent: address 1 is control/WSPACE,
// address 0 is data. Connect these ports in Platform Designer.
module jtag_uart_tx_master (
    input  wire clk,
    input  wire rst_n,
    input  wire [7:0] stream_data,
    input  wire stream_valid,
    output reg  stream_ready,
    output reg  avm_address,
    output reg  avm_read,
    output reg  avm_write,
    output reg  [31:0] avm_writedata,
    input  wire [31:0] avm_readdata,
    input  wire avm_waitrequest,
    input  wire avm_readdatavalid
);
    localparam POLL=2'd0, WAIT_CONTROL=2'd1, WRITE_DATA=2'd2;
    reg [1:0] state;
    reg [7:0] held_data;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= POLL;
            stream_ready <= 0;
            avm_address <= 1;
            avm_read <= 0;
            avm_write <= 0;
            avm_writedata <= 0;
            held_data <= 0;
        end else begin
            stream_ready <= 0;
            case (state)
                POLL: begin
                    avm_address <= 1;
                    avm_read <= 1;
                    if (!avm_waitrequest) begin
                        avm_read <= 0;
                        state <= WAIT_CONTROL;
                    end
                end
                WAIT_CONTROL: if (avm_readdatavalid) begin
                    if (avm_readdata[31:16] != 0 && stream_valid) begin
                        held_data <= stream_data;
                        state <= WRITE_DATA;
                    end else state <= POLL;
                end
                WRITE_DATA: begin
                    avm_address <= 0;
                    avm_writedata <= {24'b0,held_data};
                    avm_write <= 1;
                    if (!avm_waitrequest) begin
                        avm_write <= 0;
                        stream_ready <= 1;
                        state <= POLL;
                    end
                end
                default: state <= POLL;
            endcase
        end
    end
endmodule
