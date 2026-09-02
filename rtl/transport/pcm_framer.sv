`timescale 1ns/1ps

module pcm_framer #(
    parameter integer FRAME_SAMPLES = 256
) (
    input  wire clk,
    input  wire rst_n,
    input  wire signed [15:0] fifo_sample,
    input  wire [10:0] fifo_level,
    output reg  fifo_pop,
    output reg  [7:0] byte_data,
    output reg  byte_valid,
    input  wire byte_ready
);
    localparam IDLE=3'd0, MAGIC=3'd1, HEADER=3'd2, PAYLOAD=3'd3, CRC0=3'd4, CRC1=3'd5;
    reg [2:0] state;
    reg [2:0] magic_index;
    reg [2:0] header_index;
    reg [8:0] sample_index;
    reg payload_high;
    reg [15:0] frame_sequence;
    reg [15:0] crc;

    function automatic [15:0] crc16;
        input [15:0] prior;
        input [7:0] data;
        integer bit_index;
        reg [15:0] value;
        begin
            value = prior ^ (data << 8);
            for (bit_index=0; bit_index<8; bit_index=bit_index+1)
                value = value[15] ? (value << 1) ^ 16'h1021 : value << 1;
            crc16 = value;
        end
    endfunction

    always @* begin
        byte_valid = state != IDLE;
        case (state)
            MAGIC: case (magic_index)
                0: byte_data=8'h43; 1: byte_data=8'h54;
                2: byte_data=8'h41; default: byte_data=8'h47;
            endcase
            HEADER: case (header_index)
                0: byte_data=8'h01;
                1: byte_data=8'h01;
                2: byte_data=frame_sequence[7:0];
                3: byte_data=frame_sequence[15:8];
                4: byte_data=FRAME_SAMPLES[7:0];
                default: byte_data=FRAME_SAMPLES[15:8];
            endcase
            PAYLOAD: byte_data = payload_high ? fifo_sample[15:8] : fifo_sample[7:0];
            CRC0: byte_data=crc[7:0];
            CRC1: byte_data=crc[15:8];
            default: byte_data=0;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            magic_index <= 0;
            header_index <= 0;
            sample_index <= 0;
            payload_high <= 0;
            frame_sequence <= 0;
            crc <= 16'hffff;
            fifo_pop <= 0;
        end else begin
            fifo_pop <= 0;
            if (state == IDLE && fifo_level >= FRAME_SAMPLES) begin
                state <= MAGIC;
                magic_index <= 0;
                crc <= 16'hffff;
            end else if (byte_valid && byte_ready) begin
                case (state)
                    MAGIC: begin
                        if (magic_index == 3) begin state <= HEADER; header_index <= 0; end
                        else magic_index <= magic_index + 1'b1;
                    end
                    HEADER: begin
                        crc <= crc16(crc, byte_data);
                        if (header_index == 5) begin
                            state <= PAYLOAD;
                            sample_index <= 0;
                            payload_high <= 0;
                        end else header_index <= header_index + 1'b1;
                    end
                    PAYLOAD: begin
                        crc <= crc16(crc, byte_data);
                        if (payload_high) begin
                            fifo_pop <= 1;
                            payload_high <= 0;
                            if (sample_index == FRAME_SAMPLES-1) state <= CRC0;
                            else sample_index <= sample_index + 1'b1;
                        end else payload_high <= 1;
                    end
                    CRC0: state <= CRC1;
                    CRC1: begin
                        state <= IDLE;
                        frame_sequence <= frame_sequence + 1'b1;
                    end
                    default: state <= IDLE;
                endcase
            end
        end
    end
endmodule
