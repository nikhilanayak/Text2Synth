`timescale 1ns/1ps

module synthax_button_ui #(
    parameter integer CLOCK_HZ = 50_000_000
) (
    input  wire clk,
    input  wire rst_n,
    input  wire edit_mode,
    input  wire preset_button,
    input  wire [3:0] action_buttons,
    input  wire [2:0] selected_preset,
    input  wire [6:0] edit_index,
    input  wire [15:0] edit_value,
    output reg  preset_next,
    output reg  edit_previous,
    output reg  edit_next,
    output reg  edit_decrease,
    output reg  edit_increase,
    output wire [3:0] note_gates,
    output reg  [31:0] display_word
);
    localparam integer DEBOUNCE = CLOCK_HZ / 100;
    localparam integer PAGE_TIME = CLOCK_HZ;
    reg [19:0] debounce_count [0:4];
    reg [4:0] stable;
    reg [4:0] prior;
    reg [31:0] page_counter;
    reg display_value_page;
    integer i;
    wire [4:0] raw = {preset_button, action_buttons};
    wire [4:0] rising = stable & ~prior;
    assign note_gates = edit_mode ? 4'b0 : stable[3:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stable <= 0;
            prior <= 0;
            page_counter <= 0;
            display_value_page <= 0;
            preset_next <= 0;
            edit_previous <= 0;
            edit_next <= 0;
            edit_decrease <= 0;
            edit_increase <= 0;
            display_word <= 0;
            for (i = 0; i < 5; i = i + 1) debounce_count[i] <= 0;
        end else begin
            for (i = 0; i < 5; i = i + 1) begin
                if (raw[i] == stable[i])
                    debounce_count[i] <= 0;
                else if (debounce_count[i] == DEBOUNCE-1) begin
                    stable[i] <= raw[i];
                    debounce_count[i] <= 0;
                end else debounce_count[i] <= debounce_count[i] + 1'b1;
            end
            prior <= stable;
            preset_next <= rising[4];
            edit_previous <= edit_mode && rising[0];
            edit_next <= edit_mode && rising[1];
            edit_decrease <= edit_mode && rising[2];
            edit_increase <= edit_mode && rising[3];
            if (page_counter == PAGE_TIME-1) begin
                page_counter <= 0;
                display_value_page <= ~display_value_page;
            end else page_counter <= page_counter + 1'b1;
            if (edit_mode)
                display_word <= display_value_page ? {16'h0000, edit_value}
                    : {12'ha00, 1'b0, edit_index, 12'h000};
            else
                display_word <= {12'h500, 1'b0, selected_preset, 16'h0000};
        end
    end
endmodule
