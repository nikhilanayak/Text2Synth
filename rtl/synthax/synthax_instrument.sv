`timescale 1ns/1ps
`include "synthax_params.svh"

module synthax_instrument #(
    parameter integer CLOCK_HZ = 50_000_000
) (
    input  wire clk,
    input  wire rst_n,
    input  wire edit_mode,
    input  wire preset_button,
    input  wire [3:0] action_buttons,
    input  wire reference_mode,
    input  wire external_param_we,
    input  wire [6:0] external_param_address,
    input  wire [15:0] external_param_data,
    input  wire external_param_commit,
    output wire signed [23:0] audio_sample,
    output wire audio_valid,
    output wire [31:0] display_word,
    output wire [2:0] selected_preset,
    output wire patch_busy
);
    wire sample_tick, control_tick;
    synthax_tick_generator #(.CLOCK_HZ(CLOCK_HZ)) ticks(
        clk,rst_n,sample_tick,control_tick
    );

    wire preset_next, edit_previous, edit_next, edit_decrease, edit_increase;
    wire [3:0] note_gates;
    wire [6:0] edit_index;
    wire [`SX_PARAM_COUNT*16-1:0] parameters;
    wire [15:0] edit_value = parameters[edit_index*16 +: 16];
    synthax_button_ui #(.CLOCK_HZ(CLOCK_HZ)) ui(
        clk,rst_n,edit_mode,preset_button,action_buttons,selected_preset,
        edit_index,edit_value,preset_next,edit_previous,edit_next,
        edit_decrease,edit_increase,note_gates,display_word
    );
    synthax_patch_bank patches(
        clk,rst_n,control_tick,preset_next,edit_previous,edit_next,
        edit_decrease,edit_increase,external_param_we,external_param_address,
        external_param_data,external_param_commit,selected_preset,edit_index,
        parameters,patch_busy
    );

    reg [6:0] note;
    reg gate;
    always @* begin
        casex (note_gates)
            4'b1xxx: begin note=72; gate=1; end
            4'b01xx: begin note=67; gate=1; end
            4'b001x: begin note=64; gate=1; end
            4'b0001: begin note=60; gate=1; end
            default: begin note=60; gate=0; end
        endcase
    end
    synthax_voice_core voice(
        clk,rst_n,sample_tick,control_tick,reference_mode,note,gate,
        parameters,audio_sample,audio_valid
    );
endmodule
