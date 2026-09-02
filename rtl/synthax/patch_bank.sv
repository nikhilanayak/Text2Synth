`timescale 1ns/1ps
`include "synthax_params.svh"

module synthax_patch_bank #(
    parameter integer PRESET_COUNT = 8,
    parameter PRESET_FILE = "rtl/synthax/assets/diagnostic_presets_q0_16.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire control_tick,
    input  wire preset_next,
    input  wire edit_previous,
    input  wire edit_next,
    input  wire edit_decrease,
    input  wire edit_increase,
    input  wire external_we,
    input  wire [6:0] external_address,
    input  wire [15:0] external_data,
    input  wire external_commit,
    output reg  [2:0] selected_preset,
    output reg  [6:0] edit_index,
    output reg  [`SX_PARAM_COUNT*16-1:0] active_parameters,
    output wire busy
);
    localparam TOTAL_WORDS = PRESET_COUNT * `SX_PARAM_COUNT;
    localparam RESTORE = 2'd0, LOAD = 2'd1, READY = 2'd2, COMMIT = 2'd3;
    reg [1:0] state;
    reg [9:0] restore_index;
    reg [6:0] load_index;
    reg [15:0] preset_rom [0:TOTAL_WORDS-1];
    reg [15:0] working [0:TOTAL_WORDS-1];
    reg [15:0] external_shadow [0:`SX_PARAM_COUNT-1];
    reg commit_pending;
    integer address;

    initial $readmemh(PRESET_FILE, preset_rom);
    assign busy = state != READY;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= RESTORE;
            restore_index <= 0;
            load_index <= 0;
            selected_preset <= 0;
            edit_index <= 0;
            active_parameters <= 0;
            commit_pending <= 0;
        end else begin
            if (external_commit) commit_pending <= 1;
            case (state)
                RESTORE: begin
                    working[restore_index] <= preset_rom[restore_index];
                    if (restore_index == TOTAL_WORDS-1) begin
                        restore_index <= 0;
                        load_index <= 0;
                        state <= LOAD;
                    end else restore_index <= restore_index + 1'b1;
                end
                LOAD: begin
                    address = selected_preset * `SX_PARAM_COUNT + load_index;
                    active_parameters[load_index*16 +: 16] <= working[address];
                    external_shadow[load_index] <= working[address];
                    if (load_index == `SX_PARAM_COUNT-1) begin
                        load_index <= 0;
                        state <= READY;
                    end else load_index <= load_index + 1'b1;
                end
                COMMIT: begin
                    address = selected_preset * `SX_PARAM_COUNT + load_index;
                    working[address] <= external_shadow[load_index];
                    active_parameters[load_index*16 +: 16] <= external_shadow[load_index];
                    if (load_index == `SX_PARAM_COUNT-1) begin
                        load_index <= 0;
                        state <= READY;
                    end else load_index <= load_index + 1'b1;
                end
                default: begin
                    if (external_we && external_address < `SX_PARAM_COUNT)
                        external_shadow[external_address] <= external_data;
                    if ((commit_pending || external_commit) && control_tick) begin
                        load_index <= 0;
                        commit_pending <= 0;
                        state <= COMMIT;
                    end else if (preset_next) begin
                        if (selected_preset == PRESET_COUNT-1)
                            selected_preset <= 0;
                        else
                            selected_preset <= selected_preset + 1'b1;
                        load_index <= 0;
                        state <= LOAD;
                    end else begin
                        if (edit_previous)
                            edit_index <= edit_index == 0 ? `SX_PARAM_COUNT-1 : edit_index-1'b1;
                        if (edit_next)
                            edit_index <= edit_index == `SX_PARAM_COUNT-1 ? 0 : edit_index+1'b1;
                        address = selected_preset * `SX_PARAM_COUNT + edit_index;
                        if (edit_decrease) begin
                            if (working[address] < 16'h0100)
                                working[address] <= 0;
                            else
                                working[address] <= working[address] - 16'h0100;
                            active_parameters[edit_index*16 +: 16] <=
                                working[address] < 16'h0100 ? 0 : working[address] - 16'h0100;
                        end
                        if (edit_increase) begin
                            if (working[address] > 16'heeff)
                                working[address] <= 16'hffff;
                            else
                                working[address] <= working[address] + 16'h0100;
                            active_parameters[edit_index*16 +: 16] <=
                                working[address] > 16'heeff ? 16'hffff : working[address] + 16'h0100;
                        end
                    end
                end
            endcase
        end
    end
endmodule
