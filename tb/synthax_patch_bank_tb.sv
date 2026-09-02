`timescale 1ns/1ps
`include "synthax_params.svh"

module synthax_patch_bank_tb;
    reg clk=0, rst_n=0, control_tick=1;
    reg preset_next=0, edit_previous=0, edit_next=0, edit_decrease=0, edit_increase=0;
    reg external_we=0, external_commit=0;
    reg [6:0] external_address=0;
    reg [15:0] external_data=0;
    wire [2:0] selected_preset;
    wire [6:0] edit_index;
    wire [78*16-1:0] parameters;
    wire busy;
    always #5 clk=~clk;

    synthax_patch_bank dut(
        clk,rst_n,control_tick,preset_next,edit_previous,edit_next,
        edit_decrease,edit_increase,external_we,external_address,
        external_data,external_commit,selected_preset,edit_index,parameters,busy
    );

    task pulse;
        output reg signal;
        begin signal=1; @(posedge clk); #1 signal=0; @(posedge clk); end
    endtask

    initial begin
        repeat(3) @(posedge clk); rst_n=1;
        wait(!busy); @(posedge clk);
        if (parameters[48*16 +:16] !== 16'hffff) $fatal(1,"sine preset mixer mismatch");
        edit_increase=1; @(posedge clk); #1 edit_increase=0;
        if (parameters[15:0] == 0) $display("edit parameter 0 updated");
        preset_next=1; @(posedge clk); #1 preset_next=0;
        wait(!busy); @(posedge clk);
        if (selected_preset !== 1) $fatal(1,"preset did not advance");
        external_address=7'd10; external_data=16'h1234; external_we=1;
        @(posedge clk); #1 external_we=0;
        external_commit=1; @(posedge clk); #1 external_commit=0;
        wait(!busy); @(posedge clk);
        if (parameters[10*16 +:16] !== 16'h1234) $fatal(1,"external commit failed");
        $display("RESULT: SYNTHAX PATCH BANK PASS");
        $finish;
    end
    initial begin #30000; $fatal(1,"timeout"); end
endmodule
