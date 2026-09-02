`timescale 1ns/1ps

module synthax_instrument_tb;
    reg clk=0, rst_n=0, edit_mode=0, preset_button=0, reference_mode=0;
    reg [3:0] buttons=0;
    wire signed [23:0] sample;
    wire valid;
    wire [31:0] display;
    wire [2:0] preset;
    wire busy;
    integer samples=0, nonzero=0;
    always #5 clk=~clk;

    synthax_instrument #(.CLOCK_HZ(480_000)) dut(
        clk,rst_n,edit_mode,preset_button,buttons,reference_mode,
        1'b0,7'b0,16'b0,1'b0,sample,valid,display,preset,busy
    );
    always @(posedge clk) if(valid) begin
        samples=samples+1;
        if(sample!=0) nonzero=nonzero+1;
    end
    initial begin
        repeat(3) @(posedge clk); rst_n=1;
        wait(!busy);
        buttons=4'b0001;
        repeat(16000) @(posedge clk);
        buttons=0;
        repeat(4000) @(posedge clk);
        if(samples<1500) $fatal(1,"too few samples %0d",samples);
        if(nonzero<500) $fatal(1,"voice stayed silent %0d",nonzero);
        $display("RESULT: SYNTHAX INSTRUMENT PASS samples=%0d nonzero=%0d",samples,nonzero);
        $finish;
    end
    initial begin #500000; $fatal(1,"timeout"); end
endmodule
