`timescale 1ns/1ps

module fm_phase_accumulator_tb;

    reg clk;
    reg reset;
    reg en;
    reg [31:0] phase_inc;
    wire [9:0] phase_out;

    // Instantiate the Unit Under Test (UUT)
    fm_phase_accumulator #(
        .ACC_WIDTH(32),
        .OUT_WIDTH(10)
    ) uut (
        .clk(clk),
        .reset(reset),
        .en(en),
        .phase_inc(phase_inc),
        .phase_out(phase_out)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz clock

    initial begin
        // Initialize signals
        reset = 1;
        en = 0;
        phase_inc = 0;

        // Reset pulse
        #20 reset = 0;
        #10 en = 1;

        // Test Case 1: Slow increment
        // phase_inc = 2^32 / 100 = ~42,949,672
        phase_inc = 32'd42949673; 
        #1000;

        // Test Case 2: Faster increment
        phase_inc = 32'd214748364; 
        #1000;

        // Test Case 3: Overflow check
        phase_inc = 32'hFFFFFFFF;
        #100;

        $display("Phase Accumulator Test Finished.");
        $finish;
    end

    initial begin
        $dumpfile("build/fm_phase_accumulator_tb.vcd");
        $dumpvars(0, fm_phase_accumulator_tb);
    end

endmodule
