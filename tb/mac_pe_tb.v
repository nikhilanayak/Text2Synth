`timescale 1ns / 1ps
//============================================================================
// mac_pe_tb.v  -  Self-checking testbench for the INT8 MAC PE.
//
// Run:  iverilog -o build/mac_pe_tb.vvp rtl/mac_pe.v tb/mac_pe_tb.v
//       vvp build/mac_pe_tb.vvp
// Exits non-zero (via $fatal-style flag) on any mismatch so CI can gate on it.
//============================================================================
module mac_pe_tb;

    localparam A_WIDTH   = 8;
    localparam W_WIDTH   = 8;
    localparam ACC_WIDTH = 32;

    reg                           clk   = 1'b0;
    reg                           rst_n = 1'b0;
    reg                           en    = 1'b0;
    reg                           clear = 1'b0;
    reg  signed [A_WIDTH-1:0]     a_in  = 0;
    reg  signed [W_WIDTH-1:0]     w_in  = 0;
    wire signed [A_WIDTH-1:0]     a_out;
    wire signed [W_WIDTH-1:0]     w_out;
    wire signed [ACC_WIDTH-1:0]   acc;

    integer errors = 0;
    integer i;
    reg signed [ACC_WIDTH-1:0] expected;

    // Device under test
    mac_pe #(
        .A_WIDTH(A_WIDTH), .W_WIDTH(W_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .clear(clear),
        .a_in(a_in), .w_in(w_in),
        .a_out(a_out), .w_out(w_out), .acc(acc)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    // Directed activation/weight vectors for the dot-product test.
    reg signed [A_WIDTH-1:0] av [0:7];
    reg signed [W_WIDTH-1:0] wv [0:7];

    task check;
        input signed [ACC_WIDTH-1:0] got;
        input signed [ACC_WIDTH-1:0] exp;
        input [255:0] name;
        begin
            if (got !== exp) begin
                errors = errors + 1;
                $display("  [FAIL] %0s : got %0d, expected %0d", name, got, exp);
            end else begin
                $display("  [PASS] %0s : %0d", name, got);
            end
        end
    endtask

    initial begin
        $dumpfile("build/mac_pe_tb.vcd");
        $dumpvars(0, mac_pe_tb);

        // Directed vectors incl. INT8 extremes (-128, 127).
        av[0]=  3; wv[0]=  2;
        av[1]= -5; wv[1]=  4;
        av[2]=  7; wv[2]= -3;
        av[3]=-128; wv[3]=-128;   // max-magnitude product = +16384
        av[4]= 127; wv[4]=-128;   // large negative product
        av[5]=  0; wv[5]=  9;
        av[6]= -1; wv[6]= -7;
        av[7]= 64; wv[7]=  2;

        // ---- Reset ----
        rst_n = 1'b0; en = 1'b0; clear = 1'b0; a_in = 0; w_in = 0;
        @(negedge clk); @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        check(acc, 0, "reset clears accumulator");

        // ---- Test 1: clear ----
        clear = 1'b1; @(posedge clk); #1 clear = 1'b0;
        check(acc, 0, "clear pulse zeroes acc");

        // ---- Test 2: 8-tap signed dot product ----
        expected = 0;
        for (i = 0; i < 8; i = i + 1) begin
            @(negedge clk);
            a_in = av[i]; w_in = wv[i]; en = 1'b1;
            expected = expected + (av[i] * wv[i]);
            @(posedge clk);   // PE accumulates on this edge
        end
        #1 en = 1'b0;
        @(negedge clk);
        check(acc, expected, "8-tap INT8 dot product");

        // ---- Test 3: operand forwarding (systolic pass-through) ----
        @(negedge clk); a_in = 42; w_in = -17; en = 1'b0;
        @(posedge clk); #1;
        check(a_out, 42,  "a_in forwarded east");
        check(w_out, -17, "w_in forwarded south");

        // ---- Test 4: clear mid-stream then re-accumulate ----
        clear = 1'b1; @(posedge clk); #1 clear = 1'b0;
        check(acc, 0, "mid-stream clear");
        @(negedge clk); a_in = -128; w_in = -128; en = 1'b1;
        @(posedge clk); #1 en = 1'b0;
        @(negedge clk);
        check(acc, 16384, "(-128)*(-128) edge case");

        // ---- Summary ----
        $display("----------------------------------------------------------");
        if (errors == 0)
            $display("RESULT: ALL TESTS PASSED");
        else
            $display("RESULT: %0d TEST(S) FAILED", errors);
        $display("----------------------------------------------------------");

        if (errors != 0) $stop;   // non-zero exit for CI gating
        $finish;
    end

    // Safety timeout
    initial begin
        #10000;
        $display("RESULT: TIMEOUT - testbench did not finish");
        $stop;
    end

endmodule
