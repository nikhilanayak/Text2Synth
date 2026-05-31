`timescale 1ns / 1ps
//============================================================================
// mac_pe.v  -  INT8 Multiply-Accumulate Processing Element
//----------------------------------------------------------------------------
// The fundamental cell of the systolic GEMM array (Project.md, item 2.2).
//
// Dataflow: OUTPUT-STATIONARY (tentative; final array-level choice = item 2.1).
//   - Each PE holds one element of the output matrix in a local INT32 acc.
//   - Activations stream west -> east; weights stream north -> south.
//   - On every enabled cycle: acc += a_in * w_in, and the operands are
//     forwarded (registered) to the east/south neighbors so the next PE in
//     line sees them one cycle later (the systolic "wavefront").
//
// Arithmetic contract (must match the golden software GEMM, item 1.4):
//   INT8 (signed) * INT8 (signed) -> 16-bit signed product
//   product accumulated into a signed INT32 accumulator (no saturation here;
//   saturation/requant happens later in the requantization stage, item 2.6).
//============================================================================
module mac_pe #(
    parameter A_WIDTH   = 8,    // activation bit width
    parameter W_WIDTH   = 8,    // weight bit width
    parameter ACC_WIDTH = 32    // accumulator bit width
) (
    input  wire                          clk,
    input  wire                          rst_n,  // async active-low reset
    input  wire                          en,     // 1 = accumulate this cycle
    input  wire                          clear,  // 1 = zero the accumulator (priority over en)
    input  wire signed [A_WIDTH-1:0]     a_in,   // activation from west neighbor
    input  wire signed [W_WIDTH-1:0]     w_in,   // weight     from north neighbor
    output reg  signed [A_WIDTH-1:0]     a_out,  // activation forwarded east
    output reg  signed [W_WIDTH-1:0]     w_out,  // weight     forwarded south
    output reg  signed [ACC_WIDTH-1:0]   acc     // local INT32 partial sum
);

    // Full-precision signed product (both operands signed => signed multiply).
    wire signed [A_WIDTH+W_WIDTH-1:0] product = a_in * w_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= {A_WIDTH{1'b0}};
            w_out <= {W_WIDTH{1'b0}};
            acc   <= {ACC_WIDTH{1'b0}};
        end else begin
            // Systolic forwarding of operands to downstream PEs.
            a_out <= a_in;
            w_out <= w_in;

            // Accumulate. clear has priority so a tile can be reset in 1 cycle.
            if (clear)
                acc <= {ACC_WIDTH{1'b0}};
            else if (en)
                acc <= acc + product;
        end
    end

endmodule
