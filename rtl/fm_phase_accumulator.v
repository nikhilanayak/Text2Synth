/**
 * fm_phase_accumulator.v
 * FOUNDATION: Milestone 2.1
 * 
 * The Phase Accumulator is the heart of a Direct Digital Synthesizer (DDS).
 * It converts a frequency value (phase increment) into a continuously 
 * cycling phase index.
 * 
 * For mathematical compatibility with the DDSP PyTorch model:
 * - We use a 32-bit accumulator for high frequency precision.
 * - The top N bits will be used to index the Sine LUT.
 */

module fm_phase_accumulator #(
    parameter ACC_WIDTH = 32,
    parameter OUT_WIDTH = 10  // Top bits for LUT indexing (1024 entries)
)(
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 en,      // Global enable (e.g. for sampling rate)
    input  wire [ACC_WIDTH-1:0] phase_inc, // The "frequency" control
    output wire [OUT_WIDTH-1:0] phase_out  // The "index" for the Sine LUT
);

    reg [ACC_WIDTH-1:0] acc_reg;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            acc_reg <= 0;
        end else if (en) begin
            acc_reg <= acc_reg + phase_inc;
        end
    end

    // Use the top bits for the LUT index
    assign phase_out = acc_reg[ACC_WIDTH-1 : ACC_WIDTH-OUT_WIDTH];

endmodule
