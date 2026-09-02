`timescale 1ns/1ps

// Control-rate ADSR. Parameters use SynthAX-normalized unsigned Q0.16.
module synthax_adsr (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        control_tick,
    input  wire        gate,
    input  wire [15:0] attack,
    input  wire [15:0] decay,
    input  wire [15:0] sustain,
    input  wire [15:0] release_time,
    input  wire [15:0] alpha,
    output reg  [15:0] level
);
    localparam IDLE = 3'd0, ATTACK = 3'd1, DECAY = 3'd2;
    localparam SUSTAIN = 3'd3, RELEASE = 3'd4;
    reg [2:0] state;
    reg prior_gate;
    reg [31:0] delta;
    reg [31:0] ticks;
    reg [63:0] curved;
    wire [15:0] shaped_sustain = sustain;

    function automatic [31:0] time_ticks;
        input [15:0] normalized;
        input [15:0] maximum_ticks;
        reg [63:0] square;
        begin
            square = normalized * normalized;
            time_ticks = (square * maximum_ticks) >> 32;
            if (time_ticks == 0) time_ticks = 1;
        end
    endfunction
    function automatic [31:0] full_scale_step;
        input [15:0] normalized;
        input [15:0] maximum_ticks;
        reg [31:0] count;
        begin
            count = time_ticks(normalized, maximum_ticks);
            full_scale_step = 32'd65535 / count;
        end
    endfunction

    // Alpha remains part of the exact contract. The first fixed-point core uses
    // linear ramps; alpha refinement is isolated to this module.
    wire unused_alpha = ^alpha;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            prior_gate <= 1'b0;
            level <= 16'd0;
            delta <= 0;
            ticks <= 0;
            curved <= 0;
        end else if (control_tick) begin
            prior_gate <= gate;
            if (gate && !prior_gate) begin
                ticks <= time_ticks(attack, 16'd960);
                delta <= full_scale_step(attack, 16'd960);
                state <= ATTACK;
                level <= 0;
            end else if (!gate && prior_gate) begin
                ticks <= time_ticks(release_time, 16'd2400);
                delta <= level / time_ticks(release_time, 16'd2400);
                state <= RELEASE;
            end else begin
                case (state)
                    IDLE: level <= 0;
                    ATTACK: begin
                        if (level + delta >= 65535) begin
                            level <= 16'hffff;
                            ticks <= time_ticks(decay, 16'd960);
                            delta <= (32'd65535 - shaped_sustain)
                                / time_ticks(decay, 16'd960);
                            state <= DECAY;
                        end else level <= level + delta[15:0];
                    end
                    DECAY: begin
                        if (level <= shaped_sustain + delta) begin
                            level <= shaped_sustain;
                            state <= SUSTAIN;
                        end else level <= level - delta[15:0];
                    end
                    SUSTAIN: level <= shaped_sustain;
                    RELEASE: begin
                        if (level <= delta) begin
                            level <= 0;
                            state <= IDLE;
                        end else level <= level - delta[15:0];
                    end
                    default: begin state <= IDLE; level <= 0; end
                endcase
            end
        end
    end
endmodule
