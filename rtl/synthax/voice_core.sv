`timescale 1ns/1ps
`include "synthax_params.svh"

module synthax_voice_core (
    input  wire clk,
    input  wire rst_n,
    input  wire sample_tick,
    input  wire control_tick,
    input  wire reference_mode,
    input  wire [6:0] live_midi_note,
    input  wire live_gate,
    input  wire [`SX_PARAM_COUNT*16-1:0] parameters,
    output reg  signed [23:0] sample,
    output reg  sample_valid
);
    function automatic [15:0] p;
        input integer index;
        begin p = parameters[index*16 +: 16]; end
    endfunction
    wire [15:0] param [0:`SX_PARAM_COUNT-1];
    genvar pg;
    generate
        for (pg=0; pg<`SX_PARAM_COUNT; pg=pg+1) begin : unpack_parameters
            assign param[pg] = parameters[pg*16 +: 16];
        end
    endgenerate
    function automatic [15:0] curved_weight;
        input [15:0] x;
        reg [31:0] product;
        begin product = x*x; curved_weight = product >> 16; end
    endfunction

    reg prior_live_gate;
    reg prior_effective_gate;
    reg reference_gate;
    reg [11:0] reference_ticks;
    reg [6:0] reference_note;
    wire effective_gate = reference_mode ? reference_gate : live_gate;
    wire [6:0] midi_note = reference_mode ? reference_note : live_midi_note;

    wire [15:0] lfo1_rate_env, lfo2_rate_env, lfo1_amp_env, lfo2_amp_env;
    wire [15:0] env1, env2;
    synthax_adsr a_lfo1_rate(clk,rst_n,control_tick,effective_gate,param[26],param[27],param[29],param[28],param[25],lfo1_rate_env);
    synthax_adsr a_lfo2_rate(clk,rst_n,control_tick,effective_gate,param[44],param[45],param[47],param[46],param[43],lfo2_rate_env);
    synthax_adsr a_lfo1_amp(clk,rst_n,control_tick,effective_gate,param[21],param[22],param[24],param[23],param[20],lfo1_amp_env);
    synthax_adsr a_lfo2_amp(clk,rst_n,control_tick,effective_gate,param[39],param[40],param[42],param[41],param[38],lfo2_amp_env);
    synthax_adsr a_env1(clk,rst_n,control_tick,effective_gate,param[1],param[2],param[4],param[3],param[0],env1);
    synthax_adsr a_env2(clk,rst_n,control_tick,effective_gate,param[6],param[7],param[9],param[8],param[5],env2);

    wire [15:0] lfo1, lfo2;
    synthax_lfo lfo_1(
        clk,rst_n,control_tick,effective_gate,param[12],param[14],param[13],param[17],param[19],param[16],param[15],param[18],
        lfo1_rate_env,lfo1_amp_env,lfo1
    );
    synthax_lfo lfo_2(
        clk,rst_n,control_tick,effective_gate,param[30],param[32],param[31],param[35],param[37],param[34],param[33],param[36],
        lfo2_rate_env,lfo2_amp_env,lfo2
    );

    reg signed [18:0] mod_target [0:4];
    reg [63:0] mod_accumulator;
    integer row, column;
    reg [15:0] source_value;
    always @* begin
        for (row = 0; row < 5; row = row + 1) begin
            mod_accumulator = 0;
            for (column = 0; column < 4; column = column + 1) begin
                case (column)
                    0: source_value = env1;
                    1: source_value = env2;
                    2: source_value = lfo1;
                    default: source_value = lfo2;
                endcase
                mod_accumulator = mod_accumulator
                    + curved_weight(param[`SX_MOD_MATRIX + row*4 + column]) * source_value;
            end
            mod_target[row] = mod_accumulator >> 16;
        end
    end

    wire signed [18:0] modulation [0:4];
    genvar g;
    generate
        for (g = 0; g < 5; g = g + 1) begin : interpolators
            synthax_control_interp interp(
                clk,rst_n,sample_tick,control_tick,mod_target[g],modulation[g]
            );
        end
    endgenerate

    reg signed [31:0] pitch1_q5, pitch2_q5;
    reg signed [31:0] tuning1_q5, tuning2_q5;
    reg signed [31:0] depth1, depth2;
    reg [11:0] pitch1_address, pitch2_address;
    wire [31:0] phase_inc1, phase_inc2;
    synthax_midi_phase_rom pitch_table1(pitch1_address, phase_inc1);
    synthax_midi_phase_rom pitch_table2(pitch2_address, phase_inc2);

    always @* begin
        tuning1_q5 = (($signed({1'b0,param[73]}) - 32'sd32768) * 32'sd1536) / 32'sd32768;
        tuning2_q5 = (($signed({1'b0,param[77]}) - 32'sd32768) * 32'sd1536) / 32'sd32768;
        depth1 = (($signed({1'b0,param[72]}) - 32'sd32768) * 32'sd96) / 32'sd32768;
        depth2 = (($signed({1'b0,param[75]}) - 32'sd32768) * 32'sd96) / 32'sd32768;
        pitch1_q5 = $signed({1'b0,midi_note,5'b0}) + tuning1_q5
                  + ((depth1 * modulation[0]) >>> 11);
        pitch2_q5 = $signed({1'b0,midi_note,5'b0}) + tuning2_q5
                  + ((depth2 * modulation[2]) >>> 11);
        if (pitch1_q5 < 0) pitch1_address = 0;
        else if (pitch1_q5 > 4095) pitch1_address = 4095;
        else pitch1_address = pitch1_q5[11:0];
        if (pitch2_q5 < 0) pitch2_address = 0;
        else if (pitch2_q5 > 4095) pitch2_address = 4095;
        else pitch2_address = pitch2_q5[11:0];
    end

    reg [31:0] phase1, phase2;
    reg [31:0] noise_lfsr;
    wire signed [17:0] sine1, sine2, cosine2;
    synthax_sine_rom osc1_rom(phase1[31:22] + 10'd256, sine1);
    synthax_sine_rom osc2_rom(phase2[31:22], sine2);
    synthax_sine_rom osc2_cos_rom(phase2[31:22] + 10'd256, cosine2);
    reg signed [17:0] square2, shaped2;
    reg signed [18:0] shape_factor;
    reg signed [18:0] harmonic_factor;
    reg signed [36:0] shape_product;
    reg signed [37:0] shaped_product;
    reg signed [36:0] amplitude_product1, amplitude_product2, amplitude_product_noise;
    reg signed [19:0] wave1_q17, wave2_q17, noise_q17;
    reg signed [36:0] level_product1, level_product2, level_product_noise;
    reg signed [21:0] mix_q17;
    reg [16:0] amp1, amp2, amp_noise;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prior_live_gate <= 0;
            prior_effective_gate <= 0;
            reference_gate <= 0;
            reference_ticks <= 0;
            reference_note <= 60;
            phase1 <= 0;
            phase2 <= 0;
            noise_lfsr <= 32'h1ace_beef;
            sample <= 0;
            sample_valid <= 0;
        end else begin
            sample_valid <= 0;
            if (control_tick) begin
                prior_live_gate <= live_gate;
                prior_effective_gate <= effective_gate;
                if (reference_mode && live_gate && !prior_live_gate) begin
                    reference_note <= (param[11] * 127) >> 16;
                    reference_ticks <= 1 + (({48'b0,param[10]} *
                        {48'b0,param[10]} * 64'd1915) >> 32);
                    reference_gate <= 1;
                end else if (reference_mode && reference_gate) begin
                    if (reference_ticks == 0)
                        reference_gate <= 0;
                    else
                        reference_ticks <= reference_ticks - 1'b1;
                end
            end
            if (sample_tick) begin
                if (control_tick && effective_gate && !prior_effective_gate) begin
                    phase1 <= ({param[71],16'b0} - 32'h80000000) + phase_inc1;
                    phase2 <= ({param[74],16'b0} - 32'h80000000) + phase_inc2;
                end else begin
                    phase1 <= phase1 + phase_inc1;
                    phase2 <= phase2 + phase_inc2;
                end
                noise_lfsr <= {noise_lfsr[30:0],
                    noise_lfsr[31]^noise_lfsr[21]^noise_lfsr[1]^noise_lfsr[0]};
                amp1 = modulation[1] <= 0 ? 0 :
                    (modulation[1] > 65535 ? 65535 : modulation[1]);
                amp2 = modulation[3] <= 0 ? 0 :
                    (modulation[3] > 65535 ? 65535 : modulation[3]);
                amp_noise = modulation[4] <= 0 ? 0 :
                    (modulation[4] > 65535 ? 65535 : modulation[4]);
                square2 = sine2[17] ? -18'sd131071 : 18'sd131071;
                // SynthAX SquareSawVCO: (1-shape/2)*square*(1+shape*cos).
                shape_factor = 19'sd65535 - (param[76] >> 1);
                harmonic_factor = 19'sd65535
                    + (($signed(cosine2) * $signed({1'b0,param[76]})) >>> 17);
                shape_product = $signed(square2) * $signed(shape_factor);
                shaped_product = $signed(shape_product >>> 16) * $signed(harmonic_factor);
                shaped2 = shaped_product >>> 16;

                // Keep every multiply signed and explicitly widen unsigned Q0.16
                // controls. Implicit signed/unsigned promotion here causes wraparound
                // on several synthesis tools.
                amplitude_product1 = $signed(sine1) * $signed({1'b0,amp1});
                amplitude_product2 = $signed(shaped2) * $signed({1'b0,amp2});
                amplitude_product_noise = $signed(noise_lfsr[17:0]) * $signed({1'b0,amp_noise});
                wave1_q17 = amplitude_product1 >>> 16;
                wave2_q17 = amplitude_product2 >>> 16;
                noise_q17 = amplitude_product_noise >>> 16;
                level_product1 = $signed(wave1_q17) * $signed({1'b0,param[48]});
                level_product2 = $signed(wave2_q17) * $signed({1'b0,param[49]});
                level_product_noise = $signed(noise_q17) * $signed({1'b0,param[50]});
                // Fixed 1/3 headroom replaces SynthAX's non-causal buffer peak normalization.
                mix_q17 = (($signed(level_product1) >>> 16)
                    + ($signed(level_product2) >>> 16)
                    + ($signed(level_product_noise) >>> 16)) / 3;
                sample <= $signed(mix_q17) <<< 6;
                sample_valid <= 1;
            end
        end
    end
endmodule
