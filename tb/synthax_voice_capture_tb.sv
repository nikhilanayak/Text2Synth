`timescale 1ns/1ps

module synthax_voice_capture_tb;
    reg clk=0, rst_n=0, gate=0;
    wire sample_tick, control_tick;
    wire signed [23:0] sample;
    wire valid;
    reg [78*16-1:0] parameters;
    reg [15:0] preset_memory [0:623];
    integer output_file, count=0, i;
    always #5 clk=~clk;
    synthax_tick_generator #(.CLOCK_HZ(480_000)) ticks(clk,rst_n,sample_tick,control_tick);
    synthax_voice_core voice(clk,rst_n,sample_tick,control_tick,1'b0,7'd60,gate,parameters,sample,valid);

    initial begin
        $readmemh("rtl/synthax/assets/diagnostic_presets_q0_16.hex",preset_memory);
        for(i=0;i<78;i=i+1) parameters[i*16 +:16]=preset_memory[i];
        output_file=$fopen("build/voice_capture.hex","w");
        repeat(3) @(posedge clk); rst_n=1;
        repeat(20) @(posedge clk); gate=1;
    end
    always @(posedge clk) if(valid) begin
        $fdisplay(output_file,"%04x",sample[23:8]);
        count=count+1;
        if(count==4800) begin
            $fclose(output_file);
            $display("RESULT: VOICE CAPTURE PASS");
            $finish;
        end
    end
    initial begin #600000; $fatal(1,"timeout"); end
endmodule
