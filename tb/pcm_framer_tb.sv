`timescale 1ns/1ps

module pcm_framer_tb;
    reg clk=0, rst_n=0;
    reg sample_valid=0;
    reg signed [23:0] sample=0;
    wire [7:0] data;
    wire valid;
    reg ready=1;
    wire [31:0] overflow;
    integer i, received=0;
    reg [15:0] observed_crc=16'hffff;
    reg [7:0] prior_byte;
    always #5 clk=~clk;

    synthax_audio_transport dut(clk,rst_n,sample,sample_valid,data,valid,ready,overflow);

    function automatic [15:0] crc16;
        input [15:0] prior;
        input [7:0] value_in;
        integer b;
        reg [15:0] value;
        begin
            value=prior^(value_in<<8);
            for(b=0;b<8;b=b+1) value=value[15]?((value<<1)^16'h1021):(value<<1);
            crc16=value;
        end
    endfunction

    always @(posedge clk) if (valid && ready) begin
        if (received==0 && data!==8'h43) $fatal(1,"bad magic C");
        if (received==1 && data!==8'h54) $fatal(1,"bad magic T");
        if (received==2 && data!==8'h41) $fatal(1,"bad magic A");
        if (received==3 && data!==8'h47) $fatal(1,"bad magic G");
        if (received>=4 && received<522) observed_crc=crc16(observed_crc,data);
        if (received==522 && data!==observed_crc[7:0]) $fatal(1,"bad crc low");
        if (received==523 && data!==observed_crc[15:8]) $fatal(1,"bad crc high");
        received=received+1;
    end

    initial begin
        repeat(3) @(posedge clk); rst_n=1;
        for(i=0;i<256;i=i+1) begin
            @(negedge clk); sample=i<<<8; sample_valid=1;
        end
        @(negedge clk); sample_valid=0;
        wait(received==524); @(posedge clk);
        if (overflow!=0) $fatal(1,"unexpected overflow");
        $display("RESULT: PCM FRAMER PASS");
        $finish;
    end
    initial begin #200000; $fatal(1,"timeout %0d",received); end
endmodule
