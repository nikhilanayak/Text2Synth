# System Architecture: Streaming Vocal/Instrumental Separator

This document describes the hardware/software architecture of the ECP5-based real-time audio separator.

## Block Diagram

```mermaid
graph TB
    subgraph "Host (PC)"
        H_IN[16 kHz Mono Audio Stream]
        H_OUT[2-Stem Audio Stream]
    end

    subgraph "FPGA (ULX3S - ECP5-85F)"
        %% Transport Layer
        UART[UART / USB Transport]
        PP_BUFF[Ping-Pong Frame Buffers]

        %% DSP Pipeline: Forward
        subgraph "DSP Front-End (STFT)"
            WIN[Windowing]
            FFT[Folded FFT <br/>(Fixed-Point)]
            MAG[Magnitude / Phase]
        end

        %% NN Accelerator
        subgraph "Neural Accelerator (GEMM Core)"
            SYS_ARRAY[INT8 Systolic Array <br/>(2D PE Grid)]
            ACT_BRAM[Activation BRAMs]
            WGT_BRAM[Weight BRAMs]
            REQ[Requantization Stage <br/>(INT32 -> INT8)]
            CTL[GEMM FSM / Control]
        end

        %% Weight Management
        FLASH_CTRL[QSPI Flash Controller]
        FLASH_EXT{{External QSPI Flash <br/>(Model Storage)}}

        %% DSP Pipeline: Back-End
        subgraph "DSP Back-End (iSTFT)"
            MASK[Mask Application <br/>(Mixture x Mask)]
            IFFT[Folded IFFT <br/>(Overlap-Add)]
        end

        %% Connections
        H_IN --> UART
        UART <--> PP_BUFF
        PP_BUFF --> WIN
        WIN --> FFT
        FFT --> MAG

        MAG -- "Magnitude (Activation)" --> ACT_BRAM
        ACT_BRAM <--> SYS_ARRAY
        WGT_BRAM --> SYS_ARRAY
        SYS_ARRAY --> REQ
        REQ --> ACT_BRAM
        
        CTL --- SYS_ARRAY
        CTL --- ACT_BRAM
        CTL --- WGT_BRAM

        FLASH_EXT --- FLASH_CTRL
        FLASH_CTRL -- "Weight Load" --> WGT_BRAM

        REQ -- "Final Vocal/Inst Masks" --> MASK
        MAG -- "Mixture Complex Spectrum" --> MASK
        MASK --> IFFT
        IFFT --> PP_BUFF
        PP_BUFF --> H_OUT
    end

    style SYS_ARRAY fill:#f96,stroke:#333,stroke-width:2px
    style FFT fill:#69f,stroke:#333
    style IFFT fill:#69f,stroke:#333
    style FLASH_EXT fill:#ddd,stroke-dasharray: 5 5
```

## ASCII Block Diagram

```text
                                +---------------------------------------------+
    Host (PC)                   |           FPGA (ULX3S - ECP5-85F)           |
  +-----------+                 |                                             |
  |           |  16kHz Mono     |  +---------------------------------------+  |
  | Raw Audio |------------------->|         UART / USB Transport          |  |
  |           |     Stream      |  +-------------------|-------------------+  |
  +-----------+                 |                      |                      |
        ^                       |          +-----------V-----------+          |
        |                       |          | Ping-Pong Frame Buffs |          |
        |                       |          +-----------|-----------+          |
        |                       |                      |                      |
        |                       |  +-------------------V-------------------+  |
        |                       |  |       DSP Front-End (STFT)            |  |
        |                       |  |  [Window] -> [FFT] -> [Mag/Phase]     |  |
        |                       |  +-------------------|-------------------+  |
        |                       |                      |                      |
        |                       |          +-----------V-----------+          |
        |                       |          |    Activation BRAMs   |<---+     |
        |                       |          +-----------|-----------+    |     |
        |                       |                      |                |     |
        |        2-Stem         |  +-------------------V-------------------+  |
        |        Audio          |  |      Neural Accelerator (GEMM)        |  |
        |        Stream         |  |   [INT8 Systolic Array (2D Grid)]     |  |
        |                       |  |   [Requantization (INT32->INT8)]      |  |
        |                       |  +-------------------|-------------------+  |
        |                       |                      ^                |     |
        |                       |          +-----------|-----------+    |     |
        |                       |          |      Weight BRAMs     |----+     |
        |                       |          +-----------^-----------+          |
        |                       |                      |                      |
        |                       |          +-----------|-----------+          |
        |                       |          |  QSPI Flash Controller|          |
        |                       |          +-----------^-----------+          |
        |                       |                      |                      |
        |                       |          +-----------|-----------+          |
        |                       |          | External QSPI Flash   |          |
        |                       |          +-----------------------+          |
        |                       |                      |                      |
        |                       |  +-------------------V-------------------+  |
        |                       |  |        DSP Back-End (iSTFT)           |  |
        |                       |  |    [Mask Apply] -> [Folded IFFT]      |  |
        |                       |  +-------------------|-------------------+  |
        |                       |                      |                      |
        |                       |          +-----------V-----------+          |
        |                       +----------|         UART TX       |----------+
        +----------------------------------+-----------------------+
```

## Component Breakdown

### 1. Host Interface (USB/UART)
- **Protocol:** 3 Mbaud UART over FTDI.
- **Audio:** 16 kHz, 16-bit PCM mono.
- **Backpressure:** The FPGA uses hardware flow control or a simple XON/XOFF-style protocol to prevent buffer overflow in the ping-pong registers.

### 2. Fixed-Function DSP (STFT/iSTFT)
- **Folded FFT:** Reuses a single Radix-2 butterfly to process the N-point FFT iteratively. This saves thousands of LUTs at the cost of execution time (negligible for 16 kHz audio).
- **Coordinate Transformation:** Magnitude is calculated via CORDIC or a polynomial approximation for the mask-generating network.
- **Phase Preservation:** The original phase of the mixture is held in buffers and reapplied to the magnitude masks for reconstruction.

### 3. Neural Accelerator (GEMM Core)
- **Systolic Array:** A 2D grid of `mac_pe.v` units. It is **output-stationary**, reducing data movement by keeping partial sums local to the PEs during dot-product accumulation.
- **BRAM Hierarchy:** 
    - **Activation BRAM:** Stores input features and intermediate hidden states.
    - **Weight BRAM:** Stores the current model weights (up to ~300K INT8 parameters).
- **Requantization:** Implements the `(multiplier, shift)` logic required to bring INT32 accumulations back to INT8 range while applying the activation function (ReLU).

### 4. Memory Management
- **QSPI Flash:** The ULX3S 16MB flash stores multiple bitstreams and multiple neural model weights.
- **Weight Loading FSM:** A dedicated controller that handles the QSPI protocol to burst weights into the high-speed BRAM at boot time or during a model switch.
