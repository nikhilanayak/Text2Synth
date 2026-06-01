# CLAP-Synth: Embedded Neural FM Synthesizer
![Made With AI](https://ai-label.org/image-pack/ai-label_banner-made-with-ai.svg)

A generative synthesizer that uses on-chip Neural Inference to design its own internal FM patches based on semantic embeddings.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have the following installed:
- **macOS:** `brew install python ffmpeg icarus-verilog`
- **Linux:** `sudo apt install python3-venv ffmpeg iverilog`

### 2. Environment Setup
Run the automated setup script:
```bash
make setup
```
This will create a virtual environment, install all dependencies, and initialize data folders.

### 3. Weights & Biases
Login to W&B to track your training:
```bash
source venv/bin/activate
wandb login
```

### 4. Training the Neural Mapper
Start the two-phase training loop (Physical Grounding & Semantic Refinement):
```bash
python3 software/train_fm.py
```

## 🛠 Hardware Implementation (RTL)

The synthesizer core is located in `rtl/`. You can run simulations using Icarus Verilog:

```bash
make mac_pe                # Test the Matrix-Vector cell
make fm_phase_accumulator  # Test the DDS core
```

## 🏗 Architecture

- **Neural Engine:** INT8 MLP Decoder running on FPGA fabric.
- **Synth Engine:** 4-Operator FM Synthesizer + Filtered Noise Source.
- **Host Interface:** 512-D CLAP embeddings via UART.

---
**Architect:** Nikhil Nayak
