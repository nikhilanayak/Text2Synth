# Text2Synth: Synthesized Audio Generation Model Programmed on Cyclone IV FPGA

## How It Works
Text2Synth takes text prompts and converts them to audio in two steps: inference and synthesization

### Inference
The goal of this step is to convert a text prompt into a 78-parameter vector that encodes the sound as abstract synthesizer parameters. First, the prompt is embedded using the CLAP (Contrastive Audio-Language Pretraining) model. Then, a search algorithm is used to determine the closest synthesizer parameters that result in a similar CLAP embedding vector. This process is heavily inspired by the CTAG project, which uses the same basic process for audio synthesization. 

### Synthesization
Next, the 78-parameter vector is converted into an audio signal using the digital synthesizer and streamed back to the host computer. This process takes as input the current MIDI note being played, meaning once inference is done, the FPGA can act as a realtime instrument with relatively low latency. 

## Quick Start

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

### CTAG paper reproduction

The reference-compatible reproduction of *Creative Text-to-Audio Generation
via Synthesizer Programming* lives in [`ctag-repro/`](ctag-repro/). It includes
the official CLAP checkpoint downloader, SynthAX/LES search pipeline, tests,
reproducibility metadata, and a latest-runtime Google Colab notebook.

```bash
cd ctag-repro
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[paper]'
ctag setup --download --strict
ctag generate --profile smoke --prompt "train horn"
```

For a GPU runtime, [open the runnable Colab notebook](https://colab.research.google.com/github/nikhilanayak/Synthesizer/blob/main/ctag-repro/CTAG_Colab.ipynb)
or follow [`ctag-repro/COLAB.md`](ctag-repro/COLAB.md). It uses Colab's native
Python 3.13, JAX CUDA plugin, and PyTorch CUDA build without replacing their
accelerator wheels.

To train a live prompt-to-78-parameter network that replaces inference-time
search, open the [direct-model training notebook](https://colab.research.google.com/github/nikhilanayak/Synthesizer/blob/main/ctag-repro/DirectPatch_Train.ipynb).
It persists its dataset and checkpoints to Google Drive and exports an INT8
hardware-facing ONNX model when training completes.

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

## Hardware Implementation (RTL)

The completed first hardware milestone is a monophonic 48 kHz parameter-to-audio
core with eight compiled patches, live note/preset/edit buttons, hexadecimal
seven-segment output, atomic parameter writes for the future neural engine, and
framed PCM streaming back to the computer over Intel JTAG UART or 3 Mbaud UART.

```bash
make synthax
python3 -m pytest -q tests/test_rtl_synth.py
python3 host/play_fpga_audio.py --serial /dev/ttyUSB0 --baud 3000000
```

Start with [`hardware/README.md`](hardware/README.md) for controls, preset
burning, transport setup, parameter contract, and Cyclone IV bring-up. The
provided Quartus project is intentionally board-neutral until the exact board
part number and pin map are known.

## Architecture

- **ML:** released CLAP encoder, CTAG/SynthAX search teachers, and an eight-head
  direct mapper for live inference.
- **Synth engine:** SynthAX Voice's two oscillators, noise, six ADSRs, two LFOs,
  and 5×4 modulation matrix in fixed-point RTL.
- **Host interface:** CRC-framed 48 kHz PCM16 output. The PC plays audio because
  an on-board audio jack is not assumed.
