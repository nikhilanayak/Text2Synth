#!/bin/bash

# CLAP-Synth Setup Script
# Works on macOS (Darwin) and Linux

set -e

echo "--- CLAP-Synth Environment Setup ---"

# 1. OS Detection
OS="$(uname)"
echo "Detected OS: $OS"

# 2. Check for system dependencies
MISSING_DEPS=()
command -v python3 >/dev/null 2>&1 || MISSING_DEPS+=("python3")
command -v git >/dev/null 2>&1 || MISSING_DEPS+=("git")
command -v ffmpeg >/dev/null 2>&1 || MISSING_DEPS+=("ffmpeg")
command -v iverilog >/dev/null 2>&1 || MISSING_DEPS+=("iverilog")

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo "Error: Missing system dependencies: ${MISSING_DEPS[*]}"
    if [ "$OS" == "Darwin" ]; then
        echo "Hint: Install via Homebrew: brew install python ffmpeg icarus-verilog"
    else
        echo "Hint: Install via apt: sudo apt install python3-venv ffmpeg iverilog"
    fi
    exit 1
fi

# 3. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 4. Install Python Dependencies
echo "Installing Python requirements..."
source venv/bin/activate
pip install --upgrade pip
pip install -r software/requirements.txt

# 5. Initialize Data Folders
echo "Initializing data directories..."
mkdir -p software/data
mkdir -p software/litmus_output
mkdir -p build

# 6. Weights & Biases
echo "Checking W&B login status..."
if ! wandb status >/dev/null 2>&1; then
    echo "Hint: You will need to run 'wandb login' before training."
fi

echo ""
echo "--- Setup Complete ---"
echo "To start training, run:"
echo "  source venv/bin/activate"
echo "  python3 software/train_fm.py"
echo ""
echo "To run hardware simulations, run:"
echo "  make"
