import torch
import torchaudio
import os
import laion_clap
from diff_fm import DifferentiableFMSynth
from train_fm import Mapper # Import the exact class definition

# --- Configuration ---
SR = 16000
DURATION = 1.0
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_PATH = "software/training_checkpoint.pth"
OUTPUT_DIR = "software/litmus_output"

def run_litmus_test():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Models
    print(f"Loading Mapper from {CHECKPOINT_PATH}...")
    mapper = Mapper(output_dim=52).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Handle both full checkpoint and just state_dict
    if 'mapper_state_dict' in checkpoint:
        mapper.load_state_dict(checkpoint['mapper_state_dict'])
        print(f"Resumed from Phase {checkpoint['phase']}, Step {checkpoint['step']}")
    else:
        mapper.load_state_dict(checkpoint)
        
    mapper.eval()

    synth = DifferentiableFMSynth(sr=SR, duration=DURATION).to(DEVICE)
    synth.eval()

    print("Loading CLAP for text encoding...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
    clap_model.load_ckpt() 
    clap_model.to(DEVICE)
    clap_model.eval()

    # 2. Test Prompts (Representative ESC-50 classes)
    prompts = [
        "A recording of the sound of a barking dog",
        "A recording of the sound of heavy rain",
        "A recording of the sound of a chainsaw",
        "A recording of the sound of church bells",
        "A recording of the sound of a chirping cricket"
    ]

    print("\n--- Running Inference ---")
    with torch.no_grad():
        for i, text in enumerate(prompts):
            print(f"Synthesizing: '{text}'...")
            
            # Get Text Embedding
            text_embed = clap_model.get_text_embedding([text], use_tensor=True).to(DEVICE)
            
            # Map to Params
            predicted_params = mapper(text_embed)
            
            # Synthesize
            # We use 110Hz (A2) as the base pitch for all tests to keep it consistent
            gen_audio = synth(predicted_params, f0=110.0)
            
            # Normalize and Save
            audio_out = gen_audio[0]
            audio_out = audio_out / (torch.max(torch.abs(audio_out)) + 1e-8)
            
            fname = text.split("of ")[-1].replace(" ", "_") + ".wav"
            path = os.path.join(OUTPUT_DIR, fname)
            torchaudio.save(path, audio_out.unsqueeze(0).cpu(), SR)
            print(f"Saved to: {path}")

    print("\nLitmus test complete. Check the 'software/litmus_output' folder.")

if __name__ == "__main__":
    run_litmus_test()
