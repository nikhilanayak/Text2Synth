import torch
import numpy as np
import torchaudio
import os
from tqdm import tqdm
from diff_fm import DifferentiableAdditiveSynth
import laion_clap

# --- Configuration ---
N_SAMPLES = 20000  # Total number of random patches to bake
BATCH_SIZE = 64    # For fast encoding
SR = 16000
DURATION = 1.0
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH = "software/data/physics_grounding_data.pth"

def bake_data():
    print(f"Baking {N_SAMPLES} samples on {DEVICE}...")
    
    # 1. Initialize Components
    synth = DifferentiableAdditiveSynth(sr=SR, duration=DURATION).to(DEVICE)
    synth.eval()
    
    print("Loading CLAP for encoding...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
    clap_model.load_ckpt() 
    clap_model.to(DEVICE)
    clap_model.eval()

    resample_48k = torchaudio.transforms.Resample(SR, 48000).to(DEVICE)

    baked_embeddings = []
    baked_params = []

    # 2. Generation Loop
    n_batches = N_SAMPLES // BATCH_SIZE
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Baking Patches"):
            # Random parameters
            params = torch.rand(BATCH_SIZE, 75).to(DEVICE)
            # Random pitch for diversity
            f0 = np.exp(np.random.uniform(np.log(50), np.log(1000)))
            
            # Synthesize
            audio = synth(params, f0=f0)
            # Normalize
            audio = audio / (torch.max(torch.abs(audio), dim=1, keepdim=True)[0] + 1e-8)
            
            # Get CLAP Embedding
            audio_48k = resample_48k(audio)
            embeddings = clap_model.get_audio_embedding_from_data(x=audio_48k, use_tensor=True)
            
            # Store (Moving to CPU to save GPU memory)
            baked_embeddings.append(embeddings.cpu())
            baked_params.append(params.cpu())

    # 3. Finalize and Save
    baked_embeddings = torch.cat(baked_embeddings, dim=0)
    baked_params = torch.cat(baked_params, dim=0)
    
    data = {
        'embeddings': baked_embeddings,
        'params': baked_params,
        'sr': SR,
        'duration': DURATION
    }
    
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))
        
    torch.save(data, OUTPUT_PATH)
    print(f"\nBaking complete! Saved {N_SAMPLES} samples to {OUTPUT_PATH}")
    print(f"Total size: {os.path.getsize(OUTPUT_PATH) / 1024**2:.2f} MB")

if __name__ == "__main__":
    bake_data()
