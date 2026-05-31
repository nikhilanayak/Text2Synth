import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
from tqdm import tqdm

from diff_fm import DifferentiableFMSynth
from loss import MultiScaleSpectralLoss, CLAPLoss
import laion_clap

# --- Configuration ---
SR = 16000
DURATION = 1.0
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Training on device: {DEVICE}")

class Mapper(nn.Module):
    """
    Neural Mapper (MLP)
    Translates 512-D CLAP embedding into 48-D FM Parameters.
    """
    def __init__(self, input_dim=512, output_dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dim),
            nn.Sigmoid() # Constrain parameters to [0, 1]
        )

    def forward(self, x):
        return self.net(x)

class AKWFDataset(Dataset):
    """
    Dataset of Adventure Kid Wavetables.
    Resamples and repeats single-cycle waveforms to DURATION.
    """
    def __init__(self, root_dir, sr=SR, duration=DURATION):
        self.file_paths = glob.glob(os.path.join(root_dir, "**/*.wav"), recursive=True)
        self.sr = sr
        self.n_samples = int(sr * duration)
        print(f"Found {len(self.file_paths)} wavetables.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        audio, orig_sr = torchaudio.load(path)
        
        # Resample to target SR
        if orig_sr != self.sr:
            audio = torchaudio.transforms.Resample(orig_sr, self.sr)(audio)
            
        # Repeat to fill duration
        # audio is (channels, frames)
        n_repeats = (self.n_samples // audio.shape[1]) + 1
        audio = audio.repeat(1, n_repeats)[:, :self.n_samples]
        
        return audio.squeeze(0) # (n_samples)

def train():
    # 1. Load CLAP Model
    print("Loading CLAP model...")
    # Note: Use a smaller model version if available for memory efficiency
    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    # This downloads weights automatically on first run
    clap_model.load_ckpt() 
    clap_model.to(DEVICE)
    clap_model.eval()

    # 2. Setup Data
    dataset = AKWFDataset("software/data/wavetables/AKWF-FREE/AKWF")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Setup Models & Losses
    mapper = Mapper().to(DEVICE)
    synth = DifferentiableFMSynth(sr=SR, duration=DURATION).to(DEVICE)
    
    spec_loss_fn = MultiScaleSpectralLoss().to(DEVICE)
    clap_loss_fn = CLAPLoss().to(DEVICE)
    
    optimizer = optim.Adam(mapper.parameters(), lr=LR)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for target_audio in pbar:
            target_audio = target_audio.to(DEVICE)
            
            # A. Get Target Embedding (Self-Supervised)
            # CLAP expects 48kHz for the audio encoder usually, or it resamples
            # We'll let the library handle it or resample if needed
            with torch.no_grad():
                # clap_model.get_audio_embedding expects a list of numpy or torch
                # We need to make sure the format is correct
                # Standard CLAP audio encoder expects 48000 Hz usually
                target_audio_48k = torchaudio.transforms.Resample(SR, 48000)(target_audio)
                target_embed = clap_model.get_audio_embedding_from_data(x=target_audio_48k, use_tensor=True)
            
            # B. Map Embedding to FM Params
            optimizer.zero_grad()
            predicted_params = mapper(target_embed)
            
            # C. Synthesize Audio
            # We use a fixed f0 for now (e.g. 110Hz) to match the wavetable's tonal nature
            # In a better version, we would extract f0 from the wavetable
            gen_audio = synth(predicted_params, f0=110.0)
            
            # D. Compute Losses
            # Spectral Loss (DDSP)
            loss_spec = spec_loss_fn(target_audio, gen_audio)
            
            # Semantic Loss (CLAP)
            gen_audio_48k = torchaudio.transforms.Resample(SR, 48000)(gen_audio)
            gen_embed = clap_model.get_audio_embedding_from_data(x=gen_audio_48k, use_tensor=True)
            loss_clap = clap_loss_fn(target_embed, gen_embed)
            
            # Total Loss
            # We scale the spectral loss so it's comparable to the cosine distance
            loss = (loss_spec * 0.1) + loss_clap 
            
            # E. Backprop
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", spec=f"{loss_spec.item():.4f}", clap=f"{loss_clap.item():.4f}")
            
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(dataloader):.6f}")
        torch.save(mapper.state_dict(), f"software/mapper_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
