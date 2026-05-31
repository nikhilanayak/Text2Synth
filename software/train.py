import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import musdb
import numpy as np
from tqdm import tqdm
from model import TinyCausalTCN

# --- Configuration ---
SR = 16000
N_FFT = 512
HOP = 256
DURATION = 5.0  # 5 second chunks for training
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

# Set device: MPS (Apple Silicon), CUDA, or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class MusdbDataset(Dataset):
    def __init__(self, root, split='train', duration=DURATION):
        # subsets should be a list, e.g., ['train']
        self.db = musdb.DB(root=root, subsets=[split])
        self.duration = duration
        self.sr = SR

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        track = self.db.tracks[idx]
        
        # Random crop
        start = np.random.uniform(0, track.duration - self.duration)
        track.chunk_start = start
        track.chunk_duration = self.duration
        
        # Load mixture and stems
        # musdb returns (time, channels) at 44.1kHz
        mix = torch.from_numpy(track.audio.T).float()
        vocals = torch.from_numpy(track.targets['vocals'].audio.T).float()
        instr = torch.from_numpy(track.targets['accompaniment'].audio.T).float()
        
        # Resample to 16kHz and mix to mono
        mix = torchaudio.transforms.Resample(track.rate, self.sr)(mix).mean(0, keepdim=True)
        vocals = torchaudio.transforms.Resample(track.rate, self.sr)(vocals).mean(0, keepdim=True)
        instr = torchaudio.transforms.Resample(track.rate, self.sr)(instr).mean(0, keepdim=True)
        
        # Compute STFT Magnitudes
        spec_mix = torch.stft(mix, n_fft=N_FFT, hop_length=HOP, return_complex=True).abs()
        spec_voc = torch.stft(vocals, n_fft=N_FFT, hop_length=HOP, return_complex=True).abs()
        spec_ins = torch.stft(instr, n_fft=N_FFT, hop_length=HOP, return_complex=True).abs()
        
        # Squeeze channel dim: (1, bins, time) -> (bins, time)
        return spec_mix[0], spec_voc[0], spec_ins[0]

def train():
    # 1. Data Setup
    data_path = "software/data/musdb18"
    dataset = MusdbDataset(root=data_path, split='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model, Loss, Optimizer
    model = TinyCausalTCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for mix, voc_gt, ins_gt in pbar:
            mix, voc_gt, ins_gt = mix.to(device), voc_gt.to(device), ins_gt.to(device)
            
            optimizer.zero_grad()
            
            # Predict masks
            v_mask, i_mask = model(mix)
            
            # Apply masks to mixture to get estimates
            v_est = mix * v_mask
            i_est = mix * i_mask
            
            # Loss: MSE on magnitudes
            loss = criterion(v_est, voc_gt) + criterion(i_est, ins_gt)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.6f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"software/tiny_tcn_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
