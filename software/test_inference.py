import torch
import torchaudio
import musdb
from model import TinyCausalTCN
import os

# --- Configuration ---
SR = 16000
N_FFT = 512
HOP = 256
CHECKPOINT = "software/tiny_tcn_epoch_7.pth" # Using your latest log
OUTPUT_DIR = "software/output_test"

def test_inference():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    model = TinyCausalTCN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # 2. Load a track from musdb
    db = musdb.DB(root="software/data/musdb18", subsets=["train"])
    track = db.tracks[0] # Just the first track
    print(f"Testing on: {track.name}")

    # Process only the first 10 seconds to save time
    audio = torch.from_numpy(track.audio[:SR*10].T).float()
    
    # Resample and Mix to Mono
    mix = torchaudio.transforms.Resample(track.rate, SR)(audio).mean(0, keepdim=True)
    
    # 3. Perform STFT
    spec_complex = torch.stft(mix, n_fft=N_FFT, hop_length=HOP, return_complex=True)
    spec_mag = spec_complex.abs().to(device)
    spec_phase = spec_complex.angle()

    # 4. Predict Masks
    with torch.no_grad():
        v_mask, i_mask = model(spec_mag)

    # 5. Apply Masks and Reconstruct
    # v_mask shape: (1, bins, time)
    v_spec = (spec_mag * v_mask).cpu()
    i_spec = (spec_mag * i_mask).cpu()

    # Add phase back
    v_complex = torch.polar(v_spec, spec_phase)
    i_complex = torch.polar(i_spec, spec_phase)

    # Inverse STFT
    v_audio = torch.istft(v_complex, n_fft=N_FFT, hop_length=HOP)
    i_audio = torch.istft(i_complex, n_fft=N_FFT, hop_length=HOP)

    # v_audio shape is (batch, time), here batch=1, so squeeze to (time) then unsqueeze(0) for torchaudio
    v_audio = v_audio.squeeze(0)
    i_audio = i_audio.squeeze(0)

    # 6. Save results
    torchaudio.save(os.path.join(OUTPUT_DIR, "original_mix.wav"), mix, SR)
    torchaudio.save(os.path.join(OUTPUT_DIR, "est_vocals.wav"), v_audio.unsqueeze(0), SR)
    torchaudio.save(os.path.join(OUTPUT_DIR, "est_instr.wav"), i_audio.unsqueeze(0), SR)
    
    print(f"Done! Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    test_inference()
