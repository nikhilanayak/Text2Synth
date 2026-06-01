import torch
import os

def diagnose_cache():
    cache_path = "software/data/esc50_embeds_v6_filtered.pth"
    if not os.path.exists(cache_path):
        print("Cache not found.")
        return

    print(f"Loading cache: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu")
    
    file_to_audio = cache.get('file_to_audio', {})
    if not file_to_audio:
        print("No audio found in cache.")
        return
        
    print(f"Found {len(file_to_audio)} audio samples in cache.")
    
    # Check a few random ones
    keys = list(file_to_audio.keys())[:5]
    for k in keys:
        audio = file_to_audio[k]
        max_val = torch.max(torch.abs(audio)).item()
        mean_val = torch.mean(torch.abs(audio)).item()
        print(f"File: {os.path.basename(k)} | Max: {max_val:.6f} | Mean: {mean_val:.6f}")

if __name__ == "__main__":
    diagnose_cache()
