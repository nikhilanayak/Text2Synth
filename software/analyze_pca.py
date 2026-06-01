import torch
import numpy as np
from sklearn.decomposition import PCA
import os

def analyze_clap_variance():
    cache_path = "software/data/esc50_embeds_v6_filtered.pth"
    if not os.path.exists(cache_path):
        print("Cache not found. Run training once to generate it.")
        return

    print("Loading CLAP embeddings...")
    cache = torch.load(cache_path, map_location="cpu")
    # Use both audio and text embeds to capture the full space
    a_emb = torch.cat(cache['audio_embeds'], dim=0).numpy()
    t_emb = torch.cat(cache['text_embeds'], dim=0).numpy()
    X = np.concatenate([a_emb, t_emb], axis=0)

    print(f"Input Shape: {X.shape}")

    # Run PCA
    pca = PCA()
    pca.fit(X)

    # Calculate cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    print("\n--- PCA Explained Variance ---")
    for n in [16, 32, 64, 128, 256, 512]:
        if n <= len(cumsum):
            print(f"  {n} Components: {cumsum[n-1]*100:.2f}% of information")

    target_95 = np.where(cumsum >= 0.95)[0][0] + 1
    print(f"\nRecommendation: Use {target_95} components to keep 95% of the sonic 'soul'.")

if __name__ == "__main__":
    analyze_clap_variance()
