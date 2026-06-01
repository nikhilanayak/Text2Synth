import torch
import numpy as np
from sklearn.decomposition import PCA
import joblib
import os

# --- Configuration ---
N_COMPONENTS = 128
CACHE_PATH = "software/data/esc50_embeds_v6_filtered.pth"
PCA_MODEL_PATH = "software/data/pca_model_128.joblib"

def fit_and_save_pca():
    if not os.path.exists(CACHE_PATH):
        print(f"Error: Cache not found at {CACHE_PATH}")
        return

    print(f"Loading CLAP embeddings from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location="cpu")
    
    # Concatenate all available embeddings to define the space
    a_emb = torch.cat(cache['audio_embeds'], dim=0).numpy()
    t_emb = torch.cat(cache['text_embeds'], dim=0).numpy()
    X = np.concatenate([a_emb, t_emb], axis=0)

    print(f"Fitting PCA with {N_COMPONENTS} components on {X.shape[0]} samples...")
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(X)
    
    # Save the model
    joblib.dump(pca, PCA_MODEL_PATH)
    print(f"PCA model saved to {PCA_MODEL_PATH}")
    print(f"Total Explained Variance: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

if __name__ == "__main__":
    fit_and_save_pca()
