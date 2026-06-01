import torch
import os
from train_fm import Mapper
import laion_clap

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_PATH = "software/training_checkpoint.pth"

def check_param_spread():
    # 1. Load Model
    mapper = Mapper(output_dim=52).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'mapper_state_dict' in checkpoint:
        mapper.load_state_dict(checkpoint['mapper_state_dict'])
    else:
        mapper.load_state_dict(checkpoint)
    mapper.eval()

    # 2. Load CLAP
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
    clap_model.load_ckpt()
    clap_model.to(DEVICE)

    # 3. Get Embeddings for very different things
    prompts = ["A barking dog", "Heavy rain", "A metallic bell", "White noise"]
    with torch.no_grad():
        embeds = clap_model.get_text_embedding(prompts, use_tensor=True).to(DEVICE)
        params = mapper(embeds)
    
    print("\n--- Parameter Diagnostic ---")
    for i, p in enumerate(prompts):
        print(f"\nPrompt: {p}")
        # Print key parameter groups
        print(f"  Ratios: {params[i, 0:4].cpu().numpy()}")
        print(f"  Amps:   {params[i, 4:8].cpu().numpy()}")
        print(f"  Mod Matrix Mean: {params[i, 24:40].mean().item():.4f}")
        print(f"  Noise Amp: {params[i, 48].item():.4f}")
        print(f"  Feedback Mean: {params[i, 40:44].mean().item():.4f}")

if __name__ == "__main__":
    check_param_spread()
