import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from tqdm import tqdm
import os
import json
import pandas as pd
import wandb
import joblib

from diff_fm import DifferentiableAdditiveSynth
from loss import MultiScaleSpectralLoss, CLAPLoss
import laion_clap

# --- Configuration ---
SR = 16000
DURATION = 1.0
BATCH_SIZE = 64 
LR = 5e-4 
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_PATH = "software/training_checkpoint.pth"
PCA_MODEL_PATH = "software/data/pca_model_128.joblib"
WANDB_PROJECT = "clap-synth-fpga"

# --- Model Definitions ---

def init_weights(m):
    if isinstance(m, nn.Linear):
        w = m.weight.data.cpu()
        nn.init.orthogonal_(w)
        m.weight.data.copy_(w)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Mapper(nn.Module):
    def __init__(self, input_dim=128, output_dim=75): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.net(x)

def save_checkpoint(mapper, optimizer, stage, step, path):
    state = {
        'mapper_state_dict': mapper.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stage': stage,
        'step': step
    }
    torch.save(state, path)

def load_checkpoint(mapper, optimizer, path):
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=DEVICE)
        try:
            mapper.load_state_dict(checkpoint['mapper_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint.get('stage', 1), checkpoint.get('step', 0)
        except:
            return 1, 0
    return 1, 0

def setup_esc50():
    data_dir = "software/data/esc50"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.system(f"git clone https://github.com/karolpiczak/ESC-50.git {data_dir}")
    csv_path = os.path.join(data_dir, "meta/esc50.csv")
    df = pd.read_csv(csv_path)
    all_files = df['filename'].apply(lambda x: os.path.join(data_dir, "audio", x)).tolist()
    class_names = df['category'].unique().tolist()
    return all_files, class_names

def log_audio_to_wandb(target_audio, gen_audio, step, stage, label="Sample"):
    def norm(x): return x / (torch.max(torch.abs(x)) + 1e-8)
    wandb.log({
        f"Audio/{label}_Target": wandb.Audio(norm(target_audio).cpu().numpy(), sample_rate=SR),
        f"Audio/{label}_Generated": wandb.Audio(norm(gen_audio).detach().cpu().numpy(), sample_rate=SR),
        "Step": step,
        "Stage": stage
    }, commit=False)

def slerp(val, low, high):
    low_norm = low / (torch.norm(low, dim=1, keepdim=True) + 1e-8)
    high_norm = high / (torch.norm(high, dim=1, keepdim=True) + 1e-8)
    omega = torch.acos(torch.clamp(torch.sum(low_norm * high_norm, dim=1, keepdim=True), -1, 1))
    so = torch.sin(omega)
    res = torch.where(so > 1e-6, 
                      (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high,
                      low)
    return res

def train():
    wandb.init(project=WANDB_PROJECT)
    
    pca = joblib.load(PCA_MODEL_PATH)
    pca_V = torch.from_numpy(pca.components_).float().to(DEVICE)
    pca_mean = torch.from_numpy(pca.mean_).float().to(DEVICE)
    def apply_pca(x): return torch.mm(x - pca_mean, pca_V.T)

    mapper = Mapper(input_dim=128, output_dim=75).to(DEVICE)
    synth = DifferentiableAdditiveSynth(sr=SR, duration=DURATION).to(DEVICE)
    
    print("Loading CLAP model...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
    clap_model.load_ckpt() 
    clap_model.to(DEVICE)
    clap_model.eval()

    clap_loss_fn = CLAPLoss().to(DEVICE)
    resample_48k = torchaudio.transforms.Resample(SR, 48000).to(DEVICE)
    optimizer = optim.Adam(mapper.parameters(), lr=LR)

    current_stage, start_step = load_checkpoint(mapper, optimizer, CHECKPOINT_PATH)
    if start_step == 0 and current_stage == 1:
        mapper.apply(init_weights)

    # --- STAGE 1: Extended Physics Inversion (10,000 steps) ---
    stage_1_steps = 10000
    BAKED_DATA_PATH = "software/data/physics_grounding_data.pth"

    if current_stage == 1:
        print(f"\n--- Stage 1: Additive Physics Inversion ---")
        
        if os.path.exists(BAKED_DATA_PATH):
            print(f"Loading pre-baked grounding data from {BAKED_DATA_PATH}...")
            baked = torch.load(BAKED_DATA_PATH, map_location=DEVICE)
            baked_embeddings = baked['embeddings'].to(DEVICE)
            baked_params = baked['params'].to(DEVICE)
            use_baked = True
        else:
            print("No baked data found. Running in SLOW LIVE MODE. Run 'python3 software/bake_physics_data.py' first for speed.")
            use_baked = False

        mapper.train()
        pbar = tqdm(range(start_step, stage_1_steps), desc="Stage 1")
        for step in pbar:
            if use_baked:
                idx = torch.randint(0, len(baked_embeddings), (BATCH_SIZE,))
                input_embed_pca = apply_pca(baked_embeddings[idx])
                target_params = baked_params[idx]
            else:
                target_params = torch.rand(BATCH_SIZE, 75).to(DEVICE)
                f0 = np.exp(np.random.uniform(np.log(50), np.log(1000)))
                with torch.no_grad():
                    audio = synth(target_params, f0=f0)
                    audio = audio / (torch.max(torch.abs(audio), dim=1, keepdim=True)[0] + 1e-8)
                    audio_48k = resample_48k(audio)
                    audio_embed_raw = clap_model.get_audio_embedding_from_data(x=audio_48k, use_tensor=True)
                    input_embed_pca = apply_pca(audio_embed_raw)
            
            optimizer.zero_grad()
            predicted_params = mapper(input_embed_pca)
            loss_mse = nn.MSELoss()(predicted_params, target_params)
            diversity_loss = -predicted_params.std(dim=0).mean()
            
            total_loss = loss_mse + (diversity_loss * 0.2)
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                pred_std = predicted_params.std(dim=0).mean()

            pbar.set_postfix(mse=f"{loss_mse.item():.4f}", std=f"{pred_std.item():.4f}")
            wandb.log({
                "Stage": 1, "Loss/Param_MSE": loss_mse.item(), "Diagnostic/Prediction_STD": pred_std.item()
            })
            
            if step % 500 == 0:
                with torch.no_grad():
                    gen_audio = synth(predicted_params[0:1], f0=110.0)
                    log_audio_to_wandb(torch.zeros(16000), gen_audio[0], step, 1, label="Stage1_Inversion")
            
            if (step + 1) % 500 == 0:
                save_checkpoint(mapper, optimizer, 1, step + 1, CHECKPOINT_PATH)

        save_checkpoint(mapper, optimizer, 2, 0, CHECKPOINT_PATH)
        current_stage = 2
        start_step = 0

    # --- STAGE 2: Semantic Alignment ---
    stage_2_steps = 5000
    if current_stage == 2:
        print(f"\n--- Stage 2: Additive Semantic Alignment ---")
        embed_cache_path = "software/data/esc50_embeds_v6_filtered.pth"
        cache = torch.load(embed_cache_path, map_location=DEVICE)
        audio_embeds_raw = torch.cat(cache['audio_embeds'], dim=0).to(DEVICE)
        text_embeds_raw = torch.cat(cache['text_embeds'], dim=0).to(DEVICE)
        
        mapper.train()
        pbar = tqdm(range(start_step, stage_2_steps), desc="Stage 2")
        for step in pbar:
            use_text = (step % 2 == 0)
            idx = np.random.randint(0, len(audio_embeds_raw), BATCH_SIZE)
            target_embed_raw = text_embeds_raw[idx] if use_text else audio_embeds_raw[idx]
            
            input_pca = apply_pca(target_embed_raw)
            input_pca = input_pca + torch.randn_like(input_pca) * 0.02
            
            f0 = 110.0
            optimizer.zero_grad()
            predicted_params = mapper(input_pca)
            gen_audio = synth(predicted_params, f0=f0)
            gen_audio_48k = resample_48k(gen_audio)
            gen_embed_raw = clap_model.get_audio_embedding_from_data(x=gen_audio_48k, use_tensor=True)
            
            loss = clap_loss_fn(target_embed_raw, gen_embed_raw)
            total_loss = loss + (torch.abs(gen_audio[:, 1:] - gen_audio[:, :-1]).mean() * 0.05)
            
            total_loss.backward()
            optimizer.step()
            
            mode_str = "Text" if use_text else "Audio"
            wandb.log({"Stage": 2, f"Loss/Semantic_{mode_str}": loss.item(), "Loss/Total": total_loss.item()})
            if step % 250 == 0:
                log_audio_to_wandb(torch.zeros(int(SR*DURATION)), gen_audio[0], step, 2, label=f"Stage2_{mode_str}")
            if (step + 1) % 250 == 0:
                save_checkpoint(mapper, optimizer, 2, step + 1, CHECKPOINT_PATH)

    torch.save(mapper.state_dict(), "software/mapper_final.pth")
    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    train()
