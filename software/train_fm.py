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

from diff_fm import DifferentiableFMSynth
from loss import MultiScaleSpectralLoss, CLAPLoss
import laion_clap

# --- Configuration ---
SR = 16000 
DURATION = 1.0
BATCH_SIZE = 4 
LR = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_PATH = "software/training_checkpoint.pth"
WANDB_PROJECT = "clap-synth-fpga"

# --- Model Definitions ---

class Mapper(nn.Module):
    def __init__(self, input_dim=512, output_dim=52): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.net(x)

def slerp(val, low, high):
    low_norm = low / (torch.norm(low, dim=1, keepdim=True) + 1e-8)
    high_norm = high / (torch.norm(high, dim=1, keepdim=True) + 1e-8)
    omega = torch.acos(torch.clamp(torch.sum(low_norm * high_norm, dim=1, keepdim=True), -1, 1))
    so = torch.sin(omega)
    res = torch.where(so > 1e-6, 
                      (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high,
                      low)
    return res

def save_checkpoint(mapper, optimizer, phase, step, path):
    state = {
        'mapper_state_dict': mapper.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'phase': phase,
        'step': step
    }
    torch.save(state, path)

def load_checkpoint(mapper, optimizer, path):
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=DEVICE)
        mapper.load_state_dict(checkpoint['mapper_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['phase'], checkpoint['step']
    return 1, 0

def setup_esc50():
    data_dir = "software/data/esc50"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Downloading ESC-50...")
        os.system(f"git clone https://github.com/karolpiczak/ESC-50.git {data_dir}")
    
    csv_path = os.path.join(data_dir, "meta/esc50.csv")
    df = pd.read_csv(csv_path)
    class_to_files = df.groupby('category')['filename'].apply(list).to_dict()
    for cat in class_to_files:
        class_to_files[cat] = [os.path.join(data_dir, "audio", f) for f in class_to_files[cat]]
    
    all_files = df['filename'].apply(lambda x: os.path.join(data_dir, "audio", x)).tolist()
    return all_files, class_to_files

def log_audio_to_wandb(target_audio, gen_audio, step, phase, label="Sample"):
    def norm(x): return x / (torch.max(torch.abs(x)) + 1e-8)
    
    wandb.log({
        f"Audio/{label}_Target": wandb.Audio(norm(target_audio[0]).cpu().numpy(), sample_rate=SR),
        f"Audio/{label}_Generated": wandb.Audio(norm(gen_audio[0]).detach().cpu().numpy(), sample_rate=SR),
        "Step": step,
        "Phase": phase
    }, commit=False)

def train():
    wandb.init(project=WANDB_PROJECT)
    print(f"Starting training on {DEVICE} at {SR}Hz...")
    
    mapper = Mapper().to(DEVICE)
    synth = DifferentiableFMSynth(sr=SR, duration=DURATION).to(DEVICE)
    
    print("Loading CLAP model...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
    clap_model.load_ckpt() 
    clap_model.to(DEVICE)
    clap_model.eval()

    clap_loss_fn = CLAPLoss().to(DEVICE)
    resample_48k = torchaudio.transforms.Resample(SR, 48000).to(DEVICE)
    optimizer = optim.Adam(mapper.parameters(), lr=LR)

    current_phase, start_step = load_checkpoint(mapper, optimizer, CHECKPOINT_PATH)
    
    all_files, class_to_files = setup_esc50()
    class_names = list(class_to_files.keys())

    # 3. Pre-compute embeddings for ESC-50 (Incremental Caching with Filtering)
    embed_cache_path = "software/data/esc50_embeds_v6_filtered.pth"
    CONSISTENCY_THRESHOLD = 0.25 # Initial threshold (we can tune this)

    if os.path.exists(embed_cache_path):
        print("Loading filtered incremental embedding cache...")
        cache = torch.load(embed_cache_path, map_location=DEVICE)
    else:
        cache = {'audio_embeds': [], 'text_embeds': [], 'file_to_audio': {}, 'class_to_text_embed': {}, 'processed_files': set(), 'similarities': []}
        print("Generating filtered dual-modal incremental embedding cache...")

    files_to_process = [f for f in all_files if f not in cache['processed_files']]
    if files_to_process:
        meta_df = pd.read_csv("software/data/esc50/meta/esc50.csv")
        with torch.no_grad():
            # Pre-compute Text Embeddings
            for cat in tqdm(class_names, desc="Text Encoding"):
                prompt = f"A recording of the sound of {cat}"
                t_emb = clap_model.get_text_embedding([prompt], use_tensor=True)
                cache['class_to_text_embed'][cat] = t_emb.cpu()

            # Encode and Filter Audio
            for i, f in enumerate(tqdm(files_to_process, desc="Audio Filtering")):
                audio, orig_sr = torchaudio.load(f)
                audio_mono = audio.mean(0, keepdim=True)
                
                # Encode for CLAP
                audio_48k = torchaudio.transforms.Resample(orig_sr, 48000)(audio_mono)
                a_emb = clap_model.get_audio_embedding_from_data(x=audio_48k, use_tensor=True)
                
                # Check Consistency
                fname = os.path.basename(f)
                cat = meta_df.query(f"filename == '{fname}'")['category'].iloc[0]
                t_emb = cache['class_to_text_embed'][cat]
                
                sim = torch.nn.functional.cosine_similarity(a_emb.cpu(), t_emb, dim=1).item()
                cache['similarities'].append(sim)
                
                # We store everything but will filter during training
                audio_16k = torchaudio.transforms.Resample(orig_sr, SR)(audio_mono)
                cache['file_to_audio'][f] = audio_16k.cpu()
                cache['audio_embeds'].append(a_emb.cpu())
                cache['text_embeds'].append(t_emb.cpu())
                cache['processed_files'].add(f)
                
                if (i + 1) % 100 == 0:
                    torch.save(cache, embed_cache_path)
        torch.save(cache, embed_cache_path)

    # Dataset Analysis
    sims = np.array(cache['similarities'])
    print(f"\n--- Dataset Quality Report ---")
    print(f"Total Clips: {len(sims)}")
    print(f"Mean Consistency: {sims.mean():.4f}")
    for t in [0.15, 0.2, 0.25, 0.3, 0.4]:
        count = (sims > t).sum()
        print(f"Threshold {t}: {count} clips ({count/len(sims)*100:.1f}%)")

    # Filter tensors for training
    mask = sims > CONSISTENCY_THRESHOLD
    audio_embeds = torch.cat(cache['audio_embeds'], dim=0)[mask].to(DEVICE)
    text_embeds = torch.cat(cache['text_embeds'], dim=0)[mask].to(DEVICE)
    file_to_audio = cache['file_to_audio']
    print(f"Using {len(audio_embeds)} high-confidence clips for training (T={CONSISTENCY_THRESHOLD})")

    # --- Phase 1: Generative Dual-Modal Grounding ---
    phase_1_steps = 2000
    if current_phase == 1:
        print(f"\n--- Phase 1: Generative Grounding ({phase_1_steps} steps) ---")
        mapper.train()
        pbar = tqdm(range(start_step, phase_1_steps), desc="Phase 1")
        for step in pbar:
            idx = np.random.randint(0, len(audio_embeds), BATCH_SIZE)
            input_embed = text_embeds[idx]
            target_audio_embed = audio_embeds[idx]
            
            optimizer.zero_grad()
            predicted_params = mapper(input_embed)
            gen_audio = synth(predicted_params, f0=110.0)
            
            gen_audio_48k = resample_48k(gen_audio)
            gen_embed = clap_model.get_audio_embedding_from_data(x=gen_audio_48k, use_tensor=True)
            
            # Generative Minimum Loss
            loss_audio = clap_loss_fn(target_audio_embed, gen_embed)
            loss_text = clap_loss_fn(input_embed, gen_embed)
            
            # Select the path of least resistance to meaning
            loss = torch.min(loss_audio, loss_text).mean()
            
            # Add Temporal Smoothness Regularizer (Prevent digital clicks)
            # Penalize sudden changes in the audio wave
            smoothness_reg = torch.abs(gen_audio[:, 1:] - gen_audio[:, :-1]).mean()
            
            total_loss = loss + (smoothness_reg * 0.05)
            
            total_loss.backward()
            optimizer.step()
            
            wandb.log({
                "Phase": 1, 
                "Loss/Semantic_Min": loss.item(),
                "Loss/Smoothness": smoothness_reg.item(),
                "Loss/Total": total_loss.item()
            })
            
            if step % 200 == 0:
                log_audio_to_wandb(torch.zeros(1, int(SR*DURATION)), gen_audio[0:1], step, 1)

            if (step + 1) % 100 == 0:
                save_checkpoint(mapper, optimizer, 1, step + 1, CHECKPOINT_PATH)
        
        save_checkpoint(mapper, optimizer, 2, 0, CHECKPOINT_PATH)
        torch.save(mapper.state_dict(), "software/mapper_grounded.pth")
        current_phase = 2
        start_step = 0

    # --- Phase 2: Class Interpolation ---
    # Phase 2 is actually simpler now: we interpolate between AUDIO embeds of the same class,
    # and compare the result to the TEXT embed of that class (and the interpolated target).
    phase_2_steps = 2000
    if current_phase == 2:
        print(f"\n--- Phase 2: Leaf Interpolation ({phase_2_steps} steps) ---")
        mapper.train()
        pbar = tqdm(range(start_step, phase_2_steps), desc="Phase 2")
        
        # Group indices by class for fast lookup
        meta_df = pd.read_csv("software/data/esc50/meta/esc50.csv")
        class_to_indices = meta_df.groupby('category').indices
        
        for step in pbar:
            cat = np.random.choice(class_names)
            indices = class_to_indices[cat]
            idx1, idx2 = np.random.choice(indices, 2, replace=False)
            
            emb_a1 = audio_embeds[idx1].unsqueeze(0).repeat(BATCH_SIZE, 1)
            emb_a2 = audio_embeds[idx2].unsqueeze(0).repeat(BATCH_SIZE, 1)
            emb_text = cache['class_to_text_embed'][cat].to(DEVICE).repeat(BATCH_SIZE, 1)
            
            alpha = torch.rand(BATCH_SIZE, 1).to(DEVICE)
            target_audio_embed = slerp(alpha, emb_a1, emb_a2)
            
            optimizer.zero_grad()
            # Input is still the TEXT embedding
            predicted_params = mapper(emb_text) 
            gen_audio = synth(predicted_params, f0=110.0)
            
            gen_audio_48k = resample_48k(gen_audio)
            gen_embed = clap_model.get_audio_embedding_from_data(x=gen_audio_48k, use_tensor=True)
            
            loss_audio = clap_loss_fn(target_audio_embed, gen_embed)
            loss_text = clap_loss_fn(emb_text, gen_embed)
            
            loss = (loss_audio * 0.5) + (loss_text * 0.5)
            
            loss.backward()
            optimizer.step()
            
            wandb.log({
                "Phase": 2, 
                "Loss/Total": loss.item(),
                "Loss/Audio_Interp": loss_audio.item(),
                "Loss/Text_Alignment": loss_text.item()
            })

            if (step + 1) % 100 == 0:
                save_checkpoint(mapper, optimizer, 2, step + 1, CHECKPOINT_PATH)

    torch.save(mapper.state_dict(), "software/mapper_final.pth")
    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    train()
