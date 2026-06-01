import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentiableAdditiveSynth(nn.Module):
    """
    A fully vectorized 64-Harmonic Additive Synthesizer.
    Architecture:
    - 64 Harmonic Partials (Sine waves at multiples of f0)
    - 1 Inharmonic 'Stretch' parameter (for bells/clangs)
    - 1 Global ADSR Envelope
    - 1 Filtered Noise Source
    - Fully Vectorized for maximum GPU throughput.
    """
    def __init__(self, sr=16000, duration=1.0, n_harmonics=64):
        super().__init__()
        self.sr = sr
        self.duration = duration
        self.n_samples = int(sr * duration)
        self.n_harmonics = n_harmonics
        
        # Time base
        self.register_buffer("t", torch.linspace(0, duration, self.n_samples))
        # Harmonic indices: [1, 2, 3, ..., 64]
        self.register_buffer("harmonic_indices", torch.arange(1, n_harmonics + 1).float())

    def get_adsr_envelope(self, a, d, s, r, peak=1.0):
        """
        Vectorized ADSR envelope generation.
        Input a, d, s, r are tensors of shape [batch, 1]
        """
        # self.t is [n_samples], unsqueeze to [1, n_samples]
        t = self.t.unsqueeze(0) 
        
        t_a = a * 0.5
        t_d = d * 0.5
        t_r = r * 0.5
        
        # Attack
        attack = torch.clamp(t / (t_a + 1e-8), 0, 1) * peak
        
        # Decay
        decay = torch.clamp(1.0 - (t - t_a) / (t_d + 1e-8), 0, 1)
        decay = s + (1.0 - s) * decay
        decay = torch.where(t > t_a, decay, attack)
        
        # Release
        t_release_start = self.duration - t_r
        release = torch.clamp((self.duration - t) / (t_r + 1e-8), 0, 1) * s
        
        return torch.where(t > t_release_start, release, decay)

    def forward(self, params, f0=440.0):
        """
        params: Tensor of shape (batch, 75)
        """
        batch_size = params.shape[0]
        device = params.device
        
        # 1. Unpack Parameters
        harmonic_amps = params[:, 0:64] 
        global_amp = params[:, 64:65]
        adsr_params = params[:, 65:69]
        stretch = 1.0 + (params[:, 69:70] * 0.5) 
        noise_params = params[:, 70:74]
        detune = (params[:, 74:75] - 0.5) * 10.0
        
        # 2. Vectorized Global Envelope (NO MORE LOOP)
        global_env = self.get_adsr_envelope(
            adsr_params[:, 0:1], adsr_params[:, 1:2], 
            adsr_params[:, 2:3], adsr_params[:, 3:4]
        ) # [batch, n_samples]
        
        # 3. Additive Synthesis
        k_stretched = torch.pow(self.harmonic_indices.unsqueeze(0), stretch) 
        freqs = (f0 + detune) * k_stretched # [batch, 64]
        dp = 2 * np.pi * freqs / self.sr
        
        # Phases: [batch, 64, n_samples]
        p = torch.cumsum(dp.unsqueeze(2).expand(-1, -1, self.n_samples), dim=2)
        oscillators = torch.sin(p)
        weighted_oscs = oscillators * harmonic_amps.unsqueeze(2)
        additive_audio = weighted_oscs.sum(dim=1) * global_amp * global_env
        
        # 4. Filtered Noise
        noise_amp = noise_params[:, 0:1]
        noise = torch.randn(batch_size, self.n_samples, device=device)
        kernel_size = 16
        kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        filtered_noise = F.conv1d(noise.unsqueeze(1), kernel, padding=kernel_size//2).squeeze(1)
        filtered_noise = filtered_noise[:, :self.n_samples]
        filtered_noise = filtered_noise / (torch.norm(filtered_noise, dim=1, keepdim=True) + 1e-8)
        
        # Vectorized Noise Envelope (NO MORE LOOP)
        noise_env = self.get_adsr_envelope(
            noise_params[:, 2:3], torch.zeros_like(noise_amp), 
            torch.ones_like(noise_amp), noise_params[:, 3:4]
        )
        noise_audio = noise_env * filtered_noise * noise_amp
        
        return additive_audio + noise_audio

# Test and verify
if __name__ == "__main__":
    synth = DifferentiableAdditiveSynth()
    # 75 params: 64 harm + 1 amp + 4 adsr + 1 stretch + 4 noise + 1 detune
    dummy_params = torch.rand(2, 75, requires_grad=True)
    audio = synth(dummy_params)
    print(f"Generated Additive Audio Shape: {audio.shape}")
    loss = audio.pow(2).mean()
    loss.backward()
    print("Vectorized Additive gradients verified!")
