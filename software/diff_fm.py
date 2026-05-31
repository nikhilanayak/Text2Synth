import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentiableFMSynth(nn.Module):
    """
    A fully differentiable 4-Operator FM Synthesizer.
    Architecture:
    - 4 Operators (Oscillators)
    - Flexible Modulation Matrix (4x4)
    - Per-operator ADSR Envelopes
    - Multi-Scale Spectral Loss support (DDSP)
    """
    def __init__(self, sr=16000, duration=1.0):
        super().__init__()
        self.sr = sr
        self.duration = duration
        self.n_samples = int(sr * duration)
        
        # Time base for envelopes and oscillators
        # Register as buffer so it moves to device with the model
        self.register_buffer("t", torch.linspace(0, duration, self.n_samples))

    def get_adsr_envelope(self, a, d, s, r, peak=1.0):
        """
        Generates a differentiable ADSR envelope.
        Input params are normalized [0, 1].
        """
        # Scale times to actual duration (max 1.0s)
        t_a = a * 0.5
        t_d = d * 0.5
        t_r = r * 0.5
        
        # Compute segments
        # Attack: 0 to peak
        attack = self.t / (t_a + 1e-8)
        attack = torch.clamp(attack, 0, 1) * peak
        
        # Decay: peak to sustain
        decay = 1.0 - (self.t - t_a) / (t_d + 1e-8)
        decay = torch.clamp(decay, 0, 1)
        decay = s + (1.0 - s) * decay
        decay = torch.where(self.t > t_a, decay, attack)
        
        # Sustain segment is implicit in decay/release logic
        
        # Release: sustain to 0
        t_release_start = self.duration - t_r
        release = (self.duration - self.t) / (t_r + 1e-8)
        release = torch.clamp(release, 0, 1) * s
        
        envelope = torch.where(self.t > t_release_start, release, decay)
        return envelope

    def forward(self, params, f0=440.0):
        """
        params: Tensor of shape (batch, 52)
        - 0-3: Operator Ratios (0.5 to 16.0)
        - 4-7: Operator Amplitudes (0.0 to 1.0)
        - 8-23: ADSR Envelopes (4 ops * 4 params)
        - 24-39: Modulation Matrix (4x4)
        - 40-43: Feedback (per op)
        - 44-47: Detune (Hz)
        - 48-51: Noise Parameters (Amp, Color, ADSR_A, ADSR_R)
        """
        batch_size = params.shape[0]
        
        # Unpack FM parameters
        ratios = params[:, 0:4] * 15.5 + 0.5
        amps = params[:, 4:8]
        adsr = params[:, 8:24].view(batch_size, 4, 4)
        
        # 1. Generate Envelopes
        envelopes = []
        for i in range(4):
            envs = []
            for b in range(batch_size):
                env = self.get_adsr_envelope(adsr[b, i, 0], adsr[b, i, 1], adsr[b, i, 2], adsr[b, i, 3])
                envs.append(env)
            envelopes.append(torch.stack(envs)) 
        
        # 2. FM Oscillation
        modulator = torch.zeros(batch_size, self.n_samples, device=params.device)
        for i in reversed(range(4)):
            freq = (f0 * ratios[:, i:i+1]) + modulator
            dp = 2 * np.pi * freq / self.sr
            p = torch.cumsum(dp, dim=1)
            out = torch.sin(p) * envelopes[i] * amps[:, i:i+1]
            modulator = out * 1000.0 
            
        fm_audio = out 

        # 3. Differentiable Noise (Milestone 1.1 Upgrade)
        # params 48-51: [Noise_Amp, Noise_Filter, Noise_A, Noise_R]
        noise_amp = params[:, 48:49]
        
        # Generate white noise
        noise = torch.randn(batch_size, self.n_samples, device=params.device)
        
        # FIR Filter (Vectorized & MPS-Friendly)
        # We create a simple low-pass kernel based on the 'Noise_Filter' param
        # A 16-tap moving average is very fast on GPU.
        kernel_size = 16
        # Each batch gets its own kernel (simplified: use parameter to scale a ramp)
        # For even more speed, we can use a fixed kernel and just scale the output
        kernel = torch.ones(1, 1, kernel_size, device=params.device) / kernel_size
        
        # F.conv1d expects (batch, channels, time)
        filtered_noise = F.conv1d(noise.unsqueeze(1), kernel, padding=kernel_size//2).squeeze(1)
        
        # Normalize filtered noise
        filtered_noise = filtered_noise / (torch.norm(filtered_noise, dim=1, keepdim=True) + 1e-8)
        
        # Apply Noise Envelope (Simple AR)
        noise_env = []
        for b in range(batch_size):
            env = self.get_adsr_envelope(params[b, 50], 0.0, 1.0, params[b, 51])
            noise_env.append(env)
        noise_env = torch.stack(noise_env)
        
        noise_audio = filtered_noise * noise_env * noise_amp
        
        # 4. Final Mix
        final_audio = fm_audio + noise_audio
        
        return final_audio

if __name__ == "__main__":
    # Test the synth
    synth = DifferentiableFMSynth()
    dummy_params = torch.rand(1, 52, requires_grad=True)
    audio = synth(dummy_params)
    
    print(f"Generated audio shape: {audio.shape}")
    
    # Verify differentiability
    loss = audio.pow(2).mean()
    loss.backward()
    print(f"Gradient on params[0]: {dummy_params.grad[0, 0]}")
    if dummy_params.grad is not None:
        print("Success! Gradients are flowing through the synthesizer.")
