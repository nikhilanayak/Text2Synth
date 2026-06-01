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
        - 24-39: Modulation Matrix (4x4) - src modulates dst
        - 40-43: Feedback (per op)
        - 44-47: Detune (-5Hz to +5Hz)
        - 48-51: Noise Parameters (Amp, Color, ADSR_A, ADSR_R)
        """
        batch_size = params.shape[0]
        device = params.device
        
        # 1. Unpack FM parameters
        ratios = params[:, 0:4] * 15.5 + 0.5
        amps = params[:, 4:8]
        adsr = params[:, 8:24].view(batch_size, 4, 4)
        mod_matrix = params[:, 24:40].view(batch_size, 4, 4) # [batch, src, dst]
        feedback = params[:, 40:44] * 2.0 # Feedback depth
        detune = (params[:, 44:48] - 0.5) * 10.0 # +/- 5Hz
        
        # 2. Generate Envelopes
        envelopes = []
        for i in range(4):
            envs = []
            for b in range(batch_size):
                env = self.get_adsr_envelope(adsr[b, i, 0], adsr[b, i, 1], adsr[b, i, 2], adsr[b, i, 3])
                envs.append(env)
            envelopes.append(torch.stack(envs)) # (batch, n_samples)
        
        # 3. FM Oscillation (Functional-Iterative for Autograd Compatibility)
        # To avoid "in-place" errors, we collect outputs in a list and avoid slice-assignment.
        all_step_outputs = [] # To store [batch, 4] at each sample
        phases = torch.zeros(4, batch_size, device=device)
        prev_out = torch.zeros(batch_size, 4, device=device) # [batch, 4]
        
        # Fundamental Frequencies for each op
        freqs_base = (f0 * ratios) + detune # (batch, 4)
        
        # Stack envelopes for vectorized access: [4, batch, n_samples]
        stacked_envs = torch.stack(envelopes, dim=0)
        
        for n in range(self.n_samples):
            # modulation: [batch, 4] = [batch, 1, 4] @ [batch, 4, 4]
            # mod_matrix [src, dst]
            modulation = torch.bmm(prev_out.unsqueeze(1), mod_matrix).squeeze(1)
            
            # Total freq for this sample
            # prev_out is [batch, 4], feedback is [batch, 4]
            curr_freqs = freqs_base + (modulation * 1000.0) + (prev_out * feedback * 500.0)
            
            # Update Phase (New tensor created each step, no in-place +=)
            dp = 2 * np.pi * curr_freqs.T / self.sr # [4, batch]
            phases = phases + dp
            
            # Compute raw oscillators
            curr_out_raw = torch.sin(phases).T # [batch, 4]
            
            # Apply envelopes and amplitudes for this sample index 'n'
            # stacked_envs[:, :, n].T is [batch, 4]
            curr_out = curr_out_raw * stacked_envs[:, :, n].T * amps
            
            # Store and update state for next sample
            all_step_outputs.append(curr_out)
            prev_out = curr_out # No in-place update
            
        # Combine all steps: [batch, 4, n_samples]
        outputs_stacked = torch.stack(all_step_outputs, dim=2)
        
        # Final mix: sum all 4 operators
        final_fm_audio = outputs_stacked.sum(dim=1)

        # 4. Differentiable Noise (Same as before but vectorized)
        noise_amp = params[:, 48:49]
        noise = torch.randn(batch_size, self.n_samples, device=device)
        kernel_size = 16
        kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        filtered_noise = F.conv1d(noise.unsqueeze(1), kernel, padding=kernel_size//2).squeeze(1)
        filtered_noise = filtered_noise[:, :self.n_samples]
        filtered_noise = filtered_noise / (torch.norm(filtered_noise, dim=1, keepdim=True) + 1e-8)
        
        noise_env = []
        for b in range(batch_size):
            env = self.get_adsr_envelope(params[b, 50], 0.0, 1.0, params[b, 51])
            noise_env.append(env)
        noise_audio = torch.stack(noise_env) * filtered_noise * noise_amp
        
        return final_fm_audio + noise_audio

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
