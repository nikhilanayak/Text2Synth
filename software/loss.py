import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-Scale Spectral Loss as described in the DDSP paper.
    Computes L1 loss on both magnitude and log-magnitude spectrograms
    at multiple FFT resolutions.
    """
    def __init__(self, fft_sizes=[2048, 1024, 512, 256, 128, 64], alpha=1.0):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.alpha = alpha

    def forward(self, target, predicted):
        """
        target: (batch, n_samples)
        predicted: (batch, n_samples)
        """
        total_loss = 0
        
        for n_fft in self.fft_sizes:
            hop_length = n_fft // 4
            win_length = n_fft
            
            # Use Hann window
            window = torch.hann_window(win_length).to(target.device)
            
            # STFT
            s_target = torch.stft(target, n_fft=n_fft, hop_length=hop_length, 
                                 win_length=win_length, window=window, 
                                 return_complex=True, center=True).abs()
            
            s_predicted = torch.stft(predicted, n_fft=n_fft, hop_length=hop_length, 
                                    win_length=win_length, window=window, 
                                    return_complex=True, center=True).abs()
            
            # Linear Magnitude Loss
            l_mag = F.l1_loss(s_target, s_predicted)
            
            # Log Magnitude Loss
            l_log_mag = F.l1_loss(torch.log(s_target + 1e-7), torch.log(s_predicted + 1e-7))
            
            total_loss += l_mag + self.alpha * l_log_mag
            
        return total_loss

class CLAPLoss(nn.Module):
    """
    Semantic Loss using CLAP embeddings.
    Measures Cosine Similarity between generated audio and target embedding.
    """
    def __init__(self):
        super().__init__()

    def forward(self, target_embedding, generated_embedding):
        """
        target_embedding: (batch, 512) - The semantic target
        generated_embedding: (batch, 512) - Embedding of the generated audio
        """
        # Cosine Similarity is between -1 and 1. We want 1.
        # Loss = 1 - similarity
        similarity = F.cosine_similarity(target_embedding, generated_embedding, dim=1)
        return (1.0 - similarity).mean()

if __name__ == "__main__":
    # Simple test
    batch = 2
    samples = 16000
    t = torch.randn(batch, samples)
    p = torch.randn(batch, samples)
    
    spec_loss = MultiScaleSpectralLoss()
    loss = spec_loss(t, p)
    print(f"Multi-Scale Spectral Loss: {loss.item()}")
    
    clap_loss = CLAPLoss()
    emb_t = torch.randn(batch, 512)
    emb_p = torch.randn(batch, 512)
    l_clap = clap_loss(emb_t, emb_p)
    print(f"CLAP Semantic Loss: {l_clap.item()}")
