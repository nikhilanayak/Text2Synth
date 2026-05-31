import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        # x shape: (batch, channels, time)
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x

class TinyTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(TinyTCNBlock, self).__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out + residual # Skip connection

class TinyCausalTCN(nn.Module):
    """
    Tiny Causal TCN for Audio Source Separation.
    Designed to fit in ~300k INT8 parameters.
    Target: 16kHz mono, n_fft=512, hop=256 (257 bins).
    """
    def __init__(self, n_bins=257, n_channels=128, n_blocks=4):
        super(TinyCausalTCN, self).__init__()
        
        # 1. Encoder (Pointwise)
        self.encoder = nn.Conv1d(n_bins, n_channels, kernel_size=1)
        
        # 2. Temporal Blocks (Dilated Causal Convs)
        self.blocks = nn.ModuleList([
            TinyTCNBlock(n_channels, kernel_size=3, dilation=2**i)
            for i in range(n_blocks)
        ])
        
        # 3. Decoder (Pointwise) - Outputs 2 masks (Vocals, Instrumental)
        self.decoder = nn.Conv1d(n_channels, n_bins * 2, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, bins, time) - Magnitude Spectrogram
        
        x = F.relu(self.encoder(x))
        
        for block in self.blocks:
            x = block(x)
            
        x = self.decoder(x)
        
        # Split into two masks
        batch, _, time = x.shape
        x = x.view(batch, 2, -1, time) # (batch, 2, bins, time)
        
        vocals_mask = torch.sigmoid(x[:, 0, :, :])
        instr_mask = torch.sigmoid(x[:, 1, :, :])
        
        return vocals_mask, instr_mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = TinyCausalTCN()
    total_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    # Test with a dummy spectrogram (1 batch, 257 bins, 100 time frames)
    dummy_input = torch.randn(1, 257, 100)
    v_mask, i_mask = model(dummy_input)
    print(f"Output shapes: Vocals {v_mask.shape}, Instr {i_mask.shape}")
