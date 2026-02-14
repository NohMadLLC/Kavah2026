"""
bedrock_agi/perception/audio_encoder.py

Audio Encoder: Audio -> H^n
Placeholder for audio processing using Mel-spectrogram inputs.
"""

import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    """
    Audio -> H^n encoder.
    
    Expects Mel-spectrogram input (B, 1, n_mels, n_frames).
    Architecture:
    1. 2D CNN over time-frequency domain
    2. Global Average Pooling
    3. Projection to Tangent Space -> Exponential Map
    """
    
    def __init__(self, latent_dim: int = 16, n_mels: int = 80):
        super().__init__()
        self.latent_dim = latent_dim
        
        # CNN over mel-spectrogram
        # Input: (B, 1, 80, T)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Global Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Adapter to Tangent Space
        self.adapter = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Encode audio spectrograms to H^n.
        
        Args:
            spectrograms: (B, 1, n_mels, n_frames)
            
        Returns:
            Latents in H^n (B, latent_dim)
        """
        from ..core.hbl_geometry import PoincareMath
        
        # 1. CNN features
        features = self.cnn(spectrograms)
        
        # 2. Adapt to tangent space
        z = self.adapter(features)
        
        # 3. Map to Poincaré ball
        b = PoincareMath.exp_map_0(z)
        
        return b

if __name__ == "__main__":
    print("Testing AudioEncoder...")
    
    # Initialize
    encoder = AudioEncoder(latent_dim=16)
    
    # Test with random spectrograms (Batch=4, Channels=1, Mels=80, Frames=200)
    specs = torch.randn(4, 1, 80, 200)
    
    # Forward pass
    b = encoder(specs)
    
    # 1. Shape check
    assert b.shape == (4, 16), f"Shape mismatch: {b.shape}"
    print(f"✓ Shape correct: {b.shape}")
    
    # 2. Manifold check
    norms = torch.norm(b, dim=-1)
    max_norm = norms.max().item()
    assert max_norm < 1.0, f"Manifold violation: {max_norm} >= 1.0"
    print(f"✓ On manifold: max norm = {max_norm:.4f}")
    
    print("✓ AudioEncoder operational")