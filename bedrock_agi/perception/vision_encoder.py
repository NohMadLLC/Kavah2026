"""
bedrock_agi/perception/vision_encoder.py

Vision Encoder: Images -> H^n
Standard CNN baseline for mapping visual data to the Poincaré Ball.
Designed for modular replacement with ViT/CLIP.
"""

import torch
import torch.nn as nn
from typing import Tuple

class VisionEncoder(nn.Module):
    """
    Image -> H^n encoder.
    
    Architecture:
    1. Standard CNN Backbone (Feature Extraction)
    2. Linear Adapter (Projection to Tangent Space)
    3. Exponential Map (Projection to Manifold)
    """
    
    def __init__(self, latent_dim: int = 16, img_size: int = 224):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Robust CNN Backbone
        # Input: (B, 3, 224, 224)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Global Pooling -> (B, 128, 1, 1)
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Adapter to Tangent Space T_0 H^n
        self.adapter = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to H^n.
        
        Args:
            images: (B, 3, H, W) normalized float tensors
            
        Returns:
            Latents in H^n (B, latent_dim)
        """
        from ..core.hbl_geometry import PoincareMath
        
        # 1. Extract Visual Features (Euclidean)
        features = self.cnn(images)
        
        # 2. Project to Tangent Space
        z = self.adapter(features)
        
        # 3. Map to Poincaré Ball
        b = PoincareMath.exp_map_0(z)
        
        return b

if __name__ == "__main__":
    print("Testing VisionEncoder...")
    
    # Initialize
    encoder = VisionEncoder(latent_dim=16)
    
    # Test with random images (Batch=4, Channels=3, H=224, W=224)
    imgs = torch.randn(4, 3, 224, 224)
    
    # Forward pass
    b = encoder(imgs)
    
    # 1. Shape Check
    assert b.shape == (4, 16), f"Shape mismatch: {b.shape}"
    print(f"✓ Shape correct: {b.shape}")
    
    # 2. Manifold Check
    norms = torch.norm(b, dim=-1)
    max_norm = norms.max().item()
    assert max_norm < 1.0, f"Manifold violation: {max_norm} >= 1.0"
    print(f"✓ On manifold: max norm = {max_norm:.4f}")
    
    print("✓ VisionEncoder operational")