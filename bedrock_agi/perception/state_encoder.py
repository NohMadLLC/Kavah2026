"""
bedrock_agi/perception/state_encoder.py

State Encoder: Structured state -> H^n
For environments with explicit numeric state (robotics, games, sensors).
"""

import torch
import torch.nn as nn

class StateEncoder(nn.Module):
    """
    Structured state -> H^n encoder.
    
    Handles numeric state vectors (sensor readings, game states, etc).
    Uses an MLP with LayerNorm for stable projection.
    """
    
    def __init__(self, state_dim, latent_dim=16):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # MLP Adapter: Euclidean Input -> Tangent Space T_0 H^n
        # LayerNorm is crucial here to normalize diverse sensor scales
        self.adapter = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Encode states to H^n.
        
        Args:
            states: (B, state_dim)
            
        Returns:
            Latents in H^n (B, latent_dim)
        """
        from ..core.hbl_geometry import PoincareMath
        
        # 1. Adapt to tangent space
        z = self.adapter(states)
        
        # 2. Map to Poincaré ball
        b = PoincareMath.exp_map_0(z)
        
        return b

if __name__ == "__main__":
    print("Testing StateEncoder...")
    
    # Initialize
    encoder = StateEncoder(state_dim=32, latent_dim=16)
    
    # Test with random states (Batch=4, Dim=32)
    states = torch.randn(4, 32)
    
    # Forward pass
    b = encoder(states)
    
    # 1. Shape Check
    assert b.shape == (4, 16), f"Shape mismatch: {b.shape}"
    print(f"✓ Shape correct: {b.shape}")
    
    # 2. Manifold Check
    norms = torch.norm(b, dim=-1)
    max_norm = norms.max().item()
    assert max_norm < 1.0, f"Manifold violation: {max_norm} >= 1.0"
    print(f"✓ On manifold: max norm = {max_norm:.4f}")
    
    print("✓ StateEncoder operational")