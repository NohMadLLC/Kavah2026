"""
bedrock_agi/action/decoders.py

Decoders: H^n -> output space
Map latent states back to usable outputs (text embeddings, action vectors, etc).
"""

import torch
import torch.nn as nn

class SimpleDecoder(nn.Module):
    """
    H^n -> output space decoder.
    
    Generic MLP decoder for mapping from the Poincare Ball to any Euclidean output.
    Used for:
    - Reconstructing observations (Autoencoder)
    - Generating action parameters
    - predicting next embeddings
    """
    
    def __init__(self, latent_dim=16, output_dim=768):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # MLP Decoder
        # Input: Tangent vector (Euclidean)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to output space.
        
        Args:
            b: Latent in H^n (B, latent_dim)
            
        Returns:
            Output (B, output_dim) - Euclidean
        """
        from ..core.hbl_geometry import PoincareMath
        
        # 1. Map to Tangent Space T_0 H^n (Logarithmic Map)
        # This linearizes the hyperbolic point so the MLP can process it.
        v = PoincareMath.log_map_0(b)
        
        # 2. Decode via MLP
        return self.net(v)

if __name__ == "__main__":
    print("Testing SimpleDecoder...")
    
    from ..core.hbl_geometry import PoincareMath
    
    # Initialize
    # Example: Decoding to a text embedding size (768)
    decoder = SimpleDecoder(latent_dim=16, output_dim=768)
    
    # Create valid hyperbolic points
    b = PoincareMath.project_to_ball(torch.randn(2, 16) * 0.3)
    
    # Forward pass
    output = decoder(b)
    
    # Checks
    assert output.shape == (2, 768), f"Shape mismatch: {output.shape}"
    print(f"✓ Decoder output shape: {output.shape}")
    
    # Check gradients flow
    loss = output.sum()
    loss.backward()
    print("✓ Gradients flow confirmed")
    
    print("✓ SimpleDecoder operational")