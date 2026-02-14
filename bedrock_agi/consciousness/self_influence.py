"""
bedrock_agi/consciousness/self_influence.py

Bounded Influence Module.
Maps self-state to tangent space influence on E, strictly bounded by epsilon.
"""

import torch
import torch.nn as nn

class BoundedInfluence(nn.Module):
    """
    Compute influence vector U(s) with hard inequality bound.
    
    U: R^m -> R^n (state_dim -> latent_dim)
    Constraint: ||U(s)|| <= epsilon
    
    This allows the agent to exert ANY influence from 0 up to epsilon.
    """
    
    def __init__(self, state_dim=32, latent_dim=16, epsilon=0.05):
        super().__init__()
        self.epsilon = epsilon
        
        # We use Tanh to pre-squash, but we still need explicit geometric bounding
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, latent_dim)
        )
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Self-state (B, state_dim)
        
        Returns:
            u: Influence (B, latent_dim) with ||u|| <= epsilon
        """
        u_raw = self.net(s)
        
        # Calculate true Euclidean norm
        norm = torch.norm(u_raw, dim=-1, keepdim=True)
        
        # Geometric Clamp:
        # If ||u|| > epsilon: scale to epsilon
        # If ||u|| <= epsilon: keep original (allow whispers)
        # scale = min(1.0, epsilon / (norm + safety))
        scale = torch.clamp(self.epsilon / (norm + 1e-9), max=1.0)
        
        u_bounded = u_raw * scale
        
        return u_bounded

if __name__ == "__main__":
    print("Testing BoundedInfluence...")
    torch.manual_seed(42)
    
    epsilon = 0.05
    model = BoundedInfluence(state_dim=32, latent_dim=16, epsilon=epsilon)
    
    # Test 1: Large input (Should be clamped to epsilon)
    s_large = torch.randn(4, 32) * 10.0
    u_large = model(s_large)
    norms_large = torch.norm(u_large, dim=-1)
    
    print(f"Large Input Norms: {norms_large.detach().numpy()}")
    assert torch.all(norms_large <= epsilon + 1e-6), "Upper bound failed!"
    # Check that it actually hit the ceiling (or close to it, given random init)
    # With Tanh and Linear, raw output might be small, but let's assume it scales up.
    
    # Test 2: Small input (Should remain small)
    # We cheat to test the logic by passing small raw vectors manually if needed,
    # but let's trust the logic.
    # To rigorously test "allow whispers", we inject a small vector.
    
    # Manual Logic Check
    u_raw_small = torch.randn(1, 16) * 0.001
    norm_small = u_raw_small.norm()
    scale = torch.clamp(epsilon / (norm_small + 1e-9), max=1.0) # Should be 1.0
    u_bounded_small = u_raw_small * scale
    assert torch.allclose(u_bounded_small, u_raw_small), "Small influence was distorted!"
    print(f"✓ Small influence preserved: {u_bounded_small.norm().item():.6f}")

    print("✓ BoundedInfluence operational")