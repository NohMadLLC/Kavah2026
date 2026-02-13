"""
bedrock_agi/core/r_projector.py

Recovery/Projection Engine (R): Tangent-space denoiser with hyperbolic contraction.

Design Goals:
1. Tangent-space MLP is 1-Lipschitz in Euclidean metric (Spectral Norm + GroupSort).
2. Final Möbius scaling by beta < 1 is an exact hyperbolic contraction toward the origin.
3. Global Guarantee:
   - The composition Log0 -> MLP -> Exp0 may expand hyperbolic distances near the boundary.
   - Therefore, the overall hyperbolic Lipschitz constant is NOT analytically guaranteed.
   - Global contractivity is enforced during training via a hyperbolic contraction loss
     and verified at runtime by the CRIS monitor (lambda < 1).
   
This satisfies the Bedrock requirement for falsifiable persistence.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
from .hbl_geometry import PoincareMath, HyperbolicDistance

class GroupSort(nn.Module):
    """
    1-Lipschitz nonlinearity (Anil et al., 2019).
    Sorts groups of neurons – preserves gradient norm better than ReLU.
    """
    def __init__(self, group_size: int = 2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape
        # Pad if necessary to match group size
        if c % self.group_size != 0:
            pad = self.group_size - (c % self.group_size)
            x = torch.cat([x, torch.zeros(b, pad, device=x.device)], dim=1)
            c += pad
            
        x = x.view(b, -1, self.group_size)
        x, _ = torch.sort(x, dim=-1)
        return x.view(b, -1)

class RProjector(nn.Module):
    """
    Projector with guaranteed hyperbolic contraction via Möbius scaling.
    Projects noisy states onto the invariant manifold C (learned in tangent space).
    """
    def __init__(
        self, 
        n: int = 16, 
        hidden: int = 256, 
        depth: int = 3, 
        bias_toward_origin: float = 0.98,
        group_size: int = 2
    ):
        super().__init__()
        self.n = n
        self.beta = bias_toward_origin # The hyperbolic contraction factor
        
        # Build 1-Lipschitz MLP in Tangent Space
        layers = []
        dims = [n] + [hidden] * (depth - 1) + [n]
        
        for i in range(len(dims) - 1):
            layers.append(SN(nn.Linear(dims[i], dims[i+1])))
            if i < len(dims) - 2:
                layers.append(GroupSort(group_size))
        
        self.net = nn.Sequential(*layers)

    def forward(self, b_pred: torch.Tensor) -> torch.Tensor:
        """
        Project predicted state onto invariant manifold.
        
        Args:
            b_pred: Noisy prediction from E, shape (B, n)
        Returns:
            b_clean: Cleaned state, shape (B, n)
        """
        # 1. Lift to Tangent Space (Log Map)
        v = PoincareMath.log_map_0(b_pred)
        
        # 2. Denoise in Tangent Space (Non-expansive in Euclidean metric)
        v_clean = self.net(v)
        
        # 3. Retract to Manifold (Exp Map)
        b_intermediate = PoincareMath.exp_map_0(v_clean)
        
        # 4. Enforce Stability via Möbius Scaling
        # This guarantees: d_H(0, b_clean) = beta * d_H(0, b_intermediate)
        b_clean = PoincareMath.mobius_scale(b_intermediate, self.beta)
        
        return b_clean

if __name__ == "__main__":
    # Bedrock Verification Test
    print("Testing Projection Engine (R)...")
    torch.manual_seed(42)
    
    n_dim = 16
    model = RProjector(n=n_dim, hidden=64, bias_toward_origin=0.95)
    dist_fn = HyperbolicDistance()
    
    # Test 1: Radial Contraction (Sanity Check)
    b_noisy = torch.randn(10, n_dim)
    b_noisy = b_noisy / b_noisy.norm(dim=-1, keepdim=True) * 0.8 # Radius 0.8
    
    with torch.no_grad():
        b_clean = model(b_noisy)
        
    d_noisy = dist_fn(torch.zeros_like(b_noisy), b_noisy)
    d_clean = dist_fn(torch.zeros_like(b_clean), b_clean)
    
    avg_reduction = (d_clean / d_noisy).mean().item()
    print(f"Average Radial Contraction Ratio: {avg_reduction:.4f}")
    
    # We expect reduction roughly equal to beta
    assert avg_reduction < 0.96, "R did not contract towards identity!"
    print("✓ R enforces geometric contraction.")
    
    # Test 2: Output validity
    max_norm = b_clean.norm(dim=-1).max().item()
    assert max_norm < 1.0, f"R produced invalid point with norm {max_norm}"
    print("✓ R preserves Poincaré domain constraints.")