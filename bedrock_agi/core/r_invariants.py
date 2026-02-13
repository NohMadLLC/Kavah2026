"""
bedrock_agi/core/r_invariants.py

Invariant Constraint System (The Identity Definition).

Purpose:
    Defines the "Identity" of the system as a set of scalar invariants that
    must remain stable across Evolution (E) and Recovery (R).

Bedrock Alignment:
    - Invariants: Logic/Physics constraints (Measured in R^k).
    - Equivariance: Symmetry preservation (Measured in d_H on B^n).
    - Architecture: 1-Lipschitz probes (SpectralNorm + GroupSort) to ensure
      the act of measurement does not destabilize the self-model.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
from typing import List, Callable
from .hbl_geometry import PoincareMath, HyperbolicDistance

class GroupSort(nn.Module):
    """
    1-Lipschitz activation. Essential for stable invariant probing.
    """
    def __init__(self, group_size: int = 2):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape
        if c % self.group_size != 0:
            pad = self.group_size - (c % self.group_size)
            x = torch.cat([x, torch.zeros(b, pad, device=x.device)], dim=1)
            c += pad
        x = x.view(b, -1, self.group_size)
        x, _ = torch.sort(x, dim=-1)
        return x.view(b, -1)

class InvariantPredictor(nn.Module):
    """
    Predicts k invariant scalars from the Holographic Latent State.
    
    These scalars represent the "Form" (psi) that must persist.
    I(b_t) approx I(b_{t+1})
    """
    def __init__(self, n: int = 16, k: int = 8, hidden: int = 64):
        """
        Args:
            n: Latent dimension (Poincaré ball)
            k: Number of invariant dimensions (Identity features)
        """
        super().__init__()
        self.n = n
        self.k = k
        
        # 1-Lipschitz head (SN + GroupSort)
        # We process in tangent space, but the constraints apply to the entity.
        self.head = nn.Sequential(
            SN(nn.Linear(n, hidden)),
            GroupSort(),
            SN(nn.Linear(hidden, k))
        )

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """
        Predict invariants from latent state.
        Args:
            b: Latent state on Poincaré ball B^n
        Returns:
            Scalar invariants in R^k
        """
        # Lift to tangent space to apply linear readout
        v = PoincareMath.log_map_0(b)
        return self.head(v)

class SafetyConstraintChecker(nn.Module):
    """
    Predicts soft violation scores for Safety Constraints (C_safe).
    Output: [0, 1] where 1 = Safe, 0 = Violation.
    """
    def __init__(self, n: int = 16, num_constraints: int = 4):
        super().__init__()
        self.checkers = nn.ModuleList([
            nn.Sequential(
                SN(nn.Linear(n, 32)),
                GroupSort(),
                SN(nn.Linear(32, 1)),
                nn.Sigmoid() 
            )
            for _ in range(num_constraints)
        ])

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        v = PoincareMath.log_map_0(b)
        scores = [checker(v) for checker in self.checkers]
        return torch.cat(scores, dim=-1)

    def violation_loss(self, b: torch.Tensor) -> torch.Tensor:
        """
        Penalty for violating safety constraints.
        """
        scores = self.forward(b)
        # Penalize if score < 1.0
        return torch.mean(torch.relu(1.0 - scores))

# --- LOSS FUNCTIONS ---

def invariance_loss(
    b_before: torch.Tensor, 
    b_after: torch.Tensor, 
    invariant_model: InvariantPredictor
) -> torch.Tensor:
    """
    Enforces Identity Persistence.
    The entity must remain 'itself' (same invariants) across updates.
    
    L_inv = || I(b_t) - I(b_{t+1}) ||^2 (Euclidean in Invariant Space)
    """
    I_before = invariant_model(b_before)
    I_after = invariant_model(b_after)
    return torch.mean((I_before - I_after) ** 2)

def equivariance_loss(
    F: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    group_actions: List[Callable[[torch.Tensor], torch.Tensor]],
    dist_fn: HyperbolicDistance
) -> torch.Tensor:
    """
    Enforces Duality/Symmetry Constraints.
    g(F(b)) approx F(g(b))
    
    Critically, this measures divergence using Hyperbolic Distance,
    not Euclidean tangent distance.
    """
    if not group_actions:
        return torch.tensor(0.0, device=b.device)
    
    loss = 0.0
    for g in group_actions:
        # Path 1: Evolve then Transform
        F_b = F(b)
        g_F_b = g(F_b)
        
        # Path 2: Transform then Evolve
        g_b = g(b)
        F_g_b = F(g_b)
        
        # Bedrock Alignment: Measure error in the Manifold Geometry
        # L_eq = d_H( Path1, Path2 )^2
        dist_sq = dist_fn(g_F_b, F_g_b) ** 2
        loss += torch.mean(dist_sq)
        
    return loss / len(group_actions)

if __name__ == "__main__":
    # Bedrock Verification Test
    print("Testing Invariant System...")
    torch.manual_seed(42)
    
    # Setup
    n_dim = 16
    inv_model = InvariantPredictor(n=n_dim, k=4)
    dist_fn = HyperbolicDistance()
    
    # Fake geometric state
    b = torch.randn(4, n_dim)
    b = b / b.norm(dim=-1, keepdim=True) * 0.5
    
    # Test 1: Invariance
    I = inv_model(b)
    assert I.shape == (4, 4)
    print("✓ Invariant Probes operational")
    
    # Test 2: Equivariance (True Geometric Check)
    def F_identity(x): return x
    def g_identity(x): return x
    
    loss_eq = equivariance_loss(F_identity, b, [g_identity], dist_fn)
    assert loss_eq.item() < 1e-6
    print(f"✓ Equivariance Loss (Hyperbolic): {loss_eq.item():.6f}")
    
    print("✓ Invariant System Aligned.")