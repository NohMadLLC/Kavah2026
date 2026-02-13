"""
bedrock_agi/core/e_model.py

The Evolution (E) Operator.
Propagates the Holographic Boundary Latent (HBL) state forward in time.

Bedrock Constraint Compliance:
1. Strict Contraction Mechanism: Uses Möbius scalar multiplication (Geodesic contraction).
2. Tangent Dynamics: Uses Convex Residual Blocks to ensure 1-Lipschitz behavior in Euclidean space.
3. Global Guarantee:
   - While the tangent dynamics are non-expansive (Lip <= 1), the composition of Log/Exp maps
     can theoretically expand hyperbolic distances near the boundary.
   - Therefore, we do NOT claim an analytic global contraction proof.
   - Instead, we rely on:
     a) The strong inductive bias of Möbius scaling (eta < 1).
     b) A hyperbolic contraction regularizer during training.
     c) Runtime verification by the CRIS Monitor (lambda < 1).
   This satisfies the Bedrock requirement for falsifiable persistence.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from .hbl_geometry import PoincareMath

class GroupSort(nn.Module):
    """
    1-Lipschitz activation function. 
    Preserves gradient magnitude better than ReLU and is strictly non-expansive.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape
        # Ensure even channels for grouping
        if c % 2 != 0:
            # Pad with zero if odd to maintain dimension for sorting
            x = torch.cat([x, torch.zeros(b, 1, device=x.device)], dim=1)
            c += 1
        
        # Reshape to pairs
        x_reshaped = x.view(b, c // 2, 2)
        # Sort along the last dimension (min, max)
        x_sorted, _ = torch.sort(x_reshaped, dim=2)
        return x_sorted.view(b, c)

class ConvexResidualBlock(nn.Module):
    """
    A provably 1-Lipschitz residual block.
    Standard ResNet: y = x + f(x) -> Lip <= 1 + Lip(f) (Expansive)
    Convex ResNet:   y = (1 - alpha) * x + alpha * f(x) -> Lip <= 1 (Non-Expansive)
    """
    def __init__(self, dim, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        # The function f(x) must be 1-Lipschitz
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(dim, dim)),
            GroupSort(),
            spectral_norm(nn.Linear(dim, dim)), 
        )

    def forward(self, x):
        update = self.net(x)
        # Convex combination guarantees non-expansiveness
        return (1 - self.alpha) * x + self.alpha * update

class EvolutionModel(nn.Module):
    """
    The E-Model: Evolution Operator.
    
    Formal Definition:
    b_{t+1} = MöbiusScale( Exp_0( Update( Log_0(b_t), u_t ) ), eta )
    """
    def __init__(self, state_dim, context_dim=0, hidden_dim=256, num_layers=4, eta=0.98):
        super().__init__()
        self.state_dim = state_dim
        self.eta = eta # Target Global Contraction Factor
        
        # 1. Context Injection 
        if context_dim > 0:
            self.context_adapter = spectral_norm(nn.Linear(context_dim, state_dim))
            # Learnable scale for context injection, initialized small
            self.context_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.context_adapter = None
            self.context_scale = None

        # 2. Tangent Dynamics (Provably 1-Lipschitz in Euclidean metric)
        layers = []
        # Input projection
        layers.append(spectral_norm(nn.Linear(state_dim, hidden_dim)))
        layers.append(GroupSort())
        
        # Deep dynamics
        for _ in range(num_layers):
            layers.append(ConvexResidualBlock(hidden_dim, alpha=0.1))
        
        # Output projection
        layers.append(spectral_norm(nn.Linear(hidden_dim, state_dim)))
        
        self.tangent_dynamics = nn.Sequential(*layers)

    def forward(self, b_state: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            b_state: Current HBL state in B^n
            context: Optional control/sensory vector
        """
        # A. Lift to Tangent Space
        v_state = PoincareMath.log_map_0(b_state)
        
        # B. Inject Context (Bounded Impulse)
        if self.context_adapter is not None and context is not None:
            v_context = self.context_adapter(context)
            # Additive context is controlled by learnable scale
            v_input = v_state + self.context_scale * v_context
        else:
            v_input = v_state
            
        # C. Apply Non-Expansive Dynamics (Euclidean sense)
        v_next_tangent = self.tangent_dynamics(v_input)
        
        # D. Retract to Manifold
        b_intermediate = PoincareMath.exp_map_0(v_next_tangent)
        
        # E. Enforce Bedrock Contraction (Möbius Scaling)
        # This is the primary mechanism for thermodynamic cooling.
        # It geometrically shrinks the state towards the origin (identity).
        b_next = PoincareMath.mobius_scale(b_intermediate, self.eta)
        
        return b_next