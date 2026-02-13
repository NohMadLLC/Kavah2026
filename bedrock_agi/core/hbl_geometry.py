"""
bedrock_agi/core/hbl_geometry.py

The Axiomatic Geometry of the Bedrock Architecture.
Implements strict Poincaré ball operations for the Eliminative Ontology.

Bedrock Constraint Compliance:
- Metric: d_H is the only valid measure of difference.
- Domain Integrity: All operations guarantee outputs strictly inside B^n.
- Identity Axiom: d_H(x, x) must be exactly 0.
"""

import torch
import torch.nn as nn

class HyperbolicDistance(nn.Module):
    """
    The ground-truth metric for existence.
    d_H(x,y) = arccosh(1 + 2|x-y|^2 / ((1-|x|^2)(1-|y|^2)))
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Euclidean squared distance: |x - y|^2
        sq_dist = torch.sum((x - y) ** 2, dim=-1)
        
        # Conformal factors: (1 - |x|^2)
        x_norm_sq = torch.sum(x ** 2, dim=-1).clamp(max=1 - self.eps)
        y_norm_sq = torch.sum(y ** 2, dim=-1).clamp(max=1 - self.eps)
        
        # The Argument for arccosh
        delta = 2 * sq_dist / ((1 - x_norm_sq) * (1 - y_norm_sq))
        
        # FIX: Clamp to 1.0 (allow exact identity) plus microscopic buffer for float stability
        # We do NOT use self.eps here, as that would bias the distance away from 0.
        arg = (1 + delta).clamp(min=1.0 + 1e-15)
        
        return torch.acosh(arg)


class PoincareMath:
    """
    Stateless Riemannian operations.
    Enforces 'Structural Impossibility' of boundary violation.
    """
    
    @staticmethod
    def exp_map_0(v: torch.Tensor, eps=1e-5) -> torch.Tensor:
        """
        Exponential map at origin.
        Output is structurally guaranteed to be within ||x|| < 1 - 2*eps.
        """
        norm_v = v.norm(dim=-1, keepdim=True).clamp(min=eps)
        direction = v / norm_v
        
        # Safe scale: tanh(norm) * (1 - 2*eps)
        scale = torch.tanh(norm_v) * (1 - 2*eps)
        return direction * scale

    @staticmethod
    def log_map_0(x: torch.Tensor, eps=1e-5) -> torch.Tensor:
        """
        Logarithmic map at origin.
        Robust to inputs outside the ball (implicitly projects them).
        """
        true_norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
        direction = x / true_norm
        manifold_norm = true_norm.clamp(max=1 - eps)
        
        tangent_norm = torch.atanh(manifold_norm)
        return direction * tangent_norm

    @staticmethod
    def mobius_scale(x: torch.Tensor, alpha: float, eps=1e-5) -> torch.Tensor:
        """
        Scalar multiplication (Geodesic Contraction).
        Output is structurally guaranteed to be within ||x|| < 1 - 2*eps.
        """
        true_norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
        direction = x / true_norm
        manifold_norm = true_norm.clamp(max=1 - eps)
        
        dist = torch.atanh(manifold_norm)
        new_dist = dist * alpha
        
        new_norm = torch.tanh(new_dist) * (1 - 2*eps)
        return direction * new_norm

    @staticmethod
    def project_to_ball(x: torch.Tensor, radius=0.99, eps=1e-5) -> torch.Tensor:
        """
        Hard safety clamp.
        """
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = 1.0 - 2*eps
        
        target_radius = min(radius, max_norm)
        cond = norm > target_radius
        
        projected = x / (norm + 1e-12) * target_radius
        return torch.where(cond, projected, x)