"""
tests/test_hbl.py

Unit tests for Poincaré ball geometry (HBL).
Verifies the Axioms of the Bedrock Architecture.

Tests:
- Roundtrip: Exp_0(Log_0(x)) ≈ x
- Domain Integrity: All operations maintain ||x|| < 1
- Contraction: Möbius scaling strictly reduces hyperbolic distance
- Metric Axioms: Non-negativity, Symmetry, Identity
"""

import torch
import pytest
import sys
import os

# Ensure we can import the core module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bedrock_agi.core.hbl_geometry import (
    PoincareMath,
    HyperbolicDistance
)

# --- Helper Functions ---

def validate_on_ball(x: torch.Tensor, eps: float = 1e-5) -> bool:
    """Check if all points in x are strictly inside the unit ball."""
    norms = x.norm(dim=-1)
    return torch.all(norms < (1.0 - eps)).item()

def validate_roundtrip(x: torch.Tensor, atol: float = 1e-5) -> bool:
    """Check if x -> log -> exp -> x is identity."""
    v = PoincareMath.log_map_0(x)
    x_rec = PoincareMath.exp_map_0(v)
    return torch.allclose(x, x_rec, atol=atol)

# --- Tests ---

def test_roundtrip_identity():
    """
    Test 1: Roundtrip consistency.
    Identity: Exp_0(Log_0(x)) == x
    """
    torch.manual_seed(42)
    # Generate points inside the ball
    x = torch.randn(10, 16)
    x = x / x.norm(dim=-1, keepdim=True) * 0.8 # Scale to 0.8 radius
    
    assert validate_roundtrip(x, atol=1e-4), "Log/Exp roundtrip failed!"

def test_ball_constraint_preservation():
    """
    Test 2: Domain Integrity.
    Operations must never produce values outside B^n.
    """
    torch.manual_seed(42)
    
    # A. Exp Map with large vectors
    v_large = torch.randn(10, 16) * 100.0  # Huge tangent vectors
    b_large = PoincareMath.exp_map_0(v_large)
    assert validate_on_ball(b_large), "Exp_0 output escaped the ball!"
    
    # B. Möbius Scalar Mult (Contraction)
    x = torch.randn(10, 16) * 0.5
    x_scaled = PoincareMath.mobius_scale(x, 0.5)
    assert validate_on_ball(x_scaled), "Möbius scaling escaped the ball!"
    
    # C. Project to Ball (Safety Clamp)
    x_out = torch.randn(10, 16) * 5.0 # Outside
    x_safe = PoincareMath.project_to_ball(x_out)
    assert validate_on_ball(x_safe), "Projection failed to clamp!"

def test_mobius_contraction_property():
    """
    Test 3: Geometric Contraction.
    Möbius scaling must strictly reduce hyperbolic distance to origin.
    d_H(0, beta * x) = beta * d_H(0, x)
    """
    dist_fn = HyperbolicDistance()
    x = torch.randn(5, 16)
    x = x / x.norm(dim=-1, keepdim=True) * 0.9 # Near boundary
    
    beta = 0.5
    x_contracted = PoincareMath.mobius_scale(x, beta)
    
    origin = torch.zeros_like(x)
    d_original = dist_fn(origin, x)
    d_contracted = dist_fn(origin, x_contracted)
    
    # Check ratio
    ratio = d_contracted / d_original
    # Allow small float error
    assert torch.allclose(ratio, torch.tensor(beta), atol=1e-4), \
        f"Möbius scale did not contract distance linearly! Ratio: {ratio.mean().item()}"

def test_distance_metric_axioms():
    """
    Test 4: Metric Axioms.
    d_H(x, y) must satisfy:
    1. d(x, y) >= 0
    2. d(x, y) == d(y, x)
    3. d(x, x) == 0
    """
    dist_fn = HyperbolicDistance()
    x = torch.randn(5, 16) * 0.5
    y = torch.randn(5, 16) * 0.5
    
    # 1. Non-negative
    d_xy = dist_fn(x, y)
    assert torch.all(d_xy >= 0), "Negative distance detected!"
    
    # 2. Symmetric
    d_yx = dist_fn(y, x)
    assert torch.allclose(d_xy, d_yx, atol=1e-5), "Distance metric is asymmetric!"
    
    # 3. Identity
    d_xx = dist_fn(x, x)
    # Note: float32 precision might result in 1e-7, not exactly 0
    assert torch.all(d_xx < 1e-4), "Self-distance is non-zero!"

def test_numerical_stability_edge_cases():
    """
    Test 5: Numerical Stability.
    Behavior near boundary (norm -> 1) and origin (norm -> 0).
    """
    # Case A: Boundary
    x_bound = torch.randn(5, 16)
    x_bound = x_bound / x_bound.norm(dim=-1, keepdim=True) * 0.9999
    v_bound = PoincareMath.log_map_0(x_bound)
    
    assert not torch.any(torch.isnan(v_bound)), "NaN at boundary in Log Map!"
    assert not torch.any(torch.isinf(v_bound)), "Inf at boundary in Log Map!"
    
    # Case B: Origin
    x_zero = torch.zeros(5, 16)
    v_zero = PoincareMath.log_map_0(x_zero)
    assert torch.allclose(v_zero, torch.zeros_like(v_zero)), "Origin Log Map failed!"
    
    # Case C: Distance at identity
    dist_fn = HyperbolicDistance()
    d_zero = dist_fn(x_zero, x_zero)
    assert not torch.any(torch.isnan(d_zero)), "NaN in distance at zero!"

if __name__ == "__main__":
    print("Running HBL Geometric Unit Tests...")
    # Run tests manually if not using pytest CLI
    try:
        test_roundtrip_identity()
        print("✓ Roundtrip Identity")
        
        test_ball_constraint_preservation()
        print("✓ Domain Integrity")
        
        test_mobius_contraction_property()
        print("✓ Möbius Contraction")
        
        test_distance_metric_axioms()
        print("✓ Metric Axioms")
        
        test_numerical_stability_edge_cases()
        print("✓ Numerical Stability")
        
        print("\nALL SYSTEMS NOMINAL.")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ RUNTIME ERROR: {e}")
        sys.exit(1)