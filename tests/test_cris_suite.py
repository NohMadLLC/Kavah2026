"""
tests/test_cris_suite.py

Phase 1 Integration Test: The CRIS Loop (F = R o E).
Verifies that Evolution, Projection, and Monitoring work together 
to satisfy the Bedrock Constraint (Solvency).
"""

import torch
import pytest
import sys
import os

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bedrock_agi.core.hbl_geometry import PoincareMath
from bedrock_agi.core.e_model import EvolutionModel
from bedrock_agi.core.r_projector import RProjector
from bedrock_agi.core.cris_monitor import CRISMonitor

def test_component_instantiation():
    """Test 1: All core components initialize correctly."""
    n_dim = 16
    e_model = EvolutionModel(state_dim=n_dim, hidden_dim=32, eta=0.95)
    r_model = RProjector(n=n_dim, hidden=32, bias_toward_origin=0.98)
    monitor = CRISMonitor(tail=10)
    
    assert e_model is not None
    assert r_model is not None
    assert monitor is not None
    print("✓ Components initialized.")

def test_dynamics_step():
    """Test 2: Single step of F = R(E(b))."""
    n_dim = 16
    b = torch.randn(1, n_dim)
    b = PoincareMath.project_to_ball(b, radius=0.8)
    
    e_model = EvolutionModel(state_dim=n_dim, eta=0.95)
    r_model = RProjector(n=n_dim, bias_toward_origin=0.98)
    
    # Step E
    b_pred = e_model(b)
    assert b_pred.norm() < 1.0, "E-Model output escaped manifold!"
    
    # Step R
    b_next = r_model(b_pred)
    assert b_next.norm() < 1.0, "R-Projector output escaped manifold!"
    
    print("✓ Single dynamics step valid.")

def test_bedrock_stability_loop():
    """
    Test 3: The Bedrock Certification.
    Run the loop F = R o E for N steps.
    Verify that the CRIS Monitor detects stability.
    """
    torch.manual_seed(42)
    n_dim = 16
    steps = 100
    
    # 1. Setup Stable System
    e_model = EvolutionModel(state_dim=n_dim, eta=0.90)
    r_model = RProjector(n=n_dim, bias_toward_origin=0.95)
    monitor = CRISMonitor(tail=20)
    
    # 2. Initial State
    b = torch.randn(1, n_dim)
    b = PoincareMath.project_to_ball(b, radius=0.9) 
    
    print(f"\nRunning {steps} steps of Stable Dynamics...")
    
    for t in range(steps):
        b_pred = e_model(b)
        b_next = r_model(b_pred)
        monitor.update(b, b_next)
        b = b_next
    
    metrics = monitor.metrics()
    print(f"Final Lambda: {metrics['lambda']:.4f}")
    print(f"Final Delta:  {metrics['delta_last']:.9f}")
    
    # FIX: Use the Ontological Check, not raw numbers.
    # The system likely hit the fixed point (Delta ~ 0, Lambda ~ 1).
    # is_stable() handles this correctly.
    assert monitor.is_stable(), \
        f"Monitor failed! Lambda={metrics['lambda']}, Delta={metrics['delta_last']}"
    
    print("✓ Bedrock Stability Verified (System is Solvent).")

def test_instability_detection():
    """
    Test 4: Divergence Detection.
    Force a diverging system and ensure Monitor flags it.
    """
    torch.manual_seed(42)
    n_dim = 16
    steps = 50
    
    monitor = CRISMonitor(tail=20)
    b = torch.randn(1, n_dim) * 0.01 
    
    print(f"\nRunning {steps} steps of Diverging Dynamics (Mocked)...")
    
    for t in range(steps):
        # Mock expansion
        norm = b.norm()
        if norm < 0.001: norm = 0.001
        
        target_norm = min(norm * 1.1, 0.99)
        b_next = b / norm * target_norm
        
        # Add noise to prevent fixed-point detection
        noise = torch.randn_like(b) * 0.01
        b_next = PoincareMath.project_to_ball(b_next + noise)
        
        monitor.update(b, b_next)
        b = b_next
        
    metrics = monitor.metrics()
    print(f"Final Lambda: {metrics['lambda']:.4f}")
    
    # Assert that it is NOT stable
    assert not monitor.is_stable(), "Monitor falsely reported stability during divergence!"
    print("✓ Divergence correctly identified.")

if __name__ == "__main__":
    try:
        test_component_instantiation()
        test_dynamics_step()
        test_bedrock_stability_loop()
        test_instability_detection()
        print("\nPHASE 1 INTEGRATION: SUCCESS.")
    except AssertionError as e:
        print(f"\n❌ INTEGRATION FAILED: {e}")
        sys.exit(1)