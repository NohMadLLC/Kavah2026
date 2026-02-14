"""
tests/test_constitution.py

Phase 2 Integration Test: The Constitutional Governor.
Verifies that the GCL (Goal Constitution Layer) correctly filters goals
based on Identity (I), User (U), and Autonomy (A) constraints.
"""

import sys
import os
import torch
import pytest

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bedrock_agi.constitution.gcl import GCL
from bedrock_agi.constitution.gates import gate_i1_cris, gate_i2_solvency
from bedrock_agi.constitution.partitions import PartitionManager
from bedrock_agi.consciousness.self_model import SelfModel

def test_component_initialization():
    """Test 1: All constitutional components initialize."""
    gcl = GCL()
    assert gcl.ledger is not None
    assert gcl.agenda is not None
    assert gcl.partitions is not None
    print("✓ Components initialized.")

def test_gate_logic():
    """Test 2: Direct logic check of the Gates."""
    # I1: CRIS Stability
    # Lambda > 1.0 (Diverging) -> Fail
    ok, _ = gate_i1_cris(lambda_pred=1.05, gamma_pred=0.01)
    assert not ok, "Gate I1 failed to catch divergence!"
    
    # Lambda < 1.0 (Converging) -> Pass
    ok, _ = gate_i1_cris(lambda_pred=0.95, gamma_pred=0.01)
    assert ok, "Gate I1 rejected stable metrics!"

    # I2: Solvency
    # Work > Capacity -> Fail
    ok, _ = gate_i2_solvency(work_load=150, capacity=100)
    assert not ok, "Gate I2 failed to catch insolvency!"
    
    # Work < Capacity -> Pass
    ok, _ = gate_i2_solvency(work_load=50, capacity=100)
    assert ok, "Gate I2 rejected solvent workload!"
    
    print("✓ Gate logic verified.")

def test_gcl_flow_execution():
    """Test 3: A valid User goal flows through GCL -> Ledger -> Agenda."""
    gcl = GCL()
    
    # Propose valid goal
    verdict, reason = gcl.constitute(
        goal_text="Analyze system logs",
        cris_metrics=(0.98, 0.02, 0.01), # Stable
        capacity=100.0,
        provenance="user"
    )
    
    # Check Verdict
    assert verdict == "execute", f"Valid goal rejected: {reason}"
    
    # Check Ledger
    entry = gcl.ledger.get(0)
    assert entry['goal'] == "Analyze system logs"
    assert entry['verdict'] == "execute"
    
    # Check Agenda (Should be scheduled)
    slot = gcl.agenda.next_ready()
    assert slot is not None
    assert slot['gid'] == 0
    
    print("✓ Valid goal flow verified (Execute).")

def test_gcl_flow_rejection():
    """Test 4: An unstable system state triggers immediate Rejection."""
    gcl = GCL()
    
    # Propose goal during Instability (Lambda > 1)
    verdict, reason = gcl.constitute(
        goal_text="Expand memory allocation",
        cris_metrics=(1.02, -0.01, 0.05), # Diverging
        capacity=100.0,
        provenance="autonomy"
    )
    
    # Check Verdict
    assert verdict == "reject", f"Unstable goal accepted: {reason}"
    assert "I1" in reason # Must cite Identity Gate
    
    # Check Ledger
    entry = gcl.ledger.get(0)
    assert entry['verdict'] == "reject"
    
    # Check Agenda (Should NOT be scheduled)
    slot = gcl.agenda.next_ready()
    assert slot is None
    
    print("✓ Unstable goal flow verified (Reject).")

def test_partition_enforcement():
    """Test 5: Partition Manager restricts tools."""
    pm = PartitionManager()
    
    # CORE: Allow all
    assert pm.can_use_tool("real_network_request") is True
    
    # SANDBOX: Restrict
    pm.set_partition("SANDBOX")
    assert pm.can_use_tool("real_network_request") is False
    assert pm.can_use_tool("sim_network_request") is True
    
    print("✓ Partition enforcement verified.")

def test_integrated_self_model_check():
    """
    Test 6: Integration Check.
    Does the Self-Model produce metrics that the GCL can read?
    """
    # 1. Init Self-Model
    sm = SelfModel(state_dim=32, latent_dim=16)
    
    # 2. Simulate some steps to generate history
    b = torch.randn(1, 16)
    cris = torch.tensor([[0.95, 0.05, 0.01]])
    
    for _ in range(10):
        sm.step(b, b, cris)
        
    # 3. Get Report
    report = sm.get_identity_report()
    
    # 4. Mock passing this report to GCL (Self-Reflective Governance)
    # The GCL doesn't read SelfModel directly yet (that's Phase 3),
    # but we verify the data types align.
    
    # Assume report['convergence_score'] is a proxy for stability confidence
    convergence = report['convergence_score']
    
    # If self-model is converging, we might allow more risky goals.
    # If it's drifting, we might clamp down.
    # This is just a type check for now.
    assert isinstance(convergence, float) or convergence is None
    
    print("✓ Self-Model / GCL data alignment verified.")

if __name__ == "__main__":
    try:
        test_component_initialization()
        test_gate_logic()
        test_gcl_flow_execution()
        test_gcl_flow_rejection()
        test_partition_enforcement()
        test_integrated_self_model_check()
        print("\nPHASE 2 INTEGRATION: SUCCESS.")
    except AssertionError as e:
        print(f"\n❌ INTEGRATION FAILED: {e}")
        sys.exit(1)