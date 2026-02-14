"""
tests/test_planning_tools.py

Planning + Tools Integration Test.
Verifies the CEM Planner, MPC Objective, Decoders, and Tool Registry.
"""

import torch
import sys
import os
import math

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bedrock_agi.core.hbl_geometry import PoincareMath
from bedrock_agi.planning.cem_planner import CEMPlanner
from bedrock_agi.planning.mpc_objective import compute_objective
from bedrock_agi.action.decoders import SimpleDecoder
from bedrock_agi.action.tools.registry import REGISTRY
# Ensure tools are registered by importing them
import bedrock_agi.action.tools.calculator
import bedrock_agi.action.tools.code_exec
import bedrock_agi.action.tools.web_search

def test_planner():
    """Test CEM planner."""
    print("\nTesting CEM Planner...")
    
    # Mock world model: simple linear dynamics in tangent space
    def world_model(b, a):
        v = PoincareMath.log_map_0(b)
        # Action pushes state in tangent space
        # a is (4,)
        # v is (16,)
        # Use first 2 dims of a to push first 2 dims of v
        perturbation = torch.zeros_like(v)
        perturbation[..., :2] = a[..., :2] * 0.1
        
        v_new = v + perturbation
        return PoincareMath.exp_map_0(v_new)
    
    # Mock objective: get close to origin (minimize norm)
    def objective(b_rollout):
        # b_rollout is (T, 1, 16)
        final_state = b_rollout[-1]
        dist = torch.norm(final_state)
        return -dist.item() # Maximize negative distance
    
    # Plan
    planner = CEMPlanner(action_dim=4, horizon=5, n_samples=32)
    b_init = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.5) # Start away from origin
    
    actions = planner.plan(b_init, world_model, objective)
    
    assert actions.shape == (5, 4)
    print("✓ CEM planner produced valid plan")

def test_decoder():
    """Test decoder."""
    print("\nTesting Decoder...")
    
    decoder = SimpleDecoder(latent_dim=16, output_dim=128)
    b = PoincareMath.project_to_ball(torch.randn(2, 16) * 0.3)
    
    output = decoder(b)
    
    assert output.shape == (2, 128)
    print("✓ Decoder outputs correct shape")

def test_tool_registry():
    """Test tool execution."""
    print("\nTesting Tool Registry...")
    
    # Test calculator
    result = REGISTRY.execute('calculator', {'expression': 'sin(3.14159/2)'})
    assert result['ok']
    # sin(pi/2) = 1.0
    assert abs(result['value'] - 1.0) < 0.01
    print("✓ Calculator works")
    
    # Test code exec
    code = "result = [i**2 for i in range(5)]"
    result = REGISTRY.execute('code_exec', {'code': code})
    assert result['ok']
    assert result['value']['result'] == [0, 1, 4, 9, 16]
    print("✓ Code exec works")
    
    # Test web search (stub)
    result = REGISTRY.execute('web_search', {'query': 'bedrock', 'num_results': 2})
    assert result['ok']
    assert len(result['value']['results']) == 2
    print("✓ Web search stub works")

def test_end_to_end_plan_execute():
    """Test plan -> tool selection -> execution."""
    print("\nTesting end-to-end planning + execution...")
    
    # 1. Start state (Abstract for now)
    b_init = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.3)
    
    # 2. Plan (stub: simulator logic to select best tool)
    # In the real system, the policy decodes b_final -> tool_name, tool_args
    # Here we mock that selection
    selected_tool = 'calculator'
    selected_args = {'expression': '2 ** 10'}
    
    # 3. Execute
    print(f"  Selected Tool: {selected_tool}")
    result = REGISTRY.execute(selected_tool, selected_args)
    
    assert result['ok']
    assert result['value'] == 1024
    
    print(f"  Result: {result['value']}")
    print("✓ End-to-end plan -> execute works")

if __name__ == "__main__":
    try:
        test_planner()
        test_decoder()
        test_tool_registry()
        test_end_to_end_plan_execute()
        print("\nPLANNING + TOOLS: SUCCESS.")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)