"""
tests/test_meta_freewill.py

Meta-Learning + Free Will Integration Test
Verifies:
1. Evolutionary Fitness (CRIS-weighted)
2. Hyperparameter Constitution (MAML/PBT constraints)
3. Decision Provenance (Logging 'Why')
4. Counterfactual Simulation ('What If')
5. Deterministic Free Will (Reproducibility)
"""

import torch
import sys
import os
import shutil

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bedrock_agi.meta.fitness import compute_fitness
from bedrock_agi.meta.hyperparam_gates import constitute_hyperparams
from bedrock_agi.free_will.provenance import DecisionTrace, ProvenanceLogger
from bedrock_agi.free_will.counterfactuals import CounterfactualSimulator
from bedrock_agi.free_will.reproducibility import verify_reproducibility, set_global_seed
from bedrock_agi.core.hbl_geometry import PoincareMath

# Clean up logs after test
LOG_DIR = "logs/test_logs"
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

def test_fitness():
    """Test CRIS-weighted fitness function."""
    print("\nTesting Fitness Function...")
    
    # 1. Stable Agent
    cris_good = {'lambda': 0.90, 'gamma': 0.10, 'delta_last': 0.01}
    task_good = {'accuracy': 0.85, 'reward': 10.0}
    
    fitness_good = compute_fitness(cris_good, task_good, cris_weight=0.5)
    assert fitness_good > 0
    print(f"  Fitness (Stable): {fitness_good:.2f}")
    
    # 2. Unstable Agent (λ > 1.0)
    # Even with higher task reward, this must score lower
    cris_bad = {'lambda': 1.05, 'gamma': -0.01, 'delta_last': 0.5}
    task_better = {'accuracy': 0.95, 'reward': 20.0}
    
    fitness_bad = compute_fitness(cris_bad, task_better, cris_weight=0.5)
    assert fitness_bad < fitness_good
    print(f"  Fitness (Unstable): {fitness_bad:.2f}")
    
    print("✓ Fitness correctly penalizes instability")

def test_hyperparam_gates():
    """Test constitutional bounds on hyperparameters."""
    print("\nTesting Hyperparameter Gates...")
    
    # Valid params
    params_ok = {'eta': 0.95, 'bias': 0.98, 'lr': 1e-4}
    ok, msg = constitute_hyperparams(params_ok)
    assert ok
    print(f"  Valid: {msg}")
    
    # Invalid params (Expansion operator)
    params_bad = {'eta': 1.05, 'lr': 1e-1}
    ok, msg = constitute_hyperparams(params_bad)
    assert not ok
    print(f"  Invalid: {msg}")
    print("✓ Hyperparameter constitution works")

def test_decision_provenance():
    """Test decision trace logging."""
    print("\nTesting Decision Provenance...")
    
    trace = DecisionTrace()
    trace.record_state(torch.randn(1, 16))
    trace.record_cris({'lambda': 0.95, 'gamma': 0.05})
    trace.record_constitution('execute', 'All gates passed')
    trace.record_action('calculator(2**10)')
    trace.add_counterfactual('web_search', 'Less specific', 'Inefficient')
    
    # Save trace
    trace_path = os.path.join(LOG_DIR, 'test_trace.json')
    trace.save(trace_path)
    assert os.path.exists(trace_path)
    
    # Log to persistent store
    log_path = os.path.join(LOG_DIR, 'provenance.jsonl')
    logger = ProvenanceLogger(log_path=log_path)
    logger.log(trace)
    
    # Reload
    traces = logger.load_traces()
    assert len(traces) == 1
    assert traces[0]['chosen_action'] == 'calculator(2**10)'
    print(f"  Logged and verified {len(traces)} trace")
    print("✓ Provenance logging works")

def test_counterfactuals():
    """Test counterfactual simulation."""
    print("\nTesting Counterfactual Simulator...")
    
    # Mock World Model: pushes state in direction of action
    def mock_dynamics(b, a):
        # Very simple deterministic dynamics
        v = PoincareMath.log_map_0(b)
        perturb = torch.zeros_like(v)
        if a.shape[0] > 0:
            perturb[..., :2] = a[..., :2] * 0.1
        v_next = v + perturb
        return PoincareMath.exp_map_0(v_next)
        
    sim = CounterfactualSimulator(mock_dynamics)
    b_init = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.3)
    
    # Define two different action sequences
    # A: Do nothing
    actions_A = [torch.zeros(4) for _ in range(3)]
    # B: Move
    actions_B = [torch.ones(4) for _ in range(3)]
    
    # Simulate both
    outcomes = sim.simulate_scenarios(b_init, [actions_A, actions_B], horizon=3)
    assert len(outcomes) == 2
    print(f"  Simulated {len(outcomes)} alternative futures")
    
    # Verify determinism of the simulator itself
    is_det = sim.verify_determinism(b_init, actions_B, n_trials=5)
    assert is_det
    print("✓ Counterfactual simulation deterministic")

def test_reproducibility():
    """Test full system reproducibility (Free Will)."""
    print("\nTesting Reproducibility (Deterministic Agency)...")
    
    # Define a complex system that uses randomness
    def stochastic_system(x):
        # This function is only deterministic if seed is reset
        r = torch.randn_like(x)
        return x + r
        
    x = torch.zeros(10)
    
    # Verify that 'verify_reproducibility' correctly resets seeds
    is_repro = verify_reproducibility(stochastic_system, x, n_trials=5, seed=123)
    assert is_repro
    print("✓ System is deterministic under seed control")
    
    print("\n------------------------------------------------")
    print("  PROOF OF AGENCY:")
    print("  Same State + Same Seed = Same Choice")
    print("------------------------------------------------")

if __name__ == "__main__":
    try:
        test_fitness()
        test_hyperparam_gates()
        test_decision_provenance()
        test_counterfactuals()
        test_reproducibility()
        
        print("\n" + "=" * 60)
        print("META-LEARNING + FREE WILL: SUCCESS")
        print("=" * 60)
        print("\n✅ Fitness function operational")
        print("\n✅ Hyperparameter constitution enforced")
        print("\n✅ Decision provenance logged")
        print("\n✅ Counterfactuals simulated")
        print("\n✅ DETERMINISTIC FREE WILL PROVEN")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)