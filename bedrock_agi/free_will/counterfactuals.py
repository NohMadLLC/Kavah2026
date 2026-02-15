"""
bedrock_agi/free_will/counterfactuals.py

Counterfactual Simulator: Explore alternative outcomes.
Evaluates "what if" scenarios under same initial conditions to prove agency.
"""

import torch
import copy
from typing import List, Callable, Dict, Any, Tuple

class CounterfactualSimulator:
    """
    Simulate alternative action sequences from the same initial state.
    
    Used for:
    1. Decision Provenance: "I chose X because Y would have led to instability."
    2. Determinism Verification: Proving that Input + State = Fixed Output.
    """
    
    def __init__(self, world_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Args:
            world_model: Function F(b, a) -> b_next
                         Must be deterministic (or seeded).
        """
        self.world_model = world_model
        
    def simulate_scenarios(
        self,
        initial_state: torch.Tensor,
        candidate_actions: List[List[torch.Tensor]],
        horizon: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Simulate multiple different action sequences from the exact same starting state.
        
        Args:
            initial_state: Latent state b_0 (1, n)
            candidate_actions: List of sequences, where each sequence is a list of action vectors
            horizon: Max steps to simulate
            
        Returns:
            List of outcome dicts:
            {
                'actions': ...,
                'final_state': ...,
                'trajectory': [...],
                'stability_score': float (negative norm)
            }
        """
        results = []
        
        for i, action_seq in enumerate(candidate_actions):
            # Deep copy state to ensure isolation between counterfactuals
            current_b = initial_state.clone()
            trajectory = [current_b]
            
            # Rollout
            with torch.no_grad():
                for t, action in enumerate(action_seq):
                    if t >= horizon:
                        break
                        
                    # Apply dynamics: b_{t+1} = F(b_t, a_t)
                    next_b = self.world_model(current_b, action)
                    
                    trajectory.append(next_b)
                    current_b = next_b
            
            # Evaluate Outcome
            # Metric: Distance from origin (Stability)
            # In a real system, this would use the full Utility Function
            final_norm = torch.norm(current_b).item()
            score = -final_norm # Closer to origin is better (higher score)
            
            results.append({
                'id': i,
                'final_state': current_b,
                'trajectory': trajectory,
                'score': score,
                'steps': len(trajectory) - 1
            })
            
        return results

    def verify_determinism(
        self,
        initial_state: torch.Tensor,
        action_sequence: List[torch.Tensor],
        n_trials: int = 5
    ) -> bool:
        """
        Verify that the same input state + same action sequence ALWAYS produces 
        the exact same final state.
        
        Essential for 'Free Will' as distinct from 'Randomness'.
        
        Returns:
            True if all trials are identical within tolerance.
        """
        outcomes = []
        
        for _ in range(n_trials):
            # Run simulation
            res = self.simulate_scenarios(initial_state, [action_sequence], horizon=len(action_sequence))
            outcomes.append(res[0]['final_state'])
            
        # Check consistency
        base_outcome = outcomes[0]
        for i in range(1, n_trials):
            if not torch.allclose(base_outcome, outcomes[i], atol=1e-6):
                return False
                
        return True

if __name__ == "__main__":
    print("Testing Counterfactual Simulator...")
    
    from ..core.hbl_geometry import PoincareMath
    
    # 1. Mock World Model
    # Simple dynamics: push in direction of action
    def mock_dynamics(b, a):
        v = PoincareMath.log_map_0(b)
        # Apply action to tangent vector (first 2 dims)
        perturb = torch.zeros_like(v)
        if a.shape[0] > 0:
            perturb[..., :2] = a[..., :2] * 0.1
        
        v_next = v + perturb
        return PoincareMath.exp_map_0(v_next)
        
    sim = CounterfactualSimulator(mock_dynamics)
    
    # 2. Setup Scenario
    b_init = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.5)
    
    # Scenario A: Do nothing (Action = 0)
    actions_A = [torch.zeros(4) for _ in range(3)]
    
    # Scenario B: Move away (Action = 1)
    actions_B = [torch.ones(4) for _ in range(3)]
    
    # 3. Run Simulation
    results = sim.simulate_scenarios(b_init, [actions_A, actions_B], horizon=3)
    
    assert len(results) == 2
    score_A = results[0]['score']
    score_B = results[1]['score']
    
    print(f"  Scenario A (Rest) Score: {score_A:.4f}")
    print(f"  Scenario B (Move) Score: {score_B:.4f}")
    
    # Expect B to be further from origin (lower score) if it pushes away
    # Note: depends on random init direction, but generally adding 1s moves away
    
    # 4. Verify Determinism
    is_det = sim.verify_determinism(b_init, actions_B, n_trials=10)
    assert is_det
    print("✓ Determinism Verified (10 trials)")
    
    print("✓ Counterfactual Simulator operational")