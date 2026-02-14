"""
bedrock_agi/planning/cem_planner.py

CEM Planner: Cross-Entropy Method in H^n
Plans action sequences by:
1. Sample candidate sequences (Gaussian in action space)
2. Roll out in latent space via World Model F(b, a) -> b'
3. Score via MPC objective (Task Reward - CRIS Penalty)
4. Keep elite samples
5. Refit distribution (Update Mean/Std)
"""

import torch
import torch.nn as nn
from typing import Callable, List, Tuple

class CEMPlanner:
    """
    Cross-Entropy Method planner in latent space.
    
    Plans sequences of actions (or latent perturbations) that:
    - Achieve task goals (maximize reward)
    - Maintain CRIS stability (minimize divergence)
    - Preserve invariants (minimize drift)
    """
    
    def __init__(
        self,
        action_dim: int = 4,
        horizon: int = 8,
        n_iterations: int = 5,
        n_samples: int = 128,
        elite_frac: float = 0.1,
        init_std: float = 0.5
    ):
        """
        Args:
            action_dim: Dimension of action space (e.g., control inputs)
            horizon: Planning horizon (steps)
            n_iterations: CEM refinement iterations
            n_samples: Samples per iteration
            elite_frac: Fraction of samples to keep as elite
            init_std: Initial action standard deviation (exploration noise)
        """
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_iterations = n_iterations
        self.n_samples = n_samples
        self.elite_frac = elite_frac
        self.n_elite = max(1, int(n_samples * elite_frac))
        self.init_std = init_std
        
    def plan(
        self,
        b_init: torch.Tensor,
        world_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        objective_fn: Callable[[torch.Tensor], float]
    ) -> torch.Tensor:
        """
        Plan action sequence.
        
        Args:
            b_init: Initial latent state (1, n)
            world_model: Forward model F: (b, a) -> b'
            objective_fn: Objective function for scoring rollouts (b_seq -> scalar)
            
        Returns:
            best_actions: (horizon, action_dim) tensor
        """
        # Initialize distribution parameters
        # Mean action starts at 0, Std starts at init_std
        mu = torch.zeros(self.horizon, self.action_dim)
        std = torch.ones(self.horizon, self.action_dim) * self.init_std
        
        best_actions = None
        best_score = -float('inf')
        
        for iteration in range(self.n_iterations):
            # 1. Sample candidate action sequences: (N, H, A)
            # We use the reparameterization trick: mu + std * eps
            noise = torch.randn(self.n_samples, self.horizon, self.action_dim)
            actions_batch = mu.unsqueeze(0) + std.unsqueeze(0) * noise
            
            scores = []
            
            # 2. Roll out each sequence
            # Note: This loop can be slow if not batched. 
            # Ideally world_model supports batching. If not, we loop.
            # Here we assume world_model takes single b, a for simplicity or batched b, a.
            
            with torch.no_grad():
                for i in range(self.n_samples):
                    b_current = b_init.clone()
                    b_rollout = [b_current]
                    
                    # Unroll horizon
                    for t in range(self.horizon):
                        action_t = actions_batch[i, t] # (A,)
                        # Apply action via World Model
                        # b_next = F(b, a)
                        b_next = world_model(b_current, action_t)
                        
                        b_rollout.append(b_next)
                        b_current = b_next
                    
                    # 3. Score rollout
                    # Stack rollout: (H+1, 1, n)
                    b_seq = torch.stack(b_rollout, dim=0)
                    score = objective_fn(b_seq)
                    scores.append(score)
            
            scores_tensor = torch.tensor(scores)
            
            # 4. Select elite samples
            # We want to maximize score
            topk = torch.topk(scores_tensor, self.n_elite)
            elite_indices = topk.indices
            elite_actions = actions_batch[elite_indices] # (n_elite, H, A)
            
            # 5. Refit distribution
            # New mean and std based on elites
            mu = elite_actions.mean(dim=0) # (H, A)
            std = elite_actions.std(dim=0) + 1e-3 # (H, A) - add epsilon for numerical stability
            
            # Track best found so far
            current_best_idx = elite_indices[0]
            current_best_score = scores_tensor[current_best_idx].item()
            
            if current_best_score > best_score:
                best_score = current_best_score
                best_actions = actions_batch[current_best_idx]
                
            # Optional: Print progress
            # print(f"  Iter {iteration}: Best Score = {best_score:.4f}")
            
        return best_actions

if __name__ == "__main__":
    print("Testing CEMPlanner...")
    
    from ..core.hbl_geometry import PoincareMath
    import torch
    
    # Mock World Model
    # Simple dynamics: b' = exp( log(b) + a * 0.1 )
    # Action directly shifts latent state in tangent space
    def mock_world_model(b, a):
        # b: (1, 16), a: (4,)
        # Map to tangent
        v = PoincareMath.log_map_0(b)
        
        # Perturb tangent vector
        # Use first 4 dims of v to apply action
        perturbation = torch.zeros_like(v)
        perturbation[:, :4] = a.unsqueeze(0) * 0.1
        
        v_next = v + perturbation
        
        # Map back to manifold
        return PoincareMath.exp_map_0(v_next)
    
    # Mock Objective
    # Target is a random point. Reward is negative distance.
    target_b = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.5)
    
    def mock_objective(b_rollout):
        # b_rollout: (H+1, 1, 16)
        b_final = b_rollout[-1]
        
        # Euclidean distance in tangent space as proxy for "closeness"
        # (Real implementation uses HyperbolicDistance)
        # Using simple norm for test speed
        dist = torch.norm(b_final - target_b)
        return -dist.item() # Maximizing negative distance = minimizing distance
    
    # Setup Planner
    planner = CEMPlanner(
        action_dim=4,
        horizon=5,
        n_iterations=5, # Short run for test
        n_samples=32
    )
    
    # Initial state
    b_init = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.3)
    
    print("  Planning...")
    best_plan = planner.plan(b_init, mock_world_model, mock_objective)
    
    assert best_plan.shape == (5, 4), f"Shape mismatch: {best_plan.shape}"
    print(f"✓ Planned actions shape: {best_plan.shape}")
    print("✓ CEMPlanner operational")