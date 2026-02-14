"""
bedrock_agi/planning/mpc_objective.py

MPC Objective: Reward - CRIS penalty
Objective for Model Predictive Control in latent space.
Balances task reward with stability preservation.
"""

import torch

def compute_objective(
    b_rollout,
    task_reward_fn,
    cris_penalty_weight=0.5,
    invariant_penalty_weight=0.3
):
    """
    Compute planning objective.
    
    Objective = TaskReward - lambda_c * CRIS_penalty - lambda_inv * Invariant_penalty
    
    Args:
        b_rollout: Sequence of latent states (T, B, n) or list of tensors
        task_reward_fn: Function computing task reward from final state
        cris_penalty_weight: Weight for CRIS consistency penalty
        invariant_penalty_weight: Weight for invariant drift
        
    Returns:
        Scalar objective (higher is better)
    """
    from ..core.hbl_geometry import HyperbolicDistance
    
    # Task reward from final state
    b_final = b_rollout[-1]  # (B, n)
    # Assume task_reward_fn returns (B,) or scalar
    task_reward = task_reward_fn(b_final).mean()
    
    # CRIS penalty: consistency across rollout
    # We penalize large geometric jumps (instability)
    dist_fn = HyperbolicDistance()
    cris_penalty = 0.0
    
    # Iterate through time steps
    for t in range(len(b_rollout) - 1):
        # Calculate hyperbolic distance between t and t+1
        d = dist_fn(b_rollout[t], b_rollout[t + 1])
        cris_penalty += d.mean()
    
    # Invariant penalty: TODO - requires invariant model
    # Ideally, we check if invariants (I) are preserved: ||I(b_0) - I(b_t)||
    invariant_penalty = 0.0
    
    # Combined objective
    # We want to Maximize Reward and Minimize Penalty
    objective = (
        task_reward 
        - cris_penalty_weight * cris_penalty 
        - invariant_penalty_weight * invariant_penalty
    )
    
    return objective

if __name__ == "__main__":
    print("Testing MPC Objective...")
    
    # Mock rollout: 5 steps, Batch=2, Dim=16
    # Create valid hyperbolic points (norm < 1)
    b_rollout = [torch.randn(2, 16) * 0.3 for _ in range(5)]
    # We keep it as a list for the loop, or stack it.
    # The function handles indexing, so list is fine.
    
    # Mock task reward (prefer states near origin)
    # Higher reward for smaller norm (closer to 0)
    def task_reward(b):
        return -torch.norm(b, dim=-1)
    
    obj = compute_objective(b_rollout, task_reward)
    
    print(f"Objective: {obj.item():.4f}")
    
    # Check that penalty logic works (larger jumps = lower objective)
    b_jittery = [torch.randn(2, 16) * 0.8 for _ in range(5)] # Larger variance
    obj_jittery = compute_objective(b_jittery, task_reward)
    
    print(f"Stable Objective: {obj.item():.4f}")
    print(f"Jittery Objective: {obj_jittery.item():.4f}")
    
    assert obj > obj_jittery, "MPC failed to penalize instability"
    
    print("✓ MPC Objective operational")