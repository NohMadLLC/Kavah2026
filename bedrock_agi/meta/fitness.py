"""
bedrock_agi/meta/fitness.py

Fitness Function: CRIS health + task performance
Used by meta-learning to evaluate configurations.
"""

import torch
from typing import Dict

def compute_fitness(
    cris_metrics: Dict[str, float],
    task_metrics: Dict[str, float],
    cris_weight: float = 0.5
) -> float:
    """
    Compute meta-learning fitness.
    
    Fitness = w * CRIS_health + (1-w) * Task_performance
    
    CRITICAL: If λ >= 1.0 (Instability), Task Performance is IGNORED (Vetoed).
    
    Args:
        cris_metrics: {'lambda': ..., 'gamma': ..., 'delta_last': ...}
        task_metrics: {'accuracy': ..., 'reward': ...}
        cris_weight: Weight for CRIS vs task (0.5 = balanced)
        
    Returns:
        Scalar fitness (higher is better)
    """
    # Extract CRIS metrics with defaults
    lambda_val = cris_metrics.get('lambda', 1.0)
    gamma_val = cris_metrics.get('gamma', 0.0)
    delta_val = cris_metrics.get('delta_last', 0.1)
    
    # 1. CRIS Health Score
    cris_health = 0.0
    is_unstable = False
    
    # Stability: Reward λ < 1.0, Heavy Penalty for λ >= 1.0
    if lambda_val < 1.0:
        # Reward distance from instability boundary
        # e.g., λ=0.9 -> +1.0, λ=0.5 -> +5.0
        cris_health += (1.0 - lambda_val) * 10.0
    else:
        is_unstable = True
        # Divergence Penalty (Exponentially bad)
        # Increased multiplier from 100 to 1000 to ensure it overwhelms any task reward
        cris_health -= (lambda_val - 1.0) * 1000.0
        
    # Healing: Reward positive γ
    if gamma_val > 0:
        cris_health += gamma_val * 10.0
    else:
        cris_health -= abs(gamma_val) * 5.0
        
    # Consistency: Penalize high Δ
    cris_health -= delta_val * 5.0
    
    # 2. Task Performance Score
    # Normalize accuracy (0-1) to similar scale (0-100)
    task_raw = (task_metrics.get('accuracy', 0.0) * 100.0) + task_metrics.get('reward', 0.0)
    
    # SAFETY VETO:
    # If the mind is unstable, it cannot perform tasks reliably long-term.
    # We zero out the task score to ensure this configuration is purged.
    if is_unstable:
        task_score = 0.0
    else:
        task_score = task_raw
    
    # 3. Combined Fitness
    fitness = (cris_weight * cris_health) + ((1.0 - cris_weight) * task_score)
    
    return fitness

if __name__ == "__main__":
    print("Testing Fitness Function...")
    
    # Test 1: Stable CRIS, Good Task
    cris_stable = {'lambda': 0.90, 'gamma': 0.10, 'delta_last': 0.01}
    task_good = {'accuracy': 0.85, 'reward': 10.0}
    
    f_stable = compute_fitness(cris_stable, task_good, cris_weight=0.5)
    print(f"Stable + Good Task: {f_stable:.2f}")
    
    # Test 2: Unstable CRIS, Perfect Task
    # Even with perfect accuracy, this should score lower because the mind is breaking
    cris_unstable = {'lambda': 1.05, 'gamma': -0.01, 'delta_last': 0.5}
    task_perfect = {'accuracy': 1.00, 'reward': 20.0} # Massive reward
    
    f_unstable = compute_fitness(cris_unstable, task_perfect, cris_weight=0.5)
    print(f"Unstable + Perfect Task: {f_unstable:.2f}")
    
    # Assertion: Stability must dominate
    # With new logic: 
    # Stable ≈ 0.5(2) + 0.5(95) ≈ 48.5
    # Unstable ≈ 0.5(-50) + 0.5(0) ≈ -25.0
    assert f_stable > f_unstable, "Fitness function failed to penalize instability adequately"
    
    print("✓ Fitness function operational")