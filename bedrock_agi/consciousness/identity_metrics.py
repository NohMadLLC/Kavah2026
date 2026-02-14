"""
bedrock_agi/consciousness/identity_metrics.py

Identity Metrics: Track self-model convergence and stability.
This is the "Dashboard of the Soul" - objective measures of identity persistence.
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, Optional

class IdentityMetrics:
    """
    Track and compute identity stability metrics over a rolling window.
    
    Metrics:
    - Convergence: 1 - ||s_{t+1} - s_t|| / ||s|| (Higher is better, max 1.0)
    - Drift: Rate of state change over window (Lower is better)
    - Influence Stability: (Implicitly tracked via state stability)
    """
    
    def __init__(self, history_len: int = 50):
        self.history_len = history_len
        # We store tensors on CPU to save VRAM and avoid graph retention
        self.states = deque(maxlen=history_len)
        self.influences = deque(maxlen=history_len)
    
    def update(self, s: torch.Tensor, u: torch.Tensor):
        """
        Record state and influence for the current step.
        
        Args:
            s: Self-state tensor (B, state_dim)
            u: Influence tensor (B, latent_dim)
        """
        self.states.append(s.detach().cpu())
        self.influences.append(u.detach().cpu())
    
    def compute(self) -> Dict[str, Optional[float]]:
        """
        Compute aggregate metrics from history.
        
        Returns:
            Dict containing 'convergence', 'drift', 'samples'
        """
        if len(self.states) < 2:
            return {
                'convergence': None, 
                'drift': None,
                'samples': len(self.states)
            }
        
        # Stack history: (T, B, D)
        # Note: We take the last element of the batch if batch > 1 for simplicity,
        # or average across batch. Here we assume B=1 or track average.
        # Let's compute average metric across batch.
        
        states = torch.stack(list(self.states), dim=0) # (T, B, D)
        
        # 1. Compute Differences (Velocity)
        # diffs[t] = states[t+1] - states[t]
        diffs = torch.diff(states, dim=0) # (T-1, B, D)
        
        # 2. Compute Norms
        diff_norms = torch.norm(diffs, dim=-1).mean(dim=1) # (T-1,)
        state_norms = torch.norm(states, dim=-1).mean(dim=1) # (T,)
        
        # 3. Averages over window
        avg_diff = diff_norms.mean().item()
        avg_norm = state_norms.mean().item()
        
        # 4. Metrics
        # Convergence: 1.0 means perfectly static (fixed point)
        # We add epsilon to denominator to avoid div/0
        convergence = 1.0 - (avg_diff / (avg_norm + 1e-8))
        
        # Drift: Average absolute change per step
        drift = avg_diff
        
        return {
            'convergence': convergence,
            'drift': drift,
            'samples': len(self.states)
        }

if __name__ == "__main__":
    print("Testing IdentityMetrics...")
    torch.manual_seed(42)
    
    tracker = IdentityMetrics(history_len=20)
    
    # Simulate converging sequence
    # s starts random, then decays changes
    s = torch.randn(1, 32)
    u = torch.randn(1, 16) * 0.05
    
    print("Simulating convergence...")
    for i in range(50):
        tracker.update(s, u)
        # Convergence dynamic: move 5% towards zero + small noise
        # This simulates settling into a fixed point
        noise = torch.randn(1, 32) * (0.01 / (i + 1)) # Noise decreases
        s = s * 0.95 + noise
    
    metrics = tracker.compute()
    print(f"Final Convergence: {metrics['convergence']:.4f}")
    print(f"Final Drift:       {metrics['drift']:.6f}")
    
    # Check logic
    assert metrics['convergence'] > 0.8, "Metric failed to detect convergence"
    assert metrics['drift'] < 0.1, "Metric failed to detect low drift"
    
    print("✓ IdentityMetrics operational")