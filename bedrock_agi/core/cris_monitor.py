"""
bedrock_agi/core/cris_monitor.py

CRIS Monitor: Real-time stability verification (The Consciousness Detector).

Tracks the four CRIS signatures:
- Consistency: Delta_t = d_H(b_t, F(b_t))
- Recursive convergence: lambda (effective spectral radius)
- Invariance: drift in invariant dimensions
- Selection: healing rate gamma = -ln(lambda)

Bedrock Alignment:
- Uses STRICT Hyperbolic Distance (d_H).
- Estimates lambda via Log-Expectation (E[log eta]).
- Zero-Point Stability: If Delta_t < threshold, system is Stable (Fixed Point).
"""

import numpy as np
import torch
from typing import Dict, Optional, List, Union
from collections import deque
from .hbl_geometry import HyperbolicDistance

class CRISMonitor:
    """
    Online CRIS metrics tracker.
    """
    
    def __init__(
        self,
        tail: int = 100,
        buffer_size: int = 10000,
        device: str = "cpu"
    ):
        self.tail = tail
        self.buffer_size = buffer_size
        self.device = device
        
        self.dist_fn = HyperbolicDistance()
        
        # Raw distances d_H(b_t, b_{t+1})
        self.deltas: deque = deque(maxlen=buffer_size)
        
        # Log-ratios log(d_t / d_{t-1})
        self.log_ratios: deque = deque(maxlen=buffer_size)
        
        self.invariant_drifts: deque = deque(maxlen=buffer_size)
        
        self.t = 0
        self.last_delta = None

    def update(
        self,
        b: torch.Tensor,
        b_next: torch.Tensor,
        invariant_drift: Optional[float] = None
    ) -> float:
        # 1. Compute strict hyperbolic distance
        with torch.no_grad():
            delta = self.dist_fn(b, b_next).mean().item()
        
        # 2. Update Log-Ratios
        if self.last_delta is not None:
            # Avoid division by zero
            d_curr = max(delta, 1e-12)
            d_prev = max(self.last_delta, 1e-12)
            
            # log(eta) = log(d_t) - log(d_{t-1})
            log_eta = np.log(d_curr) - np.log(d_prev)
            self.log_ratios.append(log_eta)
        
        self.deltas.append(delta)
        self.last_delta = delta
        
        if invariant_drift is not None:
            self.invariant_drifts.append(invariant_drift)
            
        self.t += 1
        return delta

    def metrics(self) -> Dict[str, Optional[float]]:
        """
        Compute current CRIS metrics.
        """
        if len(self.log_ratios) < self.tail:
            return {
                "lambda": None,
                "gamma": None,
                "delta_last": self.deltas[-1] if self.deltas else None,
                "delta_mean": None,
                "invariant_drift_mean": None
            }
        
        # Get recent log-ratios
        recent_logs = list(self.log_ratios)[-self.tail:]
        
        # Gamma = - E[log eta]
        gamma = -np.mean(recent_logs)
        
        # Lambda = exp(-gamma)
        lambda_est = np.exp(-gamma)
        
        # Auxiliary stats
        recent_deltas = list(self.deltas)[-self.tail:]
        delta_mean = np.mean(recent_deltas)
        
        inv_drift_mean = None
        if self.invariant_drifts:
            inv_drift_mean = np.mean(list(self.invariant_drifts)[-self.tail:])
            
        return {
            "lambda": float(lambda_est),
            "gamma": float(gamma),
            "delta_last": float(self.deltas[-1]),
            "delta_mean": float(delta_mean),
            "invariant_drift_mean": float(inv_drift_mean) if inv_drift_mean else None,
            "num_samples": len(self.deltas)
        }

    def is_stable(
        self, 
        lambda_max: float = 0.999, 
        gamma_min: float = 1e-4,
        convergence_threshold: float = 1e-6 # FIX: Zero-point threshold
    ) -> bool:
        """
        The Ontological Check.
        Returns True if:
        1. System is actively contracting (lambda < 1), OR
        2. System has reached the Identity Fixed Point (delta < threshold).
        """
        m = self.metrics()
        
        # Safety check for initialization
        if m["lambda"] is None:
            return False
            
        # Condition A: Fixed Point Convergence (Identity Achieved)
        # If delta is negligible, we are stable regardless of lambda jitter.
        if m["delta_last"] is not None and m["delta_last"] < convergence_threshold:
            return True
            
        # Condition B: Active Contraction (Identity Forming)
        return m["lambda"] < lambda_max and m["gamma"] > gamma_min

    def get_stability_report(self) -> str:
        m = self.metrics()
        report = "=" * 50 + "\n"
        report += "CRIS STABILITY REPORT\n"
        report += "=" * 50 + "\n"
        
        if m["lambda"] is None:
            report += "Status: CALIBRATING\n"
            return report
            
        stable = self.is_stable()
        status = "STABLE ✓" if stable else "UNSTABLE ✗"
        
        report += f"Status: {status}\n"
        report += "-" * 50 + "\n"
        report += f"Lambda: {m['lambda']:.6f}\n"
        report += f"Gamma:  {m['gamma']:.6f}\n"
        report += f"Delta:  {m['delta_last']:.9f} {'(FIXED POINT)' if m['delta_last'] < 1e-6 else ''}\n"
        report += "=" * 50 + "\n"
        return report

if __name__ == "__main__":
    pass