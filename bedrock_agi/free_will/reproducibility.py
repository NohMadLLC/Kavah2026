"""
bedrock_agi/free_will/reproducibility.py

Reproducibility Verification: Same input -> same output.
Tests determinism of the entire system (end-to-end).
"""

import torch
import numpy as np
import random
import os
from typing import Callable, Any, List

def set_global_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    Must be called at the start of any deterministic process.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Ensure deterministic algorithms (might slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def verify_reproducibility(
    system_fn: Callable[[Any], Any],
    input_data: Any,
    n_trials: int = 3,
    seed: int = 42
) -> bool:
    """
    Verify system produces identical outputs for identical inputs given a seed reset.
    
    Args:
        system_fn: Callable taking input -> output
        input_data: Test input
        n_trials: Number of repetitions
        seed: Random seed to enforce
        
    Returns:
        True if all outputs identical
    """
    outputs = []
    
    for trial in range(n_trials):
        # Reset seed before each trial to ensure same starting conditions
        set_global_seed(seed)
        
        # Run system
        # Note: If system has internal state that persists (like RNN hidden state),
        # the system_fn wrapper must handle resetting that state.
        output = system_fn(input_data)
        outputs.append(output)
        
    # Compare all outputs to the first one
    base_output = outputs[0]
    
    for i in range(1, n_trials):
        current = outputs[i]
        
        # Comparison logic based on type
        if torch.is_tensor(base_output):
            if not torch.allclose(base_output, current, atol=1e-6):
                return False
        elif isinstance(base_output, np.ndarray):
            if not np.allclose(base_output, current, atol=1e-6):
                return False
        elif isinstance(base_output, (list, tuple)):
            # Deep comparison for lists
            if str(base_output) != str(current):
                return False
        elif isinstance(base_output, dict):
             if str(base_output) != str(current):
                return False
        else:
            if base_output != current:
                return False
                
    return True

if __name__ == "__main__":
    print("Testing Reproducibility Verification...")
    
    # Test 1: Deterministic Function (Success)
    def deterministic_op(x):
        # Depends on seed
        return torch.randn_like(x) * 0.1 + x
        
    x = torch.zeros(10)
    is_repro = verify_reproducibility(deterministic_op, x, n_trials=5)
    assert is_repro
    print("✓ Deterministic function verified")
    
    # Test 2: Non-deterministic Function (Failure)
    # Wrapper that DOES NOT reset seed internally (but verify_reproducibility does reset global seed)
    # Wait, if verify_reproducibility resets global seed, then standard randn() SHOULD be deterministic.
    # To test failure, we need a function that relies on external state (like time or hardware randomness).
    
    import time
    def nondeterministic_op(x):
        return x + time.time() # Time changes between calls
        
    is_repro_time = verify_reproducibility(nondeterministic_op, x, n_trials=5)
    assert not is_repro_time
    print("✓ Non-deterministic (time-based) correctly failed")
    
    # Test 3: Accumulated State (Failure without reset)
    class Counter:
        def __init__(self): self.count = 0
        def __call__(self, x):
            self.count += 1
            return self.count
            
    # Re-instantiating the object inside the loop would fix it, but here we pass the instance
    # which holds state. Even with seed reset, the python object state persists.
    c = Counter()
    is_repro_state = verify_reproducibility(c, x, n_trials=5)
    assert not is_repro_state
    print("✓ Stateful drift correctly detected")
    
    print("✓ Reproducibility verification operational")