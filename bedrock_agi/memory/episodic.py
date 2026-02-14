"""
bedrock_agi/memory/episodic.py

Episodic Memory: Ring buffer with disk persistence.

Stores recent (b_t, invariants, metadata) tuples.
Automatically saves to disk and loads on startup.
"""

import torch
import pickle
import os
from collections import deque
from typing import Optional, Dict, Any

class EpisodicMemory:
    """
    Ring buffer of recent latent states.
    
    Persists to disk automatically.
    """
    
    def __init__(self, capacity=10000, save_path="memory/episodic.pkl"):
        self.capacity = capacity
        self.save_path = save_path
        # We start with an empty deque; load() will populate it if file exists
        self.buffer = deque(maxlen=capacity)
        
        # Create directory if needed
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Try to load existing memory
        self.load()
    
    def write(self, b: torch.Tensor, invariants, metadata: Dict = None):
        """
        Add entry to episodic memory.
        
        Args:
            b: Latent state (tensor)
            invariants: Invariant values (tensor or array)
            metadata: Optional dict with context
        """
        entry = {
            'b': b.detach().cpu(),
            'invariants': invariants if torch.is_tensor(invariants) else torch.tensor(invariants),
            'metadata': metadata or {}
        }
        self.buffer.append(entry)
    
    def sample(self, n=64):
        """Sample random entries."""
        if len(self.buffer) == 0:
            return []
        
        import random
        n_samples = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n_samples)
    
    def recent(self, n=10):
        """Get most recent entries."""
        n_recent = min(n, len(self.buffer))
        return list(self.buffer)[-n_recent:]
    
    def save(self):
        """Save to disk."""
        try:
            with open(self.save_path, 'wb') as f:
                pickle.dump(list(self.buffer), f)
            return True
        except Exception as e:
            print(f"Warning: Failed to save episodic memory: {e}")
            return False
    
    def load(self):
        """Load from disk."""
        if not os.path.exists(self.save_path):
            return False
        
        try:
            with open(self.save_path, 'rb') as f:
                entries = pickle.load(f)
                self.buffer = deque(entries, maxlen=self.capacity)
            print(f"Loaded {len(self.buffer)} episodic memories from {self.save_path}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load episodic memory: {e}")
            return False
    
    def clear(self):
        """Clear all memories."""
        self.buffer.clear()
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
    
    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    print("Testing EpisodicMemory...")
    
    # Test 1: Write and read
    # Use a temp path for testing
    test_path = "/tmp/test_episodic.pkl"
    # Ensure tmp dir exists
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    mem = EpisodicMemory(capacity=100, save_path=test_path)
    
    b = torch.randn(1, 16) * 0.3
    inv = torch.randn(8)
    mem.write(b, inv, metadata={'task': 'test'})
    
    assert len(mem) == 1
    print("✓ Write works")
    
    # Test 2: Sample
    for i in range(20):
        mem.write(torch.randn(1, 16) * 0.3, torch.randn(8))
    
    samples = mem.sample(10)
    assert len(samples) == 10
    print("✓ Sample works")
    
    # Test 3: Save and load
    mem.save()
    
    mem2 = EpisodicMemory(capacity=100, save_path=test_path)
    assert len(mem2) == len(mem)
    print("✓ Persistence works")
    
    # Cleanup
    mem.clear()
    
    print("✓ EpisodicMemory operational")