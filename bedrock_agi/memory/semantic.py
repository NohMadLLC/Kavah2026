"""
bedrock_agi/memory/semantic.py

Semantic Memory: Long-term knowledge indexed by invariants.
Uses HyperbolicANN for retrieval.
Persists to disk.
"""

import torch
import pickle
import os
from typing import List, Dict, Any, Optional
from .ann_hyperbolic import HyperbolicANN

class SemanticMemory:
    """
    Long-term memory indexed by invariant patterns.
    Stores (b, invariants, metadata) with ANN retrieval.
    """
    
    def __init__(self, latent_dim=16, save_path="memory/semantic.pkl"):
        self.latent_dim = latent_dim
        self.save_path = save_path
        self.index = HyperbolicANN(dim=latent_dim)
        
        # Create directory
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Load if exists
        self.load()
    
    def add(self, b: torch.Tensor, invariants, metadata: Dict = None):
        """
        Add to semantic memory.
        
        Args:
            b: Latent state in H^n
            invariants: Invariant values (for metadata)
            metadata: Optional additional context
        """
        meta = metadata or {}
        # Store invariants in metadata for retrieval/filtering
        meta['invariants'] = invariants if torch.is_tensor(invariants) else torch.tensor(invariants)
        
        self.index.add(b, metadata=meta)
    
    def query(self, b: Optional[torch.Tensor] = None, invariants=None, k: int = 5):
        """
        Query semantic memory.
        
        Args:
            b: Query latent (if provided)
            invariants: Query invariants (if provided) - Currently unused for search, planned for filtering
            k: Number of results
            
        Returns:
            List of matching entries
        """
        if b is not None:
            return self.index.search(b, k=k)
        
        # TODO: If querying by invariants alone, need different search mechanism
        # For now, require latent query
        return []
    
    def save(self):
        """Save to disk."""
        try:
            # We save the raw points and metadata to reconstruct the index later
            data = {
                'points': self.index.points,
                'metadata': self.index.metadata
            }
            with open(self.save_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Warning: Failed to save semantic memory: {e}")
            return False
    
    def load(self):
        """Load from disk."""
        if not os.path.exists(self.save_path):
            return False
        
        try:
            with open(self.save_path, 'rb') as f:
                data = pickle.load(f)
            
            # Reconstruct index from raw data
            # Note: We re-add them to rebuild the tangent space cache in HyperbolicANN
            self.index = HyperbolicANN(dim=self.latent_dim)
            
            # We assume points are already tensors
            for point, meta in zip(data['points'], data['metadata']):
                # We use the internal lists directly or re-add?
                # Re-adding ensures tangent_points are calculated correctly.
                self.index.add(point, metadata=meta)
                
            print(f"Loaded {len(self.index)} semantic memories from {self.save_path}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load semantic memory: {e}")
            return False
    
    def clear(self):
        """Clear all memories."""
        self.index = HyperbolicANN(dim=self.latent_dim)
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
    
    def __len__(self):
        return len(self.index)


if __name__ == "__main__":
    print("Testing SemanticMemory...")
    
    # Lazy import for test
    from ..core.hbl_geometry import PoincareMath
    
    # Test 1: Add entries
    # Use tmp path
    test_path = "/tmp/test_semantic.pkl"
    if os.path.dirname(test_path):
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        
    mem = SemanticMemory(latent_dim=16, save_path=test_path)
    
    for i in range(10):
        # Generate valid hyperbolic points
        raw = torch.randn(1, 16) * 0.5
        b = PoincareMath.project_to_ball(raw)
        inv = torch.randn(8)
        mem.add(b, inv, metadata={'id': i})
    
    assert len(mem) == 10
    print("✓ Adding works")
    
    # Test 2: Query
    query_raw = torch.randn(1, 16) * 0.5
    query = PoincareMath.project_to_ball(query_raw)
    
    results = mem.query(b=query, k=3)
    
    assert len(results) == 3
    print(f"✓ Query works, found {len(results)} results")
    
    # Test 3: Persistence
    mem.save()
    
    mem2 = SemanticMemory(latent_dim=16, save_path=test_path)
    assert len(mem2) == len(mem)
    print("✓ Persistence works")
    
    # Cleanup
    mem.clear()
    
    print("✓ SemanticMemory operational")