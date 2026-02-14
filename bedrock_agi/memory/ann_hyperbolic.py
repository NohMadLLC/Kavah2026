"""
bedrock_agi/memory/ann_hyperbolic.py

Hyperbolic ANN: Nearest neighbor search in H^n.

Uses log_0 to map to Euclidean tangent space, then standard ANN logic.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any

class HyperbolicANN:
    """
    Approximate nearest neighbor in Poincaré ball.
    
    Strategy:
    1. Store points in H^n (Preserves Ground Truth)
    2. Map to tangent space via log_0 for search index
    3. Use Euclidean distance in tangent space as approximation for retrieval
    """
    
    def __init__(self, dim=16):
        self.dim = dim
        self.points = []  # List of tensors in H^n
        self.tangent_points = []  # log_0(points) for search
        self.metadata = []  # Associated metadata
    
    def add(self, b: torch.Tensor, metadata: Dict[str, Any] = None):
        """
        Add point to index.
        
        Args:
            b: Point in H^n (tensor)
            metadata: Optional associated data
        """
        # Lazy import to avoid circular dependencies during initialization
        from ..core.hbl_geometry import PoincareMath
        
        # Detach and move to CPU for storage
        b_cpu = b.detach().cpu()
        self.points.append(b_cpu)
        
        # Map to tangent space for search
        # log_map_0 is the inverse of exp_map_0
        v = PoincareMath.log_map_0(b_cpu)
        self.tangent_points.append(v)
        
        self.metadata.append(metadata)
    
    def search(self, query: torch.Tensor, k: int = 5) -> List[Dict]:
        """
        Find k nearest neighbors.
        
        Args:
            query: Query point in H^n
            k: Number of neighbors
        
        Returns:
            List of dicts: {'distance', 'index', 'point', 'metadata'}
        """
        if len(self.points) == 0:
            return []
        
        from ..core.hbl_geometry import PoincareMath
        
        # Map query to tangent space
        q_tangent = PoincareMath.log_map_0(query.detach().cpu())
        
        # Stack tangent points for vectorized Euclidean distance
        # tangent_points are likely (1, D), stacking makes (N, 1, D)
        # We squeeze to ensure (N, D)
        tangent_stack = torch.stack(self.tangent_points, dim=0).squeeze(1) # (N, dim)
        q_vec = q_tangent.squeeze(0) # (dim,)
        
        # Compute Euclidean distances in tangent space
        # |x - y|
        dists = torch.norm(tangent_stack - q_vec, dim=-1)  # (N,)
        
        # Get top k
        k_actual = min(k, len(self.points))
        # topk with largest=False gives smallest distances
        top_k_dists, top_k_indices = torch.topk(dists, k_actual, largest=False)
        
        results = []
        for dist, idx in zip(top_k_dists, top_k_indices):
            i = idx.item()
            results.append({
                'distance': dist.item(),
                'index': i,
                'point': self.points[i],
                'metadata': self.metadata[i]
            })
        
        return results
    
    def __len__(self):
        return len(self.points)


if __name__ == "__main__":
    print("Testing HyperbolicANN...")
    
    from ..core.hbl_geometry import PoincareMath
    import torch
    
    # Test 1: Add points
    ann = HyperbolicANN(dim=16)
    
    print("Generating synthetic hyperbolic data...")
    for i in range(20):
        # Create random points and project to ball to ensure validity
        raw = torch.randn(1, 16) * 0.5
        b = PoincareMath.project_to_ball(raw)
        ann.add(b, metadata={'id': i})
    
    assert len(ann) == 20
    print("✓ Adding points works")
    
    # Test 2: Search
    query_raw = torch.randn(1, 16) * 0.5
    query = PoincareMath.project_to_ball(query_raw)
    
    results = ann.search(query, k=5)
    
    assert len(results) == 5
    print(f"✓ Search works, nearest neighbor distance: {results[0]['distance']:.4f}")
    print(f"✓ Metadata retrieval: {results[0]['metadata']}")
    
    print("✓ HyperbolicANN operational")