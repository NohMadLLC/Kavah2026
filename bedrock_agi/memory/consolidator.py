"""
bedrock_agi/memory/consolidator.py

Consolidator: Sleep mode (episodic -> semantic).
Runs offline to distill episodic memories into semantic knowledge.
"""

import torch
from typing import List, Dict, Any
from .episodic import EpisodicMemory
from .semantic import SemanticMemory

class Consolidator:
    """
    Sleep/consolidation process.
    
    Takes episodic samples and distills them into semantic memory.
    Optimizes for CRIS (low Delta, high invariance preservation).
    Strategies:
    - Diversity Sampling: Only add if sufficiently different from existing semantic memories.
    - Invariant Reinforcement: (Future) Strengthen existing memories if invariants match.
    """
    
    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory):
        self.episodic = episodic
        self.semantic = semantic
    
    def consolidate(self, n_samples=128, threshold=0.1) -> int:
        """
        Run one consolidation cycle.
        
        Args:
            n_samples: How many episodic samples to process
            threshold: Minimum hyperbolic distance for semantic storage (Diversity)
            
        Returns:
            Number of entries added to semantic memory
        """
        if len(self.episodic) == 0:
            return 0
        
        # Sample from episodic
        samples = self.episodic.sample(n_samples)
        
        added_count = 0
        
        for entry in samples:
            b = entry['b'] # Tensor
            inv = entry['invariants']
            meta = entry.get('metadata', {})
            
            # Check if similar entry already exists in semantic memory
            # If semantic memory is empty, add immediately
            if len(self.semantic) > 0:
                # Query nearest neighbor
                results = self.semantic.query(b, k=1)
                
                # If closest match is too close, skip (redundant information)
                if len(results) > 0 and results[0]['distance'] < threshold:
                    continue 
            
            # If distinct enough, promote to Semantic
            self.semantic.add(b, inv, metadata=meta)
            added_count += 1
            
        return added_count
    
    def run_sleep_cycle(self, n_cycles=5, batch_size=64):
        """
        Run multiple consolidation cycles.
        
        Args:
            n_cycles: Number of passes
            batch_size: Samples per pass
            
        Returns:
            Total entries consolidated
        """
        total_added = 0
        print(f"Starting Sleep Cycle... (Episodic: {len(self.episodic)}, Semantic: {len(self.semantic)})")
        
        for i in range(n_cycles):
            added = self.consolidate(n_samples=batch_size)
            total_added += added
            
            # If nothing was added, we might have saturated diversity
            if added == 0:
                print(f"Cycle {i+1}: Saturation reached.")
                break
        
        # Save both memories after modification
        self.episodic.save()
        self.semantic.save()
        
        print(f"Sleep Cycle Complete. Consolidated {total_added} new memories.")
        return total_added

if __name__ == "__main__":
    print("Testing Consolidator...")
    
    # Lazy import for test setup
    from ..core.hbl_geometry import PoincareMath
    import os
    
    # Setup paths
    ep_path = "/tmp/test_ep.pkl"
    sem_path = "/tmp/test_sem.pkl"
    
    # Ensure dirs
    if os.path.dirname(ep_path): os.makedirs(os.path.dirname(ep_path), exist_ok=True)
    if os.path.dirname(sem_path): os.makedirs(os.path.dirname(sem_path), exist_ok=True)
    
    # Create memories
    episodic = EpisodicMemory(capacity=100, save_path=ep_path)
    semantic = SemanticMemory(latent_dim=16, save_path=sem_path)
    
    # Clear previous runs
    episodic.clear()
    semantic.clear()
    
    print("Filling episodic memory...")
    # Fill episodic with random data
    for i in range(50):
        # Create random point
        raw = torch.randn(1, 16) * 0.5
        b = PoincareMath.project_to_ball(raw)
        inv = torch.randn(8)
        episodic.write(b, inv, metadata={'id': i})
    
    print(f"Episodic: {len(episodic)} entries")
    print(f"Semantic: {len(semantic)} entries")
    
    # Run Consolidator
    consolidator = Consolidator(episodic, semantic)
    added = consolidator.run_sleep_cycle(n_cycles=3, batch_size=10)
    
    assert added > 0, "Consolidator failed to add any memories"
    assert len(semantic) == added, "Semantic count mismatch"
    
    print(f"✓ Consolidated {added} entries")
    print(f"Semantic now: {len(semantic)} entries")
    
    # Cleanup
    episodic.clear()
    semantic.clear()
    
    print("✓ Consolidator operational")