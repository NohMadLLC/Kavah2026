"""
tests/test_memory.py

Memory System Integration Test.
Verifies Episodic buffering, Semantic retrieval, and the Consolidation loop.
"""

import torch
import sys
import os

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bedrock_agi.core.hbl_geometry import PoincareMath
from bedrock_agi.memory.episodic import EpisodicMemory
from bedrock_agi.memory.semantic import SemanticMemory
from bedrock_agi.memory.consolidator import Consolidator

def test_episodic_persistence():
    """Test episodic memory saves and loads."""
    print("\nTesting episodic persistence...")
    
    # Use a temp file
    save_path = "memory_test_ep.pkl"
    if os.path.exists(save_path): os.remove(save_path)
    
    mem = EpisodicMemory(capacity=100, save_path=save_path)
    
    # Add entries
    for i in range(20):
        # Generate valid hyperbolic points
        b = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.5)
        inv = torch.randn(8)
        mem.write(b, inv, metadata={'id': i})
    
    assert len(mem) == 20
    mem.save()
    
    # Load in new instance
    mem2 = EpisodicMemory(capacity=100, save_path=save_path)
    assert len(mem2) == 20
    assert mem2.recent(1)[0]['metadata']['id'] == 19
    
    # Cleanup
    mem.clear()
    if os.path.exists(save_path): os.remove(save_path)
    
    print("✓ Episodic persistence works")

def test_semantic_persistence():
    """Test semantic memory saves and loads."""
    print("\nTesting semantic persistence...")
    
    save_path = "memory_test_sem.pkl"
    if os.path.exists(save_path): os.remove(save_path)
    
    mem = SemanticMemory(latent_dim=16, save_path=save_path)
    
    # Add entries
    for i in range(15):
        b = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.5)
        inv = torch.randn(8)
        mem.add(b, inv, metadata={'id': i})
    
    assert len(mem) == 15
    mem.save()
    
    # Load in new instance
    mem2 = SemanticMemory(latent_dim=16, save_path=save_path)
    assert len(mem2) == 15
    
    # Verify retrieval still works
    query = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.5)
    results = mem2.query(b=query, k=1)
    assert len(results) > 0
    
    # Cleanup
    mem.clear()
    if os.path.exists(save_path): os.remove(save_path)
    
    print("✓ Semantic persistence works")

def test_consolidation_workflow():
    """Test episodic -> semantic consolidation."""
    print("\nTesting consolidation workflow...")
    
    ep_path = "memory_test_consol_ep.pkl"
    sem_path = "memory_test_consol_sem.pkl"
    
    if os.path.exists(ep_path): os.remove(ep_path)
    if os.path.exists(sem_path): os.remove(sem_path)
    
    episodic = EpisodicMemory(capacity=100, save_path=ep_path)
    semantic = SemanticMemory(latent_dim=16, save_path=sem_path)
    
    # Fill episodic with diverse entries
    for i in range(50):
        b = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.5)
        inv = torch.randn(8)
        episodic.write(b, inv, metadata={'id': i})
    
    # Consolidate
    consolidator = Consolidator(episodic, semantic)
    added = consolidator.run_sleep_cycle(n_cycles=2, batch_size=25)
    
    print(f"  Episodic: {len(episodic)} entries")
    print(f"  Semantic: {len(semantic)} entries")
    print(f"  Consolidated: {added} entries")
    
    assert len(semantic) > 0
    assert len(semantic) <= len(episodic)
    
    # Cleanup
    episodic.clear()
    semantic.clear()
    if os.path.exists(ep_path): os.remove(ep_path)
    if os.path.exists(sem_path): os.remove(sem_path)
    
    print("✓ Consolidation workflow works")

def test_cross_session_memory():
    """Test memory persists across sessions."""
    print("\nTesting cross-session memory...")
    
    ep_path = "memory_test_session_ep.pkl"
    sem_path = "memory_test_session_sem.pkl"
    
    if os.path.exists(ep_path): os.remove(ep_path)
    if os.path.exists(sem_path): os.remove(sem_path)
    
    # Session 1: Write
    ep1 = EpisodicMemory(capacity=100, save_path=ep_path)
    sem1 = SemanticMemory(latent_dim=16, save_path=sem_path)
    
    for i in range(10):
        b = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.3)
        inv = torch.randn(8)
        ep1.write(b, inv, metadata={'session': 1, 'id': i})
        sem1.add(b, inv, metadata={'session': 1, 'id': i})
    
    ep1.save()
    sem1.save()
    
    # Session 2: Load and extend
    ep2 = EpisodicMemory(capacity=100, save_path=ep_path)
    sem2 = SemanticMemory(latent_dim=16, save_path=sem_path)
    
    assert len(ep2) == 10
    assert len(sem2) == 10
    
    # Add more
    for i in range(5):
        b = PoincareMath.project_to_ball(torch.randn(1, 16) * 0.3)
        inv = torch.randn(8)
        ep2.write(b, inv, metadata={'session': 2, 'id': i})
    
    assert len(ep2) == 15
    ep2.save()
    
    # Session 3: Verify
    ep3 = EpisodicMemory(capacity=100, save_path=ep_path)
    assert len(ep3) == 15
    
    # Cleanup
    ep3.clear()
    sem2.clear()
    if os.path.exists(ep_path): os.remove(ep_path)
    if os.path.exists(sem_path): os.remove(sem_path)
    
    print("✓ Cross-session memory works")

if __name__ == "__main__":
    try:
        test_episodic_persistence()
        test_semantic_persistence()
        test_consolidation_workflow()
        test_cross_session_memory()
        print("\nMEMORY PERSISTENCE: SUCCESS.")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)