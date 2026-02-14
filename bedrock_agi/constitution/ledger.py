"""
bedrock_agi/constitution/ledger.py

Goal Ledger: Immutable audit log.
Tracks every goal, decision, and outcome.
"""

import time
import json
import os

class GoalLedger:
    """Simple append-only log."""
    
    def __init__(self):
        self.entries = []
    
    def append(self, goal_text: str, verdict: str, provenance: str = "user") -> int:
        """
        Add entry to the ledger.
        
        Args:
            goal_text: Description of the goal
            verdict: Decision made (e.g., 'execute', 'reject', 'clarify')
            provenance: Origin of the goal
            
        Returns:
            id: The index of the new entry
        """
        entry = {
            'id': len(self.entries),
            'goal': goal_text,
            'verdict': verdict,
            'provenance': provenance,
            'timestamp': time.time()
        }
        self.entries.append(entry)
        return entry['id']
    
    def get(self, gid: int):
        """Retrieve entry by ID."""
        if 0 <= gid < len(self.entries):
            return self.entries[gid]
        return None
    
    def save(self, path: str):
        """Save ledger to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.entries, f, indent=2)
    
    def load(self, path: str):
        """Load ledger from JSON file."""
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            self.entries = json.load(f)

if __name__ == "__main__":
    print("Testing Ledger...")
    
    ledger = GoalLedger()
    
    gid = ledger.append("test goal", "execute")
    assert gid == 0
    print("✓ Append works")
    
    entry = ledger.get(0)
    assert entry['goal'] == "test goal"
    print("✓ Retrieval works")
    
    # Test persistence
    test_path = 'test_ledger.json'
    ledger.save(test_path)
    
    ledger_loaded = GoalLedger()
    ledger_loaded.load(test_path)
    assert len(ledger_loaded.entries) == 1
    assert ledger_loaded.entries[0]['verdict'] == "execute"
    print("✓ Save/load works")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("✓ Ledger operational")