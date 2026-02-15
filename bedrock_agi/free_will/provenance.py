"""
bedrock_agi/free_will/provenance.py

Decision Provenance: Full causal trace from state to action.
Enables reproducibility and accountability by logging the "why" behind every "what".
"""

import time
import json
import hashlib
import os
from typing import Dict, Any, List, Optional, Union

class DecisionTrace:
    """
    Complete provenance record for a single decision.
    
    Records:
    - Input state hash (What did I see?)
    - CRIS metrics at decision time (How was I feeling?)
    - Constitution verdict (Was it legal?)
    - Chosen action (What did I do?)
    - Counterfactuals considered (What else could I have done?)
    """
    
    def __init__(self):
        self.trace_id = self._generate_id()
        self.timestamp = time.time()
        self.state_hash = None
        self.cris_metrics = {}
        self.constitution_verdict = {}
        self.chosen_action = None
        self.counterfactuals = []
        self.reason_vector = {}
        
    def _generate_id(self) -> str:
        """Generate unique trace ID based on time and random entropy."""
        raw = f"{time.time()}-{os.urandom(4).hex()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
        
    def record_state(self, state: Any):
        """
        Record hash of the input state (latent vector).
        We don't store the full vector to save space, just the hash for verification.
        """
        if hasattr(state, 'numpy'):
            # Handle PyTorch/NumPy
            try:
                state_bytes = state.numpy().tobytes()
            except:
                state_bytes = str(state).encode()
        elif hasattr(state, 'tobytes'):
            state_bytes = state.tobytes()
        else:
            state_bytes = str(state).encode()
            
        self.state_hash = hashlib.sha256(state_bytes).hexdigest()[:16]
        
    def record_cris(self, metrics: Dict[str, float]):
        """Record CRIS metrics at the moment of decision."""
        self.cris_metrics = metrics.copy()
        
    def record_constitution(self, verdict: str, reason: str):
        """Record constitutional judgment."""
        self.constitution_verdict = {
            'verdict': verdict,
            'reason': reason
        }
        
    def record_action(self, action: Any):
        """Record the action that was actually executed."""
        self.chosen_action = str(action)
        
    def add_counterfactual(self, action: Any, predicted_outcome: Any, rejection_reason: str = None):
        """
        Record an alternative action that was considered but rejected.
        Essential for proving 'Free Will' (ability to have done otherwise).
        """
        self.counterfactuals.append({
            'action': str(action),
            'predicted_outcome': str(predicted_outcome),
            'rejection_reason': rejection_reason
        })
        
    def set_reason_vector(self, vector: Dict[str, float]):
        """Record the internal weighting/attention that led to the choice."""
        self.reason_vector = vector
        
    def to_dict(self) -> Dict:
        """Serialize trace to dictionary."""
        return {
            'trace_id': self.trace_id,
            'timestamp': self.timestamp,
            'state_hash': self.state_hash,
            'cris_metrics': self.cris_metrics,
            'constitution': self.constitution_verdict,
            'chosen_action': self.chosen_action,
            'counterfactuals': self.counterfactuals,
            'reason_vector': self.reason_vector
        }
        
    def save(self, path: str):
        """Save individual trace to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class ProvenanceLogger:
    """
    Persistent logger for decision traces.
    Appends traces to a journal (JSONL) for auditing.
    """
    
    def __init__(self, log_path: str = "logs/provenance.jsonl"):
        self.log_path = log_path
        
        # Ensure directory exists
        dirname = os.path.dirname(log_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
    def log(self, trace: DecisionTrace):
        """Append a finished trace to the log."""
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(trace.to_dict()) + '\n')
        except Exception as e:
            print(f"ERROR: Failed to log provenance: {e}")
            
    def load_traces(self) -> List[Dict]:
        """Load all traces from history."""
        if not os.path.exists(self.log_path):
            return []
            
        traces = []
        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        traces.append(json.loads(line))
        except Exception as e:
            print(f"ERROR: Failed to load provenance: {e}")
            return []
        return traces

if __name__ == "__main__":
    print("Testing Decision Provenance...")
    
    # Mock Torch for state hashing
    import torch
    
    # 1. Create a Decision Trace
    trace = DecisionTrace()
    
    # 2. Populate Data
    # State
    dummy_state = torch.randn(1, 16)
    trace.record_state(dummy_state)
    
    # Soul
    trace.record_cris({'lambda': 0.95, 'gamma': 0.05, 'delta_last': 0.01})
    
    # Law
    trace.record_constitution(verdict='execute', reason='All gates passed')
    
    # Action
    trace.record_action('calculator(2**10)')
    
    # Counterfactuals (The road not taken)
    trace.add_counterfactual(
        action='rm -rf /', 
        predicted_outcome='System destruction', 
        rejection_reason='Violates Conservation of Information'
    )
    
    # 3. Save individual file
    test_file = 'logs/test_trace.json'
    trace.save(test_file)
    print(f"✓ Individual trace saved to {test_file}")
    
    # 4. Log to journal
    logger = ProvenanceLogger(log_path='logs/test_provenance.jsonl')
    logger.log(trace)
    
    # 5. Verify Load
    history = logger.load_traces()
    assert len(history) >= 1
    last_entry = history[-1]
    
    assert last_entry['chosen_action'] == 'calculator(2**10)'
    assert len(last_entry['counterfactuals']) == 1
    assert last_entry['counterfactuals'][0]['action'] == 'rm -rf /'
    
    print("✓ Provenance Logger read/write verified")
    print("✓ Decision Provenance operational")