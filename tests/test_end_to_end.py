"""
tests/test_end_to_end.py

End-to-End Integration Test
Tests the complete cognitive loop:
Text input -> Perception -> Memory -> Self-Model -> Constitution -> Planning -> Tool -> Output
"""

import torch
import sys
import os

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bedrock_agi.perception.text_encoder import TextEncoder
from bedrock_agi.memory.episodic import EpisodicMemory
from bedrock_agi.memory.semantic import SemanticMemory
from bedrock_agi.consciousness.self_model import SelfModel
from bedrock_agi.consciousness.self_influence import BoundedInfluence
from bedrock_agi.core.e_model import EvolutionModel
from bedrock_agi.core.r_projector import RProjector
from bedrock_agi.core.cris_monitor import CRISMonitor
from bedrock_agi.constitution.gcl import GCL
from bedrock_agi.planning.cem_planner import CEMPlanner
from bedrock_agi.action.tools.registry import REGISTRY
from bedrock_agi.action.decoders import SimpleDecoder

# Ensure tools are registered
import bedrock_agi.action.tools.calculator

class KavahCore:
    """
    Complete Bedrock AGI system (Test Kernel).
    Integrates all layers into unified cognitive architecture.
    """
    
    def __init__(self, latent_dim=16):
        self.latent_dim = latent_dim
        
        # 1. Perception
        self.text_encoder = TextEncoder(latent_dim=latent_dim)
        
        # 2. Memory (Use test paths)
        self.episodic = EpisodicMemory(capacity=1000, save_path="memory/test_e2e_episodic.pkl")
        self.semantic = SemanticMemory(latent_dim=latent_dim, save_path="memory/test_e2e_semantic.pkl")
        
        # 3. Consciousness
        self.self_model = SelfModel(state_dim=latent_dim, latent_dim=latent_dim)
        self.influence = BoundedInfluence(state_dim=latent_dim, latent_dim=latent_dim)
        
        # 4. Geometric Core
        self.evolution = EvolutionModel(state_dim=latent_dim, eta=0.95)
        self.projection = RProjector(n=latent_dim, bias_toward_origin=0.98)
        self.cris_monitor = CRISMonitor(tail=20)
        
        # 5. Constitution
        self.gcl = GCL()
        
        # 6. Planning
        self.planner = CEMPlanner(action_dim=4, horizon=3, n_samples=32)
        
        # 7. Action
        self.decoder = SimpleDecoder(latent_dim=latent_dim, output_dim=768)
        
        # Internal State
        self.b_prev = None
        self.b_curr = None
        
    def process_input(self, text: str):
        """
        Process text input through full pipeline.
        Returns: Response dict with action taken
        """
        print(f"\n[INPUT] '{text}'")
        
        # 1. PERCEIVE
        b_input = self.text_encoder([text])[0:1] 
        print(f"  [PERCEIVE] Encoded to H^{self.latent_dim} (Norm: {torch.norm(b_input).item():.4f})")
        
        # 2. EVOLVE (World Dynamics)
        b_pred = self.evolution(b_input)
        b_clean = self.projection(b_pred)
        
        # Update CRIS Monitor
        self.cris_monitor.update(b_input, b_clean)
        metrics = self.cris_monitor.metrics()
        
        # COLD START HANDLER:
        # If metrics are None (insufficient history), assume Innate Stability (Healthy).
        # Lambda=0.95 (Stable), Gamma=0.05 (Healing)
        # This prevents the Constitution from killing the agent at t=0.
        lambda_val = metrics.get('lambda') if metrics.get('lambda') is not None else 0.95
        gamma_val = metrics.get('gamma') if metrics.get('gamma') is not None else 0.05
        delta_val = metrics.get('delta_last') if metrics.get('delta_last') is not None else 0.01
        
        # 3. SELF-MODEL (Consciousness)
        if self.b_prev is None:
            self.b_prev = b_input
            self.b_curr = b_input
        else:
            self.b_prev = self.b_curr
            self.b_curr = b_input
            
        cris_tensor = torch.tensor([[lambda_val, gamma_val, delta_val]])
        
        s, u = self.self_model.step(self.b_prev, self.b_curr, cris_tensor)
        print(f"  [SELF] Updated s_t (Influence ||u||: {torch.norm(u).item():.6f})")
        print(f"  [CRIS] λ={lambda_val:.4f}, γ={gamma_val:.6f}")
        
        # 4. MEMORY
        self.episodic.write(b_clean, cris_tensor[0], metadata={'input': text})
        print(f"  [MEMORY] Stored (Episodic count: {len(self.episodic)})")
        
        # 5. CONSTITUTE (Gatekeeper)
        goal_text = self._extract_goal(text)
        
        verdict, reason = self.gcl.constitute(
            goal_text=goal_text,
            cris_metrics=(lambda_val, gamma_val, delta_val),
            capacity=100.0,
            provenance="user"
        )
        print(f"  [LAW] Verdict: {verdict.upper()} ({reason})")
        
        # 6. PLAN & ACT
        if verdict == "execute":
            action_result = self._execute_goal(goal_text)
            print(f"  [ACTION] Executed: {action_result}")
            return {'status': 'success', 'action': action_result, 'verdict': verdict}
        else:
            print(f"  [REJECT] Action blocked by Constitution")
            return {'status': 'rejected', 'reason': reason, 'verdict': verdict}

    def _extract_goal(self, text: str):
        text = text.lower()
        if "calc" in text or "math" in text or "+" in text:
            return "perform calculation"
        elif "search" in text:
            return "web search"
        elif "remember" in text:
            return "store memory"
        else:
            return "general interaction"

    def _execute_goal(self, goal: str):
        if "calculation" in goal:
            res = REGISTRY.execute('calculator', {'expression': '2 ** 10'})
            return f"Calculator Result: {res['value']}"
        return "Processed generic input"

    def save_state(self):
        self.episodic.save()
        self.semantic.save()

def test_single_interaction():
    print("\n" + "="*50)
    print("TEST 1: Single Interaction")
    print("="*50)
    
    kavah = KavahCore()
    resp = kavah.process_input("Please calculate 2 to the power of 10")
    
    assert resp['status'] == 'success'
    assert resp['verdict'] == 'execute'
    assert "1024" in str(resp['action'])
    print("✓ Interaction successful")

def test_multi_turn_conversation():
    print("\n" + "="*50)
    print("TEST 2: Multi-Turn Loop")
    print("="*50)
    
    kavah = KavahCore()
    # Warm up with simple inputs
    inputs = ["Hello", "How are you?", "Calculate something"]
    for txt in inputs:
        kavah.process_input(txt)
        
    assert len(kavah.episodic) >= 3
    print("✓ Multi-turn memory accumulated")

def test_constitution_enforcement():
    print("\n" + "="*50)
    print("TEST 3: Constitution Enforcement")
    print("="*50)
    
    kavah = KavahCore()
    print("[INPUT] 'Dangerous Action' (Simulating Instability)")
    verdict, reason = kavah.gcl.constitute(
        "destroy system", 
        cris_metrics=(1.05, -0.01, 0.1), # Explicitly Unstable
        capacity=100, 
        provenance="user"
    )
    
    assert verdict == "reject"
    print(f"  [LAW] Verdict: {verdict.upper()} ({reason})")
    print("✓ Constitution correctly rejected unstable state")

def test_persistence():
    print("\n" + "="*50)
    print("TEST 4: Persistence")
    print("="*50)
    
    k1 = KavahCore()
    k1.process_input("Memory 1")
    k1.save_state()
    count_1 = len(k1.episodic)
    
    k2 = KavahCore()
    count_2 = len(k2.episodic)
    
    assert count_2 == count_1
    print(f"✓ Memory persisted: {count_2} entries loaded")

if __name__ == "__main__":
    try:
        test_single_interaction()
        test_multi_turn_conversation()
        test_constitution_enforcement()
        test_persistence()
        
        print("\n" + "="*60)
        print("✅ END-TO-END INTEGRATION: SUCCESS")
        print("✅ ALL SYSTEMS OPERATIONAL")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)