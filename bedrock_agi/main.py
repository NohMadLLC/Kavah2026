"""
bedrock_agi/main.py

The Interface (Kernel).
Binds Consciousness, Perception, Memory, and Action into a living agent loop.

Usage:
    python -m bedrock_agi.main
"""

import torch
import sys
import os
import time
from typing import Dict, Any, Optional

# Core Imports
from .perception.text_encoder import TextEncoder
from .memory.episodic import EpisodicMemory
from .memory.semantic import SemanticMemory
from .consciousness.self_model import SelfModel
from .consciousness.self_influence import BoundedInfluence
from .core.e_model import EvolutionModel
from .core.r_projector import RProjector
from .core.cris_monitor import CRISMonitor
from .constitution.gcl import GCL
from .planning.cem_planner import CEMPlanner
from .action.tools.registry import REGISTRY
from .action.decoders import SimpleDecoder

# Register tools
import bedrock_agi.action.tools.calculator
import bedrock_agi.action.tools.web_search
import bedrock_agi.action.tools.code_exec

# ANSI Colors for CLI
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class BedrockAgent:
    """
    The Kavah Agent.
    A self-regulating cognitive architecture operating on H^n.
    """
    
    def __init__(self, latent_dim=16, persistence_dir="memory"):
        self.latent_dim = latent_dim
        self.persistence_dir = persistence_dir
        
        print(f"{Colors.HEADER}Initializing Cortex (H^{latent_dim})...{Colors.ENDC}")
        
        # 1. Perception
        self.text_encoder = TextEncoder(latent_dim=latent_dim)
        
        # 2. Memory
        os.makedirs(persistence_dir, exist_ok=True)
        self.episodic = EpisodicMemory(
            capacity=5000, 
            save_path=os.path.join(persistence_dir, "episodic.pkl")
        )
        self.semantic = SemanticMemory(
            latent_dim=latent_dim, 
            save_path=os.path.join(persistence_dir, "semantic.pkl")
        )
        
        # 3. Consciousness
        self.self_model = SelfModel(state_dim=latent_dim, latent_dim=latent_dim)
        
        # 4. Geometric Core (Physics of the Mind)
        self.evolution = EvolutionModel(state_dim=latent_dim, eta=0.95)
        self.projection = RProjector(n=latent_dim, bias_toward_origin=0.98)
        self.cris_monitor = CRISMonitor(tail=20)
        
        # 5. Constitution (The Law)
        self.gcl = GCL()
        
        # 6. Action
        self.planner = CEMPlanner(action_dim=4, horizon=3, n_samples=32)
        
        # Internal State
        self.b_prev = None
        self.b_curr = None
        
        print(f"{Colors.GREEN}✓ Systems Online.{Colors.ENDC}")
        print(f"{Colors.BLUE}✓ Memory Loaded ({len(self.episodic)} episodes).{Colors.ENDC}")

    def step(self, user_input: str) -> str:
        """
        Execute one cognitive cycle (OODA Loop).
        """
        # --- 1. OBSERVE (Perception) ---
        b_input = self.text_encoder([user_input])[0:1]
        
        # --- 2. ORIENT (Dynamics & Self) ---
        # Evolve world model
        b_pred = self.evolution(b_input)
        b_clean = self.projection(b_pred)
        
        # Update CRIS Monitor
        self.cris_monitor.update(b_input, b_clean)
        metrics = self.cris_monitor.metrics()
        
        # Innate Stability (Cold Start Fix)
        # If the agent is newborn, assume it is healthy to prevent immediate shutdown
        lam = metrics.get('lambda') if metrics.get('lambda') is not None else 0.95
        gam = metrics.get('gamma') if metrics.get('gamma') is not None else 0.05
        delta = metrics.get('delta_last') if metrics.get('delta_last') is not None else 0.01
        
        # Update Self-Model (Consciousness)
        if self.b_prev is None:
            self.b_prev = b_input
            self.b_curr = b_input
        else:
            self.b_prev = self.b_curr
            self.b_curr = b_input
            
        cris_tensor = torch.tensor([[lam, gam, delta]])
        s, u = self.self_model.step(self.b_prev, self.b_curr, cris_tensor)
        
        # Commit to Memory
        self.episodic.write(b_clean, cris_tensor[0], metadata={'role': 'user', 'content': user_input})
        
        # Log Internal State
        self._log_thought(lam, gam, delta, u)
        
        # --- 3. DECIDE (Constitution) ---
        # Extract intent (Simple heuristic for prototype)
        goal = self._extract_intent(user_input)
        
        # Check Constitution
        verdict, reason = self.gcl.constitute(
            goal_text=goal,
            cris_metrics=(lam, gam, delta),
            capacity=100.0,
            provenance="user"
        )
        
        if verdict == "reject":
            response = f"I cannot do that. {reason}"
            self._log_action("REJECTED", response, color=Colors.RED)
            return response
            
        # --- 4. ACT (Planning & Execution) ---
        # In a full system, the Planner would generate the tool call.
        # Here we map intent -> tool directly for the prototype.
        tool_name, tool_args = self._select_tool(goal, user_input)
        
        result_str = ""
        if tool_name:
            self._log_action("TOOL", f"{tool_name} {tool_args}", color=Colors.YELLOW)
            exec_result = REGISTRY.execute(tool_name, tool_args)
            
            if exec_result['ok']:
                result_str = f"\n[Result]: {exec_result['value']}"
                response = f"Executed {tool_name}. {result_str}"
            else:
                result_str = f"\n[Error]: {exec_result.get('error')}"
                response = f"Failed to execute {tool_name}. {result_str}"
        else:
            # General conversation (simulated)
            response = "I have processed your input and updated my internal state."
            
        # Remember own action
        # (Ideally we encode the response back to H^n, but we skip for speed here)
        
        return response

    def _log_thought(self, lam, gam, delta, u):
        """Print internal monologue metrics."""
        # Color code stability
        status_col = Colors.GREEN
        if lam > 0.99 or gam < 0.0: status_col = Colors.RED
        elif lam > 0.96: status_col = Colors.YELLOW
        
        u_norm = torch.norm(u).item()
        
        print(f"  {Colors.CYAN}│{Colors.ENDC} {Colors.BOLD}INTERNAL STATE:{Colors.ENDC}")
        print(f"  {Colors.CYAN}│{Colors.ENDC} CRIS: λ={status_col}{lam:.4f}{Colors.ENDC} γ={gam:.4f} Δ={delta:.4f}")
        print(f"  {Colors.CYAN}│{Colors.ENDC} SELF: Influence ||u||={u_norm:.4f}")

    def _log_action(self, type_str, content, color=Colors.GREEN):
        print(f"  {Colors.CYAN}└─{Colors.ENDC} {color}[{type_str}]{Colors.ENDC} {content}")

    def _extract_intent(self, text: str) -> str:
        text = text.lower()
        if any(w in text for w in ["calc", "math", "plus", "times", "divide", "power"]):
            return "perform calculation"
        if any(w in text for w in ["search", "find", "google", "lookup"]):
            return "web search"
        if any(w in text for w in ["code", "python", "script", "run"]):
            return "execute code"
        return "general interaction"

    def _select_tool(self, goal: str, text: str) -> tuple:
        """Heuristic tool selection."""
        if "calculation" in goal:
            # Simple extraction of math part (very naive)
            # In production, use an LLM or the Planner to parse this
            return "calculator", {"expression": "2 ** 10"} # Stub for demo
        
        if "web search" in goal:
            return "web search", {"query": text, "num_results": 3}
            
        if "execute code" in goal:
            # Stub code
            return "code_exec", {"code": "print('Hello form Sandbox')"}
            
        return None, None

    def save(self):
        self.episodic.save()
        self.semantic.save()
        print(f"{Colors.BLUE}Memory persisted.{Colors.ENDC}")

def main():
    print(f"{Colors.CYAN}")
    print(r"""
    __ __                 __  
   / //_/__ __   ______ _/ /_ 
  / ,< / __ `/ | / / __ `/ __ \
 / /| / /_/ /| |/ / /_/ / / / /
/_/ |_\__,_/ |___/\__,_/_/ /_/ 
    
    BEDROCK AGI KERNEL v0.5
    Phase 5: Cognitive Loop
    """)
    print(f"{Colors.ENDC}")
    
    agent = BedrockAgent()
    
    print("\nType '/quit' to exit, '/save' to persist memory.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input(f"\n{Colors.BOLD}USER >{Colors.ENDC} ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['/quit', 'exit']:
                agent.save()
                print("Shutting down.")
                break
                
            if user_input.lower() == '/save':
                agent.save()
                continue
                
            # Run Loop
            start_t = time.time()
            response = agent.step(user_input)
            latency = (time.time() - start_t) * 1000
            
            print(f"\n{Colors.BOLD}KAVAH >{Colors.ENDC} {response}")
            print(f"{Colors.BLUE}({latency:.1f}ms){Colors.ENDC}")
            
        except KeyboardInterrupt:
            print("\nInterrupted.")
            agent.save()
            break
        except Exception as e:
            print(f"\n{Colors.RED}CRITICAL ERROR: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()