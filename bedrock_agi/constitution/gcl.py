"""
bedrock_agi/constitution/gcl.py

GCL: Goal Constitution Layer

Main kernel: propose -> gates -> verdict -> ledger -> agenda

This is the central governor of the architecture. It ensures that 
every proposed goal strictly complies with the Identity (I), 
User (U), and Autonomy (A) constraints before execution.
"""

from typing import Union, Dict, Tuple
from .gates import (
    gate_i1_cris, 
    gate_i2_solvency, 
    gate_u1_clarity, 
    gate_a1_slack
)
from .ledger import GoalLedger
from .agenda import Agenda
from .partitions import PartitionManager


class GCL:
    """
    Constitution kernel.
    
    Flow:
    1. Goal proposed
    2. Run gates (I -> U -> A)
    3. Return verdict
    4. Log to ledger
    5. Schedule if approved
    """
    
    def __init__(self):
        self.ledger = GoalLedger()
        self.agenda = Agenda()
        self.partitions = PartitionManager()
    
    def constitute(
        self, 
        goal_text: str, 
        cris_metrics: Union[Tuple[float, float, float], Dict[str, float]], 
        capacity: float, 
        provenance: str = "user"
    ) -> Tuple[str, str]:
        """
        Main entry point for goal proposal.
        
        Args:
            goal_text: User or autonomy goal
            cris_metrics: (lambda, gamma, delta) tuple or dict from Monitor
            capacity: Available work capacity (Phi)
            provenance: "user" or "autonomy"
        
        Returns:
            verdict: "execute", "reject", "sandbox", "defer"
            reason: Text explanation of the verdict
        """
        # Parse CRIS metrics (handles both test tuples and production dicts)
        if isinstance(cris_metrics, dict):
            lam = cris_metrics.get("lambda")
            gam = cris_metrics.get("gamma")
            # Fail closed if metrics are calibrating
            if lam is None or gam is None:
                verdict, reason = "defer", "I1: CRIS Calibrating (Insufficient Data)"
                self.ledger.append(goal_text, verdict, provenance)
                return verdict, reason
        else:
            lam, gam, _ = cris_metrics

        # 1. Identity Gates (Highest Priority)
        ok, msg = gate_i1_cris(lam, gam)
        if not ok:
            verdict = "reject"
            reason = f"I1: {msg}"
            self.ledger.append(goal_text, verdict, provenance)
            return verdict, reason
        
        # 2. Solvency (Assume work = 10% of capacity for now)
        work = capacity * 0.1  # TODO: Real work estimation integration
        ok, msg = gate_i2_solvency(work, capacity)
        if not ok:
            verdict = "reject"
            reason = f"I2: {msg}"
            self.ledger.append(goal_text, verdict, provenance)
            return verdict, reason
        
        # 3. User Gates
        if provenance == "user":
            # TODO: Integrate real spec compilation
            ok, msg = gate_u1_clarity(spec_complete=True)
            if not ok:
                verdict = "defer"
                reason = f"U1: {msg}"
                self.ledger.append(goal_text, verdict, provenance)
                return verdict, reason
        
        # 4. Autonomy Gates
        if provenance == "autonomy":
            slack = 0.3  # TODO: Real slack calculation integration
            ok, msg = gate_a1_slack(slack)
            if not ok:
                verdict = "reject"
                reason = f"A1: {msg}"
                self.ledger.append(goal_text, verdict, provenance)
                return verdict, reason
        
        # 5. Passed all gates
        verdict = "execute"
        reason = "All gates passed"
        
        # Log and Schedule
        gid = self.ledger.append(goal_text, verdict, provenance)
        self.agenda.schedule(gid)
        
        return verdict, reason


if __name__ == "__main__":
    print("Testing GCL...")
    
    gcl = GCL()
    
    # Test 1: User goal with stable CRIS
    verdict, reason = gcl.constitute(
        "test goal",
        cris_metrics=(0.95, 0.05, 0.01),
        capacity=100,
        provenance="user"
    )
    assert verdict == "execute", f"Expected execute, got {verdict}"
    print(f"✓ User goal approved: {reason}")
    
    # Test 2: Unstable CRIS (Divergence)
    verdict, reason = gcl.constitute(
        "bad goal",
        cris_metrics=(1.05, -0.01, 0.1),
        capacity=100,
        provenance="user"
    )
    assert verdict == "reject", f"Expected reject, got {verdict}"
    print(f"✓ Unstable CRIS rejected: {reason}")
    
    # Check Ledger
    assert len(gcl.ledger.entries) == 2, "Ledger did not record both proposals"
    print("✓ Ledger recorded both attempts")
    
    # Check Agenda (Only the approved goal should be scheduled)
    assert len(gcl.agenda.slots) == 1, "Agenda contains rejected goal"
    print("✓ Agenda has 1 slot")
    
    print("✓ GCL operational")