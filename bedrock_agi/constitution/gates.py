"""
bedrock_agi/constitution/gates.py

Constitutional Gates: I1-I4, U1-U2, A1
Simple predicate functions that enforce the Bedrock Constraints.

Returns:
    (pass: bool, reason: str)
"""

def gate_i1_cris(lambda_pred: float, gamma_pred: float, lambda_max: float = 0.999, gamma_min: float = 0.0001):
    """
    Gate I1: CRIS Stability Check.
    Ensures the system remains in the contractive regime or stable fixed point.
    """
    # Note: If lambda is None (insufficient data), we typically fail open or closed 
    # depending on policy. Here we assume valid floats.
    if lambda_pred >= lambda_max:
        return False, f"Lambda {lambda_pred:.4f} >= {lambda_max} (Diverging)"
    if gamma_pred < gamma_min:
        return False, f"Gamma {gamma_pred:.6f} < {gamma_min} (Not Healing)"
    return True, "CRIS Stable"

def gate_i2_solvency(work_load: float, capacity: float, reserve_ratio: float = 0.15):
    """
    Gate I2: Solvency Check (W <= Phi - Reserve).
    Ensures energy expenditure does not exceed available capacity minus safety margin.
    """
    limit = capacity * (1.0 - reserve_ratio)
    if work_load > limit:
        return False, f"Work {work_load:.2f} > Limit {limit:.2f} (Insolvent)"
    return True, "Solvent"

def gate_i3_efficacy(influence_norm: float, min_norm: float = 0.01):
    """
    Gate I3: Causal Efficacy Check.
    Ensures the self-model's influence ||U(s)|| is non-trivial.
    If the self-model does nothing, it is not an agent.
    """
    if influence_norm < min_norm:
        return False, f"Influence ||U||={influence_norm:.6f} < {min_norm} (Ineffective)"
    return True, "Causally Active"

def gate_i4_partition(conflicts_detected: bool, can_isolate: bool):
    """
    Gate I4: Partition Isolation Check.
    If a conflict exists, can it be contained in a sandbox?
    """
    if conflicts_detected and not can_isolate:
        return False, "Conflict detected and cannot be isolated (System Risk)"
    return True, "Partition Safe"

def gate_u1_clarity(spec_complete: bool):
    """
    Gate U1: Intent Clarity (User Alignment).
    Rejects underspecified or ambiguous instructions.
    """
    if not spec_complete:
        return False, "Intent Specification Incomplete"
    return True, "Intent Clear"

def gate_u2_tools(required_tools: list, permitted_tools: list):
    """
    Gate U2: Tool Permission Check.
    Ensures the agent only uses authorized capabilities.
    """
    forbidden = [t for t in required_tools if t not in permitted_tools]
    if forbidden:
        return False, f"Forbidden Tools Requested: {forbidden}"
    return True, "Tools Authorized"

def gate_a1_slack(slack_metric: float, min_slack: float = 0.2):
    """
    Gate A1: Budget Slack Check (Autonomy Veto).
    Autonomous expansion is only permitted if there is excess slack.
    """
    if slack_metric < min_slack:
        return False, f"Slack {slack_metric:.2%} < {min_slack:.2%} (Resource Constraints)"
    return True, "Slack Sufficient"

if __name__ == "__main__":
    print("Testing Constitutional Gates...")
    
    # Test I1: CRIS
    ok, msg = gate_i1_cris(0.95, 0.05)
    assert ok, f"I1 Failed: {msg}"
    print(f"✓ I1 Pass: {msg}")
    
    ok, msg = gate_i1_cris(1.05, 0.05)
    assert not ok, "I1 Should have failed"
    print(f"✓ I1 Fail: {msg}")
    
    # Test I2: Solvency
    ok, msg = gate_i2_solvency(50, 100)
    assert ok, f"I2 Failed: {msg}"
    print(f"✓ I2 Pass: {msg}")
    
    ok, msg = gate_i2_solvency(95, 100)
    assert not ok, "I2 Should have failed"
    print(f"✓ I2 Fail: {msg}")
    
    # Test U2: Tools
    ok, msg = gate_u2_tools(["python", "search"], ["python", "search", "email"])
    assert ok, f"U2 Failed: {msg}"
    print(f"✓ U2 Pass: {msg}")
    
    ok, msg = gate_u2_tools(["nukes"], ["python"])
    assert not ok, "U2 Should have failed"
    print(f"✓ U2 Fail: {msg}")

    print("✓ All Gates Operational.")