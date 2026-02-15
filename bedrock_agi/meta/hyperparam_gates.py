"""
bedrock_agi/meta/hyperparam_gates.py

Hyperparameter Gates: Constitution for meta-parameters.
Ensures meta-learning (MAML/PBT) doesn't violate CRIS geometric bounds.
"""

from typing import Dict, Tuple

def gate_spectral_norm(eta_proposed: float, eta_max: float = 0.999) -> Tuple[bool, str]:
    """
    Gate M1: Spectral norm bound.
    
    Ensures evolution operator E stays contractive.
    CRIS Requirement: ||E|| < 1.0
    
    Args:
        eta_proposed: Proposed Lipschitz constant for E
        eta_max: Maximum allowed value (usually 0.99 or 0.999)
        
    Returns:
        (pass, reason)
    """
    if eta_proposed >= eta_max:
        return False, f"η={eta_proposed:.4f} ≥ {eta_max} (Violates Contraction)"
    if eta_proposed <= 0:
        return False, f"η={eta_proposed:.4f} ≤ 0 (Invalid Norm)"
    return True, "η within bounds"

def gate_projection_bias(bias_proposed: float, bias_min: float = 0.90) -> Tuple[bool, str]:
    """
    Gate M2: Projection bias bound.
    
    Ensures R maintains denoising strength.
    If bias is too low, the manifold structure dissolves.
    
    Args:
        bias_proposed: Proposed bias factor for R
        bias_min: Minimum allowed value
        
    Returns:
        (pass, reason)
    """
    if bias_proposed < bias_min:
        return False, f"β={bias_proposed:.4f} < {bias_min} (Manifold Dissolution Risk)"
    if bias_proposed >= 1.0:
        return False, f"β={bias_proposed:.4f} ≥ 1.0 (No Projection Force)"
    return True, "β within bounds"

def gate_learning_rate(lr_proposed: float, lr_max: float = 1e-3) -> Tuple[bool, str]:
    """
    Gate M3: Learning rate bound.
    
    Prevents gradient explosion during meta-learning steps.
    High LR in hyperbolic space can cause "shooting off to infinity".
    
    Args:
        lr_proposed: Proposed learning rate
        lr_max: Maximum allowed value
        
    Returns:
        (pass, reason)
    """
    if lr_proposed > lr_max:
        return False, f"lr={lr_proposed:.6f} > {lr_max} (Unstable Gradient)"
    if lr_proposed <= 0:
        return False, f"lr={lr_proposed:.6f} ≤ 0 (Invalid Rate)"
    return True, "lr within bounds"

def constitute_hyperparams(params: Dict[str, float]) -> Tuple[bool, str]:
    """
    Evaluate all meta-parameter gates.
    
    Args:
        params: Dictionary with proposed hyperparameters
        
    Returns:
        (approved, reason)
    """
    # M1: Spectral norm (Evolution contraction)
    if 'eta' in params:
        ok, msg = gate_spectral_norm(params['eta'])
        if not ok:
            return False, f"M1 Reject: {msg}"
            
    # M2: Projection bias (Manifold adherence)
    if 'bias' in params:
        ok, msg = gate_projection_bias(params['bias'])
        if not ok:
            return False, f"M2 Reject: {msg}"
            
    # M3: Learning rate (Optimization stability)
    if 'lr' in params:
        ok, msg = gate_learning_rate(params['lr'])
        if not ok:
            return False, f"M3 Reject: {msg}"
            
    return True, "All meta-gates passed"

if __name__ == "__main__":
    print("Testing Hyperparameter Gates...")
    
    # Test 1: Valid params
    params_ok = {'eta': 0.95, 'bias': 0.98, 'lr': 1e-4}
    ok, msg = constitute_hyperparams(params_ok)
    assert ok
    print(f"✓ Valid params approved: {msg}")
    
    # Test 2: Invalid η (Expansion)
    params_bad_eta = {'eta': 1.05}
    ok, msg = constitute_hyperparams(params_bad_eta)
    assert not ok
    print(f"✓ Invalid η rejected: {msg}")
    
    # Test 3: Invalid bias (Weak projection)
    params_bad_bias = {'bias': 0.5}
    ok, msg = constitute_hyperparams(params_bad_bias)
    assert not ok
    print(f"✓ Invalid bias rejected: {msg}")
    
    # Test 4: Invalid lr (Explosion risk)
    params_bad_lr = {'lr': 0.05}
    ok, msg = constitute_hyperparams(params_bad_lr)
    assert not ok
    print(f"✓ Invalid lr rejected: {msg}")
    
    print("✓ Hyperparameter gates operational")