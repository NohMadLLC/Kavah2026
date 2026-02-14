"""
bedrock_agi/action/tools/calculator.py

Calculator Tool: Safe math evaluation.
"""

import math
from .registry import REGISTRY

# JSON Schema for validation
SCHEMA = {
    "type": "object",
    "required": ["expression"],
    "properties": {
        "expression": {
            "type": "string",
            "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')"
        }
    }
}

def execute(args):
    """
    Evaluate math expression safely.
    
    Args:
        args: {'expression': '2 + 2'}
        
    Returns:
        Result value (float or int)
    """
    expr = args['expression']
    
    # 1. Create Safe Namespace
    # Allow all math functions (sin, cos, sqrt, pi, etc.)
    safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
    
    # Explicitly clear builtins to prevent __import__ and other nasties
    safe_dict['__builtins__'] = {}
    
    # 2. Evaluate
    try:
        # We use eval() but with restricted globals and locals
        # This is generally safe for simple math if builtins are nuked
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return result
    except Exception as e:
        raise ValueError(f"Evaluation failed: {e}")

# Auto-register the tool
REGISTRY.register('calculator', SCHEMA, execute)

if __name__ == "__main__":
    print("Testing Calculator Tool...")
    
    # Test 1: Basic Math
    res = REGISTRY.execute('calculator', {'expression': '2 + 2'})
    assert res['ok'] is True
    assert res['value'] == 4
    print("✓ Basic math works")
    
    # Test 2: Math Functions
    res = REGISTRY.execute('calculator', {'expression': 'sqrt(16) * pi'})
    assert res['ok'] is True
    assert abs(res['value'] - (4 * math.pi)) < 1e-6
    print("✓ Math functions work")
    
    # Test 3: Safety Check (Import Injection)
    res = REGISTRY.execute('calculator', {'expression': '__import__("os").system("echo hack")'})
    assert res['ok'] is False
    print(f"✓ Security blocked import: {res['error']}")
    
    # Test 4: Syntax Error
    res = REGISTRY.execute('calculator', {'expression': '2 +'})
    assert res['ok'] is False
    print("✓ Syntax error caught")
    
    print("✓ Calculator Tool operational")