"""
bedrock_agi/action/tools/code_exec.py

Code Execution Tool: Sandboxed Python execution.
Captures stdout/stderr and returns the local scope 'result'.
"""

import sys
import io
import contextlib
from typing import Dict, Any
from .registry import REGISTRY

# Schema for validation
SCHEMA = {
    "type": "object",
    "required": ["code"],
    "properties": {
        "code": {
            "type": "string",
            "description": "Python code to execute. Assign final value to variable 'result'."
        },
        "timeout": {
            "type": "number",
            "description": "Execution timeout in seconds (not strictly enforced in this simple version)",
            "default": 5
        }
    }
}

def execute(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Python code in sandbox.
    
    Args:
        args: {'code': 'print("hi"); result = 2+2', 'timeout': 5}
        
    Returns:
        {'stdout': '...', 'stderr': '...', 'result': ...}
    """
    code = args['code']
    
    # Capture stdout/stderr
    # We use io.StringIO to capture the streams in memory
    capture_stdout = io.StringIO()
    capture_stderr = io.StringIO()
    
    result_val = None
    
    # Minimal Safe Builtins
    # We explicitly allow 'print' so the agent can debug.
    # We disable '__import__' and 'open' via the empty __builtins__, 
    # but we must re-add essential types for basic logic to work.
    safe_builtins = {
        'print': print,
        'range': range,
        'len': len,
        'int': int,
        'float': float,
        'str': str,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
        'bool': bool,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,
        'enumerate': enumerate,
        'zip': zip,
        'isinstance': isinstance,
    }
    
    namespace = {
        '__builtins__': safe_builtins
    }
    
    try:
        # Redirect standard streams
        with contextlib.redirect_stdout(capture_stdout), contextlib.redirect_stderr(capture_stderr):
            # Execute code in restricted namespace
            exec(code, namespace, namespace)
            
        # Extract 'result' variable if it exists
        result_val = namespace.get('result', None)
        
    except Exception as e:
        # Capture runtime errors into stderr
        print(f"Runtime Error: {e}", file=capture_stderr)
    
    return {
        'stdout': capture_stdout.getvalue(),
        'stderr': capture_stderr.getvalue(),
        'result': result_val
    }

# Auto-register
REGISTRY.register('code_exec', SCHEMA, execute)

if __name__ == "__main__":
    print("Testing CodeExec Tool...")
    
    # Test 1: Basic Logic
    res = REGISTRY.execute('code_exec', {'code': 'result = 2 + 2'})
    assert res['ok']
    assert res['value']['result'] == 4
    print("✓ Basic calculation works")
    
    # Test 2: Stdout Capture
    res = REGISTRY.execute('code_exec', {'code': 'print("Hello World")'})
    assert res['ok']
    assert "Hello World" in res['value']['stdout']
    print("✓ Stdout capture works")
    
    # Test 3: Safety (Import prevention)
    res = REGISTRY.execute('code_exec', {'code': 'import os'})
    # This should fail and print to stderr (captured) or raise exception caught by wrapper
    # Since exec() is inside try/except in execute(), it returns OK but stderr has error
    # Or REGISTRY catches it.
    # In this implementation, 'import os' raises ImportError because __import__ is missing.
    # The execute() function catches it and prints to stderr.
    assert "Error" in res['value']['stderr'] or "Error" in str(res.get('error'))
    print("✓ Sandbox prevents imports")
    
    print("✓ CodeExec Tool operational")