"""
bedrock_agi/action/tools/registry.py

Tool Registry: Manage available tools with schema validation.
"""

import json
from typing import Dict, Callable, Any, Tuple

class ToolRegistry:
    """
    Central registry of available tools.
    
    Each tool has:
    - name: Unique identifier
    - schema: JSON schema for arguments
    - execute: Function to run the tool
    """
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        
    def register(self, name: str, schema: dict, execute: Callable):
        """
        Register a tool.
        
        Args:
            name: Tool name
            schema: JSON schema dict (defines inputs)
            execute: Callable function receiving args dict
        """
        self.tools[name] = {
            'name': name,
            'schema': schema,
            'execute': execute
        }
        
    def get(self, name: str):
        """Get tool definition by name."""
        return self.tools.get(name)
        
    def list_tools(self):
        """List all registered tool names."""
        return list(self.tools.keys())
        
    def validate_args(self, name: str, args: dict) -> Tuple[bool, str]:
        """
        Validate arguments against schema.
        
        Returns:
            (valid: bool, error: str)
        """
        if name not in self.tools:
            return False, f"Unknown tool: {name}"
            
        tool_def = self.tools[name]
        schema = tool_def['schema']
        
        # Simple validation: Check required fields
        # Ideally, use jsonschema library for full validation
        required = schema.get('required', [])
        for field in required:
            if field not in args:
                return False, f"Missing required argument: '{field}'"
        
        return True, "OK"
        
    def execute(self, name: str, args: dict) -> Dict[str, Any]:
        """
        Execute tool with arguments.
        
        Returns:
            Result dict with 'ok' and 'value' or 'error'
        """
        # 1. Validate
        valid, error = self.validate_args(name, args)
        if not valid:
            return {'ok': False, 'error': error}
            
        # 2. Execute
        try:
            tool_def = self.tools[name]
            func = tool_def['execute']
            
            # Execute the function
            result = func(args)
            
            return {'ok': True, 'value': result}
            
        except Exception as e:
            return {'ok': False, 'error': f"Execution failed: {str(e)}"}

# Global registry instance
REGISTRY = ToolRegistry()

if __name__ == "__main__":
    print("Testing ToolRegistry...")
    
    # 1. Register a test tool
    def test_tool(args):
        return args['x'] * 2
        
    schema = {
        'type': 'object',
        'required': ['x'],
        'properties': {'x': {'type': 'number'}}
    }
    
    REGISTRY.register('test_double', schema, test_tool)
    
    # 2. Test Execution (Success)
    result = REGISTRY.execute('test_double', {'x': 5})
    assert result['ok'] is True
    assert result['value'] == 10
    print("✓ Tool execution works")
    
    # 3. Test Validation (Failure)
    result = REGISTRY.execute('test_double', {})