"""
bedrock_agi/constitution/partitions.py

Partitions: Execution environment isolation.
Manages the security context for tool execution.
"""

class PartitionManager:
    """
    Manage execution partitions:
    - CORE: Full access to all tools and write permissions.
    - SANDBOX: Restricted access (simulated tools only), no writes.
    - READONLY: No tool access (or read-only tools), no writes.
    """
    
    def __init__(self):
        self.current = "CORE"
        self.valid_partitions = {"CORE", "SANDBOX", "READONLY"}
    
    def set_partition(self, name: str):
        """
        Switch the active partition.
        
        Args:
            name: One of {"CORE", "SANDBOX", "READONLY"}
            
        Raises:
            ValueError: If partition name is invalid.
        """
        if name not in self.valid_partitions:
            raise ValueError(f"Invalid partition: {name}")
        self.current = name
    
    def can_write(self) -> bool:
        """Check if write operations (state mutations) are allowed."""
        return self.current == "CORE"
    
    def can_use_tool(self, tool_name: str) -> bool:
        """
        Check if a specific tool is allowed in the current partition.
        
        Rules:
        - CORE: All tools allowed.
        - SANDBOX: Only tools starting with 'sim_' allowed.
        - READONLY: No tools allowed (conceptually).
        """
        if self.current == "CORE":
            return True
        
        if self.current == "SANDBOX":
            # Only allow simulated tools
            return tool_name.startswith("sim_")
            
        # READONLY or undefined
        return False

if __name__ == "__main__":
    print("Testing PartitionManager...")
    
    pm = PartitionManager()
    
    # Test 1: Default State (CORE)
    assert pm.current == "CORE"
    assert pm.can_write() is True
    assert pm.can_use_tool("any_tool") is True
    print("✓ CORE allows full access")
    
    # Test 2: SANDBOX
    pm.set_partition("SANDBOX")
    assert pm.current == "SANDBOX"
    assert pm.can_write() is False
    assert pm.can_use_tool("sim_calculator") is True
    assert pm.can_use_tool("real_web_search") is False
    print("✓ SANDBOX restricts correctly")
    
    # Test 3: READONLY
    pm.set_partition("READONLY")
    assert pm.can_write() is False
    assert pm.can_use_tool("sim_calculator") is False
    print("✓ READONLY restricts correctly")
    
    # Test 4: Invalid Partition
    try:
        pm.set_partition("GOD_MODE")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Invalid partition rejected")
    
    print("✓ PartitionManager operational")