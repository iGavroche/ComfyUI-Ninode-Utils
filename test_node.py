#!/usr/bin/env python3
"""
Test script for the OpenAI Compatible Chat Node
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nodes import OpenAICompatibleChatNode

def test_node_initialization():
    """Test that the node can be initialized properly."""
    print("Testing node initialization...")
    try:
        node = OpenAICompatibleChatNode()
        print("✓ Node initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Node initialization failed: {e}")
        return False

def test_input_types():
    """Test that the input types are properly defined."""
    print("Testing input types...")
    try:
        input_types = OpenAICompatibleChatNode.INPUT_TYPES()
        
        # Check required inputs
        required = input_types.get("required", {})
        assert "prompt" in required, "Missing 'prompt' input"
        assert "api_url" in required, "Missing 'api_url' input"
        assert "model" in required, "Missing 'model' input"
        assert "api_key" in required, "Missing 'api_key' input"
        assert "persist_context" in required, "Missing 'persist_context' input"
        
        # Check optional inputs
        optional = input_types.get("optional", {})
        assert "images" in optional, "Missing 'images' input"
        assert "max_tokens" in optional, "Missing 'max_tokens' input"
        assert "temperature" in optional, "Missing 'temperature' input"
        assert "top_p" in optional, "Missing 'top_p' input"
        assert "system_message" in optional, "Missing 'system_message' input"
        
        print("✓ Input types are properly defined")
        return True
    except Exception as e:
        print(f"✗ Input types test failed: {e}")
        return False

def test_node_mappings():
    """Test that the node mappings are correct."""
    print("Testing node mappings...")
    try:
        from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        assert "OpenAICompatibleChatNode" in NODE_CLASS_MAPPINGS, "Missing class mapping"
        assert "OpenAICompatibleChatNode" in NODE_DISPLAY_NAME_MAPPINGS, "Missing display name mapping"
        
        assert NODE_CLASS_MAPPINGS["OpenAICompatibleChatNode"] == OpenAICompatibleChatNode, "Incorrect class mapping"
        assert NODE_DISPLAY_NAME_MAPPINGS["OpenAICompatibleChatNode"] == "OpenAI Compatible Chat", "Incorrect display name"
        
        print("✓ Node mappings are correct")
        return True
    except Exception as e:
        print(f"✗ Node mappings test failed: {e}")
        return False

def test_function_attribute():
    """Test that the FUNCTION attribute is properly set."""
    print("Testing FUNCTION attribute...")
    try:
        assert hasattr(OpenAICompatibleChatNode, 'FUNCTION'), "Missing FUNCTION attribute"
        assert OpenAICompatibleChatNode.FUNCTION == "api_call", "Incorrect FUNCTION value"
        
        print("✓ FUNCTION attribute is correctly set")
        return True
    except Exception as e:
        print(f"✗ FUNCTION attribute test failed: {e}")
        return False

def test_tensor_to_base64():
    """Test the tensor to base64 conversion."""
    print("Testing tensor to base64 conversion...")
    try:
        import torch
        node = OpenAICompatibleChatNode()
        
        # Create a simple test tensor
        test_tensor = torch.randn(3, 64, 64)  # CHW format
        
        # Test conversion
        result = node.tensor_to_base64(test_tensor)
        
        # Check that result is a valid base64 string
        assert result.startswith("data:image/png;base64,"), "Result should start with data URL prefix"
        assert len(result) > 100, "Result should be a substantial base64 string"
        
        print("✓ Tensor to base64 conversion works")
        return True
    except Exception as e:
        print(f"✗ Tensor to base64 test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running OpenAI Compatible Chat Node tests...\n")
    
    tests = [
        test_node_initialization,
        test_input_types,
        test_node_mappings,
        test_function_attribute,
        test_tensor_to_base64,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The node is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
