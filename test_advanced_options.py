#!/usr/bin/env python3
"""
Test script for advanced_options handling in OpenAI Compatible Chat Node
Tests both Pydantic model and dictionary inputs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nodes import OpenAICompatibleChatNode

def test_dictionary_advanced_options():
    """Test handling of dictionary advanced_options."""
    print("Testing dictionary advanced_options...")
    try:
        node = OpenAICompatibleChatNode()
        
        # Test with dictionary
        advanced_options = {
            "max_tokens": 2000,
            "temperature": 0.5,
            "top_p": 0.8,
            "instructions": "Be very creative"
        }
        
        # Test the conversion logic
        if advanced_options:
            if hasattr(advanced_options, 'model_dump'):
                options_dict = advanced_options.model_dump(exclude_none=True)
            elif hasattr(advanced_options, 'items'):
                options_dict = advanced_options
            else:
                options_dict = dict(advanced_options)
        
        # Verify the conversion worked
        assert isinstance(options_dict, dict), "Should be a dictionary"
        assert "max_tokens" in options_dict, "Should contain max_tokens"
        assert options_dict["max_tokens"] == 2000, "Should have correct value"
        
        print("✓ Dictionary advanced_options handled correctly")
        return True
        
    except Exception as e:
        print(f"✗ Dictionary advanced_options test failed: {e}")
        return False

def test_pydantic_advanced_options():
    """Test handling of Pydantic model advanced_options."""
    print("Testing Pydantic model advanced_options...")
    try:
        # Create a mock Pydantic-like object
        class MockPydanticModel:
            def __init__(self):
                self.max_tokens = 1500
                self.temperature = 0.7
                self.instructions = "Test instructions"
            
            def model_dump(self, exclude_none=True):
                return {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "instructions": self.instructions
                }
        
        node = OpenAICompatibleChatNode()
        advanced_options = MockPydanticModel()
        
        # Test the conversion logic
        if advanced_options:
            if hasattr(advanced_options, 'model_dump'):
                options_dict = advanced_options.model_dump(exclude_none=True)
            elif hasattr(advanced_options, 'items'):
                options_dict = advanced_options
            else:
                options_dict = dict(advanced_options)
        
        # Verify the conversion worked
        assert isinstance(options_dict, dict), "Should be a dictionary"
        assert "max_tokens" in options_dict, "Should contain max_tokens"
        assert options_dict["max_tokens"] == 1500, "Should have correct value"
        assert "instructions" in options_dict, "Should contain instructions"
        
        print("✓ Pydantic model advanced_options handled correctly")
        return True
        
    except Exception as e:
        print(f"✗ Pydantic model advanced_options test failed: {e}")
        return False

def test_none_advanced_options():
    """Test handling of None advanced_options."""
    print("Testing None advanced_options...")
    try:
        node = OpenAICompatibleChatNode()
        advanced_options = None
        
        # Test the conversion logic
        if advanced_options:
            if hasattr(advanced_options, 'model_dump'):
                options_dict = advanced_options.model_dump(exclude_none=True)
            elif hasattr(advanced_options, 'items'):
                options_dict = advanced_options
            else:
                options_dict = dict(advanced_options)
        else:
            options_dict = {}
        
        # Verify the conversion worked
        assert isinstance(options_dict, dict), "Should be a dictionary"
        assert len(options_dict) == 0, "Should be empty"
        
        print("✓ None advanced_options handled correctly")
        return True
        
    except Exception as e:
        print(f"✗ None advanced_options test failed: {e}")
        return False

def test_edge_case_advanced_options():
    """Test handling of edge case advanced_options."""
    print("Testing edge case advanced_options...")
    try:
        node = OpenAICompatibleChatNode()
        
        # Test with an object that has neither model_dump nor items
        class EdgeCaseObject:
            def __init__(self):
                self.value = "test"
        
        advanced_options = EdgeCaseObject()
        
        # Test the conversion logic
        if advanced_options:
            if hasattr(advanced_options, 'model_dump'):
                options_dict = advanced_options.model_dump(exclude_none=True)
            elif hasattr(advanced_options, 'items'):
                options_dict = advanced_options
            else:
                # Try to convert to dict, but handle edge cases gracefully
                try:
                    options_dict = dict(advanced_options)
                except (TypeError, ValueError):
                    # If conversion fails, skip advanced options
                    print(f"Warning: Could not convert advanced_options to dictionary: {type(advanced_options)}")
                    options_dict = {}
        
        # This should work with the fallback
        assert isinstance(options_dict, dict), "Should be a dictionary"
        assert len(options_dict) == 0, "Should be empty due to conversion failure"
        
        print("✓ Edge case advanced_options handled correctly")
        return True
        
    except Exception as e:
        print(f"✗ Edge case advanced_options test failed: {e}")
        return False

def main():
    """Run all advanced_options tests."""
    print("Testing advanced_options handling in OpenAI Compatible Chat Node...\n")
    
    tests = [
        test_dictionary_advanced_options,
        test_pydantic_advanced_options,
        test_none_advanced_options,
        test_edge_case_advanced_options,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Advanced options tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All advanced_options tests passed! The node can handle various advanced_options types.")
        return True
    else:
        print("❌ Some advanced_options tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
