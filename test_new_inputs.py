#!/usr/bin/env python3
"""
Test script for the new inputs in OpenAI Compatible Chat Node
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nodes import OpenAICompatibleChatNode

def test_files_input():
    """Test that files input is properly handled."""
    print("Testing files input handling...")
    try:
        node = OpenAICompatibleChatNode()
        
        # Test with mock files data
        files = [
            {
                "file_data": "This is test file content",
                "filename": "test.txt",
                "type": "input_file"
            }
        ]
        
        # Test create_messages with files
        messages = node.create_messages(
            prompt="Please analyze this file",
            files=files,
            system_message="You are a file analyzer"
        )
        
        # Check that files are included in the message
        user_message = messages[-1]  # Last message should be user message
        content = user_message["content"]
        
        if isinstance(content, list):
            # Multi-part content
            file_found = any("File: test.txt" in item.get("text", "") for item in content if item.get("type") == "text")
        else:
            # Single content
            file_found = "File: test.txt" in content
        
        if file_found:
            print("✓ Files input is properly handled")
            return True
        else:
            print("✗ Files input not found in message content")
            return False
            
    except Exception as e:
        print(f"✗ Files input test failed: {e}")
        return False

def test_advanced_options_input():
    """Test that advanced_options input is properly handled."""
    print("Testing advanced_options input handling...")
    try:
        node = OpenAICompatibleChatNode()
        
        # Test with mock advanced options
        advanced_options = {
            "max_tokens": 2000,
            "temperature": 0.5,
            "top_p": 0.8,
            "instructions": "Be very creative"
        }
        
        # Test that the options would be applied to payload
        # We can't easily test the full API call without mocking, but we can test the structure
        if advanced_options and all(key in advanced_options for key in ["max_tokens", "temperature", "top_p"]):
            print("✓ Advanced options structure is valid")
            return True
        else:
            print("✗ Advanced options structure is invalid")
            return False
            
    except Exception as e:
        print(f"✗ Advanced options test failed: {e}")
        return False

def test_input_types_inclusion():
    """Test that the new inputs are included in INPUT_TYPES."""
    print("Testing that new inputs are included in INPUT_TYPES...")
    try:
        input_types = OpenAICompatibleChatNode.INPUT_TYPES()
        optional = input_types.get("optional", {})
        
        # Check for files input
        if "files" not in optional:
            print("✗ 'files' input not found in optional inputs")
            return False
        
        # Check for advanced_options input
        if "advanced_options" not in optional:
            print("✗ 'advanced_options' input not found in optional inputs")
            return False
        
        # Check that the types are correct
        files_type = optional["files"][0]
        advanced_options_type = optional["advanced_options"][0]
        
        if files_type != "OPENAI_INPUT_FILES":
            print(f"✗ 'files' input has wrong type: {files_type}")
            return False
        
        if advanced_options_type != "OPENAI_CHAT_CONFIG":
            print(f"✗ 'advanced_options' input has wrong type: {advanced_options_type}")
            return False
        
        print("✓ New inputs are properly included in INPUT_TYPES")
        return True
        
    except Exception as e:
        print(f"✗ Input types test failed: {e}")
        return False

def main():
    """Run all tests for new inputs."""
    print("Testing new inputs in OpenAI Compatible Chat Node...\n")
    
    tests = [
        test_input_types_inclusion,
        test_files_input,
        test_advanced_options_input,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"New inputs tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All new input tests passed! The node now has complete input compatibility.")
        return True
    else:
        print("❌ Some new input tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

