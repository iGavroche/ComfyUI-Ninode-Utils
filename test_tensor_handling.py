#!/usr/bin/env python3
"""
Test script for tensor handling in OpenAI Compatible Chat Node
Tests various tensor shapes and data types that might cause issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from nodes import OpenAICompatibleChatNode

def test_problematic_tensor_shapes():
    """Test handling of problematic tensor shapes."""
    print("Testing problematic tensor shapes...")
    try:
        node = OpenAICompatibleChatNode()
        
        # Test the specific shape that caused the error: (1, 1, 2589)
        problematic_tensor = torch.randint(0, 255, (1, 1, 2589), dtype=torch.uint8)
        print(f"Testing tensor shape: {problematic_tensor.shape}")
        
        # This should not crash, but should handle gracefully
        result = node.tensor_to_base64(problematic_tensor)
        
        # Should return a placeholder image
        assert result.startswith("data:image/png;base64,"), "Should return a valid data URL"
        print("✓ Problematic tensor shape handled gracefully")
        return True
        
    except Exception as e:
        print(f"✗ Problematic tensor shape test failed: {e}")
        return False

def test_valid_image_tensors():
    """Test handling of valid image tensor shapes."""
    print("Testing valid image tensor shapes...")
    try:
        node = OpenAICompatibleChatNode()
        
        # Test various valid image shapes
        test_cases = [
            torch.randn(64, 64, 3),  # HWC RGB
            torch.randn(3, 64, 64),  # CHW RGB
            torch.randn(64, 64, 1),  # HWC Grayscale
            torch.randn(1, 64, 64),  # CHW Grayscale
            torch.randn(64, 64),     # 2D Grayscale
            torch.randn(1, 64, 64, 3),  # Batch RGB
        ]
        
        for i, tensor in enumerate(test_cases):
            print(f"  Testing case {i+1}: {tensor.shape}")
            result = node.tensor_to_base64(tensor)
            assert result.startswith("data:image/png;base64,"), f"Case {i+1} should return valid data URL"
        
        print("✓ All valid image tensor shapes handled correctly")
        return True
        
    except Exception as e:
        print(f"✗ Valid image tensor test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases for tensor conversion."""
    print("Testing edge cases...")
    try:
        node = OpenAICompatibleChatNode()
        
        # Test edge cases
        edge_cases = [
            torch.randn(1, 1, 1),      # Very small tensor
            torch.randn(1, 1, 1000),   # Unusual width
            torch.randn(1000, 1, 1),   # Unusual height
            torch.randn(1, 1000, 1),   # Another unusual case
            torch.zeros(64, 64, 3),    # All zeros
            torch.ones(64, 64, 3),     # All ones
        ]
        
        for i, tensor in enumerate(edge_cases):
            print(f"  Testing edge case {i+1}: {tensor.shape}")
            result = node.tensor_to_base64(tensor)
            assert result.startswith("data:image/png;base64,"), f"Edge case {i+1} should return valid data URL"
        
        print("✓ All edge cases handled correctly")
        return True
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        return False

def test_different_dtypes():
    """Test different tensor data types."""
    print("Testing different tensor data types...")
    try:
        node = OpenAICompatibleChatNode()
        
        # Test different data types
        dtypes = [torch.uint8, torch.float32, torch.float64, torch.int32]
        tensor = torch.randn(64, 64, 3)
        
        for dtype in dtypes:
            print(f"  Testing dtype: {dtype}")
            test_tensor = tensor.to(dtype)
            result = node.tensor_to_base64(test_tensor)
            assert result.startswith("data:image/png;base64,"), f"Should handle {dtype} correctly"
        
        print("✓ All data types handled correctly")
        return True
        
    except Exception as e:
        print(f"✗ Data types test failed: {e}")
        return False

def main():
    """Run all tensor handling tests."""
    print("Testing tensor handling in OpenAI Compatible Chat Node...\n")
    
    tests = [
        test_problematic_tensor_shapes,
        test_valid_image_tensors,
        test_edge_cases,
        test_different_dtypes,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tensor handling tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tensor handling tests passed! The node can handle various tensor shapes gracefully.")
        return True
    else:
        print("❌ Some tensor handling tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
