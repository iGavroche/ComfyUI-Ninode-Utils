#!/usr/bin/env python3
"""
Test script to verify the custom node structure and imports
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test all imports to identify issues"""
    print("Testing ComfyUI-Ninode-Utils imports...")
    
    try:
        # Test __init__.py
        print("1. Testing __init__.py...")
        import __init__
        print("   ✅ __init__.py imported successfully")
        print(f"   Available: {__init__.__all__}")
    except Exception as e:
        print(f"   ❌ __init__.py failed: {e}")
        return False
    
    try:
        # Test nodes.py
        print("2. Testing nodes.py...")
        import nodes
        print("   ✅ nodes.py imported successfully")
        print(f"   Node mappings: {list(nodes.NODE_CLASS_MAPPINGS.keys())}")
        print(f"   Display names: {list(nodes.NODE_DISPLAY_NAME_MAPPINGS.keys())}")
    except Exception as e:
        print(f"   ❌ nodes.py failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Test nodes_audio.py
        print("3. Testing nodes_audio.py...")
        import nodes_audio
        print("   ✅ nodes_audio.py imported successfully")
        print(f"   VibeVoice available: {nodes_audio.VIBEVOICE_AVAILABLE}")
    except Exception as e:
        print(f"   ❌ nodes_audio.py failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

