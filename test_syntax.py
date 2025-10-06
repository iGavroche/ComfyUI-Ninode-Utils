#!/usr/bin/env python3
"""
Test script to verify syntax and basic structure
"""

import sys
import os
import ast

def test_syntax():
    """Test syntax of all Python files"""
    print("Testing ComfyUI-Ninode-Utils syntax...")
    
    files_to_test = [
        "__init__.py",
        "nodes.py", 
        "nodes_audio.py"
    ]
    
    all_good = True
    
    for filename in files_to_test:
        print(f"Testing {filename}...")
        try:
            with open(filename, 'r') as f:
                source = f.read()
            
            # Parse the AST to check for syntax errors
            ast.parse(source, filename=filename)
            print(f"   ✅ {filename} syntax is valid")
            
        except SyntaxError as e:
            print(f"   ❌ {filename} syntax error: {e}")
            all_good = False
        except Exception as e:
            print(f"   ❌ {filename} error: {e}")
            all_good = False
    
    return all_good

def test_structure():
    """Test basic structure without importing"""
    print("\nTesting file structure...")
    
    # Check if required files exist
    required_files = [
        "__init__.py",
        "nodes.py",
        "pyproject.toml",
        "README.md"
    ]
    
    all_good = True
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"   ✅ {filename} exists")
        else:
            print(f"   ❌ {filename} missing")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    syntax_ok = test_syntax()
    structure_ok = test_structure()
    
    if syntax_ok and structure_ok:
        print("\n✅ All tests passed! The custom node structure is correct.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
