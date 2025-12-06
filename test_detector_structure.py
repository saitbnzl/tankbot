#!/usr/bin/env python3
"""
Basic structure test for detector module.
Tests that the code structure is correct without requiring Hailo hardware.
"""

import sys
import numpy as np

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    import os
    
    # Just check the files exist (dependencies may not be installed)
    required_files = [
        'detector.py',
        'hailo_runner.py',
        'person_follow.py',
        'tankbot_brain_server.py',
        'object_detection_post_process.py',
        'person_follow_config.py'
    ]
    
    all_exist = True
    for fname in required_files:
        if os.path.exists(fname):
            print(f"✓ {fname} exists")
        else:
            print(f"✗ {fname} not found")
            all_exist = False
    
    return all_exist

def test_detector_class():
    """Test that Detector class structure is correct"""
    print("\nTesting Detector class structure...")
    try:
        # Read the file and check for key elements
        with open('detector.py', 'r') as f:
            content = f.read()
        
        # Check for timeout parameter
        if 'timeout: float' in content or 'timeout=' in content:
            print("✓ timeout parameter found in code")
        else:
            print("✗ timeout parameter not found")
            return False
        
        # Check for threading
        if 'import threading' in content:
            print("✓ threading import found")
        else:
            print("✗ threading import not found")
            return False
        
        # Check for queue
        if 'import queue' in content:
            print("✓ queue import found")
        else:
            print("✗ queue import not found")
            return False
        
        # Check for TimeoutError handling
        if 'TimeoutError' in content:
            print("✓ TimeoutError handling found")
        else:
            print("✗ TimeoutError handling not found")
            return False
        
        print("✓ Detector class structure looks correct")
        return True
    except Exception as e:
        print(f"✗ Failed to check Detector structure: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_person_follow_config():
    """Test that person_follow_config can be imported"""
    print("\nTesting person_follow_config...")
    try:
        from person_follow_config import get_config, update_config
        print("✓ person_follow_config imported")
        
        config = get_config()
        print(f"✓ get_config() returns: {type(config)}")
        
        # Check required keys
        required_keys = [
            'CONF_THRESHOLD', 'IMG_SIZE', 'CENTER_DEADZONE',
            'TURN_SPEED', 'FORWARD_SPEED', 'SEARCH_TURN_SPEED'
        ]
        missing = [k for k in required_keys if k not in config]
        if missing:
            print(f"✗ Missing config keys: {missing}")
            return False
        else:
            print(f"✓ All required config keys present")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test person_follow_config: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_syntax():
    """Test that all Python files have valid syntax"""
    print("\nTesting Python file syntax...")
    import py_compile
    import os
    
    files = [
        'hailo_runner.py',
        'detector.py',
        'person_follow.py',
        'tankbot_brain_server.py',
        'object_detection_post_process.py',
        'person_follow_config.py'
    ]
    
    all_valid = True
    for fname in files:
        if os.path.exists(fname):
            try:
                py_compile.compile(fname, doraise=True)
                print(f"✓ {fname} has valid syntax")
            except py_compile.PyCompileError as e:
                print(f"✗ {fname} has syntax error: {e}")
                all_valid = False
        else:
            print(f"⚠ {fname} not found (skipping)")
    
    return all_valid

def main():
    """Run all tests"""
    print("=" * 60)
    print("TankBot Detector Structure Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: File syntax
    results.append(("File Syntax", test_file_syntax()))
    
    # Test 3: Detector class
    results.append(("Detector Class", test_detector_class()))
    
    # Test 4: Config
    results.append(("Config Module", test_person_follow_config()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
