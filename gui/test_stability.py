#!/usr/bin/env python3
"""
Stability Test Script for MorphoMapping GUI

Tests the GUI's stability with various data sizes and operations.
"""

import subprocess
import time
import sys
from pathlib import Path
import signal
import os

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
NC = "\033[0m"  # No Color

def print_status(message, status="info"):
    """Print colored status message."""
    colors = {
        "success": GREEN,
        "error": RED,
        "warning": YELLOW,
        "info": BLUE
    }
    color = colors.get(status, NC)
    print(f"{color}[{status.upper()}]{NC} {message}")

def test_streamlit_startup():
    """Test if Streamlit can start without errors."""
    print_status("Testing Streamlit startup...", "info")
    
    app_path = Path(__file__).parent / "app.py"
    if not app_path.exists():
        print_status(f"Error: app.py not found at {app_path}", "error")
        return False
    
    try:
        # Start Streamlit in background
        process = subprocess.Popen(
            ["streamlit", "run", str(app_path), "--server.headless", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print_status("Streamlit started successfully", "success")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            return True
        else:
            stdout, stderr = process.communicate()
            print_status(f"Streamlit crashed during startup", "error")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
    except Exception as e:
        print_status(f"Error starting Streamlit: {e}", "error")
        return False

def test_imports():
    """Test if all required imports work."""
    print_status("Testing imports...", "info")
    
    required_modules = [
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "altair",
        "sklearn",
        "hdbscan",
    ]
    
    failed = []
    for module in required_modules:
        try:
            __import__(module)
            print_status(f"  ✓ {module}", "success")
        except ImportError as e:
            print_status(f"  ✗ {module}: {e}", "error")
            failed.append(module)
    
    if failed:
        print_status(f"Failed to import: {', '.join(failed)}", "error")
        return False
    else:
        print_status("All imports successful", "success")
        return True

def test_matplotlib_backend():
    """Test matplotlib backend configuration."""
    print_status("Testing matplotlib backend...", "info")
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        # Create a test figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)
        
        print_status("Matplotlib backend working correctly", "success")
        return True
    except Exception as e:
        print_status(f"Matplotlib backend error: {e}", "error")
        return False

def test_memory_usage():
    """Test memory usage with large datasets."""
    print_status("Testing memory usage...", "info")
    
    try:
        import numpy as np
        import pandas as pd
        
        # Test with 100k rows
        test_data = pd.DataFrame({
            "x": np.random.randn(100000),
            "y": np.random.randn(100000),
            "cluster": np.random.randint(0, 10, 100000)
        })
        
        # Test matplotlib with large dataset
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(test_data["x"], test_data["y"], c=test_data["cluster"], alpha=0.5, s=1)
        plt.close(fig)
        
        print_status("Memory test passed (100k points)", "success")
        return True
    except Exception as e:
        print_status(f"Memory test failed: {e}", "error")
        return False

def main():
    """Run all stability tests."""
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}MorphoMapping GUI Stability Test{NC}")
    print(f"{BLUE}{'='*60}{NC}\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Matplotlib Backend Test", test_matplotlib_backend),
        ("Memory Usage Test", test_memory_usage),
        ("Streamlit Startup Test", test_streamlit_startup),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{YELLOW}Running: {test_name}{NC}")
        print("-" * 60)
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print_status(f"Test crashed: {e}", "error")
            results[test_name] = False
    
    # Summary
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}Test Summary{NC}")
    print(f"{BLUE}{'='*60}{NC}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = GREEN if result else RED
        print(f"{color}{status}{NC}: {test_name}")
    
    print(f"\n{BLUE}Total: {passed}/{total} tests passed{NC}\n")
    
    if passed == total:
        print_status("All tests passed! GUI should be stable.", "success")
        return 0
    else:
        print_status("Some tests failed. Please review the errors above.", "error")
        return 1

if __name__ == "__main__":
    sys.exit(main())

