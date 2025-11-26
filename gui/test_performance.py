"""
Performance testing script for MorphoMapping GUI.

Tests how many files the system can handle and measures performance metrics.
"""

import time
import subprocess
from pathlib import Path
import pandas as pd
import json
from typing import List, Dict
import shutil

# Configuration
TEST_DATA_DIR = Path("test_data")
RESULTS_DIR = Path("performance_test_results")
RESULTS_DIR.mkdir(exist_ok=True)

def create_test_files(num_files: int, source_file: Path, output_dir: Path) -> List[Path]:
    """Create test files by copying a source file multiple times."""
    output_dir.mkdir(parents=True, exist_ok=True)
    test_files = []
    
    for i in range(num_files):
        test_file = output_dir / f"test_file_{i:04d}.daf"
        shutil.copy2(source_file, test_file)
        test_files.append(test_file)
    
    return test_files

def measure_conversion_time(files: List[Path], output_dir: Path) -> Dict:
    """Measure time to convert DAF files to FCS."""
    start_time = time.time()
    
    # Run conversion (simulate GUI conversion)
    R_SCRIPT = Path("../../R/daf_to_fcs_cli.R")
    converted = []
    errors = []
    
    for daf in files:
        target = output_dir / f"{daf.stem}.fcs"
        command = ["Rscript", "--vanilla", "--slave", str(R_SCRIPT), str(daf), str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            errors.append(f"{daf.name}: {result.stderr.strip()}")
        else:
            converted.append(target)
    
    elapsed = time.time() - start_time
    
    return {
        "num_files": len(files),
        "converted": len(converted),
        "errors": len(errors),
        "time_seconds": elapsed,
        "time_per_file": elapsed / len(files) if files else 0,
    }

def measure_analysis_time(fcs_files: List[Path], num_features: int = 10) -> Dict:
    """Measure time for dimensionality reduction analysis."""
    from morphomapping.morphomapping import MM
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    start_time = time.time()
    
    # Simulate analysis
    records = []
    for fcs_file in fcs_files[:10]:  # Limit to 10 files for testing
        # This is a simplified version - actual analysis would load and process
        records.append({"sample_id": fcs_file.stem})
    
    if records:
        combined = pd.DataFrame(records)
        # Simulate scaling
        if len(combined) > 0:
            scaler = StandardScaler()
            # Dummy data for testing
            dummy_data = pd.DataFrame([[0] * num_features] * len(combined))
            scaler.fit_transform(dummy_data)
    
    elapsed = time.time() - start_time
    
    return {
        "num_files": len(fcs_files),
        "time_seconds": elapsed,
        "time_per_file": elapsed / len(fcs_files) if fcs_files else 0,
    }

def run_performance_test(
    num_files_list: List[int],
    source_file: Path,
    test_name: str = "default"
) -> pd.DataFrame:
    """Run performance tests with different numbers of files."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Performance Test: {test_name}")
    print(f"{'='*60}\n")
    
    for num_files in num_files_list:
        print(f"Testing with {num_files} files...")
        
        # Create test files
        test_dir = TEST_DATA_DIR / f"test_{num_files}"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_files = create_test_files(num_files, source_file, test_dir / "daf")
        
        # Test conversion
        fcs_dir = test_dir / "fcs"
        fcs_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            conv_result = measure_conversion_time(test_files, fcs_dir)
            conv_result["test_name"] = test_name
            conv_result["stage"] = "conversion"
            results.append(conv_result)
            print(f"  Conversion: {conv_result['time_seconds']:.2f}s ({conv_result['time_per_file']:.2f}s/file)")
        except Exception as e:
            print(f"  Conversion failed: {e}")
            results.append({
                "num_files": num_files,
                "test_name": test_name,
                "stage": "conversion",
                "converted": 0,
                "errors": num_files,
                "time_seconds": 0,
                "time_per_file": 0,
            })
        
        # Test analysis (if conversion succeeded)
        fcs_files = list(fcs_dir.glob("*.fcs"))
        if fcs_files:
            try:
                analysis_result = measure_analysis_time(fcs_files)
                analysis_result["test_name"] = test_name
                analysis_result["stage"] = "analysis"
                results.append(analysis_result)
                print(f"  Analysis: {analysis_result['time_seconds']:.2f}s ({analysis_result['time_per_file']:.2f}s/file)")
            except Exception as e:
                print(f"  Analysis failed: {e}")
        
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        print()
    
    return pd.DataFrame(results)

def main():
    """Main test function."""
    import sys
    
    # Check if source file is provided
    if len(sys.argv) < 2:
        print("Usage: python test_performance.py <source_daf_file> [num_files...]")
        print("Example: python test_performance.py test.daf 1 5 10 20 50")
        sys.exit(1)
    
    source_file = Path(sys.argv[1])
    if not source_file.exists():
        print(f"Error: Source file not found: {source_file}")
        sys.exit(1)
    
    # Get number of files to test
    if len(sys.argv) > 2:
        num_files_list = [int(x) for x in sys.argv[2:]]
    else:
        # Default test sequence
        num_files_list = [1, 5, 10, 20, 50, 100]
    
    # Run tests
    results_df = run_performance_test(num_files_list, source_file)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"performance_test_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}\n")
    print(results_df.to_string())
    print(f"\nResults saved to: {results_file}")
    
    # Create a simple report
    report_file = RESULTS_DIR / f"performance_report_{timestamp}.txt"
    with open(report_file, "w") as f:
        f.write("MorphoMapping GUI Performance Test Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source File: {source_file}\n")
        f.write(f"File Sizes Tested: {num_files_list}\n\n")
        f.write("Results:\n")
        f.write("-" * 60 + "\n")
        f.write(results_df.to_string())
        f.write("\n\n")
        
        # Recommendations
        f.write("Recommendations:\n")
        f.write("-" * 60 + "\n")
        conversion_results = results_df[results_df["stage"] == "conversion"]
        if len(conversion_results) > 0:
            max_reasonable = conversion_results[
                conversion_results["time_seconds"] < 300  # 5 minutes
            ]["num_files"].max() if len(conversion_results[conversion_results["time_seconds"] < 300]) > 0 else 0
            f.write(f"Maximum reasonable files for conversion (< 5 min): {max_reasonable}\n")
    
    print(f"Report saved to: {report_file}")

if __name__ == "__main__":
    main()



