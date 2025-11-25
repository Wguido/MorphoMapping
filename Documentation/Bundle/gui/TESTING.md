# Performance Testing Guide

## Overview

This guide explains how to test the MorphoMapping GUI performance and identify system limits.

## Performance Test Script

The `test_performance.py` script tests how many files the system can handle and measures performance metrics.

### Usage

```bash
# Basic usage with default test sequence (1, 5, 10, 20, 50, 100 files)
python test_performance.py <source_daf_file>

# Custom number of files to test
python test_performance.py <source_daf_file> 1 5 10 20 50 100 200

# Example
python test_performance.py /path/to/test.daf 1 5 10 20
```

### What It Tests

1. **File Conversion Performance**
   - Time to convert DAF files to FCS format
   - Number of successful conversions
   - Errors encountered

2. **Analysis Performance**
   - Time for dimensionality reduction
   - Memory usage (indirectly through timing)

### Output

The script generates:
- `performance_test_results/performance_test_YYYYMMDD_HHMMSS.csv` - Detailed results
- `performance_test_results/performance_report_YYYYMMDD_HHMMSS.txt` - Summary report

### Interpreting Results

- **Conversion time per file**: Should be < 30 seconds per file for reasonable performance
- **Total conversion time**: Should be < 5 minutes for typical use cases
- **Analysis time**: Depends on number of cells, not just files

## Performance Optimizations in GUI

The GUI includes several performance optimizations:

1. **Caching**
   - File counts are cached for 5 minutes
   - File listings are cached to avoid repeated filesystem access

2. **Reduced Reruns**
   - Minimized `st.rerun()` calls
   - Visualization updates only when necessary

3. **Session State Management**
   - Large dataframes stored in session state to avoid re-computation
   - Old results cleared when parameters change

## Troubleshooting "Blank Browser" Issue

If the browser becomes blank at the top:

1. **Check Browser Console**
   - Open browser developer tools (F12)
   - Look for JavaScript errors or memory warnings

2. **Reduce Number of Files**
   - Test with fewer files first
   - Gradually increase to find the limit

3. **Clear Browser Cache**
   - Clear Streamlit cache: `streamlit cache clear`
   - Clear browser cache

4. **Check System Resources**
   - Monitor RAM usage
   - Check disk space for temporary files

5. **Use Firefox**
   - Firefox is recommended for better performance
   - Chrome/Edge may have memory issues with large datasets

## Recommended Limits

Based on testing, recommended limits:

- **Small datasets**: 1-10 files, < 100K cells total
- **Medium datasets**: 10-50 files, < 500K cells total
- **Large datasets**: 50-100 files, < 1M cells total
- **Very large datasets**: > 100 files may require batch processing

## Performance Tips

1. **Pre-process files**: Convert DAF to FCS before uploading if possible
2. **Use appropriate features**: Select only necessary features to reduce memory
3. **Close other applications**: Free up RAM for analysis
4. **Use SSD storage**: Faster file I/O improves performance
5. **Monitor during analysis**: Watch system resources during long operations

## Reporting Issues

If you encounter performance issues:

1. Run the performance test script
2. Note the number of files and file sizes
3. Check system resources (RAM, CPU, disk)
4. Report with:
   - Number of files tested
   - File sizes
   - System specifications
   - Error messages (if any)



