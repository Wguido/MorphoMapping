# GitHub Issues - Preparation

## Issue 1: Cluster-Feature Heatmap Visualization

### Title
**Cluster-Feature Heatmap shows wrong axes and no visible differences**

### Description
The Cluster-Feature Heatmap is generated, but:
1. **Features are on X-axis instead of Y-axis**: Features should be displayed vertically (Y-axis), clusters horizontally (X-axis)
2. **No visible differences**: The heatmap shows no clear cluster differences, all values seem similar
3. **Heatmap is yellow**: Color coding doesn't work properly - should be red-yellow-blue (RdYlBu_r) with Z-score

### Expected Behavior
- **Y-axis**: Features (vertical, readable labels)
- **X-axis**: Clusters (horizontal, cluster IDs)
- **Color coding**: Clear differences between clusters (red = high Z-score, blue = low Z-score)
- **Z-score**: Row-wise Z-score per feature across all clusters

### Current Behavior
- Features appear on X-axis (squashed)
- Y-axis shows only "Feature" and "0"
- Heatmap is completely yellow (no color differences)
- No visible cluster differences

### Technical Details
- Function: `export_cluster_heatmap()` in `morphomapping_gui.py`
- Data structure: `heatmap_z` should have features as index (rows) and clusters as columns
- Z-score is calculated row-wise, but visualization shows no differences

### Debug Information
The function outputs debug information:
- `DEBUG: Final heatmap_z shape: ...`
- `DEBUG: heatmap_z value range: ...`
- `DEBUG: heatmap_z index (features): ...`
- `DEBUG: heatmap_z columns (clusters): ...`

### Possible Causes
1. Data linking between `embedding_df` and `feature_data` doesn't work correctly
2. Z-score calculation gives no variation (all values similar)
3. Visualization uses wrong axes (transposed?)
4. Colorbar scaling is wrong (vmin/vmax)

### Steps to Reproduce
1. Load DAF files
2. Enter metadata
3. Select features
4. Run dimensionality reduction
5. Run clustering
6. Click "🔥 Cluster-Feature Heatmap" button

### Files
- `Documentation/Bundle/gui/morphomapping_gui.py` (Line ~3346-3630)

---

## Issue 2: Top 10 Features Calculation

### Title
**Top 10 Features calculation crashes or doesn't work correctly**

### Description
The calculation of Top 10 Features for X and Y dimensions:
1. **Crash**: The calculation frequently crashes (`zsh: killed python`)
2. **Missing results**: Top 10 Features are not calculated or exported correctly
3. **Memory issues**: Large datasets cause memory problems

### Expected Behavior
- **Calculation**: Top 10 Features for X-dimension and Y-dimension are calculated
- **Export**: CSV file with features and importance scores
- **Plots**: Two PNG files (`top10_features_x_dim.png`, `top10_features_y_dim.png`)
- **Stability**: No crashes, even with large datasets

### Current Behavior
- Calculation starts but crashes
- Or: Calculation runs but no results
- Memory errors with large datasets

### Technical Details
- Function: `download_top10_features()` in `morphomapping_gui.py`
- Worker: `FeatureImportanceWorker` (QThread)
- Library: `morphomapping.morphomapping.MM.feature_importance()`
- Data: Combined features + x/y coordinates

### Debug Information
The function outputs debug information:
- `DEBUG: Combined data shape: ...`
- `DEBUG: Calculating X importance...`
- `DEBUG: X importance calculated successfully: ...`
- `DEBUG: Calculating Y importance...`

### Possible Causes
1. Memory overflow with large datasets
2. `MM.feature_importance()` has problems with the data
3. Feature data is not loaded/linked correctly
4. Threading issues (worker terminated too early)

### Steps to Reproduce
1. Load DAF files
2. Enter metadata
3. Select features
4. Run dimensionality reduction
5. Click "📊 Download Top10 Features" button

### Files
- `Documentation/Bundle/gui/morphomapping_gui.py` (Line ~2150-2250, `FeatureImportanceWorker`)

---

## Additional Information

### System
- **OS**: macOS 14.3.1
- **Python**: 3.10/3.11
- **Framework**: PySide6
- **Data**: ImageStream .daf files (100-500 MB)

### Priority
- **Issue 1 (Heatmap)**: High - Important visualization for cluster analysis
- **Issue 2 (TopFeatures)**: High - Important feature analysis

### Labels (for GitHub)
- `bug`
- `visualization`
- `heatmap`
- `feature-importance`
- `memory`
