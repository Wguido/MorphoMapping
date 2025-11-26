# GitHub Issue Templates

## Issue 1: Cluster-Feature Heatmap

**Title:**
```
Cluster-Feature Heatmap: Wrong axes and no visible differences
```

**Body (copy this into the GitHub issue):**
```markdown
## Problem

The Cluster-Feature Heatmap is generated, but:
1. **Features are on X-axis instead of Y-axis**: Features should be displayed vertically (Y-axis), clusters horizontally (X-axis)
2. **No visible differences**: The heatmap shows no clear cluster differences, all values seem similar
3. **Heatmap is yellow**: Color coding doesn't work properly - should be red-yellow-blue (RdYlBu_r) with Z-score

## Expected Behavior

- **Y-axis**: Features (vertical, readable labels)
- **X-axis**: Clusters (horizontal, cluster IDs)
- **Color coding**: Clear differences between clusters (red = high Z-score, blue = low Z-score)
- **Z-score**: Row-wise Z-score per feature across all clusters

## Current Behavior

- Features appear on X-axis (squashed)
- Y-axis shows only "Feature" and "0"
- Heatmap is completely yellow (no color differences)
- No visible cluster differences

## Technical Details

- Function: `export_cluster_heatmap()` in `morphomapping_gui.py`
- Data structure: `heatmap_z` should have features as index (rows) and clusters as columns
- Z-score is calculated row-wise, but visualization shows no differences

## Debug Information

The function outputs debug information:
- `DEBUG: Final heatmap_z shape: ...`
- `DEBUG: heatmap_z value range: ...`
- `DEBUG: heatmap_z index (features): ...`
- `DEBUG: heatmap_z columns (clusters): ...`

## Steps to Reproduce

1. Load DAF files
2. Enter metadata
3. Select features
4. Run dimensionality reduction
5. Run clustering
6. Click "🔥 Cluster-Feature Heatmap" button

## Files

- `gui/morphomapping_gui.py` (Line ~3346-3630)

## Labels

bug, visualization, heatmap
```

---

## Issue 2: Top 10 Features

**Title:**
```
Top 10 Features calculation crashes
```

**Body (copy this into the GitHub issue):**
```markdown
## Problem

The calculation of Top 10 Features for X and Y dimensions:
1. **Crash**: The calculation frequently crashes (`zsh: killed python`)
2. **Missing results**: Top 10 Features are not calculated or exported correctly
3. **Memory issues**: Large datasets cause memory problems

## Expected Behavior

- **Calculation**: Top 10 Features for X-dimension and Y-dimension are calculated
- **Export**: CSV file with features and importance scores
- **Plots**: Two PNG files (`top10_features_x_dim.png`, `top10_features_y_dim.png`)
- **Stability**: No crashes, even with large datasets

## Current Behavior

- Calculation starts but crashes
- Or: Calculation runs but no results
- Memory errors with large datasets

## Technical Details

- Function: `download_top10_features()` in `morphomapping_gui.py`
- Worker: `FeatureImportanceWorker` (QThread)
- Library: `morphomapping.morphomapping.MM.feature_importance()`
- Data: Combined features + x/y coordinates

## Debug Information

The function outputs debug information:
- `DEBUG: Combined data shape: ...`
- `DEBUG: Calculating X importance...`
- `DEBUG: X importance calculated successfully: ...`
- `DEBUG: Calculating Y importance...`

## Steps to Reproduce

1. Load DAF files
2. Enter metadata
3. Select features
4. Run dimensionality reduction
5. Click "📊 Download Top10 Features" button

## Files

- `gui/morphomapping_gui.py` (Line ~2150-2250, `FeatureImportanceWorker`)

## Labels

bug, feature-importance, memory
```

---

## How to Create Issues on GitHub

1. Go to: https://github.com/Wguido/MorphoMapping/issues/new
2. Copy the title and body for Issue 1
3. Add labels: `bug`, `visualization`, `heatmap`
4. Click "Submit new issue"
5. Repeat for Issue 2 with labels: `bug`, `feature-importance`, `memory`
