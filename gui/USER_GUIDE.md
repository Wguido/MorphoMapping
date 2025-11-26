# MorphoMapping GUI - User Guide

## 📖 Introduction

Welcome to the MorphoMapping GUI! This guide walks you through using the application step by step.

### What is MorphoMapping GUI?

MorphoMapping GUI is a desktop application for analyzing ImageStream data (.daf files). It enables:

- Dimensionality reduction (DensMAP, UMAP, t-SNE)
- Clustering (KMeans, GMM, HDBSCAN)
- Visualization and export of results
- Feature analysis and cluster statistics

### System Requirements

- Python 3.10 or 3.11
- R 4.0 or newer
- 8 GB RAM (16 GB recommended)
- macOS 10.15+, Windows 10+, or Linux

---

## Getting Started

### 1. Installation

Follow the **[INSTALLATION.md](INSTALLATION.md)** guide to install the GUI.

### 2. Start GUI

```bash
conda activate morphomapping
cd gui
python morphomapping_gui.py
```

A window should open.

---

## Step-by-Step Workflow

### Step 1: Project Setup

#### Set Run-ID

1. Find the **"Run-ID"** field at the top left
2. Enter a unique ID (e.g., `experiment_2025_01_25`)
3. The Run-ID is used for all output files

#### Choose Project Directory (Optional)

1. Click **"📁 Choose Project Directory"**
2. Select a folder where your results should be saved
3. Default: `bundle_runs/` in the MorphoMapping folder

**Tip:** Use meaningful Run-IDs like `neutrophils_day1` or `experiment_control_vs_treated`.

---

### Step 2: Load DAF Files

#### Option A: File Selection Dialog

1. Click **"📁 Select DAF Files"**
2. Select one or more .daf files
3. Click "Open"

#### Option B: Drag & Drop

1. Drag .daf files directly into the **"📁 Drop DAF files here"** field
2. Files are automatically detected

#### What Happens?

- The .daf files are converted to .fcs files
- A progress bar shows the status
- After conversion, files appear in the status overview

**Note:** Large files (100-500 MB) can take several minutes. Please be patient.

---

### Step 3: Enter Metadata

#### Option A: Manual Entry (Left)

1. The table automatically shows the loaded files
2. **file_name**: Automatically filled
3. **sample_id**: Automatically numbered (`sample_1`, `sample_2`, etc.)
4. **group**: Enter your groups (e.g., `control`, `treated`)
5. **replicate**: Enter replicate numbers (e.g., `1`, `2`, `3`)

**Tip:** You can scroll the table if you have many files.

#### Option B: CSV/Excel Upload (Right)

1. Click **"📤 Upload Metadata"**
2. Select a CSV or Excel file
3. Metadata is automatically loaded

**Required Columns:**
- `file_name`: Name of the .daf/.fcs file (without extension)
- `sample_id`: Unique sample ID (optional, auto-generated)
- `group`: Experimental group (e.g., `control`, `treated`)
- `replicate`: Replicate number (optional)

#### Save Metadata

1. Click **"💾 Save Metadata"**
2. Status changes to " Saved"

**IMPORTANT:** Save metadata before starting the analysis!

---

### Step 4: Select Features

#### Include Features

1. Scroll to the **"4⃣ Features & Gates Selection"** section
2. In the **"Features to Include"** area, you'll see blue chips
3. Click a chip to remove it (moves to "Exclude")
4. Click again to add it back

**Tip:** The first 10 features are visible. Click **"▶ Show All"** to see all of them.

#### Exclude Features

1. In the **"Features to Exclude"** area, you'll see red chips
2. Click a chip to remove it (moves to "Include")
3. Click again to exclude it

**Note:** Features that aren't present in all files are automatically excluded.

#### Select Population/Gate

1. Choose a population from the dropdown menu
2. Only cells from this population will be analyzed

**Tip:** Choose a population that exists in all files.

---

### Step 5: Dimensionality Reduction

#### Choose Method

1. Scroll to the **"5⃣ Dimensionality Reduction"** section
2. Select a method:
 - **DensMAP** (default, recommended)
 - **UMAP**
 - **t-SNE**

#### Adjust Parameters

Depending on the method, you'll see different sliders:

**DensMAP:**
- **Dens Lambda**: Density regularization (default: 2.0)
- **N Neighbors**: Number of neighbors (default: 30)
- **Min Dist**: Minimum distance (default: 0.1)

**UMAP:**
- **N Neighbors**: Number of neighbors (default: 30)
- **Min Dist**: Minimum distance (default: 0.1)

**t-SNE:**
- **Perplexity**: Perplexity parameter (default: 30.0)

#### Sampling (Optional)

- **Max cells per sample**: Limits the number of analyzed cells per sample
- Useful for very large datasets
- 0 = Analyze all cells

#### Start Analysis

1. Click **"▶ Run Analysis"**
2. A progress bar appears
3. Analysis can take several minutes (depending on data size)

**After Completion:**
- A plot appears in the **"6⃣ Visualization"** section
- The **" Download Top10 Features"** button becomes active
- The **"7⃣ Clustering"** section becomes visible

---

### Step 6: Visualization

#### Change Color Coding

1. In the **"6⃣ Visualization"** section, find the **"Color by"** dropdown
2. Select an option:
 - `sample_id`: Color by sample
 - `group`: Color by group
 - `replicate`: Color by replicate
 - Other metadata columns

**Tip:** The plot updates automatically without recalculation!

#### Adjust Axis Limits

1. Enter values in the fields:
 - **X Min / X Max**: X-axis limits
 - **Y Min / Y Max**: Y-axis limits
2. Click **"Apply Limits"**
3. Click **"Reset"** to reset

**Usage:** Useful to hide outliers or focus on specific areas.

#### Highlight Cells

1. Enter cell indices in the **"Cell Indices"** field (comma-separated, e.g., `1, 5, 10`)
2. Click **" Highlight"**
3. Cells are marked as red stars

**Tip:** You can highlight multiple cells at once.

#### Export Plot

1. Click **"📥 Export PNG"** or **"📥 Export PDF"**
2. Choose a save location
3. File is saved with 300 DPI

---

### Step 7: Clustering

#### Choose Algorithm

1. Scroll to the **"7⃣ Clustering"** section
2. Select an algorithm:
 - **KMeans**: Exact number of clusters (default: 10)
 - **Gaussian Mixture Models (GMM)**: Probabilistic clustering
 - **HDBSCAN**: Density-based clustering

#### Adjust Parameters

Parameters change depending on the algorithm:

**KMeans:**
- **N Clusters**: Number of clusters (default: 10)
- ** Download Elbow Plot**: Shows optimal cluster number

**GMM:**
- **N Clusters**: Number of clusters (default: 10)
- **Covariance Type**: Covariance type (default: `full`)

**HDBSCAN:**
- **Min Cluster Size**: Minimum cluster size (default: automatic)
- **Min Samples**: Minimum samples (default: 10)

#### Start Clustering

1. Click **"▶ Run Clustering"**
2. A progress bar appears
3. After completion, you'll see:
 - A cluster plot
 - A cluster statistics table
 - Export buttons

#### Cluster Statistics

The table shows:
- **Cluster**: Cluster ID
- **Size**: Number of cells in cluster
- **Percentage**: Percentage distribution
- **Sample Distribution**: Distribution across samples

#### Export Cluster Plot

1. Click **"📥 Export PNG"** or **"📥 Export PDF"**
2. Axis limits are applied

#### Export Cluster Statistics

1. Click **" Export Bar Chart"**
2. Creates a stacked bar chart by groups
3. Saves as PNG

**Note:** Requires a `group` column in metadata!

---

### Step 8: Advanced Analysis

#### Top 3 Features per Cluster

1. After clustering, find the **" Top 3 Features per Cluster"** button
2. Click it
3. A CSV file is created with:
 - Cluster ID
 - Top 1, 2, 3 features and their values

**Location:** `bundle_runs/run_YYYYMMDD_HHMMSS/results/top3_features_per_cluster.csv`

#### Cluster-Feature Heatmap

1. Click **"🔥 Cluster-Feature Heatmap"**
2. Creates a heatmap with:
 - **Rows**: Features
 - **Columns**: Clusters
 - **Values**: Row-wise Z-score
3. Saves:
 - PNG: `cluster_feature_heatmap.png`
 - CSV: `cluster_feature_heatmap_data.csv`

**Usage:** Identifies characteristic features per cluster.

#### Top 10 Features

1. After dimensionality reduction, find **" Download Top10 Features"**
2. Click it
3. Calculates the most important features for X and Y dimensions
4. Saves:
 - CSV: `top10_features.csv`
 - Plots: `top10_features_x_dim.png`, `top10_features_y_dim.png`

**Duration:** Can take several minutes for large datasets.

---

## Tips & Tricks

### Performance

- **Use sampling**: For very large datasets (>100,000 cells), use "Max cells per sample"
- **Reduce features**: Fewer features = faster calculation
- **Choose population**: Analyze only relevant populations

### Data Quality

- **Check metadata**: Make sure `group` is filled correctly
- **Consistent filenames**: `file_name` in metadata must exactly match filename
- **Check features**: Make sure all important features are present in all files

### Workflow Optimization

1. **Test first with few files** (2-3 files)
2. **Use sampling** for initial tests
3. **Save metadata** before analyzing
4. **Export results** regularly

---

## Frequently Asked Questions (FAQ)

### Q: GUI won't start

**A:** Check:
1. Python version: `python --version` (should be 3.10 or 3.11)
2. Environment activated: `conda activate morphomapping`
3. Dependencies installed: `pip list | grep PySide6`

### Q: "No module named 'morphomapping'"

**A:** Install the package:
```bash
cd /path/to/MorphoMapping
pip install -e ./
```

### Q: Metadata not being applied

**A:** Check:
1. `file_name` in metadata exactly matches filename (without .fcs)
2. Metadata was saved (status shows " Saved")
3. Analysis was started after saving

### Q: Plot is empty or gray

**A:** Check:
1. `sample_id` column exists and has values
2. Metadata was correctly linked with analysis data
3. Try changing "Color by"

### Q: Clustering shows no results

**A:** Check:
1. Dimensionality reduction completed successfully
2. Cluster algorithm was selected
3. Parameters are reasonable (e.g., not too many clusters for small datasets)

### Q: Export doesn't work

**A:** Check:
1. Results were calculated (plot is visible)
2. Write permissions in output folder
3. Enough disk space

---

## Troubleshooting

### "Analysis failed" Error

1. Check console/terminal for detailed error messages
2. Make sure all dependencies are installed
3. Check if R is installed: `Rscript --version`

### GUI Freezes

1. Wait - large files can take a long time
2. Check the progress bar
3. Restart if needed (progress will be lost)

### Files Not Converting

1. Check if R is installed
2. Check if Rscript is in PATH
3. Check the files - are they corrupted?

---

## Support

If you have problems:

1. **Read this guide** again
2. **Check [INSTALLATION.md](INSTALLATION.md)** for installation issues
3. **Create a GitHub Issue**: https://github.com/Wguido/MorphoMapping/issues
 - Describe the problem
 - Include error messages
 - Provide system information

---

**Last Updated:** 2025-11-25
