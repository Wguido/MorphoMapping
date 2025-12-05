# MorphoMapping GUI User Guide

This guide explains how to use MorphoMapping GUI to analyze ImageStream data. The application processes .daf files (typically 100-500 MB) and performs dimensionality reduction, clustering, and feature analysis.

## Starting the Application

```bash
conda activate morphomapping
cd gui
python morphomapping_gui.py
```

## Basic Workflow

### 1. Project Setup

**Set Run-ID:**
Enter a unique identifier in the "Run-ID" field (e.g., `experiment_2025_01_25` or `neutrophils_day1_control`). This identifier is used for all output files and folders, making it easy to organize multiple experiments. Use descriptive names that help you identify the analysis later.

**Choose Project Directory (optional):**
Click "Choose Project Directory" to select where results should be saved. If not specified, results are saved in `bundle_runs/` within the MorphoMapping folder. Each run creates a timestamped folder (e.g., `bundle_runs/run_20250125_143022/`) containing all results, plots, and data files. This keeps your analyses organized and prevents overwriting previous results.

### 2. Loading DAF Files

**File Selection:**
Click "Select DAF Files" to open a file dialog, or drag and drop .daf files directly into the drop area. You can select multiple files at once. The application supports ImageStream .daf files, which typically range from 100-500 MB in size.

**Conversion Process:**
DAF files are automatically converted to FCS (Flow Cytometry Standard) format, which is required for downstream analysis. The conversion uses an R script that reads the binary .daf format and extracts cell data along with all measured features. 

**What to expect:**
- Small files (< 50 MB): 1-2 minutes per file
- Medium files (50-200 MB): 3-5 minutes per file  
- Large files (200-500 MB): 5-10 minutes per file

A progress bar shows the conversion status. The GUI remains responsive during conversion as it runs in a background thread. After conversion, files appear in the status overview with their sample information.

**Important:** Make sure all .daf files are from the same experiment and contain compatible feature sets. Files with different feature sets may cause issues during analysis.

### 3. Metadata

Metadata links your files with experimental information like treatment groups, time points, or sample conditions. This information is used for coloring plots, grouping analyses, and generating statistics.

**Manual Entry:**
The metadata table automatically shows all loaded files. Each row represents one file:
- `file_name`: Automatically filled from the filename (cannot be changed)
- `sample_id`: Auto-numbered as `sample_1`, `sample_2`, etc. You can edit these to use custom names
- `group`: Enter your experimental groups (e.g., `control`, `treated`, `day1`, `day2`, `patient_A`)
- `replicate`: Enter replicate numbers (e.g., `1`, `2`, `3` or `rep1`, `rep2`)

You can scroll the table if you have many files. The table shows approximately 3 rows at a time.

**CSV/Excel Upload:**
For many files, it's easier to prepare a metadata file externally and upload it.

**Required Columns:**
- `file_name` (REQUIRED): Must exactly match your .daf filename WITHOUT the .daf extension
  - Example: If your file is `sample_001.daf`, use `sample_001` in the metadata

**Optional Standard Columns:**
- `sample_id`: Unique identifier (auto-generated from file_name if missing)
- `group`: Experimental group for coloring and statistics (e.g., control, treated, timepoint_1)
- `replicate`: Replicate number (e.g., 1, 2, 3)

**Custom Columns:**
You can add ANY additional columns you need. All columns will be available for:
- Coloring plots in the visualization
- Grouping and filtering data
- Statistical comparisons

Example CSV:
```csv
file_name,group,replicate,treatment,timepoint,patient_id
sample_001,control,1,untreated,0h,patient_A
sample_002,control,2,untreated,0h,patient_A
sample_003,treated,1,drug_X,24h,patient_B
sample_004,treated,2,drug_X,24h,patient_B
```

**File Format:**
- Supported formats: `.csv`, `.xlsx`, `.xls`
- Structure: Each row = one file, each column = one attribute
- Encoding: UTF-8 recommended for CSV files

**Validation:**
When you upload metadata, the app checks:
1. Is the `file_name` column present?
2. Do the file names match your loaded DAF files?
3. Are there any mismatches (typos, missing files)?

If mismatches are found, you'll see a detailed report showing:
- Which metadata entries have no matching DAF file
- Which DAF files have no metadata entry
- How many files match correctly

**Mismatch Handling:**
- Files that don't match will be EXCLUDED from analysis
- Only cells from matched files will be used
- You can choose to continue or cancel and fix the metadata first

Click "Upload Metadata" to load your file. The table updates automatically.

**Save:**
Always click "Save Metadata" before starting analysis. The status indicator changes to "Saved" when successful. Metadata is saved to `bundle_runs/run_YYYYMMDD_HHMMSS/metadata/sample_sheet.csv`. If you don't save, the analysis may not use your metadata correctly.

### 4. Feature Selection

Features are the measured parameters from your ImageStream data - fluorescence channels, morphological parameters, and other cell characteristics. Selecting the right features is crucial for meaningful analysis.

**Import/Export Features:**
You can save and load your feature selection as Excel files to easily reuse settings between runs:
- **Export Features:** Click "📤 Export Features (Excel)" to save your current included/excluded features to an Excel file
- **Import Features:** Click "📥 Import Features (Excel)" to load a previously saved feature selection
- The Excel file contains two columns: "included" (selected features) and "excluded" (deselected features)
- If features in the Excel file don't exist in the current data, you'll get a warning but valid features will still be imported
- This is especially useful when iterating between runs with similar datasets

**Include/Exclude Features:**
- Blue chips represent included features (will be used in analysis)
- Red chips represent excluded features (will be ignored)
- Click a chip to toggle between include/exclude

**Why exclude features?**
- Some features may be redundant or highly correlated
- Excluding irrelevant features speeds up computation
- Focus on biologically meaningful parameters
- Features not present in all files are automatically excluded

The first 10 features are visible by default. Click "Show All" to see and manage all features.

**Population Selection:**
Choose a population from the dropdown menu (e.g., `PMN`, `Granulocytes`, `Lymphocytes`). Only cells belonging to this population will be analyzed. This is important because:
- Different cell types have different feature distributions
- Analyzing mixed populations can obscure meaningful patterns
- Gates/populations are pre-defined in your .daf files

**Critical:** Make sure the selected population exists in all your files. If a file doesn't contain the selected population, it may be excluded from analysis or cause errors.

### 5. Dimensionality Reduction

Dimensionality reduction projects your high-dimensional data (150-200 features) into a 2D space for visualization. This is essential because humans can't visualize more than 3 dimensions. The goal is to preserve the most important structure of your data while reducing complexity.

**Understanding the Methods:**

**DensMAP (default, recommended):**
DensMAP is an extension of UMAP that explicitly preserves local density information. This means:
- **Advantages:** Better preserves density variations in your data, making it easier to identify rare cell populations and density-based structures. Particularly useful for cytometry data where cell density matters.
- **When to use:** Most cases, especially when you want to identify rare populations or density-based patterns
- **Parameters:**
  - **Dens Lambda** (default: 2.0): Controls how much emphasis is placed on preserving density. Higher values (3-5) emphasize density more, lower values (0.5-1.5) emphasize global structure. Start with default, increase if you need better separation of rare populations.
  - **N Neighbors** (default: 30): Number of neighboring points considered. Higher values (50-100) preserve more global structure but may blur local details. Lower values (10-20) preserve local structure but may fragment global patterns. For large datasets (>100k cells), increase to 50-100.
  - **Min Dist** (default: 0.1): Minimum distance between points in the embedding. Higher values (0.3-0.5) create more spread-out clusters, lower values (0.01-0.05) create tighter clusters. Use lower values if clusters appear too spread out.

**UMAP:**
UMAP (Uniform Manifold Approximation and Projection) is fast and preserves both local and global structure:
- **Advantages:** Faster than DensMAP, good balance between local and global structure, deterministic results
- **When to use:** When you need faster computation or when density preservation isn't critical
- **Parameters:**
  - **N Neighbors** (default: 30): Similar to DensMAP. Controls the balance between local and global structure.
  - **Min Dist** (default: 0.1): Same as DensMAP. Controls cluster tightness.

**t-SNE:**
t-SNE (t-distributed Stochastic Neighbor Embedding) focuses on local structure:
- **Advantages:** Excellent for visualization, creates visually appealing plots with clear separation
- **Disadvantages:** Slower than UMAP/DensMAP, doesn't preserve global structure well, results can vary between runs
- **When to use:** When you primarily need visualization and local structure is most important
- **Parameters:**
  - **Perplexity** (default: 30.0): Roughly the number of neighbors to consider. Should be less than the number of data points. For small datasets (<1000 cells), use 5-15. For large datasets, 30-50 works well. Higher values preserve more global structure but may blur local details.

**Choosing the Right Method:**
- **Start with DensMAP** - it's the default for good reason
- Use **UMAP** if DensMAP is too slow or you don't need density information
- Use **t-SNE** if you only need visualization and want the best-looking plots
- You can try all three and compare results

**Sampling:**
Set "Max cells per sample" to limit the number of analyzed cells per sample. This is useful for:
- Very large datasets (>100,000 cells per sample) where full analysis would be too slow
- Initial exploration where you want quick results
- Testing parameters before running full analysis

Set to 0 to analyze all cells. For initial tests, 5,000-10,000 cells per sample is often sufficient. For final analysis, use all cells or a larger sample size.

**Running the Analysis:**
Click "Run Analysis" to start dimensionality reduction. This process:
1. Loads all selected cells from all files
2. Extracts the selected features
3. Applies the chosen dimensionality reduction method
4. Projects the data into 2D space (x, y coordinates)

**Processing time:**
- Small datasets (<10,000 cells): 1-3 minutes
- Medium datasets (10,000-100,000 cells): 3-10 minutes
- Large datasets (>100,000 cells): 10-30 minutes or more

After completion, a plot appears in the Visualization section showing your cells in 2D space. The "Feature Importance" button also becomes active.

### 6. Visualization

The visualization section lets you explore your dimensionality reduction results interactively.

**Color Coding:**
Use the "Color by" dropdown to color points by different metadata columns:
- `sample_id`: Each sample gets a different color - useful to see if samples cluster together
- `group`: Colors by experimental group - essential for comparing treatments
- `replicate`: Colors by replicate - helps identify batch effects
- Other metadata: Any column from your metadata can be used

The plot updates automatically without recalculation. This is useful for exploring your data from different perspectives. For example, you might first color by `sample_id` to check data quality, then switch to `group` to see treatment effects.

**Axis Limits:**
Enter X Min/Max and Y Min/Max values, then click "Apply Limits" to zoom into specific regions. This is useful for:
- Hiding outliers that compress the main data
- Focusing on specific cell populations
- Creating publication-ready plots with consistent axes
- Comparing multiple analyses with the same axis limits

Click "Reset" to remove limits and show all data. The limits are also applied when exporting plots.

**Highlighting:**
Enter cell indices (comma-separated, e.g., `1, 5, 10` or `1-10`) and click "Highlight" to mark specific cells as red stars. This is useful for:
- Tracking specific cells of interest
- Identifying cells with unusual feature values
- Comparing cell positions across different visualizations

**Export:**
Click "Export PNG" or "Export PDF" to save the plot. PNG is good for presentations and quick sharing. PDF is better for publications as it's vector-based and scales without quality loss. Both are saved at 300 DPI for high quality. The exported plot includes the current color coding and axis limits.

### 7. Clustering

Clustering groups similar cells together to identify distinct cell populations. This is essential for characterizing your data and identifying cell types or states.

**Understanding the Algorithms:**

**KMeans (default):**
KMeans partitions cells into a specified number of clusters by minimizing within-cluster variance:
- **Advantages:** Fast, deterministic (same input = same output), works well with spherical clusters, easy to interpret
- **Disadvantages:** Requires specifying the number of clusters, assumes clusters are spherical and similar in size, sensitive to outliers
- **When to use:** When you know approximately how many cell types/populations to expect, when you need fast results, when clusters are roughly spherical
- **Parameters:**
  - **N Clusters:** The number of clusters to find. This is the most important parameter. Use the "Download Elbow Plot" button to help determine the optimal number. The elbow plot shows within-cluster variance vs. number of clusters - look for the "elbow" where adding more clusters doesn't significantly reduce variance. Common values: 5-20 for most datasets, but depends on your cell types.
- **Finding optimal cluster number:** The elbow plot is essential. Run KMeans with different cluster numbers (e.g., 5, 10, 15, 20) and look at the elbow plot. The optimal number is usually at the "elbow" where the curve bends. You can also use biological knowledge - if you expect 8 cell types, start with 8 clusters.

**Gaussian Mixture Models (GMM):**
GMM is a probabilistic clustering method that models clusters as Gaussian distributions:
- **Advantages:** Handles overlapping clusters, can model clusters of different shapes and sizes, provides probability of cluster membership for each cell
- **Disadvantages:** Slower than KMeans, still requires specifying cluster number, can be sensitive to initialization
- **When to use:** When clusters overlap or have different shapes/sizes, when you need probability information, when KMeans doesn't work well
- **Parameters:**
  - **N Clusters:** Number of clusters (same as KMeans)
  - **Covariance Type:**
    - `full` (default): Each cluster can have any elliptical shape and orientation - most flexible, use when clusters have different shapes
    - `diag`: Clusters are axis-aligned ellipses - faster, use when clusters are elongated along feature axes
    - `spherical`: Clusters are circular - fastest, use when clusters are roughly circular

**HDBSCAN:**
HDBSCAN (Hierarchical Density-Based Spatial Clustering) is a density-based method that automatically finds clusters:
- **Advantages:** Automatically determines number of clusters, identifies noise/outliers, handles clusters of varying densities and shapes, doesn't require cluster number
- **Disadvantages:** Can be slower, may identify many small clusters or noise, results can be sensitive to parameters
- **When to use:** When you don't know how many clusters to expect, when you want to identify outliers, when clusters have varying densities
- **Parameters:**
  - **Min Cluster Size:** Minimum number of cells required to form a cluster. Smaller values (50-100) find more, smaller clusters. Larger values (500-1000) find fewer, larger clusters. Start with 1-2% of your total cell count.
  - **Min Samples:** Minimum number of cells in a neighborhood to be considered a core point. Lower values (5-10) are more permissive and find more clusters. Higher values (20-50) are more conservative. Usually set to 10-20.

**Choosing the Right Algorithm:**
- **Start with KMeans** if you have a rough idea of cluster number - it's fast and works well for most cases
- Use **GMM** if KMeans gives poor results or clusters overlap significantly
- Use **HDBSCAN** if you don't know the cluster number or want to identify outliers
- You can try multiple algorithms and compare results

**Running Clustering:**
Click "Run Clustering" to start. The process:
1. Takes the 2D coordinates from dimensionality reduction
2. Applies the chosen clustering algorithm
3. Assigns each cell to a cluster
4. Calculates cluster statistics

Processing time: Usually 1-5 minutes depending on data size and algorithm (HDBSCAN can be slower).

After completion, you'll see:
- A cluster plot with cells colored by cluster assignment
- A statistics table showing cluster sizes and distributions

**Cluster Statistics:**
The statistics table shows:
- **Cluster:** Cluster ID (0, 1, 2, ...)
- **Size:** Number of cells in the cluster
- **Percentage:** Percentage of total cells
- **Sample Distribution:** How cells are distributed across samples (useful to see if a cluster is sample-specific)

This information helps you understand your clusters and identify interesting populations.

**Export:**
- **Cluster plot:** Click "Export PNG" or "Export PDF" to save the cluster visualization. Axis limits are automatically applied.
- **Statistics:** Click "Export Bar Chart" to create a stacked bar chart showing cluster distribution by groups. This requires a `group` column in your metadata. Useful for comparing cluster frequencies between experimental groups.

### 8. Advanced Analysis

**Top 3 Features per Cluster:**
After clustering, click "Top 3 Features per Cluster" to identify which features are most characteristic for each cluster. This generates a CSV file listing, for each cluster, the top 3 features that best distinguish it from other clusters. This is essential for:
- Understanding what makes each cluster unique
- Interpreting cluster identity (e.g., if a cluster has high CD4 and low CD8, it might be CD4+ T cells)
- Validating that clusters are biologically meaningful

The CSV is saved to `bundle_runs/run_YYYYMMDD_HHMMSS/results/top3_features_per_cluster.csv`.

**Cluster-Feature Heatmap:**
Click "Cluster-Feature Heatmap" to create a comprehensive heatmap showing feature expression across all clusters. The heatmap:
- **Rows (Y-axis, right side):** Features
- **Columns (X-axis, bottom):** Clusters
- **Values:** Row-wise Z-scores (normalized per feature) - red = high expression, blue = low expression, yellow/white = average

This visualization helps you:
- See patterns of feature expression across clusters
- Identify clusters with similar feature profiles
- Understand relationships between features and clusters
- Generate publication-quality figures

The heatmap includes dendrograms showing hierarchical relationships between features (left) and clusters (top). It's saved as PNG (high resolution) and the underlying data is saved as CSV for further analysis.

**Group-Cluster Bar Graphs:**
Click "📊 Group-Cluster Bar Graphs" to create publication-quality bar graphs showing group differences within each cluster. This analysis:
- Creates one graph per cluster
- X-axis: Groups (from metadata)
- Y-axis: Percentage of cells from each group that are in the cluster
- Includes SEM error bars and individual data points (one per sample/replicate)
- Shows statistical significance markers (*, **, ***) only for significant comparisons (p < 0.05)

This visualization is essential for:
- Comparing group frequencies within clusters
- Identifying clusters that differ significantly between groups
- Generating publication-ready figures with proper statistics

Results are saved as `cluster_{cluster}_group_bar_graph.png` for each cluster in the results folder. The graphs use consistent formatting (14pt labels, 12pt ticks, 300 DPI) matching other plots in the application.

**Feature Importance:**
After dimensionality reduction, click "Feature Importance" to determine which features drive the X and Y dimensions of your embedding. This is important because:
- It helps you understand what the dimensions represent biologically
- It identifies which features are most important for the observed structure
- It validates that relevant features are being used

The calculation uses a two-stage Random Forest approach:
1. **Stage 1:** Quick preselection with 20 trees on all features (fast)
2. **Stage 2:** Detailed calculation with 100 trees on top 50 features (accurate)

This approach is much faster than calculating importance for all features with 100 trees, especially with 150-200 features.

**Processing time:** 2-10 minutes depending on data size. For datasets with 150-200 features, expect 5-10 minutes.

Results are saved as:
- CSV: `top10_features.csv` with importance scores for X and Y dimensions
- Plots: `top10_features_x.png` and `top10_features_y.png` showing bar charts of top 10 features

**PCA Plot (Sample-Level):**
Click "📈 PCA Plot (Sample-Level)" in the Visualization section to create a PCA plot showing group differences at the sample level. This analysis:
- Calculates mean feature values per sample
- Performs PCA on sample means (not individual cells)
- Colors samples by their group assignment from metadata
- Helps visualize whether groups are separated in feature space

The plot shows PC1 vs PC2 with explained variance percentages. Each point represents one sample. This is useful for:
- Validating that experimental groups are distinct
- Identifying batch effects or outliers
- Understanding overall group differences before detailed cluster analysis

Results are saved as `pca_plot_groups_sample_level.png` in the results folder.

The top 10 features for each dimension are the ones that Random Forest found most predictive of the X or Y coordinate, meaning they're driving the structure you see in the plot.

## Tips

**Performance:**
- Use sampling for very large datasets (>100,000 cells)
- Reduce features to speed up calculations
- Start with 2-3 files to test the workflow

**Data Quality:**
- Ensure `file_name` in metadata exactly matches the filename (without .fcs)
- Make sure all important features are present in all files
- Verify conversion completed successfully

**Workflow:**
- Test with few files first
- Save metadata before analyzing
- Export results regularly
- Use meaningful Run-IDs

## Common Issues

**GUI won't start:**
Check Python version (`python --version` should be 3.10 or 3.11), verify environment is activated, check dependencies.

**Metadata not applied:**
Verify `file_name` matches filename exactly (without .fcs), ensure metadata was saved, start analysis after saving.

**Plot is empty:**
Check that `sample_id` has values, try changing "Color by", check axis limits.

**Clustering shows no results:**
Ensure dimensionality reduction completed, verify algorithm was selected, check parameters are reasonable.

**Feature Importance takes too long:**
This is normal for large datasets (150-200 features). The two-stage approach reduces computation time. Be patient - it can take 5-10 minutes.

**Export doesn't work:**
Check that results were calculated, verify write permissions, ensure sufficient disk space.

## Getting Help

For problems not covered here:
1. Check the console/terminal for error messages
2. Create an issue at https://github.com/Wguido/MorphoMapping/issues
3. Include error messages and system information
