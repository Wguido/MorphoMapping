"""
MorphoMapping GUI - PySide6 Implementation

Stable desktop GUI for processing large ImageStream .daf files (100-500 MB).
Uses PySide6 for maximum stability with large files - no upload, direct file access.

Installation:
    pip install PySide6 pandas numpy matplotlib scikit-learn hdbscan umap-learn

Start:
    python app_pyside6.py
"""

from __future__ import annotations

import datetime
import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("QtAgg")  # Use Qt backend for PySide6
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QUrl
from PySide6.QtGui import QPixmap, QFont, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QRadioButton, QButtonGroup,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QComboBox, QSlider,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QMessageBox, QProgressBar, QTextEdit, QFrame,
    QSizePolicy, QSpacerItem, QCheckBox, QDialog, QTabWidget, QListWidget, QListWidgetItem
)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Import core business logic (unchanged!)
from core import (
    BUNDLE_ROOT, PROJECT_ROOT, R_SCRIPT,
    DEFAULT_FEATURES, DEFAULT_METADATA_ROWS, COLOR_PALETTE,
    get_run_paths, get_file_counts,
    convert_daf_files,
    load_or_create_metadata,
    save_metadata as save_metadata_file,
    run_dimensionality_reduction,
    run_clustering as run_clustering_analysis,
    get_axis_labels,
)
from core.config import R_HEATMAP_SCRIPT
from morphomapping.morphomapping import MM

# Utility functions for robust path handling
def safe_str(value) -> str:
    """Safely convert any value to string, handling bytes."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except:
            return str(value)
    return str(value)

def safe_path(path) -> Path:
    """Safely create Path object, ensuring string input."""
    if isinstance(path, bytes):
        path = safe_str(path)
    elif not isinstance(path, (str, Path)):
        path = safe_str(path)
    return Path(str(path))


# ============================================================================
# Drag and Drop Widget
# ============================================================================

class DropArea(QLabel):
    """Custom widget for drag and drop of DAF files."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.parent_window = parent
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            # Check if any URL is a .daf file
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.endswith('.daf'):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event for DAF files."""
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith('.daf'):
                files.append(file_path)
        
        if files and self.parent_window:
            self.parent_window.process_daf_files(files)
        
        event.acceptProposedAction()


# ============================================================================
# Background Worker Threads
# ============================================================================

class ConversionWorker(QThread):
    """Worker thread for DAF to FCS conversion."""
    finished = Signal(str, bool, str)  # job_id, success, message
    
    def __init__(self, daf_file: Path, fcs_file: Path, job_id: str):
        super().__init__()
        self.daf_file = daf_file
        self.fcs_file = fcs_file
        self.job_id = job_id
    
    def run(self):
        try:
            command = ["Rscript", "--vanilla", "--slave", str(R_SCRIPT), str(self.daf_file), str(self.fcs_file)]
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.finished.emit(self.job_id, True, f"✅ {safe_str(self.daf_file.name)} converted")
            else:
                # Provide more detailed error message
                error_msg = result.stderr.strip() if result.stderr.strip() else result.stdout.strip()
                if not error_msg:
                    error_msg = f"Conversion failed with return code {result.returncode}"
                self.finished.emit(self.job_id, False, f"❌ Error: {error_msg[:200]}")
        except Exception as e:
            self.finished.emit(self.job_id, False, f"❌ Exception: {str(e)[:200]}")


class AnalysisWorker(QThread):
    """Worker thread for dimensionality reduction analysis."""
    finished = Signal(bool, str)  # success, message
    progress = Signal(str)  # progress message
    
    def __init__(self, paths: Dict[str, Path], features: List[str], method: str,
                 method_params: Dict, population: Optional[str], session_state: Dict):
        super().__init__()
        self.paths = paths
        self.features = features
        self.method = method
        self.method_params = method_params
        self.population = population
        self.session_state = session_state
    
    def run(self):
        try:
            self.progress.emit(f"Running {self.method}...")
            
            embedding_df, info = run_dimensionality_reduction(
                self.paths["fcs"],
                self.features,
                self.paths,
                self.method,
                self.method_params.get("dens_lambda", 2.0),
                self.method_params.get("n_neighbors", 30),
                self.method_params.get("min_dist", 0.1),
                self.method_params.get("perplexity", 30.0),
                self.population,
                self.session_state,
                max_cells_per_sample=self.method_params.get("max_cells_per_sample", 0),
            )
            
            # CRITICAL: Ensure sample_id is always present and not NaN
            if "sample_id" not in embedding_df.columns:
                print(f"DEBUG: WARNING - sample_id not in embedding_df columns: {embedding_df.columns.tolist()}")
                # This should never happen, but if it does, we need to reconstruct it
                embedding_df["sample_id"] = "unknown"
            else:
                # Ensure sample_id is string and has no NaN values
                embedding_df["sample_id"] = embedding_df["sample_id"].astype(str)
                # Replace any NaN strings with actual sample IDs from the data
                nan_mask = embedding_df["sample_id"].isna() | (embedding_df["sample_id"] == "nan") | (embedding_df["sample_id"] == "")
                if nan_mask.any():
                    print(f"DEBUG: WARNING - Found {nan_mask.sum()} rows with NaN/empty sample_id")
                    # Try to get sample_id from index or reconstruct
                    # This shouldn't happen if run_dimensionality_reduction works correctly
            
            # Merge with metadata
            metadata_path = self.paths["metadata"] / "sample_sheet.csv"
            metadata_df = load_or_create_metadata(metadata_path)
            
            # Debug: Print what we have
            print(f"DEBUG: Embedding sample_ids (first 10): {embedding_df['sample_id'].unique()[:10].tolist()}")
            print(f"DEBUG: Embedding shape: {embedding_df.shape}")
            print(f"DEBUG: Embedding sample_id NaN count: {embedding_df['sample_id'].isna().sum()}")
            print(f"DEBUG: Metadata columns: {metadata_df.columns.tolist() if not metadata_df.empty else 'No metadata'}")
            if not metadata_df.empty:
                if "sample_id" in metadata_df.columns:
                    print(f"DEBUG: Metadata sample_ids: {metadata_df['sample_id'].unique().tolist()}")
                if "file_name" in metadata_df.columns:
                    print(f"DEBUG: Metadata file_names: {metadata_df['file_name'].unique().tolist()}")
            
            # Prepare metadata: ensure sample_id exists and matches embedding
            if not metadata_df.empty:
                # Normalize file_name: remove .daf or .fcs extension if present
                if "file_name" in metadata_df.columns:
                    metadata_df["file_name"] = metadata_df["file_name"].astype(str).str.replace(r'\.(daf|fcs)$', '', regex=True)
                
                # If file_name exists but sample_id doesn't, create sample_id from file_name
                if "file_name" in metadata_df.columns and "sample_id" not in metadata_df.columns:
                    metadata_df["sample_id"] = metadata_df["file_name"].copy()
                    print(f"DEBUG: Created sample_id from file_name")
                
                # Create file_name_clean for matching (already normalized, so just copy)
                if "file_name" in metadata_df.columns:
                    metadata_df["file_name_clean"] = metadata_df["file_name"].copy()
                else:
                    metadata_df["file_name_clean"] = None
            
            if not metadata_df.empty and "sample_id" in metadata_df.columns:
                metadata_df["sample_id"] = metadata_df["sample_id"].astype(str)
                
                # Try merge on sample_id first
                embedding_with_meta = embedding_df.merge(
                    metadata_df, 
                    on="sample_id", 
                    how="left", 
                    suffixes=("_embed", "_meta")
                )
                
                # Check if merge was successful (if file_name_clean exists, try that too)
                if "file_name_clean" in metadata_df.columns:
                    # Check how many rows got matched
                    matched_count = embedding_with_meta[metadata_df.columns.difference(["sample_id", "file_name", "file_name_clean"])].notna().any(axis=1).sum()
                    print(f"DEBUG: Matched {matched_count} out of {len(embedding_with_meta)} rows on sample_id")
                    
                    # If no matches, try file_name_clean
                    if matched_count == 0:
                        print(f"DEBUG: Trying merge on file_name_clean...")
                        embedding_df["sample_id_clean"] = embedding_df["sample_id"]
                        metadata_df_clean = metadata_df.copy()
                        metadata_df_clean = metadata_df_clean.rename(columns={"sample_id": "sample_id_meta"})
                        embedding_with_meta = embedding_df.merge(
                            metadata_df_clean,
                            left_on="sample_id_clean",
                            right_on="file_name_clean",
                            how="left",
                            suffixes=("_embed", "_meta")
                        )
                        # Drop the temporary columns
                        if "sample_id_clean" in embedding_with_meta.columns:
                            embedding_with_meta = embedding_with_meta.drop(columns=["sample_id_clean"])
                        if "sample_id_meta" in embedding_with_meta.columns:
                            embedding_with_meta = embedding_with_meta.rename(columns={"sample_id_meta": "sample_id"})
                        
                        matched_count_clean = embedding_with_meta[metadata_df.columns.difference(["sample_id", "file_name", "file_name_clean"])].notna().any(axis=1).sum()
                        print(f"DEBUG: Matched {matched_count_clean} out of {len(embedding_with_meta)} rows on file_name_clean")
                
                # Clean up column names (remove suffixes)
                # First, handle sample_id specially - keep the embedding version (it's the correct one)
                if "sample_id_embed" in embedding_with_meta.columns:
                    # Keep embedding version, drop metadata version
                    if "sample_id_meta" in embedding_with_meta.columns:
                        embedding_with_meta = embedding_with_meta.drop(columns=["sample_id_meta"])
                    embedding_with_meta = embedding_with_meta.rename(columns={"sample_id_embed": "sample_id"})
                elif "sample_id_meta" in embedding_with_meta.columns and "sample_id" not in embedding_with_meta.columns:
                    # Only metadata version exists, rename it
                    embedding_with_meta = embedding_with_meta.rename(columns={"sample_id_meta": "sample_id"})
                
                # Now handle other columns
                for col in list(embedding_with_meta.columns):  # Use list() to avoid modification during iteration
                    if col.endswith("_embed"):
                        base_col = col.replace("_embed", "")
                        if base_col in metadata_df.columns and base_col != "sample_id":
                            # Keep metadata version, drop embedding version
                            embedding_with_meta = embedding_with_meta.drop(columns=[col])
                        elif base_col not in embedding_with_meta.columns:
                            # Rename back if no conflict
                            embedding_with_meta = embedding_with_meta.rename(columns={col: base_col})
                    elif col.endswith("_meta"):
                        base_col = col.replace("_meta", "")
                        if base_col not in ["sample_id", "x", "y"] and base_col not in embedding_with_meta.columns:
                            # Don't rename if it would conflict
                            embedding_with_meta = embedding_with_meta.rename(columns={col: base_col})
                
                # CRITICAL: Ensure only one sample_id column exists
                sample_id_cols = [c for c in embedding_with_meta.columns if c == "sample_id"]
                if len(sample_id_cols) > 1:
                    print(f"DEBUG: WARNING - Multiple sample_id columns found: {sample_id_cols}")
                    # Keep only the first one, drop others
                    for i, col in enumerate(sample_id_cols[1:], 1):
                        embedding_with_meta = embedding_with_meta.drop(columns=[col])
                        print(f"DEBUG: Dropped duplicate sample_id column: {col}")
                
                # Remove temporary file_name_clean column if it exists
                if "file_name_clean" in embedding_with_meta.columns:
                    embedding_with_meta = embedding_with_meta.drop(columns=["file_name_clean"])
            else:
                embedding_with_meta = embedding_df.copy()
            
            # CRITICAL: Ensure sample_id is preserved and not NaN after merge
            # First, check if sample_id exists and is a single column
            if "sample_id" not in embedding_with_meta.columns:
                print(f"DEBUG: ERROR - sample_id lost after merge! Adding from embedding_df...")
                embedding_with_meta["sample_id"] = embedding_df["sample_id"].values
            else:
                # Check if sample_id is a DataFrame (multiple columns with same name)
                sample_id_check = embedding_with_meta["sample_id"]
                if isinstance(sample_id_check, pd.DataFrame):
                    print(f"DEBUG: WARNING - sample_id is a DataFrame! Taking first column...")
                    embedding_with_meta["sample_id"] = sample_id_check.iloc[:, 0]
            
            # Check for NaN sample_id values - ensure we get a scalar, not a Series
            sample_id_col = embedding_with_meta["sample_id"]
            if isinstance(sample_id_col, pd.DataFrame):
                print(f"DEBUG: ERROR - sample_id is still a DataFrame after fix! Taking first column...")
                sample_id_col = sample_id_col.iloc[:, 0]
                embedding_with_meta["sample_id"] = sample_id_col
            nan_sample_id_count = int(sample_id_col.isna().sum())
            if nan_sample_id_count > 0:
                print(f"DEBUG: WARNING - {nan_sample_id_count} rows have NaN sample_id after merge. Filling from embedding_df...")
                # Fill NaN sample_ids from original embedding_df
                # Use the same sample_id_col we already extracted
                nan_mask = sample_id_col.isna()
                if len(embedding_df) == len(embedding_with_meta):
                    embedding_with_meta.loc[nan_mask, "sample_id"] = embedding_df.loc[nan_mask, "sample_id"].values
            
            # Ensure sample_id is string and has no NaN/empty values
            embedding_with_meta["sample_id"] = embedding_with_meta["sample_id"].astype(str)
            # Replace "nan" strings with actual sample IDs if possible
            # Use the same sample_id_col we already extracted
            nan_string_mask = (sample_id_col.astype(str) == "nan") | (sample_id_col.astype(str) == "")
            if nan_string_mask.any():
                nan_string_count = int(nan_string_mask.sum())
                print(f"DEBUG: WARNING - {nan_string_count} rows have 'nan' string in sample_id")
                # Try to get from original embedding if same length
                if len(embedding_df) == len(embedding_with_meta):
                    embedding_with_meta.loc[nan_string_mask, "sample_id"] = embedding_df.loc[nan_string_mask, "sample_id"].astype(str).values
            
            embedding_with_meta["cell_index"] = range(len(embedding_with_meta))
            
            # Debug: Print final result
            print(f"DEBUG: Available columns after merge: {embedding_with_meta.columns.tolist()}")
            
            # Safely get sample_id column
            sample_id_debug = embedding_with_meta["sample_id"]
            if isinstance(sample_id_debug, pd.DataFrame):
                print(f"DEBUG: WARNING - sample_id is a DataFrame! Columns: {sample_id_debug.columns.tolist()}")
                # Take first column
                sample_id_debug = sample_id_debug.iloc[:, 0]
                # Also fix the actual column
                embedding_with_meta["sample_id"] = sample_id_debug
                print(f"DEBUG: Fixed sample_id column to be a Series")
            
            print(f"DEBUG: sample_id column type: {type(sample_id_debug)}")
            print(f"DEBUG: sample_id unique values (first 10): {sample_id_debug.unique()[:10].tolist()}")
            print(f"DEBUG: sample_id NaN count: {int(sample_id_debug.isna().sum())}")
            print(f"DEBUG: sample_id 'nan' string count: {int((sample_id_debug.astype(str) == 'nan').sum())}")
            
            if "group" in embedding_with_meta.columns:
                # Ensure we get a scalar, not a Series
                group_col = embedding_with_meta["group"]
                if isinstance(group_col, pd.DataFrame):
                    group_col = group_col.iloc[:, 0]
                group_count = int(group_col.notna().sum())
                print(f"DEBUG: Rows with group values: {group_count} out of {len(embedding_with_meta)}")
                if group_count == 0:
                    print(f"DEBUG: WARNING - No group values found after merge!")
                    print(f"DEBUG: Embedding sample_ids: {sorted(embedding_df['sample_id'].unique().tolist())}")
                    if not metadata_df.empty and "sample_id" in metadata_df.columns:
                        print(f"DEBUG: Metadata sample_ids: {sorted(metadata_df['sample_id'].unique().tolist())}")
                    if not metadata_df.empty and "file_name" in metadata_df.columns:
                        print(f"DEBUG: Metadata file_names: {sorted(metadata_df['file_name'].unique().tolist())}")
            
            self.session_state["embedding_df"] = embedding_with_meta
            self.session_state["features"] = self.features
            self.session_state["metadata_df"] = metadata_df
            self.session_state["stored_dim_reduction_method"] = self.method
            self.session_state["selected_population"] = self.population
            # Store file processing info (skipped_files, usable_files)
            self.session_state["analysis_info"] = info
            
            # Store method and params for later saving in GUI
            self.session_state["last_dim_reduction_method"] = self.method
            self.session_state["last_dim_reduction_params"] = self.method_params
            
            self.finished.emit(True, f"✅ {self.method} analysis completed successfully!")
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()
            print(f"Analysis error: {error_msg}\n{error_trace}")
            self.finished.emit(False, f"❌ Analysis failed: {error_msg[:100]}")


class FeatureImportanceWorker(QThread):
    """Worker thread for calculating feature importance using MorphoMapping."""
    finished = Signal(bool, str)  # success, message
    
    def __init__(self, embedding_df: pd.DataFrame, features: List[str], paths: Dict[str, Path], population: Optional[str], session_state: Dict):
        super().__init__()
        self.embedding_df = embedding_df
        self.features = features
        self.paths = paths
        self.population = population
        self.session_state = session_state
    
    def run(self):
        try:
            import traceback
            import numpy as np
            from morphomapping.morphomapping import MM
            
            # Load all data from CSV cache and combine with embedding
            cache_dir = self.paths["csv_cache"]
            converter = MM()
            records = []
            
            # Get sample_ids from embedding
            sample_ids = set(self.embedding_df["sample_id"].astype(str).unique())
            
            # Load data for each sample
            for fcs_file in sorted(self.paths["fcs"].glob("*.fcs")):
                fcs_stem = str(fcs_file.stem)
                if fcs_stem not in sample_ids:
                    continue
                
                csv_path = cache_dir / f"{fcs_stem}.csv"
                if not csv_path.exists():
                    converter.convert_to_CSV(str(fcs_file), str(csv_path))
                
                df = pd.read_csv(csv_path)
                
                # Filter by population if selected
                if self.population:
                    if self.population in df.columns:
                        # Get population column as Series
                        pop_series = df[self.population]
                        if isinstance(pop_series, pd.DataFrame):
                            pop_series = pop_series.iloc[:, 0]
                        
                        # Check dtype safely - convert to numpy array first
                        pop_array = np.asarray(pop_series)
                        pop_dtype = pop_array.dtype
                        
                        # Create mask using numpy array to avoid Series ambiguity
                        if pop_dtype == bool or pop_dtype == 'bool':
                            mask = pop_array == True
                        else:
                            # For numeric types, check if value equals 1
                            mask = pop_array == 1
                        
                        # Apply mask to dataframe
                        df = df[mask]
                
                if len(df) == 0:
                    continue
                
                # Select only features that exist in the data
                available_features = [f for f in self.features if f in df.columns]
                if not available_features:
                    continue
                
                subset = df[available_features].copy()
                subset["sample_id"] = fcs_stem
                records.append(subset)
            
            if not records:
                self.finished.emit(False, "Could not load data. Check if FCS files are available.")
                return
            
            # Combine all data
            combined_data = pd.concat(records, ignore_index=True)
            
            # We need to match each cell with its embedding coordinates
            # Since we can't directly match by index, we'll sample the same number of cells
            # from the embedding for each sample
            
            # More efficient merging: use cell_index if available, otherwise sample
            embedding_subset = self.embedding_df[["sample_id", "x", "y"]].copy()
            embedding_subset["sample_id"] = embedding_subset["sample_id"].astype(str)
            
            # Try to merge on sample_id and cell_index if both have it
            if "cell_index" in combined_data.columns and "cell_index" in embedding_subset.columns:
                # Direct merge on sample_id and cell_index
                combined_data["sample_id"] = combined_data["sample_id"].astype(str)
                combined_data = combined_data.merge(
                    embedding_subset,
                    on=["sample_id", "cell_index"],
                    how="inner",
                    suffixes=("", "_emb")
                )
            else:
                # Fallback: sample-based merging
                combined_with_coords = []
                for sample_id in combined_data["sample_id"].unique():
                    sample_data = combined_data[combined_data["sample_id"] == sample_id].copy()
                    sample_embedding = embedding_subset[embedding_subset["sample_id"] == sample_id].copy()
                    
                    # Sample same number of cells from embedding
                    n_cells = len(sample_data)
                    if len(sample_embedding) >= n_cells:
                        sample_embedding = sample_embedding.sample(n=n_cells, random_state=42).reset_index(drop=True)
                    else:
                        # If embedding has fewer cells, repeat the last ones
                        while len(sample_embedding) < n_cells:
                            sample_embedding = pd.concat([sample_embedding, sample_embedding.iloc[-1:]], ignore_index=True)
                        sample_embedding = sample_embedding.iloc[:n_cells]
                    
                    # Add coordinates
                    sample_data["x"] = sample_embedding["x"].values
                    sample_data["y"] = sample_embedding["y"].values
                    combined_with_coords.append(sample_data)
                
                combined_data = pd.concat(combined_with_coords, ignore_index=True)
            
            # Remove rows without coordinates
            combined_data = combined_data[combined_data["x"].notna() & combined_data["y"].notna()]
            
            if len(combined_data) == 0:
                self.finished.emit(False, "Could not merge embedding with data. Check sample_id matching.")
                return
            
            # Validate data before creating MM object
            print(f"DEBUG: Combined data shape: {combined_data.shape}")
            print(f"DEBUG: Combined data columns: {combined_data.columns.tolist()}")
            
            # Check if we have x and y columns
            if "x" not in combined_data.columns or "y" not in combined_data.columns:
                self.finished.emit(False, f"Missing x or y coordinates in data. Columns: {combined_data.columns.tolist()}")
                return
            
            # Check if we have enough features (need at least 2 features + x + y)
            feature_cols = [c for c in combined_data.columns if c not in ["sample_id", "x", "y"]]
            if len(feature_cols) < 2:
                self.finished.emit(False, f"Not enough features for importance calculation. Found: {len(feature_cols)} features. Need at least 2.")
                return
            
            # Remove non-numeric columns and NaN values
            numeric_cols = combined_data.select_dtypes(include=[np.number]).columns.tolist()
            if "x" not in numeric_cols or "y" not in numeric_cols:
                self.finished.emit(False, "x and y must be numeric columns.")
                return
            
            # Keep only numeric columns
            combined_data_clean = combined_data[numeric_cols].copy()
            
            # Remove rows with NaN in x or y
            combined_data_clean = combined_data_clean[combined_data_clean["x"].notna() & combined_data_clean["y"].notna()]
            
            if len(combined_data_clean) < 100:
                self.finished.emit(False, f"Not enough valid data points. Found: {len(combined_data_clean)} rows. Need at least 100 for reliable feature importance calculation.")
                return
            
            # ADAPTIVE SAMPLING: Set upper limit on total values (cells × features)
            # This ensures that with many features (150-200), we use fewer cells, and vice versa
            feature_cols = [c for c in combined_data_clean.columns if c not in ["x", "y"]]
            num_features = len(feature_cols)
            
            # Upper limit: max_total_values = cells × features
            # This keeps memory usage constant regardless of feature count
            max_total_values = 10000  # Upper limit for total data points (cells × features)
            
            # Calculate max cells per sample based on number of features
            # More features → fewer cells, fewer features → more cells
            max_cells_per_sample = max_total_values // num_features
            
            # Set reasonable bounds:
            # - Minimum: 50 cells per sample (for statistical validity)
            # - Maximum: 200 cells per sample (to prevent excessive memory even with few features)
            max_cells_per_sample = max(50, min(200, max_cells_per_sample))
            
            print(f"DEBUG: Found {num_features} features")
            print(f"DEBUG: Calculated max_cells_per_sample = {max_total_values} / {num_features} = {max_total_values // num_features}")
            print(f"DEBUG: After bounds: max_cells_per_sample = {max_cells_per_sample}")
            print(f"DEBUG: Expected total values: ~{max_cells_per_sample} cells × {num_features} features = ~{max_cells_per_sample * num_features} values")
            
            # Sample per sample_id with adaptive cell count
            sampled_data = []
            for sample_id in combined_data["sample_id"].unique():
                sample_data = combined_data[combined_data["sample_id"] == sample_id].copy()
                if len(sample_data) > max_cells_per_sample:
                    sample_data = sample_data.sample(n=max_cells_per_sample, random_state=42).reset_index(drop=True)
                sampled_data.append(sample_data)
            
            combined_data_clean = pd.concat(sampled_data, ignore_index=True)
            
            # Remove non-numeric columns again (after sampling)
            numeric_cols = combined_data_clean.select_dtypes(include=[np.number]).columns.tolist()
            combined_data_clean = combined_data_clean[numeric_cols].copy()
            
            # Remove rows with NaN in x or y again
            combined_data_clean = combined_data_clean[combined_data_clean["x"].notna() & combined_data_clean["y"].notna()]
            
            print(f"DEBUG: After adaptive per-sample sampling: {len(combined_data_clean)} total cells from {len(combined_data['sample_id'].unique())} samples")
            print(f"DEBUG: Final data shape: {combined_data_clean.shape}, features: {len([c for c in combined_data_clean.columns if c not in ['x', 'y']])}")
            
            if len(combined_data_clean) < 100:
                self.finished.emit(False, f"Not enough valid data points after sampling. Found: {len(combined_data_clean)} rows. Need at least 100.")
                return
            
            # Additional global cap as safety measure (should rarely be needed with adaptive sampling)
            max_rows_global = 3000  # Global cap as backup
            if len(combined_data_clean) > max_rows_global:
                print(f"DEBUG: Still too large ({len(combined_data_clean)} rows), applying global cap to {max_rows_global}")
                combined_data_clean = combined_data_clean.sample(n=max_rows_global, random_state=42).reset_index(drop=True)
            
            # CRITICAL: Additional data validation to prevent crashes
            # Replace Inf values with NaN, then drop those rows
            import numpy as np
            combined_data_clean = combined_data_clean.replace([np.inf, -np.inf], np.nan)
            combined_data_clean = combined_data_clean.dropna()
            
            # Check for very large values that might cause numerical instability
            # Clip extreme values to reasonable range (within 10 standard deviations)
            for col in combined_data_clean.columns:
                if col in ["x", "y"]:
                    continue
                col_data = combined_data_clean[col]
                if col_data.dtype in [np.float64, np.float32]:
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    if std_val > 0:
                        # Clip to mean ± 10*std
                        combined_data_clean[col] = col_data.clip(
                            mean_val - 10 * std_val,
                            mean_val + 10 * std_val
                        )
            
            # Ensure all values are finite
            for col in combined_data_clean.columns:
                if combined_data_clean[col].dtype in [np.float64, np.float32]:
                    combined_data_clean[col] = combined_data_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Final dropna after clipping
            combined_data_clean = combined_data_clean.dropna()
            
            if len(combined_data_clean) < 100:
                self.finished.emit(False, f"Not enough valid data points after cleaning. Found: {len(combined_data_clean)} rows. Need at least 100.")
                return
            
            # Additional safety check: if still too large after adaptive sampling, reduce further
            # This should rarely be needed now with adaptive per-sample sampling
            if len(combined_data_clean) > 3000:
                print(f"DEBUG: Further reducing sample size from {len(combined_data_clean)} to 3000 for stability")
                combined_data_clean = combined_data_clean.sample(n=3000, random_state=42).reset_index(drop=True)
            
            print(f"DEBUG: Clean data shape: {combined_data_clean.shape}, columns: {len(combined_data_clean.columns)}")
            print(f"DEBUG: Memory usage: {combined_data_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Force garbage collection before creating MM object
            import gc
            gc.collect()
            
            # Create MM object and set dataframe
            try:
                mm = MM()
                print("DEBUG: MM object created")
                
                # Ensure dataframe is clean and has correct types
                # Make a fresh copy to ensure memory alignment
                mm.df = combined_data_clean.copy()
                
                # Ensure all columns are float64 for numerical stability
                for col in mm.df.columns:
                    if mm.df[col].dtype in [np.float32, np.int32, np.int64]:
                        mm.df[col] = mm.df[col].astype(np.float64)
                
                print(f"DEBUG: MM.df shape: {mm.df.shape}, columns: {mm.df.columns.tolist()[:10]}")
                
                # Verify x and y columns exist and are numeric
                if 'x' not in mm.df.columns or 'y' not in mm.df.columns:
                    self.finished.emit(False, f"Missing x or y columns in MM.df. Available: {mm.df.columns.tolist()}")
                    return
                
                # Final validation: check for any remaining invalid values
                x_valid = mm.df['x'].notna() & np.isfinite(mm.df['x'])
                y_valid = mm.df['y'].notna() & np.isfinite(mm.df['y'])
                valid_mask = x_valid & y_valid
                
                if valid_mask.sum() < len(mm.df) * 0.9:  # If more than 10% invalid
                    print(f"WARNING: {len(mm.df) - valid_mask.sum()} rows have invalid x/y values, removing them")
                    mm.df = mm.df[valid_mask].copy()
                
                if len(mm.df) < 100:
                    self.finished.emit(False, f"Not enough valid data after final validation. Found: {len(mm.df)} rows.")
                    return
                
                print(f"DEBUG: x column dtype: {mm.df['x'].dtype}, y column dtype: {mm.df['y'].dtype}")
                print(f"DEBUG: x range: [{mm.df['x'].min():.2f}, {mm.df['x'].max():.2f}]")
                print(f"DEBUG: y range: [{mm.df['y'].min():.2f}, {mm.df['y'].max():.2f}]")
                print(f"DEBUG: x has NaN: {mm.df['x'].isna().sum()}, Inf: {np.isinf(mm.df['x']).sum()}")
                print(f"DEBUG: y has NaN: {mm.df['y'].isna().sum()}, Inf: {np.isinf(mm.df['y']).sum()}")
                import sys
                sys.stdout.flush()
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"Error creating MM object: {e}\n{error_trace}")
                self.finished.emit(False, f"Error creating MM object: {str(e)[:200]}")
                return
            
            # Force garbage collection before feature importance calculation
            gc.collect()
            
            # Set multiprocessing start method to 'spawn' to avoid semaphore leaks
            import multiprocessing
            try:
                if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                    multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                # Already set, ignore
                pass
            
            # Calculate feature importance for X dimension
            features_x = pd.DataFrame()
            try:
                print("=" * 80)
                print("DEBUG: Starting X importance calculation...")
                print(f"DEBUG: MM.df shape before feature_importance: {mm.df.shape}")
                print(f"DEBUG: MM.df columns: {mm.df.columns.tolist()[:20]}")
                print(f"DEBUG: MM.df memory usage: {mm.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                # Additional validation: check for matrix singularity issues
                # Check if x and y have sufficient variance
                x_var = mm.df['x'].var()
                y_var = mm.df['y'].var()
                print(f"DEBUG: X variance: {x_var:.6f}, Y variance: {y_var:.6f}")
                
                if x_var < 1e-10 or y_var < 1e-10:
                    print("WARNING: Very low variance in x or y, this may cause numerical issues")
                
                # Check correlation between x and y (should not be too high)
                xy_corr = abs(mm.df['x'].corr(mm.df['y']))
                print(f"DEBUG: X-Y correlation: {xy_corr:.6f}")
                if xy_corr > 0.99:
                    print("WARNING: Very high correlation between x and y, this may cause numerical issues")
                
                print(f"DEBUG: Calling mm.feature_importance(dep='x', indep='y')...")
                import sys
                sys.stdout.flush()  # Force output to console
                
                # Clear any cached computations in MM object
                if hasattr(mm, 'df'):
                    # Ensure df is a fresh copy
                    mm.df = mm.df.copy()
                
                # TWO-STAGE APPROACH: Fast preselection + detailed calculation
                # This reduces computation time significantly for large feature sets (150-200 features)
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import MinMaxScaler
                
                # Prepare data
                data = mm.df.copy()
                data = data.drop('y', axis=1)  # Drop indep
                
                # Split data
                train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
                X_train_full = train_df.drop('x', axis=1)  # Drop dep
                y_train = train_df['x']
                num_features = len(X_train_full.columns)
                
                print(f"DEBUG: Starting two-stage calculation for {num_features} features...")
                
                # STAGE 1: Fast preselection with fewer trees (all features)
                # Use only 20 trees for quick preselection
                if num_features > 50:
                    print(f"DEBUG: Stage 1: Fast preselection with 20 trees (all {num_features} features)...")
                    model_preselect = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1, max_depth=10)
                    model_preselect.fit(X_train_full, y_train)
                    
                    # Get importance from preselection
                    importance_preselect = model_preselect.feature_importances_
                    
                    # Select top 50 features for detailed calculation
                    top_n_preselect = min(50, num_features)
                    s_id_preselect = np.argsort(importance_preselect)[-top_n_preselect:]
                    selected_features = X_train_full.columns[s_id_preselect].tolist()
                    
                    print(f"DEBUG: Stage 1 complete: Selected top {top_n_preselect} features for detailed calculation")
                    
                    # STAGE 2: Detailed calculation with more trees (only top features)
                    X_train_selected = X_train_full[selected_features]
                    print(f"DEBUG: Stage 2: Detailed calculation with 100 trees (top {top_n_preselect} features)...")
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
                    model.fit(X_train_selected, y_train)
                    
                    # Get feature importance (only for selected features)
                    importance_selected = model.feature_importances_
                    
                    # Map back to full feature set
                    importance = np.zeros(num_features)
                    for i, feat_idx in enumerate(s_id_preselect):
                        importance[feat_idx] = importance_selected[i]
                    
                    print(f"DEBUG: Stage 2 complete: Calculated importance for top {top_n_preselect} features")
                else:
                    # If <= 50 features, skip preselection and use full calculation
                    print(f"DEBUG: {num_features} features <= 50, skipping preselection, using full calculation...")
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
                    model.fit(X_train_full, y_train)
                    importance = model.feature_importances_
                
                # Sort and get top 10
                s_id = np.argsort(importance)
                top_n = 10
                s_id = s_id[-top_n:]
                
                # MinMax scaling
                scaler = MinMaxScaler()
                importance_scaled = scaler.fit_transform(importance.reshape(-1, 1)).flatten()
                total_importance = np.sum(importance_scaled)
                percentage_importance = (importance_scaled / total_importance) * 100
                
                # Create DataFrame matching MM.feature_importance output format
                feature_names = X_train_full.columns[s_id]
                features_x = pd.DataFrame({
                    "index1": feature_names,
                    "importance_normalized": importance_scaled[s_id],
                    "percentage_importance": percentage_importance[s_id]
                })
                
                print(f"DEBUG: X importance calculated successfully: {len(features_x)} features")
                if not features_x.empty:
                    print(f"DEBUG: X importance columns: {features_x.columns.tolist()}")
                    print(f"DEBUG: X importance head:\n{features_x.head()}")
                
                # Force garbage collection after X calculation
                gc.collect()
                sys.stdout.flush()
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print("=" * 80)
                print(f"ERROR calculating X importance: {e}")
                print(f"ERROR traceback:\n{error_trace}")
                print("=" * 80)
                import sys
                sys.stdout.flush()
                features_x = pd.DataFrame()
                gc.collect()  # Clean up on error
            
            # Force garbage collection before Y calculation
            gc.collect()
            
            # Calculate feature importance for Y dimension
            features_y = pd.DataFrame()
            try:
                print("=" * 80)
                print("DEBUG: Starting Y importance calculation...")
                print(f"DEBUG: MM.df shape before feature_importance: {mm.df.shape}")
                
                # Additional validation: check for matrix singularity issues
                x_var = mm.df['x'].var()
                y_var = mm.df['y'].var()
                print(f"DEBUG: X variance: {x_var:.6f}, Y variance: {y_var:.6f}")
                
                print(f"DEBUG: Calculating Y importance manually (no multiprocessing)...")
                import sys
                sys.stdout.flush()  # Force output to console
                
                # TWO-STAGE APPROACH: Fast preselection + detailed calculation
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import MinMaxScaler
                
                # Prepare data
                data = mm.df.copy()
                data = data.drop('x', axis=1)  # Drop indep
                
                # Split data
                train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
                X_train_full = train_df.drop('y', axis=1)  # Drop dep
                y_train = train_df['y']
                num_features = len(X_train_full.columns)
                
                print(f"DEBUG: Starting two-stage calculation for {num_features} features...")
                
                # STAGE 1: Fast preselection with fewer trees (all features)
                # Use only 20 trees for quick preselection
                if num_features > 50:
                    print(f"DEBUG: Stage 1: Fast preselection with 20 trees (all {num_features} features)...")
                    model_preselect = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1, max_depth=10)
                    model_preselect.fit(X_train_full, y_train)
                    
                    # Get importance from preselection
                    importance_preselect = model_preselect.feature_importances_
                    
                    # Select top 50 features for detailed calculation
                    top_n_preselect = min(50, num_features)
                    s_id_preselect = np.argsort(importance_preselect)[-top_n_preselect:]
                    selected_features = X_train_full.columns[s_id_preselect].tolist()
                    
                    print(f"DEBUG: Stage 1 complete: Selected top {top_n_preselect} features for detailed calculation")
                    
                    # STAGE 2: Detailed calculation with more trees (only top features)
                    X_train_selected = X_train_full[selected_features]
                    print(f"DEBUG: Stage 2: Detailed calculation with 100 trees (top {top_n_preselect} features)...")
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
                    model.fit(X_train_selected, y_train)
                    
                    # Get feature importance (only for selected features)
                    importance_selected = model.feature_importances_
                    
                    # Map back to full feature set
                    importance = np.zeros(num_features)
                    for i, feat_idx in enumerate(s_id_preselect):
                        importance[feat_idx] = importance_selected[i]
                    
                    print(f"DEBUG: Stage 2 complete: Calculated importance for top {top_n_preselect} features")
                else:
                    # If <= 50 features, skip preselection and use full calculation
                    print(f"DEBUG: {num_features} features <= 50, skipping preselection, using full calculation...")
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
                    model.fit(X_train_full, y_train)
                    importance = model.feature_importances_
                
                # Sort and get top 10
                s_id = np.argsort(importance)
                top_n = 10
                s_id = s_id[-top_n:]
                
                # MinMax scaling
                scaler = MinMaxScaler()
                importance_scaled = scaler.fit_transform(importance.reshape(-1, 1)).flatten()
                total_importance = np.sum(importance_scaled)
                percentage_importance = (importance_scaled / total_importance) * 100
                
                # Create DataFrame matching MM.feature_importance output format
                feature_names = X_train_full.columns[s_id]
                features_y = pd.DataFrame({
                    "index1": feature_names,
                    "importance_normalized": importance_scaled[s_id],
                    "percentage_importance": percentage_importance[s_id]
                })
                
                print(f"DEBUG: Y importance calculated successfully: {len(features_y)} features")
                if not features_y.empty:
                    print(f"DEBUG: Y importance columns: {features_y.columns.tolist()}")
                    print(f"DEBUG: Y importance head:\n{features_y.head()}")
                
                # Force garbage collection after Y calculation
                gc.collect()
                sys.stdout.flush()
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print("=" * 80)
                print(f"ERROR calculating Y importance: {e}")
                print(f"ERROR traceback:\n{error_trace}")
                print("=" * 80)
                import sys
                sys.stdout.flush()
                features_y = pd.DataFrame()
                gc.collect()  # Clean up on error
            
            if features_x.empty and features_y.empty:
                self.finished.emit(False, "Could not calculate feature importance. Check console for detailed error messages.")
                return
            
            # Ensure results directory exists
            self.paths["results"].mkdir(parents=True, exist_ok=True)
            
            # Save CSV with combined results
            output_data = []
            if not features_x.empty:
                for idx, row in features_x.iterrows():
                    output_data.append({
                        "Feature": row["index1"],
                        "Dimension": "X",
                        "Importance_Normalized": row["importance_normalized"],
                        "Percentage_Importance": row["percentage_importance"]
                    })
            
            if not features_y.empty:
                for idx, row in features_y.iterrows():
                    output_data.append({
                        "Feature": row["index1"],
                        "Dimension": "Y",
                        "Importance_Normalized": row["importance_normalized"],
                        "Percentage_Importance": row["percentage_importance"]
                    })
            
            if output_data:
                output_df = pd.DataFrame(output_data)
                csv_path = self.paths["results"] / "top10_features.csv"
                output_df.to_csv(csv_path, index=False)
                
                # Create plots for both dimensions
                plot_path_x = self.paths["results"] / "top10_features_x.png"
                plot_path_y = self.paths["results"] / "top10_features_y.png"
                
                # Store plot data in session_state for main thread (Matplotlib cannot be used in worker thread)
                # The plots will be created in the main thread after finished signal
                self.session_state["feature_importance_plot_data"] = {
                    "features_x": features_x.to_dict('records'),
                    "features_y": features_y.to_dict('records'),
                    "plot_path_x": str(plot_path_x),
                    "plot_path_y": str(plot_path_y)
                }
                
                self.finished.emit(True, f"✅ Top10 Features saved to:\n{csv_path}\n\nPlots will be saved to:\n{plot_path_x}\n{plot_path_y}")
            else:
                self.finished.emit(False, "No feature importance data calculated.")
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"FeatureImportanceWorker error: {str(e)}\n{error_trace}")
            self.finished.emit(False, f"❌ Error: {str(e)[:200]}")


class ClusteringWorker(QThread):
    """Worker thread for clustering analysis."""
    finished = Signal(bool, str)  # success, message
    
    def __init__(self, embedding_df: pd.DataFrame, method: str, n_clusters: int, session_state: Dict, method_params: Optional[Dict] = None):
        super().__init__()
        self.embedding_df = embedding_df
        self.method = method
        self.n_clusters = n_clusters
        self.session_state = session_state
        self.method_params = method_params or {}
    
    def run(self):
        try:
            coords = self.embedding_df[["x", "y"]].values
            cluster_labels = run_clustering_analysis(coords, self.method, self.n_clusters, self.method_params)
            
            self.session_state["cluster_labels"] = cluster_labels
            self.session_state["cluster_method"] = self.method
            self.session_state["cluster_param"] = self.n_clusters
            
            n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            self.finished.emit(True, f"✅ Clustering completed: {n_clusters_found} clusters found")
            
        except Exception as e:
            self.finished.emit(False, f"❌ Clustering failed: {str(e)[:100]}")


# ============================================================================
# Main Window
# ============================================================================

class MorphoMappingGUI(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MorphoMapping - Interactive Analysis for Imaging Flow Cytometry")
        self.setMinimumSize(1200, 800)
        
        # Session state
        self.session_state: Dict = {
            "run_id": None,
            "project_dir": str(PROJECT_ROOT),
            "embedding_df": None,
            "cluster_labels": None,
            "features": [],
            "selected_population": None,
            "metadata_df": pd.DataFrame(),
            "stored_dim_reduction_method": None,
            "cluster_method": None,
            "cluster_param": 10,
            "uploaded_files": [],
            "processing_status": {},
            "highlighted_cells": [],
        }
        
        # Initialize run_id with default
        if not self.session_state.get("run_id"):
            self.session_state["run_id"] = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        
        # Feature selection state
        self.feature_selection_state = {
            "available_features": [],
            "selected_features": [],
            "excluded_features": [],
            "populations": [],
        }
        
        # Track active worker threads to ensure proper cleanup
        self.active_workers: List[QThread] = []
        
        # Track conversion completion to show message only once
        self._conversion_complete_shown = False
        self._total_files_to_convert = 0
        
        # Setup UI
        self.setup_ui()
        
        # Setup timers for status updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)  # Update every 5 seconds
        
        # Timer to check for new FCS files and update metadata
        self.metadata_update_timer = QTimer()
        self.metadata_update_timer.timeout.connect(self.check_and_update_metadata)
        self.metadata_update_timer.start(3000)  # Check every 3 seconds
        self.last_fcs_count = 0  # Track FCS file count
    
    def check_and_update_metadata(self):
        """Check if new FCS files are available and update metadata table."""
        if not self.session_state.get("run_id"):
            return
        
        try:
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            current_fcs_count = len(list(paths["fcs"].glob("*.fcs")))
            
            # Only update if FCS count changed
            if current_fcs_count != self.last_fcs_count:
                self.last_fcs_count = current_fcs_count
                self.update_metadata_display()
        except Exception:
            pass  # Silently ignore errors during periodic check
    
    def closeEvent(self, event):
        """Handle window close event - ensure all threads are properly terminated."""
        # Stop all timers
        if hasattr(self, 'status_timer'):
            self.status_timer.stop()
        if hasattr(self, 'metadata_update_timer'):
            self.metadata_update_timer.stop()
        
        # Wait for all worker threads to finish
        for worker in self.active_workers:
            if worker.isRunning():
                worker.quit()  # Request thread to stop
                worker.wait(5000)  # Wait up to 5 seconds for thread to finish
                if worker.isRunning():
                    worker.terminate()  # Force termination if still running
                    worker.wait(1000)  # Wait a bit more
        
        self.active_workers.clear()
        event.accept()
    
    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header with logo
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Scroll area for main content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)
        
        # 1. Project Setup & Status
        project_section = self.create_project_section()
        content_layout.addWidget(project_section)
        
        # 2. DAF File Selection
        daf_section = self.create_daf_section()
        content_layout.addWidget(daf_section)
        
        # 3. Metadata
        metadata_section = self.create_metadata_section()
        content_layout.addWidget(metadata_section)
        
        # 4. Features & Gates Selection
        features_section = self.create_features_section()
        content_layout.addWidget(features_section)
        
        # 5. Dimensionality Reduction
        dr_section = self.create_dimensionality_reduction_section()
        content_layout.addWidget(dr_section)
        
        # 6. Visualization & Clustering (combined, side by side)
        self.viz_cluster_section = self.create_visualization_clustering_section()
        content_layout.addWidget(self.viz_cluster_section)
        
        content_layout.addStretch()
        
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
    
    def create_header(self) -> QWidget:
        """Create header with logo."""
        header = QWidget()
        header.setStyleSheet("background-color: #E3F2FD;")
        header.setMinimumHeight(110)
        header.setMaximumHeight(110)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(15, 5, 15, 5)  # Reduced margins
        layout.setSpacing(15)
        
        # Logo with explicit frame to prevent cropping
        logo_path = BUNDLE_ROOT / "assets" / "logo.png"
        if logo_path.exists():
            logo_frame = QFrame()
            logo_frame.setFrameShape(QFrame.NoFrame)
            logo_frame.setFixedSize(100, 100)
            logo_frame.setStyleSheet("background-color: transparent;")
            
            logo_layout = QVBoxLayout(logo_frame)
            logo_layout.setContentsMargins(5, 5, 5, 5)  # Small margins inside frame
            logo_layout.setSpacing(0)
            
            logo_label = QLabel()
            pixmap = QPixmap(str(logo_path))
            # Scale to 90x90 to leave margin inside 100x100 frame
            scaled_pixmap = pixmap.scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setScaledContents(False)
            logo_label.setFixedSize(90, 90)  # Explicit fixed size
            logo_layout.addWidget(logo_label, 0, Qt.AlignCenter)
            
            layout.addWidget(logo_frame, 0, Qt.AlignLeft | Qt.AlignVCenter)
        
        # Title - centered in container
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("MorphoMapping")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #1976D2;")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        
        subtitle = QLabel("Interactive Analysis for Imaging Flow Cytometry")
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet("color: #666;")
        subtitle.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(subtitle)
        
        # Add stretch before and after for centering
        layout.addStretch()
        layout.addWidget(title_container, 1)  # Allow stretching for centering
        layout.addStretch()
        
        # Help button on the right
        help_btn = QPushButton("📖 Help")
        help_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 15px; font-size: 12px;")
        help_btn.clicked.connect(self.show_documentation)
        layout.addWidget(help_btn, 0, Qt.AlignRight | Qt.AlignVCenter)
        
        return header
    
    def create_project_section(self) -> QGroupBox:
        """Create project setup and status section."""
        group = QGroupBox("1️⃣ Project Setup & Status")
        group.setFont(QFont("Arial", 12, QFont.Bold))
        
        layout = QHBoxLayout()
        layout.setSpacing(15)
        
        # Left: Project Setup
        setup_group = QGroupBox("Project Setup")
        setup_layout = QVBoxLayout()
        
        project_dir_input = QLineEdit()
        project_dir_input.setText(self.session_state["project_dir"])
        project_dir_input.setPlaceholderText(str(PROJECT_ROOT))
        setup_layout.addWidget(QLabel("Project Directory:"))
        setup_layout.addWidget(project_dir_input)
        
        self.run_id_input = QLineEdit()
        self.run_id_input.setText(self.session_state["run_id"])
        setup_layout.addWidget(QLabel("Run-ID:"))
        setup_layout.addWidget(self.run_id_input)
        
        def update_project():
            self.session_state["project_dir"] = project_dir_input.text()
            self.session_state["run_id"] = self.run_id_input.text()
            QMessageBox.information(self, "Success", f"✅ Project set: {self.run_id_input.text()}")
            self.update_status()
        
        set_btn = QPushButton("💾 Set Project")
        set_btn.clicked.connect(update_project)
        set_btn.setStyleSheet("background-color: #1976D2; color: white; padding: 8px;")
        setup_layout.addWidget(set_btn)
        
        # Load previous run button
        load_run_btn = QPushButton("📂 Load Previous Run")
        load_run_btn.clicked.connect(self.load_previous_run)
        load_run_btn.setStyleSheet("background-color: #388E3C; color: white; padding: 8px;")
        setup_layout.addWidget(load_run_btn)
        
        setup_group.setLayout(setup_layout)
        layout.addWidget(setup_group, 1)
        
        # Right: Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_container = QWidget()
        self.status_layout = QVBoxLayout(self.status_container)
        self.status_layout.setSpacing(5)
        
        status_scroll = QScrollArea()
        status_scroll.setWidget(self.status_container)
        status_scroll.setWidgetResizable(True)
        status_scroll.setMaximumHeight(200)
        status_scroll.setFrameShape(QFrame.NoFrame)
        
        status_layout.addWidget(status_scroll)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group, 1)
        
        group.setLayout(layout)
        return group
    
    def update_status(self):
        """Update status display and save to file."""
        # Clear existing status
        for i in reversed(range(self.status_layout.count())):
            item = self.status_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        
        # Collect status information for saving
        status_data = {}
        
        # Run-ID
        run_id = self.session_state.get("run_id", "Not set")
        run_id_label = QLabel(f"🆔 Run-ID: {run_id}")
        run_id_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.status_layout.addWidget(run_id_label)
        status_data["run_id"] = run_id
        
        if self.session_state.get("run_id"):
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            counts = get_file_counts(paths)
            
            daf_label = QLabel(f"📁 DAF files: {counts['daf']}")
            self.status_layout.addWidget(daf_label)
            status_data["daf_files"] = counts['daf']
            
            fcs_label = QLabel(f"📊 FCS files: {counts['fcs']}")
            self.status_layout.addWidget(fcs_label)
            status_data["fcs_files"] = counts['fcs']
            
            # Metadata status
            # Check if metadata file exists (was actually saved)
            metadata_path = paths["metadata"] / "sample_sheet.csv"
            if metadata_path.exists():
                label = QLabel("📋 Metadata: ✅ Saved")
                label.setStyleSheet("color: green;")
                status_data["metadata"] = "saved"
            else:
                metadata_df = self.session_state.get("metadata_df")
                if metadata_df is not None and len(metadata_df) > 0:
                    has_content = False
                    if "sample_id" in metadata_df.columns:
                        has_content = any(metadata_df["sample_id"].astype(str).str.strip() != "")
                    if not has_content and "group" in metadata_df.columns:
                        has_content = any(metadata_df["group"].astype(str).str.strip() != "")
                    
                    if has_content:
                        label = QLabel("📋 Metadata: ⚠️ Not saved (click Save)")
                        label.setStyleSheet("color: orange;")
                        status_data["metadata"] = "not_saved"
                    else:
                        label = QLabel("📋 Metadata: ❌ Not set")
                        label.setStyleSheet("color: red;")
                        status_data["metadata"] = "not_set"
                else:
                    label = QLabel("📋 Metadata: ❌ Not set")
                    label.setStyleSheet("color: red;")
                    status_data["metadata"] = "not_set"
            self.status_layout.addWidget(label)
            
            # Features status
            features_count = len(self.session_state.get("features", []))
            if features_count > 0:
                label = QLabel(f"📊 Features: {features_count} selected")
                label.setStyleSheet("color: green;")
                status_data["features_count"] = features_count
            else:
                label = QLabel("📊 Features: Not selected")
                label.setStyleSheet("color: gray;")
                status_data["features_count"] = 0
            self.status_layout.addWidget(label)
            
            # Population status
            pop = self.session_state.get("selected_population")
            if pop:
                pop_label = QLabel(f"🔬 Population: {pop}")
                self.status_layout.addWidget(pop_label)
                status_data["population"] = pop
            else:
                pop_label = QLabel("🔬 Population: All events")
                self.status_layout.addWidget(pop_label)
                status_data["population"] = None
            
            # Channel Filter status
            if hasattr(self, 'excluded_channels') and self.excluded_channels:
                excluded_str = ', '.join(self.excluded_channels)
                channel_label = QLabel(f"🔬 Channels: ⚠️ Excluded: {excluded_str}")
                channel_label.setStyleSheet("color: orange;")
                self.status_layout.addWidget(channel_label)
                status_data["excluded_channels"] = self.excluded_channels
            else:
                channel_label = QLabel("🔬 Channels: ✅ All included")
                channel_label.setStyleSheet("color: green;")
                self.status_layout.addWidget(channel_label)
                status_data["excluded_channels"] = []
            
            # Analysis status
            if self.session_state.get("embedding_df") is not None:
                method = self.session_state.get("stored_dim_reduction_method", "Unknown")
                label = QLabel(f"📈 Analysis: ✅ {method} completed")
                label.setStyleSheet("color: green;")
                self.status_layout.addWidget(label)
                status_data["analysis"] = {"method": method, "status": "completed"}
                
                # Show file processing info
                analysis_info = self.session_state.get("analysis_info", {})
                usable_files = analysis_info.get("usable_files", 0)
                skipped_files = analysis_info.get("skipped_files", [])
                
                if skipped_files:
                    skipped_info = []
                    for file_name, reasons in skipped_files:
                        reason_str = "; ".join(reasons)
                        skipped_info.append(f"{file_name}: {reason_str}")
                    skipped_text = "\n".join(skipped_info[:5])  # Show first 5
                    if len(skipped_info) > 5:
                        skipped_text += f"\n... and {len(skipped_info) - 5} more"
                    files_label = QLabel(f"📁 Files: ✅ {usable_files} usable, ⚠️ {len(skipped_files)} with notes")
                    files_label.setStyleSheet("color: orange; font-size: 10px;")
                    files_label.setWordWrap(True)
                    files_label.setToolTip(skipped_text)
                    self.status_layout.addWidget(files_label)
                    status_data["file_info"] = {"usable": usable_files, "skipped": len(skipped_files), "skipped_details": skipped_files}
                else:
                    files_label = QLabel(f"📁 Files: ✅ {usable_files} usable")
                    files_label.setStyleSheet("color: green; font-size: 10px;")
                    self.status_layout.addWidget(files_label)
                    status_data["file_info"] = {"usable": usable_files, "skipped": 0}
            else:
                label = QLabel("📈 Analysis: ❌ Not run")
                label.setStyleSheet("color: red;")
                self.status_layout.addWidget(label)
                status_data["analysis"] = {"status": "not_run"}
            
            # Clustering status
            if self.session_state.get("cluster_labels") is not None:
                cluster_method = self.session_state.get("cluster_method", "Unknown")
                n_clusters = len(set(self.session_state["cluster_labels"])) - (1 if -1 in self.session_state["cluster_labels"] else 0)
                label = QLabel(f"🔬 Clustering: ✅ {cluster_method} ({n_clusters} clusters)")
                label.setStyleSheet("color: green;")
                self.status_layout.addWidget(label)
                status_data["clustering"] = {"method": cluster_method, "n_clusters": n_clusters, "status": "completed"}
            else:
                label = QLabel("🔬 Clustering: ❌ Not run")
                label.setStyleSheet("color: red;")
                self.status_layout.addWidget(label)
                status_data["clustering"] = {"status": "not_run"}
            
            # Save status to file for later review
            try:
                status_path = paths["base"] / "run_status.json"
                # Ensure directory exists before writing
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_data["timestamp"] = datetime.datetime.now().isoformat()
                status_data["project_dir"] = project_dir
                status_path.write_text(json.dumps(status_data, indent=2))
            except Exception as e:
                print(f"Warning: Could not save status: {e}")
        
        self.status_layout.addStretch()
    
    def _save_run_info(self, paths: Dict[str, Path], daf_files: Optional[List[Dict]] = None, 
                       dim_reduction: Optional[Dict] = None, clustering: Optional[Dict] = None):
        """Save comprehensive run information to run_info.json.
        
        This file contains all information about the run:
        - DAF files used (original paths, not copies)
        - Dimensionality reduction settings (DensMAP, UMAP, etc.)
        - Clustering settings (method, parameters, etc.)
        - All other run metadata
        """
        try:
            info_path = paths["base"] / "run_info.json"
            
            # Load existing info if it exists
            if info_path.exists():
                try:
                    existing_info = json.loads(info_path.read_text())
                except:
                    existing_info = {}
            else:
                existing_info = {}
            
            # Update with new information
            if daf_files is not None:
                existing_info["daf_files"] = daf_files
                existing_info["daf_files_count"] = len(daf_files)
            
            if dim_reduction is not None:
                existing_info["dimensionality_reduction"] = dim_reduction
            
            if clustering is not None:
                existing_info["clustering"] = clustering
            
            # Always update metadata
            existing_info["run_id"] = self.session_state.get("run_id", "Unknown")
            existing_info["project_dir"] = str(paths["base"].parent.parent)
            existing_info["timestamp"] = datetime.datetime.now().isoformat()
            
            # Save features and population if available
            if "features" not in existing_info and self.session_state.get("features"):
                existing_info["features"] = self.session_state["features"]
                existing_info["features_count"] = len(self.session_state["features"])
            
            if "population" not in existing_info and self.session_state.get("selected_population"):
                existing_info["population"] = self.session_state["selected_population"]
            
            # Save to file
            info_path.parent.mkdir(parents=True, exist_ok=True)
            info_path.write_text(json.dumps(existing_info, indent=2))
            
        except Exception as e:
            print(f"Warning: Could not save run info: {e}")
    
    def load_previous_run(self):
        """Load a previous run from disk."""
        try:
            # Find all available runs
            project_dir = safe_path(self.session_state.get("project_dir", PROJECT_ROOT))
            bundle_runs_dir = project_dir / "bundle_runs"
            
            if not bundle_runs_dir.exists():
                QMessageBox.warning(self, "No Runs Found", "No previous runs found. Please run an analysis first.")
                return
            
            # Find all run directories
            run_dirs = sorted([d for d in bundle_runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], 
                            reverse=True)  # Most recent first
            
            if not run_dirs:
                QMessageBox.warning(self, "No Runs Found", "No previous runs found. Please run an analysis first.")
                return
            
            # Create dialog to select run
            dialog = QDialog(self)
            dialog.setWindowTitle("Load Previous Run")
            dialog.setMinimumWidth(600)
            layout = QVBoxLayout()
            
            layout.addWidget(QLabel("Select a run to load:"))
            
            # List widget with run information
            list_widget = QListWidget()
            list_widget.setMinimumHeight(300)
            
            for run_dir in run_dirs:
                run_id = run_dir.name
                info_path = run_dir / "run_info.json"
                
                # Try to load run info
                run_info = {}
                if info_path.exists():
                    try:
                        run_info = json.loads(info_path.read_text())
                    except:
                        pass
                
                # Create display text
                timestamp = run_info.get("timestamp", "Unknown")
                method = run_info.get("dimensionality_reduction", {}).get("method", "Unknown")
                n_files = run_info.get("daf_files_count", 0)
                has_clustering = "clustering" in run_info
                
                display_text = f"{run_id}\n"
                display_text += f"  Method: {method} | Files: {n_files} | Clustering: {'Yes' if has_clustering else 'No'}\n"
                display_text += f"  Date: {timestamp[:19] if len(timestamp) > 19 else timestamp}"
                
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, run_id)
                list_widget.addItem(item)
            
            layout.addWidget(list_widget)
            
            # Buttons
            button_layout = QHBoxLayout()
            load_btn = QPushButton("Load")
            load_btn.setStyleSheet("background-color: #388E3C; color: white; padding: 5px;")
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setStyleSheet("background-color: #757575; color: white; padding: 5px;")
            
            def on_load():
                selected_items = list_widget.selectedItems()
                if not selected_items:
                    QMessageBox.warning(dialog, "No Selection", "Please select a run to load.")
                    return
                
                selected_run_id = selected_items[0].data(Qt.UserRole)
                dialog.accept()
                self._load_run_data(selected_run_id)
            
            load_btn.clicked.connect(on_load)
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addWidget(load_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            
            if dialog.exec() == QDialog.Accepted:
                # Already handled in on_load
                pass
                
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error loading previous run: {e}\n{error_trace}")
            QMessageBox.critical(self, "Error", f"❌ Error loading run: {str(e)[:200]}")
    
    def _load_run_data(self, run_id: str):
        """Load run data from disk."""
        try:
            project_dir = safe_path(self.session_state.get("project_dir", PROJECT_ROOT))
            paths = get_run_paths(project_dir, run_id)
            
            # Load run info
            info_path = paths["base"] / "run_info.json"
            if not info_path.exists():
                QMessageBox.critical(self, "Error", f"Run info not found: {info_path}")
                return
            
            run_info = json.loads(info_path.read_text())
            
            # Update session state with run info
            self.session_state["run_id"] = run_id
            self.session_state["project_dir"] = str(project_dir)
            
            # Load embedding
            dim_reduction_info = run_info.get("dimensionality_reduction", {})
            method = dim_reduction_info.get("method", "DensMAP")
            method_name_map = {"DensMAP": "densmap", "UMAP": "umap", "t-SNE": "tsne"}
            method_name = method_name_map.get(method, "densmap")
            
            embedding_path = paths["results"] / f"{method_name}_embedding.parquet"
            if not embedding_path.exists():
                QMessageBox.critical(self, "Error", f"Embedding file not found: {embedding_path}")
                return
            
            # Try to read parquet, fallback to CSV if needed
            try:
                embedding_df = pd.read_parquet(embedding_path)
            except Exception as e:
                print(f"Warning: Could not read parquet, trying CSV: {e}")
                # Fallback: try to find CSV version
                csv_path = paths["results"] / f"{method_name}_embedding.csv"
                if csv_path.exists():
                    embedding_df = pd.read_csv(csv_path)
                else:
                    raise ValueError(f"Could not load embedding from {embedding_path}")
            
            print(f"DEBUG: Loaded embedding with {len(embedding_df)} cells")
            
            # Ensure cell_index exists (needed for cluster label matching)
            if "cell_index" not in embedding_df.columns:
                embedding_df["cell_index"] = range(len(embedding_df))
                print("DEBUG: Added cell_index to embedding_df")
            
            # Load metadata if available
            metadata_path = paths["base"] / "metadata" / "sample_sheet.csv"
            metadata_df = pd.DataFrame()
            if metadata_path.exists():
                metadata_df = pd.read_csv(metadata_path)
                print(f"DEBUG: Loaded metadata with {len(metadata_df)} samples")
            
            # Merge metadata with embedding
            if not metadata_df.empty and "file_name" in metadata_df.columns:
                # Create file_name_clean for matching
                embedding_df["file_name_clean"] = embedding_df["sample_id"].astype(str)
                metadata_df["file_name_clean"] = metadata_df["file_name"].astype(str)
                
                embedding_df = embedding_df.merge(
                    metadata_df,
                    on="file_name_clean",
                    how="left",
                    suffixes=("", "_meta")
                )
                # Clean up duplicate columns
                if "sample_id_meta" in embedding_df.columns:
                    embedding_df = embedding_df.drop(columns=["sample_id_meta"])
            
            # Load cluster labels if available
            cluster_labels = None
            cluster_labels_path = paths["results"] / "cluster_labels.csv"
            if cluster_labels_path.exists():
                cluster_df = pd.read_csv(cluster_labels_path)
                # Ensure cluster labels match embedding_df length
                if len(cluster_df) == len(embedding_df):
                    cluster_labels = cluster_df["cluster"].values
                    print(f"DEBUG: Loaded cluster labels for {len(cluster_labels)} cells")
                else:
                    print(f"WARNING: Cluster labels length ({len(cluster_df)}) doesn't match embedding ({len(embedding_df)})")
                    # Try to match by cell_index if available
                    if "cell_index" in cluster_df.columns and "cell_index" in embedding_df.columns:
                        cluster_df_merged = embedding_df[["cell_index"]].merge(
                            cluster_df[["cell_index", "cluster"]],
                            on="cell_index",
                            how="left"
                        )
                        cluster_labels = cluster_df_merged["cluster"].values
                        print(f"DEBUG: Matched cluster labels by cell_index")
                    else:
                        print(f"WARNING: Could not match cluster labels, skipping")
            
            # Update session state
            self.session_state["embedding_df"] = embedding_df
            self.session_state["metadata_df"] = metadata_df
            self.session_state["stored_dim_reduction_method"] = method
            self.session_state["features"] = run_info.get("features", [])
            self.session_state["selected_population"] = run_info.get("population")
            
            if cluster_labels is not None:
                self.session_state["cluster_labels"] = cluster_labels
                clustering_info = run_info.get("clustering", {})
                self.session_state["cluster_method"] = clustering_info.get("method", "Unknown")
                self.session_state["cluster_param"] = clustering_info.get("n_clusters", 10)
            
            # Update UI
            if hasattr(self, 'run_id_input'):
                self.run_id_input.setText(run_id)
            
            # Update visualizations
            self.update_visualization()
            if cluster_labels is not None and hasattr(self, 'update_cluster_plot'):
                self.update_cluster_plot()
            
            QMessageBox.information(self, "Success", 
                f"✅ Run loaded successfully!\n\n"
                f"Method: {method}\n"
                f"Cells: {len(embedding_df)}\n"
                f"Clustering: {'Yes' if cluster_labels is not None else 'No'}")
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error loading run data: {e}\n{error_trace}")
            QMessageBox.critical(self, "Error", f"❌ Error loading run data: {str(e)[:200]}")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event for DAF files."""
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith('.daf'):
                files.append(file_path)
        
        if files:
            self.process_daf_files(files)
    
    def process_daf_files(self, file_paths):
        """Process DAF files (called from both file dialog and drag-drop)."""
        if not self.session_state.get("run_id"):
            QMessageBox.warning(self, "Warning", "⚠️ Please set Run-ID first")
            return
        
        project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
        run_id = safe_str(self.session_state["run_id"])
        paths = get_run_paths(safe_path(project_dir), run_id)
        paths["raw_daf"].mkdir(parents=True, exist_ok=True)
        paths["fcs"].mkdir(parents=True, exist_ok=True)
        
        # Store original DAF file paths (don't copy to save disk space)
        daf_file_info = []
        files_to_convert = []
        
        # Check which files need conversion (caching)
        for file_path_str in file_paths:
            file_path = safe_path(file_path_str)
            file_name = safe_str(file_path.name)
            
            # Store original path instead of copying
            original_path = str(file_path)
            
            self.session_state["uploaded_files"].append({
                "name": file_name,
                "path": original_path,
                "size": file_path.stat().st_size,
            })
            
            daf_file_info.append({
                "name": file_name,
                "original_path": original_path,
                "size": file_path.stat().st_size,
            })
            
            # Check if FCS file already exists and is newer than DAF (caching)
            fcs_path = safe_path(paths["fcs"]) / f"{safe_str(file_path.stem)}.fcs"
            if fcs_path.exists():
                daf_mtime = file_path.stat().st_mtime
                fcs_mtime = fcs_path.stat().st_mtime
                if fcs_mtime > daf_mtime:
                    # Already converted, skip
                    self.processing_status.append(f"✅ {file_name} (already converted, skipped)")
                    continue
            
            files_to_convert.append((file_path, fcs_path, file_name))
        
        # Start parallel conversions
        if not files_to_convert:
            QMessageBox.information(self, "Info", "✅ All files are already converted!")
            self.update_status()
            return
        
        # Reset conversion completion flag
        self._conversion_complete_shown = False
        self._total_files_to_convert = len(files_to_convert)
        
        # Determine max parallel conversions (based on CPU cores, but limit to 4-8)
        import os
        try:
            cpu_count = os.cpu_count() or 4
        except:
            cpu_count = 4
        max_parallel = min(4, cpu_count, len(files_to_convert))  # Max 4 parallel to avoid memory issues
        
        # Initialize progress tracking
        self.conversion_progress.setVisible(True)
        self.conversion_progress.setRange(0, len(files_to_convert))
        self.conversion_progress.setValue(0)
        self.conversion_progress.setFormat(f"Converting 0/{len(files_to_convert)} files...")
        
        # Start conversions (with parallel limit)
        for i, (file_path, fcs_path, file_name) in enumerate(files_to_convert):
            # Wait if too many workers are active
            while len([w for w in self.active_workers if isinstance(w, ConversionWorker) and w.isRunning()]) >= max_parallel:
                QApplication.processEvents()
                time.sleep(0.1)
            
            job_id = f"convert_{file_name}_{datetime.datetime.now().timestamp()}"
            
            worker = ConversionWorker(file_path, fcs_path, job_id)
            worker.finished.connect(
                lambda job_id, success, msg, w=worker: self.on_conversion_finished(job_id, success, msg, w)
            )
            self.active_workers.append(worker)
            
            worker.start()
            self.processing_status.append(f"🔄 Converting {file_name}... ({i+1}/{len(files_to_convert)})")
        
        self.daf_files_label.setText(f"Selected {len(file_paths)} file(s) | Converting {len(files_to_convert)} file(s) in parallel (max {max_parallel})")
        
        # Save DAF file information to run info
        self._save_run_info(paths, daf_files=daf_file_info)
        
        self.update_status()
        self.load_features_and_gates()
    
    def create_daf_section(self) -> QGroupBox:
        """Create DAF file selection section."""
        group = QGroupBox("2️⃣ Select DAF Files")
        group.setFont(QFont("Arial", 12, QFont.Bold))
        
        layout = QVBoxLayout()
        
        # Drag and drop area
        drop_area = DropArea(self)
        drop_area.setText("📁 Drag & Drop DAF files here\nor click button below")
        drop_area.setAlignment(Qt.AlignCenter)
        drop_area.setStyleSheet("""
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 20px;
            background-color: #E8F5E9;
            color: #2E7D32;
            font-size: 14px;
        """)
        drop_area.setMinimumHeight(100)
        layout.addWidget(drop_area)
        
        # File selection button
        select_btn = QPushButton("📁 Select DAF Files...")
        select_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-size: 14px;")
        select_btn.clicked.connect(self.select_daf_files)
        layout.addWidget(select_btn)
        
        # Selected files list
        self.daf_files_label = QLabel("No files selected")
        self.daf_files_label.setWordWrap(True)
        layout.addWidget(self.daf_files_label)
        
        # Progress bar for conversion
        self.conversion_progress = QProgressBar()
        self.conversion_progress.setVisible(False)
        self.conversion_progress.setFormat("Converting: %p%")
        layout.addWidget(self.conversion_progress)
        
        # Processing status
        self.processing_status = QTextEdit()
        self.processing_status.setMaximumHeight(100)
        self.processing_status.setReadOnly(True)
        layout.addWidget(QLabel("Processing Status:"))
        layout.addWidget(self.processing_status)
        
        # Remove DAF files button
        remove_daf_btn = QPushButton("🗑️ Remove DAF Files...")
        remove_daf_btn.setStyleSheet("background-color: #F44336; color: white; padding: 8px; font-size: 12px;")
        remove_daf_btn.clicked.connect(self.remove_daf_files)
        layout.addWidget(remove_daf_btn)
        
        group.setLayout(layout)
        return group
    
    def select_daf_files(self):
        """Open file dialog to select DAF files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select DAF Files",
            "",
            "DAF Files (*.daf);;All Files (*)"
        )
        
        if files:
            self.process_daf_files(files)
    
    def remove_daf_files(self):
        """Remove uploaded DAF files."""
        if not self.session_state.get("run_id"):
            QMessageBox.warning(self, "Warning", "⚠️ Please set Run-ID first")
            return
        
        project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
        run_id = safe_str(self.session_state["run_id"])
        paths = get_run_paths(safe_path(project_dir), run_id)
        
        # Get list of DAF files
        daf_files = sorted(paths["raw_daf"].glob("*.daf"))
        if not daf_files:
            QMessageBox.information(self, "Info", "No DAF files found to remove.")
            return
        
        # Create dialog to select files to remove
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Remove DAF Files")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select DAF files to remove:"))
        
        file_list = QListWidget()
        file_list.setSelectionMode(QListWidget.MultiSelection)
        for daf_file in daf_files:
            file_list.addItem(daf_file.name)
        layout.addWidget(file_list)
        
        button_layout = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected")
        remove_btn.setStyleSheet("background-color: #F44336; color: white; padding: 8px;")
        remove_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(remove_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        if dialog.exec() == QDialog.Accepted:
            selected_items = file_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Warning", "No files selected.")
                return
            
            removed_count = 0
            for item in selected_items:
                file_name = item.text()
                daf_path = paths["raw_daf"] / file_name
                fcs_path = paths["fcs"] / f"{safe_str(daf_path.stem)}.fcs"
                
                # Remove DAF file
                if daf_path.exists():
                    daf_path.unlink()
                    removed_count += 1
                
                # Remove corresponding FCS file if it exists
                if fcs_path.exists():
                    fcs_path.unlink()
                
                # Remove from uploaded_files list
                self.session_state["uploaded_files"] = [
                    f for f in self.session_state["uploaded_files"] 
                    if f.get("name") != file_name
                ]
            
            QMessageBox.information(self, "Success", f"✅ Removed {removed_count} DAF file(s)")
            self.update_status()
            self.load_features_and_gates()
            self.update_metadata_display()
    
    def on_conversion_finished(self, job_id: str, success: bool, message: str, worker: ConversionWorker):
        """Handle conversion completion."""
        self.processing_status.append(message)
        
        # Update progress bar
        active_conversions = [w for w in self.active_workers if isinstance(w, ConversionWorker) and w.isRunning()]
        finished_conversions = [w for w in self.active_workers if isinstance(w, ConversionWorker) and not w.isRunning()]
        
        if self.conversion_progress.isVisible():
            total = self.conversion_progress.maximum()
            completed = len(finished_conversions)
            self.conversion_progress.setValue(completed)
            self.conversion_progress.setFormat(f"Converting {completed}/{total} files... ({len(active_conversions)} active)")
        
        # Hide progress bar if no more conversions running
        if not active_conversions:
            self.conversion_progress.setVisible(False)
            self.conversion_progress.setFormat("")
            # Show completion message only once
            if success and not self._conversion_complete_shown:
                self._conversion_complete_shown = True
                QMessageBox.information(self, "Conversion Complete", 
                    f"✅ All {self._total_files_to_convert} file(s) converted successfully!")
        
        if success:
            self.update_status()
            # Reload features after a short delay (only once when all done)
            if not active_conversions:
                QTimer.singleShot(2000, self.load_features_and_gates)
                QTimer.singleShot(2000, self.update_metadata_display)
        
        # Remove worker from active list once finished
        if worker in self.active_workers:
            self.active_workers.remove(worker)
    
    def create_metadata_section(self) -> QGroupBox:
        """Create metadata section."""
        group = QGroupBox("3️⃣ Metadata")
        group.setFont(QFont("Arial", 12, QFont.Bold))
        
        layout = QHBoxLayout()
        layout.setSpacing(15)
        
        # Left: Manual Entry
        manual_group = QGroupBox("✏️ Manual Entry")
        manual_layout = QVBoxLayout()
        
        self.metadata_table = QTableWidget()
        self.metadata_table.setColumnCount(4)
        self.metadata_table.setHorizontalHeaderLabels(["file_name", "sample_id", "group", "replicate"])
        self.metadata_table.horizontalHeader().setStretchLastSection(True)
        self.metadata_table.setMaximumHeight(200)  # Show only ~3 rows, rest scrollable
        manual_layout.addWidget(self.metadata_table)
        
        add_row_btn = QPushButton("➕ Add Row")
        add_row_btn.clicked.connect(self.add_metadata_row)
        manual_layout.addWidget(add_row_btn)
        
        save_btn = QPushButton("💾 Save Metadata")
        save_btn.setStyleSheet("background-color: #1976D2; color: white;")
        save_btn.clicked.connect(self.save_metadata)
        manual_layout.addWidget(save_btn)
        
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group, 1)
        
        # Right: Upload Metadata
        upload_group = QGroupBox("📄 Upload Metadata")
        upload_layout = QVBoxLayout()
        
        upload_btn = QPushButton("📄 Upload metadata (CSV or Excel)")
        upload_btn.clicked.connect(self.upload_metadata)
        upload_layout.addWidget(upload_btn)
        
        self.metadata_upload_status = QLabel("")
        upload_layout.addWidget(self.metadata_upload_status)
        
        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group, 1)
        
        group.setLayout(layout)
        
        # Initialize metadata display
        self.update_metadata_display()
        
        return group
    
    def update_metadata_display(self):
        """Update metadata table display."""
        # Get FCS file names for auto-fill
        fcs_files = []
        if self.session_state.get("run_id"):
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            fcs_files = sorted([safe_str(f.stem) for f in paths["fcs"].glob("*.fcs")])
        
        # Initialize or update metadata from FCS files
        if "metadata_df" not in self.session_state or self.session_state["metadata_df"] is None or len(self.session_state["metadata_df"]) == 0:
            if fcs_files:
                metadata_rows = []
                for idx, fcs_name in enumerate(fcs_files, start=1):
                    metadata_rows.append({
                        "file_name": fcs_name,
                        "sample_id": f"sample_{idx}",
                        "group": "",
                        "replicate": ""
                    })
                self.session_state["metadata_df"] = pd.DataFrame(metadata_rows)
            else:
                self.session_state["metadata_df"] = pd.DataFrame(columns=["file_name", "sample_id", "group", "replicate"])
        else:
            # Update existing metadata: add new FCS files (only if metadata was auto-generated)
            # Check if metadata looks like it was auto-generated (has empty group/replicate columns)
            is_auto_generated = False
            if "group" in self.session_state["metadata_df"].columns:
                non_empty_groups = self.session_state["metadata_df"]["group"].astype(str).str.strip()
                is_auto_generated = (non_empty_groups == "").all()
            
            existing_file_names = set()
            if "file_name" in self.session_state["metadata_df"].columns:
                # Normalize existing file_names (remove .daf/.fcs if present)
                existing_file_names = set(
                    self.session_state["metadata_df"]["file_name"]
                    .astype(str)
                    .str.replace(r'\.(daf|fcs)$', '', regex=True)
                    .dropna()
                    .values
                )
            
            # Normalize fcs_files for comparison (they're already stems, but be safe)
            fcs_files_normalized = [f.replace('.daf', '').replace('.fcs', '') for f in fcs_files]
            new_fcs_files = [f for f in fcs_files_normalized if f not in existing_file_names]
            
            # Only add new files if metadata was auto-generated (not from Excel upload)
            if new_fcs_files and is_auto_generated:
                max_sample_num = 0
                if "sample_id" in self.session_state["metadata_df"].columns:
                    for sid in self.session_state["metadata_df"]["sample_id"].astype(str).values:
                        if sid.startswith("sample_") and sid[7:].isdigit():
                            max_sample_num = max(max_sample_num, int(sid[7:]))
                
                new_rows = []
                for idx, fcs_name in enumerate(new_fcs_files, start=1):
                    new_rows.append({
                        "file_name": fcs_name,
                        "sample_id": f"sample_{max_sample_num + idx}",
                        "group": "",
                        "replicate": ""
                    })
                self.session_state["metadata_df"] = pd.concat([
                    self.session_state["metadata_df"],
                    pd.DataFrame(new_rows)
                ], ignore_index=True)
        
        # Update table
        df = self.session_state["metadata_df"]
        self.metadata_table.setRowCount(len(df))
        
        for idx in range(len(df)):
            for col_idx, col in enumerate(["file_name", "sample_id", "group", "replicate"]):
                if col not in df.columns:
                    df[col] = ""
                val = str(df.iloc[idx][col]) if pd.notna(df.iloc[idx][col]) else ""
                item = QTableWidgetItem(val)
                self.metadata_table.setItem(idx, col_idx, item)
    
    def add_metadata_row(self):
        """Add a new row to metadata table."""
        row_count = self.metadata_table.rowCount()
        self.metadata_table.insertRow(row_count)
        for col in range(4):
            self.metadata_table.setItem(row_count, col, QTableWidgetItem(""))
    
    def save_metadata(self):
        """Save metadata from table."""
        try:
            if not self.session_state.get("run_id"):
                QMessageBox.warning(self, "Warning", "⚠️ Please set Run-ID first")
                return
            
            # Collect data from table
            new_data = []
            for row in range(self.metadata_table.rowCount()):
                row_data = {
                    "file_name": self.metadata_table.item(row, 0).text() if self.metadata_table.item(row, 0) else "",
                    "sample_id": self.metadata_table.item(row, 1).text() if self.metadata_table.item(row, 1) else "",
                    "group": self.metadata_table.item(row, 2).text() if self.metadata_table.item(row, 2) else "",
                    "replicate": self.metadata_table.item(row, 3).text() if self.metadata_table.item(row, 3) else "",
                }
                if not row_data.get("sample_id") and row_data.get("file_name"):
                    row_data["sample_id"] = row_data["file_name"]
                new_data.append(row_data)
            
            self.session_state["metadata_df"] = pd.DataFrame(new_data)
            
            # Save to file
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            metadata_path = safe_path(paths["metadata"]) / "sample_sheet.csv"
            
            try:
                save_metadata_file(self.session_state["metadata_df"], metadata_path)
            except OSError as e:
                if e.errno == 28:  # No space left on device
                    import shutil
                    # Try to clean up temporary files
                    temp_dirs = list(paths["fcs"].glob(".tmp_*"))
                    for temp_dir in temp_dirs:
                        try:
                            if temp_dir.is_dir():
                                shutil.rmtree(temp_dir)
                                print(f"Cleaned up temporary directory: {temp_dir}")
                        except:
                            pass
                    
                    # Try again
                    try:
                        save_metadata_file(self.session_state["metadata_df"], metadata_path)
                    except OSError as e2:
                        if e2.errno == 28:
                            # Get free space
                            base_path = paths["base"]
                            free_space_gb = shutil.disk_usage(base_path).free / (1024**3)
                            QMessageBox.critical(
                                self,
                                "Disk Space Error",
                                f"❌ No space left on device!\n\n"
                                f"Path: {metadata_path.parent}\n\n"
                                f"Please free up disk space and try again.\n"
                                f"Current free space: {free_space_gb:.1f} GB"
                            )
                            return
                        raise
            
            QMessageBox.information(self, "Success", f"✅ Metadata saved: {len(new_data)} rows")
            self.update_status()
        except Exception as ex:
            QMessageBox.critical(self, "Error", f"❌ Save error: {str(ex)}")
    
    def upload_metadata(self):
        """Upload metadata file."""
        if not self.session_state.get("run_id"):
            QMessageBox.warning(self, "Warning", "⚠️ Please set Run-ID first")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Metadata",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            metadata_path = safe_path(paths["metadata"]) / "sample_sheet.csv"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load metadata file
            if file_path.endswith('.csv'):
                metadata_df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                metadata_df = pd.read_excel(file_path)
            else:
                QMessageBox.critical(self, "Error", "❌ Unsupported file format. Please use CSV or Excel.")
                return
            
            # VALIDATION 1: Check for required column
            if "file_name" not in metadata_df.columns:
                QMessageBox.critical(
                    self, 
                    "Missing Required Column", 
                    "❌ The metadata file MUST contain a 'file_name' column.\n\n"
                    "This column should contain the .daf filenames (with or without the .daf extension)."
                )
                return
            
            # Normalize file_name: remove .daf extension if present (for consistent matching)
            metadata_df["file_name"] = metadata_df["file_name"].astype(str).str.replace(r'\.daf$', '', regex=True)
            
            # Create sample_id from file_name if not present
            if "sample_id" not in metadata_df.columns:
                metadata_df["sample_id"] = metadata_df["file_name"].copy()
            
            # VALIDATION 2: Check for matches with loaded DAF files
            # Get list of loaded FCS files (from converted DAFs)
            fcs_dir = paths["fcs"]
            if fcs_dir.exists():
                fcs_files = [f.stem for f in fcs_dir.glob("*.fcs")]
                
                # Clean metadata file_names (remove .daf extension if present)
                metadata_df["file_name_clean"] = metadata_df["file_name"].astype(str).str.replace(r'\.daf$', '', regex=True)
                metadata_files = metadata_df["file_name_clean"].unique().tolist()
                
                # Find mismatches
                metadata_not_in_daf = set(metadata_files) - set(fcs_files)
                daf_not_in_metadata = set(fcs_files) - set(metadata_files)
                matched = set(metadata_files) & set(fcs_files)
                
                # Build warning message
                warnings = []
                if metadata_not_in_daf:
                    warnings.append(f"⚠️ Files in metadata but NOT in loaded DAF files ({len(metadata_not_in_daf)}):\n" + 
                                  "\n".join([f"  • {f}" for f in sorted(list(metadata_not_in_daf))[:5]]))
                    if len(metadata_not_in_daf) > 5:
                        warnings[-1] += f"\n  ... and {len(metadata_not_in_daf) - 5} more"
                
                if daf_not_in_metadata:
                    warnings.append(f"⚠️ Loaded DAF files NOT in metadata ({len(daf_not_in_metadata)}):\n" + 
                                  "\n".join([f"  • {f}" for f in sorted(list(daf_not_in_metadata))[:5]]))
                    if len(daf_not_in_metadata) > 5:
                        warnings[-1] += f"\n  ... and {len(daf_not_in_metadata) - 5} more"
                
                # Show validation dialog
                if warnings:
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Metadata Validation")
                    msg.setText(f"✅ Matched: {len(matched)} files\n\n" + "\n\n".join(warnings))
                    msg.setInformativeText(
                        "\n⚠️ Mismatched files will be excluded from analysis.\n"
                        "Only cells from matched files will be used.\n\n"
                        "Do you want to continue?"
                    )
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    msg.setDefaultButton(QMessageBox.No)
                    
                    if msg.exec() == QMessageBox.No:
                        return
                else:
                    QMessageBox.information(
                        self,
                        "Perfect Match",
                        f"✅ All {len(matched)} files match perfectly!\n\n"
                        "Metadata and DAF files are fully compatible."
                    )
            
            # REPLACE existing metadata with uploaded metadata (don't merge to avoid duplicates)
            # This ensures that uploaded Excel/CSV data takes precedence over auto-generated entries
            self.session_state["metadata_df"] = metadata_df.copy()
            
            # Save to standard location (with better error handling for disk space)
            try:
                # Ensure directory exists
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_df.to_csv(metadata_path, index=False)
            except OSError as e:
                if e.errno == 28:  # No space left on device
                    # Try to clean up temporary files first
                    import shutil
                    temp_dirs = list(metadata_path.parent.parent.glob(".tmp_*"))
                    for temp_dir in temp_dirs:
                        try:
                            if temp_dir.is_dir():
                                shutil.rmtree(temp_dir)
                                print(f"Cleaned up temporary directory: {temp_dir}")
                        except:
                            pass
                    
                    # Try again
                    try:
                        metadata_path.parent.mkdir(parents=True, exist_ok=True)
                        metadata_df.to_csv(metadata_path, index=False)
                    except OSError as e2:
                        if e2.errno == 28:
                            # Get free space
                            free_space_gb = shutil.disk_usage(metadata_path.parent.parent).free / (1024**3)
                            QMessageBox.critical(
                                self,
                                "Disk Space Error",
                                f"❌ No space left on device!\n\n"
                                f"Path: {metadata_path.parent}\n\n"
                                f"Please free up disk space and try again.\n"
                                f"Current free space: {free_space_gb:.1f} GB"
                            )
                            return
                        raise
            
            self.metadata_upload_status.setText(f"✅ Loaded: {len(metadata_df)} rows")
            self.metadata_upload_status.setStyleSheet("color: green;")
            self.update_metadata_display()
            self.update_status()
            
        except Exception as ex:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Metadata upload error: {ex}\n{error_trace}")
            QMessageBox.critical(self, "Error", f"❌ Metadata error: {str(ex)[:200]}")
    
    def create_features_section(self) -> QGroupBox:
        """Create features and gates selection section."""
        group = QGroupBox("4️⃣ Features & Gates Selection")
        group.setFont(QFont("Arial", 12, QFont.Bold))
        
        layout = QVBoxLayout()
        
        # Channel selection (new) - Only Ch01-Ch12, M01-M12 are automatically handled
        channel_group = QGroupBox("🔬 Channel Selection (Optional)")
        channel_layout = QVBoxLayout()
        channel_info = QLabel("Select channels to include. Corresponding M01-M12 features will be automatically excluded.")
        channel_info.setStyleSheet("color: gray; font-size: 10px;")
        channel_layout.addWidget(channel_info)
        
        # Create checkboxes for channels 1-12 only
        self.channel_checkboxes = {}
        channel_grid = QHBoxLayout()
        
        # Channels Ch01-Ch12 only
        for i in range(1, 13):
            ch_name = f"Ch{i:02d}"
            checkbox = QCheckBox(ch_name)
            checkbox.setChecked(True)  # Default: all checked
            # Don't connect - will use button to apply
            self.channel_checkboxes[ch_name] = checkbox
            channel_grid.addWidget(checkbox)
        
        channel_layout.addLayout(channel_grid)
        
        # Apply button
        apply_channel_btn = QPushButton("✅ Apply Channel Filter")
        apply_channel_btn.clicked.connect(self.apply_channel_filter)
        channel_layout.addWidget(apply_channel_btn)
        
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # Track excluded channels for status
        self.excluded_channels = []
        
        # Features container
        features_group = QGroupBox("📊 Features Selection")
        features_layout = QVBoxLayout()
        
        self.features_info_label = QLabel("Available: 0 features")
        features_layout.addWidget(self.features_info_label)
        
        # Two-column layout for Include/Exclude
        features_row = QHBoxLayout()
        
        # Included Features
        included_group = QGroupBox("✅ Included Features")
        included_layout = QVBoxLayout()
        
        self.included_scroll = QScrollArea()
        self.included_widget = QWidget()
        self.included_layout = QVBoxLayout(self.included_widget)
        self.included_layout.setSpacing(5)
        
        # First 10 features container
        self.included_first_10 = QWidget()
        self.included_first_10_layout = QVBoxLayout(self.included_first_10)
        self.included_first_10_layout.setSpacing(2)
        
        # Expansion for remaining features
        self.included_expansion_btn = QPushButton("Show remaining features")
        self.included_expansion_btn.setCheckable(True)
        self.included_expansion_btn.setVisible(False)
        self.included_expansion_widget = QWidget()
        self.included_expansion_layout = QVBoxLayout(self.included_expansion_widget)
        self.included_expansion_widget.setVisible(False)
        
        self.included_layout.addWidget(self.included_first_10)
        self.included_layout.addWidget(self.included_expansion_btn)
        self.included_layout.addWidget(self.included_expansion_widget)
        self.included_layout.addStretch()
        
        self.included_scroll.setWidget(self.included_widget)
        self.included_scroll.setWidgetResizable(True)
        self.included_scroll.setMaximumHeight(200)
        included_layout.addWidget(self.included_scroll)
        
        included_group.setLayout(included_layout)
        features_row.addWidget(included_group, 1)
        
        # Excluded Features
        excluded_group = QGroupBox("❌ Excluded Features")
        excluded_layout = QVBoxLayout()
        
        self.excluded_scroll = QScrollArea()
        self.excluded_widget = QWidget()
        self.excluded_layout = QVBoxLayout(self.excluded_widget)
        self.excluded_layout.setSpacing(5)
        
        # First 10 features container
        self.excluded_first_10 = QWidget()
        self.excluded_first_10_layout = QVBoxLayout(self.excluded_first_10)
        self.excluded_first_10_layout.setSpacing(2)
        
        # Expansion for remaining features
        self.excluded_expansion_btn = QPushButton("Show remaining features")
        self.excluded_expansion_btn.setCheckable(True)
        self.excluded_expansion_btn.setVisible(False)
        self.excluded_expansion_widget = QWidget()
        self.excluded_expansion_layout = QVBoxLayout(self.excluded_expansion_widget)
        self.excluded_expansion_widget.setVisible(False)
        
        self.excluded_layout.addWidget(self.excluded_first_10)
        self.excluded_layout.addWidget(self.excluded_expansion_btn)
        self.excluded_layout.addWidget(self.excluded_expansion_widget)
        self.excluded_layout.addStretch()
        
        self.excluded_scroll.setWidget(self.excluded_widget)
        self.excluded_scroll.setWidgetResizable(True)
        self.excluded_scroll.setMaximumHeight(200)
        excluded_layout.addWidget(self.excluded_scroll)
        
        excluded_group.setLayout(excluded_layout)
        features_row.addWidget(excluded_group, 1)
        
        features_layout.addLayout(features_row)
        
        self.selected_count_label = QLabel("✅ 0 features selected for analysis")
        self.selected_count_label.setStyleSheet("color: green;")
        features_layout.addWidget(self.selected_count_label)
        
        # Feature Import/Export buttons
        feature_io_row = QHBoxLayout()
        import_features_btn = QPushButton("📥 Import Features (Excel)")
        import_features_btn.clicked.connect(self.import_features_from_excel)
        feature_io_row.addWidget(import_features_btn)
        
        export_features_btn = QPushButton("📤 Export Features (Excel)")
        export_features_btn.clicked.connect(self.export_features_to_excel)
        feature_io_row.addWidget(export_features_btn)
        
        features_layout.addLayout(feature_io_row)
        
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)
        
        # Gates container
        gates_group = QGroupBox("🔬 Populations/Gates")
        gates_layout = QVBoxLayout()
        
        self.population_combo = QComboBox()
        self.population_combo.addItem("All events")
        gates_layout.addWidget(QLabel("Select population (optional):"))
        gates_layout.addWidget(self.population_combo)
        
        self.population_warning = QLabel("")
        self.population_warning.setWordWrap(True)
        gates_layout.addWidget(self.population_warning)
        
        gates_group.setLayout(gates_layout)
        layout.addWidget(gates_group)
        
        group.setLayout(layout)
        
        # Load features initially
        self.load_features_and_gates()
        
        return group
    
    def create_feature_chip(self, feature_name: str, is_included: bool, parent_layout) -> QPushButton:
        """Create a clickable chip button for a feature."""
        chip = QPushButton(f"{feature_name} ✕")
        chip.setMaximumHeight(25)
        
        if is_included:
            chip.setStyleSheet("""
                QPushButton {
                    background-color: #BBDEFB;
                    color: #1565C0;
                    border: none;
                    border-radius: 12px;
                    padding: 2px 8px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #90CAF9;
                }
            """)
        else:
            chip.setStyleSheet("""
                QPushButton {
                    background-color: #FFCDD2;
                    color: #C62828;
                    border: none;
                    border-radius: 12px;
                    padding: 2px 8px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #EF9A9A;
                }
            """)
        
        chip.clicked.connect(lambda: self.remove_feature(feature_name, is_included))
        parent_layout.addWidget(chip)
        return chip
    
    def remove_feature(self, feature_name: str, from_included: bool):
        """Remove feature from included/excluded and move to the other list."""
        if from_included:
            if feature_name in self.feature_selection_state["selected_features"]:
                self.feature_selection_state["selected_features"].remove(feature_name)
            if feature_name not in self.feature_selection_state.get("excluded_features", []):
                if "excluded_features" not in self.feature_selection_state:
                    self.feature_selection_state["excluded_features"] = []
                self.feature_selection_state["excluded_features"].append(feature_name)
        else:
            if feature_name in self.feature_selection_state.get("excluded_features", []):
                self.feature_selection_state["excluded_features"].remove(feature_name)
            if feature_name not in self.feature_selection_state["selected_features"]:
                self.feature_selection_state["selected_features"].append(feature_name)
        
        self.session_state["features"] = self.feature_selection_state["selected_features"]
        self.load_features_and_gates()
    
    def import_features_from_excel(self):
        """Import feature selection from Excel file."""
        if not self.session_state.get("run_id"):
            QMessageBox.warning(self, "Warning", "⚠️ Please set Run-ID first")
            return
        
        # First, ensure features are loaded
        if not self.feature_selection_state.get("available_features"):
            QMessageBox.warning(self, "Warning", "⚠️ Please load features first by selecting DAF files.")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Features from Excel",
            "",
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Load Excel file
            df = pd.read_excel(file_path)
            
            # Check for required columns
            if "included" not in df.columns and "Included" not in df.columns:
                QMessageBox.warning(
                    self, 
                    "Invalid Format", 
                    "❌ Excel file must contain an 'included' column.\n\n"
                    "Expected format:\n"
                    "- Column 'included': List of included features\n"
                    "- Column 'excluded' (optional): List of excluded features"
                )
                return
            
            # Find the correct column names (case-insensitive)
            included_col = None
            excluded_col = None
            for col in df.columns:
                if col.lower() == "included":
                    included_col = col
                elif col.lower() == "excluded":
                    excluded_col = col
            
            if included_col is None:
                QMessageBox.warning(self, "Invalid Format", "❌ Could not find 'included' column in Excel file.")
                return
            
            # Get included features from Excel
            included_from_file = []
            if included_col in df.columns:
                # Handle different formats: list in one cell, or one feature per row
                if len(df) == 1:
                    # Single row with list
                    val = df[included_col].iloc[0]
                    if isinstance(val, str):
                        # Try to parse as comma-separated or newline-separated
                        included_from_file = [f.strip() for f in val.replace('\n', ',').split(',') if f.strip()]
                    elif pd.notna(val):
                        included_from_file = [str(val)]
                else:
                    # Multiple rows, one feature per row
                    included_from_file = [str(f).strip() for f in df[included_col].dropna().unique() if str(f).strip()]
            
            # Get excluded features from Excel (optional)
            excluded_from_file = []
            if excluded_col and excluded_col in df.columns:
                if len(df) == 1:
                    val = df[excluded_col].iloc[0]
                    if isinstance(val, str):
                        excluded_from_file = [f.strip() for f in val.replace('\n', ',').split(',') if f.strip()]
                    elif pd.notna(val):
                        excluded_from_file = [str(val)]
                else:
                    excluded_from_file = [str(f).strip() for f in df[excluded_col].dropna().unique() if str(f).strip()]
            
            # Validate features against available features
            available_features_set = set(self.feature_selection_state.get("available_features", []))
            
            # Check for features that don't exist
            missing_included = [f for f in included_from_file if f not in available_features_set]
            missing_excluded = [f for f in excluded_from_file if f not in available_features_set]
            
            # Warning message for missing features
            warning_msg = ""
            if missing_included:
                warning_msg += f"⚠️ {len(missing_included)} included feature(s) not found in current data:\n"
                warning_msg += f"{', '.join(missing_included[:5])}"
                if len(missing_included) > 5:
                    warning_msg += f" ... and {len(missing_included) - 5} more"
                warning_msg += "\n\n"
            
            if missing_excluded:
                warning_msg += f"⚠️ {len(missing_excluded)} excluded feature(s) not found in current data:\n"
                warning_msg += f"{', '.join(missing_excluded[:5])}"
                if len(missing_excluded) > 5:
                    warning_msg += f" ... and {len(missing_excluded) - 5} more"
                warning_msg += "\n\n"
            
            # Filter to only include features that exist
            valid_included = [f for f in included_from_file if f in available_features_set]
            valid_excluded = [f for f in excluded_from_file if f in available_features_set]
            
            # Check if we have too few features
            if len(valid_included) == 0:
                QMessageBox.warning(
                    self,
                    "No Valid Features",
                    "❌ No valid included features found in Excel file.\n\n"
                    "All features in the 'included' column are missing from the current data."
                )
                return
            
            # Apply the feature selection
            self.feature_selection_state["selected_features"] = valid_included
            self.feature_selection_state["excluded_features"] = valid_excluded
            self.session_state["features"] = valid_included
            
            # Reload the UI
            self.load_features_and_gates()
            
            # Show success message with warnings if any
            success_msg = f"✅ Imported {len(valid_included)} included feature(s)"
            if valid_excluded:
                success_msg += f" and {len(valid_excluded)} excluded feature(s)"
            success_msg += "."
            
            if warning_msg:
                QMessageBox.warning(self, "Import with Warnings", warning_msg + success_msg)
            else:
                QMessageBox.information(self, "Success", success_msg)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Error importing features: {str(e)}")
    
    def export_features_to_excel(self):
        """Export current feature selection to Excel file."""
        if not self.session_state.get("run_id"):
            QMessageBox.warning(self, "Warning", "⚠️ Please set Run-ID first")
            return
        
        # Check if we have features to export
        selected_features = self.feature_selection_state.get("selected_features", [])
        excluded_features = self.feature_selection_state.get("excluded_features", [])
        
        if not selected_features and not excluded_features:
            QMessageBox.warning(self, "Warning", "⚠️ No features selected. Please select features first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Features to Excel",
            f"features_{self.session_state.get('run_id', 'export')}.xlsx",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Create DataFrame with features
            data = {
                "included": selected_features if selected_features else [""],
                "excluded": excluded_features if excluded_features else [""]
            }
            
            # Ensure both lists have the same length (pad with empty strings)
            max_len = max(len(selected_features), len(excluded_features))
            if max_len == 0:
                max_len = 1
            
            included_padded = selected_features + [""] * (max_len - len(selected_features))
            excluded_padded = excluded_features + [""] * (max_len - len(excluded_features))
            
            df = pd.DataFrame({
                "included": included_padded,
                "excluded": excluded_padded
            })
            
            # Remove empty rows at the end
            df = df[~(df["included"].astype(str).str.strip() == "") & 
                    ~(df["excluded"].astype(str).str.strip() == "")].copy()
            
            # If we removed everything, add at least one row
            if len(df) == 0:
                df = pd.DataFrame({
                    "included": selected_features if selected_features else [""],
                    "excluded": excluded_features if excluded_features else [""]
                })
            
            # Save to Excel
            df.to_excel(file_path, index=False, engine='openpyxl')
            
            QMessageBox.information(
                self, 
                "Success", 
                f"✅ Features exported to:\n{file_path}\n\n"
                f"Included: {len(selected_features)} feature(s)\n"
                f"Excluded: {len(excluded_features)} feature(s)"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Error exporting features: {str(e)}")
    
    def load_features_and_gates(self):
        """Load available features and gates from FCS files."""
        if not self.session_state.get("run_id"):
            self.features_info_label.setText("ℹ️ Run-ID is set. Select DAF files to load features.")
            return
        
        try:
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            fcs_files = sorted(paths["fcs"].glob("*.fcs"))
            
            if not fcs_files:
                self.features_info_label.setText("ℹ️ No FCS files found. Convert DAF files first.")
                return
            
            cache_dir = paths["csv_cache"]
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            converter = MM()
            all_populations_per_file = {}
            
            for fcs_file in fcs_files:
                csv_path = cache_dir / f"{safe_str(fcs_file.stem)}.csv"
                if not csv_path.exists():
                    converter.convert_to_CSV(safe_str(fcs_file), safe_str(csv_path))
                df = pd.read_csv(csv_path)
                
                # Identify populations
                populations = []
                for col in df.columns:
                    if col == "sample_id":
                        continue
                    if (col.startswith("R") and ("&" in col or (len(col) > 1 and col[1:].isdigit()))) or \
                       col in ["Live", "Singlet", "Dead", "Doublet", "Focused", "Viable"] or \
                       df[col].dtype == bool or \
                       (df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1, True, False, 0.0, 1.0})):
                        populations.append(col)
                all_populations_per_file[safe_str(fcs_file.name)] = set(populations)
            
            # Find common populations
            if all_populations_per_file:
                common_populations_set = set.intersection(*all_populations_per_file.values())
                all_populations_set = set.union(*all_populations_per_file.values())
                common_populations = sorted(common_populations_set)
                excluded_populations = sorted(all_populations_set - common_populations_set)
            else:
                common_populations = []
                excluded_populations = []
                all_populations_set = set()
            
            # Get features from first file
            first_fcs = fcs_files[0]
            csv_path = cache_dir / f"{safe_str(first_fcs.stem)}.csv"
            df_sample = pd.read_csv(csv_path)
            all_features = sorted([c for c in df_sample.columns if c != "sample_id" and c not in all_populations_set])
            
            # Exclude patterns
            exclude_patterns = ["intensity", "Intensity", "saturation", "Saturation", "Raw pixel", "Bkgd", "All", "Mean Pixel", "Max Pixel", "Median Pixel", "Raw", "Time", "Object Number", "Flow Speed"]
            excluded_by_default = [f for f in all_features if any(p in f for p in exclude_patterns)]
            
            # Apply channel filtering (only if channels were explicitly excluded)
            excluded_by_channel = []
            if hasattr(self, 'excluded_channels') and self.excluded_channels:
                # Get unchecked channels (Ch01-Ch12)
                unchecked_channels = self.excluded_channels
                
                # For each unchecked channel, also exclude corresponding mask (M01-M12)
                unchecked_patterns = []
                for ch_name in unchecked_channels:
                    unchecked_patterns.append(ch_name)  # e.g., "Ch04"
                    # Get corresponding mask (Ch04 -> M04)
                    mask_num = ch_name.replace("Ch", "")
                    mask_name = f"M{mask_num}"
                    unchecked_patterns.append(mask_name)  # e.g., "M04"
                
                # Find features that belong to unchecked channels/masks
                for feature in all_features:
                    # Check if feature name contains any unchecked channel/mask pattern
                    for pattern in unchecked_patterns:
                        # Pattern matching: Ch01, Ch02, ..., Ch12 or M01, M02, ..., M12
                        # Check various patterns:
                        # - Direct match: "Ch01", "M01"
                        # - With underscore: "_Ch01", "_M01"
                        # - With parentheses: "(Ch01", "(M01"
                        # - In compound names: "Area_M04", "Contrast_M07_Ch07"
                        if (pattern in feature and 
                            (f"_{pattern}" in feature or 
                             f"({pattern}" in feature or
                             feature.endswith(pattern) or
                             f"_{pattern}_" in feature)):
                            excluded_by_channel.append(feature)
                            break  # Feature already excluded, no need to check other patterns
                
                # Remove duplicates
                excluded_by_channel = list(set(excluded_by_channel))
            
            # Combine exclusions
            excluded_by_default = list(set(excluded_by_default + excluded_by_channel))
            included_by_default = [f for f in all_features if f not in excluded_by_default]
            
            # Update state
            self.feature_selection_state["available_features"] = all_features
            self.feature_selection_state["included_features"] = included_by_default
            self.feature_selection_state["excluded_features"] = excluded_by_default
            self.feature_selection_state["populations"] = common_populations
            
            # Get current selection or use defaults if empty
            current_selected = self.session_state.get("features", [])
            if not current_selected or len(current_selected) == 0:
                current_selected = included_by_default
                self.session_state["features"] = included_by_default
            self.feature_selection_state["selected_features"] = [f for f in current_selected if f in all_features]
            
            # Ensure all features are either included or excluded
            selected_set = set(self.feature_selection_state["selected_features"])
            excluded_set = set(self.feature_selection_state.get("excluded_features", excluded_by_default))
            for feat in all_features:
                if feat not in selected_set and feat not in excluded_set:
                    if feat not in excluded_by_default:
                        if "excluded_features" not in self.feature_selection_state:
                            self.feature_selection_state["excluded_features"] = []
                        self.feature_selection_state["excluded_features"].append(feat)
            
            # Update UI
            self.features_info_label.setText(f"Available: {len(all_features)} features")
            
            # Clear existing chips
            for i in reversed(range(self.included_first_10_layout.count())):
                item = self.included_first_10_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
            for i in reversed(range(self.included_expansion_layout.count())):
                item = self.included_expansion_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
            for i in reversed(range(self.excluded_first_10_layout.count())):
                item = self.excluded_first_10_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
            for i in reversed(range(self.excluded_expansion_layout.count())):
                item = self.excluded_expansion_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
            
            # Display included features: first 10, rest in expansion
            features_list = self.feature_selection_state["selected_features"]
            if features_list:
                first_10 = features_list[:10]
                remaining = features_list[10:]
                
                # Create chips for first 10
                for feat in first_10:
                    self.create_feature_chip(feat, True, self.included_first_10_layout)
                
                # Create chips for remaining
                if remaining:
                    self.included_expansion_btn.setText(f"Show remaining {len(remaining)} included features")
                    self.included_expansion_btn.setVisible(True)
                    self.included_expansion_btn.toggled.connect(self.included_expansion_widget.setVisible)
                    
                    for feat in remaining:
                        self.create_feature_chip(feat, True, self.included_expansion_layout)
                else:
                    self.included_expansion_btn.setVisible(False)
                    self.included_expansion_widget.setVisible(False)
            else:
                label = QLabel("No features included. Click features from excluded list to add.")
                label.setStyleSheet("color: gray;")
                self.included_first_10_layout.addWidget(label)
            
            # Display excluded features: first 10, rest in expansion
            excluded_list = self.feature_selection_state.get("excluded_features", excluded_by_default)
            if excluded_list:
                first_10 = excluded_list[:10]
                remaining = excluded_list[10:]
                
                # Create chips for first 10
                for feat in first_10:
                    self.create_feature_chip(feat, False, self.excluded_first_10_layout)
                
                # Create chips for remaining
                if remaining:
                    self.excluded_expansion_btn.setText(f"Show remaining {len(remaining)} excluded features")
                    self.excluded_expansion_btn.setVisible(True)
                    self.excluded_expansion_btn.toggled.connect(self.excluded_expansion_widget.setVisible)
                    
                    for feat in remaining:
                        self.create_feature_chip(feat, False, self.excluded_expansion_layout)
                else:
                    self.excluded_expansion_btn.setVisible(False)
                    self.excluded_expansion_widget.setVisible(False)
            else:
                label = QLabel("No features excluded.")
                label.setStyleSheet("color: gray;")
                self.excluded_first_10_layout.addWidget(label)
            
            # Update selected count
            selected_count = len(self.feature_selection_state["selected_features"])
            self.selected_count_label.setText(f"✅ {selected_count} features selected for analysis")
            
            # Update population combo
            self.population_combo.clear()
            self.population_combo.addItem("All events")
            if excluded_populations:
                self.population_warning.setText(f"⚠️ Populations not in all files (excluded): {', '.join(sorted(excluded_populations)[:5])}{'...' if len(excluded_populations) > 5 else ''}")
                self.population_warning.setStyleSheet("color: orange;")
            else:
                self.population_warning.setText("")
            
            if common_populations:
                for pop in sorted(common_populations):
                    self.population_combo.addItem(pop)
                
                current_pop = self.session_state.get("selected_population")
                if current_pop and current_pop in common_populations:
                    idx = self.population_combo.findText(current_pop)
                    if idx >= 0:
                        self.population_combo.setCurrentIndex(idx)
                else:
                    self.population_combo.setCurrentIndex(0)
            else:
                self.population_combo.setCurrentIndex(0)
                self.population_warning.setText("ℹ️ No common populations detected. All events will be analyzed.")
                self.population_warning.setStyleSheet("color: gray;")
            
            self.population_combo.currentTextChanged.connect(self.on_population_changed)
            
        except Exception as e:
            self.features_info_label.setText(f"⚠️ Error loading features: {str(e)}")
            self.features_info_label.setStyleSheet("color: red;")
    
    def on_population_changed(self, text: str):
        """Handle population selection change."""
        if text == "All events":
            self.session_state["selected_population"] = None
        else:
            self.session_state["selected_population"] = text
    
    def apply_channel_filter(self):
        """Apply channel filter - move features from included to excluded."""
        if not hasattr(self, 'channel_checkboxes') or not self.channel_checkboxes:
            return
        
        # Get unchecked channels
        unchecked_channels = [ch for ch, cb in self.channel_checkboxes.items() if not cb.isChecked()]
        self.excluded_channels = unchecked_channels
        
        # Reload features with updated channel filter
        self.load_features_and_gates()
        
        # IMPORTANT: Update selected_features to remove excluded channel features
        if hasattr(self, 'feature_selection_state') and self.feature_selection_state.get("selected_features"):
            # Get current selected features
            current_selected = self.feature_selection_state["selected_features"].copy()
            
            # Remove features that belong to excluded channels
            if unchecked_channels:
                # Build patterns for excluded channels and masks
                excluded_patterns = []
                for ch_name in unchecked_channels:
                    excluded_patterns.append(ch_name)  # e.g., "Ch04"
                    # Get corresponding mask
                    mask_num = ch_name.replace("Ch", "")
                    mask_name = f"M{mask_num}"
                    excluded_patterns.append(mask_name)  # e.g., "M04"
                
                # Filter out features matching excluded patterns
                filtered_selected = []
                for feature in current_selected:
                    should_exclude = False
                    for pattern in excluded_patterns:
                        if (pattern in feature and 
                            (f"_{pattern}" in feature or 
                             f"({pattern}" in feature or
                             feature.endswith(pattern) or
                             f"_{pattern}_" in feature)):
                            should_exclude = True
                            break
                    if not should_exclude:
                        filtered_selected.append(feature)
                
                # Update selected features
                self.feature_selection_state["selected_features"] = filtered_selected
                self.session_state["features"] = filtered_selected
                
                # Update UI to reflect changes - re-display features
                # Clear existing chips
                for i in reversed(range(self.included_first_10_layout.count())):
                    item = self.included_first_10_layout.itemAt(i)
                    if item and item.widget():
                        item.widget().setParent(None)
                for i in reversed(range(self.included_expansion_layout.count())):
                    item = self.included_expansion_layout.itemAt(i)
                    if item and item.widget():
                        item.widget().setParent(None)
                
                # Re-display included features
                if filtered_selected:
                    first_10 = filtered_selected[:10]
                    remaining = filtered_selected[10:]
                    
                    for feat in first_10:
                        self.create_feature_chip(feat, True, self.included_first_10_layout)
                    
                    if remaining:
                        self.included_expansion_btn.setText(f"Show remaining {len(remaining)} included features")
                        self.included_expansion_btn.setVisible(True)
                        for feat in remaining:
                            self.create_feature_chip(feat, True, self.included_expansion_layout)
                    else:
                        self.included_expansion_btn.setVisible(False)
                        self.included_expansion_widget.setVisible(False)
                
                # Update selected count
                selected_count = len(filtered_selected)
                self.selected_count_label.setText(f"✅ {selected_count} features selected for analysis")
        
        # Update status
        self.update_status()
        
        # Show confirmation
        if unchecked_channels:
            if hasattr(self, 'feature_selection_state') and self.feature_selection_state.get("selected_features"):
                excluded_count = len(current_selected) - len(self.feature_selection_state["selected_features"])
            else:
                excluded_count = 0
            QMessageBox.information(self, "Channel Filter Applied", 
                                  f"Excluded channels: {', '.join(unchecked_channels)}\n"
                                  f"Removed {excluded_count} features from selection.\n"
                                  f"Corresponding M01-M12 features have been moved to excluded.")
        else:
            QMessageBox.information(self, "Channel Filter Applied", 
                                  "All channels included. No features excluded.")
    
    def create_dimensionality_reduction_section(self) -> QGroupBox:
        """Create dimensionality reduction section."""
        group = QGroupBox("5️⃣ Run Dimensionality Reduction")
        group.setFont(QFont("Arial", 12, QFont.Bold))
        
        layout = QVBoxLayout()
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("DensMAP")  # Default first
        if UMAP_AVAILABLE:
            self.method_combo.addItem("UMAP")
        self.method_combo.addItem("t-SNE")
        self.method_combo.setCurrentText("DensMAP")  # Set DensMAP as default
        method_layout.addWidget(self.method_combo, 1)
        layout.addLayout(method_layout)
        
        # Sampling option (max cells per sample)
        sampling_layout = QHBoxLayout()
        sampling_layout.addWidget(QLabel("Max cells per sample (0 = all):"))
        self.max_cells_input = QLineEdit()
        self.max_cells_input.setText("0")
        self.max_cells_input.setPlaceholderText("0 = all cells")
        self.max_cells_input.setMaximumWidth(150)
        sampling_layout.addWidget(self.max_cells_input)
        sampling_layout.addStretch()
        layout.addLayout(sampling_layout)
        
        # Parameters container
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setSpacing(5)
        
        self.param_sliders = {}
        
        def update_params():
            # Clear existing params
            for i in reversed(range(self.params_layout.count())):
                item = self.params_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
            self.param_sliders.clear()
            
            method = self.method_combo.currentText()
            
            if method == "DensMAP":
                self.params_layout.addWidget(QLabel("dens_lambda:"))
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(5)
                slider.setMaximum(30)
                slider.setValue(20)
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(5)
                value_label = QLabel("2.0")
                slider.valueChanged.connect(lambda v: value_label.setText(f"{v/10:.1f}"))
                self.params_layout.addWidget(slider)
                self.params_layout.addWidget(value_label)
                self.param_sliders["dens_lambda"] = (slider, value_label)
                
                self.params_layout.addWidget(QLabel("n_neighbors:"))
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(5)
                slider.setMaximum(100)
                slider.setValue(30)
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(10)
                value_label = QLabel("30")
                slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
                self.params_layout.addWidget(slider)
                self.params_layout.addWidget(value_label)
                self.param_sliders["n_neighbors"] = (slider, value_label)
                
                self.params_layout.addWidget(QLabel("min_dist:"))
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(99)
                slider.setValue(10)
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(10)
                value_label = QLabel("0.1")
                slider.valueChanged.connect(lambda v: value_label.setText(f"{v/100:.2f}"))
                self.params_layout.addWidget(slider)
                self.params_layout.addWidget(value_label)
                self.param_sliders["min_dist"] = (slider, value_label)
                
            elif method == "UMAP":
                self.params_layout.addWidget(QLabel("n_neighbors:"))
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(5)
                slider.setMaximum(100)
                slider.setValue(30)
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(10)
                value_label = QLabel("30")
                slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
                self.params_layout.addWidget(slider)
                self.params_layout.addWidget(value_label)
                self.param_sliders["n_neighbors"] = (slider, value_label)
                
                self.params_layout.addWidget(QLabel("min_dist:"))
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(99)
                slider.setValue(10)
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(10)
                value_label = QLabel("0.1")
                slider.valueChanged.connect(lambda v: value_label.setText(f"{v/100:.2f}"))
                self.params_layout.addWidget(slider)
                self.params_layout.addWidget(value_label)
                self.param_sliders["min_dist"] = (slider, value_label)
                
            else:  # t-SNE
                self.params_layout.addWidget(QLabel("perplexity:"))
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(5)
                slider.setMaximum(50)
                slider.setValue(30)
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(5)
                value_label = QLabel("30.0")
                slider.valueChanged.connect(lambda v: value_label.setText(f"{v:.1f}"))
                self.params_layout.addWidget(slider)
                self.params_layout.addWidget(value_label)
                self.param_sliders["perplexity"] = (slider, value_label)
        
        self.method_combo.currentTextChanged.connect(update_params)
        update_params()
        
        layout.addWidget(self.params_container)
        
        # Run button
        run_btn = QPushButton("🚀 RUN")
        run_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; font-weight: bold;")
        run_btn.clicked.connect(self.run_analysis)
        layout.addWidget(run_btn)
        
        # Progress bar for analysis
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setVisible(False)
        self.analysis_progress.setFormat("Running analysis: %p%")
        layout.addWidget(self.analysis_progress)
        
        group.setLayout(layout)
        return group
    
    def run_analysis(self):
        """Run dimensionality reduction analysis."""
        if not self.session_state.get("run_id"):
            QMessageBox.warning(self, "Warning", "⚠️ Please set Run-ID first")
            return
        
        features = self.session_state.get("features", [])
        if not features or len(features) == 0:
            QMessageBox.warning(self, "Warning", "⚠️ Please select at least one feature")
            return
        
        project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
        run_id = safe_str(self.session_state["run_id"])
        paths = get_run_paths(safe_path(project_dir), run_id)
        fcs_files = list(paths["fcs"].glob("*.fcs"))
        
        if not fcs_files:
            QMessageBox.warning(self, "Warning", "⚠️ No FCS files found. Select and convert DAF files first.")
            return
        
        method = self.method_combo.currentText()
        
        # Get parameter values from sliders
        method_params = {}
        if "dens_lambda" in self.param_sliders:
            slider, _ = self.param_sliders["dens_lambda"]
            method_params["dens_lambda"] = slider.value() / 10.0
        else:
            method_params["dens_lambda"] = 2.0
        
        if "n_neighbors" in self.param_sliders:
            slider, _ = self.param_sliders["n_neighbors"]
            method_params["n_neighbors"] = slider.value()
        else:
            method_params["n_neighbors"] = 30
        
        if "min_dist" in self.param_sliders:
            slider, _ = self.param_sliders["min_dist"]
            method_params["min_dist"] = slider.value() / 100.0
        else:
            method_params["min_dist"] = 0.1
        
        if "perplexity" in self.param_sliders:
            slider, _ = self.param_sliders["perplexity"]
            method_params["perplexity"] = float(slider.value())
        else:
            method_params["perplexity"] = 30.0
        
        # Get max cells per sample
        try:
            max_cells_per_sample = int(self.max_cells_input.text().strip())
            if max_cells_per_sample < 0:
                max_cells_per_sample = 0
        except ValueError:
            max_cells_per_sample = 0  # Default: all cells
        
        method_params["max_cells_per_sample"] = max_cells_per_sample
        
        # Ensure all paths are safe
        safe_paths = {k: safe_path(v) for k, v in paths.items()}
        
        # Show progress bar
        self.analysis_progress.setVisible(True)
        self.analysis_progress.setRange(0, 0)  # Indeterminate progress
        self.analysis_progress.setFormat(f"Running {method} analysis...")
        
        # Start worker thread
        self.analysis_worker = AnalysisWorker(
            safe_paths,
            features,
            method,
            method_params,
            self.session_state.get("selected_population"),
            self.session_state
        )
        # Use default argument to capture worker reference correctly
        self.analysis_worker.finished.connect(
            lambda success, msg, w=self.analysis_worker: self.on_analysis_finished(success, msg, w)
        )
        self.analysis_worker.progress.connect(lambda msg: self.processing_status.append(msg))
        self.active_workers.append(self.analysis_worker)  # Track worker for cleanup
        self.analysis_worker.start()
    
    def on_analysis_finished(self, success: bool, message: str, worker: AnalysisWorker):
        """Handle analysis completion."""
        # Hide progress bar
        self.analysis_progress.setVisible(False)
        
        if success:
            QMessageBox.information(self, "Success", message)
            
            # Save comprehensive run info with all settings
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            
            method = self.session_state.get("last_dim_reduction_method", "Unknown")
            method_params = self.session_state.get("last_dim_reduction_params", {})
            features = self.session_state.get("features", [])
            population = self.session_state.get("selected_population")
            
            self._save_run_info(
                paths,
                dim_reduction={
                    "method": method,
                    "parameters": method_params,
                    "features": features,
                    "population": population,
                }
            )
            
            self.update_status()
            
            # Force update of sections visibility - show visualization immediately
            if hasattr(self, 'viz_cluster_section'):
                if self.session_state.get("embedding_df") is not None:
                    self.viz_cluster_section.setVisible(True)
                    print(f"DEBUG: Combined visualization/clustering section set to visible after analysis")
            elif hasattr(self, 'viz_section'):
                if self.session_state.get("embedding_df") is not None:
                    self.viz_section.setVisible(True)
                    print(f"DEBUG: Visualization section set to visible after analysis")
            elif hasattr(self, 'clustering_section'):
                if self.session_state.get("embedding_df") is not None:
                    self.clustering_section.setVisible(True)
                    print(f"DEBUG: Clustering section set to visible after analysis")
            
            if hasattr(self, 'top10_features_btn'):
                if self.session_state.get("embedding_df") is not None:
                    self.top10_features_btn.setVisible(True)
                    print(f"DEBUG: Top10 Features button set to visible after analysis")
            
            if hasattr(self, 'pca_plot_btn'):
                # PCA button should be visible if we have features and metadata
                self.pca_plot_btn.setVisible(True)
                print(f"DEBUG: PCA Plot button set to visible after analysis")
            
            # Update visualization immediately - this will show the plot with default "group" coloring
            self.update_visualization()
            if hasattr(self, 'update_clustering_section'):
                self.update_clustering_section()
        else:
            QMessageBox.critical(self, "Error", message)
        
        # Remove worker from active list once finished
        if worker in self.active_workers:
            self.active_workers.remove(worker)
    
    def on_top10_finished(self, success: bool, message: str, worker: FeatureImportanceWorker):
        """Handle Feature Importance calculation completion."""
        self.top10_features_btn.setEnabled(True)
        self.top10_features_btn.setText("📊 Feature Importance")
        
        # Print to console for debugging
        print(f"DEBUG on_top10_finished: success={success}, message={message}")
        
        if success:
            # Create plots in main thread (Matplotlib cannot be used in worker thread)
            plot_data = self.session_state.get("feature_importance_plot_data")
            if plot_data:
                try:
                    features_x_df = pd.DataFrame(plot_data["features_x"])
                    features_y_df = pd.DataFrame(plot_data["features_y"])
                    
                    if not features_x_df.empty:
                        self._plot_feature_importance(features_x_df, plot_data["plot_path_x"], "X Dimension")
                    if not features_y_df.empty:
                        self._plot_feature_importance(features_y_df, plot_data["plot_path_y"], "Y Dimension")
                    
                    # Clean up
                    del self.session_state["feature_importance_plot_data"]
                except Exception as e:
                    print(f"Warning: Could not create feature importance plots: {e}")
            
            QMessageBox.information(self, "Success", message)
        else:
            # Show detailed error message
            error_msg = f"{message}\n\nPlease check the console for detailed debug output."
            QMessageBox.critical(self, "Error", error_msg)
            print(f"ERROR: Feature Importance calculation failed: {message}")
        
        if worker in self.active_workers:
            self.active_workers.remove(worker)
    
    def _plot_feature_importance(self, features_df: pd.DataFrame, output_path: str, title: str):
        """Plot feature importance manually (replacement for mm.plot_feature_importance to avoid multiprocessing)."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by importance
        features_df = features_df.sort_values('importance_normalized', ascending=True)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(features_df))
        ax.barh(y_pos, features_df['importance_normalized'], color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features_df['index1'])
        ax.set_xlabel('Normalized Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Top 10 Features - {title}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Highest importance at top
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def create_visualization_clustering_section(self) -> QGroupBox:
        """Create combined visualization and clustering section with side-by-side plots."""
        group = QGroupBox("6️⃣ Visualization & Clustering")
        group.setFont(QFont("Arial", 12, QFont.Bold))
        group.setVisible(False)  # Hidden until analysis is done
        
        main_layout = QVBoxLayout()
        
        # Top controls row (shared)
        top_controls = QHBoxLayout()
        
        # Left: Visualization controls
        viz_controls = QVBoxLayout()
        viz_controls.addWidget(QLabel("📊 Visualization (Left):"))
        
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color by:"))
        
        # Radio buttons for Metadata vs Feature
        self.color_type_group = QButtonGroup()
        self.color_metadata_radio = QRadioButton("Metadata")
        self.color_metadata_radio.setChecked(True)
        self.color_feature_radio = QRadioButton("Feature")
        self.color_type_group.addButton(self.color_metadata_radio, 0)
        self.color_type_group.addButton(self.color_feature_radio, 1)
        color_layout.addWidget(self.color_metadata_radio)
        color_layout.addWidget(self.color_feature_radio)
        
        self.color_by_combo = QComboBox()
        color_layout.addWidget(self.color_by_combo, 1)
        
        # Connect radio buttons to update combo box
        # Use buttonClicked signal from QButtonGroup to ensure it fires when selection changes
        self.color_type_group.buttonClicked.connect(self.update_color_by_options)
        # Also connect individual toggled signals as backup
        self.color_metadata_radio.toggled.connect(self.update_color_by_options)
        self.color_feature_radio.toggled.connect(self.update_color_by_options)
        
        viz_controls.addLayout(color_layout)
        
        # Axis limits (shared for both plots)
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("X:"))
        self.x_min_input = QLineEdit()
        self.x_min_input.setPlaceholderText("min")
        self.x_min_input.setMaximumWidth(80)
        axis_layout.addWidget(self.x_min_input)
        axis_layout.addWidget(QLabel("to"))
        self.x_max_input = QLineEdit()
        self.x_max_input.setPlaceholderText("max")
        self.x_max_input.setMaximumWidth(80)
        axis_layout.addWidget(self.x_max_input)
        axis_layout.addWidget(QLabel("Y:"))
        self.y_min_input = QLineEdit()
        self.y_min_input.setPlaceholderText("min")
        self.y_min_input.setMaximumWidth(80)
        axis_layout.addWidget(self.y_min_input)
        axis_layout.addWidget(QLabel("to"))
        self.y_max_input = QLineEdit()
        self.y_max_input.setPlaceholderText("max")
        self.y_max_input.setMaximumWidth(80)
        axis_layout.addWidget(self.y_max_input)
        apply_axis_btn = QPushButton("Apply Limits")
        apply_axis_btn.setStyleSheet("background-color: #607D8B; color: white; padding: 3px;")
        apply_axis_btn.clicked.connect(self.apply_axis_limits)
        axis_layout.addWidget(apply_axis_btn)
        reset_axis_btn = QPushButton("Reset")
        reset_axis_btn.setStyleSheet("background-color: #9E9E9E; color: white; padding: 3px;")
        reset_axis_btn.clicked.connect(self.reset_axis_limits)
        axis_layout.addWidget(reset_axis_btn)
        viz_controls.addLayout(axis_layout)
        
        # Highlight cells
        highlight_layout = QHBoxLayout()
        highlight_layout.addWidget(QLabel("Highlight:"))
        self.highlight_input = QLineEdit()
        self.highlight_input.setPlaceholderText("e.g., 100, 200")
        highlight_layout.addWidget(self.highlight_input, 1)
        highlight_btn = QPushButton("✨ Highlight")
        highlight_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 3px;")
        highlight_btn.clicked.connect(self.highlight_cells)
        highlight_layout.addWidget(highlight_btn)
        viz_controls.addLayout(highlight_layout)
        
        # Export buttons for visualization
        viz_export_layout = QHBoxLayout()
        export_viz_png = QPushButton("💾 Export PNG")
        export_viz_png.setStyleSheet("background-color: #FF9800; color: white; padding: 3px;")
        export_viz_png.clicked.connect(lambda: self.export_plot("png"))
        viz_export_layout.addWidget(export_viz_png)
        export_viz_pdf = QPushButton("💾 Export PDF")
        export_viz_pdf.setStyleSheet("background-color: #F44336; color: white; padding: 3px;")
        export_viz_pdf.clicked.connect(lambda: self.export_plot("pdf"))
        viz_export_layout.addWidget(export_viz_pdf)
        viz_controls.addLayout(viz_export_layout)
        
        # Feature Importance button
        self.top10_features_btn = QPushButton("📊 Feature Importance")
        self.top10_features_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 3px;")
        self.top10_features_btn.clicked.connect(self.download_top10_features)
        if self.session_state.get("embedding_df") is None:
            self.top10_features_btn.setVisible(False)
        viz_controls.addWidget(self.top10_features_btn)
        
        # PCA Plot button (sample-level, grouped by groups)
        self.pca_plot_btn = QPushButton("📈 PCA Plot (Sample-Level)")
        self.pca_plot_btn.setStyleSheet("background-color: #009688; color: white; padding: 3px;")
        self.pca_plot_btn.clicked.connect(self.export_pca_plot)
        # Always visible - PCA can be run if metadata with groups is available
        viz_controls.addWidget(self.pca_plot_btn)
        
        # Right: Clustering controls
        cluster_controls = QVBoxLayout()
        cluster_controls.addWidget(QLabel("🔬 Clustering (Right):"))
        
        # Algorithm selection
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("Algorithm:"))
        self.cluster_algo_combo = QComboBox()
        self.cluster_algo_combo.addItems(["KMeans", "Gaussian Mixture Models", "HDBSCAN"])
        algo_layout.addWidget(self.cluster_algo_combo, 1)
        cluster_controls.addLayout(algo_layout)
        
        # Parameters container
        self.cluster_params_container = QWidget()
        self.cluster_params_layout = QVBoxLayout(self.cluster_params_container)
        self.cluster_n_input = QLineEdit()
        self.cluster_n_input.setText("10")
        self.cluster_n_input.setPlaceholderText("Number of clusters")
        self.cluster_params_layout.addWidget(QLabel("Number of clusters:"))
        self.cluster_params_layout.addWidget(self.cluster_n_input)
        self.elbow_plot_btn = QPushButton("📊 Download Elbow Plot")
        self.elbow_plot_btn.setStyleSheet("background-color: #FFC107; color: white; padding: 3px;")
        self.elbow_plot_btn.clicked.connect(self.download_elbow_plot)
        self.cluster_params_layout.addWidget(self.elbow_plot_btn)
        self.cluster_algo_combo.currentTextChanged.connect(self.update_cluster_params)
        cluster_controls.addWidget(self.cluster_params_container)
        
        # Run clustering button
        cluster_btn = QPushButton("🔬 Run Clustering")
        cluster_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 5px;")
        cluster_btn.clicked.connect(self.run_clustering)
        cluster_controls.addWidget(cluster_btn)
        
        # Export buttons for cluster plot
        cluster_export_layout = QHBoxLayout()
        export_cluster_png = QPushButton("💾 Export PNG")
        export_cluster_png.setStyleSheet("background-color: #FF9800; color: white; padding: 3px;")
        export_cluster_png.clicked.connect(lambda: self.export_cluster_plot("png"))
        cluster_export_layout.addWidget(export_cluster_png)
        export_cluster_pdf = QPushButton("💾 Export PDF")
        export_cluster_pdf.setStyleSheet("background-color: #F44336; color: white; padding: 3px;")
        export_cluster_pdf.clicked.connect(lambda: self.export_cluster_plot("pdf"))
        cluster_export_layout.addWidget(export_cluster_pdf)
        cluster_controls.addLayout(cluster_export_layout)
        
        # Analysis buttons
        analysis_btn_layout = QHBoxLayout()
        top5_features_btn = QPushButton("🔍 Top 5 Features")
        top5_features_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 3px;")
        top5_features_btn.clicked.connect(self.export_top3_features)
        analysis_btn_layout.addWidget(top5_features_btn)
        heatmap_btn = QPushButton("🔥 Heatmap")
        heatmap_btn.setStyleSheet("background-color: #E91E63; color: white; padding: 3px;")
        heatmap_btn.clicked.connect(self.export_cluster_heatmap)
        analysis_btn_layout.addWidget(heatmap_btn)
        cluster_cells_btn = QPushButton("📋 Export Cluster Cells")
        cluster_cells_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 3px;")
        cluster_cells_btn.clicked.connect(self.export_cluster_cells)
        analysis_btn_layout.addWidget(cluster_cells_btn)
        cluster_controls.addLayout(analysis_btn_layout)
        
        # Advanced Analysis buttons
        advanced_analysis_layout = QHBoxLayout()
        bar_graphs_btn = QPushButton("📊 Group-Cluster Bar Graphs")
        bar_graphs_btn.setStyleSheet("background-color: #795548; color: white; padding: 3px;")
        bar_graphs_btn.clicked.connect(self.export_group_cluster_bar_graphs)
        advanced_analysis_layout.addWidget(bar_graphs_btn)
        cluster_controls.addLayout(advanced_analysis_layout)
        
        # Add both control groups to top row
        top_controls.addLayout(viz_controls, 1)
        top_controls.addLayout(cluster_controls, 1)
        main_layout.addLayout(top_controls)
        
        # Plots side by side
        plots_layout = QHBoxLayout()
        
        # Left: Visualization plot
        viz_group = QGroupBox("Dimensionality Reduction Plot")
        viz_layout = QVBoxLayout()
        
        # Title label for the plot
        self.plot_title_label = QLabel("")
        self.plot_title_label.setAlignment(Qt.AlignCenter)
        self.plot_title_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        viz_layout.addWidget(self.plot_title_label)
        
        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setMinimumHeight(400)
        self.plot_label.setStyleSheet("border: 1px solid #ccc; background-color: white;")
        viz_layout.addWidget(self.plot_label)
        
        # Cluster statistics for visualization plot (table only, no export button here)
        viz_stats_header = QHBoxLayout()
        viz_stats_header.addWidget(QLabel("📊 Statistics (Table):"))
        viz_stats_header.addStretch()
        viz_layout.addLayout(viz_stats_header)
        
        self.viz_stats_table = QTableWidget()
        self.viz_stats_table.setColumnCount(4)
        self.viz_stats_table.setHorizontalHeaderLabels(["Cluster", "Size", "Percentage", "Sample Distribution"])
        self.viz_stats_table.horizontalHeader().setStretchLastSection(True)
        self.viz_stats_table.setMaximumHeight(150)
        viz_layout.addWidget(self.viz_stats_table)
        
        viz_group.setLayout(viz_layout)
        plots_layout.addWidget(viz_group, 1)
        
        # Right: Cluster plot
        cluster_group = QGroupBox("Clustering Plot")
        cluster_layout = QVBoxLayout()
        self.cluster_plot_label = QLabel()
        self.cluster_plot_label.setAlignment(Qt.AlignCenter)
        self.cluster_plot_label.setMinimumHeight(400)
        self.cluster_plot_label.setStyleSheet("border: 1px solid #ccc; background-color: white;")
        cluster_layout.addWidget(self.cluster_plot_label)
        
        # Cluster statistics for cluster plot - Bar Chart directly displayed
        cluster_stats_header = QHBoxLayout()
        cluster_stats_header.addWidget(QLabel("📊 Statistics (Bar Chart):"))
        export_cluster_stats_btn = QPushButton("💾 Export")
        export_cluster_stats_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 3px;")
        export_cluster_stats_btn.clicked.connect(self.export_cluster_stats_chart)
        cluster_stats_header.addWidget(export_cluster_stats_btn)
        cluster_stats_header.addStretch()
        cluster_layout.addLayout(cluster_stats_header)
        
        # Bar chart display (instead of table)
        self.cluster_stats_chart_label = QLabel()
        self.cluster_stats_chart_label.setAlignment(Qt.AlignCenter)
        self.cluster_stats_chart_label.setMinimumHeight(200)
        self.cluster_stats_chart_label.setStyleSheet("border: 1px solid #ccc; background-color: white;")
        self.cluster_stats_chart_label.setText("Run clustering to see statistics chart")
        cluster_layout.addWidget(self.cluster_stats_chart_label)
        
        cluster_group.setLayout(cluster_layout)
        plots_layout.addWidget(cluster_group, 1)
        
        main_layout.addLayout(plots_layout)
        
        group.setLayout(main_layout)
        return group
    
    def create_visualization_section(self) -> QGroupBox:
        """Create visualization section."""
        group = QGroupBox("6️⃣ Visualization")
        group.setFont(QFont("Arial", 12, QFont.Bold))
        group.setVisible(False)  # Hidden until analysis is done
        
        layout = QVBoxLayout()
        
        # Color by selection and export buttons
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Color by:"))
        
        # Radio buttons for Metadata vs Feature
        if not hasattr(self, 'color_type_group'):
            self.color_type_group = QButtonGroup()
            self.color_metadata_radio = QRadioButton("Metadata")
            self.color_metadata_radio.setChecked(True)
            self.color_feature_radio = QRadioButton("Feature")
            self.color_type_group.addButton(self.color_metadata_radio, 0)
            self.color_type_group.addButton(self.color_feature_radio, 1)
            # Use buttonClicked signal from QButtonGroup
            self.color_type_group.buttonClicked.connect(self.update_color_by_options)
            # Also connect individual toggled signals as backup
            self.color_metadata_radio.toggled.connect(self.update_color_by_options)
            self.color_feature_radio.toggled.connect(self.update_color_by_options)
        
        top_layout.addWidget(self.color_metadata_radio)
        top_layout.addWidget(self.color_feature_radio)
        
        if not hasattr(self, 'color_by_combo'):
            self.color_by_combo = QComboBox()
        top_layout.addWidget(self.color_by_combo, 1)
        
        # Export buttons
        export_png_btn = QPushButton("💾 Export PNG")
        export_png_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 5px;")
        export_png_btn.clicked.connect(lambda: self.export_plot("png"))
        top_layout.addWidget(export_png_btn)
        
        export_pdf_btn = QPushButton("💾 Export PDF")
        export_pdf_btn.setStyleSheet("background-color: #F44336; color: white; padding: 5px;")
        export_pdf_btn.clicked.connect(lambda: self.export_plot("pdf"))
        top_layout.addWidget(export_pdf_btn)
        
        # Feature Importance button
        self.top10_features_btn = QPushButton("📊 Feature Importance")
        self.top10_features_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        self.top10_features_btn.clicked.connect(self.download_top10_features)
        # Show immediately if embedding already exists
        if self.session_state.get("embedding_df") is not None:
            self.top10_features_btn.setVisible(True)
        else:
            self.top10_features_btn.setVisible(False)  # Only show after analysis
        
        # PCA button should always be visible (needs features and metadata, not embedding_df)
        if hasattr(self, 'pca_plot_btn'):
            self.pca_plot_btn.setVisible(True)
        top_layout.addWidget(self.top10_features_btn)
        
        layout.addLayout(top_layout)
        
        # Axis limits adjustment
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("X-axis:"))
        self.x_min_input = QLineEdit()
        self.x_min_input.setPlaceholderText("min")
        self.x_min_input.setMaximumWidth(100)
        axis_layout.addWidget(self.x_min_input)
        axis_layout.addWidget(QLabel("to"))
        self.x_max_input = QLineEdit()
        self.x_max_input.setPlaceholderText("max")
        self.x_max_input.setMaximumWidth(100)
        axis_layout.addWidget(self.x_max_input)
        
        axis_layout.addWidget(QLabel("Y-axis:"))
        self.y_min_input = QLineEdit()
        self.y_min_input.setPlaceholderText("min")
        self.y_min_input.setMaximumWidth(100)
        axis_layout.addWidget(self.y_min_input)
        axis_layout.addWidget(QLabel("to"))
        self.y_max_input = QLineEdit()
        self.y_max_input.setPlaceholderText("max")
        self.y_max_input.setMaximumWidth(100)
        axis_layout.addWidget(self.y_max_input)
        
        apply_axis_btn = QPushButton("Apply Limits")
        apply_axis_btn.setStyleSheet("background-color: #607D8B; color: white; padding: 5px;")
        apply_axis_btn.clicked.connect(self.apply_axis_limits)
        axis_layout.addWidget(apply_axis_btn)
        
        reset_axis_btn = QPushButton("Reset")
        reset_axis_btn.setStyleSheet("background-color: #9E9E9E; color: white; padding: 5px;")
        reset_axis_btn.clicked.connect(self.reset_axis_limits)
        axis_layout.addWidget(reset_axis_btn)
        
        layout.addLayout(axis_layout)
        
        # Highlight cells input
        highlight_layout = QHBoxLayout()
        highlight_layout.addWidget(QLabel("Highlight cells (comma-separated indices):"))
        self.highlight_input = QLineEdit()
        self.highlight_input.setPlaceholderText("e.g., 100, 200, 300")
        highlight_layout.addWidget(self.highlight_input, 1)
        highlight_btn = QPushButton("✨ Highlight")
        highlight_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 5px;")
        highlight_btn.clicked.connect(self.highlight_cells)
        highlight_layout.addWidget(highlight_btn)
        layout.addLayout(highlight_layout)
        
        # Plot container
        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setMinimumHeight(400)
        self.plot_label.setStyleSheet("border: 1px solid #ccc; background-color: white;")
        layout.addWidget(self.plot_label)
        
        group.setLayout(layout)
        return group
    
    def update_visualization(self):
        """Update visualization plot."""
        if self.session_state.get("embedding_df") is None:
            if hasattr(self, 'viz_cluster_section'):
                self.viz_cluster_section.setVisible(False)
            elif hasattr(self, 'viz_section'):
                self.viz_section.setVisible(False)
            return
        
        # Make section visible first
        if hasattr(self, 'viz_cluster_section'):
            self.viz_cluster_section.setVisible(True)
            print("DEBUG update_visualization: viz_cluster_section set to visible")
        elif hasattr(self, 'viz_section'):
            self.viz_section.setVisible(True)
            print("DEBUG update_visualization: viz_section set to visible")
        
        # Ensure PCA button is visible
        if hasattr(self, 'pca_plot_btn'):
            self.pca_plot_btn.setVisible(True)
        
        # Process events to ensure UI is updated before drawing plot
        QApplication.processEvents()
        
        embedding_df = self.session_state["embedding_df"]
        method = self.session_state.get("stored_dim_reduction_method", "DensMAP")
        x_label, y_label = get_axis_labels(method)
        
        # Ensure Metadata radio is checked by default FIRST (before connecting signals)
        if hasattr(self, 'color_metadata_radio'):
            # Temporarily disconnect to avoid triggering update during initialization
            try:
                self.color_metadata_radio.toggled.disconnect()
            except:
                pass
            if not self.color_metadata_radio.isChecked():
                self.color_metadata_radio.setChecked(True)
                print("DEBUG update_visualization: Set Metadata radio to checked")
        else:
            print("DEBUG update_visualization: WARNING - color_metadata_radio not found!")
        
        # Update color by combo based on current selection - do this BEFORE connecting signals
        # This ensures metadata options are immediately available
        print("DEBUG update_visualization: Calling update_color_by_options() to populate dropdown")
        self.update_color_by_options()
        
        # NOW connect radio buttons to update combo box
        if hasattr(self, 'color_type_group'):
            try:
                self.color_type_group.buttonClicked.disconnect()
            except:
                pass
            self.color_type_group.buttonClicked.connect(self.update_color_by_options)
        if hasattr(self, 'color_metadata_radio'):
            try:
                self.color_metadata_radio.toggled.disconnect()
            except:
                pass
            self.color_metadata_radio.toggled.connect(self.update_color_by_options)
        if hasattr(self, 'color_feature_radio'):
            try:
                self.color_feature_radio.toggled.disconnect()
            except:
                pass
            self.color_feature_radio.toggled.connect(self.update_color_by_options)
        
        # Disconnect previous connection if it exists
        try:
            self.color_by_combo.currentTextChanged.disconnect()
        except (TypeError, RuntimeError, SystemError):
            pass
        
        # Connect signal with immediate callback (no timer delay for user interactions)
        def on_color_changed(text):
            print(f"DEBUG: Color changed to: {text}, triggering immediate redraw")
            # Force immediate redraw without any delay
            self.redraw_plot()
            # Also force UI update
            if hasattr(self, 'plot_label') and self.plot_label is not None:
                self.plot_label.update()
                self.plot_label.repaint()
        
        self.color_by_combo.currentTextChanged.connect(on_color_changed)
        
        # Set initial selection and trigger plot
        if self.color_by_combo.count() > 0:
            # Try to set "group" as default, otherwise use first item
            group_idx = -1
            for i in range(self.color_by_combo.count()):
                if self.color_by_combo.itemText(i) == "group":
                    group_idx = i
                    break
            if group_idx >= 0:
                self.color_by_combo.setCurrentIndex(group_idx)
                print(f"DEBUG update_visualization: Set default to 'group' (index {group_idx})")
            else:
                self.color_by_combo.setCurrentIndex(0)
                print(f"DEBUG update_visualization: Set default to first item: {self.color_by_combo.itemText(0)}")
            
            # CRITICAL: Force redraw plot immediately - don't rely on signal
            # The signal might not fire if the index is set programmatically
            print("DEBUG update_visualization: Forcing immediate plot redraw")
            current_text = self.color_by_combo.currentText()
            if current_text:
                print(f"DEBUG update_visualization: Current selection: {current_text}, triggering redraw")
                # Call redraw directly AND emit signal
                self.redraw_plot()
                # Also emit signal to ensure any connected handlers run
                self.color_by_combo.currentTextChanged.emit(current_text)
            else:
                # Even if no text, try to draw
                self.redraw_plot()
        else:
            print("DEBUG update_visualization: WARNING - color_by_combo is empty!")
            # Even if empty, try to draw a basic plot
            if self.session_state.get("embedding_df") is not None:
                print("DEBUG update_visualization: Drawing basic plot without color selection")
                QTimer.singleShot(0, self.redraw_plot)  # Execute in next event loop cycle
    
    def update_color_by_options(self):
        """Update color by combo box based on Metadata/Feature selection."""
        if not hasattr(self, 'color_by_combo') or self.color_by_combo is None:
            print("DEBUG update_color_by_options: color_by_combo not found")
            return
        
        self.color_by_combo.clear()
        
        if hasattr(self, 'color_metadata_radio') and self.color_metadata_radio.isChecked():
            # Show metadata columns
            print("DEBUG update_color_by_options: Metadata selected")
            if self.session_state.get("embedding_df") is not None:
                embedding_df = self.session_state["embedding_df"]
                # Build color options with "group" as first option if available
                color_options = []
                if "group" in embedding_df.columns:
                    color_options.append("group")
                if "sample_id" not in color_options:
                    color_options.append("sample_id")
                # Add other metadata columns
                other_cols = [c for c in embedding_df.columns if c not in ["x", "y", "sample_id", "cell_index", "cluster", "cluster_numeric", "highlighted", "group"]]
                color_options.extend(other_cols)
                self.color_by_combo.addItems(color_options)
                print(f"DEBUG update_color_by_options: Added {len(color_options)} metadata options")
            else:
                print("DEBUG update_color_by_options: No embedding_df available")
        elif hasattr(self, 'color_feature_radio') and self.color_feature_radio.isChecked():
            # Show features
            print("DEBUG update_color_by_options: Feature selected")
            features = self.session_state.get("features", [])
            print(f"DEBUG update_color_by_options: Found {len(features)} features in session_state")
            
            # If no features in session_state, try to load from CSV files
            if not features:
                print("DEBUG update_color_by_options: No features in session_state, trying to load from CSV files...")
                try:
                    project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
                    run_id = safe_str(self.session_state["run_id"])
                    paths = get_run_paths(safe_path(project_dir), run_id)
                    cache_dir = paths["csv_cache"]
                    
                    # Load features from first available CSV file
                    csv_files = sorted(cache_dir.glob("*.csv"))
                    if csv_files:
                        df = pd.read_csv(csv_files[0], nrows=0)  # Read only headers
                        # Exclude non-feature columns
                        exclude_cols = ["sample_id", "cell_index", "x", "y", "cluster", "cluster_numeric", "highlighted"]
                        features = [c for c in df.columns if c not in exclude_cols]
                        print(f"DEBUG update_color_by_options: Loaded {len(features)} features from {csv_files[0].name}")
                except Exception as e:
                    print(f"DEBUG update_color_by_options: Error loading features from CSV: {e}")
            
            if features:
                self.color_by_combo.addItems(features)
                print(f"DEBUG update_color_by_options: Added {len(features)} features to combo")
            else:
                print("DEBUG update_color_by_options: No features available")
                print(f"DEBUG update_color_by_options: session_state keys: {list(self.session_state.keys())}")
        else:
            print("DEBUG update_color_by_options: Neither radio button checked or not found")
            print(f"DEBUG update_color_by_options: has color_metadata_radio: {hasattr(self, 'color_metadata_radio')}")
            print(f"DEBUG update_color_by_options: has color_feature_radio: {hasattr(self, 'color_feature_radio')}")
            if hasattr(self, 'color_metadata_radio'):
                print(f"DEBUG update_color_by_options: metadata_radio checked: {self.color_metadata_radio.isChecked()}")
            if hasattr(self, 'color_feature_radio'):
                print(f"DEBUG update_color_by_options: feature_radio checked: {self.color_feature_radio.isChecked()}")
    
    def _load_feature_values(self, feature_name: str, embedding_df: pd.DataFrame) -> Optional[pd.Series]:
        """Load feature values for a specific feature and merge with embedding_df."""
        try:
            print(f"DEBUG _load_feature_values: Loading feature '{feature_name}'")
            # Get paths
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            cache_dir = paths["csv_cache"]
            
            print(f"DEBUG _load_feature_values: cache_dir={cache_dir}, exists={cache_dir.exists()}")
            
            # Get sample IDs from embedding
            sample_ids = embedding_df["sample_id"].astype(str).unique()
            print(f"DEBUG _load_feature_values: Found {len(sample_ids)} sample_ids: {sample_ids[:5]}")
            
            # Collect feature data for all samples
            all_feature_data = []
            # Try to find CSV files directly by sample_id
            csv_files = sorted(cache_dir.glob("*.csv"))
            print(f"DEBUG _load_feature_values: Found {len(csv_files)} CSV files in cache")
            
            for csv_path in csv_files:
                csv_stem = str(csv_path.stem)
                if csv_stem not in sample_ids:
                    print(f"DEBUG _load_feature_values: Skipping {csv_stem} (not in sample_ids)")
                    continue
                
                print(f"DEBUG _load_feature_values: Processing {csv_stem}")
                
                df = pd.read_csv(csv_path)
                print(f"DEBUG _load_feature_values: Loaded {len(df)} rows from {csv_stem}.csv, columns: {len(df.columns)}")
                
                # Filter by population if selected
                population = self.session_state.get("selected_population")
                if population and population in df.columns:
                    pop_series = df[population]
                    if isinstance(pop_series, pd.DataFrame):
                        pop_series = pop_series.iloc[:, 0]
                    pop_array = np.asarray(pop_series)
                    if pop_series.dtype == bool or str(pop_series.dtype) == 'bool':
                        mask = pop_array == True
                    else:
                        mask = pop_array == 1
                    df = df[mask]
                    print(f"DEBUG _load_feature_values: After population filter: {len(df)} rows")
                
                # Check if feature exists
                if feature_name not in df.columns:
                    print(f"DEBUG _load_feature_values: Feature '{feature_name}' not in columns: {df.columns[:10].tolist()}...")
                    continue
                
                print(f"DEBUG _load_feature_values: Feature '{feature_name}' found in {csv_stem}")
                
                # Add sample_id and cell_index
                df["sample_id"] = csv_stem
                df = df.reset_index(drop=True)
                df["cell_index"] = df.groupby("sample_id").cumcount()
                
                # Select only needed columns
                df_subset = df[["sample_id", "cell_index", feature_name]].copy()
                all_feature_data.append(df_subset)
                print(f"DEBUG _load_feature_values: Added {len(df_subset)} rows for {csv_stem}")
            
            if not all_feature_data:
                print("DEBUG _load_feature_values: No feature data collected")
                return None
            
            print(f"DEBUG _load_feature_values: Collected data from {len(all_feature_data)} samples")
            
            # Combine all feature data
            feature_data = pd.concat(all_feature_data, ignore_index=True)
            print(f"DEBUG _load_feature_values: Combined feature_data shape: {feature_data.shape}")
            
            # Merge with embedding_df - need to ensure proper alignment
            embedding_df_merge = embedding_df[["sample_id", "cell_index"]].copy()
            embedding_df_merge = embedding_df_merge.reset_index(drop=True)
            print(f"DEBUG _load_feature_values: embedding_df_merge shape: {embedding_df_merge.shape}")
            print(f"DEBUG _load_feature_values: feature_data shape: {feature_data.shape}")
            
            # Merge on sample_id and cell_index
            merged = embedding_df_merge.merge(
                feature_data,
                on=["sample_id", "cell_index"],
                how="left"
            )
            print(f"DEBUG _load_feature_values: Merged shape: {merged.shape}, matched: {merged[feature_name].notna().sum()} out of {len(merged)}")
            
            # Ensure the result has the same length as embedding_df
            if len(merged) != len(embedding_df):
                print(f"DEBUG _load_feature_values: WARNING - Length mismatch! merged={len(merged)}, embedding_df={len(embedding_df)}")
                # Reindex to match embedding_df
                merged = merged.set_index(embedding_df_merge.index)
            
            # Return feature values aligned with embedding_df
            if feature_name in merged.columns:
                result = merged[feature_name]
                # Ensure result has same index as embedding_df
                if len(result) != len(embedding_df):
                    print(f"DEBUG _load_feature_values: WARNING - Result length mismatch! result={len(result)}, embedding_df={len(embedding_df)}")
                    # Try to align by index
                    result = result.reindex(embedding_df.index, fill_value=np.nan)
                print(f"DEBUG _load_feature_values: Returning {len(result)} values, {result.notna().sum()} non-NaN")
                return result
            else:
                print(f"DEBUG _load_feature_values: Feature '{feature_name}' not in merged columns: {merged.columns.tolist()[:10]}")
                return None
                
        except Exception as e:
            print(f"Error loading feature values for {feature_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def redraw_plot(self):
        """Redraw the plot with current color selection."""
        if self.session_state.get("embedding_df") is None:
            return
        
        embedding_df = self.session_state["embedding_df"]
        color_by = self.color_by_combo.currentText() if self.color_by_combo.count() > 0 else "sample_id"
        method = self.session_state.get("stored_dim_reduction_method", "DensMAP")
        x_label, y_label = get_axis_labels(method)
        
        # Check if coloring by feature or metadata
        is_feature_coloring = hasattr(self, 'color_feature_radio') and self.color_feature_radio.isChecked()
        
        # Create plot - use smaller size so both plots fit side by side
        fig, ax = plt.subplots(figsize=(7, 6))
        
        unique_vals = []
        
        if is_feature_coloring:
            # Feature-based coloring (Heatmap style)
            print(f"DEBUG redraw_plot: Feature coloring selected, feature={color_by}")
            feature_values = self._load_feature_values(color_by, embedding_df)
            print(f"DEBUG redraw_plot: feature_values type={type(feature_values)}, length={len(feature_values) if feature_values is not None else 0}")
            
            if feature_values is not None and len(feature_values) > 0:
                print(f"DEBUG redraw_plot: feature_values has {feature_values.notna().sum()} non-NaN values out of {len(feature_values)}")
                if feature_values.notna().any():
                    # Filter out NaN values
                    valid_mask = feature_values.notna()
                    valid_embedding = embedding_df[valid_mask].copy()
                    valid_values = feature_values[valid_mask]
                    
                    print(f"DEBUG redraw_plot: Plotting {len(valid_embedding)} cells with feature values")
                    print(f"DEBUG redraw_plot: Feature value range: [{valid_values.min():.2f}, {valid_values.max():.2f}]")
                    
                    if len(valid_embedding) > 0:
                        # Use heatmap colormap (viridis for continuous values)
                        # Ensure valid_values is a numpy array or list
                        if isinstance(valid_values, pd.Series):
                            valid_values_array = valid_values.values
                        else:
                            valid_values_array = valid_values
                        
                        scatter = ax.scatter(valid_embedding["x"].values, valid_embedding["y"].values, 
                                           c=valid_values_array, 
                                           cmap="viridis", 
                                           alpha=0.6, 
                                           s=10,
                                           vmin=float(valid_values.min()),
                                           vmax=float(valid_values.max()))
                        
                        # Add colorbar
                        cbar = plt.colorbar(scatter, ax=ax)
                        cbar.set_label(color_by, fontsize=12, fontweight='bold')
                        
                        # Plot NaN values in gray if any
                        nan_mask = ~valid_mask
                        if nan_mask.any():
                            nan_embedding = embedding_df[nan_mask]
                            ax.scatter(nan_embedding["x"].values, nan_embedding["y"].values, c="gray", alpha=0.3, s=10, label="N/A")
                    else:
                        print("DEBUG redraw_plot: No valid embedding after filtering")
                        ax.scatter(embedding_df["x"], embedding_df["y"], alpha=0.6, s=10, c="blue")
                else:
                    print(f"DEBUG redraw_plot: All feature values are NaN")
                    ax.scatter(embedding_df["x"], embedding_df["y"], alpha=0.6, s=10, c="blue")
            else:
                # Feature not found - plot without coloring
                print(f"WARNING: Feature '{color_by}' not found or could not be loaded.")
                print(f"DEBUG redraw_plot: feature_values is None or empty")
                ax.scatter(embedding_df["x"], embedding_df["y"], alpha=0.6, s=10, c="blue")
        elif color_by in embedding_df.columns:
            # Metadata-based coloring (categorical)
            # Filter out NaN values for the color_by column
            valid_df = embedding_df[embedding_df[color_by].notna()].copy()
            if len(valid_df) > 0:
                unique_vals = valid_df[color_by].unique()
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))
                
                for val, color in zip(unique_vals, colors):
                    subset = valid_df[valid_df[color_by] == val]
                    if len(subset) > 0:
                        ax.scatter(subset["x"], subset["y"], c=[color], label=str(val), alpha=0.6, s=10)
            
            # Plot NaN values in gray if any
            nan_df = embedding_df[embedding_df[color_by].isna()]
            if len(nan_df) > 0:
                ax.scatter(nan_df["x"], nan_df["y"], c="gray", label="N/A", alpha=0.3, s=10)
        else:
            # Column not found - show warning and plot without coloring
            print(f"WARNING: Column '{color_by}' not found in embedding_df. Available columns: {embedding_df.columns.tolist()}")
            ax.scatter(embedding_df["x"], embedding_df["y"], alpha=0.6, s=10)
        
        # Highlight selected cells
        highlighted_indices = self.session_state.get("highlighted_cells", [])
        if highlighted_indices and "cell_index" in embedding_df.columns:
            highlighted_df = embedding_df[embedding_df["cell_index"].isin(highlighted_indices)]
            if len(highlighted_df) > 0:
                ax.scatter(highlighted_df["x"], highlighted_df["y"], c="red", s=100, marker="*", 
                          edgecolors="black", linewidths=2, zorder=10)  # Removed label from legend
                # Add labels for highlighted cells
                for idx, row in highlighted_df.iterrows():
                    ax.annotate(str(int(row["cell_index"])), 
                              (row["x"], row["y"]), 
                              xytext=(5, 5), textcoords="offset points",
                              fontsize=10, fontweight="bold", color="red",
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
        # Title will be shown in the label above the plot, not in the plot itself
        
        # Update title label above the plot
        color_type = "Feature" if is_feature_coloring else "Metadata"
        title_text = f"{method} colored by {color_type.lower()}: {color_by}"
        if hasattr(self, 'plot_title_label') and self.plot_title_label is not None:
            self.plot_title_label.setText(title_text)
            print(f"DEBUG redraw_plot: Updated title to: {title_text}")
        
        # Apply axis limits if set
        axis_limits = self.session_state.get("axis_limits", {})
        if axis_limits:
            if "x_min" in axis_limits and axis_limits["x_min"] is not None:
                ax.set_xlim(left=axis_limits["x_min"])
            if "x_max" in axis_limits and axis_limits["x_max"] is not None:
                ax.set_xlim(right=axis_limits["x_max"])
            if "y_min" in axis_limits and axis_limits["y_min"] is not None:
                ax.set_ylim(bottom=axis_limits["y_min"])
            if "y_max" in axis_limits and axis_limits["y_max"] is not None:
                ax.set_ylim(top=axis_limits["y_max"])
        
        if len(unique_vals) > 0 and len(unique_vals) <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        
        plt.tight_layout()
        
        # Convert to QPixmap
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")  # Increased DPI for better resolution
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        
        # Check if plot_label exists and has valid size
        if hasattr(self, 'plot_label') and self.plot_label is not None:
            # Get label size, use minimum if size is 0
            label_width = max(self.plot_label.width(), 800) if self.plot_label.width() > 0 else 800
            label_height = max(self.plot_label.height(), 600) if self.plot_label.height() > 0 else 600
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.plot_label.setPixmap(scaled_pixmap)
            # Force immediate update of the label
            self.plot_label.update()
            self.plot_label.repaint()
            print(f"DEBUG redraw_plot: Plot displayed in plot_label (size: {label_width}x{label_height})")
        else:
            print("DEBUG redraw_plot: WARNING - plot_label not found or None!")
        
        plt.close(fig)
    
    def export_plot(self, format: str = "png"):
        """Export current plot to file."""
        if self.session_state.get("embedding_df") is None:
            QMessageBox.warning(self, "Warning", "No plot to export. Run analysis first.")
            return
        
        # Get save path
        project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
        run_id = safe_str(self.session_state["run_id"])
        paths = get_run_paths(safe_path(project_dir), run_id)
        paths["results"].mkdir(parents=True, exist_ok=True)
        
        method = self.session_state.get("stored_dim_reduction_method", "DensMAP")
        color_by = self.color_by_combo.currentText() if self.color_by_combo.count() > 0 else "sample_id"
        
        # Generate filename
        filename = f"{method.lower()}_{color_by}.{format}"
        file_path = paths["results"] / filename
        
        # Recreate plot with high DPI for export - use same logic as redraw_plot
        embedding_df = self.session_state["embedding_df"]
        x_label, y_label = get_axis_labels(method)
        
        # Check if coloring by feature or metadata
        is_feature_coloring = hasattr(self, 'color_feature_radio') and self.color_feature_radio.isChecked()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_vals = []
        
        if is_feature_coloring:
            # Feature-based coloring (Heatmap style)
            feature_values = self._load_feature_values(color_by, embedding_df)
            
            if feature_values is not None and len(feature_values) > 0 and feature_values.notna().any():
                # Filter out NaN values
                valid_mask = feature_values.notna()
                valid_embedding = embedding_df[valid_mask].copy()
                valid_values = feature_values[valid_mask]
                
                if len(valid_embedding) > 0:
                    # Use heatmap colormap (viridis for continuous values)
                    scatter = ax.scatter(valid_embedding["x"], valid_embedding["y"], 
                                       c=valid_values, 
                                       cmap="viridis", 
                                       alpha=0.6, 
                                       s=10,
                                       vmin=valid_values.min(),
                                       vmax=valid_values.max())
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(color_by, fontsize=12, fontweight='bold')
                    
                    # Plot NaN values in gray if any
                    nan_mask = ~valid_mask
                    if nan_mask.any():
                        nan_embedding = embedding_df[nan_mask]
                        ax.scatter(nan_embedding["x"], nan_embedding["y"], c="gray", alpha=0.3, s=10, label="N/A")
                else:
                    ax.scatter(embedding_df["x"], embedding_df["y"], alpha=0.6, s=10, c="blue")
            else:
                ax.scatter(embedding_df["x"], embedding_df["y"], alpha=0.6, s=10, c="blue")
        elif color_by in embedding_df.columns:
            # Get the column as Series
            color_by_series = embedding_df[color_by]
            if isinstance(color_by_series, pd.DataFrame):
                color_by_series = color_by_series.iloc[:, 0]
            
            # Filter out NaN values for the color_by column
            # Also filter out "nan" strings
            valid_mask = color_by_series.notna() & (color_by_series.astype(str) != "nan") & (color_by_series.astype(str) != "")
            valid_df = embedding_df[valid_mask].copy()
            
            # Debug: Check what we have
            print(f"DEBUG redraw_plot: color_by={color_by}, total rows={len(embedding_df)}, valid rows={len(valid_df)}")
            print(f"DEBUG redraw_plot: color_by column type: {type(color_by_series)}")
            print(f"DEBUG redraw_plot: color_by unique values (first 10): {color_by_series.unique()[:10].tolist() if len(color_by_series) > 0 else 'EMPTY'}")
            
            if len(valid_df) == 0:
                print(f"DEBUG redraw_plot: WARNING - No valid rows for color_by={color_by}")
                print(f"DEBUG redraw_plot: sample_id values (first 10): {embedding_df['sample_id'].head(10).tolist() if 'sample_id' in embedding_df.columns else 'NO SAMPLE_ID COLUMN'}")
                print(f"DEBUG redraw_plot: color_by NaN count: {color_by_series.isna().sum()}")
                print(f"DEBUG redraw_plot: color_by 'nan' string count: {(color_by_series.astype(str) == 'nan').sum()}")
                # If no valid data, plot all data without coloring
                ax.scatter(embedding_df["x"], embedding_df["y"], alpha=0.6, s=10, c="blue", label="All cells")
            else:
                # Get unique values from the valid dataframe
                color_by_valid = valid_df[color_by]
                if isinstance(color_by_valid, pd.DataFrame):
                    color_by_valid = color_by_valid.iloc[:, 0]
                unique_vals = color_by_valid.unique()
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))
                
                for val, color in zip(unique_vals, colors):
                    # Convert val to string for comparison to handle type mismatches
                    subset = valid_df[valid_df[color_by].astype(str) == str(val)]
                    if len(subset) > 0:
                        ax.scatter(subset["x"], subset["y"], c=[color], label=str(val), alpha=0.6, s=10)
            
            # Plot NaN values in gray if any
            nan_mask = color_by_series.isna() | (color_by_series.astype(str) == "nan") | (color_by_series.astype(str) == "")
            nan_df = embedding_df[nan_mask]
            if len(nan_df) > 0:
                ax.scatter(nan_df["x"], nan_df["y"], c="gray", label="N/A", alpha=0.3, s=10)
        else:
            # Color_by column doesn't exist - plot all data
            print(f"DEBUG redraw_plot: color_by={color_by} not in columns: {embedding_df.columns.tolist()}")
            ax.scatter(embedding_df["x"], embedding_df["y"], alpha=0.6, s=10, c="blue", label="All cells")
        
        # Highlight selected cells
        highlighted_indices = self.session_state.get("highlighted_cells", [])
        if highlighted_indices and "cell_index" in embedding_df.columns:
            highlighted_df = embedding_df[embedding_df["cell_index"].isin(highlighted_indices)]
            if len(highlighted_df) > 0:
                ax.scatter(highlighted_df["x"], highlighted_df["y"], c="red", s=100, marker="*", 
                          edgecolors="black", linewidths=2, zorder=10)
                # Add labels for highlighted cells
                for idx, row in highlighted_df.iterrows():
                    ax.annotate(str(int(row["cell_index"])), 
                              (row["x"], row["y"]), 
                              xytext=(5, 5), textcoords="offset points",
                              fontsize=10, fontweight="bold", color="red",
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
        ax.set_title(f"{method} Visualization - {color_by}", fontsize=16, fontweight="bold")
        
        # Apply axis limits if set (same as redraw_plot)
        axis_limits = self.session_state.get("axis_limits", {})
        if axis_limits:
            if "x_min" in axis_limits and axis_limits["x_min"] is not None:
                ax.set_xlim(left=axis_limits["x_min"])
            if "x_max" in axis_limits and axis_limits["x_max"] is not None:
                ax.set_xlim(right=axis_limits["x_max"])
            if "y_min" in axis_limits and axis_limits["y_min"] is not None:
                ax.set_ylim(bottom=axis_limits["y_min"])
            if "y_max" in axis_limits and axis_limits["y_max"] is not None:
                ax.set_ylim(top=axis_limits["y_max"])
        
        if len(unique_vals) > 0 and len(unique_vals) <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        
        plt.tight_layout()
        
        # Save with high DPI
        plt.savefig(file_path, format=format, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        QMessageBox.information(self, "Success", f"✅ Plot exported to:\n{file_path}")
    
    def apply_axis_limits(self):
        """Apply axis limits to both plots (visualization and cluster)."""
        try:
            limits = {}
            if self.x_min_input.text().strip():
                limits["x_min"] = float(self.x_min_input.text().strip())
            if self.x_max_input.text().strip():
                limits["x_max"] = float(self.x_max_input.text().strip())
            if self.y_min_input.text().strip():
                limits["y_min"] = float(self.y_min_input.text().strip())
            if self.y_max_input.text().strip():
                limits["y_max"] = float(self.y_max_input.text().strip())
            
            self.session_state["axis_limits"] = limits
            # Update both plots
            self.redraw_plot()
            # Also update cluster plot if it exists
            if hasattr(self, 'update_cluster_plot') and self.session_state.get("cluster_labels") is not None:
                self.update_cluster_plot()
                print("DEBUG apply_axis_limits: Updated both visualization and cluster plots")
            else:
                print("DEBUG apply_axis_limits: Updated visualization plot (cluster plot not available)")
        except ValueError:
            QMessageBox.warning(self, "Warning", "⚠️ Invalid number format. Please enter valid numbers.")
    
    def reset_axis_limits(self):
        """Reset axis limits to automatic for both plots."""
        self.session_state["axis_limits"] = {}
        self.x_min_input.clear()
        self.x_max_input.clear()
        self.y_min_input.clear()
        self.y_max_input.clear()
        # Update both plots
        self.redraw_plot()
        # Also update cluster plot if it exists
        if hasattr(self, 'update_cluster_plot') and self.session_state.get("cluster_labels") is not None:
            self.update_cluster_plot()
            print("DEBUG reset_axis_limits: Reset limits for both visualization and cluster plots")
        else:
            print("DEBUG reset_axis_limits: Reset limits for visualization plot (cluster plot not available)")
    
    def download_top10_features(self):
        """Calculate and download Top10 Features for x and y dimensions in background."""
        if self.session_state.get("embedding_df") is None:
            QMessageBox.warning(self, "Warning", "No analysis results available. Run analysis first.")
            return
        
        # Show progress
        self.top10_features_btn.setEnabled(False)
        self.top10_features_btn.setText("⏳ Calculating...")
        
        # Start worker thread
        embedding_df = self.session_state["embedding_df"]
        features = self.session_state.get("features", [])
        project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
        run_id = safe_str(self.session_state["run_id"])
        paths = get_run_paths(safe_path(project_dir), run_id)
        
        worker = FeatureImportanceWorker(
            embedding_df,
            features,
            paths,
            self.session_state.get("selected_population"),
            self.session_state
        )
        worker.finished.connect(lambda success, msg: self.on_top10_finished(success, msg, worker))
        self.active_workers.append(worker)
        worker.start()
    
    def highlight_cells(self):
        """Highlight selected cells in the plot."""
        text = self.highlight_input.text().strip()
        if not text:
            # Clear highlights
            self.session_state["highlighted_cells"] = []
            self.redraw_plot()
            return
        
        try:
            # Parse comma-separated indices
            indices = [int(x.strip()) for x in text.split(",") if x.strip()]
            self.session_state["highlighted_cells"] = indices
            self.redraw_plot()
        except ValueError:
            QMessageBox.warning(self, "Warning", "⚠️ Invalid format. Please use comma-separated numbers (e.g., 100, 200, 300)")
    
    def export_cluster_plot(self, format: str = "png"):
        """Export cluster plot to file."""
        if self.session_state.get("cluster_labels") is None or self.session_state.get("embedding_df") is None:
            QMessageBox.warning(self, "Warning", "No cluster plot to export. Run clustering first.")
            return
        
        # Get save path
        project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
        run_id = safe_str(self.session_state["run_id"])
        paths = get_run_paths(safe_path(project_dir), run_id)
        paths["results"].mkdir(parents=True, exist_ok=True)
        
        method = self.session_state.get("stored_dim_reduction_method", "DensMAP")
        cluster_method = self.cluster_algo_combo.currentText() if hasattr(self, 'cluster_algo_combo') else "clustering"
        
        # Generate filename
        filename = f"{method.lower()}_clusters_{cluster_method.lower().replace(' ', '_')}.{format}"
        file_path = paths["results"] / filename
        
        # Recreate plot with high DPI for export
        embedding_df = self.session_state["embedding_df"].copy()
        embedding_df["cluster"] = self.session_state["cluster_labels"]
        embedding_df["cluster"] = embedding_df["cluster"].astype(str)
        embedding_df.loc[embedding_df["cluster"] == "-1", "cluster"] = "Noise"
        
        x_label, y_label = get_axis_labels(method)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_clusters = sorted(embedding_df["cluster"].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster, color in zip(unique_clusters, colors):
            subset = embedding_df[embedding_df["cluster"] == cluster]
            ax.scatter(subset["x"], subset["y"], c=[color], label=f"Cluster {cluster}", alpha=0.6, s=10)
        
        ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
        ax.set_title(f"{method} - {cluster_method} Clustering", fontsize=16, fontweight="bold")
        
        # Apply axis limits if set (same as main plot)
        axis_limits = self.session_state.get("axis_limits", {})
        if axis_limits:
            if "x_min" in axis_limits and axis_limits["x_min"] is not None:
                ax.set_xlim(left=axis_limits["x_min"])
            if "x_max" in axis_limits and axis_limits["x_max"] is not None:
                ax.set_xlim(right=axis_limits["x_max"])
            if "y_min" in axis_limits and axis_limits["y_min"] is not None:
                ax.set_ylim(bottom=axis_limits["y_min"])
            if "y_max" in axis_limits and axis_limits["y_max"] is not None:
                ax.set_ylim(top=axis_limits["y_max"])
        
        if len(unique_clusters) <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        
        plt.tight_layout()
        
        # Save with high DPI
        plt.savefig(file_path, format=format, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        QMessageBox.information(self, "Success", f"✅ Cluster plot exported to:\n{file_path}")
    
    def create_clustering_section(self) -> QGroupBox:
        """Create clustering section."""
        group = QGroupBox("7️⃣ Clustering")
        group.setFont(QFont("Arial", 12, QFont.Bold))
        # Show immediately if embedding already exists
        if self.session_state.get("embedding_df") is not None:
            group.setVisible(True)
        else:
            group.setVisible(False)  # Hidden until DR is done
        
        layout = QVBoxLayout()
        
        # Algorithm selection
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("Algorithm:"))
        self.cluster_algo_combo = QComboBox()
        self.cluster_algo_combo.addItems(["KMeans", "Gaussian Mixture Models", "HDBSCAN"])
        algo_layout.addWidget(self.cluster_algo_combo, 1)
        layout.addLayout(algo_layout)
        
        # Parameters
        self.cluster_params_container = QWidget()
        self.cluster_params_layout = QVBoxLayout(self.cluster_params_container)
        
        self.cluster_n_input = QLineEdit()
        self.cluster_n_input.setText("10")
        self.cluster_n_input.setPlaceholderText("Number of clusters")
        self.cluster_params_layout.addWidget(QLabel("Number of clusters:"))
        self.cluster_params_layout.addWidget(self.cluster_n_input)
        
        # Elbow plot button for KMeans
        self.elbow_plot_btn = QPushButton("📊 Download Elbow Plot")
        self.elbow_plot_btn.setStyleSheet("background-color: #FFC107; color: white; padding: 5px;")
        self.elbow_plot_btn.clicked.connect(self.download_elbow_plot)
        self.cluster_params_layout.addWidget(self.elbow_plot_btn)
        
        # Update parameters when algorithm changes
        self.cluster_algo_combo.currentTextChanged.connect(self.update_cluster_params)
        
        layout.addWidget(self.cluster_params_container)
        
        # Run button
        cluster_btn = QPushButton("🔬 Run Clustering")
        cluster_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 10px; font-size: 14px;")
        cluster_btn.clicked.connect(self.run_clustering)
        layout.addWidget(cluster_btn)
        
        # Export buttons for cluster plot
        export_cluster_layout = QHBoxLayout()
        export_cluster_png_btn = QPushButton("💾 Export PNG")
        export_cluster_png_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 5px;")
        export_cluster_png_btn.clicked.connect(lambda: self.export_cluster_plot("png"))
        export_cluster_layout.addWidget(export_cluster_png_btn)
        
        export_cluster_pdf_btn = QPushButton("💾 Export PDF")
        export_cluster_pdf_btn.setStyleSheet("background-color: #F44336; color: white; padding: 5px;")
        export_cluster_pdf_btn.clicked.connect(lambda: self.export_cluster_plot("pdf"))
        export_cluster_layout.addWidget(export_cluster_pdf_btn)
        layout.addLayout(export_cluster_layout)
        
        # Cluster statistics table
        stats_header = QHBoxLayout()
        stats_header.addWidget(QLabel("📊 Cluster Statistics:"))
        export_stats_btn = QPushButton("📊 Export Bar Chart")
        export_stats_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px;")
        export_stats_btn.clicked.connect(self.export_cluster_stats_chart)
        stats_header.addWidget(export_stats_btn)
        stats_header.addStretch()
        layout.addLayout(stats_header)
        
        # Cluster-Feature Analysis buttons
        analysis_btn_layout = QHBoxLayout()
        
        top5_features_btn = QPushButton("🔍 Top 5 Features per Cluster")
        top5_features_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px;")
        top5_features_btn.clicked.connect(self.export_top3_features)  # Function name stays the same
        analysis_btn_layout.addWidget(top5_features_btn)
        
        heatmap_btn = QPushButton("🔥 Cluster-Feature Heatmap")
        heatmap_btn.setStyleSheet("background-color: #E91E63; color: white; padding: 5px;")
        heatmap_btn.clicked.connect(self.export_cluster_heatmap)
        analysis_btn_layout.addWidget(heatmap_btn)
        
        cluster_cells_btn = QPushButton("📋 Export Cluster Cells")
        cluster_cells_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 5px;")
        cluster_cells_btn.clicked.connect(self.export_cluster_cells)
        analysis_btn_layout.addWidget(cluster_cells_btn)
        
        layout.addLayout(analysis_btn_layout)
        
        self.cluster_stats_table = QTableWidget()
        self.cluster_stats_table.setColumnCount(4)
        self.cluster_stats_table.setHorizontalHeaderLabels(["Cluster", "Size", "Percentage", "Sample Distribution"])
        self.cluster_stats_table.horizontalHeader().setStretchLastSection(True)
        self.cluster_stats_table.setMaximumHeight(200)
        layout.addWidget(self.cluster_stats_table)
        
        # Cluster plot
        self.cluster_plot_label = QLabel()
        self.cluster_plot_label.setAlignment(Qt.AlignCenter)
        self.cluster_plot_label.setMinimumHeight(400)
        self.cluster_plot_label.setStyleSheet("border: 1px solid #ccc; background-color: white;")
        layout.addWidget(self.cluster_plot_label)
        
        group.setLayout(layout)
        return group
    
    def update_clustering_section(self):
        """Update clustering section visibility."""
        if self.session_state.get("embedding_df") is not None:
            if hasattr(self, 'viz_cluster_section'):
                self.viz_cluster_section.setVisible(True)
            elif hasattr(self, 'clustering_section'):
                self.clustering_section.setVisible(True)
                print(f"DEBUG: Clustering section set to visible")
            else:
                print(f"DEBUG: WARNING - clustering_section not found!")
    
    def update_cluster_params(self):
        """Update clustering parameters based on selected algorithm."""
        # Clear existing parameters
        for i in reversed(range(self.cluster_params_layout.count())):
            item = self.cluster_params_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        
        method = self.cluster_algo_combo.currentText()
        
        if method == "KMeans":
            # Number of clusters
            self.cluster_n_input = QLineEdit()
            self.cluster_n_input.setText("10")
            self.cluster_n_input.setPlaceholderText("Number of clusters")
            self.cluster_params_layout.addWidget(QLabel("Number of clusters:"))
            self.cluster_params_layout.addWidget(self.cluster_n_input)
            
            # Elbow plot button
            self.elbow_plot_btn = QPushButton("📊 Download Elbow Plot")
            self.elbow_plot_btn.setStyleSheet("background-color: #FFC107; color: white; padding: 5px;")
            self.elbow_plot_btn.clicked.connect(self.download_elbow_plot)
            self.cluster_params_layout.addWidget(self.elbow_plot_btn)
            
        elif method == "Gaussian Mixture Models":
            # Number of clusters
            self.cluster_n_input = QLineEdit()
            self.cluster_n_input.setText("10")
            self.cluster_n_input.setPlaceholderText("Number of clusters")
            self.cluster_params_layout.addWidget(QLabel("Number of clusters:"))
            self.cluster_params_layout.addWidget(self.cluster_n_input)
            
            # Covariance type
            cov_layout = QHBoxLayout()
            cov_layout.addWidget(QLabel("Covariance type:"))
            self.cov_type_combo = QComboBox()
            self.cov_type_combo.addItems(["full", "tied", "diag", "spherical"])
            cov_layout.addWidget(self.cov_type_combo, 1)
            self.cluster_params_layout.addLayout(cov_layout)
            
        elif method == "HDBSCAN":
            # Min cluster size slider
            # Default: target ~10 clusters by using larger min_cluster_size
            # Calculate default based on embedding size if available
            default_min_size = 500  # Default fallback
            if self.session_state.get("embedding_df") is not None:
                n_cells = len(self.session_state["embedding_df"])
                default_min_size = max(500, int(n_cells / 10))
                default_min_size = min(default_min_size, 5000)  # Cap at 5000
            
            self.cluster_params_layout.addWidget(QLabel("Min cluster size:"))
            min_size_slider = QSlider(Qt.Horizontal)
            min_size_slider.setMinimum(10)
            min_size_slider.setMaximum(5000)
            min_size_slider.setValue(default_min_size)
            min_size_slider.setTickPosition(QSlider.TicksBelow)
            min_size_slider.setTickInterval(500)
            min_size_label = QLabel(str(default_min_size))
            min_size_slider.valueChanged.connect(lambda v: min_size_label.setText(str(v)))
            self.cluster_params_layout.addWidget(min_size_slider)
            self.cluster_params_layout.addWidget(min_size_label)
            self.hdbscan_min_size = min_size_slider
            
            # Min samples slider
            self.cluster_params_layout.addWidget(QLabel("Min samples:"))
            min_samples_slider = QSlider(Qt.Horizontal)
            min_samples_slider.setMinimum(5)
            min_samples_slider.setMaximum(100)
            min_samples_slider.setValue(10)
            min_samples_slider.setTickPosition(QSlider.TicksBelow)
            min_samples_slider.setTickInterval(10)
            min_samples_label = QLabel("10")
            min_samples_slider.valueChanged.connect(lambda v: min_samples_label.setText(str(v)))
            self.cluster_params_layout.addWidget(min_samples_slider)
            self.cluster_params_layout.addWidget(min_samples_label)
            self.hdbscan_min_samples = min_samples_slider
    
    def run_clustering(self):
        """Run clustering analysis."""
        if self.session_state.get("embedding_df") is None:
            QMessageBox.warning(self, "Warning", "⚠️ Please run dimensionality reduction first")
            return
        
        method = self.cluster_algo_combo.currentText()
        n_clusters = 10
        method_params = {}
        
        if method == "KMeans":
            try:
                n_clusters = int(self.cluster_n_input.text())
            except (ValueError, AttributeError):
                QMessageBox.warning(self, "Warning", "⚠️ Please enter a valid number of clusters")
                return
            method = "KMeans"
            
        elif method == "Gaussian Mixture Models":
            try:
                n_clusters = int(self.cluster_n_input.text())
            except (ValueError, AttributeError):
                QMessageBox.warning(self, "Warning", "⚠️ Please enter a valid number of clusters")
                return
            method = "GMM"
            if hasattr(self, 'cov_type_combo'):
                method_params["covariance_type"] = self.cov_type_combo.currentText()
            
        else:  # HDBSCAN
            method = "HDBSCAN"
            if hasattr(self, 'hdbscan_min_size'):
                method_params["min_cluster_size"] = self.hdbscan_min_size.value()
            else:
                # Default: target ~10 clusters
                embedding_df = self.session_state.get("embedding_df")
                if embedding_df is not None:
                    n_cells = len(embedding_df)
                    default_min_size = max(500, int(n_cells / 10))
                    default_min_size = min(default_min_size, 5000)
                else:
                    default_min_size = 500
                method_params["min_cluster_size"] = default_min_size
            if hasattr(self, 'hdbscan_min_samples'):
                method_params["min_samples"] = self.hdbscan_min_samples.value()
            else:
                method_params["min_samples"] = 10
        
        embedding_df = self.session_state["embedding_df"]
        
        # Start worker thread
        self.clustering_worker = ClusteringWorker(embedding_df, method, n_clusters, self.session_state, method_params)
        # Use default argument to capture worker reference correctly
        self.clustering_worker.finished.connect(
            lambda success, msg, w=self.clustering_worker: self.on_clustering_finished(success, msg, w)
        )
        self.active_workers.append(self.clustering_worker)  # Track worker for cleanup
        self.clustering_worker.start()
    
    def on_clustering_finished(self, success: bool, message: str, worker: ClusteringWorker):
        """Handle clustering completion."""
        if success:
            QMessageBox.information(self, "Success", message)
            
            # Save clustering info to run info
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            
            cluster_method = self.session_state.get("cluster_method", "Unknown")
            n_clusters = len(set(self.session_state["cluster_labels"])) - (1 if -1 in self.session_state["cluster_labels"] else 0)
            cluster_params = {}
            if hasattr(worker, 'method_params'):
                cluster_params = worker.method_params
            
            self._save_run_info(
                paths,
                clustering={
                    "method": cluster_method,
                    "n_clusters": n_clusters,
                    "parameters": cluster_params,
                }
            )
            
            # Save cluster labels to disk for later loading
            try:
                cluster_labels_df = pd.DataFrame({
                    "cell_index": range(len(self.session_state["cluster_labels"])),
                    "cluster": self.session_state["cluster_labels"]
                })
                cluster_labels_path = paths["results"] / "cluster_labels.csv"
                cluster_labels_df.to_csv(cluster_labels_path, index=False)
                print(f"DEBUG: Saved cluster labels to {cluster_labels_path}")
            except Exception as e:
                print(f"Warning: Could not save cluster labels: {e}")
            
            self.update_status()
            self.update_cluster_plot()
        else:
            QMessageBox.critical(self, "Error", message)
        
        # Remove worker from active list once finished
        if worker in self.active_workers:
            self.active_workers.remove(worker)
    
    def update_cluster_plot(self):
        """Update cluster visualization plot and statistics."""
        if self.session_state.get("cluster_labels") is None or self.session_state.get("embedding_df") is None:
            return
        
        embedding_df = self.session_state["embedding_df"].copy()
        cluster_labels = self.session_state["cluster_labels"]
        embedding_df["cluster"] = cluster_labels
        embedding_df["cluster"] = embedding_df["cluster"].astype(str)
        embedding_df.loc[embedding_df["cluster"] == "-1", "cluster"] = "Noise"
        
        # Store paths for later use in export functions
        if not hasattr(self, 'paths'):
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            self.paths = get_run_paths(safe_path(project_dir), run_id)
        
        # Calculate cluster statistics
        cluster_counts = embedding_df["cluster"].value_counts().sort_index()
        total_cells = len(embedding_df)
        
        # Update statistics table(s) - update both if they exist
        stats_tables = []
        if hasattr(self, 'cluster_stats_table'):
            stats_tables.append(self.cluster_stats_table)
        if hasattr(self, 'viz_stats_table'):
            stats_tables.append(self.viz_stats_table)
        
        for stats_table in stats_tables:
            stats_table.setRowCount(len(cluster_counts))
            for idx, (cluster, count) in enumerate(cluster_counts.items()):
                percentage = (count / total_cells) * 100
                
                # Sample distribution
                cluster_df = embedding_df[embedding_df["cluster"] == cluster]
                if "sample_id" in cluster_df.columns:
                    sample_dist = cluster_df["sample_id"].value_counts().to_dict()
                    sample_str = ", ".join([f"{k}: {v}" for k, v in list(sample_dist.items())[:3]])
                    if len(sample_dist) > 3:
                        sample_str += f" (+{len(sample_dist) - 3} more)"
                else:
                    sample_str = "N/A"
                
                stats_table.setItem(idx, 0, QTableWidgetItem(str(cluster)))
                stats_table.setItem(idx, 1, QTableWidgetItem(str(count)))
                stats_table.setItem(idx, 2, QTableWidgetItem(f"{percentage:.2f}%"))
                stats_table.setItem(idx, 3, QTableWidgetItem(sample_str))
        
        method = self.session_state.get("stored_dim_reduction_method", "DensMAP")
        x_label, y_label = get_axis_labels(method)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_clusters = sorted(embedding_df["cluster"].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster, color in zip(unique_clusters, colors):
            subset = embedding_df[embedding_df["cluster"] == cluster]
            ax.scatter(subset["x"], subset["y"], c=[color], label=f"Cluster {cluster}", alpha=0.6, s=10)
        
        ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
        ax.set_title(f"Clustering Results ({self.session_state.get('cluster_method', 'Unknown')})", fontsize=16, fontweight="bold")
        
        # Apply axis limits if set (same as main plot)
        axis_limits = self.session_state.get("axis_limits", {})
        if axis_limits:
            if "x_min" in axis_limits and axis_limits["x_min"] is not None:
                ax.set_xlim(left=axis_limits["x_min"])
            if "x_max" in axis_limits and axis_limits["x_max"] is not None:
                ax.set_xlim(right=axis_limits["x_max"])
            if "y_min" in axis_limits and axis_limits["y_min"] is not None:
                ax.set_ylim(bottom=axis_limits["y_min"])
            if "y_max" in axis_limits and axis_limits["y_max"] is not None:
                ax.set_ylim(top=axis_limits["y_max"])
        
        if len(unique_clusters) <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        
        plt.tight_layout()
        
        # Convert to QPixmap
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")  # Increased DPI for better resolution
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        self.cluster_plot_label.setPixmap(pixmap.scaled(self.cluster_plot_label.width(), self.cluster_plot_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        plt.close(fig)
        
        # Update bar chart display if available
        if hasattr(self, 'cluster_stats_chart_label'):
            self.update_cluster_stats_chart_display()
    
    def update_cluster_stats_chart_display(self):
        """Update the cluster statistics bar chart display directly in the GUI."""
        if self.session_state.get("cluster_labels") is None or self.session_state.get("embedding_df") is None:
            if hasattr(self, 'cluster_stats_chart_label'):
                self.cluster_stats_chart_label.setText("Run clustering to see statistics chart")
            return
        
        embedding_df = self.session_state["embedding_df"].copy()
        embedding_df["cluster"] = self.session_state["cluster_labels"]
        embedding_df["cluster"] = embedding_df["cluster"].astype(str)
        embedding_df.loc[embedding_df["cluster"] == "-1", "cluster"] = "Noise"
        
        # Check if group column exists
        if "group" not in embedding_df.columns:
            if hasattr(self, 'cluster_stats_chart_label'):
                self.cluster_stats_chart_label.setText("Add 'group' column to metadata to see statistics chart")
            return
        
        # Filter out NaN and empty string groups
        embedding_df = embedding_df[embedding_df["group"].notna()]
        embedding_df = embedding_df[embedding_df["group"].astype(str).str.strip() != ""]
        
        if len(embedding_df) == 0:
            if hasattr(self, 'cluster_stats_chart_label'):
                self.cluster_stats_chart_label.setText("No valid groups found. Add group values to metadata.")
            return
        
        # Calculate percentage per cluster and group
        cluster_group_counts = embedding_df.groupby(["cluster", "group"]).size().reset_index(name="count")
        cluster_totals = embedding_df.groupby("cluster").size().reset_index(name="total")
        cluster_group_counts = cluster_group_counts.merge(cluster_totals, on="cluster")
        cluster_group_counts["percentage"] = (cluster_group_counts["count"] / cluster_group_counts["total"]) * 100
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        
        clusters = sorted(cluster_group_counts["cluster"].unique())
        groups = sorted(cluster_group_counts["group"].dropna().unique())
        
        if len(groups) == 0:
            if hasattr(self, 'cluster_stats_chart_label'):
                self.cluster_stats_chart_label.setText("No groups found in data.")
            plt.close(fig)
            return
        
        x = np.arange(len(clusters))
        width = 0.8 / len(groups) if len(groups) > 1 else 0.8
        
        # Create bars for each group (stacked)
        bottom = np.zeros(len(clusters))
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        for idx, group in enumerate(groups):
            values = []
            for cluster in clusters:
                subset = cluster_group_counts[(cluster_group_counts["cluster"] == cluster) & 
                                             (cluster_group_counts["group"] == group)]
                values.append(subset["percentage"].iloc[0] if len(subset) > 0 else 0)
            
            ax.bar(x, values, width, label=str(group), bottom=bottom, color=colors[idx])
            bottom += np.array(values)
        
        ax.set_xlabel("Cluster", fontsize=10, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
        ax.set_title("Cluster Distribution by Group", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(clusters, fontsize=8)
        if len(groups) > 0 and len(bottom) > 0:
            ax.legend(title="Group", fontsize=8, loc="upper right")
        ax.set_ylim(0, max(100, float(bottom.max()) * 1.1) if len(bottom) > 0 and bottom.max() > 0 else 100)
        plt.tight_layout()
        
        # Convert to QPixmap and display
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        if hasattr(self, 'cluster_stats_chart_label'):
            self.cluster_stats_chart_label.setPixmap(pixmap.scaled(
                self.cluster_stats_chart_label.width(), 
                self.cluster_stats_chart_label.height(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        plt.close(fig)
    
    def download_elbow_plot(self):
        """Calculate and download Elbow plot for KMeans."""
        if self.session_state.get("embedding_df") is None:
            QMessageBox.warning(self, "Warning", "No embedding data available. Run analysis first.")
            return
        
        try:
            embedding_df = self.session_state["embedding_df"]
            coords = embedding_df[["x", "y"]].values
            
            # Calculate inertia for different k values
            k_range = range(2, 21)  # Test k from 2 to 20
            inertias = []
            
            from sklearn.cluster import KMeans
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(coords)
                inertias.append(kmeans.inertia_)
            
            # Create elbow plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight="bold")
            ax.set_ylabel('Inertia', fontsize=12, fontweight="bold")
            ax.set_title('KMeans Elbow Plot', fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            paths["results"].mkdir(parents=True, exist_ok=True)
            
            output_path = paths["results"] / "kmeans_elbow_plot.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            QMessageBox.information(self, "Success", f"✅ Elbow plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ Error creating elbow plot: {str(e)}")
    
    def export_cluster_stats_chart(self):
        """Export cluster statistics as bar chart grouped by groups."""
        if self.session_state.get("cluster_labels") is None or self.session_state.get("embedding_df") is None:
            QMessageBox.warning(self, "Warning", "No cluster data available. Run clustering first.")
            return
        
        embedding_df = self.session_state["embedding_df"].copy()
        embedding_df["cluster"] = self.session_state["cluster_labels"]
        embedding_df["cluster"] = embedding_df["cluster"].astype(str)
        embedding_df.loc[embedding_df["cluster"] == "-1", "cluster"] = "Noise"
        
        # Check if group column exists
        if "group" not in embedding_df.columns:
            # Check if metadata has group but wasn't merged
            metadata_df = self.session_state.get("metadata_df")
            has_group_in_metadata = metadata_df is not None and "group" in metadata_df.columns if metadata_df is not None else False
            
            if has_group_in_metadata:
                # Metadata has group but merge failed
                embedding_sample_ids = sorted(embedding_df["sample_id"].unique().tolist())[:10]
                metadata_sample_ids = sorted(metadata_df["sample_id"].unique().tolist())[:10] if "sample_id" in metadata_df.columns else []
                metadata_file_names = sorted(metadata_df["file_name"].unique().tolist())[:10] if "file_name" in metadata_df.columns else []
                
                msg = (
                    f"Metadata has 'group' column but it wasn't merged with analysis results.\n\n"
                    f"Available columns in analysis: {', '.join(embedding_df.columns.tolist())}\n\n"
                    f"Sample IDs in analysis (first 10):\n{', '.join(map(str, embedding_sample_ids))}\n\n"
                )
                if metadata_sample_ids:
                    msg += f"Sample IDs in metadata (first 10):\n{', '.join(map(str, metadata_sample_ids))}\n\n"
                if metadata_file_names:
                    msg += f"File names in metadata (first 10):\n{', '.join(map(str, metadata_file_names))}\n\n"
                msg += (
                    "The sample_id in metadata must match the FCS file names (without .fcs extension).\n\n"
                    "Please check:\n"
                    "1. That sample_id in metadata matches the FCS file names\n"
                    "2. Save metadata and re-run analysis"
                )
                QMessageBox.warning(self, "Warning", msg)
            else:
                QMessageBox.warning(self, "Warning", f"No 'group' column found in metadata. Available columns: {', '.join(embedding_df.columns.tolist())}\n\nPlease add a 'group' column to your metadata and save it.")
            return
        
        # Filter out NaN and empty string groups
        embedding_df = embedding_df[embedding_df["group"].notna()]
        embedding_df = embedding_df[embedding_df["group"].astype(str).str.strip() != ""]
        
        if len(embedding_df) == 0:
            # Check if there are any groups at all (even if NaN)
            total_with_group_col = len(self.session_state["embedding_df"])
            
            # Check if group column exists in embedding_df
            if "group" not in self.session_state["embedding_df"].columns:
                # Try to check metadata directly
                metadata_df = self.session_state.get("metadata_df")
                if metadata_df is not None and "group" in metadata_df.columns:
                    # Metadata has group, but it wasn't merged - this is a merge issue
                    available_sample_ids = self.session_state["embedding_df"]["sample_id"].unique()[:5].tolist()
                    metadata_sample_ids = metadata_df["sample_id"].unique()[:5].tolist() if "sample_id" in metadata_df.columns else []
                    
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"Metadata has 'group' column but it wasn't merged with analysis results.\n\n"
                        f"Total cells: {total_with_group_col}\n\n"
                        f"Sample IDs in analysis (first 5): {', '.join(map(str, available_sample_ids))}\n"
                        f"Sample IDs in metadata (first 5): {', '.join(map(str, metadata_sample_ids))}\n\n"
                        "This might indicate a mismatch between sample_id in metadata and analysis.\n\n"
                        "Please check:\n"
                        "1. That sample_id in metadata matches file names (without .fcs extension)\n"
                        "2. Save metadata and re-run analysis"
                    )
                else:
                    QMessageBox.warning(
                        self, 
                        "Warning", 
                        f"No 'group' column found in metadata.\n\n"
                        f"Total cells: {total_with_group_col}\n\n"
                        "Please:\n"
                        "1. Add a 'group' column to your metadata\n"
                        "2. Fill in group values for each sample\n"
                        "3. Save the metadata\n"
                        "4. Re-run the analysis if needed"
                    )
            else:
                # Ensure we get a scalar, not a Series
                group_col = self.session_state["embedding_df"]["group"]
                if isinstance(group_col, pd.DataFrame):
                    group_col = group_col.iloc[:, 0]
                groups_found = int(group_col.notna().sum())
                if groups_found == 0:
                    QMessageBox.warning(
                        self, 
                        "Warning", 
                        f"No valid groups found in metadata.\n\n"
                        f"Total cells: {total_with_group_col}\n"
                        f"Cells with group values: {groups_found}\n\n"
                        "Please:\n"
                        "1. Add a 'group' column to your metadata\n"
                        "2. Fill in group values for each sample\n"
                        "3. Save the metadata\n"
                        "4. Re-run the analysis if needed"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"Found {groups_found} cells with group values, but none match the clusters.\n\n"
                        "This might indicate a mismatch between metadata and clustering results."
                    )
            return
        
        # Calculate percentage per cluster and group
        cluster_group_counts = embedding_df.groupby(["cluster", "group"]).size().reset_index(name="count")
        cluster_totals = embedding_df.groupby("cluster").size().reset_index(name="total")
        cluster_group_counts = cluster_group_counts.merge(cluster_totals, on="cluster")
        cluster_group_counts["percentage"] = (cluster_group_counts["count"] / cluster_group_counts["total"]) * 100
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        clusters = sorted(cluster_group_counts["cluster"].unique())
        groups = sorted(cluster_group_counts["group"].dropna().unique())
        
        if len(groups) == 0:
            QMessageBox.warning(self, "Warning", "No groups found in data.")
            plt.close(fig)
            return
        
        x = np.arange(len(clusters))
        width = 0.8 / len(groups) if len(groups) > 1 else 0.8
        
        # Create bars for each group (stacked)
        bottom = np.zeros(len(clusters))
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        for idx, group in enumerate(groups):
            values = []
            for cluster in clusters:
                subset = cluster_group_counts[(cluster_group_counts["cluster"] == cluster) & 
                                             (cluster_group_counts["group"] == group)]
                values.append(subset["percentage"].iloc[0] if len(subset) > 0 else 0)
            
            ax.bar(x, values, width, label=str(group), bottom=bottom, color=colors[idx])
            bottom += np.array(values)
        
        ax.set_xlabel("Cluster", fontsize=12, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
        ax.set_title("Cluster Distribution by Group", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(clusters)
        # Only show legend if there are labels
        if len(groups) > 0 and len(bottom) > 0:
            ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_ylim(0, max(100, float(bottom.max()) * 1.1) if len(bottom) > 0 and bottom.max() > 0 else 100)
        plt.tight_layout()
        
        # Save plot
        project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
        run_id = safe_str(self.session_state["run_id"])
        paths = get_run_paths(safe_path(project_dir), run_id)
        paths["results"].mkdir(parents=True, exist_ok=True)
        
        output_path = paths["results"] / "cluster_stats_bar_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        QMessageBox.information(self, "Success", f"✅ Cluster statistics chart saved to:\n{output_path}")
    
    def export_top3_features(self):
        """Export top 3 features per cluster as CSV."""
        if self.session_state.get("cluster_labels") is None or self.session_state.get("embedding_df") is None:
            QMessageBox.warning(self, "Warning", "No cluster data available. Run clustering first.")
            return
        
        try:
            # Get features and cluster data
            features = self.session_state.get("features", [])
            if not features:
                QMessageBox.warning(self, "Warning", "No features available. Run analysis first.")
                return
            
            embedding_df = self.session_state["embedding_df"].copy()
            cluster_labels = self.session_state["cluster_labels"]
            embedding_df["cluster"] = cluster_labels
            embedding_df["cluster"] = embedding_df["cluster"].astype(str)
            embedding_df.loc[embedding_df["cluster"] == "-1", "cluster"] = "Noise"
            
            # Get paths
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            
            # Load original feature data from CSV cache
            cache_dir = paths["csv_cache"]
            
            # Ensure sample_id is a Series, not DataFrame
            if "sample_id" not in embedding_df.columns:
                QMessageBox.warning(self, "Warning", "sample_id column missing in embedding_df.")
                return
            
            sample_id_series = embedding_df["sample_id"]
            if isinstance(sample_id_series, pd.DataFrame):
                sample_id_series = sample_id_series.iloc[:, 0]
            
            sample_ids = set(sample_id_series.astype(str).unique())
            
            # Collect feature data for all samples
            all_feature_data = []
            for fcs_file in sorted(paths["fcs"].glob("*.fcs")):
                fcs_stem = str(fcs_file.stem)
                if fcs_stem not in sample_ids:
                    continue
                
                csv_path = cache_dir / f"{fcs_stem}.csv"
                if not csv_path.exists():
                    continue
                
                df = pd.read_csv(csv_path)
                
                # Filter by population if selected
                population = self.session_state.get("selected_population")
                if population and population in df.columns:
                    # Get population column as Series
                    pop_series = df[population]
                    if isinstance(pop_series, pd.DataFrame):
                        pop_series = pop_series.iloc[:, 0]
                    
                    # Check dtype safely
                    pop_dtype = pop_series.dtype
                    if pop_dtype == bool:
                        # Use .values to avoid Series ambiguity
                        mask = pop_series.values == True
                        df = df[mask]
                    else:
                        # For numeric types, check if value equals 1
                        mask = pop_series.values == 1
                        df = df[mask]
                
                # Select only available features
                available_features = [f for f in features if f in df.columns]
                if not available_features:
                    continue
                
                subset = df[available_features].copy()
                subset["sample_id"] = fcs_stem
                all_feature_data.append(subset)
            
            if not all_feature_data:
                QMessageBox.warning(self, "Warning", "Could not load feature data from CSV cache.")
                return
            
            # Combine all data
            feature_data = pd.concat(all_feature_data, ignore_index=True)
            feature_data["sample_id"] = feature_data["sample_id"].astype(str)
            
            # Calculate mean feature values per cluster
            # Ensure cluster is a Series
            cluster_series = embedding_df["cluster"]
            if isinstance(cluster_series, pd.DataFrame):
                cluster_series = cluster_series.iloc[:, 0]
            clusters = sorted(cluster_series.unique())
            
            # First, calculate mean feature values for ALL clusters
            all_cluster_means = {}
            for cluster in clusters:
                cluster_df = embedding_df[embedding_df["cluster"] == cluster]
                sample_counts = cluster_df["sample_id"].value_counts()
                
                cluster_feature_values = {}
                for feature in features:
                    if feature not in feature_data.columns:
                        continue
                    
                    # Calculate weighted mean across samples in this cluster
                    weighted_sum = 0
                    total_weight = 0
                    for sample_id, count in sample_counts.items():
                        sample_data = feature_data[feature_data["sample_id"] == sample_id]
                        if len(sample_data) > 0:
                            # Sample same number of cells as in cluster (or all if fewer)
                            n_sample = min(count, len(sample_data))
                            sampled = sample_data.sample(n=n_sample, random_state=42) if n_sample < len(sample_data) else sample_data
                            weighted_sum += sampled[feature].mean() * count
                            total_weight += count
                    
                    if total_weight > 0:
                        cluster_feature_values[feature] = weighted_sum / total_weight
                
                all_cluster_means[cluster] = cluster_feature_values
            
            # Now calculate distinguishing features per cluster
            # A feature distinguishes a cluster if it has high variance across clusters
            # (i.e., the cluster's value is very different from other clusters)
            cluster_feature_means = []
            
            for cluster in clusters:
                cluster_feature_values = all_cluster_means.get(cluster, {})
                if not cluster_feature_values:
                    continue
                
                # Calculate distinguishing score for each feature
                # Score = how different this cluster's value is from the mean across all clusters
                distinguishing_scores = {}
                for feature in cluster_feature_values.keys():
                    # Get this cluster's value
                    cluster_value = cluster_feature_values[feature]
                    
                    # Get all other clusters' values for this feature
                    other_values = []
                    for other_cluster in clusters:
                        if other_cluster != cluster and feature in all_cluster_means.get(other_cluster, {}):
                            other_values.append(all_cluster_means[other_cluster][feature])
                    
                    if len(other_values) > 0:
                        # Calculate how different this cluster is from others
                        # Use coefficient of variation or absolute difference from mean
                        mean_other = np.mean(other_values)
                        std_other = np.std(other_values) if len(other_values) > 1 else 0
                        
                        if std_other > 0:
                            # Z-score: how many standard deviations away from mean
                            z_score = abs((cluster_value - mean_other) / std_other)
                        else:
                            # If no variance, use absolute difference
                            z_score = abs(cluster_value - mean_other) if mean_other != 0 else abs(cluster_value)
                        
                        distinguishing_scores[feature] = z_score
                    else:
                        # Only one cluster, can't distinguish
                        distinguishing_scores[feature] = 0
                
                # Get top 5 most distinguishing features
                if distinguishing_scores:
                    sorted_features = sorted(distinguishing_scores.items(), key=lambda x: x[1], reverse=True)
                    top5 = sorted_features[:5]
                    cluster_feature_means.append({
                        "Cluster": cluster,
                        "Top1_Feature": top5[0][0] if len(top5) > 0 else "",
                        "Top1_Score": f"{top5[0][1]:.4f}" if len(top5) > 0 else "",
                        "Top2_Feature": top5[1][0] if len(top5) > 1 else "",
                        "Top2_Score": f"{top5[1][1]:.4f}" if len(top5) > 1 else "",
                        "Top3_Feature": top5[2][0] if len(top5) > 2 else "",
                        "Top3_Score": f"{top5[2][1]:.4f}" if len(top5) > 2 else "",
                        "Top4_Feature": top5[3][0] if len(top5) > 3 else "",
                        "Top4_Score": f"{top5[3][1]:.4f}" if len(top5) > 3 else "",
                        "Top5_Feature": top5[4][0] if len(top5) > 4 else "",
                        "Top5_Score": f"{top5[4][1]:.4f}" if len(top5) > 4 else "",
                    })
            
            if not cluster_feature_means:
                QMessageBox.warning(self, "Warning", "Could not calculate feature means per cluster.")
                return
            
            # Save to CSV
            top5_df = pd.DataFrame(cluster_feature_means)
            paths["results"].mkdir(parents=True, exist_ok=True)
            output_path = paths["results"] / "top5_features_per_cluster.csv"
            top5_df.to_csv(output_path, index=False)
            
            QMessageBox.information(
                self, 
                "Success", 
                f"✅ Top 5 Features per Cluster saved to:\n{output_path}\n\n"
                f"Found {len(cluster_feature_means)} clusters with feature data."
            )
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error exporting top3 features: {e}\n{error_trace}")
            QMessageBox.critical(self, "Error", f"❌ Error: {str(e)[:200]}")
    
    def export_cluster_cells(self):
        """Export cell identifiers per cluster (max 100 cells per cluster)."""
        if self.session_state.get("cluster_labels") is None or self.session_state.get("embedding_df") is None:
            QMessageBox.warning(self, "Warning", "No cluster data available. Run clustering first.")
            return
        
        try:
            embedding_df = self.session_state["embedding_df"].copy()
            cluster_labels = self.session_state["cluster_labels"]
            embedding_df["cluster"] = cluster_labels
            embedding_df["cluster"] = embedding_df["cluster"].astype(str)
            embedding_df.loc[embedding_df["cluster"] == "-1", "cluster"] = "Noise"
            
            # Ensure cell_index exists
            if "cell_index" not in embedding_df.columns:
                embedding_df = embedding_df.reset_index(drop=True)
                embedding_df["cell_index"] = embedding_df.groupby("sample_id").cumcount()
            
            # Get paths
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            paths["results"].mkdir(parents=True, exist_ok=True)
            
            # Collect cells per cluster (max 100 per cluster)
            max_cells_per_cluster = 100
            cluster_cells_list = []
            
            # Get unique clusters (excluding noise if desired, but we'll include it)
            clusters = sorted([c for c in embedding_df["cluster"].unique() if pd.notna(c)])
            
            for cluster in clusters:
                cluster_data = embedding_df[embedding_df["cluster"] == cluster].copy()
                
                # Sample up to max_cells_per_cluster cells
                if len(cluster_data) > max_cells_per_cluster:
                    cluster_data = cluster_data.sample(n=max_cells_per_cluster, random_state=42)
                
                # Add to list
                for idx, row in cluster_data.iterrows():
                    cluster_cells_list.append({
                        "Cluster": cluster,
                        "Sample_ID": str(row["sample_id"]),
                        "Cell_Index": int(row["cell_index"])
                    })
            
            # Create DataFrame and save
            cluster_cells_df = pd.DataFrame(cluster_cells_list)
            output_path = paths["results"] / "cluster_cells.csv"
            cluster_cells_df.to_csv(output_path, index=False)
            
            # Count cells per cluster for message
            cells_per_cluster = cluster_cells_df.groupby("Cluster").size()
            cluster_summary = "\n".join([f"  {cluster}: {count} cells" for cluster, count in cells_per_cluster.items()])
            
            QMessageBox.information(
                self,
                "Success",
                f"✅ Cluster cells exported to:\n{output_path}\n\n"
                f"Total cells exported: {len(cluster_cells_df)}\n"
                f"Cells per cluster:\n{cluster_summary}\n\n"
                f"Note: Maximum {max_cells_per_cluster} cells per cluster (or all if fewer)."
            )
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error exporting cluster cells: {e}\n{error_trace}")
            QMessageBox.critical(self, "Error", f"❌ Error: {str(e)[:200]}")
    
    def export_cluster_heatmap(self):
        """Export cluster-feature heatmap with row-wise Z-score normalization.
        
        Creates a clean DataFrame structure:
        1. Create DataFrame with features as rows, clusters as columns
        2. Calculate row-wise Z-score
        3. Create new DataFrame with Z-scores
        4. Plot heatmap
        """
        if self.session_state.get("cluster_labels") is None or self.session_state.get("embedding_df") is None:
            QMessageBox.warning(self, "Warning", "No cluster data available. Run clustering first.")
            return
        
        try:
            # Get features and cluster data
            features = self.session_state.get("features", [])
            if not features:
                QMessageBox.warning(self, "Warning", "No features available. Run analysis first.")
                return
            
            embedding_df = self.session_state["embedding_df"].copy()
            cluster_labels = self.session_state["cluster_labels"]
            embedding_df["cluster"] = cluster_labels
            embedding_df["cluster"] = embedding_df["cluster"].astype(str)
            embedding_df.loc[embedding_df["cluster"] == "-1", "cluster"] = "Noise"
            
            # Get paths
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            
            # Load original feature data from CSV cache
            cache_dir = paths["csv_cache"]
            sample_ids = set(embedding_df["sample_id"].astype(str).unique())
            
            # Collect feature data for all samples
            all_feature_data = []
            for fcs_file in sorted(paths["fcs"].glob("*.fcs")):
                fcs_stem = str(fcs_file.stem)
                if fcs_stem not in sample_ids:
                    continue
                
                csv_path = cache_dir / f"{fcs_stem}.csv"
                if not csv_path.exists():
                    continue
                
                df = pd.read_csv(csv_path)
                
                # Filter by population if selected
                population = self.session_state.get("selected_population")
                if population and population in df.columns:
                    pop_series = df[population]
                    if isinstance(pop_series, pd.DataFrame):
                        pop_series = pop_series.iloc[:, 0]
                    pop_array = np.asarray(pop_series)
                    if pop_series.dtype == bool or str(pop_series.dtype) == 'bool':
                        mask = pop_array == True
                        df = df[mask]
                    else:
                        mask = pop_array == 1
                        df = df[mask]
                
                # Select only available features
                available_features = [f for f in features if f in df.columns]
                if not available_features:
                    continue
                
                subset = df[available_features].copy()
                subset["sample_id"] = fcs_stem
                all_feature_data.append(subset)
            
            if not all_feature_data:
                QMessageBox.warning(self, "Warning", "Could not load feature data from CSV cache.")
                return
            
            # Combine all data
            feature_data = pd.concat(all_feature_data, ignore_index=True)
            feature_data["sample_id"] = feature_data["sample_id"].astype(str)
            
            # Merge embedding_df with feature_data
            embedding_df["sample_id"] = embedding_df["sample_id"].astype(str)
            
            # Create cell_index for alignment
            if "cell_index" not in embedding_df.columns:
                embedding_df = embedding_df.reset_index(drop=True)
                embedding_df["cell_index"] = embedding_df.groupby("sample_id").cumcount()
            
            if "cell_index" not in feature_data.columns:
                feature_data = feature_data.reset_index(drop=True)
                feature_data["cell_index"] = feature_data.groupby("sample_id").cumcount()
            
            # Merge on sample_id and cell_index
            merged_df = embedding_df.merge(
                feature_data,
                on=["sample_id", "cell_index"],
                how="inner",
                suffixes=("_emb", "_feat")
            )
            
            # Get clusters
            cluster_series = merged_df["cluster"]
            if isinstance(cluster_series, pd.DataFrame):
                cluster_series = cluster_series.iloc[:, 0]
            clusters = sorted([c for c in cluster_series.unique() if pd.notna(c)])
            
            if not clusters:
                QMessageBox.warning(self, "Warning", "No valid clusters found in data.")
                return
            
            # STEP 1: Create DataFrame with mean feature values per cluster
            # Structure: Features as rows (index), Clusters as columns
            heatmap_data_dict = {}
            
            for feature in features:
                if feature not in merged_df.columns:
                    continue
                
                feature_values_per_cluster = {}
                for cluster in clusters:
                    cluster_mask = merged_df["cluster"] == cluster
                    cluster_df = merged_df[cluster_mask]
                    
                    if len(cluster_df) == 0:
                        feature_values_per_cluster[cluster] = np.nan
                        continue
                    
                    feature_values = cluster_df[feature].dropna()
                    if len(feature_values) > 0:
                        feature_values_per_cluster[cluster] = float(feature_values.mean())
                    else:
                        feature_values_per_cluster[cluster] = np.nan
                
                heatmap_data_dict[feature] = feature_values_per_cluster
            
            # Create DataFrame: Features as rows, Clusters as columns
            heatmap_df = pd.DataFrame(heatmap_data_dict).T
            # Now: heatmap_df.index = Features, heatmap_df.columns = Clusters
            
            # Remove features with all NaN
            heatmap_df = heatmap_df.dropna(how='all')
            
            if heatmap_df.empty:
                QMessageBox.warning(self, "Warning", "No valid feature data for heatmap.")
                return
            
            # Ensure all values are numeric
            heatmap_df = heatmap_df.apply(pd.to_numeric, errors='coerce')
            
            # STEP 2: Calculate row-wise Z-score
            from scipy.stats import zscore
            
            # Apply z-score row-wise (for each feature across clusters)
            # Create new DataFrame to store Z-scores
            heatmap_z_data = {}
            
            for feature in heatmap_df.index:
                row_values = heatmap_df.loc[feature].values
                # Remove NaN values for zscore calculation
                valid_mask = ~np.isnan(row_values)
                if valid_mask.sum() > 1:  # Need at least 2 values for zscore
                    valid_values = row_values[valid_mask]
                    z_scores = zscore(valid_values, nan_policy='omit')
                    # Create full array with NaN where original was NaN
                    z_full = np.full(len(row_values), np.nan)
                    z_full[valid_mask] = z_scores
                    heatmap_z_data[feature] = z_full
                else:
                    # If not enough values, set all to 0
                    heatmap_z_data[feature] = np.zeros(len(row_values))
            
            # Create DataFrame from dictionary
            # heatmap_z_data: {feature: [z_score_for_cluster1, z_score_for_cluster2, ...]}
            # We want: Features as rows (index), Clusters as columns
            # Convert dict to DataFrame: keys (features) become index, arrays become rows
            # But arrays need to be aligned with cluster columns
            heatmap_z = pd.DataFrame(heatmap_z_data, index=heatmap_df.columns).T
            # After transpose: heatmap_z.index = Features, heatmap_z.columns = Clusters
            # Ensure correct order
            heatmap_z = heatmap_z.reindex(index=heatmap_df.index, columns=heatmap_df.columns)
            
            # STEP 3: Create new DataFrame with Z-scores
            # Replace NaN with 0 (for features with no variance)
            heatmap_z = heatmap_z.fillna(0)
            
            # Ensure all values are float and scalar
            for col in heatmap_z.columns:
                for idx in heatmap_z.index:
                    val = heatmap_z.loc[idx, col]
                    if isinstance(val, (list, tuple, np.ndarray)):
                        # If somehow a sequence got in, take first element or mean
                            if len(val) > 0:
                                heatmap_z.loc[idx, col] = float(np.mean(val))
                            else:
                                heatmap_z.loc[idx, col] = 0.0
                    elif not isinstance(val, (int, float, np.number)) or (isinstance(val, float) and np.isnan(val)):
                            heatmap_z.loc[idx, col] = 0.0
            
            # Ensure all values are float
            heatmap_z = heatmap_z.astype(float)
            
            # Final structure: heatmap_z.index = Features, heatmap_z.columns = Clusters
            n_features = len(heatmap_z.index)
            n_clusters = len(heatmap_z.columns)
            
            if n_features == 0 or n_clusters == 0:
                QMessageBox.warning(self, "Warning", f"Invalid heatmap dimensions: {n_features} features, {n_clusters} clusters")
                return
            
            # STEP 4: Create heatmap using R (pheatmap) for reliable results
            # Save data to CSV for R script
            paths["results"].mkdir(parents=True, exist_ok=True)
            temp_csv = paths["results"] / "heatmap_data_temp.csv"
            heatmap_z.to_csv(temp_csv)
            
            output_path = paths["results"] / "cluster_feature_heatmap.png"
            
            # Check if R script exists
            if not R_HEATMAP_SCRIPT.exists():
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"R heatmap script not found: {R_HEATMAP_SCRIPT}\n\nFalling back to Python."
                )
                self._create_heatmap_python_fallback(heatmap_z, n_features, n_clusters, paths)
                return
            
            # Call R script to create heatmap
            command = [
                "Rscript", "--vanilla", "--slave",
                str(R_HEATMAP_SCRIPT),
                str(temp_csv),
                str(output_path),
                str(n_features),
                str(n_clusters)
            ]
            
            print(f"DEBUG: Calling R script: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                QMessageBox.information(
                    self,
                    "Success",
                    f"✅ Cluster-Feature Heatmap saved to:\n{output_path}\n\n"
                    f"Created using R (pheatmap) with dendrograms."
                )
                # Clean up temp file
                try:
                    temp_csv.unlink()
                except:
                    pass
            else:
                error_msg = result.stderr.strip() if result.stderr.strip() else result.stdout.strip()
                if not error_msg:
                    error_msg = f"R script failed with return code {result.returncode}"
                print(f"DEBUG: R script error: {error_msg}")
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"R heatmap script failed. Falling back to Python.\n\n{error_msg[:200]}"
                )
                # Fallback to Python (original code)
                self._create_heatmap_python_fallback(heatmap_z, n_features, n_clusters, paths)
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error exporting cluster heatmap: {e}\n{error_trace}")
            QMessageBox.critical(self, "Error", f"❌ Error: {str(e)[:200]}")
    
    def _create_heatmap_python_fallback(self, heatmap_z, n_features, n_clusters, paths):
        """Fallback Python heatmap creation if R script fails."""
        try:
            # Increase figure size to accommodate dendrograms
            fig_width = max(14, n_clusters * 1.5)
            fig_height = max(12, n_features * 0.3)
            
            # Calculate font size for Y-axis
            if n_features <= 20:
                y_fontsize = 28
            elif n_features <= 50:
                y_fontsize = 24
            elif n_features <= 100:
                y_fontsize = 20
            else:
                y_fontsize = 16
            
            try:
                import seaborn as sns
                
                # Try clustermap first (with dendrograms)
                # Check if scipy is available for clustering
                try:
                    from scipy.cluster.hierarchy import linkage
                    from scipy.spatial.distance import pdist
                    _ = linkage(heatmap_z.iloc[:2, :2].values, method='ward')  # Quick test
                    use_clustermap = True
                except Exception as scipy_test_error:
                    print(f"DEBUG: scipy clustering test failed ({scipy_test_error}), will use simple heatmap")
                    use_clustermap = False
                
                if use_clustermap:
                    try:
                        # Create clustermap with dendrograms
                        g = sns.clustermap(
                            heatmap_z,
                            cmap="RdYlBu_r",
                            center=0,
                            vmin=-3,
                            vmax=3,
                            row_cluster=True,  # Cluster features (Y-axis) - creates dendrogram
                            col_cluster=True,  # Cluster clusters (X-axis) - creates dendrogram
                            method='ward',
                            metric='euclidean',
                            figsize=(fig_width, fig_height),
                            cbar_kws={"label": "Z-score", "shrink": 0.6},
                            xticklabels=True,
                            yticklabels=True if n_features <= 100 else False,
                            linewidths=0.1,
                            linecolor='gray',
                            cbar_pos=(0.92, 0.05, 0.02, 0.08)  # Bottom right, smaller height
                        )
                    except Exception as clustermap_error:
                        print(f"DEBUG: clustermap creation failed ({clustermap_error})")
                        use_clustermap = False
                
                if use_clustermap:
                    
                    # Remove general Y-axis label, only show individual feature names
                    g.ax_heatmap.set_xlabel("Cluster", fontsize=24, fontweight="bold")
                    g.ax_heatmap.set_ylabel("")  # Remove "Feature" label
                    # X-axis labels: no rotation, horizontal
                    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=0, ha='center', fontsize=18, fontweight='bold')
                    
                    if n_features <= 100:
                        # Set y-axis labels with much larger, bold font
                        yticklabels = g.ax_heatmap.get_yticklabels()
                        for label in yticklabels:
                            label.set_fontsize(y_fontsize)
                            label.set_fontweight('bold')
                        g.ax_heatmap.set_yticklabels(yticklabels, rotation=0)
                    
                    # Colorbar (legend) - bottom right, smaller
                    if g.cax is not None:
                        # Position colorbar bottom right, smaller
                        g.cax.set_position([0.92, 0.05, 0.02, 0.08])  # x, y, width, height (bottom right, smaller)
                        g.cax.set_label("Z-score", fontsize=10, fontweight='bold')
                        # Make colorbar label smaller
                        g.cax.yaxis.label.set_fontsize(10)
                        g.cax.yaxis.label.set_fontweight('bold')
                        # Make tick labels smaller
                        g.cax.tick_params(labelsize=8)
                    
                    # Ensure dendrograms are visible and properly sized
                    # Row dendrogram (left side)
                    if hasattr(g, 'ax_row_dendrogram') and g.ax_row_dendrogram is not None:
                        g.ax_row_dendrogram.set_visible(True)
                        # Ensure it has proper width
                        g.ax_row_dendrogram.set_position(g.ax_row_dendrogram.get_position())
                    # Column dendrogram (top side)
                    if hasattr(g, 'ax_col_dendrogram') and g.ax_col_dendrogram is not None:
                        g.ax_col_dendrogram.set_visible(True)
                        # Ensure it has proper height
                        g.ax_col_dendrogram.set_position(g.ax_col_dendrogram.get_position())
                    
                    # Adjust layout to make room for dendrograms
                    g.fig.subplots_adjust(right=0.85)  # Leave space for colorbar
                    
                    g.fig.suptitle(
                        f"Cluster-Feature Heatmap (Row-wise Z-score)\n{n_features} Features × {n_clusters} Clusters",
                        fontsize=14,
                        fontweight="bold",
                        y=0.98
                    )
                    
                    # DO NOT use tight_layout - it overwrites our manual positions
                    # Set positions explicitly before saving
                    # Ensure dendrograms are visible
                    if hasattr(g, 'ax_row_dendrogram') and g.ax_row_dendrogram is not None:
                        g.ax_row_dendrogram.set_visible(True)
                    if hasattr(g, 'ax_col_dendrogram') and g.ax_col_dendrogram is not None:
                        g.ax_col_dendrogram.set_visible(True)
                    
                    # Adjust layout manually - leave space for dendrograms and colorbar
                    g.fig.subplots_adjust(left=0.15, right=0.90, top=0.90, bottom=0.10)
                    
                    paths["results"].mkdir(parents=True, exist_ok=True)
                    output_path = paths["results"] / "cluster_feature_heatmap.png"
                    # Save without bbox_inches="tight" to preserve manual positions
                    g.fig.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0.1)
                    plt.close(g.fig)
                    
                else:
                    # Use simple heatmap (no dendrograms)
                    print(f"DEBUG: clustermap failed ({clustermap_error}), falling back to heatmap")
                    import traceback
                    print(f"DEBUG: clustermap error traceback:\n{traceback.format_exc()}")
                    # Fallback to regular heatmap (no dendrograms, but same styling)
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    
                    # Create heatmap with manual colorbar positioning
                    im = ax.imshow(heatmap_z.values, aspect='auto', cmap='RdYlBu_r', vmin=-3, vmax=3, interpolation='nearest')
                    
                    # Set ticks and labels
                    ax.set_xticks(range(len(heatmap_z.columns)))
                    ax.set_xticklabels(heatmap_z.columns, rotation=0, ha='center', fontsize=18, fontweight='bold')
                    ax.set_xlabel("Cluster", fontsize=24, fontweight="bold")
                    
                    if n_features <= 100:
                        ax.set_yticks(range(len(heatmap_z.index)))
                        ax.set_yticklabels(heatmap_z.index, rotation=0, fontsize=y_fontsize, fontweight='bold')
                    else:
                        step = max(1, n_features // 100)
                        yticks = range(0, n_features, step)
                        ax.set_yticks(yticks)
                        ax.set_yticklabels([heatmap_z.index[i] for i in yticks], rotation=0, fontsize=12)
                    ax.set_ylabel("")  # Remove "Feature" label
                    
                    # Add colorbar - small, bottom right
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    divider = make_axes_locatable(ax)
                    cbar_ax = divider.append_axes("right", size="2%", pad=0.05)
                    cbar = plt.colorbar(im, cax=cbar_ax)
                    cbar.set_label("Z-score", fontsize=10, fontweight='bold')
                    cbar.ax.tick_params(labelsize=8)
                    # Position colorbar bottom right
                    cbar_ax.set_position([0.92, 0.05, 0.02, 0.08])
                    
                    ax.set_xlabel("Cluster", fontsize=24, fontweight="bold")
                    ax.set_ylabel("")  # Remove "Feature" label, only show individual feature names
                    ax.set_title(
                        f"Cluster-Feature Heatmap (Row-wise Z-score)\n{n_features} Features × {n_clusters} Clusters",
                        fontsize=14,
                        fontweight="bold"
                    )
                    
                    # Adjust layout manually (no tight_layout)
                    plt.subplots_adjust(left=0.15, right=0.90, top=0.90, bottom=0.10)
                    
                    paths["results"].mkdir(parents=True, exist_ok=True)
                    output_path = paths["results"] / "cluster_feature_heatmap.png"
                    plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0.1)
                    plt.close(fig)
                    
            except ImportError:
                # Fallback to matplotlib
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                
                im = ax.imshow(
                    heatmap_z.values,
                    cmap="RdYlBu_r",
                    aspect='auto',
                    vmin=-3,
                    vmax=3,
                    origin='upper',
                    interpolation='nearest'
                )
                
                ax.set_xticks(range(n_clusters))
                # X-axis labels: no rotation, horizontal
                ax.set_xticklabels(
                    [str(int(c)) if isinstance(c, (int, float)) else str(c) for c in heatmap_z.columns],
                    rotation=0,
                    ha='center',
                    fontsize=18,
                    fontweight='bold'
                )
                
                if n_features <= 100:
                    ax.set_yticks(range(n_features))
                    # Set y-axis labels with much larger, bold font
                    yticklabels = ax.get_yticklabels()
                    for label in yticklabels:
                        label.set_fontsize(y_fontsize)
                        label.set_fontweight('bold')
                    ax.set_yticklabels(heatmap_z.index, rotation=0)
                else:
                    step = max(1, n_features // 100)
                    yticks = range(0, n_features, step)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels([heatmap_z.index[i] for i in yticks], rotation=0, fontsize=6)

                # Colorbar (legend) - smaller, bottom right, larger label
                cbar = plt.colorbar(im, ax=ax, label="Z-score (row-wise)")
                cbar.ax.set_position([0.92, 0.05, 0.02, 0.15])  # x, y, width, height (bottom right)
                cbar.set_label("Z-score (row-wise)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Cluster", fontsize=24, fontweight="bold")
                ax.set_ylabel("")  # Remove "Feature" label, only show individual feature names
                ax.set_title(
                    f"Cluster-Feature Heatmap (Row-wise Z-score)\n{n_features} Features × {n_clusters} Clusters",
                    fontsize=14,
                    fontweight="bold"
                )
            
            plt.tight_layout()
            
            paths["results"].mkdir(parents=True, exist_ok=True)
            output_path = paths["results"] / "cluster_feature_heatmap.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            # Save the Z-score data
            data_path = paths["results"] / "cluster_feature_heatmap_data.csv"
            heatmap_z.to_csv(data_path)
            
            QMessageBox.information(
                self,
                "Success",
                f"✅ Cluster-Feature Heatmap saved to:\n{output_path}\n\n"
                f"Data saved to:\n{data_path}"
            )
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error exporting cluster heatmap: {e}\n{error_trace}")
            QMessageBox.critical(self, "Error", f"❌ Error: {str(e)[:200]}")
    
    def export_group_cluster_bar_graphs(self):
        """Export bar graphs showing % of cells from each group in each cluster, with SEM and statistics.
        
        Creates one graph per cluster, with groups on X-axis and % cells from group in cluster on Y-axis.
        Includes SEM error bars and statistical significance markers (stars).
        """
        if self.session_state.get("cluster_labels") is None or self.session_state.get("embedding_df") is None:
            QMessageBox.warning(self, "Warning", "No cluster data available. Run clustering first.")
            return
        
        # Check for group column
        embedding_df = self.session_state["embedding_df"].copy()
        embedding_df["cluster"] = self.session_state["cluster_labels"]
        embedding_df["cluster"] = embedding_df["cluster"].astype(str)
        embedding_df.loc[embedding_df["cluster"] == "-1", "cluster"] = "Noise"
        
        # Determine which metadata column to use for grouping
        group_column = None
        metadata_df = self.session_state.get("metadata_df")
        if metadata_df is not None:
            # Check for common group column names
            for col in ["group", "Group", "condition", "Condition", "treatment", "Treatment"]:
                if col in metadata_df.columns:
                    group_column = col
                    break
        
        if group_column is None or group_column not in embedding_df.columns:
            QMessageBox.warning(
                self,
                "Warning",
                "No group column found in metadata.\n\n"
                "Please add a 'group' column to your metadata with group labels (e.g., 'Control', 'Treatment')."
            )
            return
        
        # Filter out NaN and empty groups
        embedding_df = embedding_df[embedding_df[group_column].notna()]
        embedding_df = embedding_df[embedding_df[group_column].astype(str).str.strip() != ""]
        
        if len(embedding_df) == 0:
            QMessageBox.warning(self, "Warning", "No valid groups found in data.")
            return
        
        try:
            from scipy import stats
            from scipy.stats import ttest_ind
            
            # Get paths
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            paths["results"].mkdir(parents=True, exist_ok=True)
            
            clusters = sorted([c for c in embedding_df["cluster"].unique() if pd.notna(c)])
            groups = sorted([g for g in embedding_df[group_column].unique() if pd.notna(g)])
            
            if len(groups) < 2:
                QMessageBox.warning(self, "Warning", "Need at least 2 groups for statistical comparison.")
                return
            
            # Calculate statistics per cluster
            # For each cluster, calculate % of cells from each group that are in this cluster
            # We need to calculate this per sample (replicate) to get SEM
            
            # Get sample_id column
            if "sample_id" not in embedding_df.columns:
                QMessageBox.warning(self, "Warning", "No sample_id column found. Cannot calculate SEM.")
                return
            
            # Calculate per sample: for each sample, what % of its cells are in each cluster
            results_per_cluster = {}
            
            for cluster in clusters:
                cluster_data = []
                
                # For each group
                for group in groups:
                    group_df = embedding_df[embedding_df[group_column] == group]
                    
                    # Get unique samples in this group
                    samples_in_group = group_df["sample_id"].unique()
                    
                    percentages_per_sample = []
                    for sample_id in samples_in_group:
                        sample_df = group_df[group_df["sample_id"] == sample_id]
                        total_cells_in_sample = len(sample_df)
                        
                        if total_cells_in_sample == 0:
                            continue
                        
                        cells_in_cluster = len(sample_df[sample_df["cluster"] == cluster])
                        percentage = (cells_in_cluster / total_cells_in_sample) * 100
                        percentages_per_sample.append(percentage)
                    
                    if len(percentages_per_sample) > 0:
                        mean_pct = np.mean(percentages_per_sample)
                        sem = stats.sem(percentages_per_sample) if len(percentages_per_sample) > 1 else 0
                        cluster_data.append({
                            "group": group,
                            "mean": mean_pct,
                            "sem": sem,
                            "n": len(percentages_per_sample),
                            "values": percentages_per_sample
                        })
                
                results_per_cluster[cluster] = cluster_data
            
            # Create one plot per cluster
            for cluster in clusters:
                cluster_data = results_per_cluster[cluster]
                if not cluster_data:
                    continue
                
                # Prepare data for plotting
                groups_plot = [d["group"] for d in cluster_data]
                means = [d["mean"] for d in cluster_data]
                sems = [d["sem"] for d in cluster_data]
                values_list = [d["values"] for d in cluster_data]
                
                # Statistical testing: pairwise t-tests between groups
                # Only store significant comparisons (p < 0.05)
                significance_markers = []
                for i in range(len(groups_plot)):
                    for j in range(i + 1, len(groups_plot)):
                        # Perform t-test
                        group1_values = values_list[i]
                        group2_values = values_list[j]
                        
                        if len(group1_values) > 1 and len(group2_values) > 1:
                            try:
                                t_stat, p_value = ttest_ind(group1_values, group2_values)
                                
                                # Only add if significant (p < 0.05)
                                if p_value < 0.05:
                                    # Determine significance marker
                                    if p_value < 0.001:
                                        marker = "***"
                                    elif p_value < 0.01:
                                        marker = "**"
                                    else:
                                        marker = "*"
                                    
                                    # Store for annotation
                                    significance_markers.append({
                                        "group1_idx": i,
                                        "group2_idx": j,
                                        "p_value": p_value,
                                        "marker": marker
                                    })
                            except:
                                pass
                
                # Create figure with appropriate size (matching other plots)
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create bar plot
                x_pos = np.arange(len(groups_plot))
                bars = ax.bar(x_pos, means, yerr=sems, capsize=5, 
                             color=COLOR_PALETTE[:len(groups_plot)], 
                             alpha=0.7, edgecolor='black', linewidth=1.5)
                
                # Add individual data points
                for i, (group_idx, values) in enumerate(zip(range(len(groups_plot)), values_list)):
                    x_scatter = np.random.normal(i, 0.1, size=len(values))
                    ax.scatter(x_scatter, values, color='black', alpha=0.5, s=30, zorder=10)
                
                # Add significance markers (only for significant comparisons)
                y_max = max(means) + max(sems) if sems else max(means)
                y_range = y_max * 0.1  # 10% of max for spacing
                
                if significance_markers:
                    for sig_idx, sig in enumerate(significance_markers):
                        i1, i2 = sig["group1_idx"], sig["group2_idx"]
                        y_line = y_max + y_range * (1 + sig_idx)
                        
                        # Draw line
                        ax.plot([i1, i2], [y_line, y_line], 'k-', linewidth=1.5)
                        # Add marker
                        ax.text((i1 + i2) / 2, y_line + y_range * 0.1, sig["marker"], 
                               ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
                    # Adjust y-axis limit to accommodate significance markers
                    ax.set_ylim(bottom=0, top=y_max + y_range * (2 + len(significance_markers)))
                else:
                    # No significant differences, just set normal y-axis limit
                    ax.set_ylim(bottom=0, top=y_max * 1.15)
                
                # Formatting
                ax.set_xlabel("Group", fontsize=14, fontweight="bold")
                ax.set_ylabel("% Cells from Group in Cluster", fontsize=14, fontweight="bold")
                ax.set_title(f"Cluster {cluster}: Group Distribution", fontsize=16, fontweight="bold")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(groups_plot, fontsize=12)
                ax.tick_params(axis='y', labelsize=12)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                
                # Save plot
                output_path = paths["results"] / f"cluster_{cluster}_group_bar_graph.png"
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            
            QMessageBox.information(
                self,
                "Success",
                f"✅ Bar graphs exported for {len(clusters)} cluster(s):\n"
                f"{paths['results']}\n\n"
                f"Files: cluster_*_group_bar_graph.png"
            )
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"❌ Error creating bar graphs: {str(e)}\n\n{traceback.format_exc()}")
    
    def export_pca_plot(self):
        """Export PCA plot on sample-level, colored by groups to visualize group differences."""
        if self.session_state.get("embedding_df") is None:
            QMessageBox.warning(self, "Warning", "No embedding data available. Run analysis first.")
            return
        
        # Check for group column in metadata
        metadata_df = self.session_state.get("metadata_df")
        if metadata_df is None or len(metadata_df) == 0:
            QMessageBox.warning(self, "Warning", "No metadata available. Please add metadata with group information.")
            return
        
        # Determine which metadata column to use for grouping
        group_column = None
        for col in ["group", "Group", "condition", "Condition", "treatment", "Treatment"]:
            if col in metadata_df.columns:
                group_column = col
                break
        
        if group_column is None:
            QMessageBox.warning(
                self,
                "Warning",
                "No group column found in metadata.\n\n"
                "Please add a 'group' column to your metadata with group labels."
            )
            return
        
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Get features
            features = self.session_state.get("features", [])
            if not features:
                QMessageBox.warning(self, "Warning", "No features available. Run analysis first.")
                return
            
            # Get paths
            project_dir = safe_str(self.session_state.get("project_dir", PROJECT_ROOT))
            run_id = safe_str(self.session_state["run_id"])
            paths = get_run_paths(safe_path(project_dir), run_id)
            
            # Load feature data from CSV cache - calculate mean per sample
            cache_dir = paths["csv_cache"]
            
            # Collect mean feature values per sample
            sample_data = []
            sample_groups = {}
            
            for fcs_file in sorted(paths["fcs"].glob("*.fcs")):
                fcs_stem = str(fcs_file.stem)
                
                csv_path = cache_dir / f"{fcs_stem}.csv"
                if not csv_path.exists():
                    continue
                
                df = pd.read_csv(csv_path)
                
                # Filter by population if selected
                population = self.session_state.get("selected_population")
                if population and population in df.columns:
                    pop_series = df[population]
                    if isinstance(pop_series, pd.DataFrame):
                        pop_series = pop_series.iloc[:, 0]
                    pop_array = np.asarray(pop_series)
                    if pop_series.dtype == bool or str(pop_series.dtype) == 'bool':
                        mask = pop_array == True
                        df = df[mask]
                    else:
                        mask = pop_array == 1
                        df = df[mask]
                
                # Select only available features
                available_features = [f for f in features if f in df.columns]
                if not available_features:
                    continue
                
                # Calculate mean per feature for this sample
                sample_means = df[available_features].mean().to_dict()
                sample_means["sample_id"] = fcs_stem
                sample_data.append(sample_means)
                
                # Get group for this sample from metadata
                sample_meta = metadata_df[metadata_df["file_name"].astype(str).str.replace(r'\.(daf|fcs)$', '', regex=True) == fcs_stem]
                if len(sample_meta) > 0 and group_column in sample_meta.columns:
                    group_val = sample_meta[group_column].iloc[0]
                    if pd.notna(group_val) and str(group_val).strip() != "":
                        sample_groups[fcs_stem] = str(group_val).strip()
            
            if not sample_data:
                QMessageBox.warning(self, "Warning", "Could not load feature data from CSV cache.")
                return
            
            # Create DataFrame with sample means
            sample_df = pd.DataFrame(sample_data)
            sample_df["group"] = sample_df["sample_id"].map(sample_groups)
            
            # Filter out samples without groups
            sample_df = sample_df[sample_df["group"].notna()]
            sample_df = sample_df[sample_df["group"].astype(str).str.strip() != ""]
            
            if len(sample_df) == 0:
                QMessageBox.warning(self, "Warning", "No samples with valid groups found.")
                return
            
            # Prepare data for PCA (exclude sample_id and group columns)
            feature_cols = [f for f in features if f in sample_df.columns]
            if len(feature_cols) == 0:
                QMessageBox.warning(self, "Warning", "No valid features found in sample data.")
                return
            
            X = sample_df[feature_cols].values
            
            # Remove NaN and infinite values
            mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
            X = X[mask]
            sample_df = sample_df[mask]
            
            if len(X) == 0:
                QMessageBox.warning(self, "Warning", "No valid data after removing NaN/Inf values.")
                return
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            groups = sorted([g for g in sample_df["group"].unique() if pd.notna(g)])
            colors = COLOR_PALETTE[:len(groups)]
            
            for i, group in enumerate(groups):
                group_mask = sample_df["group"] == group
                group_pca = X_pca[group_mask]
                
                ax.scatter(group_pca[:, 0], group_pca[:, 1], 
                          label=str(group), alpha=0.7, s=100, color=colors[i], edgecolors='black', linewidth=1.5)
            
            # Formatting
            explained_var = pca.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)", fontsize=14, fontweight="bold")
            ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)", fontsize=14, fontweight="bold")
            ax.set_title("PCA Plot: Sample-Level Group Comparison", fontsize=16, fontweight="bold")
            ax.legend(title="Group", fontsize=12, loc="best")
            ax.grid(alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=12)
            
            plt.tight_layout()
            
            # Save plot
            paths["results"].mkdir(parents=True, exist_ok=True)
            output_path = paths["results"] / "pca_plot_groups_sample_level.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            QMessageBox.information(
                self,
                "Success",
                f"✅ PCA plot (sample-level) saved to:\n{output_path}\n\n"
                f"PC1 explains {explained_var[0]*100:.1f}% of variance\n"
                f"PC2 explains {explained_var[1]*100:.1f}% of variance\n\n"
                f"Number of samples: {len(sample_df)}"
            )
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"❌ Error creating PCA plot: {str(e)}\n\n{traceback.format_exc()}")
    
    def show_documentation(self):
        """Show documentation in a dialog window."""
        dialog = QDialog(self)
        dialog.setWindowTitle("MorphoMapping - Documentation")
        dialog.setMinimumSize(1000, 800)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Create tab widget for different documentation sections
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                background-color: #fafafa;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #c0c0c0;
            }
        """)
        
        # Common style for all text editors
        text_edit_style = """
            QTextEdit {
                background-color: #ffffff;
                color: #212121;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 13px;
                line-height: 1.6;
                padding: 15px;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
            }
        """
        
        # Helper function to convert Markdown to HTML with formatting
        def markdown_to_html(markdown_text: str) -> str:
            """Convert Markdown text to HTML with proper formatting."""
            import re
            
            lines = markdown_text.split('\n')
            result_lines = []
            in_list = False
            list_type = None
            
            for line in lines:
                line_stripped = line.strip()
                
                # Headers (process first to avoid conflicts)
                if re.match(r'^#### ', line):
                    text = re.sub(r'^#### ', '', line).strip()
                    result_lines.append(f'<h4 style="color: #1976D2; margin-top: 18px; margin-bottom: 8px; font-size: 15px;"><b>{text}</b></h4>')
                    if in_list:
                        result_lines.append(f'</{list_type}>')
                        in_list = False
                elif re.match(r'^### ', line):
                    text = re.sub(r'^### ', '', line).strip()
                    result_lines.append(f'<h3 style="color: #1976D2; margin-top: 20px; margin-bottom: 10px; font-size: 16px;"><b>{text}</b></h3>')
                    if in_list:
                        result_lines.append(f'</{list_type}>')
                        in_list = False
                elif re.match(r'^## ', line):
                    text = re.sub(r'^## ', '', line).strip()
                    result_lines.append(f'<h2 style="color: #1565C0; margin-top: 25px; margin-bottom: 12px; font-size: 18px;"><b>{text}</b></h2>')
                    if in_list:
                        result_lines.append(f'</{list_type}>')
                        in_list = False
                elif re.match(r'^# ', line):
                    text = re.sub(r'^# ', '', line).strip()
                    result_lines.append(f'<h1 style="color: #0D47A1; margin-top: 30px; margin-bottom: 15px; font-size: 22px;"><b>{text}</b></h1>')
                    if in_list:
                        result_lines.append(f'</{list_type}>')
                        in_list = False
                # Lists
                elif re.match(r'^[-*] ', line):
                    if not in_list:
                        result_lines.append('<ul style="margin-left: 20px; margin-top: 10px; margin-bottom: 10px;">')
                        in_list = True
                        list_type = 'ul'
                    item_text = re.sub(r'^[-*] ', '', line).strip()
                    # Process inline formatting in list items
                    item_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', item_text)
                    item_text = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', item_text)
                    item_text = re.sub(r'`([^`]+)`', r'<code style="background-color: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: monospace;">\1</code>', item_text)
                    result_lines.append(f'<li style="margin-bottom: 5px;">{item_text}</li>')
                elif re.match(r'^\d+\. ', line):
                    if not in_list or list_type != 'ol':
                        if in_list:
                            result_lines.append(f'</{list_type}>')
                        result_lines.append('<ol style="margin-left: 20px; margin-top: 10px; margin-bottom: 10px;">')
                        in_list = True
                        list_type = 'ol'
                    item_text = re.sub(r'^\d+\. ', '', line).strip()
                    # Process inline formatting in list items
                    item_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', item_text)
                    item_text = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', item_text)
                    item_text = re.sub(r'`([^`]+)`', r'<code style="background-color: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: monospace;">\1</code>', item_text)
                    result_lines.append(f'<li style="margin-bottom: 5px;">{item_text}</li>')
                # Empty line
                elif not line_stripped:
                    if in_list:
                        result_lines.append(f'</{list_type}>')
                        in_list = False
                    result_lines.append('<br>')
                # Regular paragraph
                else:
                    if in_list:
                        result_lines.append(f'</{list_type}>')
                        in_list = False
                    
                    # Check for tables (Markdown table format: | col1 | col2 |)
                    if re.match(r'^\|', line) and '|' in line[1:]:
                        # This is a table row
                        if not hasattr(markdown_to_html, '_in_table'):
                            markdown_to_html._in_table = True
                            markdown_to_html._table_rows = []
                            markdown_to_html._is_header = True
                        
                        cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
                        if markdown_to_html._is_header:
                            # Header row
                            markdown_to_html._table_headers = cells
                            markdown_to_html._is_header = False
                        elif re.match(r'^[\s\-:]+$', line):  # Separator row (|---|---|)
                            # Skip separator
                            pass
                        else:
                            # Data row
                            markdown_to_html._table_rows.append(cells)
                        continue  # Skip normal paragraph processing for table rows
                    else:
                        # Not a table row - close table if we were in one
                        if hasattr(markdown_to_html, '_in_table') and markdown_to_html._in_table:
                            # Generate table HTML
                            table_html = '<table style="border-collapse: collapse; width: 100%; margin: 15px 0; border: 1px solid #ddd;">'
                            if hasattr(markdown_to_html, '_table_headers') and markdown_to_html._table_headers:
                                table_html += '<thead><tr style="background-color: #4CAF50; color: white;">'
                                for header in markdown_to_html._table_headers:
                                    table_html += f'<th style="padding: 10px; border: 1px solid #ddd; text-align: left;"><b>{header}</b></th>'
                                table_html += '</tr></thead>'
                            table_html += '<tbody>'
                            for row in markdown_to_html._table_rows:
                                table_html += '<tr>'
                                for cell in row:
                                    table_html += f'<td style="padding: 8px; border: 1px solid #ddd;">{cell}</td>'
                                table_html += '</tr>'
                            table_html += '</tbody></table>'
                            result_lines.append(table_html)
                            delattr(markdown_to_html, '_in_table')
                            delattr(markdown_to_html, '_table_rows')
                            if hasattr(markdown_to_html, '_table_headers'):
                                delattr(markdown_to_html, '_table_headers')
                    
                    # Process inline formatting
                    para_text = line_stripped
                    # Code blocks first (to avoid conflicts)
                    para_text = re.sub(r'```(.*?)```', r'<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px; border-left: 3px solid #4CAF50; margin: 10px 0;"><code>\1</code></pre>', para_text, flags=re.DOTALL)
                    # Links [text](url)
                    para_text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2" style="color: #1976D2; text-decoration: none;"><b>\1</b></a>', para_text)
                    # Bold
                    para_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', para_text)
                    # Italic (avoid conflicts with bold)
                    para_text = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', para_text)
                    # Inline code
                    para_text = re.sub(r'`([^`]+)`', r'<code style="background-color: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: monospace;">\1</code>', para_text)
                    
                    result_lines.append(f'<p style="margin-bottom: 10px; line-height: 1.6;">{para_text}</p>')
            
            if in_list:
                result_lines.append(f'</{list_type}>')
            
            # Close table if still open
            if hasattr(markdown_to_html, '_in_table') and markdown_to_html._in_table:
                table_html = '<table style="border-collapse: collapse; width: 100%; margin: 15px 0; border: 1px solid #ddd;">'
                if hasattr(markdown_to_html, '_table_headers') and markdown_to_html._table_headers:
                    table_html += '<thead><tr style="background-color: #4CAF50; color: white;">'
                    for header in markdown_to_html._table_headers:
                        table_html += f'<th style="padding: 10px; border: 1px solid #ddd; text-align: left;"><b>{header}</b></th>'
                    table_html += '</tr></thead>'
                table_html += '<tbody>'
                for row in markdown_to_html._table_rows:
                    table_html += '<tr>'
                    for cell in row:
                        table_html += f'<td style="padding: 8px; border: 1px solid #ddd;">{cell}</td>'
                    table_html += '</tr>'
                table_html += '</tbody></table>'
                result_lines.append(table_html)
            
            html = '\n'.join(result_lines)
            
            # Wrap in HTML structure
            return f"""
            <html>
            <head>
                <style>
                    body {{
                        font-family: 'Segoe UI', 'Arial', sans-serif;
                        font-size: 13px;
                        line-height: 1.6;
                        color: #212121;
                        padding: 15px;
                    }}
                </style>
            </head>
            <body>
                {html}
            </body>
            </html>
            """
        
        # Read documentation files
        doc_dir = BUNDLE_ROOT
        
        # Installation Guide
        install_text = QTextEdit()
        install_text.setReadOnly(True)
        install_text.setStyleSheet(text_edit_style)
        install_path = doc_dir / "INSTALLATION.md"
        if install_path.exists():
            markdown_content = install_path.read_text(encoding='utf-8')
            html_content = markdown_to_html(markdown_content)
            install_text.setHtml(html_content)
        else:
            install_text.setPlainText("Installation guide not found.")
        tabs.addTab(install_text, "📦 Installation")
        
        # User Guide
        user_guide_text = QTextEdit()
        user_guide_text.setReadOnly(True)
        user_guide_text.setStyleSheet(text_edit_style)
        user_guide_path = doc_dir / "USER_GUIDE.md"
        if user_guide_path.exists():
            markdown_content = user_guide_path.read_text(encoding='utf-8')
            html_content = markdown_to_html(markdown_content)
            user_guide_text.setHtml(html_content)
        else:
            user_guide_text.setPlainText("User guide not found.")
        tabs.addTab(user_guide_text, "📖 User Guide")
        
        # README
        readme_text = QTextEdit()
        readme_text.setReadOnly(True)
        readme_text.setStyleSheet(text_edit_style)
        readme_path = doc_dir / "README.md"
        if readme_path.exists():
            markdown_content = readme_path.read_text(encoding='utf-8')
            html_content = markdown_to_html(markdown_content)
            readme_text.setHtml(html_content)
        else:
            readme_text.setPlainText("README not found.")
        tabs.addTab(readme_text, "📋 Overview")
        
        # Documentation Index
        index_text = QTextEdit()
        index_text.setReadOnly(True)
        index_text.setStyleSheet(text_edit_style)
        index_path = doc_dir / "DOCUMENTATION_INDEX.md"
        if index_path.exists():
            markdown_content = index_path.read_text(encoding='utf-8')
            html_content = markdown_to_html(markdown_content)
            index_text.setHtml(html_content)
        else:
            index_text.setPlainText("Documentation index not found.")
        tabs.addTab(index_text, "📑 Index")
        
        layout.addWidget(tabs)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                padding: 10px 25px;
                font-size: 13px;
                font-weight: bold;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look
    
    window = MorphoMappingGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

