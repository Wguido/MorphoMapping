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
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QComboBox, QSlider,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QMessageBox, QProgressBar, QTextEdit, QFrame,
    QSizePolicy, QSpacerItem, QCheckBox
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
                self.finished.emit(self.job_id, False, f"❌ Error: {result.stderr[:100]}")
        except Exception as e:
            self.finished.emit(self.job_id, False, f"❌ Exception: {str(e)[:100]}")


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
                # If file_name exists but sample_id doesn't, create sample_id from file_name
                if "file_name" in metadata_df.columns and "sample_id" not in metadata_df.columns:
                    # Remove .fcs extension if present
                    metadata_df["sample_id"] = metadata_df["file_name"].astype(str).str.replace(r'\.fcs$', '', regex=True)
                    print(f"DEBUG: Created sample_id from file_name")
                
                # Also try matching by file_name (without .fcs extension)
                if "file_name" in metadata_df.columns:
                    metadata_df["file_name_clean"] = metadata_df["file_name"].astype(str).str.replace(r'\.fcs$', '', regex=True)
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
            
            # Save config
            run_config = {
                "project_dir": str(self.paths["base"].parent.parent),
                "run_id": self.session_state["run_id"],
                "features": self.features,
                "population": self.population,
                "method": self.method,
                "parameters": self.method_params,
            }
            (self.paths["base"] / "run_config.json").write_text(json.dumps(run_config, indent=2))
            
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
    
    def __init__(self, embedding_df: pd.DataFrame, features: List[str], paths: Dict[str, Path], population: Optional[str]):
        super().__init__()
        self.embedding_df = embedding_df
        self.features = features
        self.paths = paths
        self.population = population
    
    def run(self):
        try:
            import traceback
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
            
            # Group by sample_id and sample same number of cells from embedding
            combined_with_coords = []
            embedding_subset = self.embedding_df[["sample_id", "x", "y"]].copy()
            embedding_subset["sample_id"] = embedding_subset["sample_id"].astype(str)
            
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
            
            # Sample if too large (to prevent memory issues)
            max_rows = 50000
            if len(combined_data) > max_rows:
                combined_data = combined_data.sample(n=max_rows, random_state=42).reset_index(drop=True)
            
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
            
            print(f"DEBUG: Clean data shape: {combined_data_clean.shape}, columns: {combined_data_clean.columns.tolist()}")
            
            # Create MM object and set dataframe
            mm = MM()
            mm.df = combined_data_clean.copy()
            
            # Calculate feature importance for X dimension
            try:
                print("DEBUG: Calculating X importance...")
                features_x = mm.feature_importance(dep='x', indep='y')
                print(f"DEBUG: X importance calculated successfully: {len(features_x)} features")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"Error calculating X importance: {e}\n{error_trace}")
                features_x = pd.DataFrame()
            
            # Calculate feature importance for Y dimension
            try:
                print("DEBUG: Calculating Y importance...")
                features_y = mm.feature_importance(dep='y', indep='x')
                print(f"DEBUG: Y importance calculated successfully: {len(features_y)} features")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"Error calculating Y importance: {e}\n{error_trace}")
                features_y = pd.DataFrame()
            
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
                
                if not features_x.empty:
                    mm.plot_feature_importance(features_x, str(plot_path_x), base_width=10, base_height=6)
                
                if not features_y.empty:
                    mm.plot_feature_importance(features_y, str(plot_path_y), base_width=10, base_height=6)
                
                self.finished.emit(True, f"✅ Top10 Features saved to:\n{csv_path}\n\nPlots saved to:\n{plot_path_x}\n{plot_path_y}")
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
        
        # 6. Visualization
        self.viz_section = self.create_visualization_section()
        content_layout.addWidget(self.viz_section)
        
        # 7. Clustering
        self.clustering_section = self.create_clustering_section()
        content_layout.addWidget(self.clustering_section)
        
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
        
        run_id_input = QLineEdit()
        run_id_input.setText(self.session_state["run_id"])
        setup_layout.addWidget(QLabel("Run-ID:"))
        setup_layout.addWidget(run_id_input)
        
        def update_project():
            self.session_state["project_dir"] = project_dir_input.text()
            self.session_state["run_id"] = run_id_input.text()
            QMessageBox.information(self, "Success", f"✅ Project set: {run_id_input.text()}")
            self.update_status()
        
        set_btn = QPushButton("💾 Set Project")
        set_btn.clicked.connect(update_project)
        set_btn.setStyleSheet("background-color: #1976D2; color: white; padding: 8px;")
        setup_layout.addWidget(set_btn)
        
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
                status_data["timestamp"] = datetime.datetime.now().isoformat()
                status_data["project_dir"] = project_dir
                status_path.write_text(json.dumps(status_data, indent=2))
            except Exception as e:
                print(f"Warning: Could not save status: {e}")
        
        self.status_layout.addStretch()
    
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
        
        for file_path_str in file_paths:
            file_path = safe_path(file_path_str)
            file_name = safe_str(file_path.name)
            
            # Copy to raw_daf directory
            dest_path = safe_path(paths["raw_daf"]) / file_name
            import shutil
            shutil.copy2(file_path, dest_path)
            
            self.session_state["uploaded_files"].append({
                "name": file_name,
                "path": str(dest_path),
                "size": dest_path.stat().st_size,
            })
            
            # Start background conversion
            fcs_path = safe_path(paths["fcs"]) / f"{safe_str(dest_path.stem)}.fcs"
            job_id = f"convert_{file_name}_{datetime.datetime.now().timestamp()}"
            
            worker = ConversionWorker(dest_path, fcs_path, job_id)
            worker.finished.connect(
                lambda job_id, success, msg, w=worker: self.on_conversion_finished(job_id, success, msg, w)
            )
            self.active_workers.append(worker)
            
            # Show progress bar
            self.conversion_progress.setVisible(True)
            self.conversion_progress.setRange(0, 0)  # Indeterminate progress
            self.conversion_progress.setFormat(f"Converting {file_name}...")
            
            worker.start()
            
            self.processing_status.append(f"🔄 Converting {file_name}...")
        
        self.daf_files_label.setText(f"Selected {len(file_paths)} file(s)")
        self.update_status()
        self.load_features_and_gates()
    
    def create_daf_section(self) -> QGroupBox:
        """Create DAF file selection section."""
        group = QGroupBox("2️⃣ Select DAF Files")
        group.setFont(QFont("Arial", 12, QFont.Bold))
        
        layout = QVBoxLayout()
        
        # Drag and drop area
        drop_area = QLabel("📁 Drag & Drop DAF files here\nor click button below")
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
        drop_area.setAcceptDrops(True)
        drop_area.dragEnterEvent = lambda e: self.dragEnterEvent(e)
        drop_area.dropEvent = lambda e: self.dropEvent(e)
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
    
    def on_conversion_finished(self, job_id: str, success: bool, message: str, worker: ConversionWorker):
        """Handle conversion completion."""
        self.processing_status.append(message)
        
        # Hide progress bar if no more conversions running
        active_conversions = [w for w in self.active_workers if isinstance(w, ConversionWorker) and w.isRunning()]
        if not active_conversions:
            self.conversion_progress.setVisible(False)
        
        if success:
            self.update_status()
            # Reload features after a short delay
            QTimer.singleShot(2000, self.load_features_and_gates)
            # Update metadata display to include new FCS files
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
            # Update existing metadata: add new FCS files
            existing_file_names = set()
            if "file_name" in self.session_state["metadata_df"].columns:
                existing_file_names = set(self.session_state["metadata_df"]["file_name"].astype(str).dropna().values)
            
            new_fcs_files = [f for f in fcs_files if f not in existing_file_names]
            if new_fcs_files:
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
            save_metadata_file(self.session_state["metadata_df"], metadata_path)
            
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
            metadata_path = safe_path(paths["metadata"]) / safe_str(Path(file_path).name)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(file_path, metadata_path)
            
            # Load and store
            if file_path.endswith('.csv'):
                self.session_state["metadata_df"] = pd.read_csv(metadata_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.session_state["metadata_df"] = pd.read_excel(metadata_path)
            
            if "file_name" in self.session_state["metadata_df"].columns and "sample_id" not in self.session_state["metadata_df"].columns:
                self.session_state["metadata_df"]["sample_id"] = self.session_state["metadata_df"]["file_name"]
            
            self.metadata_upload_status.setText(f"✅ Loaded: {len(self.session_state['metadata_df'])} rows")
            self.metadata_upload_status.setStyleSheet("color: green;")
            self.update_metadata_display()
            self.update_status()
        except Exception as ex:
            QMessageBox.critical(self, "Error", f"❌ Metadata error: {str(ex)}")
    
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
            exclude_patterns = ["intensity", "Intensity", "saturation", "Saturation", "Raw pixel", "Bkgd", "All", "Mean Pixel", "Max Pixel", "Median Pixel", "Raw", "Time", "Object Number"]
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
            self.update_status()
            
            # Force update of sections visibility
            if hasattr(self, 'clustering_section'):
                if self.session_state.get("embedding_df") is not None:
                    self.clustering_section.setVisible(True)
                    print(f"DEBUG: Clustering section set to visible after analysis")
            
            if hasattr(self, 'top10_features_btn'):
                if self.session_state.get("embedding_df") is not None:
                    self.top10_features_btn.setVisible(True)
                    print(f"DEBUG: Top10 Features button set to visible after analysis")
            
            self.update_visualization()
            self.update_clustering_section()
        else:
            QMessageBox.critical(self, "Error", message)
        
        # Remove worker from active list once finished
        if worker in self.active_workers:
            self.active_workers.remove(worker)
    
    def on_top10_finished(self, success: bool, message: str, worker: FeatureImportanceWorker):
        """Handle Top10 Features calculation completion."""
        self.top10_features_btn.setEnabled(True)
        self.top10_features_btn.setText("📊 Download Top10 Features")
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)
        
        if worker in self.active_workers:
            self.active_workers.remove(worker)
    
    def create_visualization_section(self) -> QGroupBox:
        """Create visualization section."""
        group = QGroupBox("6️⃣ Visualization")
        group.setFont(QFont("Arial", 12, QFont.Bold))
        group.setVisible(False)  # Hidden until analysis is done
        
        layout = QVBoxLayout()
        
        # Color by selection and export buttons
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Color by:"))
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
        
        # Top10 Features download button
        self.top10_features_btn = QPushButton("📊 Download Top10 Features")
        self.top10_features_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        self.top10_features_btn.clicked.connect(self.download_top10_features)
        # Show immediately if embedding already exists
        if self.session_state.get("embedding_df") is not None:
            self.top10_features_btn.setVisible(True)
        else:
            self.top10_features_btn.setVisible(False)  # Only show after analysis
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
            self.viz_section.setVisible(False)
            return
        
        self.viz_section.setVisible(True)
        
        embedding_df = self.session_state["embedding_df"]
        method = self.session_state.get("stored_dim_reduction_method", "DensMAP")
        x_label, y_label = get_axis_labels(method)
        
        # Update color by combo
        self.color_by_combo.clear()
        color_options = ["sample_id"] + [c for c in embedding_df.columns if c not in ["x", "y", "sample_id", "cell_index", "cluster", "cluster_numeric", "highlighted"]]
        self.color_by_combo.addItems(color_options)
        
        # Disconnect previous connection if it exists
        # Use a flag to track if we need to disconnect
        try:
            # Try to disconnect the specific slot
            self.color_by_combo.currentTextChanged.disconnect(self.redraw_plot)
        except (TypeError, RuntimeError, SystemError):
            # No previous connection - that's fine
            pass
        
        # Connect the signal
        self.color_by_combo.currentTextChanged.connect(self.redraw_plot)
        
        # Set initial selection and trigger plot
        if self.color_by_combo.count() > 0:
            self.color_by_combo.setCurrentIndex(0)
        self.redraw_plot()
    
    def redraw_plot(self):
        """Redraw the plot with current color selection."""
        if self.session_state.get("embedding_df") is None:
            return
        
        embedding_df = self.session_state["embedding_df"]
        color_by = self.color_by_combo.currentText() if self.color_by_combo.count() > 0 else "sample_id"
        method = self.session_state.get("stored_dim_reduction_method", "DensMAP")
        x_label, y_label = get_axis_labels(method)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_vals = []
        if color_by in embedding_df.columns:
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
        ax.set_title(f"{method} Visualization", fontsize=16, fontweight="bold")
        
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
        self.plot_label.setPixmap(pixmap.scaled(self.plot_label.width(), self.plot_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
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
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_vals = []
        if color_by in embedding_df.columns:
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
        """Apply axis limits to the plot."""
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
            self.redraw_plot()
        except ValueError:
            QMessageBox.warning(self, "Warning", "⚠️ Invalid number format. Please enter valid numbers.")
    
    def reset_axis_limits(self):
        """Reset axis limits to automatic."""
        self.session_state["axis_limits"] = {}
        self.x_min_input.clear()
        self.x_max_input.clear()
        self.y_min_input.clear()
        self.y_max_input.clear()
        self.redraw_plot()
    
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
            self.session_state.get("selected_population")
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
        
        top3_features_btn = QPushButton("🔍 Top 3 Features per Cluster")
        top3_features_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px;")
        top3_features_btn.clicked.connect(self.export_top3_features)
        analysis_btn_layout.addWidget(top3_features_btn)
        
        heatmap_btn = QPushButton("🔥 Cluster-Feature Heatmap")
        heatmap_btn.setStyleSheet("background-color: #E91E63; color: white; padding: 5px;")
        heatmap_btn.clicked.connect(self.export_cluster_heatmap)
        analysis_btn_layout.addWidget(heatmap_btn)
        
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
            if hasattr(self, 'clustering_section'):
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
            self.cluster_params_layout.addWidget(QLabel("Min cluster size:"))
            min_size_slider = QSlider(Qt.Horizontal)
            min_size_slider.setMinimum(10)
            min_size_slider.setMaximum(500)
            min_size_slider.setValue(50)
            min_size_slider.setTickPosition(QSlider.TicksBelow)
            min_size_slider.setTickInterval(50)
            min_size_label = QLabel("50")
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
                method_params["min_cluster_size"] = 50
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
        
        # Update statistics table
        self.cluster_stats_table.setRowCount(len(cluster_counts))
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
            
            self.cluster_stats_table.setItem(idx, 0, QTableWidgetItem(str(cluster)))
            self.cluster_stats_table.setItem(idx, 1, QTableWidgetItem(str(count)))
            self.cluster_stats_table.setItem(idx, 2, QTableWidgetItem(f"{percentage:.2f}%"))
            self.cluster_stats_table.setItem(idx, 3, QTableWidgetItem(sample_str))
        
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
            cluster_feature_means = []
            
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
                
                # Get top 3 features
                if cluster_feature_values:
                    sorted_features = sorted(cluster_feature_values.items(), key=lambda x: x[1], reverse=True)
                    top3 = sorted_features[:3]
                    cluster_feature_means.append({
                        "Cluster": cluster,
                        "Top1_Feature": top3[0][0] if len(top3) > 0 else "",
                        "Top1_Value": f"{top3[0][1]:.4f}" if len(top3) > 0 else "",
                        "Top2_Feature": top3[1][0] if len(top3) > 1 else "",
                        "Top2_Value": f"{top3[1][1]:.4f}" if len(top3) > 1 else "",
                        "Top3_Feature": top3[2][0] if len(top3) > 2 else "",
                        "Top3_Value": f"{top3[2][1]:.4f}" if len(top3) > 2 else "",
                    })
            
            if not cluster_feature_means:
                QMessageBox.warning(self, "Warning", "Could not calculate feature means per cluster.")
                return
            
            # Save to CSV
            top3_df = pd.DataFrame(cluster_feature_means)
            paths["results"].mkdir(parents=True, exist_ok=True)
            output_path = paths["results"] / "top3_features_per_cluster.csv"
            top3_df.to_csv(output_path, index=False)
            
            QMessageBox.information(
                self, 
                "Success", 
                f"✅ Top 3 Features per Cluster saved to:\n{output_path}\n\n"
                f"Found {len(cluster_feature_means)} clusters with feature data."
            )
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error exporting top3 features: {e}\n{error_trace}")
            QMessageBox.critical(self, "Error", f"❌ Error: {str(e)[:200]}")
    
    def export_cluster_heatmap(self):
        """Export cluster-feature heatmap with row-wise Z-score normalization."""
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
            
            print(f"DEBUG: Combined feature_data shape: {feature_data.shape}")
            print(f"DEBUG: feature_data columns (first 10): {feature_data.columns.tolist()[:10]}")
            
            # Merge embedding_df with feature_data using sample_id
            # First, ensure both have matching sample_id format
            embedding_df["sample_id"] = embedding_df["sample_id"].astype(str)
            
            # Create cell_index in both DataFrames for proper alignment
            # embedding_df should already have cell_index from analysis
            if "cell_index" not in embedding_df.columns:
                # Create cell_index as 0-based index within each sample
                embedding_df = embedding_df.reset_index(drop=True)
                embedding_df["cell_index"] = embedding_df.groupby("sample_id").cumcount()
            
            # Create cell_index in feature_data if not present
            if "cell_index" not in feature_data.columns:
                feature_data = feature_data.reset_index(drop=True)
                feature_data["cell_index"] = feature_data.groupby("sample_id").cumcount()
            
            print(f"DEBUG: embedding_df sample_ids: {embedding_df['sample_id'].unique()[:5]}")
            print(f"DEBUG: feature_data sample_ids: {feature_data['sample_id'].unique()[:5]}")
            print(f"DEBUG: embedding_df cell_index range: {embedding_df['cell_index'].min()} to {embedding_df['cell_index'].max()}")
            print(f"DEBUG: feature_data cell_index range: {feature_data['cell_index'].min()} to {feature_data['cell_index'].max()}")
            
            # Merge embedding with feature data on sample_id and cell_index
            merged_df = embedding_df.merge(
                feature_data,
                on=["sample_id", "cell_index"],
                how="inner",  # Use inner join to only keep matched rows
                suffixes=("_emb", "_feat")
            )
            
            print(f"DEBUG: Merged data shape: {merged_df.shape}")
            print(f"DEBUG: Features available in merged: {sum([f in merged_df.columns for f in features])} out of {len(features)}")
            
            # Calculate mean feature values per cluster
            # Ensure cluster is a Series
            cluster_series = merged_df["cluster"]
            if isinstance(cluster_series, pd.DataFrame):
                cluster_series = cluster_series.iloc[:, 0]
            clusters = sorted([c for c in cluster_series.unique() if pd.notna(c)])
            
            print(f"DEBUG: Clusters found: {clusters}")
            
            if not clusters:
                QMessageBox.warning(self, "Warning", "No valid clusters found in data.")
                return
            
            heatmap_data = []
            
            for cluster in clusters:
                cluster_mask = merged_df["cluster"] == cluster
                cluster_df = merged_df[cluster_mask]
                
                if len(cluster_df) == 0:
                    continue
                
                cluster_means = {"Cluster": cluster}
                for feature in features:
                    if feature not in cluster_df.columns:
                        cluster_means[feature] = np.nan
                        continue
                    
                    # Calculate mean directly from merged data
                    feature_values = cluster_df[feature].dropna()
                    if len(feature_values) > 0:
                        mean_val = float(feature_values.mean())
                        cluster_means[feature] = mean_val
                    else:
                        cluster_means[feature] = np.nan
                
                heatmap_data.append(cluster_means)
            
            print(f"DEBUG: Heatmap data entries: {len(heatmap_data)}")
            if heatmap_data:
                print(f"DEBUG: First cluster features count: {len([k for k in heatmap_data[0].keys() if k != 'Cluster'])}")
            
            # Create DataFrame: rows = clusters, columns = features
            heatmap_df = pd.DataFrame(heatmap_data)
            
            if heatmap_df.empty:
                QMessageBox.warning(self, "Warning", "No heatmap data created.")
                return
            
            print(f"DEBUG: heatmap_df shape before transpose: {heatmap_df.shape}")
            print(f"DEBUG: heatmap_df columns (first 5): {heatmap_df.columns.tolist()[:5]}")
            
            heatmap_df = heatmap_df.set_index("Cluster")  # Cluster as index (rows)
            # Now transpose: features as rows (index), clusters as columns
            heatmap_df = heatmap_df.T
            # After transpose: heatmap_df.index = Features, heatmap_df.columns = Clusters
            
            print(f"DEBUG: heatmap_df shape after transpose: {heatmap_df.shape}")
            print(f"DEBUG: heatmap_df index (first 5 features): {heatmap_df.index.tolist()[:5]}")
            print(f"DEBUG: heatmap_df columns (clusters): {heatmap_df.columns.tolist()}")
            print(f"DEBUG: heatmap_df value range before z-score: min={heatmap_df.values.min():.2f}, max={heatmap_df.values.max():.2f}")
            
            # Remove features with all NaN
            heatmap_df = heatmap_df.dropna(how='all')
            
            if heatmap_df.empty:
                QMessageBox.warning(self, "Warning", "No valid feature data for heatmap after removing NaN.")
                return
            
            print(f"DEBUG: heatmap_df shape after dropna: {heatmap_df.shape}")
            
            # Row-wise Z-score normalization
            from scipy.stats import zscore
            
            # Ensure all values are numeric before applying zscore
            # Convert to numeric, coercing errors to NaN
            heatmap_df_numeric = heatmap_df.apply(pd.to_numeric, errors='coerce')
            
            # Remove rows/columns with all NaN
            heatmap_df_numeric = heatmap_df_numeric.dropna(how='all').dropna(axis=1, how='all')
            
            if heatmap_df_numeric.empty:
                QMessageBox.warning(self, "Warning", "No valid numeric feature data for heatmap.")
                return
            
            # Apply z-score
            heatmap_z = heatmap_df_numeric.apply(zscore, axis=1, nan_policy='omit')
            
            # Ensure heatmap_z is a DataFrame (not Series)
            # If only one row, apply() returns a Series - convert to DataFrame
            if isinstance(heatmap_z, pd.Series):
                heatmap_z = heatmap_z.to_frame().T
            
            # Replace NaN with 0 (for features with no variance)
            heatmap_z = heatmap_z.fillna(0)
            
            # Ensure numeric dtype - handle any remaining non-numeric values
            # Convert each column individually to handle mixed types
            for col in heatmap_z.columns:
                heatmap_z[col] = pd.to_numeric(heatmap_z[col], errors='coerce')
            
            # Fill NaN again after conversion
            heatmap_z = heatmap_z.fillna(0)
            
            # Final conversion to float - ensure all values are scalars
            # Check for any remaining non-scalar values and convert them
            for col in heatmap_z.columns:
                for idx in heatmap_z.index:
                    val = heatmap_z.loc[idx, col]
                    if not isinstance(val, (int, float, np.number)) or (isinstance(val, float) and np.isnan(val)):
                        heatmap_z.loc[idx, col] = 0.0
                    elif isinstance(val, (list, tuple, np.ndarray)):
                        # If somehow a sequence got in, take first element or mean
                        try:
                            if len(val) > 0:
                                heatmap_z.loc[idx, col] = float(np.mean(val))
                            else:
                                heatmap_z.loc[idx, col] = 0.0
                        except:
                            heatmap_z.loc[idx, col] = 0.0
            
            # Now safe to convert to float
            heatmap_z = heatmap_z.astype(float)
            
            # Create heatmap
            # heatmap_z structure: rows (index) = Features, columns = Clusters
            n_features = len(heatmap_z.index)
            n_clusters = len(heatmap_z.columns)
            
            print(f"DEBUG: Final heatmap_z shape: {heatmap_z.shape}")
            print(f"DEBUG: n_features: {n_features}, n_clusters: {n_clusters}")
            print(f"DEBUG: heatmap_z value range: min={heatmap_z.values.min():.2f}, max={heatmap_z.values.max():.2f}")
            print(f"DEBUG: heatmap_z index (first 5): {heatmap_z.index.tolist()[:5]}")
            print(f"DEBUG: heatmap_z columns: {heatmap_z.columns.tolist()}")
            
            if n_features == 0 or n_clusters == 0:
                QMessageBox.warning(self, "Warning", f"Invalid heatmap dimensions: {n_features} features, {n_clusters} clusters")
                return
            
            # Calculate figure size - make it larger for better visibility
            fig_width = max(10, n_clusters * 1.0)
            fig_height = max(8, n_features * 0.4)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Use seaborn if available, otherwise matplotlib
            try:
                import seaborn as sns
                # seaborn.heatmap: rows (y-axis) = index, columns (x-axis) = columns
                # heatmap_z: index = Features, columns = Clusters
                # So: Y-axis = Features, X-axis = Clusters (correct!)
                sns.heatmap(
                    heatmap_z,
                    cmap="RdYlBu_r",
                    center=0,
                    vmin=-3,
                    vmax=3,
                    cbar_kws={"label": "Z-score"},
                    ax=ax,
                    xticklabels=True,
                    yticklabels=True if n_features <= 50 else False,  # Hide y-labels if too many
                    linewidths=0.1,
                    linecolor='gray',
                    square=False
                )
                # Show y-labels for features if not too many
                if n_features <= 50:
                    ax.set_yticklabels(heatmap_z.index, rotation=0, fontsize=8)
                else:
                    # Show every Nth feature label
                    step = max(1, n_features // 50)
                    ax.set_yticks(range(0, n_features, step))
                    ax.set_yticklabels([heatmap_z.index[i] for i in range(0, n_features, step)], rotation=0, fontsize=7)
                
                # Ensure correct axis labels
                ax.set_xlabel("Cluster", fontsize=12, fontweight="bold")
                ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
            except ImportError:
                # Fallback to matplotlib
                # imshow: first dimension (rows) = y-axis, second dimension (columns) = x-axis
                # heatmap_z.values: rows = Features, columns = Clusters
                # So: Y-axis = Features, X-axis = Clusters (correct!)
                im = ax.imshow(heatmap_z.values, cmap="RdYlBu_r", aspect='auto', vmin=-3, vmax=3, origin='upper', interpolation='nearest')
                
                # X-axis: clusters (columns)
                ax.set_xticks(range(n_clusters))
                ax.set_xticklabels([str(int(c)) if isinstance(c, (int, float)) else str(c) for c in heatmap_z.columns], rotation=45, ha='right', fontsize=10)
                
                # Y-axis: features (index/rows)
                if n_features <= 50:
                    ax.set_yticks(range(n_features))
                    ax.set_yticklabels(heatmap_z.index, rotation=0, fontsize=8)
                else:
                    # Show every Nth label
                    step = max(1, n_features // 50)
                    ax.set_yticks(range(0, n_features, step))
                    ax.set_yticklabels([heatmap_z.index[i] for i in range(0, n_features, step)], rotation=0, fontsize=7)
                
                plt.colorbar(im, ax=ax, label="Z-score")
                
                ax.set_xlabel("Cluster", fontsize=12, fontweight="bold")
                ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
            
            ax.set_title(f"Cluster-Feature Heatmap (Row-wise Z-score)\n{n_features} Features × {n_clusters} Clusters", fontsize=14, fontweight="bold")
            
            plt.tight_layout()
            
            # Save
            paths["results"].mkdir(parents=True, exist_ok=True)
            output_path = paths["results"] / "cluster_feature_heatmap.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            # Also save the data
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


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look
    
    window = MorphoMappingGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

