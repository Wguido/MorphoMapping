"""
Visualization utilities.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from morphomapping.morphomapping import MM


def get_axis_labels(method: str) -> Tuple[str, str]:
    """Get axis labels based on dimensionality reduction method."""
    method_map = {
        "DensMAP": ("DensMAP_1", "DensMAP_2"),
        "UMAP": ("UMAP_1", "UMAP_2"),
        "t-SNE": ("t-SNE_1", "t-SNE_2"),
    }
    return method_map.get(method, ("Component_1", "Component_2"))


def calculate_feature_importance(
    embedding_df: pd.DataFrame,
    features: List[str],
    run_paths: dict,
    population: str = None,
) -> pd.DataFrame:
    """Calculate feature importance (correlation) for x and y dimensions.
    
    This function uses a memory-efficient approach by processing one feature at a time
    and sampling data aggressively to prevent memory issues.
    """
    import gc
    
    # Aggressively sample embedding to reduce memory usage
    max_samples = 10000  # Further reduced to 10k cells
    if len(embedding_df) > max_samples:
        embedding_df = embedding_df.sample(n=max_samples, random_state=42).copy()
    
    # Get sample_ids from embedding
    embedding_sample_ids = set(embedding_df["sample_id"].astype(str).unique())
    
    # Process one feature at a time to reduce memory footprint
    importance_data = []
    cache_dir = run_paths["csv_cache"]
    converter = MM()
    
    for feature in features:
        try:
            # Collect data for this feature only
            feature_records = []
            fcs_files = sorted(run_paths["fcs"].glob("*.fcs"))
            
            for fcs_file in fcs_files:
                fcs_stem = fcs_file.stem
                if fcs_stem not in embedding_sample_ids:
                    continue
                
                csv_path = cache_dir / f"{fcs_stem}.csv"
                if not csv_path.exists():
                    converter.convert_to_CSV(str(fcs_file), str(csv_path))
                
                try:
                    # Read only this feature and population column
                    needed_cols = [feature]
                    if population:
                        needed_cols.append(population)
                    
                    # Read in small chunks
                    chunk_size = 5000
                    chunks = []
                    for chunk in pd.read_csv(csv_path, usecols=needed_cols, chunksize=chunk_size):
                        # Filter by population if selected
                        if population and population in chunk.columns:
                            # Get population column as Series
                            pop_series = chunk[population]
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
                            chunk = chunk[mask]
                        
                        if len(chunk) > 0 and feature in chunk.columns:
                            # Sample chunk if too large
                            if len(chunk) > 5000:
                                chunk = chunk.sample(n=5000, random_state=42)
                            chunks.append(chunk[[feature]])
                    
                    if chunks:
                        df = pd.concat(chunks, ignore_index=True)
                        # Sample if still too large
                        if len(df) > 5000:
                            df = df.sample(n=5000, random_state=42)
                        df["sample_id"] = fcs_stem
                        feature_records.append(df)
                    
                    del chunks
                    gc.collect()
                except Exception as e:
                    print(f"Warning: Could not process {fcs_file.name} for {feature}: {e}")
                    continue
            
            if not feature_records:
                continue
            
            # Combine for this feature
            feature_data = pd.concat(feature_records, ignore_index=True)
            feature_data["sample_id"] = feature_data["sample_id"].astype(str)
            
            # Merge with embedding
            merged = embedding_df[["sample_id", "x", "y"]].merge(
                feature_data[[feature, "sample_id"]], on="sample_id", how="inner"
            )
            
            del feature_data, feature_records
            gc.collect()
            
            if len(merged) == 0:
                continue
            
            # Calculate correlation
            valid_mask = merged[feature].notna() & merged["x"].notna() & merged["y"].notna()
            if valid_mask.sum() > 10:
                subset = merged[valid_mask]
                corr_x = subset[feature].corr(subset["x"])
                corr_y = subset[feature].corr(subset["y"])
                if pd.notna(corr_x) and pd.notna(corr_y):
                    importance_data.append({
                        "feature": feature,
                        "x_impact": abs(corr_x),
                        "y_impact": abs(corr_y),
                        "x_correlation": corr_x,
                        "y_correlation": corr_y,
                    })
            
            del merged
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Could not calculate correlation for {feature}: {e}")
            continue
    
    return pd.DataFrame(importance_data)

