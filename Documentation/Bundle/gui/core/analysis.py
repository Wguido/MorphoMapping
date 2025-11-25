"""
Dimensionality reduction and clustering analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hdbscan

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from morphomapping.morphomapping import MM


def run_dimensionality_reduction(
    fcs_dir: Path,
    features: List[str],
    run_paths: Dict[str, Path],
    method: str,
    dens_lambda: float = 2.0,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    perplexity: float = 30.0,
    population: Optional[str] = None,
    session_state: Optional[Dict] = None,
    max_cells_per_sample: int = 0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run dimensionality reduction on FCS files.
    
    Returns:
        Tuple of (embedding DataFrame, skipped_files info)
    """
    cache_dir = run_paths["csv_cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    converter = MM()
    skipped_files = []
    
    for fcs_file in sorted(fcs_dir.glob("*.fcs")):
        # Safely convert to string to avoid bytes issues
        fcs_file_stem = str(fcs_file.stem) if hasattr(fcs_file, 'stem') else str(Path(fcs_file).stem)
        fcs_file_name = str(fcs_file.name) if hasattr(fcs_file, 'name') else str(Path(fcs_file).name)
        
        csv_path = cache_dir / f"{fcs_file_stem}.csv"
        if not csv_path.exists():
            converter.convert_to_CSV(str(fcs_file), str(csv_path))
        df = pd.read_csv(csv_path)
        
        # Skip files with missing features
        missing = [f for f in features if f not in df.columns]
        if missing:
            skipped_files.append((fcs_file_name, missing))
            continue
        
        # Filter by population if selected
        if population:
            if population not in df.columns:
                skipped_files.append((fcs_file_name, [f"Population '{population}' not found - using all events"]))
            else:
                # Get population column as Series
                pop_series = df[population]
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
                    skipped_files.append((fcs_file_name, ["No events after population filtering"]))
                    continue
        
        # Sample cells if max_cells_per_sample is set
        original_count = len(df)
        if max_cells_per_sample > 0 and len(df) > max_cells_per_sample:
            df = df.sample(n=max_cells_per_sample, random_state=42).reset_index(drop=True)
            skipped_files.append((fcs_file_name, [f"Sampled {max_cells_per_sample} cells from {original_count} total"]))
        
        subset = df[features].copy()
        subset["sample_id"] = fcs_file_stem
        records.append(subset)
    
    if not records:
        raise ValueError("No files found or all files were empty after filtering. Please upload DAF files first or check your population selection.")
    
    combined = pd.concat(records, ignore_index=True)
    
    # Prepare data
    data = combined[features].copy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply dimensionality reduction
    if method == "DensMAP":
        embedder = MM()
        embedder.df = pd.DataFrame(data_scaled, columns=features)
        embedder.dmap(dens_lambda, n_neighbors, min_dist, met="euclidean")
        coords = embedder.df[["x", "y"]].values
        method_name = "densmap"
    elif method == "UMAP":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP is not installed. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        coords = reducer.fit_transform(data_scaled)
        method_name = "umap"
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, verbose=0)
        coords = reducer.fit_transform(data_scaled)
        method_name = "tsne"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embedding = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "sample_id": combined["sample_id"].values,
    })
    
    run_paths["results"].mkdir(parents=True, exist_ok=True)
    embedding.to_parquet(run_paths["results"] / f"{method_name}_embedding.parquet", index=False)
    
    return embedding, {"skipped_files": skipped_files, "usable_files": len(records)}


def run_clustering(
    coords: np.ndarray,
    method: str,
    n_clusters: int = 10,
    method_params: Optional[Dict] = None,
) -> np.ndarray:
    """
    Run clustering on coordinates.
    
    Args:
        coords: 2D array of coordinates (n_samples, 2)
        method: Clustering method ("KMeans", "HDBSCAN", "GMM")
        n_clusters: Number of clusters (for KMeans/GMM)
        method_params: Additional parameters for clustering methods
    
    Returns:
        Cluster labels array
    """
    if method_params is None:
        method_params = {}
    
    if method == "HDBSCAN":
        min_cluster_size = method_params.get("min_cluster_size", max(50, int(len(coords) / 15)))
        min_cluster_size = min(min_cluster_size, 500)  # Cap at 500
        min_samples = method_params.get("min_samples", 10)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(coords)
    elif method == "GMM":
        covariance_type = method_params.get("covariance_type", "full")
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10, covariance_type=covariance_type)
        cluster_labels = gmm.fit_predict(coords)
    else:  # KMeans (default)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
    
    return cluster_labels

