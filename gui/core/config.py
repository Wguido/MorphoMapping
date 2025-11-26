"""
Configuration constants for MorphoMapping GUI.
"""

from pathlib import Path

# Default features
DEFAULT_FEATURES = [
    "Area_Ch01",
    "Aspect.Ratio_Ch01",
    "Contrast_Ch01",
    "Intensity_MC_Ch01",
    "Intensity_MC_Ch02",
    "Std.Dev._Ch03",
    "Perimeter_Ch01",
    "Circularity_Ch01",
]

DEFAULT_METADATA_ROWS = [{"sample_id": "", "group": "", "replicate": ""}]

# Color palette for visualization
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d3", "#c7c7c7", "#dbdb8d", "#9edae5"
]

# Performance settings
MAX_POINTS_FOR_VISUALIZATION = 1000000  # Very high limit - effectively disabled
SAMPLING_METHOD = "random"  # "random" or "stratified"

# Paths
BUNDLE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BUNDLE_ROOT.parents[2]
R_SCRIPT = PROJECT_ROOT / "R" / "daf_to_fcs_cli.R"

