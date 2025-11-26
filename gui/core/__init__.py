"""
Core business logic for MorphoMapping GUI.

This module contains all business logic that is independent of the UI framework.
It can be used by both Streamlit and NiceGUI implementations.
"""

from .config import (
    BUNDLE_ROOT, PROJECT_ROOT, R_SCRIPT,
    DEFAULT_FEATURES, DEFAULT_METADATA_ROWS, COLOR_PALETTE,
    MAX_POINTS_FOR_VISUALIZATION, SAMPLING_METHOD,
)
from .file_handling import (
    get_run_paths, get_file_counts,
)
from .conversion import (
    convert_daf_files,
)
from .metadata import (
    load_or_create_metadata, save_metadata,
)
from .analysis import (
    run_dimensionality_reduction, run_clustering,
)
from .visualization import (
    get_axis_labels, calculate_feature_importance,
)

__all__ = [
    "BUNDLE_ROOT", "PROJECT_ROOT", "R_SCRIPT",
    "DEFAULT_FEATURES", "DEFAULT_METADATA_ROWS", "COLOR_PALETTE",
    "MAX_POINTS_FOR_VISUALIZATION", "SAMPLING_METHOD",
    "get_run_paths", "get_file_counts",
    "convert_daf_files",
    "load_or_create_metadata", "save_metadata",
    "run_dimensionality_reduction", "run_clustering",
    "get_axis_labels", "calculate_feature_importance",
]
