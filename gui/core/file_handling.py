"""
File handling utilities.
"""

from pathlib import Path
from typing import Dict, List


def get_run_paths(project_dir: Path, run_id: str) -> Dict[str, Path]:
    """Get paths for a run."""
    base = project_dir / "bundle_runs" / run_id
    return {
        "base": base,
        "raw_daf": base / "raw_daf",
        "fcs": base / "fcs",
        "metadata": base / "metadata",
        "results": base / "results",
        "csv_cache": base / "csv_cache",
    }


def get_file_counts(paths: Dict[str, Path]) -> Dict[str, int]:
    """Get file counts for DAF and FCS files."""
    return {
        "daf": len(list(paths["raw_daf"].glob("*.daf"))),
        "fcs": len(list(paths["fcs"].glob("*.fcs"))),
    }

