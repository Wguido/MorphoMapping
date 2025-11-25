"""
DAF to FCS conversion utilities.
"""

import subprocess
from pathlib import Path
from typing import List

from .config import R_SCRIPT


def convert_daf_files(daf_files: List[Path], output_dir: Path) -> List[Path]:
    """Convert DAF files to FCS format silently in the background."""
    if not R_SCRIPT.exists():
        raise FileNotFoundError(f"R script not found: {R_SCRIPT}")
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    errors = []
    
    for daf in daf_files:
        target = output_dir / f"{daf.stem}.fcs"
        command = ["Rscript", "--vanilla", "--slave", str(R_SCRIPT), str(daf), str(target)]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            errors.append(f"{daf.name}: {result.stderr.strip()}")
        else:
            converted.append(target)
    
    if errors:
        error_msg = "Errors during conversion:\n" + "\n".join(errors)
        raise RuntimeError(error_msg)
    
    return converted

