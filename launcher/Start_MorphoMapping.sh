#!/usr/bin/env bash
# Start MorphoMapping GUI (macOS / OS X)

INSTALL_DIR="${HOME}/MorphoMapping"
CONDA_ENV="morphomapping"

if command -v conda &>/dev/null; then
  :
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] Conda not found. Run Install_MorphoMapping.sh first."
  exit 1
fi

conda activate "$CONDA_ENV" 2>/dev/null || {
  echo "[ERROR] Environment \"$CONDA_ENV\" not found. Run Install_MorphoMapping.sh first."
  exit 1
}

if [[ ! -f "$INSTALL_DIR/gui/morphomapping_gui.py" ]]; then
  echo "[ERROR] MorphoMapping not found at $INSTALL_DIR."
  echo "Edit INSTALL_DIR in this script or run Install_MorphoMapping.sh first."
  exit 1
fi

cd "$INSTALL_DIR/gui"
exec python morphomapping_gui.py
