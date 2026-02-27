#!/usr/bin/env bash
# MorphoMapping Installer (macOS / OS X)
# Run once: installs Conda env, repo, and all packages.

set -e
INSTALL_DIR="${HOME}/MorphoMapping"
CONDA_ENV="morphomapping"

echo ""
echo "============================================"
echo "  MorphoMapping Installer (macOS)"
echo "============================================"
echo ""

CONDA_CMD=""
if command -v conda &>/dev/null; then
  CONDA_CMD="conda"
  echo "[OK] Conda found."
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  export PATH="${HOME}/miniconda3/bin:${PATH}"
  CONDA_CMD="${HOME}/miniconda3/bin/conda"
  echo "[OK] Miniconda found at ${HOME}/miniconda3."
elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
  export PATH="${HOME}/anaconda3/bin:${PATH}"
  CONDA_CMD="${HOME}/anaconda3/bin/conda"
  echo "[OK] Anaconda found at ${HOME}/anaconda3."
else
  echo "Conda/Miniconda not found. Install Miniconda now? (y/n)"
  read -r answer
  if [[ "$answer" =~ ^[jJyY] ]]; then
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
      URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    else
      URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    fi
    echo "Downloading Miniconda ..."
    curl -sL "$URL" -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
    rm -f /tmp/miniconda.sh
    export PATH="${HOME}/miniconda3/bin:${PATH}"
    CONDA_CMD="${HOME}/miniconda3/bin/conda"
    echo "[OK] Miniconda installed. Restart the terminal or run: source ~/.zshrc"
  else
    echo "Install Miniconda manually:"
    echo "  brew install --cask miniconda"
    echo "  or: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
  fi
fi

if [[ ! -d "$INSTALL_DIR" ]]; then
  mkdir -p "$INSTALL_DIR"
  if command -v git &>/dev/null; then
    echo "Cloning MorphoMapping to $INSTALL_DIR ..."
    git clone https://github.com/Wguido/MorphoMapping.git "$INSTALL_DIR"
  else
    echo "[NOTE] Git not found."
    echo "Install MorphoMapping manually:"
    echo "  1. https://github.com/Wguido/MorphoMapping → Code → Download ZIP"
    echo "  2. Extract and copy contents to $INSTALL_DIR"
    echo "  3. Run this script again."
    exit 1
  fi
else
  echo "Directory already exists: $INSTALL_DIR"
  if command -v git &>/dev/null && [[ -d "$INSTALL_DIR/.git" ]]; then
    (cd "$INSTALL_DIR" && git pull --quiet 2>/dev/null || true)
  fi
fi

if [[ ! -f "$INSTALL_DIR/gui/morphomapping_gui.py" ]]; then
  echo "[ERROR] Expected file not found: $INSTALL_DIR/gui/morphomapping_gui.py"
  echo "Ensure the repo is fully present under $INSTALL_DIR."
  exit 1
fi

echo ""
echo "Creating Conda environment \"$CONDA_ENV\" (Python 3.11) ..."
"$CONDA_CMD" create -n "$CONDA_ENV" python=3.11 -y

echo ""
echo "Installing Python packages (may take a few minutes) ..."
CONDA_SH="$(dirname "$CONDA_CMD")/../etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] && source "$CONDA_SH"
conda activate "$CONDA_ENV" 2>/dev/null || { eval "$("$CONDA_CMD" shell.bash hook)" 2>/dev/null && conda activate "$CONDA_ENV"; }

pip install PySide6 pandas numpy matplotlib scikit-learn hdbscan umap-learn openpyxl scipy -q
cd "$INSTALL_DIR"
pip install -e . -q
cd - >/dev/null

echo ""
echo "============================================"
echo "  Installation complete."
echo "============================================"
echo ""
echo "MorphoMapping is at: $INSTALL_DIR"
echo "To start the GUI: ./Start_MorphoMapping.sh"
echo ""
