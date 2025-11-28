# Installing MorphoMapping GUI

This guide covers installing MorphoMapping GUI on macOS, Windows, and Linux. You don't need programming experience, but you should be comfortable using a terminal.

## System Requirements

**Operating System:**
- macOS 10.15 or newer
- Windows 10 or newer
- Linux (Ubuntu 20.04+ or Fedora 33+)

**Hardware:**
- 8 GB RAM minimum (16 GB recommended for large files)
- 2 GB free disk space
- Modern CPU (Intel Core i5 or better, Apple Silicon M1/M2)

**Software:**
- Python 3.10 or 3.11 (not 3.12+ due to compatibility issues)
- R 4.0 or newer (for DAF-to-FCS conversion)

## Installing Python

### macOS

**Option 1: Homebrew**
```bash
brew install python@3.11
```

**Option 2: Download from python.org**
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 for macOS
3. Run the installer
4. Make sure to check "Add Python to PATH"

### Windows

1. Download Python 3.11 from https://www.python.org/downloads/
2. Run the installer
3. Check "Add Python to PATH" during installation

### Linux

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**Fedora:**
```bash
sudo dnf install python3.11 python3-pip
```

**Verify installation:**
```bash
python --version
```
Should show Python 3.10.x or 3.11.x

## Installing R

### macOS
```bash
brew install r
```

Or download from https://cran.r-project.org/bin/macosx/

### Windows
1. Download R from https://cran.r-project.org/bin/windows/base/
2. Run the installer
3. Check "Add R to PATH"

### Linux

**Ubuntu/Debian:**
```bash
sudo apt install r-base
```

**Fedora:**
```bash
sudo dnf install R
```

**Verify installation:**
```bash
Rscript --version
```

## Downloading MorphoMapping

**Option 1: Download ZIP**
1. Go to https://github.com/Wguido/MorphoMapping
2. Click "Code" → "Download ZIP"
3. Extract to your preferred location

**Option 2: Git clone**
```bash
git clone https://github.com/Wguido/MorphoMapping.git
cd MorphoMapping
```

## Setting Up the Environment

We use Conda to manage dependencies. If you don't have Conda installed:

**macOS/Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

**Windows:**
Download from https://docs.conda.io/en/latest/miniconda.html

**Create the environment:**
```bash
cd /path/to/MorphoMapping
conda create -n morphomapping python=3.11
conda activate morphomapping
```

Note: You'll need to run `conda activate morphomapping` each time you open a new terminal.

## Installing Dependencies

Make sure the environment is activated (you should see `(morphomapping)` in your prompt):

```bash
cd gui
pip install PySide6 pandas numpy matplotlib scikit-learn hdbscan umap-learn openpyxl scipy
pip install -e ../../../
```

This installs:
- PySide6: GUI framework
- pandas, numpy: Data processing
- matplotlib: Plotting
- scikit-learn, hdbscan, umap-learn: Machine learning algorithms
- openpyxl: Excel file support
- scipy: Scientific computing

Installation takes 5-10 minutes depending on your connection.

## Testing the Installation

```bash
conda activate morphomapping
cd gui
python morphomapping_gui.py
```

If a window opens with the MorphoMapping interface, installation was successful.

## Troubleshooting

**"python: command not found"**
- macOS/Linux: Try `python3` instead
- Windows: Make sure Python was added to PATH during installation

**"conda: command not found"**
- Close and reopen your terminal
- Or add Conda to PATH manually (macOS):
  ```bash
  echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
  source ~/.zshrc
  ```

**"Rscript: command not found"**
- Make sure R is installed and added to PATH
- Verify with `Rscript --version`

**"No module named 'PySide6'"**
```bash
conda activate morphomapping
pip install PySide6
```

**"No module named 'morphomapping'"**
```bash
conda activate morphomapping
cd /path/to/MorphoMapping
pip install -e ./
```

**GUI won't start**
1. Check Python version: `python --version` (should be 3.10 or 3.11)
2. Verify dependencies: `pip list | grep PySide6`
3. Run with debug output: `python morphomapping_gui.py 2>&1 | tee debug.log`

## Creating a Standalone Executable

If you want to create an executable that doesn't require Python:

```bash
conda activate morphomapping
pip install pyinstaller
cd gui
pyinstaller --name="MorphoMapping" --windowed --onefile \
  --add-data "core:core" --add-data "../assets:assets" \
  morphomapping_gui.py
```

The executable will be in `dist/MorphoMapping.app` (macOS) or `dist/MorphoMapping.exe` (Windows).

## Next Steps

Once installation is complete:
1. Read [USER_GUIDE.md](USER_GUIDE.md) for usage instructions
2. Check [README.md](README.md) for an overview of features

For help, create an issue at https://github.com/Wguido/MorphoMapping/issues
