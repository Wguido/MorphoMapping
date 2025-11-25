# MorphoMapping GUI - Installation Guide

## 📋 Overview

This guide walks you through installing the MorphoMapping GUI step by step. No programming experience needed - just follow the instructions.

## ⚙️ System Requirements

### Operating System
- **macOS**: 10.15 (Catalina) or newer
- **Windows**: Windows 10 or newer
- **Linux**: Ubuntu 20.04 or newer / Fedora 33 or newer

### Hardware
- **RAM**: At least 8 GB (16 GB recommended for large files)
- **Disk Space**: At least 2 GB free space
- **Processor**: Modern CPU (Intel Core i5 or better, Apple Silicon M1/M2)

### Software
- **Python**: Version 3.10 or 3.11 (not 3.12 or newer)
- **R**: Version 4.0 or newer (for DAF-to-FCS conversion)

---

## 🚀 Installation - Step by Step

### Step 1: Install Python

#### macOS

1. **Option A: Homebrew (Recommended)**
   ```bash
   # Install Homebrew (if not already installed)
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install Python
   brew install python@3.11
   ```

2. **Option B: Python.org**
   - Go to https://www.python.org/downloads/
   - Download Python 3.11 for macOS
   - Run the installer
   - **IMPORTANT**: Check "Add Python to PATH" during installation

#### Windows

1. Go to https://www.python.org/downloads/
2. Download Python 3.11 for Windows
3. Run the installer
4. **IMPORTANT**: Check "Add Python to PATH" during installation
5. Click "Install Now"

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

#### Linux (Fedora)

```bash
sudo dnf install python3.11 python3-pip
```

**Verification:**
Open a terminal (macOS/Linux) or Command Prompt (Windows) and type:
```bash
python --version
```
You should see `Python 3.10.x` or `Python 3.11.x`.

---

### Step 2: Install R

#### macOS

```bash
# With Homebrew
brew install r
```

Or download R from https://cran.r-project.org/bin/macosx/.

#### Windows

1. Go to https://cran.r-project.org/bin/windows/base/
2. Download R for Windows
3. Run the installer
4. **IMPORTANT**: Check "Add R to PATH" during installation

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install r-base
```

#### Linux (Fedora)

```bash
sudo dnf install R
```

**Verification:**
```bash
Rscript --version
```
You should see a version number (e.g., `R scripting front-end version 4.3.0`).

---

### Step 3: Download MorphoMapping

#### Option A: From GitHub (Recommended)

1. Go to https://github.com/Wguido/MorphoMapping
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file to a folder of your choice (e.g., `~/Documents/MorphoMapping`)

#### Option B: With Git (for advanced users)

```bash
git clone https://github.com/Wguido/MorphoMapping.git
cd MorphoMapping
```

---

### Step 4: Create Conda Environment (Recommended)

**What is Conda?**
Conda is a package manager that helps install and manage all required programs.

#### Install Conda

**macOS/Linux:**
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

**Windows:**
1. Go to https://docs.conda.io/en/latest/miniconda.html
2. Download Miniconda for Windows
3. Run the installer

#### Create Environment

Open a terminal and navigate to the MorphoMapping folder:

```bash
cd /path/to/MorphoMapping
```

Create the environment:

```bash
conda create -n morphomapping python=3.11
conda activate morphomapping
```

**Note:** You need to run `conda activate morphomapping` every time you open a new terminal.

---

### Step 5: Install Dependencies

Make sure the `morphomapping` environment is activated:

```bash
conda activate morphomapping
```

Install all required packages:

```bash
# Navigate to the GUI folder
cd Documentation/Bundle/gui

# Install dependencies
pip install PySide6 pandas numpy matplotlib scikit-learn hdbscan umap-learn openpyxl scipy

# Install the MorphoMapping package
pip install -e ../../../
```

**What's happening here?**
- `PySide6`: The GUI framework
- `pandas`, `numpy`: Data processing
- `matplotlib`: Plots and visualizations
- `scikit-learn`, `hdbscan`, `umap-learn`: Machine learning algorithms
- `openpyxl`: Read/write Excel files
- `scipy`: Scientific computations

**Duration:** 5-10 minutes (depending on your internet connection)

---

### Step 6: Test Installation

Make sure the environment is activated:

```bash
conda activate morphomapping
```

Navigate to the GUI folder:

```bash
cd Documentation/Bundle/gui
```

Start the GUI:

```bash
python morphomapping_gui.py
```

**Expected Result:**
A window should open with:
- Logo at the top
- "Project Setup" section
- "Status" section
- Various buttons and input fields

If the window appears: ✅ **Installation successful!**

---

## 🔧 Troubleshooting

### Problem: "python: command not found"

**Solution:**
- **macOS/Linux**: Use `python3` instead of `python`
- **Windows**: Make sure Python is installed and "Add to PATH" was checked
- Verify installation: `python --version` or `python3 --version`

### Problem: "conda: command not found"

**Solution:**
1. Close the terminal and open it again
2. Or reinstall Conda (see Step 4)
3. **macOS**: Add Conda to PATH:
   ```bash
   echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

### Problem: "Rscript: command not found"

**Solution:**
- **macOS**: `brew install r` or download R from CRAN
- **Windows**: Make sure R is installed and "Add to PATH" was checked
- Verify: `Rscript --version`

### Problem: "No module named 'PySide6'"

**Solution:**
```bash
conda activate morphomapping
pip install PySide6
```

### Problem: "No module named 'morphomapping'"

**Solution:**
```bash
conda activate morphomapping
cd /path/to/MorphoMapping
pip install -e ./
```

### Problem: GUI won't open / Error on startup

**Solution:**
1. Check if all dependencies are installed:
   ```bash
   pip list | grep -E "PySide6|pandas|numpy|matplotlib|scikit-learn|hdbscan|umap-learn"
   ```

2. Check Python version:
   ```bash
   python --version
   ```
   Should be 3.10 or 3.11, NOT 3.12 or newer!

3. Try starting the GUI with debug output:
   ```bash
   python morphomapping_gui.py 2>&1 | tee gui_debug.log
   ```
   Send the `gui_debug.log` file to support.

---

## 📦 Alternative: Standalone Executable (For End Users)

If you don't want a Python installation, you can use a standalone version:

### For Developers: Create Executable

```bash
conda activate morphomapping
pip install pyinstaller

cd Documentation/Bundle/gui
pyinstaller --name="MorphoMapping" \
            --windowed \
            --onefile \
            --add-data "core:core" \
            --add-data "../assets:assets" \
            morphomapping_gui.py
```

The executable will be in `dist/MorphoMapping.app` (macOS) or `dist/MorphoMapping.exe` (Windows).

### For End Users: Use Executable

1. Download the `.app` (macOS) or `.exe` (Windows) file
2. Double-click the file
3. The GUI starts automatically - no installation needed!

**Note:** The executable version is larger (~200-500 MB) but works without a Python installation.

---

## ✅ Installation Successful?

After successful installation, you should be able to:

1. ✅ Start the GUI: `python morphomapping_gui.py`
2. ✅ See a window with logo and various sections
3. ✅ Select DAF files
4. ✅ Enter metadata

**Next Steps:**
- Read the [User Guide](USER_GUIDE.md) for usage instructions
- Or check out the [README](README.md) for an overview

---

## 🆘 Need Help?

If you're having problems:

1. **Check the system requirements** (see above)
2. **Read the troubleshooting section** (see above)
3. **Create a GitHub Issue**: https://github.com/Wguido/MorphoMapping/issues
   - Describe your problem
   - Include error messages
   - Provide your system information (OS, Python version, etc.)

---

## 📝 Checklist

Use this checklist to make sure everything is installed:

- [ ] Python 3.10 or 3.11 installed
- [ ] R installed and `Rscript` works
- [ ] MorphoMapping downloaded
- [ ] Conda environment created (`morphomapping`)
- [ ] All dependencies installed
- [ ] MorphoMapping package installed (`pip install -e ../../../`)
- [ ] GUI starts successfully (`python morphomapping_gui.py`)

If all items are checked: ✅ **You're ready!**

---

**Last Updated:** 2025-11-25
