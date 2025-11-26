# MorphoMapping GUI

A stable desktop GUI for analyzing large ImageStream .daf files (100-500 MB).

## 🚀 Quick Start

```bash
# 1. Activate environment
conda activate morphomapping

# 2. Install dependencies (if not already done)
pip install PySide6 pandas numpy matplotlib scikit-learn hdbscan umap-learn openpyxl scipy

# 3. Install MorphoMapping
pip install -e ../../../

# 4. Start GUI
cd gui
python morphomapping_gui.py
```

## 📚 Documentation

- **[INSTALLATION.md](INSTALLATION.md)** - Detailed installation guide (beginner-friendly)
- **[USER_GUIDE.md](USER_GUIDE.md)** - User manual with step-by-step instructions
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Documentation overview

## ✨ Features

### Core Functionality

1. **Project Setup**
   - Set project directory
   - Manage Run-ID
   - Status overview

2. **DAF File Processing**
   - Native file selection (no upload needed!)
   - Drag & drop support
   - Automatic conversion to FCS
   - Progress indicator

3. **Metadata Management**
   - Manual entry (left) with auto-fill
   - CSV/Excel upload (right)
   - Scrollable table (shows ~3 rows)
   - Auto-numbering for sample_id

4. **Feature Selection**
   - Clickable feature chips (blue = included, red = excluded)
   - First 10 features visible, rest collapsible
   - Population/gate selection
   - Channel filtering (Ch01-Ch12 with auto M01-M12 exclusion)

5. **Dimensionality Reduction**
   - **DensMAP** (default)
   - **UMAP**
   - **t-SNE**
   - Parameter sliders for all methods
   - Max cells per sample (sampling)

6. **Visualization**
   - Interactive plots with color coding
   - Color by metadata (sample_id, group, etc.)
   - Adjustable axis limits
   - Cell highlighting
   - Export as PNG/PDF (300 DPI)

7. **Clustering**
   - **KMeans** (10 clusters default)
   - **Gaussian Mixture Models (GMM)**
   - **HDBSCAN**
   - Dynamic parameter sliders
   - Elbow plot for KMeans
   - Cluster statistics

8. **Advanced Analysis**
   - **Top 3 Features per Cluster** (CSV export)
   - **Cluster-Feature Heatmap** (row-wise Z-score, PNG + CSV)
   - **Cluster Statistics Bar Chart** (by groups)
   - **Top 10 Features** for X/Y dimensions

## 🎯 Why PySide6?

- ✅ **Maximum stability**: No WebSocket drops, no browser issues
- ✅ **Direct file access**: Files are read directly from filesystem (no upload!)
- ✅ **Native desktop app**: Better performance with large data
- ✅ **No connection errors**: Native threading, no network issues
- ✅ **Standalone executable**: Can be built as .app (macOS) or .exe (Windows)

## 📋 System Requirements

- **Python**: 3.10 or 3.11 (not 3.12+)
- **R**: 4.0+ (for DAF-to-FCS conversion)
- **RAM**: 8 GB minimum (16 GB recommended)
- **OS**: macOS 10.15+, Windows 10+, Linux (Ubuntu 20.04+)

## 🛠️ Installation

### For Beginners

Follow the detailed **[INSTALLATION.md](INSTALLATION.md)** guide.

### For Advanced Users

```bash
# 1. Create Conda environment
conda create -n morphomapping python=3.11
conda activate morphomapping

# 2. Install dependencies
pip install PySide6 pandas numpy matplotlib scikit-learn hdbscan umap-learn openpyxl scipy

# 3. Install MorphoMapping
cd /path/to/MorphoMapping
pip install -e ./

# 4. Start GUI
cd gui
python morphomapping_gui.py
```

## 📦 Dependencies

```
PySide6>=6.5.0          # GUI Framework
pandas>=1.5.0           # Data processing
numpy>=1.23.0            # Numerical computations
matplotlib>=3.6.0        # Visualization
scikit-learn>=1.2.0      # Machine Learning
hdbscan>=0.8.0           # Clustering
umap-learn>=0.5.0        # Dimensionality reduction
openpyxl>=3.0.0          # Excel files
scipy>=1.9.0             # Scientific computations
```

## 🏗️ Architecture

```
gui/
├── morphomapping_gui.py  # PySide6 main application
├── core/                 # Business logic (framework-independent)
│   ├── config.py        # Configuration constants
│   ├── file_handling.py # File operations
│   ├── conversion.py    # DAF to FCS conversion
│   ├── metadata.py       # Metadata management
│   ├── analysis.py      # Dimensionality reduction & clustering
│   └── visualization.py # Plotting utilities
└── assets/               # Logo and static assets
```

## 🔄 Background Processing

All compute-intensive operations run in background threads:

- **DAF to FCS conversion**: `ConversionWorker` (QThread)
- **Dimensionality reduction**: `AnalysisWorker` (QThread)
- **Clustering**: `ClusteringWorker` (QThread)
- **Feature importance**: `FeatureImportanceWorker` (QThread)

The UI remains responsive during processing.

## 📊 Workflow

1. **Project Setup**: Set Run-ID, choose project directory
2. **DAF Files**: Select files (drag & drop or dialog)
3. **Metadata**: Enter manually or upload CSV/Excel
4. **Features**: Select features (include/exclude), choose population
5. **Dimensionality Reduction**: Choose method, adjust parameters, start analysis
6. **Visualization**: Adjust plot (colors, limits, highlights)
7. **Clustering**: Choose algorithm, adjust parameters, start clustering
8. **Export**: Export results (plots, statistics, features)

## 🎨 Design Features

- ✅ Header with logo
- ✅ Project Setup & Status side by side
- ✅ Metadata: Manual (left) and Upload (right) side by side
- ✅ Feature chips: Blue for included, red for excluded (same size)
- ✅ First 10 features visible, rest collapsible
- ✅ Status with color coding (green/yellow/red)
- ✅ Auto-numbering for sample_id
- ✅ Scrollable metadata table (shows ~3 rows)

## 🚢 Deployment

### Create Standalone Executable

```bash
pip install pyinstaller

pyinstaller --name="MorphoMapping" \
            --windowed \
            --onefile \
            --add-data "core:core" \
            --add-data "../assets:assets" \
            morphomapping_gui.py
```

The executable will be in `dist/MorphoMapping.app` (macOS) or `dist/MorphoMapping.exe` (Windows).

See **[PYSIDE6_DEPLOYMENT.md](PYSIDE6_DEPLOYMENT.md)** for details.

## 🐛 Troubleshooting

### "No module named 'PySide6'"
```bash
pip install PySide6
```

### "Rscript not found"
Make sure R is installed and `Rscript` is in your PATH.

### "Plot not updating"
Make sure matplotlib backend is set to `QtAgg` (already in code).

### GUI won't start
1. Check Python version: `python --version` (should be 3.10 or 3.11)
2. Check all dependencies: `pip list`
3. Start with debug output: `python morphomapping_gui.py 2>&1 | tee debug.log`

## 🌍 Platform Compatibility

**✅ PySide6 is fully cross-platform!**

- ✅ **macOS**: Native .app (tested on macOS 13+)
- ✅ **Windows**: Native .exe (Windows 10/11)
- ✅ **Linux**: Native application (Ubuntu, Fedora, etc.)

The same code runs on all platforms.

## 📝 License

Same license as the MorphoMapping project.

## 🤝 Contributing

Contributions are welcome! Please create an issue or pull request on GitHub.

## 📞 Support

- **GitHub Issues**: https://github.com/Wguido/MorphoMapping/issues
- **Documentation**: See [INSTALLATION.md](INSTALLATION.md) and [USER_GUIDE.md](USER_GUIDE.md)

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-25
