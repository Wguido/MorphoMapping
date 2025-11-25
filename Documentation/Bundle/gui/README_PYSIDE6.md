# MorphoMapping GUI - PySide6 Implementation

Stable desktop GUI for processing large ImageStream .daf files (100-500 MB).

## Why PySide6?

- ✅ **Maximum stability**: No WebSocket disconnections, no browser issues
- ✅ **Direct file access**: Files read directly from filesystem (no upload!)
- ✅ **Native desktop app**: Better performance with large data
- ✅ **No connection errors**: Native threading, no network issues

## Installation

```bash
# Activate conda environment
conda activate morphomapping

# Install PySide6 and dependencies
pip install PySide6 pandas numpy matplotlib scikit-learn hdbscan umap-learn openpyxl

# Install MorphoMapping package
pip install -e ../../
```

## Running the GUI

```bash
cd /Users/labor/Documents/Projects/MorphoMapping/upstream_repo/MorphoMapping/Documentation/Bundle/gui
conda activate morphomapping
python app_pyside6.py
```

The GUI will open as a native desktop application window.

## Features

All features from NiceGUI/Streamlit are preserved:

1. **Project Setup**: Set project directory and Run-ID
2. **DAF File Selection**: Select .daf files using native file dialog (no upload!)
3. **Metadata Management**: 
   - Manual entry (left side) with auto-fill from FCS files
   - Upload CSV or Excel metadata files (right side)
   - Scrollable table (shows ~3 rows, rest scrollable)
4. **Features & Gates Selection**: 
   - Clickable feature chips (blue for included, red for excluded)
   - First 10 features visible, rest in expandable section
   - Population/gate selection dropdown
5. **Dimensionality Reduction**: Run DensMAP, UMAP, or t-SNE with parameter sliders
6. **Visualization**: Interactive plots with color coding by metadata
7. **Clustering**: KMeans, Gaussian Mixture Models, or HDBSCAN

## Design Features Preserved

All design improvements from NiceGUI/Streamlit are maintained:

- ✅ Header with logo
- ✅ Project Setup & Status side-by-side
- ✅ Metadata: Manual (left) and Upload (right) side-by-side
- ✅ Feature chips: Blue for included, red for excluded (same size)
- ✅ First 10 features visible, rest in expandable section
- ✅ Status with color coding (green/yellow/red)
- ✅ Auto-numbering for sample_id
- ✅ Scrollable metadata table (shows ~3 rows)

## Key Differences from NiceGUI

| Feature | NiceGUI | PySide6 |
|---------|---------|---------|
| **File Selection** | Web upload | Native file dialog |
| **File Access** | Upload to server | Direct filesystem access |
| **Connection** | WebSocket (can drop) | Native (always stable) |
| **UI Updates** | HTTP requests | Native Qt signals |
| **Background Tasks** | asyncio tasks | QThread workers |
| **Deployment** | Server + browser | Standalone executable |

## Architecture

```
gui/
├── app_pyside6.py          # PySide6 main application
├── core/                    # Business logic (unchanged!)
│   ├── config.py          # Configuration constants
│   ├── file_handling.py   # File operations
│   ├── conversion.py      # DAF to FCS conversion
│   ├── metadata.py         # Metadata handling
│   ├── analysis.py        # Dimensionality reduction & clustering
│   └── visualization.py    # Plotting utilities
└── assets/                 # Logo and static assets
```

## Background Processing

All heavy operations run in background threads:

- **DAF to FCS conversion**: `ConversionWorker` (QThread)
- **Dimensionality reduction**: `AnalysisWorker` (QThread)
- **Clustering**: `ClusteringWorker` (QThread)

The UI remains responsive during processing.

## Creating a Standalone Executable

To create a standalone `.app` (macOS) or `.exe` (Windows):

```bash
# Install PyInstaller
pip install pyinstaller

# Create executable
pyinstaller --name="MorphoMapping" \
            --windowed \
            --onefile \
            --add-data "gui/core:gui/core" \
            --add-data "assets:assets" \
            app_pyside6.py
```

The executable will be in `dist/MorphoMapping.app` (macOS) or `dist/MorphoMapping.exe` (Windows).

## Troubleshooting

### "No module named 'PySide6'"
```bash
pip install PySide6
```

### "Rscript not found"
Make sure R is installed and `Rscript` is in your PATH.

### Plot not updating
Make sure matplotlib backend is set to `QtAgg` (already set in code).

## Platform Compatibility

**✅ PySide6 is fully cross-platform!**

- ✅ **macOS**: Native .app (tested on macOS 13+)
- ✅ **Windows**: Native .exe (Windows 10/11)
- ✅ **Linux**: Native application (Ubuntu, Fedora, etc.)

The same code runs on all platforms. Only the executable packaging differs:

```bash
# macOS
pyinstaller --name="MorphoMapping" --windowed --onefile app_pyside6.py
# → Creates MorphoMapping.app

# Windows
pyinstaller --name="MorphoMapping" --windowed --onefile app_pyside6.py
# → Creates MorphoMapping.exe

# Linux
pyinstaller --name="MorphoMapping" --onefile app_pyside6.py
# → Creates MorphoMapping binary
```

**Note**: The Python code itself is identical across platforms. PySide6 handles all platform-specific differences automatically.

## Comparison with NiceGUI

**PySide6 Advantages:**
- ✅ No connection errors
- ✅ Direct file access (no upload)
- ✅ Native desktop app
- ✅ Better performance
- ✅ Standalone executable possible
- ✅ **Cross-platform** (macOS, Windows, Linux)

**NiceGUI Advantages:**
- ✅ Web-based (accessible from anywhere)
- ✅ No installation needed (just browser)
- ✅ Easier to share (just send URL)

For internal tools with large files, **PySide6 is recommended**.

