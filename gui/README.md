# MorphoMapping GUI

Desktop application for analyzing ImageStream .daf files (100-500 MB). Built with PySide6.

## Quick Start

```bash
conda activate morphomapping
pip install PySide6 pandas numpy matplotlib scikit-learn hdbscan umap-learn openpyxl scipy
pip install -e ../../../
cd gui
python morphomapping_gui.py
```

See [INSTALLATION.md](INSTALLATION.md) for detailed installation instructions.

## Documentation

- [INSTALLATION.md](INSTALLATION.md) - Installation guide
- [USER_GUIDE.md](USER_GUIDE.md) - User manual
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Documentation overview

## Features

**Project Setup:**
- Run-ID management
- Project directory selection
- Status overview

**DAF File Processing:**
- File selection and drag & drop
- Automatic conversion to FCS
- Progress indicators

**Metadata Management:**
- Manual entry with auto-fill
- CSV/Excel upload
- Auto-numbering for sample_id

**Feature Selection:**
- Clickable feature chips (include/exclude)
- Population/gate selection
- Channel filtering

**Dimensionality Reduction:**
- DensMAP (default)
- UMAP
- t-SNE
- Parameter sliders
- Sampling for large datasets

**Visualization:**
- Interactive plots with color coding
- Adjustable axis limits
- Cell highlighting
- Export as PNG/PDF (300 DPI)

**Clustering:**
- KMeans (default: 10 clusters)
- Gaussian Mixture Models (GMM)
- HDBSCAN
- Elbow plot for KMeans
- Cluster statistics

**Advanced Analysis:**
- Top 3 features per cluster
- Cluster-feature heatmap
- Cluster statistics bar chart
- Feature importance (two-stage approach)

## System Requirements

- Python 3.10 or 3.11 (not 3.12+)
- R 4.0+ (for DAF-to-FCS conversion)
- 8 GB RAM minimum (16 GB recommended)
- macOS 10.15+, Windows 10+, or Linux

## Dependencies

```
PySide6>=6.5.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
hdbscan>=0.8.0
umap-learn>=0.5.0
openpyxl>=3.0.0
scipy>=1.9.0
```

## Architecture

```
gui/
├── morphomapping_gui.py    # Main application
├── core/                   # Business logic
│   ├── config.py
│   ├── file_handling.py
│   ├── conversion.py
│   ├── metadata.py
│   ├── analysis.py
│   └── visualization.py
└── assets/                 # Static assets
```

All compute-intensive operations run in background threads to keep the UI responsive.

## Workflow

1. Set Run-ID and project directory
2. Load DAF files (drag & drop or file dialog)
3. Enter metadata (manual or CSV/Excel upload)
4. Select features and population
5. Run dimensionality reduction
6. Visualize and adjust plot
7. Run clustering
8. Export results

See [USER_GUIDE.md](USER_GUIDE.md) for detailed instructions.

## Deployment

Create a standalone executable:

```bash
pip install pyinstaller
pyinstaller --name="MorphoMapping" --windowed --onefile \
  --add-data "core:core" --add-data "../assets:assets" \
  morphomapping_gui.py
```

Executable will be in `dist/MorphoMapping.app` (macOS) or `dist/MorphoMapping.exe` (Windows).

## Troubleshooting

**"No module named 'PySide6'"**
```bash
pip install PySide6
```

**"Rscript not found"**
Make sure R is installed and in PATH.

**GUI won't start**
Check Python version (should be 3.10 or 3.11), verify dependencies, run with debug output.

See [INSTALLATION.md](INSTALLATION.md) and [USER_GUIDE.md](USER_GUIDE.md) for more troubleshooting help.

## Platform Support

Works on macOS, Windows, and Linux. The same code runs on all platforms.

## License

Same license as the MorphoMapping project.

## Contributing

Contributions welcome. Create an issue or pull request on GitHub.

## Support

- GitHub Issues: https://github.com/Wguido/MorphoMapping/issues
- Documentation: See INSTALLATION.md and USER_GUIDE.md
