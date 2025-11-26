#!/bin/bash
# Script zum Vorbereiten des GitHub Push
# Verwendung: ./PREPARE_PUSH.sh

set -e  # Exit on error

cd "$(dirname "$0")/../../.."  # Go to MorphoMapping root

echo "🔍 Checking Git status..."
git status

echo ""
echo "📦 Adding GUI files..."
echo ""

# Neue GUI-Datei
if [ -f "gui/morphomapping_gui.py" ]; then
    echo "✅ Adding morphomapping_gui.py"
    git add gui/morphomapping_gui.py
else
    echo "⚠️  morphomapping_gui.py not found!"
fi

# Dokumentation
echo "✅ Adding documentation files..."
git add gui/INSTALLATION.md
git add gui/USER_GUIDE.md
git add gui/README.md
git add gui/DOCUMENTATION_INDEX.md
git add gui/GITHUB_ISSUES.md
git add gui/PUSH_INSTRUCTIONS.md

# Git-Ignore
if [ -f "gui/.gitignore" ]; then
    echo "✅ Adding .gitignore"
    git add gui/.gitignore
fi

# Core-Module (nur wenn geändert)
if [ -n "$(git status --short gui/core/ 2>/dev/null)" ]; then
    echo "✅ Adding core modules"
    git add gui/core/
fi

echo ""
echo "📋 Files staged for commit:"
git status --short

echo ""
echo "⚠️  WICHTIG: Prüfen Sie die Dateien oben!"
echo "   Nur GUI-Dateien sollten hinzugefügt werden."
echo "   Das ursprüngliche Package (morphomapping/, R/) bleibt unverändert."
echo ""
read -p "Fortfahren mit Commit? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Abgebrochen."
    exit 1
fi

echo ""
echo "💾 Creating commit..."
git commit -m "Add PySide6 GUI (morphomapping_gui.py) with comprehensive documentation

- New stable PySide6 desktop GUI for large .daf files (100-500 MB)
- Comprehensive installation guide (INSTALLATION.md) - DAU-tauglich
- Complete user guide (USER_GUIDE.md) with step-by-step workflows
- GitHub-ready documentation structure
- Channel filtering for features (Ch01-Ch12 with auto M01-M12 exclusion)
- Cluster analysis and visualization
- Feature selection with include/exclude chips
- Known issues: Heatmap visualization and Top10 Features (see GITHUB_ISSUES.md)

Features:
- Native file dialogs (no upload needed)
- Drag & drop support for DAF files
- Background processing with QThread workers
- Progress bars for long-running operations
- Export functionality (PNG/PDF, 300 DPI)
- Axis limits adjustment
- Cell highlighting
- Cluster statistics
- Top 3 features per cluster
- Cluster-feature heatmap (row-wise Z-score)

Technical:
- PySide6 for maximum stability
- Cross-platform (macOS, Windows, Linux)
- Standalone executable support (PyInstaller)
- Comprehensive error handling
- Debug output for troubleshooting"

echo ""
echo "✅ Commit created successfully!"
echo ""
echo "📤 Next step: Push to GitHub"
echo "   git push origin main"
echo ""
echo "🐛 After push, create GitHub Issues:"
echo "   See GITHUB_ISSUES.md for issue descriptions"
echo "   Or use: ./create_github_issues.sh (requires GitHub token)"

