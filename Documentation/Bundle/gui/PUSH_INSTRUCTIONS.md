# GitHub Push - Anleitung

## ✅ Sicherheit: Was wird gepusht?

**NUR die GUI-Dateien werden hinzugefügt/geändert:**
- `Documentation/Bundle/gui/morphomapping_gui.py` (neue Datei)
- `Documentation/Bundle/gui/app_pyside6.py` (wird NICHT gepusht, nur lokal)
- `Documentation/Bundle/gui/core/` (Business-Logik, unverändert)
- `Documentation/Bundle/gui/*.md` (Dokumentation)
- `Documentation/Bundle/gui/.gitignore` (Git-Ignore-Regeln)

**Das ursprüngliche Package bleibt unverändert:**
- `morphomapping/` - **UNVERÄNDERT**
- `R/` - **UNVERÄNDERT**
- Alle anderen Package-Dateien - **UNVERÄNDERT**

## 📋 Push-Schritte

### 1. Status prüfen
```bash
cd /Users/labor/Documents/Projects/MorphoMapping/upstream_repo/MorphoMapping
git status
```

### 2. Nur GUI-Dateien hinzufügen
```bash
# Neue GUI-Datei
git add Documentation/Bundle/gui/morphomapping_gui.py

# Dokumentation
git add Documentation/Bundle/gui/INSTALLATION.md
git add Documentation/Bundle/gui/USER_GUIDE.md
git add Documentation/Bundle/gui/README.md
git add Documentation/Bundle/gui/DOCUMENTATION_INDEX.md
git add Documentation/Bundle/gui/GITHUB_ISSUES.md

# Git-Ignore
git add Documentation/Bundle/gui/.gitignore

# Core-Module (falls geändert)
git add Documentation/Bundle/gui/core/
```

### 3. Commit erstellen
```bash
git commit -m "Add PySide6 GUI (morphomapping_gui.py) with comprehensive documentation

- New stable PySide6 desktop GUI for large .daf files
- Comprehensive installation guide (INSTALLATION.md)
- Complete user guide (USER_GUIDE.md)
- GitHub-ready documentation structure
- Channel filtering for features
- Cluster analysis and visualization
- Known issues: Heatmap visualization and Top10 Features (see GITHUB_ISSUES.md)"
```

### 4. Push zu GitHub
```bash
git push origin main
# oder
git push origin master
```

## ⚠️ WICHTIG: Was NICHT gepusht wird

Die folgenden Dateien werden **NICHT** gepusht (durch .gitignore):
- `app_pyside6.py` (alte Datei, bleibt lokal)
- `app.py` (Streamlit-Version, bleibt lokal)
- `app_nicegui_prototype.py` (Prototype, bleibt lokal)
- `session_cache/` (lokale Cache-Dateien)
- `bundle_runs/` (Ergebnisse)
- `*.log` (Log-Dateien)
- `__pycache__/` (Python-Cache)

## 🐛 Offene Issues

Zwei bekannte Issues werden als GitHub Issues erstellt:
1. **Cluster-Feature Heatmap**: Visualisierung zeigt keine Unterschiede, falsche Achsen
2. **Top 10 Features**: Berechnung stürzt ab bei großen Datensätzen

Siehe `GITHUB_ISSUES.md` für Details.

## ✅ Nach dem Push

1. GitHub Issues erstellen (siehe `GITHUB_ISSUES.md`)
2. README.md prüfen (wird als Hauptseite angezeigt)
3. Releases/Tags erstellen (optional)

