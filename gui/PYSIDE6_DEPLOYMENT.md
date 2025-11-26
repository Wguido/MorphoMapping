# PySide6 Deployment & GitHub-Kompatibilität

## GitHub-Kompatibilität: ✅ Perfekt

**PySide6 ist eine normale Python-Bibliothek** - keine speziellen GitHub-Anforderungen!

### Was funktioniert:

1. **Code auf GitHub**: PySide6-Code ist reines Python → perfekt für Git
2. **Version Control**: Alle `.py` Dateien sind textbasiert → Git-friendly
3. **Dependencies**: PySide6 wird über `pip` installiert → in `requirements.txt` oder `pyproject.toml`
4. **Keine Binaries im Repo**: PySide6 wird zur Laufzeit installiert → kein Problem mit großen Binaries

### Beispiel-Struktur für GitHub:

```
MorphoMapping/
├── .gitignore              # Standard Python .gitignore
├── pyproject.toml          # Dependencies (inkl. PySide6)
├── README.md
├── morphomapping/
│   ├── __init__.py
│   └── ...
└── gui/
    ├── app_pyside6.py      # PySide6 GUI
    ├── core/               # Business-Logik (unverändert)
    └── ...
```

**Keine Änderungen nötig** - funktioniert wie bisher!

---

## Installation für Endanwender

### Option 1: Über pip (Empfohlen für Entwickler)

```bash
# 1. Environment erstellen
conda create -n morphomapping python=3.10
conda activate morphomapping

# 2. Von GitHub installieren
pip install git+https://github.com/Wguido/MorphoMapping@main

# 3. GUI starten
morphomapping-gui
# oder
python -m morphomapping.gui.app_pyside6
```

### Option 2: Standalone Executable (Empfohlen für Endanwender)

**Mit PyInstaller oder cx_Freeze** kann man eine `.app` (macOS) oder `.exe` (Windows) erstellen:

```bash
# Installation von PyInstaller
pip install pyinstaller

# Executable erstellen
pyinstaller --name="MorphoMapping" \
            --windowed \
            --onefile \
            --add-data "gui/core:gui/core" \
            gui/app_pyside6.py
```

**Ergebnis**: Eine einzige `.app` Datei, die alles enthält!

**Vorteile für Endanwender**:
- ✅ Keine Python-Installation nötig
- ✅ Keine Dependencies installieren
- ✅ Einfach doppelklicken und starten
- ✅ Funktioniert auf macOS, Windows, Linux

### Option 3: Conda Package (Für wissenschaftliche Umgebungen)

```bash
# Conda environment mit allen Dependencies
conda create -n morphomapping python=3.10
conda activate morphomapping
conda install -c conda-forge pyside6 pandas numpy scikit-learn
pip install morphomapping
```

---

## Dependencies in pyproject.toml

```toml
[project]
name = "morphomapping"
version = "0.1.0"
dependencies = [
    "pandas",
    "numpy",
    "scikit-learn",
    "hdbscan",
    "umap-learn",
    "matplotlib",
    "PySide6>=6.5.0",  # GUI Framework
    # ... andere Dependencies
]

[project.optional-dependencies]
gui = [
    "PySide6>=6.5.0",
]
```

**Installation**:
```bash
pip install morphomapping[gui]  # Mit GUI
pip install morphomapping        # Ohne GUI
```

---

## Vergleich: NiceGUI vs. PySide6 für GitHub/Deployment

| Aspekt | NiceGUI | PySide6 |
|--------|---------|---------|
| **GitHub-Kompatibilität** | ✅ Gut | ✅ Perfekt |
| **Code-Größe** | Klein | Mittel |
| **Dependencies** | NiceGUI + FastAPI | PySide6 (Qt) |
| **Installation** | `pip install nicegui` | `pip install PySide6` |
| **Executable** | Möglich (aber komplexer) | ✅ Einfach mit PyInstaller |
| **Standalone-App** | ❌ Braucht Server | ✅ Native Desktop-App |
| **Für Endanwender** | ⚠️ Komplex (Server starten) | ✅ Einfach (Doppelklick) |

---

## Empfohlene Deployment-Strategie

### Für Entwickler/Wissenschaftler:

```bash
# Einfach über pip
pip install morphomapping[gui]
morphomapping-gui
```

### Für Endanwender (ohne Python-Kenntnisse):

**Option A: Executable erstellen**
```bash
# Entwickler erstellt einmalig:
pyinstaller --windowed --onefile gui/app_pyside6.py

# Endanwender bekommt:
MorphoMapping.app  # macOS
MorphoMapping.exe  # Windows
```

**Option B: Installer (macOS)**
```bash
# Mit create-dmg oder ähnlich
create-dmg MorphoMapping.app
# → MorphoMapping-Installer.dmg
```

**Option C: Homebrew Cask (macOS)**
```ruby
# MorphoMapping.rb
cask 'morphomapping' do
  version '0.1.0'
  url "https://github.com/Wguido/MorphoMapping/releases/download/v#{version}/MorphoMapping-#{version}.dmg"
  # ...
end
```

---

## Beispiel: pyproject.toml für PySide6

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "morphomapping"
version = "0.1.0"
description = "MorphoMapping - Imaging Flow Cytometry Analysis"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "pandas>=1.5.0",
    "numpy>=1.23.0",
    "scikit-learn>=1.2.0",
    "hdbscan>=0.8.0",
    "umap-learn>=0.5.0",
    "matplotlib>=3.6.0",
    "flowkit>=1.0.0",
]

[project.optional-dependencies]
gui = [
    "PySide6>=6.5.0",
]

[project.scripts]
morphomapping-gui = "morphomapping.gui.app_pyside6:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["morphomapping*", "gui*"]
```

---

## Fazit

### GitHub-Kompatibilität: ✅ Kein Problem
- PySide6 ist reines Python
- Funktioniert wie jede andere Python-Bibliothek
- Keine speziellen Git-Anforderungen

### Für Endanwender: ✅ Sehr einfach möglich

**Einfachste Lösung**: Executable mit PyInstaller
- Entwickler: Einmalig `pyinstaller` ausführen
- Endanwender: `.app` oder `.exe` doppelklicken
- Keine Python-Kenntnisse nötig

**Alternative**: Über pip (für technische Nutzer)
- `pip install morphomapping[gui]`
- `morphomapping-gui` starten

**PySide6 ist sogar einfacher zu deployen als NiceGUI**, weil:
- Native Desktop-App (kein Server nötig)
- Einfache Executable-Erstellung
- Keine Browser-Abhängigkeit

