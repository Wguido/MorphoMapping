# Migration Plan: Streamlit → NiceGUI

## Wichtig: Kein Code-Verlust!

**Der meiste Code bleibt erhalten!** Nur die UI-Layer muss neu geschrieben werden.

## Code-Wiederverwendbarkeit

### ✅ 100% Wiederverwendbar (Business Logic)

Diese Funktionen bleiben **komplett unverändert**:

```python
# Alle diese Funktionen funktionieren in beiden Frameworks:

def get_run_paths(project_dir: Path, run_id: str) -> Dict[str, Path]
def save_uploaded_files(files, destination: Path) -> List[Path]
def convert_daf_files(daf_files: List[Path], output_dir: Path) -> List[Path]
def load_or_create_metadata(path: Path) -> pd.DataFrame
def save_metadata(df: pd.DataFrame, path: Path) -> None
def run_dimensionality_reduction(...) -> pd.DataFrame
def calculate_feature_importance(...) -> pd.DataFrame
def get_axis_labels(method: str) -> Tuple[str, str]
```

**Das sind ~80% des Codes!**

### ⚠️ Muss angepasst werden (UI-Layer)

Nur die UI-Komponenten müssen neu geschrieben werden:

**Streamlit:**
```python
st.file_uploader(...)
st.button(...)
st.selectbox(...)
st.pyplot(fig)
st.altair_chart(chart)
st.session_state[...]
```

**NiceGUI:**
```python
ui.upload(...)
ui.button(...)
ui.select(...)
ui.plotly(...)  # oder matplotlib
ui.column(...)
session_state[...]  # ähnlich!
```

### 📊 Code-Aufteilung

```
app.py (aktuell ~1500 Zeilen):
├── Business Logic:        ~800 Zeilen (✅ 100% wiederverwendbar)
├── Streamlit UI:         ~600 Zeilen (⚠️ muss neu geschrieben)
└── Konfiguration:         ~100 Zeilen (✅ 100% wiederverwendbar)
```

## Migration-Strategie

### Option 1: Parallele Entwicklung (Empfohlen)

**Streamlit bleibt funktionsfähig, NiceGUI wird parallel entwickelt:**

```
gui/
├── app.py                    # Streamlit (bleibt!)
├── app_nicegui.py            # NiceGUI (neu)
├── core/                     # Gemeinsame Business Logic
│   ├── file_handling.py      # get_run_paths, save_uploaded_files, etc.
│   ├── conversion.py         # convert_daf_files
│   ├── analysis.py           # run_dimensionality_reduction
│   └── visualization.py       # calculate_feature_importance
├── streamlit_ui/             # Streamlit-spezifische UI
│   └── components.py
└── nicegui_ui/               # NiceGUI-spezifische UI
    └── components.py
```

**Vorteile:**
- ✅ Streamlit bleibt funktionsfähig
- ✅ Beide Versionen können parallel getestet werden
- ✅ Business Logic wird geteilt (DRY)
- ✅ Einfacher Rollback falls NiceGUI Probleme hat

### Option 2: Refactoring zuerst, dann Migration

**Schritt 1: Refactoring (1-2 Tage)**
- Business Logic in separate Module extrahieren
- UI-Code in separate Module trennen
- Streamlit-Version bleibt funktionsfähig

**Schritt 2: NiceGUI-UI entwickeln (2-3 Tage)**
- Neue UI mit NiceGUI
- Verwendet die gleiche Business Logic
- Streamlit-Version bleibt parallel

**Schritt 3: Testing & Entscheidung (1 Woche)**
- Beide Versionen testen
- Performance-Vergleich
- User-Feedback
- Entscheidung: NiceGUI oder Streamlit

### Option 3: Schrittweise Migration

**Feature für Feature migrieren:**

1. **Phase 1: File Upload** (1 Tag)
   - NiceGUI-Version: Upload funktioniert
   - Streamlit-Version: Bleibt für Rest

2. **Phase 2: Conversion** (1 Tag)
   - NiceGUI-Version: Upload + Conversion
   - Streamlit-Version: Bleibt für Rest

3. **Phase 3: Analysis** (2 Tage)
   - NiceGUI-Version: Vollständig
   - Streamlit-Version: Als Backup

4. **Phase 4: Vollständige Migration**
   - NiceGUI wird Standard
   - Streamlit als Legacy behalten

## Konkrete Code-Beispiele

### Beispiel 1: Business Logic (Unverändert)

```python
# core/analysis.py - Funktioniert in BEIDEN Frameworks!

def run_dimensionality_reduction(
    fcs_dir: Path,
    features: List[str],
    run_paths: Dict[str, Path],
    method: str,
    dens_lambda: float = 2.0,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    perplexity: float = 30.0,
    population: str | None = None,
) -> pd.DataFrame:
    # Komplett identisch in beiden Versionen!
    # ... existing code ...
    return embedding
```

### Beispiel 2: UI-Layer (Unterschiedlich)

**Streamlit:**
```python
# streamlit_ui/components.py
import streamlit as st

def render_file_upload(paths):
    uploaded = st.file_uploader("Upload files", type=["daf"])
    if uploaded:
        saved = save_uploaded_files(uploaded, paths["raw_daf"])
        st.success(f"Uploaded {len(saved)} files")
        return saved
    return []
```

**NiceGUI:**
```python
# nicegui_ui/components.py
from nicegui import ui

def render_file_upload(paths, on_upload_callback):
    upload = ui.upload(
        label="Upload files",
        auto_upload=True,
        max_file_size=500 * 1024 * 1024
    )
    upload.on('upload', lambda e: handle_upload(e, paths, on_upload_callback))
    return upload
```

**Aber beide verwenden:**
```python
# core/file_handling.py - GETEILT!
def save_uploaded_files(files, destination: Path) -> List[Path]:
    # Identischer Code in beiden Versionen!
    ...
```

## Refactoring-Plan

### Schritt 1: Code-Struktur erstellen

```bash
gui/
├── core/                          # NEU: Gemeinsame Business Logic
│   ├── __init__.py
│   ├── file_handling.py          # get_run_paths, save_uploaded_files
│   ├── conversion.py              # convert_daf_files
│   ├── metadata.py                # load_or_create_metadata, save_metadata
│   ├── analysis.py                # run_dimensionality_reduction
│   ├── clustering.py              # Clustering-Logik
│   ├── visualization.py           # calculate_feature_importance, get_axis_labels
│   └── config.py                  # DEFAULT_FEATURES, COLOR_PALETTE, etc.
│
├── streamlit_ui/                  # NEU: Streamlit-spezifische UI
│   ├── __init__.py
│   ├── components.py              # UI-Komponenten
│   └── app.py                     # Haupt-App (refactored)
│
├── nicegui_ui/                    # NEU: NiceGUI-spezifische UI
│   ├── __init__.py
│   ├── components.py              # UI-Komponenten
│   └── app.py                     # Haupt-App
│
└── app.py                         # ALT: Original (wird refactored)
```

### Schritt 2: Business Logic extrahieren

**Vorher (app.py):**
```python
def run_dimensionality_reduction(...):
    # 100 Zeilen Code
    ...

# Später im Code:
if st.button("RUN"):
    embedding = run_dimensionality_reduction(...)
    st.session_state["embedding_df"] = embedding
```

**Nachher (core/analysis.py):**
```python
# core/analysis.py - Unverändert!
def run_dimensionality_reduction(...):
    # 100 Zeilen Code - identisch!
    ...
```

**Nachher (streamlit_ui/app.py):**
```python
from core.analysis import run_dimensionality_reduction

if st.button("RUN"):
    embedding = run_dimensionality_reduction(...)
    st.session_state["embedding_df"] = embedding
```

**Nachher (nicegui_ui/app.py):**
```python
from core.analysis import run_dimensionality_reduction

def on_run_click():
    embedding = run_dimensionality_reduction(...)
    session_state["embedding_df"] = embedding
```

### Schritt 3: UI-Komponenten abstrahieren

**Gemeinsame Abstraktion:**
```python
# core/interfaces.py
from abc import ABC, abstractmethod

class UIComponent(ABC):
    @abstractmethod
    def render_file_upload(self, paths, callback):
        pass
    
    @abstractmethod
    def render_button(self, label, callback):
        pass
```

**Streamlit-Implementierung:**
```python
# streamlit_ui/components.py
class StreamlitUI(UIComponent):
    def render_file_upload(self, paths, callback):
        uploaded = st.file_uploader(...)
        if uploaded:
            callback(uploaded)
```

**NiceGUI-Implementierung:**
```python
# nicegui_ui/components.py
class NiceGUIUI(UIComponent):
    def render_file_upload(self, paths, callback):
        upload = ui.upload(...)
        upload.on('upload', callback)
```

## Empfehlung

### Phase 1: Refactoring (2-3 Tage)
1. Business Logic in `core/` extrahieren
2. Streamlit-UI in `streamlit_ui/` refactoren
3. **Streamlit bleibt vollständig funktionsfähig!**

### Phase 2: NiceGUI-UI (3-4 Tage)
1. NiceGUI-UI in `nicegui_ui/` entwickeln
2. Verwendet die gleiche Business Logic
3. **Streamlit bleibt parallel verfügbar!**

### Phase 3: Testing (1 Woche)
1. Beide Versionen testen
2. Performance-Vergleich
3. User-Feedback
4. **Entscheidung: NiceGUI oder Streamlit**

### Phase 4: Entscheidung
- **Wenn NiceGUI besser**: Streamlit als Legacy behalten
- **Wenn Streamlit ausreicht**: NiceGUI als Alternative behalten
- **Beide können parallel existieren!**

## Zusammenfassung

✅ **Kein Code-Verlust!**
- ~80% des Codes (Business Logic) bleibt unverändert
- Nur UI-Layer muss neu geschrieben werden
- Beide Versionen können parallel existieren

✅ **Sichere Migration**
- Streamlit bleibt funktionsfähig während Migration
- Einfacher Rollback möglich
- Schrittweise Migration möglich

✅ **Code-Wiederverwendung**
- Business Logic wird geteilt
- DRY-Prinzip
- Einfacher zu warten

## Nächste Schritte

1. **Refactoring starten?**
   - Business Logic extrahieren
   - Streamlit-UI refactoren
   - Streamlit bleibt funktionsfähig

2. **NiceGUI-Prototype erweitern?**
   - Mit extrahierter Business Logic
   - Vollständige Feature-Liste
   - Vergleichstest

3. **Beide parallel entwickeln?**
   - Streamlit für Stabilität
   - NiceGUI für neue Features
   - Später entscheiden

