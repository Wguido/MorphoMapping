# Framework Evaluation für MorphoMapping GUI

## Problemstellung

- **Dateigröße**: 100-500 MB pro .daf File
- **Anzahl**: 1-3 Files pro Session
- **Symptome**: 
  - Leere Seiten / halbe UI
  - Connection Loss
  - Session Resets
  - WebSocket Instabilität

## Root Cause Analysis

### Streamlit's Architektur-Probleme:

1. **Rerun-basiertes Modell**: Jede UI-Änderung triggert vollständigen Script-Rerun
2. **In-Memory State**: Große Objekte müssen durch Streamlit's State-System
3. **WebSocket-Limitierungen**: Große Datenmengen überlasten die Verbindung
4. **Keine echte Background-Jobs**: Alles läuft im Request-Context

### Warum Streamlit bei großen Files versagt:

- Upload + Processing im selben Request-Context
- Streamlit versucht State zu serialisieren, der nie das Backend verlassen sollte
- Memory-Pressure führt zu WebSocket-Instabilität
- UI-Resets bei jedem Rerun mit großen Objekten

## Framework-Vergleich

### 1. NiceGUI ⭐ (Top-Empfehlung)

**Warum für uns:**
- ✅ **Stabiler**: Kein permanentes Script-Rerun
- ✅ **FastAPI-Foundation**: Normale Web-App Architektur
- ✅ **Upload + Background-Jobs**: Getrennte Prozesse möglich
- ✅ **Große Dateien**: Handhabt 100-500 MB Files zuverlässig
- ✅ **Python-only**: Keine JavaScript-Kenntnisse nötig
- ✅ **Streamlit-Feeling**: Ähnliche API, aber stabiler

**Nachteile:**
- ⚠️ Relativ neu (weniger Community als Streamlit)
- ⚠️ Weniger Beispiele für wissenschaftliche Apps

**Migration-Aufwand**: Mittel (2-3 Tage)
- API ähnlich zu Streamlit
- Upload-Handling muss angepasst werden
- Background-Jobs müssen implementiert werden

### 2. Dash ⭐ (Production-Ready)

**Warum für uns:**
- ✅ **Sehr stabil**: Etabliert seit Jahren
- ✅ **Production-Ready**: Gunicorn, Docker, Nginx Support
- ✅ **Wissenschaftliche Apps**: Standard für Bioinformatik
- ✅ **Skalierbar**: Kann große Dateien handhaben (mit richtiger Architektur)
- ✅ **Flask + React**: Solide Foundation

**Nachteile:**
- ⚠️ Mehr Boilerplate als Streamlit/NiceGUI
- ⚠️ React-Kenntnisse hilfreich (aber nicht nötig)
- ⚠️ Steilerer Learning-Curve

**Migration-Aufwand**: Hoch (5-7 Tage)
- Komplett andere API
- Callback-basiertes System
- Upload-Handling muss neu implementiert werden

### 3. PySide6 (Desktop-App)

**Warum für uns:**
- ✅ **Maximal stabil**: Keine WebSocket-Probleme
- ✅ **Direkter File-Zugriff**: Kein Upload nötig
- ✅ **Native Performance**: Keine Browser-Limitierungen
- ✅ **Ideal für interne Tools**: Perfekt für Lab-Umgebung

**Nachteile:**
- ⚠️ Höherer Entwicklungsaufwand
- ⚠️ Plattform-spezifische Builds
- ⚠️ Keine Web-basierte Lösung

**Migration-Aufwand**: Sehr hoch (10-14 Tage)
- Komplett neue GUI-Architektur
- Qt-Kenntnisse nötig
- Alle UI-Komponenten müssen neu gebaut werden

### 4. Streamlit (Aktuell)

**Warum es nicht funktioniert:**
- ❌ Instabil bei großen Dateien
- ❌ Nicht für lange Berechnungen gemacht
- ❌ Verliert Sessions bei Reruns
- ❌ WebSocket-Instabilität

**Kann es verbessert werden?**
- ⚠️ Teilweise: Session State Persistierung hilft
- ⚠️ Teilweise: Matplotlib statt Altair reduziert Probleme
- ❌ Aber: Fundamentale Architektur-Probleme bleiben

## Empfehlung

### Kurzfristig (Sofort):
1. **Streamlit stabilisieren** (bereits implementiert):
   - Session State Persistierung
   - Matplotlib statt Altair
   - Besseres Memory-Management
   - Auto-Save/Load

2. **Workarounds implementieren**:
   - File-Upload auf Disk statt in Memory
   - Background-Processing mit Threading
   - Progress-Bars für lange Operationen
   - Chunked Processing für große Files

### Mittelfristig (1-2 Wochen):
**NiceGUI Prototype entwickeln**

**Warum NiceGUI:**
- Beste Balance zwischen Stabilität und Entwicklungsaufwand
- Ähnliche API zu Streamlit → einfache Migration
- Kann große Files handhaben
- Background-Jobs möglich

**Prototype-Scope:**
- Upload-Funktionalität
- DAF → FCS Conversion (Background)
- Basis-Visualisierung
- Vergleich mit Streamlit-Version

### Langfristig (1-2 Monate):
**Entscheidung basierend auf Prototype:**
- Wenn NiceGUI stabil läuft → Vollständige Migration
- Wenn Probleme → Dash evaluieren
- Wenn Desktop bevorzugt → PySide6 evaluieren

## Implementierungs-Plan

### Phase 1: Streamlit Stabilisierung (Bereits gemacht)
- ✅ Session State Manager
- ✅ Matplotlib statt Altair
- ✅ Auto-Save/Load
- ✅ Besseres Figure-Handling

### Phase 2: NiceGUI Prototype
- [ ] NiceGUI Installation & Setup
- [ ] Basis-Layout (ähnlich Streamlit-Version)
- [ ] File-Upload mit Background-Processing
- [ ] DAF → FCS Conversion (Background-Job)
- [ ] Basis-Visualisierung
- [ ] Vergleichstest mit Streamlit

### Phase 3: Evaluation
- [ ] Stabilitätstest mit 3x 500MB Files
- [ ] Performance-Vergleich
- [ ] User-Feedback
- [ ] Entscheidung: Migration oder weitere Optimierung

## Technische Details

### NiceGUI Architektur-Vorteile:

```python
# Streamlit (Problem):
uploaded_file = st.file_uploader(...)  # In Memory!
process_file(uploaded_file)  # Im Request-Context

# NiceGUI (Lösung):
@ui.page('/upload')
def upload_page():
    uploaded_file = ui.upload(...).on('upload', save_to_disk)  # Sofort auf Disk
    ui.timer(1.0, check_background_job)  # Status-Updates
```

### Background-Job Pattern:

```python
# NiceGUI ermöglicht echte Background-Jobs
import asyncio
from nicegui import ui

async def process_daf_file(file_path):
    # Läuft in separatem Prozess
    # Kein UI-Block
    result = convert_daf_to_fcs(file_path)
    return result

def start_processing(file_path):
    asyncio.create_task(process_daf_file(file_path))
    ui.notify("Processing started in background")
```

## Fazit

**Für euer konkretes Szenario:**

1. **NiceGUI** ist die beste Wahl für Migration
   - Stabiler als Streamlit
   - Ähnliche API
   - Kann große Files handhaben
   - Background-Jobs möglich

2. **Dash** wenn Production-Stabilität kritisch ist
   - Etablierter Standard
   - Aber höherer Entwicklungsaufwand

3. **PySide6** wenn Desktop-App gewünscht
   - Maximal stabil
   - Aber komplett andere Architektur

4. **Streamlit** nur für kleine Demos
   - Nicht für 100-500 MB Files geeignet
   - Fundamentale Architektur-Probleme

## Nächste Schritte

1. ✅ Streamlit stabilisieren (bereits gemacht)
2. ⏳ NiceGUI Prototype entwickeln
3. ⏳ Vergleichstest durchführen
4. ⏳ Entscheidung treffen

