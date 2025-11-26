# Framework Migration Guide - Streamlit zu NiceGUI/Dash

## Problem-Analyse

**Aktuelle Situation:**
- Streamlit wird instabil bei 100-500 MB .daf Files
- WebSocket-Verbindungen brechen ab
- UI wird leer/blank
- Session-Verlust bei Reruns

**Root Cause:**
- Streamlit's Rerun-Modell + große Dateien in Memory = Instabilität
- Upload + Processing im selben Request-Context
- Keine echte Trennung zwischen UI und Backend

## Empfohlene Lösung: NiceGUI

### Warum NiceGUI?

1. **FastAPI-Foundation** → Normale HTTP-Requests, keine Reruns
2. **Stabiler bei großen Dateien** → Upload als normaler HTTP-Request
3. **Background Jobs** → Processing getrennt von UI
4. **Python-only** → Keine JavaScript-Kenntnisse nötig
5. **Streamlit-ähnliches Feeling** → Einfache Migration

### Migration-Plan

#### Phase 1: Upload/Processing trennen (Schnell-Win)

**Aktuell (Streamlit):**
```python
uploaded_dafs = st.file_uploader(...)  # Lädt alles in Memory
if uploaded_dafs:
    saved = save_uploaded_files(...)  # Synchron
    converted = convert_daf_files(...)  # Blockiert UI
    st.rerun()  # Triggert Neustart
```

**Verbessert (Streamlit mit Background):**
```python
# Upload direkt zu Disk, nicht in Memory
# Processing in Background-Thread
# Status-Polling statt Rerun
```

#### Phase 2: NiceGUI Migration (Langfristig)

**NiceGUI Struktur:**
```python
from nicegui import ui, app
import asyncio
from pathlib import Path

@ui.page('/')
async def main():
    # Upload direkt zu Disk
    async def handle_upload(e):
        file_path = Path(f"uploads/{e.name}")
        file_path.write_bytes(await e.read())
        # Start Background Job
        asyncio.create_task(process_file(file_path))
    
    ui.upload(on_upload=handle_upload, 
              max_file_size=500_000_000)  # 500 MB
    
    # Status Display (wird automatisch aktualisiert)
    status_label = ui.label("Ready")
    
    # Background Processing
    async def process_file(file_path: Path):
        status_label.text = "Processing..."
        # Lange Operation hier
        await convert_daf_file(file_path)
        status_label.text = "Done"
```

### Vergleich: Streamlit vs NiceGUI

| Feature | Streamlit | NiceGUI |
|---------|-----------|---------|
| **Upload große Dateien** | ❌ Instabil | ✅ Stabil |
| **Background Jobs** | ❌ Schwierig | ✅ Einfach |
| **WebSocket** | ❌ Instabil | ✅ Stabil |
| **Reruns** | ❌ Immer | ✅ Nur bei Bedarf |
| **Memory** | ❌ Alles in Session | ✅ Getrennt |
| **Learning Curve** | ✅ Sehr einfach | ✅ Einfach |
| **Production Ready** | ⚠️ Begrenzt | ✅ Ja |

## Alternative: Dash (wenn Production wichtig)

**Dash Vorteile:**
- Sehr etabliert (Plotly)
- Production-ready (Gunicorn, Docker)
- Sehr stabil
- Große Community

**Dash Nachteile:**
- Mehr Boilerplate-Code
- React-Kenntnisse für Custom-UI hilfreich
- Weniger "Pythonic" als NiceGUI

## Alternative: PySide6 (Desktop-App)

**PySide6 Vorteile:**
- Maximale Stabilität (keine Browser-Probleme)
- Direkter Dateisystem-Zugriff (kein Upload!)
- Native Performance

**PySide6 Nachteile:**
- Höherer Entwicklungsaufwand
- Keine Web-Version
- Installation pro Rechner nötig

## Empfehlung

### Kurzfristig (1-2 Wochen):
**Streamlit mit Background-Processing**
- Upload direkt zu Disk
- Processing in Thread/Process
- Status-Polling statt Reruns
- Kann 80% der Probleme lösen

### Mittelfristig (1-2 Monate):
**NiceGUI Migration**
- Vollständige Neuimplementierung
- Stabiler, moderner, zukunftssicher
- Ähnliche Code-Struktur wie Streamlit

### Langfristig (wenn Desktop-App gewünscht):
**PySide6**
- Nur wenn Web-Version nicht nötig
- Maximale Stabilität für interne Tools

## Konkrete nächste Schritte

1. **Sofort:** Background-Processing in Streamlit implementieren
2. **Diese Woche:** NiceGUI Prototype erstellen
3. **Nächster Monat:** Entscheidung: NiceGUI oder Dash
4. **Migration:** Schrittweise, Feature für Feature

