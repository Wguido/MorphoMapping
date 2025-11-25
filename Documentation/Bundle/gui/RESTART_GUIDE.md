# MorphoMapping GUI - Neustart-Anleitung

## Problem: Streamlit ist abgestürzt - wie starte ich neu ohne Datenverlust?

### Automatische Session-Wiederherstellung

Die GUI speichert automatisch den Session State nach wichtigen Operationen:
- Nach Dimensionality Reduction
- Nach Clustering
- Nach Metadaten-Speicherung

### Neustart-Schritte

1. **Streamlit neu starten:**
   ```bash
   cd /Users/labor/Documents/Projects/MorphoMapping/upstream_repo/MorphoMapping
   conda activate morphomapping
   streamlit run Documentation/Bundle/gui/app.py
   ```

2. **Session wird automatisch wiederhergestellt:**
   - Beim Start prüft die GUI, ob eine gespeicherte Session existiert
   - Wenn ja, wird eine Info-Meldung angezeigt: "🔄 Restored session from..."
   - Alle wichtigen Daten werden automatisch wiederhergestellt:
     - Run-ID
     - Embedding DataFrame (DR-Ergebnisse)
     - Cluster Labels
     - Features
     - Metadaten
     - Einstellungen

3. **Falls keine automatische Wiederherstellung:**
   - Die Session-Dateien befinden sich in: `Documentation/Bundle/gui/session_cache/`
   - Prüfen Sie, ob die Dateien existieren:
     ```bash
     ls -lh Documentation/Bundle/gui/session_cache/
     ```

### Manuelle Session-Verwaltung

#### Session löschen (wenn Probleme auftreten):
```python
# In der Python-Konsole oder als temporäre Funktion in app.py:
from session_state_manager import SessionStateManager
from pathlib import Path

manager = SessionStateManager(Path("Documentation/Bundle/gui/session_cache"))
manager.clear_session_state()
```

#### Session-Info anzeigen:
```python
from session_state_manager import SessionStateManager
from pathlib import Path

manager = SessionStateManager(Path("Documentation/Bundle/gui/session_cache"))
info = manager.get_session_info()
print(info)
```

## Warum stürzt Streamlit ab?

### Häufige Ursachen:

1. **Zu große Datenmengen:**
   - Mehr als 1 Million Zellen können zu Memory-Problemen führen
   - Lösung: Daten filtern oder Sampling verwenden

2. **Matplotlib Figure Leaks:**
   - Figures werden nicht richtig geschlossen
   - Lösung: `plt.close()` wird jetzt automatisch aufgerufen

3. **Browser-Überlastung:**
   - Zu viele interaktive Elemente
   - Lösung: Altair Charts wurden durch statische Matplotlib-Plots ersetzt

4. **Memory-Probleme:**
   - Zu viele große DataFrames im Session State
   - Lösung: Session State speichert nur wichtige, serialisierbare Daten

### Stabilitätsverbesserungen:

1. **Matplotlib Backend:**
   - Verwendet "Agg" Backend (non-interactive)
   - Figures werden explizit geschlossen

2. **Session State Management:**
   - Auto-Save nach wichtigen Operationen
   - Auto-Load beim Neustart
   - Große DataFrames werden nicht gespeichert (>1M rows)

3. **Error Handling:**
   - Try-finally Blöcke für Figure-Cleanup
   - Graceful degradation bei fehlenden Daten

## Stabilität testen

### Test-Skript ausführen:

```bash
cd /Users/labor/Documents/Projects/MorphoMapping/upstream_repo/MorphoMapping
conda activate morphomapping
python Documentation/Bundle/gui/test_stability.py
```

Das Skript testet:
- ✅ Alle Imports funktionieren
- ✅ Matplotlib Backend funktioniert
- ✅ Memory-Handling mit großen Datasets
- ✅ Streamlit kann starten

### Manuelle Tests:

1. **Kleine Datenmenge testen:**
   - 1-5 DAF-Dateien hochladen
   - DR durchführen
   - Clustering durchführen
   - Prüfen ob alles stabil läuft

2. **Größere Datenmenge testen:**
   - 10-20 DAF-Dateien hochladen
   - Prüfen ob Memory-Probleme auftreten

3. **Neustart-Test:**
   - DR durchführen
   - Streamlit stoppen (Ctrl+C)
   - Streamlit neu starten
   - Prüfen ob Session wiederhergestellt wird

## Tipps für Stabilität:

1. **Regelmäßig speichern:**
   - Metadaten regelmäßig speichern
   - Ergebnisse exportieren (PNG/PDF)

2. **Browser-Cache leeren:**
   - Bei Problemen: Browser-Cache leeren
   - Hard Refresh: Cmd+Shift+R (Mac) oder Ctrl+Shift+R (Linux)

3. **Streamlit Cache leeren:**
   ```bash
   streamlit cache clear
   ```

4. **Memory überwachen:**
   - Activity Monitor (Mac) oder htop (Linux) verwenden
   - Prüfen ob Python-Prozess zu viel Memory verwendet

## Bei anhaltenden Problemen:

1. **Logs prüfen:**
   - Terminal-Output für Fehlermeldungen
   - Browser-Konsole (F12) für JavaScript-Fehler

2. **Session-Cache löschen:**
   ```bash
   rm -rf Documentation/Bundle/gui/session_cache/*
   ```

3. **Streamlit neu installieren:**
   ```bash
   pip install --upgrade streamlit
   ```

4. **Minimal-Test:**
   - Nur eine DAF-Datei hochladen
   - Minimal Features auswählen
   - Prüfen ob grundlegende Funktionalität funktioniert

