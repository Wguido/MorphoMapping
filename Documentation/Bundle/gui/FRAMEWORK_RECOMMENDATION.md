# Framework-Empfehlung für MorphoMapping GUI

## Aktuelle Situation

NiceGUI zeigt trotz mehrfacher Fixes weiterhin Probleme:
- **TypeError mit bytes**: NiceGUI's File-Upload-API gibt bytes zurück, die nicht konsistent behandelt werden können
- **Verbindungsabbrüche**: WebSocket-Probleme bei großen Dateien (100-500 MB)
- **Instabilität**: Häufige Server-Fehler und Verbindungsverluste

## Empfehlung: **PySide6 (Qt für Python)**

### Warum PySide6?

1. **Maximale Stabilität**
   - Keine WebSocket-Abbrüche (kein Browser!)
   - Keine Seitenupdates, kein UI-Rerendering
   - Native Desktop-App = keine Netzwerk-Probleme

2. **Perfekt für große Dateien**
   - `.daf`-Files werden direkt vom Dateisystem gelesen (kein Upload!)
   - Keine Memory-Probleme durch Upload-Buffer
   - Native File-Dialog statt Web-Upload

3. **Ideal für interne Tools**
   - Keine Browser-Abhängigkeit
   - Bessere Performance bei großen Daten
   - Native Look & Feel

4. **Etabliert und stabil**
   - Qt ist seit Jahrzehnten bewährt
   - Große Community
   - Gute Dokumentation

### Migration-Plan

**Gute Nachricht**: Die Business-Logik in `core/` ist bereits framework-unabhängig!

```python
# Diese Module funktionieren OHNE Änderung:
- core/file_handling.py
- core/conversion.py
- core/metadata.py
- core/analysis.py
- core/visualization.py
```

**Nur der UI-Layer muss neu geschrieben werden:**

```python
# NiceGUI (aktuell)
ui.upload(...)
ui.button(...)
ui.select(...)

# PySide6 (neu)
QFileDialog.getOpenFileNames(...)
QPushButton(...)
QComboBox(...)
```

### Vorteile der Migration

1. **Keine Upload-Probleme**: Direkter File-Zugriff
2. **Keine bytes-Probleme**: Native Python-Pfade
3. **Bessere Performance**: Native Threading
4. **Stabiler**: Keine WebSocket/HTTP-Probleme

### Nachteile

1. **Mehr Code**: Desktop-Apps brauchen mehr UI-Code
2. **Lernkurve**: Qt/PySide6 hat eigene Konzepte
3. **Nur Desktop**: Keine Web-Version (aber das ist OK für interne Tools)

## Alternative: Dash

**Dash** wäre auch stabil, aber:
- Immer noch Web-basiert (Browser nötig)
- Immer noch Upload-Probleme möglich
- Komplexer als PySide6 für Desktop-Apps

**Dash macht Sinn, wenn:**
- Web-Zugriff wichtig ist
- Mehrere Nutzer gleichzeitig arbeiten sollen
- Remote-Zugriff benötigt wird

## Empfehlung

**Für MorphoMapping: PySide6**

- Interne Tool → Desktop-App ist perfekt
- Große Dateien → Direkter File-Zugriff ist essentiell
- Stabilität → Qt ist bewährt
- Business-Logic bleibt → Nur UI-Layer ändern

### Nächste Schritte

1. **Prototyp erstellen**: Einfache PySide6-Version mit File-Dialog
2. **UI-Komponenten portieren**: Schritt für Schritt von NiceGUI zu PySide6
3. **Testen**: Mit echten .daf-Files testen
4. **Produktiv**: NiceGUI-Version durch PySide6 ersetzen

### Code-Beispiel (PySide6)

```python
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton
from pathlib import Path

class MorphoMappingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MorphoMapping")
        
        # File-Dialog Button
        btn = QPushButton("Select DAF Files", self)
        btn.clicked.connect(self.select_files)
        
    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select DAF Files",
            "",
            "DAF Files (*.daf)"
        )
        
        # Direkter Zugriff - keine Upload-Probleme!
        for file_path in files:
            path = Path(file_path)  # Native Path, keine bytes!
            # Weiterverarbeitung...
```

## GitHub-Kompatibilität & Deployment

**✅ PySide6 ist perfekt mit GitHub kompatibel!**

- Reines Python → keine speziellen Git-Anforderungen
- Dependencies über `pip` → in `pyproject.toml` oder `requirements.txt`
- Keine Binaries im Repo → alles textbasiert

**Für Endanwender:**

1. **Einfachste Lösung**: Executable mit PyInstaller
   - Entwickler erstellt einmalig: `pyinstaller --windowed --onefile gui/app_pyside6.py`
   - Endanwender bekommt: `.app` (macOS) oder `.exe` (Windows)
   - **Keine Python-Installation nötig!** Einfach doppelklicken.

2. **Alternative**: Über pip (für technische Nutzer)
   ```bash
   pip install morphomapping[gui]
   morphomapping-gui
   ```

**Siehe**: `PYSIDE6_DEPLOYMENT.md` für Details.

## Fazit

**NiceGUI war ein guter Versuch**, aber für große Dateien und Desktop-Apps ist **PySide6 die bessere Wahl**.

Die Migration ist machbar, da die Business-Logik bereits getrennt ist.

**PySide6 ist sogar einfacher zu deployen als NiceGUI**, weil:
- Native Desktop-App (kein Server nötig)
- Einfache Executable-Erstellung
- Keine Browser-Abhängigkeit

