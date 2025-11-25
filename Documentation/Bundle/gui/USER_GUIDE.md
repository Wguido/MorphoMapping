# MorphoMapping GUI - Benutzerhandbuch

## 📖 Einführung

Willkommen beim MorphoMapping GUI! Diese Anleitung führt Sie Schritt für Schritt durch die Verwendung der Anwendung.

### Was ist MorphoMapping GUI?

MorphoMapping GUI ist eine Desktop-Anwendung zur Analyse von ImageStream-Daten (.daf Dateien). Sie ermöglicht:

- Dimensionsreduktion (DensMAP, UMAP, t-SNE)
- Clustering (KMeans, GMM, HDBSCAN)
- Visualisierung und Export von Ergebnissen
- Feature-Analyse und Cluster-Statistiken

### Systemanforderungen

- Python 3.10 oder 3.11
- R 4.0 oder neuer
- 8 GB RAM (16 GB empfohlen)
- macOS 10.15+, Windows 10+, oder Linux

---

## 🚀 Erste Schritte

### 1. Installation

Folgen Sie der **[INSTALLATION.md](INSTALLATION.md)** Anleitung, um die GUI zu installieren.

### 2. GUI starten

```bash
conda activate morphomapping
cd Documentation/Bundle/gui
python app_pyside6.py
```

Ein Fenster sollte sich öffnen.

---

## 📋 Schritt-für-Schritt Workflow

### Schritt 1: Projekt Setup

#### Run-ID setzen

1. Finden Sie das Feld **"Run-ID"** oben links
2. Geben Sie eine eindeutige ID ein (z.B. `experiment_2025_01_25`)
3. Die Run-ID wird für alle Ausgabedateien verwendet

#### Projektverzeichnis wählen (Optional)

1. Klicken Sie auf **"📁 Choose Project Directory"**
2. Wählen Sie einen Ordner, in dem Ihre Ergebnisse gespeichert werden sollen
3. Standard: `bundle_runs/` im MorphoMapping Ordner

**Tipp:** Verwenden Sie aussagekräftige Run-IDs wie `neutrophils_day1` oder `experiment_control_vs_treated`.

---

### Schritt 2: DAF-Dateien laden

#### Option A: Dateiauswahl-Dialog

1. Klicken Sie auf **"📁 Select DAF Files"**
2. Wählen Sie eine oder mehrere .daf Dateien aus
3. Klicken Sie auf "Öffnen"

#### Option B: Drag & Drop

1. Ziehen Sie .daf Dateien direkt in das **"📁 Drop DAF files here"** Feld
2. Die Dateien werden automatisch erkannt

#### Was passiert?

- Die .daf Dateien werden zu .fcs Dateien konvertiert
- Ein Fortschrittsbalken zeigt den Status
- Nach der Konvertierung erscheinen die Dateien in der Status-Übersicht

**Hinweis:** Große Dateien (100-500 MB) können einige Minuten dauern. Bitte haben Sie Geduld.

---

### Schritt 3: Metadaten eingeben

#### Option A: Manuelle Eingabe (Links)

1. Die Tabelle zeigt automatisch die geladenen Dateien
2. **file_name**: Wird automatisch ausgefüllt
3. **sample_id**: Wird automatisch nummeriert (`sample_1`, `sample_2`, etc.)
4. **group**: Geben Sie Ihre Gruppen ein (z.B. `control`, `treated`)
5. **replicate**: Geben Sie Replikat-Nummern ein (z.B. `1`, `2`, `3`)

**Tipp:** Sie können die Tabelle scrollen, wenn Sie viele Dateien haben.

#### Option B: CSV/Excel Upload (Rechts)

1. Klicken Sie auf **"📤 Upload Metadata"**
2. Wählen Sie eine CSV oder Excel-Datei
3. Die Metadaten werden automatisch geladen

**Erforderliche Spalten:**
- `file_name`: Name der .daf/.fcs Datei (ohne Endung)
- `sample_id`: Eindeutige Sample-ID (optional, wird auto-generiert)
- `group`: Experimentelle Gruppe (z.B. `control`, `treated`)
- `replicate`: Replikat-Nummer (optional)

#### Metadaten speichern

1. Klicken Sie auf **"💾 Save Metadata"**
2. Der Status ändert sich zu "✅ Saved"

**WICHTIG:** Speichern Sie die Metadaten, bevor Sie mit der Analyse beginnen!

---

### Schritt 4: Features auswählen

#### Features einbeziehen (Include)

1. Scrollen Sie zur Sektion **"4️⃣ Features & Gates Selection"**
2. Im **"Features to Include"** Bereich sehen Sie blaue Chips
3. Klicken Sie auf einen Chip, um ihn zu entfernen (wird zu "Exclude")
4. Klicken Sie erneut, um ihn wieder hinzuzufügen

**Tipp:** Die ersten 10 Features sind sichtbar. Klicken Sie auf **"▶ Show All"**, um alle zu sehen.

#### Features ausschließen (Exclude)

1. Im **"Features to Exclude"** Bereich sehen Sie rote Chips
2. Klicken Sie auf einen Chip, um ihn zu entfernen (wird zu "Include")
3. Klicken Sie erneut, um ihn wieder auszuschließen

**Hinweis:** Features, die nicht in allen Dateien vorhanden sind, werden automatisch ausgeschlossen.

#### Population/Gate auswählen

1. Wählen Sie eine Population aus dem Dropdown-Menü
2. Nur Zellen aus dieser Population werden analysiert

**Tipp:** Wählen Sie eine Population, die in allen Dateien vorhanden ist.

---

### Schritt 5: Dimensionsreduktion

#### Methode wählen

1. Scrollen Sie zur Sektion **"5️⃣ Dimensionality Reduction"**
2. Wählen Sie eine Methode:
   - **DensMAP** (Standard, empfohlen)
   - **UMAP**
   - **t-SNE**

#### Parameter anpassen

Je nach Methode sehen Sie verschiedene Slider:

**DensMAP:**
- **Dens Lambda**: Dichte-Regularisierung (Standard: 2.0)
- **N Neighbors**: Anzahl Nachbarn (Standard: 30)
- **Min Dist**: Minimale Distanz (Standard: 0.1)

**UMAP:**
- **N Neighbors**: Anzahl Nachbarn (Standard: 30)
- **Min Dist**: Minimale Distanz (Standard: 0.1)

**t-SNE:**
- **Perplexity**: Perplexity-Parameter (Standard: 30.0)

#### Sampling (Optional)

- **Max cells per sample**: Begrenzt die Anzahl analysierter Zellen pro Probe
- Nützlich für sehr große Datensätze
- 0 = Alle Zellen analysieren

#### Analyse starten

1. Klicken Sie auf **"▶ Run Analysis"**
2. Ein Fortschrittsbalken erscheint
3. Die Analyse kann einige Minuten dauern (abhängig von Datenmenge)

**Nach Abschluss:**
- Ein Plot erscheint in der **"6️⃣ Visualization"** Sektion
- Der **"📊 Download Top10 Features"** Button wird aktiv
- Die **"7️⃣ Clustering"** Sektion wird sichtbar

---

### Schritt 6: Visualisierung

#### Farbcodierung ändern

1. In der **"6️⃣ Visualization"** Sektion finden Sie das Dropdown **"Color by"**
2. Wählen Sie eine Option:
   - `sample_id`: Nach Probe färben
   - `group`: Nach Gruppe färben
   - `replicate`: Nach Replikat färben
   - Andere Metadaten-Spalten

**Tipp:** Der Plot aktualisiert sich automatisch, ohne Neuberechnung!

#### Achsen-Limits anpassen

1. Geben Sie Werte in die Felder ein:
   - **X Min / X Max**: X-Achse Limits
   - **Y Min / Y Max**: Y-Achse Limits
2. Klicken Sie auf **"Apply Limits"**
3. Klicken Sie auf **"Reset"**, um zurückzusetzen

**Verwendung:** Nützlich, um Ausreißer auszublenden oder bestimmte Bereiche zu fokussieren.

#### Zellen hervorheben

1. Geben Sie Zell-Indizes in das Feld **"Cell Indices"** ein (komma-separiert, z.B. `1, 5, 10`)
2. Klicken Sie auf **"✨ Highlight"**
3. Die Zellen werden als rote Sterne markiert

**Tipp:** Sie können mehrere Zellen gleichzeitig hervorheben.

#### Plot exportieren

1. Klicken Sie auf **"📥 Export PNG"** oder **"📥 Export PDF"**
2. Wählen Sie einen Speicherort
3. Die Datei wird mit 300 DPI gespeichert

---

### Schritt 7: Clustering

#### Algorithmus wählen

1. Scrollen Sie zur Sektion **"7️⃣ Clustering"**
2. Wählen Sie einen Algorithmus:
   - **KMeans**: Exakte Anzahl Cluster (Standard: 10)
   - **Gaussian Mixture Models (GMM)**: Probabilistisches Clustering
   - **HDBSCAN**: Dichte-basiertes Clustering

#### Parameter anpassen

Die Parameter ändern sich je nach Algorithmus:

**KMeans:**
- **N Clusters**: Anzahl Cluster (Standard: 10)
- **📊 Download Elbow Plot**: Zeigt optimale Cluster-Anzahl

**GMM:**
- **N Clusters**: Anzahl Cluster (Standard: 10)
- **Covariance Type**: Kovarianz-Typ (Standard: `full`)

**HDBSCAN:**
- **Min Cluster Size**: Minimale Clustergröße (Standard: automatisch)
- **Min Samples**: Minimale Samples (Standard: 10)

#### Clustering starten

1. Klicken Sie auf **"▶ Run Clustering"**
2. Ein Fortschrittsbalken erscheint
3. Nach Abschluss erscheint:
   - Ein Cluster-Plot
   - Eine Cluster-Statistik-Tabelle
   - Export-Buttons

#### Cluster-Statistiken

Die Tabelle zeigt:
- **Cluster**: Cluster-ID
- **Size**: Anzahl Zellen im Cluster
- **Percentage**: Prozentuale Verteilung
- **Sample Distribution**: Verteilung über Samples

#### Cluster-Plot exportieren

1. Klicken Sie auf **"📥 Export PNG"** oder **"📥 Export PDF"**
2. Die Achsen-Limits werden übernommen

#### Cluster-Statistiken exportieren

1. Klicken Sie auf **"📊 Export Bar Chart"**
2. Erstellt einen gestapelten Bar Chart nach Groups
3. Speichert als PNG

**Hinweis:** Erfordert eine `group` Spalte in den Metadaten!

---

### Schritt 8: Erweiterte Analyse

#### Top 3 Features pro Cluster

1. Nach dem Clustering finden Sie den Button **"🔍 Top 3 Features per Cluster"**
2. Klicken Sie darauf
3. Eine CSV-Datei wird erstellt mit:
   - Cluster-ID
   - Top 1, 2, 3 Features und deren Werte

**Speicherort:** `bundle_runs/run_YYYYMMDD_HHMMSS/results/top3_features_per_cluster.csv`

#### Cluster-Feature Heatmap

1. Klicken Sie auf **"🔥 Cluster-Feature Heatmap"**
2. Erstellt eine Heatmap mit:
   - **Zeilen**: Features
   - **Spalten**: Cluster
   - **Werte**: Row-wise Z-Score
3. Speichert:
   - PNG: `cluster_feature_heatmap.png`
   - CSV: `cluster_feature_heatmap_data.csv`

**Verwendung:** Identifiziert charakteristische Features pro Cluster.

#### Top 10 Features

1. Nach der Dimensionsreduktion finden Sie **"📊 Download Top10 Features"**
2. Klicken Sie darauf
3. Berechnet die wichtigsten Features für X- und Y-Dimensionen
4. Speichert:
   - CSV: `top10_features.csv`
   - Plots: `top10_features_x_dim.png`, `top10_features_y_dim.png`

**Dauer:** Kann einige Minuten dauern für große Datensätze.

---

## 💡 Tipps & Tricks

### Performance

- **Sampling verwenden**: Bei sehr großen Datensätzen (>100.000 Zellen) verwenden Sie "Max cells per sample"
- **Features reduzieren**: Weniger Features = schnellere Berechnung
- **Population wählen**: Analysieren Sie nur relevante Populationen

### Datenqualität

- **Metadaten prüfen**: Stellen Sie sicher, dass `group` korrekt ausgefüllt ist
- **Dateinamen konsistent**: `file_name` in Metadaten muss exakt mit Dateinamen übereinstimmen
- **Features prüfen**: Stellen Sie sicher, dass alle wichtigen Features in allen Dateien vorhanden sind

### Workflow-Optimierung

1. **Testen Sie zuerst mit wenigen Dateien** (2-3 Dateien)
2. **Verwenden Sie Sampling** für erste Tests
3. **Speichern Sie Metadaten** bevor Sie analysieren
4. **Exportieren Sie Ergebnisse** regelmäßig

---

## ❓ Häufige Fragen (FAQ)

### Q: Die GUI startet nicht

**A:** Überprüfen Sie:
1. Python-Version: `python --version` (sollte 3.10 oder 3.11 sein)
2. Environment aktiviert: `conda activate morphomapping`
3. Dependencies installiert: `pip list | grep PySide6`

### Q: "No module named 'morphomapping'"

**A:** Installieren Sie das Paket:
```bash
cd /Pfad/zum/MorphoMapping
pip install -e ./
```

### Q: Metadaten werden nicht übernommen

**A:** Überprüfen Sie:
1. `file_name` in Metadaten entspricht exakt dem Dateinamen (ohne .fcs)
2. Metadaten wurden gespeichert (Status zeigt "✅ Saved")
3. Analyse wurde nach dem Speichern gestartet

### Q: Plot ist leer oder grau

**A:** Überprüfen Sie:
1. `sample_id` Spalte existiert und hat Werte
2. Metadaten wurden korrekt mit Analyse-Daten verknüpft
3. Versuchen Sie, "Color by" zu ändern

### Q: Clustering zeigt keine Ergebnisse

**A:** Überprüfen Sie:
1. Dimensionsreduktion wurde erfolgreich abgeschlossen
2. Cluster-Algorithmus wurde ausgewählt
3. Parameter sind sinnvoll (z.B. nicht zu viele Cluster für kleine Datensätze)

### Q: Export funktioniert nicht

**A:** Überprüfen Sie:
1. Ergebnisse wurden berechnet (Plot ist sichtbar)
2. Schreibrechte im Ausgabeordner
3. Genug Festplattenspeicher

---

## 🐛 Fehlerbehebung

### "Analysis failed" Fehler

1. Überprüfen Sie die Konsole/Terminal für detaillierte Fehlermeldungen
2. Stellen Sie sicher, dass alle Dependencies installiert sind
3. Überprüfen Sie, ob R installiert ist: `Rscript --version`

### GUI friert ein

1. Warten Sie - große Dateien können lange dauern
2. Überprüfen Sie den Fortschrittsbalken
3. Bei Bedarf neu starten (Fortschritt geht verloren)

### Dateien werden nicht konvertiert

1. Überprüfen Sie, ob R installiert ist
2. Überprüfen Sie, ob Rscript im PATH ist
3. Überprüfen Sie die Dateien - sind sie korrupt?

---

## 📞 Support

Bei Problemen:

1. **Lesen Sie diese Anleitung** noch einmal durch
2. **Überprüfen Sie die [INSTALLATION.md](INSTALLATION.md)** für Installationsprobleme
3. **Erstellen Sie ein GitHub Issue**: https://github.com/Wguido/MorphoMapping/issues
   - Beschreiben Sie das Problem
   - Fügen Sie Fehlermeldungen hinzu
   - Geben Sie Systeminformationen an

---

**Letzte Aktualisierung:** 2025-01-25

