#!/bin/bash
# Script zum Erstellen von GitHub Issues
# Verwendung: ./create_github_issues.sh

# GitHub Repository
REPO="Wguido/MorphoMapping"
GITHUB_API="https://api.github.com/repos/${REPO}/issues"

# Issue 1: Cluster-Feature Heatmap
echo "Creating Issue 1: Cluster-Feature Heatmap..."
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  "${GITHUB_API}" \
  -d '{
    "title": "Cluster-Feature Heatmap: Falsche Achsen und keine sichtbaren Unterschiede",
    "body": "## Problem\n\nDie Cluster-Feature Heatmap wird generiert, aber:\n1. **Features sind auf X-Achse statt Y-Achse**: Features sollten vertikal (Y-Achse) angezeigt werden, Clusters horizontal (X-Achse)\n2. **Keine sichtbaren Unterschiede**: Die Heatmap zeigt keine klaren Cluster-Unterschiede, alle Werte scheinen ähnlich zu sein\n3. **Heatmap ist gelb**: Die Farbcodierung funktioniert nicht richtig - sollte rot-gelb-blau (RdYlBu_r) mit Z-Score sein\n\n## Erwartetes Verhalten\n\n- **Y-Achse**: Features (vertikal, lesbare Labels)\n- **X-Achse**: Clusters (horizontal, Cluster-IDs)\n- **Farbcodierung**: Klare Unterschiede zwischen Clusters (rot = hoher Z-Score, blau = niedriger Z-Score)\n- **Z-Score**: Row-wise Z-Score pro Feature über alle Clusters\n\n## Aktuelles Verhalten\n\n- Features erscheinen auf X-Achse (gequetscht)\n- Y-Achse zeigt nur \"Feature\" und \"0\"\n- Heatmap ist komplett gelb (keine Farbunterschiede)\n- Keine sichtbaren Cluster-Unterschiede\n\n## Technische Details\n\n- Funktion: `export_cluster_heatmap()` in `morphomapping_gui.py`\n- Datenstruktur: `heatmap_z` sollte Features als Index (rows) und Clusters als Spalten (columns) haben\n- Z-Score wird row-wise berechnet, aber Visualisierung zeigt keine Unterschiede\n\n## Schritte zur Reproduktion\n\n1. DAF-Dateien laden\n2. Metadaten eingeben\n3. Features auswählen\n4. Dimensionsreduktion durchführen\n5. Clustering durchführen\n6. \"🔥 Cluster-Feature Heatmap\" Button klicken\n\n## Dateien\n\n- `gui/morphomapping_gui.py` (Zeile ~3346-3630)\n\n## Labels\n\nbug, visualization, heatmap",
    "labels": ["bug", "visualization", "heatmap"]
  }'

echo ""
echo "Creating Issue 2: Top 10 Features..."
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  "${GITHUB_API}" \
  -d '{
    "title": "Top 10 Features Berechnung stürzt ab",
    "body": "## Problem\n\nDie Berechnung der Top 10 Features für X- und Y-Dimensionen:\n1. **Crash**: Die Berechnung stürzt häufig ab (`zsh: killed python`)\n2. **Fehlende Ergebnisse**: Top 10 Features werden nicht korrekt berechnet oder exportiert\n3. **Memory Issues**: Bei großen Datensätzen kommt es zu Memory-Problemen\n\n## Erwartetes Verhalten\n\n- **Berechnung**: Top 10 Features für X-Dimension und Y-Dimension werden berechnet\n- **Export**: CSV-Datei mit Features und Importance-Scores\n- **Plots**: Zwei PNG-Dateien (`top10_features_x_dim.png`, `top10_features_y_dim.png`)\n- **Stabilität**: Keine Crashes, auch bei großen Datensätzen\n\n## Aktuelles Verhalten\n\n- Berechnung startet, aber stürzt ab\n- Oder: Berechnung läuft, aber keine Ergebnisse\n- Memory-Fehler bei großen Datensätzen\n\n## Technische Details\n\n- Funktion: `download_top10_features()` in `morphomapping_gui.py`\n- Worker: `FeatureImportanceWorker` (QThread)\n- Library: `morphomapping.morphomapping.MM.feature_importance()`\n- Daten: Kombiniert Features + x/y Koordinaten\n\n## Schritte zur Reproduktion\n\n1. DAF-Dateien laden\n2. Metadaten eingeben\n3. Features auswählen\n4. Dimensionsreduktion durchführen\n5. \"📊 Download Top10 Features\" Button klicken\n\n## Dateien\n\n- `gui/morphomapping_gui.py` (Zeile ~2150-2250, `FeatureImportanceWorker`)\n\n## Labels\n\nbug, feature-importance, memory",
    "labels": ["bug", "feature-importance", "memory"]
  }'

echo ""
echo "Issues created successfully!"
echo ""
echo "HINWEIS: Ersetzen Sie 'YOUR_GITHUB_TOKEN' mit Ihrem GitHub Personal Access Token"
echo "Token erstellen: https://github.com/settings/tokens"

