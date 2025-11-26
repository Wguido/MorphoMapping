"""
NiceGUI Prototype für MorphoMapping GUI

Dies ist ein Prototype zur Evaluation von NiceGUI als Streamlit-Ersatz.
Fokus: Stabilität bei großen .daf Files (100-500 MB).

Installation:
    pip install nicegui

Start:
    python app_nicegui_prototype.py
"""

from pathlib import Path
from typing import Dict, List, Optional
import asyncio
import subprocess
from datetime import datetime

from nicegui import ui, app
import pandas as pd
import numpy as np

# Konfiguration
BUNDLE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BUNDLE_ROOT.parents[2]
R_SCRIPT = PROJECT_ROOT / "R" / "daf_to_fcs_cli.R"

# Session State (ähnlich Streamlit, aber stabiler)
session_state = {
    "run_id": None,
    "project_dir": str(PROJECT_ROOT),
    "uploaded_files": [],
    "converted_files": [],
    "processing_status": {},
}

# Background-Job Queue
background_jobs = {}


def get_run_paths(project_dir: Path, run_id: str) -> Dict[str, Path]:
    """Erstelle Pfad-Struktur für einen Run."""
    base = Path(project_dir) / "runs" / run_id
    return {
        "base": base,
        "raw_daf": base / "raw_daf",
        "fcs": base / "fcs",
        "results": base / "results",
        "metadata": base / "metadata",
        "csv_cache": base / "csv_cache",
    }


async def convert_daf_to_fcs_background(daf_file: Path, fcs_file: Path, job_id: str):
    """
    Konvertiere DAF zu FCS im Background.
    Läuft asynchron und blockiert nicht die UI.
    """
    try:
        session_state["processing_status"][job_id] = {
            "status": "processing",
            "progress": 0,
            "message": f"Converting {daf_file.name}..."
        }
        
        # R-Script aufrufen
        command = ["Rscript", "--vanilla", "--slave", str(R_SCRIPT), str(daf_file), str(fcs_file)]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            session_state["processing_status"][job_id] = {
                "status": "completed",
                "progress": 100,
                "message": f"✅ {daf_file.name} converted successfully"
            }
            session_state["converted_files"].append(fcs_file)
        else:
            session_state["processing_status"][job_id] = {
                "status": "error",
                "progress": 0,
                "message": f"❌ Error: {result.stderr}"
            }
    except Exception as e:
        session_state["processing_status"][job_id] = {
            "status": "error",
            "progress": 0,
            "message": f"❌ Exception: {str(e)}"
        }


@ui.page('/')
async def main_page():
    """Hauptseite der GUI."""
    
    # Header
    with ui.header().classes('items-center justify-between'):
        ui.label('MorphoMapping').classes('text-2xl font-bold')
        ui.label('NiceGUI Prototype').classes('text-sm text-gray-500')
    
    # Main Content
    with ui.column().classes('w-full max-w-6xl mx-auto p-4 gap-4'):
        
        # Project Setup
        with ui.card().classes('w-full'):
            ui.label('1️⃣ Project Setup').classes('text-xl font-bold mb-4')
            
            project_dir_input = ui.input(
                'Project Directory',
                value=session_state["project_dir"]
            ).classes('w-full')
            
            run_id_input = ui.input(
                'Run-ID',
                value=datetime.now().strftime("run_%Y%m%d_%H%M%S")
            ).classes('w-full')
            
            def update_run_id():
                session_state["run_id"] = run_id_input.value
                ui.notify(f"Run-ID set to: {run_id_input.value}")
            
            ui.button('Set Run-ID', on_click=update_run_id)
        
        # File Upload
        with ui.card().classes('w-full'):
            ui.label('2️⃣ Upload DAF Files').classes('text-xl font-bold mb-4')
            
            upload_area = ui.upload(
                label='Upload .daf files',
                auto_upload=True,
                max_file_size=500 * 1024 * 1024,  # 500 MB
                multiple=True,
                on_upload=lambda e: handle_file_upload(e, run_id_input.value)
            ).classes('w-full')
            
            # Status Display
            status_container = ui.column().classes('w-full mt-4')
        
        # File List
        with ui.card().classes('w-full'):
            ui.label('Uploaded Files').classes('text-lg font-bold mb-2')
            file_list = ui.column().classes('w-full')
            update_file_list(file_list)
        
        # Processing Status
        with ui.card().classes('w-full'):
            ui.label('Processing Status').classes('text-lg font-bold mb-2')
            processing_status = ui.column().classes('w-full')
            update_processing_status(processing_status)
        
        # Auto-Update Status (alle 2 Sekunden)
        ui.timer(2.0, lambda: update_processing_status(processing_status))
        ui.timer(2.0, lambda: update_file_list(file_list))


def handle_file_upload(event, run_id: str):
    """Handle file upload - speichere sofort auf Disk."""
    try:
        # Erstelle Run-Pfade
        paths = get_run_paths(Path(session_state["project_dir"]), run_id)
        paths["raw_daf"].mkdir(parents=True, exist_ok=True)
        
        # Speichere File sofort auf Disk (nicht in Memory!)
        file_path = paths["raw_daf"] / event.name
        file_path.write_bytes(event.content.read())
        
        session_state["uploaded_files"].append({
            "name": event.name,
            "path": file_path,
            "size": file_path.stat().st_size,
            "uploaded_at": datetime.now().isoformat()
        })
        
        # Starte Background-Conversion
        fcs_path = paths["fcs"] / f"{file_path.stem}.fcs"
        paths["fcs"].mkdir(parents=True, exist_ok=True)
        
        job_id = f"{run_id}_{event.name}"
        asyncio.create_task(convert_daf_to_fcs_background(file_path, fcs_path, job_id))
        background_jobs[job_id] = {
            "daf_file": file_path,
            "fcs_file": fcs_path,
            "started_at": datetime.now().isoformat()
        }
        
        ui.notify(f"✅ {event.name} uploaded and queued for conversion")
        
    except Exception as e:
        ui.notify(f"❌ Upload error: {str(e)}", type='negative')


def update_file_list(container: ui.column):
    """Update file list display."""
    container.clear()
    
    for file_info in session_state["uploaded_files"]:
        size_mb = file_info["size"] / (1024 * 1024)
        with container:
            ui.label(f"📄 {file_info['name']} ({size_mb:.1f} MB)").classes('text-sm')


def update_processing_status(container: ui.column):
    """Update processing status display."""
    container.clear()
    
    for job_id, status in session_state["processing_status"].items():
        with container:
            if status["status"] == "processing":
                ui.label(f"⏳ {status['message']}").classes('text-blue-600')
            elif status["status"] == "completed":
                ui.label(f"✅ {status['message']}").classes('text-green-600')
            elif status["status"] == "error":
                ui.label(f"❌ {status['message']}").classes('text-red-600')


# Start NiceGUI
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="MorphoMapping - NiceGUI Prototype",
        port=8502,  # Anderer Port als Streamlit
        show=True,
        reload=False,  # Wichtig für Stabilität bei großen Files
    )

