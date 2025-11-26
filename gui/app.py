"""
MorphoMapping GUI - NiceGUI Implementation

Stable GUI for processing large ImageStream .daf files (100-500 MB).
Uses NiceGUI for better stability with large files compared to Streamlit.

Installation:
    pip install nicegui pandas numpy matplotlib scikit-learn hdbscan umap-learn

Start:
    python app.py
"""

from __future__ import annotations

import asyncio
import datetime
import io
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nicegui import ui, app

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Import core business logic
from core import (
    BUNDLE_ROOT, PROJECT_ROOT, R_SCRIPT,
    DEFAULT_FEATURES, DEFAULT_METADATA_ROWS, COLOR_PALETTE,
    get_run_paths, get_file_counts,
    convert_daf_files,
    load_or_create_metadata,
    save_metadata as save_metadata_file,
    run_dimensionality_reduction,
    run_clustering as run_clustering_analysis,
    get_axis_labels, calculate_feature_importance,
)
from morphomapping.morphomapping import MM

# Session state (global, shared across all users - in production, use proper session management)
session_state: Dict = {
    "run_id": None,
    "project_dir": str(PROJECT_ROOT),
    "embedding_df": None,
    "cluster_labels": None,
    "features": [],
    "selected_population": None,
    "metadata_df": pd.DataFrame(),
    "stored_dim_reduction_method": None,
    "cluster_method": None,
    "cluster_param": 10,
    "uploaded_files": [],
    "processing_status": {},
    "highlighted_cells": [],
}

# Background job tracking
background_jobs: Dict[str, Dict] = {}

# ============================================================================
# Utility Functions for Robust Path Handling
# ============================================================================

def safe_str(value) -> str:
    """Safely convert any value to string, handling bytes."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except:
            return str(value)
    return str(value)

def safe_path(path) -> Path:
    """Safely create Path object, ensuring string input."""
    if isinstance(path, bytes):
        path = safe_str(path)
    elif not isinstance(path, (str, Path)):
        path = safe_str(path)
    return Path(str(path))


# ============================================================================
# Background Processing Functions
# ============================================================================

async def convert_daf_to_fcs_background(daf_file: Path, fcs_file: Path, job_id: str):
    """Convert DAF to FCS in background."""
    try:
        session_state["processing_status"][job_id] = {
            "status": "processing",
            "message": f"Converting {str(daf_file.name)}...",
        }
        
        command = ["Rscript", "--vanilla", "--slave", str(R_SCRIPT), str(daf_file), str(fcs_file)]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            session_state["processing_status"][job_id] = {
                "status": "completed",
                "message": f"✅ {str(daf_file.name)} converted",
            }
        else:
            session_state["processing_status"][job_id] = {
                "status": "error",
                "message": f"❌ Error: {result.stderr[:100]}",
            }
    except Exception as e:
        session_state["processing_status"][job_id] = {
            "status": "error",
            "message": f"❌ Exception: {str(e)[:100]}",
        }


async def run_analysis_background(
    paths: Dict[str, Path],
    features: List[str],
    method: str,
    method_params: Dict,
    population: Optional[str],
    job_id: str,
):
    """Run dimensionality reduction in background."""
    try:
        session_state["processing_status"][job_id] = {
            "status": "processing",
            "message": f"Running {method}...",
        }
        
        # Ensure all paths are safe Path objects
        safe_paths = {k: safe_path(v) for k, v in paths.items()}
        
        embedding_df, info = run_dimensionality_reduction(
            safe_paths["fcs"],
            features,
            safe_paths,
            method,
            method_params.get("dens_lambda", 2.0),
            method_params.get("n_neighbors", 30),
            method_params.get("min_dist", 0.1),
            method_params.get("perplexity", 30.0),
            population,
            session_state,
        )
        
        # Merge with metadata
        metadata_path = safe_paths["metadata"] / "sample_sheet.csv"
        metadata_df = load_or_create_metadata(metadata_path)
        
        if "file_name" in metadata_df.columns and "sample_id" not in metadata_df.columns:
            metadata_df["sample_id"] = metadata_df["file_name"]
        
        embedding_df["sample_id"] = embedding_df["sample_id"].astype(str)
        
        if not metadata_df.empty and "sample_id" in metadata_df.columns:
            metadata_df["sample_id"] = metadata_df["sample_id"].astype(str)
            embedding_with_meta = embedding_df.merge(metadata_df, on="sample_id", how="left")
        else:
            embedding_with_meta = embedding_df.copy()
        
        embedding_with_meta["cell_index"] = range(len(embedding_with_meta))
        
        session_state["embedding_df"] = embedding_with_meta
        session_state["features"] = features
        session_state["metadata_df"] = metadata_df
        session_state["stored_dim_reduction_method"] = method
        session_state["selected_population"] = population
        session_state["processing_status"][job_id] = {
            "status": "completed",
            "message": f"✅ {method} completed",
        }
        
        # Save config
        run_config = {
            "project_dir": str(paths["base"].parent.parent),
            "run_id": session_state["run_id"],
            "features": features,
            "population": population,
            "method": method,
            "parameters": method_params,
        }
        (safe_paths["base"] / "run_config.json").write_text(json.dumps(run_config, indent=2))
        
        # Notify completion
        ui.notify(f"✅ {method} analysis completed successfully!", type='positive')
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"Analysis error: {error_msg}\n{error_trace}")  # Debug output
        
        session_state["processing_status"][job_id] = {
            "status": "error",
            "message": f"❌ Error: {error_msg[:200]}",
        }
        
        # Notify error
        ui.notify(f"❌ Analysis failed: {error_msg[:100]}", type='negative')


# ============================================================================
# UI Helper Functions
# ============================================================================

def create_matplotlib_plot(data: pd.DataFrame, x_col: str, y_col: str, color_col: str, 
                          title: str, x_label: str, y_label: str, highlighted: List[int] = None) -> bytes:
    """Create matplotlib plot and return as PNG bytes."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique values for coloring
    unique_vals = sorted([v for v in data[color_col].unique() if pd.notna(v)])
    
    # Plot all points
    for i, val in enumerate(unique_vals):
        subset = data[data[color_col] == val]
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        ax.scatter(subset[x_col], subset[y_col], label=str(val), alpha=0.7, s=40, color=color)
    
    # Add highlighted cells
    if highlighted and len(data[data["cell_index"].isin(highlighted)]) > 0:
        highlighted_subset = data[data["cell_index"].isin(highlighted)]
        ax.scatter(highlighted_subset[x_col], highlighted_subset[y_col],
                  s=200, c="red", marker="o", edgecolors="black",
                  linewidths=2, alpha=1.0, zorder=10)
        # Add labels
        for idx, row in highlighted_subset.iterrows():
            if "cell_index" in row:
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                x_offset = x_range * 0.02
                y_offset = y_range * 0.02
                ax.annotate(str(row["cell_index"]),
                          (row[x_col], row[y_col]),
                          xytext=(x_offset, y_offset),
                          textcoords="offset points",
                          fontsize=14, fontweight="bold",
                          color="red", ha="left", va="bottom",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                   edgecolor="black", linewidth=2),
                          arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=10, width=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.grid(False)
    if len(unique_vals) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    
    # Convert to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


# ============================================================================
# Main UI Page
# ============================================================================

@ui.page('/')
async def main_page():
    """Main application page."""
    
    # Heartbeat to keep connection alive
    async def heartbeat():
        """Send periodic heartbeat to keep connection alive."""
        try:
            # Just update a hidden element to keep connection active
            pass
        except:
            pass
    
    ui.timer(30.0, heartbeat)  # Heartbeat every 30 seconds
    
    # Header with logo
    with ui.header().classes('items-center justify-between bg-blue-50 p-4'):
        logo_path = BUNDLE_ROOT / "assets" / "logo.png"
        if logo_path.exists():
            with ui.row().classes('items-center gap-4'):
                ui.image(str(logo_path)).classes('w-24')
                with ui.column():
                    ui.label('MorphoMapping').classes('text-3xl font-bold text-blue-600')
                    ui.label('Interactive Analysis for Imaging Flow Cytometry').classes('text-sm text-gray-600')
        else:
            ui.label('MorphoMapping').classes('text-3xl font-bold text-blue-600')
    
    # Main content container
    with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-6'):
        
        # ====================================================================
        # 1. Project Setup & Status
        # ====================================================================
        with ui.row().classes('w-full gap-4'):
            # Left: Project Setup
            with ui.card().classes('flex-1'):
                ui.label('1️⃣ Project Setup').classes('text-xl font-bold mb-4')
                
                # Initialize run_id if not set
                if not session_state.get("run_id"):
                    session_state["run_id"] = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
                
                project_dir_input = ui.input(
                    'Project Directory',
                    value=session_state["project_dir"],
                    placeholder=str(PROJECT_ROOT)
                ).classes('w-full mb-2')
                
                run_id_input = ui.input(
                    'Run-ID',
                    value=session_state["run_id"],
                    placeholder=session_state["run_id"]
                ).classes('w-full mb-2')
                
                def update_project():
                    session_state["project_dir"] = project_dir_input.value
                    session_state["run_id"] = run_id_input.value
                    ui.notify(f"✅ Project set: {run_id_input.value}", type='positive')
                
                ui.button('💾 Set Project', on_click=update_project, color='primary').classes('w-full')
            
            # Right: Status
            with ui.card().classes('flex-1'):
                ui.label('Status').classes('text-xl font-bold mb-4')
                
                status_container = ui.column().classes('w-full gap-2')
                
                def update_status():
                    status_container.clear()
                    with status_container:
                        # Always show Run-ID
                        run_id = session_state.get("run_id", "Not set")
                        ui.label(f"🆔 Run-ID: {run_id}").classes('text-sm font-bold')
                        
                        if session_state.get("run_id"):
                            project_dir = safe_str(session_state.get("project_dir", PROJECT_ROOT))
                            run_id = safe_str(session_state["run_id"])
                            paths = get_run_paths(safe_path(project_dir), run_id)
                            counts = get_file_counts(paths)
                            
                            ui.label(f"📁 DAF files: {counts['daf']}").classes('text-sm')
                            ui.label(f"📊 FCS files: {counts['fcs']}").classes('text-sm')
                            
                            # Metadata status
                            metadata_df = session_state.get("metadata_df")
                            if metadata_df is not None and len(metadata_df) > 0:
                                # Check if metadata has been edited (has non-empty values beyond file_name)
                                has_content = False
                                if "sample_id" in metadata_df.columns:
                                    has_content = any(metadata_df["sample_id"].astype(str).str.strip() != "")
                                if not has_content and "group" in metadata_df.columns:
                                    has_content = any(metadata_df["group"].astype(str).str.strip() != "")
                                
                                if has_content:
                                    ui.label("📋 Metadata: ✅ Saved").classes('text-sm text-green-600')
                                else:
                                    ui.label("📋 Metadata: ⚠️ Not edited").classes('text-sm text-yellow-600')
                            else:
                                ui.label("📋 Metadata: ❌ Not set").classes('text-sm text-red-600')
                            
                            # Features status
                            features_count = len(session_state.get("features", []))
                            if features_count > 0:
                                ui.label(f"📊 Features: {features_count} selected").classes('text-sm text-green-600')
                            else:
                                ui.label("📊 Features: Not selected").classes('text-sm text-gray-500')
                            
                            # Population status
                            pop = session_state.get("selected_population")
                            if pop:
                                ui.label(f"🔬 Population: {pop}").classes('text-sm')
                            else:
                                ui.label("🔬 Population: All events").classes('text-sm')
                            
                            # Analysis status
                            if session_state.get("embedding_df") is not None:
                                method = session_state.get("stored_dim_reduction_method", "Unknown")
                                ui.label(f"✅ Analysis: {method} completed").classes('text-sm text-green-600')
                            else:
                                ui.label("⏳ Analysis: Pending").classes('text-sm text-gray-500')
                            
                            # Clustering status
                            if session_state.get("cluster_labels") is not None:
                                n_clusters = len(set(session_state["cluster_labels"])) - (1 if -1 in session_state["cluster_labels"] else 0)
                                ui.label(f"🔬 Clusters: {n_clusters}").classes('text-sm text-green-600')
                
                update_status()
                ui.timer(5.0, update_status)  # Less frequent for stability
        
        # ====================================================================
        # 2. Upload DAF Files
        # ====================================================================
        with ui.card().classes('w-full'):
            ui.label('2️⃣ Upload DAF Files').classes('text-xl font-bold mb-4')
            
            async def handle_upload(e):
                try:
                    if not session_state.get("run_id"):
                        ui.notify("⚠️ Please set Run-ID first", type='warning')
                        return
                    
                    # NiceGUI upload event structure: e.file.name, e.file.read()
                    # For large files, NiceGUI uses LargeFileUpload which has async read() method
                    file = e.file
                    file_name = str(file.name) if file.name else "uploaded_file.daf"
                    file_content = await file.read()  # Async read() method
                    
                    # Ensure file_content is bytes
                    if isinstance(file_content, str):
                        file_content = file_content.encode('utf-8')
                    
                    # Ensure project_dir is str, not bytes - use safe_str
                    project_dir = safe_str(session_state.get("project_dir", PROJECT_ROOT))
                    run_id = safe_str(session_state.get("run_id", "default"))
                    
                    paths = get_run_paths(safe_path(project_dir), run_id)
                    paths["raw_daf"].mkdir(parents=True, exist_ok=True)
                    
                    # Save file immediately to disk - use safe_path
                    file_path = safe_path(paths["raw_daf"]) / safe_str(file_name)
                    file_path.write_bytes(file_content)
                    
                    session_state["uploaded_files"].append({
                        "name": safe_str(file_name),
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                    })
                    
                    # Start background conversion - ensure all paths are str
                    file_stem = safe_str(file_path.stem)
                    fcs_path = safe_path(paths["fcs"]) / f"{file_stem}.fcs"
                    paths["fcs"].mkdir(parents=True, exist_ok=True)
                    
                    job_id = f"convert_{safe_str(file_name)}_{datetime.datetime.now().timestamp()}"
                    asyncio.create_task(convert_daf_to_fcs_background(file_path, fcs_path, job_id))
                    
                    ui.notify(f"✅ {file_name} uploaded and conversion started", type='positive')
                    update_status()
                    
                    # Trigger features reload after a delay (to allow conversion to start)
                    async def reload_features_delayed():
                        await asyncio.sleep(2)
                        # This will be handled by the timer that calls load_features_and_gates
                    
                    asyncio.create_task(reload_features_delayed())
                except Exception as ex:
                    ui.notify(f"❌ Upload error: {str(ex)}", type='negative')
            
            ui.upload(
                label='📁 Drag & drop .daf files here',
                auto_upload=True,
                max_file_size=500 * 1024 * 1024,  # 500 MB
                multiple=True,
                on_upload=handle_upload
            ).classes('w-full')
            
            # Processing status
            processing_status_container = ui.column().classes('w-full mt-4 gap-2')
            
            def update_processing_status():
                processing_status_container.clear()
                with processing_status_container:
                    for job_id, status in list(session_state["processing_status"].items())[-5:]:  # Show last 5
                        color = {
                            "processing": "text-blue-600",
                            "completed": "text-green-600",
                            "error": "text-red-600"
                        }.get(status["status"], "text-gray-600")
                        ui.label(f"{status['message']}").classes(f'text-sm {color}')
            
            ui.timer(3.0, update_processing_status)  # Less frequent for stability
        
        # ====================================================================
        # 3. Metadata
        # ====================================================================
        with ui.card().classes('w-full'):
            ui.label('3️⃣ Metadata').classes('text-xl font-bold mb-4')
            
            # Mode selection: Upload (right) or Manual (left) - side by side
            with ui.row().classes('w-full gap-4 mb-4'):
                # Left: Manual Entry
                with ui.column().classes('flex-1'):
                    ui.label('✏️ Manual Entry').classes('text-sm font-bold mb-2')
                    metadata_manual_container = ui.column().classes('w-full')
                
                # Right: Upload Metadata
                with ui.column().classes('flex-1'):
                    ui.label('📄 Upload Metadata').classes('text-sm font-bold mb-2')
                    metadata_upload_container = ui.column().classes('w-full')
                    
                    async def handle_metadata_upload(e):
                        try:
                            if not session_state.get("run_id"):
                                ui.notify("⚠️ Please set Run-ID first", type='warning')
                                return
                            
                            file = e.file
                            file_name = str(file.name) if file.name else "metadata.csv"
                            file_content = await file.read()
                            
                            # Ensure file_content is bytes
                            if isinstance(file_content, str):
                                file_content = file_content.encode('utf-8')
                            
                            # Ensure project_dir and run_id are str, not bytes - use safe_str
                            project_dir = safe_str(session_state.get("project_dir", PROJECT_ROOT))
                            run_id = safe_str(session_state.get("run_id", "default"))
                            
                            paths = get_run_paths(safe_path(project_dir), run_id)
                            metadata_path = safe_path(paths["metadata"]) / safe_str(file_name)
                            metadata_path.parent.mkdir(parents=True, exist_ok=True)
                            metadata_path.write_bytes(file_content)
                            
                            # Load and store
                            if file_name.endswith('.csv'):
                                session_state["metadata_df"] = pd.read_csv(metadata_path)
                            elif file_name.endswith(('.xlsx', '.xls')):
                                session_state["metadata_df"] = pd.read_excel(metadata_path)
                            
                            if "file_name" in session_state["metadata_df"].columns and "sample_id" not in session_state["metadata_df"].columns:
                                session_state["metadata_df"]["sample_id"] = session_state["metadata_df"]["file_name"]
                            
                            ui.notify(f"✅ Metadata loaded: {len(session_state['metadata_df'])} rows", type='positive')
                            update_metadata_display()
                        except Exception as ex:
                            ui.notify(f"❌ Metadata error: {str(ex)}", type='negative')
                    
                    with metadata_upload_container:
                        ui.upload(
                            label='📄 Upload metadata (CSV or Excel)',
                            auto_upload=True,
                            on_upload=handle_metadata_upload
                        ).classes('w-full')
                        
                        # Show loaded metadata
                        if "metadata_df" in session_state and len(session_state["metadata_df"]) > 0:
                            ui.label(f"✅ Loaded: {len(session_state['metadata_df'])} rows").classes('text-sm text-green-600 mt-2')
            
            metadata_container = ui.column().classes('w-full')
            
            # Track if metadata is being edited to prevent timer from overwriting
            if "metadata_editing" not in session_state:
                session_state["metadata_editing"] = False
            
            def update_metadata_display(force=False):
                """Update metadata table display. Only updates if not being edited."""
                # Don't update if user is editing (unless forced)
                if session_state.get("metadata_editing", False) and not force:
                    return
                
                metadata_manual_container.clear()
                
                with metadata_manual_container:
                    # Manual mode - always show
                    # Get FCS file names for auto-fill
                    fcs_files = []
                    if session_state.get("run_id"):
                        project_dir = safe_str(session_state.get("project_dir", PROJECT_ROOT))
                        run_id = safe_str(session_state["run_id"])
                        paths = get_run_paths(safe_path(project_dir), run_id)
                        fcs_files = sorted([safe_str(f.stem) for f in paths["fcs"].glob("*.fcs")])
                    
                    # Initialize or update metadata from FCS files
                    if "metadata_df" not in session_state or session_state["metadata_df"] is None or len(session_state["metadata_df"]) == 0:
                        # Create metadata from FCS files - no empty rows, auto-number sample_id
                        if fcs_files:
                            metadata_rows = []
                            for idx, fcs_name in enumerate(fcs_files, start=1):
                                metadata_rows.append({
                                    "file_name": fcs_name,
                                    "sample_id": f"sample_{idx}",  # Auto-number instead of copying file_name
                                    "group": "",
                                    "replicate": ""
                                })
                            session_state["metadata_df"] = pd.DataFrame(metadata_rows)
                        else:
                            # Only create default if no FCS files yet
                            session_state["metadata_df"] = pd.DataFrame(columns=["file_name", "sample_id", "group", "replicate"])
                    else:
                        # Update existing metadata: add new FCS files that aren't in metadata yet
                        existing_file_names = set()
                        if "file_name" in session_state["metadata_df"].columns:
                            existing_file_names = set(session_state["metadata_df"]["file_name"].astype(str).dropna().values)
                        
                        new_fcs_files = [f for f in fcs_files if f not in existing_file_names]
                        if new_fcs_files:
                            # Get next sample number
                            max_sample_num = 0
                            if "sample_id" in session_state["metadata_df"].columns:
                                for sid in session_state["metadata_df"]["sample_id"].astype(str).values:
                                    if sid.startswith("sample_") and sid[7:].isdigit():
                                        max_sample_num = max(max_sample_num, int(sid[7:]))
                            
                            new_rows = []
                            for idx, fcs_name in enumerate(new_fcs_files, start=1):
                                new_rows.append({
                                    "file_name": fcs_name,
                                    "sample_id": f"sample_{max_sample_num + idx}",  # Auto-number
                                    "group": "",
                                    "replicate": ""
                                })
                            session_state["metadata_df"] = pd.concat([
                                session_state["metadata_df"],
                                pd.DataFrame(new_rows)
                            ], ignore_index=True)
                    
                    df = session_state["metadata_df"]
                    
                    ui.label(f"Metadata entries: {len(df)} row(s)").classes('text-sm font-bold mb-2')
                    
                    # Create editable table with scrollable container (show only 3 rows)
                    metadata_inputs = []
                    with ui.column().classes('w-full gap-2'):
                        # Header
                        with ui.row().classes('w-full gap-2 font-bold border-b pb-2'):
                            ui.label("file_name").classes('flex-1 text-sm')
                            ui.label("sample_id").classes('flex-1 text-sm')
                            ui.label("group").classes('flex-1 text-sm')
                            ui.label("replicate").classes('flex-1 text-sm')
                        
                        # Scrollable container for data rows (show only 3, rest scrollable)
                        with ui.scroll_area().classes('w-full').style('max-height: 200px;'):
                            with ui.column().classes('w-full gap-2'):
                                # Data rows (all rows, but container is scrollable)
                                for idx in range(len(df)):
                                    row_inputs = {}
                                    with ui.row().classes('w-full gap-2'):
                                        for col in ["file_name", "sample_id", "group", "replicate"]:
                                            if col not in df.columns:
                                                df[col] = ""
                                            val = str(df.iloc[idx][col]) if pd.notna(df.iloc[idx][col]) else ""
                                            # Track focus to prevent timer updates
                                            def on_focus():
                                                session_state["metadata_editing"] = True
                                            
                                            def on_blur():
                                                # Small delay before allowing updates
                                                import asyncio
                                                async def delayed():
                                                    await asyncio.sleep(2)
                                                    session_state["metadata_editing"] = False
                                                asyncio.create_task(delayed())
                                            
                                            input_field = ui.input(
                                                label="",
                                                value=val,
                                                placeholder=col
                                            ).classes('flex-1 text-sm')
                                            input_field.on('focus', on_focus)
                                            input_field.on('blur', on_blur)
                                            row_inputs[col] = input_field
                                    metadata_inputs.append(row_inputs)
                        
                        if len(df) > 3:
                            ui.label(f"Showing 3 of {len(df)} rows (scroll to see more)").classes('text-xs text-gray-500')
                        
                        # Add row button
                        def add_metadata_row():
                            new_row = {"file_name": "", "sample_id": "", "group": "", "replicate": ""}
                            session_state["metadata_df"] = pd.concat([
                                session_state["metadata_df"],
                                pd.DataFrame([new_row])
                            ], ignore_index=True)
                            update_metadata_display()
                        
                        ui.button('➕ Add Row', on_click=add_metadata_row, color='primary').classes('w-full')
                        
                        # Save button
                        def save_metadata():
                            try:
                                if not session_state.get("run_id"):
                                    ui.notify("⚠️ Please set Run-ID first", type='warning')
                                    return
                                
                                # Collect data from inputs (all rows are now in metadata_inputs)
                                new_data = []
                                for row_inputs in metadata_inputs:
                                    row_data = {col: row_inputs[col].value for col in row_inputs.keys()}
                                    # Ensure sample_id = file_name if sample_id is empty
                                    if not row_data.get("sample_id") and row_data.get("file_name"):
                                        row_data["sample_id"] = row_data["file_name"]
                                    new_data.append(row_data)
                                
                                # Update session state
                                session_state["metadata_df"] = pd.DataFrame(new_data)
                                
                                # Save to file
                                project_dir = safe_str(session_state.get("project_dir", PROJECT_ROOT))
                                run_id = safe_str(session_state["run_id"])
                                paths = get_run_paths(safe_path(project_dir), run_id)
                                metadata_path = safe_path(paths["metadata"]) / "sample_sheet.csv"
                                save_metadata_file(session_state["metadata_df"], metadata_path)
                                
                                ui.notify(f"✅ Metadata saved: {len(new_data)} rows", type='positive')
                            except Exception as ex:
                                ui.notify(f"❌ Save error: {str(ex)}", type='negative')
                        
                        ui.button('💾 Save Metadata', on_click=save_metadata, color='primary').classes('w-full')
            
            update_metadata_display()
            # Auto-update metadata when FCS files are added (much less frequent to avoid connection issues)
            # Only check if new files exist, don't refresh the whole display
            def check_new_files():
                if not session_state.get("metadata_editing", False) and session_state.get("run_id"):
                    project_dir = safe_str(session_state.get("project_dir", PROJECT_ROOT))
                    run_id = safe_str(session_state["run_id"])
                    paths = get_run_paths(safe_path(project_dir), run_id)
                    fcs_files = sorted([safe_str(f.stem) for f in paths["fcs"].glob("*.fcs")])
                    if fcs_files:
                        existing_files = set()
                        if "metadata_df" in session_state and "file_name" in session_state["metadata_df"].columns:
                            existing_files = set(session_state["metadata_df"]["file_name"].astype(str).dropna().values)
                        new_files = [f for f in fcs_files if f not in existing_files]
                        if new_files:
                            # Force update only if new files detected
                            update_metadata_display(force=True)
            
            ui.timer(10.0, check_new_files)  # Check every 10 seconds, much less frequent
        
        # ====================================================================
        # 4. Features & Gates Selection
        # ====================================================================
        with ui.card().classes('w-full'):
            ui.label('4️⃣ Features & Gates Selection').classes('text-xl font-bold mb-4')
            
            # This section will be populated dynamically based on uploaded files
            features_container = ui.column().classes('w-full')
            gates_container = ui.column().classes('w-full')
            
            # Store feature selection state
            feature_selection_state = {
                "available_features": [],
                "included_features": [],
                "excluded_features": [],
                "selected_features": [],
                "populations": [],
            }
            
            def load_features_and_gates():
                """Load available features and gates from FCS files."""
                features_container.clear()
                gates_container.clear()
                
                if not session_state.get("run_id"):
                    with features_container:
                        ui.label("ℹ️ Run-ID is set. Upload DAF files to load features.").classes('text-sm text-gray-500')
                    return
                
                project_dir = safe_str(session_state.get("project_dir", PROJECT_ROOT))
                run_id = safe_str(session_state["run_id"])
                paths = get_run_paths(safe_path(project_dir), run_id)
                fcs_files = sorted(paths["fcs"].glob("*.fcs"))
                
                # Check if conversion is in progress
                converting = any(
                    status.get("status") == "processing" 
                    for job_id, status in session_state.get("processing_status", {}).items()
                    if "convert_" in job_id
                )
                
                if not fcs_files:
                    with features_container:
                        if converting:
                            ui.label("⏳ Converting DAF files... Please wait.").classes('text-sm text-blue-600')
                        else:
                            ui.label("ℹ️ No FCS files found. Upload DAF files first.").classes('text-sm text-gray-500')
                    return
                
                try:
                    cache_dir = paths["csv_cache"]
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    converter = MM()
                    
                    # Load all files to find common populations
                    all_populations_per_file = {}
                    all_columns_per_file = {}
                    
                    for fcs_file in fcs_files:
                        csv_path = safe_path(cache_dir) / f"{safe_str(fcs_file.stem)}.csv"
                        if not csv_path.exists():
                            converter.convert_to_CSV(safe_str(fcs_file), safe_str(csv_path))
                        df = pd.read_csv(csv_path)
                        all_columns_per_file[safe_str(fcs_file.name)] = set(df.columns)
                        
                        # Identify populations
                        populations = []
                        for col in df.columns:
                            if col == "sample_id":
                                continue
                            if (col.startswith("R") and ("&" in col or (len(col) > 1 and col[1:].isdigit()))) or \
                               col in ["Live", "Singlet", "Dead", "Doublet", "Focused", "Viable"] or \
                               df[col].dtype == bool or \
                               (df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1, True, False, 0.0, 1.0})):
                                populations.append(col)
                        all_populations_per_file[safe_str(fcs_file.name)] = set(populations)
                    
                    # Find common populations
                    if all_populations_per_file:
                        common_populations_set = set.intersection(*all_populations_per_file.values())
                        all_populations_set = set.union(*all_populations_per_file.values())
                        common_populations = sorted(common_populations_set)
                        excluded_populations = sorted(all_populations_set - common_populations_set)
                    else:
                        common_populations = []
                        excluded_populations = []
                        all_populations_set = set()
                    
                    # Get features from first file
                    first_fcs = fcs_files[0]
                    csv_path = cache_dir / f"{first_fcs.stem}.csv"
                    df_sample = pd.read_csv(csv_path)
                    all_features = sorted([c for c in df_sample.columns if c != "sample_id" and c not in all_populations_set])
                    
                    # Exclude patterns
                    exclude_patterns = ["intensity", "Intensity", "saturation", "Saturation", "Raw pixel", "Bkgd", "All", "Max Pixel", "Median Pixel", "Raw", "Time"]
                    excluded_by_default = [f for f in all_features if any(p in f for p in exclude_patterns)]
                    included_by_default = [f for f in all_features if f not in excluded_by_default]
                    
                    # Update state
                    feature_selection_state["available_features"] = all_features
                    feature_selection_state["included_features"] = included_by_default
                    feature_selection_state["excluded_features"] = excluded_by_default
                    feature_selection_state["populations"] = common_populations
                    
                    # Get current selection or use defaults if empty
                    current_selected = session_state.get("features", [])
                    if not current_selected or len(current_selected) == 0:
                        # Use defaults if no selection
                        current_selected = included_by_default
                        session_state["features"] = included_by_default
                    feature_selection_state["selected_features"] = [f for f in current_selected if f in all_features]
                    
                    # Ensure all features are either included or excluded (except those not in all files)
                    selected_set = set(feature_selection_state["selected_features"])
                    excluded_set = set(feature_selection_state.get("excluded_features", excluded_by_default))
                    # Features that are neither included nor excluded should be excluded by default
                    for feat in all_features:
                        if feat not in selected_set and feat not in excluded_set:
                            if feat not in excluded_by_default:
                                # Add to excluded if not in default excluded
                                if "excluded_features" not in feature_selection_state:
                                    feature_selection_state["excluded_features"] = []
                                feature_selection_state["excluded_features"].append(feat)
                    
                    with features_container:
                        ui.label(f"📊 Features Selection").classes('text-lg font-bold mb-2')
                        ui.label(f"Available: {len(all_features)} features").classes('text-sm mb-4')
                        
                        # Two-column layout for Include/Exclude with clickable chips
                        with ui.row().classes('w-full gap-4'):
                            # Included Features - clickable chips
                            with ui.column().classes('flex-1'):
                                ui.label("✅ Included Features").classes('text-sm font-bold mb-2')
                                included_chips_container = ui.column().classes('w-full gap-2 flex-wrap')
                                
                                def create_feature_chip(feature_name: str, is_included: bool):
                                    """Create a clickable chip for a feature (smaller size)."""
                                    if is_included:
                                        # Blue for included - smaller size
                                        chip_btn = ui.button(
                                            f'{feature_name} ✕',
                                            on_click=lambda f=feature_name, inc=is_included: remove_feature(f, inc)
                                        ).props('flat dense').classes('bg-blue-100 text-blue-800 hover:bg-blue-200 rounded-full px-2 py-0.5 text-xs')
                                    else:
                                        # Red for excluded - smaller size
                                        chip_btn = ui.button(
                                            f'{feature_name} ✕',
                                            on_click=lambda f=feature_name, inc=is_included: remove_feature(f, inc)
                                        ).props('flat dense').classes('bg-red-100 text-red-800 hover:bg-red-200 rounded-full px-2 py-0.5 text-xs')
                                    return chip_btn
                                
                                def remove_feature(feature_name: str, from_included: bool):
                                    """Remove feature from included/excluded and move to the other list."""
                                    if from_included:
                                        # Move from included to excluded
                                        if feature_name in feature_selection_state["selected_features"]:
                                            feature_selection_state["selected_features"].remove(feature_name)
                                        if feature_name not in feature_selection_state.get("excluded_features", []):
                                            if "excluded_features" not in feature_selection_state:
                                                feature_selection_state["excluded_features"] = []
                                            feature_selection_state["excluded_features"].append(feature_name)
                                    else:
                                        # Move from excluded to included
                                        if feature_name in feature_selection_state.get("excluded_features", []):
                                            feature_selection_state["excluded_features"].remove(feature_name)
                                        if feature_name not in feature_selection_state["selected_features"]:
                                            feature_selection_state["selected_features"].append(feature_name)
                                    
                                    session_state["features"] = feature_selection_state["selected_features"]
                                    load_features_and_gates()  # Refresh display
                                
                                def add_feature_to_included(feature_name: str):
                                    """Add feature to included list."""
                                    if feature_name not in feature_selection_state["selected_features"]:
                                        feature_selection_state["selected_features"].append(feature_name)
                                    if feature_name in feature_selection_state.get("excluded_features", []):
                                        feature_selection_state["excluded_features"].remove(feature_name)
                                    session_state["features"] = feature_selection_state["selected_features"]
                                    load_features_and_gates()
                                
                                # Display included features: show first 10, rest in collapsible expansion
                                with included_chips_container:
                                    if feature_selection_state["selected_features"]:
                                        features_list = feature_selection_state["selected_features"]
                                        
                                        # Show first 10 features directly
                                        first_10 = features_list[:10]
                                        with ui.row().classes('flex-wrap gap-1'):
                                            for feat in first_10:
                                                create_feature_chip(feat, is_included=True)
                                        
                                        # Show rest in collapsible expansion
                                        if len(features_list) > 10:
                                            remaining = features_list[10:]
                                            with ui.expansion(
                                                f'Show remaining {len(remaining)} included features',
                                                value=False
                                            ).classes('w-full mt-2'):
                                                with ui.row().classes('flex-wrap gap-1'):
                                                    for feat in remaining:
                                                        create_feature_chip(feat, is_included=True)
                                    else:
                                        ui.label("No features included. Click features from excluded list to add.").classes('text-sm text-gray-500')
                                
                                # Quick add from excluded (if any)
                                excluded_list = feature_selection_state.get("excluded_features", excluded_by_default)
                                if excluded_list:
                                    ui.label("Quick add from excluded:").classes('text-xs text-gray-500 mt-2')
                                    quick_add_container = ui.row().classes('flex-wrap gap-1')
                                    with quick_add_container:
                                        for feat in excluded_list[:15]:  # Show first 15
                                            ui.button(feat, on_click=lambda f=feat: add_feature_to_included(f)).props('flat dense size=sm').classes('text-xs bg-gray-100 hover:bg-gray-200 rounded px-2 py-1')
                            
                            # Excluded Features - clickable chips
                            with ui.column().classes('flex-1'):
                                ui.label("❌ Excluded Features").classes('text-sm font-bold mb-2')
                                excluded_chips_container = ui.column().classes('w-full gap-2 flex-wrap')
                                
                                # Display excluded features: show first 10, rest in collapsible expansion
                                with excluded_chips_container:
                                    excluded_list = feature_selection_state.get("excluded_features", excluded_by_default)
                                    if excluded_list:
                                        # Show first 10 features directly
                                        first_10 = excluded_list[:10]
                                        with ui.row().classes('flex-wrap gap-1'):
                                            for feat in first_10:
                                                create_feature_chip(feat, is_included=False)
                                        
                                        # Show rest in collapsible expansion
                                        if len(excluded_list) > 10:
                                            remaining = excluded_list[10:]
                                            with ui.expansion(
                                                f'Show remaining {len(remaining)} excluded features',
                                                value=False
                                            ).classes('w-full mt-2'):
                                                with ui.row().classes('flex-wrap gap-1'):
                                                    for feat in remaining:
                                                        create_feature_chip(feat, is_included=False)
                                    else:
                                        ui.label("No features excluded.").classes('text-sm text-gray-500')
                        
                        # Show selected count
                        selected_count = len(feature_selection_state["selected_features"])
                        ui.label(f"✅ {selected_count} features selected for analysis").classes('text-sm text-green-600 mt-2')
                    
                    with gates_container:
                        ui.label(f"🔬 Populations/Gates").classes('text-lg font-bold mb-2')
                        
                        if excluded_populations:
                            ui.label(f"⚠️ Populations not in all files (excluded): {', '.join(sorted(excluded_populations)[:5])}{'...' if len(excluded_populations) > 5 else ''}").classes('text-sm text-yellow-600 mb-2')
                        
                        if common_populations:
                            ui.label(f"Available: {len(common_populations)} common populations").classes('text-sm mb-2')
                            
                            # Use a more stable select widget
                            population_options = ["All events"] + sorted(common_populations)
                            current_pop = session_state.get("selected_population")
                            current_value = current_pop if current_pop and current_pop in common_populations else "All events"
                            
                            population_select = ui.select(
                                options=population_options,
                                label="Select population (optional)",
                                value=current_value
                            ).classes('w-full')
                            
                            def update_population():
                                val = population_select.value
                                session_state["selected_population"] = None if val == "All events" else val
                            
                            # Use change event instead of update:model-value for stability
                            population_select.on('change', lambda: update_population())
                        else:
                            ui.label("ℹ️ No common populations detected. All events will be analyzed.").classes('text-sm text-gray-500')
                            session_state["selected_population"] = None
                
                except Exception as e:
                    with features_container:
                        ui.label(f"⚠️ Error loading features: {str(e)}").classes('text-sm text-red-600')
            
            load_features_and_gates()
            # Only refresh features if FCS files change, not constantly
            def check_features_update():
                if session_state.get("run_id"):
                    project_dir = safe_str(session_state.get("project_dir", PROJECT_ROOT))
                    run_id = safe_str(session_state["run_id"])
                    paths = get_run_paths(safe_path(project_dir), run_id)
                    fcs_count = len(list(paths["fcs"].glob("*.fcs")))
                    last_count = session_state.get("_last_fcs_count", 0)
                    if fcs_count != last_count:
                        session_state["_last_fcs_count"] = fcs_count
                        load_features_and_gates()
            
            ui.timer(10.0, check_features_update)  # Check every 10 seconds, much less frequent
        
        # ====================================================================
        # 5. Dimensionality Reduction
        # ====================================================================
        with ui.card().classes('w-full'):
            ui.label('5️⃣ Run Dimensionality Reduction').classes('text-xl font-bold mb-4')
            
            method_select = ui.select(
                options=["DensMAP"] + (["UMAP"] if UMAP_AVAILABLE else []) + ["t-SNE"],
                label="Method",
                value="DensMAP"
            ).classes('w-full mb-4')
            
            # Parameters (simplified - show only when method selected)
            params_container = ui.column().classes('w-full gap-2')
            
            # Store slider references and parameter values
            param_sliders = {}
            param_values = {
                "dens_lambda": 2.0,
                "n_neighbors": 30,
                "min_dist": 0.1,
                "perplexity": 30.0,
            }
            
            def update_params():
                params_container.clear()
                method = method_select.value
                param_sliders.clear()
                
                with params_container:
                    if method == "DensMAP":
                        ui.label("dens_lambda").classes('text-sm font-bold')
                        param_sliders["dens_lambda"] = ui.slider(min=0.5, max=3.0, value=2.0, step=0.1)
                        
                        ui.label("n_neighbors").classes('text-sm font-bold')
                        param_sliders["n_neighbors"] = ui.slider(min=5, max=100, value=30, step=1)
                        
                        ui.label("min_dist").classes('text-sm font-bold')
                        param_sliders["min_dist"] = ui.slider(min=0.0, max=0.99, value=0.1, step=0.01)
                    elif method == "UMAP":
                        ui.label("n_neighbors").classes('text-sm font-bold')
                        param_sliders["n_neighbors"] = ui.slider(min=5, max=100, value=30, step=1)
                        
                        ui.label("min_dist").classes('text-sm font-bold')
                        param_sliders["min_dist"] = ui.slider(min=0.0, max=0.99, value=0.1, step=0.01)
                    else:  # t-SNE
                        ui.label("perplexity").classes('text-sm font-bold')
                        param_sliders["perplexity"] = ui.slider(min=5.0, max=50.0, value=30.0, step=1.0)
            
            method_select.on('update:model-value', lambda: update_params())
            update_params()
            
            def run_analysis():
                if not session_state.get("run_id"):
                    ui.notify("⚠️ Please set Run-ID first", type='warning')
                    return
                
                features = session_state.get("features", [])
                if not features or len(features) == 0:
                    ui.notify("⚠️ Please select at least one feature", type='warning')
                    return
                
                project_dir = safe_str(session_state.get("project_dir", PROJECT_ROOT))
                run_id = safe_str(session_state["run_id"])
                paths = get_run_paths(safe_path(project_dir), run_id)
                fcs_files = list(paths["fcs"].glob("*.fcs"))
                
                if not fcs_files:
                    ui.notify("⚠️ No FCS files found. Upload and convert DAF files first.", type='warning')
                    return
                
                method = method_select.value
                # Get current values from sliders
                method_params = {
                    "dens_lambda": param_sliders.get("dens_lambda", ui.slider(min=0.5, max=3.0, value=2.0)).value if "dens_lambda" in param_sliders else 2.0,
                    "n_neighbors": param_sliders.get("n_neighbors", ui.slider(min=5, max=100, value=30)).value if "n_neighbors" in param_sliders else 30,
                    "min_dist": param_sliders.get("min_dist", ui.slider(min=0.0, max=0.99, value=0.1)).value if "min_dist" in param_sliders else 0.1,
                    "perplexity": param_sliders.get("perplexity", ui.slider(min=5.0, max=50.0, value=30.0)).value if "perplexity" in param_sliders else 30.0,
                }
                
                job_id = f"analysis_{datetime.datetime.now().timestamp()}"
                
                # Show progress notification
                ui.notify(f"🚀 Starting {method} analysis with {len(features)} features...", type='info')
                
                try:
                    asyncio.create_task(run_analysis_background(
                        paths,
                        features,
                        method,
                        method_params,
                        session_state.get("selected_population"),
                        job_id,
                    ))
                except Exception as e:
                    ui.notify(f"❌ Error starting analysis: {str(e)}", type='negative')
                    session_state["processing_status"][job_id] = {
                        "status": "error",
                        "message": f"Error: {str(e)}",
                    }
            
            ui.button('🚀 RUN', on_click=run_analysis, color='primary').classes('w-full mt-4')
        
        # ====================================================================
        # 6. Visualization
        # ====================================================================
        if session_state.get("embedding_df") is not None:
            with ui.card().classes('w-full'):
                ui.label('6️⃣ Visualization').classes('text-xl font-bold mb-4')
                
                embedding_df = session_state["embedding_df"]
                method = session_state.get("stored_dim_reduction_method", "DensMAP")
                x_label, y_label = get_axis_labels(method)
                
                # Color by selection
                color_options = ["sample_id"] + [c for c in embedding_df.columns if c not in ["x", "y", "sample_id", "cell_index", "cluster", "cluster_numeric", "highlighted"]]
                color_by_select = ui.select(
                    options=color_options,
                    label="Color by",
                    value="sample_id"
                ).classes('w-full mb-4')
                
                # Create plot
                plot_container = ui.column().classes('w-full')
                
                def update_plot():
                    plot_container.clear()
                    
                    color_col = color_by_select.value
                    plot_data = embedding_df.copy()
                    plot_data[color_col] = plot_data[color_col].fillna("Unknown")
                    
                    highlighted = session_state.get("highlighted_cells", [])
                    
                    plot_bytes = create_matplotlib_plot(
                        plot_data, "x", "y", color_col,
                        f"{method} Visualization", x_label, y_label, highlighted
                    )
                    
                    with plot_container:
                        ui.image(plot_bytes).classes('w-full')
                
                color_by_select.on('update:model-value', lambda: update_plot())
                update_plot()
        
        # ====================================================================
        # 7. Clustering
        # ====================================================================
        if session_state.get("embedding_df") is not None:
            with ui.card().classes('w-full'):
                ui.label('7️⃣ Clustering').classes('text-xl font-bold mb-4')
                
                cluster_method_select = ui.select(
                    options=["KMeans", "Gaussian Mixture Models", "HDBSCAN"],
                    label="Algorithm",
                    value="KMeans"
                ).classes('w-full mb-2')
                
                n_clusters_input = ui.number(
                    label="Number of clusters",
                    value=10,
                    min=2,
                    max=100,
                    precision=0
                ).classes('w-full mb-4')
                
                def run_clustering():
                    if session_state.get("embedding_df") is None:
                        ui.notify("⚠️ Please run dimensionality reduction first", type='warning')
                        return
                    
                    coords = session_state["embedding_df"][["x", "y"]].values
                    cluster_labels = run_clustering_analysis(
                        coords,
                        cluster_method_select.value,
                        int(n_clusters_input.value),
                    )
                    
                    # Update embedding with clusters
                    embedding_with_clusters = session_state["embedding_df"].copy()
                    embedding_with_clusters["cluster_numeric"] = cluster_labels
                    cluster_str = cluster_labels.astype(str)
                    cluster_str[cluster_labels == -1] = "Noise"
                    embedding_with_clusters["cluster"] = cluster_str
                    
                    session_state["embedding_df"] = embedding_with_clusters
                    session_state["cluster_labels"] = cluster_labels
                    session_state["cluster_method"] = cluster_method_select.value
                    session_state["cluster_param"] = int(n_clusters_input.value)
                    
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    ui.notify(f"✅ Clustering completed: {n_clusters} clusters", type='positive')
                    update_plot()  # Refresh plot
                
                ui.button('🚀 Run Clustering', on_click=run_clustering, color='primary').classes('w-full mb-4')
                
                # Cluster plot
                if session_state.get("cluster_labels") is not None:
                    cluster_plot_container = ui.column().classes('w-full')
                    
                    def update_cluster_plot():
                        cluster_plot_container.clear()
                        
                        embedding_with_clusters = session_state["embedding_df"]
                        method = session_state.get("stored_dim_reduction_method", "DensMAP")
                        x_label, y_label = get_axis_labels(method)
                        
                        # Create cluster plot
                        fig, ax = plt.subplots(figsize=(10, 8))
                        unique_clusters = sorted([c for c in embedding_with_clusters["cluster"].unique() if pd.notna(c)])
                        
                        for i, cluster_id in enumerate(unique_clusters):
                            subset = embedding_with_clusters[embedding_with_clusters["cluster"] == cluster_id]
                            color = plt.cm.tab20(i % 20)
                            ax.scatter(subset["x"], subset["y"], label=str(cluster_id), alpha=0.7, s=40, color=color)
                        
                        ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
                        ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
                        ax.tick_params(axis="both", which="major", labelsize=10, width=2)
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.grid(False)
                        if len(unique_clusters) <= 20:
                            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                        plt.tight_layout()
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                        buf.seek(0)
                        plt.close(fig)
                        
                        with cluster_plot_container:
                            ui.image(buf.read()).classes('w-full')
                    
                    update_cluster_plot()


# ============================================================================
# Start Application
# ============================================================================

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="MorphoMapping",
        port=8501,
        show=True,
        reload=False,  # Important for stability with large files
    )
