"""
Background Processing für Streamlit - Schnell-Win Lösung

Trennt Upload und Processing, um Streamlit stabiler zu machen.
"""

import threading
import queue
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingJob:
    """Repräsentiert einen Background-Processing-Job."""
    job_id: str
    file_path: Path
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    result: Optional[Path] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None


class BackgroundProcessor:
    """
    Background Processor für Streamlit.
    
    Verarbeitet große Dateien in separaten Threads, um die UI nicht zu blockieren.
    """
    
    def __init__(self, status_file: Path):
        """
        Initialize Background Processor.
        
        Args:
            status_file: Path to JSON file where job status is stored
        """
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, ProcessingJob] = {}
        self._lock = threading.Lock()
        self._load_status()
    
    def _load_status(self):
        """Load job status from disk."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    for job_id, job_data in data.items():
                        job = ProcessingJob(
                            job_id=job_data['job_id'],
                            file_path=Path(job_data['file_path']),
                            status=JobStatus(job_data['status']),
                            progress=job_data.get('progress', 0.0),
                            message=job_data.get('message', ''),
                            error=job_data.get('error'),
                            result=Path(job_data['result']) if job_data.get('result') else None,
                            created_at=job_data.get('created_at', 0.0),
                            completed_at=job_data.get('completed_at')
                        )
                        self._jobs[job_id] = job
            except Exception as e:
                print(f"Error loading status: {e}")
    
    def _save_status(self):
        """Save job status to disk."""
        try:
            data = {}
            with self._lock:
                for job_id, job in self._jobs.items():
                    data[job_id] = {
                        'job_id': job.job_id,
                        'file_path': str(job.file_path),
                        'status': job.status.value,
                        'progress': job.progress,
                        'message': job.message,
                        'error': job.error,
                        'result': str(job.result) if job.result else None,
                        'created_at': job.created_at,
                        'completed_at': job.completed_at
                    }
            with open(self.status_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving status: {e}")
    
    def submit_job(self, job_id: str, file_path: Path, 
                   processor_func: Callable[[Path], Path]) -> ProcessingJob:
        """
        Submit a new processing job.
        
        Args:
            job_id: Unique job identifier
            file_path: Path to file to process
            processor_func: Function that processes the file and returns output path
        
        Returns:
            ProcessingJob object
        """
        job = ProcessingJob(
            job_id=job_id,
            file_path=file_path,
            status=JobStatus.PENDING,
            created_at=time.time()
        )
        
        with self._lock:
            self._jobs[job_id] = job
        
        # Start processing in background thread
        thread = threading.Thread(
            target=self._process_job,
            args=(job, processor_func),
            daemon=True
        )
        thread.start()
        
        self._save_status()
        return job
    
    def _process_job(self, job: ProcessingJob, processor_func: Callable[[Path], Path]):
        """Process a job in background thread."""
        try:
            with self._lock:
                job.status = JobStatus.RUNNING
                job.message = "Processing..."
            self._save_status()
            
            # Call processor function
            result_path = processor_func(job.file_path)
            
            with self._lock:
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.message = "Completed"
                job.result = result_path
                job.completed_at = time.time()
            self._save_status()
            
        except Exception as e:
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.message = f"Failed: {str(e)}"
                job.completed_at = time.time()
            self._save_status()
    
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job status."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> Dict[str, ProcessingJob]:
        """Get all jobs."""
        with self._lock:
            return self._jobs.copy()
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove old completed/failed jobs."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        with self._lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.completed_at and (current_time - job.completed_at) > max_age_seconds:
                    to_remove.append(job_id)
            
            for job_id in to_remove:
                del self._jobs[job_id]
        
        self._save_status()


# Global instance (wird in app.py initialisiert)
_processor: Optional[BackgroundProcessor] = None


def get_processor(status_file: Path) -> BackgroundProcessor:
    """Get or create global processor instance."""
    global _processor
    if _processor is None:
        _processor = BackgroundProcessor(status_file)
    return _processor

