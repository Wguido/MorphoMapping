"""
Session State Manager for MorphoMapping GUI

Provides functions to save and restore Streamlit session state to prevent data loss
when the app crashes or is restarted.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


class SessionStateManager:
    """Manages saving and loading of session state to/from disk."""
    
    def __init__(self, cache_dir: Path):
        """
        Initialize the session state manager.
        
        Args:
            cache_dir: Directory where session state files will be stored
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.cache_dir / "session_state.pkl"
        self.metadata_file = self.cache_dir / "session_metadata.json"
    
    def save_session_state(self, session_state: Dict[str, Any], run_id: Optional[str] = None) -> bool:
        """
        Save session state to disk.
        
        Args:
            session_state: Dictionary containing session state data
            run_id: Optional run ID to include in filename
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Filter out non-serializable items and large DataFrames
            saveable_state = {}
            skipped_keys = []
            
            for key, value in session_state.items():
                # Skip certain keys that shouldn't be saved
                if key in ["_last_rerun", "_last_widget_values", "_widget_states"]:
                    continue
                
                try:
                    # Try to serialize to check if it's saveable
                    if isinstance(value, (pd.DataFrame, pd.Series)):
                        # Save DataFrames separately if they're too large
                        if len(value) > 1000000:  # More than 1M rows
                            skipped_keys.append(f"{key} (too large: {len(value)} rows)")
                            continue
                        saveable_state[key] = value
                    elif isinstance(value, (np.ndarray, np.generic)):
                        if value.size > 10000000:  # More than 10M elements
                            skipped_keys.append(f"{key} (too large: {value.size} elements)")
                            continue
                        saveable_state[key] = value
                    elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        saveable_state[key] = value
                    else:
                        # Try to pickle it
                        pickle.dumps(value)
                        saveable_state[key] = value
                except Exception as e:
                    skipped_keys.append(f"{key} (not serializable: {type(value).__name__})")
            
            # Save the state
            with open(self.session_file, "wb") as f:
                pickle.dump(saveable_state, f)
            
            # Save metadata
            metadata = {
                "run_id": run_id,
                "timestamp": pd.Timestamp.now().isoformat(),
                "keys_saved": list(saveable_state.keys()),
                "keys_skipped": skipped_keys,
                "num_keys": len(saveable_state)
            }
            
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving session state: {e}")
            return False
    
    def load_session_state(self) -> Optional[Dict[str, Any]]:
        """
        Load session state from disk.
        
        Returns:
            Dictionary containing session state, or None if loading failed
        """
        try:
            if not self.session_file.exists():
                return None
            
            with open(self.session_file, "rb") as f:
                state = pickle.load(f)
            
            return state
        except Exception as e:
            print(f"Error loading session state: {e}")
            return None
    
    def clear_session_state(self) -> bool:
        """
        Clear saved session state.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.session_file.exists():
                self.session_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            return True
        except Exception as e:
            print(f"Error clearing session state: {e}")
            return False
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the saved session state.
        
        Returns:
            Dictionary with session metadata, or None if no session exists
        """
        try:
            if not self.metadata_file.exists():
                return None
            
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading session info: {e}")
            return None

