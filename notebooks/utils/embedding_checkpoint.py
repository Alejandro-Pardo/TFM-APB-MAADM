"""
Checkpoint management system for the embeddings creation process.

This module provides functionality to save and restore embedding creation progress,
allowing the process to resume from where it left off in case of interruption.
"""

import json
from pathlib import Path
from typing import List
from datetime import datetime


class EmbeddingCheckpointManager:
    """Simple checkpoint manager for embeddings creation process."""
    
    def __init__(self, checkpoint_file: str = "embeddings_checkpoint.json"):
        """Initialize the checkpoint manager."""
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data = self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint data from file if it exists."""
        if Path(self.checkpoint_file).exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading checkpoint file: {str(e)}")
                return self._create_default_checkpoint()
        else:
            return self._create_default_checkpoint()
    
    def _create_default_checkpoint(self):
        """Create a default checkpoint structure."""
        return {
            "last_updated": None,
            "services_processed": [],
            "methods_processed": {},
            "current_service": None,
            "phase": "services"  # "services" or "methods"
        }
    
    def save_checkpoint(self):
        """Save the current checkpoint data to file."""
        self.checkpoint_data["last_updated"] = datetime.now().isoformat()
        
        try:
            # Ensure the directory exists before saving
            checkpoint_path = Path(self.checkpoint_file)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=2)
        except Exception as e:
            print(f"Error saving checkpoint file: {str(e)}")
    
    def is_service_processed(self, service_name: str) -> bool:
        """Check if a service has been processed."""
        return service_name in self.checkpoint_data["services_processed"]
    
    def is_method_processed(self, service_name: str, method_name: str) -> bool:
        """Check if a method has been processed."""
        if service_name not in self.checkpoint_data["methods_processed"]:
            return False
        return method_name in self.checkpoint_data["methods_processed"][service_name]
    
    def mark_service_as_processed(self, service_name: str):
        """Mark a service as processed."""
        if service_name not in self.checkpoint_data["services_processed"]:
            self.checkpoint_data["services_processed"].append(service_name)
            self.save_checkpoint()
    
    def mark_method_as_processed(self, service_name: str, method_name: str):
        """Mark a method as processed."""
        if service_name not in self.checkpoint_data["methods_processed"]:
            self.checkpoint_data["methods_processed"][service_name] = []
        
        if method_name not in self.checkpoint_data["methods_processed"][service_name]:
            self.checkpoint_data["methods_processed"][service_name].append(method_name)
            self.save_checkpoint()
    
    def set_phase(self, phase: str):
        """Set the current processing phase."""
        self.checkpoint_data["phase"] = phase
        self.save_checkpoint()
    
    def get_phase(self) -> str:
        """Get the current processing phase."""
        return self.checkpoint_data.get("phase", "services")
    
    def get_processed_methods(self, service_name: str) -> List[str]:
        """Get the list of processed methods for a service."""
        return self.checkpoint_data["methods_processed"].get(service_name, [])
    
    def get_resume_info(self):
        """Get information about what has been processed for resuming."""
        return {
            "phase": self.get_phase(),
            "services_processed": len(self.checkpoint_data["services_processed"]),
            "methods_processed": sum(len(methods) for methods in self.checkpoint_data["methods_processed"].values())
        }
    
    def reset_checkpoint(self):
        """Reset the checkpoint to start fresh."""
        self.checkpoint_data = self._create_default_checkpoint()
        self.save_checkpoint()
