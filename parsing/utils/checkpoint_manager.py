"""
Checkpoint management system for the AWS API parser.

This module provides functionality to save and restore parsing progress,
allowing the parser to resume from where it left off in case of interruption.
"""

import json
import os
from datetime import datetime
from .config import logger


class CheckpointManager:
    """
    Manages checkpoints to allow resuming from where processing stopped.
    
    This class handles saving and loading progress information, including
    which services and methods have been processed.
    """
    
    def __init__(self, checkpoint_file="checkpoint.json"):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_file (str): Path to the checkpoint file
        """
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data = self._load_checkpoint()
    
    def _load_checkpoint(self):
        """
        Load checkpoint data from file if it exists.
        
        Returns:
            dict: Checkpoint data or default structure if file doesn't exist
        """
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading checkpoint file: {str(e)}")
                return self._create_default_checkpoint()
        else:
            return self._create_default_checkpoint()
    
    def _create_default_checkpoint(self):
        """
        Create a default checkpoint structure.
        
        Returns:
            dict: Default checkpoint data structure
        """
        return {
            "last_updated": datetime.now().isoformat(),
            "services_processed": [],
            "methods_processed": {},
            "current_service": None
        }
    
    def save_checkpoint(self):
        """Save the current checkpoint data to file."""
        self.checkpoint_data["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving checkpoint file: {str(e)}")
    
    def is_service_processed(self, service_name):
        """
        Check if a service has been fully processed.
        
        Args:
            service_name (str): Name of the service to check
        
        Returns:
            bool: True if service has been processed, False otherwise
        """
        return service_name in self.checkpoint_data["services_processed"]
    
    def is_method_processed(self, service_name, method_name):
        """
        Check if a method has been processed.
        
        Args:
            service_name (str): Name of the service
            method_name (str): Name of the method
        
        Returns:
            bool: True if method has been processed, False otherwise
        """
        if service_name not in self.checkpoint_data["methods_processed"]:
            return False
        return method_name in self.checkpoint_data["methods_processed"][service_name]
    
    def mark_service_as_current(self, service_name):
        """
        Mark a service as the current one being processed.
        
        Args:
            service_name (str): Name of the service being processed
        """
        self.checkpoint_data["current_service"] = service_name
        self.save_checkpoint()
    
    def mark_service_as_processed(self, service_name):
        """
        Mark a service as fully processed.
        
        Args:
            service_name (str): Name of the processed service
        """
        if service_name not in self.checkpoint_data["services_processed"]:
            self.checkpoint_data["services_processed"].append(service_name)
            self.checkpoint_data["current_service"] = None
            self.save_checkpoint()
    
    def mark_method_as_processed(self, service_name, method_name):
        """
        Mark a method as processed.
        
        Args:
            service_name (str): Name of the service
            method_name (str): Name of the processed method
        """
        if service_name not in self.checkpoint_data["methods_processed"]:
            self.checkpoint_data["methods_processed"][service_name] = []
        
        if method_name not in self.checkpoint_data["methods_processed"][service_name]:
            self.checkpoint_data["methods_processed"][service_name].append(method_name)
            self.save_checkpoint()
    
    def get_resume_position(self):
        """
        Get the position to resume processing from.
        
        Returns:
            str or None: Name of the service to resume from, or None if starting fresh
        """
        return self.checkpoint_data["current_service"]
    
    def get_processed_methods(self, service_name):
        """
        Get the list of processed methods for a service.
        
        Args:
            service_name (str): Name of the service
        
        Returns:
            list: List of processed method names for the service
        """
        if service_name not in self.checkpoint_data["methods_processed"]:
            return []
        return self.checkpoint_data["methods_processed"][service_name]
