"""
Configuration and logging setup for the AWS API parser.

This module contains logging configuration and global settings
used throughout the parsing application.
"""

import logging
import os


class CleanFormatter(logging.Formatter):
    """Custom formatter that provides clean, simple log formatting without timestamps."""
    
    LEVEL_PREFIXES = {
        'DEBUG': '[DEBUG] 🐛',
        'INFO': '[INFO] 📄',
        'WARNING': '[WARN] ⚠️ ',
        'ERROR': '[ERROR] ❌',
        'CRITICAL': '[CRIT] 🚨'
    }
    
    def format(self, record):
        # Get clean prefix for the log level
        prefix = self.LEVEL_PREFIXES.get(record.levelname, '[INFO]')
        
        # Format the message without timestamp for cleaner look
        formatted_message = f"{prefix} {record.getMessage()}"
        
        return formatted_message


def setup_logging(log_file='parser.log', level=logging.INFO):
    """
    Set up logging configuration for the application.
    
    Args:
        log_file (str): Name of the log file
        level: Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get logger instance
    logger = logging.getLogger(__name__)
    
    # Only set up handlers if they haven't been added yet
    if not logger.handlers:
        # Create formatter for file (with timestamp) and console (without timestamp)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = CleanFormatter()
        
        # Create handlers
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        
        # Configure logger
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


# Global logger instance
logger = setup_logging()

# Get the directory of this config file and construct absolute paths
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parsing_dir = os.path.dirname(_current_dir)  # Go up from utils/ to parsing/
_project_root = os.path.dirname(_parsing_dir)  # Go up from parsing/ to project root/

# Default configuration with absolute paths
DEFAULT_CONFIG = {
    'services_folder': os.path.join(_project_root, "docs", "services"),
    'output_folder': os.path.join(_project_root, "docs", "methods"),
    'checkpoint_file': os.path.join(_parsing_dir, "checkpoint.json"),
    'request_timeout': 50,
    'processing_timeout': 300,
    'sleep_between_requests': 1.0,
    'sleep_between_services': 0.5 
}
