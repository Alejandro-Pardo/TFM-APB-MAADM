"""
Configuration and logging setup for the AWS API parser.

This module contains logging configuration and global settings
used throughout the parsing application.
"""

import logging
import os

def setup_logging(log_file='parser.log', level=logging.INFO):
    """
    Set up logging configuration for the application.
    
    Args:
        log_file (str): Name of the log file
        level: Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

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
