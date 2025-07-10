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

# Default configuration
DEFAULT_CONFIG = {
    'services_folder': "../docs/services",
    'output_folder': "../docs/methods",
    'checkpoint_file': "checkpoint.json",
    'request_timeout': 50,
    'processing_timeout': 300,
    'sleep_between_requests': 1.0,
    'sleep_between_services': 2.0
}
