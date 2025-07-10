"""
Main entry point for the AWS API documentation parser.

This script coordinates the parsing of AWS service documentation
to extract structured information about API methods, parameters,
and return values.

Usage:
    python main.py

The script will:
1. Set up logging and configuration
2. Initialize checkpoint management for resumable processing
3. Process all AWS services or resume from a previous checkpoint
4. Generate structured JSON output for all methods
"""

import os
from utils.config import logger, DEFAULT_CONFIG
from utils.checkpoint_manager import CheckpointManager
from service_processor import ServiceProcessor


def main():
    """
    Main function to coordinate the AWS API documentation parsing process.
    
    This function:
    - Sets up the necessary managers and processors
    - Handles graceful interruption and resumption
    - Coordinates the overall parsing workflow
    """
    # Use configuration from config module
    services_folder = DEFAULT_CONFIG['services_folder']
    output_folder = DEFAULT_CONFIG['output_folder']
    checkpoint_file = DEFAULT_CONFIG['checkpoint_file']
    
    logger.info("Starting AWS API documentation parser")
    logger.info(f"Services folder: {services_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Checkpoint file: {checkpoint_file}")
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_file)
    
    # Check if we're resuming from a previous run
    resume_service = checkpoint_manager.get_resume_position()
    if resume_service:
        logger.info(f"Resuming processing from service: {resume_service}")
    else:
        logger.info("Starting fresh processing run")
    
    # Create service processor with checkpoint manager
    processor = ServiceProcessor(services_folder, output_folder, checkpoint_manager)
    
    # For testing with a single file (uncomment to use)
    # service_file = "AccessAnalyzer.json"
    # service_file_path = os.path.join(services_folder, service_file)
    # logger.info(f"Processing single service file: {service_file}")
    # processor.process_service_file(service_file_path)
    
    # For processing all services
    try:
        processor.process_all_services()
        logger.info("Successfully completed processing all services")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user. Progress has been saved to checkpoint file.")
        print("\nProcessing interrupted. You can resume from the last checkpoint later.")
    except Exception as e:
        logger.error(f"Unexpected error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
