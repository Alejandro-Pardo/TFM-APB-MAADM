"""
AWS service documentation processor.

This module contains the ServiceProcessor class responsible for processing
entire AWS services, coordinating the parsing of individual methods,
and managing the overall workflow of the documentation extraction process.
"""

import json
import os
import time
import traceback
from tqdm.auto import tqdm

from utils.config import logger, DEFAULT_CONFIG
from utils.timeout import timeout, TimeoutException
from parsers.method_parser import MethodParser


class ServiceProcessor:
    """
    Processor for AWS service documentation.
    
    This class coordinates the processing of entire AWS services by:
    - Reading service JSON files
    - Managing method extraction for each service
    - Handling progress tracking and resumption
    - Creating service summaries and overall reports
    """
    
    def __init__(self, services_folder, output_folder, checkpoint_manager=None):
        """
        Initialize the service processor.
        
        Args:
            services_folder (str): Path to folder containing service JSON files
            output_folder (str): Path to folder for output files
            checkpoint_manager (CheckpointManager, optional): Manager for tracking progress
        """
        self.services_folder = services_folder
        self.output_folder = output_folder
        self.checkpoint_manager = checkpoint_manager
        self.method_parser = MethodParser(checkpoint_manager)
    
    @timeout(300)  # 5 minute timeout
    def process_service_file(self, service_file_path):
        """
        Process a service JSON file to extract method details.
        
        Args:
            service_file_path (str): Path to the service JSON file
        
        Returns:
            dict or None: Dictionary with service name and methods, or None if failed
        """
        # Read the service file
        try:
            with open(service_file_path, 'r') as f:
                service_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading service file {service_file_path}: {str(e)}")
            return None
        
        service_name = service_data['service_name']
        logger.info(f"Processing methods for {service_name}")
        
        # Check if this service has already been processed
        if self.checkpoint_manager and self.checkpoint_manager.is_service_processed(service_name):
            logger.info(f"Skipping already processed service: {service_name}")
            return None
        
        # Mark this service as current in progress
        if self.checkpoint_manager:
            self.checkpoint_manager.mark_service_as_current(service_name)
        
        # Get methods and their URLs
        methods, method_links = self._extract_methods_data(service_data, service_name)
        if not methods:
            if self.checkpoint_manager:
                self.checkpoint_manager.mark_service_as_processed(service_name)
            return None
        
        # Create output folder for this service
        service_output_folder = os.path.join(self.output_folder, service_name.replace(' ', '_'))
        if not os.path.exists(service_output_folder):
            os.makedirs(service_output_folder)
        
        # Process each method
        service_methods = []
        
        # Get already processed methods
        processed_methods = []
        if self.checkpoint_manager:
            processed_methods = self.checkpoint_manager.get_processed_methods(service_name)
        
        # Calculate initial progress for the progress bar
        initial_progress = len(processed_methods)
        total_methods = len(methods)
        
        # Log summary of skipped methods instead of individual entries
        if initial_progress > 0:
            logger.info(f"Skipping {initial_progress} already processed methods")
        
        # Use tqdm to show progress with correct initial position
        progress_bar = tqdm(total=total_methods, 
                          desc=f"Methods for {service_name}",
                          leave=False,
                          initial=initial_progress)
        
        for method_name, method_url in zip(methods, method_links):
            
            # Skip already processed methods
            if method_name in processed_methods:
                # Don't log individual skipped methods to reduce verbosity
                method_filename = method_name.replace(' ', '_')
                method_output_path = os.path.join(service_output_folder, f"{method_filename}.json")
                
                # Add to service methods list even if skipped (to maintain complete service summary)
                service_methods.append({
                    'name': method_name,
                    'url': method_url,
                    'file_path': method_output_path
                })
                continue
            
            # Scrape method details with proper error handling
            method_details = None
            try:
                method_details = self.method_parser.scrape_method_details(method_url, method_name, service_name)
            except TimeoutException:
                logger.error(f"Timeout processing method {method_name}")
                # Continue to next method after recording failure
                self._record_failure("failed_methods.txt", f"{service_name} - {method_name}: Timed out")
                progress_bar.update(1)
                time.sleep(DEFAULT_CONFIG['sleep_between_requests'])
                continue
            except Exception as e:
                logger.error(f"Error processing method {method_name}: {str(e)}")
                self._record_failure("failed_methods.txt", f"{service_name} - {method_name}: {str(e)}")
                progress_bar.update(1)
                time.sleep(DEFAULT_CONFIG['sleep_between_requests'])
                continue
            
            if method_details:
                # Save method details to file
                method_filename = method_name.replace(' ', '_')
                method_output_path = os.path.join(service_output_folder, f"{method_filename}.json")
                
                with open(method_output_path, 'w', encoding='utf-8') as f:
                    json.dump(method_details, f, indent=2)
                
                # Add to service methods list
                service_methods.append({
                    'name': method_name,
                    'url': method_url,
                    'file_path': method_output_path
                })
                
                # Mark method as processed
                if self.checkpoint_manager:
                    self.checkpoint_manager.mark_method_as_processed(service_name, method_name)
            
            # Update progress bar for processed method
            progress_bar.update(1)
            # Be nice to the server and avoid rate limiting
            time.sleep(DEFAULT_CONFIG['sleep_between_requests'])
        
        # Close the progress bar
        progress_bar.close()
        
        # Create and save service summary
        service_summary = self._create_service_summary(service_name, service_data['url'], service_methods)
        summary_path = os.path.join(service_output_folder, "_summary.json")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(service_summary, f, indent=2)
        
        # Mark service as fully processed
        if self.checkpoint_manager:
            self.checkpoint_manager.mark_service_as_processed(service_name)
        
        return service_summary
    
    def _extract_methods_data(self, service_data, service_name):
        """
        Extract methods and their URLs from service data.
        
        Args:
            service_data (dict): Service data from JSON file
            service_name (str): Name of the service
        
        Returns:
            tuple: (methods_list, method_links_list) or ([], []) if invalid data
        """
        methods = []
        method_links = []
        
        if 'client' in service_data:
            if 'methods_names' in service_data['client']:
                methods = service_data['client']['methods_names']
            elif 'methods' in service_data['client']:
                methods = service_data['client']['methods']
                
            if 'methods_links' in service_data['client']:
                method_links = service_data['client']['methods_links']
            elif 'method_links' in service_data['client']:
                method_links = service_data['client']['method_links']
        
        if not methods or not method_links or len(methods) != len(method_links):
            logger.error(f"Error: Invalid or missing method information for {service_name}")
            return [], []
        
        return methods, method_links
    
    def _create_service_summary(self, service_name, service_url, service_methods):
        """
        Create a summary of the service and its methods.
        
        Args:
            service_name (str): Name of the service
            service_url (str): URL of the service documentation
            service_methods (list): List of processed methods
        
        Returns:
            dict: Service summary dictionary
        """
        return {
            'service_name': service_name,
            'url': service_url,
            'method_count': len(service_methods),
            'methods': service_methods
        }
    
    def _record_failure(self, failure_file, message):
        """
        Record a failure message to a file.
        
        Args:
            failure_file (str): Name of the failure file
            message (str): Failure message to record
        """
        failure_path = os.path.join(self.output_folder, failure_file)
        with open(failure_path, "a", encoding='utf-8') as f:
            f.write(f"{message}\n")
    
    def process_all_services(self):
        """
        Process all service JSON files to extract method details.
        
        This method coordinates the processing of all services, handles
        resumption from checkpoints, and creates overall summaries.
        """
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Get all service JSON files
        service_files = [f for f in os.listdir(self.services_folder) if f.endswith('.json')]
        
        # Process each service file
        service_summaries = []
        
        # Check if we need to resume from a specific service
        resume_service = None
        resume_index = 0
        if self.checkpoint_manager:
            resume_service = self.checkpoint_manager.get_resume_position()
            
        resume_processing = False if resume_service else True
        
        # Calculate starting position based on already processed services
        if resume_service:
            # If resuming, find the index of the service to resume from
            for i, service_file in enumerate(service_files):
                service_file_path = os.path.join(self.services_folder, service_file)
                try:
                    with open(service_file_path, 'r') as f:
                        service_data = json.load(f)
                    if service_data['service_name'] == resume_service:
                        resume_index = i
                        break
                except Exception:
                    continue
        else:
            # If starting fresh, count already processed services
            if self.checkpoint_manager:
                for i, service_file in enumerate(service_files):
                    service_file_path = os.path.join(self.services_folder, service_file)
                    try:
                        with open(service_file_path, 'r') as f:
                            service_data = json.load(f)
                        if self.checkpoint_manager.is_service_processed(service_data['service_name']):
                            resume_index = i + 1  # Start from the next service
                        else:
                            break  # Found first unprocessed service
                    except Exception:
                        continue
        
        # Initialize progress bar with correct starting position
        progress_bar = tqdm(total=len(service_files), desc="Processing services", initial=resume_index)
        
        for i, service_file in enumerate(service_files):
            service_file_path = os.path.join(self.services_folder, service_file)
            
            # Skip files before our starting point
            if i < resume_index:
                progress_bar.update(1)
                continue
            
            # If we're resuming and haven't reached the resume point yet, skip
            if not resume_processing:
                # Read the service file to check if it's the one to resume from
                try:
                    with open(service_file_path, 'r') as f:
                        service_data = json.load(f)
                    
                    if service_data['service_name'] == resume_service:
                        resume_processing = True
                    else:
                        progress_bar.update(1)
                        continue
                except Exception:
                    progress_bar.update(1)
                    continue
            
            try:
                # Process service file with timeout handler
                service_summary = self.process_service_file(service_file_path)
                if service_summary:
                    service_summaries.append(service_summary)
                # Update progress bar for processed files
                progress_bar.update(1)
                # Sleep between services only when actually processing (not skipping)
                time.sleep(DEFAULT_CONFIG['sleep_between_services'])
            except TimeoutException:
                logger.error(f"Processing {service_file} timed out after 5 minutes, skipping")
                self._record_failure("failed_services.txt", f"{service_file}: Timed out")
                # Update progress bar for failed files
                progress_bar.update(1)
                # Sleep after timeout
                time.sleep(DEFAULT_CONFIG['sleep_between_services'])
            except Exception as e:
                logger.error(f"Error processing {service_file}: {str(e)}")
                traceback.print_exc()
                self._record_failure("failed_services.txt", f"{service_file}: {str(e)}")
                # Update progress bar for failed files
                progress_bar.update(1)
                # Sleep after error
                time.sleep(DEFAULT_CONFIG['sleep_between_services'])
        
        # Close the progress bar
        progress_bar.close()
        
        # Create and save overall summary
        self._save_overall_summary(service_summaries)
        
        logger.info(f"Processed {len(service_summaries)} services with methods")
    
    def _save_overall_summary(self, service_summaries):
        """
        Save an overall summary of all processed services.
        
        Args:
            service_summaries (list): List of service summary dictionaries
        """
        overall_summary = {
            'total_services': len(service_summaries),
            'services': service_summaries
        }
        
        summary_path = os.path.join(self.output_folder, "all_services_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, indent=2)
