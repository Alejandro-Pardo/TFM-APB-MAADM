import requests
import json
import os
import re
import time
import html
import threading
import sys
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom timeout exception
class TimeoutException(Exception):
    """Raised when a function execution times out."""
    pass

# Create a timeout decorator using threading.Timer instead of signals
def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [TimeoutException("Function call timed out")]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            t = threading.Thread(target=target)
            t.daemon = True
            t.start()
            t.join(seconds)
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

class CheckpointManager:
    """Manages checkpoints to allow resuming from where processing stopped."""
    
    def __init__(self, checkpoint_file="checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data = self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint data from file if it exists."""
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
        """Create a default checkpoint structure."""
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
        """Check if a service has been fully processed."""
        return service_name in self.checkpoint_data["services_processed"]
    
    def is_method_processed(self, service_name, method_name):
        """Check if a method has been processed."""
        if service_name not in self.checkpoint_data["methods_processed"]:
            return False
        return method_name in self.checkpoint_data["methods_processed"][service_name]
    
    def mark_service_as_current(self, service_name):
        """Mark a service as the current one being processed."""
        self.checkpoint_data["current_service"] = service_name
        self.save_checkpoint()
    
    def mark_service_as_processed(self, service_name):
        """Mark a service as fully processed."""
        if service_name not in self.checkpoint_data["services_processed"]:
            self.checkpoint_data["services_processed"].append(service_name)
            self.checkpoint_data["current_service"] = None
            self.save_checkpoint()
    
    def mark_method_as_processed(self, service_name, method_name):
        """Mark a method as processed."""
        if service_name not in self.checkpoint_data["methods_processed"]:
            self.checkpoint_data["methods_processed"][service_name] = []
        
        if method_name not in self.checkpoint_data["methods_processed"][service_name]:
            self.checkpoint_data["methods_processed"][service_name].append(method_name)
            self.save_checkpoint()
    
    def get_resume_position(self):
        """Get the position to resume processing from."""
        return self.checkpoint_data["current_service"]
    
    def get_processed_methods(self, service_name):
        """Get the list of processed methods for a service."""
        if service_name not in self.checkpoint_data["methods_processed"]:
            return []
        return self.checkpoint_data["methods_processed"][service_name]

def preprocess_html(html_content):
    """Pre-process the HTML to remove all admonition notes before parsing."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find and remove all admonition notes
    admonition_notes = soup.find_all('div', class_='admonition')
    for note in admonition_notes:
        note.extract()
    
    return str(soup)

class DescriptionCleaner:
    """Utility class for cleaning and formatting description text."""
    
    @staticmethod
    def clean(text):
        """
        Clean description text, handling encodings and removing duplicates.
        
        Args:
            text: Raw description text
        
        Returns:
            str: Cleaned description text
        """
        if not text:
            return ""
            
        # Decode HTML entities
        text = html.unescape(text)
        
        # Fix specific problematic Unicode sequences
        text = text.replace('\u00e2\u0080\u0099', "'")
        text = text.replace('\u00e2\u0080\u009c', "'")
        text = text.replace('\u00e2\u0080\u009d', "'")
        text = re.sub(r'[\u00e2\u0080\u0093–—]', '-', text)
        
        # Fix common patterns of duplication
        parts = re.split(r'\s*–\s*', text)
        if len(parts) > 1:
            text = parts[-1].strip()
        
        # Remove [REQUIRED] markers and parameter format text
        text = re.sub(r'\s*\*?\[REQUIRED\]\*?\s*', ' ', text)
        text = re.sub(r'^[a-zA-Z0-9_]+ \([^)]+\)\s*-?\s*', '', text)
        text = re.sub(r'^-+\s*', '', text)
        
        # Fix double descriptions (common issue in the AWS docs)
        sentences = text.split('. ')
        unique_sentences = []
        
        for sentence in sentences:
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        text = '. '.join(unique_sentences)
        
        # Ensure the text ends with a period if it's a sentence
        if text and text[-1].isalpha():
            text += '.'
        
        return text.strip()

class MethodParser:
    """Parser for AWS API method documentation."""
    
    def __init__(self, checkpoint_manager=None):
        self.description_cleaner = DescriptionCleaner()
        self.checkpoint_manager = checkpoint_manager
        
    @timeout(50)
    def fetch_page(self, url):
        """Fetch a web page with timeout."""
        # Set up retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        try:
            # Set a 30-second timeout
            response = session.get(url, timeout=50)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch the page {url}. Status code: {response.status_code}")
                return None
            return response
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out for {url}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for {url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
        
    def scrape_method_details(self, method_url, method_name, service_name=None):
        """
        Scrape details for a specific API method.
        
        Args:
            method_url (str): URL of the method documentation
            method_name (str): Name of the method
            service_name (str, optional): Name of the service
        
        Returns:
            dict: Dictionary with method details or None if failed
        """
        # Check if this method has already been processed
        if service_name and self.checkpoint_manager and self.checkpoint_manager.is_method_processed(service_name, method_name):
            logger.info(f"Skipping already processed method: {method_name}")
            return None

        # Extract base URL and anchor
        base_url = method_url
        anchor = ""
        if '#' in method_url:
            base_url, anchor = method_url.split('#', 1)
        
        try:
            # Fetch the HTML content
            response = self.fetch_page(base_url)
            if not response:
                return None
                
            # Preprocess HTML to remove admonition notes
            preprocessed_html = preprocess_html(response.text)
            
            # Parse the preprocessed HTML content
            soup = BeautifulSoup(preprocessed_html, 'html.parser')
            
            # Find the main article and method section
            main_article = soup.find('article', attrs={'role': 'main'})
            if not main_article:
                logger.warning(f"Could not find the main article for {method_url}")
                return None
            
            method_section = self._find_method_section(main_article, anchor, method_name)
            if not method_section:
                logger.warning(f"Could not find method section for {method_name} at {method_url}")
                return None
            
            # Initialize result
            result = {
                'method_name': method_name,
                'url': method_url,
                'description': '',
                'parameters': [],
                'return_structure': []
            }
            
            # Process method content
            try:
                dd_element = method_section.find('dd')
                if dd_element:
                    result['description'] = self._extract_description(dd_element)
                    result['parameters'] = self._extract_parameters_section(dd_element)
                    result['return_structure'] = self._extract_return_section(dd_element)
                
                # Mark method as processed if we have a checkpoint manager
                if service_name and self.checkpoint_manager:
                    self.checkpoint_manager.mark_method_as_processed(service_name, method_name)
                
                return result
                
            except Exception as e:
                logger.error(f"Error parsing method content for {method_name}: {str(e)}")
                return None
                
        except TimeoutException:
            logger.error(f"Timeout while fetching or processing {method_name}")
            return None
        except Exception as e:
            logger.error(f"Error processing method {method_name}: {str(e)}")
            return None
    
    def _find_method_section(self, main_article, anchor, method_name):
        """Find the method section in the document."""
        try:
            if anchor:
                method_section = main_article.find(id=anchor)
                # Navigate to the parent dl tag if needed
                if method_section and not method_section.name == 'dl':
                    method_section = method_section.find_next('dl', class_='py method')
            else:
                # If no anchor, try to find the method by name
                method_sections = main_article.find_all('dl', class_='py method')
                method_section = None
                for section in method_sections:
                    if section.find('dt') and method_name.lower() in section.find('dt').text.lower():
                        method_section = section
                        break
            
            return method_section
        except Exception as e:
            logger.error(f"Error finding method section for {method_name}: {str(e)}")
            return None
    
    def _extract_description(self, dd_element):
        """Extract method description from dd element."""
        try:
            description_paragraphs = []
            
            # Process each element in the dd to extract description
            for element in dd_element.children:
                if isinstance(element, Tag):
                    # Stop when we reach parameters section or request syntax
                    if (element.name == 'dl' and element.find('dt', class_='field-odd')) or \
                      (element.name == 'h3' and 'Request Syntax' in element.text):
                        break
                    
                    # Skip "See also" paragraphs
                    if element.name == 'p' and 'See also' in element.text:
                        continue
                        
                    # Add paragraph text to description
                    if element.name == 'p':
                        description_paragraphs.append(self.description_cleaner.clean(element.text.strip()))
            
            return ' '.join(description_paragraphs)
        except Exception as e:
            logger.error(f"Error extracting description: {str(e)}")
            return ""
    
    def _extract_parameters_section(self, dd_element):
        """Extract the parameters section from the method details."""
        try:
            # Check for Parameters field using multiple methods
            params_dt = dd_element.find('dt', class_='field-odd')
            params_header = dd_element.find('h3', string=re.compile('Parameters'))
            
            if params_dt and 'Parameters' in params_dt.text:
                params_dd = params_dt.find_next('dd', class_='field-odd')
                if params_dd:
                    return self._extract_parameters(params_dd)
            elif params_header:
                params_ul = params_header.find_next('ul')
                if params_ul:
                    return self._extract_parameters_from_ul(params_ul)
            
            return []
        except Exception as e:
            logger.error(f"Error extracting parameters section: {str(e)}")
            return []
    
    def _extract_parameters(self, params_dd):
        """Extract parameter information from the parameter section."""
        try:
            parameters = []
            
            # Find the simple list of parameters
            ul = params_dd.find('ul', class_='simple')
            if ul:
                for li in ul.find_all('li', recursive=False):
                    param_info = self._extract_parameter_info(li)
                    if param_info:
                        parameters.append(param_info)
            else:
                # Handle case where there is a single parameter not in a list
                strong_tag = params_dd.find('strong')
                if strong_tag:
                    param = self._create_basic_param(strong_tag.text.strip('*'))
                    
                    # Extract type
                    em_tag = params_dd.find('em')
                    if em_tag:
                        param['type'] = em_tag.text.strip()
                    
                    # Check if required
                    if params_dd.find(string=re.compile(r'\[REQUIRED\]', re.IGNORECASE)):
                        param['required'] = True
                    
                    # Extract description from paragraphs
                    param['description'] = self._extract_single_param_description(params_dd, strong_tag)
                    
                    parameters.append(param)
            
            return parameters
        except Exception as e:
            logger.error(f"Error extracting parameters: {str(e)}")
            return []
    
    def _extract_single_param_description(self, params_dd, strong_tag):
        """Extract description for a single parameter."""
        try:
            description_parts = []
            for p in params_dd.find_all('p'):
                # Skip paragraph with parameter name and type or just [REQUIRED]
                if p.find('strong') == strong_tag:
                    continue
                if p.text.strip() in ['[REQUIRED]', '**[REQUIRED]**']:
                    continue
                
                if p.text.strip():
                    description_parts.append(p.text.strip())
            
            if description_parts:
                return self.description_cleaner.clean(' '.join(description_parts))
            return ''
        except Exception as e:
            logger.error(f"Error extracting single param description: {str(e)}")
            return ""
    
    def _extract_parameters_from_ul(self, ul_element, depth=0):
        """Extract parameters from an unordered list element with proper depth tracking."""
        try:
            parameters = []
            
            for li in ul_element.find_all('li', recursive=False):
                if not isinstance(li, Tag):
                    continue
                    
                first_p = li.find('p')
                if first_p and first_p.find('em'):
                    # Check if this is a type declaration for a parent parameter (e.g., "(dict)")
                    type_match = re.search(r'\(([^)]+)\)', first_p.text)
                    if type_match:
                        param_type = type_match.group(1).strip()
                        # Check if there's a parent parameter name (strong tag)
                        strong_tag = first_p.find('strong')
                        if strong_tag:
                            # This is a named parameter with type (e.g., "configurations (dict)")
                            param_name = strong_tag.text.strip('*')
                            param = self._create_basic_param(param_name)
                            param['type'] = param_type
                            # Extract description from subsequent elements
                            desc_parts = []
                            for elem in first_p.next_siblings:
                                if isinstance(elem, Tag):
                                    if elem.name == 'ul':
                                        break  # Stop at nested list
                                    if elem.name == 'p' and not elem.find('strong'):
                                        desc_parts.append(elem.text.strip())
                            param['description'] = self.description_cleaner.clean(' '.join(desc_parts))
                            # Process nested ULs for this parameter
                            nested_ul = li.find('ul')
                            if nested_ul:
                                param['nested_params'] = self._extract_parameters_from_ul(nested_ul, depth + 1)
                            parameters.append(param)
                        else:
                            # This is a standalone type declaration, process nested ULs
                            nested_ul = li.find('ul')
                            if nested_ul:
                                parameters.extend(self._extract_parameters_from_ul(nested_ul, depth))
                else:
                    # Regular parameter extraction
                    param_info = self._extract_parameter_info(li)
                    if param_info:
                        parameters.append(param_info)
            
            return parameters
        except Exception as e:
            logger.error(f"Error extracting parameters from UL at depth {depth}: {str(e)}")
            return []

    def _extract_parameter_info(self, li_element):
        """Extract parameter information from a list item."""
        try:
            # Find all paragraphs in the list item
            paragraphs = li_element.find_all('p')
            if not paragraphs:
                return None
            
            # First paragraph should contain the parameter name
            main_p = paragraphs[0]
            strong_tag = main_p.find('strong')
            if not strong_tag:
                return None
            
            param = self._create_basic_param(strong_tag.text.strip('*'))
            
            # Extract type from the main paragraph
            type_match = re.search(r'\(([^)]+)\)', main_p.text)
            if type_match:
                param['type'] = type_match.group(1).strip()
            
            # Check if required - look through all paragraphs
            for p in paragraphs:
                if '[REQUIRED]' in p.text:
                    param['required'] = True
                    break
            
            # Extract description - look for paragraphs after the parameter name and [REQUIRED] ones
            desc_parts = []
            for p in paragraphs:
                # Skip the parameter name paragraph and [REQUIRED] paragraphs
                if p == main_p or '[REQUIRED]' in p.text:
                    continue
                
                # Add this paragraph text to description
                desc_text = p.text.strip()
                if desc_text:
                    desc_parts.append(desc_text)
            
            if desc_parts:
                param['description'] = self.description_cleaner.clean(' '.join(desc_parts))
            
            # Process nested parameters
            nested_ul = li_element.find('ul')
            if nested_ul:
                param['nested_params'] = self._extract_parameters_from_ul(nested_ul)
            
            return param
        except Exception as e:
            logger.error(f"Error extracting parameter info: {str(e)}")
            return None
    
    def _create_basic_param(self, name):
        """Create a basic parameter dictionary."""
        return {
            'name': name,
            'type': '',
            'required': False,
            'description': '',
            'nested_params': []
        }
    
    def _extract_return_section(self, dd_element):
        """Extract the return structure section from the method details."""
        try:
            # Extract return structure - specifically look for Returns, not Return type
            returns_dt = None
            for dt in dd_element.find_all('dt'):
                if dt.text.strip() == 'Returns:':
                    returns_dt = dt
                    break

            if not returns_dt:
                return []
                
            returns_dd = returns_dt.find_next('dd')
            if not returns_dd:
                return []
                
            # Check if return is None
            if "None" in returns_dd.text and not returns_dd.find('h3', string=re.compile('Response Structure')):
                return [{'type': None}]
                
            # Look for Response Structure section
            response_structure_h3 = dd_element.find('h3', string=re.compile('Response Structure'))
            
            if response_structure_h3:
                response_ul = response_structure_h3.find_next('ul')
                if response_ul:
                    return self._extract_return_structure(response_ul)
            
            # Try alternate approach - look for return information in the text
            return_p = returns_dd.find('p')
            if return_p:
                clean_text = self.description_cleaner.clean(return_p.text.strip())
                if clean_text.lower() == "none":
                    return [{'type': None}]
                else:
                    return [{'type': '', 'description': clean_text}]
            
            return []
        except Exception as e:
            logger.error(f"Error extracting return section: {str(e)}")
            return []
    
    def _extract_return_structure(self, ul_element, current_depth=0):
        """Extract return structure information from the Response Structure section."""
        try:
            structure = []
            
            for li in ul_element.find_all('li', recursive=False):
                item = self._create_return_item()
                
                # Extract type from the list item text
                type_match = re.search(r'\(([^)]+)\)', li.text)
                if type_match:
                    item['type'] = type_match.group(1).strip()
                
                # Get the first paragraph to check if it's just a type declaration
                first_p = li.find('p')
                is_just_type = first_p and len(first_p.text.strip()) < 20 and '(dict)' in first_p.text
                
                # Process name and description if not just a type
                if not is_just_type:
                    # Try to extract name from strong tag
                    strong_tag = li.find('strong')
                    if strong_tag:
                        item['name'] = strong_tag.text.strip()
                    
                    # Extract description from paragraph
                    p_tags = li.find_all('p')
                    for p in p_tags:
                        if not p.find('strong') or p.find('strong') != strong_tag:
                            desc_text = self.description_cleaner.clean(p.text.strip())
                            # Clean up the description
                            desc_text = re.sub(r'^\([^)]+\)\s*–\s*', '', desc_text)
                            item['description'] = desc_text
                            break
                
                # Process nested items if present
                nested_ul = li.find('ul')
                if nested_ul:
                    item['nested_items'] = self._extract_return_structure(nested_ul, current_depth + 1)
                
                structure.append(item)
            
            return structure
        except Exception as e:
            logger.error(f"Error extracting return structure: {str(e)}")
            return []
    
    def _create_return_item(self):
        """Create a basic return item dictionary."""
        return {
            'name': '',
            'type': '',
            'description': '',
            'nested_items': []
        }


class ServiceProcessor:
    """Processor for AWS service documentation."""
    
    def __init__(self, services_folder, output_folder, checkpoint_manager=None):
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
            dict: Dictionary with service name and methods
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
        
        # Use tqdm to show progress
        for method_name, method_url in tqdm(list(zip(methods, method_links)), 
                                         desc=f"Methods for {service_name}",
                                         leave=False):
            
            # Skip already processed methods
            if method_name in processed_methods:
                logger.info(f"Skipping already processed method: {method_name}")
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
                with open(os.path.join(self.output_folder, "failed_methods.txt"), "a") as f:
                    f.write(f"{service_name} - {method_name}: Timed out\n")
                time.sleep(1.0)  # Sleep before continuing
                continue
            except Exception as e:
                logger.error(f"Error processing method {method_name}: {str(e)}")
                with open(os.path.join(self.output_folder, "failed_methods.txt"), "a") as f:
                    f.write(f"{service_name} - {method_name}: {str(e)}\n")
                time.sleep(1.0)  # Sleep before continuing
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
            
            # Be nice to the server and avoid rate limiting
            time.sleep(1.0)
        
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
        """Extract methods and their URLs from service data."""
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
        """Create a summary of the service and its methods."""
        return {
            'service_name': service_name,
            'url': service_url,
            'method_count': len(service_methods),
            'methods': service_methods
        }
    
    def process_all_services(self):
        """Process all service JSON files to extract method details."""
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Get all service JSON files
        service_files = [f for f in os.listdir(self.services_folder) if f.endswith('.json')]
        
        # Process each service file
        service_summaries = []
        
        # Check if we need to resume from a specific service
        resume_service = None
        if self.checkpoint_manager:
            resume_service = self.checkpoint_manager.get_resume_position()
            
        resume_processing = False if resume_service else True
        
        for service_file in tqdm(service_files, desc="Processing services"):
            service_file_path = os.path.join(self.services_folder, service_file)
            
            # If we're resuming and haven't reached the resume point yet, skip
            if not resume_processing:
                # Read the service file to check if it's the one to resume from
                try:
                    with open(service_file_path, 'r') as f:
                        service_data = json.load(f)
                    
                    if service_data['service_name'] == resume_service:
                        resume_processing = True
                    else:
                        continue
                except Exception:
                    continue
            
            try:
                # Process service file with timeout handler
                service_summary = self.process_service_file(service_file_path)
                if service_summary:
                    service_summaries.append(service_summary)
            except TimeoutException:
                logger.error(f"Processing {service_file} timed out after 5 minutes, skipping")
                # Write a failure record to track skipped services
                with open(os.path.join(self.output_folder, "failed_services.txt"), "a") as f:
                    f.write(f"{service_file}: Timed out\n")
            except Exception as e:
                logger.error(f"Error processing {service_file}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Write a failure record
                with open(os.path.join(self.output_folder, "failed_services.txt"), "a") as f:
                    f.write(f"{service_file}: {str(e)}\n")
            
            # Increased sleep time to avoid rate limiting and allow system resources to free up
            time.sleep(2.0)
        
        # Create and save overall summary
        self._save_overall_summary(service_summaries)
        
        logger.info(f"Processed {len(service_summaries)} services with methods")
    
    def _save_overall_summary(self, service_summaries):
        """Save an overall summary of all processed services."""
        overall_summary = {
            'total_services': len(service_summaries),
            'services': service_summaries
        }
        
        with open(os.path.join(self.output_folder, "all_services_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, indent=2)


def main():
    # Define folder paths
    services_folder = "TFM-APB-MAADM/parsing/services"
    output_folder = "TFM-APB-MAADM/parsing/methods"
    checkpoint_file = "TFM-APB-MAADM/parsing/checkpoint.json"
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_file)
    
    # Create service processor with checkpoint manager
    processor = ServiceProcessor(services_folder, output_folder, checkpoint_manager)
    
    # For testing with a single file (uncomment to use)
    # service_file = "AccessAnalyzer.json"
    # service_file_path = os.path.join(services_folder, service_file)
    # processor.process_service_file(service_file_path)
    
    # For processing all services
    try:
        processor.process_all_services()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user. Progress has been saved to checkpoint file.")
        print("\nProcessing interrupted. You can resume from the last checkpoint later.")


if __name__ == "__main__":
    main()