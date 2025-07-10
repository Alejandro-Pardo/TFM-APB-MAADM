"""
AWS API method documentation parser.

This module contains the MethodParser class responsible for scraping and parsing
individual AWS API method documentation pages to extract structured information
about methods, their parameters, and return values.
"""

import re
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from utils.config import logger
from utils.timeout import timeout, TimeoutException
from utils.text_cleaner import DescriptionCleaner


class MethodParser:
    """
    Parser for AWS API method documentation.
    
    This class handles scraping individual method documentation pages from AWS
    and extracting structured information including descriptions, parameters,
    and return structures.
    """
    
    def __init__(self, checkpoint_manager=None):
        """
        Initialize the method parser.
        
        Args:
            checkpoint_manager (CheckpointManager, optional): Manager for tracking progress
        """
        self.description_cleaner = DescriptionCleaner()
        self.checkpoint_manager = checkpoint_manager
        
    @timeout(50)
    def fetch_page(self, url):
        """
        Fetch a web page with timeout and retry logic.
        
        Args:
            url (str): URL to fetch
        
        Returns:
            requests.Response or None: Response object if successful, None otherwise
        """
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
            dict or None: Dictionary with method details or None if failed
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
            preprocessed_html = DescriptionCleaner.delete_admonition_notes(response.text)
            
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
        """
        Find the method section in the document.
        
        Args:
            main_article: BeautifulSoup element containing the main article
            anchor (str): Anchor fragment from the URL
            method_name (str): Name of the method to find
        
        Returns:
            BeautifulSoup element or None: Method section element if found
        """
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
        """
        Extract method description from dd element.
        
        Args:
            dd_element: BeautifulSoup element containing method details
        
        Returns:
            str: Cleaned method description
        """
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
        """
        Extract the parameters section from the method details.
        
        Args:
            dd_element: BeautifulSoup element containing method details
        
        Returns:
            list: List of parameter dictionaries
        """
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
        """
        Extract parameter information from the parameter section.
        
        Args:
            params_dd: BeautifulSoup element containing parameters
        
        Returns:
            list: List of parameter dictionaries
        """
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
        """
        Extract description for a single parameter.
        
        Args:
            params_dd: BeautifulSoup element containing parameter details
            strong_tag: BeautifulSoup element containing parameter name
        
        Returns:
            str: Cleaned parameter description
        """
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
        """
        Extract parameters from an unordered list element with proper depth tracking.
        
        Args:
            ul_element: BeautifulSoup ul element containing parameters
            depth (int): Current nesting depth for recursion tracking
        
        Returns:
            list: List of parameter dictionaries
        """
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
        """
        Extract parameter information from a list item.
        
        Args:
            li_element: BeautifulSoup li element containing parameter info
        
        Returns:
            dict or None: Parameter dictionary if successful, None otherwise
        """
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
        """
        Create a basic parameter dictionary structure.
        
        Args:
            name (str): Parameter name
        
        Returns:
            dict: Basic parameter dictionary
        """
        return {
            'name': name,
            'type': '',
            'required': False,
            'description': '',
            'nested_params': []
        }
    
    def _extract_return_section(self, dd_element):
        """
        Extract the return structure section from the method details.
        
        Args:
            dd_element: BeautifulSoup element containing method details
        
        Returns:
            list: List of return structure dictionaries
        """
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
        """
        Extract return structure information from the Response Structure section.
        
        Args:
            ul_element: BeautifulSoup ul element containing return structure
            current_depth (int): Current nesting depth for recursion tracking
        
        Returns:
            list: List of return structure dictionaries
        """
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
        """
        Create a basic return item dictionary structure.
        
        Returns:
            dict: Basic return item dictionary
        """
        return {
            'name': '',
            'type': '',
            'description': '',
            'nested_items': []
        }
