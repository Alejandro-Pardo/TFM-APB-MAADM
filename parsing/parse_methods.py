import requests
import json
import os
import re
import time
import html
from tqdm import tqdm
from bs4 import BeautifulSoup, Tag


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
    
    def __init__(self):
        self.description_cleaner = DescriptionCleaner()
        
    def scrape_method_details(self, method_url, method_name):
        """
        Scrape details for a specific API method.
        
        Args:
            method_url (str): URL of the method documentation
            method_name (str): Name of the method
        
        Returns:
            dict: Dictionary with method details or None if failed
        """
        # Extract base URL and anchor
        base_url = method_url
        anchor = ""
        if '#' in method_url:
            base_url, anchor = method_url.split('#', 1)
        
        # Fetch the HTML content
        response = self._fetch_page(base_url)
        if not response:
            return None
        
        # Preprocess HTML to remove admonition notes
        preprocessed_html = preprocess_html(response.text)
        
        # Parse the preprocessed HTML content
        soup = BeautifulSoup(preprocessed_html, 'html.parser')
        
        # Find the main article and method section
        main_article = soup.find('article', attrs={'role': 'main'})
        if not main_article:
            print(f"Could not find the main article for {method_url}")
            return None
        
        method_section = self._find_method_section(main_article, anchor, method_name)
        if not method_section:
            print(f"Could not find method section for {method_name} at {method_url}")
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
        dd_element = method_section.find('dd')
        if dd_element:
            result['description'] = self._extract_description(dd_element)
            result['parameters'] = self._extract_parameters_section(dd_element)
            result['return_structure'] = self._extract_return_section(dd_element)
        
        return result
    
    def _fetch_page(self, url):
        """Fetch a web page and handle errors."""
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to fetch the page {url}. Status code: {response.status_code}")
                return None
            return response
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
    
    def _find_method_section(self, main_article, anchor, method_name):
        """Find the method section in the document."""
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
    
    def _extract_description(self, dd_element):
        """Extract method description from dd element."""
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
    
    def _extract_parameters_section(self, dd_element):
        """Extract the parameters section from the method details."""
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
    
    def _extract_parameters(self, params_dd):
        """Extract parameter information from the parameter section."""
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
    
    def _extract_single_param_description(self, params_dd, strong_tag):
        """Extract description for a single parameter."""
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
    
    def _extract_parameters_from_ul(self, ul_element, depth=0):
        """Extract parameters from an unordered list element with proper depth tracking."""
        parameters = []
        
        for li in ul_element.find_all('li', recursive=False):
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
                        # This is a standalone type declaration (e.g., "(string)"), process nested ULs
                        nested_ul = li.find('ul')
                        if nested_ul:
                            parameters.extend(self._extract_parameters_from_ul(nested_ul, depth))
                else:
                    # Regular parameter extraction
                    param_info = self._extract_parameter_info(li)
                    if param_info:
                        parameters.append(param_info)
            else:
                # Regular parameter extraction
                param_info = self._extract_parameter_info(li)
                if param_info:
                    parameters.append(param_info)
        
        return parameters

    def _extract_parameter_info(self, li_element):
        """Extract parameter information from a list item."""
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
    
    def _extract_return_structure(self, ul_element):
        """Extract return structure information from the Response Structure section."""
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
                item['nested_items'] = self._extract_return_structure(nested_ul)
            
            structure.append(item)
        
        return structure
    
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
    
    def __init__(self, services_folder, output_folder):
        self.services_folder = services_folder
        self.output_folder = output_folder
        self.method_parser = MethodParser()
    
    def process_service_file(self, service_file_path):
        """
        Process a service JSON file to extract method details.
        
        Args:
            service_file_path (str): Path to the service JSON file
        
        Returns:
            dict: Dictionary with service name and methods
        """
        # Read the service file
        with open(service_file_path, 'r') as f:
            service_data = json.load(f)
        
        service_name = service_data['service_name']
        print(f"Processing methods for {service_name}")
        
        # Get methods and their URLs
        methods, method_links = self._extract_methods_data(service_data, service_name)
        if not methods:
            return None
        
        # Create output folder for this service
        service_output_folder = os.path.join(self.output_folder, service_name.replace(' ', '_'))
        if not os.path.exists(service_output_folder):
            os.makedirs(service_output_folder)
        
        # Process each method
        service_methods = []
        
        # Use tqdm to show progress
        for method_name, method_url in tqdm(list(zip(methods, method_links)), 
                                         desc=f"Methods for {service_name}",
                                         leave=False):
            
            # Scrape method details
            method_details = self.method_parser.scrape_method_details(method_url, method_name)
            
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
            
            # Be nice to the server and avoid rate limiting
            time.sleep(0.5)
        
        # Create and save service summary
        service_summary = self._create_service_summary(service_name, service_data['url'], service_methods)
        summary_path = os.path.join(service_output_folder, "_summary.json")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(service_summary, f, indent=2)
        
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
            print(f"Error: Invalid or missing method information for {service_name}")
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
        
        for service_file in tqdm(service_files, desc="Processing services"):
            service_file_path = os.path.join(self.services_folder, service_file)
            
            try:
                # Process service file
                service_summary = self.process_service_file(service_file_path)
                if service_summary:
                    service_summaries.append(service_summary)
            except Exception as e:
                print(f"Error processing {service_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create and save overall summary
        self._save_overall_summary(service_summaries)
        
        print(f"Processed {len(service_summaries)} services with methods")
    
    def _save_overall_summary(self, service_summaries):
        """Save an overall summary of all processed services."""
        overall_summary = {
            'total_services': len(service_summaries),
            'services': service_summaries
        }
        
        with open(os.path.join(self.output_folder, "all_services_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, indent=2)


def main():
    services_folder = "TFM-APB-MAADM/parsing/services"
    output_folder = "TFM-APB-MAADM/parsing/methods"
    
    processor = ServiceProcessor(services_folder, output_folder)
    
    # For testing with a single file
    service_file = "AccessAnalyzer.json"
    service_file_path = os.path.join(services_folder, service_file)
    processor.process_service_file(service_file_path)
    
    # For processing all services (uncomment to use):
    # processor.process_all_services()


if __name__ == "__main__":
    main()