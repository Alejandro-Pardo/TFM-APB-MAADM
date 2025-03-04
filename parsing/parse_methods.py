import requests
import json
import os
import re
import time
import html
from tqdm import tqdm
from bs4 import BeautifulSoup, Tag

def scrape_method_details(method_url, method_name):
    """
    Scrape details for a specific API method.
    
    Args:
        method_url (str): URL of the method documentation
        method_name (str): Name of the method
    
    Returns:
        dict: Dictionary with method details
    """
    # Extract base URL and anchor
    base_url = method_url
    anchor = ""
    if '#' in method_url:
        base_url, anchor = method_url.split('#', 1)
    
    # Fetch the HTML content
    try:
        response = requests.get(base_url)
        if response.status_code != 200:
            print(f"Failed to fetch the page {base_url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {base_url}: {str(e)}")
        return None
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the main article
    main_article = soup.find('article', attrs={'role': 'main'})
    if not main_article:
        print(f"Could not find the main article for {method_url}")
        return None
    
    # Find the method section using the anchor
    method_section = None
    if anchor:
        method_section = main_article.find(id=anchor)
        # Navigate to the parent dl tag if needed
        if method_section and not method_section.name == 'dl':
            # Look for closest dl with class "py method"
            method_section = method_section.find_next('dl', class_='py method')
    else:
        # If no anchor, try to find the method by name
        method_sections = main_article.find_all('dl', class_='py method')
        for section in method_sections:
            if section.find('dt') and method_name.lower() in section.find('dt').text.lower():
                method_section = section
                break
    
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
    
    # Find the description in dd element
    dd_element = method_section.find('dd')
    if dd_element:
        # Extract description from paragraphs before parameters field
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
                    description_paragraphs.append(clean_description(element.text.strip()))
        
        result['description'] = ' '.join(description_paragraphs)
    
    # Extract parameters - check if there's a Parameters field
    params_dt = dd_element.find('dt', class_='field-odd')
    params_header = dd_element.find('h3', string=re.compile('Parameters'))
    
    if params_dt and 'Parameters' in params_dt.text:
        params_dd = params_dt.find_next('dd', class_='field-odd')
        if params_dd:
            # Extract the parameter information recursively
            result['parameters'] = extract_parameters(params_dd)
    elif params_header:
        # Find the UL that follows the Parameters header
        params_ul = params_header.find_next('ul')
        if params_ul:
            # Extract parameters from this UL
            result['parameters'] = extract_parameters_from_ul(params_ul)
    
    # Extract return structure - specifically look for Returns, not Return type
    returns_dt = None
    for dt in dd_element.find_all('dt'):
        if dt.text.strip() == 'Returns:':
            returns_dt = dt
            break

    if returns_dt:
        returns_dd = returns_dt.find_next('dd')
        if returns_dd:
            # Check if return is None
            if "None" in returns_dd.text and not returns_dd.find('h3', string=re.compile('Response Structure')):
                result['return_structure'] = [{'type': 'None'}]
            else:
                # Look for Response Structure section anywhere in the dd_element
                response_structure_h3 = dd_element.find('h3', string=re.compile('Response Structure'))
                
                if response_structure_h3:
                    # Find the UL element that follows the Response Structure h3
                    response_ul = response_structure_h3.find_next('ul')
                    if response_ul:
                        result['return_structure'] = extract_return_structure(response_ul)
                else:
                    # Try alternate approach - look for return information in the text
                    return_p = returns_dd.find('p')
                    if return_p:
                        clean_text = (clean_description(return_p.text.strip()))
                        if clean_text.lower() == "none":
                            result['return_structure'] = [{'type': 'None'}]
                        else:
                            result['return_structure'] = [{'type': '', 'description': clean_text}]
        
        return result

def extract_parameters(params_dd):
    """
    Extract parameter information from the parameter section.
    
    Args:
        params_dd: BeautifulSoup element containing parameter information
    
    Returns:
        list: List of parameter dictionaries
    """
    parameters = []
    
    # Find the simple list of parameters
    ul = params_dd.find('ul', class_='simple')
    if ul:
        # Process each list item (parameter)
        for li in ul.find_all('li', recursive=False):
            param_info = extract_parameter_info(li)
            if param_info:
                parameters.append(param_info)
    else:
        # Handle case where there is a single parameter not in a list
        # Check for strong tag directly in the params_dd
        strong_tag = params_dd.find('strong')
        if strong_tag:
            # Create a parameter entry
            param = {
                'name': strong_tag.text.strip('*'),
                'type': '',
                'required': False,
                'description': '',
                'nested_params': []
            }
            
            # Try to extract type
            em_tag = params_dd.find('em')
            if em_tag:
                param['type'] = em_tag.text.strip()
            
            # Check if required
            if params_dd.find(string=re.compile(r'\[REQUIRED\]', re.IGNORECASE)):
                param['required'] = True
            
            # Extract description from paragraphs
            description_parts = []
            for p in params_dd.find_all('p'):
                # Skip paragraph with parameter name and type
                if p.find('strong') == strong_tag:
                    continue
                # Skip paragraph with just [REQUIRED]
                if p.text.strip() == '[REQUIRED]' or p.text.strip() == '**[REQUIRED]**':
                    continue
                
                if p.text.strip():
                    description_parts.append(p.text.strip())
            
            if description_parts:
                param['description'] = clean_description(' '.join(description_parts))
            
            parameters.append(param)
    
    return parameters

def extract_parameters_from_ul(ul_element):
    """
    Extract parameters from an unordered list element.
    
    Args:
        ul_element: BeautifulSoup unordered list element
    
    Returns:
        list: List of parameter dictionaries
    """
    parameters = []
    
    # Process each list item (parameter)
    for li in ul_element.find_all('li', recursive=False):
        param_info = extract_parameter_info(li)
        if param_info:
            parameters.append(param_info)
    
    return parameters


def clean_description(text):
    """
    Clean description text, handling encodings and removing duplicates.
    
    Args:
        text: Raw description text
    
    Returns:
        str: Cleaned description text
    """
    # Decode HTML entities
    text = html.unescape(text)
    
    # Fix specific problematic Unicode sequences (like apostrophes)
    text = text.replace('\u00e2\u0080\u0099', "'")

    # Remove unicode dash and similar characters
    text = re.sub(r'[\u00e2\u0080\u0093–—]', '-', text)
    
    # Fix common patterns of duplication
    # Some descriptions appear twice, separated by the parameter declaration
    parts = re.split(r'\s*–\s*', text)
    if len(parts) > 1:
        # Take the last part after any dashes
        text = parts[-1].strip()
    
    # Remove [REQUIRED] markers from descriptions
    text = re.sub(r'\s*\*?\[REQUIRED\]\*?\s*', ' ', text)
    
    # Remove parameter format text (name (type))
    text = re.sub(r'^[a-zA-Z0-9_]+ \([^)]+\)\s*-?\s*', '', text)
    
    # Clean up any leading dashes
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

def extract_parameter_info(li_element):
    """
    Extract parameter information from a list item.
    
    Args:
        li_element: BeautifulSoup list item element
    
    Returns:
        dict: Parameter information
    """
    # Initialize parameter dictionary
    param = {
        'name': '',
        'type': '',
        'required': False,
        'description': '',
        'nested_params': []
    }
    
    # Find the main parameter description
    main_p = li_element.find('p')
    if not main_p:
        return None
    
    # Extract parameter name and type
    strong_tag = main_p.find('strong')
    if strong_tag:
        param['name'] = strong_tag.text.strip('*')
        
        # Extract parameter type
        type_match = re.search(r'\(([^)]+)\)', main_p.text)
        if type_match:
            param['type'] = type_match.group(1).strip()
    
    # Check if parameter is required
    if li_element.find(string=re.compile(r'\[REQUIRED\]', re.IGNORECASE)):
        param['required'] = True
    
    # Extract and clean description
    description_parts = []
    
    for p in li_element.find_all('p'):
        text = p.text.strip()
        if text:
            description_parts.append(text)
    
    if description_parts:
        # Join all parts and clean
        full_description = ' '.join(description_parts)
        param['description'] = clean_description(full_description)
    
    # Check for nested parameters (sub-lists)
    sub_ul = li_element.find('ul')
    if sub_ul:
        for sub_li in sub_ul.find_all('li', recursive=False):
            # Check if this is just a dict type declaration
            first_p = sub_li.find('p')
            is_just_type = False
            
            if first_p and len(first_p.text.strip()) < 20 and '(dict)' in first_p.text:
                # This is likely just a type declaration like "*(dict) –*"
                is_just_type = True
            
            if is_just_type:
                # Add a dict container with the right type
                dict_param = {
                    'name': '',
                    'type': 'dict',
                    'required': False,
                    'description': '',
                    'nested_params': []
                }
                
                # Look for additional paragraphs that might contain description
                next_elements = first_p.find_next_siblings()
                for elem in next_elements:
                    if elem.name == 'ul':
                        # Stop at the next list which contains nested params
                        break
                    elif elem.name == 'p':
                        # Add this paragraph to the description
                        if elem.text.strip():
                            if dict_param['description']:
                                dict_param_description = re.sub(r'^[a-zA-Z0-9_]+ \([^)]+\)\s*-?\s*', '', elem.text.strip())
                                dict_param['description'] += ' ' + clean_description(dict_param_description)
                            else:
                                dict_param_description = re.sub(r'^[a-zA-Z0-9_]+ \([^)]+\)\s*-?\s*', '', elem.text.strip())
                                dict_param['description'] = clean_description(dict_param_description)
                
                # Find any sub-elements for this dict
                dict_ul = sub_li.find('ul')
                if dict_ul:
                    # Extract the actual parameters from this list
                    for dict_li in dict_ul.find_all('li', recursive=False):
                        sub_param = extract_parameter_info(dict_li)
                        if sub_param:
                            dict_param['nested_params'].append(sub_param)
                
                param['nested_params'].append(dict_param)
    
    return param

def extract_return_structure(ul_element):
    """
    Extract return structure information from the Response Structure section.
    
    Args:
        ul_element: BeautifulSoup unordered list element
    
    Returns:
        list: List of return structure items
    """
    structure = []
    
    for li in ul_element.find_all('li', recursive=False):
        item = {
            'name': '',
            'type': '',
            'description': '',
            'nested_items': []
        }
        
        # Extract type from the list item text
        type_match = re.search(r'\(([^)]+)\)', li.text)
        if type_match:
            item['type'] = type_match.group(1).strip()
        
        # Get the first paragraph to check if it's just a type declaration
        first_p = li.find('p')
        is_just_type = False
        
        if first_p and len(first_p.text.strip()) < 20 and '(dict)' in first_p.text:
            # This is likely just a type declaration like "*(dict) –*"
            is_just_type = True
            
        # Try to extract name from strong tag if present, but only if not just a type
        if not is_just_type:
            strong_tag = li.find('strong')
            if strong_tag:
                item['name'] = strong_tag.text.strip()
        
        # Extract description from paragraph only if not just a type
        if not is_just_type:
            p_tags = li.find_all('p')
            if p_tags:
                # Use the paragraph after the strong tag (if any)
                for p in p_tags:
                    if not p.find('strong') or p.find('strong') != strong_tag:
                        desc_text = clean_description(p.text.strip())
                        # Clean up the description
                        desc_text = re.sub(r'^\([^)]+\)\s*–\s*', '', desc_text)
                        item['description'] = desc_text
                        break
        
        # Process nested items if present
        nested_ul = li.find('ul')
        if nested_ul:
            item['nested_items'] = extract_return_structure(nested_ul)
        
        structure.append(item)
    
    return structure

def process_service_file(service_file_path, output_folder):
    """
    Process a service JSON file to extract method details.
    
    Args:
        service_file_path (str): Path to the service JSON file
        output_folder (str): Path to the output folder
    
    Returns:
        dict: Dictionary with service name and methods
    """
    # Read the service file
    with open(service_file_path, 'r') as f:
        service_data = json.load(f)
    
    service_name = service_data['service_name']
    print(f"Processing methods for {service_name}")
    
    # Get methods and their URLs - adjusted for your JSON structure
    # Check which fields exist in your JSON structure
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
        return None
    
    # Create output folder for this service
    service_output_folder = os.path.join(output_folder, service_name.replace(' ', '_'))
    if not os.path.exists(service_output_folder):
        os.makedirs(service_output_folder)
    
    # Process each method
    service_methods = []
    
    # Use tqdm to show progress for methods within this service
    for i, (method_name, method_url) in enumerate(tqdm(list(zip(methods, method_links)), 
                                                    desc=f"Methods for {service_name}",
                                                    leave=False)):
        
        print(f"  Scraping method {i+1}/{len(methods)}: {method_name}")
        
        # Scrape method details
        method_details = scrape_method_details(method_url, method_name)
        
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
    
    # Create a service summary
    service_summary = {
        'service_name': service_name,
        'url': service_data['url'],
        'method_count': len(service_methods),
        'methods': service_methods
    }
    
    # Save service summary
    summary_path = os.path.join(service_output_folder, "_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(service_summary, f, indent=2)
    
    return service_summary

def process_all_services(services_folder, output_folder):
    """
    Process all service JSON files to extract method details.
    
    Args:
        services_folder (str): Path to the folder containing service JSON files
        output_folder (str): Path to the output folder
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all service JSON files
    service_files = [f for f in os.listdir(services_folder) if f.endswith('.json')]
    
    # Process each service file
    service_summaries = []
    
    for i, service_file in enumerate(tqdm(service_files, desc="Processing services")):
        service_file_path = os.path.join(services_folder, service_file)
        
        try:
            # Process service file
            service_summary = process_service_file(service_file_path, output_folder)
            if service_summary:
                service_summaries.append(service_summary)
        except Exception as e:
            print(f"Error processing {service_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create overall summary
    overall_summary = {
        'total_services': len(service_summaries),
        'services': service_summaries
    }
    
    # Save overall summary
    with open(os.path.join(output_folder, "all_services_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=2)
    
    print(f"Processed {len(service_summaries)} services with methods")

if __name__ == "__main__":
    services_folder = "TFM-APB-MAADM/parsing/services"
    output_folder = "TFM-APB-MAADM/parsing/methods"
    
    # For testing with a single file, uncomment these lines:
    # service_file = "AccessAnalyzer.json"
    # service_file_path = os.path.join(services_folder, service_file)
    # process_service_file(service_file_path, output_folder)
    
    # For processing all services:
    process_all_services(services_folder, output_folder)