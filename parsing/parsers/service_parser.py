import requests
from bs4 import BeautifulSoup
import json
import os
from tqdm.auto import tqdm  # For progress bar

def read_api_links(input_file):
    """
    Read the API links from a text file that contains only links (one per line).
    
    Args:
        input_file (str): Path to the input text file
    
    Returns:
        list: List of URLs
    """
    api_links = []
    with open(input_file, 'r') as f:
        for line in f:
            url = line.strip()
            if url:
                api_links.append(url)
    return api_links

def scrape_api_details(url):
    """
    Scrape the details of a specific API page.
    
    Args:
        url (str): URL of the API documentation page
    
    Returns:
        dict: Dictionary with client and paginator details
    """    
    # Fetch the HTML content
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch the page. Status code: {response.status_code}")
        return {
            'service_name': "Unknown",
            'url': url,
            'client': {
                'description': "Failed to fetch page",
                'methods_names': [],
                'methods_links': []
            },
            'paginators': {
                'paginators_names': [],
                'paginators_links': []
            }
        }
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the main article
    main_article = soup.find('article', attrs={'role': 'main'})
    if not main_article:
        print(f"Could not find the main article for {url}")
        return None
    
    # Get service name from h1 tag
    service_name = "Unknown"
    h1_tag = main_article.find('h1')
    if h1_tag:
        service_name = h1_tag.text.strip().rstrip('#').strip()
    
    result = {
        'service_name': service_name,
        'url': url,
        'client': {
            'description': "",
            'methods_names': [],
            'methods_links': []
        },
        'paginators': {
            'paginators_names': [],
            'paginators_links': []
        }
    }
    
    # Extract client information
    client_section = main_article.find('section', id='client')
    if client_section:
        # Find client description
        py_class = client_section.find('dl', class_='py class')
        if py_class:
            dd_element = py_class.find('dd')
            if dd_element:
                paragraphs = dd_element.find_all('p')
                description = ' '.join([p.text.strip() for p in paragraphs])
                result['client']['description'] = description
        
        # Find client methods
        toctree_items = client_section.find_all('li', class_='toctree-l1')
        for item in toctree_items:
            link = item.find('a')
            if link:
                method_name = link.text.strip()
                method_link = link.get('href')
                if method_name and method_link:
                    # Make the link absolute
                    if method_link.startswith('#'):
                        method_link = f"{url}{method_link}"
                    elif not method_link.startswith('http'):
                        method_link = f"{url.rsplit('/', 1)[0]}/{method_link}"
                    
                    result['client']['methods_names'].append(method_name)
                    result['client']['methods_links'].append(method_link)
    
    # Extract paginator information
    paginator_section = main_article.find('section', id='paginators')
    if paginator_section:
        toctree_items = paginator_section.find_all('li', class_='toctree-l1')
        for item in toctree_items:
            link = item.find('a')
            if link:
                paginator_name = link.text.strip()
                paginator_link = link.get('href')
                if paginator_name and paginator_link:
                    # Make the link absolute
                    if paginator_link.startswith('#'):
                        paginator_link = f"{url}{paginator_link}"
                    elif not paginator_link.startswith('http'):
                        paginator_link = f"{url.rsplit('/', 1)[0]}/{paginator_link}"
                    
                    result['paginators']['paginators_names'].append(paginator_name)
                    result['paginators']['paginators_links'].append(paginator_link)
    
    return result

def scrape_all_apis(input_file, output_folder):
    """
    Scrape all API details and save them to separate JSON files in the specified folder.
    
    Args:
        input_file (str): Path to the input text file with API links
        output_folder (str): Path to the output folder to save JSON files
    """
    api_links = read_api_links(input_file)
    
    print(f"Found {len(api_links)} API links to scrape")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    for url in tqdm(api_links, desc="Scraping APIs"):
        api_details = scrape_api_details(url)
        if api_details:
            # Create a valid filename from the service name
            service_name = api_details['service_name']
            # Replace any characters that aren't valid in filenames
            filename = ''.join(c if c.isalnum() else '_' for c in service_name)
            filename = filename.rstrip('_')
            # Ensure the filename is valid and meaningful
            if not filename or filename == "Unknown":
                # Use the last part of the URL as a fallback
                url_parts = url.split('/')
                filename = url_parts[-1].split('.')[0]
            
            # Save to JSON file
            output_file = os.path.join(output_folder, f"{filename}.json")
            with open(output_file, 'w') as f:
                json.dump(api_details, f, indent=2)
    
    print(f"Scraped all API details and saved to individual files in {output_folder}")

if __name__ == "__main__":
    input_file = "../../docs/aws_api_urls.txt"
    output_folder = "../../docs/services"
    scrape_all_apis(input_file, output_folder)