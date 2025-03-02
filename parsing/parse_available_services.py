import requests
from bs4 import BeautifulSoup

def extract_aws_api_links(url, output_file):
    """
    Extract all the main AWS API links from the boto3 documentation page
    and save them to a text file. Only extracts links from the 'available-services' section.
    
    Args:
        url (str): URL of the boto3 documentation page
        output_file (str): Path to the output text file
    
    Returns:
        list: List of dictionaries with service names and their URLs
    """
    # Fetch the HTML content
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch the page. Status code: {response.status_code}")
        return []
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the "Available Services" section by its ID
    available_services_section = soup.find('section', id='available-services')
    
    if not available_services_section:
        print("Could not find the 'available-services' section.")
        return []
    
    # Find all the main service links within this section
    service_links = []
    
    # Find all li elements with class "toctree-l1" within the available-services section
    toctree_items = available_services_section.find_all('li', class_='toctree-l1')
    
    for item in toctree_items:
        # Find the first "a" element within each li
        link = item.find('a', class_='reference internal')
        if link:
            service_name = link.text.strip()
            href = link.get('href')
            
            # Make sure it's not a client or paginator link
            if '#' not in href:
                full_url = f"{url.rsplit('/', 1)[0]}/{href}"
                service_links.append({
                    'name': service_name,
                    'url': full_url
                })
    
    # Sort the links by service name for better readability
    service_links.sort(key=lambda x: x['name'].lower())
    
    # Save the links to a text file
    with open(output_file, 'w') as f:
        for service in service_links:
            #f.write(f"{service['name']}: {service['url']}\n")
            f.write(f"{service['url']}\n")
    
    print(f"Extracted {len(service_links)} AWS API links and saved to {output_file}")
    return service_links

if __name__ == "__main__":
    url = "https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/index.html"
    output_file = "aws_api_links.txt"
    extract_aws_api_links(url, output_file)