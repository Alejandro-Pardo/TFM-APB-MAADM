"""
Utility functions and classes for the AWS API parser.

This module contains common utilities like timeout decorators,
custom exceptions, and HTML preprocessing functions.
"""

import threading
import time
from bs4 import BeautifulSoup


class TimeoutException(Exception):
    """Raised when a function execution times out."""
    pass


def timeout(seconds):
    """
    Decorator to add timeout functionality to functions using threading.
    
    Args:
        seconds (int): Timeout duration in seconds
    
    Returns:
        function: Decorated function with timeout capability
    
    Raises:
        TimeoutException: If function execution exceeds timeout
    """
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


def preprocess_html(html_content):
    """
    Pre-process HTML content to remove admonition notes before parsing.
    
    Args:
        html_content (str): Raw HTML content
    
    Returns:
        str: Preprocessed HTML content with admonition notes removed
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find and remove all admonition notes
    admonition_notes = soup.find_all('div', class_='admonition')
    for note in admonition_notes:
        note.extract()
    
    return str(soup)
