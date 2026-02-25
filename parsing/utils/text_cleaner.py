"""
Text cleaning utilities for AWS API documentation.

This module provides functionality to clean and format description text
from AWS API documentation, handling encoding issues and removing duplicates.
"""

import html
import re
from bs4 import BeautifulSoup


class DescriptionCleaner:
    """
    Utility class for cleaning and formatting description text from AWS documentation.
    
    This class handles common issues in AWS documentation text like:
    - HTML entity decoding
    - Unicode character cleanup
    - Duplicate text removal
    - Parameter marker cleanup
    - HTML admonition note removal
    """
    
    @staticmethod
    def delete_admonition_notes(html_content):
        """
        Remove admonition notes from HTML content before parsing.
        
        Args:
            html_content (str): Raw HTML content
        
        Returns:
            str: HTML content with admonition notes removed
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find and remove all admonition notes
        admonition_notes = soup.find_all('div', class_='admonition')
        for note in admonition_notes:
            note.extract()
        
        return str(soup)
    
    @staticmethod
    def clean(text):
        """
        Clean description text, handling encodings and removing duplicates.
        
        This method performs comprehensive text cleaning including:
        - HTML entity decoding
        - Unicode character normalization
        - Duplicate sentence removal
        - Parameter marker cleanup
        - Proper sentence formatting
        
        Args:
            text (str): Raw description text from AWS documentation
        
        Returns:
            str: Cleaned and formatted description text
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
